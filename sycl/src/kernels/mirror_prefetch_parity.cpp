// sycl/src/kernels/mirror_prefetch_parity.cpp - T2a (STRATA_PF_SLOTS): the prefetch's select and copy kernels
// (mirror_prefetch_tables, resident_plan_mirror.hpp) against a host replay of their contract. Synthetic, no model:
// 512 experts of which a third are resident, a third mirrored (blobs of random bytes in host USM, as the pinned
// mirror is) and a third neither; an S-slot ring in device memory; predicted id lists of 1..80 entries with
// duplicates. After every call: the ring table, the cursor and the pair count equal the replay (first occurrence,
// res < 0 && mir != 0, not in the ring, round-robin from the cursor skipping slots whose expert is predicted this
// round, at most S - protected), every occupied slot's bytes equal its source blob, no expert sits in two slots, at
// most S copies, and a second call with the same predictions copies nothing. The stats word [5] sums the copies.
#ifdef STRATA_SYCL_PROFILING_QUEUES
#define DPCT_PROFILING_ENABLED
#endif
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/resident_plan_mirror.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
// the kernels' contract on the host: updates `eid` and `cursor`, returns the selected (expert, slot) pairs in order
std::vector<std::pair<int32_t, int>> replay(const std::vector<int32_t>& pred, const std::vector<int32_t>& res,
                                             const std::vector<unsigned long long>& mir, std::vector<int32_t>& eid,
                                             uint32_t& cursor, int S) {
    const int n = (int) pred.size();
    std::vector<int> prot(S, 0);
    int nprot = 0;
    for (int s = 0; s < S; ++s)
        if (eid[s] >= 0 && std::find(pred.begin(), pred.end(), eid[s]) != pred.end()) { prot[s] = 1; ++nprot; }
    const int room = S - nprot;
    int c = (int) (cursor % (uint32_t) S);
    std::vector<std::pair<int32_t, int>> sel;
    for (int i = 0; i < n && (int) sel.size() < room; ++i) {
        const int32_t e = pred[i];
        if (e < 0 || e >= (int32_t) res.size()) continue;
        bool first = true;
        for (int j = 0; j < i; ++j)
            if (pred[j] == e) first = false;
        if (!first || res[e] >= 0 || mir[e] == 0) continue;
        if (std::find(eid.begin(), eid.end(), e) != eid.end()) continue;
        while (prot[c]) c = (c + 1) % S;
        sel.push_back({e, c});
        c = (c + 1) % S;
    }
    for (const auto& pr : sel) eid[pr.second] = pr.first;
    cursor = (uint32_t) c;
    return sel;
}
}  // namespace

int main() {
    const int NE = 512, S = 8;
    const long long blob_bytes = 4096 + 48;   // a multiple of 16, not of 256: the copy's 16-byte loads at any such size
    std::mt19937 rng(2026);
    auto& q = dpct::get_in_order_queue();

    // residency, the mirror (host USM blobs of random bytes) and the ring (device memory)
    std::vector<int32_t> res(NE);
    std::vector<unsigned long long> mir(NE, 0ull);
    std::vector<int> mir_index(NE, -1);
    int n_mir = 0;
    for (int e = 0; e < NE; ++e) {
        res[e] = e % 3 == 0 ? e : -1;
        if (e % 3 == 1) mir_index[e] = n_mir++;
    }
    uint8_t* h_blobs = (uint8_t*) sycl::malloc_host((size_t) n_mir * (size_t) blob_bytes, q);
    if (h_blobs == nullptr) { std::printf("mirror_prefetch_parity: host USM allocation failed\n"); return 1; }
    for (size_t i = 0; i < (size_t) n_mir * (size_t) blob_bytes; ++i) h_blobs[i] = (uint8_t) rng();
    for (int e = 0; e < NE; ++e)
        if (mir_index[e] >= 0) mir[e] = (unsigned long long) (h_blobs + (size_t) mir_index[e] * (size_t) blob_bytes);
    uint8_t* d_ring_mem = sycl::malloc_device<uint8_t>((size_t) S * (size_t) blob_bytes, q);
    int32_t* d_pred = sycl::malloc_device<int32_t>(128, q);
    int32_t* d_res = sycl::malloc_device<int32_t>(NE, q);
    unsigned long long* d_mir = sycl::malloc_device<unsigned long long>(NE, q);
    int32_t* d_eid = sycl::malloc_device<int32_t>(S, q);
    unsigned long long* d_base = sycl::malloc_device<unsigned long long>(S, q);
    uint32_t* d_cursor = sycl::malloc_device<uint32_t>(1, q);
    unsigned long long* d_pairs = sycl::malloc_device<unsigned long long>(2 * S, q);
    int32_t* d_nsel = sycl::malloc_device<int32_t>(1, q);
    uint32_t* d_stats = sycl::malloc_device<uint32_t>(k::kResidentPlanStatWords, q);
    if (!d_ring_mem || !d_pred || !d_res || !d_mir || !d_eid || !d_base || !d_cursor || !d_pairs || !d_nsel || !d_stats) {
        std::printf("mirror_prefetch_parity: device allocation failed\n");
        return 1;
    }
    std::vector<unsigned long long> base(S);
    for (int s = 0; s < S; ++s) base[s] = (unsigned long long) (d_ring_mem + (size_t) s * (size_t) blob_bytes);
    std::vector<int32_t> eid(S, -1);
    uint32_t cursor = 0;
    q.memcpy(d_res, res.data(), NE * 4).wait();
    q.memcpy(d_mir, mir.data(), NE * 8).wait();
    q.memcpy(d_eid, eid.data(), S * 4).wait();
    q.memcpy(d_base, base.data(), S * 8).wait();
    q.memset(d_cursor, 0, 4).wait();
    q.memset(d_pairs, 0, 2 * S * 8).wait();
    q.memset(d_nsel, 0, 4).wait();
    q.memset(d_stats, 0, k::kResidentPlanStatWords * 4).wait();
    q.memset(d_ring_mem, 0, (size_t) S * (size_t) blob_bytes).wait();

    int cases = 0, bad = 0;
    unsigned long long copies_total = 0;
    std::vector<uint8_t> ring_host((size_t) S * (size_t) blob_bytes);
    for (int c = 0; c < 240; ++c) {
        const int n = 1 + (int) (rng() % 80);
        // duplicates: a third of the entries repeat an earlier one; a narrow pool now and then so the ring turns over slowly
        const int pool = c % 5 == 0 ? 24 : NE;
        std::vector<int32_t> pred(n);
        for (int i = 0; i < n; ++i)
            pred[i] = (i > 0 && rng() % 3 == 0) ? pred[rng() % i] : (int32_t) (rng() % pool);
        q.memcpy(d_pred, pred.data(), n * 4).wait();
        for (int call = 0; call < 2; ++call) {   // the same prediction twice: the second copies nothing
            std::vector<int32_t> want_eid = eid;
            uint32_t want_cursor = cursor;
            const auto sel = replay(pred, res, mir, want_eid, want_cursor, S);
            k::mirror_prefetch_tables(d_pred, n, d_res, d_mir, NE, d_eid, d_base, d_cursor, d_pairs, d_nsel, S, blob_bytes,
                                      d_stats, nullptr);
            q.wait();
            std::vector<int32_t> got_eid(S);
            uint32_t got_cursor = 0;
            int32_t got_nsel = -1;
            q.memcpy(got_eid.data(), d_eid, S * 4).wait();
            q.memcpy(&got_cursor, d_cursor, 4).wait();
            q.memcpy(&got_nsel, d_nsel, 4).wait();
            q.memcpy(ring_host.data(), d_ring_mem, (size_t) S * (size_t) blob_bytes).wait();
            bool ok = got_eid == want_eid && got_cursor == want_cursor && got_nsel == (int32_t) sel.size();
            ok = ok && got_nsel <= S && (call == 0 || got_nsel == 0);
            for (int s = 0; s < S && ok; ++s) {   // no expert twice; occupied slots hold their blob's bytes
                if (got_eid[s] < 0) continue;
                for (int s2 = s + 1; s2 < S; ++s2)
                    if (got_eid[s2] == got_eid[s]) ok = false;
                const int mi = mir_index[got_eid[s]];
                if (mi < 0 || std::memcmp(ring_host.data() + (size_t) s * (size_t) blob_bytes,
                                          h_blobs + (size_t) mi * (size_t) blob_bytes, (size_t) blob_bytes) != 0)
                    ok = false;
            }
            copies_total += (unsigned long long) sel.size();
            eid = want_eid;
            cursor = want_cursor;
            ++cases;
            if (!ok) {
                std::printf("    *** case %d call %d n=%d: got n_sel %d cursor %u (want %zu, %u); ring", c, call, n, got_nsel,
                            got_cursor, sel.size(), want_cursor);
                for (int s = 0; s < S; ++s) std::printf(" %d/%d", got_eid[s], want_eid[s]);
                std::printf(" (got/want) ***\n");
                ++bad;
                // resynchronise the model with the device so one failure does not cascade
                q.memcpy(eid.data(), d_eid, S * 4).wait();
                cursor = got_cursor;
            }
        }
    }
    uint32_t stats[k::kResidentPlanStatWords] = {0, 0, 0, 0, 0, 0};
    q.memcpy(stats, d_stats, sizeof stats).wait();
    ++cases;
    if (stats[5] != copies_total || stats[0] + stats[1] + stats[2] + stats[3] + stats[4] != 0) {
        std::printf("    *** stats: copies word %u (want %llu), other words %u %u %u %u %u (want 0) ***\n", stats[5],
                    copies_total, stats[0], stats[1], stats[2], stats[3], stats[4]);
        ++bad;
    }

    sycl::free(h_blobs, q); sycl::free(d_ring_mem, q); sycl::free(d_pred, q); sycl::free(d_res, q); sycl::free(d_mir, q);
    sycl::free(d_eid, q); sycl::free(d_base, q); sycl::free(d_cursor, q); sycl::free(d_pairs, q); sycl::free(d_nsel, q);
    sycl::free(d_stats, q);
    std::printf("mirror_prefetch: %d cases, %d failures, %llu experts copied\n", cases, bad, copies_total);
    if (bad) return 1;
    std::printf("mirror_prefetch_parity OK\n");
    return 0;
}
