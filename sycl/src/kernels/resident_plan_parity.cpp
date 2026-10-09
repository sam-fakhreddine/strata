// src/kernels/resident_plan_parity.cpp - the verify window's device-built expert plan (#783 PR-e: the parallel
// grouping and the prefix scan that replaced thread 0's serial pass) against a host replay of its contract:
// distinct experts in routing order, each with its entries ascending, `ptr[g]` = the expert's slot, counts / start /
// dst / tok / start2 in the host pool's layout.
//
// n = 1..80 entries (a window of up to kVerifyMaxT tokens x 10 experts), ids drawn from pools of 3..512 experts so
// duplicates run from "every entry the same expert" to "all distinct", with and without a slot table (`slot_off`).
// Both modes are run: with a `skip` word (ring when the plan was built, 0 when an expert was not resident - and
// then the plan is untouched) and without one (the all-resident graph: a non-resident expert gives an empty plan and
// sets *plan_err).
// Every word of the plan the host pool would read is compared with memcmp. GPU, synthetic, no model.
// T2a: a third block registers a mirror table and a prefetch ring (resident_plan_set_mirror / resident_plan_set_ring,
// as generate.cpp does) with a stats buffer: a mirrored expert's group pointer must be its ring slot when the ring
// holds it, its mirror address otherwise, a resident one its cache slot whatever the ring says; the rest of the plan
// memcmp-equal; the stats words count what was served from where.
#ifdef STRATA_SYCL_PROFILING_QUEUES
#define DPCT_PROFILING_ENABLED
#endif
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/resident_plan_mirror.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(dpct::err0 e, const char *w) {
}
}  // namespace

int main() {
    const int K = 10, NE = 512;
    const long long capx = (long long) k::kVerifyMaxT * K;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    const size_t plan_words = (size_t) (ptr_off + 4 * capx + capx + 16);
    std::mt19937 rng(783);
    const long long blob = 1337;
    uint8_t* const cache_base = reinterpret_cast<uint8_t*>(uintptr_t(0x40000000));   // never dereferenced

    int32_t *d_ids, *d_res, *d_plan;
    unsigned long long* d_off;
    uint32_t* d_skip;
    ck(DPCT_CHECK_ERROR(d_ids = (int32_t *)sycl::malloc_device(
                            128 * 4, dpct::get_in_order_queue())),
       "ids");
    ck(DPCT_CHECK_ERROR(d_res = (int32_t *)sycl::malloc_device(
                            NE * 4, dpct::get_in_order_queue())),
       "res");
    ck(DPCT_CHECK_ERROR(d_off = (unsigned long long *)sycl::malloc_device(
                            1024 * 8, dpct::get_in_order_queue())),
       "off");
    ck(DPCT_CHECK_ERROR(d_plan = (int32_t *)sycl::malloc_device(
                            plan_words * 4, dpct::get_in_order_queue())),
       "plan");
    ck(DPCT_CHECK_ERROR(d_skip = (uint32_t *)sycl::malloc_device(
                            4, dpct::get_in_order_queue())),
       "skip");
    uint32_t* d_err = nullptr;
    ck(DPCT_CHECK_ERROR(d_err = (uint32_t *)sycl::malloc_device(
                            4, dpct::get_in_order_queue())),
       "err");

    std::vector<int32_t> res(NE);
    for (int e = 0; e < NE; ++e) res[e] = (e * 37 + 11) % 1000;      // distinct slots, not the identity
    std::vector<unsigned long long> off(1024);
    for (int s = 0; s < 1024; ++s) off[s] = (unsigned long long) s * 4096 + (s % 7) * 64;
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    ck(DPCT_CHECK_ERROR(
           (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_res, res.data(), NE * 4).wait()),
       "res");
    /*
    DPCT1114: cudaMemcpy is migrated to asynchronization memcpy, assuming
    in the original code the source host memory is pageable memory. If the
    memory is not pageable, call wait() on event return by memcpy API to ensure
    synchronization behavior.
    */
    ck(DPCT_CHECK_ERROR(
           (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_off, off.data(), 1024 * 8).wait()),
       "off");

    int cases = 0, bad = 0;
    for (int n = 1; n <= 80; ++n) {
        for (int variant = 0; variant < 4; ++variant) {
            const int pool = variant == 0 ? 3 : (variant == 1 ? 12 : (variant == 2 ? 40 : 512));
            std::vector<int32_t> ids(n);
            for (int i = 0; i < n; ++i) ids[i] = (int32_t) ((rng() % pool) * (512 / pool));
            const bool use_off = (n + variant) % 2 == 1;
            /*
            DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
            assuming in the original code the source host memory is pageable
            memory. If the memory is not pageable, call wait() on event return
            by memcpy API to ensure synchronization behavior.
            */
            ck(DPCT_CHECK_ERROR(
                   (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_ids, ids.data(), n * 4).wait()),
               "ids");

            // host replay
            std::vector<int32_t> counts = {0, n, 0}, start, dst(n), tok(n);
            std::vector<unsigned long long> ptr;
            std::vector<int> first_seen;     // group -> expert
            for (int i = 0; i < n; ++i) {
                int g = -1;
                for (size_t q = 0; q < first_seen.size(); ++q)
                    if (first_seen[q] == ids[i]) g = (int) q;
                if (g < 0) { first_seen.push_back(ids[i]); g = (int) first_seen.size() - 1; }
            }
            const int groups = (int) first_seen.size();
            counts[0] = groups;
            start.assign(groups + 1, 0);
            int pos = 0;
            for (int g = 0; g < groups; ++g) {
                start[g] = pos;
                ptr.push_back((unsigned long long) (uintptr_t) cache_base +
                              (use_off ? off[res[first_seen[g]]] : (unsigned long long) res[first_seen[g]] * (unsigned long long) blob));
                for (int i = 0; i < n; ++i)
                    if (ids[i] == first_seen[g]) { dst[pos] = i; tok[pos] = i / K; ++pos; }
            }
            start[groups] = n;

            for (int mode = 0; mode < 2; ++mode) {   // 0: general kernel with a skip word, 1: all resident
                ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                        .memset(d_plan, 0xAB, plan_words * 4)
                                        .wait()),
                   "fill");
                ck(DPCT_CHECK_ERROR(
                       (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memset(d_skip, 0, 4).wait()),
                   "skip0");
                ck(DPCT_CHECK_ERROR(
                       (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memset(d_err, 0, 4).wait()),
                   "err0");
                k::resident_plan(d_ids, n, K, d_res, NE, cache_base, use_off ? d_off : nullptr, blob, d_plan, capx,
                                 mode == 0 ? d_skip : nullptr, 7u, nullptr, d_err);
                ck(DPCT_CHECK_ERROR(
                       dpct::get_current_device().queues_wait_and_throw()),
                   "sync");
                std::vector<int32_t> plan(plan_words);
                ck(DPCT_CHECK_ERROR(
                       (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                           .memcpy(plan.data(), d_plan, plan_words * 4)
                           .wait()),
                   "plan");
                uint32_t skip = 0;
                ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                        .memcpy(&skip, d_skip, 4)
                                        .wait()),
                   "skipr");
                bool ok = mode == 1 || skip == 7u;
                ok = ok && std::memcmp(plan.data(), counts.data(), 3 * 4) == 0;
                ok = ok && std::memcmp(plan.data() + 4, start.data(), (size_t) (groups + 1) * 4) == 0;
                const int32_t* pdst = plan.data() + 4 + capx + 1;
                ok = ok && std::memcmp(pdst, dst.data(), (size_t) n * 4) == 0;
                ok = ok && std::memcmp(pdst + capx, tok.data(), (size_t) n * 4) == 0;
                ok = ok && std::memcmp(plan.data() + ptr_off, ptr.data(), (size_t) groups * 8) == 0;
                ok = ok && plan[(size_t) (ptr_off + 4 * capx)] == n;
                ++cases;
                if (!ok) {
                    std::printf("    *** n=%d pool=%d %s %s: the plan differs from the host replay ***\n", n, pool,
                                use_off ? "slot_off" : "blob", mode ? "all_resident" : "general");
                    ++bad;
                }
            }
        }
    }

    // a routed expert that is not resident: the general kernel leaves the plan alone and zeroes *skip
    {
        std::vector<int32_t> res2 = res;
        res2[5] = -1;
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR(
               (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_res, res2.data(), NE * 4).wait()),
           "res2");
        const int32_t ids[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        ck(DPCT_CHECK_ERROR(
               (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_ids, ids, sizeof ids).wait()),
           "ids");
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memset(d_plan, 0xAB, plan_words * 4)
                                .wait()),
           "fill");
        const uint32_t seven = 7;
        ck(DPCT_CHECK_ERROR(
               (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(d_skip, &seven, 4).wait()),
           "skip7");
        k::resident_plan(d_ids, 10, K, d_res, NE, cache_base, nullptr, blob, d_plan, capx, d_skip, 7u, nullptr, d_err);
        ck(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()),
           "sync");
        std::vector<int32_t> plan(plan_words);
        ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                                .memcpy(plan.data(), d_plan, plan_words * 4)
                                .wait()),
           "plan");
        uint32_t skip = 99;
        ck(DPCT_CHECK_ERROR(
               (dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue()).memcpy(&skip, d_skip, 4).wait()),
           "skipr");
        bool untouched = true;
        for (int32_t w : plan) untouched = untouched && (uint32_t) w == 0xABABABABu;
        ++cases;
        if (skip != 0 || !untouched) {
            std::printf("    *** a non-resident expert: skip=%u untouched=%d (want 0 / 1) ***\n", skip, (int) untouched);
            ++bad;
        }
    }

    // T2a: the mirror and the prefetch ring. Odd experts are not resident but mirrored (fake addresses, never
    // dereferenced: the kernel only copies them into the plan); even ones keep their slots. The ring holds five
    // mirrored experts, one resident one (ignored: the residency check comes first) and two empty slots.
    {
        const int S = 8;
        std::vector<int32_t> res3 = res;
        std::vector<unsigned long long> mir(NE, 0ull);
        for (int e = 1; e < NE; e += 2) { res3[e] = -1; mir[e] = 0x70000000ull + (unsigned long long) e * 4096; }
        std::vector<int32_t> ring_eid(S, -1);
        std::vector<unsigned long long> ring_base(S);
        for (int s = 0; s < S; ++s) ring_base[s] = 0x90000000ull + (unsigned long long) s * 8192;
        for (int s = 0; s < 5; ++s) ring_eid[s] = 2 * s + 1;   // 1, 3, 5, 7, 9
        ring_eid[5] = 2;                                      // resident: res wins
        auto& q = dpct::get_in_order_queue();
        unsigned long long* d_mir = sycl::malloc_device<unsigned long long>(NE, q);
        int32_t* d_reid = sycl::malloc_device<int32_t>(S, q);
        unsigned long long* d_rbase = sycl::malloc_device<unsigned long long>(S, q);
        uint32_t* d_cursor = sycl::malloc_device<uint32_t>(1, q);
        unsigned long long* d_pairs = sycl::malloc_device<unsigned long long>(2 * S, q);
        int32_t* d_nsel = sycl::malloc_device<int32_t>(1, q);
        uint32_t* d_stats = sycl::malloc_device<uint32_t>(k::kResidentPlanStatWords, q);
        q.memcpy(d_res, res3.data(), NE * 4).wait();
        q.memcpy(d_mir, mir.data(), NE * 8).wait();
        q.memcpy(d_reid, ring_eid.data(), S * 4).wait();
        q.memcpy(d_rbase, ring_base.data(), S * 8).wait();
        q.memset(d_cursor, 0, 4).wait();
        q.memset(d_pairs, 0, 2 * S * 8).wait();
        q.memset(d_nsel, 0, 4).wait();
        k::MirrorRing ring;
        ring.slots = S; ring.eid = d_reid; ring.base = d_rbase; ring.cursor = d_cursor; ring.pairs = d_pairs; ring.n_sel = d_nsel;
        k::resident_plan_set_mirror(d_res, d_mir);
        k::resident_plan_set_stats(d_res, d_stats, 1, NE);
        for (int with_ring = 0; with_ring < 2; ++with_ring) {   // 0: mirror only (D7's path), 1: mirror and ring
            if (with_ring) k::resident_plan_set_ring(d_res, ring, 1, NE);
            for (int n = 1; n <= 80; n += 3) {
                std::vector<int32_t> ids(n);
                for (int i = 0; i < n; ++i)   // a quarter of the entries name a ring expert, so hits are frequent
                    ids[i] = (rng() % 4 == 0) ? (int32_t) (2 * (rng() % 5) + 1) : (int32_t) (rng() % NE);
                q.memcpy(d_ids, ids.data(), n * 4).wait();
                std::vector<int32_t> counts = {0, n, 0}, start, dst(n), tok(n);
                std::vector<unsigned long long> ptr;
                std::vector<int> first_seen;
                for (int i = 0; i < n; ++i) {
                    int g = -1;
                    for (size_t q2 = 0; q2 < first_seen.size(); ++q2)
                        if (first_seen[q2] == ids[i]) g = (int) q2;
                    if (g < 0) first_seen.push_back(ids[i]);
                }
                const int groups = (int) first_seen.size();
                counts[0] = groups;
                start.assign(groups + 1, 0);
                int pos = 0;
                uint32_t want_stats[k::kResidentPlanStatWords] = {0, 0, 0, (uint32_t) groups, 0, 0};
                for (int g = 0; g < groups; ++g) {
                    start[g] = pos;
                    const int e = first_seen[g];
                    unsigned long long a;
                    if (res3[e] >= 0) a = (unsigned long long) (uintptr_t) cache_base + (unsigned long long) res3[e] * (unsigned long long) blob;
                    else {
                        a = mir[e];
                        if (with_ring)
                            for (int s = 0; s < S; ++s)
                                if (ring_eid[s] == e) a = ring_base[s];
                    }
                    ptr.push_back(a);
                    for (int i = 0; i < n; ++i)
                        if (ids[i] == e) { dst[pos] = i; tok[pos] = i / K; ++pos; }
                }
                start[groups] = n;
                for (int i = 0; i < n; ++i) {
                    const int e = ids[i];
                    if (res3[e] >= 0) ++want_stats[0];
                    else {
                        ++want_stats[1];
                        if (with_ring)
                            for (int s = 0; s < S; ++s)
                                if (ring_eid[s] == e) ++want_stats[4];
                    }
                }
                q.memset(d_plan, 0xAB, plan_words * 4).wait();
                q.memset(d_skip, 0, 4).wait();
                q.memset(d_stats, 0, k::kResidentPlanStatWords * 4).wait();
                k::resident_plan(d_ids, n, K, d_res, NE, cache_base, nullptr, blob, d_plan, capx, d_skip, 7u, nullptr, d_err);
                q.wait();
                std::vector<int32_t> plan(plan_words);
                q.memcpy(plan.data(), d_plan, plan_words * 4).wait();
                uint32_t skip = 0, got_stats[k::kResidentPlanStatWords];
                q.memcpy(&skip, d_skip, 4).wait();
                q.memcpy(got_stats, d_stats, k::kResidentPlanStatWords * 4).wait();
                bool ok = skip == 7u;
                ok = ok && std::memcmp(plan.data(), counts.data(), 3 * 4) == 0;
                ok = ok && std::memcmp(plan.data() + 4, start.data(), (size_t) (groups + 1) * 4) == 0;
                const int32_t* pdst = plan.data() + 4 + capx + 1;
                ok = ok && std::memcmp(pdst, dst.data(), (size_t) n * 4) == 0;
                ok = ok && std::memcmp(pdst + capx, tok.data(), (size_t) n * 4) == 0;
                ok = ok && std::memcmp(plan.data() + ptr_off, ptr.data(), (size_t) groups * 8) == 0;
                ok = ok && plan[(size_t) (ptr_off + 4 * capx)] == n;
                const bool stats_ok = std::memcmp(got_stats, want_stats, sizeof want_stats) == 0;
                ++cases;
                if (!ok || !stats_ok) {
                    std::printf("    *** n=%d %s: %s ***\n", n, with_ring ? "mirror+ring" : "mirror",
                                !ok ? "the plan differs from the host replay" : "the stats words differ");
                    if (!stats_ok)
                        std::printf("        stats got %u/%u/%u/%u/%u/%u want %u/%u/%u/%u/%u/%u\n", got_stats[0], got_stats[1],
                                    got_stats[2], got_stats[3], got_stats[4], got_stats[5], want_stats[0], want_stats[1],
                                    want_stats[2], want_stats[3], want_stats[4], want_stats[5]);
                    ++bad;
                }
            }
        }
        k::resident_plan_set_ring(nullptr, k::MirrorRing{}, 0, 0);
        k::resident_plan_set_stats(nullptr, nullptr, 0, 0);
        k::resident_plan_set_mirror(nullptr, nullptr);
        sycl::free(d_mir, q); sycl::free(d_reid, q); sycl::free(d_rbase, q); sycl::free(d_cursor, q);
        sycl::free(d_pairs, q); sycl::free(d_nsel, q); sycl::free(d_stats, q);
    }

    std::printf("resident_plan: %d cases, %d failures\n", cases, bad);
    if (bad) return 1;
    std::printf("resident_plan_parity OK\n");
    return 0;
}
