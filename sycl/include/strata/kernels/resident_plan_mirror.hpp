// SYCL port: experts mirrored in pinned host memory (GgufExpertSource::mirror) in the device-built verify plan.
#pragma once
#include <cstdint>
namespace strata::kernels {
/// From now on (and in every graph captured after this call) the device-built plan (resident_plan) treats an expert
/// that is not in the VRAM cache but has a non-zero entry in `mirror_table` ([n_layers][n_expert] device-readable
/// addresses, 0 = none) as planned: the expert kernels read it from that address, over PCIe. `d_res` is the residency
/// table whose per-layer slices resident_plan receives, so the layer of a call is found from its pointer.
void resident_plan_set_mirror(const int32_t* d_res, const unsigned long long* mirror_table);
/// STRATA_MIRROR_STATS: the counter words per layer in the `stats` buffer of resident_plan_set_stats:
/// [0] routed entries served from a VRAM slot, [1] from the mirror (a ring hit counts here too: the expert is
/// mirrored), [2] from neither, [3] the groups the plan formed, [4] of the mirrored entries, those pointed at a
/// prefetch ring slot (T2a), [5] experts the prefetch copied into the ring (T2a). Entries served over PCIe = [1] - [4].
constexpr int kResidentPlanStatWords = 6;
/// STRATA_MIRROR_STATS: from now on (and in every graph captured after this call) resident_plan counts, per layer, in
/// `stats` ([n_layers][kResidentPlanStatWords] device uint32, the words above) with relaxed device-scope atomics.
/// `d_res` as for resident_plan_set_mirror. The host reads the buffer back and zeroes it between requests; a null
/// `stats` turns the counting off for later captures.
void resident_plan_set_stats(const int32_t* d_res, uint32_t* stats, long long n_layers, long long n_expert);

// ---- T2a (STRATA_PF_SLOTS=S): a per-layer ring of S VRAM slots that holds predicted mirror misses one layer ahead.
/// The ring's device tables, all allocated by the caller (generate.cpp) before any graph is captured; the pointers are
/// baked into the captured graphs. Per layer `slots` entries: `eid[l * slots + s]` the expert slot s holds (-1 empty),
/// `base[l * slots + s]` the slot's device address (fixed), `cursor[l]` the next slot to refill, and the scratch the
/// select kernel hands the copy kernel: `pairs[l * 2 * slots + 2 * i]` = source (mirror) address, `+ 1` = the slot
/// address, `n_sel[l]` how many pairs. Everything is read and written on one in-order queue, so one scratch per layer is
/// enough (a side-queue form would still be per layer).
struct MirrorRing {
    int slots = 0;                        ///< S (1..kMirrorRingMaxSlots); 0 = no ring
    int32_t* eid = nullptr;               ///< [n_layers][S]
    unsigned long long* base = nullptr;   ///< [n_layers][S]
    uint32_t* cursor = nullptr;           ///< [n_layers]
    unsigned long long* pairs = nullptr;  ///< [n_layers][2 * S]
    int32_t* n_sel = nullptr;             ///< [n_layers]
};
/// The plan kernel compares each mirrored entry against S ring entries and the select kernel's one-thread slot pass is
/// S + n long: S is capped so neither grows past a few hundred operations.
constexpr int kMirrorRingMaxSlots = 32;
/// From now on (and in every graph captured after this call) resident_plan points a mirrored expert that sits in its
/// layer's ring slot at that slot instead of the mirror (nullptr tables / slots 0: today's behaviour, byte-identical),
/// and mirror_prefetch finds its layer's tables from `res_layer`. `d_res` as for resident_plan_set_mirror.
void resident_plan_set_ring(const int32_t* d_res, const MirrorRing& ring, long long n_layers, long long n_expert);
/// The S registered by resident_plan_set_ring (0: none): the verifier captures the prediction only when a ring exists.
int mirror_ring_slots();
/// Inside the captured window, after layer l's routing: predict layer l + 1's mirror misses from `pred_ids` (layer
/// l + 1's router run on layer l's MoE input; n ids, duplicates allowed, n <= kResidentPlanMax) and copy them into layer
/// l + 1's ring. `res_layer` is layer l + 1's residency slice (its layer index, and so its mirror and ring slices, is
/// found from the pointer as resident_plan does), `blob_bytes` that layer's expert blob size (a multiple of 16). Two
/// kernels on `stream`: select (dedupe, keep `res < 0 && mir != 0` and not already in the ring, at most S - protected,
/// slots round-robin from the layer's cursor skipping slots whose expert is predicted this round) and copy (16-byte
/// grid-stride loads from host USM, as fetch_blobs). No-op when no ring or mirror is registered.
void mirror_prefetch(const int32_t* pred_ids, int n, const int32_t* res_layer, int n_expert, long long blob_bytes,
                     void* stream);
/// The same two kernels on explicit per-layer tables (mirror_prefetch_parity): `ring_eid` / `ring_base` are one
/// layer's S entries, `cursor`, `pairs` (2 * S words) and `n_sel` that layer's words, `stats` that layer's
/// kResidentPlanStatWords counters or nullptr.
void mirror_prefetch_tables(const int32_t* pred_ids, int n, const int32_t* res, const unsigned long long* mir,
                            int n_expert, int32_t* ring_eid, const unsigned long long* ring_base, uint32_t* cursor,
                            unsigned long long* pairs, int32_t* n_sel, int slots, long long blob_bytes, uint32_t* stats,
                            void* stream);
}
