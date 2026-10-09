// SYCL port: experts mirrored in pinned host memory (GgufExpertSource::mirror) in the device-built verify plan.
#pragma once
#include <cstdint>
namespace strata::kernels {
/// From now on (and in every graph captured after this call) the device-built plan (resident_plan) treats an expert
/// that is not in the VRAM cache but has a non-zero entry in `mirror_table` ([n_layers][n_expert] device-readable
/// addresses, 0 = none) as planned: the expert kernels read it from that address, over PCIe. `d_res` is the residency
/// table whose per-layer slices resident_plan receives, so the layer of a call is found from its pointer.
void resident_plan_set_mirror(const int32_t* d_res, const unsigned long long* mirror_table);
/// STRATA_MIRROR_STATS: from now on (and in every graph captured after this call) resident_plan counts, per layer, in
/// `stats` ([n_layers][4] device uint32: routed entries served from a VRAM slot, from the mirror, from neither, and
/// the groups it formed) with relaxed device-scope atomics. `d_res` as for resident_plan_set_mirror. The host reads
/// the buffer back and zeroes it between requests; a null `stats` turns the counting off for later captures.
void resident_plan_set_stats(const int32_t* d_res, uint32_t* stats, long long n_layers, long long n_expert);
}
