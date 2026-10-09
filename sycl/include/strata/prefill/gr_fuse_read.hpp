// sycl/include/strata/prefill/gr_fuse_read.hpp - the SYCL port's STRATA_GR_FUSE_READ=1 kernel of the prompt path's
// hyper-connection read (the upstream header include/strata/prefill/kernels.hpp is not edited; see docs/INTEL.md).
//
// The read is two passes over the FP32 residual R (T x 10240 floats): gr_norm_rs reads it for the row scales and the
// BF16 image the three GEMMs consume, and gr_mix_r reads it again for x = R * rs * w, which it mixes with the up
// projection's output (sum_c x_c * sigmoid(gated_c) / 4).  The mix needs every element of x in FP32 and the GEMM
// output, so no single kernel exists; an FP32 side buffer of x is STRATA_GR_UNFUSED=1 (gr_norm + gr_mix: more
// bytes, not fewer).  What halves the second pass is reading x from the BF16 image gr_norm_rs already wrote:
// gr_mix_x16.  Rounding-level (x with 8 mantissa bits, as the down and inject projections see it), the same
// summation order and output images as gr_mix_r.
#pragma once

#include <cstdint>

namespace strata::prefill {

/// mixed[t, d] = mean_c bf16(xn16[t*ldx + c*2560 + d]) * sigmoid(gated[t, c, d]); FP32, BF16 (+ its low part) and FP16
/// images as gr_mix_r writes them.  xn16: gr_norm_rs's image with token stride ldx (0 = 10240).
void gr_mix_x16(const uint16_t* xn16, int64_t ldx, const float* gated, float* mixed, uint16_t* mixed16, int64_t T,
                void* stream, uint16_t* mixed_h = nullptr, uint16_t* mixed16_lo = nullptr);

}  // namespace strata::prefill
