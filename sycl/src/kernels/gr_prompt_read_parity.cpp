// sycl/src/kernels/gr_prompt_read_parity.cpp - the prompt path's hyper-connection read with STRATA_GR_FUSE_READ=1
// (gr_mix_x16: the mix over the BF16 image) against the default pair (gr_norm_rs, then gr_mix_r over the FP32
// residual).  Synthetic prompt shape (n_embd 2560, hc 4), GPU, no model.
//
// The fused read is rounding-level by design (x = R * rs * w carries BF16's 8 mantissa bits instead of FP32's 24), so
// the FP32 mix is not compared bitwise: per element it must sit inside the BF16 rounding bound of its four terms
// (2^-8 of sum_c |x_c sigmoid(g_c)| / 4, plus a sigmoid-precision allowance), and it must match a host model of the
// same arithmetic over the device's own image to 1e-4 of scale.  Everything that is not rounding is bitwise: the
// BF16 image of `mixed` and its low part follow the kernel's rounding rule exactly, and the FP16 image is the RTE
// half of the FP32 value.  Both xn16 token strides (10240 and 10240 + 64) and both low-part modes run.
#ifdef STRATA_SYCL_PROFILING_QUEUES   // D1: the CMake option decides, as in every other translation unit
#define DPCT_PROFILING_ENABLED
#endif
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/gr_fuse_read.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace pf = strata::prefill;

namespace {
constexpr long long N = 2560, HC = 4, D = N * HC, XN_PAD = 64;

void ck(dpct::err0 e, const char* w) { (void) e; (void) w; }
template <typename T>
T* up(const std::vector<T>& h) {
    void* p = nullptr;
    ck(DPCT_CHECK_ERROR(p = (void*) sycl::malloc_device(h.size() * sizeof(T) + 64, dpct::get_in_order_queue())), "malloc");
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(p, h.data(), h.size() * sizeof(T)).wait()), "h2d");
    return (T*) p;
}
template <typename T>
T* dev(size_t n) {
    void* p = nullptr;
    ck(DPCT_CHECK_ERROR(p = (void*) sycl::malloc_device(n * sizeof(T) + 64, dpct::get_in_order_queue())), "malloc");
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memset(p, 0, n * sizeof(T) + 64).wait()), "memset");
    return (T*) p;
}
template <typename T>
std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(DPCT_CHECK_ERROR((dpct::get_current_device().queues_wait_and_throw(), dpct::get_in_order_queue())
                            .memcpy(h.data(), d, n * sizeof(T)).wait()), "d2h");
    return h;
}
template <typename T>
void drop(T* p) { sycl::free(p, dpct::get_in_order_queue()); }

// the kernels' rounding rules, on the host (kernels.dp.cpp: bf, bf_lo, hf)
uint16_t bf(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
float bf16_to_f32(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
uint16_t bf_lo(float f, uint16_t hi) { return bf(f - bf16_to_f32(hi)); }
uint16_t hf(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
float sigm(float x) { return 1.0f / (1.0f + std::exp(-x)); }

struct Case { long long T; bool pad; bool lo; };

// one shape: returns the number of failed checks, prints one line
int run(const Case& cs, std::mt19937& rng) {
    const long long T = cs.T, ldx = cs.pad ? D + XN_PAD : D;
    std::normal_distribution<float> gauss(0.f, 1.f);
    // the residual: the four streams at different scales (gr_multi_parity's recipe), so rs differs per (t, c)
    std::vector<float> R((size_t) T * D), w((size_t) D), g((size_t) T * D);
    for (long long t = 0; t < T; ++t)
        for (long long c = 0; c < HC; ++c)
            for (long long d = 0; d < N; ++d) R[(size_t) (t * D + c * N + d)] = gauss(rng) * (float) (1 << (2 * c));
    for (auto& x : w) x = 1.0f + 0.1f * gauss(rng);
    for (auto& x : g) x = 2.0f * gauss(rng);   // the up projection's output: the gates' logits
    const float eps = 1e-6f;

    float *d_R = up(R), *d_w = up(w), *d_g = up(g);
    float* d_rs = dev<float>((size_t) T * HC);
    uint16_t* d_xn16 = dev<uint16_t>((size_t) T * ldx);
    float *d_mix_ref = dev<float>((size_t) T * N), *d_mix_f = dev<float>((size_t) T * N);
    uint16_t *d_bf_ref = dev<uint16_t>((size_t) T * N), *d_bf_f = dev<uint16_t>((size_t) T * N);
    uint16_t *d_h_ref = dev<uint16_t>((size_t) T * N), *d_h_f = dev<uint16_t>((size_t) T * N);
    uint16_t* d_lo_ref = cs.lo ? dev<uint16_t>((size_t) T * N) : nullptr;
    uint16_t* d_lo_f = cs.lo ? dev<uint16_t>((size_t) T * N) : nullptr;

    // the first pass (shared by both paths), then the default mix and the fused-read mix
    pf::gr_norm_rs(d_R, d_w, eps, d_rs, d_xn16, T, nullptr, nullptr, ldx);
    pf::gr_mix_r(d_R, d_rs, d_w, d_g, d_mix_ref, d_bf_ref, T, nullptr, d_h_ref, d_lo_ref);
    pf::gr_mix_x16(d_xn16, ldx, d_g, d_mix_f, d_bf_f, T, nullptr, d_h_f, d_lo_f);
    ck(DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()), "sync");

    const std::vector<uint16_t> xn16 = down(d_xn16, (size_t) T * ldx);
    const std::vector<float> ref = down(d_mix_ref, (size_t) T * N), got = down(d_mix_f, (size_t) T * N);
    const std::vector<uint16_t> bf_f = down(d_bf_f, (size_t) T * N), h_f = down(d_h_f, (size_t) T * N);
    const std::vector<uint16_t> lo_f = cs.lo ? down(d_lo_f, (size_t) T * N) : std::vector<uint16_t>();

    // scale of the output, for the sigmoid-precision allowance
    double r2 = 0;
    for (float v : ref) r2 += (double) v * v;
    const double scale = std::sqrt(r2 / (double) ref.size());

    long long out_of_bound = 0, off_model = 0, bad_bf = 0, bad_h = 0, bad_lo = 0;
    double e2 = 0, emax = 0;
    for (long long t = 0; t < T; ++t) {
        for (long long d = 0; d < N; ++d) {
            const size_t i = (size_t) (t * N + d);
            // the host model of gr_mix_x16 over the device's own image: fma over c in order, / 4
            float s = 0.0f;
            double terms = 0;
            for (long long c = 0; c < HC; ++c) {
                const float x = bf16_to_f32(xn16[(size_t) (t * ldx + c * N + d)]);
                const float sg = sigm(g[(size_t) (t * D + c * N + d)]);
                s = std::fma(x, sg, s);
                terms += std::fabs((double) x * sg);
            }
            s /= (float) HC;
            const double dd = (double) got[i] - ref[i];
            e2 += dd * dd;
            emax = std::max(emax, std::fabs(dd));
            // BF16 rounding of each x_c: |x_bf - x| <= 2^-8 |x|; the mean of the four terms carries at most that share
            const double bound = std::ldexp(terms / (double) HC, -8) + 1e-4 * (std::fabs((double) ref[i]) + scale);
            if (std::fabs(dd) > bound) ++out_of_bound;
            // the device's native sigmoid against std::exp: 1e-4 of scale is generous; a layout slip is 1e-1 and more
            if (std::fabs((double) got[i] - s) > 1e-4 * (std::fabs((double) s) + scale)) ++off_model;
            // the images: the kernel's own rounding rules of its FP32 value, bitwise
            const uint16_t hb = bf(got[i]);
            if (bf_f[i] != hb) ++bad_bf;
            if (h_f[i] != hf(got[i])) ++bad_h;
            if (cs.lo && lo_f[i] != bf_lo(got[i], hb)) ++bad_lo;
        }
    }
    const double rel_rms = std::sqrt(e2 / (double) ref.size()) / std::max(scale, 1e-30);
    const int fails = (out_of_bound ? 1 : 0) + (off_model ? 1 : 0) + (bad_bf ? 1 : 0) + (bad_h ? 1 : 0) + (bad_lo ? 1 : 0);
    std::printf("T %3lld ldx %5lld lo %d: fused vs default rel RMS %.3e, max |diff| %.3e (scale %.3e); out of bound %lld, "
                "off model %lld, bf16 image %lld, fp16 image %lld, low part %lld: %s\n",
                T, ldx, cs.lo ? 1 : 0, rel_rms, emax, scale, out_of_bound, off_model, bad_bf, bad_h, bad_lo,
                fails ? "FAIL" : "ok");

    drop(d_R); drop(d_w); drop(d_g); drop(d_rs); drop(d_xn16);
    drop(d_mix_ref); drop(d_mix_f); drop(d_bf_ref); drop(d_bf_f); drop(d_h_ref); drop(d_h_f);
    if (d_lo_ref) drop(d_lo_ref);
    if (d_lo_f) drop(d_lo_f);
    return fails;
}
}  // namespace

int main(int, char**) {   // --selftest or no argument: the same run
    std::mt19937 rng(20261009);
    const Case cases[] = {{1, false, false}, {7, false, false}, {7, true, false}, {64, false, false},
                          {64, true, false}, {64, false, true}, {64, true, true}, {333, true, false}};
    int fails = 0;
    for (const Case& c : cases) fails += run(c, rng);
    std::printf("gr_prompt_read_parity: %d failure(s)\n", fails);
    return fails ? 1 : 0;
}
