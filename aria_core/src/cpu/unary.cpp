#ifndef ARIA_KERNELS_COMMON_H
#include "kernels_common.h"
#endif
#include "simd_elementwise.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Elementwise unary ─────────────────────────────────────────────── */

void aria_relu_f32(const float *x, float *y, int64_t n) {
    ARIA_EW_UNARY(x, y, n, (AriaOpRelu::vec), (AriaOpRelu::scalar));
}

void aria_gelu_f32(const float *x, float *y, int64_t n) {
    ARIA_EW_UNARY(x, y, n, (AriaOpGelu::vec), (AriaOpGelu::scalar));
}

void aria_silu_f32(const float *x, float *y, int64_t n) {
    ARIA_EW_UNARY(x, y, n, (AriaOpSilu::vec), (AriaOpSilu::scalar));
}

void aria_silu_mul_f32(const float *gate, const float *up, float *y, int64_t n) {
    ARIA_EW_BINARY(gate, up, y, n,
        ([](aria_simd_ps vg, aria_simd_ps vu) {
            aria_simd_ps sig = aria_simd_sigmoid_ps(vg);
            return aria_simd_mul_ps(aria_simd_mul_ps(vg, sig), vu);
        }),
        ([](float g, float u) {
            float s = 1.0f / (1.0f + expf(-g));
            return (g * s) * u;
        }));
}

void aria_square_f32(const float *x, float *y, int64_t n) {
    ARIA_EW_UNARY(x, y, n,
        ([](aria_simd_ps vx) { return aria_simd_mul_ps(vx, vx); }),
        ([](float v) { return v * v; }));
}

void aria_abs_f32(const float *x, float *y, int64_t n) {
    ARIA_EW_UNARY(x, y, n,
        ([](aria_simd_ps vx) {
            /* clear sign bit via AND with 0x7FFFFFFF */
            return aria_simd_and_ps(
                vx, aria_simd_castsi_ps(aria_simd_set1_epi32(0x7FFFFFFF)));
        }),
        ([](float v) { return fabsf(v); }));
}

void aria_neg_f32(const float *x, float *y, int64_t n) {
    ARIA_EW_UNARY(x, y, n,
        ([](aria_simd_ps vx) { return aria_simd_sub_ps(aria_simd_zero_ps, vx); }),
        ([](float v) { return -v; }));
}

void aria_reciprocal_f32(const float *x, float *y, int64_t n) {
    aria_ew_unary_scalar_f32(x, y, n, [](float v) { return 1.0f / v; });
}

void aria_log_f32(const float *x, float *y, int64_t n) {
    aria_ew_unary_scalar_f32(x, y, n, [](float v) { return logf(v); });
}

void aria_sqrt_f32(const float *x, float *y, int64_t n) {
    aria_ew_unary_scalar_f32(x, y, n, [](float v) { return sqrtf(v); });
}

void aria_sin_f32(const float *x, float *y, int64_t n) {
    aria_ew_unary_scalar_f32(x, y, n, [](float v) { return sinf(v); });
}

void aria_cos_f32(const float *x, float *y, int64_t n) {
    aria_ew_unary_scalar_f32(x, y, n, [](float v) { return cosf(v); });
}

/* tanh(x) = 2*sigmoid(2x) - 1 in the SIMD body */
void aria_tanh_f32(const float *x, float *y, int64_t n) {
    ARIA_EW_UNARY(x, y, n,
        ([](aria_simd_ps vx) {
            aria_simd_ps two = aria_simd_set1_ps(2.0f);
            aria_simd_ps one = aria_simd_set1_ps(1.0f);
            aria_simd_ps sig = aria_simd_sigmoid_ps(aria_simd_mul_ps(two, vx));
            return aria_simd_sub_ps(aria_simd_mul_ps(two, sig), one);
        }),
        ([](float v) { return tanhf(v); }));
}

void aria_sigmoid_f32(const float *x, float *y, int64_t n) {
    ARIA_EW_UNARY(x, y, n, (AriaOpSigmoid::vec), (AriaOpSigmoid::scalar));
}

void aria_exp_f32(const float *x, float *y, int64_t n) {
    ARIA_EW_UNARY(x, y, n,
        ([](aria_simd_ps vx) { return aria_simd_exp_ps(vx); }),
        ([](float v) { return expf(v); }));
}

void aria_sign_ste_f32(const float *x, float *y, int64_t n) {
    aria_ew_unary_scalar_f32(x, y, n, [](float v) {
        if (v > 0.0f) return 1.0f;
        if (v < 0.0f) return -1.0f;
        return 0.0f;
    });
}

#ifdef __cplusplus
}
#endif
