#ifndef ARIA_KERNELS_COMMON_H
#include "kernels_common.h"
#endif
#include "simd_elementwise.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Elementwise binary ────────────────────────────────────────────── */

void aria_add_f32(const float *a, const float *b, float *y, int64_t n) {
    ARIA_EW_BINARY(a, b, y, n, (AriaOpAdd::vec), (AriaOpAdd::scalar));
}

void aria_mul_f32(const float *a, const float *b, float *y, int64_t n) {
    ARIA_EW_BINARY(a, b, y, n, (AriaOpMul::vec), (AriaOpMul::scalar));
}

void aria_sub_f32(const float *a, const float *b, float *y, int64_t n) {
    ARIA_EW_BINARY(a, b, y, n,
        ([](aria_simd_ps va, aria_simd_ps vb) { return aria_simd_sub_ps(va, vb); }),
        ([](float av, float bv) { return av - bv; }));
}

void aria_tropical_add_f32(const float *a, const float *b, float *y, int64_t n) {
    ARIA_EW_BINARY(a, b, y, n,
        ([](aria_simd_ps va, aria_simd_ps vb) { return aria_simd_min_ps(va, vb); }),
        ([](float av, float bv) { return fminf(av, bv); }));
}

void aria_maximum_f32(const float *a, const float *b, float *y, int64_t n) {
    ARIA_EW_BINARY(a, b, y, n,
        ([](aria_simd_ps va, aria_simd_ps vb) { return aria_simd_max_ps(va, vb); }),
        ([](float av, float bv) { return fmaxf(av, bv); }));
}

void aria_minimum_f32(const float *a, const float *b, float *y, int64_t n) {
    ARIA_EW_BINARY(a, b, y, n,
        ([](aria_simd_ps va, aria_simd_ps vb) { return aria_simd_min_ps(va, vb); }),
        ([](float av, float bv) { return fminf(av, bv); }));
}

void aria_div_safe_f32(const float *a, const float *b, float *y, int64_t n) {
    aria_ew_binary_scalar_f32(a, b, y, n, [](float av, float bv) {
        const float EPS = 1e-7f;
        float denom = bv;
        if (denom >= 0.0f && denom < EPS) denom = EPS;
        else if (denom < 0.0f && denom > -EPS) denom = -EPS;
        return av / denom;
    });
}

void aria_outer_product_f32(const float *a, const float *b, float *y, int64_t n) {
    aria_mul_f32(a, b, y, n);
}

#ifdef __cplusplus
}
#endif
