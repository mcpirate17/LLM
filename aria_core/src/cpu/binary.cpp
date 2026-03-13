#ifndef ARIA_KERNELS_COMMON_H
#include "kernels_common.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── Elementwise binary ────────────────────────────────────────────── */

void aria_add_f32(const float *a, const float *b, float *y, int64_t n) {
#if defined(ARIA_SIMD_WIDTH)
    int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
        aria_simd_ps va = aria_simd_loadu_ps(a + i);
        aria_simd_ps vb = aria_simd_loadu_ps(b + i);
        aria_simd_ps vy = aria_simd_add_ps(va, vb);
        aria_simd_storeu_ps(y + i, vy);
    }
    for (int64_t i = vec_end; i < n; i++) y[i] = a[i] + b[i];
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = a[i] + b[i];
    }
#endif
}

void aria_mul_f32(const float *a, const float *b, float *y, int64_t n) {
#if defined(ARIA_SIMD_WIDTH)
    int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
        aria_simd_ps va = aria_simd_loadu_ps(a + i);
        aria_simd_ps vb = aria_simd_loadu_ps(b + i);
        aria_simd_ps vy = aria_simd_mul_ps(va, vb);
        aria_simd_storeu_ps(y + i, vy);
    }
    for (int64_t i = vec_end; i < n; i++) y[i] = a[i] * b[i];
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = a[i] * b[i];
    }
#endif
}

void aria_sub_f32(const float *a, const float *b, float *y, int64_t n) {
#if defined(ARIA_SIMD_WIDTH)
    int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
        aria_simd_ps va = aria_simd_loadu_ps(a + i);
        aria_simd_ps vb = aria_simd_loadu_ps(b + i);
        aria_simd_ps vy = aria_simd_sub_ps(va, vb);
        aria_simd_storeu_ps(y + i, vy);
    }
    for (int64_t i = vec_end; i < n; i++) y[i] = a[i] - b[i];
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = a[i] - b[i];
    }
#endif
}

void aria_tropical_add_f32(const float *a, const float *b, float *y, int64_t n) {
#if defined(ARIA_SIMD_WIDTH)
    int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
        aria_simd_ps va = aria_simd_loadu_ps(a + i);
        aria_simd_ps vb = aria_simd_loadu_ps(b + i);
        aria_simd_ps vy = aria_simd_min_ps(va, vb);
        aria_simd_storeu_ps(y + i, vy);
    }
    for (int64_t i = vec_end; i < n; i++) y[i] = fminf(a[i], b[i]);
#else
    for (int64_t i = 0; i < n; i++) {
        y[i] = fminf(a[i], b[i]);
    }
#endif
}

void aria_maximum_f32(const float *a, const float *b, float *y, int64_t n) {
#ifdef __AVX2__
    {
        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(y + i, _mm256_max_ps(va, vb));
        }
        for (int64_t i = vec_end; i < n; i++) {
            y[i] = fmaxf(a[i], b[i]);
        }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = fmaxf(a[i], b[i]);
    }
#endif
}

void aria_minimum_f32(const float *a, const float *b, float *y, int64_t n) {
#ifdef __AVX2__
    {
        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(y + i, _mm256_min_ps(va, vb));
        }
        for (int64_t i = vec_end; i < n; i++) {
            y[i] = fminf(a[i], b[i]);
        }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = fminf(a[i], b[i]);
    }
#endif
}

void aria_div_safe_f32(const float *a, const float *b, float *y, int64_t n) {
    static const float EPS = 1e-7f;
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        float denom = b[i];
        if (denom >= 0.0f && denom < EPS) denom = EPS;
        else if (denom < 0.0f && denom > -EPS) denom = -EPS;
        y[i] = a[i] / denom;
    }
}

void aria_outer_product_f32(const float *a, const float *b, float *y, int64_t n) {
    aria_mul_f32(a, b, y, n);
}

#ifdef __cplusplus
}
#endif
