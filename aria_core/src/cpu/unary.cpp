#ifndef ARIA_KERNELS_COMMON_H
#include "kernels_common.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── Elementwise unary ─────────────────────────────────────────────── */

void aria_relu_f32(const float *x, float *y, int64_t n) {
#if defined(ARIA_SIMD_WIDTH)
    int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
    aria_simd_ps zero = aria_simd_zero_ps;
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
        aria_simd_ps vx = aria_simd_loadu_ps(x + i);
        aria_simd_ps vy = aria_simd_max_ps(vx, zero);
        aria_simd_storeu_ps(y + i, vy);
    }
    for (int64_t i = vec_end; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
#endif
}

/* GELU with AVX2 SIMD + OpenMP — uses tanh approximation with vectorized exp */
void aria_gelu_f32(const float *x, float *y, int64_t n) {
#if defined(ARIA_SIMD_WIDTH)
    int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
    aria_simd_ps half = aria_simd_set1_ps(0.5f);
    aria_simd_ps one = aria_simd_set1_ps(1.0f);
    aria_simd_ps coeff = aria_simd_set1_ps(GELU_COEFF);
    aria_simd_ps cubic = aria_simd_set1_ps(GELU_CUBIC);
    aria_simd_ps two = aria_simd_set1_ps(2.0f);
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
        aria_simd_ps vx = aria_simd_loadu_ps(x + i);
        /* inner = coeff * (x + cubic * x^3) */
        aria_simd_ps vx2 = aria_simd_mul_ps(vx, vx);
        aria_simd_ps vx3 = aria_simd_mul_ps(vx2, vx);
        aria_simd_ps inner = aria_simd_mul_ps(coeff, aria_simd_fmadd_ps(cubic, vx3, vx));
        /* tanh(x) = 2*sigmoid(2x) - 1 */
        aria_simd_ps two_inner = aria_simd_mul_ps(two, inner);
        aria_simd_ps sig = aria_simd_sigmoid_ps(two_inner);
        aria_simd_ps tanh_val = aria_simd_sub_ps(aria_simd_mul_ps(two, sig), one);
        /* gelu = 0.5 * x * (1 + tanh) */
        aria_simd_ps vy = aria_simd_mul_ps(half, aria_simd_mul_ps(vx, aria_simd_add_ps(one, tanh_val)));
        aria_simd_storeu_ps(y + i, vy);
    }
    for (int64_t i = vec_end; i < n; i++) {
        float v = x[i];
        float v3 = v * v * v;
        y[i] = 0.5f * v * (1.0f + tanhf(GELU_COEFF * (v + GELU_CUBIC * v3)));
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        float v3 = v * v * v;
        y[i] = 0.5f * v * (1.0f + tanhf(GELU_COEFF * (v + GELU_CUBIC * v3)));
    }
#endif
}

void aria_silu_f32(const float *x, float *y, int64_t n) {
#if defined(ARIA_SIMD_WIDTH)
    {
        int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
            aria_simd_ps vx = aria_simd_loadu_ps(x + i);
            aria_simd_ps sig = aria_simd_sigmoid_ps(vx);
            aria_simd_ps vy = aria_simd_mul_ps(vx, sig);
            aria_simd_storeu_ps(y + i, vy);
        }
        for (int64_t i = vec_end; i < n; i++) {
            float v = x[i];
            y[i] = v / (1.0f + expf(-v));
        }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v / (1.0f + expf(-v));
    }
#endif
}

void aria_silu_mul_f32(const float *gate, const float *up, float *y, int64_t n) {
#if defined(ARIA_SIMD_WIDTH)
    {
        int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
            aria_simd_ps vg = aria_simd_loadu_ps(gate + i);
            aria_simd_ps vu = aria_simd_loadu_ps(up + i);
            aria_simd_ps sig = aria_simd_sigmoid_ps(vg);
            aria_simd_ps vy = aria_simd_mul_ps(aria_simd_mul_ps(vg, sig), vu);
            aria_simd_storeu_ps(y + i, vy);
        }
        for (int64_t i = vec_end; i < n; i++) {
            float g = gate[i];
            float s = 1.0f / (1.0f + expf(-g));
            y[i] = (g * s) * up[i];
        }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        float g = gate[i];
        float s = 1.0f / (1.0f + expf(-g));
        y[i] = (g * s) * up[i];
    }
#endif
}

void aria_square_f32(const float *x, float *y, int64_t n) {
#if defined(ARIA_SIMD_WIDTH)
    int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
        aria_simd_ps vx = aria_simd_loadu_ps(x + i);
        aria_simd_storeu_ps(y + i, aria_simd_mul_ps(vx, vx));
    }
    for (int64_t i = vec_end; i < n; i++) { y[i] = x[i] * x[i]; }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) { y[i] = x[i] * x[i]; }
#endif
}

void aria_abs_f32(const float *x, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) { y[i] = fabsf(x[i]); }
}

void aria_neg_f32(const float *x, float *y, int64_t n) {
#if defined(ARIA_SIMD_WIDTH)
    int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
    aria_simd_ps sign_mask = aria_simd_set1_ps(-0.0f);
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
        aria_simd_ps vx = aria_simd_loadu_ps(x + i);
        aria_simd_storeu_ps(y + i, aria_simd_sub_ps(aria_simd_zero_ps, vx));
    }
    for (int64_t i = vec_end; i < n; i++) { y[i] = -x[i]; }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) { y[i] = -x[i]; }
#endif
}

void aria_reciprocal_f32(const float *x, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) { y[i] = 1.0f / x[i]; }
}

void aria_log_f32(const float *x, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) { y[i] = logf(x[i]); }
}

void aria_sqrt_f32(const float *x, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) { y[i] = sqrtf(x[i]); }
}

void aria_sin_f32(const float *x, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) { y[i] = sinf(x[i]); }
}

void aria_cos_f32(const float *x, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) { y[i] = cosf(x[i]); }
}

/* Tanh with AVX2 SIMD: tanh(x) = 2*sigmoid(2x) - 1 */
void aria_tanh_f32(const float *x, float *y, int64_t n) {
#if defined(ARIA_SIMD_WIDTH)
    int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
    aria_simd_ps two = aria_simd_set1_ps(2.0f);
    aria_simd_ps one = aria_simd_set1_ps(1.0f);
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
        aria_simd_ps vx = aria_simd_loadu_ps(x + i);
        aria_simd_ps two_x = aria_simd_mul_ps(two, vx);
        aria_simd_ps sig = aria_simd_sigmoid_ps(two_x);
        aria_simd_ps vy = aria_simd_sub_ps(aria_simd_mul_ps(two, sig), one);
        aria_simd_storeu_ps(y + i, vy);
    }
    for (int64_t i = vec_end; i < n; i++) {
        y[i] = tanhf(x[i]);
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) { y[i] = tanhf(x[i]); }
#endif
}

void aria_sigmoid_f32(const float *x, float *y, int64_t n) {
#if defined(ARIA_SIMD_WIDTH)
    {
        int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
            aria_simd_ps vx = aria_simd_loadu_ps(x + i);
            aria_simd_ps vy = aria_simd_sigmoid_ps(vx);
            aria_simd_storeu_ps(y + i, vy);
        }
        for (int64_t i = vec_end; i < n; i++) {
            y[i] = 1.0f / (1.0f + expf(-x[i]));
        }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
#endif
}

void aria_exp_f32(const float *x, float *y, int64_t n) {
#if defined(ARIA_SIMD_WIDTH)
    {
        int64_t vec_end = n - (n % ARIA_SIMD_WIDTH);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += ARIA_SIMD_WIDTH) {
            aria_simd_ps vx = aria_simd_loadu_ps(x + i);
            aria_simd_ps vy = aria_simd_exp_ps(vx);
            aria_simd_storeu_ps(y + i, vy);
        }
        for (int64_t i = vec_end; i < n; i++) {
            y[i] = expf(x[i]);
        }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) { y[i] = expf(x[i]); }
#endif
}

void aria_sign_ste_f32(const float *x, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        if (x[i] > 0.0f) y[i] = 1.0f;
        else if (x[i] < 0.0f) y[i] = -1.0f;
        else y[i] = 0.0f;
    }
}

#ifdef __cplusplus
}
#endif
