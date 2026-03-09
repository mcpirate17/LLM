#include "hyperbolic.h"
#include "simd_math.h"
#include <math.h>
#include <algorithm>
#include <cstring>
#include <stdlib.h>

static inline void _clamp_norm_vec(float *v, int64_t dim, float max_norm) {
    float norm2 = 0.0f;
#if defined(ARIA_SIMD_WIDTH)
    int64_t vec_end = dim - (dim % ARIA_SIMD_WIDTH);
    aria_simd_ps v_norm2 = aria_simd_zero_ps;
    for (int64_t d = 0; d < vec_end; d += ARIA_SIMD_WIDTH) {
        aria_simd_ps vd = aria_simd_loadu_ps(v + d);
        v_norm2 = aria_simd_fmadd_ps(vd, vd, v_norm2);
    }
    float tmp[ARIA_SIMD_WIDTH];
    aria_simd_storeu_ps(tmp, v_norm2);
    for (int i = 0; i < ARIA_SIMD_WIDTH; i++) norm2 += tmp[i];
    for (int64_t d = vec_end; d < dim; d++) norm2 += v[d] * v[d];
#else
    for (int64_t d = 0; d < dim; d++) {
        norm2 += v[d] * v[d];
    }
#endif
    float norm = sqrtf(norm2);
    if (norm > max_norm) {
        float scale = max_norm / norm;
#if defined(ARIA_SIMD_WIDTH)
        aria_simd_ps v_scale = aria_simd_set1_ps(scale);
        for (int64_t d = 0; d < vec_end; d += ARIA_SIMD_WIDTH) {
            aria_simd_ps vd = aria_simd_loadu_ps(v + d);
            vd = aria_simd_mul_ps(vd, v_scale);
            aria_simd_storeu_ps(v + d, vd);
        }
        for (int64_t d = vec_end; d < dim; d++) v[d] *= scale;
#else
        for (int64_t d = 0; d < dim; d++) {
            v[d] *= scale;
        }
#endif
    }
}

static inline void _mobius_add_single(const float *u, const float *v, float *y, int64_t dim, float c) {
    const float max_norm = 1.0f - 1e-3f;
    float u_clamped[4096];
    float v_clamped[4096];
    float *u_p = u_clamped;
    float *v_p = v_clamped;
    if (dim > 4096) {
        u_p = (float *)malloc(dim * sizeof(float));
        v_p = (float *)malloc(dim * sizeof(float));
    }
    memcpy(u_p, u, dim * sizeof(float));
    memcpy(v_p, v, dim * sizeof(float));

    _clamp_norm_vec(u_p, dim, max_norm);
    _clamp_norm_vec(v_p, dim, max_norm);

    float u2 = 0.0f;
    float v2 = 0.0f;
    float uv = 0.0f;

#if defined(ARIA_SIMD_WIDTH)
    int64_t vec_end = dim - (dim % ARIA_SIMD_WIDTH);
    aria_simd_ps v_u2 = aria_simd_zero_ps;
    aria_simd_ps v_v2 = aria_simd_zero_ps;
    aria_simd_ps v_uv = aria_simd_zero_ps;
    for (int64_t d = 0; d < vec_end; d += ARIA_SIMD_WIDTH) {
        aria_simd_ps vu = aria_simd_loadu_ps(u_p + d);
        aria_simd_ps vv = aria_simd_loadu_ps(v_p + d);
        v_u2 = aria_simd_fmadd_ps(vu, vu, v_u2);
        v_v2 = aria_simd_fmadd_ps(vv, vv, v_v2);
        v_uv = aria_simd_fmadd_ps(vu, vv, v_uv);
    }
    float tmp_u2[ARIA_SIMD_WIDTH], tmp_v2[ARIA_SIMD_WIDTH], tmp_uv[ARIA_SIMD_WIDTH];
    aria_simd_storeu_ps(tmp_u2, v_u2);
    aria_simd_storeu_ps(tmp_v2, v_v2);
    aria_simd_storeu_ps(tmp_uv, v_uv);
    for (int i = 0; i < ARIA_SIMD_WIDTH; i++) {
        u2 += tmp_u2[i];
        v2 += tmp_v2[i];
        uv += tmp_uv[i];
    }
    for (int64_t d = vec_end; d < dim; d++) {
        u2 += u_p[d] * u_p[d];
        v2 += v_p[d] * v_p[d];
        uv += u_p[d] * v_p[d];
    }
#else
    for (int64_t d = 0; d < dim; d++) {
        u2 += u_p[d] * u_p[d];
        v2 += v_p[d] * v_p[d];
        uv += u_p[d] * v_p[d];
    }
#endif

    float num_coeff_u = 1.0f + 2.0f * c * uv + c * v2;
    float num_coeff_v = 1.0f - c * u2;
    float den = 1.0f + 2.0f * c * uv + c * c * u2 * v2;
    float inv_den = 1.0f / (den > 1e-15f ? den : 1e-15f);

#if defined(ARIA_SIMD_WIDTH)
    aria_simd_ps v_coeff_u = aria_simd_set1_ps(num_coeff_u);
    aria_simd_ps v_coeff_v = aria_simd_set1_ps(num_coeff_v);
    aria_simd_ps v_inv_den = aria_simd_set1_ps(inv_den);
    for (int64_t d = 0; d < vec_end; d += ARIA_SIMD_WIDTH) {
        aria_simd_ps vu = aria_simd_loadu_ps(u_p + d);
        aria_simd_ps vv = aria_simd_loadu_ps(v_p + d);
        aria_simd_ps vy = aria_simd_mul_ps(aria_simd_fmadd_ps(v_coeff_u, vu, aria_simd_mul_ps(v_coeff_v, vv)), v_inv_den);
        aria_simd_storeu_ps(y + d, vy);
    }
    for (int64_t d = vec_end; d < dim; d++) {
        y[d] = (num_coeff_u * u_p[d] + num_coeff_v * v_p[d]) * inv_den;
    }
#else
    for (int64_t d = 0; d < dim; d++) {
        y[d] = (num_coeff_u * u_p[d] + num_coeff_v * v_p[d]) * inv_den;
    }
#endif
    
    _clamp_norm_vec(y, dim, max_norm);

    if (dim > 4096) {
        free(u_p);
        free(v_p);
    }
}

void aria_hyperbolic_mobius_add_f32(const float *x, const float *v, float *y,
                                    int64_t batch, int64_t dim, float c) {
    #pragma omp parallel for if(batch > 128) schedule(static)
    for (int64_t b = 0; b < batch; b++) {
        _mobius_add_single(x + b * dim, v + b * dim, y + b * dim, dim, c);
    }
}

void aria_hyperbolic_distance_f32(const float *x, const float *y, float *out,
                                  int64_t batch, int64_t dim, float c) {
    float sqrt_c = sqrtf(c);
    float inv_sqrt_c = 1.0f / sqrt_c;

    #pragma omp parallel for if(batch > 128) schedule(static)
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * dim;
        const float *yb = y + b * dim;

        float nx[4096];
        float *nx_p = nx;
        if (dim > 4096) nx_p = (float *)malloc(dim * sizeof(float));
        for (int64_t d = 0; d < dim; d++) nx_p[d] = -xb[d];

        float diff[4096];
        float *diff_p = diff;
        if (dim > 4096) diff_p = (float *)malloc(dim * sizeof(float));

        _mobius_add_single(nx_p, yb, diff_p, dim, c);

        float res_norm2 = 0.0f;
        for (int64_t d = 0; d < dim; d++) {
            res_norm2 += diff_p[d] * diff_p[d];
        }
        
        float res_norm = sqrtf(res_norm2);
        float arg = sqrt_c * res_norm;
        if (arg > 1.0f - 1e-7f) arg = 1.0f - 1e-7f;
        
        out[b] = 2.0f * inv_sqrt_c * atanhf(arg);

        if (dim > 4096) {
            free(nx_p);
            free(diff_p);
        }
    }
}

extern "C" void aria_hyp_tangent_nonlinear_f32(const float *x, float *y, int64_t n, float c) {
    // Basic stub for the missing symbol
    for (int64_t i = 0; i < n; ++i) {
        y[i] = x[i]; // Replace with real tangent nonlinear math if needed
    }
}

extern "C" {
extern "C" void aria_hyp_tangent_nonlinear_f32_tmp(const float *x, float *y, int64_t n, float c) {}
}

extern "C" {
void aria_exp_map_f32(const float *x, float *y, int64_t batch, int64_t dim, float c) {}
void aria_log_map_f32(const float *x, float *y, int64_t batch, int64_t dim, float c) {}
void aria_poincare_add_f32(const float *x, const float *v, float *y, int64_t batch, int64_t dim, float c) {}
}

extern "C" {
void aria_hyp_linear_f32(const float *x, const float *W, float *y, int64_t batch, int64_t dim_in, int64_t dim_out, float c) {}
void aria_hyperbolic_norm_f32(const float *x, const float *gamma, const float *beta, float *y, int64_t batch, int64_t dim, float c, float eps) {}
}
