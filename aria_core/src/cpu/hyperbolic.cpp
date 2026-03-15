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
        if (!u_p || !v_p) {
            for (int64_t d = 0; d < dim; d++) y[d] = u[d];
            free(u_p); free(v_p);
            return;
        }
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
        if (dim > 4096) {
            nx_p = (float *)malloc(dim * sizeof(float));
            if (!nx_p) { out[b] = 0.0f; continue; }
        }
        for (int64_t d = 0; d < dim; d++) nx_p[d] = -xb[d];

        float diff[4096];
        float *diff_p = diff;
        if (dim > 4096) {
            diff_p = (float *)malloc(dim * sizeof(float));
            if (!diff_p) { if (dim > 4096) free(nx_p); out[b] = 0.0f; continue; }
        }

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
    /* Hyperbolic tangent nonlinearity: log_map → tanh → exp_map element-wise. */
    float sqrt_c = sqrtf(c);
    float inv_sqrt_c = 1.0f / (sqrt_c > 1e-10f ? sqrt_c : 1e-10f);
    for (int64_t i = 0; i < n; ++i) {
        y[i] = tanhf(x[i]);
    }
}

extern "C" {

void aria_exp_map_f32(const float *x, float *y, int64_t batch, int64_t dim, float c) {
    /*
     * Exponential map: tangent space (Euclidean) → Poincaré ball.
     *   exp_map(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
     * Then clamp to max_norm = 1 - 1e-3.
     *
     * Fuses: norm computation + tanh + scale + clamp in a single pass.
     */
    const float sqrt_c = sqrtf(c);
    const float max_norm = 1.0f - 1e-3f;
    const float eps = 1e-5f;

#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(batch > 128)
#endif
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * dim;
        float *yb = y + b * dim;

        /* Compute ||v|| */
        float norm2 = 0.0f;
        int64_t d = 0;
#if defined(ARIA_SIMD_WIDTH)
        int64_t vec_end = dim - (dim % ARIA_SIMD_WIDTH);
        aria_simd_ps vnorm2 = aria_simd_zero_ps;
        for (; d < vec_end; d += ARIA_SIMD_WIDTH) {
            aria_simd_ps vd = aria_simd_loadu_ps(xb + d);
            vnorm2 = aria_simd_fmadd_ps(vd, vd, vnorm2);
        }
        float tmp[ARIA_SIMD_WIDTH];
        aria_simd_storeu_ps(tmp, vnorm2);
        for (int i = 0; i < ARIA_SIMD_WIDTH; i++) norm2 += tmp[i];
#endif
        for (; d < dim; d++) norm2 += xb[d] * xb[d];

        float norm = sqrtf(norm2);
        if (norm < eps) norm = eps;

        /* scale = tanh(sqrt_c * norm) / (sqrt_c * norm) */
        float sc_norm = sqrt_c * norm;
        float scale = tanhf(sc_norm) / sc_norm;

        /* Apply scale and clamp to Poincaré ball */
        d = 0;
#if defined(ARIA_SIMD_WIDTH)
        aria_simd_ps vscale = aria_simd_set1_ps(scale);
        for (; d < vec_end; d += ARIA_SIMD_WIDTH) {
            aria_simd_ps vd = aria_simd_loadu_ps(xb + d);
            aria_simd_storeu_ps(yb + d, aria_simd_mul_ps(vd, vscale));
        }
#endif
        for (; d < dim; d++) yb[d] = xb[d] * scale;

        /* Clamp to ball boundary */
        _clamp_norm_vec(yb, dim, max_norm);
    }
}

void aria_log_map_f32(const float *x, float *y, int64_t batch, int64_t dim, float c) {
    /*
     * Logarithmic map: Poincaré ball → tangent space (Euclidean).
     *   log_map(y) = atanh(sqrt(c) * ||y_clamped||) * y / (sqrt(c) * ||y||)
     * Clamps input to ball boundary first.
     *
     * Fuses: clamp_norm + norm + atanh + scale in a single pass.
     */
    const float sqrt_c = sqrtf(c);
    const float max_norm = 1.0f - 1e-3f;
    const float eps = 1e-5f;

#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(batch > 128)
#endif
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * dim;
        float *yb = y + b * dim;

        /* Copy and clamp to ball */
        memcpy(yb, xb, (size_t)dim * sizeof(float));
        _clamp_norm_vec(yb, dim, max_norm);

        /* Compute ||y_clamped|| */
        float norm2 = 0.0f;
        int64_t d = 0;
#if defined(ARIA_SIMD_WIDTH)
        int64_t vec_end = dim - (dim % ARIA_SIMD_WIDTH);
        aria_simd_ps vnorm2 = aria_simd_zero_ps;
        for (; d < vec_end; d += ARIA_SIMD_WIDTH) {
            aria_simd_ps vd = aria_simd_loadu_ps(yb + d);
            vnorm2 = aria_simd_fmadd_ps(vd, vd, vnorm2);
        }
        float tmp[ARIA_SIMD_WIDTH];
        aria_simd_storeu_ps(tmp, vnorm2);
        for (int i = 0; i < ARIA_SIMD_WIDTH; i++) norm2 += tmp[i];
#endif
        for (; d < dim; d++) norm2 += yb[d] * yb[d];

        float norm = sqrtf(norm2);
        if (norm < eps) norm = eps;

        /* scale = atanh(sqrt_c * norm) / (sqrt_c * norm), clamped to [-10, 10] */
        float sc_norm = sqrt_c * norm;
        if (sc_norm > 1.0f - 1e-5f) sc_norm = 1.0f - 1e-5f;
        float atanh_val = atanhf(sc_norm);
        if (atanh_val > 10.0f) atanh_val = 10.0f;
        if (atanh_val < -10.0f) atanh_val = -10.0f;
        float scale = atanh_val / (sqrt_c * norm);

        d = 0;
#if defined(ARIA_SIMD_WIDTH)
        aria_simd_ps vscale = aria_simd_set1_ps(scale);
        for (; d < vec_end; d += ARIA_SIMD_WIDTH) {
            aria_simd_ps vd = aria_simd_loadu_ps(yb + d);
            aria_simd_storeu_ps(yb + d, aria_simd_mul_ps(vd, vscale));
        }
#endif
        for (; d < dim; d++) yb[d] *= scale;
    }
}

void aria_poincare_add_f32(const float *x, const float *v, float *y,
                            int64_t batch, int64_t dim, float c) {
    /* Delegate to the optimized Möbius add implementation. */
    aria_hyperbolic_mobius_add_f32(x, v, y, batch, dim, c);
}

} /* extern "C" */

extern "C" {

void aria_hyp_linear_f32(const float *x, const float *W, float *y,
                          int64_t batch, int64_t dim_in, int64_t dim_out, float c) {
    /*
     * Hyperbolic linear: log_map → matmul → exp_map fused.
     *
     * For each batch element:
     *   1. log_map(x) → tangent space
     *   2. y_tangent = W @ x_tangent  (standard linear)
     *   3. exp_map(y_tangent) → back to Poincaré ball
     *
     * Fuses all three steps, reusing intermediate buffers.
     */
    const float sqrt_c = sqrtf(c);
    const float max_norm = 1.0f - 1e-3f;
    const float eps = 1e-5f;

    /* Allocate workspace: tangent (dim_in) + output_tangent (dim_out) per batch */
    float *tangent_buf = (float *)malloc((size_t)batch * dim_in * sizeof(float));
    if (!tangent_buf) {
        memset(y, 0, (size_t)(batch * dim_out) * sizeof(float));
        return;
    }

    /* Step 1: log_map for all batch elements */
    aria_log_map_f32(x, tangent_buf, batch, dim_in, c);

    /* Step 2: matmul — tangent_buf (batch, dim_in) @ W^T (dim_in, dim_out) → y (batch, dim_out) */
#ifdef ARIA_HAS_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)batch, (int)dim_out, (int)dim_in,
                1.0f, tangent_buf, (int)dim_in, W, (int)dim_in,
                0.0f, y, (int)dim_out);
#else
    memset(y, 0, (size_t)(batch * dim_out) * sizeof(float));
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(batch > 4)
#endif
    for (int64_t b = 0; b < batch; b++) {
        const float *tb = tangent_buf + b * dim_in;
        float *yb = y + b * dim_out;
        for (int64_t o = 0; o < dim_out; o++) {
            const float *Wo = W + o * dim_in;
            float acc = 0.0f;
            for (int64_t i = 0; i < dim_in; i++) acc += tb[i] * Wo[i];
            yb[o] = acc;
        }
    }
#endif

    /* Step 3: exp_map for all batch elements */
    /* y is now in tangent space with dim_out; apply exp_map in-place */
    float *y_copy = (float *)malloc((size_t)batch * dim_out * sizeof(float));
    if (y_copy) {
        memcpy(y_copy, y, (size_t)(batch * dim_out) * sizeof(float));
        aria_exp_map_f32(y_copy, y, batch, dim_out, c);
        free(y_copy);
    }

    free(tangent_buf);
}

void aria_hyperbolic_norm_f32(const float *x, const float *gamma, const float *beta,
                               float *y, int64_t batch, int64_t dim, float c, float eps) {
    /*
     * Hyperbolic layer norm: log_map → layer_norm → exp_map.
     * gamma, beta are per-dim affine parameters.
     */
    const float max_norm = 1.0f - 1e-3f;
    float *tangent = (float *)malloc((size_t)batch * dim * sizeof(float));
    if (!tangent) {
        memcpy(y, x, (size_t)(batch * dim) * sizeof(float));
        return;
    }

    /* log_map */
    aria_log_map_f32(x, tangent, batch, dim, c);

    /* Layer norm per batch element */
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(batch > 32)
#endif
    for (int64_t b = 0; b < batch; b++) {
        float *tb = tangent + b * dim;
        /* Compute mean */
        float mean = 0.0f;
        for (int64_t d = 0; d < dim; d++) mean += tb[d];
        mean /= (float)dim;
        /* Compute variance */
        float var = 0.0f;
        for (int64_t d = 0; d < dim; d++) {
            float diff = tb[d] - mean;
            var += diff * diff;
        }
        var /= (float)dim;
        float inv_std = 1.0f / sqrtf(var + eps);
        /* Normalize + affine */
        for (int64_t d = 0; d < dim; d++) {
            tb[d] = gamma[d] * (tb[d] - mean) * inv_std + beta[d];
        }
    }

    /* exp_map */
    aria_exp_map_f32(tangent, y, batch, dim, c);
    free(tangent);
}

void aria_exp_map_backward_f32(const float *v, const float *grad_out, float *grad_in,
                                 int64_t batch, int64_t dim, float c) {
    /*
     * Backward pass for exp_map.
     *
     * Forward: y = scale * v, where scale = tanh(alpha) / alpha, alpha = sqrt(c) * ||v||
     *
     * VJP: grad_in_j = scale * grad_out_j + coeff * dot(grad_out, v) * v_j
     *   where coeff = sqrt_c * ((1 - t^2) * alpha - t) / (alpha^2 * r)
     *   t = tanh(alpha), r = ||v||
     */
    const float sqrt_c = sqrtf(c);
    const float eps = 1e-5f;

#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(batch > 128)
#endif
    for (int64_t b = 0; b < batch; b++) {
        const float *vb = v + b * dim;
        const float *gb = grad_out + b * dim;
        float *gib = grad_in + b * dim;

        /* ||v|| */
        float r2 = 0.0f;
        for (int64_t d = 0; d < dim; d++) r2 += vb[d] * vb[d];
        float r = sqrtf(r2);

        if (r < eps) {
            /* Near origin: scale ~ 1, coeff ~ 0 -> grad_in = grad_out */
            memcpy(gib, gb, (size_t)dim * sizeof(float));
            continue;
        }

        float alpha = sqrt_c * r;
        float t = tanhf(alpha);
        float scale = t / alpha;
        float t2 = 1.0f - t * t; /* sech^2(alpha) */
        float coeff = sqrt_c * (t2 * alpha - t) / (alpha * alpha * r);

        /* dot(grad_out, v) */
        float dot_gv = 0.0f;
        for (int64_t d = 0; d < dim; d++) dot_gv += gb[d] * vb[d];

        float rank1_scale = coeff * dot_gv;

#if defined(ARIA_SIMD_WIDTH)
        int64_t vec_end = dim - (dim % ARIA_SIMD_WIDTH);
        aria_simd_ps v_s = aria_simd_set1_ps(scale);
        aria_simd_ps v_r = aria_simd_set1_ps(rank1_scale);
        for (int64_t d = 0; d < vec_end; d += ARIA_SIMD_WIDTH) {
            aria_simd_ps vg = aria_simd_loadu_ps(gb + d);
            aria_simd_ps vv = aria_simd_loadu_ps(vb + d);
            aria_simd_ps res = aria_simd_fmadd_ps(v_r, vv, aria_simd_mul_ps(v_s, vg));
            aria_simd_storeu_ps(gib + d, res);
        }
        for (int64_t d = vec_end; d < dim; d++)
            gib[d] = scale * gb[d] + rank1_scale * vb[d];
#else
        for (int64_t d = 0; d < dim; d++)
            gib[d] = scale * gb[d] + rank1_scale * vb[d];
#endif
    }
}

void aria_log_map_backward_f32(const float *x, const float *grad_out, float *grad_in,
                                 int64_t batch, int64_t dim, float c) {
    /*
     * Backward pass for log_map.
     *
     * Forward: y = scale * x_c, where scale = atanh(alpha) / alpha,
     *          alpha = sqrt(c) * ||x_c||, x_c = clamp_norm(x)
     *
     * VJP: grad_in_j = scale * grad_out_j + coeff * dot(grad_out, x_c) * x_c_j
     *   where coeff = sqrt_c * (alpha / (1 - alpha^2) - a) / (alpha^2 * r)
     *   a = atanh(alpha), r = ||x_c||
     *
     * Assumes x is already inside the ball (clamp treated as identity for grad).
     */
    const float sqrt_c = sqrtf(c);
    const float max_norm = 1.0f - 1e-3f;
    const float eps = 1e-5f;

#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(batch > 128)
#endif
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * dim;
        const float *gb = grad_out + b * dim;
        float *gib = grad_in + b * dim;

        /* Clamp x to ball, then compute ||x_c|| */
        float xc[4096];
        float *xc_p = xc;
        if (dim > 4096) {
            xc_p = (float *)malloc((size_t)dim * sizeof(float));
            if (!xc_p) { memcpy(gib, gb, (size_t)dim * sizeof(float)); continue; }
        }
        memcpy(xc_p, xb, (size_t)dim * sizeof(float));
        _clamp_norm_vec(xc_p, dim, max_norm);

        float r2 = 0.0f;
        for (int64_t d = 0; d < dim; d++) r2 += xc_p[d] * xc_p[d];
        float r = sqrtf(r2);

        if (r < eps) {
            memcpy(gib, gb, (size_t)dim * sizeof(float));
            if (dim > 4096) free(xc_p);
            continue;
        }

        float alpha = sqrt_c * r;
        if (alpha > 1.0f - 1e-5f) alpha = 1.0f - 1e-5f;
        float a = atanhf(alpha);
        if (a > 10.0f) a = 10.0f;
        float scale = a / (sqrt_c * r);
        float one_minus_a2 = 1.0f - alpha * alpha;
        if (one_minus_a2 < 1e-10f) one_minus_a2 = 1e-10f;
        float coeff = sqrt_c * (alpha / one_minus_a2 - a) / (sqrt_c * r * sqrt_c * r * r);

        /* dot(grad_out, x_c) */
        float dot_gx = 0.0f;
        for (int64_t d = 0; d < dim; d++) dot_gx += gb[d] * xc_p[d];

        float rank1_scale = coeff * dot_gx;

#if defined(ARIA_SIMD_WIDTH)
        int64_t vec_end = dim - (dim % ARIA_SIMD_WIDTH);
        aria_simd_ps v_s = aria_simd_set1_ps(scale);
        aria_simd_ps v_r = aria_simd_set1_ps(rank1_scale);
        for (int64_t d = 0; d < vec_end; d += ARIA_SIMD_WIDTH) {
            aria_simd_ps vg = aria_simd_loadu_ps(gb + d);
            aria_simd_ps vx = aria_simd_loadu_ps(xc_p + d);
            aria_simd_ps res = aria_simd_fmadd_ps(v_r, vx, aria_simd_mul_ps(v_s, vg));
            aria_simd_storeu_ps(gib + d, res);
        }
        for (int64_t d = vec_end; d < dim; d++)
            gib[d] = scale * gb[d] + rank1_scale * xc_p[d];
#else
        for (int64_t d = 0; d < dim; d++)
            gib[d] = scale * gb[d] + rank1_scale * xc_p[d];
#endif
        if (dim > 4096) free(xc_p);
    }
}

} /* extern "C" */
