#ifndef ARIA_KERNELS_COMMON_H
#include "kernels_common.h"
#endif
#include <string.h>
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Math space (tropical) ────────────────────────────────────────── */

void aria_tropical_center_f32(const float *x, float *y,
                              int64_t batch, int64_t seq, int64_t dim) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static) if(batch * dim > 1024)
#endif
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t d = 0; d < dim; d++) {
            float max_val = -INFINITY;
            for (int64_t s = 0; s < seq; s++) {
                float v = x[(b * seq + s) * dim + d];
                if (v > max_val) max_val = v;
            }
            for (int64_t s = 0; s < seq; s++) {
                int64_t idx = (b * seq + s) * dim + d;
                y[idx] = x[idx] - max_val;
            }
        }
    }
}

void aria_tropical_attention_f32(const float *x, float *y,
                                 int64_t batch, int64_t seq, int64_t dim,
                                 float temperature) {
    if (temperature <= 0.0f) temperature = 0.1f;
    float inv_temp = 1.0f / temperature;

#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(batch > 1)
#endif
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * seq * dim;
        float *yb = y + b * seq * dim;
        float *dist = (float *)malloc(sizeof(float) * (size_t)seq);
        float *weights = (float *)malloc(sizeof(float) * (size_t)seq);

        for (int64_t i = 0; i < seq; i++) {
            const float *xi = xb + i * dim;
            for (int64_t j = 0; j <= i; j++) {
                const float *xj = xb + j * dim;
                float best = -INFINITY;
                int64_t k = 0;
#if defined(ARIA_SIMD_WIDTH)
                aria_simd_ps vbest = aria_simd_set1_ps(-INFINITY);
                for (; k <= dim - ARIA_SIMD_WIDTH; k += ARIA_SIMD_WIDTH) {
                    aria_simd_ps vxi = aria_simd_loadu_ps(xi + k);
                    aria_simd_ps vxj = aria_simd_loadu_ps(xj + k);
                    vbest = aria_simd_max_ps(vbest, aria_simd_add_ps(vxi, vxj));
                }
                float tmp[ARIA_SIMD_WIDTH]; aria_simd_storeu_ps(tmp, vbest);
                for (int h = 0; h < ARIA_SIMD_WIDTH; h++) if (tmp[h] > best) best = tmp[h];
#endif
                for (; k < dim; k++) {
                    float v = xi[k] + xj[k];
                    if (v > best) best = v;
                }
                dist[j] = best;
            }

            float max_logit = -INFINITY;
            for (int64_t j = 0; j <= i; j++) {
                float logit = dist[j] * inv_temp;
                if (logit > max_logit) max_logit = logit;
            }

            float sum = 0.0f;
            for (int64_t j = 0; j <= i; j++) {
                float w = expf((dist[j] * inv_temp) - max_logit);
                weights[j] = w;
                sum += w;
            }
            if (sum < 1e-12f) sum = 1e-12f;
            float inv_sum = 1.0f / sum;

            float *yi = yb + i * dim;
            memset(yi, 0, sizeof(float) * (size_t)dim);
            for (int64_t j = 0; j <= i; j++) {
                float w = weights[j] * inv_sum;
                const float *xj = xb + j * dim;
                int64_t d = 0;
#if defined(ARIA_SIMD_WIDTH)
                aria_simd_ps vw = aria_simd_set1_ps(w);
                for (; d <= dim - ARIA_SIMD_WIDTH; d += ARIA_SIMD_WIDTH) {
                    aria_simd_ps vyi = aria_simd_loadu_ps(yi + d);
                    aria_simd_ps vxj = aria_simd_loadu_ps(xj + d);
                    aria_simd_storeu_ps(yi + d, aria_simd_fmadd_ps(vw, vxj, vyi));
                }
#endif
                for (; d < dim; d++) yi[d] += w * xj[d];
            }
        }
        free(dist); free(weights);
    }
}

void aria_tropical_gate_f32(const float *x, float *y,
                            int64_t batch, int64_t seq, int64_t dim,
                            float temperature) {
    if (temperature <= 0.0f) temperature = 0.1f;
    float inv_temp = 1.0f / temperature;

#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(batch > 1)
#endif
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * seq * dim;
        float *yb = y + b * seq * dim;
        float *dist = (float *)malloc(sizeof(float) * (size_t)seq);
        float *weights = (float *)malloc(sizeof(float) * (size_t)seq);
        float *gated = (float *)malloc(sizeof(float) * (size_t)dim);

        for (int64_t i = 0; i < seq; i++) {
            const float *xi = xb + i * dim;
            for (int64_t j = 0; j <= i; j++) {
                const float *xj = xb + j * dim;
                float best = -INFINITY;
                int64_t k = 0;
#if defined(ARIA_SIMD_WIDTH)
                aria_simd_ps vbest = aria_simd_set1_ps(-INFINITY);
                for (; k <= dim - ARIA_SIMD_WIDTH; k += ARIA_SIMD_WIDTH) {
                    aria_simd_ps vxi = aria_simd_loadu_ps(xi + k);
                    aria_simd_ps vxj = aria_simd_loadu_ps(xj + k);
                    vbest = aria_simd_max_ps(vbest, aria_simd_add_ps(vxi, vxj));
                }
                float tmp[ARIA_SIMD_WIDTH]; aria_simd_storeu_ps(tmp, vbest);
                for (int h = 0; h < ARIA_SIMD_WIDTH; h++) if (tmp[h] > best) best = tmp[h];
#endif
                for (; k < dim; k++) {
                    float v = xi[k] + xj[k];
                    if (v > best) best = v;
                }
                dist[j] = best;
            }

            float max_logit = -INFINITY;
            for (int64_t j = 0; j <= i; j++) {
                float logit = dist[j] * inv_temp;
                if (logit > max_logit) max_logit = logit;
            }

            float sum = 0.0f;
            for (int64_t j = 0; j <= i; j++) {
                float w = expf((dist[j] * inv_temp) - max_logit);
                weights[j] = w;
                sum += w;
            }
            if (sum < 1e-12f) sum = 1e-12f;
            float inv_sum = 1.0f / sum;

            memset(gated, 0, sizeof(float) * (size_t)dim);
            for (int64_t j = 0; j <= i; j++) {
                float w = weights[j] * inv_sum;
                const float *xj = xb + j * dim;
                int64_t d = 0;
#if defined(ARIA_SIMD_WIDTH)
                aria_simd_ps vw = aria_simd_set1_ps(w);
                for (; d <= dim - ARIA_SIMD_WIDTH; d += ARIA_SIMD_WIDTH) {
                    aria_simd_ps vgate = aria_simd_loadu_ps(gated + d);
                    aria_simd_ps vxj = aria_simd_loadu_ps(xj + d);
                    aria_simd_storeu_ps(gated + d, aria_simd_fmadd_ps(vw, vxj, vgate));
                }
#endif
                for (; d < dim; d++) gated[d] += w * xj[d];
            }

            float *yi = yb + i * dim;
            int64_t d = 0;
#if defined(ARIA_SIMD_WIDTH)
            for (; d <= dim - ARIA_SIMD_WIDTH; d += ARIA_SIMD_WIDTH) {
                aria_simd_ps vxi = aria_simd_loadu_ps(xi + d);
                aria_simd_ps vg = aria_simd_loadu_ps(gated + d);
                aria_simd_ps vgate = aria_simd_sigmoid_ps(vg);
                aria_simd_storeu_ps(yi + d, aria_simd_mul_ps(vxi, vgate));
            }
#endif
            for (; d < dim; d++) {
                float gate = 1.0f / (1.0f + expf(-gated[d]));
                yi[d] = xi[d] * gate;
            }
        }
        free(dist); free(weights); free(gated);
    }
}

/* ── Math space (hyperbolic) ─────────────────────────────── */

/* Logic delegated to hyperbolic.cpp for optimized Mobius operations.
   Common hyperbolic utilities remain here for shared use. */

static inline void _aria_hyp_clamp_norm(float *v, int64_t dim, float max_norm) {
    float norm2 = 0.0f;
    int64_t d = 0;
#if defined(ARIA_SIMD_WIDTH)
        aria_simd_ps vnorm2 = aria_simd_zero_ps;
        for (; d <= dim - ARIA_SIMD_WIDTH; d += ARIA_SIMD_WIDTH) {
            aria_simd_ps vd = aria_simd_loadu_ps(v + d);
            vnorm2 = aria_simd_fmadd_ps(vd, vd, vnorm2);
        }
        float tmp[ARIA_SIMD_WIDTH]; aria_simd_storeu_ps(tmp, vnorm2);
        for (int i = 0; i < ARIA_SIMD_WIDTH; i++) norm2 += tmp[i];
#endif
    for (; d < dim; d++) norm2 += v[d] * v[d];
    
    float norm = sqrtf(norm2);
    if (norm > max_norm) {
        float scale = max_norm / (norm + 1e-10f);
        d = 0;
#if defined(ARIA_SIMD_WIDTH)
        aria_simd_ps vscale = aria_simd_set1_ps(scale);
        for (; d <= dim - ARIA_SIMD_WIDTH; d += ARIA_SIMD_WIDTH) {
            aria_simd_ps vd = aria_simd_loadu_ps(v + d);
            aria_simd_storeu_ps(v + d, aria_simd_mul_ps(vd, vscale));
        }
#endif
        for (; d < dim; d++) v[d] *= scale;
    }
}

void aria_hyp_distance_f32(const float *x, const float *y, float *out,
                           int64_t batch, int64_t seq, int64_t dim) {
    /* High-level distance implementation using poincare add */
    float c = 1.0f;
    float sqrt_c = 1.0f;
    float inv_sqrt_c = 1.0f;

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t s = 0; s < seq; s++) {
            const float *xv = x + (b * seq + s) * dim;
            const float *yv = y + (b * seq + s) * dim;
            
            float nx[4096];
            float *nx_ptr = nx;
            if (dim > 4096) nx_ptr = (float*)malloc(dim * sizeof(float));
            
            for(int64_t d=0; d<dim; d++) nx_ptr[d] = -xv[d];
            
            float diff[4096];
            float *diff_ptr = diff;
            if (dim > 4096) diff_ptr = (float*)malloc(dim * sizeof(float));

            extern void aria_poincare_add_f32(const float *x, const float *v, float *y,
                                             int64_t batch, int64_t dim, float c);
            aria_poincare_add_f32(nx_ptr, yv, diff_ptr, 1, dim, c);
            
            float norm2 = 0.0f;
            for(int64_t d=0; d<dim; d++) norm2 += diff_ptr[d]*diff_ptr[d];
            float norm = sqrtf(norm2);
            float arg = sqrt_c * norm;
            if (arg > 1.0f - 1e-4f) arg = 1.0f - 1e-4f;
            out[b * seq + s] = 2.0f * inv_sqrt_c * atanhf(arg);

            if (dim > 4096) { free(nx_ptr); free(diff_ptr); }
        }
    }
}

/* ── P-adic Math Space ─────────────────────────────────────────────── */

void aria_padic_gate_f32(const float *x, float *y, int64_t n, float p) {
    float log_p = logf(p);
    if (log_p == 0.0f) log_p = logf(2.0f);
    int64_t i = 0;
#if defined(ARIA_SIMD_WIDTH)
    aria_simd_ps vlp = aria_simd_set1_ps(log_p);
    for (; i <= n - ARIA_SIMD_WIDTH; i += ARIA_SIMD_WIDTH) {
        aria_simd_ps vx = aria_simd_loadu_ps(x + i);
        aria_simd_ps vax = aria_simd_and_ps(vx, aria_simd_castsi_ps(aria_simd_set1_epi32(0x7fffffff)));
        vax = aria_simd_max_ps(vax, aria_simd_set1_ps(1e-10f));
        aria_simd_ps val = aria_simd_div_ps(aria_simd_log_ps(vax), vlp);
        val = aria_simd_sub_ps(aria_simd_zero_ps, val);
        aria_simd_ps gate = aria_simd_sigmoid_ps(val);
        aria_simd_storeu_ps(y + i, aria_simd_mul_ps(vx, gate));
    }
#endif
    for (; i < n; i++) {
        float v = x[i];
        float abs_v = fabsf(v);
        if (abs_v < 1e-10f) abs_v = 1e-10f;
        float valuation = -(logf(abs_v) / log_p);
        float gate = 1.0f / (1.0f + expf(-valuation));
        y[i] = v * gate;
    }
}

void aria_padic_expand_f32(const float *x, const float *W, float *y,
                             int64_t batch, int64_t dim, float p, int64_t n_digits) {
    if (n_digits < 1) n_digits = 4;
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * dim;
        float *yb = y + b * dim;
        memset(yb, 0, (size_t)dim * sizeof(float));
        for (int64_t k = 0; k < n_digits; k++) {
            float scale = powf(p, (float)k);
            for (int64_t d = 0; d < dim; d++) {
                float digit = fmodf(fabsf(xb[d] * scale), p) / p;
                const float *wrow = W + (k * dim + d) * dim;
                for (int64_t o = 0; o < dim; o++) yb[o] += digit * wrow[o];
            }
        }
    }
}

void aria_padic_residual_f32(const float *x, const float *W, float *y,
                               int64_t batch, int64_t dim, float p, int64_t n_digits) {
    aria_padic_expand_f32(x, W, y, batch, dim, p, n_digits);
    for (int64_t i = 0; i < batch * dim; i++) y[i] += x[i];
}

void aria_ultrametric_attention_f32(const float *x, float *y,
                                      int64_t batch, int64_t seq, int64_t dim,
                                      float p) {
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < seq; i++) {
            const float *qi = x + (b * seq + i) * dim;
            float *yi = y + (b * seq + i) * dim;
            float *scores = (float *)malloc((size_t)seq * sizeof(float));
            if (!scores) { memcpy(yi, qi, (size_t)dim * sizeof(float)); continue; }
            float max_score = -INFINITY;
            for (int64_t j = 0; j <= i; j++) {
                const float *kj = x + (b * seq + j) * dim;
                float dist = 0.0f;
                for (int64_t d = 0; d < dim; d++) {
                    float diff = fabsf(qi[d] - kj[d]);
                    if (diff > dist) dist = diff;
                }
                scores[j] = -dist;
                if (scores[j] > max_score) max_score = scores[j];
            }
            float sum_exp = 0.0f;
            for (int64_t j = 0; j <= i; j++) {
                scores[j] = expf(scores[j] - max_score);
                sum_exp += scores[j];
            }
            float inv_sum = 1.0f / (sum_exp + 1e-7f);
            memset(yi, 0, (size_t)dim * sizeof(float));
            for (int64_t j = 0; j <= i; j++) {
                float w = scores[j] * inv_sum;
                const float *vj = x + (b * seq + j) * dim;
                for (int64_t d = 0; d < dim; d++) yi[d] += w * vj[d];
            }
            free(scores);
        }
    }
}

/* ── Clifford Math Space (Cl30 delegated to clifford.cpp) ─────────── */

extern void _gp_cl30_single(const float *ai, const float *bi, float *yi);

void aria_rotor_transform_f32(const float *x, const float *rotor, float *y, int64_t batch, int64_t dim) {
    int64_t n = (batch * dim) / 8;
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(n > 256)
#endif
    for (int64_t i = 0; i < n; i++) {
        const float *xi = x + i*8, *ri = rotor + i*8; float *yi = y + i*8;
        float r_rev[8] = {ri[0], ri[1], ri[2], ri[3], -ri[4], -ri[5], -ri[6], -ri[7]};
        float tmp[8]; _gp_cl30_single(ri, xi, tmp); _gp_cl30_single(tmp, r_rev, yi);
    }
}

void aria_grade_select_f32(const float *x, float *y, int64_t batch, int64_t dim, int32_t grade) {
    int64_t n = (batch * dim) / 8;
    for (int64_t i = 0; i < n; i++) {
        const float *xi = x + i*8; float *yi = y + i*8;
        memset(yi, 0, 8 * sizeof(float));
        if (grade == 0) yi[0] = xi[0];
        else if (grade == 1) { yi[1] = xi[1]; yi[2] = xi[2]; yi[3] = xi[3]; }
        else if (grade == 2) { yi[4] = xi[4]; yi[5] = xi[5]; yi[6] = xi[6]; }
        else if (grade == 3) yi[7] = xi[7];
    }
}

void aria_grade_mix_f32(const float *x, const float *alpha, float *y, int64_t batch, int64_t dim) {
    int64_t n = (batch * dim) / 8;
    for (int64_t i = 0; i < n; i++) {
        const float *xi = x + i*8; float *yi = y + i*8;
        yi[0] = xi[0] * alpha[0];
        yi[1] = xi[1] * alpha[1]; yi[2] = xi[2] * alpha[1]; yi[3] = xi[3] * alpha[1];
        yi[4] = xi[4] * alpha[2]; yi[5] = xi[5] * alpha[2]; yi[6] = xi[6] * alpha[2];
        yi[7] = xi[7] * alpha[3];
    }
}

void aria_clifford_attention_f32(const float *x, float *y, int64_t batch, int64_t seq, int64_t dim) {
    int64_t n_mv = dim / 8;
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(batch > 1)
#endif
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < seq; i++) {
            const float *qi = x + (b*seq + i)*dim; float *yi = y + (b*seq + i)*dim;
            float *scores = (float *)malloc((size_t)seq * sizeof(float));
            if (!scores) continue;
            float max_score = -INFINITY;
            for (int64_t j = 0; j <= i; j++) {
                const float *kj = x + (b*seq + j)*dim;
                float total_mag = 0.0f;
                for (int64_t m = 0; m < n_mv; m++) {
                    const float *qim = qi + m*8, *kjm = kj + m*8;
                    float res[8]; _gp_cl30_single(qim, kjm, res);
                    float mag2 = 0.0f; for(int k=0; k<8; k++) mag2 += res[k]*res[k];
                    total_mag += sqrtf(mag2);
                }
                scores[j] = total_mag;
                if (scores[j] > max_score) max_score = scores[j];
            }
            float sum_exp = 0.0f;
            for (int64_t j = 0; j <= i; j++) {
                scores[j] = expf((scores[j] - max_score) / sqrtf((float)dim));
                sum_exp += scores[j];
            }
            float inv_sum = 1.0f / (sum_exp + 1e-7f);
            memset(yi, 0, (size_t)dim * sizeof(float));
            for (int64_t j = 0; j <= i; j++) {
                float w = scores[j] * inv_sum;
                const float *vj = x + (b*seq + j)*dim;
                for (int64_t d = 0; d < dim; d++) yi[d] += w * vj[d];
            }
            free(scores);
        }
    }
}

/* ── Spiking Math Space ────────────────────────────────────────────── */

void aria_lif_neuron_f32(const float *x, float *y,
                           int64_t batch, int64_t seq, int64_t dim,
                           float tau, float threshold) {
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t d = 0; d < dim; d++) {
            float v = 0.0f;
            for (int64_t s = 0; s < seq; s++) {
                v = tau * v + x[(b * seq + s) * dim + d];
                float spike = (v > threshold) ? 1.0f : 0.0f;
                y[(b * seq + s) * dim + d] = spike;
                if (spike > 0.0f) v = 0.0f;
            }
        }
    }
}

void aria_spike_rate_code_f32(const float *x, float *y,
                                int64_t batch, int64_t seq, int64_t dim) {
    for (int64_t i = 0; i < batch * seq * dim; i++) {
        float prob = 1.0f / (1.0f + expf(-x[i]));
        y[i] = prob >= 0.5f ? 1.0f : 0.0f;
    }
}

void aria_stdp_attention_f32(const float *x, float *y,
                               int64_t batch, int64_t seq, int64_t dim,
                               float tau_plus, float tau_minus) {
    (void)tau_minus;
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < seq; i++) {
            const float *qi = x + (b * seq + i) * dim;
            float *yi = y + (b * seq + i) * dim;
            float *scores = (float *)malloc((size_t)seq * sizeof(float));
            if (!scores) { memcpy(yi, qi, (size_t)dim * sizeof(float)); continue; }
            float max_score = -INFINITY;
            for (int64_t j = 0; j <= i; j++) {
                float stdp = expf(-(float)(i - j) / tau_plus);
                float dot = 0.0f;
                const float *kj = x + (b * seq + j) * dim;
                for (int64_t d = 0; d < dim; d++) dot += qi[d] * kj[d];
                scores[j] = dot * stdp / sqrtf((float)dim);
                if (scores[j] > max_score) max_score = scores[j];
            }
            float sum_exp = 0.0f;
            for (int64_t j = 0; j <= i; j++) {
                scores[j] = expf(scores[j] - max_score);
                sum_exp += scores[j];
            }
            float inv_sum = 1.0f / (sum_exp + 1e-7f);
            memset(yi, 0, (size_t)dim * sizeof(float));
            for (int64_t j = 0; j <= i; j++) {
                float w = scores[j] * inv_sum;
                const float *vj = x + (b * seq + j) * dim;
                for (int64_t d = 0; d < dim; d++) yi[d] += w * vj[d];
            }
            free(scores);
        }
    }
}

#ifdef __cplusplus
}
#endif
