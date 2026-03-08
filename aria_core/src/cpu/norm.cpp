#ifndef ARIA_KERNELS_COMMON_H
#include "kernels_common.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── RMSNorm (already optimized with AVX2 + OpenMP) ───────────────── */

void aria_rmsnorm_f32(const float *x, const float *weight, float *y,
                      int64_t batch, int64_t dim, float eps) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(batch > ARIA_OMP_BATCH_THRESHOLD) schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * dim;
        float *yb = y + b * dim;

        double ss = 0.0;
        int64_t i = 0;
#ifdef __AVX2__
        __m256 vss = _mm256_setzero_ps();
        for (; i <= dim - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(xb + i);
            vss = _mm256_fmadd_ps(vx, vx, vss);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, vss);
        for (int j = 0; j < 8; j++) ss += (double)tmp[j];
#endif
        for (; i < dim; i++) {
            ss += (double)xb[i] * (double)xb[i];
        }

        float inv_rms = 1.0f / (float)sqrt(ss / (double)dim + (double)eps);

#ifdef __AVX2__
        __m256 vinv = _mm256_set1_ps(inv_rms);
        for (i = 0; i <= dim - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(xb + i);
            __m256 vw = _mm256_loadu_ps(weight + i);
            __m256 vy = _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vw);
            _mm256_storeu_ps(yb + i, vy);
        }
#else
        i = 0;
#endif
        for (; i < dim; i++) {
            yb[i] = xb[i] * inv_rms * weight[i];
        }
    }
}

/* ── LayerNorm (AVX2 + OpenMP) ────────────────────────────────────── */

void aria_layernorm_f32(const float *x, const float *weight, const float *bias,
                        float *y, int64_t batch, int64_t dim, float eps) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(batch > ARIA_OMP_BATCH_THRESHOLD) schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * dim;
        float *yb = y + b * dim;

        /* Mean with AVX2 */
        double sum = 0.0;
        int64_t i = 0;
#ifdef __AVX2__
        __m256 vsum = _mm256_setzero_ps();
        for (; i <= dim - 8; i += 8) {
            vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(xb + i));
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, vsum);
        for (int j = 0; j < 8; j++) sum += (double)tmp[j];
#endif
        for (; i < dim; i++) sum += (double)xb[i];
        float mean = (float)(sum / (double)dim);

        /* Variance with AVX2 */
        double var = 0.0;
        i = 0;
#ifdef __AVX2__
        __m256 vmean = _mm256_set1_ps(mean);
        __m256 vvar = _mm256_setzero_ps();
        for (; i <= dim - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(xb + i);
            __m256 vd = _mm256_sub_ps(vx, vmean);
            vvar = _mm256_fmadd_ps(vd, vd, vvar);
        }
        _mm256_storeu_ps(tmp, vvar);
        for (int j = 0; j < 8; j++) var += (double)tmp[j];
#endif
        for (; i < dim; i++) {
            double d = (double)xb[i] - (double)mean;
            var += d * d;
        }
        float inv_std = 1.0f / sqrtf((float)(var / (double)dim) + eps);

        /* Normalize + scale + shift with AVX2 */
        i = 0;
#ifdef __AVX2__
        __m256 vinv = _mm256_set1_ps(inv_std);
        // reuse vmean from above
        for (; i <= dim - 8; i += 8) {
            __m256 vx = _mm256_loadu_ps(xb + i);
            __m256 vw = _mm256_loadu_ps(weight + i);
            __m256 normed = _mm256_mul_ps(_mm256_sub_ps(vx, vmean), vinv);
            __m256 vy = _mm256_mul_ps(normed, vw);
            if (bias) {
                __m256 vb = _mm256_loadu_ps(bias + i);
                vy = _mm256_add_ps(vy, vb);
            }
            _mm256_storeu_ps(yb + i, vy);
        }
#endif
        for (; i < dim; i++) {
            float normed = (xb[i] - mean) * inv_std;
            yb[i] = normed * weight[i] + (bias ? bias[i] : 0.0f);
        }
    }
}

/* ── Softmax (AVX2 + OpenMP) ──────────────────────────────────────── */

void aria_softmax_f32(const float *x, float *y, int64_t batch, int64_t dim) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(batch > ARIA_OMP_BATCH_THRESHOLD) schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * dim;
        float *yb = y + b * dim;

        /* Find max with AVX2 */
        float max_val = xb[0];
        int64_t i = 1;
#ifdef __AVX2__
        if (dim >= 8) {
            __m256 vmax = _mm256_loadu_ps(xb);
            for (i = 8; i <= dim - 8; i += 8) {
                vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(xb + i));
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, vmax);
            max_val = tmp[0];
            for (int j = 1; j < 8; j++) if (tmp[j] > max_val) max_val = tmp[j];
        }
#endif
        for (; i < dim; i++) {
            if (xb[i] > max_val) max_val = xb[i];
        }

        /* Exp + sum with AVX2 */
        float sum = 0.0f;
        i = 0;
#ifdef __AVX2__
        __m256 vmax2 = _mm256_set1_ps(max_val);
        __m256 vsum = _mm256_setzero_ps();
        for (; i + 8 <= dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(xb + i);
            __m256 ve = _mm256_exp_ps(_mm256_sub_ps(vx, vmax2));
            _mm256_storeu_ps(yb + i, ve);
            vsum = _mm256_add_ps(vsum, ve);
        }
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        sum = _mm_cvtss_f32(lo);
#endif
        for (; i < dim; i++) {
            yb[i] = expf(xb[i] - max_val);
            sum += yb[i];
        }

        /* Normalize with AVX2 */
        if (sum < 1e-12f) sum = 1e-12f;
        float inv_sum = 1.0f / sum;
        i = 0;
#ifdef __AVX2__
        __m256 vinv = _mm256_set1_ps(inv_sum);
        for (; i + 8 <= dim; i += 8) {
            __m256 vy = _mm256_loadu_ps(yb + i);
            _mm256_storeu_ps(yb + i, _mm256_mul_ps(vy, vinv));
        }
#endif
        for (; i < dim; i++) {
            yb[i] *= inv_sum;
        }
    }
}

void aria_softmax_seq_f32(const float *x, float *y,
                            int64_t batch, int64_t seq, int64_t dim) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(batch * dim > ARIA_OMP_THRESHOLD) collapse(2) schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t d = 0; d < dim; d++) {
            float max_val = -INFINITY;
            for (int64_t s = 0; s < seq; s++) {
                float v = x[(b * seq + s) * dim + d];
                if (v > max_val) max_val = v;
            }
            float sum_exp = 0.0f;
            for (int64_t s = 0; s < seq; s++) {
                float e = expf(x[(b * seq + s) * dim + d] - max_val);
                y[(b * seq + s) * dim + d] = e;
                sum_exp += e;
            }
            float inv_sum = 1.0f / sum_exp;
            for (int64_t s = 0; s < seq; s++) {
                y[(b * seq + s) * dim + d] *= inv_sum;
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
