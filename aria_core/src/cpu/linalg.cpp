#ifndef ARIA_KERNELS_COMMON_H
#include "kernels_common.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── Reductions ────────────────────────────────────────────────────── */

float aria_sum_f32(const float *x, int64_t n) {
    float sum = 0.0f;
    float c = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float y = x[i] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

float aria_mean_f32(const float *x, int64_t n) {
    if (n == 0) return 0.0f;
    return aria_sum_f32(x, n) / (float)n;
}

float aria_linear_cka_f32(const float *X, const float *Y, int64_t n) {
    if (n <= 0) return 0.0f;
    int64_t size = n * n;
    float mean_x = aria_mean_f32(X, size);
    float mean_y = aria_mean_f32(Y, size);
    double hsic_xy = 0.0, hsic_xx = 0.0, hsic_yy = 0.0;
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for reduction(+:hsic_xy, hsic_xx, hsic_yy) schedule(static)
#endif
    for (int64_t i = 0; i < size; i++) {
        double xi = (double)X[i] - (double)mean_x;
        double yi = (double)Y[i] - (double)mean_y;
        hsic_xy += xi * yi; hsic_xx += xi * xi; hsic_yy += yi * yi;
    }
    double denom = sqrt(hsic_xx * hsic_yy);
    if (denom < 1e-10) return 0.0f;
    float result = (float)(hsic_xy / denom);
    if (result < 0.0f) return 0.0f;
    if (result > 1.0f) return 1.0f;
    return result;
}

/* ── Matrix multiply (tiled) ───────────────────────────────────────── */

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

void aria_matmul_f32(const float *A, const float *B, float *C,
                     int64_t M, int64_t K, int64_t N) {
#ifdef ARIA_HAS_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K,
                1.0f, A, (int)K, B, (int)N,
                0.0f, C, (int)N);
#else
    memset(C, 0, (size_t)(M * N) * sizeof(float));
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static) if(M*N > 4096)
#endif
    for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
        for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
            int64_t iend = i0 + TILE_M < M ? i0 + TILE_M : M;
            int64_t kend = k0 + TILE_K < K ? k0 + TILE_K : K;
            for (int64_t j0 = 0; j0 < N; j0 += TILE_N) {
                int64_t jend = j0 + TILE_N < N ? j0 + TILE_N : N;
                for (int64_t i = i0; i < iend; i++) {
                    for (int64_t k = k0; k < kend; k++) {
                        float a_ik = A[i * K + k];
                        for (int64_t j = j0; j < jend; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
#endif
}

void aria_tropical_matmul_f32(const float *A, const float *B, float *C,
                              int64_t M, int64_t K, int64_t N) {
    float *BT = (float *)malloc((size_t)(N * K) * sizeof(float));
    if (!BT) return;
    for (int64_t k = 0; k < K; k++) {
        for (int64_t j = 0; j < N; j++) {
            BT[j * K + k] = B[k * N + j];
        }
    }
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(M*N > 1024)
#endif
    for (int64_t i = 0; i < M; i++) {
        const float *Ai = A + i * K;
        for (int64_t j = 0; j < N; j++) {
            const float *BTj = BT + j * K;
            float best = INFINITY;
#ifdef __AVX2__
            __m256 vbest = _mm256_set1_ps(INFINITY);
            int64_t k = 0;
            for (; k <= K - 8; k += 8) {
                __m256 va = _mm256_loadu_ps(Ai + k);
                __m256 vb = _mm256_loadu_ps(BTj + k);
                vbest = _mm256_min_ps(vbest, _mm256_add_ps(va, vb));
            }
            float tmp[8]; _mm256_storeu_ps(tmp, vbest);
            for (int h = 0; h < 8; h++) if (tmp[h] < best) best = tmp[h];
            for (; k < K; k++) { float v = Ai[k] + BTj[k]; if (v < best) best = v; }
#else
            for (int64_t k = 0; k < K; k++) { float v = Ai[k] + BTj[k]; if (v < best) best = v; }
#endif
            C[i * N + j] = best;
        }
    }
    free(BT);
}

void aria_tropical_matmul_batched_f32(const float *A, const float *B, float *C,
                                      int64_t batch, int64_t M, int64_t K, int64_t N) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        aria_tropical_matmul_f32(A + b * M * K, B + b * K * N, C + b * M * N, M, K, N);
    }
}

void aria_linear_f32(const float *x, const float *W, const float *bias,
                     float *y, int64_t batch, int64_t dim_in, int64_t dim_out) {
#ifdef ARIA_HAS_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)batch, (int)dim_out, (int)dim_in,
                1.0f, x, (int)dim_in, W, (int)dim_in,
                0.0f, y, (int)dim_out);
    if (bias) {
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if((batch * dim_out) > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t b = 0; b < batch; b++) {
            float *yb = y + b * dim_out;
            for (int64_t o = 0; o < dim_out; o++) yb[o] += bias[o];
        }
    }
#else
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * dim_in;
        float *yb = y + b * dim_out;
        for (int64_t o = 0; o < dim_out; o++) {
            const float *Wo = W + o * dim_in;
            double acc = 0.0;
            int64_t i = 0;
#ifdef __AVX2__
            __m256 vacc = _mm256_setzero_ps();
            for (; i <= dim_in - 8; i += 8) {
                __m256 vx = _mm256_loadu_ps(xb + i);
                __m256 vw = _mm256_loadu_ps(Wo + i);
                vacc = _mm256_fmadd_ps(vx, vw, vacc);
            }
            float tmp[8]; _mm256_storeu_ps(tmp, vacc);
            for (int j = 0; j < 8; j++) acc += (double)tmp[j];
#endif
            for (; i < dim_in; i++) acc += (double)xb[i] * (double)Wo[i];
            yb[o] = (float)acc + (bias ? bias[o] : 0.0f);
        }
    }
#endif
}

void aria_transpose2d_f32(const float *input, float *output, int64_t rows, int64_t cols) {
    for (int64_t i = 0; i < rows; i++) for (int64_t j = 0; j < cols; j++) output[j * rows + i] = input[i * cols + j];
}

/* ── Gromov 4-Point Delta-Hyperbolicity ────────────────────────────── */
/*
 * Computes Gromov's delta for a distance matrix d[n x n].
 *
 * For all 4-tuples (x,y,z,w) from the index set:
 *   S1 = d(x,y) + d(z,w)
 *   S2 = d(x,z) + d(y,w)
 *   S3 = d(x,w) + d(y,z)
 *   Sort so S1 <= S2 <= S3
 *   delta = (S3 - S2) / 2
 *   max_delta = max over all 4-tuples
 *
 * Inputs:
 *   d:       (n * n) row-major distance matrix
 *   indices: array of index values to iterate over (length n_idx)
 *   n:       full matrix dimension
 *   n_idx:   number of indices to use
 * Returns:
 *   max delta value
 */
float aria_gromov_delta_f32(const float *d, const int32_t *indices,
                             int64_t n, int64_t n_idx) {
    if (n_idx < 4) return 0.0f;

    float max_delta = 0.0f;

#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for reduction(max:max_delta) schedule(dynamic, 4) if(n_idx > 10)
#endif
    for (int64_t i = 0; i < n_idx; i++) {
        int32_t xi = indices[i];
        float local_max = 0.0f;

        for (int64_t j = i + 1; j < n_idx; j++) {
            int32_t yj = indices[j];
            float dxy = d[xi * n + yj];

            for (int64_t k = j + 1; k < n_idx; k++) {
                int32_t zk = indices[k];
                float dxz = d[xi * n + zk];
                float dyz = d[yj * n + zk];

                for (int64_t l = k + 1; l < n_idx; l++) {
                    int32_t wl = indices[l];
                    float dzw = d[zk * n + wl];
                    float dyw = d[yj * n + wl];
                    float dxw = d[xi * n + wl];

                    float s1 = dxy + dzw;
                    float s2 = dxz + dyw;
                    float s3 = dxw + dyz;

                    /* Sort three values: we only need the top two */
                    float mid, top;
                    if (s1 <= s2) {
                        if (s2 <= s3) { mid = s2; top = s3; }
                        else if (s1 <= s3) { mid = s3; top = s2; }
                        else { mid = s1; top = s2; }
                    } else {
                        if (s1 <= s3) { mid = s1; top = s3; }
                        else if (s2 <= s3) { mid = s3; top = s1; }
                        else { mid = s2; top = s1; }
                    }

                    float delta_val = (top - mid) * 0.5f;
                    if (delta_val > local_max) local_max = delta_val;
                }
            }
        }

        if (local_max > max_delta) max_delta = local_max;
    }

    return max_delta;
}

#ifdef __cplusplus
}
#endif
