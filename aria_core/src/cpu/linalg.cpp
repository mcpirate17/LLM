#ifndef ARIA_KERNELS_COMMON_H
#include "kernels_common.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <algorithm>
#include <limits>
#include <vector>

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

void aria_sequence_self_similarity_f32(
    const float *reps, float *out, int64_t n_probes, int64_t seq_len, int64_t dim
) {
    if (n_probes <= 0 || seq_len <= 0 || dim <= 0) return;
    const int64_t sim_size = seq_len * seq_len;
    memset(out, 0, (size_t)sim_size * sizeof(float));

#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static) if(seq_len * seq_len > 64)
#endif
    for (int64_t i = 0; i < seq_len; i++) {
        for (int64_t j = 0; j < seq_len; j++) {
            double acc = 0.0;
            for (int64_t p = 0; p < n_probes; p++) {
                const float *vi = reps + ((p * seq_len + i) * dim);
                const float *vj = reps + ((p * seq_len + j) * dim);
                double dot = 0.0;
                double norm_i = 0.0;
                double norm_j = 0.0;
                for (int64_t k = 0; k < dim; k++) {
                    const double a = (double)vi[k];
                    const double b = (double)vj[k];
                    dot += a * b;
                    norm_i += a * a;
                    norm_j += b * b;
                }
                const double denom = sqrt(norm_i * norm_j);
                if (denom > 1e-12) {
                    acc += dot / denom;
                }
            }
            out[i * seq_len + j] = (float)(acc / (double)n_probes);
        }
    }
}

void aria_mean_abs_linear_delta_f32(
    const float *delta, const float *weight, float *out,
    int64_t batch, int64_t seq_len, int64_t dim, int64_t vocab
) {
    if (batch <= 0 || seq_len <= 0 || dim <= 0 || vocab <= 0) return;

#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for collapse(2) schedule(static) if(batch * seq_len >= 8)
#endif
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            const float *delta_row = delta + ((b * seq_len + s) * dim);
            double acc = 0.0;
            for (int64_t v = 0; v < vocab; ++v) {
                const float *weight_row = weight + (v * dim);
                double dot = 0.0;
                for (int64_t k = 0; k < dim; ++k) {
                    dot += (double)delta_row[k] * (double)weight_row[k];
                }
                acc += fabs(dot);
            }
            out[b * seq_len + s] = (float)(acc / (double)vocab);
        }
    }
}

static void jacobi_eigenvalues_symmetric(std::vector<double> &mat, int64_t n, std::vector<double> &eigs) {
    eigs.assign((size_t)n, 0.0);
    if (n <= 0) return;
    const int max_sweeps = 64;
    const double tol = 1e-10;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double max_off = 0.0;
        int64_t p = 0;
        int64_t q = 1;
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = i + 1; j < n; ++j) {
                const double off = fabs(mat[(size_t)i * (size_t)n + (size_t)j]);
                if (off > max_off) {
                    max_off = off;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_off < tol) {
            break;
        }

        const size_t pp = (size_t)p * (size_t)n + (size_t)p;
        const size_t qq = (size_t)q * (size_t)n + (size_t)q;
        const size_t pq = (size_t)p * (size_t)n + (size_t)q;
        const double app = mat[pp];
        const double aqq = mat[qq];
        const double apq = mat[pq];
        if (fabs(apq) < tol) {
            continue;
        }

        const double tau = (aqq - app) / (2.0 * apq);
        const double t = (tau >= 0.0)
            ? 1.0 / (tau + sqrt(1.0 + tau * tau))
            : -1.0 / (-tau + sqrt(1.0 + tau * tau));
        const double c = 1.0 / sqrt(1.0 + t * t);
        const double s = t * c;

        for (int64_t k = 0; k < n; ++k) {
            if (k == p || k == q) continue;
            const size_t kp = (size_t)k * (size_t)n + (size_t)p;
            const size_t kq = (size_t)k * (size_t)n + (size_t)q;
            const size_t pk = (size_t)p * (size_t)n + (size_t)k;
            const size_t qk = (size_t)q * (size_t)n + (size_t)k;
            const double akp = mat[kp];
            const double akq = mat[kq];
            mat[kp] = c * akp - s * akq;
            mat[pk] = mat[kp];
            mat[kq] = s * akp + c * akq;
            mat[qk] = mat[kq];
        }

        mat[pp] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        mat[qq] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        mat[pq] = 0.0;
        mat[(size_t)q * (size_t)n + (size_t)p] = 0.0;
    }

    for (int64_t i = 0; i < n; ++i) {
        eigs[(size_t)i] = mat[(size_t)i * (size_t)n + (size_t)i];
    }
    std::sort(eigs.begin(), eigs.end());
}

void aria_geometry_metrics_f32(
    const float *reps,
    const int64_t *row_indices,
    float *out,
    int64_t total_rows,
    int64_t sample_rows,
    int64_t dim
) {
    if (total_rows < 2 || sample_rows < 2 || dim < 2) {
        out[0] = 0.0f;
        out[1] = 0.0f;
        out[2] = 0.0f;
        return;
    }

    std::vector<double> mean((size_t)dim, 0.0);
    for (int64_t r = 0; r < total_rows; ++r) {
        const float *row = reps + (size_t)r * (size_t)dim;
        for (int64_t d = 0; d < dim; ++d) {
            mean[(size_t)d] += (double)row[d];
        }
    }
    const double inv_total_rows = 1.0 / (double)total_rows;
    for (int64_t d = 0; d < dim; ++d) {
        mean[(size_t)d] *= inv_total_rows;
    }

    const bool use_feature_gram = sample_rows >= dim;
    const int64_t gram_n = use_feature_gram ? dim : sample_rows;
    std::vector<double> gram((size_t)gram_n * (size_t)gram_n, 0.0);
    std::vector<double> centered_row((size_t)dim, 0.0);

    if (use_feature_gram) {
        for (int64_t sample = 0; sample < sample_rows; ++sample) {
            const int64_t row_idx = row_indices != nullptr ? row_indices[sample] : sample;
            const float *row = reps + (size_t)row_idx * (size_t)dim;
            for (int64_t d = 0; d < dim; ++d) {
                centered_row[(size_t)d] = (double)row[d] - mean[(size_t)d];
            }
            for (int64_t i = 0; i < dim; ++i) {
                const double vi = centered_row[(size_t)i];
                for (int64_t j = i; j < dim; ++j) {
                    gram[(size_t)i * (size_t)dim + (size_t)j] += vi * centered_row[(size_t)j];
                }
            }
        }
        for (int64_t i = 0; i < dim; ++i) {
            for (int64_t j = i + 1; j < dim; ++j) {
                gram[(size_t)j * (size_t)dim + (size_t)i] = gram[(size_t)i * (size_t)dim + (size_t)j];
            }
        }
    } else {
        std::vector<double> centered((size_t)sample_rows * (size_t)dim, 0.0);
        for (int64_t sample = 0; sample < sample_rows; ++sample) {
            const int64_t row_idx = row_indices != nullptr ? row_indices[sample] : sample;
            const float *row = reps + (size_t)row_idx * (size_t)dim;
            double *dst = centered.data() + (size_t)sample * (size_t)dim;
            for (int64_t d = 0; d < dim; ++d) {
                dst[d] = (double)row[d] - mean[(size_t)d];
            }
        }
        for (int64_t i = 0; i < sample_rows; ++i) {
            const double *ri = centered.data() + (size_t)i * (size_t)dim;
            for (int64_t j = i; j < sample_rows; ++j) {
                const double *rj = centered.data() + (size_t)j * (size_t)dim;
                double dot = 0.0;
                for (int64_t d = 0; d < dim; ++d) {
                    dot += ri[d] * rj[d];
                }
                gram[(size_t)i * (size_t)sample_rows + (size_t)j] = dot;
                gram[(size_t)j * (size_t)sample_rows + (size_t)i] = dot;
            }
        }
    }

    std::vector<double> eigs;
    jacobi_eigenvalues_symmetric(gram, gram_n, eigs);

    std::vector<double> singular_values;
    singular_values.reserve((size_t)gram_n);
    double sum_sv = 0.0;
    double min_sv = std::numeric_limits<double>::infinity();
    double max_sv = 0.0;
    for (double eig : eigs) {
        const double sv = sqrt(std::max(eig, 1e-20));
        const double clamped = std::max(sv, 1e-10);
        singular_values.push_back(clamped);
        sum_sv += clamped;
        min_sv = std::min(min_sv, clamped);
        max_sv = std::max(max_sv, clamped);
    }

    double sum_sq = 0.0;
    double entropy = 0.0;
    for (double sv : singular_values) {
        const double p = sv / sum_sv;
        sum_sq += p * p;
        entropy -= p * log(p);
    }

    out[0] = (float)(1.0 / sum_sq);
    out[1] = max_sv > 0.0 ? (float)(min_sv / max_sv) : 0.0f;
    out[2] = (float)(exp(entropy) / (double)singular_values.size());
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
