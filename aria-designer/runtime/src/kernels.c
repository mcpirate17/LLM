/**
 * kernels.c — Optimized tensor operation kernels.
 *
 * Compiled with -O3 -march=native for auto-vectorization.
 * Manual SIMD not needed: GCC/Clang vectorize these loops well.
 *
 * Performance targets (single-threaded, L1-resident data):
 *   relu/silu/gelu: ~2 GB/s (memory-bound)
 *   matmul 256x256: ~8 GFLOPS (compute-bound, tiled)
 *   rmsnorm: ~1.5 GB/s (reduction + elementwise)
 */
#include "kernels.h"
#include "simd_math.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef ARIA_HAS_OPENMP
#include <omp.h>
#define ARIA_OMP_THRESHOLD 16384
#define ARIA_OMP_BATCH_THRESHOLD 4
#endif

#ifdef ARIA_HAS_BLAS
#ifdef ARIA_BLAS_SCIPY_PREFIX
/* scipy-bundled OpenBLAS uses prefixed symbols; provide our own declarations */
enum CBLAS_ORDER { CblasRowMajor = 101 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };
extern void scipy_cblas_sgemm(enum CBLAS_ORDER, enum CBLAS_TRANSPOSE, enum CBLAS_TRANSPOSE,
                               int, int, int, float, const float *, int,
                               const float *, int, float, float *, int);
#define cblas_sgemm scipy_cblas_sgemm
#else
#include <cblas.h>
#endif
#endif

/* ── Constants ─────────────────────────────────────────────────────── */

static const float GELU_COEFF = 0.7978845608028654f;  /* sqrt(2/pi) */
static const float GELU_CUBIC = 0.044715f;

/* ── Elementwise unary ─────────────────────────────────────────────── */

void aria_relu_f32(const float *x, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

void aria_gelu_f32(const float *x, float *y, int64_t n) {
    /* GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
     * tanh(z) = 2*sigmoid(2z) - 1, reusing fast AVX2 sigmoid */
#ifdef __AVX2__
    {
        const __m256 half   = _mm256_set1_ps(0.5f);
        const __m256 one    = _mm256_set1_ps(1.0f);
        const __m256 two    = _mm256_set1_ps(2.0f);
        const __m256 coeff  = _mm256_set1_ps(GELU_COEFF);
        const __m256 cubic  = _mm256_set1_ps(GELU_CUBIC);

        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            /* inner = sqrt(2/pi) * (x + 0.044715 * x^3) */
            __m256 x2 = _mm256_mul_ps(vx, vx);
            __m256 x3 = _mm256_mul_ps(x2, vx);
            __m256 inner = _mm256_fmadd_ps(cubic, x3, vx);
            inner = _mm256_mul_ps(coeff, inner);
            /* tanh(inner) = 2 * sigmoid(2 * inner) - 1 */
            __m256 two_inner = _mm256_mul_ps(two, inner);
            __m256 sig = _mm256_sigmoid_ps(two_inner);
            __m256 tanh_val = _mm256_fmsub_ps(two, sig, one);
            /* gelu = 0.5 * x * (1 + tanh) */
            __m256 vy = _mm256_mul_ps(half, _mm256_mul_ps(vx, _mm256_add_ps(one, tanh_val)));
            _mm256_storeu_ps(y + i, vy);
        }
        /* Scalar tail */
        for (int64_t i = vec_end; i < n; i++) {
            float v = x[i];
            float inner = GELU_COEFF * (v + GELU_CUBIC * v * v * v);
            y[i] = 0.5f * v * (1.0f + tanhf(inner));
        }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        float inner = GELU_COEFF * (v + GELU_CUBIC * v * v * v);
        y[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
#endif
}

void aria_silu_f32(const float *x, float *y, int64_t n) {
    /* SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)) */
#ifdef __AVX2__
    {
        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 sig = _mm256_sigmoid_ps(vx);
            __m256 vy = _mm256_mul_ps(vx, sig);
            _mm256_storeu_ps(y + i, vy);
        }
        /* Scalar tail (at most 7 elements, no need for OpenMP) */
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

void aria_square_f32(const float *x, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v * v;
    }
}

void aria_abs_f32(const float *x, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = fabsf(x[i]);
    }
}

void aria_neg_f32(const float *x, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = -x[i];
    }
}

void aria_reciprocal_f32(const float *x, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = 1.0f / x[i];
    }
}

void aria_log_f32(const float *x, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = logf(x[i]);
    }
}

void aria_sqrt_f32(const float *x, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = sqrtf(x[i]);
    }
}

void aria_sin_f32(const float *x, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = sinf(x[i]);
    }
}

void aria_cos_f32(const float *x, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = cosf(x[i]);
    }
}

void aria_tanh_f32(const float *x, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = tanhf(x[i]);
    }
}

void aria_sigmoid_f32(const float *x, float *y, int64_t n) {
#ifdef __AVX2__
    {
        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vy = _mm256_sigmoid_ps(vx);
            _mm256_storeu_ps(y + i, vy);
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
#ifdef __AVX2__
    {
        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vy = _mm256_exp_ps(vx);
            _mm256_storeu_ps(y + i, vy);
        }
        for (int64_t i = vec_end; i < n; i++) {
            y[i] = expf(x[i]);
        }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = expf(x[i]);
    }
#endif
}

/* ── Elementwise binary ────────────────────────────────────────────── */

void aria_add_f32(const float *a, const float *b, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = a[i] + b[i];
    }
}

void aria_mul_f32(const float *a, const float *b, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = a[i] * b[i];
    }
}

void aria_sub_f32(const float *a, const float *b, float *y, int64_t n) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        y[i] = a[i] - b[i];
    }
}

void aria_tropical_add_f32(const float *a, const float *b, float *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = fminf(a[i], b[i]);
    }
}

/* ── Reductions ────────────────────────────────────────────────────── */

float aria_sum_f32(const float *x, int64_t n) {
    /* Kahan summation for numerical stability */
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

/* ── Matrix multiply (tiled) ───────────────────────────────────────── */

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

void aria_matmul_f32(const float *A, const float *B, float *C,
                     int64_t M, int64_t K, int64_t N) {
#ifdef ARIA_HAS_BLAS
    /* C = 1.0 * A @ B + 0.0 * C */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K,
                1.0f, A, (int)K, B, (int)N,
                0.0f, C, (int)N);
#else
    /* Fallback: tiled matmul */
    memset(C, 0, sizeof(float) * M * N);
    for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
        int64_t iend = i0 + TILE_M < M ? i0 + TILE_M : M;
        for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
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
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float best = INFINITY;
            const float *Ai = A + i * K;
            for (int64_t k = 0; k < K; k++) {
                float v = Ai[k] + B[k * N + j];
                if (v < best) best = v;
            }
            C[i * N + j] = best;
        }
    }
}

/* ── Linear projection ─────────────────────────────────────────────── */

void aria_linear_f32(const float *x, const float *W, const float *bias,
                     float *y, int64_t batch, int64_t dim_in, int64_t dim_out) {
#ifdef ARIA_HAS_BLAS
    /* y = x @ W^T : x[batch, dim_in] @ W^T[dim_in, dim_out] = y[batch, dim_out]
     * W is stored as [dim_out, dim_in], so W^T means CblasTrans on W */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)batch, (int)dim_out, (int)dim_in,
                1.0f, x, (int)dim_in, W, (int)dim_in,
                0.0f, y, (int)dim_out);
    /* Add bias if present */
    if (bias) {
        for (int64_t b = 0; b < batch; b++) {
            float *yb = y + b * dim_out;
            for (int64_t o = 0; o < dim_out; o++) {
                yb[o] += bias[o];
            }
        }
    }
#else
    /* y[b, o] = sum_i(x[b, i] * W[o, i]) + bias[o]
     * W is stored as [dim_out, dim_in] (row-major) */
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * dim_in;
        float *yb = y + b * dim_out;
        for (int64_t o = 0; o < dim_out; o++) {
            const float *Wo = W + o * dim_in;
            float acc = 0.0f;
            for (int64_t i = 0; i < dim_in; i++) {
                acc += xb[i] * Wo[i];
            }
            yb[o] = bias ? acc + bias[o] : acc;
        }
    }
#endif
}

/* ── IO ────────────────────────────────────────────────────────────── */

int aria_read_csv_f32(const char *filename, float *out_data, int64_t max_rows, int64_t max_cols, char delimiter) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;

    char line[4096];
    int64_t row = 0;
    int64_t col = 0;
    
    while (fgets(line, sizeof(line), fp) && row < max_rows) {
        char *ptr = line;
        col = 0;
        while (*ptr && col < max_cols) {
            float val = strtof(ptr, &ptr);
            out_data[row * max_cols + col] = val;
            col++;
            if (*ptr == delimiter) ptr++;
            else break; 
        }
        row++;
    }
    
    fclose(fp);
    return row;
}

int aria_filter_f32(const float *data, float *out_data, int64_t rows, int64_t cols, int64_t col_idx, float val, int op) {
    int64_t out_row = 0;
    for (int64_t i = 0; i < rows; i++) {
        float v = data[i * cols + col_idx];
        int keep = 0;
        switch (op) {
            case 0: keep = v > val; break;  // >
            case 1: keep = v < val; break;  // <
            case 2: keep = v >= val; break; // >=
            case 3: keep = v <= val; break; // <=
            case 4: keep = fabsf(v - val) < 1e-6; break; // ==
            case 5: keep = fabsf(v - val) > 1e-6; break; // !=
        }
        if (keep) {
            memcpy(out_data + out_row * cols, data + i * cols, cols * sizeof(float));
            out_row++;
        }
    }
    return out_row;
}

int aria_file_loader_csv_f32(const char *filename, float *out_data,
                             int64_t max_rows, int64_t max_cols,
                             char delimiter, int has_header) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;

    char line[8192];
    int64_t row = 0;
    int skip_first = has_header ? 1 : 0;

    char delim_str[2] = {delimiter, '\0'};
    while (fgets(line, sizeof(line), fp) && row < max_rows) {
        if (skip_first) {
            skip_first = 0;
            continue;
        }

        int64_t col = 0;
        char *tok = strtok(line, delim_str);
        while (tok && col < max_cols) {
            char *end = tok;
            float val = strtof(tok, &end);
            if (end != tok) {
                out_data[row * max_cols + col] = val;
                col++;
            }
            tok = strtok(NULL, delim_str);
        }

        if (col > 0) row++;
    }

    fclose(fp);
    return (int)row;
}

int aria_binary_file_reader_f32(const char *filename, float *out_data,
                                int64_t max_elems, int64_t offset_bytes) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) return -1;

    if (offset_bytes > 0) {
        if (fseek(fp, (long)offset_bytes, SEEK_SET) != 0) {
            fclose(fp);
            return -2;
        }
    }

    size_t n = fread(out_data, sizeof(float), (size_t)max_elems, fp);
    fclose(fp);
    return (int)n;
}

int aria_file_writer_txt_f32(const char *filename, const float *data,
                             int64_t n, int overwrite) {
    if (!overwrite) {
        FILE *chk = fopen(filename, "r");
        if (chk) {
            fclose(chk);
            return -1;
        }
    }

    FILE *fp = fopen(filename, "w");
    if (!fp) return -2;

    for (int64_t i = 0; i < n; i++) {
        fprintf(fp, "%g\n", (double)data[i]);
    }

    fclose(fp);
    return (int)n;
}

/* ── Math space (tropical) ────────────────────────────────────────── */

void aria_tropical_center_f32(const float *x, float *y,
                              int64_t batch, int64_t seq, int64_t dim) {
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t d = 0; d < dim; d++) {
            float baseline = INFINITY;
            for (int64_t s = 0; s < seq; s++) {
                float v = x[(b * seq + s) * dim + d];
                if (v < baseline) baseline = v;
            }
            for (int64_t s = 0; s < seq; s++) {
                int64_t idx = (b * seq + s) * dim + d;
                y[idx] = x[idx] - baseline;
            }
        }
    }
}

/* ── Math space (hyperbolic + p-adic) ─────────────────────────────── */

static void aria_clamp_norm_vec(const float *x, float *y, int64_t dim,
                                float max_norm, float eps) {
    float ss = 0.0f;
    for (int64_t i = 0; i < dim; i++) {
        ss += x[i] * x[i];
    }
    float norm = sqrtf(ss);
    if (norm < eps) norm = eps;
    float scaled = norm;
    if (scaled > max_norm) scaled = max_norm;
    float scale = scaled / norm;
    for (int64_t i = 0; i < dim; i++) {
        y[i] = x[i] * scale;
    }
}

void aria_hyp_distance_f32(const float *x, const float *y, float *out,
                           int64_t batch, int64_t seq, int64_t dim) {
    const float c = 1.0f;
    const float sqrt_c = 1.0f;
    const float eps = 1e-5f;
    const float max_norm = 1.0f - 1e-3f;
    const float scale = 2.0f / sqrt_c;

    float *x_clamped = (float *)malloc(sizeof(float) * (size_t)dim);
    float *y_clamped = (float *)malloc(sizeof(float) * (size_t)dim);
    float *num = (float *)malloc(sizeof(float) * (size_t)dim);
    float *tmp = (float *)malloc(sizeof(float) * (size_t)dim);

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t s = 0; s < seq; s++) {
            const float *xv = x + (b * seq + s) * dim;
            const float *yv = y + (b * seq + s) * dim;

            aria_clamp_norm_vec(xv, x_clamped, dim, max_norm, eps);
            aria_clamp_norm_vec(yv, y_clamped, dim, max_norm, eps);

            float x_sq = 0.0f;
            float y_sq = 0.0f;
            float xy = 0.0f;
            for (int64_t i = 0; i < dim; i++) {
                float xi = x_clamped[i];
                float yi = y_clamped[i];
                x_sq += xi * xi;
                y_sq += yi * yi;
                xy += (-xi) * yi;
            }

            float a = 1.0f + 2.0f * c * xy + c * y_sq;
            float bcoef = 1.0f - c * x_sq;
            float denom = 1.0f + 2.0f * c * xy + c * c * x_sq * y_sq;
            if (denom < eps) denom = eps;

            for (int64_t i = 0; i < dim; i++) {
                num[i] = a * (-x_clamped[i]) + bcoef * y_clamped[i];
                tmp[i] = num[i] / denom;
            }

            aria_clamp_norm_vec(tmp, num, dim, max_norm, eps);

            float ss = 0.0f;
            for (int64_t i = 0; i < dim; i++) {
                ss += num[i] * num[i];
            }
            float diff_norm = sqrtf(ss);
            if (diff_norm < eps) diff_norm = eps;
            float arg = sqrt_c * diff_norm;
            if (arg > 0.999999f) arg = 0.999999f;
            float dist = scale * atanhf(arg);
            if (dist > 10.0f) dist = 10.0f;
            if (dist < -10.0f) dist = -10.0f;
            out[b * seq + s] = dist;
        }
    }

    free(x_clamped);
    free(y_clamped);
    free(num);
    free(tmp);
}

void aria_padic_gate_f32(const float *x, float *y, int64_t n, float p) {
    float log_p = logf(p);
    if (log_p == 0.0f) log_p = logf(2.0f);
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        float abs_v = fabsf(v);
        if (abs_v < 1e-10f) abs_v = 1e-10f;
        float valuation = -(logf(abs_v) / log_p);
        if (valuation > 10.0f) valuation = 10.0f;
        if (valuation < -10.0f) valuation = -10.0f;
        float gate = 1.0f / (1.0f + expf(-valuation));
        y[i] = v * gate;
    }
}

void aria_tropical_attention_f32(const float *x, float *y,
                                 int64_t batch, int64_t seq, int64_t dim,
                                 float temperature) {
    if (temperature <= 0.0f) temperature = 0.1f;
    float *dist = (float *)malloc(sizeof(float) * (size_t)seq);
    float *weights = (float *)malloc(sizeof(float) * (size_t)seq);

    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * seq * dim;
        float *yb = y + b * seq * dim;

        for (int64_t i = 0; i < seq; i++) {
            const float *xi = xb + i * dim;

            for (int64_t j = 0; j < seq; j++) {
                const float *xj = xb + j * dim;
                float best = INFINITY;
                for (int64_t k = 0; k < dim; k++) {
                    float v = xi[k] + xj[k];
                    if (v < best) best = v;
                }
                dist[j] = best;
            }

            float max_logit = -INFINITY;
            for (int64_t j = 0; j < seq; j++) {
                float logit = -dist[j] / temperature;
                if (logit > max_logit) max_logit = logit;
            }

            float sum = 0.0f;
            for (int64_t j = 0; j < seq; j++) {
                float w = expf((-dist[j] / temperature) - max_logit);
                weights[j] = w;
                sum += w;
            }
            if (sum < 1e-12f) sum = 1e-12f;

            float *yi = yb + i * dim;
            for (int64_t d = 0; d < dim; d++) {
                float acc = 0.0f;
                for (int64_t j = 0; j < seq; j++) {
                    const float *xj = xb + j * dim;
                    acc += (weights[j] / sum) * xj[d];
                }
                yi[d] = acc;
            }
        }
    }

    free(dist);
    free(weights);
}

void aria_tropical_gate_f32(const float *x, float *y,
                            int64_t batch, int64_t seq, int64_t dim,
                            float temperature) {
    if (temperature <= 0.0f) temperature = 0.1f;
    float *dist = (float *)malloc(sizeof(float) * (size_t)seq);
    float *weights = (float *)malloc(sizeof(float) * (size_t)seq);
    float *gated = (float *)malloc(sizeof(float) * (size_t)dim);

    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * seq * dim;
        float *yb = y + b * seq * dim;

        for (int64_t i = 0; i < seq; i++) {
            const float *xi = xb + i * dim;

            for (int64_t j = 0; j < seq; j++) {
                const float *xj = xb + j * dim;
                float best = INFINITY;
                for (int64_t k = 0; k < dim; k++) {
                    float v = xi[k] + xj[k];
                    if (v < best) best = v;
                }
                dist[j] = best;
            }

            float max_logit = -INFINITY;
            for (int64_t j = 0; j < seq; j++) {
                float logit = -dist[j] / temperature;
                if (logit > max_logit) max_logit = logit;
            }

            float sum = 0.0f;
            for (int64_t j = 0; j < seq; j++) {
                float w = expf((-dist[j] / temperature) - max_logit);
                weights[j] = w;
                sum += w;
            }
            if (sum < 1e-12f) sum = 1e-12f;

            for (int64_t d = 0; d < dim; d++) {
                float acc = 0.0f;
                for (int64_t j = 0; j < seq; j++) {
                    const float *xj = xb + j * dim;
                    acc += (weights[j] / sum) * xj[d];
                }
                gated[d] = acc;
            }

            float *yi = yb + i * dim;
            for (int64_t d = 0; d < dim; d++) {
                float gate = 1.0f / (1.0f + expf(-gated[d]));
                yi[d] = xi[d] * gate;
            }
        }
    }

    free(dist);
    free(weights);
    free(gated);
}

/* ── Softmax ───────────────────────────────────────────────────────── */

void aria_softmax_f32(const float *x, float *y, int64_t batch, int64_t dim) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if((batch * dim) > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * dim;
        float *yb = y + b * dim;

        /* Find max (scalar is fine, memory-bound) */
        float max_val = xb[0];
        for (int64_t i = 1; i < dim; i++) {
            if (xb[i] > max_val) max_val = xb[i];
        }

        /* Vectorized exp(x - max) */
        float sum = 0.0f;
#ifdef __AVX2__
        __m256 vmax = _mm256_set1_ps(max_val);
        __m256 vsum = _mm256_setzero_ps();
        int64_t i = 0;
        for (; i + 8 <= dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(xb + i);
            __m256 ve = _mm256_exp_ps(_mm256_sub_ps(vx, vmax));
            _mm256_storeu_ps(yb + i, ve);
            vsum = _mm256_add_ps(vsum, ve);
        }
        /* Horizontal sum */
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        sum = _mm_cvtss_f32(lo);
        /* Scalar tail */
        for (; i < dim; i++) {
            yb[i] = expf(xb[i] - max_val);
            sum += yb[i];
        }
#else
        for (int64_t i = 0; i < dim; i++) {
            yb[i] = expf(xb[i] - max_val);
            sum += yb[i];
        }
#endif
        if (sum < 1e-12f) sum = 1e-12f;
        float inv_sum = 1.0f / sum;
        for (int64_t i = 0; i < dim; i++) {
            yb[i] *= inv_sum;
        }
    }
}

/* ── Concat ────────────────────────────────────────────────────────── */

void aria_concat_f32(const float **inputs, const int64_t *sizes,
                     int32_t n_inputs, float *output) {
    int64_t offset = 0;
    for (int32_t i = 0; i < n_inputs; i++) {
        memcpy(output + offset, inputs[i], (size_t)sizes[i] * sizeof(float));
        offset += sizes[i];
    }
}

/* ── Split ─────────────────────────────────────────────────────────── */

void aria_split_f32(const float *input, float **outputs,
                    const int64_t *sizes, int32_t n_outputs) {
    int64_t offset = 0;
    for (int32_t i = 0; i < n_outputs; i++) {
        memcpy(outputs[i], input + offset, (size_t)sizes[i] * sizeof(float));
        offset += sizes[i];
    }
}

/* ── LayerNorm ─────────────────────────────────────────────────────── */

void aria_layernorm_f32(const float *x, const float *weight, const float *bias,
                        float *y, int64_t batch, int64_t dim, float eps) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if((batch * dim) > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * dim;
        float *yb = y + b * dim;

        /* Compute mean */
        float mean = 0.0f;
        for (int64_t i = 0; i < dim; i++) {
            mean += xb[i];
        }
        mean /= (float)dim;

        /* Compute variance */
        float var = 0.0f;
        for (int64_t i = 0; i < dim; i++) {
            float d = xb[i] - mean;
            var += d * d;
        }
        var /= (float)dim;

        float inv_std = 1.0f / sqrtf(var + eps);

        /* Normalize, scale, shift */
        for (int64_t i = 0; i < dim; i++) {
            float normed = (xb[i] - mean) * inv_std;
            yb[i] = normed * weight[i] + (bias ? bias[i] : 0.0f);
        }
    }
}

/* ── Transpose 2D ──────────────────────────────────────────────────── */

void aria_transpose2d_f32(const float *input, float *output,
                           int64_t rows, int64_t cols) {
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

/* ── RMSNorm ───────────────────────────────────────────────────────── */

void aria_rmsnorm_f32(const float *x, const float *weight, float *y,
                      int64_t batch, int64_t dim, float eps) {
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if((batch * dim) > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        const float *xb = x + b * dim;
        float *yb = y + b * dim;

        /* Compute RMS = sqrt(mean(x^2)) */
        float ss = 0.0f;
        for (int64_t i = 0; i < dim; i++) {
            ss += xb[i] * xb[i];
        }
        float rms = sqrtf(ss / (float)dim + eps);
        float inv_rms = 1.0f / rms;

        /* Normalize and scale */
        for (int64_t i = 0; i < dim; i++) {
            yb[i] = xb[i] * inv_rms * weight[i];
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════
 * FUSED KERNELS
 *
 * These kernels fuse common op sequences to eliminate intermediate buffer
 * writes, reducing memory bandwidth pressure. The matmul portion uses BLAS
 * when available; the key optimization is applying activation/bias inline
 * on the output rather than writing → reading → writing again.
 * ══════════════════════════════════════════════════════════════════════ */

/* ── matmul + relu ─────────────────────────────────────────────────── */

void aria_matmul_relu_f32(const float *A, const float *B, float *C,
                           int64_t M, int64_t K, int64_t N) {
#ifdef ARIA_HAS_BLAS
    /* BLAS matmul into C, then fuse relu in a single pass */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K,
                1.0f, A, (int)K, B, (int)N,
                0.0f, C, (int)N);
    /* Fused relu pass — single traversal, no intermediate buffer */
    int64_t total = M * N;
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(total > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < total; i++) {
        if (C[i] < 0.0f) C[i] = 0.0f;
    }
#else
    /* Fallback: tiled matmul with inline relu after each output row */
    memset(C, 0, sizeof(float) * M * N);
    for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
        int64_t iend = i0 + TILE_M < M ? i0 + TILE_M : M;
        for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
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
    /* Apply relu inline */
    int64_t total = M * N;
    for (int64_t i = 0; i < total; i++) {
        if (C[i] < 0.0f) C[i] = 0.0f;
    }
#endif
}

/* ── matmul + bias + relu ──────────────────────────────────────────── */

void aria_matmul_bias_relu_f32(const float *A, const float *B,
                                const float *bias, float *C,
                                int64_t M, int64_t K, int64_t N) {
#ifdef ARIA_HAS_BLAS
    /* BLAS matmul into C */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K,
                1.0f, A, (int)K, B, (int)N,
                0.0f, C, (int)N);
    /* Single fused pass: add bias + relu */
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(M * N > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < M; i++) {
        float *Ci = C + i * N;
        for (int64_t j = 0; j < N; j++) {
            float v = Ci[j] + bias[j];
            Ci[j] = v > 0.0f ? v : 0.0f;
        }
    }
#else
    /* Fallback: tiled matmul then fused bias+relu */
    memset(C, 0, sizeof(float) * M * N);
    for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
        int64_t iend = i0 + TILE_M < M ? i0 + TILE_M : M;
        for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
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
    /* Fused bias + relu pass */
    for (int64_t i = 0; i < M; i++) {
        float *Ci = C + i * N;
        for (int64_t j = 0; j < N; j++) {
            float v = Ci[j] + bias[j];
            Ci[j] = v > 0.0f ? v : 0.0f;
        }
    }
#endif
}

/* ── layernorm + residual ──────────────────────────────────────────── */

void aria_layernorm_residual_f32(const float *x, const float *residual,
                                  const float *gamma, const float *beta,
                                  float *y, int64_t rows, int64_t cols,
                                  float eps) {
    /* Fuses: y = layernorm(x + residual, gamma, beta)
     * Uses output buffer as scratch: write (x + residual) into y first,
     * then compute layernorm in-place. Eliminates separate temp buffer
     * and reduces total memory passes vs sequential add + layernorm. */
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(rows > ARIA_OMP_BATCH_THRESHOLD) schedule(static)
#endif
    for (int64_t b = 0; b < rows; b++) {
        const float *xb = x + b * cols;
        const float *rb = residual + b * cols;
        float *yb = y + b * cols;

        /* Pass 1: fused add + mean accumulation */
        float mean = 0.0f;
        for (int64_t i = 0; i < cols; i++) {
            float v = xb[i] + rb[i];
            yb[i] = v;
            mean += v;
        }
        mean /= (float)cols;

        /* Pass 2: variance (read from yb, no redundant add) */
        float var = 0.0f;
        for (int64_t i = 0; i < cols; i++) {
            float d = yb[i] - mean;
            var += d * d;
        }
        var /= (float)cols;

        float inv_std = 1.0f / sqrtf(var + eps);

        /* Pass 3: normalize, scale, shift in-place */
        for (int64_t i = 0; i < cols; i++) {
            float normed = (yb[i] - mean) * inv_std;
            yb[i] = normed * gamma[i] + (beta ? beta[i] : 0.0f);
        }
    }
}

/* ── matmul + gelu ─────────────────────────────────────────────────── */

void aria_matmul_gelu_f32(const float *A, const float *B, float *C,
                           int64_t M, int64_t K, int64_t N) {
#ifdef ARIA_HAS_BLAS
    /* BLAS matmul into C */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K,
                1.0f, A, (int)K, B, (int)N,
                0.0f, C, (int)N);
    /* Fused GELU pass — single traversal over output */
    int64_t total = M * N;
#ifdef __AVX2__
    {
        const __m256 half   = _mm256_set1_ps(0.5f);
        const __m256 one    = _mm256_set1_ps(1.0f);
        const __m256 two    = _mm256_set1_ps(2.0f);
        const __m256 coeff  = _mm256_set1_ps(GELU_COEFF);
        const __m256 cubic  = _mm256_set1_ps(GELU_CUBIC);

        int64_t vec_end = total - (total % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(total > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vx = _mm256_loadu_ps(C + i);
            __m256 x2 = _mm256_mul_ps(vx, vx);
            __m256 x3 = _mm256_mul_ps(x2, vx);
            __m256 inner = _mm256_fmadd_ps(cubic, x3, vx);
            inner = _mm256_mul_ps(coeff, inner);
            __m256 two_inner = _mm256_mul_ps(two, inner);
            __m256 sig = _mm256_sigmoid_ps(two_inner);
            __m256 tanh_val = _mm256_fmsub_ps(two, sig, one);
            __m256 vy = _mm256_mul_ps(half, _mm256_mul_ps(vx, _mm256_add_ps(one, tanh_val)));
            _mm256_storeu_ps(C + i, vy);
        }
        for (int64_t i = vec_end; i < total; i++) {
            float v = C[i];
            float inner = GELU_COEFF * (v + GELU_CUBIC * v * v * v);
            C[i] = 0.5f * v * (1.0f + tanhf(inner));
        }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(total > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < total; i++) {
        float v = C[i];
        float inner = GELU_COEFF * (v + GELU_CUBIC * v * v * v);
        C[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
#endif
#else
    /* Fallback: tiled matmul then gelu */
    memset(C, 0, sizeof(float) * M * N);
    for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
        int64_t iend = i0 + TILE_M < M ? i0 + TILE_M : M;
        for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
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
    /* Apply gelu */
    int64_t total = M * N;
    for (int64_t i = 0; i < total; i++) {
        float v = C[i];
        float inner = GELU_COEFF * (v + GELU_CUBIC * v * v * v);
        C[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
#endif
}

/* ══════════════════════════════════════════════════════════════════════
 * FP16 (HALF-PRECISION) KERNELS
 *
 * Strategy: F16C convert-at-boundaries.
 * - Load fp16 (uint16_t) → convert to f32 via _mm256_cvtph_ps
 * - Compute in f32 (reuse existing SIMD paths)
 * - Convert f32 → fp16 via _mm256_cvtps_ph → store
 *
 * Scalar fallback uses union-based bit manipulation when F16C is not
 * available (should never happen on AVX2 hardware, but keeps it safe).
 * ══════════════════════════════════════════════════════════════════════ */

/* ── Scalar fp16 ↔ fp32 conversion (fallback) ────────────────────── */

static inline float aria_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            /* Denormalized: convert to normalized f32 */
            exp = 1;
            while (!(mant & 0x400)) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | (uint32_t)((127 - 15 + exp) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13); /* Inf/NaN */
    } else {
        f = sign | (uint32_t)((exp - 15 + 127) << 23) | (mant << 13);
    }
    union { uint32_t u; float f; } u;
    u.u = f;
    return u.f;
}

static inline uint16_t aria_f32_to_f16(float val) {
    union { float f; uint32_t u; } u;
    u.f = val;
    uint32_t f = u.u;
    uint32_t sign = (f >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = f & 0x7FFFFF;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;  /* Too small → zero */
        mant = (mant | 0x800000) >> (1 - exp);
        return (uint16_t)(sign | (mant >> 13));
    } else if (exp >= 31) {
        if (exp == 143 && mant) {
            return (uint16_t)(sign | 0x7C00 | (mant >> 13)); /* NaN */
        }
        return (uint16_t)(sign | 0x7C00);  /* Inf */
    }
    return (uint16_t)(sign | (uint32_t)(exp << 10) | (mant >> 13));
}

/* ── Unary fp16 kernels ──────────────────────────────────────────── */

void aria_relu_f16(const uint16_t *x, uint16_t *y, int64_t n) {
#ifdef __F16C__
    {
        int64_t vec_end = n - (n % 8);
        const __m256 zero = _mm256_setzero_ps();
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m128i vh = _mm_loadu_si128((const __m128i *)(x + i));
            __m256 vf = _mm256_cvtph_ps(vh);
            vf = _mm256_max_ps(vf, zero);
            __m128i out = _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(y + i), out);
        }
        for (int64_t i = vec_end; i < n; i++) {
            float v = aria_f16_to_f32(x[i]);
            y[i] = aria_f32_to_f16(v > 0.0f ? v : 0.0f);
        }
    }
#else
    for (int64_t i = 0; i < n; i++) {
        float v = aria_f16_to_f32(x[i]);
        y[i] = aria_f32_to_f16(v > 0.0f ? v : 0.0f);
    }
#endif
}

void aria_gelu_f16(const uint16_t *x, uint16_t *y, int64_t n) {
#if defined(__F16C__) && defined(__AVX2__)
    {
        const __m256 half  = _mm256_set1_ps(0.5f);
        const __m256 one   = _mm256_set1_ps(1.0f);
        const __m256 two   = _mm256_set1_ps(2.0f);
        const __m256 coeff = _mm256_set1_ps(GELU_COEFF);
        const __m256 cubic = _mm256_set1_ps(GELU_CUBIC);
        int64_t vec_end = n - (n % 8);
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m128i vh = _mm_loadu_si128((const __m128i *)(x + i));
            __m256 vx = _mm256_cvtph_ps(vh);
            __m256 x2 = _mm256_mul_ps(vx, vx);
            __m256 x3 = _mm256_mul_ps(x2, vx);
            __m256 inner = _mm256_fmadd_ps(cubic, x3, vx);
            inner = _mm256_mul_ps(coeff, inner);
            __m256 two_inner = _mm256_mul_ps(two, inner);
            __m256 sig = _mm256_sigmoid_ps(two_inner);
            __m256 tanh_val = _mm256_fmsub_ps(two, sig, one);
            __m256 vy = _mm256_mul_ps(half, _mm256_mul_ps(vx, _mm256_add_ps(one, tanh_val)));
            __m128i out = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(y + i), out);
        }
        for (int64_t i = vec_end; i < n; i++) {
            float v = aria_f16_to_f32(x[i]);
            float inner = GELU_COEFF * (v + GELU_CUBIC * v * v * v);
            y[i] = aria_f32_to_f16(0.5f * v * (1.0f + tanhf(inner)));
        }
    }
#else
    for (int64_t i = 0; i < n; i++) {
        float v = aria_f16_to_f32(x[i]);
        float inner = GELU_COEFF * (v + GELU_CUBIC * v * v * v);
        y[i] = aria_f32_to_f16(0.5f * v * (1.0f + tanhf(inner)));
    }
#endif
}

void aria_silu_f16(const uint16_t *x, uint16_t *y, int64_t n) {
#if defined(__F16C__) && defined(__AVX2__)
    {
        int64_t vec_end = n - (n % 8);
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m128i vh = _mm_loadu_si128((const __m128i *)(x + i));
            __m256 vx = _mm256_cvtph_ps(vh);
            __m256 sig = _mm256_sigmoid_ps(vx);
            __m256 vy = _mm256_mul_ps(vx, sig);
            __m128i out = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(y + i), out);
        }
        for (int64_t i = vec_end; i < n; i++) {
            float v = aria_f16_to_f32(x[i]);
            y[i] = aria_f32_to_f16(v / (1.0f + expf(-v)));
        }
    }
#else
    for (int64_t i = 0; i < n; i++) {
        float v = aria_f16_to_f32(x[i]);
        y[i] = aria_f32_to_f16(v / (1.0f + expf(-v)));
    }
#endif
}

void aria_sigmoid_f16(const uint16_t *x, uint16_t *y, int64_t n) {
#if defined(__F16C__) && defined(__AVX2__)
    {
        int64_t vec_end = n - (n % 8);
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m128i vh = _mm_loadu_si128((const __m128i *)(x + i));
            __m256 vx = _mm256_cvtph_ps(vh);
            __m256 vy = _mm256_sigmoid_ps(vx);
            __m128i out = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(y + i), out);
        }
        for (int64_t i = vec_end; i < n; i++) {
            float v = aria_f16_to_f32(x[i]);
            y[i] = aria_f32_to_f16(1.0f / (1.0f + expf(-v)));
        }
    }
#else
    for (int64_t i = 0; i < n; i++) {
        float v = aria_f16_to_f32(x[i]);
        y[i] = aria_f32_to_f16(1.0f / (1.0f + expf(-v)));
    }
#endif
}

/* ── Binary fp16 kernels ─────────────────────────────────────────── */

void aria_add_f16(const uint16_t *a, const uint16_t *b, uint16_t *y, int64_t n) {
#ifdef __F16C__
    {
        int64_t vec_end = n - (n % 8);
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m128i va_h = _mm_loadu_si128((const __m128i *)(a + i));
            __m128i vb_h = _mm_loadu_si128((const __m128i *)(b + i));
            __m256 va_f = _mm256_cvtph_ps(va_h);
            __m256 vb_f = _mm256_cvtph_ps(vb_h);
            __m256 vy = _mm256_add_ps(va_f, vb_f);
            __m128i out = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(y + i), out);
        }
        for (int64_t i = vec_end; i < n; i++) {
            float fa = aria_f16_to_f32(a[i]);
            float fb = aria_f16_to_f32(b[i]);
            y[i] = aria_f32_to_f16(fa + fb);
        }
    }
#else
    for (int64_t i = 0; i < n; i++) {
        float fa = aria_f16_to_f32(a[i]);
        float fb = aria_f16_to_f32(b[i]);
        y[i] = aria_f32_to_f16(fa + fb);
    }
#endif
}

void aria_mul_f16(const uint16_t *a, const uint16_t *b, uint16_t *y, int64_t n) {
#ifdef __F16C__
    {
        int64_t vec_end = n - (n % 8);
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m128i va_h = _mm_loadu_si128((const __m128i *)(a + i));
            __m128i vb_h = _mm_loadu_si128((const __m128i *)(b + i));
            __m256 va_f = _mm256_cvtph_ps(va_h);
            __m256 vb_f = _mm256_cvtph_ps(vb_h);
            __m256 vy = _mm256_mul_ps(va_f, vb_f);
            __m128i out = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(y + i), out);
        }
        for (int64_t i = vec_end; i < n; i++) {
            float fa = aria_f16_to_f32(a[i]);
            float fb = aria_f16_to_f32(b[i]);
            y[i] = aria_f32_to_f16(fa * fb);
        }
    }
#else
    for (int64_t i = 0; i < n; i++) {
        float fa = aria_f16_to_f32(a[i]);
        float fb = aria_f16_to_f32(b[i]);
        y[i] = aria_f32_to_f16(fa * fb);
    }
#endif
}

/* ── Matmul fp16 ─────────────────────────────────────────────────── */

void aria_matmul_f16(const uint16_t *A, const uint16_t *B, uint16_t *C,
                     int64_t M, int64_t K, int64_t N) {
    /* Convert inputs to f32, compute via f32 matmul, convert output back.
     * For large matrices this is memory-optimal vs. converting in tiles,
     * but for bandwidth savings the key win is halved storage on disk/transfer. */
    float *Af = (float *)malloc(sizeof(float) * (size_t)(M * K));
    float *Bf = (float *)malloc(sizeof(float) * (size_t)(K * N));
    float *Cf = (float *)malloc(sizeof(float) * (size_t)(M * N));

    /* Convert A: fp16 → f32 */
    int64_t total_a = M * K;
#ifdef __F16C__
    {
        int64_t vec_end = total_a - (total_a % 8);
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m128i vh = _mm_loadu_si128((const __m128i *)(A + i));
            __m256 vf = _mm256_cvtph_ps(vh);
            _mm256_storeu_ps(Af + i, vf);
        }
        for (int64_t i = vec_end; i < total_a; i++) {
            Af[i] = aria_f16_to_f32(A[i]);
        }
    }
#else
    for (int64_t i = 0; i < total_a; i++) Af[i] = aria_f16_to_f32(A[i]);
#endif

    /* Convert B: fp16 → f32 */
    int64_t total_b = K * N;
#ifdef __F16C__
    {
        int64_t vec_end = total_b - (total_b % 8);
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m128i vh = _mm_loadu_si128((const __m128i *)(B + i));
            __m256 vf = _mm256_cvtph_ps(vh);
            _mm256_storeu_ps(Bf + i, vf);
        }
        for (int64_t i = vec_end; i < total_b; i++) {
            Bf[i] = aria_f16_to_f32(B[i]);
        }
    }
#else
    for (int64_t i = 0; i < total_b; i++) Bf[i] = aria_f16_to_f32(B[i]);
#endif

    /* Compute in f32 (reuse BLAS/tiled path) */
    aria_matmul_f32(Af, Bf, Cf, M, K, N);

    /* Convert C: f32 → fp16 */
    int64_t total_c = M * N;
#ifdef __F16C__
    {
        int64_t vec_end = total_c - (total_c % 8);
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vf = _mm256_loadu_ps(Cf + i);
            __m128i vh = _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(C + i), vh);
        }
        for (int64_t i = vec_end; i < total_c; i++) {
            C[i] = aria_f32_to_f16(Cf[i]);
        }
    }
#else
    for (int64_t i = 0; i < total_c; i++) C[i] = aria_f32_to_f16(Cf[i]);
#endif

    free(Af);
    free(Bf);
    free(Cf);
}

/* ── Softmax fp16 ────────────────────────────────────────────────── */

void aria_softmax_f16(const uint16_t *x, uint16_t *y, int64_t batch, int64_t dim) {
    /* Convert row to f32, run softmax, convert back */
    float *xf = (float *)malloc(sizeof(float) * (size_t)dim);
    float *yf = (float *)malloc(sizeof(float) * (size_t)dim);
    for (int64_t b = 0; b < batch; b++) {
        const uint16_t *xb = x + b * dim;
        uint16_t *yb = y + b * dim;
        /* Convert row to f32 */
#ifdef __F16C__
        {
            int64_t vec_end = dim - (dim % 8);
            for (int64_t i = 0; i < vec_end; i += 8) {
                __m128i vh = _mm_loadu_si128((const __m128i *)(xb + i));
                _mm256_storeu_ps(xf + i, _mm256_cvtph_ps(vh));
            }
            for (int64_t i = vec_end; i < dim; i++) xf[i] = aria_f16_to_f32(xb[i]);
        }
#else
        for (int64_t i = 0; i < dim; i++) xf[i] = aria_f16_to_f32(xb[i]);
#endif
        /* Compute softmax in f32 (single row) */
        aria_softmax_f32(xf, yf, 1, dim);
        /* Convert back to fp16 */
#ifdef __F16C__
        {
            int64_t vec_end = dim - (dim % 8);
            for (int64_t i = 0; i < vec_end; i += 8) {
                __m256 vf = _mm256_loadu_ps(yf + i);
                _mm_storeu_si128((__m128i *)(yb + i),
                                 _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT));
            }
            for (int64_t i = vec_end; i < dim; i++) yb[i] = aria_f32_to_f16(yf[i]);
        }
#else
        for (int64_t i = 0; i < dim; i++) yb[i] = aria_f32_to_f16(yf[i]);
#endif
    }
    free(xf);
    free(yf);
}

/* ── RMSNorm fp16 ────────────────────────────────────────────────── */

void aria_rmsnorm_f16(const uint16_t *x, const uint16_t *weight, uint16_t *y,
                      int64_t batch, int64_t dim, float eps) {
    /* Convert weight to f32 once */
    float *wf = (float *)malloc(sizeof(float) * (size_t)dim);
#ifdef __F16C__
    {
        int64_t vec_end = dim - (dim % 8);
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m128i vh = _mm_loadu_si128((const __m128i *)(weight + i));
            _mm256_storeu_ps(wf + i, _mm256_cvtph_ps(vh));
        }
        for (int64_t i = vec_end; i < dim; i++) wf[i] = aria_f16_to_f32(weight[i]);
    }
#else
    for (int64_t i = 0; i < dim; i++) wf[i] = aria_f16_to_f32(weight[i]);
#endif

    float *xf = (float *)malloc(sizeof(float) * (size_t)dim);
    float *yf = (float *)malloc(sizeof(float) * (size_t)dim);
    for (int64_t b = 0; b < batch; b++) {
        const uint16_t *xb = x + b * dim;
        uint16_t *yb = y + b * dim;
        /* Convert input row */
#ifdef __F16C__
        {
            int64_t vec_end = dim - (dim % 8);
            for (int64_t i = 0; i < vec_end; i += 8) {
                __m128i vh = _mm_loadu_si128((const __m128i *)(xb + i));
                _mm256_storeu_ps(xf + i, _mm256_cvtph_ps(vh));
            }
            for (int64_t i = vec_end; i < dim; i++) xf[i] = aria_f16_to_f32(xb[i]);
        }
#else
        for (int64_t i = 0; i < dim; i++) xf[i] = aria_f16_to_f32(xb[i]);
#endif
        /* Compute RMSNorm in f32 (single row) */
        aria_rmsnorm_f32(xf, wf, yf, 1, dim, eps);
        /* Convert output back */
#ifdef __F16C__
        {
            int64_t vec_end = dim - (dim % 8);
            for (int64_t i = 0; i < vec_end; i += 8) {
                __m256 vf = _mm256_loadu_ps(yf + i);
                _mm_storeu_si128((__m128i *)(yb + i),
                                 _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT));
            }
            for (int64_t i = vec_end; i < dim; i++) yb[i] = aria_f32_to_f16(yf[i]);
        }
#else
        for (int64_t i = 0; i < dim; i++) yb[i] = aria_f32_to_f16(yf[i]);
#endif
    }
    free(wf);
    free(xf);
    free(yf);
}

/* ══════════════════════════════════════════════════════════════════════
 * BACKWARD (GRADIENT) KERNELS
 *
 * Naming convention: aria_<op>_backward_f32(...)
 * All backward kernels take grad_output as first argument and write
 * gradients w.r.t. the forward inputs.
 * ══════════════════════════════════════════════════════════════════════ */

/* ── Unary backward ops ───────────────────────────────────────────── */

void aria_relu_backward_f32(const float *grad_out, const float *input,
                             float *grad_in, int64_t n) {
    /* grad_in = grad_out * (input > 0 ? 1 : 0) */
#ifdef __AVX2__
    {
        const __m256 zero = _mm256_setzero_ps();
        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vx = _mm256_loadu_ps(input + i);
            __m256 vg = _mm256_loadu_ps(grad_out + i);
            /* mask: input > 0 */
            __m256 mask = _mm256_cmp_ps(vx, zero, _CMP_GT_OQ);
            __m256 result = _mm256_and_ps(vg, mask);
            _mm256_storeu_ps(grad_in + i, result);
        }
        for (int64_t i = vec_end; i < n; i++) {
            grad_in[i] = input[i] > 0.0f ? grad_out[i] : 0.0f;
        }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        grad_in[i] = input[i] > 0.0f ? grad_out[i] : 0.0f;
    }
#endif
}

void aria_sigmoid_backward_f32(const float *grad_out, const float *output,
                                float *grad_in, int64_t n) {
    /* grad_in = grad_out * output * (1 - output) */
#ifdef __AVX2__
    {
        const __m256 one = _mm256_set1_ps(1.0f);
        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vo = _mm256_loadu_ps(output + i);
            __m256 vg = _mm256_loadu_ps(grad_out + i);
            __m256 one_minus_o = _mm256_sub_ps(one, vo);
            __m256 result = _mm256_mul_ps(vg, _mm256_mul_ps(vo, one_minus_o));
            _mm256_storeu_ps(grad_in + i, result);
        }
        for (int64_t i = vec_end; i < n; i++) {
            float o = output[i];
            grad_in[i] = grad_out[i] * o * (1.0f - o);
        }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        float o = output[i];
        grad_in[i] = grad_out[i] * o * (1.0f - o);
    }
#endif
}

void aria_tanh_backward_f32(const float *grad_out, const float *output,
                             float *grad_in, int64_t n) {
    /* grad_in = grad_out * (1 - output^2) */
#ifdef __AVX2__
    {
        const __m256 one = _mm256_set1_ps(1.0f);
        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vo = _mm256_loadu_ps(output + i);
            __m256 vg = _mm256_loadu_ps(grad_out + i);
            __m256 o2 = _mm256_mul_ps(vo, vo);
            __m256 result = _mm256_mul_ps(vg, _mm256_sub_ps(one, o2));
            _mm256_storeu_ps(grad_in + i, result);
        }
        for (int64_t i = vec_end; i < n; i++) {
            float o = output[i];
            grad_in[i] = grad_out[i] * (1.0f - o * o);
        }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        float o = output[i];
        grad_in[i] = grad_out[i] * (1.0f - o * o);
    }
#endif
}

void aria_gelu_backward_f32(const float *grad_out, const float *input,
                             float *grad_in, int64_t n) {
    /* GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
     * Let inner = sqrt(2/pi) * (x + 0.044715 * x^3)
     * Let t = tanh(inner)
     * d(GELU)/dx = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
     */
#ifdef __AVX2__
    {
        const __m256 half   = _mm256_set1_ps(0.5f);
        const __m256 one    = _mm256_set1_ps(1.0f);
        const __m256 two    = _mm256_set1_ps(2.0f);
        const __m256 three  = _mm256_set1_ps(3.0f);
        const __m256 coeff  = _mm256_set1_ps(GELU_COEFF);
        const __m256 cubic  = _mm256_set1_ps(GELU_CUBIC);

        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vx = _mm256_loadu_ps(input + i);
            __m256 vg = _mm256_loadu_ps(grad_out + i);

            /* inner = sqrt(2/pi) * (x + 0.044715 * x^3) */
            __m256 x2 = _mm256_mul_ps(vx, vx);
            __m256 x3 = _mm256_mul_ps(x2, vx);
            __m256 inner = _mm256_fmadd_ps(cubic, x3, vx);
            inner = _mm256_mul_ps(coeff, inner);

            /* tanh(inner) via 2*sigmoid(2*inner) - 1 */
            __m256 two_inner = _mm256_mul_ps(two, inner);
            __m256 sig = _mm256_sigmoid_ps(two_inner);
            __m256 t = _mm256_fmsub_ps(two, sig, one);

            /* d_inner_dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2) */
            __m256 d_inner = _mm256_fmadd_ps(_mm256_mul_ps(three, cubic), x2, one);
            d_inner = _mm256_mul_ps(coeff, d_inner);

            /* dgelu = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * d_inner */
            __m256 t2 = _mm256_mul_ps(t, t);
            __m256 one_minus_t2 = _mm256_sub_ps(one, t2);
            __m256 term1 = _mm256_mul_ps(half, _mm256_add_ps(one, t));
            __m256 term2 = _mm256_mul_ps(half, _mm256_mul_ps(vx, _mm256_mul_ps(one_minus_t2, d_inner)));
            __m256 dgelu = _mm256_add_ps(term1, term2);

            _mm256_storeu_ps(grad_in + i, _mm256_mul_ps(vg, dgelu));
        }
        for (int64_t i = vec_end; i < n; i++) {
            float v = input[i];
            float inner = GELU_COEFF * (v + GELU_CUBIC * v * v * v);
            float t = tanhf(inner);
            float d_inner = GELU_COEFF * (1.0f + 3.0f * GELU_CUBIC * v * v);
            float dgelu = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * d_inner;
            grad_in[i] = grad_out[i] * dgelu;
        }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        float v = input[i];
        float inner = GELU_COEFF * (v + GELU_CUBIC * v * v * v);
        float t = tanhf(inner);
        float d_inner = GELU_COEFF * (1.0f + 3.0f * GELU_CUBIC * v * v);
        float dgelu = 0.5f * (1.0f + t) + 0.5f * v * (1.0f - t * t) * d_inner;
        grad_in[i] = grad_out[i] * dgelu;
    }
#endif
}

void aria_silu_backward_f32(const float *grad_out, const float *input,
                             float *grad_in, int64_t n) {
    /* SiLU(x) = x * sigmoid(x)
     * d(SiLU)/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
     *            = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
     */
#ifdef __AVX2__
    {
        const __m256 one = _mm256_set1_ps(1.0f);
        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vx = _mm256_loadu_ps(input + i);
            __m256 vg = _mm256_loadu_ps(grad_out + i);
            __m256 sig = _mm256_sigmoid_ps(vx);
            /* dsilu = sig * (1 + x * (1 - sig)) */
            __m256 one_minus_sig = _mm256_sub_ps(one, sig);
            __m256 x_term = _mm256_fmadd_ps(vx, one_minus_sig, one);
            __m256 dsilu = _mm256_mul_ps(sig, x_term);
            _mm256_storeu_ps(grad_in + i, _mm256_mul_ps(vg, dsilu));
        }
        for (int64_t i = vec_end; i < n; i++) {
            float v = input[i];
            float sig = 1.0f / (1.0f + expf(-v));
            float dsilu = sig * (1.0f + v * (1.0f - sig));
            grad_in[i] = grad_out[i] * dsilu;
        }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        float v = input[i];
        float sig = 1.0f / (1.0f + expf(-v));
        float dsilu = sig * (1.0f + v * (1.0f - sig));
        grad_in[i] = grad_out[i] * dsilu;
    }
#endif
}

/* ── Binary backward ops ──────────────────────────────────────────── */

void aria_add_backward_f32(const float *grad_out,
                            float *grad_a, float *grad_b, int64_t n) {
    /* add(a, b) = a + b  =>  grad_a = grad_out, grad_b = grad_out */
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        grad_a[i] = grad_out[i];
        grad_b[i] = grad_out[i];
    }
}

void aria_mul_backward_f32(const float *grad_out,
                            const float *a, const float *b,
                            float *grad_a, float *grad_b, int64_t n) {
    /* mul(a, b) = a * b  =>  grad_a = grad_out * b, grad_b = grad_out * a */
#ifdef __AVX2__
    {
        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vg = _mm256_loadu_ps(grad_out + i);
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(grad_a + i, _mm256_mul_ps(vg, vb));
            _mm256_storeu_ps(grad_b + i, _mm256_mul_ps(vg, va));
        }
        for (int64_t i = vec_end; i < n; i++) {
            grad_a[i] = grad_out[i] * b[i];
            grad_b[i] = grad_out[i] * a[i];
        }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        grad_a[i] = grad_out[i] * b[i];
        grad_b[i] = grad_out[i] * a[i];
    }
#endif
}

void aria_sub_backward_f32(const float *grad_out,
                            float *grad_a, float *grad_b, int64_t n) {
    /* sub(a, b) = a - b  =>  grad_a = grad_out, grad_b = -grad_out */
#ifdef __AVX2__
    {
        const __m256 neg_one = _mm256_set1_ps(-1.0f);
        int64_t vec_end = n - (n % 8);
#ifdef ARIA_HAS_OPENMP
        #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
        for (int64_t i = 0; i < vec_end; i += 8) {
            __m256 vg = _mm256_loadu_ps(grad_out + i);
            _mm256_storeu_ps(grad_a + i, vg);
            _mm256_storeu_ps(grad_b + i, _mm256_mul_ps(vg, neg_one));
        }
        for (int64_t i = vec_end; i < n; i++) {
            grad_a[i] = grad_out[i];
            grad_b[i] = -grad_out[i];
        }
    }
#else
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(n > ARIA_OMP_THRESHOLD) schedule(static)
#endif
    for (int64_t i = 0; i < n; i++) {
        grad_a[i] = grad_out[i];
        grad_b[i] = -grad_out[i];
    }
#endif
}

/* ── Matmul backward ──────────────────────────────────────────────── */

void aria_matmul_backward_f32(const float *grad_out,
                               const float *A, const float *B,
                               float *grad_A, float *grad_B,
                               int64_t M, int64_t K, int64_t N) {
    /* C = A[M,K] @ B[K,N]
     * grad_A = grad_out[M,N] @ B^T[N,K]  =>  [M,K]
     * grad_B = A^T[K,M] @ grad_out[M,N]  =>  [K,N]
     */
#ifdef ARIA_HAS_BLAS
    /* grad_A = grad_out @ B^T : grad_out[M,N] x B^T[N,K] = grad_A[M,K] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)M, (int)K, (int)N,
                1.0f, grad_out, (int)N, B, (int)N,
                0.0f, grad_A, (int)K);
    /* grad_B = A^T @ grad_out : A^T[K,M] x grad_out[M,N] = grad_B[K,N] */
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                (int)K, (int)N, (int)M,
                1.0f, A, (int)K, grad_out, (int)N,
                0.0f, grad_B, (int)N);
#else
    /* Fallback: naive matmul for grad_A = grad_out @ B^T */
    memset(grad_A, 0, sizeof(float) * M * K);
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float g = grad_out[i * N + j];
            for (int64_t k = 0; k < K; k++) {
                grad_A[i * K + k] += g * B[k * N + j];
            }
        }
    }
    /* Fallback: naive matmul for grad_B = A^T @ grad_out */
    memset(grad_B, 0, sizeof(float) * K * N);
    for (int64_t k = 0; k < K; k++) {
        for (int64_t i = 0; i < M; i++) {
            float a_val = A[i * K + k];
            for (int64_t j = 0; j < N; j++) {
                grad_B[k * N + j] += a_val * grad_out[i * N + j];
            }
        }
    }
#endif
}

/* ── Softmax backward ─────────────────────────────────────────────── */

void aria_softmax_backward_f32(const float *grad_out, const float *output,
                                float *grad_in, int64_t batch, int64_t dim) {
    /* dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))  per row */
#ifdef ARIA_HAS_OPENMP
    #pragma omp parallel for if(batch > ARIA_OMP_BATCH_THRESHOLD) schedule(static)
#endif
    for (int64_t b = 0; b < batch; b++) {
        const float *go = grad_out + b * dim;
        const float *y  = output + b * dim;
        float *gi       = grad_in + b * dim;

        /* dot = sum(dL/dy * y) */
        float dot = 0.0f;
        for (int64_t i = 0; i < dim; i++) {
            dot += go[i] * y[i];
        }

        /* grad_in = y * (grad_out - dot) */
        for (int64_t i = 0; i < dim; i++) {
            gi[i] = y[i] * (go[i] - dot);
        }
    }
}

/* ── LayerNorm backward ───────────────────────────────────────────── */

void aria_layernorm_backward_f32(const float *grad_out, const float *input,
                                  const float *gamma,
                                  float *grad_in, float *grad_gamma,
                                  float *grad_beta,
                                  int64_t batch, int64_t dim, float eps) {
    /*
     * Forward: y = gamma * (x - mean) / sqrt(var + eps) + beta
     * Let x_hat = (x - mean) * inv_std
     *
     * grad_gamma = sum_over_batch(grad_out * x_hat)         [dim]
     * grad_beta  = sum_over_batch(grad_out)                 [dim]
     * grad_in    = inv_std * (grad_out * gamma
     *              - mean(grad_out * gamma)
     *              - x_hat * mean(grad_out * gamma * x_hat))
     */

    /* Zero-init parameter gradients (accumulated across batch) */
    memset(grad_gamma, 0, sizeof(float) * dim);
    memset(grad_beta, 0, sizeof(float) * dim);

    /* NOTE: We don't parallelize the outer loop because grad_gamma/grad_beta
     * are shared accumulators. Instead, each row is processed sequentially
     * for grad_gamma/grad_beta, but the inner loops are fast (memory-bound). */
    for (int64_t b = 0; b < batch; b++) {
        const float *go = grad_out + b * dim;
        const float *x  = input + b * dim;
        float *gi       = grad_in + b * dim;

        /* Recompute mean and variance (same as forward) */
        float mean = 0.0f;
        for (int64_t i = 0; i < dim; i++) {
            mean += x[i];
        }
        mean /= (float)dim;

        float var = 0.0f;
        for (int64_t i = 0; i < dim; i++) {
            float d = x[i] - mean;
            var += d * d;
        }
        var /= (float)dim;

        float inv_std = 1.0f / sqrtf(var + eps);

        /* Accumulate grad_gamma and grad_beta */
        for (int64_t i = 0; i < dim; i++) {
            float x_hat = (x[i] - mean) * inv_std;
            grad_gamma[i] += go[i] * x_hat;
            grad_beta[i]  += go[i];
        }

        /* Compute grad_in for this row:
         * Let g = grad_out * gamma
         * mean_g = mean(g)
         * mean_gx = mean(g * x_hat)
         * grad_in = inv_std * (g - mean_g - x_hat * mean_gx)
         */
        float mean_g = 0.0f;
        float mean_gx = 0.0f;
        for (int64_t i = 0; i < dim; i++) {
            float g = go[i] * gamma[i];
            float x_hat = (x[i] - mean) * inv_std;
            mean_g += g;
            mean_gx += g * x_hat;
        }
        mean_g /= (float)dim;
        mean_gx /= (float)dim;

        for (int64_t i = 0; i < dim; i++) {
            float g = go[i] * gamma[i];
            float x_hat = (x[i] - mean) * inv_std;
            gi[i] = inv_std * (g - mean_g - x_hat * mean_gx);
        }
    }
}

/* ── RMSNorm backward ─────────────────────────────────────────────── */

void aria_rmsnorm_backward_f32(const float *grad_out, const float *input,
                                const float *gamma,
                                float *grad_in, float *grad_gamma,
                                int64_t batch, int64_t dim, float eps) {
    /*
     * Forward: y = gamma * x / rms,  rms = sqrt(mean(x^2) + eps)
     *
     * grad_gamma = sum_over_batch(grad_out * x / rms)       [dim]
     * grad_in    = gamma / rms * (grad_out - x * mean(grad_out * gamma * x) / rms^2)
     */

    /* Zero-init parameter gradient */
    memset(grad_gamma, 0, sizeof(float) * dim);

    for (int64_t b = 0; b < batch; b++) {
        const float *go = grad_out + b * dim;
        const float *x  = input + b * dim;
        float *gi       = grad_in + b * dim;

        /* Recompute RMS (same as forward) */
        float ss = 0.0f;
        for (int64_t i = 0; i < dim; i++) {
            ss += x[i] * x[i];
        }
        float rms_sq = ss / (float)dim + eps;
        float rms = sqrtf(rms_sq);
        float inv_rms = 1.0f / rms;

        /* Accumulate grad_gamma */
        for (int64_t i = 0; i < dim; i++) {
            grad_gamma[i] += go[i] * x[i] * inv_rms;
        }

        /* Compute grad_in:
         * Let g = grad_out * gamma
         * mean_gx = mean(g * x) = (1/dim) * sum(g * x)
         * grad_in = inv_rms * (g - x * mean_gx / rms^2)
         *         = inv_rms * g - x * mean_gx / (rms * rms^2)
         *         = inv_rms * g - x * mean_gx * inv_rms / rms^2
         */
        float sum_gx = 0.0f;
        for (int64_t i = 0; i < dim; i++) {
            sum_gx += go[i] * gamma[i] * x[i];
        }
        float mean_gx = sum_gx / (float)dim;
        float coeff = mean_gx / rms_sq;  /* mean_gx / rms^2 */

        for (int64_t i = 0; i < dim; i++) {
            float g = go[i] * gamma[i];
            gi[i] = inv_rms * (g - x[i] * coeff);
        }
    }
}
