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
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>

/* ── Constants ─────────────────────────────────────────────────────── */

static const float GELU_COEFF = 0.7978845608028654f;  /* sqrt(2/pi) */
static const float GELU_CUBIC = 0.044715f;

/* ── Elementwise unary ─────────────────────────────────────────────── */

void aria_relu_f32(const float *x, float *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

void aria_gelu_f32(const float *x, float *y, int64_t n) {
    /* GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        float inner = GELU_COEFF * (v + GELU_CUBIC * v * v * v);
        y[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

void aria_silu_f32(const float *x, float *y, int64_t n) {
    /* SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)) */
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v / (1.0f + expf(-v));
    }
}

void aria_sin_f32(const float *x, float *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = sinf(x[i]);
    }
}

void aria_cos_f32(const float *x, float *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = cosf(x[i]);
    }
}

void aria_tanh_f32(const float *x, float *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = tanhf(x[i]);
    }
}

void aria_sigmoid_f32(const float *x, float *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

void aria_exp_f32(const float *x, float *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = expf(x[i]);
    }
}

/* ── Elementwise binary ────────────────────────────────────────────── */

void aria_add_f32(const float *a, const float *b, float *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = a[i] + b[i];
    }
}

void aria_mul_f32(const float *a, const float *b, float *y, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        y[i] = a[i] * b[i];
    }
}

void aria_sub_f32(const float *a, const float *b, float *y, int64_t n) {
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
    /* C[M,N] = A[M,K] @ B[K,N], tiled for L1 cache */
    memset(C, 0, sizeof(float) * M * N);

    for (int64_t i0 = 0; i0 < M; i0 += TILE_M) {
        int64_t iend = i0 + TILE_M < M ? i0 + TILE_M : M;

        for (int64_t k0 = 0; k0 < K; k0 += TILE_K) {
            int64_t kend = k0 + TILE_K < K ? k0 + TILE_K : K;

            for (int64_t j0 = 0; j0 < N; j0 += TILE_N) {
                int64_t jend = j0 + TILE_N < N ? j0 + TILE_N : N;

                /* Micro-kernel: C[i,j] += A[i,k] * B[k,j] */
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

/* ── RMSNorm ───────────────────────────────────────────────────────── */

void aria_rmsnorm_f32(const float *x, const float *weight, float *y,
                      int64_t batch, int64_t dim, float eps) {
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
