/**
 * kernels.h — Optimized tensor operation kernels.
 *
 * All kernels operate on flat float arrays with explicit strides.
 * SIMD-optimized where beneficial (SSE/AVX auto-vectorized).
 */
#ifndef ARIA_KERNELS_H
#define ARIA_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Elementwise unary ─────────────────────────────────────────────── */

void aria_relu_f32(const float *x, float *y, int64_t n);
void aria_gelu_f32(const float *x, float *y, int64_t n);
void aria_silu_f32(const float *x, float *y, int64_t n);
void aria_sin_f32(const float *x, float *y, int64_t n);
void aria_cos_f32(const float *x, float *y, int64_t n);
void aria_tanh_f32(const float *x, float *y, int64_t n);
void aria_sigmoid_f32(const float *x, float *y, int64_t n);
void aria_exp_f32(const float *x, float *y, int64_t n);

/* ── Elementwise binary ────────────────────────────────────────────── */

void aria_add_f32(const float *a, const float *b, float *y, int64_t n);
void aria_mul_f32(const float *a, const float *b, float *y, int64_t n);
void aria_sub_f32(const float *a, const float *b, float *y, int64_t n);
void aria_tropical_add_f32(const float *a, const float *b, float *y, int64_t n);

/* ── Reductions ────────────────────────────────────────────────────── */

float aria_sum_f32(const float *x, int64_t n);
float aria_mean_f32(const float *x, int64_t n);

/* ── Linear algebra ────────────────────────────────────────────────── */

/**
 * Dense matrix multiply: C[M,N] = A[M,K] @ B[K,N]
 * Tiled for cache efficiency.
 */
void aria_matmul_f32(const float *A, const float *B, float *C,
                     int64_t M, int64_t K, int64_t N);
void aria_tropical_matmul_f32(const float *A, const float *B, float *C,
                              int64_t M, int64_t K, int64_t N);

/**
 * Linear projection: y = x @ W^T + bias
 * x: [batch, dim_in], W: [dim_out, dim_in], bias: [dim_out] (may be NULL)
 */
void aria_linear_f32(const float *x, const float *W, const float *bias,
                     float *y, int64_t batch, int64_t dim_in, int64_t dim_out);

/* ── IO ────────────────────────────────────────────────────────────── */

int aria_read_csv_f32(const char *filename, float *out_data, int64_t max_rows, int64_t max_cols, char delimiter);
int aria_filter_f32(const float *data, float *out_data, int64_t rows, int64_t cols, int64_t col_idx, float val, int op);
int aria_file_loader_csv_f32(const char *filename, float *out_data,
                             int64_t max_rows, int64_t max_cols,
                             char delimiter, int has_header);
int aria_binary_file_reader_f32(const char *filename, float *out_data,
                                int64_t max_elems, int64_t offset_bytes);
int aria_file_writer_txt_f32(const char *filename, const float *data,
                             int64_t n, int overwrite);

/* ── Normalization ─────────────────────────────────────────────────── */

/**
 * RMSNorm: y = x / rms(x) * weight
 * x: [batch, dim], weight: [dim]
 */
void aria_rmsnorm_f32(const float *x, const float *weight, float *y,
                      int64_t batch, int64_t dim, float eps);

/* ── Math space (tropical) ────────────────────────────────────────── */

void aria_tropical_center_f32(const float *x, float *y,
                              int64_t batch, int64_t seq, int64_t dim);

/* ── Math space (hyperbolic + p-adic) ─────────────────────────────── */

void aria_hyp_distance_f32(const float *x, const float *y, float *out,
                           int64_t batch, int64_t seq, int64_t dim);

void aria_padic_gate_f32(const float *x, float *y, int64_t n, float p);

void aria_tropical_attention_f32(const float *x, float *y,
                                 int64_t batch, int64_t seq, int64_t dim,
                                 float temperature);

void aria_tropical_gate_f32(const float *x, float *y,
                            int64_t batch, int64_t seq, int64_t dim,
                            float temperature);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_KERNELS_H */
