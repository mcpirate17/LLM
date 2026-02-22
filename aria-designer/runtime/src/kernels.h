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
void aria_square_f32(const float *x, float *y, int64_t n);
void aria_abs_f32(const float *x, float *y, int64_t n);
void aria_neg_f32(const float *x, float *y, int64_t n);
void aria_reciprocal_f32(const float *x, float *y, int64_t n);
void aria_log_f32(const float *x, float *y, int64_t n);
void aria_sqrt_f32(const float *x, float *y, int64_t n);
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

/* ── Softmax ──────────────────────────────────────────────────────── */
void aria_softmax_f32(const float *x, float *y, int64_t batch, int64_t dim);

/* ── LayerNorm ────────────────────────────────────────────────────── */
void aria_layernorm_f32(const float *x, const float *weight, const float *bias,
                        float *y, int64_t batch, int64_t dim, float eps);

/* ── Structural ops ───────────────────────────────────────────────── */
void aria_concat_f32(const float **inputs, const int64_t *sizes,
                     int32_t n_inputs, float *output);
void aria_split_f32(const float *input, float **outputs,
                    const int64_t *sizes, int32_t n_outputs);
void aria_transpose2d_f32(const float *input, float *output,
                           int64_t rows, int64_t cols);

/* ── Fused kernels ────────────────────────────────────────────────
 *
 * These kernels fuse common op sequences (matmul+activation, layernorm+residual)
 * to eliminate intermediate buffer writes, reducing memory bandwidth.
 * The matmul portion uses BLAS when available (ARIA_HAS_BLAS).
 * ──────────────────────────────────────────────────────────────────── */

/** matmul + relu: C[M,N] = max(0, A[M,K] @ B[K,N]) */
void aria_matmul_relu_f32(const float *A, const float *B, float *C,
                           int64_t M, int64_t K, int64_t N);

/** matmul + bias + relu: C[M,N] = max(0, A[M,K] @ B[K,N] + bias[N]) */
void aria_matmul_bias_relu_f32(const float *A, const float *B,
                                const float *bias, float *C,
                                int64_t M, int64_t K, int64_t N);

/** layernorm(x + residual): y = layernorm(x + residual, gamma, beta)
 *  x, residual: [rows, cols], gamma, beta: [cols], y: [rows, cols] */
void aria_layernorm_residual_f32(const float *x, const float *residual,
                                  const float *gamma, const float *beta,
                                  float *y, int64_t rows, int64_t cols,
                                  float eps);

/** matmul + gelu: C[M,N] = gelu(A[M,K] @ B[K,N]) */
void aria_matmul_gelu_f32(const float *A, const float *B, float *C,
                           int64_t M, int64_t K, int64_t N);

/* ── FP16 (half-precision) kernels ────────────────────────────────
 *
 * fp16 stored as uint16_t. Compute internally in f32 using F16C
 * convert-at-boundaries: cvtph→f32, compute, cvtps→f16.
 * This halves memory bandwidth; compute stays f32.
 * ──────────────────────────────────────────────────────────────────── */

/* Unary fp16 */
void aria_relu_f16(const uint16_t *x, uint16_t *y, int64_t n);
void aria_gelu_f16(const uint16_t *x, uint16_t *y, int64_t n);
void aria_silu_f16(const uint16_t *x, uint16_t *y, int64_t n);
void aria_sigmoid_f16(const uint16_t *x, uint16_t *y, int64_t n);

/* Binary fp16 */
void aria_add_f16(const uint16_t *a, const uint16_t *b, uint16_t *y, int64_t n);
void aria_mul_f16(const uint16_t *a, const uint16_t *b, uint16_t *y, int64_t n);

/* Matmul fp16: C[M,N] = A[M,K] @ B[K,N] */
void aria_matmul_f16(const uint16_t *A, const uint16_t *B, uint16_t *C,
                     int64_t M, int64_t K, int64_t N);

/* Softmax fp16 */
void aria_softmax_f16(const uint16_t *x, uint16_t *y, int64_t batch, int64_t dim);

/* RMSNorm fp16 */
void aria_rmsnorm_f16(const uint16_t *x, const uint16_t *weight, uint16_t *y,
                      int64_t batch, int64_t dim, float eps);

/* ── Backward (gradient) kernels ─────────────────────────────────── */

/* Unary backward: grad_in = f'(input_or_output) * grad_out */
void aria_relu_backward_f32(const float *grad_out, const float *input,
                             float *grad_in, int64_t n);
void aria_sigmoid_backward_f32(const float *grad_out, const float *output,
                                float *grad_in, int64_t n);
void aria_tanh_backward_f32(const float *grad_out, const float *output,
                             float *grad_in, int64_t n);
void aria_gelu_backward_f32(const float *grad_out, const float *input,
                             float *grad_in, int64_t n);
void aria_silu_backward_f32(const float *grad_out, const float *input,
                             float *grad_in, int64_t n);

/* Binary backward: compute gradients w.r.t. both inputs */
void aria_add_backward_f32(const float *grad_out,
                            float *grad_a, float *grad_b, int64_t n);
void aria_mul_backward_f32(const float *grad_out,
                            const float *a, const float *b,
                            float *grad_a, float *grad_b, int64_t n);
void aria_sub_backward_f32(const float *grad_out,
                            float *grad_a, float *grad_b, int64_t n);

/**
 * Matmul backward: C = A[M,K] @ B[K,N]
 * grad_A[M,K] = grad_out[M,N] @ B^T[N,K]
 * grad_B[K,N] = A^T[K,M] @ grad_out[M,N]
 */
void aria_matmul_backward_f32(const float *grad_out,
                               const float *A, const float *B,
                               float *grad_A, float *grad_B,
                               int64_t M, int64_t K, int64_t N);

/**
 * Softmax backward: dx = y * (dL/dy - sum(dL/dy * y)) per row
 * output: forward softmax result [batch, dim]
 */
void aria_softmax_backward_f32(const float *grad_out, const float *output,
                                float *grad_in, int64_t batch, int64_t dim);

/**
 * LayerNorm backward.
 * grad_gamma[dim], grad_beta[dim] accumulated across batch.
 * grad_in[batch, dim] = gradient w.r.t. input.
 */
void aria_layernorm_backward_f32(const float *grad_out, const float *input,
                                  const float *gamma,
                                  float *grad_in, float *grad_gamma,
                                  float *grad_beta,
                                  int64_t batch, int64_t dim, float eps);

/**
 * RMSNorm backward.
 * grad_gamma[dim] accumulated across batch.
 * grad_in[batch, dim] = gradient w.r.t. input.
 */
void aria_rmsnorm_backward_f32(const float *grad_out, const float *input,
                                const float *gamma,
                                float *grad_in, float *grad_gamma,
                                int64_t batch, int64_t dim, float eps);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_KERNELS_H */
