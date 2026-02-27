#ifndef ARIA_RESEARCH_KERNEL_ABI_H
#define ARIA_RESEARCH_KERNEL_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  NK_OK = 0,
  NK_ERR_UNSUPPORTED = -1,
  NK_ERR_INVALID_ARGUMENT = -2,
  NK_ERR_INTERNAL = -3
} nk_status_t;

/* --------------- function pointer signatures --------------- */

typedef nk_status_t (*nk_unary_f32_fn)(const float* x, float* y, int64_t n);
typedef nk_status_t (*nk_binary_f32_fn)(const float* a, const float* b, float* y, int64_t n);

typedef nk_status_t (*nk_matmul_f32_fn)(
    const float* A, const float* B, float* C,
    int64_t M, int64_t K, int64_t N);

typedef nk_status_t (*nk_linear_f32_fn)(
    const float* x, const float* W, const float* bias, float* y,
    int64_t batch, int64_t dim_in, int64_t dim_out);

typedef nk_status_t (*nk_softmax_f32_fn)(
    const float* x, float* y,
    int64_t batch, int64_t dim);

typedef nk_status_t (*nk_rmsnorm_f32_fn)(
    const float* x, const float* weight, float* y,
    int64_t batch, int64_t dim, float eps);

typedef nk_status_t (*nk_concat_f32_fn)(
    const float** inputs, const int64_t* sizes, int32_t n_inputs,
    float* output, int64_t dim);

typedef nk_status_t (*nk_split_f32_fn)(
    const float* input, float** outputs, const int64_t* sizes,
    int32_t n_outputs, int64_t dim);

/* --------------- fp16 function pointer signatures --------------- */

typedef nk_status_t (*nk_unary_f16_fn)(const uint16_t* x, uint16_t* y, int64_t n);
typedef nk_status_t (*nk_binary_f16_fn)(const uint16_t* a, const uint16_t* b, uint16_t* y, int64_t n);

typedef nk_status_t (*nk_matmul_f16_fn)(
    const uint16_t* A, const uint16_t* B, uint16_t* C,
    int64_t M, int64_t K, int64_t N);

typedef nk_status_t (*nk_softmax_f16_fn)(
    const uint16_t* x, uint16_t* y,
    int64_t batch, int64_t dim);

typedef nk_status_t (*nk_rmsnorm_f16_fn)(
    const uint16_t* x, const uint16_t* weight, uint16_t* y,
    int64_t batch, int64_t dim, float eps);

/* --------------- fused kernel function pointer signatures --------------- */

/** matmul + relu: C = max(0, A @ B) */
typedef nk_status_t (*nk_matmul_relu_f32_fn)(
    const float* A, const float* B, float* C,
    int64_t M, int64_t K, int64_t N);

/** matmul + bias + relu: C = max(0, A @ B + bias) */
typedef nk_status_t (*nk_matmul_bias_relu_f32_fn)(
    const float* A, const float* B, const float* bias, float* C,
    int64_t M, int64_t K, int64_t N);

/** layernorm(x + residual): y = LN(x + residual, gamma, beta) */
typedef nk_status_t (*nk_layernorm_residual_f32_fn)(
    const float* x, const float* residual,
    const float* gamma, const float* beta,
    float* y, int64_t rows, int64_t cols, float eps);

/** matmul + gelu: C = gelu(A @ B) */
typedef nk_status_t (*nk_matmul_gelu_f32_fn)(
    const float* A, const float* B, float* C,
    int64_t M, int64_t K, int64_t N);

/** swiglu: y = down(silu(gate) * up) */
typedef nk_status_t (*nk_swiglu_f32_fn)(
    const float* x,
    const float* W_gate, const float* W_up, const float* W_down,
    const float* bias_gate, const float* bias_up, const float* bias_down,
    float* y, float* tmp_gate, float* tmp_up,
    int64_t batch, int64_t dim, int64_t hidden_dim);

/** rwkv_channel: time-shift mixing + gated MLP */
typedef nk_status_t (*nk_rwkv_channel_f32_fn)(
    const float* x,
    const float* mix_k, const float* mix_r,
    const float* W_k, const float* W_r, const float* W_v,
    float* y, float* tmp_xk, float* tmp_xr, float* tmp_k,
    int64_t batch, int64_t seq, int64_t dim, int64_t hidden_dim);

/* --------------- reference architecture kernel signatures --------------- */

/** embedding_lookup: y = table[indices] + pos_embed */
typedef nk_status_t (*nk_embedding_lookup_f32_fn)(
    const float* table, const int32_t* indices, const float* pos_embed,
    float* y, int64_t batch, int64_t dim, int64_t vocab_size);

/** rope_rotate: rotary position embedding */
typedef nk_status_t (*nk_rope_rotate_f32_fn)(
    const float* x, float* y,
    int64_t batch, int64_t seq, int64_t dim, float theta_base);

/** gated_linear: y = (x @ W + b) * sigmoid(x @ W_gate + b_gate) */
typedef nk_status_t (*nk_gated_linear_f32_fn)(
    const float* x, const float* W, const float* b,
    const float* W_gate, const float* b_gate,
    float* y, float* tmp_gate,
    int64_t batch, int64_t dim_in, int64_t dim_out);

/** cosine_similarity: out[batch, seq] = cos_sim(a, b) */
typedef nk_status_t (*nk_cosine_similarity_f32_fn)(
    const float* a, const float* b, float* out,
    int64_t batch, int64_t seq, int64_t dim);

/** gather_topk: select top-k vectors by scores */
typedef nk_status_t (*nk_gather_topk_f32_fn)(
    const float* scores, const float* values,
    float* out, int32_t* out_indices,
    int64_t batch, int64_t n_items, int64_t dim, int64_t k);

/** rwkv_time_mixing: linear attention with learned exponential decay */
typedef nk_status_t (*nk_rwkv_time_mixing_f32_fn)(
    const float* x, const float* w_decay, const float* u_bonus,
    const float* W_k, const float* W_v, const float* W_r,
    float* y, int64_t batch, int64_t seq, int64_t dim);

/* --------------- backward function pointer signatures --------------- */

/* Unary backward: grad_in = f'(input_or_output) * grad_out
 * Second arg is either the forward input (relu, gelu, silu) or the forward
 * output (sigmoid, tanh) depending on the op. */
typedef nk_status_t (*nk_unary_backward_f32_fn)(
    const float* grad_out, const float* fwd_data, float* grad_in, int64_t n);

/* Binary backward for add/sub: grad_a, grad_b from grad_out only */
typedef nk_status_t (*nk_binary_backward_simple_f32_fn)(
    const float* grad_out, float* grad_a, float* grad_b, int64_t n);

/* Binary backward for mul: needs forward inputs a, b */
typedef nk_status_t (*nk_binary_backward_f32_fn)(
    const float* grad_out, const float* a, const float* b,
    float* grad_a, float* grad_b, int64_t n);

/* Matmul backward: grad_A, grad_B from grad_out, A, B */
typedef nk_status_t (*nk_matmul_backward_f32_fn)(
    const float* grad_out, const float* A, const float* B,
    float* grad_A, float* grad_B,
    int64_t M, int64_t K, int64_t N);

/* --------------- registration entry --------------- */

typedef struct {
  const char* op_name;
  /* Forward */
  nk_unary_f32_fn   unary_fn;
  nk_binary_f32_fn  binary_fn;
  nk_matmul_f32_fn  matmul_fn;
  nk_linear_f32_fn  linear_fn;
  nk_softmax_f32_fn softmax_fn;
  nk_rmsnorm_f32_fn rmsnorm_fn;
  nk_concat_f32_fn  concat_fn;
  nk_split_f32_fn   split_fn;
  /* FP16 forward */
  nk_unary_f16_fn   unary_f16_fn;
  nk_binary_f16_fn  binary_f16_fn;
  nk_matmul_f16_fn  matmul_f16_fn;
  nk_softmax_f16_fn softmax_f16_fn;
  nk_rmsnorm_f16_fn rmsnorm_f16_fn;
  /* Fused kernels */
  nk_matmul_relu_f32_fn             matmul_relu_fn;
  nk_matmul_bias_relu_f32_fn        matmul_bias_relu_fn;
  nk_layernorm_residual_f32_fn      layernorm_residual_fn;
  nk_matmul_gelu_f32_fn             matmul_gelu_fn;
  nk_swiglu_f32_fn                  swiglu_fn;
  nk_rwkv_channel_f32_fn            rwkv_channel_fn;
  /* Reference architecture ops */
  nk_embedding_lookup_f32_fn        embedding_lookup_fn;
  nk_rope_rotate_f32_fn             rope_rotate_fn;
  nk_gated_linear_f32_fn            gated_linear_fn;
  nk_cosine_similarity_f32_fn       cosine_similarity_fn;
  nk_gather_topk_f32_fn             gather_topk_fn;
  nk_rwkv_time_mixing_f32_fn        rwkv_time_mixing_fn;
  /* Backward */
  nk_unary_backward_f32_fn          unary_backward_fn;
  nk_binary_backward_simple_f32_fn  binary_backward_simple_fn;
  nk_binary_backward_f32_fn         binary_backward_fn;
  nk_matmul_backward_f32_fn         matmul_backward_fn;
} nk_registration_t;

/* --------------- registry API --------------- */

nk_status_t nk_register(const nk_registration_t* reg);
int32_t nk_is_registered(const char* op_name);

/* Return the registration entry for `op_name`, or NULL if not found. */
const nk_registration_t* nk_dispatch(const char* op_name);

/* Fill `names` with pointers to the registered op names (up to *count).
   On return *count contains the total number of registered ops. */
nk_status_t nk_list_registered(const char** names, int32_t* count);

#ifdef __cplusplus
}
#endif

#endif
