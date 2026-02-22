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
