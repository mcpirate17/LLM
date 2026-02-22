# aria_kernels.pxd — C declarations for kernel dispatch
from libc.stdint cimport int32_t, int64_t

cdef extern from "kernel_abi.h":
    ctypedef enum nk_status_t:
        NK_OK = 0
        NK_ERR_UNSUPPORTED = -1
        NK_ERR_INVALID_ARGUMENT = -2
        NK_ERR_INTERNAL = -3

    ctypedef nk_status_t (*nk_unary_f32_fn)(const float* x, float* y, int64_t n)
    ctypedef nk_status_t (*nk_binary_f32_fn)(const float* a, const float* b, float* y, int64_t n)

    ctypedef struct nk_registration_t:
        const char* op_name
        nk_unary_f32_fn unary_fn
        nk_binary_f32_fn binary_fn

    nk_status_t nk_register(const nk_registration_t* reg)
    int32_t nk_is_registered(const char* op_name)

cdef extern from "kernels.h":
    # Elementwise unary
    void aria_relu_f32(const float* x, float* y, int64_t n)
    void aria_gelu_f32(const float* x, float* y, int64_t n)
    void aria_silu_f32(const float* x, float* y, int64_t n)
    void aria_square_f32(const float* x, float* y, int64_t n)
    void aria_abs_f32(const float* x, float* y, int64_t n)
    void aria_neg_f32(const float* x, float* y, int64_t n)
    void aria_reciprocal_f32(const float* x, float* y, int64_t n)
    void aria_log_f32(const float* x, float* y, int64_t n)
    void aria_sqrt_f32(const float* x, float* y, int64_t n)
    void aria_sin_f32(const float* x, float* y, int64_t n)
    void aria_cos_f32(const float* x, float* y, int64_t n)
    void aria_sigmoid_f32(const float* x, float* y, int64_t n)
    void aria_tanh_f32(const float* x, float* y, int64_t n)
    void aria_exp_f32(const float* x, float* y, int64_t n)

    # Elementwise binary
    void aria_add_f32(const float* a, const float* b, float* y, int64_t n)
    void aria_mul_f32(const float* a, const float* b, float* y, int64_t n)
    void aria_sub_f32(const float* a, const float* b, float* y, int64_t n)

    # Reductions
    float aria_sum_f32(const float* x, int64_t n)
    float aria_mean_f32(const float* x, int64_t n)

    # Linear algebra
    void aria_matmul_f32(const float* A, const float* B, float* C, int64_t M, int64_t K, int64_t N)
    void aria_linear_f32(const float* x, const float* W, const float* bias, float* y, int64_t batch, int64_t dim_in, int64_t dim_out)

    # Normalization
    void aria_rmsnorm_f32(const float* x, const float* weight, float* y, int64_t batch, int64_t dim, float eps)
    void aria_layernorm_f32(const float* x, const float* weight, const float* bias, float* y, int64_t batch, int64_t dim, float eps)

    # Softmax
    void aria_softmax_f32(const float* x, float* y, int64_t batch, int64_t dim)

    # Structural ops
    void aria_transpose2d_f32(const float* input, float* output, int64_t rows, int64_t cols)
    void aria_concat_f32(const float** inputs, const int64_t* sizes, int32_t n_inputs, float* output)
    void aria_split_f32(const float* input, float** outputs, const int64_t* sizes, int32_t n_outputs)

    # Backward (gradient) kernels — unary
    void aria_relu_backward_f32(const float* grad_out, const float* input, float* grad_in, int64_t n)
    void aria_sigmoid_backward_f32(const float* grad_out, const float* output, float* grad_in, int64_t n)
    void aria_tanh_backward_f32(const float* grad_out, const float* output, float* grad_in, int64_t n)
    void aria_gelu_backward_f32(const float* grad_out, const float* input, float* grad_in, int64_t n)
    void aria_silu_backward_f32(const float* grad_out, const float* input, float* grad_in, int64_t n)

    # Backward (gradient) kernels — binary
    void aria_add_backward_f32(const float* grad_out, float* grad_a, float* grad_b, int64_t n)
    void aria_mul_backward_f32(const float* grad_out, const float* a, const float* b, float* grad_a, float* grad_b, int64_t n)
    void aria_sub_backward_f32(const float* grad_out, float* grad_a, float* grad_b, int64_t n)

    # Backward (gradient) kernels — matmul
    void aria_matmul_backward_f32(const float* grad_out, const float* A, const float* B, float* grad_A, float* grad_B, int64_t M, int64_t K, int64_t N)

    # Backward (gradient) kernels — normalization / softmax
    void aria_softmax_backward_f32(const float* grad_out, const float* output, float* grad_in, int64_t batch, int64_t dim)
    void aria_layernorm_backward_f32(const float* grad_out, const float* input, const float* gamma, float* grad_in, float* grad_gamma, float* grad_beta, int64_t batch, int64_t dim, float eps)
    void aria_rmsnorm_backward_f32(const float* grad_out, const float* input, const float* gamma, float* grad_in, float* grad_gamma, int64_t batch, int64_t dim, float eps)
