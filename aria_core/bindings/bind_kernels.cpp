/**
 * bind_kernels.cpp — Elementwise, reduction, linear algebra, normalization,
 *                    softmax, structural, fused, FP16, and backward kernel bindings.
 */
#include "bind_common.h"

// ═══ Elementwise unary ═══

#define DEFINE_UNARY_F32(name, c_func) \
torch::Tensor name(torch::Tensor x) { \
    CHECK_INPUT(x); \
    auto y = torch::empty_like(x); \
    c_func(x.data_ptr<float>(), y.data_ptr<float>(), x.numel()); \
    return y; \
}

DEFINE_UNARY_F32(relu_f32, aria_relu_f32)
DEFINE_UNARY_F32(gelu_f32, aria_gelu_f32)
DEFINE_UNARY_F32(silu_f32, aria_silu_f32)
DEFINE_UNARY_F32(square_f32, aria_square_f32)
DEFINE_UNARY_F32(abs_f32, aria_abs_f32)
DEFINE_UNARY_F32(neg_f32, aria_neg_f32)
DEFINE_UNARY_F32(reciprocal_f32, aria_reciprocal_f32)
DEFINE_UNARY_F32(log_f32, aria_log_f32)
DEFINE_UNARY_F32(sqrt_f32, aria_sqrt_f32)
DEFINE_UNARY_F32(sin_f32, aria_sin_f32)
DEFINE_UNARY_F32(cos_f32, aria_cos_f32)
DEFINE_UNARY_F32(tanh_f32, aria_tanh_f32)
DEFINE_UNARY_F32(sigmoid_f32, aria_sigmoid_f32)
DEFINE_UNARY_F32(exp_f32, aria_exp_f32)
DEFINE_UNARY_F32(sign_ste_f32, aria_sign_ste_f32)

// ═══ Elementwise binary ═══

#define DEFINE_BINARY_F32(name, c_func) \
torch::Tensor name(torch::Tensor a, torch::Tensor b) { \
    CHECK_INPUT(a); CHECK_INPUT(b); \
    TORCH_CHECK(a.numel() == b.numel(), "a and b must have same numel"); \
    auto y = torch::empty_like(a); \
    c_func(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel()); \
    return y; \
}

DEFINE_BINARY_F32(add_f32, aria_add_f32)
DEFINE_BINARY_F32(mul_f32, aria_mul_f32)
DEFINE_BINARY_F32(sub_f32, aria_sub_f32)
DEFINE_BINARY_F32(silu_mul_f32, aria_silu_mul_f32)
static torch::Tensor tropical_add_f32(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT_ANY(a); CHECK_INPUT_ANY(b);
    TORCH_CHECK(a.numel() == b.numel(), "a and b must have same numel");
    auto y = torch::empty_like(a);
    if (a.is_cuda()) {
        launch_cuda_tropical_add_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel());
    } else {
        aria_tropical_add_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel());
    }
    return y;
}
DEFINE_BINARY_F32(maximum_f32, aria_maximum_f32)
DEFINE_BINARY_F32(minimum_f32, aria_minimum_f32)
DEFINE_BINARY_F32(div_safe_f32, aria_div_safe_f32)
DEFINE_BINARY_F32(outer_product_f32, aria_outer_product_f32)

// ═══ Reductions ═══

static float sum_f32(torch::Tensor x) { CHECK_INPUT(x); return aria_sum_f32(x.data_ptr<float>(), x.numel()); }
static float mean_f32(torch::Tensor x) { CHECK_INPUT(x); return aria_mean_f32(x.data_ptr<float>(), x.numel()); }

static float linear_cka_f32(torch::Tensor X, torch::Tensor Y) {
    CHECK_INPUT(X); CHECK_INPUT(Y);
    TORCH_CHECK(X.numel() == Y.numel(), "X and Y must have same number of elements");
    int64_t n = (int64_t)sqrt(X.numel());
    TORCH_CHECK(n * n == X.numel(), "X must be a square matrix [n, n]");
    return aria_linear_cka_f32(X.data_ptr<float>(), Y.data_ptr<float>(), n);
}

// ═══ Linear algebra ═══

static torch::Tensor matmul_f32(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    aria_matmul_f32(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    return C;
}

static torch::Tensor tropical_matmul_f32(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT_ANY(A); CHECK_INPUT_ANY(B);
    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    if (A.is_cuda()) {
        launch_cuda_tropical_matmul_f32(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    } else {
        aria_tropical_matmul_f32(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    }
    return C;
}

static torch::Tensor tropical_matmul_batched_f32(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT_ANY(A); CHECK_INPUT_ANY(B);
    int64_t batch = A.size(0), M = A.size(1), K = A.size(2), N = B.size(1);
    auto C = torch::zeros({batch, M, N}, A.options());
    if (A.is_cuda()) {
        launch_cuda_tropical_matmul_batched_f32(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), batch, M, K, N);
    } else {
        aria_tropical_matmul_batched_f32(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), batch, M, K, N);
    }
    return C;
}

static std::vector<torch::Tensor> tropical_matmul_batched_backward_f32(
    torch::Tensor grad_out, torch::Tensor A, torch::Tensor B, float tau) {
    CHECK_INPUT_ANY(grad_out); CHECK_INPUT_ANY(A); CHECK_INPUT_ANY(B);
    int64_t batch = A.size(0), M = A.size(1), K = A.size(2), N = B.size(1);
    auto grad_A = torch::zeros_like(A);
    auto grad_B = torch::zeros_like(B);
    if (A.is_cuda()) {
        TORCH_CHECK(false, "CUDA backward not implemented natively yet, use Triton");
    } else {
        aria_tropical_matmul_batched_backward_f32(
            grad_out.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(),
            grad_A.data_ptr<float>(), grad_B.data_ptr<float>(),
            batch, M, K, N, tau
        );
    }
    return {grad_A, grad_B};
}

static torch::Tensor linear_f32(torch::Tensor x, torch::Tensor W, c10::optional<torch::Tensor> bias) {
    CHECK_INPUT(x); CHECK_INPUT(W);
    int64_t batch = x.size(0), dim_in = x.size(1), dim_out = W.size(0);
    const float *b_ptr = nullptr;
    if (bias.has_value()) { CHECK_INPUT(bias.value()); b_ptr = bias.value().data_ptr<float>(); }
    auto y = torch::empty({batch, dim_out}, x.options());
    aria_linear_f32(x.data_ptr<float>(), W.data_ptr<float>(), b_ptr, y.data_ptr<float>(), batch, dim_in, dim_out);
    return y;
}

// ═══ Normalization ═══

static torch::Tensor rmsnorm_f32(torch::Tensor x, torch::Tensor weight, float eps) {
    CHECK_INPUT(x); CHECK_INPUT(weight);
    auto y = torch::empty_like(x);
    int64_t dim = x.size(-1);
    int64_t batch = x.numel() / dim;
    aria_rmsnorm_f32(x.data_ptr<float>(), weight.data_ptr<float>(), y.data_ptr<float>(), batch, dim, eps);
    return y;
}

static torch::Tensor layernorm_f32(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float eps) {
    CHECK_INPUT(x); CHECK_INPUT(weight); CHECK_INPUT(bias);
    auto y = torch::empty_like(x);
    int64_t dim = x.size(-1);
    int64_t batch = x.numel() / dim;
    aria_layernorm_f32(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(), batch, dim, eps);
    return y;
}

// ═══ Softmax ═══

static torch::Tensor softmax_f32(torch::Tensor x) {
    CHECK_INPUT(x); auto y = torch::empty_like(x);
    int64_t dim = x.size(-1);
    int64_t batch = x.numel() / dim;
    aria_softmax_f32(x.data_ptr<float>(), y.data_ptr<float>(), batch, dim);
    return y;
}

static torch::Tensor softmax_seq_f32(torch::Tensor x) {
    CHECK_INPUT(x); auto y = torch::empty_like(x);
    aria_softmax_seq_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2));
    return y;
}

// ═══ Structural ═══

static torch::Tensor transpose2d_f32(torch::Tensor x) {
    CHECK_INPUT(x);
    auto y = torch::empty({x.size(1), x.size(0)}, x.options());
    aria_transpose2d_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1));
    return y;
}

static torch::Tensor causal_mask_f32(torch::Tensor x) {
    CHECK_INPUT(x); auto y = torch::empty_like(x);
    aria_causal_mask_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2));
    return y;
}

// ═══ Fused ═══

static torch::Tensor matmul_relu_f32(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    aria_matmul_relu_f32(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    return C;
}

static torch::Tensor matmul_gelu_f32(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    aria_matmul_gelu_f32(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    return C;
}

static torch::Tensor matmul_bias_relu_f32(torch::Tensor A, torch::Tensor B, torch::Tensor bias) {
    CHECK_INPUT(A); CHECK_INPUT(B); CHECK_INPUT(bias);
    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    aria_matmul_bias_relu_f32(A.data_ptr<float>(), B.data_ptr<float>(), bias.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    return C;
}

static torch::Tensor layernorm_residual_f32(torch::Tensor x, torch::Tensor residual, torch::Tensor gamma, torch::Tensor beta, float eps) {
    CHECK_INPUT(x); CHECK_INPUT(residual); CHECK_INPUT(gamma); CHECK_INPUT(beta);
    auto y = torch::empty_like(x);
    aria_layernorm_residual_f32(x.data_ptr<float>(), residual.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), eps);
    return y;
}

static torch::Tensor fused_linear_gelu_f32(torch::Tensor x, torch::Tensor W, torch::Tensor bias) {
    CHECK_INPUT(x); CHECK_INPUT(W); CHECK_INPUT(bias);
    int64_t batch = x.size(0), dim_in = x.size(1), dim_out = W.size(0);
    auto y = torch::empty({batch, dim_out}, x.options());
    aria_fused_linear_gelu_f32(x.data_ptr<float>(), W.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(), batch, dim_in, dim_out);
    return y;
}

// ═══ FP16 kernels ═══

#define DEFINE_UNARY_F16(name, c_func) \
torch::Tensor name(torch::Tensor x) { \
    CHECK_INPUT_F16(x); \
    auto y = torch::empty_like(x); \
    c_func(half_ptr_const(x), half_ptr(y), x.numel()); \
    return y; \
}

#define DEFINE_BINARY_F16(name, c_func) \
torch::Tensor name(torch::Tensor a, torch::Tensor b) { \
    CHECK_INPUT_F16(a); CHECK_INPUT_F16(b); \
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape"); \
    auto y = torch::empty_like(a); \
    c_func(half_ptr_const(a), half_ptr_const(b), half_ptr(y), a.numel()); \
    return y; \
}

DEFINE_UNARY_F16(relu_f16, aria_relu_f16)
DEFINE_UNARY_F16(gelu_f16, aria_gelu_f16)
DEFINE_UNARY_F16(silu_f16, aria_silu_f16)
DEFINE_UNARY_F16(sigmoid_f16, aria_sigmoid_f16)
DEFINE_BINARY_F16(add_f16, aria_add_f16)
DEFINE_BINARY_F16(mul_f16, aria_mul_f16)

static torch::Tensor matmul_f16(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT_F16(A); CHECK_INPUT_F16(B);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "matmul_f16 expects 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "inner dimensions must match");
    auto C = torch::empty({A.size(0), B.size(1)}, A.options());
    aria_matmul_f16(half_ptr_const(A), half_ptr_const(B), half_ptr(C), A.size(0), A.size(1), B.size(1));
    return C;
}

static torch::Tensor softmax_f16(torch::Tensor x) {
    CHECK_INPUT_F16(x);
    auto y = torch::empty_like(x);
    int64_t dim = x.size(-1);
    int64_t batch = x.numel() / dim;
    aria_softmax_f16(half_ptr_const(x), half_ptr(y), batch, dim);
    return y;
}

static torch::Tensor rmsnorm_f16(torch::Tensor x, torch::Tensor weight, float eps) {
    CHECK_INPUT_F16(x); CHECK_INPUT_F16(weight);
    auto y = torch::empty_like(x);
    int64_t dim = x.size(-1);
    int64_t batch = x.numel() / dim;
    aria_rmsnorm_f16(half_ptr_const(x), half_ptr_const(weight), half_ptr(y), batch, dim, eps);
    return y;
}

// ═══ Backward kernels ═══

static torch::Tensor relu_backward_f32(torch::Tensor go, torch::Tensor inp) { CHECK_INPUT(go); CHECK_INPUT(inp); auto g = torch::empty_like(go); aria_relu_backward_f32(go.data_ptr<float>(), inp.data_ptr<float>(), g.data_ptr<float>(), go.numel()); return g; }
static torch::Tensor sigmoid_backward_f32(torch::Tensor go, torch::Tensor out) { CHECK_INPUT(go); CHECK_INPUT(out); auto g = torch::empty_like(go); aria_sigmoid_backward_f32(go.data_ptr<float>(), out.data_ptr<float>(), g.data_ptr<float>(), go.numel()); return g; }
static torch::Tensor tanh_backward_f32(torch::Tensor go, torch::Tensor out) { CHECK_INPUT(go); CHECK_INPUT(out); auto g = torch::empty_like(go); aria_tanh_backward_f32(go.data_ptr<float>(), out.data_ptr<float>(), g.data_ptr<float>(), go.numel()); return g; }
static torch::Tensor gelu_backward_f32(torch::Tensor go, torch::Tensor inp) { CHECK_INPUT(go); CHECK_INPUT(inp); auto g = torch::empty_like(go); aria_gelu_backward_f32(go.data_ptr<float>(), inp.data_ptr<float>(), g.data_ptr<float>(), go.numel()); return g; }
static torch::Tensor silu_backward_f32(torch::Tensor go, torch::Tensor inp) { CHECK_INPUT(go); CHECK_INPUT(inp); auto g = torch::empty_like(go); aria_silu_backward_f32(go.data_ptr<float>(), inp.data_ptr<float>(), g.data_ptr<float>(), go.numel()); return g; }

static std::tuple<torch::Tensor, torch::Tensor> add_backward_f32(torch::Tensor go) { CHECK_INPUT(go); auto ga = torch::empty_like(go), gb = torch::empty_like(go); aria_add_backward_f32(go.data_ptr<float>(), ga.data_ptr<float>(), gb.data_ptr<float>(), go.numel()); return {ga, gb}; }
static std::tuple<torch::Tensor, torch::Tensor> sub_backward_f32(torch::Tensor go) { CHECK_INPUT(go); auto ga = torch::empty_like(go), gb = torch::empty_like(go); aria_sub_backward_f32(go.data_ptr<float>(), ga.data_ptr<float>(), gb.data_ptr<float>(), go.numel()); return {ga, gb}; }
static std::tuple<torch::Tensor, torch::Tensor> mul_backward_f32(torch::Tensor go, torch::Tensor a, torch::Tensor b) { CHECK_INPUT(go); CHECK_INPUT(a); CHECK_INPUT(b); auto ga = torch::empty_like(a), gb = torch::empty_like(b); aria_mul_backward_f32(go.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), ga.data_ptr<float>(), gb.data_ptr<float>(), go.numel()); return {ga, gb}; }

static std::tuple<torch::Tensor, torch::Tensor> matmul_backward_f32(torch::Tensor go, torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(go); CHECK_INPUT(A); CHECK_INPUT(B);
    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    auto gA = torch::zeros_like(A), gB = torch::zeros_like(B);
    aria_matmul_backward_f32(go.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), gA.data_ptr<float>(), gB.data_ptr<float>(), M, K, N);
    return {gA, gB};
}

static torch::Tensor softmax_backward_f32(torch::Tensor go, torch::Tensor out) { CHECK_INPUT(go); CHECK_INPUT(out); auto g = torch::empty_like(go); aria_softmax_backward_f32(go.data_ptr<float>(), out.data_ptr<float>(), g.data_ptr<float>(), out.size(0), out.size(1)); return g; }

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> layernorm_backward_f32(torch::Tensor go, torch::Tensor inp, torch::Tensor gamma, float eps) {
    CHECK_INPUT(go); CHECK_INPUT(inp); CHECK_INPUT(gamma);
    int64_t B = inp.size(0), D = inp.size(1);
    auto gi = torch::empty_like(inp); auto gg = torch::zeros({D}, inp.options()); auto gb = torch::zeros({D}, inp.options());
    aria_layernorm_backward_f32(go.data_ptr<float>(), inp.data_ptr<float>(), gamma.data_ptr<float>(), gi.data_ptr<float>(), gg.data_ptr<float>(), gb.data_ptr<float>(), B, D, eps);
    return {gi, gg, gb};
}

static std::tuple<torch::Tensor, torch::Tensor> rmsnorm_backward_f32(torch::Tensor go, torch::Tensor inp, torch::Tensor gamma, float eps) {
    CHECK_INPUT(go); CHECK_INPUT(inp); CHECK_INPUT(gamma);
    int64_t B = inp.size(0), D = inp.size(1);
    auto gi = torch::empty_like(inp); auto gg = torch::zeros({D}, inp.options());
    aria_rmsnorm_backward_f32(go.data_ptr<float>(), inp.data_ptr<float>(), gamma.data_ptr<float>(), gi.data_ptr<float>(), gg.data_ptr<float>(), B, D, eps);
    return {gi, gg};
}

static std::tuple<torch::Tensor, torch::Tensor> maximum_backward_f32(torch::Tensor go, torch::Tensor a, torch::Tensor b) { CHECK_INPUT(go); CHECK_INPUT(a); CHECK_INPUT(b); auto ga = torch::empty_like(a), gb = torch::empty_like(b); aria_maximum_backward_f32(go.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), ga.data_ptr<float>(), gb.data_ptr<float>(), a.numel()); return {ga, gb}; }
static std::tuple<torch::Tensor, torch::Tensor> minimum_backward_f32(torch::Tensor go, torch::Tensor a, torch::Tensor b) { CHECK_INPUT(go); CHECK_INPUT(a); CHECK_INPUT(b); auto ga = torch::empty_like(a), gb = torch::empty_like(b); aria_minimum_backward_f32(go.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), ga.data_ptr<float>(), gb.data_ptr<float>(), a.numel()); return {ga, gb}; }
static std::tuple<torch::Tensor, torch::Tensor> div_safe_backward_f32(torch::Tensor go, torch::Tensor a, torch::Tensor b) { CHECK_INPUT(go); CHECK_INPUT(a); CHECK_INPUT(b); auto ga = torch::empty_like(a), gb = torch::empty_like(b); aria_div_safe_backward_f32(go.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), ga.data_ptr<float>(), gb.data_ptr<float>(), a.numel()); return {ga, gb}; }

// ═══ Registration ═══

void bind_kernels(py::module_ &m) {
    // Unary f32
    m.def("relu_f32", &relu_f32); m.def("gelu_f32", &gelu_f32); m.def("silu_f32", &silu_f32);
    m.def("square_f32", &square_f32); m.def("abs_f32", &abs_f32); m.def("neg_f32", &neg_f32);
    m.def("reciprocal_f32", &reciprocal_f32); m.def("log_f32", &log_f32); m.def("sqrt_f32", &sqrt_f32);
    m.def("sin_f32", &sin_f32); m.def("cos_f32", &cos_f32); m.def("tanh_f32", &tanh_f32);
    m.def("sigmoid_f32", &sigmoid_f32); m.def("exp_f32", &exp_f32); m.def("sign_ste_f32", &sign_ste_f32);
    // Binary f32
    m.def("add_f32", &add_f32); m.def("mul_f32", &mul_f32); m.def("sub_f32", &sub_f32);
    m.def("silu_mul_f32", &silu_mul_f32);
    m.def("tropical_add_f32", &tropical_add_f32); m.def("maximum_f32", &maximum_f32);
    m.def("minimum_f32", &minimum_f32); m.def("div_safe_f32", &div_safe_f32); m.def("outer_product_f32", &outer_product_f32);
    // Reductions
    m.def("sum_f32", &sum_f32); m.def("mean_f32", &mean_f32);
    m.def("linear_cka_f32", &linear_cka_f32, "Linear CKA similarity score");
    // Linear algebra
    m.def("matmul_f32", &matmul_f32);
    m.def("tropical_matmul_f32", &tropical_matmul_f32);
    m.def("tropical_matmul_batched_f32", &tropical_matmul_batched_f32);
    m.def("tropical_matmul_batched_backward_f32", &tropical_matmul_batched_backward_f32);
    m.def("linear_f32", &linear_f32);
    // Normalization
    m.def("rmsnorm_f32", &rmsnorm_f32); m.def("layernorm_f32", &layernorm_f32);
    // Softmax
    m.def("softmax_f32", &softmax_f32); m.def("softmax_seq_f32", &softmax_seq_f32);
    // Structural
    m.def("transpose2d_f32", &transpose2d_f32); m.def("causal_mask_f32", &causal_mask_f32);
    // Fused
    m.def("matmul_relu_f32", &matmul_relu_f32); m.def("matmul_gelu_f32", &matmul_gelu_f32);
    m.def("matmul_bias_relu_f32", &matmul_bias_relu_f32); m.def("layernorm_residual_f32", &layernorm_residual_f32);
    m.def("fused_linear_gelu_f32", &fused_linear_gelu_f32);
    // FP16
    m.def("relu_f16", &relu_f16); m.def("gelu_f16", &gelu_f16); m.def("silu_f16", &silu_f16);
    m.def("sigmoid_f16", &sigmoid_f16); m.def("add_f16", &add_f16); m.def("mul_f16", &mul_f16);
    m.def("matmul_f16", &matmul_f16); m.def("softmax_f16", &softmax_f16);
    m.def("rmsnorm_f16", &rmsnorm_f16);
    // Backward
    m.def("relu_backward_f32", &relu_backward_f32); m.def("sigmoid_backward_f32", &sigmoid_backward_f32);
    m.def("tanh_backward_f32", &tanh_backward_f32); m.def("gelu_backward_f32", &gelu_backward_f32);
    m.def("silu_backward_f32", &silu_backward_f32); m.def("add_backward_f32", &add_backward_f32);
    m.def("mul_backward_f32", &mul_backward_f32); m.def("sub_backward_f32", &sub_backward_f32);
    m.def("matmul_backward_f32", &matmul_backward_f32); m.def("softmax_backward_f32", &softmax_backward_f32);
    m.def("layernorm_backward_f32", &layernorm_backward_f32); m.def("rmsnorm_backward_f32", &rmsnorm_backward_f32);
    m.def("maximum_backward_f32", &maximum_backward_f32); m.def("minimum_backward_f32", &minimum_backward_f32);
    m.def("div_safe_backward_f32", &div_safe_backward_f32);
    // Cumulative ops
    m.def("cumsum_f32", [](torch::Tensor x) -> torch::Tensor {
        CHECK_INPUT(x);
        auto flat = x.reshape({-1, x.size(-1)});
        auto y = torch::empty_like(flat);
        aria_cumsum_f32(flat.data_ptr<float>(), y.data_ptr<float>(), flat.size(0), flat.size(1));
        return y.reshape_as(x);
    }, "Cumulative sum along last dim (AVX2+OpenMP)");
    m.def("cumprod_safe_f32", [](torch::Tensor x, float lo, float hi) -> torch::Tensor {
        CHECK_INPUT(x);
        auto flat = x.reshape({-1, x.size(-1)});
        auto y = torch::empty_like(flat);
        aria_cumprod_safe_f32(flat.data_ptr<float>(), y.data_ptr<float>(), flat.size(0), flat.size(1), lo, hi);
        return y.reshape_as(x);
    }, "Cumulative product with clamping", py::arg("x"), py::arg("lo") = 1e-6f, py::arg("hi") = 1e6f);
}
