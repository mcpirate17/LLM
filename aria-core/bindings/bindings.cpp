/**
 * bindings.cpp — pybind11 bindings for all aria-core kernels.
 *
 * Wraps C kernels with torch::Tensor interfaces for zero-copy interop.
 * All tensors must be contiguous CPU float32.
 *
 * Expanded from 10 → 92 bindings by claude-opus (2026-02-26).
 * Preserves gemini-cli's clifford.h / hyperbolic.h kernel bindings.
 */
#include <torch/extension.h>
#include <pybind11/stl.h>
#include "kernels.h"
#include "clifford.h"
#include "hyperbolic.h"
#include "graph_validator.h"
#include "shape_inference.h"

#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_F32(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) do { CHECK_CPU(x); CHECK_CONTIGUOUS(x); CHECK_F32(x); } while(0)

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
DEFINE_BINARY_F32(tropical_add_f32, aria_tropical_add_f32)
DEFINE_BINARY_F32(maximum_f32, aria_maximum_f32)
DEFINE_BINARY_F32(minimum_f32, aria_minimum_f32)
DEFINE_BINARY_F32(div_safe_f32, aria_div_safe_f32)
DEFINE_BINARY_F32(outer_product_f32, aria_outer_product_f32)

// ═══ Reductions ═══

float sum_f32(torch::Tensor x) { CHECK_INPUT(x); return aria_sum_f32(x.data_ptr<float>(), x.numel()); }
float mean_f32(torch::Tensor x) { CHECK_INPUT(x); return aria_mean_f32(x.data_ptr<float>(), x.numel()); }

// ═══ Linear algebra ═══

torch::Tensor matmul_f32(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    aria_matmul_f32(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    return C;
}

torch::Tensor tropical_matmul_f32(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    aria_tropical_matmul_f32(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    return C;
}

torch::Tensor linear_f32(torch::Tensor x, torch::Tensor W, c10::optional<torch::Tensor> bias) {
    CHECK_INPUT(x); CHECK_INPUT(W);
    int64_t batch = x.size(0), dim_in = x.size(1), dim_out = W.size(0);
    const float *b_ptr = nullptr;
    if (bias.has_value()) { CHECK_INPUT(bias.value()); b_ptr = bias.value().data_ptr<float>(); }
    auto y = torch::empty({batch, dim_out}, x.options());
    aria_linear_f32(x.data_ptr<float>(), W.data_ptr<float>(), b_ptr, y.data_ptr<float>(), batch, dim_in, dim_out);
    return y;
}

// ═══ Normalization ═══

torch::Tensor rmsnorm_f32(torch::Tensor x, torch::Tensor weight, float eps) {
    CHECK_INPUT(x); CHECK_INPUT(weight);
    auto y = torch::empty_like(x);
    int64_t dim = x.size(-1);
    int64_t batch = x.numel() / dim;
    aria_rmsnorm_f32(x.data_ptr<float>(), weight.data_ptr<float>(), y.data_ptr<float>(), batch, dim, eps);
    return y;
}

torch::Tensor layernorm_f32(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float eps) {
    CHECK_INPUT(x); CHECK_INPUT(weight); CHECK_INPUT(bias);
    auto y = torch::empty_like(x);
    int64_t dim = x.size(-1);
    int64_t batch = x.numel() / dim;
    aria_layernorm_f32(x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(), batch, dim, eps);
    return y;
}

// ═══ Softmax ═══

torch::Tensor softmax_f32(torch::Tensor x) {
    CHECK_INPUT(x); auto y = torch::empty_like(x);
    int64_t dim = x.size(-1);
    int64_t batch = x.numel() / dim;
    aria_softmax_f32(x.data_ptr<float>(), y.data_ptr<float>(), batch, dim);
    return y;
}

torch::Tensor softmax_seq_f32(torch::Tensor x) {
    CHECK_INPUT(x); auto y = torch::empty_like(x);
    aria_softmax_seq_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2));
    return y;
}

// ═══ Structural ═══

torch::Tensor transpose2d_f32(torch::Tensor x) {
    CHECK_INPUT(x);
    auto y = torch::empty({x.size(1), x.size(0)}, x.options());
    aria_transpose2d_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1));
    return y;
}

torch::Tensor causal_mask_f32(torch::Tensor x) {
    CHECK_INPUT(x); auto y = torch::empty_like(x);
    aria_causal_mask_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2));
    return y;
}

// ═══ Fused ═══

torch::Tensor matmul_relu_f32(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    aria_matmul_relu_f32(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    return C;
}

torch::Tensor matmul_gelu_f32(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    aria_matmul_gelu_f32(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    return C;
}

torch::Tensor matmul_bias_relu_f32(torch::Tensor A, torch::Tensor B, torch::Tensor bias) {
    CHECK_INPUT(A); CHECK_INPUT(B); CHECK_INPUT(bias);
    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    aria_matmul_bias_relu_f32(A.data_ptr<float>(), B.data_ptr<float>(), bias.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    return C;
}

torch::Tensor layernorm_residual_f32(torch::Tensor x, torch::Tensor residual, torch::Tensor gamma, torch::Tensor beta, float eps) {
    CHECK_INPUT(x); CHECK_INPUT(residual); CHECK_INPUT(gamma); CHECK_INPUT(beta);
    auto y = torch::empty_like(x);
    aria_layernorm_residual_f32(x.data_ptr<float>(), residual.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), eps);
    return y;
}

torch::Tensor fused_linear_gelu_f32(torch::Tensor x, torch::Tensor W, torch::Tensor bias) {
    CHECK_INPUT(x); CHECK_INPUT(W); CHECK_INPUT(bias);
    int64_t batch = x.size(0), dim_in = x.size(1), dim_out = W.size(0);
    auto y = torch::empty({batch, dim_out}, x.options());
    aria_fused_linear_gelu_f32(x.data_ptr<float>(), W.data_ptr<float>(), bias.data_ptr<float>(), y.data_ptr<float>(), batch, dim_in, dim_out);
    return y;
}

// ═══ Backward kernels ═══

torch::Tensor relu_backward_f32(torch::Tensor go, torch::Tensor inp) { CHECK_INPUT(go); CHECK_INPUT(inp); auto g = torch::empty_like(go); aria_relu_backward_f32(go.data_ptr<float>(), inp.data_ptr<float>(), g.data_ptr<float>(), go.numel()); return g; }
torch::Tensor sigmoid_backward_f32(torch::Tensor go, torch::Tensor out) { CHECK_INPUT(go); CHECK_INPUT(out); auto g = torch::empty_like(go); aria_sigmoid_backward_f32(go.data_ptr<float>(), out.data_ptr<float>(), g.data_ptr<float>(), go.numel()); return g; }
torch::Tensor tanh_backward_f32(torch::Tensor go, torch::Tensor out) { CHECK_INPUT(go); CHECK_INPUT(out); auto g = torch::empty_like(go); aria_tanh_backward_f32(go.data_ptr<float>(), out.data_ptr<float>(), g.data_ptr<float>(), go.numel()); return g; }
torch::Tensor gelu_backward_f32(torch::Tensor go, torch::Tensor inp) { CHECK_INPUT(go); CHECK_INPUT(inp); auto g = torch::empty_like(go); aria_gelu_backward_f32(go.data_ptr<float>(), inp.data_ptr<float>(), g.data_ptr<float>(), go.numel()); return g; }
torch::Tensor silu_backward_f32(torch::Tensor go, torch::Tensor inp) { CHECK_INPUT(go); CHECK_INPUT(inp); auto g = torch::empty_like(go); aria_silu_backward_f32(go.data_ptr<float>(), inp.data_ptr<float>(), g.data_ptr<float>(), go.numel()); return g; }

std::tuple<torch::Tensor, torch::Tensor> add_backward_f32(torch::Tensor go) { CHECK_INPUT(go); auto ga = torch::empty_like(go), gb = torch::empty_like(go); aria_add_backward_f32(go.data_ptr<float>(), ga.data_ptr<float>(), gb.data_ptr<float>(), go.numel()); return {ga, gb}; }
std::tuple<torch::Tensor, torch::Tensor> sub_backward_f32(torch::Tensor go) { CHECK_INPUT(go); auto ga = torch::empty_like(go), gb = torch::empty_like(go); aria_sub_backward_f32(go.data_ptr<float>(), ga.data_ptr<float>(), gb.data_ptr<float>(), go.numel()); return {ga, gb}; }
std::tuple<torch::Tensor, torch::Tensor> mul_backward_f32(torch::Tensor go, torch::Tensor a, torch::Tensor b) { CHECK_INPUT(go); CHECK_INPUT(a); CHECK_INPUT(b); auto ga = torch::empty_like(a), gb = torch::empty_like(b); aria_mul_backward_f32(go.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), ga.data_ptr<float>(), gb.data_ptr<float>(), go.numel()); return {ga, gb}; }

std::tuple<torch::Tensor, torch::Tensor> matmul_backward_f32(torch::Tensor go, torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(go); CHECK_INPUT(A); CHECK_INPUT(B);
    int64_t M = A.size(0), K = A.size(1), N = B.size(1);
    auto gA = torch::zeros_like(A), gB = torch::zeros_like(B);
    aria_matmul_backward_f32(go.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), gA.data_ptr<float>(), gB.data_ptr<float>(), M, K, N);
    return {gA, gB};
}

torch::Tensor softmax_backward_f32(torch::Tensor go, torch::Tensor out) { CHECK_INPUT(go); CHECK_INPUT(out); auto g = torch::empty_like(go); aria_softmax_backward_f32(go.data_ptr<float>(), out.data_ptr<float>(), g.data_ptr<float>(), out.size(0), out.size(1)); return g; }

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> layernorm_backward_f32(torch::Tensor go, torch::Tensor inp, torch::Tensor gamma, float eps) {
    CHECK_INPUT(go); CHECK_INPUT(inp); CHECK_INPUT(gamma);
    int64_t B = inp.size(0), D = inp.size(1);
    auto gi = torch::empty_like(inp); auto gg = torch::zeros({D}, inp.options()); auto gb = torch::zeros({D}, inp.options());
    aria_layernorm_backward_f32(go.data_ptr<float>(), inp.data_ptr<float>(), gamma.data_ptr<float>(), gi.data_ptr<float>(), gg.data_ptr<float>(), gb.data_ptr<float>(), B, D, eps);
    return {gi, gg, gb};
}

std::tuple<torch::Tensor, torch::Tensor> rmsnorm_backward_f32(torch::Tensor go, torch::Tensor inp, torch::Tensor gamma, float eps) {
    CHECK_INPUT(go); CHECK_INPUT(inp); CHECK_INPUT(gamma);
    int64_t B = inp.size(0), D = inp.size(1);
    auto gi = torch::empty_like(inp); auto gg = torch::zeros({D}, inp.options());
    aria_rmsnorm_backward_f32(go.data_ptr<float>(), inp.data_ptr<float>(), gamma.data_ptr<float>(), gi.data_ptr<float>(), gg.data_ptr<float>(), B, D, eps);
    return {gi, gg};
}

std::tuple<torch::Tensor, torch::Tensor> maximum_backward_f32(torch::Tensor go, torch::Tensor a, torch::Tensor b) { CHECK_INPUT(go); CHECK_INPUT(a); CHECK_INPUT(b); auto ga = torch::empty_like(a), gb = torch::empty_like(b); aria_maximum_backward_f32(go.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), ga.data_ptr<float>(), gb.data_ptr<float>(), a.numel()); return {ga, gb}; }
std::tuple<torch::Tensor, torch::Tensor> minimum_backward_f32(torch::Tensor go, torch::Tensor a, torch::Tensor b) { CHECK_INPUT(go); CHECK_INPUT(a); CHECK_INPUT(b); auto ga = torch::empty_like(a), gb = torch::empty_like(b); aria_minimum_backward_f32(go.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), ga.data_ptr<float>(), gb.data_ptr<float>(), a.numel()); return {ga, gb}; }
std::tuple<torch::Tensor, torch::Tensor> div_safe_backward_f32(torch::Tensor go, torch::Tensor a, torch::Tensor b) { CHECK_INPUT(go); CHECK_INPUT(a); CHECK_INPUT(b); auto ga = torch::empty_like(a), gb = torch::empty_like(b); aria_div_safe_backward_f32(go.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), ga.data_ptr<float>(), gb.data_ptr<float>(), a.numel()); return {ga, gb}; }

// ═══ Parameterized ═══

torch::Tensor sliding_window_mask_f32(torch::Tensor x, int64_t ws) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_sliding_window_mask_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), ws); return y; }
std::tuple<torch::Tensor, torch::Tensor> sort_seq_f32(torch::Tensor x) { CHECK_INPUT(x); auto y = torch::empty_like(x); auto idx = torch::empty({x.size(0), x.size(1), x.size(2)}, torch::dtype(torch::kInt64)); aria_sort_seq_f32(x.data_ptr<float>(), y.data_ptr<float>(), idx.data_ptr<int64_t>(), x.size(0), x.size(1), x.size(2)); return {y, idx}; }
torch::Tensor conv1d_seq_f32(torch::Tensor x, torch::Tensor w, torch::Tensor b) { CHECK_INPUT(x); CHECK_INPUT(w); CHECK_INPUT(b); auto y = torch::empty_like(x); aria_conv1d_seq_f32(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2)); return y; }
torch::Tensor selective_scan_f32(torch::Tensor x, torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor D) { CHECK_INPUT(x); CHECK_INPUT(A); CHECK_INPUT(B); CHECK_INPUT(C); CHECK_INPUT(D); auto y = torch::empty_like(x); aria_selective_scan_f32(x.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2)); return y; }
torch::Tensor topk_gate_f32(torch::Tensor x, torch::Tensor Wg, int64_t k) { CHECK_INPUT(x); CHECK_INPUT(Wg); auto y = torch::empty_like(x); aria_topk_gate_f32(x.data_ptr<float>(), Wg.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), k); return y; }
torch::Tensor basis_expansion_f32(torch::Tensor x, torch::Tensor f, int64_t nb) { CHECK_INPUT(x); CHECK_INPUT(f); auto y = torch::empty({x.size(0), x.size(1), x.size(2)*nb}, x.options()); aria_basis_expansion_f32(x.data_ptr<float>(), f.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), nb); return y; }
torch::Tensor sparse_threshold_f32(torch::Tensor x) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_sparse_threshold_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2)); return y; }
torch::Tensor token_pool_restore_f32(torch::Tensor x) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_token_pool_restore_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2)); return y; }

// ═══ Hyperbolic ═══

torch::Tensor exp_map_f32(torch::Tensor x, float c) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_exp_map_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.numel(), c); return y; }
torch::Tensor log_map_f32(torch::Tensor x, float c) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_log_map_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.numel(), c); return y; }
torch::Tensor poincare_add_f32(torch::Tensor x, torch::Tensor v, float c) { CHECK_INPUT(x); CHECK_INPUT(v); auto y = torch::empty_like(x); aria_poincare_add_f32(x.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), c); return y; }
torch::Tensor hyp_linear_f32(torch::Tensor x, torch::Tensor W, float c) { CHECK_INPUT(x); CHECK_INPUT(W); auto y = torch::empty({x.size(0), W.size(0)}, x.options()); aria_hyp_linear_f32(x.data_ptr<float>(), W.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), W.size(0), c); return y; }
torch::Tensor hyperbolic_norm_f32(torch::Tensor x, torch::Tensor g, torch::Tensor b, float c, float eps) { CHECK_INPUT(x); CHECK_INPUT(g); CHECK_INPUT(b); auto y = torch::empty_like(x); aria_hyperbolic_norm_f32(x.data_ptr<float>(), g.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), c, eps); return y; }
torch::Tensor hyp_tangent_nonlinear_f32(torch::Tensor x, float c) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_hyp_tangent_nonlinear_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.numel(), c); return y; }
torch::Tensor hyp_distance_f32(torch::Tensor x, torch::Tensor yi) { CHECK_INPUT(x); CHECK_INPUT(yi); auto out = torch::empty({x.size(0), x.size(1)}, x.options()); aria_hyp_distance_f32(x.data_ptr<float>(), yi.data_ptr<float>(), out.data_ptr<float>(), x.size(0), x.size(1), x.size(2)); return out; }
torch::Tensor hyperbolic_mobius_add_f32(torch::Tensor x, torch::Tensor v, float c) { CHECK_INPUT(x); CHECK_INPUT(v); auto y = torch::empty_like(x); aria_hyperbolic_mobius_add_f32(x.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), c); return y; }
torch::Tensor hyperbolic_distance_f32(torch::Tensor x, torch::Tensor yi, float c) { CHECK_INPUT(x); CHECK_INPUT(yi); auto out = torch::empty({x.size(0)}, x.options()); aria_hyperbolic_distance_f32(x.data_ptr<float>(), yi.data_ptr<float>(), out.data_ptr<float>(), x.size(0), x.size(1), c); return out; }

// ═══ Tropical ═══

torch::Tensor tropical_center_f32(torch::Tensor x) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_tropical_center_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2)); return y; }
torch::Tensor tropical_attention_f32(torch::Tensor x, float t) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_tropical_attention_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), t); return y; }
torch::Tensor tropical_gate_f32(torch::Tensor x, float t) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_tropical_gate_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), t); return y; }

// ═══ P-adic ═══

torch::Tensor padic_gate_f32(torch::Tensor x, float p) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_padic_gate_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.numel(), p); return y; }
torch::Tensor padic_expand_f32(torch::Tensor x, torch::Tensor W, float p, int64_t nd) { CHECK_INPUT(x); CHECK_INPUT(W); auto y = torch::empty({x.size(0), x.size(1)*nd}, x.options()); aria_padic_expand_f32(x.data_ptr<float>(), W.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), p, nd); return y; }
torch::Tensor padic_residual_f32(torch::Tensor x, torch::Tensor W, float p, int64_t nd) { CHECK_INPUT(x); CHECK_INPUT(W); auto y = torch::empty_like(x); aria_padic_residual_f32(x.data_ptr<float>(), W.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), p, nd); return y; }
torch::Tensor ultrametric_attention_f32(torch::Tensor x, float p) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_ultrametric_attention_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), p); return y; }

// ═══ Clifford ═══

torch::Tensor rotor_transform_f32(torch::Tensor x, torch::Tensor r) { CHECK_INPUT(x); CHECK_INPUT(r); auto y = torch::empty_like(x); aria_rotor_transform_f32(x.data_ptr<float>(), r.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1)); return y; }
torch::Tensor grade_select_f32(torch::Tensor x, int32_t g) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_grade_select_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), g); return y; }
torch::Tensor grade_mix_f32(torch::Tensor x, torch::Tensor a) { CHECK_INPUT(x); CHECK_INPUT(a); auto y = torch::empty_like(x); aria_grade_mix_f32(x.data_ptr<float>(), a.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1)); return y; }
torch::Tensor clifford_attention_f32(torch::Tensor x) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_clifford_attention_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2)); return y; }
torch::Tensor clifford_geometric_product_cl30_f32(torch::Tensor a, torch::Tensor b) { CHECK_INPUT(a); CHECK_INPUT(b); auto y = torch::empty_like(a); aria_clifford_geometric_product_cl30_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel()/8); return y; }
torch::Tensor clifford_rotor_transform_cl30_f32(torch::Tensor x, torch::Tensor r) { CHECK_INPUT(x); CHECK_INPUT(r); auto y = torch::empty_like(x); aria_clifford_rotor_transform_cl30_f32(x.data_ptr<float>(), r.data_ptr<float>(), y.data_ptr<float>(), x.numel()/8); return y; }

// ═══ Spiking ═══

torch::Tensor lif_neuron_f32(torch::Tensor x, float tau, float thr) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_lif_neuron_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), tau, thr); return y; }
torch::Tensor spike_rate_code_f32(torch::Tensor x) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_spike_rate_code_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2)); return y; }
torch::Tensor stdp_attention_f32(torch::Tensor x, float tp, float tm) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_stdp_attention_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), tp, tm); return y; }

// ═══ Reference Architecture ═══

torch::Tensor embedding_lookup_f32(torch::Tensor table, torch::Tensor idx, c10::optional<torch::Tensor> pe) {
    CHECK_INPUT(table);
    const float *pp = nullptr; if (pe.has_value()) { CHECK_INPUT(pe.value()); pp = pe.value().data_ptr<float>(); }
    auto y = torch::empty({idx.size(0), table.size(1)}, table.options());
    aria_embedding_lookup_f32(table.data_ptr<float>(), idx.data_ptr<int32_t>(), pp, y.data_ptr<float>(), idx.size(0), table.size(1), table.size(0));
    return y;
}
torch::Tensor rope_rotate_f32(torch::Tensor x, float tb) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_rope_rotate_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), tb); return y; }
torch::Tensor gated_linear_f32(torch::Tensor x, torch::Tensor W, c10::optional<torch::Tensor> b, torch::Tensor Wg, c10::optional<torch::Tensor> bg) {
    CHECK_INPUT(x); CHECK_INPUT(W); CHECK_INPUT(Wg);
    const float *bp=nullptr, *bgp=nullptr;
    if (b.has_value()) { CHECK_INPUT(b.value()); bp = b.value().data_ptr<float>(); }
    if (bg.has_value()) { CHECK_INPUT(bg.value()); bgp = bg.value().data_ptr<float>(); }
    int64_t batch=x.size(0), di=x.size(1), do_=W.size(0);
    auto y = torch::empty({batch, do_}, x.options()); auto tmp = torch::empty({batch, do_}, x.options());
    aria_gated_linear_f32(x.data_ptr<float>(), W.data_ptr<float>(), bp, Wg.data_ptr<float>(), bgp, y.data_ptr<float>(), tmp.data_ptr<float>(), batch, di, do_);
    return y;
}
torch::Tensor cosine_similarity_f32(torch::Tensor a, torch::Tensor b) { CHECK_INPUT(a); CHECK_INPUT(b); auto out = torch::empty({a.size(0), a.size(1)}, a.options()); aria_cosine_similarity_f32(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), a.size(0), a.size(1), a.size(2)); return out; }
torch::Tensor rwkv_time_mixing_f32(torch::Tensor x, torch::Tensor wd, torch::Tensor ub, torch::Tensor Wk, torch::Tensor Wv, torch::Tensor Wr) {
    CHECK_INPUT(x); CHECK_INPUT(wd); CHECK_INPUT(ub); CHECK_INPUT(Wk); CHECK_INPUT(Wv); CHECK_INPUT(Wr);
    auto y = torch::empty_like(x);
    aria_rwkv_time_mixing_f32(x.data_ptr<float>(), wd.data_ptr<float>(), ub.data_ptr<float>(), Wk.data_ptr<float>(), Wv.data_ptr<float>(), Wr.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2));
    return y;
}

// ── GraphExecutor Wrapper ──────────────────────────────────────────

class PyGraphExecutor {
public:
    PyGraphExecutor(int32_t n_tensors) : tensors_(n_tensors) {}

    void set_tensor(int32_t index, torch::Tensor t) {
        if (index >= 0 && index < (int32_t)tensors_.size()) {
            tensors_[index] = t;
        }
    }

    torch::Tensor get_tensor(int32_t index) {
        if (index >= 0 && index < (int32_t)tensors_.size()) {
            return tensors_[index];
        }
        return torch::Tensor();
    }

    void execute(const py::list& py_nodes) {
        for (auto item : py_nodes) {
            auto node_dict = item.cast<py::dict>();
            int type_int = node_dict["type"].cast<int>();
            
            py::list inputs = node_dict["inputs"].cast<py::list>();
            py::list outputs = node_dict["outputs"].cast<py::list>();
            py::list params = node_dict.contains("params") ? node_dict["params"].cast<py::list>() : py::list();

            if (type_int == 0) { // RELU
                auto x = tensors_[inputs[0].cast<int>()];
                auto y = tensors_[outputs[0].cast<int>()];
                aria_relu_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.numel());
            } else if (type_int == 3) { // ADD
                auto a = tensors_[inputs[0].cast<int>()];
                auto b = tensors_[inputs[1].cast<int>()];
                auto y = tensors_[outputs[0].cast<int>()];
                aria_add_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel());
            } else if (type_int == 6) { // RMSNORM
                auto x = tensors_[inputs[0].cast<int>()];
                auto w = tensors_[inputs[1].cast<int>()];
                auto y = tensors_[outputs[0].cast<int>()];
                float eps = params.size() > 0 ? params[0].cast<float>() : 1e-6f;
                int64_t dim = x.size(-1);
                int64_t batch = x.numel() / dim;
                aria_rmsnorm_f32(x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(), batch, dim, eps);
            } else if (type_int == 8) { // MATMUL
                auto a = tensors_[inputs[0].cast<int>()];
                auto b = tensors_[inputs[1].cast<int>()];
                auto y = tensors_[outputs[0].cast<int>()];
                int64_t M = a.size(0), K = a.size(1), N = b.size(1);
                aria_matmul_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), M, K, N);
            }
        }
    }

private:
    std::vector<torch::Tensor> tensors_;
};

// ═══ Graph validation & shape inference ═══

py::dict validate_graph(int32_t n_nodes, std::vector<std::vector<int32_t>> edges) {
    AriaGraph graph;
    graph.n_nodes = n_nodes;
    graph.n_edges = static_cast<int32_t>(edges.size());
    for (int32_t i = 0; i < graph.n_edges; i++) {
        graph.edges[i].source = edges[i][0];
        graph.edges[i].target = edges[i][1];
        graph.edges[i].src_port = edges[i].size() > 2 ? edges[i][2] : 0;
        graph.edges[i].tgt_port = edges[i].size() > 3 ? edges[i][3] : 0;
    }
    AriaValidationResult result;
    memset(&result, 0, sizeof(result));
    AriaResult rc = aria_validate_graph(&graph, &result);

    py::dict out;
    if (rc == ARIA_OK) {
        out["valid"] = true;
        std::vector<int32_t> topo(result.topo_order, result.topo_order + result.topo_len);
        std::vector<int32_t> in_deg(result.in_degree, result.in_degree + n_nodes);
        std::vector<int32_t> out_deg(result.out_degree, result.out_degree + n_nodes);
        out["topo_order"] = topo;
        out["in_degrees"] = in_deg;
        out["out_degrees"] = out_deg;
    } else {
        out["valid"] = false;
        out["error"] = std::string(result.error);
        out["code"] = static_cast<int>(rc);
    }
    return out;
}

py::dict propagate_shapes(
    std::vector<int32_t> topo_order,
    std::vector<std::vector<int32_t>> edges,
    std::vector<py::dict> node_rules
) {
    ShapeInferenceResult res;
    memset(&res, 0, sizeof(res));
    res.n_nodes = static_cast<int32_t>(node_rules.size());

    for (int32_t i = 0; i < res.n_nodes; i++) {
        auto& rule_data = node_rules[i];
        NodeShapeSpec& node = res.nodes[i];
        node.rule = static_cast<ShapeRule>(rule_data["rule"].cast<int>());
        node.n_inputs = rule_data.contains("n_inputs") ? rule_data["n_inputs"].cast<int32_t>() : 1;
        node.n_outputs = rule_data.contains("n_outputs") ? rule_data["n_outputs"].cast<int32_t>() : 1;
        node.split_n = rule_data.contains("split_n") ? rule_data["split_n"].cast<int32_t>() : 0;
        node.out_dim = rule_data.contains("out_dim") ? rule_data["out_dim"].cast<int32_t>() : -1;
        node.orig_seq_len = rule_data.contains("orig_seq_len") ? rule_data["orig_seq_len"].cast<int32_t>() : 0;

        if (rule_data.contains("input_shapes")) {
            auto shapes = rule_data["input_shapes"].cast<std::vector<py::object>>();
            for (size_t p = 0; p < shapes.size(); p++) {
                if (!shapes[p].is_none()) {
                    auto dims = shapes[p].cast<std::vector<int32_t>>();
                    node.input_shapes[p].shape.ndim = static_cast<int32_t>(dims.size());
                    for (size_t d = 0; d < dims.size(); d++)
                        node.input_shapes[p].shape.dims[d] = dims[d];
                    node.input_shapes[p].shape.valid = 1;
                }
            }
        }
    }

    // Prepare edges
    int32_t (*c_edges)[4] = new int32_t[edges.size()][4];
    for (size_t i = 0; i < edges.size(); i++) {
        c_edges[i][0] = edges[i][0];
        c_edges[i][1] = edges[i][1];
        c_edges[i][2] = edges[i].size() > 2 ? edges[i][2] : 0;
        c_edges[i][3] = edges[i].size() > 3 ? edges[i][3] : 0;
    }

    int rc = aria_propagate_shapes(&res, topo_order.data(),
                                    static_cast<int32_t>(topo_order.size()),
                                    c_edges, static_cast<int32_t>(edges.size()));
    delete[] c_edges;

    py::dict out;
    if (rc == 0) {
        out["valid"] = true;
        py::list all_shapes;
        for (int32_t i = 0; i < res.n_nodes; i++) {
            py::list node_out;
            for (int32_t p = 0; p < res.nodes[i].n_outputs; p++) {
                auto& shape = res.nodes[i].output_shapes[p].shape;
                if (shape.valid) {
                    py::list dims;
                    for (int32_t d = 0; d < shape.ndim; d++)
                        dims.append(shape.dims[d]);
                    node_out.append(dims);
                } else {
                    node_out.append(py::none());
                }
            }
            all_shapes.append(node_out);
        }
        out["output_shapes"] = all_shapes;
    } else {
        out["valid"] = false;
        out["error"] = std::string(res.error);
    }
    return out;
}

// ═══ Module registration — 97 bindings ═══

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "aria_core: Unified high-performance kernel library for Aria";
    m.def("relu_f32", &relu_f32); m.def("gelu_f32", &gelu_f32); m.def("silu_f32", &silu_f32);
    m.def("square_f32", &square_f32); m.def("abs_f32", &abs_f32); m.def("neg_f32", &neg_f32);
    m.def("reciprocal_f32", &reciprocal_f32); m.def("log_f32", &log_f32); m.def("sqrt_f32", &sqrt_f32);
    m.def("sin_f32", &sin_f32); m.def("cos_f32", &cos_f32); m.def("tanh_f32", &tanh_f32);
    m.def("sigmoid_f32", &sigmoid_f32); m.def("exp_f32", &exp_f32); m.def("sign_ste_f32", &sign_ste_f32);
    m.def("add_f32", &add_f32); m.def("mul_f32", &mul_f32); m.def("sub_f32", &sub_f32);
    m.def("silu_mul_f32", &silu_mul_f32);
    m.def("tropical_add_f32", &tropical_add_f32); m.def("maximum_f32", &maximum_f32);
    m.def("minimum_f32", &minimum_f32); m.def("div_safe_f32", &div_safe_f32); m.def("outer_product_f32", &outer_product_f32);
    m.def("sum_f32", &sum_f32); m.def("mean_f32", &mean_f32);
    m.def("matmul_f32", &matmul_f32); m.def("tropical_matmul_f32", &tropical_matmul_f32); m.def("linear_f32", &linear_f32);
    m.def("rmsnorm_f32", &rmsnorm_f32); m.def("layernorm_f32", &layernorm_f32);
    m.def("softmax_f32", &softmax_f32); m.def("softmax_seq_f32", &softmax_seq_f32);
    m.def("transpose2d_f32", &transpose2d_f32); m.def("causal_mask_f32", &causal_mask_f32);
    m.def("matmul_relu_f32", &matmul_relu_f32); m.def("matmul_gelu_f32", &matmul_gelu_f32);
    m.def("matmul_bias_relu_f32", &matmul_bias_relu_f32); m.def("layernorm_residual_f32", &layernorm_residual_f32);
    m.def("fused_linear_gelu_f32", &fused_linear_gelu_f32);
    m.def("relu_backward_f32", &relu_backward_f32); m.def("sigmoid_backward_f32", &sigmoid_backward_f32);
    m.def("tanh_backward_f32", &tanh_backward_f32); m.def("gelu_backward_f32", &gelu_backward_f32);
    m.def("silu_backward_f32", &silu_backward_f32); m.def("add_backward_f32", &add_backward_f32);
    m.def("mul_backward_f32", &mul_backward_f32); m.def("sub_backward_f32", &sub_backward_f32);
    m.def("matmul_backward_f32", &matmul_backward_f32); m.def("softmax_backward_f32", &softmax_backward_f32);
    m.def("layernorm_backward_f32", &layernorm_backward_f32); m.def("rmsnorm_backward_f32", &rmsnorm_backward_f32);
    m.def("maximum_backward_f32", &maximum_backward_f32); m.def("minimum_backward_f32", &minimum_backward_f32);
    m.def("div_safe_backward_f32", &div_safe_backward_f32);
    m.def("sliding_window_mask_f32", &sliding_window_mask_f32); m.def("sort_seq_f32", &sort_seq_f32);
    m.def("conv1d_seq_f32", &conv1d_seq_f32); m.def("selective_scan_f32", &selective_scan_f32);
    m.def("topk_gate_f32", &topk_gate_f32); m.def("basis_expansion_f32", &basis_expansion_f32);
    m.def("sparse_threshold_f32", &sparse_threshold_f32); m.def("token_pool_restore_f32", &token_pool_restore_f32);
    m.def("exp_map_f32", &exp_map_f32); m.def("log_map_f32", &log_map_f32);
    m.def("poincare_add_f32", &poincare_add_f32); m.def("hyp_linear_f32", &hyp_linear_f32);
    m.def("hyperbolic_norm_f32", &hyperbolic_norm_f32); m.def("hyp_tangent_nonlinear_f32", &hyp_tangent_nonlinear_f32);
    m.def("hyp_distance_f32", &hyp_distance_f32); m.def("hyperbolic_mobius_add_f32", &hyperbolic_mobius_add_f32);
    m.def("hyperbolic_distance_f32", &hyperbolic_distance_f32);
    m.def("tropical_center_f32", &tropical_center_f32); m.def("tropical_attention_f32", &tropical_attention_f32);
    m.def("tropical_gate_f32", &tropical_gate_f32);
    m.def("padic_gate_f32", &padic_gate_f32); m.def("padic_expand_f32", &padic_expand_f32);
    m.def("padic_residual_f32", &padic_residual_f32); m.def("ultrametric_attention_f32", &ultrametric_attention_f32);
    m.def("rotor_transform_f32", &rotor_transform_f32); m.def("grade_select_f32", &grade_select_f32);
    m.def("grade_mix_f32", &grade_mix_f32); m.def("clifford_attention_f32", &clifford_attention_f32);
    m.def("clifford_geometric_product_cl30_f32", &clifford_geometric_product_cl30_f32);
    m.def("clifford_rotor_transform_cl30_f32", &clifford_rotor_transform_cl30_f32);
    m.def("lif_neuron_f32", &lif_neuron_f32); m.def("spike_rate_code_f32", &spike_rate_code_f32);
    m.def("stdp_attention_f32", &stdp_attention_f32);
    m.def("embedding_lookup_f32", &embedding_lookup_f32); m.def("rope_rotate_f32", &rope_rotate_f32);
    m.def("gated_linear_f32", &gated_linear_f32); m.def("cosine_similarity_f32", &cosine_similarity_f32);
    m.def("rwkv_time_mixing_f32", &rwkv_time_mixing_f32);
    // Graph validation & shape inference
    m.def("validate_graph", &validate_graph, "Validate a DAG: cycle detection, topological sort",
          py::arg("n_nodes"), py::arg("edges"));
    m.def("propagate_shapes", &propagate_shapes, "Propagate tensor shapes through a graph",
          py::arg("topo_order"), py::arg("edges"), py::arg("node_rules"));

    py::class_<PyGraphExecutor>(m, "GraphExecutor")
        .def(py::init<int32_t>())
        .def("set_tensor", &PyGraphExecutor::set_tensor)
        .def("get_tensor", &PyGraphExecutor::get_tensor)
        .def("execute", &PyGraphExecutor::execute);
}
