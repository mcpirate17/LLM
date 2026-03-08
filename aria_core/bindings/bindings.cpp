/**
 * bindings.cpp — pybind11 bindings for all aria_core kernels.
 *
 * Wraps C kernels with torch::Tensor interfaces for zero-copy interop.
 * All tensors must be contiguous CPU float32.
 *
 * Expanded from 10 → 92 bindings by claude-opus (2026-02-26).
 * Preserves gemini-cli's clifford.h / hyperbolic.h kernel bindings.
 */
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <unordered_set>
#include "kernels.h"
#include "clifford.h"
#include "hyperbolic.h"
#include "graph_validator.h"
#include "shape_inference.h"

#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_F32(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) do { CHECK_CPU(x); CHECK_CONTIGUOUS(x); CHECK_F32(x); } while(0)
#define CHECK_I64(x) TORCH_CHECK(x.dtype() == torch::kInt64, #x " must be int64")

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

float linear_cka_f32(torch::Tensor X, torch::Tensor Y) {
    CHECK_INPUT(X); CHECK_INPUT(Y);
    TORCH_CHECK(X.numel() == Y.numel(), "X and Y must have same number of elements");
    int64_t n = (int64_t)sqrt(X.numel());
    TORCH_CHECK(n * n == X.numel(), "X must be a square matrix [n, n]");
    return aria_linear_cka_f32(X.data_ptr<float>(), Y.data_ptr<float>(), n);
}

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
std::tuple<torch::Tensor, torch::Tensor> route_topk_indices_f32(torch::Tensor scores, int64_t k) {
    CHECK_INPUT(scores);
    auto idx = torch::empty({scores.size(0), k}, torch::dtype(torch::kInt64));
    auto w = torch::empty({scores.size(0), k}, scores.options());
    aria_route_topk_indices_f32(scores.data_ptr<float>(), idx.data_ptr<int64_t>(), w.data_ptr<float>(),
                                scores.size(0), scores.size(1), k);
    return {idx, w};
}
torch::Tensor difficulty_scorer_f32(torch::Tensor x, torch::Tensor w1, c10::optional<torch::Tensor> b1,
                                    torch::Tensor w2, c10::optional<torch::Tensor> b2) {
    CHECK_INPUT(x); CHECK_INPUT(w1); CHECK_INPUT(w2);
    TORCH_CHECK(x.dim() == 3, "x must be [B,S,D]");
    TORCH_CHECK(w1.dim() == 2, "w1 must be [H,D]");
    TORCH_CHECK(w2.dim() == 2 || w2.dim() == 1, "w2 must be [1,H] or [H]");
    int64_t B = x.size(0), S = x.size(1), D = x.size(2), H = w1.size(0);
    TORCH_CHECK(w1.size(1) == D, "w1 second dim must equal D");

    const float *b1p = nullptr;
    if (b1.has_value()) { CHECK_INPUT(b1.value()); b1p = b1.value().data_ptr<float>(); }
    const float *b2p = nullptr;
    if (b2.has_value()) { CHECK_INPUT(b2.value()); b2p = b2.value().data_ptr<float>(); }
    if (w2.dim() == 2) {
        TORCH_CHECK(w2.size(0) == 1 && w2.size(1) == H, "w2 must be [1,H]");
    } else {
        TORCH_CHECK(w2.size(0) == H, "w2 must be [H]");
    }

    auto scores = torch::empty({B, S, 1}, x.options());
    aria_difficulty_scorer_f32(x.data_ptr<float>(), w1.data_ptr<float>(), b1p, w2.data_ptr<float>(), b2p,
                               scores.data_ptr<float>(), B, S, D, H);
    return scores;
}
std::tuple<torch::Tensor, torch::Tensor> lane_router_threshold_f32(
    torch::Tensor scores, int64_t lanes, c10::optional<torch::Tensor> thresholds) {
    CHECK_INPUT(scores);
    TORCH_CHECK(scores.dim() == 2 || scores.dim() == 3, "scores must be [B,S] or [B,S,1]");
    TORCH_CHECK(lanes > 0, "lanes must be positive");
    auto flat_scores = scores.dim() == 3 ? scores.squeeze(-1).contiguous() : scores.contiguous();
    int64_t B = flat_scores.size(0), S = flat_scores.size(1);

    const float *threshold_ptr = nullptr;
    if (thresholds.has_value()) {
        CHECK_INPUT(thresholds.value());
        TORCH_CHECK(thresholds.value().dim() == 1, "thresholds must be 1D");
        TORCH_CHECK(thresholds.value().size(0) == lanes - 1, "thresholds size must be lanes-1");
        threshold_ptr = thresholds.value().data_ptr<float>();
    }

    auto assignments = torch::empty({B, S}, torch::dtype(torch::kInt64));
    auto weights = torch::empty({B, S, lanes}, scores.options());
    aria_lane_router_threshold_f32(flat_scores.data_ptr<float>(),
                                   assignments.data_ptr<int64_t>(), weights.data_ptr<float>(),
                                   B, S, lanes, threshold_ptr);
    return {assignments, weights};
}
std::tuple<torch::Tensor, torch::Tensor> load_balance_loss_f32(
    torch::Tensor assignments, int64_t lanes, float loss_weight,
    c10::optional<torch::Tensor> target_distribution) {
    CHECK_CPU(assignments); CHECK_CONTIGUOUS(assignments); CHECK_I64(assignments);
    TORCH_CHECK(assignments.dim() == 2, "assignments must be [B,S]");
    TORCH_CHECK(lanes > 0, "lanes must be positive");

    const float *target_ptr = nullptr;
    if (target_distribution.has_value()) {
        CHECK_INPUT(target_distribution.value());
        TORCH_CHECK(target_distribution.value().dim() == 1, "target_distribution must be 1D");
        TORCH_CHECK(target_distribution.value().size(0) == lanes, "target_distribution size must equal lanes");
        target_ptr = target_distribution.value().data_ptr<float>();
    }

    auto lane_fractions = torch::empty({lanes}, torch::dtype(torch::kFloat32));
    auto loss = torch::empty({1}, torch::dtype(torch::kFloat32));
    aria_load_balance_loss_f32(assignments.data_ptr<int64_t>(), target_ptr, lane_fractions.data_ptr<float>(),
                               loss.data_ptr<float>(), assignments.size(0), assignments.size(1), lanes, loss_weight);
    return {loss, lane_fractions};
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conditional_dispatch_f32(
    torch::Tensor x, torch::Tensor assignments, int64_t lane_id) {
    CHECK_INPUT(x);
    CHECK_CPU(assignments); CHECK_CONTIGUOUS(assignments); CHECK_I64(assignments);
    TORCH_CHECK(x.dim() == 3, "x must be [B,S,D]");
    TORCH_CHECK(assignments.dim() == 2, "assignments must be [B,S]");
    TORCH_CHECK(x.size(0) == assignments.size(0) && x.size(1) == assignments.size(1),
                "assignments must match x batch/seq");
    int64_t B = x.size(0), S = x.size(1), D = x.size(2);
    auto lane_out = torch::zeros_like(x);
    auto index_map = torch::empty({B, S}, torch::dtype(torch::kInt64));
    auto lane_counts = torch::empty({B}, torch::dtype(torch::kInt64));
    aria_conditional_dispatch_f32(x.data_ptr<float>(), assignments.data_ptr<int64_t>(), lane_id,
                                  lane_out.data_ptr<float>(), index_map.data_ptr<int64_t>(), lane_counts.data_ptr<int64_t>(),
                                  B, S, D);
    return {lane_out, index_map, lane_counts};
}
torch::Tensor conditional_dispatch_backward_f32(torch::Tensor lane_grad, torch::Tensor index_map) {
    CHECK_INPUT(lane_grad);
    CHECK_CPU(index_map); CHECK_CONTIGUOUS(index_map); CHECK_I64(index_map);
    TORCH_CHECK(lane_grad.dim() == 3, "lane_grad must be [B,S,D]");
    TORCH_CHECK(index_map.dim() == 2, "index_map must be [B,S]");
    TORCH_CHECK(lane_grad.size(0) == index_map.size(0) && lane_grad.size(1) == index_map.size(1),
                "index_map must match lane_grad batch/seq");
    int64_t B = lane_grad.size(0), S = lane_grad.size(1), D = lane_grad.size(2);
    auto grad_x = torch::zeros_like(lane_grad);
    aria_conditional_dispatch_backward_f32(lane_grad.data_ptr<float>(), index_map.data_ptr<int64_t>(),
                                           grad_x.data_ptr<float>(), B, S, D);
    return grad_x;
}
torch::Tensor conditional_gather_f32(torch::Tensor lane_out, torch::Tensor index_map, torch::Tensor weights) {
    CHECK_INPUT(lane_out);
    CHECK_CPU(index_map); CHECK_CONTIGUOUS(index_map); CHECK_I64(index_map);
    CHECK_INPUT(weights);
    TORCH_CHECK(lane_out.dim() == 3, "lane_out must be [B,S,D]");
    TORCH_CHECK(index_map.dim() == 2, "index_map must be [B,S]");
    TORCH_CHECK(weights.dim() == 2, "weights must be [B,S]");
    TORCH_CHECK(lane_out.size(0) == index_map.size(0) && lane_out.size(1) == index_map.size(1),
                "index_map must match lane_out batch/seq");
    TORCH_CHECK(weights.size(0) == lane_out.size(0) && weights.size(1) == lane_out.size(1),
                "weights must match lane_out batch/seq");
    int64_t B = lane_out.size(0), S = lane_out.size(1), D = lane_out.size(2);
    auto y = torch::zeros_like(lane_out);
    aria_conditional_gather_f32(lane_out.data_ptr<float>(), index_map.data_ptr<int64_t>(), weights.data_ptr<float>(),
                                y.data_ptr<float>(), B, S, D);
    return y;
}
std::tuple<torch::Tensor, torch::Tensor> conditional_gather_backward_f32(
    torch::Tensor grad_y, torch::Tensor lane_out, torch::Tensor index_map, torch::Tensor weights) {
    CHECK_INPUT(grad_y); CHECK_INPUT(lane_out);
    CHECK_CPU(index_map); CHECK_CONTIGUOUS(index_map); CHECK_I64(index_map);
    CHECK_INPUT(weights);
    TORCH_CHECK(grad_y.dim() == 3 && lane_out.dim() == 3, "grad_y/lane_out must be [B,S,D]");
    TORCH_CHECK(index_map.dim() == 2 && weights.dim() == 2, "index_map/weights must be [B,S]");
    TORCH_CHECK(grad_y.sizes() == lane_out.sizes(), "grad_y and lane_out shape mismatch");
    TORCH_CHECK(index_map.size(0) == grad_y.size(0) && index_map.size(1) == grad_y.size(1),
                "index_map must match grad_y batch/seq");
    TORCH_CHECK(weights.size(0) == grad_y.size(0) && weights.size(1) == grad_y.size(1),
                "weights must match grad_y batch/seq");
    int64_t B = grad_y.size(0), S = grad_y.size(1), D = grad_y.size(2);
    auto grad_lane = torch::zeros_like(lane_out);
    auto grad_weights = torch::zeros_like(weights);
    aria_conditional_gather_backward_f32(
        grad_y.data_ptr<float>(), lane_out.data_ptr<float>(), index_map.data_ptr<int64_t>(), weights.data_ptr<float>(),
        grad_lane.data_ptr<float>(), grad_weights.data_ptr<float>(), B, S, D
    );
    return {grad_lane, grad_weights};
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
adaptive_route_dispatch_f32(torch::Tensor x, torch::Tensor w1, c10::optional<torch::Tensor> b1,
                            torch::Tensor w2, c10::optional<torch::Tensor> b2,
                            int64_t lanes, c10::optional<torch::Tensor> thresholds) {
    CHECK_INPUT(x); CHECK_INPUT(w1); CHECK_INPUT(w2);
    TORCH_CHECK(x.dim() == 3, "x must be [B,S,D]");
    TORCH_CHECK(w1.dim() == 2, "w1 must be [H,D]");
    TORCH_CHECK(w2.dim() == 2 || w2.dim() == 1, "w2 must be [1,H] or [H]");
    TORCH_CHECK(lanes > 0, "lanes must be positive");
    int64_t B = x.size(0), S = x.size(1), D = x.size(2), H = w1.size(0);
    TORCH_CHECK(w1.size(1) == D, "w1 second dim must equal D");
    if (w2.dim() == 2) {
        TORCH_CHECK(w2.size(0) == 1 && w2.size(1) == H, "w2 must be [1,H]");
    } else {
        TORCH_CHECK(w2.size(0) == H, "w2 must be [H]");
    }

    const float *b1p = nullptr;
    if (b1.has_value()) { CHECK_INPUT(b1.value()); b1p = b1.value().data_ptr<float>(); }
    const float *b2p = nullptr;
    if (b2.has_value()) { CHECK_INPUT(b2.value()); b2p = b2.value().data_ptr<float>(); }
    const float *threshold_ptr = nullptr;
    if (thresholds.has_value()) {
        CHECK_INPUT(thresholds.value());
        TORCH_CHECK(thresholds.value().dim() == 1, "thresholds must be 1D");
        TORCH_CHECK(thresholds.value().size(0) == lanes - 1, "thresholds size must be lanes-1");
        threshold_ptr = thresholds.value().data_ptr<float>();
    }

    auto scores_flat = torch::empty({B, S}, x.options());
    auto scores = scores_flat.unsqueeze(-1);
    auto assignments = torch::empty({B, S}, torch::dtype(torch::kInt64));
    auto weights = torch::empty({B, S, lanes}, x.options());
    auto lane_out = torch::zeros({lanes, B, S, D}, x.options());
    auto index_map = torch::empty({lanes, B, S}, torch::dtype(torch::kInt64));
    auto lane_counts = torch::empty({lanes, B}, torch::dtype(torch::kInt64));

    aria_adaptive_route_dispatch_f32(
        x.data_ptr<float>(), w1.data_ptr<float>(), b1p, w2.data_ptr<float>(), b2p,
        lanes, threshold_ptr,
        scores_flat.data_ptr<float>(), assignments.data_ptr<int64_t>(), weights.data_ptr<float>(),
        lane_out.data_ptr<float>(), index_map.data_ptr<int64_t>(), lane_counts.data_ptr<int64_t>(),
        B, S, D, H
    );

    return {scores, assignments, weights, lane_out, index_map, lane_counts};
}
torch::Tensor route_lane_argmax_f32(torch::Tensor scores) {
    CHECK_INPUT(scores);
    auto idx = torch::empty({scores.size(0), scores.size(1)}, torch::dtype(torch::kInt64));
    aria_route_lane_argmax_f32(scores.data_ptr<float>(), idx.data_ptr<int64_t>(),
                               scores.size(0), scores.size(1), scores.size(2));
    return idx;
}
torch::Tensor route_recursion_depth_f32(torch::Tensor scores) {
    CHECK_INPUT(scores);
    auto depth = torch::empty({scores.size(0), scores.size(1)}, torch::dtype(torch::kInt64));
    aria_route_recursion_depth_f32(scores.data_ptr<float>(), depth.data_ptr<int64_t>(),
                                   scores.size(0), scores.size(1), scores.size(2));
    return depth;
}
std::tuple<torch::Tensor, torch::Tensor> token_merge_simple_f32(torch::Tensor x, int64_t n_keep) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 3, "x must be [B, S, D]");
    TORCH_CHECK(x.size(1) > 0, "sequence length must be > 0");
    int64_t nk = n_keep;
    if (nk < 1) nk = 1;
    if (nk > x.size(1)) nk = x.size(1);
    auto y = torch::empty({x.size(0), nk, x.size(2)}, x.options());
    auto restore = torch::empty({x.size(0), x.size(1)}, torch::dtype(torch::kInt64));
    aria_token_merge_simple_f32(x.data_ptr<float>(), y.data_ptr<float>(), restore.data_ptr<int64_t>(),
                                x.size(0), x.size(1), x.size(2), nk);
    return {y, restore};
}

torch::Tensor linear_low_rank_f32(torch::Tensor x, torch::Tensor U, torch::Tensor V, c10::optional<torch::Tensor> bias) {
    CHECK_INPUT(x); CHECK_INPUT(U); CHECK_INPUT(V);
    int64_t batch = x.size(0), dim_in = x.size(1), dim_out = V.size(0), rank = U.size(0);
    const float *bp = nullptr;
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
        bp = bias.value().data_ptr<float>();
    }
    
    auto y = torch::empty({batch, dim_out}, x.options());
    aria_linear_low_rank_f32(x.data_ptr<float>(), U.data_ptr<float>(), V.data_ptr<float>(), bp, y.data_ptr<float>(), batch, dim_in, dim_out, rank);
    return y;
}

torch::Tensor linear_block_sparse_f32(torch::Tensor x, torch::Tensor W, torch::Tensor mask, c10::optional<torch::Tensor> bias, int64_t bs) {
    CHECK_INPUT(x); CHECK_INPUT(W);
    TORCH_CHECK(mask.dtype() == torch::kUInt8, "mask must be uint8");
    int64_t batch = x.size(0), dim_in = x.size(1), dim_out = W.size(0);
    const float *bp = nullptr;
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
        bp = bias.value().data_ptr<float>();
    }
    auto y = torch::empty({batch, dim_out}, x.options());
    aria_linear_block_sparse_f32(x.data_ptr<float>(), W.data_ptr<float>(), bp, mask.data_ptr<uint8_t>(), y.data_ptr<float>(), batch, dim_in, dim_out, bs);
    return y;
}

torch::Tensor nm_sparse_mask_f32(torch::Tensor W, int32_t n, int32_t m) {
    CHECK_INPUT(W);
    auto mask = torch::empty_like(W, torch::kUInt8);
    aria_nm_sparse_mask_f32(W.data_ptr<float>(), mask.data_ptr<uint8_t>(), W.size(0), W.size(1), n, m);
    return mask;
}

torch::Tensor linear_grouped_f32(torch::Tensor x, torch::Tensor W, c10::optional<torch::Tensor> bias, int64_t groups) {
    CHECK_INPUT(x); CHECK_INPUT(W);
    int64_t n_tokens = x.numel() / x.size(-1);
    int64_t dim = x.size(-1);
    const float *bp = nullptr;
    if (bias.has_value()) {
        CHECK_INPUT(bias.value());
        bp = bias.value().data_ptr<float>();
    }
    auto y = torch::empty_like(x);
    aria_linear_grouped_f32(x.data_ptr<float>(), W.data_ptr<float>(), bp, y.data_ptr<float>(), n_tokens, dim, groups);
    return y;
}

torch::Tensor linear_bottleneck_f32(torch::Tensor x, torch::Tensor W_down, torch::Tensor W_up,
                                     c10::optional<torch::Tensor> b_down, c10::optional<torch::Tensor> b_up) {
    CHECK_INPUT(x); CHECK_INPUT(W_down); CHECK_INPUT(W_up);
    int64_t n_tokens = x.numel() / x.size(-1);
    int64_t dim_in = x.size(-1), dim_out = W_up.size(0), rank = W_down.size(0);
    const float *bd = nullptr;
    if (b_down.has_value()) {
        CHECK_INPUT(b_down.value());
        bd = b_down.value().data_ptr<float>();
    }
    const float *bu = nullptr;
    if (b_up.has_value()) {
        CHECK_INPUT(b_up.value());
        bu = b_up.value().data_ptr<float>();
    }
    auto y_shape = x.sizes().vec();
    y_shape.back() = dim_out;
    auto y = torch::empty(y_shape, x.options());
    aria_linear_bottleneck_f32(x.data_ptr<float>(), W_down.data_ptr<float>(), W_up.data_ptr<float>(), bd, bu, y.data_ptr<float>(), n_tokens, dim_in, dim_out, rank);
    return y;
}

torch::Tensor linear_shared_basis_f32(torch::Tensor x, torch::Tensor Mixing, torch::Tensor Basis) {
    CHECK_INPUT(x); CHECK_INPUT(Mixing); CHECK_INPUT(Basis);
    int64_t n_tokens = x.numel() / x.size(-1);
    int64_t dim = x.size(-1), k_basis = Mixing.size(0);
    auto y = torch::empty_like(x);
    aria_linear_shared_basis_f32(x.data_ptr<float>(), Mixing.data_ptr<float>(), Basis.data_ptr<float>(), y.data_ptr<float>(), n_tokens, dim, k_basis);
    return y;
}

torch::Tensor linear_tied_f32(torch::Tensor x, torch::Tensor W, c10::optional<torch::Tensor> b_down, c10::optional<torch::Tensor> b_up) {
    CHECK_INPUT(x); CHECK_INPUT(W);
    int64_t n_tokens = x.numel() / x.size(-1);
    int64_t dim_in = x.size(-1), rank = W.size(0);
    const float *bd = nullptr;
    if (b_down.has_value()) {
        CHECK_INPUT(b_down.value());
        bd = b_down.value().data_ptr<float>();
    }
    const float *bu = nullptr;
    if (b_up.has_value()) {
        CHECK_INPUT(b_up.value());
        bu = b_up.value().data_ptr<float>();
    }
    auto y = torch::empty_like(x);
    aria_linear_tied_f32(x.data_ptr<float>(), W.data_ptr<float>(), bd, bu, y.data_ptr<float>(), n_tokens, dim_in, rank);
    return y;
}

// ═══ Hyperbolic ═══

torch::Tensor exp_map_f32(torch::Tensor x, float c) {
    CHECK_INPUT(x); auto y = torch::empty_like(x);
    int64_t dim = x.size(-1);
    int64_t batch = x.numel() / dim;
    aria_exp_map_f32(x.data_ptr<float>(), y.data_ptr<float>(), batch, dim, c);
    return y;
}
torch::Tensor log_map_f32(torch::Tensor x, float c) {
    CHECK_INPUT(x); auto y = torch::empty_like(x);
    int64_t dim = x.size(-1);
    int64_t batch = x.numel() / dim;
    aria_log_map_f32(x.data_ptr<float>(), y.data_ptr<float>(), batch, dim, c);
    return y;
}
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

// ═══ SwiGLU / RWKV Channel / Gather Top-K ═══

torch::Tensor swiglu_f32(torch::Tensor x, torch::Tensor W_gate, torch::Tensor W_up, torch::Tensor W_down,
                          c10::optional<torch::Tensor> b_gate, c10::optional<torch::Tensor> b_up, c10::optional<torch::Tensor> b_down) {
    CHECK_INPUT(x); CHECK_INPUT(W_gate); CHECK_INPUT(W_up); CHECK_INPUT(W_down);
    int64_t batch = x.size(0), dim = x.size(1), hidden_dim = W_gate.size(0);
    const float *bg = nullptr, *bu = nullptr, *bd = nullptr;
    if (b_gate.has_value()) { CHECK_INPUT(b_gate.value()); bg = b_gate.value().data_ptr<float>(); }
    if (b_up.has_value()) { CHECK_INPUT(b_up.value()); bu = b_up.value().data_ptr<float>(); }
    if (b_down.has_value()) { CHECK_INPUT(b_down.value()); bd = b_down.value().data_ptr<float>(); }
    auto y = torch::empty_like(x);
    auto tmp_gate = torch::empty({batch, hidden_dim}, x.options());
    auto tmp_up = torch::empty({batch, hidden_dim}, x.options());
    aria_swiglu_f32(x.data_ptr<float>(), W_gate.data_ptr<float>(), W_up.data_ptr<float>(), W_down.data_ptr<float>(),
                    bg, bu, bd, y.data_ptr<float>(), tmp_gate.data_ptr<float>(), tmp_up.data_ptr<float>(),
                    batch, dim, hidden_dim);
    return y;
}

torch::Tensor rwkv_channel_f32(torch::Tensor x, torch::Tensor mix_k, torch::Tensor mix_r,
                                torch::Tensor W_k, torch::Tensor W_r, torch::Tensor W_v) {
    CHECK_INPUT(x); CHECK_INPUT(mix_k); CHECK_INPUT(mix_r);
    CHECK_INPUT(W_k); CHECK_INPUT(W_r); CHECK_INPUT(W_v);
    int64_t batch = x.size(0), seq = x.size(1), dim = x.size(2), hidden_dim = W_k.size(0);
    auto y = torch::empty_like(x);
    auto tmp_xk = torch::empty({batch, seq, dim}, x.options());
    auto tmp_xr = torch::empty({batch, seq, dim}, x.options());
    auto tmp_k = torch::empty({batch, seq, hidden_dim}, x.options());
    aria_rwkv_channel_f32(x.data_ptr<float>(), mix_k.data_ptr<float>(), mix_r.data_ptr<float>(),
                          W_k.data_ptr<float>(), W_r.data_ptr<float>(), W_v.data_ptr<float>(),
                          y.data_ptr<float>(), tmp_xk.data_ptr<float>(), tmp_xr.data_ptr<float>(), tmp_k.data_ptr<float>(),
                          batch, seq, dim, hidden_dim);
    return y;
}

std::tuple<torch::Tensor, torch::Tensor> gather_topk_f32(torch::Tensor scores, torch::Tensor values, int64_t k) {
    CHECK_INPUT(scores); CHECK_INPUT(values);
    int64_t batch = scores.size(0), n_items = scores.size(1), dim = values.size(2);
    auto out = torch::empty({batch, k, dim}, values.options());
    auto out_indices = torch::empty({batch, k}, torch::dtype(torch::kInt32));
    aria_gather_topk_f32(scores.data_ptr<float>(), values.data_ptr<float>(),
                         out.data_ptr<float>(), out_indices.data_ptr<int32_t>(),
                         batch, n_items, dim, k);
    return {out, out_indices};
}

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

struct ExecNode {
    int type;
    std::vector<int> inputs;
    std::vector<int> outputs;
    std::vector<float> params;
};

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

    void bake(const py::list& py_nodes) {
        baked_nodes_.clear();
        for (auto item : py_nodes) {
            auto node_dict = item.cast<py::dict>();
            ExecNode node;
            node.type = node_dict["type"].cast<int>();
            node.inputs = node_dict["inputs"].cast<std::vector<int>>();
            node.outputs = node_dict["outputs"].cast<std::vector<int>>();
            if (node_dict.contains("params")) {
                node.params = node_dict["params"].cast<std::vector<float>>();
            }
            baked_nodes_.push_back(node);
        }
    }

    void execute() {
        for (size_t i = 0; i < baked_nodes_.size(); ++i) {
            _execute_node(baked_nodes_[i]);
        }
    }

    // Legacy support
    void execute_list(const py::list& py_nodes) {
        for (auto item : py_nodes) {
            auto node_dict = item.cast<py::dict>();
            ExecNode node;
            node.type = node_dict["type"].cast<int>();
            node.inputs = node_dict["inputs"].cast<std::vector<int>>();
            node.outputs = node_dict["outputs"].cast<std::vector<int>>();
            if (node_dict.contains("params")) {
                node.params = node_dict["params"].cast<std::vector<float>>();
            }
            _execute_node(node);
        }
    }

private:
    void _execute_node(const ExecNode& node) {
        // Bounds check
        for (int in_idx : node.inputs) {
            if (in_idx < 0 || in_idx >= (int)tensors_.size()) {
                return;
            }
        }
        for (int out_idx : node.outputs) {
            if (out_idx < 0 || out_idx >= (int)tensors_.size()) {
                return;
            }
        }

        switch (node.type) {
            case 0: { // RELU
                auto& x = tensors_[node.inputs[0]];
                auto& y = tensors_[node.outputs[0]];
                aria_relu_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.numel());
                break;
            }
            case 1: { // GELU
                auto& x = tensors_[node.inputs[0]];
                auto& y = tensors_[node.outputs[0]];
                aria_gelu_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.numel());
                break;
            }
            case 2: { // SILU
                auto& x = tensors_[node.inputs[0]];
                auto& y = tensors_[node.outputs[0]];
                aria_silu_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.numel());
                break;
            }
            case 3: { // ADD
                auto& a = tensors_[node.inputs[0]];
                auto& b = tensors_[node.inputs[1]];
                auto& y = tensors_[node.outputs[0]];
                aria_add_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel());
                break;
            }
            case 4: { // MUL
                auto& a = tensors_[node.inputs[0]];
                auto& b = tensors_[node.inputs[1]];
                auto& y = tensors_[node.outputs[0]];
                aria_mul_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel());
                break;
            }
            case 5: { // SUB
                auto& a = tensors_[node.inputs[0]];
                auto& b = tensors_[node.inputs[1]];
                auto& y = tensors_[node.outputs[0]];
                aria_sub_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel());
                break;
            }
            case 6: { // RMSNORM
                if (node.inputs.size() < 2) return;
                auto& x = tensors_[node.inputs[0]];
                auto& w = tensors_[node.inputs[1]];
                auto& y = tensors_[node.outputs[0]];
                float eps = node.params.size() > 0 ? node.params[0] : 1e-6f;
                int64_t dim = x.size(-1);
                int64_t batch = x.numel() / dim;
                aria_rmsnorm_f32(x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(), batch, dim, eps);
                break;
            }
            case 7: { // LAYERNORM
                if (node.inputs.size() < 3) return;
                auto& x = tensors_[node.inputs[0]];
                auto& w = tensors_[node.inputs[1]];
                auto& b = tensors_[node.inputs[2]];
                auto& y = tensors_[node.outputs[0]];
                float eps = node.params.size() > 0 ? node.params[0] : 1e-6f;
                int64_t dim = x.size(-1);
                int64_t batch = x.numel() / dim;
                float* bias_ptr = (b.numel() > 0) ? b.data_ptr<float>() : nullptr;
                aria_layernorm_f32(x.data_ptr<float>(), w.data_ptr<float>(), bias_ptr, y.data_ptr<float>(), batch, dim, eps);
                break;
            }
            case 8: { // MATMUL
                auto& a = tensors_[node.inputs[0]];
                auto& b = tensors_[node.inputs[1]];
                auto& y = tensors_[node.outputs[0]];
                int64_t M = a.size(0), K = a.size(1), N = b.size(1);
                aria_matmul_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), M, K, N);
                break;
            }
            case 9: { // LINEAR
                if (node.inputs.size() < 2) return;
                auto& x = tensors_[node.inputs[0]];
                auto& w = tensors_[node.inputs[1]];
                auto& y = tensors_[node.outputs[0]];
                
                float* bias_ptr = nullptr;
                if (node.inputs.size() >= 3) {
                    auto& b = tensors_[node.inputs[2]];
                    if (b.numel() > 0) bias_ptr = b.data_ptr<float>();
                }
                
                int64_t dim_in = x.size(-1);
                int64_t batch = x.numel() / dim_in;
                int64_t dim_out = w.size(0);
                aria_linear_f32(x.data_ptr<float>(), w.data_ptr<float>(), bias_ptr, y.data_ptr<float>(), batch, dim_in, dim_out);
                break;
            }
            case 10: { // SOFTMAX
                auto& x = tensors_[node.inputs[0]];
                auto& y = tensors_[node.outputs[0]];
                int64_t dim = x.size(-1);
                int64_t batch = x.numel() / dim;
                aria_softmax_f32(x.data_ptr<float>(), y.data_ptr<float>(), batch, dim);
                break;
            }
        }
    }

    std::vector<torch::Tensor> tensors_;
    std::vector<ExecNode> baked_nodes_;
};

// ═══ Graph validation & shape inference ═══

py::dict validate_graph(int32_t n_nodes, std::vector<std::vector<int32_t>> edges, std::vector<int32_t> op_codes) {
    AriaGraph graph;
    memset(&graph, 0, sizeof(graph));
    graph.n_nodes = n_nodes;
    graph.n_edges = static_cast<int32_t>(edges.size());
    for (int32_t i = 0; i < graph.n_edges; i++) {
        graph.edges[i].source = edges[i][0];
        graph.edges[i].target = edges[i][1];
        graph.edges[i].src_port = edges[i].size() > 2 ? edges[i][2] : 0;
        graph.edges[i].tgt_port = edges[i].size() > 3 ? edges[i][3] : 0;
    }
    for (int32_t i = 0; i < n_nodes && i < (int32_t)op_codes.size(); i++) {
        graph.op_codes[i] = op_codes[i];
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

py::dict proactive_gating(int32_t n_nodes, std::vector<std::vector<int32_t>> edges, std::vector<int32_t> op_codes,
                          std::vector<int32_t> norm_opcodes, std::vector<int32_t> param_opcodes, std::vector<int32_t> linear_opcodes) {
    AriaGraph graph;
    memset(&graph, 0, sizeof(graph));
    graph.n_nodes = n_nodes;
    graph.n_edges = static_cast<int32_t>(edges.size());
    for (int32_t i = 0; i < graph.n_edges; i++) {
        graph.edges[i].source = edges[i][0];
        graph.edges[i].target = edges[i][1];
    }
    // Build opcode lookup sets
    std::unordered_set<int32_t> norm_set(norm_opcodes.begin(), norm_opcodes.end());
    std::unordered_set<int32_t> param_set(param_opcodes.begin(), param_opcodes.end());
    std::unordered_set<int32_t> linear_set(linear_opcodes.begin(), linear_opcodes.end());
    for (int32_t i = 0; i < n_nodes && i < (int32_t)op_codes.size(); i++) {
        graph.op_codes[i] = op_codes[i];
        graph.is_norm[i] = norm_set.count(op_codes[i]) ? 1 : 0;
        graph.is_parameterized[i] = param_set.count(op_codes[i]) ? 1 : 0;
        graph.is_linear[i] = linear_set.count(op_codes[i]) ? 1 : 0;
    }

    AriaValidationResult val;
    aria_validate_graph(&graph, &val);

    AriaProactiveGatingResult res;
    aria_proactive_gating(&graph, &val, &res);

    py::dict out;
    out["passed"] = res.passed != 0;
    out["reason"] = std::string(res.reason);
    out["max_depth"] = res.max_depth;
    out["n_toxic_motifs"] = res.n_toxic_motifs;
    out["has_normalization_gap"] = res.has_normalization_gap != 0;
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

py::list canonical_topo_sort(
    int32_t n_nodes,
    std::vector<std::pair<int32_t, int32_t>> edges,
    std::vector<std::string> op_names,
    std::vector<std::string> config_strs,
    std::vector<std::vector<int32_t>> node_inputs
) {
    if (n_nodes == 0) return py::list();

    std::vector<int32_t> in_degree(n_nodes, 0);
    std::vector<std::vector<int32_t>> children(n_nodes);

    for (const auto& edge : edges) {
        if (edge.first >= 0 && edge.first < n_nodes && edge.second >= 0 && edge.second < n_nodes) {
            children[edge.first].push_back(edge.second);
            in_degree[edge.second]++;
        }
    }

    struct CanonicalKey {
        std::string op_name;
        std::vector<int32_t> input_ranks;
        std::string config_str;
        int32_t node_id;

        bool operator>(const CanonicalKey& other) const {
            if (op_name != other.op_name) return op_name > other.op_name;
            if (input_ranks != other.input_ranks) return input_ranks > other.input_ranks;
            if (config_str != other.config_str) return config_str > other.config_str;
            return node_id > other.node_id;
        }
    };

    std::priority_queue<CanonicalKey, std::vector<CanonicalKey>, std::greater<CanonicalKey>> ready;
    std::vector<int32_t> canonical_id_map(n_nodes, -1);
    std::vector<int32_t> order;
    order.reserve(n_nodes);

    auto push_node = [&](int32_t nid) {
        CanonicalKey key;
        key.node_id = nid;
        key.op_name = op_names[nid];
        key.config_str = config_strs[nid];
        for (int32_t iid : node_inputs[nid]) {
            key.input_ranks.push_back(canonical_id_map[iid]);
        }
        ready.push(std::move(key));
    };

    for (int32_t i = 0; i < n_nodes; i++) {
        if (in_degree[i] == 0) {
            push_node(i);
        }
    }

    while (!ready.empty()) {
        CanonicalKey key = std::move(ready.top());
        ready.pop();

        int32_t u = key.node_id;
        canonical_id_map[u] = static_cast<int32_t>(order.size());
        order.push_back(u);

        for (int32_t v : children[u]) {
            in_degree[v]--;
            if (in_degree[v] == 0) {
                push_node(v);
            }
        }
    }

    py::list res;
    for (int32_t nid : order) res.append(nid);
    return res;
}

// ═══ Module registration — 98 bindings ═══

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "aria_core: Unified high-performance kernel library for Aria";
    m.def("canonical_topo_sort", &canonical_topo_sort, "Stable topological sort for graph fingerprinting");
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
    m.def("linear_cka_f32", &linear_cka_f32, "Linear CKA similarity score");
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
    m.def("difficulty_scorer_f32", &difficulty_scorer_f32);
    m.def("lane_router_threshold_f32", &lane_router_threshold_f32);
    m.def("load_balance_loss_f32", &load_balance_loss_f32);
    m.def("conditional_dispatch_f32", &conditional_dispatch_f32);
    m.def("conditional_dispatch_backward_f32", &conditional_dispatch_backward_f32);
    m.def("conditional_gather_f32", &conditional_gather_f32);
    m.def("conditional_gather_backward_f32", &conditional_gather_backward_f32);
    m.def("adaptive_route_dispatch_f32", &adaptive_route_dispatch_f32);
    m.def("route_topk_indices_f32", &route_topk_indices_f32);
    m.def("route_lane_argmax_f32", &route_lane_argmax_f32);
    m.def("route_recursion_depth_f32", &route_recursion_depth_f32);
    m.def("token_merge_simple_f32", &token_merge_simple_f32);
    
    // Phase 3 Compression (Standardized 2D Linear API)
    m.def("linear_low_rank_f32", &linear_low_rank_f32);
    m.def("linear_block_sparse_f32", &linear_block_sparse_f32);
    m.def("nm_sparse_mask_f32", &nm_sparse_mask_f32);
    m.def("linear_grouped_f32", &linear_grouped_f32);
    m.def("linear_bottleneck_f32", &linear_bottleneck_f32);
    m.def("linear_shared_basis_f32", &linear_shared_basis_f32);
    m.def("linear_tied_f32", &linear_tied_f32);
    
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
    m.def("swiglu_f32", &swiglu_f32);
    m.def("rwkv_channel_f32", &rwkv_channel_f32);
    m.def("gather_topk_f32", &gather_topk_f32);
    // Graph validation & shape inference
    m.def("validate_graph", &validate_graph, "Validate a DAG: cycle detection, topological sort",
          py::arg("n_nodes"), py::arg("edges"), py::arg("op_codes") = std::vector<int32_t>());
    m.def("proactive_gating", &proactive_gating, "Native proactive stability and toxicity gating",
          py::arg("n_nodes"), py::arg("edges"), py::arg("op_codes"),
          py::arg("norm_opcodes"), py::arg("param_opcodes"), py::arg("linear_opcodes"));
    m.def("propagate_shapes", &propagate_shapes, "Propagate tensor shapes through a graph",
          py::arg("topo_order"), py::arg("edges"), py::arg("node_rules"));

    py::class_<PyGraphExecutor>(m, "GraphExecutor")
        .def(py::init<int32_t>())
        .def("set_tensor", &PyGraphExecutor::set_tensor)
        .def("get_tensor", &PyGraphExecutor::get_tensor)
        .def("bake", &PyGraphExecutor::bake)
        .def("execute", &PyGraphExecutor::execute)
        .def("execute_list", &PyGraphExecutor::execute_list);
}
