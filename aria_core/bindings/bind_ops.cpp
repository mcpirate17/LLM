/**
 * bind_ops.cpp — Parameterized ops, routing, compression, geometry
 *                (hyperbolic, tropical, p-adic, clifford), spiking,
 *                SwiGLU, RWKV, IO, and reference architecture bindings.
 */
#include "bind_common.h"

// ═══ Parameterized ═══

static torch::Tensor sliding_window_mask_f32(torch::Tensor x, int64_t ws) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_sliding_window_mask_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), ws); return y; }
static std::tuple<torch::Tensor, torch::Tensor> sort_seq_f32(torch::Tensor x) { CHECK_INPUT(x); auto y = torch::empty_like(x); auto idx = torch::empty({x.size(0), x.size(1), x.size(2)}, torch::dtype(torch::kInt64)); aria_sort_seq_f32(x.data_ptr<float>(), y.data_ptr<float>(), idx.data_ptr<int64_t>(), x.size(0), x.size(1), x.size(2)); return {y, idx}; }
static torch::Tensor argsort_seq_f32(torch::Tensor x) { CHECK_INPUT(x); auto idx = torch::empty({x.size(0), x.size(1), x.size(2)}, torch::dtype(torch::kInt64)); aria_argsort_seq_f32(x.data_ptr<float>(), idx.data_ptr<int64_t>(), x.size(0), x.size(1), x.size(2)); return idx; }
static torch::Tensor conv1d_seq_f32(torch::Tensor x, torch::Tensor w, torch::Tensor b) { CHECK_INPUT(x); CHECK_INPUT(w); CHECK_INPUT(b); auto y = torch::empty_like(x); aria_conv1d_seq_f32(x.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2)); return y; }
static torch::Tensor selective_scan_f32(torch::Tensor x, torch::Tensor A, torch::Tensor B, torch::Tensor C, torch::Tensor D) { CHECK_INPUT(x); CHECK_INPUT(A); CHECK_INPUT(B); CHECK_INPUT(C); CHECK_INPUT(D); auto y = torch::empty_like(x); aria_selective_scan_f32(x.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), D.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2)); return y; }
static torch::Tensor topk_gate_f32(torch::Tensor x, torch::Tensor Wg, int64_t k) { CHECK_INPUT(x); CHECK_INPUT(Wg); auto y = torch::empty_like(x); aria_topk_gate_f32(x.data_ptr<float>(), Wg.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), k); return y; }
static torch::Tensor basis_expansion_f32(torch::Tensor x, torch::Tensor f, int64_t nb) { CHECK_INPUT(x); CHECK_INPUT(f); auto y = torch::empty({x.size(0), x.size(1), x.size(2)*nb}, x.options()); aria_basis_expansion_f32(x.data_ptr<float>(), f.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), nb); return y; }
static torch::Tensor sparse_threshold_f32(torch::Tensor x) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_sparse_threshold_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2)); return y; }
static torch::Tensor token_pool_restore_f32(torch::Tensor x) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_token_pool_restore_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2)); return y; }

static std::tuple<torch::Tensor, torch::Tensor> route_topk_indices_f32(torch::Tensor scores, int64_t k) {
    CHECK_INPUT(scores);
    auto idx = torch::empty({scores.size(0), k}, torch::dtype(torch::kInt64));
    auto w = torch::empty({scores.size(0), k}, scores.options());
    aria_route_topk_indices_f32(scores.data_ptr<float>(), idx.data_ptr<int64_t>(), w.data_ptr<float>(),
                                scores.size(0), scores.size(1), k);
    return {idx, w};
}

static torch::Tensor difficulty_scorer_f32(torch::Tensor x, torch::Tensor w1, c10::optional<torch::Tensor> b1,
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

static std::tuple<torch::Tensor, torch::Tensor> lane_router_threshold_f32(
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

static std::tuple<torch::Tensor, torch::Tensor> load_balance_loss_f32(
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

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> conditional_dispatch_f32(
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

static torch::Tensor conditional_dispatch_backward_f32(torch::Tensor lane_grad, torch::Tensor index_map) {
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

static torch::Tensor conditional_gather_f32(torch::Tensor lane_out, torch::Tensor index_map, torch::Tensor weights) {
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

static std::tuple<torch::Tensor, torch::Tensor> conditional_gather_backward_f32(
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

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
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

static torch::Tensor route_lane_argmax_f32(torch::Tensor scores) {
    CHECK_INPUT(scores);
    auto idx = torch::empty({scores.size(0), scores.size(1)}, torch::dtype(torch::kInt64));
    aria_route_lane_argmax_f32(scores.data_ptr<float>(), idx.data_ptr<int64_t>(),
                               scores.size(0), scores.size(1), scores.size(2));
    return idx;
}

static torch::Tensor route_recursion_depth_f32(torch::Tensor scores) {
    CHECK_INPUT(scores);
    auto depth = torch::empty({scores.size(0), scores.size(1)}, torch::dtype(torch::kInt64));
    aria_route_recursion_depth_f32(scores.data_ptr<float>(), depth.data_ptr<int64_t>(),
                                   scores.size(0), scores.size(1), scores.size(2));
    return depth;
}

static std::tuple<torch::Tensor, torch::Tensor> token_merge_simple_f32(torch::Tensor x, int64_t n_keep) {
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

// ═══ Compression / Linear variants ═══

static torch::Tensor linear_low_rank_f32(torch::Tensor x, torch::Tensor U, torch::Tensor V, c10::optional<torch::Tensor> bias) {
    CHECK_INPUT(x); CHECK_INPUT(U); CHECK_INPUT(V);
    int64_t batch = x.size(0), dim_in = x.size(1), dim_out = V.size(0), rank = U.size(0);
    const float *bp = nullptr;
    if (bias.has_value()) { CHECK_INPUT(bias.value()); bp = bias.value().data_ptr<float>(); }
    auto y = torch::empty({batch, dim_out}, x.options());
    aria_linear_low_rank_f32(x.data_ptr<float>(), U.data_ptr<float>(), V.data_ptr<float>(), bp, y.data_ptr<float>(), batch, dim_in, dim_out, rank);
    return y;
}

static torch::Tensor linear_block_sparse_f32(torch::Tensor x, torch::Tensor W, torch::Tensor mask, c10::optional<torch::Tensor> bias, int64_t bs) {
    CHECK_INPUT(x); CHECK_INPUT(W);
    TORCH_CHECK(mask.dtype() == torch::kUInt8, "mask must be uint8");
    int64_t batch = x.size(0), dim_in = x.size(1), dim_out = W.size(0);
    const float *bp = nullptr;
    if (bias.has_value()) { CHECK_INPUT(bias.value()); bp = bias.value().data_ptr<float>(); }
    auto y = torch::empty({batch, dim_out}, x.options());
    aria_linear_block_sparse_f32(x.data_ptr<float>(), W.data_ptr<float>(), bp, mask.data_ptr<uint8_t>(), y.data_ptr<float>(), batch, dim_in, dim_out, bs);
    return y;
}

static torch::Tensor nm_sparse_mask_f32(torch::Tensor W, int32_t n, int32_t m) {
    CHECK_INPUT(W);
    auto mask = torch::empty_like(W, torch::kUInt8);
    aria_nm_sparse_mask_f32(W.data_ptr<float>(), mask.data_ptr<uint8_t>(), W.size(0), W.size(1), n, m);
    return mask;
}

static torch::Tensor linear_grouped_f32(torch::Tensor x, torch::Tensor W, c10::optional<torch::Tensor> bias, int64_t groups) {
    CHECK_INPUT(x); CHECK_INPUT(W);
    int64_t n_tokens = x.numel() / x.size(-1);
    int64_t dim = x.size(-1);
    const float *bp = nullptr;
    if (bias.has_value()) { CHECK_INPUT(bias.value()); bp = bias.value().data_ptr<float>(); }
    auto y = torch::empty_like(x);
    aria_linear_grouped_f32(x.data_ptr<float>(), W.data_ptr<float>(), bp, y.data_ptr<float>(), n_tokens, dim, groups);
    return y;
}

static torch::Tensor linear_bottleneck_f32(torch::Tensor x, torch::Tensor W_down, torch::Tensor W_up,
                                     c10::optional<torch::Tensor> b_down, c10::optional<torch::Tensor> b_up) {
    CHECK_INPUT(x); CHECK_INPUT(W_down); CHECK_INPUT(W_up);
    int64_t n_tokens = x.numel() / x.size(-1);
    int64_t dim_in = x.size(-1), dim_out = W_up.size(0), rank = W_down.size(0);
    const float *bd = nullptr;
    if (b_down.has_value()) { CHECK_INPUT(b_down.value()); bd = b_down.value().data_ptr<float>(); }
    const float *bu = nullptr;
    if (b_up.has_value()) { CHECK_INPUT(b_up.value()); bu = b_up.value().data_ptr<float>(); }
    auto y_shape = x.sizes().vec();
    y_shape.back() = dim_out;
    auto y = torch::empty(y_shape, x.options());
    aria_linear_bottleneck_f32(x.data_ptr<float>(), W_down.data_ptr<float>(), W_up.data_ptr<float>(), bd, bu, y.data_ptr<float>(), n_tokens, dim_in, dim_out, rank);
    return y;
}

static torch::Tensor linear_shared_basis_f32(torch::Tensor x, torch::Tensor Mixing, torch::Tensor Basis) {
    CHECK_INPUT(x); CHECK_INPUT(Mixing); CHECK_INPUT(Basis);
    int64_t n_tokens = x.numel() / x.size(-1);
    int64_t dim = x.size(-1), k_basis = Mixing.size(0);
    auto y = torch::empty_like(x);
    aria_linear_shared_basis_f32(x.data_ptr<float>(), Mixing.data_ptr<float>(), Basis.data_ptr<float>(), y.data_ptr<float>(), n_tokens, dim, k_basis);
    return y;
}

static torch::Tensor linear_tied_f32(torch::Tensor x, torch::Tensor W, c10::optional<torch::Tensor> b_down, c10::optional<torch::Tensor> b_up) {
    CHECK_INPUT(x); CHECK_INPUT(W);
    int64_t n_tokens = x.numel() / x.size(-1);
    int64_t dim_in = x.size(-1), rank = W.size(0);
    const float *bd = nullptr;
    if (b_down.has_value()) { CHECK_INPUT(b_down.value()); bd = b_down.value().data_ptr<float>(); }
    const float *bu = nullptr;
    if (b_up.has_value()) { CHECK_INPUT(b_up.value()); bu = b_up.value().data_ptr<float>(); }
    auto y = torch::empty_like(x);
    aria_linear_tied_f32(x.data_ptr<float>(), W.data_ptr<float>(), bd, bu, y.data_ptr<float>(), n_tokens, dim_in, rank);
    return y;
}

// ═══ Hyperbolic ═══

static torch::Tensor exp_map_f32(torch::Tensor x, float c) {
    CHECK_INPUT(x); auto y = torch::empty_like(x);
    int64_t dim = x.size(-1), batch = x.numel() / dim;
    aria_exp_map_f32(x.data_ptr<float>(), y.data_ptr<float>(), batch, dim, c);
    return y;
}
static torch::Tensor log_map_f32(torch::Tensor x, float c) {
    CHECK_INPUT(x); auto y = torch::empty_like(x);
    int64_t dim = x.size(-1), batch = x.numel() / dim;
    aria_log_map_f32(x.data_ptr<float>(), y.data_ptr<float>(), batch, dim, c);
    return y;
}
static torch::Tensor poincare_add_f32(torch::Tensor x, torch::Tensor v, float c) { CHECK_INPUT(x); CHECK_INPUT(v); auto y = torch::empty_like(x); aria_poincare_add_f32(x.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), c); return y; }
static torch::Tensor hyp_linear_f32(torch::Tensor x, torch::Tensor W, float c) { CHECK_INPUT(x); CHECK_INPUT(W); auto y = torch::empty({x.size(0), W.size(0)}, x.options()); aria_hyp_linear_f32(x.data_ptr<float>(), W.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), W.size(0), c); return y; }
static torch::Tensor hyperbolic_norm_f32(torch::Tensor x, torch::Tensor g, torch::Tensor b, float c, float eps) { CHECK_INPUT(x); CHECK_INPUT(g); CHECK_INPUT(b); auto y = torch::empty_like(x); aria_hyperbolic_norm_f32(x.data_ptr<float>(), g.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), c, eps); return y; }
static torch::Tensor hyp_tangent_nonlinear_f32(torch::Tensor x, float c) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_hyp_tangent_nonlinear_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.numel(), c); return y; }
static torch::Tensor hyp_distance_f32(torch::Tensor x, torch::Tensor yi) { CHECK_INPUT(x); CHECK_INPUT(yi); auto out = torch::empty({x.size(0), x.size(1)}, x.options()); aria_hyp_distance_f32(x.data_ptr<float>(), yi.data_ptr<float>(), out.data_ptr<float>(), x.size(0), x.size(1), x.size(2)); return out; }
static torch::Tensor hyperbolic_mobius_add_f32(torch::Tensor x, torch::Tensor v, float c) { CHECK_INPUT(x); CHECK_INPUT(v); auto y = torch::empty_like(x); aria_hyperbolic_mobius_add_f32(x.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), c); return y; }
static torch::Tensor hyperbolic_distance_f32(torch::Tensor x, torch::Tensor yi, float c) { CHECK_INPUT(x); CHECK_INPUT(yi); auto out = torch::empty({x.size(0)}, x.options()); aria_hyperbolic_distance_f32(x.data_ptr<float>(), yi.data_ptr<float>(), out.data_ptr<float>(), x.size(0), x.size(1), c); return out; }

static torch::Tensor exp_map_backward_f32(torch::Tensor v, torch::Tensor grad_out, float c) {
    CHECK_INPUT(v); CHECK_INPUT(grad_out);
    auto grad_in = torch::empty_like(v);
    int64_t dim = v.size(-1), batch = v.numel() / dim;
    aria_exp_map_backward_f32(v.data_ptr<float>(), grad_out.data_ptr<float>(), grad_in.data_ptr<float>(), batch, dim, c);
    return grad_in;
}
static torch::Tensor log_map_backward_f32(torch::Tensor x, torch::Tensor grad_out, float c) {
    CHECK_INPUT(x); CHECK_INPUT(grad_out);
    auto grad_in = torch::empty_like(x);
    int64_t dim = x.size(-1), batch = x.numel() / dim;
    aria_log_map_backward_f32(x.data_ptr<float>(), grad_out.data_ptr<float>(), grad_in.data_ptr<float>(), batch, dim, c);
    return grad_in;
}

// ═══ Tropical ═══

static torch::Tensor tropical_center_f32(torch::Tensor x) {
    CHECK_INPUT_ANY(x);
    auto y = torch::empty_like(x);
    if (x.is_cuda()) {
        launch_cuda_tropical_center_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2));
    } else {
        aria_tropical_center_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2));
    }
    return y;
}
static torch::Tensor tropical_attention_f32(torch::Tensor x, float t) {
    CHECK_INPUT_ANY(x);
    auto y = torch::empty_like(x);
    if (x.is_cuda()) {
        launch_cuda_tropical_attention_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), t);
    } else {
        aria_tropical_attention_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), t);
    }
    return y;
}
static torch::Tensor tropical_gate_f32(torch::Tensor x, float t) {
    CHECK_INPUT_ANY(x);
    auto y = torch::empty_like(x);
    if (x.is_cuda()) {
        launch_cuda_tropical_gate_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), t);
    } else {
        aria_tropical_gate_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), t);
    }
    return y;
}
static torch::Tensor tropical_router_f32(torch::Tensor x, torch::Tensor centroids) {
    CHECK_INPUT(x); CHECK_INPUT(centroids);
    int64_t B = x.size(0), S = x.size(1), D = x.size(2), E = centroids.size(0);
    auto y = torch::empty({B, S, E}, x.options());
    aria_tropical_router_f32(x.data_ptr<float>(), centroids.data_ptr<float>(), y.data_ptr<float>(), B, S, D, E);
    return y;
}

// ═══ P-adic ═══

static torch::Tensor padic_gate_f32(torch::Tensor x, float p) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_padic_gate_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.numel(), p); return y; }
static torch::Tensor padic_expand_f32(torch::Tensor x, torch::Tensor W, float p, int64_t nd) { CHECK_INPUT(x); CHECK_INPUT(W); auto y = torch::empty({x.size(0), x.size(1)*nd}, x.options()); aria_padic_expand_f32(x.data_ptr<float>(), W.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), p, nd); return y; }
static torch::Tensor padic_residual_f32(torch::Tensor x, torch::Tensor W, float p, int64_t nd) { CHECK_INPUT(x); CHECK_INPUT(W); auto y = torch::empty_like(x); aria_padic_residual_f32(x.data_ptr<float>(), W.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), p, nd); return y; }
static torch::Tensor ultrametric_attention_f32(torch::Tensor x, float p) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_ultrametric_attention_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), p); return y; }

// ═══ Clifford ═══

static torch::Tensor rotor_transform_f32(torch::Tensor x, torch::Tensor r) { CHECK_INPUT(x); CHECK_INPUT(r); auto y = torch::empty_like(x); aria_rotor_transform_f32(x.data_ptr<float>(), r.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1)); return y; }
static torch::Tensor grade_select_f32(torch::Tensor x, int32_t g) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_grade_select_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), g); return y; }
static torch::Tensor grade_mix_f32(torch::Tensor x, torch::Tensor a) { CHECK_INPUT(x); CHECK_INPUT(a); auto y = torch::empty_like(x); aria_grade_mix_f32(x.data_ptr<float>(), a.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1)); return y; }
static torch::Tensor clifford_attention_f32(torch::Tensor x) {
    CHECK_INPUT_ANY(x);
    auto y = torch::empty_like(x);
    if (x.is_cuda()) {
        launch_cuda_clifford_attention_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2));
    } else {
        aria_clifford_attention_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2));
    }
    return y;
}
static torch::Tensor clifford_geometric_product_cl30_f32(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT_ANY(a); CHECK_INPUT_ANY(b);
    auto y = torch::empty_like(a);
    if (a.is_cuda()) {
        launch_cuda_clifford_geometric_product_cl30_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel());
    } else {
        aria_clifford_geometric_product_cl30_f32(a.data_ptr<float>(), b.data_ptr<float>(), y.data_ptr<float>(), a.numel()/8);
    }
    return y;
}
static torch::Tensor clifford_rotor_transform_cl30_f32(torch::Tensor x, torch::Tensor r) { CHECK_INPUT(x); CHECK_INPUT(r); auto y = torch::empty_like(x); aria_clifford_rotor_transform_cl30_f32(x.data_ptr<float>(), r.data_ptr<float>(), y.data_ptr<float>(), x.numel()/8); return y; }

// ═══ Spiking ═══

static torch::Tensor lif_neuron_f32(torch::Tensor x, float tau, float thr) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_lif_neuron_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), tau, thr); return y; }
static std::vector<torch::Tensor> lif_neuron_with_state_f32(torch::Tensor x, float tau, float thr) {
    CHECK_INPUT(x);
    auto spikes = torch::empty_like(x);
    auto membrane = torch::empty_like(x);
    aria_lif_neuron_with_state_f32(x.data_ptr<float>(), spikes.data_ptr<float>(),
                                    membrane.data_ptr<float>(),
                                    x.size(0), x.size(1), x.size(2), tau, thr);
    return {spikes, membrane};
}
static torch::Tensor spike_rate_code_f32(torch::Tensor x) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_spike_rate_code_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2)); return y; }
static torch::Tensor stdp_attention_f32(torch::Tensor x, float tp, float tm) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_stdp_attention_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), tp, tm); return y; }

// ═══ SwiGLU / RWKV Channel / Gather Top-K ═══

static torch::Tensor swiglu_f32(torch::Tensor x, torch::Tensor W_gate, torch::Tensor W_up, torch::Tensor W_down,
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

static torch::Tensor rwkv_channel_f32(torch::Tensor x, torch::Tensor mix_k, torch::Tensor mix_r,
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

static std::tuple<torch::Tensor, torch::Tensor> gather_topk_f32(torch::Tensor scores, torch::Tensor values, int64_t k) {
    CHECK_INPUT(scores); CHECK_INPUT(values);
    int64_t batch = scores.size(0), n_items = scores.size(1), dim = values.size(2);
    auto out = torch::empty({batch, k, dim}, values.options());
    auto out_indices = torch::empty({batch, k}, torch::dtype(torch::kInt32));
    aria_gather_topk_f32(scores.data_ptr<float>(), values.data_ptr<float>(),
                         out.data_ptr<float>(), out_indices.data_ptr<int32_t>(),
                         batch, n_items, dim, k);
    return {out, out_indices};
}

// ═══ IO kernels ═══

static torch::Tensor read_csv_f32(const std::string &filename, int64_t max_rows, int64_t max_cols, char delimiter) {
    auto out = torch::zeros({max_rows, max_cols}, torch::dtype(torch::kFloat32));
    int rows = aria_read_csv_f32(filename.c_str(), out.data_ptr<float>(), max_rows, max_cols, delimiter);
    TORCH_CHECK(rows >= 0, "failed to read csv: ", filename);
    return out.narrow(0, 0, rows).clone();
}

static torch::Tensor filter_f32(torch::Tensor data, int64_t col_idx, float val, int op) {
    CHECK_INPUT(data);
    TORCH_CHECK(data.dim() == 2, "data must be 2D");
    auto out = torch::empty_like(data);
    int rows = aria_filter_f32(data.data_ptr<float>(), out.data_ptr<float>(), data.size(0), data.size(1), col_idx, val, op);
    TORCH_CHECK(rows >= 0, "filter_f32 failed");
    return out.narrow(0, 0, rows).clone();
}

static torch::Tensor file_loader_csv_f32(const std::string &filename, int64_t max_rows, int64_t max_cols, char delimiter, bool has_header) {
    auto out = torch::zeros({max_rows, max_cols}, torch::dtype(torch::kFloat32));
    int rows = aria_file_loader_csv_f32(filename.c_str(), out.data_ptr<float>(), max_rows, max_cols, delimiter, has_header ? 1 : 0);
    TORCH_CHECK(rows >= 0, "failed to load csv: ", filename);
    return out.narrow(0, 0, rows).clone();
}

static torch::Tensor binary_file_reader_f32(const std::string &filename, int64_t max_elems, int64_t offset_bytes) {
    auto out = torch::zeros({max_elems}, torch::dtype(torch::kFloat32));
    int n = aria_binary_file_reader_f32(filename.c_str(), out.data_ptr<float>(), max_elems, offset_bytes);
    TORCH_CHECK(n >= 0, "failed to read binary file: ", filename);
    return out.narrow(0, 0, n).clone();
}

static int file_writer_txt_f32(const std::string &filename, torch::Tensor data, bool overwrite) {
    CHECK_INPUT(data);
    auto flat = data.reshape({-1}).contiguous();
    return aria_file_writer_txt_f32(filename.c_str(), flat.data_ptr<float>(), flat.numel(), overwrite ? 1 : 0);
}

// ═══ Reference Architecture ═══

static torch::Tensor embedding_lookup_f32(torch::Tensor table, torch::Tensor idx, c10::optional<torch::Tensor> pe) {
    CHECK_INPUT(table);
    const float *pp = nullptr; if (pe.has_value()) { CHECK_INPUT(pe.value()); pp = pe.value().data_ptr<float>(); }
    auto y = torch::empty({idx.size(0), table.size(1)}, table.options());
    aria_embedding_lookup_f32(table.data_ptr<float>(), idx.data_ptr<int32_t>(), pp, y.data_ptr<float>(), idx.size(0), table.size(1), table.size(0));
    return y;
}
static torch::Tensor rope_rotate_f32(torch::Tensor x, float tb) { CHECK_INPUT(x); auto y = torch::empty_like(x); aria_rope_rotate_f32(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2), tb); return y; }
static torch::Tensor gated_linear_f32(torch::Tensor x, torch::Tensor W, c10::optional<torch::Tensor> b, torch::Tensor Wg, c10::optional<torch::Tensor> bg) {
    CHECK_INPUT(x); CHECK_INPUT(W); CHECK_INPUT(Wg);
    const float *bp=nullptr, *bgp=nullptr;
    if (b.has_value()) { CHECK_INPUT(b.value()); bp = b.value().data_ptr<float>(); }
    if (bg.has_value()) { CHECK_INPUT(bg.value()); bgp = bg.value().data_ptr<float>(); }
    int64_t batch=x.size(0), di=x.size(1), do_=W.size(0);
    auto y = torch::empty({batch, do_}, x.options()); auto tmp = torch::empty({batch, do_}, x.options());
    aria_gated_linear_f32(x.data_ptr<float>(), W.data_ptr<float>(), bp, Wg.data_ptr<float>(), bgp, y.data_ptr<float>(), tmp.data_ptr<float>(), batch, di, do_);
    return y;
}
static torch::Tensor cosine_similarity_f32(torch::Tensor a, torch::Tensor b) { CHECK_INPUT(a); CHECK_INPUT(b); auto out = torch::empty({a.size(0), a.size(1)}, a.options()); aria_cosine_similarity_f32(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), a.size(0), a.size(1), a.size(2)); return out; }
static torch::Tensor rwkv_time_mixing_f32(torch::Tensor x, torch::Tensor wd, torch::Tensor ub, torch::Tensor Wk, torch::Tensor Wv, torch::Tensor Wr) {
    CHECK_INPUT(x); CHECK_INPUT(wd); CHECK_INPUT(ub); CHECK_INPUT(Wk); CHECK_INPUT(Wv); CHECK_INPUT(Wr);
    auto y = torch::empty_like(x);
    aria_rwkv_time_mixing_f32(x.data_ptr<float>(), wd.data_ptr<float>(), ub.data_ptr<float>(), Wk.data_ptr<float>(), Wv.data_ptr<float>(), Wr.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1), x.size(2));
    return y;
}
static torch::Tensor rwkv_wkv_scan_f32(torch::Tensor k, torch::Tensor v, torch::Tensor r,
                                  torch::Tensor w_decay, torch::Tensor u_bonus) {
    CHECK_INPUT(k); CHECK_INPUT(v); CHECK_INPUT(r);
    CHECK_INPUT(w_decay); CHECK_INPUT(u_bonus);
    auto out = torch::empty_like(k);
    aria_rwkv_wkv_scan_f32(k.data_ptr<float>(), v.data_ptr<float>(), r.data_ptr<float>(),
                            w_decay.data_ptr<float>(), u_bonus.data_ptr<float>(),
                            out.data_ptr<float>(), k.size(0), k.size(1), k.size(2));
    return out;
}

static std::tuple<torch::Tensor, torch::Tensor> embedding_lookup_backward_f32(
    torch::Tensor grad_out, torch::Tensor indices, int64_t vocab_size, bool with_pos_embed) {
    CHECK_INPUT(grad_out);
    TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must be int32");
    CHECK_CPU(indices); CHECK_CONTIGUOUS(indices);
    TORCH_CHECK(grad_out.dim() == 2, "grad_out must be [batch, dim]");
    auto grad_table = torch::zeros({vocab_size, grad_out.size(1)}, grad_out.options());
    auto grad_pos_embed = with_pos_embed ? torch::zeros_like(grad_out) : torch::Tensor();
    aria_embedding_lookup_backward_f32(
        grad_out.data_ptr<float>(), indices.data_ptr<int32_t>(), grad_table.data_ptr<float>(),
        with_pos_embed ? grad_pos_embed.data_ptr<float>() : nullptr,
        grad_out.size(0), grad_out.size(1), vocab_size
    );
    return {grad_table, grad_pos_embed};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> gated_linear_backward_f32(
    torch::Tensor grad_out, torch::Tensor x, torch::Tensor W, torch::Tensor W_gate, torch::Tensor gate_sigmoid) {
    CHECK_INPUT(grad_out); CHECK_INPUT(x); CHECK_INPUT(W); CHECK_INPUT(W_gate); CHECK_INPUT(gate_sigmoid);
    TORCH_CHECK(grad_out.dim() == 2 && x.dim() == 2, "grad_out and x must be 2D");
    auto grad_x = torch::zeros_like(x);
    auto grad_W = torch::zeros_like(W);
    auto grad_W_gate = torch::zeros_like(W_gate);
    auto grad_b = torch::zeros({W.size(0)}, x.options());
    auto grad_b_gate = torch::zeros({W_gate.size(0)}, x.options());
    aria_gated_linear_backward_f32(
        grad_out.data_ptr<float>(), x.data_ptr<float>(), W.data_ptr<float>(), W_gate.data_ptr<float>(),
        gate_sigmoid.data_ptr<float>(), grad_x.data_ptr<float>(), grad_W.data_ptr<float>(),
        grad_W_gate.data_ptr<float>(), grad_b.data_ptr<float>(), grad_b_gate.data_ptr<float>(),
        x.size(0), x.size(1), grad_out.size(1)
    );
    return {grad_x, grad_W, grad_W_gate, grad_b, grad_b_gate};
}

// ── Gromov Delta ──

static float gromov_delta_f32(torch::Tensor distance_matrix, torch::Tensor indices) {
    CHECK_INPUT(distance_matrix);
    int64_t n = distance_matrix.size(0);
    int64_t n_idx = indices.numel();
    auto indices_int = indices.to(torch::kInt32).contiguous();
    return aria_gromov_delta_f32(distance_matrix.data_ptr<float>(),
                                 indices_int.data_ptr<int32_t>(), n, n_idx);
}

// ═══ Registration ═══

void bind_ops(py::module_ &m) {
    // Parameterized
    m.def("sliding_window_mask_f32", &sliding_window_mask_f32); m.def("sort_seq_f32", &sort_seq_f32);
    m.def("argsort_seq_f32", &argsort_seq_f32);
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
    // Compression / Linear variants
    m.def("linear_low_rank_f32", &linear_low_rank_f32);
    m.def("linear_block_sparse_f32", &linear_block_sparse_f32);
    m.def("nm_sparse_mask_f32", &nm_sparse_mask_f32);
    m.def("linear_grouped_f32", &linear_grouped_f32);
    m.def("linear_bottleneck_f32", &linear_bottleneck_f32);
    m.def("linear_shared_basis_f32", &linear_shared_basis_f32);
    m.def("linear_tied_f32", &linear_tied_f32);
    // Hyperbolic
    m.def("exp_map_f32", &exp_map_f32); m.def("log_map_f32", &log_map_f32);
    m.def("exp_map_backward_f32", &exp_map_backward_f32); m.def("log_map_backward_f32", &log_map_backward_f32);
    m.def("poincare_add_f32", &poincare_add_f32); m.def("hyp_linear_f32", &hyp_linear_f32);
    m.def("hyperbolic_norm_f32", &hyperbolic_norm_f32); m.def("hyp_tangent_nonlinear_f32", &hyp_tangent_nonlinear_f32);
    m.def("hyp_distance_f32", &hyp_distance_f32); m.def("hyperbolic_mobius_add_f32", &hyperbolic_mobius_add_f32);
    m.def("hyperbolic_distance_f32", &hyperbolic_distance_f32);
    // Tropical
    m.def("tropical_center_f32", &tropical_center_f32);
    m.def("tropical_attention_f32", &tropical_attention_f32);
    m.def("tropical_gate_f32", &tropical_gate_f32);
    m.def("tropical_router_f32", &tropical_router_f32);
    // P-adic
    m.def("padic_gate_f32", &padic_gate_f32);
    m.def("padic_expand_f32", &padic_expand_f32);
    m.def("padic_residual_f32", &padic_residual_f32); m.def("ultrametric_attention_f32", &ultrametric_attention_f32);
    // Clifford
    m.def("rotor_transform_f32", &rotor_transform_f32); m.def("grade_select_f32", &grade_select_f32);
    m.def("grade_mix_f32", &grade_mix_f32); m.def("clifford_attention_f32", &clifford_attention_f32);
    m.def("clifford_geometric_product_cl30_f32", &clifford_geometric_product_cl30_f32);
    m.def("clifford_rotor_transform_cl30_f32", &clifford_rotor_transform_cl30_f32);
    // Spiking
    m.def("lif_neuron_f32", &lif_neuron_f32); m.def("lif_neuron_with_state_f32", &lif_neuron_with_state_f32);
    m.def("spike_rate_code_f32", &spike_rate_code_f32);
    m.def("stdp_attention_f32", &stdp_attention_f32);
    // SwiGLU / RWKV / Gather
    m.def("swiglu_f32", &swiglu_f32);
    m.def("rwkv_channel_f32", &rwkv_channel_f32);
    m.def("gather_topk_f32", &gather_topk_f32);
    // IO
    m.def("read_csv_f32", &read_csv_f32, py::arg("filename"), py::arg("max_rows"), py::arg("max_cols"), py::arg("delimiter") = ',');
    m.def("filter_f32", &filter_f32, py::arg("data"), py::arg("col_idx"), py::arg("val"), py::arg("op"));
    m.def("file_loader_csv_f32", &file_loader_csv_f32,
          py::arg("filename"), py::arg("max_rows"), py::arg("max_cols"),
          py::arg("delimiter") = ',', py::arg("has_header") = false);
    m.def("binary_file_reader_f32", &binary_file_reader_f32,
          py::arg("filename"), py::arg("max_elems"), py::arg("offset_bytes") = 0);
    m.def("file_writer_txt_f32", &file_writer_txt_f32,
          py::arg("filename"), py::arg("data"), py::arg("overwrite") = false);
    // Reference arch
    m.def("embedding_lookup_f32", &embedding_lookup_f32); m.def("rope_rotate_f32", &rope_rotate_f32);
    m.def("gated_linear_f32", &gated_linear_f32); m.def("cosine_similarity_f32", &cosine_similarity_f32);
    m.def("rwkv_time_mixing_f32", &rwkv_time_mixing_f32);
    m.def("rwkv_wkv_scan_f32", &rwkv_wkv_scan_f32);
    m.def("embedding_lookup_backward_f32", &embedding_lookup_backward_f32);
    m.def("gated_linear_backward_f32", &gated_linear_backward_f32);
    m.def("gromov_delta_f32", &gromov_delta_f32);
}
