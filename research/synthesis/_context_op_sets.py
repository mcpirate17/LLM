"""Context rules — shared frozen op sets used by the registry and validation."""

from __future__ import annotations

from typing import FrozenSet


# ── Shared forbidden-predecessor sets ─────────────────────────────
# Many ops share the same "not after reduce" constraint.

_REDUCE_OPS: FrozenSet[str] = frozenset(
    {
        "sum_last",
        "mean_last",
        "max_last",
        "norm_last",
    }
)

_STRUCTURAL_SPLIT_OPS: FrozenSet[str] = frozenset(
    {
        "split2",
        "split3",
    }
)

# Gating/mixture ops — must not directly chain (double-gating produces
# conflicting gradient signals: 100% failure in investigation data).
_GATING_OPS: FrozenSet[str] = frozenset(
    {
        "difficulty_blend_3way",  # was: adaptive_lane_mixer
        "depth_weighted_proj",  # was: adaptive_recursion
        "feature_sparsity",
        "gated_lane_blend",
        "depth_gated_transform",
        "moe_topk",
        "moe_2expert",
        "sparse_bottleneck_moe",  # was: n_way_sparse_router
        "adaptive_rank_gate",  # was: progressive_compression_gate
        "relu_gated_moe",
        "signal_conditioned_compression",
        "confidence_token_gate",  # was: early_exit
        "learned_token_gate",  # was: cascade
        # True routing ops (heterogeneous experts)
        "hetero_moe",
        "arch_router",
        "compute_budget_router",
    }
)
# Backward-compat alias
_ROUTING_OPS = _GATING_OPS

# Mask ops — produce structural masks, not data tensors. Must only feed
# into attention/mixing ops that consume masks.
_MASK_OPS: FrozenSet[str] = frozenset(
    {
        "sliding_window_mask",
        "causal_mask",
    }
)

# Mixing ops that can consume mask output
_MIXING_OPS: FrozenSet[str] = frozenset(
    {
        "softmax_attention",
        "linear_attention",
        "graph_attention",
        "diff_attention",
        "local_window_attn",
        "clifford_attention",
        "difficulty_routed_attention",
        "strided_attention",
        "gated_progressive_attention",
        "gated_linear_attention",
        "associative_memory",
    }
)

# Math-space ops — different algebraic metrics, must not directly chain
# across spaces without normalization.
# NOTE: This is a curated subset of primitives._ALGEBRAIC_SPACE_TAGS (34 ops).
# Only ops with cross-space chaining risks are listed here. If you add a new
# algebraic-space op to _ALGEBRAIC_SPACE_TAGS, check whether it needs a
# chaining constraint here too.
_MATH_SPACE_OPS: FrozenSet[str] = frozenset(
    {
        "clifford_attention",
        "geometric_product",
        "grade_select",
        "rotor_transform",
        "padic_expand",
        "ultrametric_attention",
        "tropical_center",
        "tropical_matmul",
        "tropical_gate",
        "tropical_router",
        "hyp_linear",
        "hyp_tangent_nonlinear",
        "chebyshev_spectral_mix",
    }
)

# Sparse linear ops — internal sparsity patterns break when chained with
# routing or full-width ops expecting dense input
_SPARSE_LINEAR_OPS: FrozenSet[str] = frozenset(
    {
        "nm_sparse_linear",
        "semi_structured_2_4_linear",
        "block_sparse_linear",
    }
)

# Ops that require causal masking context — identity strips it,
# causing guaranteed causality violations when directly preceding these.
_CAUSAL_SENSITIVE_OPS: FrozenSet[str] = frozenset(
    {
        "selective_scan",
        "softmax_attention",
        "linear_attention",
        "graph_attention",
        "diff_attention",
        "state_space",
        "rwkv_channel",
        "rwkv_time_mixing",
        "conv1d_seq",
        "integral_kernel",
        "transpose_sd",
        "difficulty_routed_attention",
        "strided_attention",
        "gated_progressive_attention",
        "gated_linear_attention",
        "long_conv_hyena",
        "associative_memory",
        "mixture_of_recursions",
    }
)

# Ops that do internal Q/K/V or channel expansion — break when receiving
# half-width input from split2 (100% RuntimeError in failure data).
_FULL_DIM_OPS: FrozenSet[str] = frozenset(
    {
        "softmax_attention",
        "linear_attention",
        "graph_attention",
        "diff_attention",
        "state_space",
        "selective_scan",
        "rwkv_channel",
        "rwkv_time_mixing",
        "conv1d_seq",
        "causal_mask",
        "rope_rotate",
        "sliding_window_mask",
        "padic_expand",
        "softmax_attention",
        "gated_delta",
        "difficulty_routed_attention",
        "strided_attention",
        "gated_progressive_attention",
        "gated_linear_attention",
        "long_conv_hyena",
        "associative_memory",
        "mixture_of_recursions",
        "low_rank_proj",
        "basis_expansion",
        "ternary_projection",
    }
)

# Ops that need full model dim to function — down-projected input causes
# RuntimeError (weight dim mismatch) or dead networks (capacity collapse).
_FULL_DIM_CONSUMERS: FrozenSet[str] = frozenset(
    {
        "relu_gated_moe",
        "adaptive_recursion",
        "swiglu_mlp",
        "gated_linear",
        "conv1d_seq",
    }
)

# ── Helper sets used by validation ──────────────────────────────────

_LOCAL_WINDOW_VALID_PREDS = frozenset({"rmsnorm", "layernorm"})
_MASK_VALID_SUCCESSORS = frozenset(
    {"softmax_attention", "linear_attention", "linear_proj"}
)
_REDUCTION_RESTORE_OPS = frozenset({"linear_proj", "linear_proj_up", "concat", "add"})
_TROPICAL_BRIDGE_PREDS = frozenset(
    {"rmsnorm", "layernorm", "tropical_attention", "tropical_gate"}
)
_RESTRICTED_LINEAR_SUCCESSORS = frozenset({"linear_proj", "linear_proj_up", "add"})
_ROUTER_VALID_SUCCESSORS = frozenset({"rmsnorm", "layernorm", "linear_proj"})
# Ops that stabilize unbounded output: projection rescales, norm bounds, mul gates
_STABILIZER_SUCCESSORS = frozenset(
    {
        "linear_proj",
        "linear_proj_down",
        "linear_proj_up",
        "rmsnorm",
        "layernorm",
        "mul",
    }
)
