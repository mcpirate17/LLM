"""
Op Role Classification for Motif-Based Grammar

Every op in PRIMITIVE_REGISTRY gets a functional role tag.
The grammar composes ops in role-valid sequences only.

Roles:
  project   — Learnable weight matrix. Must appear in every block.
  normalize — Stabilizes activations. Placed before mix or project.
  activate  — Pointwise nonlinearity. Follows project or gate.
  mix       — Sequence-level mixing (attention, SSM, conv).
  route     — Controls information flow (splits, MoE, routing).
  gate      — Multiplicative modulation. Pairs with project.
  position  — Positional info. Applied once near input.
  reduce    — Dimension reduction ops. Used in specific contexts only.
  residual  — Binary ops for skip connections (add, mul).
  unsafe    — Fallback for truly unknown ops. Not used for known ops;
              context rules enforce valid placement instead of blanket exclusion.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet


class OpRole(Enum):
    __slots__ = ()
    PROJECT = "project"
    NORMALIZE = "normalize"
    ACTIVATE = "activate"
    MIX = "mix"
    ROUTE = "route"
    GATE = "gate"
    POSITION = "position"
    REDUCE = "reduce"
    RESIDUAL = "residual"
    UNSAFE = "unsafe"


# ── Explicit role assignments for all known ops ─────────────────────

_OP_ROLE_MAP: Dict[str, OpRole] = {
    # ── PROJECT: learnable weight matrices ──────────────────────────
    "linear_proj": OpRole.PROJECT,
    "linear_proj_down": OpRole.PROJECT,
    "linear_proj_up": OpRole.PROJECT,
    "fused_linear_gelu": OpRole.PROJECT,
    "nm_sparse_linear": OpRole.PROJECT,
    "block_sparse_linear": OpRole.PROJECT,
    "semi_structured_2_4_linear": OpRole.PROJECT,
    "ternary_projection": OpRole.PROJECT,
    "conv1d_seq": OpRole.PROJECT,
    "basis_expansion": OpRole.PROJECT,
    "embedding_lookup": OpRole.PROJECT,  # learnable codebook projection (soft VQ)
    # ── NORMALIZE: stabilize activations ────────────────────────────
    "rmsnorm": OpRole.NORMALIZE,
    "layernorm": OpRole.NORMALIZE,
    "qk_norm": OpRole.NORMALIZE,
    "logit_softcap": OpRole.NORMALIZE,
    # ── ACTIVATE: pointwise nonlinearities ──────────────────────────
    "relu": OpRole.ACTIVATE,
    "gelu": OpRole.ACTIVATE,
    "silu": OpRole.ACTIVATE,
    "tanh": OpRole.ACTIVATE,
    "sigmoid": OpRole.ACTIVATE,
    "sin": OpRole.ACTIVATE,
    "cos": OpRole.ACTIVATE,
    "softmax_last": OpRole.ACTIVATE,
    # ── MIX: sequence-level mixing ──────────────────────────────────
    "softmax_attention": OpRole.MIX,
    "sparsemax_attention": OpRole.MIX,
    "entmax_attention": OpRole.MIX,
    "linear_attention": OpRole.MIX,
    "graph_attention": OpRole.MIX,
    "local_window_attn": OpRole.MIX,
    "state_space": OpRole.MIX,
    "selective_scan": OpRole.MIX,
    "rwkv_time_mixing": OpRole.MIX,
    "rwkv_channel": OpRole.MIX,
    "integral_kernel": OpRole.MIX,
    "fixed_point_iter": OpRole.MIX,
    "latent_attention_compressor": OpRole.MIX,
    "diff_attention": OpRole.MIX,
    "gated_delta": OpRole.MIX,
    "dplr_gated_delta": OpRole.MIX,
    "difficulty_routed_attention": OpRole.MIX,
    "strided_attention": OpRole.MIX,
    "gated_progressive_attention": OpRole.MIX,
    # gated_linear_attention disabled 2026-05-23: proven anti-causal (next-token
    # leak via whole-chunk kv/decay), records purged, NOT fixed. Removed from the
    # MIX pool so the grammar can't insert it. Fixed replacement pending (gemini).
    # See Obsidian note `adjacent_token_merge_leak_2026-05-23`
    "long_conv_hyena": OpRole.MIX,
    "associative_memory": OpRole.MIX,
    "mixture_of_recursions": OpRole.MIX,
    "token_hodge_mixer": OpRole.MIX,
    "wavelet_packet_mix": OpRole.MIX,
    "retention_mix": OpRole.MIX,
    "product_key_memory": OpRole.MIX,
    "conv_only": OpRole.MIX,
    "multi_head_mix": OpRole.MIX,
    # ── ROUTE: information flow control ─────────────────────────────
    "split2": OpRole.ROUTE,
    "split3": OpRole.ROUTE,
    "concat": OpRole.ROUTE,
    "moe_topk": OpRole.ROUTE,
    "moe_2expert": OpRole.ROUTE,
    "depth_token_mask": OpRole.ROUTE,  # was: mod_topk
    "confidence_token_gate": OpRole.ROUTE,  # was: early_exit
    "depth_weighted_proj": OpRole.ROUTE,  # was: adaptive_recursion
    "adjacent_token_merge": OpRole.ROUTE,  # was: token_merge
    "learned_token_gate": OpRole.ROUTE,  # was: cascade
    "cheap_verify_blend": OpRole.ROUTE,  # was: speculative
    "hybrid_token_gate": OpRole.ROUTE,
    "sparse_span_builder": OpRole.ROUTE,
    "hybrid_sparse_router": OpRole.ROUTE,
    "lane_conditioned_block": OpRole.ROUTE,
    "default_path": OpRole.ROUTE,
    "feature_sparsity": OpRole.ROUTE,
    "route_topk": OpRole.ROUTE,  # alias for feature_sparsity
    "gated_lane_blend": OpRole.ROUTE,
    "route_lanes": OpRole.ROUTE,  # alias for gated_lane_blend
    "depth_gated_transform": OpRole.ROUTE,
    "route_recursion": OpRole.ROUTE,  # alias for depth_gated_transform
    "difficulty_blend_3way": OpRole.ROUTE,  # was: adaptive_lane_mixer
    "score_depth_blend": OpRole.ROUTE,  # was: mixed_recursion_gate
    "signal_conditioned_compression": OpRole.ROUTE,
    # True token-routing ops (heterogeneous experts)
    "hetero_moe": OpRole.ROUTE,
    "arch_router": OpRole.ROUTE,
    "compute_budget_router": OpRole.ROUTE,
    "dual_compression_blend": OpRole.ROUTE,  # was: compression_mixture_experts
    "gather_topk": OpRole.ROUTE,
    "sparse_bottleneck_moe": OpRole.ROUTE,  # was: n_way_sparse_router
    # ── GATE: multiplicative modulation ─────────────────────────────
    "gated_linear": OpRole.GATE,
    "swiglu_mlp": OpRole.GATE,
    "learnable_scale": OpRole.GATE,
    "learnable_bias": OpRole.GATE,
    "topk_gate": OpRole.GATE,
    "relu_gated_moe": OpRole.GATE,
    "adaptive_rank_gate": OpRole.GATE,  # was: progressive_compression_gate
    "token_class_proj": OpRole.GATE,  # was: token_type_classifier
    "token_entropy": OpRole.GATE,  # was: entropy_score
    # ── POSITION: positional information ────────────────────────────
    "rope_rotate": OpRole.POSITION,
    "causal_mask": OpRole.POSITION,
    "sliding_window_mask": OpRole.POSITION,
    # ── RESIDUAL: binary skip-connection ops ────────────────────────
    "add": OpRole.RESIDUAL,
    "mul": OpRole.RESIDUAL,
    "sub": OpRole.RESIDUAL,
    "maximum": OpRole.RESIDUAL,
    "minimum": OpRole.RESIDUAL,
    # ── REDUCE: dimension reduction ─────────────────────────────────
    "sum_last": OpRole.REDUCE,
    "mean_last": OpRole.REDUCE,
    "max_last": OpRole.REDUCE,
    "norm_last": OpRole.REDUCE,
    "cumsum": OpRole.REDUCE,
    # ── Context-safe activations (safe within motif-provided context) ─
    "exp": OpRole.ACTIVATE,
    "log": OpRole.ACTIVATE,
    "sqrt": OpRole.ACTIVATE,
    "abs": OpRole.ACTIVATE,
    "square": OpRole.ACTIVATE,
    "neg": OpRole.ACTIVATE,
    "sign_ste": OpRole.ACTIVATE,
    "reciprocal": OpRole.ACTIVATE,
    # ── Context-safe projections (learnable, unconditionally safe) ──
    "bottleneck_proj": OpRole.PROJECT,
    "grouped_linear": OpRole.PROJECT,
    "low_rank_proj": OpRole.PROJECT,
    "shared_basis_proj": OpRole.PROJECT,
    "tied_proj": OpRole.PROJECT,
    # ── Context-safe mixing ───────────────────────────────────────
    "rotor_transform": OpRole.MIX,
    "grade_select": OpRole.MIX,
    "transpose_sd": OpRole.MIX,
    "stdp_attention": OpRole.MIX,
    "hyperbolic_norm": OpRole.MIX,
    # ── Context-safe routing/gating ───────────────────────────────
    "tropical_moe": OpRole.ROUTE,
    "tropical_router": OpRole.GATE,
    "padic_gate": OpRole.GATE,
    # ── Context-safe spiking activations ──────────────────────────
    "lif_neuron": OpRole.ACTIVATE,
    "sparse_threshold": OpRole.ACTIVATE,
    "spike_rate_code": OpRole.ACTIVATE,
    # ── Context-safe residual ─────────────────────────────────────
    "identity": OpRole.RESIDUAL,
    "poincare_add": OpRole.RESIDUAL,
    # ── Learnable-semiring attention ─────────────────────────────
    "learnable_semiring_attention": OpRole.MIX,
    "reciprocal_rank_attention": OpRole.MIX,
    "phase_lock_attention": OpRole.MIX,
    # ── PQ-bottleneck MoE block ──────────────────────────────────
    "pq_embedding_moe_block": OpRole.MIX,
    # ── Math-space: tropical ─────────────────────────────────────
    "tropical_attention": OpRole.MIX,
    "tropical_gate": OpRole.GATE,
    "tropical_center": OpRole.NORMALIZE,
    "tropical_add": OpRole.RESIDUAL,
    "tropical_matmul": OpRole.MIX,
    # ── Math-space: clifford ─────────────────────────────────────
    "clifford_attention": OpRole.MIX,
    "grade_mix": OpRole.MIX,
    # ── Math-space: hyperbolic ─────────────────────────────────
    "hyp_distance": OpRole.REDUCE,
    "hyp_linear": OpRole.PROJECT,
    "exp_map": OpRole.MIX,
    "log_map": OpRole.MIX,
    # ── Math-space: p-adic ───────────────────────────────────────
    "padic_expand": OpRole.PROJECT,
    "padic_residual": OpRole.RESIDUAL,
    "ultrametric_attention": OpRole.MIX,
    # ── Frequency ────────────────────────────────────────────────
    "spectral_filter": OpRole.MIX,
    "chebyshev_spectral_mix": OpRole.MIX,
    "kronecker_linear": OpRole.PROJECT,
    "hyp_tangent_nonlinear": OpRole.ACTIVATE,
    # ── Binary / multi-input ops (context rules enforce valid placement) ──
    "div_safe": OpRole.RESIDUAL,
    "cumprod_safe": OpRole.REDUCE,
    "matmul": OpRole.PROJECT,
    "outer_product": OpRole.PROJECT,
    "cosine_similarity": OpRole.REDUCE,
    "geometric_product": OpRole.MIX,
    # ── Virtual graph nodes (not real ops, neutral role) ──────────
    "input": OpRole.RESIDUAL,
    "output": OpRole.RESIDUAL,
}


# ── Fallback classification by OpCategory ───────────────────────────
# For dynamically loaded ops (designer manifests, math spaces).

_CATEGORY_ROLE_FALLBACK: Dict[str, OpRole] = {
    "elementwise_unary": OpRole.ACTIVATE,
    "elementwise_binary": OpRole.RESIDUAL,
    "reduction": OpRole.REDUCE,
    "linear_algebra": OpRole.PROJECT,
    "structural": OpRole.ROUTE,
    "parameterized": OpRole.PROJECT,
    "mixing": OpRole.MIX,
    "sequence": OpRole.MIX,
    "frequency": OpRole.MIX,
    "math_space": OpRole.MIX,
    "functional": OpRole.ROUTE,
}


def get_role(op_name: str) -> OpRole:
    """Get the functional role for an op. O(1) dict lookup."""
    role = _OP_ROLE_MAP.get(op_name)
    if role is not None:
        return role
    # Fallback: classify by OpCategory from registry
    from .primitives import PRIMITIVE_REGISTRY

    prim = PRIMITIVE_REGISTRY.get(op_name)
    if prim is not None:
        return _CATEGORY_ROLE_FALLBACK.get(prim.category.value, OpRole.ACTIVATE)
    return OpRole.ACTIVATE


# ── Role-valid transition rules ─────────────────────────────────────
# Maps each role to the set of roles that can legally follow it.
# Used by motif validation and template slot constraints.

VALID_SUCCESSORS: Dict[OpRole, FrozenSet[OpRole]] = {
    OpRole.PROJECT: frozenset(
        {
            OpRole.ACTIVATE,
            OpRole.NORMALIZE,
            OpRole.GATE,
            OpRole.RESIDUAL,
            OpRole.ROUTE,
            OpRole.REDUCE,
        }
    ),
    OpRole.NORMALIZE: frozenset(
        {
            OpRole.PROJECT,
            OpRole.MIX,
            OpRole.GATE,
            OpRole.ROUTE,
        }
    ),
    OpRole.ACTIVATE: frozenset(
        {
            OpRole.PROJECT,
            OpRole.NORMALIZE,
            OpRole.RESIDUAL,
            OpRole.GATE,
            OpRole.ROUTE,
        }
    ),
    OpRole.MIX: frozenset(
        {
            OpRole.PROJECT,
            OpRole.NORMALIZE,
            OpRole.ACTIVATE,
            OpRole.RESIDUAL,
            OpRole.GATE,
        }
    ),
    OpRole.ROUTE: frozenset(
        {
            OpRole.PROJECT,
            OpRole.MIX,
            OpRole.GATE,
            OpRole.NORMALIZE,
            OpRole.ACTIVATE,
            OpRole.ROUTE,
            OpRole.RESIDUAL,
        }
    ),
    OpRole.GATE: frozenset(
        {
            OpRole.PROJECT,
            OpRole.RESIDUAL,
            OpRole.NORMALIZE,
            OpRole.ACTIVATE,
            OpRole.ROUTE,
        }
    ),
    OpRole.POSITION: frozenset(
        {
            OpRole.PROJECT,
            OpRole.MIX,
            OpRole.NORMALIZE,
        }
    ),
    OpRole.REDUCE: frozenset(
        {
            OpRole.PROJECT,
            OpRole.ACTIVATE,
            OpRole.RESIDUAL,
        }
    ),
    OpRole.RESIDUAL: frozenset(
        {
            OpRole.PROJECT,
            OpRole.NORMALIZE,
            OpRole.ACTIVATE,
            OpRole.MIX,
            OpRole.GATE,
            OpRole.ROUTE,
            OpRole.REDUCE,
        }
    ),
    OpRole.UNSAFE: frozenset(),  # Fallback for truly unknown ops
}


# ── Convenience sets ────────────────────────────────────────────────

#: MoE / routing-with-experts ops. Total params != active params per token.
#: Used to exempt MoE models from param-count penalties in scoring.
MOE_OPS: FrozenSet[str] = frozenset(
    {
        "moe_topk",
        "moe_2expert",
        "relu_gated_moe",
        "sparse_bottleneck_moe",
        "hetero_moe",
        "arch_router",
        "compute_budget_router",
        "tropical_moe",
    }
)

#: Roles that contribute learnable parameters.
PARAM_ROLES: FrozenSet[OpRole] = frozenset(
    {
        OpRole.PROJECT,
        OpRole.GATE,
        OpRole.MIX,
        OpRole.NORMALIZE,
    }
)
