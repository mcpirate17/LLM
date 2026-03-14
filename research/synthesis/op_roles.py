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
  unsafe    — Gradient-killing or numerically unstable standalone.
              Never placed by grammar.
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
    "conv_only": OpRole.PROJECT,
    "basis_expansion": OpRole.PROJECT,
    "embedding_lookup": OpRole.PROJECT,

    # ── NORMALIZE: stabilize activations ────────────────────────────
    "rmsnorm": OpRole.NORMALIZE,
    "layernorm": OpRole.NORMALIZE,

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
    "multi_head_mix": OpRole.MIX,

    # ── ROUTE: information flow control ─────────────────────────────
    "split2": OpRole.ROUTE,
    "split3": OpRole.ROUTE,
    "concat": OpRole.ROUTE,
    "moe_topk": OpRole.ROUTE,
    "moe_2expert": OpRole.ROUTE,
    "mod_topk": OpRole.ROUTE,
    "early_exit": OpRole.ROUTE,
    "adaptive_recursion": OpRole.ROUTE,
    "token_merge": OpRole.ROUTE,
    "token_merging": OpRole.ROUTE,
    "cascade": OpRole.ROUTE,
    "speculative": OpRole.ROUTE,
    "route_topk": OpRole.ROUTE,
    "route_lanes": OpRole.ROUTE,
    "route_recursion": OpRole.ROUTE,
    "adaptive_lane_mixer": OpRole.ROUTE,
    "mixed_recursion_gate": OpRole.ROUTE,
    "routing_conditioned_compression": OpRole.ROUTE,
    "compression_mixture_experts": OpRole.ROUTE,
    "gather_topk": OpRole.ROUTE,

    # ── GATE: multiplicative modulation ─────────────────────────────
    "gated_linear": OpRole.GATE,
    "swiglu_mlp": OpRole.GATE,
    "learnable_scale": OpRole.GATE,
    "learnable_bias": OpRole.GATE,
    "topk_gate": OpRole.GATE,
    "relu_gate_routing": OpRole.GATE,
    "progressive_compression_gate": OpRole.GATE,
    "token_type_classifier": OpRole.GATE,
    "entropy_router": OpRole.GATE,

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

    # ── UNSAFE: gradient-killing or numerically unstable ────────────
    "sign_ste": OpRole.UNSAFE,
    "abs": OpRole.UNSAFE,
    "log": OpRole.UNSAFE,
    "exp": OpRole.UNSAFE,
    "reciprocal": OpRole.UNSAFE,
    "square": OpRole.UNSAFE,
    "neg": OpRole.UNSAFE,
    "sqrt": OpRole.UNSAFE,
    "div_safe": OpRole.UNSAFE,
    "cumprod_safe": OpRole.UNSAFE,
    "matmul": OpRole.UNSAFE,
    "outer_product": OpRole.UNSAFE,
    "cosine_similarity": OpRole.UNSAFE,
    "transpose_sd": OpRole.UNSAFE,
    "identity": OpRole.UNSAFE,

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
    "linear_algebra": OpRole.UNSAFE,
    "structural": OpRole.ROUTE,
    "parameterized": OpRole.PROJECT,
    "mixing": OpRole.MIX,
    "sequence": OpRole.MIX,
    "frequency": OpRole.UNSAFE,
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
        return _CATEGORY_ROLE_FALLBACK.get(prim.category.value, OpRole.UNSAFE)
    return OpRole.UNSAFE


def ops_by_role(role: OpRole) -> FrozenSet[str]:
    """Return all explicitly classified ops with the given role."""
    return frozenset(name for name, r in _OP_ROLE_MAP.items() if r == role)


# ── Role-valid transition rules ─────────────────────────────────────
# Maps each role to the set of roles that can legally follow it.
# Used by motif validation and template slot constraints.

VALID_SUCCESSORS: Dict[OpRole, FrozenSet[OpRole]] = {
    OpRole.PROJECT: frozenset({
        OpRole.ACTIVATE, OpRole.NORMALIZE, OpRole.GATE,
        OpRole.RESIDUAL, OpRole.ROUTE, OpRole.REDUCE,
    }),
    OpRole.NORMALIZE: frozenset({
        OpRole.PROJECT, OpRole.MIX, OpRole.GATE, OpRole.ROUTE,
    }),
    OpRole.ACTIVATE: frozenset({
        OpRole.PROJECT, OpRole.NORMALIZE, OpRole.RESIDUAL,
        OpRole.GATE, OpRole.ROUTE,
    }),
    OpRole.MIX: frozenset({
        OpRole.PROJECT, OpRole.NORMALIZE, OpRole.ACTIVATE,
        OpRole.RESIDUAL, OpRole.GATE,
    }),
    OpRole.ROUTE: frozenset({
        OpRole.PROJECT, OpRole.MIX, OpRole.GATE, OpRole.NORMALIZE,
        OpRole.ACTIVATE, OpRole.ROUTE, OpRole.RESIDUAL,
    }),
    OpRole.GATE: frozenset({
        OpRole.PROJECT, OpRole.RESIDUAL, OpRole.NORMALIZE,
        OpRole.ACTIVATE, OpRole.ROUTE,
    }),
    OpRole.POSITION: frozenset({
        OpRole.PROJECT, OpRole.MIX, OpRole.NORMALIZE,
    }),
    OpRole.REDUCE: frozenset({
        OpRole.PROJECT, OpRole.ACTIVATE, OpRole.RESIDUAL,
    }),
    OpRole.RESIDUAL: frozenset({
        OpRole.PROJECT, OpRole.NORMALIZE, OpRole.ACTIVATE,
        OpRole.MIX, OpRole.GATE, OpRole.ROUTE, OpRole.REDUCE,
    }),
    OpRole.UNSAFE: frozenset(),  # Never placed by grammar
}


# ── Convenience sets ────────────────────────────────────────────────

#: Ops that must never be sampled by the grammar.
GRAMMAR_EXCLUDED_ROLES: FrozenSet[OpRole] = frozenset({OpRole.UNSAFE})

#: Roles that contribute learnable parameters.
PARAM_ROLES: FrozenSet[OpRole] = frozenset({
    OpRole.PROJECT, OpRole.GATE, OpRole.MIX, OpRole.NORMALIZE,
})
