"""
Component Context Rules — Enforcement Layer

Encodes placement constraints for ops that have been audited as
context-sensitive. These rules prevent the grammar from generating
graphs where ops appear in invalid predecessor/successor chains,
and classify ops by search-mode so niche/restricted ops are not
sprayed into default search blindly.

Sources:
  - artifacts/component_context_rules.md
  - artifacts/component_context_rules.json
  - artifacts/low_s1_root_cause_audit.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from typing import Dict, FrozenSet, Iterable, List, Optional

from .graph import ComputationGraph
from .motifs import MOTIFS_BY_CLASS, Motif
from .primitives import PRIMITIVE_REGISTRY


# ── Search-mode classification ────────────────────────────────────


class SearchMode(Enum):
    __slots__ = ()
    GENERAL = "general"


# ── Per-op context rule ───────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class ContextRule:
    """Machine-actionable placement constraint for a single op."""

    search_mode: SearchMode
    # Concrete op names that are forbidden as direct predecessors.
    forbidden_predecessors: FrozenSet[str] = field(default_factory=frozenset)
    # Concrete op names that are forbidden as direct successors.
    forbidden_successors: FrozenSet[str] = field(default_factory=frozenset)
    # If True, the op must sit inside a residual bypass (add consuming same input).
    requires_residual_context: bool = False


CONTEXT_CLASS_GENERAL = "general-use"
CONTEXT_CLASS_RESTRICTED = "restricted-use"
CONTEXT_CLASS_STRUCTURAL = "structural"
CONTEXT_CLASS_REHAB = "rehab"


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

# ── Context Rules Registry ────────────────────────────────────────
# Only ops with non-trivial constraints are listed.
# Ops not listed here are treated as general-use with no extra constraints.

CONTEXT_RULES: Dict[str, ContextRule] = {
    # ── Attention/position ops: promoted to GENERAL with dedicated templates ──
    "local_window_attn": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    # sliding_window_mask: moved to routing section below
    "causal_mask": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    # ── Restricted-use: structural ops ──────────────────────────────
    "split2": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        # split2 halves dim — ops with internal projections built for full
        # model_dim get RuntimeError shape mismatches (100% fail in data).
        forbidden_successors=frozenset({"output_head"}) | _FULL_DIM_OPS,
    ),
    "split3": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
    ),
    # concat: moved to extended rules section with failure-derived constraints
    # identity: moved to extended rules below
    # ── Restricted-use: dimension-changing ops ───────────────────────
    "linear_proj_down": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        # Down-projection collapses dim — feeding into ops that expect
        # full model_dim causes RuntimeError or capacity starvation
        # (85-100% fail rate across swiglu_mlp, relu_gate_routing, etc.).
        forbidden_successors=_FULL_DIM_CONSUMERS
        | frozenset(
            {
                "identity",  # 100% fail — dead passthrough after dim collapse
                "linear_proj_up",  # up after down with no activation = broken bottleneck
                "concat",  # dim mismatch on concat with full-width sibling
                "linear_proj",  # 100% fail (12/12)
                "adaptive_recursion",  # 100% fail (7/7) — needs full dim
            }
        ),
    ),
    # ── Restricted-use: reduce ops (must not chain or feed output directly) ─
    "norm_last": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "max_last": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "sum_last": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "mean_last": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    # ── General-use ops with forbidden predecessor constraints ──────
    "graph_attention": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    # softmax_attention, linear_attention: moved to extended rules below
    "state_space": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "diff_attention": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "gated_delta": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "fused_linear_gelu": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "rwkv_time_mixing": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "integral_kernel": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "fixed_point_iter": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    # ── Promoted from NICHE to GENERAL: have dedicated templates + MATH_SPACE_RULES ──
    # tropical_center: moved to math-space rules section below
    "tropical_matmul": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "lif_neuron": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "sparse_threshold": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
    ),
    "stdp_attention": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
    ),
    "sparse_bottleneck_moe": ContextRule(  # was: n_way_sparse_router
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "confidence_token_gate": ContextRule(  # was: early_exit
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS
        | frozenset(
            {
                "add",  # 100% fail (37/37) — double residual, conflicting exits
            }
        ),
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
        requires_residual_context=True,
    ),
    "learned_token_gate": ContextRule(  # was: cascade
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS,
    ),
    "hyp_linear": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
    ),
    "hyp_tangent_nonlinear": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
    ),
    # ── Gradient-sensitive ops: require bounded input, must feed projection ──
    "div_safe": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
        requires_residual_context=True,  # division output range depends on inputs; residual stabilizes
    ),
    "log": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "reciprocal": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    # ── Elementwise ops: require normalized input, must feed projection ──
    "minimum": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "sub": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "exp": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head", "mul", "linear_proj"})
        | _STRUCTURAL_SPLIT_OPS,
    ),
    "sign_ste": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    # ── Routing ops: no double-routing, no feeding masks ────────────
    # Failure data: adaptive_recursion->progressive_compression_gate,
    # adaptive_lane_mixer->progressive_compression_gate, route_topk->add,
    # moe_topk->rmsnorm all 100% fail. Routing ops must not chain.
    "depth_weighted_proj": ContextRule(  # was: adaptive_recursion
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS | frozenset({"linear_proj_down"}),
        forbidden_successors=_ROUTING_OPS
        | _MASK_OPS
        | _MATH_SPACE_OPS
        | _STRUCTURAL_SPLIT_OPS
        | _MIXING_OPS  # 100% fail: recursion strips residual/mask context
        | frozenset(
            {
                "linear_proj_up",  # 100% fail — redundant projection after recursion
            }
        ),
    ),
    "difficulty_blend_3way": ContextRule(  # was: adaptive_lane_mixer
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_ROUTING_OPS
        | _MASK_OPS
        | frozenset(
            {
                "learnable_bias",  # redundant after routing
            }
        ),
    ),
    "feature_sparsity": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_GATING_OPS
        | frozenset(
            {
                "add",
                "mul",  # 100% fail: gated output fed to raw arithmetic
                "linear_proj_up",  # 100% fail
            }
        ),
    ),
    "gated_lane_blend": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_GATING_OPS | _MASK_OPS,
    ),
    "depth_gated_transform": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_GATING_OPS | _STRUCTURAL_SPLIT_OPS,
    ),
    "signal_conditioned_compression": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        # Allow moe_topk/moe_2expert after compression — data-mined high-signal combo
        forbidden_successors=(_GATING_OPS - frozenset({"moe_topk", "moe_2expert"}))
        | _STRUCTURAL_SPLIT_OPS,
    ),
    # ── Mask ops: must feed mixing ops only ───────────────────────
    "sliding_window_mask": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_ROUTING_OPS | _REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | (_ROUTING_OPS - _MIXING_OPS),
        requires_residual_context=True,
    ),
    # ── Sparse linear ops: break with routing and MoE ────────────
    "nm_sparse_linear": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=_ROUTING_OPS
        | frozenset(
            {
                "moe_topk",
                "moe_2expert",  # 100% fail: sparse->MoE
                "tanh",  # 100% fail
            }
        ),
    ),
    "semi_structured_2_4_linear": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=_ROUTING_OPS,
    ),
    "block_sparse_linear": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset(
            {
                "swiglu_mlp",  # 100% fail
            }
        ),
    ),
    # ── Down-projection: extend existing rule with more forbidden successors ─
    # linear_proj_down->adaptive_recursion: 100% fail (7/7)
    # linear_proj_down->add: 100% fail (11/11)
    # linear_proj_down->linear_proj: 100% fail (12/12)
    # (base rule already exists — updating handled below)
    # ── Attention ops: must not feed raw linear_proj ──────────────
    # softmax_attention->linear_proj: 100% fail (13/13)
    # linear_attention->linear_proj: 100% fail (11/11)
    "softmax_attention": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(
            {
                "causal_mask",  # 100% fail (58/58) — mask tensor fed as data input
                "split2",
                "split3",  # halved dim breaks multi-head structure
                "linear_proj_down",  # reduced dim breaks head dimension
                "token_merge",  # destroyed token order breaks attention
                "transpose_sd",  # wrong axis orientation
            }
        )
        | _REDUCE_OPS,
        forbidden_successors=frozenset(
            {
                "output_head",
                "linear_proj",  # 100% fail (13/13) — raw attention output needs norm first
                "identity",  # strips causal context
                "softmax_attention",  # stacking raw attention 100% fail
            }
        ),
        requires_residual_context=True,
    ),
    "linear_attention": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    # ── High-value ops with strict placement requirements ────────
    # selective_scan: 54% S0 fail (causality), 90% S1 fail when S0 passes.
    # Every success with loss < 0.05 has norm → conv1d_seq → silu → selective_scan → add.
    "selective_scan": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS
        | _STRUCTURAL_SPLIT_OPS
        | frozenset(
            {
                "identity",  # strips causal context
                "token_merge",  # destroys token ordering SSM depends on
                "transpose_sd",  # transposes (S,D) axes, breaks SSM state shape
            }
        ),
        forbidden_successors=frozenset(
            {
                "output_head",
                "selective_scan",  # SSM→SSM chaining 96% fail
                "state_space",  # same failure mode
            }
        ),
        requires_residual_context=True,  # SSM output needs residual path
    ),
    # transpose_sd: 95% S1 fail in random placement. Works in cross_dim_mixer (9%)
    # / dual_axis_block (5%) where templates manage the transpose lifecycle.
    # Keep forbidden_successors narrow — blanket bans on _CAUSAL_SENSITIVE_OPS /
    # _FULL_DIM_OPS block template-internal edges (4932/4940 grammar rejections).
    "transpose_sd": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS
        | frozenset(
            {
                "transpose_sd",  # 100% fail (10/10) — double transpose = noop or wrong
            }
        ),
        forbidden_successors=frozenset(
            {
                "output_head",
                "split2",
                "split3",  # splitting transposed tensor
            }
        ),
        requires_residual_context=True,  # transposed output MUST rejoin through residual
    ),
    # ── MLP ops: must not feed down-projection or sparse ─────────
    "swiglu_mlp": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS | _SPARSE_LINEAR_OPS,
        forbidden_successors=frozenset(
            {
                "linear_proj_down",  # 100% fail (15/15)
                "nm_sparse_linear",  # 100% fail (6/6)
            }
        ),
    ),
    # ── Norm ops: must not feed identity or raw add ──────────────
    "rmsnorm": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(
            {
                "progressive_compression_gate",  # 100% fail (17/17)
                "moe_topk",  # 100% fail (40/40)
                "chebyshev_spectral_mix",  # 100% fail (5/5)
            }
        ),
        forbidden_successors=frozenset(
            {
                "identity",  # 100% fail (17/17) — strips causal mask
                # rwkv_channel removed: data mining (n=11,447) shows
                # rmsnorm → rwkv_channel → add at loss=0.049 when in
                # proper residual scaffold. Original 17/17 failures lacked
                # the residual add; templates now enforce it.
            }
        ),
    ),
    # layernorm->add has 30% fail rate (122F/289S) — not a hard ban.
    # Failures are from direct layernorm->add with no op in between (empty residual).
    # Most successes have layernorm->op->add with add as residual connection.
    # ── Identity: passthrough. Causality violations correlate with identity
    # but the root cause is graphs lacking causal attention — fixed by
    # grammar-level causal_mask injection before sequence-mixing ops.
    "identity": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(
            {
                "rmsnorm",  # 100% fail (8/8)
            }
        ),
        forbidden_successors=frozenset({"output_head"})
        | _CAUSAL_SENSITIVE_OPS
        | frozenset({"selective_scan"}),  # 100% fail (5/5)
    ),
    # ── Projection ops: chain prevention ─────────────────────────
    "linear_proj_up": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset({"cos"}),  # 100% fail (17/17)
        forbidden_successors=frozenset(
            {
                "mul",  # 100% fail (6/6)
                "linear_proj_up",  # 100% fail (5/5) — double up-project
                "linear_proj",  # 100% fail (5/5)
            }
        ),
    ),
    # ── Math-space ops: prevent cross-space chaining ─────────────
    "padic_expand": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset(
            {
                "ultrametric_attention",  # 100% fail (5/5) — padic->ultrametric
            }
        )
        | (_MATH_SPACE_OPS - frozenset({"padic_expand"})),
    ),
    "tropical_center": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset(
            {
                "linear_proj",  # 100% fail (5/5)
            }
        )
        | (_MATH_SPACE_OPS - frozenset({"tropical_center", "tropical_matmul"})),
    ),
    # geometric_product: has dedicated template (geometric_product_block).
    # Data: standalone 100% fail (11/11), doubled 58% fail — needs residual + companion.
    "geometric_product": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS | frozenset({"output_head"}),
        requires_residual_context=True,  # multivector output unbounded without skip
    ),
    "adaptive_rank_gate": ContextRule(  # was: progressive_compression_gate
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS | _SPARSE_LINEAR_OPS,
        forbidden_successors=_STRUCTURAL_SPLIT_OPS
        | frozenset(
            {
                "rmsnorm",  # 100% fail (17/17)
            }
        ),
    ),
    # ── Token merge: destroys token ordering ──────────────────────
    # Data (2026-03-29): 89% S0 failure from causality gate when downstream
    # ops assume causal token ordering. Successes: token_merge → conv1d_seq
    # (loss 0.006), token_merge → swiglu_mlp (loss 0.006). Requires residual.
    "token_merge": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS | _STRUCTURAL_SPLIT_OPS,
        forbidden_successors=_CAUSAL_SENSITIVE_OPS
        | frozenset(
            {
                "output_head",
                "softmax_attention",  # needs original token order
                "linear_attention",
                "selective_scan",  # SSM needs causal token ordering
                "state_space",
                "identity",  # strips context
            }
        ),
        requires_residual_context=True,
    ),
    # ── Routing ops: require linear_proj predecessor ──────────────
    # Data (2026-03-29): These routing ops only succeed when preceded by
    # a projection. Direct norm→router: 0-5% S1. proj→router: 10-70%.
    # The router needs a transformed representation, not raw normed activations.
    "arch_router": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(
            {"rmsnorm", "layernorm"}  # 2% consec S1 (63 samples) — needs proj between
        )
        | _REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    "compute_budget_router": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(
            {"rmsnorm", "layernorm"}  # 0% consec S1 (15 samples) — needs proj between
        )
        | _REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    "hetero_moe": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(
            {"rmsnorm", "layernorm"}  # 0% consec S1 (13 samples) — needs proj between
        )
        | _REDUCE_OPS,
        forbidden_successors=frozenset(
            {"output_head", "linear_proj"}
        ),  # 0% (15 samples)
        requires_residual_context=True,
    ),
    # moe_topk: norm→moe_topk 5% consec, only works after linear_proj (data: 2/2 S1)
    "moe_topk": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(
            {"rmsnorm", "layernorm"}  # 5% consec S1 (20 samples) — needs proj between
        )
        | _REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    # moe_2expert: norm→moe_2expert 4% consec vs 19% co-occur
    # Works after gelu (0.161 loss), conv1d_seq (0.008 loss), silu (0.436 loss)
    "moe_2expert": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(
            {
                "rmsnorm",
                "layernorm",
            }  # 4% consec S1 (28 samples) — needs activation/proj between
        )
        | _REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    # ── From user-reported 5% penalty pairs (all 100% fail in program_results) ──
    "spectral_filter": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(
            {
                "add",  # 100% fail (56/56) — residual sum into FFT = garbage spectrum
            }
        ),
        forbidden_successors=frozenset({"output_head"}),
    ),
    "bottleneck_proj": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset(
            {
                "cos",  # 100% fail (10/10) — reduced dim into cos = numerically unstable
            }
        ),
    ),
    "concat": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(
            {
                "concat",  # 100% fail (14/14) — double concat = 4x dim explosion
                "cos",  # 100% fail (10/10) — cos ∈ [-1,1] width != sibling width
            }
        ),
        forbidden_successors=frozenset({"output_head"}),
    ),
    "conv1d_seq": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset(
            {
                "neg",  # 100% fail (11/11) — negation destroys conv features
            }
        ),
    ),
    "cos": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset(
            {
                "concat",  # 100% fail (10/10) — bounded output width mismatch
                "linear_proj_down",  # 100% fail (16/16) — bounded input starves projection
                "linear_proj_up",  # 100% fail (17/17)
            }
        )
        | _STRUCTURAL_SPLIT_OPS,
    ),
    "gated_linear": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset(
            {
                "mul",  # 100% fail (10/10) — double gating
                "ternary_projection",  # 100% fail (27/27) — gated output into {-1,0,1} kills gradients
            }
        ),
    ),
    # ── Comprehensive coverage: remaining standalone ops ─────────
    #
    # Elementwise unary — safe activations need no rules, but risky
    # ones (sqrt, abs) and gradient-killers need successors constrained.
    "abs": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "neg": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "sqrt": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "square": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "relu": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
    ),
    "sigmoid": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
    ),
    "tanh": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
    ),
    "sin": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "gelu": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
    ),
    "silu": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
    ),
    # Elementwise binary — add/mul are fundamental, need minimal rules.
    "add": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
    ),
    "mul": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
    ),
    "maximum": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    # Linear algebra — binary ops need bounded input and stabilized output.
    # Data: matmul 36% S1, outer_product 32%, cosine_similarity 23%.
    # Successes cluster around norm→proj→op→proj→residual patterns.
    "matmul": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS
        | frozenset(
            {"rmsnorm", "layernorm"}
        ),  # 68% fail without proj between norm and matmul
        forbidden_successors=frozenset({"output_head", "gather_topk"})
        | _STRUCTURAL_SPLIT_OPS,
        requires_residual_context=True,  # unbounded output needs skip connection
    ),
    "outer_product": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head", "gather_topk"})
        | _STRUCTURAL_SPLIT_OPS,
        requires_residual_context=True,  # unbounded output needs skip connection
    ),
    "cosine_similarity": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS
        | frozenset(
            {"rmsnorm", "layernorm"}
        ),  # 95% fail (148 samples) — needs proj first
        forbidden_successors=frozenset(
            {"output_head", "gather_topk"}
        )  # 95% fail (305 samples)
        | _STRUCTURAL_SPLIT_OPS,
        requires_residual_context=True,  # [-1,1] output needs residual to preserve signal
    ),
    # Reductions — cumsum/cumprod need norm after; cumprod is risky.
    "cumsum": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "cumprod_safe": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    # Sequence — softmax_last produces probabilities, must feed projection.
    "softmax_last": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    # Structural — multi_head_mix, gather_topk.
    "multi_head_mix": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
    ),
    "gather_topk": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    # Parameterized — projections, norms, routing gaps.
    "linear_proj": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
    ),
    "layernorm": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset(
            {
                "identity",  # strips causal context, same as rmsnorm
                # rwkv_channel removed: data mining (n=11,447) shows
                # layernorm → rwkv_channel → add → layernorm at loss=0.054.
                # The winning RWKV pattern requires norm before the op.
            }
        ),
    ),
    "learnable_scale": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
    ),
    "learnable_bias": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
    ),
    "grouped_linear": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "kronecker_linear": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "embedding_lookup": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(),
        forbidden_successors=frozenset({"output_head"}),
    ),
    "latent_attention_compressor": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "conv_only": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    # Gating gaps — relu_gated_moe, speculative, topk_gate, mod_topk, token_merge.
    "relu_gated_moe": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS | frozenset({"linear_proj_down"}),
        forbidden_successors=_GATING_OPS | frozenset({"output_head"}),
    ),
    "cheap_verify_blend": ContextRule(  # was: speculative
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_ROUTING_OPS | frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    "topk_gate": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_ROUTING_OPS | frozenset({"output_head"}),
    ),
    "depth_token_mask": ContextRule(  # was: mod_topk
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS | _SPARSE_LINEAR_OPS,
        forbidden_successors=_ROUTING_OPS | frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    "adjacent_token_merge": ContextRule(  # was: token_merge
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_ROUTING_OPS | frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    # 2-input routing/compression ops — template-confined but need rules
    # in case grammar places them.
    "dual_compression_blend": ContextRule(  # was: compression_mixture_experts
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        # Allow moe_topk/moe_2expert after compression — data-mined high-signal combo
        forbidden_successors=(_ROUTING_OPS - frozenset({"moe_topk", "moe_2expert"}))
        | frozenset({"output_head"}),
    ),
    "score_depth_blend": ContextRule(  # was: mixed_recursion_gate
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_ROUTING_OPS | frozenset({"output_head"}),
    ),
    # Math-space gaps — ops in math_space category but not in _MATH_SPACE_OPS
    # or without ContextRule.
    "exp_map": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "log_map": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "grade_mix": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "hyp_distance": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "hyperbolic_norm": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "padic_gate": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"})
        | (_MATH_SPACE_OPS - frozenset({"padic_expand", "padic_gate"})),
    ),
    "padic_residual": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"})
        | (_MATH_SPACE_OPS - frozenset({"padic_expand", "padic_residual"})),
    ),
    "poincare_add": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "spike_rate_code": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "tropical_add": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"})
        | (
            _MATH_SPACE_OPS
            - frozenset({"tropical_add", "tropical_matmul", "tropical_center"})
        ),
    ),
    "tropical_attention": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    "tropical_moe": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS | _SPARSE_LINEAR_OPS,
        forbidden_successors=_ROUTING_OPS | frozenset({"output_head"}),
    ),
}

_OP_CONTEXT_CLASS: Dict[str, str] = {
    # Structural ops: scaffolding, not standalone learners
    "causal_mask": CONTEXT_CLASS_STRUCTURAL,
    "split2": CONTEXT_CLASS_STRUCTURAL,
    "split3": CONTEXT_CLASS_STRUCTURAL,
    "concat": CONTEXT_CLASS_STRUCTURAL,
    "identity": CONTEXT_CLASS_STRUCTURAL,
    # geometric_product: promoted to GENERAL — context rules enforce valid placement
}

_MOTIF_TEMPLATE_ALLOWLIST: Dict[str, FrozenSet[str]] = {
    "attn_causal_mask": frozenset(
        {
            "residual_block",
            "transformer_block",
            "hybrid_parallel",
            "gated_residual",
            "difficulty_routed_block",
            "three_lane_adaptive",
            # Attention templates that use attention motifs
            "attn_residual_block",
            "attn_gated_residual",
            "attn_three_way_split",
            "attn_dense_cascade",
            "attn_conditional_compute",
            "attn_dual_axis",
            "attn_cross_dim",
            "attn_multi_head_mix",
            "attn_ssm_hybrid",
            "attn_conv_hybrid",
            "attn_rwkv_hybrid",
            "attn_moe_block",
            "attn_bottleneck_hybrid",
            "attn_routing_block",
            "dual_attn_block",
            "attn_state_space_hybrid",
            "cascaded_attn_ffn",
        }
    ),
    "attn_local_window": frozenset(
        {
            "residual_block",
            "transformer_block",
            "hybrid_parallel",
            "gated_residual",
            "difficulty_routed_block",
            "three_lane_adaptive",
            "local_attention_block",
            # Attention templates
            "attn_residual_block",
            "attn_gated_residual",
            "attn_three_way_split",
            "attn_dense_cascade",
            "attn_conditional_compute",
            "attn_dual_axis",
            "attn_cross_dim",
            "attn_multi_head_mix",
            "attn_ssm_hybrid",
            "attn_conv_hybrid",
            "attn_rwkv_hybrid",
            "attn_moe_block",
            "attn_bottleneck_hybrid",
            "attn_routing_block",
            "dual_attn_block",
            "attn_state_space_hybrid",
            "cascaded_attn_ffn",
            "local_attn_ffn_block",
            "local_attn_swiglu",
            "local_attn_routing",
            "local_attn_moe",
            "local_attn_ssm_hybrid",
        }
    ),
    "attn_sliding_window": frozenset(
        {
            "residual_block",
            "transformer_block",
            "hybrid_parallel",
            "gated_residual",
            "difficulty_routed_block",
            "three_lane_adaptive",
            "windowed_attention",
            # Attention templates
            "attn_residual_block",
            "attn_gated_residual",
            "attn_three_way_split",
            "attn_dense_cascade",
            "attn_conditional_compute",
            "attn_dual_axis",
            "attn_cross_dim",
            "attn_multi_head_mix",
            "attn_ssm_hybrid",
            "attn_conv_hybrid",
            "attn_rwkv_hybrid",
            "attn_moe_block",
            "attn_bottleneck_hybrid",
            "attn_routing_block",
            "dual_attn_block",
            "attn_state_space_hybrid",
            "cascaded_attn_ffn",
        }
    ),
    "attn_graph": frozenset(
        {
            "residual_block",
            "transformer_block",
            "hybrid_parallel",
            "gated_residual",
            "graph_attention_block",
            # Attention templates
            "attn_residual_block",
            "attn_gated_residual",
            "attn_three_way_split",
            "attn_dense_cascade",
            "attn_conditional_compute",
            "attn_dual_axis",
            "attn_cross_dim",
            "attn_multi_head_mix",
            "attn_ssm_hybrid",
            "attn_conv_hybrid",
            "attn_rwkv_hybrid",
            "attn_moe_block",
            "attn_bottleneck_hybrid",
            "attn_routing_block",
            "dual_attn_block",
            "attn_state_space_hybrid",
            "cascaded_attn_ffn",
            "graph_attn_ffn_block",
            "graph_attn_moe",
            "graph_attn_sparse_ffn",
        }
    ),
    "ssm_state_space": frozenset(
        {
            "residual_block",
            "transformer_block",
            "gated_residual",
            "state_space_block",
        }
    ),
    "mix_integral_kernel": frozenset(
        {
            "residual_block",
            "gated_residual",
            "integral_kernel_block",
        }
    ),
    "ffn_fused_gelu": frozenset(
        {
            "residual_block",
            "transformer_block",
            "gated_residual",
            "fused_gelu_ffn",
        }
    ),
    "reduce_sum": frozenset({"parallel_split", "three_way_split", "reduce_attend"}),
    "reduce_mean": frozenset({"parallel_split", "three_way_split", "reduce_attend"}),
    "reduce_max": frozenset({"parallel_split", "three_way_split", "reduce_attend"}),
    "reduce_norm": frozenset({"parallel_split", "three_way_split", "reduce_attend"}),
    "route_identity": frozenset(
        {
            "difficulty_routed_block",
            "three_lane_adaptive",
            "cascaded_early_exit",
            "recursive_depth_router",
            "conditional_compute",
        }
    ),
    "route_early_exit": frozenset(
        {
            "difficulty_routed_block",
            "three_lane_adaptive",
            "cascaded_early_exit",
            "recursive_depth_router",
            "conditional_compute",
        }
    ),
    "spiking_lif_rate": frozenset(
        {
            "residual_block",
            "transformer_block",
            "gated_residual",
            "spiking_residual_block",
            "spiking_moe_block",
        }
    ),
    "spiking_threshold_stdp": frozenset(
        {
            "residual_block",
            "transformer_block",
            "gated_residual",
            "spiking_residual_block",
            "spiking_moe_block",
        }
    ),
    "spiking_tropical_gate": frozenset(
        {"spiking_moe_block", "residual_block", "gated_residual"}
    ),
    "spiking_rate_tropical_gate": frozenset(
        {"spiking_moe_block", "residual_block", "gated_residual"}
    ),
    "spiking_threshold_tropical_gate": frozenset(
        {"spiking_moe_block", "residual_block", "gated_residual"}
    ),
    "tropical_center_norm": frozenset(
        {
            "residual_block",
            "transformer_block",
            "gated_residual",
            "tropical_center_block",
        }
    ),
    "hyperbolic_residual_bridge": frozenset(
        {"residual_block", "hyperbolic_bridge_block"}
    ),
    "poincare_add_bridge": frozenset({"residual_block", "hyperbolic_bridge_block"}),
    "conv_only_block": frozenset(
        {"residual_block", "gated_residual", "conv_residual_block"}
    ),
    "attn_gated_delta": frozenset(
        {
            "residual_block",
            "transformer_block",
            "recurrent_delta_block",
            "attn_residual_block",
            "attn_gated_residual",
            "attn_three_way_split",
            "attn_dense_cascade",
            "dual_attn_block",
            "cascaded_attn_ffn",
        }
    ),
    "mix_fixed_point": frozenset({"residual_block", "iterative_refinement"}),
    "n_way_routing": frozenset({"moe", "n_way_moe_block"}),
    "proj_bottleneck": frozenset({"residual_block", "bottleneck", "gated_residual"}),
    "proj_low_rank": frozenset({"residual_block", "bottleneck", "gated_residual"}),
    "route_adaptive_recursion": frozenset(
        {"recursive_depth_router", "difficulty_routed_block", "conditional_compute"}
    ),
}

_CONTEXT_CLASS_PRIORS = {
    CONTEXT_CLASS_GENERAL: 1.0,
    CONTEXT_CLASS_RESTRICTED: 0.55,
    CONTEXT_CLASS_STRUCTURAL: 0.30,
    CONTEXT_CLASS_REHAB: 0.15,
}

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


# ── Derived sets (frozen at import time) ──────────────────────────

STRUCTURAL_OPS: FrozenSet[str] = frozenset(
    {
        "identity",
        "split2",
        "split3",
        "concat",
    }
)

# Ops exempt from per-op S1 attribution: scaffolding (splits, concat,
# identity), positional masks (causal_mask, sliding_window_mask),
# dimension-reduction ops (norm_last, sum_last, mean_last, max_last),
# and parameter-free elementwise/sequence transforms (minimum, maximum,
# sub, cumprod_safe, cumsum). None have learnable parameters — they
# should not be judged as standalone learning carriers.
S1_EXEMPT_OPS: FrozenSet[str] = frozenset(
    {
        "identity",
        "split2",
        "split3",
        "concat",
        "causal_mask",
        "sliding_window_mask",
        "norm_last",
        "sum_last",
        "mean_last",
        "max_last",
        # Parameter-free elementwise transforms
        "minimum",
        "maximum",
        "sub",
        # Parameter-free sequence transforms
        "cumprod_safe",
        "cumsum",
    }
)

# Ops that require residual bypass context per audit.
REQUIRES_RESIDUAL_CONTEXT: FrozenSet[str] = frozenset(
    name for name, rule in CONTEXT_RULES.items() if rule.requires_residual_context
)


# ── Query helpers ─────────────────────────────────────────────────


def is_structural(op_name: str) -> bool:
    return op_name in STRUCTURAL_OPS


def get_op_context_class(op_name: str) -> str:
    return _OP_CONTEXT_CLASS.get(op_name, CONTEXT_CLASS_GENERAL)


def _motif_context_class(motif: Motif) -> str:
    classes = {get_op_context_class(step.op_name) for step in motif.steps}
    if CONTEXT_CLASS_REHAB in classes:
        return CONTEXT_CLASS_REHAB
    if CONTEXT_CLASS_STRUCTURAL in classes:
        return CONTEXT_CLASS_STRUCTURAL
    if CONTEXT_CLASS_RESTRICTED in classes:
        return CONTEXT_CLASS_RESTRICTED
    return CONTEXT_CLASS_GENERAL


def apply_context_rule_priors(
    motif_weights: Optional[Dict[str, float]],
    exploration_targets: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, float]]:
    weights = dict(motif_weights) if motif_weights else {}
    targeted_ops = set(exploration_targets or ())
    seen_motifs: set[str] = set()
    for motifs in MOTIFS_BY_CLASS.values():
        for motif in motifs:
            if motif.name in seen_motifs:
                continue
            seen_motifs.add(motif.name)
            motif_ops = {step.op_name for step in motif.steps}
            if motif_ops & targeted_ops:
                continue
            factor = _CONTEXT_CLASS_PRIORS[_motif_context_class(motif)]
            if motif.name == "attn_local_window":
                factor *= 0.7
            base = weights.get(motif.name, motif.lift)
            weights[motif.name] = max(0.05, base * factor)
    return weights or None


def motif_allowed_in_template(motif: Motif, template_name: Optional[str]) -> bool:
    if template_name is None:
        return True
    allowed_templates = _MOTIF_TEMPLATE_ALLOWLIST.get(motif.name)
    if allowed_templates is None:
        return True
    return template_name in allowed_templates


def _child_map(graph: ComputationGraph) -> Dict[int, List[int]]:
    children = {nid: [] for nid in graph.nodes}
    for nid, node in graph.nodes.items():
        for inp_id in node.input_ids:
            if inp_id in children:
                children[inp_id].append(nid)
    return children


def _has_descendant_op(
    graph: ComputationGraph,
    start_id: int,
    allowed_ops: Iterable[str],
    children: Optional[Dict[int, List[int]]] = None,
) -> bool:
    children = children or _child_map(graph)
    q = deque(children.get(start_id, ()))
    seen: set = set()
    while q:
        nid = q.popleft()
        if nid in seen:
            continue
        seen.add(nid)
        node = graph.nodes.get(nid)
        if node is None:
            continue
        if node.op_name in allowed_ops:
            return True
        q.extend(children.get(nid, ()))
    return False


def _has_ancestor_op(
    graph: ComputationGraph,
    start_id: int,
    allowed_ops: Iterable[str],
) -> bool:
    start_node = graph.nodes.get(start_id)
    if start_node is None:
        return False
    q = deque(start_node.input_ids)
    seen: set = set()
    while q:
        nid = q.popleft()
        if nid in seen:
            continue
        seen.add(nid)
        node = graph.nodes.get(nid)
        if node is None:
            continue
        if node.op_name in allowed_ops:
            return True
        q.extend(node.input_ids)
    return False


def _has_immediate_predecessor_op(
    graph: ComputationGraph,
    node_id: int,
    allowed_ops: Iterable[str],
) -> bool:
    """Check if any direct parent of node_id has op_name in allowed_ops."""
    node = graph.nodes.get(node_id)
    if node is None:
        return False
    for pid in node.input_ids:
        parent = graph.nodes.get(pid)
        if parent is not None and parent.op_name in allowed_ops:
            return True
    return False


def _has_immediate_successor_op(
    children: Dict[int, List[int]],
    graph: ComputationGraph,
    node_id: int,
    allowed_ops: Iterable[str],
) -> bool:
    """Check if any direct child of node_id has op_name in allowed_ops."""
    for cid in children.get(node_id, ()):
        child = graph.nodes.get(cid)
        if child is not None and child.op_name in allowed_ops:
            return True
    return False


def find_graph_context_violations(graph: ComputationGraph) -> List[str]:
    violations: List[str] = []
    # Build child (successor) map once — used for both successor checks and
    # descendant traversals. Previously built twice (successors + _child_map).
    children: Dict[int, List[int]] = _child_map(graph)
    non_input_nodes = [node for node in graph.nodes.values() if not node.is_input]

    for nid in graph.topological_order():
        node = graph.nodes[nid]
        if node.is_input:
            continue

        rule = CONTEXT_RULES.get(node.op_name)
        if rule is not None:
            if rule.forbidden_predecessors:
                for pid in node.input_ids:
                    parent = graph.nodes.get(pid)
                    if (
                        parent is not None
                        and parent.op_name in rule.forbidden_predecessors
                    ):
                        violations.append(
                            f"Context rule: {parent.op_name} (id={pid}) -> {node.op_name} (id={nid}) is forbidden"
                        )
            if rule.forbidden_successors:
                for sid in children.get(nid, ()):
                    succ = graph.nodes.get(sid)
                    if succ is not None and succ.op_name in rule.forbidden_successors:
                        violations.append(
                            f"Context rule: {node.op_name} (id={nid}) -> {succ.op_name} (id={sid}) is forbidden"
                        )

        child_ops = {
            graph.nodes[cid].op_name
            for cid in children.get(nid, ())
            if cid in graph.nodes
        }
        parent_ops = {
            graph.nodes[parent_id].op_name
            for parent_id in node.input_ids
            if parent_id in graph.nodes and not graph.nodes[parent_id].is_input
        }
        if node.op_name == "local_window_attn":
            if not parent_ops & _LOCAL_WINDOW_VALID_PREDS:
                violations.append(
                    "local_window_attn requires rmsnorm/layernorm predecessor"
                )
            if "linear_proj" not in child_ops:
                violations.append(
                    "local_window_attn requires immediate linear_proj successor"
                )
            if not _has_descendant_op(graph, nid, {"add"}, children):
                violations.append(
                    "local_window_attn must remain inside a residual attention block"
                )
        elif node.op_name in {"causal_mask", "sliding_window_mask"}:
            if not child_ops & _MASK_VALID_SUCCESSORS:
                violations.append(
                    f"{node.op_name} must feed attention/projection, not stand alone"
                )
        elif node.op_name in {"sum_last", "mean_last", "max_last", "norm_last"}:
            if not child_ops & _REDUCTION_RESTORE_OPS:
                violations.append(
                    f"{node.op_name} must rejoin through projection/merge, not stand alone"
                )
        elif node.op_name == "identity":
            if (
                graph.output_node is not None
                and graph.output_node.id == nid
                and len(non_input_nodes) <= 2
            ):
                violations.append("identity cannot be the primary learning carrier")
        elif node.op_name in ("split2", "split3"):
            if not _has_descendant_op(graph, nid, {"concat", "add"}, children):
                violations.append(
                    f"{node.op_name} must rejoin through concat or add before output"
                )
        elif node.op_name == "lif_neuron":
            if not _has_descendant_op(
                graph,
                nid,
                {"spike_rate_code", "stdp_attention", "tropical_gate"},
                children,
            ):
                violations.append("lif_neuron requires spiking successor context")
        elif node.op_name == "sparse_threshold":
            if not (child_ops & {"stdp_attention", "tropical_gate"}):
                violations.append(
                    "sparse_threshold requires stdp_attention or tropical_gate successor"
                )
        elif node.op_name == "stdp_attention":
            if not (parent_ops & {"sparse_threshold", "spike_rate_code", "lif_neuron"}):
                violations.append("stdp_attention requires spiking predecessor context")
        elif node.op_name in {"geometric_product", "tropical_matmul"}:
            if not _has_ancestor_op(graph, nid, _LOCAL_WINDOW_VALID_PREDS):
                violations.append(
                    f"{node.op_name} requires normalized predecessor context"
                )
            if not _has_descendant_op(
                graph, nid, _RESTRICTED_LINEAR_SUCCESSORS, children
            ):
                violations.append(
                    f"{node.op_name} must feed projection/residual context"
                )
        elif node.op_name == "n_way_sparse_router":
            if not _has_ancestor_op(graph, nid, _LOCAL_WINDOW_VALID_PREDS):
                violations.append(
                    "n_way_sparse_router requires normalized predecessor context"
                )
            if not child_ops & _ROUTER_VALID_SUCCESSORS:
                violations.append(
                    "n_way_sparse_router must feed rmsnorm/layernorm/linear_proj, not stand alone"
                )
        elif node.op_name == "tropical_center":
            if not parent_ops & _TROPICAL_BRIDGE_PREDS:
                violations.append(
                    "tropical_center requires tropical or normalized predecessor context"
                )
            if not _has_descendant_op(
                graph, nid, _RESTRICTED_LINEAR_SUCCESSORS, children
            ):
                violations.append(
                    "tropical_center must feed projection/residual context"
                )
        elif node.op_name == "early_exit":
            if not _has_descendant_op(graph, nid, {"add"}, children):
                violations.append("early_exit must sit inside a residual/routing block")
        elif node.op_name == "reciprocal":
            if not _has_immediate_predecessor_op(
                graph, nid, {"rmsnorm", "layernorm", "sigmoid", "tanh"}
            ):
                violations.append(
                    "reciprocal requires immediate bounded predecessor (norm/sigmoid/tanh)"
                )
        elif node.op_name == "exp":
            if not _has_immediate_predecessor_op(
                graph, nid, {"rmsnorm", "layernorm", "sigmoid", "tanh"}
            ):
                violations.append(
                    "exp requires immediate bounded predecessor (norm/sigmoid/tanh)"
                )
            if not _has_immediate_successor_op(
                children, graph, nid, {"rmsnorm", "layernorm", "sigmoid", "tanh", "mul"}
            ):
                violations.append(
                    "exp requires immediate stabilizer successor (norm/sigmoid/tanh/mul)"
                )
        elif node.op_name == "log":
            if not _has_immediate_predecessor_op(
                graph,
                nid,
                {"rmsnorm", "layernorm", "sigmoid", "softmax_last", "exp", "abs"},
            ):
                violations.append(
                    "log requires immediate positive-bounded predecessor (norm/sigmoid/exp/abs)"
                )
            if not _has_immediate_successor_op(
                children,
                graph,
                nid,
                {"rmsnorm", "layernorm", "sigmoid", "tanh", "mul", "linear_proj"},
            ):
                violations.append(
                    "log requires immediate stabilizer successor (norm/sigmoid/tanh/mul/linear_proj)"
                )
        elif node.op_name == "div_safe":
            if not _has_immediate_predecessor_op(
                graph, nid, {"rmsnorm", "layernorm", "sigmoid", "softmax_last"}
            ):
                violations.append(
                    "div_safe requires immediate normalized predecessor (norm/sigmoid/softmax)"
                )
            if not _has_immediate_successor_op(
                children, graph, nid, _STABILIZER_SUCCESSORS | {"add"}
            ):
                violations.append(
                    "div_safe requires immediate stabilizer/merge successor (proj/norm/mul/add)"
                )
        elif node.op_name == "sign_ste":
            if not _has_immediate_successor_op(
                children, graph, nid, {"mul", "linear_proj"}
            ):
                violations.append(
                    "sign_ste must immediately feed mul/linear_proj (STE gradient flow)"
                )
        elif node.op_name == "sub":
            if not _has_immediate_successor_op(
                children, graph, nid, _STABILIZER_SUCCESSORS
            ):
                violations.append(
                    "sub requires immediate stabilizer successor (proj/norm/mul)"
                )
        elif node.op_name == "minimum":
            if not _has_descendant_op(
                graph, nid, _RESTRICTED_LINEAR_SUCCESSORS, children
            ):
                violations.append("minimum must feed projection/residual context")
        elif node.op_name == "cumsum":
            if not _has_immediate_successor_op(
                children, graph, nid, {"rmsnorm", "layernorm"}
            ):
                violations.append(
                    "cumsum requires immediate norm successor (running sum grows unbounded)"
                )
        elif node.op_name == "matmul":
            if not _has_immediate_predecessor_op(
                graph, nid, {"linear_proj", "linear_proj_up", "linear_proj_down"}
            ):
                violations.append(
                    "matmul requires immediate projection predecessor (68% fail without)"
                )
            if not _has_immediate_successor_op(
                children, graph, nid, _STABILIZER_SUCCESSORS | {"add"}
            ):
                violations.append(
                    "matmul requires immediate stabilizer/merge successor (proj/norm/mul/add)"
                )
        elif node.op_name == "cosine_similarity":
            if not _has_immediate_predecessor_op(
                graph, nid, {"linear_proj", "linear_proj_up", "linear_proj_down"}
            ):
                violations.append(
                    "cosine_similarity requires immediate projection predecessor (95% fail without)"
                )
        elif node.op_name == "outer_product":
            if not _has_immediate_successor_op(
                children, graph, nid, _STABILIZER_SUCCESSORS | {"add"}
            ):
                violations.append(
                    "outer_product requires immediate stabilizer/merge successor (proj/norm/mul/add)"
                )
        elif node.op_name == "tropical_matmul":
            if not _has_immediate_successor_op(
                children,
                graph,
                nid,
                {"linear_proj", "linear_proj_down", "rmsnorm", "layernorm"},
            ):
                violations.append(
                    "tropical_matmul requires immediate projection/norm successor"
                )

    return violations


# ── Graph validation ──────────────────────────────────────────────


def validate_context_rules(graph: ComputationGraph) -> Optional[str]:
    violations = find_graph_context_violations(graph)
    return violations[0] if violations else None


# ── Byte-safety enforcement ──────────────────────────────────────

# Lazily computed set of ops with byte_safe=False in the registry.
_BYTE_UNSAFE_OPS: FrozenSet[str] = frozenset(
    name for name, op in PRIMITIVE_REGISTRY.items() if not op.byte_safe
)


def find_byte_safety_violations(graph: ComputationGraph) -> List[str]:
    """Check that no byte-unsafe ops appear in the graph.

    Call this when the graph will run in native or quantized execution
    modes where token reordering/merging breaks tensor layout assumptions.
    """
    violations: List[str] = []
    for nid, node in graph.nodes.items():
        if node.is_input:
            continue
        if node.op_name in _BYTE_UNSAFE_OPS:
            violations.append(
                f"Byte-unsafe op '{node.op_name}' (node {nid}) is not "
                f"allowed in native/quantized execution mode"
            )
    return violations


# ── 5.5: Weight sharing context rules ────────────────────────────
# tied_proj and shared_basis_proj are efficient alternatives to
# linear_proj. They should follow the same placement rules as
# linear_proj but are additionally safe to chain (weight sharing
# reduces redundant parameters).

# ── True routing ops: dispatch tokens to heterogeneous experts ────
CONTEXT_RULES["hetero_moe"] = ContextRule(
    search_mode=SearchMode.GENERAL,
    forbidden_predecessors=_REDUCE_OPS,
    forbidden_successors=_GATING_OPS | frozenset({"output_head"}),
    requires_residual_context=True,
)

CONTEXT_RULES["arch_router"] = ContextRule(
    search_mode=SearchMode.GENERAL,
    forbidden_predecessors=_REDUCE_OPS,
    forbidden_successors=_GATING_OPS | frozenset({"output_head"}),
    requires_residual_context=True,
)

CONTEXT_RULES["compute_budget_router"] = ContextRule(
    search_mode=SearchMode.GENERAL,
    forbidden_predecessors=_REDUCE_OPS,
    forbidden_successors=_GATING_OPS | frozenset({"output_head"}),
    requires_residual_context=True,
)

CONTEXT_RULES["tied_proj"] = ContextRule(
    search_mode=SearchMode.GENERAL,
    forbidden_predecessors=_REDUCE_OPS,
    forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
)

CONTEXT_RULES["shared_basis_proj"] = ContextRule(
    search_mode=SearchMode.GENERAL,
    forbidden_predecessors=_REDUCE_OPS,
    forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
)
