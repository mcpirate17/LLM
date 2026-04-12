"""Context rules — per-op rule registry and derived sets."""

from __future__ import annotations

from typing import Dict, FrozenSet

from ._context_types import (
    CONTEXT_CLASS_STRUCTURAL,
    ContextRule,
    SearchMode,
)
from ._context_op_sets import (
    _CAUSAL_SENSITIVE_OPS,
    _FULL_DIM_CONSUMERS,
    _FULL_DIM_OPS,
    _GATING_OPS,
    _MASK_OPS,
    _MATH_SPACE_OPS,
    _MIXING_OPS,
    _REDUCE_OPS,
    _ROUTING_OPS,
    _SPARSE_LINEAR_OPS,
    _STRUCTURAL_SPLIT_OPS,
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
        forbidden_predecessors=frozenset(
            {
                "neg",  # common up->neg->down collapse chain with poor screening outcomes
            }
        ),
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
                "token_merge",  # old alias kept for notebook compatibility
                "adjacent_token_merge",  # destroyed token order breaks attention
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
    # Every success with loss < 0.05 has norm -> conv1d_seq -> silu -> selective_scan -> add.
    "selective_scan": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS
        | _STRUCTURAL_SPLIT_OPS
        | frozenset(
            {
                "add",  # dominant crash site: residual sum destabilizes scan state
                "identity",  # strips causal context
                "token_merge",  # old alias kept for notebook compatibility
                "adjacent_token_merge",  # destroys token ordering SSM depends on
                "transpose_sd",  # transposes (S,D) axes, breaks SSM state shape
            }
        ),
        forbidden_successors=frozenset(
            {
                "output_head",
                "linear_proj",  # raw scan output into projection is a common crash/no-learn pattern
                "selective_scan",  # SSM->SSM chaining 96% fail
                "state_space",  # same failure mode
                "ternary_projection",  # quantized ternary branch kills already fragile scan signal
            }
        ),
        requires_residual_context=True,  # SSM output needs residual path
    ),
    # transpose_sd: 95% S1 fail in random placement. Works in cross_dim_mixer (9%)
    # / dual_axis_block (5%) where templates manage the transpose lifecycle.
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
                "rmsnorm",  # historical low-value washout path after gated MLP
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
            }
        ),
    ),
    # layernorm->add has 30% fail rate (122F/289S) — not a hard ban.
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
    # moe_topk: norm->moe_topk 5% consec, only works after linear_proj (data: 2/2 S1)
    "moe_topk": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=frozenset(
            {
                "rmsnorm",
                "layernorm",
                "linear_proj_up",
            }  # norm needs proj between; up-project before gate is a recurrent bad route
        )
        | _REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    # moe_2expert: norm->moe_2expert 4% consec vs 19% co-occur
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
                "cos",  # 100% fail (10/10) — cos in [-1,1] width != sibling width
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
    # Elementwise binary
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
    # Linear algebra
    "matmul": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS
        | frozenset(
            {"rmsnorm", "layernorm"}
        ),  # 68% fail without proj between norm and matmul
        forbidden_successors=frozenset({"output_head", "gather_topk"})
        | _STRUCTURAL_SPLIT_OPS,
        requires_residual_context=True,
    ),
    "outer_product": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head", "gather_topk"})
        | _STRUCTURAL_SPLIT_OPS,
        requires_residual_context=True,
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
        requires_residual_context=True,
    ),
    # Reductions
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
    # Sequence
    "softmax_last": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}) | _STRUCTURAL_SPLIT_OPS,
    ),
    # Structural
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
    # Parameterized
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
    # Gating gaps
    "relu_gated_moe": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS
        | frozenset({"linear_proj_down", "rmsnorm", "layernorm"}),
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
        forbidden_predecessors=_REDUCE_OPS | _STRUCTURAL_SPLIT_OPS,
        forbidden_successors=_ROUTING_OPS
        | _CAUSAL_SENSITIVE_OPS
        | frozenset(
            {
                "output_head",
                "identity",
            }
        ),
        requires_residual_context=True,
    ),
    "hybrid_token_gate": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    "sparse_span_builder": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head", "add"}),
        requires_residual_context=True,
    ),
    "hybrid_sparse_router": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    "lane_conditioned_block": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    "default_path": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    "calibrated_branch_merge": ContextRule(
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=frozenset({"output_head"}),
        requires_residual_context=True,
    ),
    # 2-input routing/compression ops
    "dual_compression_blend": ContextRule(  # was: compression_mixture_experts
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=(_ROUTING_OPS - frozenset({"moe_topk", "moe_2expert"}))
        | frozenset({"output_head"}),
    ),
    "score_depth_blend": ContextRule(  # was: mixed_recursion_gate
        search_mode=SearchMode.GENERAL,
        forbidden_predecessors=_REDUCE_OPS,
        forbidden_successors=_ROUTING_OPS | frozenset({"output_head"}),
    ),
    # Math-space gaps
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
        forbidden_successors=frozenset({"output_head", "linear_proj"})
        | _STRUCTURAL_SPLIT_OPS,
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


# ── Derived sets (frozen at import time) ──────────────────────────

STRUCTURAL_OPS: FrozenSet[str] = frozenset(
    {
        "identity",
        "split2",
        "split3",
        "concat",
    }
)

# Ops exempt from per-op S1 attribution: scaffolding, positional masks,
# dimension-reduction ops, and parameter-free transforms.
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


# ── 5.5: Weight sharing context rules ────────────────────────────
# tied_proj and shared_basis_proj are efficient alternatives to
# linear_proj. They should follow the same placement rules as
# linear_proj but are additionally safe to chain (weight sharing
# reduces redundant parameters).

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


# ── Query helpers ─────────────────────────────────────────────────


def is_structural(op_name: str) -> bool:
    return op_name in STRUCTURAL_OPS
