"""Template Registry — maps names to template functions + weights.

Template implementations live in submodules:
  _templates_core.py     — workhorse templates (residual, transformer, etc.)
  _templates_routing.py  — routing-first templates (difficulty-gated, etc.)
  _templates_exotic.py   — binary-op safety, math-space, spiking templates
  _templates_attention.py — attention-heavy structural templates
  _templates_attention_tail.py — generated attention wrappers and tail templates
  _templates_research.py — 0% S1 fixes, zero-coverage ops, reference architectures
  _template_helpers.py   — shared helpers (motif picking, instantiation)
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple

if TYPE_CHECKING:
    from .graph import ComputationGraph
from .native_template_selection import (
    pick_template_index_native,
    pick_template_index_python,
)

# Ops the grammar can emit through split template modules, alias substitution,
# or helper-inserted structural nodes even when the literal op name does not
# appear in this file's template registry body.
GRAMMAR_REACHABLE_OPS = frozenset(
    {
        "adaptive_lane_mixer",
        "adaptive_recursion",
        "add",
        "cascade",
        "causal_mask",
        "compression_mixture_experts",
        "concat",
        "cosine_similarity",
        "div_safe",
        "early_exit",
        "entropy_score",
        "gather_topk",
        "geometric_product",
        "hyp_distance",
        "matmul",
        "maximum",
        "minimum",
        "mixed_recursion_gate",
        "mod_topk",
        "mul",
        "n_way_sparse_router",
        "outer_product",
        "progressive_compression_gate",
        "relu_gate_routing",
        "route_lanes",
        "route_recursion",
        "route_topk",
        "routing_conditioned_compression",
        "softmax_last",
        "speculative",
        "split2",
        "split3",
        "sub",
        "token_merge",
        "token_type_classifier",
        "tropical_add",
    }
)

# Template families that intentionally do not require a 1:1 dedicated component
# graph. Many are variant wrappers around the same validated structure or
# reference architecture generators better covered by native hotpath and
# grammar-level tests.
COMPONENT_GRAPH_EXEMPT_TEMPLATE_PREFIXES = (
    "adaptive_",
    "attn_",
    "diff_attn_",
    "graph_attn_",
    "latent_attn_",
    "linear_attn_",
    "local_attn_",
)

COMPONENT_GRAPH_EXEMPT_TEMPLATES = frozenset(
    {
        "arch_router_block",
        "causal_mix_block",
        "chebyshev_block",
        "compute_budget_block",
        "conv_residual_block",
        "cross_dim_mixer",
        "cumulative_sequence",
        "depth_gated_block",
        "depth_gated_block_matmul",
        "depth_gated_block_matmul_norm",
        "depth_gated_block_matmul_stable",
        "depth_token_mask_block",
        "diff_attention_block",
        "dual_attn_block",
        "dual_axis_block",
        "dual_routing_deep",
        "dual_routing_stack",
        "exp_gated_residual",
        "feature_sparse_block",
        "fused_gelu_ffn",
        "gpt2_reference",
        "hetero_moe_block",
        "hyperbolic_bridge_block",
        "integral_kernel_block",
        "iterative_refinement",
        "kronecker_block",
        "log_gated",
        "mamba_reference",
        "intelligent_multilane_router",
        "multiscale_difficulty_router",
        "multiscale_difficulty_router_adaptive_attn_ssm",
        "multiscale_rich_lane_router",
        "multi_head_mix_block",
        "n_way_moe_block",
        "poincare_add_bridge",
        "reciprocal_gated",
        "recurrent_delta_block",
        "reduce_attend",
        "rope_attention_block",
        "routing_conditioned_moe",
        "rwkv_block",
        "rwkv_double_norm",
        "rwkv_sparse_chain",
        "sign_ste_gated",
        "spiking_moe_block",
        "spiking_residual_block",
        "spiking_stdp_block",
        "sqrt_gated_ffn",
        "state_space_block",
        "token_merge_conv",
        "tropical_center_block",
        "ultrametric_attention_block",
        "windowed_attention",
        "recursive_attn_ssm_hybrid",
        "recursive_attn_ssm_depth",
        "attn_normalized_matmul_pinned",
        "difficulty_routed_attention_block",
        "strided_attention_block",
        "gated_progressive_attention_block",
        "gated_linear_attention_block",
        "long_conv_hyena_block",
        "associative_memory_block",
        "mixture_of_recursions_block",
        "codex_ssm_retention_block",
        "codex_ssm_delta_memory_block",
        "codex_ssm_mla_gated_block",
        "codex_ssm_local_recall_block",
        "induction_matmul_block",
        "recursive_moe_attn",
        "typed_slot_memory_block",
        "sparse_relation_graph_block",
        "token_program_interpreter_block",
        "conv_residual_retrieval_v2",
        "state_space_retrieval_v2",
        "latent_attn_retrieval_v2",
    }
)

RETIRED_TEMPLATE_NAMES = frozenset(
    {
        "attn_reciprocal_gated",
        "attn_softmax_router_sidecar",
        "multiscale_difficulty_router_blocksparse_attn_ssm",
        "multiscale_difficulty_router_easy_attn_ssm",
    }
)


def is_component_graph_exempt_template(template_name: str) -> bool:
    return (
        template_name in COMPONENT_GRAPH_EXEMPT_TEMPLATES
        or template_name.startswith(COMPONENT_GRAPH_EXEMPT_TEMPLATE_PREFIXES)
    )


# Re-export public API used by grammar.py, tests, etc.
from ._template_helpers import (  # noqa: F401
    MotifWeights,
    TemplateFn,
    TemplateBuildError,
    _instantiate_motif,
    _pick_compatible_motif,
    _pick_compatible_motif_from_classes,
    _fix_dim,
    _motif_is_compatible,
)
from ._template_attention_manifest import (
    ATTENTION_TEMPLATE_DEFAULT_WEIGHTS,
    ATTENTION_TEMPLATE_REGISTRY,
)
from ._template_role_slot_manifest import (
    ROLE_SLOT_TEMPLATE_DEFAULT_WEIGHTS,
    ROLE_SLOT_TEMPLATE_REGISTRY,
)
from ._template_research_manifest import (
    RESEARCH_TEMPLATE_DEFAULT_WEIGHTS,
    RESEARCH_TEMPLATE_REGISTRY,
)
from ._template_routing_manifest import (
    ROUTING_TEMPLATE_DEFAULT_WEIGHTS,
    ROUTING_TEMPLATE_REGISTRY,
)

# ── Import all template functions from submodules ──────────────────

from ._templates_core import (  # noqa: F401
    tpl_residual_block,
    tpl_sequential,
    tpl_transformer_block,
    tpl_parallel_split,
    tpl_gated_maximum,
    tpl_three_way_split,
    tpl_bottleneck,
    tpl_moe,
    tpl_hybrid_parallel,
    tpl_gated_residual,
    tpl_dense_cascade,
    tpl_sparse_ffn,
    tpl_sparse_moe_block,
    tpl_routed_bottleneck,
    tpl_token_merge_block,
    tpl_token_merge_conv,
    tpl_conditional_compute,
    tpl_recursive_attn_ssm_hybrid,
    tpl_induction_matmul_block,
    tpl_recursive_moe_attn,
)

from ._templates_routing import (  # noqa: F401
    CAPABILITY_FIRST_TEMPLATES,
    ROUTING_TEMPLATES,
    # Retired 2026-04-17: tpl_multiscale_difficulty_router_blocksparse_attn_ssm
    # and tpl_multiscale_difficulty_router_easy_attn_ssm — both 0% S1 after
    # 24-25 runs; over-complex routing dominated signal. Names retained in
    # RETIRED_TEMPLATE_NAMES below for back-compat dedup.
    # Retired 2026-04-17: tpl_depth_gated_block_matmul — depth signal (B,S,1)
    # cannot meaningfully gate O(S²) matmul compute; weight=0.25 with
    # _matmul_stable variant already covering the use case.
)

from ._templates_exotic import (  # noqa: F401
    tpl_normalized_matmul,
    tpl_gated_product,
    tpl_safe_division,
    tpl_cosine_scoring,
    tpl_decay_sequence,
    tpl_hyp_distance_scoring,
    tpl_tropical_residual,
    tpl_tropical_center_block,
    tpl_geometric_product_block,
    tpl_residual_difference,
    tpl_tropical_matmul_block,
    tpl_gated_minimum,
    tpl_spiking_residual_block,
    tpl_spiking_moe_block,
    tpl_hyperbolic_bridge_block,
    tpl_poincare_add_bridge,
    tpl_n_way_moe_block,
    tpl_conv_residual_block,
    tpl_causal_mix_block,
    tpl_iterative_refinement,
    tpl_recurrent_delta_block,
)

# ── Template Registry ───────────────────────────────────────────────

TEMPLATES: Dict[str, TemplateFn] = {
    "residual_block": tpl_residual_block,
    "sequential": tpl_sequential,
    "transformer_block": tpl_transformer_block,
    "parallel_split": tpl_parallel_split,
    "bottleneck": tpl_bottleneck,
    "moe": tpl_moe,
    "hybrid_parallel": tpl_hybrid_parallel,
    "gated_residual": tpl_gated_residual,
    "dense_cascade": tpl_dense_cascade,
    "sparse_ffn": tpl_sparse_ffn,
    "sparse_moe_block": tpl_sparse_moe_block,
    "routed_bottleneck": tpl_routed_bottleneck,
    "token_merge_block": tpl_token_merge_block,
    "token_merge_conv": tpl_token_merge_conv,
    "conditional_compute": tpl_conditional_compute,
    "recursive_attn_ssm_hybrid": tpl_recursive_attn_ssm_hybrid,
    "induction_matmul_block": tpl_induction_matmul_block,
    "recursive_moe_attn": tpl_recursive_moe_attn,
    **ROUTING_TEMPLATE_REGISTRY,
    "normalized_matmul": tpl_normalized_matmul,
    "gated_product": tpl_gated_product,
    "safe_division": tpl_safe_division,
    "cosine_scoring": tpl_cosine_scoring,
    "decay_sequence": tpl_decay_sequence,
    "residual_difference": tpl_residual_difference,
    "gated_minimum": tpl_gated_minimum,
    "hyp_distance_scoring": tpl_hyp_distance_scoring,
    "tropical_residual": tpl_tropical_residual,
    "tropical_matmul_block": tpl_tropical_matmul_block,
    "geometric_product_block": tpl_geometric_product_block,
    "gated_maximum": tpl_gated_maximum,
    "three_way_split": tpl_three_way_split,
    **RESEARCH_TEMPLATE_REGISTRY,
    "spiking_residual_block": tpl_spiking_residual_block,
    "spiking_moe_block": tpl_spiking_moe_block,
    "hyperbolic_bridge_block": tpl_hyperbolic_bridge_block,
    "poincare_add_bridge": tpl_poincare_add_bridge,
    "n_way_moe_block": tpl_n_way_moe_block,
    "conv_residual_block": tpl_conv_residual_block,
    "causal_mix_block": tpl_causal_mix_block,
    "iterative_refinement": tpl_iterative_refinement,
    "recurrent_delta_block": tpl_recurrent_delta_block,
    "tropical_center_block": tpl_tropical_center_block,
    **ROLE_SLOT_TEMPLATE_REGISTRY,
    **ATTENTION_TEMPLATE_REGISTRY,
}

DEFAULT_TEMPLATE_WEIGHTS: Dict[str, float] = {
    "residual_block": 3.0,
    "transformer_block": 3.0,
    "sequential": 2.0,
    "parallel_split": 1.5,
    "bottleneck": 1.5,
    "moe": 2.0,
    "hybrid_parallel": 1.0,
    "gated_residual": 1.5,
    "dense_cascade": 0.8,
    "sparse_ffn": 2.0,
    "sparse_moe_block": 4.0,
    "routed_bottleneck": 4.0,
    "token_merge_block": 7.0,
    "token_merge_conv": 6.0,
    "conditional_compute": 3.5,
    "recursive_attn_ssm_hybrid": 5.0,
    "induction_matmul_block": 4.5,
    "recursive_moe_attn": 5.0,
    **ROUTING_TEMPLATE_DEFAULT_WEIGHTS,
    "normalized_matmul": 2.0,
    "gated_product": 2.0,
    "safe_division": 1.5,
    "cosine_scoring": 2.0,
    "decay_sequence": 3.0,
    "residual_difference": 2.5,
    "gated_minimum": 2.5,
    "hyp_distance_scoring": 1.5,
    "tropical_residual": 2.5,
    "tropical_matmul_block": 2.5,
    "geometric_product_block": 1.5,
    "gated_maximum": 1.5,
    "three_way_split": 2.5,
    **RESEARCH_TEMPLATE_DEFAULT_WEIGHTS,
    "spiking_residual_block": 3.0,
    "spiking_moe_block": 4.0,
    "hyperbolic_bridge_block": 3.0,
    "poincare_add_bridge": 3.0,
    "n_way_moe_block": 3.5,
    "conv_residual_block": 3.0,
    "causal_mix_block": 2.5,
    "iterative_refinement": 2.5,
    "recurrent_delta_block": 3.5,
    "tropical_center_block": 2.5,
    **ROLE_SLOT_TEMPLATE_DEFAULT_WEIGHTS,
    # New high-performance templates: proven parallel attn+X + FFN pattern
    "recursive_attn_ssm_depth": 5.5,  # latent_attn||SSM + adaptive_recursion + FFN
    "latent_attn_padic_hybrid": 5.0,  # latent_attn||padic_expand + FFN
    "graph_attn_ssm_recursive": 4.5,  # graph_attn||SSM + FFN
    # NB: retired template names are tracked in RETIRED_TEMPLATE_NAMES above
    # for dedup/back-compat. Their templates and weight entries have been
    # removed — pick_template iterates TEMPLATES.keys() so dangling weight
    # entries here would just be dead lookups.
    **ATTENTION_TEMPLATE_DEFAULT_WEIGHTS,
}

_TEMPLATE_NAME_ORDER: Tuple[str, ...] = tuple(TEMPLATES.keys())
_TEMPLATE_DEFAULT_WEIGHT_VECTOR: Tuple[float, ...] = tuple(
    float(DEFAULT_TEMPLATE_WEIGHTS.get(name, 1.0)) for name in _TEMPLATE_NAME_ORDER
)


def pick_template(
    rng: random.Random,
    weights: Optional[Dict[str, float]] = None,
    exploration_budget: float = 0.0,
    allowed_template_names: Optional[Iterable[str]] = None,
) -> Tuple[str, TemplateFn, bool]:
    """Pick a template weighted by success priors.

    When exploration_budget > 0, that fraction of picks ignore weights
    and select uniformly from ALL templates — including zero-weighted ones.
    This guarantees every template gets coverage.

    Returns (name, fn, was_exploration).
    """
    exploration_draw = rng.random() if exploration_budget > 0.0 else 1.0
    selection_draw = rng.random()
    selection = pick_template_index_native(
        _TEMPLATE_NAME_ORDER,
        _TEMPLATE_DEFAULT_WEIGHT_VECTOR,
        weights,
        allowed_names=allowed_template_names,
        exploration_budget=exploration_budget,
        exploration_draw=exploration_draw,
        selection_draw=selection_draw,
    )
    if selection is None:
        index, was_exploration = pick_template_index_python(
            _TEMPLATE_NAME_ORDER,
            _TEMPLATE_DEFAULT_WEIGHT_VECTOR,
            weights,
            allowed_names=allowed_template_names,
            exploration_budget=exploration_budget,
            exploration_draw=exploration_draw,
            selection_draw=selection_draw,
        )
    else:
        index, was_exploration = selection
    name = _TEMPLATE_NAME_ORDER[index]
    return name, TEMPLATES[name], was_exploration


def apply_template(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    template_name: Optional[str] = None,
    template_weights: Optional[Dict[str, float]] = None,
    motif_weights: MotifWeights = None,
    op_weights: Optional[Dict[str, float]] = None,
    exploration_budget: float = 0.0,
    allowed_template_names: Optional[Iterable[str]] = None,
) -> int:
    """Apply a template to the graph. Main entry point for grammar."""
    prev_next_id = graph._next_id
    prev_output_id = graph._output_node_id
    prev_ir_version = graph._ir_version
    prev_metadata = dict(graph.metadata)

    was_exploration = False
    if template_name and template_name in TEMPLATES:
        name = template_name
        fn = TEMPLATES[name]
    else:
        name, fn, was_exploration = pick_template(
            rng, template_weights, exploration_budget, allowed_template_names
        )
    if was_exploration:
        graph.metadata["_template_exploration_used"] = True
    if op_weights:
        graph.metadata["_op_weights"] = op_weights
    graph.metadata.setdefault("templates_used", []).append(name)
    prev_template = graph.metadata.get("_active_template")
    prev_slot_counter = graph.metadata.get("_active_template_slot_counter")
    prev_template_instance = graph.metadata.get("_active_template_instance")
    graph.metadata["_active_template"] = name
    graph.metadata["_active_template_slot_counter"] = 0
    graph.metadata["_active_template_instance"] = (
        len(graph.metadata.get("templates_used", [])) - 1
    )
    try:
        return fn(graph, input_id, rng, motif_weights)
    except Exception:
        for nid in range(prev_next_id, graph._next_id):
            del graph.nodes[nid]
        graph._next_id = prev_next_id
        graph._output_node_id = prev_output_id
        graph._ir_version = prev_ir_version
        graph.metadata = prev_metadata
        graph._cache.clear()
        raise
    finally:
        if prev_template is None:
            graph.metadata.pop("_active_template", None)
        else:
            graph.metadata["_active_template"] = prev_template
        if prev_slot_counter is None:
            graph.metadata.pop("_active_template_slot_counter", None)
        else:
            graph.metadata["_active_template_slot_counter"] = prev_slot_counter
        if prev_template_instance is None:
            graph.metadata.pop("_active_template_instance", None)
        else:
            graph.metadata["_active_template_instance"] = prev_template_instance
