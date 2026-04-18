"""Routing-first templates — mandatory routing structure."""

from __future__ import annotations

import random

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph
from ._template_helpers import (
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_CHANNEL,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_EFFICIENT_PROJ,
    MOTIF_CLASS_MATH_SPACE,
    MOTIF_CLASS_NORM,
    MOTIF_CLASS_SSM,
    MotifWeights,
    _FFN_CLASSES,
    _MIXER_CLASSES,
    _fix_dim,
    _instantiate_motif,
    _pick_compatible_motif,
    _pick_compatible_motif_from_classes,
    record_template_slot_binding,
    template_add_op as _add,
    template_add_residual as _residual,
)


# ── Routing-First Templates (Phase 2) ──────────────────────────────
#
# These templates MANDATE routing structure: every graph produced by
# these templates has a difficulty scorer and differential compute paths.
# The grammar fills motif slots, but the routing skeleton is fixed.

# Set of all routing-first template names for grammar filtering.
ROUTING_TEMPLATES: frozenset = frozenset(
    {
        "difficulty_routed_block",
        "three_lane_adaptive",
        "cascaded_early_exit",
        "hybrid_sparse_triplet_router",
        "multiscale_difficulty_router",
        "multiscale_difficulty_router_easy_attn_ssm",
        "multiscale_difficulty_router_adaptive_attn_ssm",
        "multiscale_difficulty_router_blocksparse_attn_ssm",
        "multiscale_rich_lane_router",
        "intelligent_multilane_router",
        "recursive_depth_router",
        "conditional_compute",
        "token_merge_block",
        "routed_bottleneck",
        "sparse_moe_block",
        # Attention templates that produce routing ops internally
        "attn_routing_block",
        "attn_moe_block",
        "attn_three_way_split",
        "attn_conditional_compute",
        "attn_sparse_moe",
        "diff_attn_routing",
        "local_attn_routing",
        "latent_attn_moe",
        "local_attn_moe",
        "diff_attn_moe",
        "graph_attn_moe",
        # Role-slot trunk+sidecar templates (capability-first family).
        # These enforce explicit binding_write → global_retrieval → binding_read
        # with a typed-entropy or sparse-router controller and gated merge.
        "typed_slot_memory_block",
        "sparse_relation_graph_block",
        "token_program_interpreter_block",
        # Role-slot v2: proven trunks + retrieval sidecar.
        "conv_residual_retrieval_v2",
        "state_space_retrieval_v2",
        "latent_attn_retrieval_v2",
    }
)


# Capability-first subset: templates that wire an explicit exact-retrieval
# sidecar merged into a compression trunk via a gated add. Used by the
# ``capability_first`` grammar preset to pressure the search toward graphs
# that can win ppl AND binding/induction/ar simultaneously.
CAPABILITY_FIRST_TEMPLATES: frozenset = frozenset(
    {
        "typed_slot_memory_block",
        "sparse_relation_graph_block",
        "token_program_interpreter_block",
        "conv_residual_retrieval_v2",
        "state_space_retrieval_v2",
        "latent_attn_retrieval_v2",
    }
)

# Curated candidate pools for the promoted multiscale rich router.
# The broader manifest-compatible menus remain documented in component metadata,
# but generation is intentionally narrower here because short-run audit data
# showed several medium/hard options were either underpowered, collapse-prone,
# or outright non-viable in this template context.
NEXT_MULTISCALE_MEDIUM_LANE_OPS: tuple[str, ...] = (
    "conv1d_seq",
    "semi_structured_2_4_linear",
    "block_sparse_linear",
    "adaptive_lane_mixer",
    "cheap_verify_blend",
)

DIFFICULTY_ROUTER_MEDIUM_LANE_OPS: tuple[str, ...] = (
    "semi_structured_2_4_linear",
    "block_sparse_linear",
    "adaptive_lane_mixer",
)

NEXT_MULTISCALE_HARD_LANE_OPS: tuple[str, ...] = (
    "mixed_recursion_gate",
    "dual_compression_blend",
    "adaptive_recursion",
    "route_recursion",
    "moe_topk",
)

DIFFICULTY_ROUTER_HARD_LANE_OPS: tuple[str, ...] = ("route_recursion",)

DIFFICULTY_ROUTER_HARD_ATTENTION_OPS: tuple[str, ...] = ("graph_attention",)

INTELLIGENT_PRE_ROUTER_OPTIONAL_OPS: tuple[str, ...] = (
    "cheap_verify_blend",
    "linear_proj",
)

INTELLIGENT_EASY_MANDATORY_OPS: tuple[str, ...] = (
    "cheap_verify_blend",
    "conv_only",
    "conv1d_seq",
)

INTELLIGENT_EASY_OPTIONAL_OPS: tuple[str, ...] = (
    "linear_proj",
    "nm_sparse_linear",
    "default_path",
)

INTELLIGENT_MEDIUM_MANDATORY_OPS: tuple[str, ...] = (
    "route_lanes",
    "adaptive_lane_mixer",
    "semi_structured_2_4_linear",
    "block_sparse_linear",
)

INTELLIGENT_MEDIUM_OPTIONAL_OPS: tuple[str, ...] = (
    "linear_proj",
    "nm_sparse_linear",
)

INTELLIGENT_HARD_MANDATORY_OPS: tuple[str, ...] = (
    "adaptive_recursion",
    "route_recursion",
    "moe_topk",
    "moe_2expert",
)

INTELLIGENT_HARD_OPTIONAL_OPS: tuple[str, ...] = (
    "state_space",
    "linear_proj",
)

INTELLIGENT_POST_MERGE_OPTIONAL_OPS: tuple[str, ...] = (
    "linear_proj",
    "nm_sparse_linear",
)


def _next_multiscale_medium_config(op_name: str, rng: random.Random) -> dict:
    if op_name == "route_lanes":
        return {"n_lanes": 3}
    if op_name == "adaptive_lane_mixer":
        return {"n_lanes": 3}
    return {}


def _next_multiscale_hard_config(op_name: str, rng: random.Random) -> dict:
    if op_name in {"route_recursion", "adaptive_recursion", "mixed_recursion_gate"}:
        max_depth = rng.choice([2, 3, 4])
        return {
            "max_depth": max_depth,
            "curriculum_enabled": True,
            "curriculum_warmup_frac": 0.25,
            "curriculum_mid_frac": 0.65,
            "active_depth_start": 1,
            "active_depth_mid": min(2, max_depth),
            "active_depth_end": max_depth,
        }
    if op_name == "moe_topk":
        return {"num_experts": rng.choice([2, 4]), "top_k": 1}
    if op_name == "moe_2expert":
        return {}
    if op_name == "state_space":
        return {}
    return {}


def _multiscale_gate_config() -> dict:
    return {
        "threshold": 0.5,
        "gate_temperature": 1.0,
        "curriculum_enabled": True,
        "curriculum_warmup_frac": 0.25,
        "curriculum_mid_frac": 0.65,
        "threshold_start": 0.34,
        "threshold_mid": 0.4,
        "threshold_end": 0.46,
        "gate_temperature_start": 1.35,
        "gate_temperature_mid": 1.1,
        "gate_temperature_end": 1.0,
    }


def _multiscale_sparse_router_config(span_width: int) -> dict:
    return {
        "span_width": span_width,
        "lane_count": span_width,
        "confidence_threshold": 0.55,
        "min_keep_fraction": 0.125,
        "route_temperature": 0.85,
        "curriculum_enabled": True,
        "curriculum_warmup_frac": 0.25,
        "curriculum_mid_frac": 0.65,
        "confidence_threshold_start": 0.3,
        "confidence_threshold_mid": 0.4,
        "confidence_threshold_end": 0.48,
        "min_keep_fraction_start": 0.28,
        "min_keep_fraction_mid": 0.2,
        "min_keep_fraction_end": 0.16,
        "route_temperature_start": 1.35,
        "route_temperature_mid": 1.05,
        "route_temperature_end": 0.9,
    }


def _multiscale_merge_config(
    *,
    primary_role: str,
    secondary_role: str,
    min_secondary_share: float,
    max_secondary_share: float,
    min_secondary_start: float | None = None,
    min_secondary_mid: float | None = None,
    min_secondary_end: float | None = None,
    max_secondary_start: float | None = None,
    max_secondary_mid: float | None = None,
    max_secondary_end: float | None = None,
) -> dict:
    return {
        "n_branches": 2,
        "primary_role": primary_role,
        "secondary_role": secondary_role,
        "normalize_inputs": True,
        "merge_temperature": 0.9,
        "min_secondary_share": min_secondary_share,
        "max_secondary_share": max_secondary_share,
        "curriculum_enabled": True,
        "curriculum_warmup_frac": 0.25,
        "curriculum_mid_frac": 0.65,
        "min_secondary_share_start": min_secondary_share
        if min_secondary_start is None
        else min_secondary_start,
        "min_secondary_share_mid": min_secondary_share
        if min_secondary_mid is None
        else min_secondary_mid,
        "min_secondary_share_end": min_secondary_share
        if min_secondary_end is None
        else min_secondary_end,
        "max_secondary_share_start": max_secondary_share
        if max_secondary_start is None
        else max_secondary_start,
        "max_secondary_share_mid": max_secondary_share
        if max_secondary_mid is None
        else max_secondary_mid,
        "max_secondary_share_end": max_secondary_share
        if max_secondary_end is None
        else max_secondary_end,
    }


def _single_input_op_config(op_name: str, model_dim: int, rng: random.Random) -> dict:
    if op_name == "linear_proj":
        return {"out_dim": model_dim}
    if op_name in {"route_lanes", "adaptive_lane_mixer"}:
        return {"n_lanes": 3}
    if op_name in {"route_recursion", "adaptive_recursion", "mixed_recursion_gate"}:
        return {"max_depth": rng.choice([2, 3, 4])}
    if op_name == "moe_topk":
        return {"num_experts": rng.choice([2, 4]), "top_k": 1}
    return {}


def _apply_optional_single_input_ops(
    graph: ComputationGraph,
    start_id: int,
    rng: random.Random,
    candidates: tuple[str, ...],
    count: int,
    *,
    context_prefix: str,
) -> tuple[int, list[str]]:
    node_id = start_id
    selected: list[str] = []
    for idx, op_name in enumerate(rng.sample(list(candidates), k=count), start=1):
        node_id = _add(
            graph,
            op_name,
            [node_id],
            _single_input_op_config(op_name, graph.model_dim, rng),
            context=f"{context_prefix}.optional_{idx}",
        )
        selected.append(op_name)
    return node_id, selected



# ── Template implementations live in split modules ─────────────────
# Each module imports the helpers above and contributes a disjoint set
# of tpl_* functions. Re-exported here so `from ._templates_routing
# import tpl_X` continues to work for all legacy call sites.

from ._templates_routing_core import (  # noqa: E402,F401
    tpl_difficulty_routed_block,
    tpl_three_lane_adaptive,
    tpl_cascaded_early_exit,
    tpl_hybrid_sparse_triplet_router,
    tpl_multiscale_difficulty_router,
    tpl_multiscale_difficulty_router_adaptive_attn_ssm,
    tpl_multiscale_rich_lane_router,
)
from ._templates_routing_multilane import (  # noqa: E402,F401
    tpl_intelligent_multilane_router,
    tpl_recursive_depth_router,
    tpl_depth_token_mask_block,
)
from ._templates_routing_gated import (  # noqa: E402,F401
    tpl_latent_compress_block,
    tpl_latent_compress_rwkv,
    tpl_signal_routed_compression,
    tpl_dual_routing_stack,
    tpl_dual_routing_deep,
    tpl_routing_conditioned_moe,
    tpl_mixed_recursion,
    tpl_depth_gated_block,
    tpl_depth_gated_block_matmul_stable,
    tpl_depth_gated_block_matmul_norm,
    tpl_gated_lane_blend_block,
    tpl_feature_sparse_block,
    tpl_topk_retrieval,
    tpl_adaptive_sparse,
    tpl_adaptive_conv_ffn,
    tpl_adaptive_ssm_chain,
    tpl_adaptive_lane_recursion,
)
