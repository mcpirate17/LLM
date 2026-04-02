"""Shared helpers for template implementations.

All template modules import from here — types, class groupings, motif
picking, and the motif instantiation engine.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, Optional, Tuple

from .graph import ComputationGraph
from .motifs import (
    MATH_SPACE_RULES,
    MOTIFS_BY_CLASS,
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_CHANNEL,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_EFFICIENT_PROJ,
    MOTIF_CLASS_FFN,
    MOTIF_CLASS_GATE,
    MOTIF_CLASS_GUARDED_ACT,
    MOTIF_CLASS_MATH_SPACE,
    MOTIF_CLASS_MOE,
    MOTIF_CLASS_NORM,  # noqa: F401 — re-exported for _templates_*.py
    MOTIF_CLASS_REDUCE,
    MOTIF_CLASS_SPARSE,
    MOTIF_CLASS_SSM,
    Motif,
    resolve_step,
)
from .primitives import (
    AlgebraicType,
    PRIMITIVE_REGISTRY,
    REQUIRES_RESIDUAL_BYPASS,
    algebraic_types_compatible,
)
from .context_rules import (
    motif_allowed_in_template as _motif_allowed_in_template,
)

# Type alias for motif weight dicts passed from judgment engine
MotifWeights = Optional[Dict[str, float]]

# Type for template callables
TemplateFn = Callable[
    [ComputationGraph, int, random.Random, MotifWeights],
    int,
]

# ── Motif class groupings for slot constraints ──────────────────────

# Slots that accept any sequence mixer
_MIXER_CLASSES: Tuple[str, ...] = (
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_SSM,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_CHANNEL,
    MOTIF_CLASS_MATH_SPACE,
)

# Slots inside bottleneck (D/2 → core → D): only ops that adapt to input dim.
# Excludes attention/MoE/FFN/math_space which build internal params at model_dim,
# wasting half their parameters on reduced-rank input.
_BOTTLENECK_CLASSES: Tuple[str, ...] = (
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_GATE,
    MOTIF_CLASS_GUARDED_ACT,
    MOTIF_CLASS_EFFICIENT_PROJ,
    MOTIF_CLASS_SPARSE,
)

# Slots that accept any FFN-like transform
_FFN_CLASSES: Tuple[str, ...] = (
    MOTIF_CLASS_FFN,
    MOTIF_CLASS_GATE,
    MOTIF_CLASS_SPARSE,
    MOTIF_CLASS_EFFICIENT_PROJ,
    MOTIF_CLASS_MOE,
    MOTIF_CLASS_GUARDED_ACT,
    MOTIF_CLASS_REDUCE,
)

# All motif classes
_ALL_CLASSES: Tuple[str, ...] = (
    MOTIF_CLASS_FFN,
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_SSM,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_GATE,
    MOTIF_CLASS_SPARSE,
    MOTIF_CLASS_MOE,
    MOTIF_CLASS_CHANNEL,
    MOTIF_CLASS_EFFICIENT_PROJ,
    MOTIF_CLASS_REDUCE,
    MOTIF_CLASS_GUARDED_ACT,
    MOTIF_CLASS_MATH_SPACE,
)


# ── Helper: instantiate a motif into a graph ────────────────────────

_INPUT_TYPE = AlgebraicType("euclidean", "real", "real")


def _node_output_type(graph: ComputationGraph, node_id: int) -> AlgebraicType:
    node = graph.nodes[node_id]
    if node.is_input:
        return _INPUT_TYPE
    return PRIMITIVE_REGISTRY[node.op_name].algebraic_type


def _step_is_compatible(graph: ComputationGraph, node_id: int, op_name: str) -> bool:
    current_type = _node_output_type(graph, node_id)
    next_op = PRIMITIVE_REGISTRY.get(op_name)
    if next_op is None:
        return False
    return algebraic_types_compatible(current_type, next_op.algebraic_type)


def _motif_is_compatible(graph: ComputationGraph, node_id: int, motif: Motif) -> bool:
    current_type = _node_output_type(graph, node_id)
    # Approximate depth of node_id for min_layer_depth check
    depth = 0
    nid = node_id
    while nid in graph.nodes:
        node = graph.nodes[nid]
        if node.is_input or not node.input_ids:
            break
        nid = node.input_ids[0]
        depth += 1
    for step in motif.steps:
        step_op = PRIMITIVE_REGISTRY.get(step.op_name)
        if step_op is None or not algebraic_types_compatible(
            current_type, step_op.algebraic_type
        ):
            return False
        # Reject motif if op requires deeper placement than current position
        if step_op.min_layer_depth > 0 and depth < step_op.min_layer_depth:
            return False
        current_type = step_op.algebraic_type
    return True


def _compatible_from_classes(
    graph: ComputationGraph,
    node_id: int,
    classes: Tuple[str, ...] | list[str],
) -> list[Motif]:
    """Return motifs from *classes* that are compatible at *node_id*."""
    active_tpl = graph.metadata.get("_active_template")
    pool: list[Motif] = []
    for cls in classes:
        pool.extend(MOTIFS_BY_CLASS.get(cls, []))
    return [
        m
        for m in pool
        if _motif_is_compatible(graph, node_id, m)
        and _motif_allowed_in_template(m, active_tpl)
    ]


def _select_from_candidates(
    candidates: list[Motif],
    rng: random.Random,
    weights: MotifWeights,
) -> Optional[Motif]:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    candidate_weights = [
        weights.get(m.name, m.lift) if weights else m.lift for m in candidates
    ]
    return rng.choices(candidates, weights=candidate_weights, k=1)[0]


def _pick_compatible_motif(
    graph: ComputationGraph,
    node_id: int,
    rng: random.Random,
    motif_class: str,
    weights: MotifWeights = None,
) -> Optional[Motif]:
    wildcard_prob = graph.metadata.get("_wildcard_slot_prob", 0.0)
    is_wildcard = wildcard_prob > 0 and rng.random() < wildcard_prob

    if is_wildcard:
        candidates = _compatible_from_classes(graph, node_id, _ALL_CLASSES)
    else:
        candidates = _compatible_from_classes(graph, node_id, (motif_class,))
        # Fallback to wildcard when prescribed class yields zero candidates
        if not candidates and wildcard_prob > 0:
            candidates = _compatible_from_classes(graph, node_id, _ALL_CLASSES)
            is_wildcard = True

    selected = _select_from_candidates(candidates, rng, weights)
    _record_slot_usage(
        graph,
        node_id=node_id,
        slot_classes=(motif_class,),
        candidates=candidates,
        selected=selected,
        wildcard=is_wildcard,
    )
    return selected


def _pick_compatible_motif_from_classes(
    graph: ComputationGraph,
    node_id: int,
    rng: random.Random,
    classes: Tuple[str, ...] | list[str],
    weights: MotifWeights = None,
) -> Optional[Motif]:
    wildcard_prob = graph.metadata.get("_wildcard_slot_prob", 0.0)
    is_wildcard = wildcard_prob > 0 and rng.random() < wildcard_prob

    # Check for slot adaptations (learned class expansions from DB)
    slot_adaptations = graph.metadata.get("_slot_adaptations")
    if slot_adaptations:
        slot_key = _current_slot_key(graph)
        extra_classes = slot_adaptations.get(slot_key, ())
        if extra_classes:
            classes = tuple(set(classes) | set(extra_classes))

    if is_wildcard:
        candidates = _compatible_from_classes(graph, node_id, _ALL_CLASSES)
    else:
        candidates = _compatible_from_classes(graph, node_id, classes)
        # Fallback to wildcard when prescribed classes yield zero candidates
        if not candidates and wildcard_prob > 0:
            candidates = _compatible_from_classes(graph, node_id, _ALL_CLASSES)
            is_wildcard = True

    selected = _select_from_candidates(candidates, rng, weights)
    _record_slot_usage(
        graph,
        node_id=node_id,
        slot_classes=tuple(classes),
        candidates=candidates,
        selected=selected,
        wildcard=is_wildcard,
    )
    return selected


def _current_slot_key(graph: ComputationGraph) -> str:
    """Build the slot_key for the currently active template slot."""
    tpl = graph.metadata.get("_active_template", "unknown")
    idx = graph.metadata.get("_active_template_slot_counter", 0)
    return f"{tpl}.slot{idx}"


def _record_slot_usage(
    graph: ComputationGraph,
    node_id: int,
    slot_classes: Tuple[str, ...] | list[str],
    candidates: list[Motif],
    selected: Optional[Motif],
    wildcard: bool = False,
) -> None:
    template_name = graph.metadata.get("_active_template")
    if not template_name:
        return
    slot_index = int(graph.metadata.get("_active_template_slot_counter", 0) or 0)
    graph.metadata["_active_template_slot_counter"] = slot_index + 1
    template_instance = int(graph.metadata.get("_active_template_instance", 0) or 0)
    class_list = [str(cls) for cls in slot_classes]
    entry = {
        "template_name": str(template_name),
        "template_instance": template_instance,
        "slot_index": slot_index,
        "slot_key": f"{template_name}[{template_instance}].slot{slot_index}",
        "slot_classes": class_list,
        "selected_motif": selected.name if selected else None,
        "selected_motif_class": selected.motif_class if selected else None,
        "candidate_count": len(candidates),
        "input_node_id": int(node_id),
        "wildcard": wildcard,
    }
    graph.metadata.setdefault("template_slot_usage", []).append(entry)


def _fix_dim(graph: ComputationGraph, node_id: int) -> int:
    """Add linear_proj to fix dimension back to model_dim if needed."""
    if graph.nodes[node_id].output_shape.dim != graph.model_dim:
        try:
            return graph.add_op(
                "linear_proj", [node_id], config={"out_dim": graph.model_dim}
            )
        except ValueError:
            return node_id
    return node_id


def _shuffle_wrap(
    graph: ComputationGraph,
    node_id: int,
    rng: random.Random,
    motif_classes: Tuple[str, ...] | list[str],
    weights: MotifWeights = None,
    prob: float = 1.0,
) -> int:
    """Optionally wrap a motif in transpose_sd (channel interleave/deinterleave).

    With probability `prob`, inserts transpose_sd before and after the motif.
    Returns the output node ID.
    """
    use_shuffle = prob >= 1.0 or rng.random() < prob
    current = node_id
    if use_shuffle:
        try:
            current = graph.add_op("transpose_sd", [current])
        except (ValueError, KeyError):
            use_shuffle = False
            current = node_id

    motif = _pick_compatible_motif_from_classes(
        graph, current, rng, motif_classes, weights
    )
    result = _instantiate_motif(graph, current, motif, rng) if motif else current

    if use_shuffle:
        result = _fix_dim(graph, result)
        try:
            result = graph.add_op("transpose_sd", [result])
        except (ValueError, KeyError):
            pass

    return result


def _instantiate_motif(
    graph: ComputationGraph,
    node_id: int,
    motif: Motif,
    rng: random.Random,
) -> int:
    """Add a motif's ops to the graph starting from node_id.

    Returns the output node ID. On shape mismatch, returns input node_id.
    Reads op_weights from graph.metadata["_op_weights"] if present.
    """
    current = node_id
    D = graph.model_dim
    _op_weights = graph.metadata.get("_op_weights")
    prev_op = (
        graph.nodes[current].op_name if not graph.nodes[current].is_input else None
    )
    for i, step in enumerate(motif.steps):
        # Peek at next step's op to inform "before" constraint
        next_step_op = motif.steps[i + 1].op_name if i + 1 < len(motif.steps) else None
        op_name, config = resolve_step(
            step, rng, prev_op=prev_op, next_op=next_step_op, op_weights=_op_weights
        )
        if not _step_is_compatible(graph, current, op_name):
            return node_id
        # Math-space safety: auto-insert rmsnorm if must_precede is unsatisfied
        ms_rules = MATH_SPACE_RULES.get(op_name)
        if ms_rules and "must_precede" in ms_rules:
            if prev_op not in ms_rules["must_precede"]:
                try:
                    current = graph.add_op("rmsnorm", [current])
                    prev_op = "rmsnorm"
                except (ValueError, KeyError):
                    return node_id
        # Auto-fix dim=1 outputs (from reduce_last ops like entropy_router):
        # if the current node reduced to dim=1 and the next op is parameterized,
        # insert a linear_proj to restore model_dim before the next op.
        cur_dim = graph.nodes[current].output_shape.dim
        if cur_dim == 1 and cur_dim != D:
            prim = PRIMITIVE_REGISTRY.get(op_name)
            if prim and prim.has_params:
                try:
                    current = graph.add_op(
                        "linear_proj", [current], config={"out_dim": D}
                    )
                except ValueError:
                    return node_id
                cur_dim = D
        if op_name in ("linear_proj", "fused_linear_gelu", "gated_linear"):
            config.setdefault("out_dim", D)
        elif op_name == "linear_proj_down":
            config.setdefault("out_dim", cur_dim // 2)
        elif op_name == "linear_proj_up":
            config.setdefault("out_dim", cur_dim * 2)
        elif op_name in (
            "nm_sparse_linear",
            "block_sparse_linear",
            "semi_structured_2_4_linear",
            "ternary_projection",
        ):
            config.setdefault("out_dim", cur_dim)
            if op_name == "nm_sparse_linear":
                config.setdefault("n", 2)
                config.setdefault("m", 4)
            elif op_name == "block_sparse_linear":
                config.setdefault("block_size", rng.choice([8, 16, 32]))
                config.setdefault("block_density", rng.uniform(0.05, 0.5))
        elif op_name in (
            "bottleneck_proj",
            "low_rank_proj",
            "grouped_linear",
            "shared_basis_proj",
            "tied_proj",
        ):
            config.setdefault("out_dim", cur_dim)
        elif op_name == "multi_head_mix":
            config.setdefault("n_heads", rng.choice([2, 4, 8]))
        elif op_name == "local_window_attn":
            # Cap window_size to avoid Triton shared memory overflow.
            # At D>=256, W=32 exceeds GPU shared memory (151KB > 100KB).
            choices = [8, 16] if cur_dim >= 256 else [8, 16, 32]
            config.setdefault("window_size", rng.choice(choices))
        elif op_name == "sliding_window_mask":
            config.setdefault("window_size", rng.choice([8, 16, 32]))
        elif op_name == "tropical_moe":
            config.setdefault("num_experts", rng.choice([2, 4]))
        elif op_name == "gather_topk":
            config.setdefault("k", rng.choice([4, 8, 16]))
        elif op_name in ("swiglu_mlp", "rwkv_channel", "moe_topk", "rwkv_time_mixing"):
            config.setdefault("mlp_ratio", rng.choice([2.0, 3.0, 4.0]))
        pre_op = current
        try:
            current = graph.add_op(op_name, [current], config=config)
        except (ValueError, KeyError):
            return node_id  # Bail on shape error
        # Auto-wrap REQUIRES_RESIDUAL_BYPASS ops with add(input, gated)
        if op_name in REQUIRES_RESIDUAL_BYPASS:
            current = graph.add_op("add", [pre_op, current])
        prev_op = op_name
    # Record motif usage for analytics feedback loop
    if current != node_id:
        graph.metadata.setdefault("motifs_used", []).append(motif.name)
    return current


# ── Template Factories ──────────────────────────────────────────────
#
# These cover the three most common template patterns, eliminating
# copy-paste boilerplate across _templates_*.py.


def _tpl_norm_op_residual(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
    *,
    op_name: str,
    op_config: Optional[dict] = None,
    post_norm: bool = False,
) -> int:
    """Factory: norm → op → fix_dim → add(input).

    Covers ~13 templates that apply a single op with residual connection.
    Falls back to tpl_residual_block on error.
    """
    from ._templates_core import tpl_residual_block

    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id
    try:
        processed = graph.add_op(op_name, [normed], config=op_config or {})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)
    if post_norm:
        try:
            processed = graph.add_op("rmsnorm", [processed])
        except (ValueError, KeyError):
            pass
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def _tpl_norm_dual_op_residual(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
    *,
    merge_op: str,
    path_a_config: Optional[dict] = None,
    path_b_config: Optional[dict] = None,
) -> int:
    """Factory: norm → proj_a → proj_b → merge_op → fix_dim → add(input).

    Covers ~8 binary-op templates (matmul, gated product, cosine, tropical, etc.).
    Falls back to tpl_residual_block on error.
    """
    from ._templates_core import tpl_residual_block

    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id
    try:
        proj_a = graph.add_op(
            "linear_proj", [normed], config=path_a_config or {"out_dim": D}
        )
        proj_b = graph.add_op(
            "linear_proj", [normed], config=path_b_config or {"out_dim": D}
        )
        merged = graph.add_op(merge_op, [proj_a, proj_b])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)
    projected = _fix_dim(graph, merged)
    try:
        return graph.add_op("add", [input_id, projected])
    except ValueError:
        return projected


def _tpl_norm_op_motif_residual(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
    *,
    op_name: str,
    op_config: Optional[dict] = None,
    motif_classes: Tuple[str, ...] = _FFN_CLASSES,
) -> int:
    """Factory: norm → op → proj → motif_slot → fix_dim → add(input).

    Covers ~6 templates that apply a fixed op then a motif slot
    (integral_kernel, windowed_attention, local_attention, state_space, etc.).
    Falls back to tpl_residual_block on error.
    """
    from ._templates_core import tpl_residual_block

    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id
    try:
        mixed = graph.add_op(op_name, [normed], config=op_config or {})
        projected = graph.add_op("linear_proj", [mixed], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)
    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, motif_classes, weights
    )
    processed = _instantiate_motif(graph, projected, ffn, rng) if ffn else projected
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed
