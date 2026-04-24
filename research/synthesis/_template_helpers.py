"""Shared helpers for template implementations.

All template modules import from here — types, class groupings, motif
picking, and the motif instantiation engine.
"""

from __future__ import annotations

import random
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from .graph import ComputationGraph
else:
    ComputationGraph = Any
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
from ._selection_utils import context_pair_allowed
from .primitives import (
    AlgebraicType,
    PRIMITIVE_REGISTRY,
    REQUIRES_RESIDUAL_BYPASS,
    algebraic_types_compatible,
)
from .context_rules import motif_allowed_in_template as _motif_allowed_in_template

# Type alias for motif weight dicts passed from judgment engine
MotifWeights = Optional[Dict[str, float]]

# Type for template callables
TemplateFn = Callable[
    [ComputationGraph, int, random.Random, MotifWeights],
    int,
]


class TemplateBuildError(ValueError):
    """Raised when a template cannot be lowered into a valid graph."""


_TEMPLATE_PRE_NORM_OPS = frozenset(
    op_name for op_name, rules in MATH_SPACE_RULES.items() if "must_precede" in rules
)


def _template_inputs_with_required_pre_norm(
    graph: ComputationGraph,
    op_name: str,
    input_ids: list[int],
    context: str,
) -> list[int]:
    required = MATH_SPACE_RULES[op_name]["must_precede"]
    for input_id in input_ids:
        parent = graph.nodes.get(input_id)
        if parent is not None and not parent.is_input and parent.op_name in required:
            return input_ids

    if not input_ids:
        return input_ids

    normalized = list(input_ids)
    try:
        normalized[0] = graph.add_op("rmsnorm", [normalized[0]])
    except (ValueError, KeyError) as exc:
        raise TemplateBuildError(f"{context}: failed to add required pre-norm") from exc
    return normalized


def template_add_op(
    graph: ComputationGraph,
    op_name: str,
    input_ids: list[int],
    config: Optional[Dict[str, object]] = None,
    *,
    context: str,
) -> int:
    """Add an op during template lowering and fail with template context."""
    if op_name in _TEMPLATE_PRE_NORM_OPS:
        input_ids = _template_inputs_with_required_pre_norm(
            graph,
            op_name,
            input_ids,
            context,
        )
    try:
        return graph.add_op(op_name, input_ids, config=config)
    except (ValueError, KeyError) as exc:
        raise TemplateBuildError(f"{context}: failed to add {op_name}") from exc


def template_add_residual(
    graph: ComputationGraph,
    skip_id: int,
    value_id: int,
    *,
    context: str,
) -> int:
    """Add a residual edge during template lowering and fail explicitly."""
    return template_add_op(graph, "add", [skip_id, value_id], context=context)


# ── Motif class groupings for slot constraints ──────────────────────

# Slots that accept any sequence mixer
_MIXER_CLASSES: Tuple[str, ...] = (
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_SSM,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_CHANNEL,
    MOTIF_CLASS_MATH_SPACE,
)

# Slots that guarantee attention (no 5-class lottery dilution)
_ATTENTION_ONLY_CLASSES: Tuple[str, ...] = (MOTIF_CLASS_ATTENTION,)

# Constrained FFN classes for attention templates where sparse/efficient
# outperform random FFN (empirical: 25% vs 10% S1 rate)
_SPARSE_FFN_CLASSES: Tuple[str, ...] = (
    MOTIF_CLASS_SPARSE,
    MOTIF_CLASS_EFFICIENT_PROJ,
    MOTIF_CLASS_GATE,
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
_MAX_CONTEXT_DEPTH = max(
    (op.min_layer_depth for op in PRIMITIVE_REGISTRY.values()),
    default=0,
)


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


def _node_depth_and_previous_op(
    graph: ComputationGraph,
    node_id: int,
) -> tuple[int, str | None]:
    current_node = graph.nodes[node_id]
    previous_op = None if current_node.is_input else current_node.op_name
    return current_node.depth, previous_op


def _motif_is_compatible_from_context(
    current_type: AlgebraicType,
    previous_op: str | None,
    depth: int,
    motif: Motif,
) -> bool:
    for step in motif.steps:
        step_op = PRIMITIVE_REGISTRY.get(step.op_name)
        if (
            step_op is None
            or step_op.n_inputs != 1
            or not algebraic_types_compatible(current_type, step_op.algebraic_type)
            or not context_pair_allowed(previous_op, step.op_name)
        ):
            return False
        # Reject motif if op requires deeper placement than current position
        if step_op.min_layer_depth > 0 and depth < step_op.min_layer_depth:
            return False
        current_type = step_op.algebraic_type
        previous_op = step.op_name
    return True


def _motif_is_compatible(graph: ComputationGraph, node_id: int, motif: Motif) -> bool:
    depth, previous_op = _node_depth_and_previous_op(graph, node_id)
    return _motif_is_compatible_from_context(
        _node_output_type(graph, node_id),
        previous_op,
        depth,
        motif,
    )


@lru_cache(maxsize=128)
def _motif_pool_for_classes(classes: Tuple[str, ...]) -> tuple[Motif, ...]:
    pool: list[Motif] = []
    for cls in classes:
        pool.extend(MOTIFS_BY_CLASS.get(cls, ()))
    return tuple(pool)


def _compatible_from_classes(
    graph: ComputationGraph,
    node_id: int,
    classes: Tuple[str, ...] | list[str],
) -> tuple[Motif, ...]:
    """Return motifs from *classes* that are compatible at *node_id*."""
    active_tpl = graph.metadata.get("_active_template")
    classes_tuple = tuple(classes)
    current_type = _node_output_type(graph, node_id)
    depth, previous_op = _node_depth_and_previous_op(graph, node_id)
    depth = min(depth, _MAX_CONTEXT_DEPTH)
    return _compatible_from_context(
        classes_tuple,
        current_type,
        previous_op,
        depth,
        active_tpl,
    )


@lru_cache(maxsize=2048)
def _compatible_base_from_context(
    classes: Tuple[str, ...],
    current_type: AlgebraicType,
    previous_op: str | None,
    depth: int,
) -> tuple[Motif, ...]:
    pool = _motif_pool_for_classes(classes)
    return tuple(
        m
        for m in pool
        if _motif_is_compatible_from_context(current_type, previous_op, depth, m)
    )


@lru_cache(maxsize=4096)
def _compatible_from_context(
    classes: Tuple[str, ...],
    current_type: AlgebraicType,
    previous_op: str | None,
    depth: int,
    active_tpl: str | None,
) -> tuple[Motif, ...]:
    base = _compatible_base_from_context(
        classes,
        current_type,
        previous_op,
        depth,
    )
    return tuple(m for m in base if _motif_allowed_in_template(m, active_tpl))


_SLOT_MOTIF_DENYLIST: dict[str, frozenset[str]] = {
    # Post-mask signal is fragile — routing motifs corrupt the masked stream.
    "depth_token_mask_block.slot1": frozenset(
        {
            "route_mod_topk",
            "route_lanes_block",
            "route_recursion_block",
            "route_speculative",
            "route_topk_gate",
            "route_topk_sparse",
        }
    ),
}

_SLOT_MOTIF_ALLOWLIST: dict[str, frozenset[str]] = {}

_SLOT_MOTIF_WEIGHT_MULTIPLIERS: dict[str, dict[str, float]] = {}


def _normalize_slot_key(slot_key: str) -> str:
    """Map instanceful telemetry keys to the canonical template.slotN form."""
    if "[" not in slot_key:
        return slot_key
    return re.sub(r"\[\d+\]", "", slot_key)


def get_slot_rule_summary() -> list[dict[str, object]]:
    """Return the configured slot compatibility rules in canonical form."""
    slot_keys = sorted(
        set(_SLOT_MOTIF_DENYLIST)
        | set(_SLOT_MOTIF_ALLOWLIST)
        | set(_SLOT_MOTIF_WEIGHT_MULTIPLIERS)
    )
    summary: list[dict[str, object]] = []
    for slot_key in slot_keys:
        multipliers = _SLOT_MOTIF_WEIGHT_MULTIPLIERS.get(slot_key, {})
        summary.append(
            {
                "slot_key": slot_key,
                "template_name": slot_key.split(".slot", 1)[0],
                "slot_index": int(slot_key.rsplit("slot", 1)[1]),
                "allowed_motifs": sorted(_SLOT_MOTIF_ALLOWLIST.get(slot_key, ())),
                "blocked_motifs": sorted(_SLOT_MOTIF_DENYLIST.get(slot_key, ())),
                "weight_multipliers": {
                    name: multipliers[name] for name in sorted(multipliers)
                },
            }
        )
    return summary


def _filter_slot_candidates(
    graph: ComputationGraph,
    candidates: Sequence[Motif],
    slot_key: str | None = None,
) -> Sequence[Motif]:
    """Drop motifs that are known-bad for the active template slot."""
    if slot_key is None:
        slot_key = _normalize_slot_key(_current_slot_key(graph))
    allowed = _SLOT_MOTIF_ALLOWLIST.get(slot_key)
    if allowed is not None:
        candidates = tuple(motif for motif in candidates if motif.name in allowed)
    denied_static = _SLOT_MOTIF_DENYLIST.get(slot_key)
    dynamic_denied = graph.metadata.get("_slot_motif_denylist", {})
    denied_dynamic = ()
    if isinstance(dynamic_denied, dict):
        denied_dynamic = dynamic_denied.get(slot_key, ()) or ()
    if denied_static or denied_dynamic:
        denied = (
            denied_static
            if denied_static and not denied_dynamic
            else frozenset((*tuple(denied_static or ()), *tuple(denied_dynamic)))
        )
        candidates = tuple(motif for motif in candidates if motif.name not in denied)
    return candidates


def _select_by_lift(candidates: Sequence[Motif], rng: random.Random) -> Motif:
    total = 0.0
    for motif in candidates:
        total += float(motif.lift)
    if total <= 0.0:
        return rng.choice(candidates)
    threshold = rng.random() * total
    for motif in candidates:
        threshold -= float(motif.lift)
        if threshold < 0.0:
            return motif
    return candidates[-1]


def _select_from_candidates(
    graph: ComputationGraph,
    candidates: Sequence[Motif],
    rng: random.Random,
    weights: MotifWeights,
    slot_key: str | None = None,
) -> Optional[Motif]:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if slot_key is None:
        slot_key = _normalize_slot_key(_current_slot_key(graph))
    slot_multipliers = _SLOT_MOTIF_WEIGHT_MULTIPLIERS.get(slot_key)
    dynamic_slot_multipliers = graph.metadata.get("_slot_motif_weight_multipliers", {})
    dynamic_multipliers = None
    if isinstance(dynamic_slot_multipliers, dict):
        dynamic_multipliers = dynamic_slot_multipliers.get(slot_key, {}) or None
    if weights is None and not slot_multipliers and not dynamic_multipliers:
        return _select_by_lift(candidates, rng)

    merged_multipliers = dict(slot_multipliers or {})
    if dynamic_multipliers:
        for motif_name, multiplier in dynamic_multipliers.items():
            try:
                merged_multipliers[str(motif_name)] = float(multiplier)
            except (TypeError, ValueError):
                continue
    candidate_weights = [
        (weights.get(m.name, m.lift) if weights else m.lift)
        * merged_multipliers.get(m.name, 1.0)
        for m in candidates
    ]
    return rng.choices(candidates, weights=candidate_weights, k=1)[0]


def _pick_compatible_motif(
    graph: ComputationGraph,
    node_id: int,
    rng: random.Random,
    motif_class_or_classes,
    weights: MotifWeights = None,
    *,
    wildcard_prob: Optional[float] = None,
) -> Optional[Motif]:
    """Pick a compatible motif from one or more classes.

    Args:
        motif_class_or_classes: A single class string or a tuple/list of classes.
    """
    if isinstance(motif_class_or_classes, str):
        classes: Tuple[str, ...] = (motif_class_or_classes,)
    else:
        classes = tuple(motif_class_or_classes)

    if wildcard_prob is None:
        wildcard_prob = graph.metadata.get("_wildcard_slot_prob", 0.0)
    is_wildcard = wildcard_prob > 0 and rng.random() < wildcard_prob

    # Slot adaptations: learned class expansions from DB (multi-class only)
    if len(classes) > 1:
        slot_adaptations = graph.metadata.get("_slot_adaptations")
        if slot_adaptations:
            slot_key = _normalize_slot_key(_current_slot_key(graph))
            extra_classes = slot_adaptations.get(slot_key, ())
            if extra_classes:
                classes = tuple(set(classes) | set(extra_classes))

    if is_wildcard:
        candidates = _compatible_from_classes(graph, node_id, _ALL_CLASSES)
    else:
        candidates = _compatible_from_classes(graph, node_id, classes)
        if not candidates and wildcard_prob > 0:
            candidates = _compatible_from_classes(graph, node_id, _ALL_CLASSES)
            is_wildcard = True
    slot_key = _normalize_slot_key(_current_slot_key(graph))
    candidates = _filter_slot_candidates(graph, candidates, slot_key=slot_key)

    # Graph-level wildcard breadcrumb. Per-slot `wildcard` is already on
    # _record_slot_usage; this aggregate makes it cheap for downstream
    # filtering to ask "was this graph touched by wildcard fallback at all?"
    if is_wildcard:
        graph.metadata["_template_wildcard_used"] = True
        wildcards = graph.metadata.setdefault("_template_wildcard_slot_keys", [])
        wildcards.append(_current_slot_key(graph))

    selected = _select_from_candidates(
        graph, candidates, rng, weights, slot_key=slot_key
    )
    _record_slot_usage(
        graph,
        node_id=node_id,
        slot_classes=classes,
        candidates=candidates,
        selected=selected,
        wildcard=is_wildcard,
    )
    return selected


# Backward-compat alias — callers using the old multi-class name still work.
_pick_compatible_motif_from_classes = _pick_compatible_motif


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
        "slot_key_canonical": f"{template_name}.slot{slot_index}",
        "slot_classes": class_list,
        "selected_motif": selected.name if selected else None,
        "selected_motif_class": selected.motif_class if selected else None,
        "candidate_count": len(candidates),
        "input_node_id": int(node_id),
        "wildcard": wildcard,
    }
    graph.metadata.setdefault("template_slot_usage", []).append(entry)


def record_template_slot_binding(
    graph: ComputationGraph,
    *,
    template_name: str,
    template_instance: int,
    slot_index: int,
    slot_key: str,
    slot_classes: Tuple[str, ...] | list[str],
    selected_name: str,
    selected_class: str,
    input_node_id: int,
) -> None:
    """Record a non-motif structural slot binding for observability."""
    entry = {
        "template_name": str(template_name),
        "template_instance": int(template_instance),
        "slot_index": int(slot_index),
        "slot_key": str(slot_key),
        "slot_key_canonical": _normalize_slot_key(str(slot_key)),
        "slot_classes": [str(cls) for cls in slot_classes],
        "selected_motif": str(selected_name),
        "selected_motif_class": str(selected_class),
        "candidate_count": 1,
        "input_node_id": int(input_node_id),
        "wildcard": False,
    }
    graph.metadata.setdefault("template_slot_usage", []).append(entry)


def _fix_dim(graph: ComputationGraph, node_id: int) -> int:
    """Add projection to fix dimension back to model_dim if needed.

    Uses linear_proj_down when current dim > model_dim and linear_proj_up when
    current dim < model_dim so reduced-rank trunks recover through the explicit
    up-projection path rather than the stale linear_proj shortcut.

    Also inserts a stabilizing norm after terminal depth_token_mask nodes.
    """
    op_name = graph.nodes[node_id].op_name
    if op_name == "depth_token_mask":
        try:
            node_id = graph.add_op("rmsnorm", [node_id])
        except ValueError as exc:
            raise TemplateBuildError(
                "Failed to stabilize depth_token_mask before dimension repair"
            ) from exc
    cur_dim = graph.nodes[node_id].output_shape.dim
    if cur_dim != graph.model_dim:
        op = "linear_proj_down" if cur_dim > graph.model_dim else "linear_proj_up"
        try:
            return graph.add_op(op, [node_id], config={"out_dim": graph.model_dim})
        except ValueError as exc:
            raise TemplateBuildError(
                f"Failed to restore model_dim={graph.model_dim} from dim={cur_dim}"
            ) from exc
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
        except (ValueError, KeyError) as exc:
            raise TemplateBuildError("Failed to add pre-motif transpose_sd") from exc

    motif = _pick_compatible_motif_from_classes(
        graph, current, rng, motif_classes, weights
    )
    result = _instantiate_motif(graph, current, motif, rng) if motif else current

    if use_shuffle:
        result = _fix_dim(graph, result)
        try:
            result = graph.add_op("transpose_sd", [result])
        except (ValueError, KeyError) as exc:
            raise TemplateBuildError("Failed to add post-motif transpose_sd") from exc

    return result


def _instantiate_motif(
    graph: ComputationGraph,
    node_id: int,
    motif: Motif,
    rng: random.Random,
) -> int:
    """Add a motif's ops to the graph starting from node_id.

    Returns the output node ID.
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
            raise TemplateBuildError(
                f"Motif '{motif.name}' step '{op_name}' is algebraically incompatible"
            )
        # Math-space safety: auto-insert rmsnorm if must_precede is unsatisfied
        ms_rules = MATH_SPACE_RULES.get(op_name)
        if ms_rules and "must_precede" in ms_rules:
            if prev_op not in ms_rules["must_precede"]:
                try:
                    current = graph.add_op("rmsnorm", [current])
                    prev_op = "rmsnorm"
                except (ValueError, KeyError) as exc:
                    raise TemplateBuildError(
                        f"Motif '{motif.name}' could not insert required rmsnorm"
                    ) from exc
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
                except ValueError as exc:
                    raise TemplateBuildError(
                        f"Motif '{motif.name}' could not restore reduced dim=1 to {D}"
                    ) from exc
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
        if (
            op_name in ("linear_proj_up", "linear_proj")
            and prev_op == "linear_proj_down"
        ):
            try:
                current = graph.add_op("rmsnorm", [current])
                prev_op = "rmsnorm"
            except ValueError as exc:
                raise TemplateBuildError(
                    f"Motif '{motif.name}' failed to stabilize after linear_proj_down"
                ) from exc
        pre_op = current
        try:
            current = graph.add_op(op_name, [current], config=config)
        except (ValueError, KeyError) as exc:
            raise TemplateBuildError(
                f"Motif '{motif.name}' failed on step '{op_name}'"
            ) from exc
        if op_name == "depth_token_mask":
            try:
                current = graph.add_op("rmsnorm", [current])
            except ValueError as exc:
                raise TemplateBuildError(
                    f"Motif '{motif.name}' failed to stabilize depth_token_mask"
                ) from exc
        # Auto-wrap REQUIRES_RESIDUAL_BYPASS ops with add(input, gated)
        if op_name in REQUIRES_RESIDUAL_BYPASS:
            try:
                current = graph.add_op("add", [pre_op, current])
            except ValueError as exc:
                raise TemplateBuildError(
                    f"Motif '{motif.name}' failed to add required residual bypass"
                ) from exc
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
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id
    try:
        processed = template_add_op(
            graph,
            op_name,
            [normed],
            config=op_config or {},
            context=f"template_op_residual.{op_name}",
        )
    except (ValueError, KeyError) as exc:
        raise TemplateBuildError(f"Template op '{op_name}' failed to lower") from exc
    if post_norm:
        try:
            processed = graph.add_op("rmsnorm", [processed])
        except (ValueError, KeyError) as exc:
            raise TemplateBuildError(
                f"Template op '{op_name}' failed to add post rmsnorm"
            ) from exc
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError as exc:
        raise TemplateBuildError(
            f"Template op '{op_name}' failed to add residual connection"
        ) from exc


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
        merged = template_add_op(
            graph,
            merge_op,
            [proj_a, proj_b],
            context=f"template_dual_op_residual.{merge_op}",
        )
    except (ValueError, KeyError) as exc:
        raise TemplateBuildError(
            f"Template merge op '{merge_op}' failed to lower"
        ) from exc
    projected = _fix_dim(graph, merged)
    try:
        return graph.add_op("add", [input_id, projected])
    except ValueError as exc:
        raise TemplateBuildError(
            f"Template merge op '{merge_op}' failed to add residual connection"
        ) from exc


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
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id
    try:
        mixed = template_add_op(
            graph,
            op_name,
            [normed],
            config=op_config or {},
            context=f"template_op_motif_residual.{op_name}",
        )
        projected = graph.add_op("linear_proj", [mixed], config={"out_dim": D})
    except (ValueError, KeyError) as exc:
        raise TemplateBuildError(
            f"Template op '{op_name}' failed before motif slot lowering"
        ) from exc
    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, motif_classes, weights
    )
    processed = _instantiate_motif(graph, projected, ffn, rng) if ffn else projected
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError as exc:
        raise TemplateBuildError(
            f"Template op '{op_name}' failed to add residual connection"
        ) from exc


def _tpl_attention_ffn_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
    *,
    attn_op: Optional[str] = None,
    attn_config: Optional[dict] = None,
    ffn_classes: Tuple[str, ...] = _FFN_CLASSES,
) -> int:
    """Factory: norm → attention → add → norm → FFN → add.

    Pre-norm transformer pattern with **forced attention** in the mixer slot.
    If attn_op is None, picks a random attention motif. If attn_op is a string,
    uses that specific attention op (e.g., 'latent_attention_compressor').
    Falls back to tpl_residual_block on error.
    """
    D = graph.model_dim

    # Attention sub-block
    norm1 = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    if attn_op is not None:
        try:
            mixed = template_add_op(
                graph,
                attn_op,
                [normed1],
                config=attn_config or {},
                context=f"attention_ffn_block.{attn_op}",
            )
            mixed = graph.add_op("linear_proj", [mixed], config={"out_dim": D})
        except (ValueError, KeyError) as exc:
            raise TemplateBuildError(
                f"Forced attention op '{attn_op}' failed to lower"
            ) from exc
    else:
        attn = _pick_compatible_motif(
            graph, normed1, rng, _ATTENTION_ONLY_CLASSES, weights
        )
        if attn:
            mixed = _instantiate_motif(graph, normed1, attn, rng)
        else:
            raise TemplateBuildError("No compatible attention motif available")
    mixed = _fix_dim(graph, mixed)

    try:
        mid = graph.add_op("add", [input_id, mixed])
    except ValueError as exc:
        raise TemplateBuildError(
            "Attention sub-block failed to add residual connection"
        ) from exc

    # FFN sub-block
    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid

    ffn = _pick_compatible_motif_from_classes(graph, normed2, rng, ffn_classes, weights)
    if ffn:
        ffned = _instantiate_motif(graph, normed2, ffn, rng)
    else:
        ffned = normed2
    ffned = _fix_dim(graph, ffned)

    try:
        return graph.add_op("add", [mid, ffned])
    except ValueError as exc:
        raise TemplateBuildError(
            "FFN sub-block failed to add residual connection"
        ) from exc
