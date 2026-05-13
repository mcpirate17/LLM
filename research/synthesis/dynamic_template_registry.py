"""Descriptor-backed dynamic template candidates for graph generation.

This module intentionally does not mutate the global ``TEMPLATES`` registry.
It loads pre-validated mined-chain descriptors, filters them against the live
primitive registry, and lowers a selected descriptor directly into a graph.
The lowering topology mirrors ``meta_analysis.template_validator`` so offline
validation evidence matches online generation behavior.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from ._template_helpers import (
    TemplateBuildError,
    record_template_slot_binding,
    template_add_op,
    template_add_residual,
)
from .component_rules import (
    ComponentRuleConfig,
    DEFAULT_MIN_LOWERED_OPS,
    estimated_chain_lowered_op_count,
    validate_component_op_chain,
)
from .graph import ComputationGraph
from .primitives import PRIMITIVE_REGISTRY


DEFAULT_DYNAMIC_TEMPLATE_CANDIDATE_PATH = Path(
    "research/notes/dynamic_component_candidates.json"
)
DEFAULT_DYNAMIC_TEMPLATE_MIN_LOWERED_OPS = DEFAULT_MIN_LOWERED_OPS
_MAX_SCORE_WEIGHT = 100.0
_MIN_SCORE_WEIGHT = 0.05
_MAX_EFFECTIVE_WEIGHT = 8.0
_LOWERING_SELECTION_MULTIPLIERS = {
    "trunk_sidecar_merge_v1": 1.10,
    "mixer_sidecar_restore_v1": 1.10,
    "router_lane_blend_v1": 0.75,
    "rmsnorm_chain_with_binary_skip": 1.10,
}


@dataclass(frozen=True, slots=True)
class DynamicTemplateCandidate:
    """Validated candidate descriptor that can be lowered as a template block."""

    template_id: str
    display_name: str
    chain: tuple[str, ...]
    weight: float
    lowered_op_count: int
    source_path: str
    source: str = "validated_template_candidates"
    evidence: Mapping[str, Any] = field(default_factory=dict)
    validation: Mapping[str, Any] = field(default_factory=dict)
    component_descriptor: Mapping[str, Any] = field(default_factory=dict)


def load_dynamic_template_candidates(
    path: str | Path = DEFAULT_DYNAMIC_TEMPLATE_CANDIDATE_PATH,
    *,
    max_candidates: int = 32,
    min_lowered_ops: int = DEFAULT_DYNAMIC_TEMPLATE_MIN_LOWERED_OPS,
    require_validated: bool = True,
) -> tuple[DynamicTemplateCandidate, ...]:
    """Load validated dynamic template candidates from a JSON artifact.

    Invalid, unknown, or unsupported chains are skipped. The result is sorted
    by evidence weight and capped so opt-in generation cannot accidentally turn
    a large mining artifact into a hot-path scan.
    """
    artifact = Path(path)
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()

    raw_candidates = _candidate_records(payload)
    candidates: list[DynamicTemplateCandidate] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_candidates):
        candidate = _coerce_candidate(
            raw,
            index=index,
            source_path=str(artifact),
            min_lowered_ops=min_lowered_ops,
            require_validated=require_validated,
        )
        if candidate is None:
            continue
        template_id = candidate.template_id
        if template_id in seen_ids:
            template_id = f"{template_id}_{len(seen_ids):04d}"
            candidate = DynamicTemplateCandidate(
                template_id=template_id,
                display_name=candidate.display_name,
                chain=candidate.chain,
                weight=candidate.weight,
                lowered_op_count=candidate.lowered_op_count,
                source_path=candidate.source_path,
                source=candidate.source,
                evidence=candidate.evidence,
                validation=candidate.validation,
                component_descriptor=candidate.component_descriptor,
            )
        seen_ids.add(template_id)
        candidates.append(candidate)

    candidates.sort(key=lambda item: item.weight, reverse=True)
    limit = max(0, int(max_candidates))
    if limit <= 0:
        return ()
    return tuple(candidates[:limit])


def choose_dynamic_template_candidate(
    rng: random.Random,
    candidates: Sequence[DynamicTemplateCandidate],
    *,
    strength: float = 1.0,
) -> DynamicTemplateCandidate:
    """Choose a dynamic candidate with bounded evidence weighting."""
    if not candidates:
        raise ValueError("no dynamic template candidates available")
    if len(candidates) == 1:
        return candidates[0]

    weights = _candidate_selection_weights(candidates, strength=strength)
    return rng.choices(list(candidates), weights=weights, k=1)[0]


def apply_dynamic_template_candidate(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    candidate: DynamicTemplateCandidate,
) -> int:
    """Lower a dynamic candidate into ``graph`` and return its tail node id."""
    del rng  # descriptor lowering is deterministic; selection owns randomness.
    name = candidate.template_id
    prev_next_id = graph._next_id
    prev_output_id = graph._output_node_id
    prev_ir_version = graph._ir_version
    prev_metadata = copy.deepcopy(graph.metadata)

    graph.metadata.setdefault("templates_used", []).append(name)
    template_instance = len(graph.metadata.get("templates_used", [])) - 1
    usage = _dynamic_usage_record(candidate)
    graph.metadata.setdefault("dynamic_templates_used", []).append(
        _dynamic_template_usage_record(usage)
    )
    graph.metadata.setdefault("dynamic_components_used", []).append(usage)
    lowering = _candidate_lowering(candidate)
    prev_template = graph.metadata.get("_active_template")
    prev_slot_counter = graph.metadata.get("_active_template_slot_counter")
    prev_template_instance = graph.metadata.get("_active_template_instance")
    graph.metadata["_active_template"] = name
    graph.metadata["_active_template_slot_counter"] = 0
    graph.metadata["_active_template_instance"] = template_instance

    try:
        if lowering == "router_lane_blend_v1":
            return _apply_router_lane_blend_dynamic_candidate(
                graph,
                input_id,
                candidate,
                template_instance=template_instance,
            )
        if lowering in {"trunk_sidecar_merge_v1", "mixer_sidecar_restore_v1"}:
            return _apply_trunk_sidecar_dynamic_candidate(
                graph,
                input_id,
                candidate,
                template_instance=template_instance,
            )
        return _apply_linear_dynamic_candidate(
            graph,
            input_id,
            candidate,
            template_instance=template_instance,
        )
    except Exception:
        for nid in range(prev_next_id, graph._next_id):
            graph.nodes.pop(nid, None)
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


def _dynamic_usage_record(candidate: DynamicTemplateCandidate) -> dict[str, Any]:
    descriptor = dict(candidate.component_descriptor)
    return {
        "template_id": candidate.template_id,
        "component_id": str(descriptor.get("component_id") or candidate.template_id),
        "display_name": candidate.display_name,
        "chain": list(candidate.chain),
        "weight": float(candidate.weight),
        "lowered_op_count": int(candidate.lowered_op_count),
        "source": candidate.source,
        "source_path": candidate.source_path,
        "lowering": str(descriptor.get("lowering") or "rmsnorm_chain_with_binary_skip"),
        "component_descriptor": descriptor,
    }


def _dynamic_template_usage_record(usage: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "template_id": str(usage.get("template_id") or ""),
        "display_name": str(usage.get("display_name") or ""),
        "chain": list(usage.get("chain") or ()),
        "weight": float(usage.get("weight") or 0.0),
        "lowered_op_count": int(usage.get("lowered_op_count") or 0),
        "source": str(usage.get("source") or ""),
        "source_path": str(usage.get("source_path") or ""),
        "component_id": str(usage.get("component_id") or ""),
        "lowering": str(usage.get("lowering") or ""),
    }


def _apply_linear_dynamic_candidate(
    graph: ComputationGraph,
    input_id: int,
    candidate: DynamicTemplateCandidate,
    *,
    template_instance: int,
) -> int:
    name = candidate.template_id
    current = template_add_op(
        graph,
        "rmsnorm",
        [input_id],
        context=f"{name}.input_norm",
    )
    prev_snapshot = input_id
    for index, op_name in enumerate(candidate.chain):
        current, prev_snapshot = _add_dynamic_chain_op(
            graph,
            op_name,
            current,
            prev_snapshot,
            context=f"{name}.step{index}.{op_name}",
        )
        _record_dynamic_slot(
            graph,
            candidate,
            template_instance=template_instance,
            slot_index=index,
            input_node_id=current,
        )
    return _fix_dynamic_dim(graph, current, context=f"{name}.fix_dim")


def _apply_trunk_sidecar_dynamic_candidate(
    graph: ComputationGraph,
    input_id: int,
    candidate: DynamicTemplateCandidate,
    *,
    template_instance: int,
) -> int:
    name = candidate.template_id
    plan = _candidate_branch_plan(candidate)
    trunk_indices = _branch_indices(plan, "trunk_indices", candidate)
    sidecar_indices = _branch_indices(plan, "sidecar_indices", candidate)

    branch_input = template_add_op(
        graph,
        "rmsnorm",
        [input_id],
        context=f"{name}.input_norm",
    )
    trunk = _apply_dynamic_branch(
        graph,
        candidate,
        indices=trunk_indices,
        branch_input=branch_input,
        template_instance=template_instance,
        branch_name="trunk",
    )
    sidecar = _apply_dynamic_branch(
        graph,
        candidate,
        indices=sidecar_indices,
        branch_input=branch_input,
        template_instance=template_instance,
        branch_name="sidecar",
    )
    trunk = _fix_dynamic_dim(graph, trunk, context=f"{name}.trunk_fix_dim")
    sidecar = _fix_dynamic_dim(graph, sidecar, context=f"{name}.sidecar_fix_dim")
    merge_op = str(plan.get("merge_op") or "add")
    if merge_op != "add":
        raise TemplateBuildError(f"{name}: unsupported dynamic branch merge {merge_op}")
    merged = template_add_op(
        graph,
        "add",
        [trunk, sidecar],
        context=f"{name}.branch_merge",
    )
    if bool(plan.get("post_merge_norm", True)):
        merged = template_add_op(
            graph,
            "rmsnorm",
            [merged],
            context=f"{name}.branch_merge_norm",
        )
    if bool(plan.get("residual_output", True)):
        return template_add_residual(
            graph,
            input_id,
            merged,
            context=f"{name}.output_residual",
        )
    return _fix_dynamic_dim(graph, merged, context=f"{name}.output_fix_dim")


def _apply_router_lane_blend_dynamic_candidate(
    graph: ComputationGraph,
    input_id: int,
    candidate: DynamicTemplateCandidate,
    *,
    template_instance: int,
) -> int:
    name = candidate.template_id
    plan = _candidate_branch_plan(candidate)
    branch_input = template_add_op(
        graph,
        "rmsnorm",
        [input_id],
        context=f"{name}.input_norm",
    )
    value_index, matmul_index, score_index, route_index, gate_index = (
        _router_lane_indices(plan, candidate)
    )
    value = _add_router_lane_op(
        graph,
        candidate,
        template_instance=template_instance,
        slot_index=value_index,
        inputs=[branch_input],
        context=f"{name}.router.value_project",
    )
    scores = _add_router_lane_op(
        graph,
        candidate,
        template_instance=template_instance,
        slot_index=matmul_index,
        inputs=[value, value],
        context=f"{name}.router.score_matmul",
    )
    score_projected = _add_router_lane_op(
        graph,
        candidate,
        template_instance=template_instance,
        slot_index=score_index,
        inputs=[scores],
        context=f"{name}.router.score_project",
    )
    routed = _add_router_lane_op(
        graph,
        candidate,
        template_instance=template_instance,
        slot_index=route_index,
        inputs=[value, score_projected],
        context=f"{name}.router.route",
    )
    gated = _add_router_lane_op(
        graph,
        candidate,
        template_instance=template_instance,
        slot_index=gate_index,
        inputs=[routed],
        context=f"{name}.router.gate",
    )
    return _finish_router_lane_blend(
        graph,
        input_id=input_id,
        value=value,
        gated=gated,
        plan=plan,
        name=name,
    )


def _router_lane_indices(
    plan: Mapping[str, Any],
    candidate: DynamicTemplateCandidate,
) -> tuple[int, int, int, int, int]:
    return (
        _branch_index(plan, "value_project_index", candidate),
        _branch_index(plan, "matmul_index", candidate),
        _branch_index(plan, "score_project_index", candidate),
        _branch_index(plan, "route_index", candidate),
        _branch_index(plan, "gate_index", candidate),
    )


def _add_router_lane_op(
    graph: ComputationGraph,
    candidate: DynamicTemplateCandidate,
    *,
    template_instance: int,
    slot_index: int,
    inputs: list[int],
    context: str,
) -> int:
    node_id = template_add_op(
        graph, candidate.chain[slot_index], inputs, context=context
    )
    _record_dynamic_slot(
        graph,
        candidate,
        template_instance=template_instance,
        slot_index=slot_index,
        input_node_id=node_id,
        branch_name="router",
    )
    return node_id


def _finish_router_lane_blend(
    graph: ComputationGraph,
    *,
    input_id: int,
    value: int,
    gated: int,
    plan: Mapping[str, Any],
    name: str,
) -> int:
    blend_op = str(plan.get("blend_op") or "gated_lane_blend")
    blended = template_add_op(
        graph,
        blend_op,
        [gated],
        {"n_lanes": 2},
        context=f"{name}.router.blend",
    )
    value = _fix_dynamic_dim(graph, value, context=f"{name}.router.value_fix_dim")
    blended = _fix_dynamic_dim(graph, blended, context=f"{name}.router.blend_fix_dim")
    merged = template_add_op(
        graph,
        "add",
        [value, blended],
        context=f"{name}.router.merge",
    )
    if bool(plan.get("post_merge_norm", True)):
        merged = template_add_op(
            graph,
            "rmsnorm",
            [merged],
            context=f"{name}.router.merge_norm",
        )
    if bool(plan.get("residual_output", True)):
        return template_add_residual(
            graph,
            input_id,
            merged,
            context=f"{name}.output_residual",
        )
    return _fix_dynamic_dim(graph, merged, context=f"{name}.output_fix_dim")


def _apply_dynamic_branch(
    graph: ComputationGraph,
    candidate: DynamicTemplateCandidate,
    *,
    indices: tuple[int, ...],
    branch_input: int,
    template_instance: int,
    branch_name: str,
) -> int:
    current = branch_input
    prev_snapshot = branch_input
    for index in indices:
        op_name = candidate.chain[index]
        current, prev_snapshot = _add_dynamic_chain_op(
            graph,
            op_name,
            current,
            prev_snapshot,
            context=f"{candidate.template_id}.{branch_name}.step{index}.{op_name}",
        )
        _record_dynamic_slot(
            graph,
            candidate,
            template_instance=template_instance,
            slot_index=index,
            input_node_id=current,
            branch_name=branch_name,
        )
    return current


def _record_dynamic_slot(
    graph: ComputationGraph,
    candidate: DynamicTemplateCandidate,
    *,
    template_instance: int,
    slot_index: int,
    input_node_id: int,
    branch_name: str | None = None,
) -> None:
    op_name = candidate.chain[slot_index]
    slot_suffix = (
        f"{branch_name}.step{slot_index}" if branch_name else f"step{slot_index}"
    )
    record_template_slot_binding(
        graph,
        template_name=candidate.template_id,
        template_instance=template_instance,
        slot_index=slot_index,
        slot_key=f"{candidate.template_id}[{template_instance}].{slot_suffix}",
        slot_classes=_candidate_slot_classes(candidate, slot_index),
        selected_name=op_name,
        selected_class=f"dynamic_op_arity{PRIMITIVE_REGISTRY[op_name].n_inputs}",
        input_node_id=input_node_id,
    )


def _fix_dynamic_dim(graph: ComputationGraph, current: int, *, context: str) -> int:
    cur_dim = graph.nodes[current].output_shape.dim
    if cur_dim == graph.model_dim:
        return current
    fix_op = "linear_proj_down" if cur_dim > graph.model_dim else "linear_proj_up"
    return template_add_op(
        graph,
        fix_op,
        [current],
        {"out_dim": graph.model_dim},
        context=context,
    )


def _candidate_lowering(candidate: DynamicTemplateCandidate) -> str:
    descriptor = candidate.component_descriptor
    if not isinstance(descriptor, Mapping):
        return "rmsnorm_chain_with_binary_skip"
    return str(descriptor.get("lowering") or "rmsnorm_chain_with_binary_skip")


def _candidate_branch_plan(candidate: DynamicTemplateCandidate) -> Mapping[str, Any]:
    descriptor = candidate.component_descriptor
    plan = descriptor.get("branch_plan") if isinstance(descriptor, Mapping) else None
    if not isinstance(plan, Mapping):
        raise TemplateBuildError(f"{candidate.template_id}: missing branch_plan")
    return plan


def _branch_indices(
    plan: Mapping[str, Any],
    key: str,
    candidate: DynamicTemplateCandidate,
) -> tuple[int, ...]:
    raw = plan.get(key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TemplateBuildError(f"{candidate.template_id}: invalid {key}")
    out: list[int] = []
    for value in raw:
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise TemplateBuildError(f"{candidate.template_id}: invalid {key}") from exc
        if index < 0 or index >= len(candidate.chain):
            raise TemplateBuildError(f"{candidate.template_id}: {key} out of range")
        out.append(index)
    if not out:
        raise TemplateBuildError(f"{candidate.template_id}: empty {key}")
    return tuple(out)


def _branch_index(
    plan: Mapping[str, Any],
    key: str,
    candidate: DynamicTemplateCandidate,
) -> int:
    try:
        index = int(plan.get(key))
    except (TypeError, ValueError) as exc:
        raise TemplateBuildError(f"{candidate.template_id}: invalid {key}") from exc
    if index < 0 or index >= len(candidate.chain):
        raise TemplateBuildError(f"{candidate.template_id}: {key} out of range")
    return index


def _candidate_records(payload: Any) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    ready = payload.get("ready_for_registration")
    if isinstance(ready, list):
        return [item for item in ready if isinstance(item, Mapping)]
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        return [item for item in candidates if isinstance(item, Mapping)]
    return []


def _coerce_candidate(
    raw: Mapping[str, Any],
    *,
    index: int,
    source_path: str,
    min_lowered_ops: int,
    require_validated: bool,
) -> DynamicTemplateCandidate | None:
    raw_validation = raw.get("validation")
    validation = raw_validation if isinstance(raw_validation, Mapping) else {}
    if require_validated and not _candidate_is_validated(validation):
        return None

    chain = _coerce_supported_chain(raw.get("chain"), min_lowered_ops=min_lowered_ops)
    if not chain:
        return None
    lowered_op_count = estimated_chain_lowered_op_count(chain)

    display_name = str(raw.get("proposed_template_name") or "dynamic_template").strip()
    if not display_name:
        display_name = "dynamic_template"
    template_id = _unique_template_id(display_name, chain, index)
    evidence = _candidate_evidence(raw)
    return DynamicTemplateCandidate(
        template_id=template_id,
        display_name=display_name,
        chain=chain,
        weight=_candidate_weight(raw),
        lowered_op_count=lowered_op_count,
        source_path=source_path,
        source=str(raw.get("source") or "dynamic_component_candidates"),
        evidence=evidence,
        validation=dict(validation),
        component_descriptor=_candidate_component_descriptor(raw),
    )


def _candidate_is_validated(validation: Mapping[str, Any]) -> bool:
    return bool(
        validation.get("validate_passed")
        and validation.get("compile_passed")
        and validation.get("backward_passed")
    )


def _coerce_supported_chain(value: Any, *, min_lowered_ops: int) -> tuple[str, ...]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        return ()
    chain = tuple(str(raw_op) for raw_op in value)
    violations = validate_component_op_chain(
        chain,
        config=ComponentRuleConfig(
            min_lowered_ops=int(min_lowered_ops),
            min_distinct_roles=1,
        ),
    )
    if violations:
        return ()
    return chain


def _unique_template_id(display_name: str, chain: tuple[str, ...], index: int) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", display_name.strip()).strip("_").lower()
    if not slug:
        slug = "dynamic_template"
    digest = hashlib.blake2b(
        f"{index}|{'|'.join(chain)}".encode("utf-8"), digest_size=4
    ).hexdigest()
    return f"dynamic_{slug}_{index:04d}_{digest}"


def _candidate_evidence(raw: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "n_total",
        "pass_rate",
        "lift_vs_cohort",
        "promotion_score",
        "cohort_pass_rate",
        "chain_length",
        "lowered_op_count",
        "mean_loss_ratio",
    )
    return {key: raw[key] for key in keys if key in raw}


def _candidate_component_descriptor(raw: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = raw.get("component_descriptor")
    return dict(descriptor) if isinstance(descriptor, Mapping) else {}


def _candidate_slot_classes(
    candidate: DynamicTemplateCandidate,
    index: int,
) -> tuple[str, ...]:
    descriptor = candidate.component_descriptor
    slot_plan = descriptor.get("slot_plan") if isinstance(descriptor, Mapping) else None
    if isinstance(slot_plan, Sequence) and not isinstance(slot_plan, (str, bytes)):
        for item in slot_plan:
            if not isinstance(item, Mapping):
                continue
            try:
                slot_index = int(item.get("slot_index"))
            except (TypeError, ValueError):
                continue
            if slot_index != index:
                continue
            classes = item.get("slot_classes")
            if isinstance(classes, Sequence) and not isinstance(classes, (str, bytes)):
                out = tuple(str(cls) for cls in classes if str(cls))
                if out:
                    return out
    return ("dynamic_step",)


def _candidate_weight(raw: Mapping[str, Any]) -> float:
    explicit = _safe_float(raw.get("promotion_score"))
    if explicit is not None and explicit > 0.0:
        return max(_MIN_SCORE_WEIGHT, min(_MAX_SCORE_WEIGHT, explicit))

    n_total = _safe_float(raw.get("n_total")) or 1.0
    pass_rate = _safe_float(raw.get("pass_rate")) or 1.0
    lift = _safe_float(raw.get("lift_vs_cohort")) or 1.0
    score = math.sqrt(max(1.0, n_total)) * max(0.0, pass_rate) * max(0.0, lift)
    return max(_MIN_SCORE_WEIGHT, min(_MAX_SCORE_WEIGHT, score))


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _candidate_selection_weights(
    candidates: Sequence[DynamicTemplateCandidate],
    *,
    strength: float,
) -> list[float]:
    # guardrail: allow-complexity - bounded candidate pool (default max 32).
    clamped_strength = max(0.0, float(strength))
    raw_scores = [
        max(
            _MIN_SCORE_WEIGHT,
            min(
                _MAX_SCORE_WEIGHT,
                float(candidate.weight) * _lowering_selection_multiplier(candidate),
            ),
        )
        for candidate in candidates
    ]
    if clamped_strength == 0.0:
        return [1.0 for _ in raw_scores]

    baseline = max(_MIN_SCORE_WEIGHT, float(median(raw_scores)))
    weights: list[float] = []
    for score in raw_scores:
        normalized = score / baseline
        effective = 1.0 + (normalized - 1.0) * clamped_strength
        weights.append(max(_MIN_SCORE_WEIGHT, min(_MAX_EFFECTIVE_WEIGHT, effective)))
    return weights


def _lowering_selection_multiplier(candidate: DynamicTemplateCandidate) -> float:
    descriptor = candidate.component_descriptor
    lowering = (
        str(descriptor.get("lowering") or "") if isinstance(descriptor, Mapping) else ""
    )
    return float(_LOWERING_SELECTION_MULTIPLIERS.get(lowering, 1.0))


def _add_dynamic_chain_op(
    graph: ComputationGraph,
    op_name: str,
    current: int,
    prev_snapshot: int,
    *,
    context: str,
) -> tuple[int, int]:
    prim = PRIMITIVE_REGISTRY.get(op_name)
    if prim is None:
        raise TemplateBuildError(f"{context}: unknown op {op_name}")
    if prim.n_inputs == 1:
        return template_add_op(graph, op_name, [current], context=context), current
    if prim.n_inputs == 2:
        return (
            template_add_op(graph, op_name, [current, prev_snapshot], context=context),
            current,
        )
    raise TemplateBuildError(
        f"{context}: unsupported dynamic chain arity {prim.n_inputs}"
    )
