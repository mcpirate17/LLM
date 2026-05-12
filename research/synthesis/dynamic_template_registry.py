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
)
from .graph import ComputationGraph
from .primitives import PRIMITIVE_REGISTRY, get_wiring_rule


DEFAULT_DYNAMIC_TEMPLATE_CANDIDATE_PATH = Path(
    "research/notes/validated_template_candidates.json"
)
_MAX_SCORE_WEIGHT = 100.0
_MIN_SCORE_WEIGHT = 0.05
_MAX_EFFECTIVE_WEIGHT = 8.0


@dataclass(frozen=True, slots=True)
class DynamicTemplateCandidate:
    """Validated candidate descriptor that can be lowered as a template block."""

    template_id: str
    display_name: str
    chain: tuple[str, ...]
    weight: float
    source_path: str
    source: str = "validated_template_candidates"
    evidence: Mapping[str, Any] = field(default_factory=dict)
    validation: Mapping[str, Any] = field(default_factory=dict)


def load_dynamic_template_candidates(
    path: str | Path = DEFAULT_DYNAMIC_TEMPLATE_CANDIDATE_PATH,
    *,
    max_candidates: int = 32,
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
                source_path=candidate.source_path,
                source=candidate.source,
                evidence=candidate.evidence,
                validation=candidate.validation,
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
    graph.metadata.setdefault("dynamic_templates_used", []).append(
        {
            "template_id": name,
            "display_name": candidate.display_name,
            "chain": list(candidate.chain),
            "weight": float(candidate.weight),
            "source": candidate.source,
            "source_path": candidate.source_path,
        }
    )
    prev_template = graph.metadata.get("_active_template")
    prev_slot_counter = graph.metadata.get("_active_template_slot_counter")
    prev_template_instance = graph.metadata.get("_active_template_instance")
    graph.metadata["_active_template"] = name
    graph.metadata["_active_template_slot_counter"] = 0
    graph.metadata["_active_template_instance"] = template_instance

    try:
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
            record_template_slot_binding(
                graph,
                template_name=name,
                template_instance=template_instance,
                slot_index=index,
                slot_key=f"{name}[{template_instance}].step{index}",
                slot_classes=("dynamic_step",),
                selected_name=op_name,
                selected_class=(
                    f"dynamic_op_arity{PRIMITIVE_REGISTRY[op_name].n_inputs}"
                ),
                input_node_id=current,
            )
        cur_dim = graph.nodes[current].output_shape.dim
        if cur_dim != graph.model_dim:
            fix_op = (
                "linear_proj_down" if cur_dim > graph.model_dim else "linear_proj_up"
            )
            current = template_add_op(
                graph,
                fix_op,
                [current],
                {"out_dim": graph.model_dim},
                context=f"{name}.fix_dim",
            )
        return current
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
    require_validated: bool,
) -> DynamicTemplateCandidate | None:
    raw_validation = raw.get("validation")
    validation = raw_validation if isinstance(raw_validation, Mapping) else {}
    if require_validated and not _candidate_is_validated(validation):
        return None

    chain = _coerce_supported_chain(raw.get("chain"))
    if not chain:
        return None

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
        source_path=source_path,
        evidence=evidence,
        validation=dict(validation),
    )


def _candidate_is_validated(validation: Mapping[str, Any]) -> bool:
    return bool(
        validation.get("validate_passed")
        and validation.get("compile_passed")
        and validation.get("backward_passed")
    )


def _coerce_supported_chain(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        return ()
    chain: list[str] = []
    for raw_op in value:
        op_name = str(raw_op)
        prim = PRIMITIVE_REGISTRY.get(op_name)
        if prim is None or prim.n_inputs not in (1, 2):
            return ()
        chain.append(op_name)
    if chain and _op_requires_restricted_consumer(chain[-1]):
        return ()
    return tuple(chain)


def _op_requires_restricted_consumer(op_name: str) -> bool:
    """Return True when an op cannot safely be a reusable template tail."""
    rule = get_wiring_rule(op_name)
    return bool(rule and rule.get("valid_consumers"))


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
    )
    return {key: raw[key] for key in keys if key in raw}


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
    clamped_strength = max(0.0, float(strength))
    raw_scores = [
        max(_MIN_SCORE_WEIGHT, min(_MAX_SCORE_WEIGHT, float(candidate.weight)))
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
