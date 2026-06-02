"""Build validated dynamic component candidates from mined rule evidence.

This is the bridge between read-only historical mining and live dynamic
generation. It consumes ``mine_component_rules`` reports, applies cheap
component-structure rules, validates surviving chains with the existing
compile/forward/backward smoke validator, then writes a descriptor artifact
that ``dynamic_template_registry`` can load directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.meta_analysis.template_validator import (
    _finalize_compile_and_smoke,
    validate_chain,
)
from research.synthesis.component_rules import (
    ComponentRuleConfig,
    component_role_counts,
    estimated_chain_lowered_op_count,
    validate_component_op_chain,
)
from research.synthesis.component_rule_engine import (
    component_slot_plan,
    load_component_rule_set,
)
from research.synthesis.compiler import _compile_layer_module
from research.synthesis.dynamic_template_registry import (
    DynamicTemplateCandidate,
    apply_dynamic_template_candidate,
)
from research.synthesis.graph import ComputationGraph
from research.synthesis.op_roles import OpRole, get_role
from research.synthesis.validator import validate_graph


DEFAULT_INPUT = Path("research/reports/component_rule_mining_20260511_222730.json")
DEFAULT_OUTPUT = Path(
    "research/data/synthesis_candidates/dynamic_component_candidates.json"
)
PREFERRED_PAIR_MIN_PASS_RATE = 0.60
PREFERRED_PAIR_MIN_PASS_RATE_LIFT = -0.25
_BRANCH_LOWERINGS = frozenset(
    {
        "trunk_sidecar_merge_v1",
        "mixer_sidecar_restore_v1",
        "router_lane_blend_v1",
    }
)


def build_dynamic_component_candidates(
    *,
    mining_report_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path | None = DEFAULT_OUTPUT,
    max_candidates: int = 32,
    min_lowered_ops: int = 8,
    min_support: int = 8,
    min_pass_rate: float = 0.70,
    min_pass_rate_lift: float = 0.0,
    negative_pair_max_pass_rate: float = 0.50,
    negative_pair_max_lift: float = -0.20,
    model_dim: int = 64,
    run_smoke: bool = True,
    validate_candidates: bool = True,
) -> dict[str, Any]:
    """Return and optionally write a validated dynamic component artifact."""
    report_path, report, rows = _load_candidate_rows(mining_report_path)
    rule_set = load_component_rule_set()
    blocked_pairs = (
        _blocked_negative_pairs(
            report,
            max_pass_rate=float(negative_pair_max_pass_rate),
            max_lift=float(negative_pair_max_lift),
        )
        - rule_set.preferred_op_pairs
    )

    rule_config = ComponentRuleConfig(
        min_lowered_ops=int(min_lowered_ops),
        min_distinct_roles=2,
        rule_set=rule_set,
    )
    candidates, ready = _collect_candidates(
        rows=rows,
        report_path=report_path,
        rule_config=rule_config,
        preferred_pairs=rule_set.preferred_op_pairs,
        blocked_pairs=blocked_pairs,
        max_candidates=max(1, int(max_candidates)),
        min_support=int(min_support),
        min_pass_rate=float(min_pass_rate),
        min_pass_rate_lift=float(min_pass_rate_lift),
        model_dim=int(model_dim),
        run_smoke=bool(run_smoke),
        validate_candidates=bool(validate_candidates),
    )
    ready.sort(key=lambda item: float(item.get("promotion_score") or 0.0), reverse=True)
    payload = _candidate_artifact_payload(
        report_path=report_path,
        rows=rows,
        candidates=candidates,
        ready=ready,
        min_lowered_ops=int(min_lowered_ops),
        min_support=int(min_support),
        min_pass_rate=float(min_pass_rate),
        min_pass_rate_lift=float(min_pass_rate_lift),
        negative_pair_max_pass_rate=float(negative_pair_max_pass_rate),
        negative_pair_max_lift=float(negative_pair_max_lift),
        negative_pairs_blocked=len(blocked_pairs),
        model_dim=int(model_dim),
        run_smoke=bool(run_smoke),
        schema_versions=rule_set.schema_versions,
    )
    _write_candidate_artifact(payload, output_path)
    return payload


def _collect_candidates(
    *,
    rows: Sequence[Any],
    report_path: Path,
    rule_config: ComponentRuleConfig,
    preferred_pairs: frozenset[tuple[str, str]],
    blocked_pairs: frozenset[tuple[str, str]],
    max_candidates: int,
    min_support: int,
    min_pass_rate: float,
    min_pass_rate_lift: float,
    model_dim: int,
    run_smoke: bool,
    validate_candidates: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    ready: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        candidate = _candidate_from_report_row(
            row,
            report_path=report_path,
            rule_config=rule_config,
            preferred_pairs=preferred_pairs,
            blocked_pairs=blocked_pairs,
            candidate_index=len(candidates),
            seen=seen,
            min_support=min_support,
            min_pass_rate=min_pass_rate,
            min_pass_rate_lift=min_pass_rate_lift,
            model_dim=model_dim,
            run_smoke=run_smoke,
            validate_candidates=validate_candidates,
        )
        if candidate is None:
            continue
        candidates.append(candidate)
        if _is_ready(candidate["validation"]):
            ready.append(candidate)
        if len(candidates) >= max_candidates:
            break
    return candidates, ready


def _candidate_from_report_row(
    row: Mapping[str, Any],
    *,
    report_path: Path,
    rule_config: ComponentRuleConfig,
    preferred_pairs: frozenset[tuple[str, str]],
    blocked_pairs: frozenset[tuple[str, str]],
    candidate_index: int,
    seen: set[tuple[str, ...]],
    min_support: int,
    min_pass_rate: float,
    min_pass_rate_lift: float,
    model_dim: int,
    run_smoke: bool,
    validate_candidates: bool,
) -> dict[str, Any] | None:
    chain = _coerce_chain(row.get("pattern"))
    if not chain or chain in seen:
        return None
    seen.add(chain)
    has_preferred_pair = _has_blocked_pair(chain, preferred_pairs)
    if not _passes_row_filters(
        row,
        min_support=min_support,
        min_pass_rate=min_pass_rate,
        min_pass_rate_lift=min_pass_rate_lift,
        allow_preferred_pair=has_preferred_pair,
    ):
        return None
    if validate_component_op_chain(chain, config=rule_config):
        return None
    if not has_preferred_pair and _has_blocked_pair(chain, blocked_pairs):
        return None

    candidate = _candidate_from_window(
        row,
        chain=chain,
        index=candidate_index,
        source_path=str(report_path),
    )
    candidate["validation"] = _candidate_validation(
        candidate,
        model_dim=model_dim,
        run_smoke=run_smoke,
        validate_candidates=validate_candidates,
    )
    return candidate


def _candidate_validation(
    candidate: Mapping[str, Any],
    *,
    model_dim: int,
    run_smoke: bool,
    validate_candidates: bool,
) -> dict[str, Any]:
    chain = _coerce_chain(candidate.get("chain"))
    if validate_candidates:
        descriptor = candidate.get("component_descriptor")
        if _candidate_lowering(descriptor) in _BRANCH_LOWERINGS:
            return _validate_lowered_dynamic_candidate(
                candidate,
                model_dim=model_dim,
                run_smoke=run_smoke,
            )
        return _validate_candidate_chain(
            chain, model_dim=model_dim, run_smoke=run_smoke
        )
    return {
        "compile_passed": False,
        "validate_passed": False,
        "forward_passed": False,
        "backward_passed": False,
        "failure_mode": "not_run",
    }


def _candidate_lowering(descriptor: Any) -> str:
    if not isinstance(descriptor, Mapping):
        return "rmsnorm_chain_with_binary_skip"
    return str(descriptor.get("lowering") or "rmsnorm_chain_with_binary_skip")


def _load_candidate_rows(
    mining_report_path: str | Path,
) -> tuple[Path, Mapping[str, Any], list[Any]]:
    report_path = Path(mining_report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report.get("candidate_windows") if isinstance(report, Mapping) else []
    return report_path, report, rows if isinstance(rows, list) else []


def _candidate_artifact_payload(
    *,
    report_path: Path,
    rows: Sequence[Any],
    candidates: list[dict[str, Any]],
    ready: list[dict[str, Any]],
    min_lowered_ops: int,
    min_support: int,
    min_pass_rate: float,
    min_pass_rate_lift: float,
    negative_pair_max_pass_rate: float,
    negative_pair_max_lift: float,
    negative_pairs_blocked: int,
    model_dim: int,
    run_smoke: bool,
    schema_versions: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": "dynamic_component_candidates_v1",
        "metadata": {
            "created_at": time.time(),
            "input_source": str(report_path),
            "n_input_windows": len(rows),
            "n_candidates": len(candidates),
            "n_ready_for_registration": len(ready),
            "min_lowered_ops": int(min_lowered_ops),
            "min_support": int(min_support),
            "min_pass_rate": float(min_pass_rate),
            "min_pass_rate_lift": float(min_pass_rate_lift),
            "negative_pair_max_pass_rate": float(negative_pair_max_pass_rate),
            "negative_pair_max_lift": float(negative_pair_max_lift),
            "negative_pairs_blocked": int(negative_pairs_blocked),
            "model_dim": int(model_dim),
            "run_smoke": bool(run_smoke),
            "validation_required_for_ready": True,
            "component_rule_schema_versions": list(schema_versions),
        },
        "candidates": candidates,
        "ready_for_registration": ready,
    }


def _write_candidate_artifact(
    payload: Mapping[str, Any],
    output_path: str | Path | None,
) -> None:
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _blocked_negative_pairs(
    report: Mapping[str, Any],
    *,
    max_pass_rate: float,
    max_lift: float,
) -> frozenset[tuple[str, str]]:
    rules = report.get("op_pair_rules")
    if not isinstance(rules, Mapping):
        return frozenset()
    negative = rules.get("negative")
    if not isinstance(negative, list):
        return frozenset()
    out: set[tuple[str, str]] = set()
    for row in negative:
        if not isinstance(row, Mapping):
            continue
        pattern = row.get("pattern")
        if not isinstance(pattern, Sequence) or isinstance(pattern, (str, bytes)):
            continue
        if len(pattern) != 2:
            continue
        pass_rate = float(row.get("pass_rate") or 0.0)
        lift = float(row.get("pass_rate_lift") or 0.0)
        if pass_rate <= max_pass_rate and lift <= max_lift:
            out.add((str(pattern[0]), str(pattern[1])))
    return frozenset(out)


def _has_blocked_pair(
    chain: tuple[str, ...],
    blocked_pairs: frozenset[tuple[str, str]],
) -> bool:
    if not blocked_pairs:
        return False
    return any((left, right) in blocked_pairs for left, right in zip(chain, chain[1:]))


def _passes_row_filters(
    row: Mapping[str, Any],
    *,
    min_support: int,
    min_pass_rate: float,
    min_pass_rate_lift: float,
    allow_preferred_pair: bool,
) -> bool:
    if int(row.get("n") or 0) < min_support:
        return False
    return _passes_candidate_thresholds(
        pass_rate=float(row.get("pass_rate") or 0.0),
        pass_rate_lift=float(row.get("pass_rate_lift") or 0.0),
        min_pass_rate=min_pass_rate,
        min_pass_rate_lift=min_pass_rate_lift,
        allow_preferred_pair=allow_preferred_pair,
    )


def _passes_candidate_thresholds(
    *,
    pass_rate: float,
    pass_rate_lift: float,
    min_pass_rate: float,
    min_pass_rate_lift: float,
    allow_preferred_pair: bool,
) -> bool:
    if pass_rate >= min_pass_rate and pass_rate_lift >= min_pass_rate_lift:
        return True
    if not allow_preferred_pair:
        return False
    return (
        pass_rate >= PREFERRED_PAIR_MIN_PASS_RATE
        and pass_rate_lift >= PREFERRED_PAIR_MIN_PASS_RATE_LIFT
    )


def _coerce_chain(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(op) for op in value if str(op))


def _candidate_from_window(
    row: Mapping[str, Any],
    *,
    chain: tuple[str, ...],
    index: int,
    source_path: str,
) -> dict[str, Any]:
    lowered_op_count = estimated_chain_lowered_op_count(chain)
    role_counts = component_role_counts(chain)
    descriptor = _component_descriptor(chain, index, role_counts)
    candidate = {
        "proposed_template_name": _candidate_name(chain),
        "chain": list(chain),
        "chain_length": len(chain),
        "lowered_op_count": lowered_op_count,
        "n_total": int(row.get("n") or 0),
        "n_pass": int(row.get("stage1_passed") or 0),
        "pass_rate": float(row.get("pass_rate") or 0.0),
        "lift_vs_cohort": float(row.get("pass_rate_lift") or 0.0),
        "mean_loss_ratio": row.get("mean_loss_ratio"),
        "promotion_score": _promotion_score(row),
        "source": "component_rule_mining",
        "source_path": source_path,
        "component_descriptor": descriptor,
    }
    return candidate


def _component_descriptor(
    chain: tuple[str, ...],
    index: int,
    role_counts: Mapping[str, int],
) -> dict[str, Any]:
    lowering, branch_plan = _candidate_branch_lowering(chain)
    descriptor = {
        "descriptor_version": "component_chain_v1",
        "component_id": _component_id(chain, index),
        "roles": [get_role(op).value for op in chain],
        "role_counts": dict(role_counts),
        "slot_plan": [
            {
                "slot_index": int(slot["slot_index"]),
                "op_name": str(slot["op_name"]),
                "role": str(slot["role"]),
                "slot_classes": list(slot["slot_classes"]),
            }
            for slot in component_slot_plan(chain)
        ],
        "mixer_count": int(role_counts.get(OpRole.MIX.value, 0)),
        "route_count": int(role_counts.get(OpRole.ROUTE.value, 0)),
        "has_recursion_signal": any(_is_recursion_op(op) for op in chain),
        "has_multi_mixer": int(role_counts.get(OpRole.MIX.value, 0)) >= 2,
        "lowering": "rmsnorm_chain_with_binary_skip",
    }
    if branch_plan is not None:
        descriptor["lowering"] = lowering
        descriptor["branch_plan"] = branch_plan
    return descriptor


def _candidate_branch_lowering(
    chain: tuple[str, ...],
) -> tuple[str, dict[str, Any] | None]:
    trunk_sidecar = _trunk_sidecar_branch_plan(chain)
    if trunk_sidecar is not None:
        return "trunk_sidecar_merge_v1", trunk_sidecar
    router_lane = _router_lane_blend_plan(chain)
    if router_lane is not None:
        return "router_lane_blend_v1", router_lane
    mixer_restore = _mixer_sidecar_restore_branch_plan(chain)
    if mixer_restore is not None:
        return "mixer_sidecar_restore_v1", mixer_restore
    return "rmsnorm_chain_with_binary_skip", None


def _trunk_sidecar_branch_plan(chain: tuple[str, ...]) -> dict[str, Any] | None:
    roles = tuple(get_role(op) for op in chain)
    mixer_indices = [idx for idx, role in enumerate(roles) if role is OpRole.MIX]
    if len(mixer_indices) < 2:
        return None

    first_mixer, second_mixer = mixer_indices[:2]
    trunk_start = 0
    while trunk_start < first_mixer and roles[trunk_start] is OpRole.NORMALIZE:
        trunk_start += 1
    trunk_end = first_mixer + 1
    if trunk_end < len(chain) and roles[trunk_end] is OpRole.PROJECT:
        trunk_end += 1

    trunk_indices = tuple(
        idx
        for idx in range(trunk_start, trunk_end)
        if roles[idx] is not OpRole.RESIDUAL
    )
    sidecar_indices = tuple(
        idx
        for idx in range(trunk_end, second_mixer + 1)
        if roles[idx] is not OpRole.RESIDUAL
    )
    if not trunk_indices or not sidecar_indices:
        return None
    return {
        "trunk_indices": list(trunk_indices),
        "sidecar_indices": list(sidecar_indices),
        "merge_op": "add",
        "post_merge_norm": True,
        "residual_output": True,
    }


def _mixer_sidecar_restore_branch_plan(chain: tuple[str, ...]) -> dict[str, Any] | None:
    roles = tuple(get_role(op) for op in chain)
    mixer_indices = [idx for idx, role in enumerate(roles) if role is OpRole.MIX]
    if len(mixer_indices) != 1:
        return None

    mixer_index = mixer_indices[0]
    trunk_indices = _single_mixer_trunk_indices(roles, mixer_index)
    sidecar_indices = _single_mixer_sidecar_indices(roles, mixer_index, trunk_indices)
    if not trunk_indices or not sidecar_indices:
        return None
    return {
        "trunk_indices": list(trunk_indices),
        "sidecar_indices": list(sidecar_indices),
        "merge_op": "add",
        "post_merge_norm": True,
        "residual_output": True,
    }


def _router_lane_blend_plan(chain: tuple[str, ...]) -> dict[str, Any] | None:
    roles = tuple(get_role(op) for op in chain)
    route_indices = [idx for idx, role in enumerate(roles) if role is OpRole.ROUTE]
    gate_indices = [idx for idx, role in enumerate(roles) if role is OpRole.GATE]
    matmul_indices = [idx for idx, op in enumerate(chain) if op == "matmul"]
    if not route_indices or not gate_indices or not matmul_indices:
        return None

    route_index = route_indices[0]
    gate_index = next((idx for idx in gate_indices if idx > route_index), None)
    matmul_index = next((idx for idx in matmul_indices if idx < route_index), None)
    if gate_index is None or matmul_index is None:
        return None

    value_project_index = _nearest_role_before(roles, OpRole.PROJECT, matmul_index)
    score_project_index = _nearest_role_between(
        roles, OpRole.PROJECT, matmul_index + 1, route_index
    )
    if value_project_index is None or score_project_index is None:
        return None
    return {
        "value_project_index": int(value_project_index),
        "matmul_index": int(matmul_index),
        "score_project_index": int(score_project_index),
        "route_index": int(route_index),
        "gate_index": int(gate_index),
        "blend_op": "gated_lane_blend",
        "post_merge_norm": True,
        "residual_output": True,
    }


def _nearest_role_before(
    roles: tuple[OpRole, ...],
    role: OpRole,
    end: int,
) -> int | None:
    for index in range(end - 1, -1, -1):
        if roles[index] is role:
            return index
    return None


def _nearest_role_between(
    roles: tuple[OpRole, ...],
    role: OpRole,
    start: int,
    end: int,
) -> int | None:
    for index in range(start, end):
        if roles[index] is role:
            return index
    return None


def _single_mixer_trunk_indices(
    roles: tuple[OpRole, ...],
    mixer_index: int,
) -> tuple[int, ...]:
    trunk = [mixer_index]
    if mixer_index + 1 < len(roles) and roles[mixer_index + 1] is OpRole.PROJECT:
        trunk.append(mixer_index + 1)
    elif mixer_index > 0 and roles[mixer_index - 1] is OpRole.PROJECT:
        trunk.insert(0, mixer_index - 1)
    return tuple(trunk)


def _single_mixer_sidecar_indices(
    roles: tuple[OpRole, ...],
    mixer_index: int,
    trunk_indices: tuple[int, ...],
) -> tuple[int, ...]:
    trunk_set = set(trunk_indices)
    start = max(trunk_indices) + 1
    while start < len(roles) and roles[start] is OpRole.RESIDUAL:
        start += 1
    sidecar = _restore_sidecar_segment(roles, range(start, len(roles)), trunk_set)
    if sidecar:
        return sidecar

    for start in range(0, mixer_index):
        if roles[start] is not OpRole.NORMALIZE:
            continue
        sidecar = _restore_sidecar_segment(roles, range(start, mixer_index), trunk_set)
        if sidecar:
            return sidecar
    return _restore_sidecar_segment(roles, range(0, mixer_index), trunk_set)


def _restore_sidecar_segment(
    roles: tuple[OpRole, ...],
    indices: range,
    excluded_indices: set[int],
) -> tuple[int, ...]:
    out = [
        idx
        for idx in indices
        if idx not in excluded_indices and roles[idx] is not OpRole.RESIDUAL
    ]
    while out and roles[out[-1]] is OpRole.NORMALIZE:
        out.pop()
    if len(out) < 2:
        return ()
    sidecar_roles = {roles[idx] for idx in out}
    if OpRole.PROJECT not in sidecar_roles:
        return ()
    if not sidecar_roles.intersection(
        {OpRole.ACTIVATE, OpRole.GATE, OpRole.ROUTE, OpRole.POSITION}
    ):
        return ()
    return tuple(out)


def _candidate_name(chain: tuple[str, ...]) -> str:
    anchors = [
        op for op in chain if get_role(op) in {OpRole.MIX, OpRole.ROUTE, OpRole.GATE}
    ][:3]
    if not anchors:
        anchors = list(chain[:2])
    slug = "_".join(anchors)
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", slug).strip("_").lower()
    return f"mined_component_{slug or 'chain'}_block"


def _component_id(chain: tuple[str, ...], index: int) -> str:
    digest = hashlib.blake2b("|".join(chain).encode("utf-8"), digest_size=5).hexdigest()
    return f"component_chain_{index:04d}_{digest}"


def _promotion_score(row: Mapping[str, Any]) -> float:
    n = max(1.0, float(row.get("n") or 1.0))
    pass_rate = max(0.0, float(row.get("pass_rate") or 0.0))
    lift = float(row.get("pass_rate_lift") or 0.0)
    loss_ratio = row.get("mean_loss_ratio")
    loss_bonus = 1.0
    if loss_ratio is not None:
        try:
            loss_bonus = max(0.2, min(2.0, 1.0 / max(0.05, float(loss_ratio))))
        except (TypeError, ValueError):
            loss_bonus = 1.0
    return round(math.sqrt(n) * pass_rate * (1.0 + max(0.0, lift)) * loss_bonus, 6)


def _validate_candidate_chain(
    chain: tuple[str, ...],
    *,
    model_dim: int,
    run_smoke: bool,
) -> dict[str, Any]:
    # The user explicitly does not want a small fixed max-op ceiling here.
    # These bounds only keep the existing validator from rejecting larger
    # mined components because it was originally tuned for 3-4 op chains.
    lowered_ops = estimated_chain_lowered_op_count(chain)
    return validate_chain(
        chain,
        model_dim=model_dim,
        max_ops=max(32, lowered_ops + 8),
        max_depth=max(32, lowered_ops + 8),
        run_smoke=run_smoke,
    )


def _validate_lowered_dynamic_candidate(
    candidate: Mapping[str, Any],
    *,
    model_dim: int,
    run_smoke: bool,
) -> dict[str, Any]:
    chain = _coerce_chain(candidate.get("chain"))
    lowered_ops = estimated_chain_lowered_op_count(chain)
    result: dict[str, Any] = {
        "compile_passed": False,
        "validate_passed": False,
        "forward_passed": False,
        "backward_passed": False,
        "n_ops": 0,
        "failure_mode": None,
        "error": None,
        "lowering_validated": _candidate_lowering(
            candidate.get("component_descriptor")
        ),
    }
    try:
        graph = _build_lowered_dynamic_candidate_graph(candidate, model_dim=model_dim)
    except Exception as exc:
        result["failure_mode"] = "build"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["n_ops"] = len(graph.nodes) - 1
    validation = validate_graph(
        graph,
        max_ops=max(32, lowered_ops + 12),
        max_depth=max(32, lowered_ops + 12),
    )
    if validation.errors:
        result["failure_mode"] = "validate"
        result["error"] = "; ".join(validation.errors[:3])
        return result
    result["validate_passed"] = True

    return _finalize_compile_and_smoke(
        graph, result, _compile_layer_module, model_dim=model_dim, run_smoke=run_smoke
    )


def _build_lowered_dynamic_candidate_graph(
    candidate: Mapping[str, Any],
    *,
    model_dim: int,
) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    input_node = graph.add_input()
    dynamic_candidate = DynamicTemplateCandidate(
        template_id=str(candidate.get("proposed_template_name") or "dynamic_component"),
        display_name=str(
            candidate.get("proposed_template_name") or "dynamic_component"
        ),
        chain=_coerce_chain(candidate.get("chain")),
        weight=float(candidate.get("promotion_score") or 1.0),
        lowered_op_count=int(candidate.get("lowered_op_count") or 0),
        source_path=str(candidate.get("source_path") or ""),
        source=str(candidate.get("source") or "component_rule_mining"),
        evidence={},
        validation={},
        component_descriptor=(
            dict(candidate["component_descriptor"])
            if isinstance(candidate.get("component_descriptor"), Mapping)
            else {}
        ),
    )
    output = apply_dynamic_template_candidate(
        graph,
        input_node,
        random.Random(0),
        dynamic_candidate,
    )
    graph.set_output(output)
    return graph


def _is_ready(validation: Mapping[str, Any]) -> bool:
    return bool(
        validation.get("validate_passed")
        and validation.get("compile_passed")
        and validation.get("forward_passed")
        and validation.get("backward_passed")
    )


def _is_recursion_op(op_name: str) -> bool:
    return op_name in {
        "fixed_point_iter",
        "mixture_of_recursions",
        "depth_gated_transform",
        "score_depth_blend",
        "depth_weighted_proj",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--min-lowered-ops", type=int, default=8)
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--min-pass-rate", type=float, default=0.70)
    parser.add_argument("--min-pass-rate-lift", type=float, default=0.0)
    parser.add_argument("--negative-pair-max-pass-rate", type=float, default=0.50)
    parser.add_argument("--negative-pair-max-lift", type=float, default=-0.20)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args(argv)

    payload = build_dynamic_component_candidates(
        mining_report_path=args.input,
        output_path=args.output,
        max_candidates=args.max_candidates,
        min_lowered_ops=args.min_lowered_ops,
        min_support=args.min_support,
        min_pass_rate=args.min_pass_rate,
        min_pass_rate_lift=args.min_pass_rate_lift,
        negative_pair_max_pass_rate=args.negative_pair_max_pass_rate,
        negative_pair_max_lift=args.negative_pair_max_lift,
        model_dim=args.model_dim,
        run_smoke=not args.no_smoke,
        validate_candidates=not args.skip_validation,
    )
    meta = payload["metadata"]
    print(
        "dynamic_component_candidates "
        f"candidates={meta['n_candidates']} "
        f"ready={meta['n_ready_for_registration']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
