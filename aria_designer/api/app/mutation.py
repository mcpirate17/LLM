from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from aria_designer.component_identity import component_leaf
from .database import get_workflow, save_proposal
from .intent_parser import IntentConstraints, component_groups, parse_intent_constraints
from .models import AriaPatchProposalModel, PatchOpModel
from .research_signals import fetch_research_recommendation_signals
from .type_utils import safe_float, safe_str

_IO_GROUPS = {"io"}
_MUTATION_PENALTIES = {
    "mutate_param": 0.015,
    "replace_node": 0.03,
    "add_node": 0.035,
    "remove_node": 0.08,
    "rewire": 0.02,
}


def _loss_ratio_value(parent_scores: Dict[str, Any]) -> float:
    return safe_float(parent_scores.get("loss_ratio"), default=0.5)


def _param_delta_window(
    loss_ratio: float, max_param_delta: float
) -> Tuple[float, float]:
    if loss_ratio < 0.3:
        upper = max(max_param_delta, 0.3)
        return 0.12, min(0.3, upper)
    if loss_ratio > 0.7:
        upper = max(0.02, min(max_param_delta, 0.1))
        return 0.02, upper
    upper = max(0.02, max_param_delta)
    return max(0.02, min(0.1, upper / 2.0)), upper


def _signal_pair_rates(signals: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if not isinstance(signals, dict):
        return {}
    rates: Dict[str, float] = {}
    for row in signals.get("op_pair_priors") or ():
        signature = safe_str(row.get("signature"))
        success_rate = safe_float(row.get("success_rate"))
        if signature and success_rate > 0.0:
            rates[signature] = success_rate
    return rates


def _component_allowed(component_type: str, constraints: IntentConstraints) -> bool:
    groups = set(component_groups(component_type))
    if groups & _IO_GROUPS or groups & set(constraints.blocked_component_groups):
        return False
    preferred = set(constraints.preferred_component_groups)
    return not preferred or bool(groups & preferred)


def _resolve_signal_component(
    op_name: str,
    options: Sequence[str] = (),
    fallback_component: str = "",
) -> str:
    normalized = component_leaf(op_name)
    option_map = {
        component_leaf(component_type): component_type
        for component_type in options
        if component_type
    }
    if normalized in option_map:
        return option_map[normalized]
    if "/" in fallback_component:
        return f"{fallback_component.rsplit('/', 1)[0]}/{normalized}"
    return normalized


def _graph_adjacency(
    graph: Dict[str, Any],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    nodes_by_id = {str(node.get("id")): node for node in graph.get("nodes", [])}
    parents: Dict[str, List[str]] = {}
    children: Dict[str, List[str]] = {}
    for edge in graph.get("edges") or ():
        source_id = str(edge.get("source"))
        target_id = str(edge.get("target"))
        if source_id not in nodes_by_id or target_id not in nodes_by_id:
            continue
        children.setdefault(source_id, []).append(target_id)
        parents.setdefault(target_id, []).append(source_id)
    return parents, children, nodes_by_id


def _best_neighbor_component(
    prev_ops: Sequence[str],
    next_ops: Sequence[str],
    pair_rates: Dict[str, float],
    constraints: IntentConstraints,
    options: Sequence[str] = (),
    fallback_component: str = "",
) -> str:
    candidate_scores: Dict[str, float] = {}
    for signature, score in pair_rates.items():
        left, _, right = signature.partition("->")
        if not right:
            continue
        if left in prev_ops:
            candidate_scores[right] = candidate_scores.get(right, 0.0) + score
        if right in next_ops:
            candidate_scores[left] = candidate_scores.get(left, 0.0) + score
    ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
    for op_name, _ in ranked:
        component_type = _resolve_signal_component(op_name, options, fallback_component)
        if _component_allowed(component_type, constraints):
            return component_type
    return ""


@dataclass(slots=True, frozen=True)
class MutationPlan:
    ops: Tuple[PatchOpModel, ...]
    rationale: str
    predicted_multiplier: float


def refine_winner(
    workflow_id: str,
    num_variations: int = 3,
    intent: Optional[str] = None,
    parent_scores: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Generate constrained workflow refinements and return proposal IDs."""
    workflow = get_workflow(workflow_id)
    if not workflow:
        raise ValueError("Workflow not found")

    graph = json.loads(workflow["graph_json"])
    constraints = parse_intent_constraints(intent, parent_scores)
    rng = random.Random()
    proposal_ids: List[str] = []
    attempts = 0
    max_attempts = max(num_variations * 12, 12)

    while len(proposal_ids) < num_variations and attempts < max_attempts:
        attempts += 1
        plan = _build_mutation_plan(graph, constraints, parent_scores or {}, rng)
        if plan is None or not _meets_parent_score_floor(plan, parent_scores or {}):
            continue
        proposal_id = f"evo_{uuid4().hex[:10]}"
        rationale = _format_rationale(plan.rationale, parent_scores or {}, constraints)
        patch = AriaPatchProposalModel(
            workflow_id=workflow_id,
            base_version=workflow["version"],
            rationale=rationale,
            expected_impact={
                "intent": constraints.intent_key,
                "predicted_retention": f"{plan.predicted_multiplier:.3f}",
            },
            ops=list(plan.ops),
        )
        save_proposal(
            proposal_id=proposal_id,
            workflow_id=workflow_id,
            patch_json=json.dumps(patch.model_dump()),
            rationale=rationale,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        proposal_ids.append(proposal_id)
    return proposal_ids


def _build_mutation_plan(
    graph: Dict[str, Any],
    constraints: IntentConstraints,
    parent_scores: Dict[str, Any],
    rng: random.Random,
) -> Optional[MutationPlan]:
    builders = {
        "mutate_param": _build_param_mutation,
        "replace_activation": _build_replacement_mutation,
        "add_layer": _build_add_layer_mutation,
    }
    strategies = list(constraints.allowed_mutations)
    rng.shuffle(strategies)
    for strategy in strategies:
        plan = builders[strategy](graph, constraints, parent_scores, rng)
        if plan is not None:
            return plan
    return None


def _build_param_mutation(
    graph: Dict[str, Any],
    constraints: IntentConstraints,
    parent_scores: Dict[str, Any],
    rng: random.Random,
) -> Optional[MutationPlan]:
    candidates = _numeric_param_candidates(graph, constraints)
    if not candidates:
        return None
    touches = min(constraints.max_nodes_touched, len(candidates))
    selected = rng.sample(candidates, k=max(1, touches))
    ops: List[PatchOpModel] = []
    labels: List[str] = []
    for node, param_name, value in selected:
        new_value = _mutated_value(param_name, value, constraints, parent_scores, rng)
        if new_value == value:
            continue
        ops.append(
            PatchOpModel(
                op="mutate_param",
                node_id=str(node.get("id")),
                payload={param_name: new_value},
            )
        )
        labels.append(f"{node.get('id')}:{param_name} {value}->{new_value}")
    if not ops:
        return None
    multiplier = _predicted_multiplier(parent_scores, ops, constraints)
    return MutationPlan(tuple(ops), f"Evolution: tuned {'; '.join(labels)}", multiplier)


def _build_replacement_mutation(
    graph: Dict[str, Any],
    constraints: IntentConstraints,
    parent_scores: Dict[str, Any],
    rng: random.Random,
) -> Optional[MutationPlan]:
    candidates = _target_nodes(graph, constraints, {"activation", "normalization"})
    if not candidates:
        candidates = _target_nodes(
            graph, constraints, set(constraints.preferred_component_groups)
        )
    if not candidates:
        return None
    node = rng.choice(candidates)
    current = str(node.get("component_type") or "")
    current_leaf = component_leaf(current)
    options = [
        item
        for item in constraints.replacement_components
        if component_leaf(item) != current_leaf
    ]
    if not options:
        return None
    parents, children, nodes_by_id = _graph_adjacency(graph)
    node_id = str(node.get("id"))
    prev_ops = [
        component_leaf(nodes_by_id[parent_id].get("component_type"))
        for parent_id in parents.get(node_id, [])
    ]
    next_ops = [
        component_leaf(nodes_by_id[child_id].get("component_type"))
        for child_id in children.get(node_id, [])
    ]
    replacement = ""
    pair_rates = _signal_pair_rates(fetch_research_recommendation_signals())
    if pair_rates:
        replacement = _best_neighbor_component(
            prev_ops,
            next_ops,
            pair_rates,
            constraints,
            options=options,
            fallback_component=current,
        )
    if not replacement:
        replacement = rng.choice(options)
    ops = [
        PatchOpModel(
            op="replace_node",
            node_id=str(node.get("id")),
            payload={
                "component_type": replacement,
                "params": dict(node.get("params") or {}),
            },
        )
    ]
    if constraints.intent_key == "improve_stability":
        tweak = _stability_followup_op(
            graph, constraints, rng, exclude_node=str(node.get("id"))
        )
        if tweak is not None:
            ops.append(tweak)
    multiplier = _predicted_multiplier(parent_scores, ops, constraints)
    return MutationPlan(
        tuple(ops), f"Evolution: stabilized {current} at {node.get('id')}", multiplier
    )


def _build_add_layer_mutation(
    graph: Dict[str, Any],
    constraints: IntentConstraints,
    parent_scores: Dict[str, Any],
    rng: random.Random,
) -> Optional[MutationPlan]:
    edges = list(graph.get("edges") or [])
    if not edges:
        return None
    _, _, nodes_by_id = _graph_adjacency(graph)
    candidates = [
        edge for edge in edges if _can_insert_clone(edge, nodes_by_id, constraints)
    ]
    if not candidates:
        return None
    edge = rng.choice(candidates)
    source_id = str(edge.get("source"))
    target_id = str(edge.get("target"))
    source_node = nodes_by_id[source_id]
    target_node = nodes_by_id[target_id]
    inserted_component = str(source_node.get("component_type") or "")
    pair_rates = _signal_pair_rates(fetch_research_recommendation_signals())
    if pair_rates:
        preferred_component = _best_neighbor_component(
            (component_leaf(source_node.get("component_type")),),
            (component_leaf(target_node.get("component_type")),),
            pair_rates,
            constraints,
            fallback_component=inserted_component
            or str(target_node.get("component_type") or ""),
        )
        if preferred_component:
            inserted_component = preferred_component
    inserted_params = (
        dict(source_node.get("params") or {})
        if inserted_component == str(source_node.get("component_type") or "")
        else {}
    )
    new_node_id = f"refine_{uuid4().hex[:8]}"
    new_edge_a = f"refine_e_{uuid4().hex[:6]}"
    new_edge_b = f"refine_e_{uuid4().hex[:6]}"
    ops = [
        PatchOpModel(
            op="add_node",
            payload={
                "id": new_node_id,
                "component_type": inserted_component,
                "params": inserted_params,
                "ui_meta": {"generated_by": "refine_winner"},
            },
        ),
        PatchOpModel(
            op="rewire",
            payload={"action": "remove", "source": source_id, "target": target_id},
        ),
        PatchOpModel(
            op="rewire",
            edge_id=new_edge_a,
            payload={"action": "add", "source": source_id, "target": new_node_id},
        ),
        PatchOpModel(
            op="rewire",
            edge_id=new_edge_b,
            payload={"action": "add", "source": new_node_id, "target": target_id},
        ),
    ]
    multiplier = _predicted_multiplier(parent_scores, ops, constraints)
    rationale = (
        f"Evolution: inserted {inserted_component} between {source_id} and {target_id}"
    )
    return MutationPlan(tuple(ops), rationale, multiplier)


def _numeric_param_candidates(
    graph: Dict[str, Any],
    constraints: IntentConstraints,
) -> List[Tuple[Dict[str, Any], str, float]]:
    allowed = set(constraints.preferred_component_groups)
    candidates: List[Tuple[Dict[str, Any], str, float]] = []
    for node in _target_nodes(graph, constraints, allowed):
        for name, value in (node.get("params") or {}).items():
            if name not in constraints.target_param_names or not isinstance(
                value, (int, float)
            ):
                continue
            candidates.append((node, str(name), float(value)))
    if candidates:
        return candidates
    for node in _target_nodes(graph, constraints, allowed):
        for name, value in (node.get("params") or {}).items():
            if isinstance(value, (int, float)):
                candidates.append((node, str(name), float(value)))
    return candidates


def _target_nodes(
    graph: Dict[str, Any],
    constraints: IntentConstraints,
    preferred_groups: set[str],
) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    blocked = set(constraints.blocked_component_groups)
    for node in graph.get("nodes", []):
        groups = set(component_groups(str(node.get("component_type") or "")))
        if groups & _IO_GROUPS or groups & blocked:
            continue
        if not preferred_groups or groups & preferred_groups:
            matched.append(node)
    return matched


def _mutated_value(
    param_name: str,
    value: float,
    constraints: IntentConstraints,
    parent_scores: Dict[str, Any],
    rng: random.Random,
) -> int | float:
    direction = constraints.param_direction
    loss_ratio = _loss_ratio_value(parent_scores)
    min_delta, max_delta = _param_delta_window(loss_ratio, constraints.max_param_delta)
    bias_by_quality = 0.55 if loss_ratio > 0.7 else 0.8
    signed_delta = rng.uniform(-max_delta, max_delta)
    if direction != 0:
        sign = direction if rng.random() <= bias_by_quality else -direction
        signed_delta = sign * rng.uniform(min_delta, max_delta)
    elif abs(signed_delta) < min_delta:
        signed_delta = min_delta if signed_delta >= 0.0 else -min_delta
    scale = 1.0 + signed_delta
    new_value = value * scale
    if any(token in param_name for token in ("head", "rank", "dim", "width", "hidden")):
        new_value = max(1.0, round(new_value))
    if float(value).is_integer():
        return max(1, int(round(new_value)))
    return round(float(new_value), 6)


def _stability_followup_op(
    graph: Dict[str, Any],
    constraints: IntentConstraints,
    rng: random.Random,
    exclude_node: str,
) -> Optional[PatchOpModel]:
    candidates = [
        candidate
        for candidate in _numeric_param_candidates(graph, constraints)
        if str(candidate[0].get("id")) != exclude_node
    ]
    if not candidates:
        return None
    node, param_name, value = rng.choice(candidates)
    return PatchOpModel(
        op="mutate_param",
        node_id=str(node.get("id")),
        payload={
            param_name: _mutated_value(
                param_name, value, constraints, {"loss_ratio": 0.8}, rng
            )
        },
    )


def _can_insert_clone(
    edge: Dict[str, Any],
    nodes_by_id: Dict[str, Dict[str, Any]],
    constraints: IntentConstraints,
) -> bool:
    source = nodes_by_id.get(str(edge.get("source")))
    target = nodes_by_id.get(str(edge.get("target")))
    if not source or not target:
        return False
    source_groups = set(component_groups(str(source.get("component_type") or "")))
    target_groups = set(component_groups(str(target.get("component_type") or "")))
    if source_groups & _IO_GROUPS or target_groups & _IO_GROUPS:
        return False
    if source_groups & set(constraints.blocked_component_groups):
        return False
    return bool(source_groups & set(constraints.preferred_component_groups))


def _predicted_multiplier(
    parent_scores: Dict[str, Any],
    ops: Sequence[PatchOpModel],
    constraints: IntentConstraints,
) -> float:
    penalty = sum(_MUTATION_PENALTIES.get(op.op, 0.04) for op in ops)
    if constraints.preserve_novelty:
        penalty *= 0.85
    tier = str(parent_scores.get("tier") or "").lower()
    if tier in {"investigation", "validation", "breakthrough"}:
        penalty *= 1.15
    return max(0.75, 1.0 - penalty)


def _meets_parent_score_floor(
    plan: MutationPlan, parent_scores: Dict[str, Any]
) -> bool:
    parent_composite = float(parent_scores.get("composite_score") or 0.0)
    if parent_composite <= 0.0:
        return True
    predicted = parent_composite * plan.predicted_multiplier
    return predicted >= (parent_composite * 0.95)


def _format_rationale(
    rationale: str,
    parent_scores: Dict[str, Any],
    constraints: IntentConstraints,
) -> str:
    if not parent_scores:
        return rationale
    tier = str(parent_scores.get("tier") or "unknown")
    score = float(parent_scores.get("composite_score") or 0.0)
    novelty = float(
        parent_scores.get("screening_novelty")
        or parent_scores.get("novelty_score")
        or 0.0
    )
    return (
        f"{rationale} "
        f"[intent={constraints.intent_key}; parent_tier={tier}; parent_composite={score:.3f}; novelty={novelty:.3f}]"
    )
