"""Causal attribution planning for survivor counterfactual tests.

This module deliberately stops at planning and evidence summarization. The
runner owns model compilation/training so causal tests reuse the existing
ablation evaluator instead of adding a parallel execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Iterable, Mapping, Optional, Sequence

from research.synthesis.graph import ComputationGraph
from research.synthesis.serializer import graph_from_json
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value

LOGGER = logging.getLogger(__name__)

SCAFFOLD_OPS: frozenset[str] = frozenset(
    {
        "add",
        "linear_proj",
        "linear_proj_up",
        "linear_proj_down",
        "rmsnorm",
        "layernorm",
        "mul",
        "sigmoid",
        "tanh",
        "relu",
        "gelu",
        "silu",
        "identity",
    }
)

HIGH_VALUE_TOKENS: tuple[str, ...] = (
    "route",
    "routing",
    "sparse",
    "compression",
    "compress",
    "moe",
    "expert",
    "token",
    "selective",
    "scan",
    "topk",
    "entropy",
    "gather",
    "cosine",
    "outer",
    "matmul",
    "binding",
)


@dataclass(frozen=True, slots=True)
class CausalAblationCandidate:
    parent_experiment_id: str
    parent_result_id: str
    parent_fingerprint: str
    parent_loss_ratio: Optional[float]
    graph: ComputationGraph
    rule_type: str
    rule_key: str
    hypothesis: str
    context: Mapping[str, Any]


def annotate_observations(
    observations: Iterable[Mapping[str, Any]],
    meta_by_fingerprint: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Overlay per-fingerprint provenance metadata onto observation rows.

    Used by both `champion_exhaustive_ablation` and any other campaign that
    wants to attach the "what did we change in this child" context to each
    observation before recording.
    """
    annotated: list[dict[str, Any]] = []
    for observation in observations:
        item = dict(observation)
        fp = str(item.get("child_fingerprint") or "")
        provenance = dict(item.get("provenance") or {})
        provenance.update(meta_by_fingerprint.get(fp, {}))
        item["provenance"] = provenance
        annotated.append(item)
    return annotated


def run_ablation_suite(
    *,
    nb: Any,
    runner: Any,
    config: Any,
    candidate: CausalAblationCandidate,
    graphs: Sequence[ComputationGraph],
    child_meta_by_fingerprint: Optional[Mapping[str, Mapping[str, Any]]] = None,
    campaign: str = "ablation_suite",
    extra_evidence_fields: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Single canonical entry point for executing one ablation suite.

    All ablation paths (HTTP-triggered, continuous-loop, exhaustive champion
    CLI) flow through here so they share dedup against historical children,
    use the same `_run_ablation_experiment` execution path, summarize
    evidence the same way, and persist the same observation/evidence rows.

    Returns a summary dict (evidence_id, outcome, counts) or None if the
    suite produced no observations.
    """
    if not graphs:
        return None
    historical = find_historical_ablation_observations(
        nb,
        candidate=candidate,
        graphs=list(graphs),
    )
    historically_seen = {
        str(obs.get("child_fingerprint") or "")
        for obs in historical
        if str(obs.get("child_fingerprint") or "")
    }
    graphs_to_run = [
        graph for graph in graphs if graph.fingerprint() not in historically_seen
    ]
    hypothesis = (
        f"{campaign}: {candidate.rule_type}:{candidate.rule_key}; "
        f"parent_result_id={candidate.parent_result_id}; {candidate.hypothesis}"
    )
    exp_ids, outcome = runner._run_ablation_experiment(
        nb=nb,
        config=config,
        hypothesis=hypothesis,
        ablation_graphs=graphs_to_run,
        original_loss_ratio=candidate.parent_loss_ratio,
    )
    executed = find_executed_ablation_observations(
        nb,
        candidate=candidate,
        experiment_ids=exp_ids,
    )
    meta = dict(child_meta_by_fingerprint or {})
    observations = (
        annotate_observations(historical + executed, meta)
        if meta
        else (historical + executed)
    )
    if not observations and not exp_ids:
        return None

    evidence = summarize_ablation_effect(
        nb,
        candidate=candidate,
        ablation_experiment_ids=exp_ids,
        ablation_outcome=outcome,
        child_observations=observations,
    )
    evidence_payload = json.loads(evidence.get("evidence_json") or "{}")
    evidence_payload["campaign"] = campaign
    evidence_payload["planned_child_count"] = len(graphs)
    evidence_payload["executed_child_count"] = len(executed)
    evidence_payload["historical_child_count"] = len(historical)
    if extra_evidence_fields:
        evidence_payload.update(dict(extra_evidence_fields))
    evidence["evidence_json"] = json.dumps(evidence_payload, sort_keys=True)
    evidence_id = nb.record_causal_rule_evidence(evidence)
    inserted = nb.record_causal_ablation_child_observations(evidence_id, observations)
    nb.flush_writes()

    return {
        "evidence_id": evidence_id,
        "rule_type": evidence["rule_type"],
        "rule_key": evidence["rule_key"],
        "outcome": evidence["outcome"],
        "confidence": evidence["confidence"],
        "effect_size": evidence["effect_size"],
        "ablation_total": evidence["ablation_total"],
        "ablation_stage1_pass_count": evidence["ablation_stage1_pass_count"],
        "ablation_experiment_ids": exp_ids,
        "planned_children": len(graphs),
        "executed_children": len(executed),
        "historical_children": len(historical),
        "inserted_child_observations": inserted,
    }


def select_causal_ablation_candidates(
    nb: Any,
    *,
    experiment_id: str,
    max_survivors: int,
    max_signals_per_survivor: int,
) -> list[CausalAblationCandidate]:
    """Return bounded survivor signals worth testing with ablations."""

    if max_survivors <= 0 or max_signals_per_survivor <= 0:
        return []

    rows = nb.conn.execute(
        """SELECT result_id, graph_fingerprint, graph_json, loss_ratio
           FROM program_results_compat
           WHERE experiment_id = ?
             AND COALESCE(stage1_passed, 0) = 1
             AND graph_json IS NOT NULL
             AND TRIM(CAST(graph_json AS TEXT)) <> ''
           ORDER BY loss_ratio ASC NULLS LAST, result_id ASC
           LIMIT ?""",
        (experiment_id, int(max_survivors)),
    ).fetchall()

    candidates: list[CausalAblationCandidate] = []
    seen_rules: set[tuple[str, str, str]] = set()
    for row in rows:
        try:
            graph_json = resolve_graph_json_value(
                nb.conn,
                nb.db_path,
                row["graph_json"],
            )
            graph = graph_from_json(graph_json)
        except (TypeError, ValueError, RuntimeError, json.JSONDecodeError):
            continue
        parent_loss = _optional_float(row["loss_ratio"])
        parent_result_id = str(row["result_id"] or "")
        parent_fp = str(row["graph_fingerprint"] or graph.fingerprint())
        signals = _rank_graph_signals(graph)[: int(max_signals_per_survivor)]
        for signal in signals:
            dedup_key = (parent_result_id, signal["rule_type"], signal["rule_key"])
            if dedup_key in seen_rules:
                continue
            seen_rules.add(dedup_key)
            candidates.append(
                CausalAblationCandidate(
                    parent_experiment_id=experiment_id,
                    parent_result_id=parent_result_id,
                    parent_fingerprint=parent_fp,
                    parent_loss_ratio=parent_loss,
                    graph=graph,
                    rule_type=signal["rule_type"],
                    rule_key=signal["rule_key"],
                    hypothesis=signal["hypothesis"],
                    context=signal["context"],
                )
            )
    return candidates


def summarize_ablation_effect(
    nb: Any,
    *,
    candidate: CausalAblationCandidate,
    ablation_experiment_ids: Iterable[str],
    ablation_outcome: str,
    child_observations: Optional[Sequence[Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Summarize result deltas for one causal ablation test."""

    exp_ids = [str(exp_id) for exp_id in ablation_experiment_ids if str(exp_id)]
    observations = list(child_observations or ())
    stats = (
        _observation_stats(observations)
        if observations
        else _ablation_result_stats(nb, exp_ids)
    )
    original = candidate.parent_loss_ratio
    ablated_best = stats["best_loss_ratio"]
    effect_size = None
    if original is not None and ablated_best is not None:
        effect_size = float(ablated_best) - float(original)

    effective_total = int(stats.get("unique_fingerprint_count") or stats["total"])
    outcome = _classify_effect(
        ablation_outcome=ablation_outcome,
        effect_size=effect_size,
        ablation_total=int(stats["total"]),
        ablation_stage1=int(stats["stage1_pass_count"]),
    )
    confidence = _effect_confidence(
        effect_size=effect_size,
        ablation_total=effective_total,
        ablation_stage1=int(stats["stage1_pass_count"]),
    )
    all_exp_ids = sorted(
        {
            str(obs.get("child_experiment_id") or obs.get("experiment_id") or "")
            for obs in observations
            if str(obs.get("child_experiment_id") or obs.get("experiment_id") or "")
        }
    )
    executed_ids = set(exp_ids)
    historical_exp_ids = [
        exp_id for exp_id in all_exp_ids if exp_id not in executed_ids
    ]
    return {
        "parent_experiment_id": candidate.parent_experiment_id,
        "parent_result_id": candidate.parent_result_id,
        "parent_fingerprint": candidate.parent_fingerprint,
        "ablation_experiment_id": ",".join(exp_ids or all_exp_ids),
        "rule_type": candidate.rule_type,
        "rule_key": candidate.rule_key,
        "rule_context": json.dumps(candidate.context, sort_keys=True),
        "original_loss_ratio": original,
        "ablation_best_loss_ratio": ablated_best,
        "effect_size": effect_size,
        "original_stage1_passed": 1,
        "ablation_stage1_pass_count": int(stats["stage1_pass_count"]),
        "ablation_total": int(stats["total"]),
        "outcome": outcome,
        "confidence": confidence,
        "evidence_json": json.dumps(
            {
                "ablation_outcome": ablation_outcome,
                "ablation_experiment_ids": exp_ids,
                "historical_experiment_ids": historical_exp_ids,
                "hypothesis": candidate.hypothesis,
                "stats": stats,
            },
            sort_keys=True,
        ),
    }


def find_historical_ablation_observations(
    nb: Any,
    *,
    candidate: CausalAblationCandidate,
    graphs: Sequence[ComputationGraph],
    max_rows_per_fingerprint: int = 12,
) -> list[dict[str, Any]]:
    """Attach prior rows whose fingerprints match proposed ablation children."""

    fingerprints = tuple(dict.fromkeys(graph.fingerprint() for graph in graphs))
    if not fingerprints:
        return []
    placeholders = ",".join("?" for _ in fingerprints)
    params: list[Any] = list(fingerprints)
    params.append(candidate.parent_result_id)
    rows = nb.conn.execute(
        f"""SELECT result_id, experiment_id, graph_fingerprint, timestamp,
                  stage0_passed, stage05_passed, stage1_passed,
                  loss_ratio, final_loss, model_source, trust_label,
                  comparability_label
           FROM program_results_compat
           WHERE graph_fingerprint IN ({placeholders})
             AND result_id <> ?
           ORDER BY graph_fingerprint ASC, timestamp DESC""",
        tuple(params),
    ).fetchall()
    out: list[dict[str, Any]] = []
    per_fp: dict[str, int] = {}
    for row in rows:
        fp = str(row["graph_fingerprint"] or "")
        count = per_fp.get(fp, 0)
        if count >= max(1, int(max_rows_per_fingerprint or 1)):
            continue
        per_fp[fp] = count + 1
        out.append(_observation_from_row(row, candidate, source="historical"))
    return out


def find_executed_ablation_observations(
    nb: Any,
    *,
    candidate: CausalAblationCandidate,
    experiment_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Return observations produced by newly executed ablation experiments."""

    exp_ids = tuple(dict.fromkeys(str(exp_id) for exp_id in experiment_ids if exp_id))
    if not exp_ids:
        return []
    placeholders = ",".join("?" for _ in exp_ids)
    rows = nb.conn.execute(
        f"""SELECT result_id, experiment_id, graph_fingerprint, timestamp,
                  stage0_passed, stage05_passed, stage1_passed,
                  loss_ratio, final_loss, model_source, trust_label,
                  comparability_label
           FROM program_results_compat
           WHERE experiment_id IN ({placeholders})
           ORDER BY timestamp DESC""",
        exp_ids,
    ).fetchall()
    return [
        _observation_from_row(row, candidate, source="executed")
        for row in rows
        if str(row["result_id"] or "")
    ]


def causal_generation_adjustments(
    nb: Any,
    *,
    min_confidence: float = 0.35,
    limit: int = 200,
) -> dict[str, Any]:
    """Return conservative generation priors from causal ablation evidence."""

    try:
        rows = nb.get_causal_rule_evidence(limit=limit)
    except AttributeError:
        return {
            "op_weights": {},
            "slot_motif_multipliers": {},
            "slot_motif_denylist": {},
        }

    op_weights: dict[str, float] = {}
    slot_motif_multipliers: dict[str, dict[str, float]] = {}
    slot_motif_denylist: dict[str, set[str]] = {}
    aggregates: dict[tuple[str, str], dict[str, float]] = {}

    for row in rows:
        confidence = _optional_float(row.get("confidence")) or 0.0
        outcome = str(row.get("outcome") or "")
        rule_type = str(row.get("rule_type") or "")
        rule_key = str(row.get("rule_key") or "")
        if not rule_key:
            continue
        supported = outcome == "supported"
        refuted = outcome.startswith("refuted")
        if not supported and not refuted:
            continue
        effect = abs(_optional_float(row.get("effect_size")) or 0.0)
        # Low-confidence rows still carry signal when repeated. Blend the
        # measured confidence with effect magnitude, then aggregate by rule.
        weight = max(confidence, min(effect / 0.20, 0.35), 0.02)
        bucket = aggregates.setdefault(
            (rule_type, rule_key),
            {
                "supported": 0.0,
                "refuted": 0.0,
                "max_confidence": 0.0,
                "count": 0.0,
            },
        )
        bucket["count"] += 1.0
        bucket["max_confidence"] = max(bucket["max_confidence"], confidence)
        if supported:
            bucket["supported"] += weight
        else:
            bucket["refuted"] += weight

    for (rule_type, rule_key), bucket in aggregates.items():
        supported_mass = bucket["supported"]
        refuted_mass = bucket["refuted"]
        net_mass = supported_mass - refuted_mass
        directional_mass = abs(net_mass)
        if (
            directional_mass < min_confidence
            and bucket["max_confidence"] < min_confidence
        ):
            continue
        if max(supported_mass, refuted_mass) <= 0.0:
            continue
        supported = net_mass > 0.0
        confidence = min(1.0, directional_mass)

        if supported:
            multiplier = 1.0 + min(0.6, 0.5 * confidence)
        else:
            multiplier = max(0.25, 1.0 - min(0.7, 0.6 * confidence))

        if rule_type == "op":
            _merge_multiplier(op_weights, rule_key, multiplier)
        elif rule_type == "op_pair":
            for op_name in rule_key.split("->", 1):
                if op_name and op_name not in SCAFFOLD_OPS:
                    _merge_multiplier(op_weights, op_name, multiplier**0.5)
        elif rule_type == "slot_motif":
            slot_key, motif_name = _split_slot_motif_rule(rule_key)
            if not slot_key or not motif_name:
                continue
            if supported:
                slot_motif_multipliers.setdefault(slot_key, {})
                _merge_multiplier(
                    slot_motif_multipliers[slot_key],
                    motif_name,
                    multiplier,
                )
            elif confidence >= 0.65:
                slot_motif_denylist.setdefault(slot_key, set()).add(motif_name)
            else:
                slot_motif_multipliers.setdefault(slot_key, {})
                _merge_multiplier(
                    slot_motif_multipliers[slot_key],
                    motif_name,
                    multiplier,
                )

    return {
        "op_weights": op_weights,
        "slot_motif_multipliers": slot_motif_multipliers,
        "slot_motif_denylist": {
            key: frozenset(values) for key, values in slot_motif_denylist.items()
        },
    }


def _rank_graph_signals(graph: ComputationGraph) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    ops = _graph_ops(graph)
    pairs = _graph_op_pairs(graph)
    slot_signals = _slot_signals(graph)

    for slot in slot_signals:
        signals.append(slot)
    for pair in pairs:
        left, right = pair.split("->", 1)
        if left in SCAFFOLD_OPS and right in SCAFFOLD_OPS:
            continue
        signals.append(
            {
                "rule_type": "op_pair",
                "rule_key": pair,
                "hypothesis": f"op_pair:{pair} ops:{left} {right}",
                "context": {"op_pair": pair},
            }
        )
    for op_name in ops:
        if op_name in SCAFFOLD_OPS:
            continue
        signals.append(
            {
                "rule_type": "op",
                "rule_key": op_name,
                "hypothesis": f"op:{op_name}",
                "context": {"op": op_name},
            }
        )

    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for signal in signals:
        unique.setdefault((signal["rule_type"], signal["rule_key"]), signal)
    return sorted(unique.values(), key=_signal_priority, reverse=True)


def _graph_ops(graph: ComputationGraph) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for node_id in graph.topological_order():
        node = graph.nodes[node_id]
        if node.is_input or node.op_name in seen:
            continue
        seen.add(node.op_name)
        out.append(node.op_name)
    return out


def _graph_op_pairs(graph: ComputationGraph) -> list[str]:
    by_id = graph.nodes
    pairs: set[str] = set()
    for node in by_id.values():
        if node.is_input:
            continue
        for input_id in node.input_ids:
            parent = by_id.get(input_id)
            if parent is None or parent.is_input:
                continue
            pairs.add(f"{parent.op_name}->{node.op_name}")
    return sorted(pairs)


def _slot_signals(graph: ComputationGraph) -> list[dict[str, Any]]:
    usage = (getattr(graph, "metadata", {}) or {}).get("template_slot_usage")
    if not isinstance(usage, list):
        return []
    motif_ops = _motif_ops_by_name()
    out: list[dict[str, Any]] = []
    for item in usage:
        if not isinstance(item, Mapping):
            continue
        motif = str(item.get("selected_motif") or "").strip()
        if not motif:
            continue
        ops = tuple(op for op in motif_ops.get(motif, ()) if op not in SCAFFOLD_OPS)
        if not ops:
            continue
        template = str(item.get("template_name") or "").strip()
        slot_index = item.get("slot_index")
        slot_key = str(item.get("slot_key") or f"{template}.slot{slot_index}")
        rule_key = f"{slot_key}:{motif}"
        out.append(
            {
                "rule_type": "slot_motif",
                "rule_key": rule_key,
                "hypothesis": f"slot_motif:{rule_key} ops:{' '.join(ops)}",
                "context": {
                    "template_name": template,
                    "slot_key": slot_key,
                    "slot_index": slot_index,
                    "selected_motif": motif,
                    "selected_motif_class": item.get("selected_motif_class"),
                    "ops": ops,
                    "wildcard": bool(item.get("wildcard")),
                },
            }
        )
    return out


def _motif_ops_by_name() -> dict[str, tuple[str, ...]]:
    from research.synthesis.motifs import ALL_MOTIFS

    out: dict[str, tuple[str, ...]] = {}
    for motif in ALL_MOTIFS:
        ops: list[str] = []
        for step in getattr(motif, "steps", ()):
            op_name = str(getattr(step, "op_name", "") or "")
            if op_name:
                ops.append(op_name)
        out[str(motif.name)] = tuple(dict.fromkeys(ops))
    return out


def _signal_priority(signal: Mapping[str, Any]) -> tuple[int, int, int, str]:
    rule_type = str(signal.get("rule_type") or "")
    rule_key = str(signal.get("rule_key") or "")
    type_score = {"slot_motif": 3, "op_pair": 2, "op": 1}.get(rule_type, 0)
    token_score = sum(1 for token in HIGH_VALUE_TOKENS if token in rule_key)
    non_scaffold_score = 0 if rule_key in SCAFFOLD_OPS else 1
    return (type_score, token_score, non_scaffold_score, rule_key)


def _ablation_result_stats(nb: Any, experiment_ids: list[str]) -> dict[str, Any]:
    if not experiment_ids:
        return {"total": 0, "stage1_pass_count": 0, "best_loss_ratio": None}
    placeholders = ",".join("?" for _ in experiment_ids)
    row = nb.conn.execute(
        f"""SELECT COUNT(*) AS total,
                  SUM(CASE WHEN COALESCE(stage1_passed, 0) THEN 1 ELSE 0 END)
                      AS stage1_pass_count,
                  MIN(loss_ratio) AS best_loss_ratio
           FROM program_results_compat
           WHERE experiment_id IN ({placeholders})""",
        tuple(experiment_ids),
    ).fetchone()
    if row is None:
        return {"total": 0, "stage1_pass_count": 0, "best_loss_ratio": None}
    return {
        "total": int(row["total"] or 0),
        "stage1_pass_count": int(row["stage1_pass_count"] or 0),
        "best_loss_ratio": _optional_float(row["best_loss_ratio"]),
        "unique_fingerprint_count": int(row["total"] or 0),
    }


def _observation_from_row(
    row: Any,
    candidate: CausalAblationCandidate,
    *,
    source: str,
) -> dict[str, Any]:
    child_experiment_id = str(row["experiment_id"] or "")
    child_result_id = str(row["result_id"] or "")
    child_fingerprint = str(row["graph_fingerprint"] or "")
    return {
        "parent_result_id": candidate.parent_result_id,
        "parent_experiment_id": candidate.parent_experiment_id,
        "parent_fingerprint": candidate.parent_fingerprint,
        "child_result_id": child_result_id,
        "child_experiment_id": child_experiment_id,
        "child_fingerprint": child_fingerprint,
        "ablation_experiment_id": (
            child_experiment_id if source == "executed" else None
        ),
        "source": source,
        "rule_type": candidate.rule_type,
        "rule_key": candidate.rule_key,
        "timestamp": _optional_float(row["timestamp"]),
        "stage0_passed": int(row["stage0_passed"] or 0),
        "stage05_passed": int(row["stage05_passed"] or 0),
        "stage1_passed": int(row["stage1_passed"] or 0),
        "loss_ratio": _optional_float(row["loss_ratio"]),
        "final_loss": _optional_float(row["final_loss"]),
        "model_source": row["model_source"],
        "trust_label": row["trust_label"],
        "comparability_label": row["comparability_label"],
        "provenance": {
            "source": source,
            "child_result_id": child_result_id,
            "child_experiment_id": child_experiment_id,
            "child_fingerprint": child_fingerprint,
            "parent_result_id": candidate.parent_result_id,
            "rule_type": candidate.rule_type,
            "rule_key": candidate.rule_key,
        },
    }


def _observation_stats(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(observations)
    stage1 = sum(1 for obs in observations if int(obs.get("stage1_passed") or 0))
    best_loss: Optional[float] = None
    fingerprints: set[str] = set()
    source_counts: dict[str, int] = {}
    for obs in observations:
        fp = str(obs.get("child_fingerprint") or obs.get("graph_fingerprint") or "")
        if fp:
            fingerprints.add(fp)
        source = str(obs.get("source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
        loss = _optional_float(obs.get("loss_ratio"))
        if loss is not None and (best_loss is None or loss < best_loss):
            best_loss = loss
    return {
        "total": total,
        "stage1_pass_count": stage1,
        "best_loss_ratio": best_loss,
        "unique_fingerprint_count": len(fingerprints),
        "source_counts": source_counts,
    }


def _classify_effect(
    *,
    ablation_outcome: str,
    effect_size: Optional[float],
    ablation_total: int,
    ablation_stage1: int,
) -> str:
    if ablation_total <= 0:
        return "inconclusive"
    if effect_size is not None:
        if effect_size >= 0.02:
            return "supported"
        if effect_size <= -0.02:
            return "refuted_ablation_improved"
    if ablation_stage1 == 0:
        return "supported"
    return "inconclusive"


def _effect_confidence(
    *,
    effect_size: Optional[float],
    ablation_total: int,
    ablation_stage1: int,
) -> float:
    if ablation_total <= 0:
        return 0.0
    support = ablation_total / (ablation_total + 4.0)
    pass_drop = (
        1.0
        if ablation_stage1 == 0
        else max(0.0, 1.0 - ablation_stage1 / ablation_total)
    )
    delta = min(abs(float(effect_size or 0.0)) / 0.10, 1.0)
    return round(
        max(support * 0.35, support * ((0.65 * delta) + (0.35 * pass_drop))), 4
    )


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _merge_multiplier(target: dict[str, float], key: str, multiplier: float) -> None:
    current = float(target.get(key, 1.0))
    if multiplier >= 1.0:
        target[key] = max(current, float(multiplier))
    else:
        target[key] = min(current, float(multiplier))


def _split_slot_motif_rule(rule_key: str) -> tuple[str, str]:
    if ":" not in rule_key:
        return "", ""
    slot_key, motif_name = rule_key.rsplit(":", 1)
    return slot_key.strip(), motif_name.strip()
