"""Recommendation, evidence, enrichment, and readiness helpers."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import time as _time

from ..shared_utils import safe_float

if TYPE_CHECKING:
    from ..notebook import LabNotebook

_logger = logging.getLogger(__name__)


def infer_tier_for_program(nb: LabNotebook, program: dict) -> str:
    """Infer tier for a raw program_results row by checking the leaderboard."""
    result_id = program.get("result_id")
    if not result_id:
        return "screening"
    row = nb.conn.execute(
        "SELECT tier FROM leaderboard WHERE result_id = ?", (result_id,)
    ).fetchone()
    return row["tier"] if row else "screening"


_tier_cache: dict = {}
_tier_cache_ts: float = 0.0
_TIER_CACHE_TTL: float = 60.0


def count_discovery_tiers(nb: LabNotebook) -> dict:
    """Count discovery rows by tier, excluding references from stage buckets.

    Uses stage-based counting: entries that *passed* a stage count for that
    stage, even if their tier column has since been promoted further.
    """
    global _tier_cache, _tier_cache_ts
    now = _time.monotonic()
    if _tier_cache and (now - _tier_cache_ts) < _TIER_CACHE_TTL:
        return _tier_cache
    _NON_REF = "COALESCE(is_reference, 0) = 0"
    rows = nb.conn.execute(
        f"SELECT tier, COUNT(*) AS cnt FROM leaderboard WHERE {_NON_REF} GROUP BY tier"
    ).fetchall()
    tier_counts = {r["tier"]: r["cnt"] for r in rows}
    # Stage-based counts: investigation/validation count entries that *reached*
    # and *passed* each stage, regardless of current tier value.
    inv_row = nb.conn.execute(
        f"SELECT COUNT(*) AS cnt FROM leaderboard "
        f"WHERE {_NON_REF} AND investigation_passed = 1"
    ).fetchone()
    val_row = nb.conn.execute(
        f"SELECT COUNT(*) AS cnt FROM leaderboard "
        f"WHERE {_NON_REF} AND validation_passed = 1"
    ).fetchone()
    counts = {
        "screening": int(tier_counts.get("screening", 0) or 0),
        "screened_out": int(tier_counts.get("screened_out", 0) or 0),
        "investigation": int(inv_row["cnt"] if inv_row else 0),
        "validation": int(val_row["cnt"] if val_row else 0),
        "breakthrough": int(tier_counts.get("breakthrough", 0) or 0),
    }
    ref_row = nb.conn.execute(
        "SELECT COUNT(*) AS cnt FROM leaderboard WHERE COALESCE(is_reference, 0) = 1"
    ).fetchone()
    counts["references"] = int(ref_row["cnt"] if ref_row else 0)
    counts["all"] = sum(
        int(tier_counts.get(tier, 0) or 0)
        for tier in (
            "screening",
            "screened_out",
            "investigation",
            "validation",
            "breakthrough",
        )
    )
    total_s1 = nb.conn.execute(
        "SELECT COUNT(*) AS cnt FROM program_results WHERE stage1_passed = 1"
    ).fetchone()
    counts["total_survivors"] = total_s1["cnt"] if total_s1 else 0
    _tier_cache = counts
    _tier_cache_ts = now
    return counts


def _rank_label(delta: Optional[int], seen_runs: int) -> str:
    if seen_runs <= 1:
        return "new"
    if delta is None:
        return "new"
    if delta == 0:
        return "stable"
    return "up" if delta < 0 else "down"


def compute_cross_run_stability(nb: LabNotebook, top_programs: Any) -> dict:
    """Compute rank movement for top candidates across recent experiments."""
    normalized_programs: list[dict[str, Any]]
    if isinstance(top_programs, str):
        normalized_programs = [{"graph_fingerprint": top_programs}]
    elif isinstance(top_programs, dict):
        normalized_programs = [top_programs]
    elif isinstance(top_programs, list):
        normalized_programs = [p for p in top_programs if isinstance(p, dict)]
    else:
        normalized_programs = []

    fingerprints = [
        fp
        for fp in (
            str(program.get("graph_fingerprint") or "").strip()
            for program in normalized_programs[:20]
        )
        if fp
    ]
    if not fingerprints:
        return {
            "summary": {"stable": 0, "up": 0, "down": 0, "new": 0},
            "candidates": [],
            "window_size": 0,
        }

    experiments = [
        exp
        for exp in nb.get_recent_experiments(40)
        if exp.get("status") == "completed" and exp.get("experiment_id")
    ]
    if not experiments:
        return {
            "summary": {"stable": 0, "up": 0, "down": 0, "new": 0},
            "candidates": [],
            "window_size": 0,
        }

    experiment_ids = [str(exp["experiment_id"]) for exp in experiments]
    experiment_order = {
        experiment_id: index for index, experiment_id in enumerate(experiment_ids)
    }
    placeholders = ",".join("?" for _ in experiment_ids)
    fp_placeholders = ",".join("?" for _ in fingerprints)
    rows = nb.conn.execute(
        f"""
        WITH deduped_programs AS (
            SELECT
                experiment_id,
                graph_fingerprint,
                loss_ratio,
                timestamp,
                ROW_NUMBER() OVER (
                    PARTITION BY experiment_id, graph_fingerprint
                    ORDER BY loss_ratio ASC, timestamp DESC
                ) AS fingerprint_rank
            FROM program_results
            WHERE stage1_passed = 1
              AND loss_ratio IS NOT NULL
              AND experiment_id IN ({placeholders})
              AND graph_fingerprint IN ({fp_placeholders})
        ),
        ranked_programs AS (
            SELECT
                experiment_id,
                graph_fingerprint,
                ROW_NUMBER() OVER (
                    PARTITION BY experiment_id
                    ORDER BY loss_ratio ASC, timestamp DESC
                ) AS rank
            FROM deduped_programs
            WHERE fingerprint_rank = 1
        )
        SELECT experiment_id, graph_fingerprint, rank
        FROM ranked_programs
        ORDER BY experiment_id, rank
        """,
        (*experiment_ids, *fingerprints),
    ).fetchall()

    history_by_fingerprint: dict[str, list[dict[str, Any]]] = {
        fp: [] for fp in fingerprints
    }
    experiment_ts = {
        str(exp["experiment_id"]): exp.get("timestamp") for exp in experiments
    }
    for row in rows:
        fp = str(row["graph_fingerprint"] or "")
        if not fp:
            continue
        history_by_fingerprint.setdefault(fp, []).append(
            {
                "experiment_id": row["experiment_id"],
                "timestamp": experiment_ts.get(str(row["experiment_id"])),
                "rank": row["rank"],
            }
        )
    for history in history_by_fingerprint.values():
        history.sort(
            key=lambda item: experiment_order.get(str(item["experiment_id"]), 10**9)
        )

    candidates = []
    summary = {"stable": 0, "up": 0, "down": 0, "new": 0}
    for index, program in enumerate(normalized_programs[:20], start=1):
        fp = program.get("graph_fingerprint")
        if not fp:
            continue
        history = history_by_fingerprint.get(str(fp), [])

        seen_runs = len(history)
        latest_rank = history[0]["rank"] if history else None
        previous_rank = history[1]["rank"] if len(history) > 1 else None
        delta = None
        if latest_rank is not None and previous_rank is not None:
            delta = latest_rank - previous_rank
        trend = _rank_label(delta, seen_runs)
        summary[trend] = summary.get(trend, 0) + 1

        candidates.append(
            {
                "result_id": program.get("result_id"),
                "graph_fingerprint": fp,
                "current_overall_rank": index,
                "seen_runs": seen_runs,
                "latest_rank": latest_rank,
                "previous_rank": previous_rank,
                "rank_delta": delta,
                "trend": trend,
            }
        )

    return {
        "summary": summary,
        "candidates": candidates,
        "window_size": len(experiments),
    }


def compute_recommendation(program: dict, leaderboard_entry: Optional[dict]) -> dict:
    """Deterministic next-action recommendation based on tier and pass/fail."""
    tier = (leaderboard_entry or {}).get("tier", "screening")
    s1 = program.get("stage1_passed", False)

    if not s1:
        return {
            "action": "archive",
            "rationale": "Program did not pass Stage 1 learning evaluation.",
            "confidence": "high",
        }
    if tier == "breakthrough":
        return {
            "action": "publish",
            "rationale": "Breakthrough-tier architecture with validated performance.",
            "confidence": "high",
        }
    if tier == "validation":
        passed = (leaderboard_entry or {}).get("validation_passed", False)
        if passed:
            return {
                "action": "scale up or publish",
                "rationale": "Validation passed with multi-seed stability confirmed.",
                "confidence": "high",
                "bias_check": "grammar_independence_verified",
            }
        return {
            "action": "re-validate",
            "rationale": "Validation tier but not yet passed; may need more seeds or longer training.",
            "confidence": "medium",
        }
    if tier == "investigation":
        passed = (leaderboard_entry or {}).get("investigation_passed", False)
        if passed:
            return {
                "action": "validate",
                "rationale": "Investigation passed; promote to validation for multi-seed confirmation.",
                "confidence": "high",
            }
        return {
            "action": "re-investigate or archive",
            "rationale": "Investigation tier but not yet passed; re-run or archive if stale.",
            "confidence": "medium",
        }
    return {
        "action": "investigate",
        "rationale": "Screening-tier candidate; needs deeper investigation to confirm potential.",
        "confidence": "medium",
    }


def promotion_evidence_for_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    seen_runs = int(((entry.get("cross_run_stability") or {}).get("seen_runs") or 0))
    baseline_ratio = safe_float(entry.get("validation_baseline_ratio"))
    std = safe_float(entry.get("validation_multi_seed_std"))

    # Replication aggregates from leaderboard
    replication_n = int(entry.get("replication_n") or 0)
    replication_loss_mean = safe_float(entry.get("replication_loss_mean"))
    replication_loss_std = safe_float(entry.get("replication_loss_std"))
    replication_gap = safe_float(entry.get("replication_best_vs_mean_gap"))

    checks = {
        "baselineEvidence": baseline_ratio is not None,
        "baselineBeat": baseline_ratio is not None and baseline_ratio < 1.0,
        "multiSeedStd": std is not None,
        "boundedStd": std is not None and std <= 0.12,
        "ckaArtifactBacked": entry.get("cka_source") == "artifact",
        "repeatObserved": seen_runs >= 3,
        "replicatedEvidence": replication_n >= 3,
    }
    evidence_count = sum(1 for ok in checks.values() if ok)
    total_checks = len(checks)
    completeness = evidence_count / total_checks if total_checks else 0.0

    std_signal = 0.0
    if std is not None:
        if std <= 0.05:
            std_signal = 1.0
        elif std <= 0.12:
            std_signal = 0.65
        elif std <= 0.2:
            std_signal = 0.35
        else:
            std_signal = 0.1

    if seen_runs >= 5:
        repeat_signal = 1.0
    elif seen_runs >= 3:
        repeat_signal = 0.65
    elif seen_runs >= 2:
        repeat_signal = 0.4
    elif seen_runs >= 1:
        repeat_signal = 0.2
    else:
        repeat_signal = 0.0

    margin_signal = 0.0
    if baseline_ratio is not None:
        margin = 1.0 - baseline_ratio
        if margin >= 0.1:
            margin_signal = 1.0
        elif margin > 0:
            margin_signal = 0.7
        else:
            margin_signal = 0.15

    score = round(
        (
            completeness * 0.5
            + std_signal * 0.2
            + repeat_signal * 0.2
            + margin_signal * 0.1
        )
        * 100
    )
    missing = [name for name, ok in checks.items() if not ok]

    # Replication summary: use mean±std instead of single best run
    insufficient_replication = replication_n < 3
    replication_summary = None
    if replication_n >= 1 and replication_loss_mean is not None:
        replication_summary = {
            "n_runs": replication_n,
            "loss_mean": round(replication_loss_mean, 4),
            "loss_std": round(replication_loss_std, 4)
            if replication_loss_std is not None
            else None,
            "best_vs_mean_gap": round(replication_gap, 4)
            if replication_gap is not None
            else None,
            "sufficient": not insufficient_replication,
        }

    return {
        "score": score,
        "seen_runs": seen_runs,
        "std": std,
        "evidence_count": evidence_count,
        "total_checks": total_checks,
        "missing": missing,
        "replication": replication_summary,
        "insufficient_replication": insufficient_replication,
    }


def decision_gate_for_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    investigation_robustness = safe_float(entry.get("investigation_robustness"))
    validation_baseline_ratio = safe_float(entry.get("validation_baseline_ratio"))
    validation_multi_seed_std = safe_float(entry.get("validation_multi_seed_std"))

    checks = {
        "screeningEvidence": entry.get("screening_loss_ratio") is not None
        and entry.get("screening_novelty") is not None,
        "investigationEvidence": entry.get("investigation_loss_ratio") is not None
        and entry.get("investigation_robustness") is not None,
        "robustnessFloor": investigation_robustness is not None
        and investigation_robustness >= 0.5,
        "validationEvidence": (
            entry.get("validation_loss_ratio") is not None
            and entry.get("validation_baseline_ratio") is not None
            and entry.get("validation_multi_seed_std") is not None
        ),
        "baselineBeatsReference": validation_baseline_ratio is not None
        and validation_baseline_ratio < 1.0,
        "consistencyBounded": validation_multi_seed_std is not None
        and validation_multi_seed_std <= 0.12,
    }
    decision_ready = all(checks.values())
    missing = [name for name, ok in checks.items() if not ok]
    return {
        "decision_ready": decision_ready,
        "missing": missing,
    }


def build_scale_up_templates_for_result(
    result_id: Optional[str],
) -> List[Dict[str, Any]]:
    normalized = str(result_id or "").strip()
    if not normalized:
        return []
    return [
        {
            "template_id": "multi_seed_stress",
            "title": "Multi-seed stress validation",
            "description": "Run deeper multi-seed validation to confirm consistency and variance bounds.",
            "start_payload": {
                "mode": "validation",
                "result_ids": [normalized],
                "validation_steps": 12000,
                "validation_n_seeds": 7,
                "validation_batch_size": 8,
                "validation_seq_len": 512,
            },
        },
        {
            "template_id": "robustness_recheck",
            "title": "Robustness re-check",
            "description": "Re-run investigation-level robustness checks before heavier scale-up spend.",
            "start_payload": {
                "mode": "investigation",
                "result_ids": [normalized],
                "investigation_steps": 3500,
                "investigation_batch_size": 4,
                "n_training_programs": 4,
            },
        },
        {
            "template_id": "efficiency_scale_up",
            "title": "Scale-up + efficiency profile",
            "description": "Run scale-up training with one-shot pruning baseline to profile efficiency/quality trade-offs.",
            "start_payload": {
                "mode": "scale_up",
                "result_ids": [normalized],
                "scale_up_steps": 8000,
                "scale_up_batch_size": 8,
                "scale_up_seq_len": 512,
                "one_shot_pruning_baseline": True,
                "one_shot_pruning_method": "wanda",
                "one_shot_pruning_sparsity": 0.5,
            },
        },
    ]


def build_reproducibility_workflow(
    repro_packet: Dict[str, Any],
    scale_up_templates: List[Dict[str, Any]],
    result_id: Optional[str] = None,
    graph_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    packet = repro_packet or {}
    ready_count = int(packet.get("ready_count") or 0)
    total_checks = int(packet.get("total_checks") or 6)
    missing = {str(item) for item in (packet.get("missing") or []) if item}

    template_by_id = {
        str(template.get("template_id")): template
        for template in (scale_up_templates or [])
        if isinstance(template, dict) and template.get("template_id")
    }

    def _payload(template_id: str) -> Optional[Dict[str, Any]]:
        template = template_by_id.get(template_id)
        if not template:
            return None
        payload = template.get("start_payload")
        return dict(payload) if isinstance(payload, dict) else None

    checks = [
        (
            "result_id",
            "Result identifier captured",
            None,
            "Program result must be persisted before repro closure.",
        ),
        (
            "graph_fingerprint",
            "Fingerprint captured",
            None,
            "Graph fingerprint is required for cross-run traceability.",
        ),
        (
            "arch_spec",
            "Architecture spec recorded",
            "robustness_recheck",
            "Re-run robustness check if architecture metadata is incomplete.",
        ),
        (
            "baseline_ratio",
            "Baseline ratio measured",
            "multi_seed_stress",
            "Run multi-seed validation to compute baseline ratio.",
        ),
        (
            "multi_seed_std",
            "Multi-seed variance measured",
            "multi_seed_stress",
            "Run multi-seed validation to measure stability variance.",
        ),
        (
            "cka_artifact",
            "Artifact-backed CKA recorded",
            "efficiency_scale_up",
            "After run completion, stamp CKA artifact integrity in artifact references.",
        ),
    ]

    steps: List[Dict[str, Any]] = []
    for check_id, label, template_id, guidance in checks:
        is_complete = check_id not in missing
        step: Dict[str, Any] = {
            "check_id": check_id,
            "label": label,
            "status": "complete" if is_complete else "missing",
            "guidance": guidance,
        }
        if result_id:
            step["result_id"] = result_id
        if graph_fingerprint:
            step["graph_fingerprint"] = graph_fingerprint
        if not is_complete and template_id:
            payload = _payload(template_id)
            if payload:
                step["action_label"] = "Run template"
                step["start_payload"] = payload
        steps.append(step)

    next_actions = [
        {
            "check_id": step.get("check_id"),
            "label": step.get("label"),
            "action_label": step.get("action_label"),
            "start_payload": step.get("start_payload"),
            "guidance": step.get("guidance"),
        }
        for step in steps
        if step.get("status") == "missing"
    ][:3]

    return {
        "status": "ready" if ready_count >= total_checks else "in_progress",
        "ready_count": ready_count,
        "total_checks": total_checks,
        "progress_label": f"{ready_count}/{total_checks}",
        "remaining": max(0, total_checks - ready_count),
        "steps": steps,
        "next_actions": next_actions,
        "result_id": result_id,
        "graph_fingerprint": graph_fingerprint,
    }


def annotate_qkv_usage(programs: list, analytics) -> None:
    for program in programs:
        if not isinstance(program, dict):
            continue
        qkv_usage = analytics.qkv_usage_enum(program)
        program["qkv_usage"] = qkv_usage
        program["uses_qkv"] = qkv_usage != "qkv_free"
        program["compression_metrics"] = analytics.canonical_compression_metrics(
            program
        )
        program["reproducibility_packet"] = analytics.reproducibility_packet_status(
            program
        )


def _empty_breakthrough_readiness() -> Dict[str, Any]:
    return {
        "breakthrough_count": 0,
        "decision_ready_count": 0,
        "high_confidence_count": 0,
        "full_repro_packet_count": 0,
        "artifact_cka_count": 0,
        "epic_switch_recommendation": {
            "action": "stay_current_epic",
            "reason": "No breakthrough-tier candidates are available yet.",
        },
        "top_candidates": [],
        "scale_up_templates": [],
        "reproducibility_workflow": None,
    }


def _evaluate_breakthrough_entry(
    entry: Dict[str, Any],
    stability_by_result: Dict[str, Dict[str, Any]],
    analytics: Any,
) -> Dict[str, Any]:
    row = dict(entry)
    row["cross_run_stability"] = stability_by_result.get(
        row.get("result_id"),
        {
            "trend": "unknown",
            "seen_runs": 0,
            "latest_rank": None,
            "previous_rank": None,
            "rank_delta": None,
        },
    )
    row["reproducibility_packet"] = analytics.reproducibility_packet_status(row)
    promotion = promotion_evidence_for_entry(row)
    gate = decision_gate_for_entry(row)
    scale_templates = build_scale_up_templates_for_result(row.get("result_id"))
    reproducibility_workflow = build_reproducibility_workflow(
        row["reproducibility_packet"],
        scale_templates,
        result_id=row.get("result_id"),
        graph_fingerprint=row.get("graph_fingerprint"),
    )
    return {
        "result_id": row.get("result_id"),
        "architecture_family": row.get("architecture_family"),
        "composite_score": safe_float(row.get("composite_score"), 0.0),
        "promotion_confidence_score": promotion["score"],
        "seen_runs": promotion["seen_runs"],
        "decision_ready": gate["decision_ready"],
        "decision_missing": gate["missing"],
        "repro_packet": row["reproducibility_packet"],
        "cka_source": row.get("cka_source"),
        "scale_up_templates": scale_templates,
        "reproducibility_workflow": reproducibility_workflow,
    }


def _switch_recommendation(evaluated: List[Dict[str, Any]]) -> Dict[str, str]:
    switch_ready = any(
        row.get("decision_ready")
        and int(row.get("promotion_confidence_score") or 0) >= 75
        and (row.get("repro_packet") or {}).get("status") == "ready"
        and row.get("cka_source") == "artifact"
        for row in evaluated
    )
    if switch_ready:
        return {
            "action": "switch_to_scale_up_epic",
            "reason": "At least one breakthrough candidate meets decision, confidence, repro, and artifact-backed CKA gates.",
        }
    return {
        "action": "stay_current_epic",
        "reason": "Breakthrough evidence is still incomplete; continue hardening reproducibility and validation before switching epics.",
    }


def compute_breakthrough_production_readiness(
    nb: LabNotebook, analytics: Any
) -> Dict[str, Any]:
    breakthroughs = nb.get_leaderboard(
        tier="breakthrough",
        limit=20,
        sort_by="composite_score",
        include_references=False,
        trusted_only=True,
    )
    if not breakthroughs:
        return _empty_breakthrough_readiness()

    stability = compute_cross_run_stability(
        nb, nb.get_top_programs(20, sort_by="loss_ratio")
    )
    stability_by_result = {
        c.get("result_id"): c
        for c in stability.get("candidates", [])
        if c.get("result_id")
    }

    evaluated = [
        _evaluate_breakthrough_entry(entry, stability_by_result, analytics)
        for entry in breakthroughs
    ]

    breakthrough_count = len(evaluated)
    decision_ready_count = sum(1 for row in evaluated if row.get("decision_ready"))
    high_confidence_count = sum(
        1 for row in evaluated if int(row.get("promotion_confidence_score") or 0) >= 75
    )
    full_repro_packet_count = sum(
        1
        for row in evaluated
        if (row.get("repro_packet") or {}).get("status") == "ready"
    )
    artifact_cka_count = sum(
        1 for row in evaluated if row.get("cka_source") == "artifact"
    )
    recommendation = _switch_recommendation(evaluated)

    top_candidates = sorted(
        evaluated,
        key=lambda row: (
            int(bool(row.get("decision_ready"))),
            int(row.get("promotion_confidence_score") or 0),
            safe_float(row.get("composite_score"), 0.0),
        ),
        reverse=True,
    )[:3]
    scale_up_templates = (
        top_candidates[0].get("scale_up_templates", []) if top_candidates else []
    )

    return {
        "breakthrough_count": breakthrough_count,
        "decision_ready_count": decision_ready_count,
        "high_confidence_count": high_confidence_count,
        "full_repro_packet_count": full_repro_packet_count,
        "artifact_cka_count": artifact_cka_count,
        "epic_switch_recommendation": recommendation,
        "top_candidates": top_candidates,
        "scale_up_templates": scale_up_templates,
        "reproducibility_workflow": (
            top_candidates[0].get("reproducibility_workflow")
            if top_candidates
            else None
        ),
    }


def _is_reference_like(entry: Dict[str, Any]) -> bool:
    if not isinstance(entry, dict):
        return False
    rid = str(entry.get("result_id") or "").strip().lower()
    model_source = str(entry.get("model_source") or "").strip().lower()
    reference_name = str(entry.get("reference_name") or "").strip()
    return (
        bool(entry.get("is_reference"))
        or bool(reference_name)
        or model_source == "reference"
        or rid.startswith("ref_")
    )


def _append_breakthrough_actions(
    actions: List[Dict[str, Any]], nb: LabNotebook
) -> None:
    breakthroughs = nb.get_leaderboard(
        tier="breakthrough",
        limit=5,
        sort_by="composite_score",
        include_references=False,
        trusted_only=True,
    )
    for entry in breakthroughs:
        if _is_reference_like(entry):
            continue
        rid = entry.get("result_id", "")
        actions.append(
            {
                "id": f"breakthrough_{rid[:12]}",
                "type": "breakthrough",
                "priority": 1,
                "icon": "trophy",
                "title": f"Architecture {rid[:8]} — Breakthrough",
                "summary": f"Composite score {safe_float(entry.get('composite_score'), 0.0):.3f}. Tier: breakthrough.",
                "detail": {
                    "result_id": rid,
                    "composite_score": safe_float(entry.get("composite_score"), 0.0),
                    "screening_loss_ratio": entry.get("screening_loss_ratio"),
                    "tier": "breakthrough",
                },
                "actions": [
                    {
                        "label": "View Details",
                        "action": "navigate",
                        "payload": {"tab": "discoveries", "result_id": rid},
                    },
                ],
                "dismissable": True,
                "source": "leaderboard",
            }
        )


def _append_stalled_run_warning(actions: List[Dict[str, Any]], nb: LabNotebook) -> None:
    recent = nb.get_recent_experiments(5)
    completed = [e for e in recent if e.get("status") == "completed"]
    if len(completed) < 3 or not all(
        (e.get("n_stage1_passed") or 0) == 0 for e in completed[:3]
    ):
        return
    actions.append(
        {
            "id": "warning_stalled_runs",
            "type": "warning",
            "priority": 2,
            "icon": "warning",
            "title": "Pipeline stalled — zero S1 survivors",
            "summary": f"Last {len(completed[:3])} completed runs produced no Stage 1 survivors.",
            "detail": {
                "recent_experiments": [
                    {
                        "id": e.get("experiment_id", "")[:12],
                        "s1": e.get("n_stage1_passed", 0),
                    }
                    for e in completed[:3]
                ],
            },
            "actions": [
                {
                    "label": "Run Novelty Search",
                    "action": "start",
                    "payload": {"mode": "novelty"},
                },
            ],
            "dismissable": True,
            "source": "experiments",
        }
    )


def _append_healer_actions(actions: List[Dict[str, Any]], nb: LabNotebook) -> None:
    healer_tasks = nb.get_recent_healer_tasks(limit=5)
    active = [t for t in healer_tasks if t.get("state") not in ("completed", "failed")]
    for task in active[:2]:
        tid = task.get("task_id", "")
        actions.append(
            {
                "id": f"healer_{tid[:12]}",
                "type": "healer",
                "priority": 4,
                "icon": "wrench",
                "title": f"Code healer: {task.get('trigger_type', 'repair')}",
                "summary": f"Task {tid[:12]} — {task.get('state', 'active')}. {task.get('scope', '')[:80]}",
                "detail": {
                    "task_id": tid,
                    "state": task.get("state"),
                    "trigger_type": task.get("trigger_type"),
                    "experiment_id": task.get("experiment_id"),
                },
                "actions": [],
                "dismissable": True,
                "source": "healer",
            }
        )


def _append_first_run_strategy(actions: List[Dict[str, Any]], nb: LabNotebook) -> None:
    summary = nb.get_dashboard_headline_summary()
    if summary.get("total_experiments", 0) != 0:
        return
    actions.append(
        {
            "id": "strategy_first_run",
            "type": "strategy",
            "priority": 5,
            "icon": "lightbulb",
            "title": "Get started",
            "summary": "No experiments yet. Start your first continuous run to begin exploring architectures.",
            "detail": {},
            "actions": [
                {
                    "label": "Start Continuous",
                    "action": "start",
                    "payload": {"mode": "continuous"},
                },
            ],
            "dismissable": False,
            "source": "strategy",
        }
    )


def compute_action_queue(nb, analytics=None) -> List[Dict[str, Any]]:
    """Aggregate prioritized actions from existing data sources."""
    from ._helpers import _DISMISSED_ACTIONS

    actions: List[Dict[str, Any]] = []

    try:
        _append_breakthrough_actions(actions, nb)
    except Exception as exc:
        _logger.debug("Failed to append breakthrough actions: %s", exc, exc_info=True)

    try:
        _append_stalled_run_warning(actions, nb)
    except Exception as exc:
        _logger.debug("Failed to append stalled-run warning: %s", exc, exc_info=True)

    try:
        _append_healer_actions(actions, nb)
    except Exception as exc:
        _logger.debug("Failed to append healer actions: %s", exc, exc_info=True)

    try:
        _append_first_run_strategy(actions, nb)
    except Exception as exc:
        _logger.debug(
            "Failed to append first-run strategy action: %s", exc, exc_info=True
        )

    actions = [a for a in actions if a["id"] not in _DISMISSED_ACTIONS]
    actions.sort(key=lambda a: a.get("priority", 10))
    return actions[:8]


def attach_long_context_breakdown(nb: LabNotebook, entries: list) -> None:
    """Attach long-context eval breakdown scores to leaderboard/discovery entries."""
    if not entries:
        return
    result_ids = [e.get("result_id") for e in entries if e.get("result_id")]
    if not result_ids:
        return
    try:
        placeholders = ",".join("?" for _ in result_ids)
        rows = nb.conn.execute(
            f"""SELECT result_id, robustness_long_ctx_score
                FROM leaderboard WHERE result_id IN ({placeholders})""",
            tuple(result_ids),
        ).fetchall()
        score_map = {r["result_id"]: r["robustness_long_ctx_score"] for r in rows}
        for entry in entries:
            rid = entry.get("result_id")
            if rid and rid in score_map:
                entry["long_context_score"] = score_map[rid]
    except Exception as exc:
        _logger.debug("Failed to attach long-context breakdown: %s", exc, exc_info=True)


def enrich_program_detail(nb: LabNotebook, program: dict) -> dict:
    """Enrich a program detail dict with leaderboard and analytics data."""
    result_id = program.get("result_id")
    if not result_id:
        return program

    try:
        lb = nb.conn.execute(
            "SELECT * FROM leaderboard WHERE result_id = ?", (result_id,)
        ).fetchone()
        if lb:
            lb_dict = dict(lb)
            program["tier"] = lb_dict.get("tier", "screening")
            program["leaderboard"] = lb_dict
            program["recommendation"] = compute_recommendation(program, lb_dict)
            program["promotion_evidence"] = promotion_evidence_for_entry(lb_dict)
            program["decision_gate"] = decision_gate_for_entry(lb_dict)
        else:
            program["tier"] = "screening"
    except Exception as e:
        _logger.debug(f"enrich_program_detail leaderboard lookup failed: {e}")

    return program


def program_lineage_chain(nb: LabNotebook, result_id: str) -> List[Dict[str, Any]]:
    """Trace the parent_result_id chain for lineage visualization."""
    chain: List[Dict[str, Any]] = []
    visited: set = set()
    current_id = result_id

    for _ in range(20):
        if not current_id or current_id in visited:
            break
        visited.add(current_id)

        row = nb.conn.execute(
            """SELECT result_id, experiment_id, graph_fingerprint,
                      graph_json, loss_ratio, stage1_passed, timestamp
               FROM program_results WHERE result_id = ?""",
            (current_id,),
        ).fetchone()
        if not row:
            break

        parent_result_id = None
        if "parent_result_id" in row.keys():
            parent_result_id = row["parent_result_id"]
        else:
            try:
                graph_json = row["graph_json"]
                parsed = (
                    json.loads(graph_json)
                    if isinstance(graph_json, str)
                    else (graph_json or {})
                )
                metadata = parsed.get("metadata") if isinstance(parsed, dict) else {}
                refinement = (
                    metadata.get("refinement") if isinstance(metadata, dict) else {}
                )
                source = (
                    refinement.get("source_result_id")
                    if isinstance(refinement, dict)
                    else None
                )
                if isinstance(source, str) and source.strip():
                    parent_result_id = source.strip()
            except Exception:  # noqa: BLE001 — best-effort lineage inference
                _logger.debug(
                    "Failed to infer parent_result_id from graph_json for result_id=%s",
                    current_id,
                    exc_info=True,
                )
                parent_result_id = None

        chain.append(
            {
                "result_id": row["result_id"],
                "experiment_id": row["experiment_id"],
                "graph_fingerprint": row["graph_fingerprint"],
                "parent_result_id": parent_result_id,
                "loss_ratio": row["loss_ratio"],
                "stage1_passed": bool(row["stage1_passed"]),
                "timestamp": row["timestamp"],
            }
        )
        current_id = parent_result_id

    return chain


def compute_compression_opportunities(coverage: Dict[str, Any]) -> Dict[str, Any]:
    """Transform compression coverage data into ranked opportunity suggestions."""
    by_technique = coverage.get("by_technique", {})
    if not by_technique:
        return {
            "available": False,
            "top_techniques": [],
            "explanation": "No compression technique data available.",
        }

    techniques = []
    for name, stats in by_technique.items():
        n_evaluated = int(stats.get("n_evaluated") or 0)
        n_survived = int(stats.get("n_survived") or 0)
        survival_rate = (n_survived / n_evaluated) if n_evaluated > 0 else 0.0
        techniques.append(
            {
                "technique": name,
                "n_evaluated": n_evaluated,
                "n_survived": n_survived,
                "survival_rate": round(survival_rate, 4),
            }
        )

    techniques.sort(key=lambda t: (t["survival_rate"], t["n_evaluated"]), reverse=True)

    return {
        "available": len(techniques) > 0,
        "top_techniques": techniques[:15],
        "total_techniques": len(techniques),
        "explanation": "Ranked by S1 survival rate among programs using each technique.",
    }


def compute_sparse_evidence(nb: LabNotebook) -> Dict[str, Any]:
    """Compute sparse-training evidence from the notebook."""
    try:
        rows = nb.conn.execute(
            """SELECT COUNT(*) as n,
                      AVG(CASE WHEN sparsity_density_mean IS NOT NULL
                          THEN sparsity_density_mean END) as avg_density,
                      AVG(CASE WHEN sparsity_nm_compliance IS NOT NULL
                          THEN sparsity_nm_compliance END) as avg_nm
               FROM program_results
               WHERE sparsity_density_mean IS NOT NULL"""
        ).fetchone()
        if rows and rows["n"] > 0:
            return {
                "n_sparse_programs": int(rows["n"]),
                "avg_density_mean": float(rows["avg_density"])
                if rows["avg_density"] is not None
                else None,
                "avg_nm_compliance": float(rows["avg_nm"])
                if rows["avg_nm"] is not None
                else None,
            }
    except Exception as exc:
        _logger.debug("Failed to compute sparse evidence: %s", exc, exc_info=True)
    return {"n_sparse_programs": 0, "avg_density_mean": None, "avg_nm_compliance": None}


def sparse_coverage_summary(sparse_coverage_data: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize sparse coverage data for briefing context."""
    sparse_share = float(sparse_coverage_data.get("sparse_share", 0))
    target_share = float(sparse_coverage_data.get("target_share", 0.15))
    sparse_survival = float(sparse_coverage_data.get("sparse_survival_rate", 0))
    return {
        "sparse_share": sparse_share,
        "target_share": target_share,
        "sparse_survival_rate": sparse_survival,
        "below_target": sparse_share < target_share,
    }
