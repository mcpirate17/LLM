"""Strategy, report, recommendation, and evidence helper functions.

Contains domain logic for leaderboard analysis, breakthrough readiness,
cross-run stability, tier eligibility, report filtering, and program
enrichment helpers.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..shared_utils import safe_float
from ..notebook import LabNotebook

# Re-export safe_float under the old name used by blueprints
_to_safe_float = safe_float


# ── Report date/filter helpers ──────────────────────────────────────────

def parse_report_date(value: Optional[str], end_of_day: bool = False) -> Optional[float]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if len(raw) == 10:
            dt = datetime.strptime(raw, "%Y-%m-%d")
            if end_of_day:
                dt = dt.replace(hour=23, minute=59, second=59)
            return dt.timestamp()
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return None


def report_program_matches_theme(program: Dict[str, Any], theme: str) -> bool:
    normalized = str(theme or "").strip().lower()
    if not normalized or normalized in {"all", "any"}:
        return True
    graph_json = str(program.get("graph_json") or "").lower()
    arch_spec = str(program.get("arch_spec_json") or "").lower()
    pruning_method = str(program.get("pruning_method") or "").lower()
    if normalized == "sparsity":
        return (
            program.get("sparse_density_mean") is not None
            or "sparse" in graph_json
            or "sparse" in arch_spec
            or bool(pruning_method)
        )
    if normalized == "compression":
        compression_markers = (
            "low_rank", "shared_basis", "tied_proj", "grouped_linear", "bottleneck", "quant", "compressed"
        )
        return any(marker in graph_json or marker in arch_spec for marker in compression_markers)
    if normalized == "routing":
        return (
            program.get("routing_confidence_mean") is not None
            or "routing" in graph_json
            or "moe" in graph_json
            or "gate" in graph_json
        )
    if normalized == "mathspace":
        return (
            bool(program.get("graph_uses_math_spaces"))
            or "mathspace" in graph_json
            or "clifford" in graph_json
            or "hyperbolic" in graph_json
            or "padic" in graph_json
            or "tropical" in graph_json
        )
    if normalized == "failure_modes":
        return (program.get("stage1_passed") or 0) == 0 or bool(program.get("error_type"))
    return True


def experiment_s1_rate(exp: Dict[str, Any]) -> Optional[float]:
    generated = exp.get("n_programs_generated")
    if generated is None:
        generated = exp.get("n_programs")
    passed = exp.get("n_stage1_passed")
    if passed is None:
        passed = exp.get("s1_passed")
    try:
        gen = float(generated or 0)
        s1 = float(passed or 0)
    except Exception:
        return None
    if gen <= 0:
        return None
    return s1 / gen


def report_experiment_matches_trend(exp: Dict[str, Any], trend: str) -> bool:
    normalized = str(trend or "").strip().lower()
    if not normalized or normalized in {"all", "any"}:
        return True
    rate = experiment_s1_rate(exp)
    novelty = exp.get("best_novelty_score")
    if normalized == "high_novelty":
        return isinstance(novelty, (int, float)) and float(novelty) >= 0.5
    if rate is None:
        return False
    if normalized in {"improving", "high_survival"}:
        return rate >= 0.08
    if normalized == "declining":
        return rate < 0.03
    if normalized == "plateaued":
        return 0.03 <= rate < 0.08
    return True


def build_filtered_report_summary(
    base_summary: Dict[str, Any],
    experiments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not experiments:
        return dict(base_summary or {})
    total_programs = 0
    total_survivors = 0
    for exp in experiments:
        total_programs += int(exp.get("n_programs_generated") or 0)
        total_survivors += int(exp.get("n_stage1_passed") or 0)
    out = dict(base_summary or {})
    out["total_experiments"] = len(experiments)
    out["total_programs_evaluated"] = total_programs
    out["stage1_survivors"] = total_survivors
    return out


def build_report_snapshot_key(scope: str, query_payload: Dict[str, Any]) -> str:
    raw = json.dumps(
        {"scope": scope, "query": query_payload or {}},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# ── Tier / discovery helpers ────────────────────────────────────────────

def infer_tier_for_program(nb: LabNotebook, program: dict) -> str:
    """Infer tier for a raw program_results row by checking the leaderboard."""
    result_id = program.get("result_id")
    if not result_id:
        return "screening"
    row = nb.conn.execute(
        "SELECT tier FROM leaderboard WHERE result_id = ?", (result_id,)
    ).fetchone()
    return row["tier"] if row else "screening"


def count_discovery_tiers(nb: LabNotebook) -> dict:
    """Count unique fingerprints per tier + total S1 survivors."""
    rows = nb.conn.execute(
        "SELECT tier, COUNT(*) AS cnt FROM leaderboard GROUP BY tier"
    ).fetchall()
    counts = {r["tier"]: r["cnt"] for r in rows}
    total_s1 = nb.conn.execute(
        "SELECT COUNT(*) AS cnt FROM program_results WHERE stage1_passed = 1"
    ).fetchone()
    counts["total_survivors"] = total_s1["cnt"] if total_s1 else 0
    return counts


# ── Cross-run stability ────────────────────────────────────────────────

def _rank_label(delta: Optional[int], seen_runs: int) -> str:
    if seen_runs <= 1:
        return "new"
    if delta is None:
        return "new"
    if delta == 0:
        return "stable"
    return "up" if delta < 0 else "down"


def compute_cross_run_stability(nb: LabNotebook, top_programs: list) -> dict:
    """Compute rank movement for top candidates across recent experiments."""
    experiments = [
        exp for exp in nb.get_recent_experiments(40)
        if exp.get("status") == "completed"
    ]
    if not top_programs or not experiments:
        return {
            "summary": {"stable": 0, "up": 0, "down": 0, "new": 0},
            "candidates": [],
            "window_size": len(experiments),
        }

    fingerprint_ranks_by_experiment: dict[str, dict[str, int]] = {}
    for exp in experiments:
        experiment_id = exp.get("experiment_id")
        if not experiment_id:
            continue
        programs = nb.get_program_results(experiment_id)
        ranked = sorted(
            [
                p for p in programs
                if p.get("stage1_passed") and p.get("loss_ratio") is not None
            ],
            key=lambda p: p.get("loss_ratio", float("inf")),
        )
        ranks = {}
        for idx, program in enumerate(ranked, start=1):
            fp = program.get("graph_fingerprint")
            if fp and fp not in ranks:
                ranks[fp] = idx
        fingerprint_ranks_by_experiment[experiment_id] = ranks

    candidates = []
    summary = {"stable": 0, "up": 0, "down": 0, "new": 0}
    for index, program in enumerate(top_programs[:20], start=1):
        fp = program.get("graph_fingerprint")
        if not fp:
            continue

        history = []
        for exp in experiments:
            experiment_id = exp.get("experiment_id")
            if not experiment_id:
                continue
            rank = fingerprint_ranks_by_experiment.get(experiment_id, {}).get(fp)
            if rank is None:
                continue
            history.append({
                "experiment_id": experiment_id,
                "timestamp": exp.get("timestamp"),
                "rank": rank,
            })

        seen_runs = len(history)
        latest_rank = history[0]["rank"] if history else None
        previous_rank = history[1]["rank"] if len(history) > 1 else None
        delta = None
        if latest_rank is not None and previous_rank is not None:
            delta = latest_rank - previous_rank
        trend = _rank_label(delta, seen_runs)
        summary[trend] = summary.get(trend, 0) + 1

        candidates.append({
            "result_id": program.get("result_id"),
            "graph_fingerprint": fp,
            "current_overall_rank": index,
            "seen_runs": seen_runs,
            "latest_rank": latest_rank,
            "previous_rank": previous_rank,
            "rank_delta": delta,
            "trend": trend,
        })

    return {
        "summary": summary,
        "candidates": candidates,
        "window_size": len(experiments),
    }


# ── Recommendation / evidence ──────────────────────────────────────────

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

    checks = {
        "baselineEvidence": baseline_ratio is not None,
        "baselineBeat": baseline_ratio is not None and baseline_ratio < 1.0,
        "multiSeedStd": std is not None,
        "boundedStd": std is not None and std <= 0.12,
        "ckaArtifactBacked": entry.get("cka_source") == "artifact",
        "repeatObserved": seen_runs >= 3,
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

    score = round((completeness * 0.5 + std_signal * 0.2 + repeat_signal * 0.2 + margin_signal * 0.1) * 100)
    missing = [name for name, ok in checks.items() if not ok]

    return {
        "score": score,
        "seen_runs": seen_runs,
        "std": std,
        "evidence_count": evidence_count,
        "total_checks": total_checks,
        "missing": missing,
    }


def decision_gate_for_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    investigation_robustness = safe_float(entry.get("investigation_robustness"))
    validation_baseline_ratio = safe_float(entry.get("validation_baseline_ratio"))
    validation_multi_seed_std = safe_float(entry.get("validation_multi_seed_std"))

    checks = {
        "screeningEvidence": entry.get("screening_loss_ratio") is not None and entry.get("screening_novelty") is not None,
        "investigationEvidence": entry.get("investigation_loss_ratio") is not None and entry.get("investigation_robustness") is not None,
        "robustnessFloor": investigation_robustness is not None and investigation_robustness >= 0.5,
        "validationEvidence": (
            entry.get("validation_loss_ratio") is not None
            and entry.get("validation_baseline_ratio") is not None
            and entry.get("validation_multi_seed_std") is not None
        ),
        "baselineBeatsReference": validation_baseline_ratio is not None and validation_baseline_ratio < 1.0,
        "consistencyBounded": validation_multi_seed_std is not None and validation_multi_seed_std <= 0.12,
    }
    decision_ready = all(checks.values())
    missing = [name for name, ok in checks.items() if not ok]
    return {
        "decision_ready": decision_ready,
        "missing": missing,
    }


def build_scale_up_templates_for_result(result_id: Optional[str]) -> List[Dict[str, Any]]:
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
        ("result_id", "Result identifier captured", None, "Program result must be persisted before repro closure."),
        ("graph_fingerprint", "Fingerprint captured", None, "Graph fingerprint is required for cross-run traceability."),
        ("arch_spec", "Architecture spec recorded", "robustness_recheck", "Re-run robustness check if architecture metadata is incomplete."),
        ("baseline_ratio", "Baseline ratio measured", "multi_seed_stress", "Run multi-seed validation to compute baseline ratio."),
        ("multi_seed_std", "Multi-seed variance measured", "multi_seed_stress", "Run multi-seed validation to measure stability variance."),
        ("cka_artifact", "Artifact-backed CKA recorded", "efficiency_scale_up", "After run completion, stamp CKA artifact integrity in artifact references."),
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
        program["compression_metrics"] = analytics.canonical_compression_metrics(program)
        program["reproducibility_packet"] = analytics.reproducibility_packet_status(program)


# ── Scale-up resolution ─────────────────────────────────────────────────

def resolve_scale_up_result_ids(
    nb: LabNotebook,
    result_ids: List[str],
    graph_fingerprints: List[str],
) -> Dict[str, Any]:
    """Resolve explicit result IDs and/or fingerprint prefixes for scale-up."""
    merged_result_ids: List[str] = []
    seen: set = set()
    for result_id in result_ids:
        if result_id in seen:
            continue
        seen.add(result_id)
        merged_result_ids.append(result_id)

    resolved: List[Dict[str, Any]] = []
    unresolved: List[str] = []

    for fingerprint in graph_fingerprints:
        rows = nb.conn.execute(
            """
            SELECT result_id, graph_fingerprint, experiment_id, stage1_passed,
                   loss_ratio, timestamp
            FROM program_results
            WHERE graph_fingerprint LIKE ?
            ORDER BY stage1_passed DESC,
                     (loss_ratio IS NULL) ASC,
                     loss_ratio ASC,
                     timestamp DESC
            LIMIT 5
            """,
            (f"{fingerprint}%",),
        ).fetchall()

        if not rows:
            unresolved.append(fingerprint)
            continue

        chosen = dict(rows[0])
        chosen_result_id = str(chosen.get("result_id") or "")
        if chosen_result_id and chosen_result_id not in seen:
            seen.add(chosen_result_id)
            merged_result_ids.append(chosen_result_id)

        candidates = [
            {
                "result_id": row["result_id"],
                "graph_fingerprint": row["graph_fingerprint"],
                "experiment_id": row["experiment_id"],
                "stage1_passed": bool(row["stage1_passed"]),
                "loss_ratio": row["loss_ratio"],
            }
            for row in rows
        ]
        resolved.append({
            "requested_fingerprint": fingerprint,
            "selected_result_id": chosen.get("result_id"),
            "selected_graph_fingerprint": chosen.get("graph_fingerprint"),
            "selected_experiment_id": chosen.get("experiment_id"),
            "candidate_count": len(rows),
            "candidates": candidates,
        })

    return {
        "result_ids": merged_result_ids,
        "resolved_fingerprints": resolved,
        "unresolved_fingerprints": unresolved,
    }


# ── Start mode eligibility ──────────────────────────────────────────────

def build_start_mode_eligibility(
    nb: LabNotebook,
    mode: str,
    result_ids: List[str],
) -> Dict[str, Any]:
    """Validate candidate progression eligibility for start modes."""
    payload: Dict[str, Any] = {
        "mode": mode,
        "requested_result_ids": list(result_ids),
        "eligible_result_ids": [],
        "ineligible": [],
        "all_eligible": False,
    }
    if not result_ids:
        return payload

    placeholders = ",".join("?" for _ in result_ids)
    leaderboard_rows = nb.conn.execute(
        f"""
        SELECT result_id, tier, investigation_passed, validation_passed,
               investigation_loss_ratio, validation_loss_ratio
        FROM leaderboard
        WHERE result_id IN ({placeholders})
        """,
        tuple(result_ids),
    ).fetchall()
    program_rows = nb.conn.execute(
        f"""
        SELECT result_id, stage1_passed
        FROM program_results
        WHERE result_id IN ({placeholders})
        """,
        tuple(result_ids),
    ).fetchall()

    leaderboard_by_id = {row["result_id"]: dict(row) for row in leaderboard_rows}
    program_by_id = {row["result_id"]: dict(row) for row in program_rows}

    for result_id in result_ids:
        lb = leaderboard_by_id.get(result_id)
        program = program_by_id.get(result_id)

        if lb is None:
            if program is None:
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "result_not_found",
                    "detail": "Result ID was not found in program results.",
                })
            elif not bool(program.get("stage1_passed")):
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_stage1_survivor",
                    "detail": "Result exists but is not a Stage-1 survivor.",
                })
            else:
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_in_leaderboard",
                    "detail": "Result exists but has no leaderboard progression record.",
                })
            continue

        tier = str(lb.get("tier") or "").lower()

        if mode == "investigation":
            if tier != "screening":
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_screening_tier",
                    "detail": f"Current tier is '{tier or 'unknown'}'; only screening tier can be investigated.",
                    "tier": tier or None,
                })
                continue
            payload["eligible_result_ids"].append(result_id)
            continue

        if mode == "validation":
            if tier != "investigation":
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_investigation_tier",
                    "detail": f"Current tier is '{tier or 'unknown'}'; validation requires investigation tier.",
                    "tier": tier or None,
                })
                continue
            if not bool(lb.get("investigation_passed")):
                payload["ineligible"].append({
                    "result_id": result_id,
                    "reason": "not_investigation_passed",
                    "detail": "Investigation evidence did not pass robustness gate.",
                    "tier": tier,
                })
                continue
            payload["eligible_result_ids"].append(result_id)
            continue

        payload["ineligible"].append({
            "result_id": result_id,
            "reason": "unsupported_mode",
            "detail": f"Eligibility checks are not implemented for mode '{mode}'.",
        })

    payload["all_eligible"] = len(payload["ineligible"]) == 0 and len(payload["eligible_result_ids"]) > 0
    payload["summary"] = {
        "requested": len(result_ids),
        "eligible": len(payload["eligible_result_ids"]),
        "ineligible": len(payload["ineligible"]),
    }
    return payload


def build_report_action_eligibility(
    nb: LabNotebook,
    result_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Build per-result report action eligibility aligned with start guardrails."""
    from ._helpers import normalize_result_ids
    normalized_ids = normalize_result_ids(result_ids)
    if not normalized_ids:
        return {}

    inv = build_start_mode_eligibility(nb, "investigation", normalized_ids)
    val = build_start_mode_eligibility(nb, "validation", normalized_ids)

    inv_eligible = set(inv.get("eligible_result_ids") or [])
    val_eligible = set(val.get("eligible_result_ids") or [])
    inv_reason = {
        row.get("result_id"): row.get("reason")
        for row in (inv.get("ineligible") or [])
        if row.get("result_id")
    }
    val_reason = {
        row.get("result_id"): row.get("reason")
        for row in (val.get("ineligible") or [])
        if row.get("result_id")
    }

    eligibility_by_id: Dict[str, Dict[str, Any]] = {}
    for result_id in normalized_ids:
        investigation_eligible = result_id in inv_eligible
        validation_eligible = result_id in val_eligible
        queue_eligible = investigation_eligible or validation_eligible
        queue_reason = None
        if not queue_eligible:
            queue_reason = inv_reason.get(result_id) or val_reason.get(result_id) or "not_progression_eligible"

        eligibility_by_id[result_id] = {
            "investigationEligible": investigation_eligible,
            "validationEligible": validation_eligible,
            "queueEligible": queue_eligible,
            "queueReason": queue_reason,
            "investigationReason": inv_reason.get(result_id),
            "validationReason": val_reason.get(result_id),
        }

    return eligibility_by_id


# ── Breakthrough production readiness ───────────────────────────────────

def compute_breakthrough_production_readiness(nb: LabNotebook, analytics: Any) -> Dict[str, Any]:
    breakthroughs = nb.get_leaderboard(
        tier="breakthrough", limit=20, sort_by="composite_score", include_references=False
    )
    if not breakthroughs:
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

    stability = compute_cross_run_stability(nb, nb.get_top_programs(20, sort_by="loss_ratio"))
    stability_by_result = {
        c.get("result_id"): c
        for c in stability.get("candidates", [])
        if c.get("result_id")
    }

    evaluated: List[Dict[str, Any]] = []
    for entry in breakthroughs:
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
        evaluated.append({
            "result_id": row.get("result_id"),
            "architecture_family": row.get("architecture_family"),
            "composite_score": _to_safe_float(row.get("composite_score"), 0.0),
            "promotion_confidence_score": promotion["score"],
            "seen_runs": promotion["seen_runs"],
            "decision_ready": gate["decision_ready"],
            "decision_missing": gate["missing"],
            "repro_packet": row["reproducibility_packet"],
            "cka_source": row.get("cka_source"),
            "scale_up_templates": scale_templates,
            "reproducibility_workflow": reproducibility_workflow,
        })

    breakthrough_count = len(evaluated)
    decision_ready_count = sum(1 for row in evaluated if row.get("decision_ready"))
    high_confidence_count = sum(1 for row in evaluated if int(row.get("promotion_confidence_score") or 0) >= 75)
    full_repro_packet_count = sum(1 for row in evaluated if (row.get("repro_packet") or {}).get("status") == "ready")
    artifact_cka_count = sum(1 for row in evaluated if row.get("cka_source") == "artifact")

    switch_ready = any(
        row.get("decision_ready")
        and int(row.get("promotion_confidence_score") or 0) >= 75
        and (row.get("repro_packet") or {}).get("status") == "ready"
        and row.get("cka_source") == "artifact"
        for row in evaluated
    )

    if switch_ready:
        recommendation = {
            "action": "switch_to_scale_up_epic",
            "reason": "At least one breakthrough candidate meets decision, confidence, repro, and artifact-backed CKA gates.",
        }
    else:
        recommendation = {
            "action": "stay_current_epic",
            "reason": "Breakthrough evidence is still incomplete; continue hardening reproducibility and validation before switching epics.",
        }

    top_candidates = sorted(
        evaluated,
        key=lambda row: (
            int(bool(row.get("decision_ready"))),
            int(row.get("promotion_confidence_score") or 0),
            _to_safe_float(row.get("composite_score"), 0.0),
        ),
        reverse=True,
    )[:3]
    scale_up_templates = top_candidates[0].get("scale_up_templates", []) if top_candidates else []

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


# ── Action queue ────────────────────────────────────────────────────────

def compute_action_queue(nb, analytics=None) -> List[Dict[str, Any]]:
    """Aggregate prioritized actions from existing data sources."""
    from ._helpers import _DISMISSED_ACTIONS

    actions: List[Dict[str, Any]] = []

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

    # 1. Breakthrough candidates from leaderboard
    try:
        breakthroughs = nb.get_leaderboard(
            tier="breakthrough", limit=5, sort_by="composite_score", include_references=False
        )
        for entry in breakthroughs:
            if _is_reference_like(entry):
                continue
            rid = entry.get("result_id", "")
            actions.append({
                "id": f"breakthrough_{rid[:12]}",
                "type": "breakthrough",
                "priority": 1,
                "icon": "trophy",
                "title": f"Architecture {rid[:8]} — Breakthrough",
                "summary": f"Composite score {_to_safe_float(entry.get('composite_score'), 0.0):.3f}. Tier: breakthrough.",
                "detail": {
                    "result_id": rid,
                    "composite_score": _to_safe_float(entry.get("composite_score"), 0.0),
                    "screening_loss_ratio": entry.get("screening_loss_ratio"),
                    "tier": "breakthrough",
                },
                "actions": [
                    {"label": "View Details", "action": "navigate", "payload": {"tab": "discoveries", "result_id": rid}},
                ],
                "dismissable": True,
                "source": "leaderboard",
            })
    except Exception:
        pass

    # 2. Stalled run warning
    try:
        recent = nb.get_recent_experiments(5)
        completed = [e for e in recent if e.get("status") == "completed"]
        if len(completed) >= 3 and all(
            (e.get("n_stage1_passed") or 0) == 0 for e in completed[:3]
        ):
            actions.append({
                "id": "warning_stalled_runs",
                "type": "warning",
                "priority": 2,
                "icon": "warning",
                "title": "Pipeline stalled — zero S1 survivors",
                "summary": f"Last {len(completed[:3])} completed runs produced no Stage 1 survivors.",
                "detail": {
                    "recent_experiments": [
                        {"id": e.get("experiment_id", "")[:12], "s1": e.get("n_stage1_passed", 0)}
                        for e in completed[:3]
                    ],
                },
                "actions": [
                    {"label": "Run Novelty Search", "action": "start", "payload": {"mode": "novelty"}},
                ],
                "dismissable": True,
                "source": "experiments",
            })
    except Exception:
        pass

    # 3. Healer fixes
    try:
        healer_tasks = nb.get_recent_healer_tasks(limit=5)
        active = [t for t in healer_tasks if t.get("state") not in ("completed", "failed")]
        for task in active[:2]:
            tid = task.get("task_id", "")
            actions.append({
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
            })
    except Exception:
        pass

    # 4. Diagnosis issues
    try:
        if analytics:
            analytics_data = analytics.get_analytics_data() if hasattr(analytics, "get_analytics_data") else {}
        else:
            from ..analytics import ExperimentAnalytics
            analytics_obj = ExperimentAnalytics(nb)
            analytics_data = analytics_obj.get_analytics_data() if hasattr(analytics_obj, "get_analytics_data") else {}
        # _diagnose_research_issues is defined inside blueprint (misc_bp)
        # For now we skip this if not available
        import logging as _log
        _log.getLogger(__name__).debug("Diagnosis issues require blueprint-local _diagnose_research_issues")
    except Exception:
        pass

    # 5. Strategy suggestion
    try:
        summary = nb.get_dashboard_summary()
        total_exp = summary.get("total_experiments", 0)
        if total_exp == 0:
            actions.append({
                "id": "strategy_first_run",
                "type": "strategy",
                "priority": 5,
                "icon": "lightbulb",
                "title": "Get started",
                "summary": "No experiments yet. Start your first continuous run to begin exploring architectures.",
                "detail": {},
                "actions": [
                    {"label": "Start Continuous", "action": "start", "payload": {"mode": "continuous"}},
                ],
                "dismissable": False,
                "source": "strategy",
            })
    except Exception:
        pass

    # Filter out dismissed actions
    actions = [a for a in actions if a["id"] not in _DISMISSED_ACTIONS]
    actions.sort(key=lambda a: a.get("priority", 10))
    return actions[:8]


# ── Program enrichment helpers ──────────────────────────────────────────

_logger = logging.getLogger(__name__)


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
    except Exception:
        pass


def enrich_program_detail(nb: LabNotebook, program: dict) -> dict:
    """Enrich a program detail dict with leaderboard and analytics data."""
    result_id = program.get("result_id")
    if not result_id:
        return program

    # Attach leaderboard tier and evidence
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

    for _ in range(20):  # max depth guard
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
                parsed = json.loads(graph_json) if isinstance(graph_json, str) else (graph_json or {})
                metadata = parsed.get("metadata") if isinstance(parsed, dict) else {}
                refinement = metadata.get("refinement") if isinstance(metadata, dict) else {}
                source = refinement.get("source_result_id") if isinstance(refinement, dict) else None
                if isinstance(source, str) and source.strip():
                    parent_result_id = source.strip()
            except Exception:
                parent_result_id = None

        chain.append({
            "result_id": row["result_id"],
            "experiment_id": row["experiment_id"],
            "graph_fingerprint": row["graph_fingerprint"],
            "parent_result_id": parent_result_id,
            "loss_ratio": row["loss_ratio"],
            "stage1_passed": bool(row["stage1_passed"]),
            "timestamp": row["timestamp"],
        })
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
        techniques.append({
            "technique": name,
            "n_evaluated": n_evaluated,
            "n_survived": n_survived,
            "survival_rate": round(survival_rate, 4),
        })

    techniques.sort(key=lambda t: (t["survival_rate"], t["n_evaluated"]), reverse=True)

    return {
        "available": len(techniques) > 0,
        "top_techniques": techniques[:15],
        "total_techniques": len(techniques),
        "explanation": "Ranked by S1 survival rate among programs using each technique.",
    }


# ── Experiment launch helpers ─────────────────────────────────────────

_VALID_START_MODES = frozenset({
    "single", "continuous", "evolve", "novelty",
    "investigation", "validation", "scale_up",
    "refine_fingerprint", "compact_synthesis", "sparse_morph",
})


def normalize_start_mode(raw_mode: str) -> str:
    """Normalize and validate experiment start mode string."""
    mode = str(raw_mode or "single").strip().lower().replace("-", "_")
    if mode in _VALID_START_MODES:
        return mode
    return "single"


def run_launch_preflight(
    *,
    config,
    mode: str,
    prescreen: Dict[str, Any],
    notebook_path: str,
    sample_n: int = 4,
) -> Dict[str, Any]:
    """Run preflight checks before launching an experiment.

    Returns a dict with 'verdict' ('pass', 'warn', 'fail') and 'checks'.
    """
    checks: List[Dict[str, Any]] = []
    verdict = "pass"

    # Check prescreen warnings
    prescreen_warnings = prescreen.get("warnings", [])
    if prescreen_warnings:
        checks.append({
            "name": "prescreen_warnings",
            "status": "warn",
            "details": prescreen_warnings,
        })
        verdict = "warn"

    # Check prescreen blockers
    prescreen_blockers = prescreen.get("blockers", [])
    if prescreen_blockers:
        checks.append({
            "name": "prescreen_blockers",
            "status": "fail",
            "details": prescreen_blockers,
        })
        verdict = "fail"

    # Check for active experiment conflicts
    nb = LabNotebook(notebook_path)
    try:
        active = nb.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE status = 'running'"
        ).fetchone()[0]
        if active > 0:
            checks.append({
                "name": "active_experiment",
                "status": "warn",
                "details": f"{active} experiment(s) marked as running",
            })
            if verdict == "pass":
                verdict = "warn"
    except Exception:
        pass
    finally:
        nb.close()

    if not checks:
        checks.append({"name": "all_clear", "status": "pass", "details": None})

    return {"verdict": verdict, "checks": checks, "sample_n": sample_n}


def apply_compact_synthesis_bias(config) -> Dict[str, Any]:
    """Apply compact-synthesis mode biases to RunConfig.

    Returns dict of changes applied (for logging/response).
    """
    changes: Dict[str, Any] = {}
    if hasattr(config, "max_nodes") and (config.max_nodes is None or config.max_nodes > 12):
        changes["max_nodes"] = {"from": config.max_nodes, "to": 12}
        config.max_nodes = 12
    if hasattr(config, "grammar_config") and config.grammar_config is not None:
        gc = config.grammar_config
        if hasattr(gc, "max_depth") and (gc.max_depth is None or gc.max_depth > 5):
            changes["grammar_max_depth"] = {"from": gc.max_depth, "to": 5}
            gc.max_depth = 5
    return changes


def apply_sparse_morph_bias(config) -> Dict[str, Any]:
    """Apply sparse-morph mode biases to RunConfig.

    Returns dict of changes applied.
    """
    changes: Dict[str, Any] = {}
    if hasattr(config, "grammar_config") and config.grammar_config is not None:
        gc = config.grammar_config
        if hasattr(gc, "sparsity_bias"):
            changes["sparsity_bias"] = {"from": gc.sparsity_bias, "to": 0.7}
            gc.sparsity_bias = 0.7
    return changes


def extract_hypothesis_missing_fields(critique: Optional[Dict[str, Any]]) -> List[str]:
    """Extract list of missing required fields from a hypothesis critique dict."""
    if not critique or not isinstance(critique, dict):
        return []
    missing = critique.get("missing_fields", [])
    if isinstance(missing, list):
        return [str(f) for f in missing if f]
    return []


# ── Briefing helpers ──────────────────────────────────────────────────

_BRIEFING_MODE_MAP = {
    "synthesis": "single",
    "single": "single",
    "continuous": "continuous",
    "evolve": "evolve",
    "evolution": "evolve",
    "novelty": "novelty",
    "novelty_search": "novelty",
    "investigation": "investigation",
    "investigate": "investigation",
    "validation": "validation",
    "scale_up": "scale_up",
    "compact_synthesis": "compact_synthesis",
    "sparse_morph": "sparse_morph",
}


def normalize_briefing_mode(raw_mode: Optional[str]) -> Optional[str]:
    """Normalize LLM-suggested briefing mode to a valid start mode."""
    if not raw_mode:
        return None
    mode = str(raw_mode).strip().lower().replace("-", "_")
    return _BRIEFING_MODE_MAP.get(mode, mode if mode in _VALID_START_MODES else None)


def briefing_action_from_mode(mode: Optional[str]) -> Optional[str]:
    """Map a normalized mode to a briefing action key."""
    if not mode:
        return None
    action_map = {
        "single": "continuous",
        "continuous": "continuous",
        "evolve": "novelty_search",
        "novelty": "novelty_search",
        "investigation": "investigate",
        "validation": "validate",
        "scale_up": "scale_up",
        "compact_synthesis": "compact_synthesis",
        "sparse_morph": "novelty_search",
    }
    return action_map.get(mode, mode)


def briefing_action_label(mode: Optional[str], hypothesis: Optional[str] = None) -> str:
    """Human-readable label for a briefing action."""
    label_map = {
        "single": "Run Synthesis",
        "continuous": "Continue Research",
        "evolve": "Run Evolution Search",
        "novelty": "Run Novelty Search",
        "investigation": "Investigate Candidates",
        "validation": "Validate Candidates",
        "scale_up": "Scale Up",
        "compact_synthesis": "Run Compact Synthesis",
        "sparse_morph": "Run Sparse Morphology",
    }
    label = label_map.get(str(mode or ""), "Start Experiment")
    if hypothesis:
        label += f": {hypothesis[:60]}"
    return label


def augment_sparse_action_config(
    config: Optional[Dict[str, Any]],
    mode: Optional[str],
    sparse_coverage_data: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Augment a suggested config with sparse coverage hints when appropriate."""
    if config is None or not isinstance(config, dict):
        return config
    if mode not in ("novelty", "evolve", "single", "continuous"):
        return config

    sparse_share = float(sparse_coverage_data.get("sparse_share", 0))
    target_share = float(sparse_coverage_data.get("target_share", 0.15))
    if sparse_share < target_share:
        config.setdefault("morph_focus_sparse", True)
        config.setdefault("morph_sparse_weight_storage", "semi_structured_2_4")
    return config


# ── Sparse evidence helpers ───────────────────────────────────────────

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
                "avg_density_mean": float(rows["avg_density"]) if rows["avg_density"] is not None else None,
                "avg_nm_compliance": float(rows["avg_nm"]) if rows["avg_nm"] is not None else None,
            }
    except Exception:
        pass
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


# ── Diagnosis helpers ──────────────────────────────────────────────────

def diagnose_research_issues(
    analytics_data: Dict[str, Any],
    nb: LabNotebook,
) -> List[Dict[str, Any]]:
    """Diagnose common research pipeline issues from analytics data.

    Returns list of issue dicts with 'issue', 'action_type', and optional 'config_fix'.
    """
    issues: List[Dict[str, Any]] = []

    # Check S1 pass rate
    op_rates = analytics_data.get("op_success_rates") or {}
    if isinstance(op_rates, dict):
        total_uses = sum(v.get("total_uses", 0) for v in op_rates.values() if isinstance(v, dict))
        total_passes = sum(v.get("s1_passes", 0) for v in op_rates.values() if isinstance(v, dict))
        if total_uses > 50 and total_passes == 0:
            issues.append({
                "issue": "Zero S1 passes across all ops — grammar may be misconfigured",
                "action_type": "info",
            })

    # Check for grammar weight staleness
    grammar = analytics_data.get("grammar_weights") or {}
    if isinstance(grammar, dict):
        learned = grammar.get("learned") or {}
        if not learned:
            issues.append({
                "issue": "No learned grammar weights — consider running more experiments",
                "action_type": "info",
            })

    # Check for stuck experiments
    try:
        stuck = nb.conn.execute(
            "SELECT COUNT(*) FROM experiments WHERE status = 'running' "
            "AND timestamp < ?",
            (time.time() - 7200,),  # 2 hours old
        ).fetchone()[0]
        if stuck > 0:
            issues.append({
                "issue": f"{stuck} experiment(s) stuck in 'running' for >2 hours",
                "action_type": "info",
            })
    except Exception:
        pass

    return issues


# ── Pipeline sample check ─────────────────────────────────────────────

def run_pipeline_sample_check(*, config, sample_n: int = 5) -> Dict[str, Any]:
    """Run a quick pipeline sample check: generate, compile, test S0.

    Returns dict with 'generated', 'compiled', 'passed_s0', 'errors'.
    """
    generated = 0
    compiled = 0
    passed_s0 = 0
    errors: List[str] = []

    try:
        from ...synthesis.grammar import GrammarConfig, random_graph
        from ...synthesis.compiler import compile_model

        gc = GrammarConfig()
        for _ in range(sample_n):
            try:
                graph = random_graph(gc)
                generated += 1
                model = compile_model([graph], vocab_size=256, max_seq_len=64)
                compiled += 1
                # Quick forward pass check
                import torch
                x = torch.randint(0, 256, (1, 16))
                out = model(x)
                if out is not None:
                    passed_s0 += 1
            except Exception as exc:
                errors.append(str(exc)[:200])
    except ImportError as exc:
        errors.append(f"Import error: {exc}")

    return {
        "generated": generated,
        "compiled": compiled,
        "passed_s0": passed_s0,
        "errors": errors,
    }


# ── Entry normalization ───────────────────────────────────────────────

def normalize_entries(entries: list) -> list:
    """Normalize notebook entry dicts for API consumption.

    Parses JSON metadata strings, ensures consistent field types.
    """
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        d = dict(entry)
        # Parse metadata if it's a JSON string
        meta = d.get("metadata") or d.get("metadata_json", "{}") or d.get("metadata_json", "{}")
        if isinstance(meta, str):
            try:
                d["metadata"] = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                d["metadata"] = {}
        elif meta is None:
            d["metadata"] = {}
        result.append(d)
    return result


def parse_bool_query(value: Optional[str], default: bool = False) -> bool:
    """Parse a query parameter as a boolean."""
    if value is None:
        return default
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default
