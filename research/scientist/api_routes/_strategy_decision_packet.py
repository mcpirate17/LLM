"""Decision packet builder for /api/decision-packet/<result_id>."""

from __future__ import annotations

import logging
from typing import Any, Optional

from ._strategy_recommendations import (
    compute_cross_run_stability,
    compute_recommendation,
)

logger = logging.getLogger(__name__)


def build_decision_packet(nb, result_id: str) -> Optional[dict[str, Any]]:
    """Build the full evidence bundle for a promotion decision.

    Returns None if the result_id is not found.
    """
    program = nb.get_program_detail(result_id)
    if program is None:
        return None

    fingerprint = program.get("graph_fingerprint", "")
    experiment_id = program.get("experiment_id")
    leaderboard_entry = _get_leaderboard_entry(nb, result_id)
    failure_analysis = _get_failure_analysis(nb, experiment_id)
    hypothesis_chain = _get_hypothesis_chain(nb, experiment_id)
    cross_run = _get_cross_run_stability(nb, result_id)
    outcomes = _build_outcomes(program, leaderboard_entry)

    bl_ratio = program.get("baseline_loss_ratio")
    baseline_comparison = _interpret_baseline(bl_ratio)
    recommendation = compute_recommendation(program, leaderboard_entry)

    from ..analytics import ExperimentAnalytics

    analytics = ExperimentAnalytics(nb)
    entry_or_prog = leaderboard_entry if leaderboard_entry else program
    packet_status = analytics.reproducibility_packet_status(entry_or_prog)

    return {
        "result_id": result_id,
        "fingerprint": fingerprint,
        "experiment_id": experiment_id,
        "hypothesis_chain": hypothesis_chain,
        "outcomes": outcomes,
        "baseline_comparison": baseline_comparison,
        "failure_context": {
            "stage_at_death": program.get("stage_at_death"),
            "error_type": program.get("error_type"),
            "experiment_errors": failure_analysis.get("errors", {}),
            "experiment_funnel": failure_analysis.get("funnel", {}),
        },
        "cross_run_stability": cross_run,
        "recommendation": recommendation,
        "evidence_flags": {
            "has_baseline": bl_ratio is not None,
            "has_cka_artifact": program.get("cka_source") == "artifact",
            "has_multi_seed": outcomes["validation"] is not None,
            "has_hypothesis": len(hypothesis_chain) > 0,
            "repro_packet_ready": packet_status.get("status") == "ready",
        },
        "compression_metrics": analytics.canonical_compression_metrics(entry_or_prog),
        "reproducibility_packet": packet_status,
    }


def _get_leaderboard_entry(nb, result_id: str):
    try:
        return nb.get_leaderboard_entry(result_id)
    except Exception as exc:
        logger.debug(
            "Failed to load leaderboard entry for result_id=%s: %s",
            result_id,
            exc,
        )
        return None


def _get_failure_analysis(nb, experiment_id: Optional[str]) -> dict:
    if not experiment_id:
        return {"funnel": {}, "errors": {}, "stage_deaths": {}}
    try:
        nb.get_experiment(experiment_id)
    except Exception as exc:
        logger.debug(
            "Experiment lookup failed during decision packet build for experiment_id=%s: %s",
            experiment_id,
            exc,
        )
    try:
        return nb.get_failure_analysis(experiment_id)
    except Exception as exc:
        logger.debug(
            "Failure analysis lookup failed for experiment_id=%s: %s",
            experiment_id,
            exc,
        )
        return {"funnel": {}, "errors": {}, "stage_deaths": {}}


def _get_hypothesis_chain(nb, experiment_id: Optional[str]) -> list:
    if not experiment_id:
        return []
    try:
        hyp_row = nb.conn.execute(
            "SELECT hypothesis_id FROM hypotheses WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if hyp_row:
            hid = hyp_row["hypothesis_id"] if isinstance(hyp_row, dict) else hyp_row[0]
            return nb.get_hypothesis_chain(hid)
    except Exception as exc:
        logger.debug(
            "Hypothesis chain lookup failed for experiment_id=%s: %s",
            experiment_id,
            exc,
        )
    return []


def _get_cross_run_stability(nb, result_id: str) -> dict:
    try:
        top = nb.get_top_programs(20, sort_by="loss_ratio")
        stability = compute_cross_run_stability(nb, top)
        for c in stability.get("candidates", []):
            if c.get("result_id") == result_id:
                return {
                    "trend": c.get("trend", "unknown"),
                    "seen_runs": c.get("seen_runs", 0),
                }
    except Exception as exc:
        logger.debug(
            "Cross-run stability lookup failed for result_id=%s: %s",
            result_id,
            exc,
        )
    return {"trend": "unknown", "seen_runs": 0}


def _build_outcomes(program: dict, leaderboard_entry: Optional[dict]) -> dict:
    outcomes: dict[str, Any] = {
        "screening": {
            "loss_ratio": program.get("loss_ratio"),
            "novelty": program.get("novelty_score"),
        },
        "investigation": None,
        "validation": None,
    }
    if not leaderboard_entry:
        return outcomes
    inv_lr = leaderboard_entry.get("investigation_loss_ratio")
    if inv_lr is not None:
        outcomes["investigation"] = {
            "loss_ratio": inv_lr,
            "robustness": leaderboard_entry.get("investigation_robustness"),
            "passed": bool(leaderboard_entry.get("investigation_passed")),
        }
    val_lr = leaderboard_entry.get("validation_loss_ratio")
    if val_lr is not None:
        outcomes["validation"] = {
            "loss_ratio": val_lr,
            "baseline_ratio": leaderboard_entry.get("validation_baseline_ratio"),
            "multi_seed_std": leaderboard_entry.get("validation_multi_seed_std"),
            "passed": bool(leaderboard_entry.get("validation_passed")),
        }
    return outcomes


def _interpret_baseline(bl_ratio: Optional[float]) -> dict:
    result = {"ratio": bl_ratio, "interpretation": "unknown"}
    if bl_ratio is None:
        return result
    if bl_ratio < 0.95:
        result["interpretation"] = "outperforms"
    elif bl_ratio <= 1.05:
        result["interpretation"] = "comparable"
    else:
        result["interpretation"] = "underperforms"
    return result
