"""Extracted logic for the /api/strategy/briefing endpoint.

Splits the 682-line api_strategy_briefing handler into four focused functions:
- gather_briefing_data: collect pipeline/trajectory/compression/sparse data
- try_llm_briefing: attempt LLM-powered briefing via Aria
- build_deterministic_briefing: sentence-based fallback briefing
- determine_recommended_action: action selection + config building
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any, Optional

from ._strategy_preflight import (
    build_start_mode_eligibility,
    normalize_briefing_mode,
    briefing_action_from_mode,
    briefing_action_label,
    augment_sparse_action_config,
)
from ._strategy_recommendations import (
    compute_compression_opportunities,
    compute_sparse_evidence,
    sparse_coverage_summary,
)
from ._helpers import normalize_result_ids

logger = logging.getLogger(__name__)


_briefing_cache: dict[str, Any] = {}
_briefing_cache_ts: float = 0.0
_BRIEFING_CACHE_TTL: float = 60.0


def gather_briefing_data(
    nb,
    analytics,
    recent: list[dict],
) -> dict[str, Any]:
    """Collect all data needed by both LLM and deterministic briefing paths.

    Returns a dict with keys: summary, trajectory, compression_coverage,
    compression_opportunities, primitive_effectiveness, sparse_evidence,
    sparse_coverage_data, sparse_coverage_overview, pipeline, data_block,
    recommendation_evidence, completed, avg_recent_s1, ref_comparison.
    """
    global _briefing_cache, _briefing_cache_ts
    now = _time.monotonic()
    if _briefing_cache and (now - _briefing_cache_ts) < _BRIEFING_CACHE_TTL:
        return _briefing_cache

    summary = nb.get_dashboard_summary()
    trajectory = analytics.learning_trajectory() or {}
    compression_coverage = analytics.compression_coverage() or {}
    compression_opportunities = compute_compression_opportunities(compression_coverage)
    primitive_effectiveness = analytics.compression_primitive_effectiveness() or {}
    sparse_evidence = compute_sparse_evidence(nb)
    sparse_coverage_data = analytics.sparse_coverage() or {}
    sparse_coverage_overview = sparse_coverage_summary(sparse_coverage_data)

    # Pipeline counts (exclude pinned reference architectures)
    leaderboard_rows = nb.conn.execute(
        "SELECT tier, COUNT(*) as cnt FROM leaderboard "
        "WHERE COALESCE(is_reference, 0) = 0 GROUP BY tier"
    ).fetchall()
    tiers = {r["tier"]: r["cnt"] for r in leaderboard_rows}
    pipeline = {
        "screening": tiers.get("screening", 0),
        "investigation": tiers.get("investigation", 0),
        "validation": tiers.get("validation", 0),
        "breakthrough": tiers.get("breakthrough", 0),
    }

    # Recent outcomes
    completed = [e for e in recent if e.get("status") == "completed"]
    recent_s1_rates = []
    for e in completed[:5]:
        gen = e.get("n_programs_generated") or 0
        passed = e.get("n_stage1_passed") or 0
        if gen > 0:
            recent_s1_rates.append(passed / gen)

    avg_recent_s1 = (
        sum(recent_s1_rates) / len(recent_s1_rates) if recent_s1_rates else None
    )

    trend = trajectory.get("trend", "insufficient_data")
    slope = trajectory.get("slope")

    total_exp = summary.get("total_experiments", 0)
    total_progs = summary.get("total_programs_evaluated", 0)
    s1_survivors = summary.get("stage1_survivors", 0)

    compression_summary = compression_opportunities.get("summary") or {}
    data_block = {
        "total_experiments": total_exp,
        "total_programs": total_progs,
        "s1_survivors": s1_survivors,
        "avg_recent_s1_rate": avg_recent_s1,
        "learning_trend": trend,
        "learning_slope": slope,
        "pipeline": pipeline,
        "compression": compression_summary,
        "compression_primitives": primitive_effectiveness.get("primitives", []),
        "sparse": sparse_evidence,
    }

    recommendation_evidence = _build_recommendation_evidence(
        recent,
        completed,
        avg_recent_s1,
        trend,
        slope,
        pipeline,
        compression_summary,
        primitive_effectiveness,
        sparse_evidence,
        sparse_coverage_overview,
    )
    ref_comparison = _build_ref_comparison(nb)

    result = {
        "summary": summary,
        "trajectory": trajectory,
        "tiers": tiers,
        "compression_opportunities": compression_opportunities,
        "primitive_effectiveness": primitive_effectiveness,
        "sparse_evidence": sparse_evidence,
        "sparse_coverage_data": sparse_coverage_data,
        "sparse_coverage_overview": sparse_coverage_overview,
        "pipeline": pipeline,
        "data_block": data_block,
        "recommendation_evidence": recommendation_evidence,
        "completed": completed,
        "avg_recent_s1": avg_recent_s1,
        "recent_s1_rates": recent_s1_rates,
        "ref_comparison": ref_comparison,
    }
    _briefing_cache = result
    _briefing_cache_ts = now
    return result


def _build_recommendation_evidence(
    recent,
    completed,
    avg_recent_s1,
    trend,
    slope,
    pipeline,
    compression_summary,
    primitive_effectiveness,
    sparse_evidence,
    sparse_coverage_overview,
) -> dict[str, Any]:
    """Build the recommendation_evidence dict from pipeline signals."""
    recent_window = recent[:10]
    recent_cancelled = 0
    recent_failed = 0
    for exp in recent_window:
        status = str(exp.get("status") or "").strip().lower()
        if status in {"cancelled", "canceled"}:
            recent_cancelled += 1
        elif status == "failed":
            recent_failed += 1

    recent_completed_window = completed[:5]
    recent_zero_s1_runs = sum(
        1
        for exp in recent_completed_window
        if (exp.get("n_programs_generated") or 0) > 0
        and (exp.get("n_stage1_passed") or 0) == 0
    )

    return {
        "learning_trend": trend,
        "learning_slope": slope,
        "avg_recent_s1_rate": avg_recent_s1,
        "recent_completed_runs": len(recent_completed_window),
        "recent_zero_s1_runs": recent_zero_s1_runs,
        "recent_cancelled_runs": recent_cancelled,
        "recent_failed_runs": recent_failed,
        "pipeline": pipeline,
        "compression": compression_summary,
        "compression_primitives": primitive_effectiveness.get("primitives", []),
        "sparse": sparse_evidence,
        "sparse_coverage": sparse_coverage_overview,
    }


def _build_ref_comparison(nb) -> Optional[dict]:
    """Compare best synthesized model against reference baselines."""
    try:
        ref_rows = nb.conn.execute(
            "SELECT reference_name, composite_score, loss_ratio "
            "FROM leaderboard WHERE COALESCE(is_reference, 0) = 1 "
            "ORDER BY composite_score DESC"
        ).fetchall()
        best_ref_score = max((r["composite_score"] for r in ref_rows), default=None)
        if not best_ref_score:
            return None

        best_synth = nb.conn.execute(
            "SELECT composite_score FROM leaderboard "
            "WHERE COALESCE(is_reference, 0) = 0 "
            "ORDER BY composite_score DESC LIMIT 1"
        ).fetchone()

        refs_list = [
            {"name": r["reference_name"], "score": float(r["composite_score"])}
            for r in ref_rows
        ]

        if best_synth and best_synth["composite_score"] > best_ref_score:
            return {
                "beats_all_references": True,
                "best_synthesized_score": float(best_synth["composite_score"]),
                "best_reference_score": float(best_ref_score),
                "margin_pct": round(
                    100.0
                    * (best_synth["composite_score"] - best_ref_score)
                    / best_ref_score,
                    1,
                ),
                "references": refs_list,
            }
        return {
            "beats_all_references": False,
            "best_reference_score": float(best_ref_score),
            "references": refs_list,
        }
    except Exception as exc:
        logger.debug("Returning default due to error: %s", exc)
        return None


def try_llm_briefing(
    nb,
    aria,
    analytics,
    data: dict[str, Any],
    recent: list[dict],
    just_completed_exp: Optional[dict],
) -> Optional[dict]:
    """Attempt LLM-powered briefing via Aria.

    Returns a complete response dict on success, or None to fall back
    to deterministic briefing.  Sets 'fallback_reason' in the returned
    dict when the LLM path is unavailable.
    """
    llm = aria._get_llm()
    if llm is None:
        return None

    try:
        llm_reachable = (
            bool(llm.is_available()) if hasattr(llm, "is_available") else True
        )
    except Exception as exc:
        logger.debug("LLM reachability check failed: %s", exc)
        llm_reachable = False
    if not llm_reachable:
        return None

    try:
        briefing_context = _gather_llm_context(
            nb, analytics, data, recent, just_completed_exp
        )
        ai_briefing = aria.generate_briefing(context=briefing_context)
        if not ai_briefing or not ai_briefing.get("briefing_text"):
            return None

        action_key, suggested_config, normalized_mode, hypothesis = (
            _resolve_llm_suggestion(nb, ai_briefing, data["sparse_coverage_data"])
        )

        return {
            "briefing": ai_briefing["briefing_text"],
            "action": action_key or normalized_mode or "continuous",
            "action_label": briefing_action_label(normalized_mode, hypothesis),
            "action_rationale": (ai_briefing.get("suggested_action") or {}).get(
                "reasoning", ""
            ),
            "ai_powered": True,
            "confidence": ai_briefing.get("confidence", 0.5),
            "suggested_config": suggested_config or None,
            "evidence": data["recommendation_evidence"],
            "data": data["data_block"],
            "compression_opportunities": data["compression_opportunities"],
            "ref_comparison": data["ref_comparison"],
        }
    except Exception as e:
        logger.warning(f"LLM briefing unavailable, using deterministic: {e}")
        return None


def _gather_llm_context(
    nb,
    analytics,
    data: dict,
    recent: list[dict],
    just_completed_exp,
) -> dict:
    """Build the context dict for LLM briefing generation."""
    from ..llm.context_briefing import build_briefing_context

    try:
        active_campaigns = nb.get_active_campaigns()
        campaign = active_campaigns[0] if active_campaigns else None
    except Exception as exc:
        logger.debug("Failed to load campaigns: %s", exc)
        campaign = None

    try:
        dw = analytics.get_current_grammar_weights() or {}
    except Exception as exc:
        logger.debug("Falling back to default: %s", exc)
        dw = {}

    try:
        gw = analytics.compute_grammar_weights() or {}
    except Exception as exc:
        logger.debug("Falling back to default: %s", exc)
        gw = {}

    try:
        top_programs = nb.conn.execute(
            "SELECT graph_fingerprint, loss_ratio, novelty_score, tier "
            "FROM leaderboard WHERE COALESCE(is_reference, 0) = 0 "
            "ORDER BY composite_score DESC LIMIT 3"
        ).fetchall()
        top_progs = [dict(r) for r in top_programs] if top_programs else None
    except Exception as exc:
        logger.debug("Failed to load top programs: %s", exc)
        top_progs = None

    try:
        scaling_summary_data = nb.get_scaling_summary()
    except Exception as exc:
        logger.debug("Failed to load scaling summary: %s", exc)
        scaling_summary_data = None

    try:
        return build_briefing_context(
            recent_experiments=recent,
            pipeline_tiers=data["tiers"],
            learning_trajectory=data["trajectory"],
            campaign=campaign,
            grammar_weights=gw,
            default_weights=dw,
            top_programs=top_progs,
            just_completed=just_completed_exp,
            sparse_coverage=data["sparse_coverage_data"],
            scaling_summary=scaling_summary_data,
            ref_comparison=data["ref_comparison"],
        )
    except Exception as exc:
        logger.debug("build_briefing_context failed, using fallback: %s", exc)
        trajectory = data["trajectory"]
        return {
            "pipeline": data["pipeline"],
            "learning": {
                "trend": trajectory.get("trend", "insufficient_data"),
                "slope": trajectory.get("slope"),
                "avg_recent_s1_rate": data["avg_recent_s1"],
            },
            "recent_experiments": recent[:5],
            "campaign": campaign,
        }


def _resolve_llm_suggestion(
    nb,
    ai_briefing: dict,
    sparse_coverage_data: dict,
) -> tuple[str, dict, str, Optional[str]]:
    """Process LLM suggestion into action_key, config, mode, hypothesis."""
    suggested = ai_briefing.get("suggested_action") or {}
    normalized_mode = normalize_briefing_mode(suggested.get("mode"))
    action_key = briefing_action_from_mode(normalized_mode)
    suggested_config = dict(suggested.get("config") or {})
    hypothesis = suggested.get("hypothesis")
    if normalized_mode:
        suggested_config["mode"] = normalized_mode
    if hypothesis:
        suggested_config["hypothesis"] = hypothesis

    # Modes that require result_ids — resolve them automatically
    if normalized_mode in (
        "investigation",
        "validation",
    ) and not suggested_config.get("result_ids"):
        _tier = "screening" if normalized_mode == "investigation" else "investigation"
        _TIER_SQL = {
            "screening": "SELECT result_id FROM leaderboard WHERE tier = ? AND screening_passed = 1 ORDER BY screening_loss_ratio ASC LIMIT 20",
            "investigation": "SELECT result_id FROM leaderboard WHERE tier = ? AND investigation_passed = 1 ORDER BY investigation_loss_ratio ASC LIMIT 20",
        }
        _tier_rows = nb.conn.execute(_TIER_SQL[_tier], (_tier,)).fetchall()
        suggested_config["result_ids"] = [
            r["result_id"] for r in _tier_rows if r["result_id"]
        ]

    if normalized_mode in ("investigation", "validation"):
        _requested = normalize_result_ids(suggested_config.get("result_ids", []))
        _eligibility = build_start_mode_eligibility(nb, normalized_mode, _requested)
        _eligible = _eligibility.get("eligible_result_ids") or []
        if _eligible:
            suggested_config["result_ids"] = _eligible
        else:
            normalized_mode = "continuous"
            action_key = "continuous"
            _hypothesis = suggested_config.get("hypothesis")
            suggested_config = {"mode": "continuous", "model_source": "mixed"}
            if _hypothesis:
                suggested_config["hypothesis"] = _hypothesis

    suggested_config = augment_sparse_action_config(
        suggested_config, normalized_mode, sparse_coverage_data
    )
    return action_key, suggested_config, normalized_mode, hypothesis


def build_deterministic_briefing(nb, data: dict[str, Any]) -> str:
    """Build a sentence-based briefing from pipeline data."""
    sentences = []
    db = data["data_block"]
    pipeline = data["pipeline"]
    total_exp = db["total_experiments"]
    total_progs = db["total_programs"]
    s1_survivors = db["s1_survivors"]
    avg_recent_s1 = data["avg_recent_s1"]
    recent_s1_rates = data["recent_s1_rates"]
    completed = data["completed"]
    trend = db["learning_trend"]
    slope = db["learning_slope"]
    compression_summary = db["compression"]
    sparse_evidence = db["sparse"]
    ref_comparison = data["ref_comparison"]

    screening = pipeline["screening"]
    investigation = pipeline["investigation"]
    validation = pipeline["validation"]
    breakthrough = pipeline["breakthrough"]

    if total_exp > 0:
        sentences.append(
            f"Across {total_exp} experiments, {total_progs:,} architectures "
            f"have been evaluated with {s1_survivors} stage-1 survivors "
            f"({s1_survivors / max(total_progs, 1) * 100:.1f}% overall pass rate)."
        )

    if avg_recent_s1 is not None:
        n_recent = len(recent_s1_rates)
        sentences.append(
            f"The last {n_recent} completed experiment{'s' if n_recent != 1 else ''} "
            f"averaged a {avg_recent_s1 * 100:.1f}% S1 pass rate."
        )

    if trend == "improving" and slope is not None:
        sentences.append(
            f"The system is learning — S1 rate is improving at "
            f"+{abs(slope) * 100:.2f} percentage points per experiment."
        )
    elif trend == "declining" and slope is not None:
        sentences.append(
            f"S1 rate is declining ({slope * 100:.2f} pp/experiment). "
            f"Consider switching search strategy or trying evolution mode."
        )
    elif trend == "plateaued":
        sentences.append(
            "S1 rate has plateaued — a novelty search or evolution run "
            "could help escape the current local optimum."
        )

    pipeline_parts = []
    if screening > 0:
        pipeline_parts.append(f"{screening} at screening")
    if investigation > 0:
        pipeline_parts.append(f"{investigation} under investigation")
    if validation > 0:
        pipeline_parts.append(f"{validation} in validation")
    if breakthrough > 0:
        pipeline_parts.append(
            f"{breakthrough} breakthrough{'s' if breakthrough != 1 else ''}"
        )
    if pipeline_parts:
        sentences.append(f"Candidate pipeline: {', '.join(pipeline_parts)}.")

    _append_telemetry_sentences(
        sentences, compression_summary, sparse_evidence, completed
    )
    _append_diversity_sentences(sentences, nb, pipeline)

    if ref_comparison and ref_comparison.get("beats_all_references"):
        margin = ref_comparison.get("margin_pct", 0)
        sentences.append(
            f"Milestone: a synthesized architecture now beats ALL "
            f"reference baselines by {margin}%."
        )
    elif ref_comparison and ref_comparison.get("references"):
        best_ref = ref_comparison["best_reference_score"]
        sentences.append(
            f"Best reference baseline score: {best_ref:.1f}. "
            f"No synthesized model has surpassed it yet."
        )

    return " ".join(sentences)


def _append_telemetry_sentences(
    sentences: list[str],
    compression_summary: dict,
    sparse_evidence: dict,
    completed: list[dict],
) -> None:
    """Append compression, sparse, and last-experiment sentences."""
    compressed_share = float(compression_summary.get("compressed_test_share") or 0.0)
    compressed_survival = float(
        compression_summary.get("compressed_survival_rate") or 0.0
    )
    if compression_summary:
        sentences.append(
            "Compression coverage: "
            f"{compressed_share * 100:.1f}% of tested candidates use compact techniques; "
            f"compressed survival is {compressed_survival * 100:.1f}%."
        )

    sparse_n = int(sparse_evidence.get("n_sparse_programs") or 0)
    if sparse_n > 0:
        sparse_density = float(sparse_evidence.get("avg_density_mean") or 0.0)
        sparse_nm = sparse_evidence.get("avg_nm_compliance")
        sparse_fragment = f"Sparse telemetry: {sparse_n} runs with mean density {sparse_density * 100:.1f}%"
        if sparse_nm is not None:
            sparse_fragment += f", N:M compliance {float(sparse_nm) * 100:.1f}%"
        sparse_fragment += "."
        sentences.append(sparse_fragment)

    if completed:
        last = completed[0]
        last_s1 = last.get("n_stage1_passed") or 0
        last_gen = last.get("n_programs_generated") or 0
        last_loss = last.get("best_loss_ratio")
        last_id = last.get("experiment_id", "")[:8]
        parts = [f"Last experiment ({last_id}): {last_s1}/{last_gen} passed S1"]
        if last_loss is not None:
            parts.append(f"best loss {last_loss:.4f}")
        aria_sum = last.get("aria_summary")
        if aria_sum:
            parts.append(f"— {aria_sum}")
        sentences.append(". ".join(parts) + ".")


def _append_diversity_sentences(
    sentences: list[str],
    nb,
    pipeline: dict[str, int],
) -> None:
    """Append DB-backed diversity analysis sentences (optional, swallows errors)."""
    if nb is None:
        return
    try:
        op_rows = nb.conn.execute(
            "SELECT op_name, s1_passes, total_uses FROM op_success_rates "
            "WHERE total_uses >= 5 ORDER BY "
            "CAST(s1_passes AS REAL) / CAST(total_uses AS REAL) DESC LIMIT 3"
        ).fetchall()
        if op_rows:
            top_ops = [
                f"{r['op_name']} ({r['s1_passes']}/{r['total_uses']})" for r in op_rows
            ]
            sentences.append(f"Top-performing operators: {', '.join(top_ops)}.")

        failure_rows = nb.conn.execute(
            "SELECT stage_at_death, COUNT(*) as cnt FROM program_results "
            "WHERE stage1_passed = 0 AND stage_at_death IS NOT NULL "
            "GROUP BY stage_at_death ORDER BY cnt DESC LIMIT 2"
        ).fetchall()
        if failure_rows:
            failure_parts = [
                f"{r['stage_at_death']} ({r['cnt']})" for r in failure_rows
            ]
            sentences.append(f"Dominant failure stages: {', '.join(failure_parts)}.")

        total_leaderboard = sum(pipeline.values())
        unique_fps = nb.conn.execute(
            "SELECT COUNT(DISTINCT SUBSTR(graph_fingerprint, 1, 8)) FROM leaderboard"
        ).fetchone()[0]
        if unique_fps is not None and total_leaderboard > 0:
            diversity_ratio = unique_fps / total_leaderboard
            if diversity_ratio < 0.5:
                sentences.append(
                    f"Warning: only {unique_fps} unique architecture "
                    f"families in {total_leaderboard} "
                    f"leaderboard entries — search may be converging."
                )
    except Exception as exc:
        logger.debug("Suppressed error: %s", exc)


def determine_recommended_action(nb, data: dict[str, Any]) -> dict[str, Any]:
    """Select recommended action and config based on pipeline state."""
    p, db = data["pipeline"], data["data_block"]
    sparse_coverage_overview = data["sparse_coverage_overview"]
    avg_recent_s1, recent_s1_rates = data["avg_recent_s1"], data["recent_s1_rates"]
    total_exp = db["total_experiments"]
    compressed_share = float(db["compression"].get("compressed_test_share") or 0.0)
    screening, investigation = p["screening"], p["investigation"]
    validation, breakthrough = p["validation"], p["breakthrough"]

    action = action_label = action_rationale = None
    screening_result_ids: list[str] = []

    if breakthrough > 0:
        action = "export_breakthrough"
        action_label = "Export Breakthrough Report"
        action_rationale = (
            f"{breakthrough} candidate{'s have' if breakthrough != 1 else ' has'} "
            f"reached breakthrough tier — ready for publication review."
        )
    elif compressed_share < 0.2 and total_exp >= 3:
        action = "compact_synthesis"
        action_label = "Run Compactness-Focused Synthesis"
        action_rationale = (
            "Compression techniques are underexplored in this campaign. "
            "Run a compactness-focused synthesis batch to improve model efficiency coverage."
        )
    elif sparse_coverage_overview.get("below_target") and total_exp >= 3:
        sparse_share = float(sparse_coverage_overview.get("sparse_share") or 0.0)
        sparse_survival = float(
            sparse_coverage_overview.get("sparse_survival_rate") or 0.0
        )
        target_share = float(sparse_coverage_overview.get("target_share") or 0.15)
        action = "novelty_search"
        action_label = "Run Sparse-Focused Novelty Search"
        action_rationale = (
            f"Sparse coverage is below target ({sparse_share * 100:.1f}% < {target_share * 100:.0f}%) "
            f"with {sparse_survival * 100:.1f}% sparse survival. "
            "Run novelty search with sparse-focused morphological sampling to explore high-upside sparse candidates."
        )
    elif validation > 0 and screening == 0 and investigation == 0:
        action = "monitor_validation"
        action_label = "Review Validation Progress"
        action_rationale = (
            f"{validation} candidate{'s are' if validation != 1 else ' is'} "
            f"in validation. Monitor results before starting new experiments."
        )
    elif screening > 0:
        action, action_label, action_rationale, screening_result_ids = (
            _resolve_screening_action(nb, avg_recent_s1)
        )
    elif total_exp == 0:
        action = "start_first"
        action_label = "Run First Experiment"
        action_rationale = (
            "No experiments yet. Start a mixed continuous run to begin "
            "exploring the architecture space."
        )
    elif data["data_block"]["learning_trend"] == "declining" or (
        len(recent_s1_rates) >= 3 and all(r == 0 for r in recent_s1_rates[:3])
    ):
        action = "novelty_search"
        action_label = "Try Evolution / Novelty Search"
        action_rationale = (
            "Recent experiments are underperforming. An evolution or "
            "novelty-driven search can escape the current local minimum."
        )
    else:
        trend = data["data_block"]["learning_trend"]
        action = "continuous"
        action_label = "Continue Research"
        action_rationale = (
            "The pipeline is active and the system is "
            + ("learning" if trend == "improving" else "exploring")
            + ". Continue generating and evaluating new architectures."
        )

    suggested_config = _build_action_config(
        action,
        screening_result_ids,
        sparse_coverage_overview,
        data["sparse_coverage_data"],
    )

    return {
        "action": action,
        "action_label": action_label,
        "action_rationale": action_rationale,
        "suggested_config": suggested_config,
    }


def _resolve_screening_action(
    nb,
    avg_recent_s1: Optional[float],
) -> tuple[str, str, str, list[str]]:
    """Determine action when screening survivors exist."""
    inv_failed = nb.conn.execute(
        "SELECT COUNT(*) FROM leaderboard "
        "WHERE tier = 'investigation' AND investigation_passed = 0"
    ).fetchone()[0]

    screening_rows = nb.conn.execute(
        "SELECT result_id FROM leaderboard "
        "WHERE tier = 'screening' AND screening_passed = 1 "
        "AND COALESCE(is_reference, 0) = 0 "
        "ORDER BY composite_score DESC LIMIT 20"
    ).fetchall()
    candidate_ids = [r["result_id"] for r in screening_rows if r["result_id"]]

    screening_result_ids: list[str] = []
    if candidate_ids:
        eligibility = build_start_mode_eligibility(nb, "investigation", candidate_ids)
        screening_result_ids = eligibility.get("eligible_result_ids") or []

    if not screening_result_ids:
        return (
            "continuous",
            "Continue Research",
            "Screening survivors exist but are not currently eligible for investigation reruns. "
            "Continue generating new architectures.",
            [],
        )

    n = len(screening_result_ids)
    rationale_parts = [
        f"{n} candidate{'s' if n != 1 else ''} passed screening and "
        f"{'are' if n != 1 else 'is'} awaiting deeper investigation"
    ]
    if inv_failed > 0:
        rationale_parts.append(
            f"({inv_failed} prior investigation{'s' if inv_failed != 1 else ''} "
            f"failed — fresh candidates may outperform)"
        )
    if avg_recent_s1 is not None:
        rationale_parts.append(f"with recent {avg_recent_s1 * 100:.0f}% hit rate")

    return (
        "investigate",
        f"Investigate {n} Screening Survivor{'s' if n != 1 else ''}",
        ", ".join(rationale_parts) + ".",
        screening_result_ids,
    )


def _build_action_config(
    action: str,
    screening_result_ids: list[str],
    sparse_coverage_overview: dict,
    sparse_coverage_data: dict,
) -> Optional[dict]:
    """Build suggested_config dict from the determined action."""
    det_mode_map = {
        "investigate": "investigation",
        "continuous": "continuous",
        "start_first": "continuous",
        "novelty_search": "novelty",
        "compact_synthesis": "synthesis",
        "export_breakthrough": None,
        "monitor_validation": None,
    }
    det_mode = det_mode_map.get(action, "continuous")

    if action == "compact_synthesis":
        config = {
            "mode": "synthesis",
            "model_source": "mixed",
            "morph_ratio": 0.85,
            "max_depth": 5,
            "max_ops": 8,
            "math_space_weight": 1.8,
            "residual_prob": 0.85,
            "n_programs": 80,
        }
    elif action == "novelty_search" and sparse_coverage_overview.get("below_target"):
        config = {
            "mode": "novelty",
            "model_source": "mixed",
            "morph_ratio": 0.8,
            "morph_focus_sparse": True,
            "morph_sparse_weight_storage": "semi_structured_2_4",
            "use_synthesized_training": True,
            "math_space_weight": 2.2,
            "max_depth": 6,
            "max_ops": 10,
            "n_programs": 120,
        }
    elif action == "investigate" and screening_result_ids:
        config = {
            "mode": "investigation",
            "model_source": "mixed",
            "result_ids": screening_result_ids,
        }
    elif det_mode:
        config = {"mode": det_mode, "model_source": "mixed"}
    else:
        return None

    return augment_sparse_action_config(
        config,
        config.get("mode", det_mode),
        sparse_coverage_data,
    )
