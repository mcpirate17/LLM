"""Dashboard and reporting route registration."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, List

from flask import jsonify, request
from .deps import ApiRouteContext
from ._utils import (
    bind_notebook_view,
    is_malformed_db_error,
    malformed_db_response_payload,
    with_notebook_context,
)
from ..persona import get_aria
from ._helpers import (
    get_runner,
    get_run_trigger_snapshot,
    deduplicate_insights,
    resolve_runner_status,
)
from ._strategy_recommendations import (
    annotate_qkv_usage,
    compute_cross_run_stability,
    compute_breakthrough_production_readiness,
)
from ._strategy_report import (
    parse_report_date,
    report_program_matches_theme,
    report_experiment_matches_trend,
    build_filtered_report_summary,
    build_report_snapshot_key,
    build_report_action_eligibility,
    normalize_entries,
    parse_bool_query,
)

logger = logging.getLogger(__name__)


def _trusted_report_mode() -> bool:
    return parse_bool_query(request.args.get("trusted_only"), default=True)


def _degraded_dashboard_summary(exc: Exception) -> Dict[str, Any]:
    payload = malformed_db_response_payload(exc)
    return {
        "total_experiments": 0,
        "completed_experiments": 0,
        "repaired_result_experiments": 0,
        "resultful_experiments": 0,
        "total_programs_evaluated": 0,
        "stage1_survivors": 0,
        "survival_rate": 0.0,
        "avg_novelty_score": 0.0,
        "top_novelty_score": 0.0,
        "active_insights": 0,
        "learning_events": 0,
        "latest_learning": None,
        "avg_step_time_ms": 0.0,
        "avg_throughput_tok_s": 0.0,
        "avg_routing_entropy": None,
        "avg_depth_savings": None,
        "avg_recursion_savings": None,
        "avg_routing_token_retention": None,
        "avg_sparsity_ratio": None,
        "latest_perf_report": None,
        "unique_fingerprints": 0,
        "data_accounting": {
            "row_volume": {},
            "run_volume": {},
            "graph_volume": {},
            "filtering": {},
            "training_curve_density": {},
            "leaderboard_tiers": {},
        },
        "latest_dedup": None,
        "template_observability": {},
        "leaderboard_consistency": {},
        "degraded": True,
        "database_status": payload["database_status"],
    }


def _enrich_dashboard_summary_campaigns(nb, summary: Dict[str, Any]) -> None:
    try:
        active_campaigns = nb.get_active_campaigns()
        total_hypotheses = nb.conn.execute(
            "SELECT COUNT(*) FROM hypotheses"
        ).fetchone()[0]
        knowledge_entries = nb.conn.execute(
            "SELECT COUNT(*) FROM knowledge_base WHERE status = 'active'"
        ).fetchone()[0]
        summary["active_campaigns"] = len(active_campaigns)
        summary["total_hypotheses"] = total_hypotheses
        summary["knowledge_entries"] = knowledge_entries
    except Exception as exc:
        logger.warning("Failed enriching dashboard campaign metadata: %s", exc)


def _load_recent_experiments_with_funnel(nb, limit: int) -> List[Dict[str, Any]]:
    recent_experiments = nb.get_recent_experiments(limit)
    recent_ids = [
        str(exp.get("experiment_id"))
        for exp in recent_experiments
        if exp.get("experiment_id")
    ]
    if not recent_ids:
        return recent_experiments
    placeholders = ",".join("?" for _ in recent_ids)
    rows = nb.conn.execute(
        f"""
        SELECT experiment_id, results_json
        FROM experiments
        WHERE experiment_id IN ({placeholders})
        """,
        recent_ids,
    ).fetchall()
    recent_results = {}
    for row in rows:
        parsed = nb._decompress(row["results_json"]) if row["results_json"] else None
        if isinstance(parsed, dict):
            recent_results[str(row["experiment_id"])] = parsed
    for exp in recent_experiments:
        parsed = recent_results.get(str(exp.get("experiment_id"))) or {}
        funnel_counts = parsed.get("funnel_counts")
        if isinstance(funnel_counts, dict) and funnel_counts:
            exp["funnel_counts"] = funnel_counts
        if "skipped_dedup" in parsed:
            exp["skipped_dedup"] = parsed.get("skipped_dedup")
        if "dedup_rate" in parsed:
            exp["dedup_rate"] = parsed.get("dedup_rate")
    return recent_experiments


def _compact_dashboard_payload(
    recent_experiments: List[Dict[str, Any]],
    top_programs: List[Dict[str, Any]],
    insights: List[Dict[str, Any]],
    recent_entries: List[Dict[str, Any]],
) -> tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    compact_experiments = [
        {
            "experiment_id": exp.get("experiment_id"),
            "timestamp": exp.get("timestamp"),
            "status": exp.get("status"),
            "mode": exp.get("mode"),
            "n_generated": exp.get("n_generated"),
            "n_stage0_passed": exp.get("n_stage0_passed"),
            "n_stage1_passed": exp.get("n_stage1_passed"),
            "best_loss_ratio": exp.get("best_loss_ratio"),
            "aria_summary": exp.get("aria_summary"),
            "funnel_counts": exp.get("funnel_counts"),
            "skipped_dedup": exp.get("skipped_dedup"),
            "dedup_rate": exp.get("dedup_rate"),
        }
        for exp in recent_experiments[:8]
    ]
    compact_programs = [
        {
            "result_id": row.get("result_id"),
            "experiment_id": row.get("experiment_id"),
            "timestamp": row.get("timestamp"),
            "loss_ratio": row.get("loss_ratio"),
            "novelty_score": row.get("novelty_score"),
            "composite_score": row.get("composite_score"),
            "architecture_family": row.get("architecture_family"),
            "qkv_usage": row.get("qkv_usage"),
            "routing_mode": row.get("routing_mode"),
            "param_count": row.get("param_count"),
            "stage1_passed": row.get("stage1_passed"),
            "graph_fingerprint": row.get("graph_fingerprint"),
        }
        for row in top_programs[:5]
    ]
    compact_insights = [
        {
            "insight_id": ins.get("insight_id"),
            "experiment_id": ins.get("experiment_id"),
            "timestamp": ins.get("timestamp"),
            "op_name": ins.get("op_name"),
            "text": str(ins.get("text") or "")[:220],
            "confidence": ins.get("confidence"),
            "signal": ins.get("signal"),
        }
        for ins in insights[:12]
    ]
    compact_entries = [
        {
            "entry_id": row.get("entry_id"),
            "experiment_id": row.get("experiment_id"),
            "entry_type": row.get("entry_type"),
            "timestamp": row.get("timestamp"),
            "content": str(row.get("content") or "")[:180],
        }
        for row in recent_entries[:8]
    ]
    return compact_experiments, compact_programs, compact_insights, compact_entries


def _compute_dashboard_deltas(
    data: Dict[str, Any], recent_experiments: List[Dict[str, Any]]
) -> None:
    try:
        completed = [
            exp for exp in recent_experiments if exp.get("status") == "completed"
        ]
        if len(completed) < 2:
            return
        latest = completed[0]
        previous = completed[1]
        data["deltas"] = {
            "experiment_id": latest.get("experiment_id"),
            "programs": (latest.get("n_programs_generated") or 0)
            - (previous.get("n_programs_generated") or 0),
            "stage1": (latest.get("n_stage1_passed") or 0)
            - (previous.get("n_stage1_passed") or 0),
            "best_loss": round(
                (latest.get("best_loss_ratio") or 1)
                - (previous.get("best_loss_ratio") or 1),
                4,
            )
            if latest.get("best_loss_ratio")
            else None,
            "best_novelty": round(
                (latest.get("best_novelty_score") or 0)
                - (previous.get("best_novelty_score") or 0),
                4,
            )
            if latest.get("best_novelty_score")
            else None,
        }
    except Exception as exc:
        logger.debug("Suppressed error: %s", exc)


def _attach_dashboard_learning_trend(summary: Dict[str, Any], analytics) -> None:
    try:
        trajectory = analytics.learning_trajectory()
        if trajectory and trajectory.get("trend") != "insufficient_data":
            summary["learning_trend"] = trajectory.get("trend")
            summary["learning_slope"] = trajectory.get("slope")
            summary["recent_s1_rate"] = trajectory.get("recent_s1_rate")
    except Exception as exc:
        logger.debug("Suppressed error: %s", exc)


def _build_report_payload(
    nb,
    analytics,
    *,
    fast_mode: bool,
    trusted_only: bool,
    include_heavy: bool,
    include_narrative: bool,
) -> Dict[str, Any]:
    top_limit = 20 if not fast_mode else 12
    expanded_limit = 80 if include_heavy else 0
    recent_limit = 100 if include_heavy else 30
    summary = nb.get_dashboard_summary(
        include_data_accounting=not fast_mode,
        include_template_observability=False,
    )
    payload = {
        "summary": summary,
        "top_programs": nb.get_report_top_programs_grouped_by_fingerprint(
            top_limit, sort_by="loss_ratio", trusted_only=trusted_only
        ),
        "top_programs_expanded": nb.get_top_programs(
            expanded_limit, sort_by="loss_ratio", trusted_only=trusted_only
        )
        if include_heavy
        else [],
        "recent_experiments": nb.get_recent_experiments(recent_limit),
        "op_success_rates": analytics.op_success_rates(),
        "failure_patterns": analytics.failure_patterns(),
        "grammar_weights": {
            "learned": analytics.compute_grammar_weights(),
            "default": analytics.get_current_grammar_weights(),
            "control_comparison": analytics.control_experiment_comparison(),
            "holdout_validation": analytics.holdout_validation(),
            "learning_diagnostics": analytics.grammar_weight_learning_diagnostics(),
        },
        "learning_log": nb.get_learning_log(limit=20 if fast_mode else 50),
        "insights": nb.get_insights(),
        "report_mode": {
            "fast": fast_mode,
            "include_heavy": include_heavy,
            "include_narrative": include_narrative,
            "trusted_only": trusted_only,
        },
    }
    if include_heavy:
        payload.update(
            {
                "math_family_coverage": analytics.math_family_coverage(),
                "mathspace_operator_impact": analytics.mathspace_operator_impact(),
                "routing_mode_comparison": analytics.routing_mode_comparison(),
                "gating_behavior_diagnostics": analytics.gating_behavior_diagnostics(),
                "structural_correlations": analytics.structural_correlations(),
                "top_op_combinations": analytics.top_op_combinations(10),
                "efficiency_frontier": analytics.efficiency_frontier(),
                "experiment_clusters": analytics.experiment_clusters(),
            }
        )
    return payload


def _attach_report_program_metadata(nb, analytics, data: Dict[str, Any]) -> None:
    learning_diagnostics = data["grammar_weights"].get("learning_diagnostics") or {}
    data["architecture_rerun_telemetry"] = {
        "unique_fingerprint_count": int(
            learning_diagnostics.get("unique_fingerprints") or 0
        ),
        "total_result_rows": int(learning_diagnostics.get("total_rows") or 0),
        "repeat_result_rows": int(learning_diagnostics.get("repeat_rows") or 0),
        "rerun_ratio": float(learning_diagnostics.get("rerun_ratio") or 0.0),
        "top_fingerprint_concentration": float(
            learning_diagnostics.get("top_fingerprint_concentration") or 0.0
        ),
        "weighting_mode": str(learning_diagnostics.get("mode") or "unknown"),
    }
    data["action_eligibility"] = build_report_action_eligibility(
        nb,
        [
            row.get("result_id")
            for row in [
                *(data["top_programs"] or []),
                *(data["top_programs_expanded"] or []),
            ]
            if row.get("result_id")
        ],
    )
    annotate_qkv_usage(data["top_programs"], analytics)
    annotate_qkv_usage(data["top_programs_expanded"], analytics)


def _attach_report_cross_run_stability(nb, data: Dict[str, Any]) -> None:
    expanded_by_fingerprint: Dict[str, List[Dict[str, Any]]] = {}
    for row in data["top_programs_expanded"]:
        fp = row.get("graph_fingerprint")
        if not fp:
            continue
        expanded_by_fingerprint.setdefault(fp, []).append(row)

    grouped_rank_by_fingerprint = {
        row.get("graph_fingerprint"): index
        for index, row in enumerate(data["top_programs"], start=1)
        if row.get("graph_fingerprint")
    }
    for fp, rows in expanded_by_fingerprint.items():
        repeat_count = len(rows)
        grouped_rank = grouped_rank_by_fingerprint.get(fp)
        for repeat_index, row in enumerate(rows, start=1):
            row["group_repeat_count"] = repeat_count
            row["group_repeat_index"] = repeat_index
            row["grouped_fingerprint_rank"] = grouped_rank

    data["cross_run_stability"] = compute_cross_run_stability(nb, data["top_programs"])
    stability_by_result = {
        candidate.get("result_id"): candidate
        for candidate in data["cross_run_stability"].get("candidates", [])
        if candidate.get("result_id")
    }
    stability_by_fingerprint = {
        candidate.get("graph_fingerprint"): candidate
        for candidate in data["cross_run_stability"].get("candidates", [])
        if candidate.get("graph_fingerprint")
    }
    fallback_stability = {
        "trend": "unknown",
        "seen_runs": 0,
        "latest_rank": None,
        "previous_rank": None,
        "rank_delta": None,
    }
    for program in [
        *(data["top_programs"] or []),
        *(data["top_programs_expanded"] or []),
    ]:
        by_result = stability_by_result.get(program.get("result_id"))
        by_fingerprint = stability_by_fingerprint.get(program.get("graph_fingerprint"))
        program["cross_run_stability"] = (
            by_result or by_fingerprint or fallback_stability
        )


def _load_filtered_report_experiments(
    nb, *, start_ts, end_ts, trend: str
) -> List[Dict[str, Any]]:
    experiments = nb.get_recent_experiments(500)
    filtered_experiments = []
    for exp in experiments:
        ts = exp.get("timestamp")
        if isinstance(ts, (int, float)):
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
        if not report_experiment_matches_trend(exp, trend):
            continue
        filtered_experiments.append(exp)
    return filtered_experiments


def _load_filtered_report_programs(
    nb,
    *,
    start_ts,
    end_ts,
    theme: str,
    trend: str,
    limit: int,
    trusted_only: bool,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sort_by = "novelty_score" if trend == "high_novelty" else "loss_ratio"
    expanded = nb.get_top_programs(
        max(limit * 3, 120),
        sort_by=sort_by,
        trusted_only=trusted_only,
    )
    filtered_programs: List[Dict[str, Any]] = []
    for program in expanded:
        ts = program.get("timestamp")
        if isinstance(ts, (int, float)):
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts > end_ts:
                continue
        if not report_program_matches_theme(program, theme):
            continue
        filtered_programs.append(program)
    grouped = []
    seen = set()
    for row in filtered_programs:
        fp = row.get("graph_fingerprint")
        if fp and fp in seen:
            continue
        if fp:
            seen.add(fp)
        grouped.append(row)
        if len(grouped) >= limit:
            break
    return grouped, filtered_programs


def _api_status(notebook_path: str, nb=None):
    """Get Aria's current status and dashboard summary."""
    runner = get_runner(notebook_path)
    aria = get_aria()
    runner_state = resolve_runner_status(nb, runner)
    try:
        summary = nb.get_dashboard_headline_summary()
        summary["leaderboard_consistency"] = nb.get_leaderboard_consistency_report()
    except sqlite3.DatabaseError as exc:
        if not is_malformed_db_error(exc):
            raise
        logger.warning("Status endpoint degraded due to malformed DB: %s", exc)
        summary = _degraded_dashboard_summary(exc)
    progress_payload = runner_state["progress"]
    trigger = get_run_trigger_snapshot(progress_payload.get("experiment_id"))
    progress_payload["run_trigger_source"] = trigger.get("source")
    progress_payload["run_trigger"] = trigger
    return jsonify(
        {
            "aria": aria.get_status(db_summary=summary),
            "summary": summary,
            "is_running": runner_state["is_running"],
            "progress": progress_payload,
            "native_runner": progress_payload.get("native_runner"),
            "run_trigger_source": trigger.get("source"),
            "run_trigger": trigger,
        }
    )


def _api_recompute_failure_signatures(nb=None):
    """Delete and rebuild failure_signatures using S1-only failures."""
    count = nb.recompute_failure_signatures()
    return jsonify({"status": "ok", "signatures_created": count})


def _api_reset_op_stats(nb=None):
    """Reset op_success_rates for specific ops so they get a fresh start.

    POST body: {"ops": ["op1", "op2", ...]}
    If no ops specified, resets all ops with 0 S1 passes.
    """
    data = request.get_json(silent=True) or {}
    ops = data.get("ops")
    if ops:
        nb.conn.executemany(
            "DELETE FROM op_success_rates WHERE op_name = ?",
            [(op_name,) for op_name in ops],
        )
        nb.conn.commit()
        count = len(ops)
    else:
        cur = nb.conn.execute(
            "DELETE FROM op_success_rates WHERE n_stage1_passed = 0 AND n_used >= 5"
        )
        nb.conn.commit()
        count = cur.rowcount
    return jsonify({"status": "ok", "ops_reset": count})


def _api_healer_tasks(nb=None):
    """List recent Code Healer tasks."""
    limit = request.args.get("limit", 20, type=int)
    return jsonify(nb.get_recent_healer_tasks(limit=max(1, min(limit, 200))))


def _api_healer_task_detail(task_id: str, nb=None):
    """Get one healer task with state history."""
    task = nb.get_healer_task(task_id)
    if task is None:
        return jsonify({"error": "Not found"}), 404
    return jsonify(
        {
            "task": task,
            "events": nb.get_healer_events(task_id, limit=200),
        }
    )


def _api_entries(nb=None):
    """List notebook entries."""
    exp_id = request.args.get("experiment_id")
    entry_type = request.args.get("type")
    n = request.args.get("n", 50, type=int)
    entries = nb.get_entries(experiment_id=exp_id, entry_type=entry_type, limit=n)
    return jsonify(normalize_entries(entries))


def _api_metrics(metric_name, nb=None):
    """Get time-series metrics."""
    exp_id = request.args.get("experiment_id")
    return jsonify(nb.get_metrics(metric_name, experiment_id=exp_id))


def _api_dashboard(notebook_path: str, nb=None):
    """Get all dashboard data in one call."""
    runner = get_runner(notebook_path)
    aria = get_aria()
    trusted_only = _trusted_report_mode()
    compact = request.path.endswith("/summary") or (
        str(request.args.get("compact", "0")).strip().lower() in {"1", "true", "yes"}
    )
    runner_state = resolve_runner_status(nb, runner)
    try:
        summary = nb.get_dashboard_summary(
            include_data_accounting=not compact,
            include_template_observability=not compact,
        )
        summary["leaderboard_consistency"] = nb.get_leaderboard_consistency_report()
    except sqlite3.DatabaseError as exc:
        if not is_malformed_db_error(exc):
            raise
        logger.warning(
            "Dashboard endpoint degraded due to malformed DB pages: %s",
            exc,
        )
        summary = _degraded_dashboard_summary(exc)
        return jsonify(
            {
                "aria": aria.get_status(db_summary=summary),
                "summary": summary,
                "recent_experiments": [],
                "top_programs": [],
                "production_readiness": {
                    "breakthrough_count": 0,
                    "epic_switch_recommendation": {
                        "action": "stay_current_epic",
                        "reason": "Notebook database is currently degraded",
                    },
                    "scale_up_templates": [],
                    "reproducibility_workflow": None,
                    "top_candidates": [],
                },
                "insights": [],
                "recent_entries": [],
                "is_running": runner_state["is_running"],
                "progress": runner_state["progress"],
                "degraded": True,
                "database_status": summary["database_status"],
            }
        )

    _enrich_dashboard_summary_campaigns(nb, summary)
    recent_experiments = _load_recent_experiments_with_funnel(nb, limit=30)
    from ..analytics import ExperimentAnalytics

    analytics = ExperimentAnalytics(nb)
    top_programs = nb.get_top_programs(10, trusted_only=trusted_only)
    annotate_qkv_usage(top_programs, analytics)
    production_readiness = compute_breakthrough_production_readiness(nb, analytics)
    insights = deduplicate_insights(nb.get_insights(limit=50))
    recent_entries = normalize_entries(nb.get_entries(limit=20))

    if compact:
        recent_experiments, top_programs, insights, recent_entries = (
            _compact_dashboard_payload(
                recent_experiments,
                top_programs,
                insights,
                recent_entries,
            )
        )

    data = {
        "aria": aria.get_status(db_summary=summary),
        "summary": summary,
        "recent_experiments": recent_experiments,
        "top_programs": top_programs,
        "production_readiness": production_readiness,
        "insights": insights,
        "recent_entries": recent_entries,
        "is_running": runner_state["is_running"],
        "progress": runner_state["progress"],
        "trusted_only": trusted_only,
    }
    _compute_dashboard_deltas(data, recent_experiments)
    _attach_dashboard_learning_trend(summary, analytics)
    last_rec = runner.last_recommendation
    if last_rec:
        data["last_recommendation"] = last_rec

    return jsonify(data)


def _api_data_accounting(nb=None):
    """Return explicit row/run/graph/cohort accounting for reporting consumers."""
    try:
        return jsonify(nb.get_data_accounting_summary())
    except sqlite3.DatabaseError as exc:
        if not is_malformed_db_error(exc):
            raise
        payload = malformed_db_response_payload(exc)
        return jsonify(
            {
                "row_volume": {},
                "run_volume": {},
                "graph_volume": {},
                "filtering": {},
                "training_curve_density": {},
                "leaderboard_tiers": {},
                "degraded": True,
                "database_status": payload["database_status"],
            }
        )


def _api_model_strength(nb=None):
    """Confounder-aware component/template/slot strength report."""
    from ..analytics.model_strength import build_model_strength_report

    min_support = request.args.get("min_support", 12, type=int)
    top_k = request.args.get("top_k", 20, type=int)
    payload = build_model_strength_report(
        nb.db_path,
        min_support=max(1, min_support),
        top_k=max(5, min(top_k, 50)),
    )
    return jsonify(payload)


def _api_slot_compatibility(nb=None):
    """Return the generated slot compatibility rules from synthesis."""
    from research.synthesis._template_helpers import get_slot_rule_summary

    return jsonify({"slot_rules": get_slot_rule_summary()})


def _api_report(nb=None):
    """Consolidated research report with all data."""
    aria = get_aria()
    from ..analytics import ExperimentAnalytics

    analytics = ExperimentAnalytics(nb)

    fast_mode = parse_bool_query(request.args.get("fast"), default=False)
    trusted_only = _trusted_report_mode()
    include_heavy = parse_bool_query(
        request.args.get("include_heavy"),
        default=not fast_mode,
    )
    include_narrative = parse_bool_query(
        request.args.get("include_narrative"),
        default=not fast_mode,
    )

    data = _build_report_payload(
        nb,
        analytics,
        fast_mode=fast_mode,
        trusted_only=trusted_only,
        include_heavy=include_heavy,
        include_narrative=include_narrative,
    )
    _attach_report_program_metadata(nb, analytics, data)
    _attach_report_cross_run_stability(nb, data)
    data["narrative"] = None
    if include_narrative:
        try:
            data["narrative"] = aria.generate_report_narrative(data)
        except Exception as exc:
            logger.debug("Report narrative generation failed: %s", exc)
            data["narrative"] = None

    return jsonify(data)


def _api_report_query(nb=None):
    """Scoped report payload for date/theme/trend report generation."""
    aria = get_aria()
    from ..analytics import ExperimentAnalytics

    analytics = ExperimentAnalytics(nb)

    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    start_ts = parse_report_date(start_date, end_of_day=False)
    end_ts = parse_report_date(end_date, end_of_day=True)
    theme = str(request.args.get("theme") or "all").strip().lower()
    trend = str(request.args.get("trend") or "all").strip().lower()
    include_narrative = parse_bool_query(
        request.args.get("include_narrative"),
        default=False,
    )
    trusted_only = _trusted_report_mode()
    try:
        limit = int(request.args.get("limit") or 20)
    except (TypeError, ValueError):
        limit = 20
    limit = max(5, min(120, limit))

    snapshot_query = {
        "start_date": start_date,
        "end_date": end_date,
        "theme": theme,
        "trend": trend,
        "limit": limit,
        "include_narrative": bool(include_narrative),
        "trusted_only": bool(trusted_only),
    }
    latest_completed_ts = nb.get_latest_completed_experiment_timestamp()
    snapshot_key = build_report_snapshot_key("report_query", snapshot_query)

    if not include_narrative:
        cached = nb.get_report_snapshot(
            snapshot_key=snapshot_key,
            scope="report_query",
            min_latest_completed_ts=latest_completed_ts,
        )
        if isinstance(cached, dict):
            cached["snapshot_cache"] = {
                "enabled": True,
                "hit": True,
                "key": snapshot_key,
                "latest_completed_ts": latest_completed_ts,
            }
            return jsonify(cached)

    filtered_experiments = _load_filtered_report_experiments(
        nb,
        start_ts=start_ts,
        end_ts=end_ts,
        trend=trend,
    )
    grouped, filtered_programs = _load_filtered_report_programs(
        nb,
        start_ts=start_ts,
        end_ts=end_ts,
        theme=theme,
        trend=trend,
        limit=limit,
        trusted_only=trusted_only,
    )

    base_summary = nb.get_dashboard_headline_summary()
    summary = build_filtered_report_summary(base_summary, filtered_experiments)

    data = {
        "summary": summary,
        "top_programs": grouped,
        "top_programs_expanded": filtered_programs[: max(limit * 2, 40)],
        "recent_experiments": filtered_experiments[: max(limit * 5, 40)],
        "op_success_rates": analytics.op_success_rates(),
        "failure_patterns": analytics.failure_patterns(),
        "insights": nb.get_insights(),
        "learning_log": nb.get_learning_log(limit=30),
        "narrative": None,
        "query": {
            "start_date": start_date,
            "end_date": end_date,
            "theme": theme,
            "trend": trend,
            "limit": limit,
            "matched_experiments": len(filtered_experiments),
            "matched_programs": len(filtered_programs),
            "trusted_only": bool(trusted_only),
        },
        "snapshot_cache": {
            "enabled": True,
            "hit": False,
            "key": snapshot_key,
            "latest_completed_ts": latest_completed_ts,
        },
    }

    if include_narrative:
        try:
            data["narrative"] = aria.generate_report_narrative(data)
        except Exception as exc:
            logger.debug("Scoped report narrative generation failed: %s", exc)
            data["narrative"] = None

    if not include_narrative:
        try:
            nb.save_report_snapshot(
                snapshot_key=snapshot_key,
                scope="report_query",
                query=snapshot_query,
                payload=data,
                latest_completed_ts=latest_completed_ts,
            )
        except Exception as exc:
            logger.debug("Scoped report snapshot save failed: %s", exc)

    return jsonify(data)


def register_reporting_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)

    app.add_url_rule(
        "/api/status", "api_status", bind_notebook_view(wnb, _api_status, notebook_path)
    )
    app.add_url_rule(
        "/api/recompute-failure-signatures",
        "api_recompute_failure_signatures",
        bind_notebook_view(wnb, _api_recompute_failure_signatures),
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/reset-op-stats",
        "api_reset_op_stats",
        bind_notebook_view(wnb, _api_reset_op_stats),
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/healer/tasks",
        "api_healer_tasks",
        bind_notebook_view(wnb, _api_healer_tasks),
    )
    app.add_url_rule(
        "/api/healer/tasks/<task_id>",
        "api_healer_task_detail",
        bind_notebook_view(wnb, _api_healer_task_detail),
    )
    app.add_url_rule(
        "/api/entries", "api_entries", bind_notebook_view(wnb, _api_entries)
    )
    app.add_url_rule(
        "/api/metrics/<metric_name>",
        "api_metrics",
        bind_notebook_view(wnb, _api_metrics),
    )
    app.add_url_rule(
        "/api/dashboard",
        "api_dashboard",
        bind_notebook_view(wnb, _api_dashboard, notebook_path),
    )
    app.add_url_rule(
        "/api/dashboard/summary",
        "api_dashboard_2",
        bind_notebook_view(wnb, _api_dashboard, notebook_path),
    )
    app.add_url_rule(
        "/api/reporting/data-accounting",
        "api_data_accounting",
        bind_notebook_view(wnb, _api_data_accounting),
    )
    app.add_url_rule(
        "/api/reporting/model-strength",
        "api_model_strength",
        bind_notebook_view(wnb, _api_model_strength),
    )
    app.add_url_rule(
        "/api/reporting/slot-compatibility",
        "api_slot_compatibility",
        bind_notebook_view(wnb, _api_slot_compatibility),
    )
    app.add_url_rule("/api/report", "api_report", bind_notebook_view(wnb, _api_report))
    app.add_url_rule(
        "/api/report/query",
        "api_report_query",
        bind_notebook_view(wnb, _api_report_query),
    )
