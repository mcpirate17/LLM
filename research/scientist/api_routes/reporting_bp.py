"""Dashboard and reporting route registration."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from flask import jsonify, request
from .deps import ApiRouteContext
from ._utils import with_notebook_context
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


def register_reporting_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)

    @app.route("/api/status")
    @wnb
    def api_status(nb=None):
        """Get Aria's current status and dashboard summary."""
        runner = get_runner(notebook_path)
        aria = get_aria()
        summary = nb.get_dashboard_summary()
        summary["leaderboard_consistency"] = nb.get_leaderboard_consistency_report()
        runner_state = resolve_runner_status(nb, runner)
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

    @app.route("/api/recompute-failure-signatures", methods=["POST"])
    @wnb
    def api_recompute_failure_signatures(nb=None):
        """Delete and rebuild failure_signatures using S1-only failures."""
        count = nb.recompute_failure_signatures()
        return jsonify({"status": "ok", "signatures_created": count})

    @app.route("/api/reset-op-stats", methods=["POST"])
    @wnb
    def api_reset_op_stats(nb=None):
        """Reset op_success_rates for specific ops so they get a fresh start.

        POST body: {"ops": ["op1", "op2", ...]}
        If no ops specified, resets all ops with 0 S1 passes.
        """
        data = request.get_json(silent=True) or {}
        ops = data.get("ops")
        if ops:
            for op_name in ops:
                nb.conn.execute(
                    "DELETE FROM op_success_rates WHERE op_name = ?",
                    (op_name,),
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

    @app.route("/api/healer/tasks")
    @wnb
    def api_healer_tasks(nb=None):
        """List recent Code Healer tasks."""
        limit = request.args.get("limit", 20, type=int)
        return jsonify(nb.get_recent_healer_tasks(limit=max(1, min(limit, 200))))

    @app.route("/api/healer/tasks/<task_id>")
    @wnb
    def api_healer_task_detail(task_id: str, nb=None):
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

    @app.route("/api/entries")
    @wnb
    def api_entries(nb=None):
        """List notebook entries."""
        exp_id = request.args.get("experiment_id")
        entry_type = request.args.get("type")
        n = request.args.get("n", 50, type=int)
        entries = nb.get_entries(experiment_id=exp_id, entry_type=entry_type, limit=n)
        return jsonify(normalize_entries(entries))

    @app.route("/api/metrics/<metric_name>")
    @wnb
    def api_metrics(metric_name, nb=None):
        """Get time-series metrics."""
        exp_id = request.args.get("experiment_id")
        return jsonify(nb.get_metrics(metric_name, experiment_id=exp_id))

    @app.route("/api/dashboard")
    @app.route("/api/dashboard/summary")
    @wnb
    def api_dashboard(nb=None):
        """Get all dashboard data in one call."""
        runner = get_runner(notebook_path)
        aria = get_aria()
        compact = request.path.endswith("/summary") or (
            str(request.args.get("compact", "0")).strip().lower()
            in {"1", "true", "yes"}
        )
        summary = nb.get_dashboard_summary()
        summary["leaderboard_consistency"] = nb.get_leaderboard_consistency_report()
        runner_state = resolve_runner_status(nb, runner)

        # Add campaign/hypothesis/knowledge counts
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
        except Exception as e:
            logger.warning("Failed enriching dashboard campaign metadata: %s", e)

        recent_experiments = nb.get_recent_experiments(30)
        recent_ids = [
            str(exp.get("experiment_id"))
            for exp in recent_experiments
            if exp.get("experiment_id")
        ]
        recent_results = {}
        if recent_ids:
            placeholders = ",".join("?" for _ in recent_ids)
            rows = nb.conn.execute(
                f"""
                SELECT experiment_id, results_json
                FROM experiments
                WHERE experiment_id IN ({placeholders})
                """,
                recent_ids,
            ).fetchall()
            for row in rows:
                parsed = (
                    nb._decompress(row["results_json"]) if row["results_json"] else None
                )
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
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        top_programs = nb.get_top_programs(10)
        annotate_qkv_usage(top_programs, analytics)
        production_readiness = compute_breakthrough_production_readiness(nb, analytics)
        insights = deduplicate_insights(nb.get_insights(limit=50))
        recent_entries = normalize_entries(nb.get_entries(limit=20))

        if compact:
            recent_experiments = [
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
            top_programs = [
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
            insights = [
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
            recent_entries = [
                {
                    "entry_id": row.get("entry_id"),
                    "experiment_id": row.get("experiment_id"),
                    "entry_type": row.get("entry_type"),
                    "timestamp": row.get("timestamp"),
                    "content": str(row.get("content") or "")[:180],
                }
                for row in recent_entries[:8]
            ]

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
        }

        # Compute deltas from latest completed experiment
        try:
            completed = [
                e for e in recent_experiments if e.get("status") == "completed"
            ]
            if len(completed) >= 2:
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

        # Include learning trajectory trend in summary
        try:
            trajectory = analytics.learning_trajectory()
            if trajectory and trajectory.get("trend") != "insufficient_data":
                summary["learning_trend"] = trajectory.get("trend")
                summary["learning_slope"] = trajectory.get("slope")
                summary["recent_s1_rate"] = trajectory.get("recent_s1_rate")
        except Exception as exc:
            logger.debug("Suppressed error: %s", exc)

        # Include latest auto-recommendation if experiment just completed
        last_rec = runner.last_recommendation
        if last_rec:
            data["last_recommendation"] = last_rec

        return jsonify(data)

    @app.route("/api/report")
    @wnb
    def api_report(nb=None):
        """Consolidated research report with all data."""
        aria = get_aria()
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)

        fast_mode = parse_bool_query(request.args.get("fast"), default=False)
        include_heavy = parse_bool_query(
            request.args.get("include_heavy"),
            default=not fast_mode,
        )
        include_narrative = parse_bool_query(
            request.args.get("include_narrative"),
            default=not fast_mode,
        )

        top_limit = 20 if not fast_mode else 12
        expanded_limit = 80 if include_heavy else 0
        recent_limit = 100 if include_heavy else 30

        data = {
            "summary": nb.get_dashboard_summary(),
            "top_programs": nb.get_report_top_programs_grouped_by_fingerprint(
                top_limit, sort_by="loss_ratio"
            ),
            "top_programs_expanded": nb.get_top_programs(
                expanded_limit, sort_by="loss_ratio"
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
            },
        }
        if include_heavy:
            data.update(
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

        data["cross_run_stability"] = compute_cross_run_stability(
            nb, data["top_programs"]
        )
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
            by_fingerprint = stability_by_fingerprint.get(
                program.get("graph_fingerprint")
            )
            program["cross_run_stability"] = (
                by_result or by_fingerprint or fallback_stability
            )

        # Generate narrative only when explicitly enabled
        data["narrative"] = None
        if include_narrative:
            try:
                narrative = aria.generate_report_narrative(data)
                data["narrative"] = narrative
            except Exception as e:
                logger.debug(f"Report narrative generation failed: {e}")
                data["narrative"] = None

        return jsonify(data)

    @app.route("/api/report/query")
    @wnb
    def api_report_query(nb=None):
        """Scoped report payload for date/theme/trend report generation."""
        aria = get_aria()
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)

        start_ts = parse_report_date(request.args.get("start_date"), end_of_day=False)
        end_ts = parse_report_date(request.args.get("end_date"), end_of_day=True)
        theme = str(request.args.get("theme") or "all").strip().lower()
        trend = str(request.args.get("trend") or "all").strip().lower()
        include_narrative = parse_bool_query(
            request.args.get("include_narrative"),
            default=False,
        )
        try:
            limit = int(request.args.get("limit") or 20)
        except (TypeError, ValueError):
            limit = 20
        limit = max(5, min(120, limit))

        snapshot_query = {
            "start_date": request.args.get("start_date"),
            "end_date": request.args.get("end_date"),
            "theme": theme,
            "trend": trend,
            "limit": limit,
            "include_narrative": bool(include_narrative),
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

        sort_by = "novelty_score" if trend == "high_novelty" else "loss_ratio"
        expanded = nb.get_top_programs(max(limit * 3, 120), sort_by=sort_by)
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

        base_summary = nb.get_dashboard_summary()
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
                "start_date": request.args.get("start_date"),
                "end_date": request.args.get("end_date"),
                "theme": theme,
                "trend": trend,
                "limit": limit,
                "matched_experiments": len(filtered_experiments),
                "matched_programs": len(filtered_programs),
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
            except Exception as e:
                logger.debug(f"Scoped report narrative generation failed: {e}")
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
            except Exception as e:
                logger.debug(f"Scoped report snapshot save failed: {e}")

        return jsonify(data)
