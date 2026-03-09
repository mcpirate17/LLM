"""misc API route registration."""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import traceback
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import jsonify, request, Response, send_from_directory
from ..json_utils import json_safe as _json_safe
from ..notebook import LabNotebook
from ..runner import RunConfig
from ..persona import get_aria
from ..native_runner import native_runner_capability_report
from ..code_agent import _spawn_code_agent_task
from ..evidence import build_evidence_pack
from ..designer_utils import get_designer_components
from ._helpers import (
    get_runner, with_native_runner_progress, get_run_trigger_snapshot,
    deduplicate_insights, normalize_result_ids, record_run_trigger,
    get_autonomy, native_runner_canary_status_payload,
)
from ._strategy import (
    parse_report_date, report_program_matches_theme,
    report_experiment_matches_trend, build_filtered_report_summary,
    build_report_snapshot_key, build_report_action_eligibility,
    annotate_qkv_usage, compute_cross_run_stability,
    compute_breakthrough_production_readiness, compute_recommendation,
    normalize_start_mode, build_start_mode_eligibility,
    compute_compression_opportunities, compute_sparse_evidence,
    sparse_coverage_summary, diagnose_research_issues,
    run_pipeline_sample_check, run_launch_preflight,
    normalize_entries, parse_bool_query,
    normalize_briefing_mode, briefing_action_from_mode,
    briefing_action_label, augment_sparse_action_config,
)
from ._designer import (
    designer_service_status, designer_touch_activity, designer_idle_state,
    start_designer_services, stop_designer_services,
    designer_proxy, proxy_or_error, proxy_stream,
)
from ._chat import (
    chat_requests_detailed_response, chat_requests_summary_response,
    chat_requests_brief_response, chat_requests_self_fix_now,
    chat_requests_codebase_fix,
    record_chat_guardrail_event, chat_guardrail_snapshot,
    code_agent_task_snapshot, summarize_agent_task,
    run_local_chat_agent, chat_workspace_root, query_file_index,
    parse_action_contract_response, truncate_summary, estimate_tokens,
    local_ollama_helper_status, get_local_ollama_settings,
)
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)

# Designer dist path for static serving
_designer_dist = str(Path(__file__).resolve().parents[3] / "aria_designer" / "ui" / "dist")


def register_misc_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    _dashboard_index_path = context.dashboard_index_path
    _dashboard_missing_response = context.dashboard_missing_response
    _is_asset_path = context.is_asset_path

    @app.route("/api/status")
    def api_status():
        """Get Aria's current status and dashboard summary."""
        nb = LabNotebook(notebook_path)
        runner = get_runner(notebook_path)
        aria = get_aria()
        try:
            summary = nb.get_dashboard_summary()
            progress_payload = with_native_runner_progress(runner.progress.to_dict())
            trigger = get_run_trigger_snapshot(progress_payload.get("experiment_id"))
            progress_payload["run_trigger_source"] = trigger.get("source")
            progress_payload["run_trigger"] = trigger
            return jsonify({
                "aria": aria.get_status(db_summary=summary),
                "summary": summary,
                "is_running": runner.is_running,
                "progress": progress_payload,
                "native_runner": progress_payload.get("native_runner"),
                "run_trigger_source": trigger.get("source"),
                "run_trigger": trigger,
            })
        except Exception as e:
            logger.error(f"Error in /api/status: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/recompute-failure-signatures", methods=["POST"])
    def api_recompute_failure_signatures():
        """Delete and rebuild failure_signatures using S1-only failures."""
        nb = LabNotebook(notebook_path)
        try:
            count = nb.recompute_failure_signatures()
            return jsonify({"status": "ok", "signatures_created": count})
        except Exception as e:
            logger.error(f"Error in /api/recompute-failure-signatures: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/reset-op-stats", methods=["POST"])
    def api_reset_op_stats():
        """Reset op_success_rates for specific ops so they get a fresh start.

        POST body: {"ops": ["op1", "op2", ...]}
        If no ops specified, resets all ops with 0 S1 passes.
        """
        nb = LabNotebook(notebook_path)
        try:
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
        except Exception as e:
            logger.error(f"Error in /api/reset-op-stats: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/healer/tasks")
    def api_healer_tasks():
        """List recent Code Healer tasks."""
        nb = LabNotebook(notebook_path)
        try:
            limit = request.args.get("limit", 20, type=int)
            return jsonify(nb.get_recent_healer_tasks(limit=max(1, min(limit, 200))))
        except Exception as e:
            logger.error(f"Error in /api/healer/tasks: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/healer/tasks/<task_id>")
    def api_healer_task_detail(task_id: str):
        """Get one healer task with state history."""
        nb = LabNotebook(notebook_path)
        try:
            task = nb.get_healer_task(task_id)
            if task is None:
                return jsonify({"error": "Not found"}), 404
            return jsonify({
                "task": task,
                "events": nb.get_healer_events(task_id, limit=200),
            })
        except Exception as e:
            logger.error(f"Error in /api/healer/tasks/{task_id}: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/entries")
    def api_entries():
        """List notebook entries."""
        exp_id = request.args.get("experiment_id")
        entry_type = request.args.get("type")
        n = request.args.get("n", 50, type=int)
        nb = LabNotebook(notebook_path)
        try:
            entries = nb.get_entries(
                experiment_id=exp_id, entry_type=entry_type, limit=n
            )
            return jsonify(normalize_entries(entries))
        except Exception as e:
            logger.error(f"Error in /api/entries: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/metrics/<metric_name>")
    def api_metrics(metric_name):
        """Get time-series metrics."""
        exp_id = request.args.get("experiment_id")
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_metrics(metric_name, experiment_id=exp_id))
        except Exception as e:
            logger.error(f"Error in /api/metrics/{metric_name}: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/dashboard")
    def api_dashboard():
        """Get all dashboard data in one call."""
        runner = get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            summary = nb.get_dashboard_summary()

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
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            top_programs = nb.get_top_programs(10)
            annotate_qkv_usage(top_programs, analytics)
            production_readiness = compute_breakthrough_production_readiness(nb, analytics)

            data = {
                "aria": aria.get_status(db_summary=summary),
                "summary": summary,
                "recent_experiments": recent_experiments,
                "top_programs": top_programs,
                "production_readiness": production_readiness,
                "insights": deduplicate_insights(nb.get_insights(limit=50)),
                "recent_entries": normalize_entries(nb.get_entries(limit=20)),
                "is_running": runner.is_running,
                "progress": with_native_runner_progress(runner.progress.to_dict()),
            }

            # Compute deltas from latest completed experiment
            try:
                completed = [e for e in recent_experiments
                             if e.get("status") == "completed"]
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
                            - (previous.get("best_loss_ratio") or 1), 4
                        ) if latest.get("best_loss_ratio") else None,
                        "best_novelty": round(
                            (latest.get("best_novelty_score") or 0)
                            - (previous.get("best_novelty_score") or 0), 4
                        ) if latest.get("best_novelty_score") else None,
                    }
            except Exception:
                pass

            # Include learning trajectory trend in summary
            try:
                trajectory = analytics.learning_trajectory()
                if trajectory and trajectory.get("trend") != "insufficient_data":
                    summary["learning_trend"] = trajectory.get("trend")
                    summary["learning_slope"] = trajectory.get("slope")
                    summary["recent_s1_rate"] = trajectory.get("recent_s1_rate")
            except Exception:
                pass

            # Include latest auto-recommendation if experiment just completed
            last_rec = runner.last_recommendation
            if last_rec:
                data["last_recommendation"] = last_rec

            return jsonify(data)
        except Exception as e:
            logger.error(f"Error in /api/dashboard: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/report")
    def api_report():
        """Consolidated research report with all data."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
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
                "top_programs": nb.get_report_top_programs_grouped_by_fingerprint(top_limit, sort_by="loss_ratio"),
                "top_programs_expanded": nb.get_top_programs(expanded_limit, sort_by="loss_ratio") if include_heavy else [],
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
                data.update({
                    "math_family_coverage": analytics.math_family_coverage(),
                    "mathspace_operator_impact": analytics.mathspace_operator_impact(),
                    "routing_mode_comparison": analytics.routing_mode_comparison(),
                    "gating_behavior_diagnostics": analytics.gating_behavior_diagnostics(),
                    "structural_correlations": analytics.structural_correlations(),
                    "top_op_combinations": analytics.top_op_combinations(10),
                    "efficiency_frontier": analytics.efficiency_frontier(),
                    "experiment_clusters": analytics.experiment_clusters(),
                })
            learning_diagnostics = data["grammar_weights"].get("learning_diagnostics") or {}
            data["architecture_rerun_telemetry"] = {
                "unique_fingerprint_count": int(learning_diagnostics.get("unique_fingerprints") or 0),
                "total_result_rows": int(learning_diagnostics.get("total_rows") or 0),
                "repeat_result_rows": int(learning_diagnostics.get("repeat_rows") or 0),
                "rerun_ratio": float(learning_diagnostics.get("rerun_ratio") or 0.0),
                "top_fingerprint_concentration": float(learning_diagnostics.get("top_fingerprint_concentration") or 0.0),
                "weighting_mode": str(learning_diagnostics.get("mode") or "unknown"),
            }
            data["action_eligibility"] = build_report_action_eligibility(
                nb,
                [
                    row.get("result_id")
                    for row in [*(data["top_programs"] or []), *(data["top_programs_expanded"] or [])]
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
            for program in [*(data["top_programs"] or []), *(data["top_programs_expanded"] or [])]:
                by_result = stability_by_result.get(program.get("result_id"))
                by_fingerprint = stability_by_fingerprint.get(program.get("graph_fingerprint"))
                program["cross_run_stability"] = by_result or by_fingerprint or fallback_stability

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
        except Exception as e:
            logger.error(f"Error in /api/report: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/report/query")
    def api_report_query():
        """Scoped report payload for date/theme/trend report generation."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
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
            except Exception:
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
        except Exception as e:
            logger.error(f"Error in /api/report/query: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/decision-packet/<result_id>")
    def api_decision_packet(result_id):
        """One-click evidence bundle for promotion decisions."""
        nb = LabNotebook(notebook_path)
        try:
            program = nb.get_program_detail(result_id)
            if program is None:
                return jsonify({"error": "Not found"}), 404

            fingerprint = program.get("graph_fingerprint", "")
            experiment_id = program.get("experiment_id")

            # Leaderboard entry (targeted)
            leaderboard_entry = None
            try:
                leaderboard_entry = nb.get_leaderboard_entry(result_id)
            except Exception:
                leaderboard_entry = None

            # Experiment data + failure analysis
            experiment = None
            failure_analysis = {"funnel": {}, "errors": {}, "stage_deaths": {}}
            if experiment_id:
                try:
                    experiment = nb.get_experiment(experiment_id)
                except Exception:
                    pass
                try:
                    failure_analysis = nb.get_failure_analysis(experiment_id)
                except Exception:
                    pass

            # Hypothesis chain — find hypothesis linked to this experiment
            hypothesis_chain = []
            if experiment_id:
                try:
                    hyp_row = nb.conn.execute(
                        "SELECT hypothesis_id FROM hypotheses WHERE experiment_id = ?",
                        (experiment_id,),
                    ).fetchone()
                    if hyp_row:
                        hypothesis_chain = nb.get_hypothesis_chain(
                            hyp_row["hypothesis_id"] if isinstance(hyp_row, dict)
                            else hyp_row[0]
                        )
                except Exception:
                    pass

            # Cross-run stability for this specific result
            cross_run = {"trend": "unknown", "seen_runs": 0}
            try:
                top = nb.get_top_programs(20, sort_by="loss_ratio")
                stability = compute_cross_run_stability(nb, top)
                for c in stability.get("candidates", []):
                    if c.get("result_id") == result_id:
                        cross_run = {
                            "trend": c.get("trend", "unknown"),
                            "seen_runs": c.get("seen_runs", 0),
                        }
                        break
            except Exception:
                pass

            # Build outcomes by phase
            tier = (leaderboard_entry or {}).get("tier", "screening")
            outcomes = {
                "screening": {
                    "loss_ratio": program.get("loss_ratio"),
                    "novelty": program.get("novelty_score"),
                },
                "investigation": None,
                "validation": None,
            }
            if leaderboard_entry:
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

            # Baseline comparison
            bl_ratio = program.get("baseline_loss_ratio")
            baseline_comparison = {"ratio": bl_ratio, "interpretation": "unknown"}
            if bl_ratio is not None:
                if bl_ratio < 0.95:
                    baseline_comparison["interpretation"] = "outperforms"
                elif bl_ratio <= 1.05:
                    baseline_comparison["interpretation"] = "comparable"
                else:
                    baseline_comparison["interpretation"] = "underperforms"

            # Failure context
            failure_context = {
                "stage_at_death": program.get("stage_at_death"),
                "error_type": program.get("error_type"),
                "experiment_errors": failure_analysis.get("errors", {}),
                "experiment_funnel": failure_analysis.get("funnel", {}),
            }

            # Recommendation
            recommendation = compute_recommendation(program, leaderboard_entry)

            # Evidence flags
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            packet_status = analytics.reproducibility_packet_status(
                leaderboard_entry if leaderboard_entry else program
            )
            evidence_flags = {
                "has_baseline": bl_ratio is not None,
                "has_cka_artifact": program.get("cka_source") == "artifact",
                "has_multi_seed": outcomes["validation"] is not None,
                "has_hypothesis": len(hypothesis_chain) > 0,
                "repro_packet_ready": packet_status.get("status") == "ready",
            }

            return jsonify({
                "result_id": result_id,
                "fingerprint": fingerprint,
                "experiment_id": experiment_id,
                "hypothesis_chain": hypothesis_chain,
                "outcomes": outcomes,
                "baseline_comparison": baseline_comparison,
                "failure_context": failure_context,
                "cross_run_stability": cross_run,
                "recommendation": recommendation,
                "evidence_flags": evidence_flags,
                "compression_metrics": analytics.canonical_compression_metrics(
                    leaderboard_entry if leaderboard_entry else program
                ),
                "reproducibility_packet": packet_status,
            })
        except Exception as e:
            logger.error(f"Error in /api/decision-packet/{result_id}: {e}\n"
                         f"{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/reproducibility-manifest/<result_id>")
    def api_reproducibility_manifest(result_id):
        """Exportable reproducibility manifest for a program result."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            program = nb.get_program_detail(result_id)
            if program is None:
                return jsonify({"error": "Not found"}), 404

            experiment_id = program.get("experiment_id")
            experiment = None
            if experiment_id:
                try:
                    experiment = nb.get_experiment(experiment_id)
                except Exception:
                    pass

            config = (experiment or {}).get("config", {}) or {}
            training = {}
            try:
                tp = json.loads(program.get("training_program_json") or "{}")
                training = tp
            except (json.JSONDecodeError, TypeError):
                pass

            # Grammar weights snapshot from experiment config
            grammar_weights = config.get("applied_grammar_weights") or config.get("grammar_weights")
            grammar_config = config.get("grammar_config", {})

            manifest = {
                "result_id": result_id,
                "graph_fingerprint": program.get("graph_fingerprint"),
                "experiment_id": experiment_id,
                "experiment_type": (experiment or {}).get("experiment_type"),
                "timestamp": program.get("timestamp"),
                "code_version": config.get("code_version"),
                "seeds": {
                    "experiment_seed": config.get("seed"),
                    "training_seed": training.get("seed"),
                },
                "data": {
                    "data_mode": config.get("data_mode"),
                    "dataset": config.get("dataset"),
                    "seq_len": training.get("seq_len") or config.get("seq_len"),
                    "batch_size": training.get("batch_size") or config.get("batch_size"),
                    "vocab_size": training.get("vocab_size") or config.get("vocab_size"),
                },
                "grammar": {
                    "max_ops": grammar_config.get("max_ops"),
                    "max_depth": grammar_config.get("max_depth"),
                    "weights_snapshot": grammar_weights,
                },
                "training": {
                    "learning_rate": training.get("learning_rate") or training.get("lr"),
                    "steps": training.get("steps") or training.get("n_steps"),
                    "warmup_steps": training.get("warmup_steps"),
                },
                "architecture": {
                    "param_count": program.get("param_count"),
                    "graph_json": program.get("graph_json"),
                },
                "outcomes": {
                    "stage0_passed": bool(program.get("stage0_passed")),
                    "stage05_passed": bool(program.get("stage05_passed")),
                    "stage1_passed": bool(program.get("stage1_passed")),
                    "loss_ratio": program.get("loss_ratio"),
                    "discovery_loss_ratio": program.get("discovery_loss_ratio"),
                    "validation_loss_ratio": program.get("validation_loss_ratio"),
                    "novelty_score": program.get("novelty_score"),
                    "baseline_loss_ratio": program.get("baseline_loss_ratio"),
                    "validation_baseline_ratio": program.get("validation_baseline_ratio"),
                },
                "canonical_metrics": {
                    "compression": analytics.canonical_compression_metrics(program),
                },
                "packet_status": analytics.reproducibility_packet_status(program),
            }
            return jsonify(manifest)
        except Exception as e:
            logger.error(f"Error in /api/reproducibility-manifest/{result_id}: {e}\n"
                         f"{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/reproducibility-manifest/<result_id>/workflow", methods=["GET"])
    def api_workflow_export(result_id: str):
        """Export a program result as an aria_designer workflow JSON."""
        nb = LabNotebook(notebook_path)
        try:
            row = nb.conn.execute(
                "SELECT graph_json, model_dim FROM program_results WHERE result_id = ?",
                (result_id,),
            ).fetchone()
            if not row or not row["graph_json"]:
                return jsonify({"error": "Program not found or has no graph"}), 404
            
            from ..synthesis.serializer import graph_from_json
            from ..synthesis.workflow_converter import graph_to_workflow
            
            graph = graph_from_json(row["graph_json"], model_dim=row["model_dim"])
            workflow = graph_to_workflow(
                graph, 
                workflow_id=f"aria_{result_id[:8]}",
                name=f"Aria Discovery {result_id[:8]}",
                metadata={"result_id": result_id}
            )
            return jsonify(workflow)
        except Exception as e:
            logger.error(f"Error exporting workflow for {result_id}: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/references")
    def api_references():
        """Get pinned reference architectures."""
        nb = LabNotebook(notebook_path)
        try:
            from ..naming import annotate_display_names
            refs = nb.get_references()
            annotate_display_names(refs)
            return jsonify({
                "entries": _json_safe(refs),
                "total": len(refs),
            })
        except Exception as e:
            logger.error(f"Error in /api/references: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/fingerprint/resolve")
    def api_fingerprint_resolve():
        """Resolve a result_id or fingerprint prefix to a concrete program result.

        Preference order for fingerprint prefixes:
        1) Best leaderboard-backed run (highest composite score)
        2) Best surviving run by loss ratio
        """
        value = str(request.args.get("value") or "").strip()
        if not value:
            return jsonify({"error": "value query param required"}), 400
        nb = LabNotebook(notebook_path)
        try:
            direct = nb.conn.execute(
                "SELECT result_id, graph_fingerprint FROM program_results WHERE result_id = ?",
                (value,),
            ).fetchone()
            if direct:
                return jsonify({
                    "result_id": direct["result_id"],
                    "graph_fingerprint": direct.get("graph_fingerprint"),
                    "resolved_from": "result_id",
                    "candidates": [],
                })
            rows = nb.conn.execute(
                """
                SELECT
                    pr.result_id,
                    pr.graph_fingerprint,
                    pr.experiment_id,
                    pr.stage1_passed,
                    pr.loss_ratio,
                    pr.timestamp,
                    lb.tier,
                    lb.composite_score,
                    lb.screening_loss_ratio,
                    lb.investigation_loss_ratio,
                    lb.validation_loss_ratio
                FROM program_results pr
                LEFT JOIN leaderboard lb ON lb.result_id = pr.result_id
                WHERE pr.graph_fingerprint LIKE ?
                ORDER BY
                    CASE WHEN lb.result_id IS NULL THEN 1 ELSE 0 END ASC,
                    COALESCE(lb.composite_score, -1e9) DESC,
                    pr.stage1_passed DESC,
                    (pr.loss_ratio IS NULL) ASC,
                    pr.loss_ratio ASC,
                    pr.timestamp DESC
                LIMIT 50
                """,
                (f"{value}%",),
            ).fetchall()
            if rows:
                chosen_row = dict(rows[0])
                candidates = []
                for row in rows:
                    candidates.append({
                        "result_id": row["result_id"],
                        "graph_fingerprint": row["graph_fingerprint"],
                        "experiment_id": row["experiment_id"],
                        "stage1_passed": bool(row["stage1_passed"]),
                        "loss_ratio": row["loss_ratio"],
                        "timestamp": row["timestamp"],
                        "tier": row["tier"],
                        "composite_score": row["composite_score"],
                        "screening_loss_ratio": row["screening_loss_ratio"],
                        "investigation_loss_ratio": row["investigation_loss_ratio"],
                        "validation_loss_ratio": row["validation_loss_ratio"],
                    })
                return jsonify({
                    "result_id": chosen_row.get("result_id"),
                    "graph_fingerprint": chosen_row.get("graph_fingerprint"),
                    "resolved_from": "graph_fingerprint",
                    "candidate_count": len(candidates),
                    "selection_policy": "leaderboard_composite_then_loss",
                    "candidates": candidates,
                })
            return jsonify({"error": "No matching fingerprint or result_id found."}), 404
        except Exception as e:
            logger.error(f"Error in /api/fingerprint/resolve: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/fingerprint/history")
    def api_fingerprint_history():
        """Return chronological run history for a fingerprint prefix/result_id."""
        value = str(request.args.get("value") or "").strip()
        limit = int(request.args.get("limit", 100) or 100)
        limit = max(1, min(limit, 500))
        if not value:
            return jsonify({"error": "value query param required"}), 400
        nb = LabNotebook(notebook_path)
        try:
            direct = nb.conn.execute(
                "SELECT graph_fingerprint FROM program_results WHERE result_id = ? LIMIT 1",
                (value,),
            ).fetchone()
            fingerprint_like = (direct["graph_fingerprint"] if direct else value) + "%"
            rows = nb.conn.execute(
                """
                SELECT
                    pr.result_id,
                    pr.graph_fingerprint,
                    pr.experiment_id,
                    pr.timestamp,
                    pr.stage0_passed,
                    pr.stage05_passed,
                    pr.stage1_passed,
                    pr.loss_ratio,
                    pr.discovery_loss_ratio,
                    pr.validation_loss_ratio,
                    lb.tier,
                    lb.composite_score,
                    lb.screening_loss_ratio,
                    lb.investigation_loss_ratio,
                    lb.validation_loss_ratio AS lb_validation_loss_ratio,
                    lb.investigation_passed,
                    lb.validation_passed
                FROM program_results pr
                LEFT JOIN leaderboard lb ON lb.result_id = pr.result_id
                WHERE pr.graph_fingerprint LIKE ?
                ORDER BY pr.timestamp DESC
                LIMIT ?
                """,
                (fingerprint_like, limit),
            ).fetchall()
            history = [dict(r) for r in rows]
            best_row = nb.conn.execute(
                """
                SELECT
                    pr.result_id,
                    pr.graph_fingerprint,
                    pr.experiment_id,
                    pr.timestamp,
                    pr.loss_ratio,
                    lb.tier,
                    lb.composite_score,
                    lb.validation_loss_ratio
                FROM program_results pr
                JOIN leaderboard lb ON lb.result_id = pr.result_id
                WHERE pr.graph_fingerprint LIKE ?
                  AND lb.composite_score IS NOT NULL
                ORDER BY lb.composite_score DESC, pr.timestamp DESC
                LIMIT 1
                """,
                (fingerprint_like,),
            ).fetchone()
            best_by_composite = dict(best_row) if best_row else None
            return jsonify({
                "query": value,
                "resolved_graph_fingerprint": history[0]["graph_fingerprint"] if history else None,
                "total": len(history),
                "best_leaderboard_run": best_by_composite,
                "runs": history,
            })
        except Exception as e:
            logger.error(f"Error in /api/fingerprint/history: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/worker/evaluate", methods=["POST"])
    def api_worker_evaluate():
        """Z12: Distributed worker endpoint for evaluating a computation graph."""
        runner = get_runner(notebook_path)
        body = request.get_json(silent=True) or {}
        
        graph_json = body.get("graph_json")
        config_dict = body.get("config")
        seed = body.get("seed", 42)
        
        if not graph_json or not config_dict:
            return jsonify({"error": "Missing graph_json or config"}), 400
            
        try:
            from ..synthesis.graph import json_to_graph
            from ..synthesis.compiler import compile_model
            import torch
            
            graph = json_to_graph(graph_json)
            config = RunConfig.from_dict(config_dict)
            
            dev_str = config.device if torch.cuda.is_available() else "cpu"
            dev = torch.device(dev_str)
            
            # Compile model locally on worker
            layer_graphs = [graph] * config.n_layers
            model = compile_model(
                layer_graphs,
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
            ).to(dev)
            
            # Use the runner's async-friendly training method
            result = runner._micro_train_async(model, config, seed, dev)
            
            return jsonify({
                "status": "ok",
                "result": result,
                "device": dev_str,
                "worker_id": os.environ.get("ARIA_WORKER_ID", "anonymous")
            })
            
        except Exception as e:
            logger.error("Worker evaluation failed: %s", e)
            return jsonify({"error": str(e), "passed": False}), 500


    @app.route("/api/progress")
    def api_progress():
        """Get current experiment progress (poll-based alternative to SSE)."""
        runner = get_runner(notebook_path)
        progress_payload = with_native_runner_progress(runner.progress.to_dict())
        trigger = get_run_trigger_snapshot(progress_payload.get("experiment_id"))
        progress_payload["run_trigger_source"] = trigger.get("source")
        progress_payload["run_trigger"] = trigger
        return jsonify({
            "is_running": runner.is_running,
            "progress": progress_payload,
            "native_runner": progress_payload.get("native_runner"),
            "run_trigger_source": trigger.get("source"),
            "run_trigger": trigger,
        })


    @app.route("/api/strategy/briefing")
    def api_strategy_briefing():
        """Data-driven strategy briefing for the overview page.

        Tries LLM-powered briefing first (via Aria), falls back to
        deterministic rules.  Always returns a valid response.
        """
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            summary = nb.get_dashboard_summary()
            recent = nb.get_recent_experiments(10)
            trajectory = analytics.learning_trajectory() or {}
            compression_coverage = analytics.compression_coverage() or {}
            compression_opportunities = compute_compression_opportunities(compression_coverage)
            primitive_effectiveness = analytics.compression_primitive_effectiveness() or {}
            sparse_evidence = compute_sparse_evidence(nb)
            sparse_coverage_data = analytics.sparse_coverage() or {}
            sparse_coverage_overview = sparse_coverage_summary(sparse_coverage_data)

            # Optional: highlight a just-completed experiment
            just_completed_id = request.args.get("just_completed")
            just_completed_exp = None
            if just_completed_id:
                for e in recent:
                    if (e.get("experiment_id") or "").startswith(just_completed_id):
                        just_completed_exp = e
                        break
                # Clear briefing cache so LLM sees the new context
                aria_inst = get_aria()
                if hasattr(aria_inst, "_briefing_cache"):
                    aria_inst._briefing_cache = None

            # --- Pipeline counts (exclude pinned reference architectures) ---
            leaderboard_rows = nb.conn.execute(
                "SELECT tier, COUNT(*) as cnt FROM leaderboard "
                "WHERE COALESCE(is_reference, 0) = 0 GROUP BY tier"
            ).fetchall()
            tiers = {r["tier"]: r["cnt"] for r in leaderboard_rows}
            screening = tiers.get("screening", 0)
            investigation = tiers.get("investigation", 0)
            validation = tiers.get("validation", 0)
            breakthrough = tiers.get("breakthrough", 0)

            # --- Recent outcomes ---
            completed = [e for e in recent if e.get("status") == "completed"]
            recent_s1_rates = []
            for e in completed[:5]:
                gen = e.get("n_programs_generated") or 0
                passed = e.get("n_stage1_passed") or 0
                if gen > 0:
                    recent_s1_rates.append(passed / gen)

            avg_recent_s1 = (
                sum(recent_s1_rates) / len(recent_s1_rates)
                if recent_s1_rates
                else None
            )

            # --- Learning trend ---
            trend = trajectory.get("trend", "insufficient_data")
            slope = trajectory.get("slope")

            # --- Common data block (used by both LLM and deterministic) ---
            total_exp = summary.get("total_experiments", 0)
            total_progs = summary.get("total_programs_evaluated", 0)
            s1_survivors = summary.get("stage1_survivors", 0)

            pipeline_data = {
                "screening": screening,
                "investigation": investigation,
                "validation": validation,
                "breakthrough": breakthrough,
            }
            compression_summary = (compression_opportunities.get("summary") or {})
            data_block = {
                "total_experiments": total_exp,
                "total_programs": total_progs,
                "s1_survivors": s1_survivors,
                "avg_recent_s1_rate": avg_recent_s1,
                "learning_trend": trend,
                "learning_slope": slope,
                "pipeline": pipeline_data,
                "compression": compression_summary,
                "compression_primitives": primitive_effectiveness.get("primitives", []),
                "sparse": sparse_evidence,
            }

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
            recent_zero_s1_runs = 0
            for exp in recent_completed_window:
                gen = exp.get("n_programs_generated") or 0
                passed = exp.get("n_stage1_passed") or 0
                if gen > 0 and passed == 0:
                    recent_zero_s1_runs += 1

            recommendation_evidence = {
                "learning_trend": trend,
                "learning_slope": slope,
                "avg_recent_s1_rate": avg_recent_s1,
                "recent_completed_runs": len(recent_completed_window),
                "recent_zero_s1_runs": recent_zero_s1_runs,
                "recent_cancelled_runs": recent_cancelled,
                "recent_failed_runs": recent_failed,
                "pipeline": pipeline_data,
                "compression": compression_summary,
                "compression_primitives": primitive_effectiveness.get("primitives", []),
                "sparse": sparse_evidence,
                "sparse_coverage": sparse_coverage_overview,
            }

            # --- Try LLM-powered briefing first ---
            aria = get_aria()
            fallback_reason: Optional[str] = None
            llm = aria._get_llm()
            llm_reachable = False
            if llm is None:
                fallback_reason = "llm_not_configured"
            else:
                try:
                    llm_reachable = bool(llm.is_available()) if hasattr(llm, "is_available") else True
                except Exception:
                    llm_reachable = False
                if not llm_reachable:
                    fallback_reason = "llm_unreachable"
            ref_comparison = None
            try:
                from ..llm.context import build_briefing_context

                # Gather extra context for LLM
                try:
                    active_campaigns = nb.get_active_campaigns()
                    campaign = active_campaigns[0] if active_campaigns else None
                except Exception:
                    campaign = None

                try:
                    dw = analytics.get_current_grammar_weights() or {}
                except Exception:
                    dw = {}

                try:
                    gw = analytics.compute_grammar_weights() or {}
                except Exception:
                    gw = {}

                try:
                    top_programs = nb.conn.execute(
                        "SELECT graph_fingerprint, loss_ratio, novelty_score, tier "
                        "FROM leaderboard WHERE COALESCE(is_reference, 0) = 0 "
                        "ORDER BY composite_score DESC LIMIT 3"
                    ).fetchall()
                    top_progs = [dict(r) for r in top_programs] if top_programs else None
                except Exception:
                    top_progs = None

                # --- Reference comparison: surface when synthesized models beat references ---
                try:
                    ref_rows = nb.conn.execute(
                        "SELECT reference_name, composite_score, loss_ratio "
                        "FROM leaderboard WHERE COALESCE(is_reference, 0) = 1 "
                        "ORDER BY composite_score DESC"
                    ).fetchall()
                    best_ref_score = max((r["composite_score"] for r in ref_rows), default=None)
                    if best_ref_score and top_progs:
                        best_synth_score = nb.conn.execute(
                            "SELECT composite_score FROM leaderboard "
                            "WHERE COALESCE(is_reference, 0) = 0 "
                            "ORDER BY composite_score DESC LIMIT 1"
                        ).fetchone()
                        if best_synth_score and best_synth_score["composite_score"] > best_ref_score:
                            ref_comparison = {
                                "beats_all_references": True,
                                "best_synthesized_score": float(best_synth_score["composite_score"]),
                                "best_reference_score": float(best_ref_score),
                                "margin_pct": round(
                                    100.0 * (best_synth_score["composite_score"] - best_ref_score) / best_ref_score, 1
                                ),
                                "references": [
                                    {"name": r["reference_name"], "score": float(r["composite_score"])}
                                    for r in ref_rows
                                ],
                            }
                        else:
                            ref_comparison = {
                                "beats_all_references": False,
                                "best_reference_score": float(best_ref_score),
                                "references": [
                                    {"name": r["reference_name"], "score": float(r["composite_score"])}
                                    for r in ref_rows
                                ],
                            }
                    else:
                        ref_comparison = None
                except Exception:
                    ref_comparison = None

                try:
                    scaling_summary_data = None
                    try:
                        scaling_summary_data = nb.get_scaling_summary()
                    except Exception:
                        pass
                    briefing_context = build_briefing_context(
                        recent_experiments=recent,
                        pipeline_tiers=tiers,
                        learning_trajectory=trajectory,
                        campaign=campaign,
                        grammar_weights=gw,
                        default_weights=dw,
                        top_programs=top_progs,
                        just_completed=just_completed_exp,
                        sparse_coverage=sparse_coverage_data,
                        scaling_summary=scaling_summary_data,
                        ref_comparison=ref_comparison,
                    )
                except Exception:
                    briefing_context = {
                        "pipeline": pipeline_data,
                        "learning": {
                            "trend": trend,
                            "slope": slope,
                            "avg_recent_s1_rate": avg_recent_s1,
                        },
                        "recent_experiments": recent[:5],
                        "campaign": campaign,
                    }

                ai_briefing = aria.generate_briefing(context=briefing_context)
                if ai_briefing and ai_briefing.get("briefing_text"):
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
                    if normalized_mode in ("investigation", "validation") and not suggested_config.get("result_ids"):
                        _tier = "screening" if normalized_mode == "investigation" else "investigation"
                        _tier_rows = nb.conn.execute(
                            f"SELECT result_id FROM leaderboard "
                            f"WHERE tier = ? AND {_tier}_passed = 1 "
                            f"ORDER BY {_tier}_loss_ratio ASC LIMIT 20",
                            (_tier,),
                        ).fetchall()
                        _rids = [r["result_id"] for r in _tier_rows if r["result_id"]]
                        suggested_config["result_ids"] = _rids

                    if normalized_mode in ("investigation", "validation"):
                        _requested = normalize_result_ids(suggested_config.get("result_ids", []))
                        _eligibility = build_start_mode_eligibility(nb, normalized_mode, _requested)
                        _eligible = _eligibility.get("eligible_result_ids") or []
                        if _eligible:
                            suggested_config["result_ids"] = _eligible
                        else:
                            # No actionable candidates under start-mode guardrails — downgrade to continuous
                            normalized_mode = "continuous"
                            action_key = "continuous"
                            _hypothesis = suggested_config.get("hypothesis")
                            suggested_config = {
                                "mode": "continuous",
                                "model_source": "mixed",
                            }
                            if _hypothesis:
                                suggested_config["hypothesis"] = _hypothesis

                    suggested_config = augment_sparse_action_config(
                        suggested_config,
                        normalized_mode,
                        sparse_coverage_data,
                    )
                    return jsonify({
                        "briefing": ai_briefing["briefing_text"],
                        "action": action_key or normalized_mode or "continuous",
                        "action_label": briefing_action_label(
                            normalized_mode, hypothesis),
                        "action_rationale": suggested.get("reasoning", ""),
                        "ai_powered": True,
                        "confidence": ai_briefing.get("confidence", 0.5),
                        "suggested_config": suggested_config or None,
                        "evidence": recommendation_evidence,
                        "data": data_block,
                        "compression_opportunities": compression_opportunities,
                        "ref_comparison": ref_comparison,
                    })
                if fallback_reason is None:
                    fallback_reason = "llm_empty_response"
            except Exception as e:
                logger.warning(f"LLM briefing unavailable, using deterministic: {e}")
                err_msg = str(e)[:120]
                fallback_reason = f"llm_error:{type(e).__name__}: {err_msg}"

            # --- Deterministic fallback: build briefing sentences ---
            sentences = []
            if total_exp > 0:
                sentences.append(
                    f"Across {total_exp} experiments, {total_progs:,} architectures "
                    f"have been evaluated with {s1_survivors} stage-1 survivors "
                    f"({s1_survivors / max(total_progs, 1) * 100:.1f}% overall pass rate)."
                )

            # 2. Recent performance
            if avg_recent_s1 is not None:
                n_recent = len(recent_s1_rates)
                sentences.append(
                    f"The last {n_recent} completed experiment{'s' if n_recent != 1 else ''} "
                    f"averaged a {avg_recent_s1 * 100:.1f}% S1 pass rate."
                )

            # 3. Learning trajectory
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

            # 4. Pipeline state
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
                sentences.append(
                    f"Candidate pipeline: {', '.join(pipeline_parts)}."
                )

            compressed_share = float(compression_summary.get("compressed_test_share") or 0.0)
            compressed_survival = float(compression_summary.get("compressed_survival_rate") or 0.0)
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
                sparse_fragment = (
                    f"Sparse telemetry: {sparse_n} runs with mean density {sparse_density * 100:.1f}%"
                )
                if sparse_nm is not None:
                    sparse_fragment += f", N:M compliance {float(sparse_nm) * 100:.1f}%"
                sparse_fragment += "."
                sentences.append(sparse_fragment)

            # 5. Last experiment outcome
            if completed:
                last = completed[0]
                last_s1 = last.get("n_stage1_passed") or 0
                last_gen = last.get("n_programs_generated") or 0
                last_loss = last.get("best_loss_ratio")
                last_id = last.get("experiment_id", "")[:8]
                parts = [
                    f"Last experiment ({last_id}): "
                    f"{last_s1}/{last_gen} passed S1"
                ]
                if last_loss is not None:
                    parts.append(f"best loss {last_loss:.4f}")
                aria_sum = last.get("aria_summary")
                if aria_sum:
                    parts.append(f"— {aria_sum}")
                sentences.append(". ".join(parts) + ".")

            # 6. Data-driven diversity analysis
            try:
                # Op category distribution from learning log
                op_rows = nb.conn.execute(
                    "SELECT op_name, s1_passes, total_uses FROM op_success_rates "
                    "WHERE total_uses >= 5 ORDER BY "
                    "CAST(s1_passes AS REAL) / CAST(total_uses AS REAL) DESC LIMIT 3"
                ).fetchall()
                if op_rows:
                    top_ops = [f"{r['op_name']} ({r['s1_passes']}/{r['total_uses']})"
                               for r in op_rows]
                    sentences.append(
                        f"Top-performing operators: {', '.join(top_ops)}."
                    )

                # Failure mode analysis
                failure_rows = nb.conn.execute(
                    "SELECT stage_at_death, COUNT(*) as cnt FROM program_results "
                    "WHERE stage1_passed = 0 AND stage_at_death IS NOT NULL "
                    "GROUP BY stage_at_death ORDER BY cnt DESC LIMIT 2"
                ).fetchall()
                if failure_rows:
                    failure_parts = [f"{r['stage_at_death']} ({r['cnt']})"
                                     for r in failure_rows]
                    sentences.append(
                        f"Dominant failure stages: {', '.join(failure_parts)}."
                    )

                # Architecture diversity check
                unique_fps = nb.conn.execute(
                    "SELECT COUNT(DISTINCT SUBSTR(graph_fingerprint, 1, 8)) "
                    "FROM leaderboard"
                ).fetchone()[0]
                total_leaderboard = screening + investigation + validation + breakthrough
                if unique_fps is not None and total_leaderboard > 0:
                    diversity_ratio = unique_fps / total_leaderboard
                    if diversity_ratio < 0.5:
                        sentences.append(
                            f"Warning: only {unique_fps} unique architecture "
                            f"families in {total_leaderboard} "
                            f"leaderboard entries — search may be converging."
                        )
            except Exception:
                pass  # Analytics are optional enhancements

            # 7. Reference architecture comparison
            try:
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
            except Exception:
                pass

            briefing = " ".join(sentences)

            # --- Determine recommended action ---
            action = None
            action_label = None
            action_rationale = None
            screening_result_ids = []

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
                sparse_survival = float(sparse_coverage_overview.get("sparse_survival_rate") or 0.0)
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
                inv_failed = nb.conn.execute(
                    "SELECT COUNT(*) FROM leaderboard "
                    "WHERE tier = 'investigation' AND investigation_passed = 0"
                ).fetchone()[0]
                # Fetch actual result_ids for screening survivors
                screening_rows = nb.conn.execute(
                    "SELECT result_id FROM leaderboard "
                    "WHERE tier = 'screening' AND screening_passed = 1 "
                    "AND COALESCE(is_reference, 0) = 0 "
                    "ORDER BY composite_score DESC LIMIT 20"
                ).fetchall()
                screening_candidate_ids = [r["result_id"] for r in screening_rows if r["result_id"]]
                screening_result_ids = []
                if screening_candidate_ids:
                    screening_eligibility = build_start_mode_eligibility(
                        nb,
                        "investigation",
                        screening_candidate_ids,
                    )
                    screening_result_ids = screening_eligibility.get("eligible_result_ids") or []
                if not screening_result_ids:
                    # No actionable screening survivors — fall through to default
                    action = "continuous"
                    action_label = "Continue Research"
                    action_rationale = (
                        "Screening survivors exist but are not currently eligible for investigation reruns. "
                        "Continue generating new architectures."
                    )
                else:
                    action = "investigate"
                    action_label = (
                        f"Investigate {len(screening_result_ids)} Screening "
                        f"Survivor{'s' if len(screening_result_ids) != 1 else ''}"
                    )
                    rationale_parts = [
                        f"{len(screening_result_ids)} candidate{'s' if len(screening_result_ids) != 1 else ''} passed "
                        f"screening and "
                        f"{'are' if len(screening_result_ids) != 1 else 'is'} awaiting deeper investigation"
                    ]
                    if inv_failed > 0:
                        rationale_parts.append(
                            f"({inv_failed} prior investigation"
                            f"{'s' if inv_failed != 1 else ''} "
                            f"failed — fresh candidates may outperform)"
                        )
                    if avg_recent_s1 is not None:
                        rationale_parts.append(
                            f"with recent {avg_recent_s1 * 100:.0f}% hit rate"
                        )
                    action_rationale = ", ".join(rationale_parts) + "."
            elif total_exp == 0:
                action = "start_first"
                action_label = "Run First Experiment"
                action_rationale = (
                    "No experiments yet. Start a mixed continuous run to begin "
                    "exploring the architecture space."
                )
            elif trend == "declining" or (
                len(recent_s1_rates) >= 3
                and all(r == 0 for r in recent_s1_rates[:3])
            ):
                action = "novelty_search"
                action_label = "Try Evolution / Novelty Search"
                action_rationale = (
                    "Recent experiments are underperforming. An evolution or "
                    "novelty-driven search can escape the current local minimum."
                )
            else:
                action = "continuous"
                action_label = "Continue Research"
                action_rationale = (
                    "The pipeline is active and the system is "
                    + ("learning" if trend == "improving" else "exploring")
                    + ". Continue generating and evaluating new architectures."
                )

            # Build deterministic suggested_config from action
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
                det_config = {
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
                det_config = {
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
                det_config = {
                    "mode": "investigation",
                    "model_source": "mixed",
                    "result_ids": screening_result_ids,
                }
            else:
                det_config = (
                    {"mode": det_mode, "model_source": "mixed"}
                    if det_mode
                    else None
                )

            det_config = augment_sparse_action_config(
                det_config,
                det_config.get("mode") if isinstance(det_config, dict) else det_mode,
                sparse_coverage_data,
            ) if isinstance(det_config, dict) else det_config

            return jsonify({
                "briefing": briefing,
                "action": action,
                "action_label": action_label,
                "action_rationale": action_rationale,
                "ai_powered": False,
                "fallback_reason": fallback_reason,
                "suggested_config": det_config,
                "evidence": recommendation_evidence,
                "data": data_block,
                "compression_opportunities": compression_opportunities,
                "ref_comparison": ref_comparison,
            })
        except Exception as e:
            logger.error(f"Error in /api/strategy/briefing: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/aria/cycle-status")
    def api_aria_cycle_status():
        """Get Aria continuous-cycle status (planning/running/analyzing)."""
        runner = get_runner(notebook_path)
        try:
            return jsonify(runner.get_aria_cycle_status())
        except Exception as e:
            logger.error(f"Error in /api/aria/cycle-status: {e}")
            return jsonify({"error": str(e)}), 500


    @app.route("/api/aria/cycle-history")
    def api_aria_cycle_history():
        """Get persisted Aria cycle summaries from notebook live-feed entries."""
        n = request.args.get("n", 100, type=int)
        mode_filter = str(request.args.get("mode") or "").strip().lower()
        status_filter = str(request.args.get("status") or "").strip().lower()
        query_text = str(request.args.get("q") or "").strip().lower()
        output_format = str(request.args.get("format") or "json").strip().lower()
        nb = LabNotebook(notebook_path)
        try:
            entries = normalize_entries(nb.get_entries(entry_type="live_feed", limit=n * 4))
            history: List[Dict[str, Any]] = []
            for entry in reversed(entries):
                metadata = entry.get("metadata") or {}
                if not isinstance(metadata, dict):
                    continue
                if metadata.get("live_feed_type") != "aria_cycle":
                    continue
                payload = metadata.get("payload") or {}
                if not isinstance(payload, dict):
                    continue
                row = dict(payload)
                row["entry_id"] = entry.get("entry_id")
                row["experiment_id"] = entry.get("experiment_id")
                row["entry_timestamp"] = entry.get("timestamp")

                row_mode = str(row.get("mode") or "").strip().lower()
                row_status = str(row.get("status") or "").strip().lower()
                if mode_filter and row_mode != mode_filter:
                    continue
                if status_filter and row_status != status_filter:
                    continue
                if query_text:
                    searchable = " ".join([
                        str(row.get("mode") or ""),
                        str(row.get("status") or ""),
                        str(row.get("reasoning") or ""),
                        str(row.get("error") or ""),
                    ]).lower()
                    if query_text not in searchable:
                        continue

                history.append(row)
                if len(history) >= n:
                    break

            if output_format == "csv":
                fieldnames = [
                    "cycle_index",
                    "mode",
                    "status",
                    "timestamp",
                    "delta_programs",
                    "delta_stage1_survivors",
                    "stage1_survivors",
                    "confidence",
                    "experiment_id",
                    "reasoning",
                    "error",
                ]
                buffer = io.StringIO()
                writer = csv.DictWriter(buffer, fieldnames=fieldnames)
                writer.writeheader()
                for row in history:
                    writer.writerow({k: row.get(k) for k in fieldnames})
                csv_payload = buffer.getvalue()
                return Response(
                    csv_payload,
                    mimetype="text/csv",
                    headers={
                        "Content-Disposition": "attachment; filename=aria_cycle_history.csv",
                    },
                )

            return jsonify(history)
        except Exception as e:
            logger.error(f"Error in /api/aria/cycle-history: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/aria/cycle-control", methods=["POST"])
    def api_aria_cycle_control():
        """Control Aria cycle policy: start, pause, resume."""
        runner = get_runner(notebook_path)
        body = request.get_json(silent=True) or {}
        action = str(body.get("action") or "").strip().lower()

        if action == "pause":
            status = runner.pause_aria_cycle()
            return jsonify({"ok": True, "action": "pause", "cycle": status})

        if action == "resume":
            status = runner.resume_aria_cycle()
            return jsonify({"ok": True, "action": "resume", "cycle": status})

        if action == "start":
            if runner.is_running:
                return jsonify({"error": "An experiment is already running"}), 409

            auto_harden = bool(body.get("auto_harden", True))
            config_payload = body.get("config") if isinstance(body.get("config"), dict) else body
            config_payload = dict(config_payload or {})
            config_payload.pop("action", None)
            config_payload.pop("auto_harden", None)
            config_payload["continuous"] = True

            try:
                config = RunConfig.from_dict(config_payload)
                config, prescreen = runner.prescreen_run_config(
                    config,
                    mode="continuous",
                    auto_harden=auto_harden,
                )
                exp_id = runner.start_continuous(config)
                record_run_trigger(
                    experiment_id=exp_id,
                    source="cycle_control",
                    mode="continuous",
                    details={
                        "endpoint": "/api/aria/cycle-control",
                        "action": "start",
                        "auto_harden": auto_harden,
                    },
                )
                return jsonify({
                    "ok": True,
                    "action": "start",
                    "experiment_id": exp_id,
                    "config": config.to_dict(),
                    "prescreen": prescreen,
                    "cycle": runner.get_aria_cycle_status(),
                })
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                logger.error(f"Error starting cycle control: {e}")
                return jsonify({"error": str(e)}), 500

        return jsonify({"error": "action must be one of: start, pause, resume"}), 400


    @app.route("/api/aria/recommendation")
    def api_aria_recommendation():
        """Get Aria's experiment recommendation based on all data."""
        runner = get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            analytics_data = runner._gather_analytics_data(nb)
            history = nb.get_recent_experiments(10)
            past_hypotheses = runner._get_past_hypotheses(nb)
            from ..llm.context import build_rich_context
            context = build_rich_context(
                results={"total": 0, "stage0_passed": 0, "stage05_passed": 0,
                         "stage1_passed": 0, "novel_count": 0},
                analytics_data=analytics_data,
                history=history,
                past_hypotheses=past_hypotheses,
            )
            suggestion = aria.suggest_experiment(
                context, op_success_rates=analytics_data.get("op_success_rates"),
                compression_coverage=analytics_data.get("compression_coverage"))
            if suggestion:
                suggestion["evidence_pack"] = build_evidence_pack(
                    nb,
                    analytics=None,
                    recommendation=suggestion,
                    decision_type="api_recommendation",
                    recent_experiments=history,
                )
            return jsonify(suggestion)
        except Exception as e:
            logger.error(f"Error in /api/aria/recommendation: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/aria/strategy")
    def api_aria_strategy():
        """Get Aria's research strategy recommendation."""
        runner = get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            analytics_data = runner._gather_analytics_data(nb)
            history = nb.get_recent_experiments(10)
            past_hypotheses = runner._get_past_hypotheses(nb)
            from ..llm.context import build_rich_context
            context = build_rich_context(
                results={"total": 0, "stage0_passed": 0, "stage05_passed": 0,
                         "stage1_passed": 0, "novel_count": 0},
                analytics_data=analytics_data,
                history=history,
                past_hypotheses=past_hypotheses,
            )
            strategy = aria.plan_strategy(context)
            return jsonify({
                "strategy": strategy,
                "available": strategy is not None,
            })
        except Exception as e:
            logger.error(f"Error in /api/aria/strategy: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/aria/tools")
    def api_aria_tools():
        """Report Aria tool capabilities and current operational readiness."""
        runner = get_runner(notebook_path)
        aria = get_aria()
        llm = aria._get_llm()
        llm_available = False
        llm_reason = "not_configured"
        if llm:
            try:
                llm_available = bool(getattr(llm, "is_available", lambda: True)())
                llm_reason = "ok" if llm_available else "unreachable"
            except Exception:
                llm_available = False
                llm_reason = "unreachable"

        cycle_status = runner.get_aria_cycle_status()
        ollama_helper = local_ollama_helper_status(llm)
        return jsonify({
            "codebase_agent": {
                "spawn_endpoint": True,
                "status_endpoint": True,
                "workspace_scoped": True,
                "allow_write_default": True,
                "execution_first_for_fix_requests": True,
                "small_model_swarm_enabled": True,
                "small_model_swarm_max_workers": get_local_ollama_settings().get("max_small_workers", 3),
                "simple_task_policy": "prefer_3b_swarm_then_7b",
                "complex_task_policy": "prefer_7b_single",
            },
            "local_ollama_helper": ollama_helper,
            "chat_actions": ["adjust_config", "adjust_grammar", "start_experiment", "edit_file", "spawn_agent"],
            "chat_guardrails": chat_guardrail_snapshot(window=200),
            "local_context_tools": ["runner.progress", "notebook.get_recent_experiments", "workspace.search"],
            "llm": {
                "available": llm_available,
                "reason": llm_reason,
            },
            "runner": {
                "is_running": bool(runner.is_running),
                "progress_status": (runner.progress.to_dict() or {}).get("status"),
            },
            "run_trigger": get_run_trigger_snapshot((runner.progress.to_dict() or {}).get("experiment_id")),
            "continuous": {
                "active": bool(cycle_status.get("continuous_active")),
                "phase": cycle_status.get("phase"),
            },
        })


    @app.route("/api/aria/chat/guardrails")
    def api_aria_chat_guardrails():
        """Expose chat action/summarization guardrail metrics."""
        try:
            window = int(request.args.get("window", 200))
        except Exception:
            window = 200
        return jsonify(chat_guardrail_snapshot(window=window))


    @app.route("/api/aria/agent/spawn", methods=["POST"])
    def api_aria_agent_spawn():
        """Spawn a background Aria codebase agent task for autonomous repair/refactor."""
        body = request.get_json(silent=True) or {}
        goal = str(body.get("goal") or "").strip()
        allow_write = bool(body.get("allow_write", True))

        if not goal:
            return jsonify({"error": "goal is required"}), 400

        spawn_session_id = str(body.get("session_id") or "").strip()
        task = _spawn_code_agent_task(
            goal=goal,
            notebook_path=notebook_path,
            allow_write=allow_write,
            session_id=spawn_session_id,
        )
        return jsonify({"ok": True, "task": task}), 202


    @app.route("/api/aria/agent/status/<task_id>")
    def api_aria_agent_status(task_id: str):
        """Get status/result for a background Aria codebase agent task."""
        task = code_agent_task_snapshot(task_id)
        if not task:
            return jsonify({"error": "task not found"}), 404
        detail = str(request.args.get("detail") or "").strip().lower()
        if detail != "full":
            task = {
                **task,
                **summarize_agent_task(task),
            }
        return jsonify({"ok": True, "task": task})


    @app.route("/api/aria/agent/status/<task_id>/summary")
    def api_aria_agent_status_summary(task_id: str):
        """Get concise milestone summary for a background Aria codebase agent task."""
        task = code_agent_task_snapshot(task_id)
        if not task:
            return jsonify({"error": "task not found"}), 404
        return jsonify({"ok": True, "task": summarize_agent_task(task)})


    @app.route("/api/aria/diagnose", methods=["POST"])
    def api_aria_diagnose():
        """Run Aria's self-diagnosis: gather analytics, identify issues, apply fixes."""
        runner = get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        try:
            analytics_data = {}
            try:
                analytics_data = runner._gather_analytics_data(nb)
            except Exception as exc:
                logger.debug(f"Analytics gather failed during diagnosis: {exc}")

            diagnosed_issues = diagnose_research_issues(analytics_data, nb)
            actions_applied: List[Dict[str, Any]] = []

            for issue in diagnosed_issues:
                cfg_fix = issue.get("config_fix")
                if cfg_fix and issue.get("action_type") in ("config_fix", "grammar_fix"):
                    try:
                        result = runner.execute_chat_action(cfg_fix, nb)
                        if result.get("status") == "applied":
                            applied_keys = list((result.get("changes") or result.get("weights") or {}).keys())
                            actions_applied.append({
                                "issue": issue["issue"],
                                "action_type": issue["action_type"],
                                "keys_applied": applied_keys,
                            })
                    except Exception as exc:
                        logger.debug(f"Diagnosis config fix failed: {exc}")

            return jsonify({
                "ok": True,
                "issues_found": len(diagnosed_issues),
                "issues": [
                    {
                        "issue": i["issue"],
                        "action_type": i.get("action_type", "info"),
                        "fixed": i["issue"] in [a["issue"] for a in actions_applied],
                    }
                    for i in diagnosed_issues
                ],
                "actions_applied": actions_applied,
                "summary": (
                    f"Found {len(diagnosed_issues)} issue(s), applied {len(actions_applied)} fix(es)."
                    if diagnosed_issues
                    else "No issues found in current analytics."
                ),
            })
        except Exception as e:
            logger.error(f"Diagnosis failed: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/aria/chat", methods=["POST"])
    def api_aria_chat():
        """Interactive Aria chat response grounded in current research context."""
        runner = get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()

        try:
            body = request.get_json(silent=True) or {}
            question = str(body.get("message") or "").strip()
            history_raw = body.get("history") or []
            session_id = str(body.get("session_id") or "").strip()
            spawn_agent = bool(body.get("spawn_agent", False))
            allow_code_writes = bool(body.get("allow_code_writes", True))
            explicit_detailed = chat_requests_detailed_response(question)
            summary_requested = chat_requests_summary_response(question)
            brief_response_requested = (
                bool(body.get("brief_response", False))
                or chat_requests_brief_response(question)
            )
            concise_default_mode = not explicit_detailed and not summary_requested
            brief_response = bool(brief_response_requested or concise_default_mode)
            self_fix_now = chat_requests_self_fix_now(question)
            fix_request = spawn_agent or chat_requests_codebase_fix(question) or self_fix_now
            execution_first_mode = bool(fix_request)
            fallback_reason: Optional[str] = None
            local_agent_result: Dict[str, Any] = {"tools_used": [], "summary": "", "code_hits": []}
            code_agent_task: Optional[Dict[str, Any]] = None

            if not question:
                return jsonify({"error": "message is required"}), 400

            if execution_first_mode:
                # Diagnose → Act → Report instead of blindly spawning agents
                analytics_data = {}
                try:
                    analytics_data = runner._gather_analytics_data(nb)
                except Exception as exc:
                    logger.debug(f"Analytics gather failed during diagnosis: {exc}")

                diagnosed_issues = diagnose_research_issues(analytics_data, nb)
                actions_taken: List[str] = []
                config_keys_applied: List[str] = []

                # Apply config/grammar fixes directly
                for issue in diagnosed_issues:
                    cfg_fix = issue.get("config_fix")
                    if cfg_fix and issue.get("action_type") in ("config_fix", "grammar_fix"):
                        try:
                            result = runner.execute_chat_action(cfg_fix, nb)
                            if result.get("status") == "applied":
                                applied = result.get("changes") or result.get("weights") or {}
                                config_keys_applied.extend(applied.keys())
                                actions_taken.append(issue["issue"])
                        except Exception as exc:
                            logger.debug(f"Config fix failed: {exc}")

                # Decide whether to spawn an agent
                is_vague = self_fix_now  # "fix yourself", "fix what's wrong", etc.
                if not is_vague and fix_request:
                    # Specific fix request — spawn agent with enriched goal
                    diag_context = "; ".join(i["issue"] for i in diagnosed_issues) if diagnosed_issues else "No issues diagnosed"
                    enriched_goal = f"{question}\n\nDiagnosis context: {diag_context}"
                    try:
                        code_agent_task = _spawn_code_agent_task(
                            goal=enriched_goal,
                            notebook_path=notebook_path,
                            allow_write=allow_code_writes,
                            session_id=session_id,
                        )
                    except Exception as exc:
                        logger.warning(f"Unable to spawn codebase agent from chat: {exc}")

                # Build reply
                if diagnosed_issues:
                    reply_parts = []
                    for issue in diagnosed_issues:
                        if issue.get("action_type") == "info":
                            reply_parts.append(issue["issue"] + ".")
                        elif issue["issue"] in actions_taken:
                            reply_parts.append(f"Diagnosed: {issue['issue']}. Applied config fix ({', '.join(issue.get('config_fix', {}).get('changes', issue.get('config_fix', {}).get('weights', {})).keys())}).")
                        else:
                            reply_parts.append(f"Diagnosed: {issue['issue']}.")
                    if code_agent_task:
                        task_id = code_agent_task.get("task_id")
                        reply_parts.append(f"Agent `{task_id}` working on the code-level fix.")
                    concise_reply = " ".join(reply_parts)
                elif code_agent_task:
                    task_id = code_agent_task.get("task_id")
                    concise_reply = f"No config issues found. Spawned agent `{task_id}` to investigate."
                else:
                    concise_reply = "Ran diagnostics — no actionable issues found in current analytics."

                if session_id:
                    try:
                        nb.save_chat_message(
                            session_id=session_id,
                            role="aria",
                            text=concise_reply,
                            label="Aria",
                        )
                    except Exception:
                        pass
                record_chat_guardrail_event(
                    actionable=bool(actions_taken or code_agent_task),
                    advice_only=not bool(actions_taken or code_agent_task),
                    summary_text=concise_reply,
                )
                return jsonify({
                    "reply": concise_reply,
                    "ai_powered": False,
                    "used_context": True,
                    "fallback_reason": None,
                    "brief_mode": True,
                    "execution_first_mode": True,
                    "advice_only": not bool(actions_taken or code_agent_task),
                    "agent_task": code_agent_task,
                    "actions_taken": actions_taken,
                    "local_tools_used": [],
                    "local_code_hits": [],
                })

            # Persist user message to DB if session_id provided
            if session_id:
                try:
                    nb.save_chat_message(
                        session_id=session_id, role="user", text=question,
                        label="You",
                    )
                except Exception:
                    pass  # Non-fatal — don't block chat on persistence failure

            # Build history lines: prefer DB history when session_id given
            history_lines: List[str] = []
            if session_id:
                try:
                    db_messages = nb.get_chat_history(session_id, limit=12)
                    for msg in db_messages:
                        role = str(msg.get("role") or "user").strip().lower()
                        text = str(msg.get("text") or "").strip()
                        if not text:
                            continue
                        label = "ARIA" if role in {"aria", "assistant"} else role.upper()
                        history_lines.append(f"{label}: {text}")
                except Exception:
                    pass  # Fall through to request-body history
            if not history_lines and isinstance(history_raw, list):
                for entry in history_raw[-8:]:
                    if not isinstance(entry, dict):
                        continue
                    role = str(entry.get("role") or "user").strip().lower()
                    if role not in {"user", "aria", "assistant", "system"}:
                        role = "user"
                    text = str(entry.get("text") or "").strip()
                    if not text:
                        continue
                    label = "ARIA" if role in {"aria", "assistant"} else role.upper()
                    history_lines.append(f"{label}: {text}")

            try:
                analytics_data = runner._gather_analytics_data(nb)
            except Exception:
                analytics_data = {}

            try:
                history = nb.get_recent_experiments(10)
            except Exception:
                history = []

            try:
                past_hypotheses = runner._get_past_hypotheses(nb)
            except Exception:
                past_hypotheses = []

            try:
                from ..llm.context import build_rich_context
                context = build_rich_context(
                    results={"total": 0, "stage0_passed": 0, "stage05_passed": 0,
                             "stage1_passed": 0, "novel_count": 0},
                    analytics_data=analytics_data,
                    history=history,
                    past_hypotheses=past_hypotheses,
                )
            except Exception:
                context = (
                    "Context fallback:\n"
                    f"- Recent experiments: {len(history)}\n"
                    f"- Analytics keys: {len(analytics_data) if isinstance(analytics_data, dict) else 0}\n"
                    f"- Past hypotheses: {len(past_hypotheses) if isinstance(past_hypotheses, list) else 0}"
                )

            local_agent_result = run_local_chat_agent(
                question=question,
                runner=runner,
                nb=nb,
                notebook_path=notebook_path,
                enable_code_tools=True,
            )
            if local_agent_result.get("summary"):
                context = f"{context}\n\n{local_agent_result['summary']}"
            # Cap context to ~2000 chars to prevent LLM from echoing data back
            if len(context) > 2000:
                context = context[:2000] + "\n[context truncated]"
            if code_agent_task:
                task_id = code_agent_task.get("task_id")
                context = (
                    f"{context}\n\n"
                    "Autonomous codebase agent was spawned for this request:\n"
                    f"- task_id={task_id}\n"
                    f"- allow_write={bool(code_agent_task.get('allow_write'))}\n"
                    "- can inspect and patch any workspace file with safety checks"
                )

            llm = aria._get_llm()
            if llm:
                try:
                    if hasattr(llm, "is_available") and not llm.is_available():
                        fallback_reason = "llm_unreachable"
                except Exception:
                    fallback_reason = "llm_unreachable"
                try:
                    from ..llm.prompts import SYSTEM_PROMPT, CHAT_PROMPT
                    prompt_question = question
                    prompt_question = (
                        f"{prompt_question}\n\n"
                        "STRICT CONTRACT:\n"
                        "1) Return only typed actions using ```action JSON blocks.\n"
                        "2) Allowed type values: adjust_config, adjust_grammar, start_experiment, edit_file, spawn_agent.\n"
                        "3) Do not output execution plans, pseudo-code, or non-action code blocks.\n"
                        "4) If no action is appropriate, return one short plain sentence only."
                    )
                    # Keep only last 5 history lines, each capped at 100 chars
                    trimmed_history = [
                        (line[:100] + "..." if len(line) > 100 else line)
                        for line in history_lines[-5:]
                    ]
                    prompt = CHAT_PROMPT.format(
                        context=context,
                        history="\n".join(trimmed_history) if trimmed_history else "(none)",
                        question=prompt_question,
                    )
                    max_tokens = 200 if brief_response else 384
                    resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=max_tokens)
                    aria._track_cost(resp)
                    text = (resp.text or "").strip()
                    if text:
                        parsed = parse_action_contract_response(text)
                        actions = parsed.get("actions") or []
                        advice_only = bool(parsed.get("advice_only"))
                        actions_taken = []
                        for action in actions:
                            try:
                                if str(action.get("type") or "") == "spawn_agent":
                                    goal = str(action.get("goal") or "").strip() or question
                                    if goal:
                                        # Route technical planning details to local planner context
                                        context_lines = [f"Original request: {question}"]
                                        local_summary = str(local_agent_result.get("summary") or "").strip()
                                        if local_summary:
                                            context_lines.append(f"Local evidence summary: {local_summary}")
                                        hits = local_agent_result.get("code_hits") or []
                                        if hits:
                                            top_hits = ", ".join(
                                                f"{str(h.get('path') or '?')}:{int(h.get('line') or 0)}"
                                                for h in hits[:5]
                                            )
                                            context_lines.append(f"Relevant code hits: {top_hits}")
                                        try:
                                            ws = chat_workspace_root(notebook_path)
                                            idx_hits = query_file_index(goal, ws, max_results=6)
                                            if idx_hits:
                                                files_hint = ", ".join(h["rel_path"] for h in idx_hits[:6])
                                                context_lines.append(f"Indexed files: {files_hint}")
                                        except Exception:
                                            pass
                                        history_tail = " | ".join(history_lines[-3:]) if history_lines else ""
                                        if history_tail:
                                            context_lines.append(f"Chat context: {history_tail}")
                                        goal = f"{goal}\n\nTechnical plan context:\n- " + "\n- ".join(context_lines)
                                        agent_task = _spawn_code_agent_task(
                                            goal=goal,
                                            notebook_path=notebook_path,
                                            allow_write=allow_code_writes,
                                            session_id=session_id,
                                        )
                                        result = {
                                            "status": "spawned",
                                            "task_id": agent_task.get("task_id"),
                                            "goal": truncate_summary(str(action.get("goal") or question), 120),
                                        }
                                        if not code_agent_task:
                                            code_agent_task = agent_task
                                    else:
                                        result = {"status": "error", "error": "No goal provided"}
                                else:
                                    result = runner.execute_chat_action(action, nb)
                                if (
                                    str(action.get("type") or "").strip() == "start_experiment"
                                    and str(result.get("status") or "").strip() == "started"
                                    and result.get("experiment_id")
                                ):
                                    record_run_trigger(
                                        experiment_id=str(result.get("experiment_id")),
                                        source="chat_action",
                                        mode=str(result.get("mode") or "single").strip() or "single",
                                        details={
                                            "endpoint": "/api/aria/chat",
                                            "session_id": session_id or None,
                                        },
                                    )
                                actions_taken.append({
                                    "type": action.get("type"),
                                    "status": result.get("status", "unknown"),
                                    "detail": result,
                                })
                            except Exception as action_err:
                                actions_taken.append({
                                    "type": action.get("type"),
                                    "status": "error",
                                    "detail": {"error": str(action_err)},
                                })
                        actionable = any(
                            str(a.get("status") or "").lower() in {"applied", "started", "spawned"}
                            for a in actions_taken
                        )
                        if actionable:
                            action_types = ", ".join(
                                sorted({str(a.get("type") or "?") for a in actions_taken})
                            )
                            status_bits = []
                            for item in actions_taken:
                                t = str(item.get("type") or "?")
                                s = str(item.get("status") or "unknown")
                                status_bits.append(f"{t}:{s}")
                            reply_text = truncate_summary(
                                f"Action started: {action_types}. "
                                f"Status: {'; '.join(status_bits[:4])}. "
                                f"Next checkpoint: monitor task progress and report completion.",
                                240,
                            )
                        else:
                            summary = str(parsed.get("summary") or "").strip()
                            reply_text = truncate_summary(
                                summary or "advice_only: no valid executable actions were produced.",
                                220,
                            )
                            advice_only = True
                        if code_agent_task and code_agent_task.get("task_id"):
                            snap = summarize_agent_task(code_agent_task)
                            reply_text = truncate_summary(
                                f"{reply_text} Task {snap.get('task_id')} queued ({snap.get('milestone_summary')}).",
                                260,
                            )
                        record_chat_guardrail_event(
                            actionable=actionable,
                            advice_only=advice_only,
                            summary_text=reply_text,
                        )
                        if session_id:
                            try:
                                nb.save_chat_message(
                                    session_id=session_id, role="aria",
                                    text=reply_text, label="Aria",
                                )
                            except Exception:
                                pass
                        return jsonify({
                            "reply": reply_text,
                            "ai_powered": True,
                            "used_context": True,
                            "fallback_reason": None,
                            "brief_mode": brief_response,
                            "agent_task": code_agent_task,
                            "actions_taken": actions_taken,
                            "advice_only": advice_only,
                            "local_tools_used": local_agent_result.get("tools_used", []),
                            "local_code_hits": [
                                {
                                    "path": hit.get("path"),
                                    "abs_path": hit.get("abs_path"),
                                    "line": hit.get("line"),
                                    "score": hit.get("score"),
                                }
                                for hit in local_agent_result.get("code_hits", [])
                            ],
                        })
                    fallback_reason = fallback_reason or "llm_empty_response"
                except Exception as e:
                    logger.warning(f"Aria chat LLM failed, using fallback: {e}")
                    err_msg = str(e)[:120]
                    fallback_reason = f"llm_error:{type(e).__name__}: {err_msg}"
            else:
                fallback_reason = "llm_not_configured"

            # Fallback: no LLM available. Keep it short.
            if code_agent_task:
                task_id = code_agent_task.get("task_id")
                fallback_reply = f"Agent `{task_id}` is working on it. No LLM available for chat right now."
            elif summary_requested:
                fallback_reply = "LLM unavailable. Check Strategy Advisor for current recommendations."
            else:
                fallback_reply = "LLM unavailable. Try a fix-intent request (e.g. 'fix X') to spawn an agent."
            if session_id:
                try:
                    nb.save_chat_message(
                        session_id=session_id, role="aria",
                        text=fallback_reply,
                        label=f"Aria (fallback: {fallback_reason})",
                    )
                except Exception:
                    pass
            record_chat_guardrail_event(
                actionable=False,
                advice_only=True,
                summary_text=fallback_reply,
            )
            return jsonify({
                "reply": fallback_reply,
                "ai_powered": False,
                "used_context": True,
                "fallback_reason": fallback_reason,
                "brief_mode": brief_response,
                "advice_only": True,
                "agent_task": code_agent_task,
                "local_tools_used": local_agent_result.get("tools_used", []),
                "local_code_hits": [
                    {
                        "path": hit.get("path"),
                        "abs_path": hit.get("abs_path"),
                        "line": hit.get("line"),
                        "score": hit.get("score"),
                    }
                    for hit in local_agent_result.get("code_hits", [])
                ],
            })
        except Exception as e:
            logger.error(f"Error in /api/aria/chat: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/aria/chat/history")
    def api_aria_chat_history():
        """Load chat history from the database."""
        nb = LabNotebook(notebook_path)
        try:
            session_id = request.args.get("session_id", "default")
            limit = min(int(request.args.get("limit", 50)), 200)
            messages = nb.get_chat_history(session_id, limit=limit)
            return jsonify({"messages": messages, "session_id": session_id})
        except Exception as e:
            logger.error(f"Error in /api/aria/chat/history: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/aria/chat/message", methods=["POST"])
    def api_aria_chat_message():
        """Save a single chat message to the database."""
        nb = LabNotebook(notebook_path)
        try:
            body = request.get_json(silent=True) or {}
            session_id = body.get("session_id", "default")
            role = body.get("role", "user")
            text = body.get("text", "")
            label = body.get("label")
            message_id = body.get("message_id")
            metadata = body.get("metadata")
            if not text:
                return jsonify({"error": "text is required"}), 400
            mid = nb.save_chat_message(
                session_id=session_id, role=role, text=text,
                label=label, message_id=message_id, metadata=metadata,
            )
            return jsonify({"message_id": mid, "saved": True})
        except Exception as e:
            logger.error(f"Error in /api/aria/chat/message: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/aria/chat/compact", methods=["POST"])
    def api_aria_chat_compact():
        """Compact older chat messages into a summary when token budget exceeded."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            body = request.get_json(silent=True) or {}
            session_id = body.get("session_id", "default")
            token_budget = int(body.get("token_budget", 4000))

            messages = nb.get_chat_history(session_id, limit=200)
            if not messages:
                return jsonify({"compacted": False, "reason": "no messages"})

            # Calculate tokens for active messages
            total_tokens = sum(estimate_tokens(m.get("text", "")) for m in messages)
            if total_tokens <= token_budget:
                return jsonify({"compacted": False, "reason": "within budget",
                                "total_tokens": total_tokens})

            # Find oldest messages that exceed the budget
            # Keep recent messages within budget, compact the rest
            keep_tokens = 0
            keep_from = len(messages)
            for i in range(len(messages) - 1, -1, -1):
                msg_tokens = estimate_tokens(messages[i].get("text", ""))
                if keep_tokens + msg_tokens > token_budget * 0.7:  # Keep 70% budget for recent
                    keep_from = i + 1
                    break
                keep_tokens += msg_tokens

            to_compact = messages[:keep_from]
            if not to_compact:
                return jsonify({"compacted": False, "reason": "nothing to compact"})

            # Build text for summarization
            compact_text = "\n".join(
                f"{m.get('role', 'unknown').upper()}: {m.get('text', '')}"
                for m in to_compact
            )

            # Try LLM summarization, fall back to first-sentence extraction
            summary_text = None
            llm = aria._get_llm()
            if llm:
                try:
                    from ..llm.prompts import SYSTEM_PROMPT, CHAT_COMPACTION_PROMPT
                    prompt = CHAT_COMPACTION_PROMPT.format(messages=compact_text[:3000])
                    resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=300)
                    aria._track_cost(resp)
                    summary_text = (resp.text or "").strip()
                except Exception as e:
                    logger.warning(f"Chat compaction LLM failed: {e}")

            if not summary_text:
                # Fallback: extract first sentence from each message
                lines = []
                for m in to_compact:
                    text = (m.get("text") or "").strip()
                    first_sentence = text.split(".")[0].strip()
                    if first_sentence and len(first_sentence) > 10:
                        role = m.get("role", "?").upper()
                        lines.append(f"- [{role}] {first_sentence}.")
                    if len(lines) >= 5:
                        break
                summary_text = "\n".join(lines) if lines else "Previous conversation summarized."

            # Save summary message
            import uuid as _uuid
            summary_id = f"summary-{_uuid.uuid4().hex[:8]}"
            compact_ids = [m["message_id"] for m in to_compact if m.get("message_id")]

            nb.save_chat_message(
                session_id=session_id, role="system",
                text=summary_text, label="Summary",
                message_id=summary_id,
                metadata={"compaction": True, "summarized_count": len(compact_ids)},
            )
            nb.mark_messages_compacted(compact_ids, summary_id)

            return jsonify({
                "compacted": True,
                "messages_compacted": len(compact_ids),
                "summary_id": summary_id,
                "summary_tokens": estimate_tokens(summary_text),
                "original_tokens": total_tokens,
            })
        except Exception as e:
            logger.error(f"Error in /api/aria/chat/compact: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/system/status")
    def api_system_status():
        """Report system status: CUDA, LLM, database, runner state."""
        import torch
        runner = get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        refresh_canary = parse_bool_query(request.args.get("refresh_canary"), default=False)
        try:
            # CUDA info
            cuda_available = torch.cuda.is_available()
            cuda_info = {}
            if cuda_available:
                try:
                    cuda_info = {
                        "device_name": torch.cuda.get_device_name(0),
                        "device_count": torch.cuda.device_count(),
                    }
                    mem = torch.cuda.mem_get_info(0)
                    cuda_info["memory_free_gb"] = round(mem[0] / 1e9, 1)
                    cuda_info["memory_total_gb"] = round(mem[1] / 1e9, 1)
                except Exception as e:
                    logger.warning("Failed collecting CUDA details: %s", e)

            # LLM backend
            llm = aria._get_llm()
            llm_reachable = False
            if llm is not None:
                try:
                    llm_reachable = bool(llm.is_available()) if hasattr(llm, "is_available") else True
                except Exception:
                    llm_reachable = False
            llm_info = {
                "available": llm_reachable,
                "configured": llm is not None,
                "backend": llm.name if llm else None,
            }

            # Database stats
            summary = nb.get_dashboard_summary()
            db_info = {
                "path": notebook_path,
                "total_experiments": summary.get("total_experiments", 0),
                "total_programs": summary.get("total_programs_evaluated", 0),
            }

            return jsonify({
                "cuda": {"available": cuda_available, **cuda_info},
                "llm": llm_info,
                "database": db_info,
                "native_runner": native_runner_capability_report(),
                "native_runner_canary": native_runner_canary_status_payload(force_refresh=refresh_canary),
                "is_running": runner.is_running,
            })
        except Exception as e:
            logger.error(f"Error in /api/system/status: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/native-runner/capability")
    def api_native_runner_capability():
        """Report native-runner adapter capability and current mode flags."""
        try:
            return jsonify(native_runner_capability_report())
        except Exception as e:
            logger.error(f"Error in /api/native-runner/capability: {e}")
            return jsonify({"error": str(e)}), 500


    @app.route("/api/native-runner/canary/refresh", methods=["POST"])
    def api_native_runner_canary_refresh():
        """Force-refresh native runner canary payload (bypass TTL cache)."""
        try:
            payload = native_runner_canary_status_payload(force_refresh=True)
            return jsonify(
                {
                    "status": "ok",
                    "native_runner_canary": payload,
                    "refreshed_at": datetime.now(timezone.utc)
                    .isoformat(timespec="seconds")
                    .replace("+00:00", "Z"),
                }
            )
        except Exception as e:
            logger.error(f"Error in /api/native-runner/canary/refresh: {e}")
            return jsonify({"error": str(e)}), 500


    @app.route("/api/native-runner/telemetry")
    def api_native_runner_telemetry():
        """Return native runner fallback metrics for dashboard consumption."""
        try:
            report = native_runner_capability_report()
            return jsonify({
                "status": "ok",
                "metrics": report.get("fallback_metrics", {}),
                "capability": {
                    "enabled": report.get("enabled"),
                    "strict": report.get("strict"),
                    "designer_runtime_available": report.get("designer_runtime_available"),
                    "status": report.get("status"),
                },
                "op_support": report.get("native_op_support", {}),
            })
        except Exception as e:
            logger.error(f"Error in /api/native-runner/telemetry: {e}")
            return jsonify({"error": str(e)}), 500


    @app.route("/api/validate", methods=["POST"])
    def api_validate_pipeline():
        """Validate the synthesis pipeline by generating and testing programs."""
        body = request.get_json(silent=True) or {}
        n = min(int(body.get("n", body.get("sample_n", 5)) or 5), 20)
        mode = normalize_start_mode(body.pop("mode", "single"))
        auto_harden = bool(body.pop("auto_harden", True))
        runner = get_runner(notebook_path)
        config = RunConfig.from_dict(body) if body else RunConfig()
        config, prescreen = runner.prescreen_run_config(
            config,
            mode=mode,
            auto_harden=auto_harden,
        )

        try:
            sample = run_pipeline_sample_check(config=config, sample_n=n)
            preflight = run_launch_preflight(
                config=config,
                mode=mode,
                prescreen=prescreen,
                notebook_path=notebook_path,
                sample_n=n,
            )
            healthy = preflight.get("verdict") != "fail"
            return jsonify({
                "generated": sample.get("generated", 0),
                "compiled": sample.get("compiled", 0),
                "passed_s0": sample.get("passed_s0", 0),
                "errors": sample.get("errors", [])[:5],
                "healthy": healthy,
                "mode": mode,
                "config": config.to_dict(),
                "prescreen": prescreen,
                "preflight": preflight,
            })
        except Exception as e:
            logger.error(f"Error in pipeline validation: {e}")
            return jsonify({
                "generated": 0,
                "compiled": 0,
                "passed_s0": 0,
                "errors": [str(e)],
                "healthy": False,
                "mode": mode,
                "config": config.to_dict(),
                "prescreen": prescreen,
            })


    @app.route("/api/designer/lifecycle")
    def api_designer_lifecycle():
        """Return current aria_designer service status."""
        payload = designer_service_status()
        payload.update(designer_idle_state())
        return jsonify(payload)


    @app.route("/api/designer/ensure-running", methods=["POST"])
    def api_designer_ensure_running():
        """Ensure aria_designer API+UI are running for seamless UX."""
        body = request.get_json(silent=True) or {}
        force_restart = bool(body.get("force_restart", False))
        result = start_designer_services(force_restart=force_restart)
        if result.get("ok"):
            result.update(designer_touch_activity("ensure-running"))
        status = 200 if result.get("ok") else 503
        return jsonify(result), status


    @app.route("/api/designer/stop", methods=["POST"])
    def api_designer_stop():
        """Stop aria_designer API+UI services."""
        result = stop_designer_services()
        status = 200 if result.get("ok") else 500
        return jsonify(result), status


    @app.route("/api/designer/touch", methods=["POST"])
    def api_designer_touch():
        """Refresh designer activity for idle auto-stop policy."""
        body = request.get_json(silent=True) or {}
        reason = str(body.get("reason") or "manual-touch")
        payload = {"ok": True}
        payload.update(designer_touch_activity(reason))
        payload.update(designer_idle_state())
        return jsonify(payload), 200


    @app.route("/api/designer/compile", methods=["POST"])
    def api_designer_compile():
        """Accept graph JSON from designer and return compiled module info."""
        workflow_json = request.get_json(silent=True)
        if not workflow_json:
            return jsonify({"success": False, "error": "Missing workflow JSON"}), 400

        # Proxy: POST /api/v1/workflows/compile
        proxy_body = {"workflow": workflow_json, "target": "auto"}
        proxied = proxy_or_error(
            designer_proxy("POST", "/api/v1/workflows/compile", json_body=proxy_body)
        )
        if proxied is not None:
            return proxied
        
        return jsonify({"success": False, "error": "Designer service unavailable"}), 503


    @app.route("/api/designer/validate", methods=["POST"])
    def api_designer_validate():
        """Accept graph JSON from designer and return validation results."""
        workflow_json = request.get_json(silent=True)
        if not workflow_json:
            return jsonify({"success": False, "error": "Missing workflow JSON"}), 400

        # Proxy: POST /api/v1/workflows/validate
        proxy_body = {"workflow": workflow_json}
        proxied = proxy_or_error(
            designer_proxy("POST", "/api/v1/workflows/validate", json_body=proxy_body)
        )
        if proxied is not None:
            return proxied

        return jsonify({"success": False, "error": "Designer service unavailable"}), 503


    @app.route("/api/designer/run", methods=["POST"])
    def api_designer_run():
        """Accept graph JSON from designer, run forward pass, and return metrics."""
        workflow_json = request.get_json(silent=True)
        if not workflow_json:
            return jsonify({"success": False, "error": "Missing workflow JSON"}), 400

        device = request.args.get("device", "cpu")

        # Proxy: POST /api/v1/workflows/run
        proxy_body = {"workflow": workflow_json, "budget": {"device": device}}
        proxied = proxy_or_error(
            designer_proxy("POST", "/api/v1/workflows/run", json_body=proxy_body)
        )
        if proxied is not None:
            return proxied

        return jsonify({"success": False, "error": "Designer service unavailable"}), 503


    @app.route("/api/designer/components", methods=["GET"])
    def api_designer_components():
        """Return all available primitives formatted for the designer."""
        # Proxy: GET /api/v1/components
        proxied = proxy_or_error(
            designer_proxy("GET", "/api/v1/components")
        )
        if proxied is not None:
            return proxied

        return jsonify({"success": False, "error": "Designer service unavailable"}), 503


    @app.route("/api/designer/save", methods=["POST"])
    def api_designer_save():
        """Save a workflow definition to the notebook."""
        body = request.get_json(silent=True) or {}
        workflow_id = body.get("workflow_id")
        name = body.get("name", "Untitled Workflow")
        if not workflow_id:
            return jsonify({"success": False, "error": "Missing workflow_id"}), 400

        # Proxy: PUT /api/v1/workflows/{workflow_id}
        proxy_body = {
            "schema_version": "workflow_graph.v1",
            "workflow_id": workflow_id,
            "name": name,
            "nodes": body.get("nodes", []),
            "edges": body.get("edges", []),
            "metadata": body.get("metadata", {}),
        }
        proxied = proxy_or_error(
            designer_proxy("PUT", f"/api/v1/workflows/{workflow_id}", json_body=proxy_body)
        )
        if proxied is not None:
            return proxied

        return jsonify({"success": False, "error": "Designer service unavailable"}), 503


    @app.route("/api/designer/commit", methods=["POST"])
    def api_designer_commit():
        """Commit a designer architecture as a new program result in the research pipeline."""
        body = request.get_json(silent=True) or {}
        workflow = body.get("workflow")
        if not workflow:
            return jsonify({"success": False, "error": "Missing workflow data"}), 400

        # Proxy: POST /api/v1/workflows/evaluate
        # Note: evaluate is effectively a commit to the evaluation database in the designer
        # which our dashboard syncs from.
        proxied = proxy_or_error(
            designer_proxy("POST", "/api/v1/workflows/evaluate", json_body={"workflow": workflow})
        )
        if proxied is not None:
            return proxied

        return jsonify({"success": False, "error": "Designer service unavailable"}), 503


    @app.route("/api/v1/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
    def designer_v1_proxy(path):
        """Catch-all proxy for designer API v1 routes when embedded."""
        result = proxy_or_error(
            designer_proxy(request.method, f"/api/v1/{path}",
                            json_body=request.get_json(silent=True) if request.method in ("POST", "PUT") else None,
                            params=request.args)
        )
        if result is not None:
            return result
        return jsonify({"error": "Designer API proxy failed"}), 502


    @app.route("/api/designer/load/<workflow_id>")
    def api_designer_load(workflow_id):
        """Load a specific workflow definition."""
        # Proxy: GET /api/v1/workflows/{workflow_id}
        proxied = proxy_or_error(
            designer_proxy("GET", f"/api/v1/workflows/{workflow_id}")
        )
        if proxied is not None:
            return proxied

        return jsonify({"success": False, "error": "Designer service unavailable"}), 503


    @app.route("/api/designer/list")
    def api_designer_list_workflows():
        """List all saved workflows."""
        # Proxy: GET /api/v1/workflows
        proxied = proxy_or_error(
            designer_proxy("GET", "/api/v1/workflows")
        )
        if proxied is not None:
            return proxied

        return jsonify({"success": False, "error": "Designer service unavailable"}), 503


    @app.route("/api/designer/templates")
    def api_designer_templates():
        """Return hardcoded starter templates for the designer.

        No proxy equivalent — templates are served locally.
        """
        templates = [
            {
                "id": "tpl_linear",
                "name": "Simple Linear",
                "description": "Single linear projection.",
                "workflow": {
                    "nodes": [
                        {"id": "n0", "component_type": "io/input", "params": {}, "ui_meta": {"position": {"x": 100, "y": 100}}},
                        {"id": "n1", "component_type": "linear_algebra/linear_proj", "params": {}, "ui_meta": {"position": {"x": 100, "y": 200}}},
                        {"id": "n2", "component_type": "io/output", "params": {}, "ui_meta": {"position": {"x": 100, "y": 300}}}
                    ],
                    "edges": [
                        {"id": "e0", "source": "n0", "target": "n1"},
                        {"id": "e1", "source": "n1", "target": "n2"}
                    ]
                }
            },
            {
                "id": "tpl_mlp",
                "name": "Standard MLP",
                "description": "Two-layer MLP with ReLU.",
                "workflow": {
                    "nodes": [
                        {"id": "in", "component_type": "io/input", "params": {}, "ui_meta": {"position": {"x": 100, "y": 50}}},
                        {"id": "l1", "component_type": "linear_algebra/linear_proj", "params": {"out_dim": 512}, "ui_meta": {"position": {"x": 100, "y": 150}}},
                        {"id": "act", "component_type": "math/relu", "params": {}, "ui_meta": {"position": {"x": 100, "y": 250}}},
                        {"id": "l2", "component_type": "linear_algebra/linear_proj", "params": {"out_dim": 256}, "ui_meta": {"position": {"x": 100, "y": 350}}},
                        {"id": "out", "component_type": "io/output", "params": {}, "ui_meta": {"position": {"x": 100, "y": 450}}}
                    ],
                    "edges": [
                        {"id": "e1", "source": "in", "target": "l1"},
                        {"id": "e2", "source": "l1", "target": "act"},
                        {"id": "e3", "source": "act", "target": "l2"},
                        {"id": "e4", "source": "l2", "target": "out"}
                    ]
                }
            }
        ]
        return jsonify(templates)


    @app.route("/api/designer/export/python", methods=["POST"])
    def api_designer_export_python():
        """Generate standalone PyTorch module code for a workflow.

        No proxy equivalent — uses local generation.
        """
        workflow_json = request.get_json(silent=True)
        if not workflow_json:
            return jsonify({"success": False, "error": "Missing workflow JSON"}), 400

        from ..designer_utils import generate_python_module
        code = generate_python_module(workflow_json)
        return jsonify({"success": True, "code": code})


    @app.route("/api/designer/import/survivors")
    def api_designer_survivors():
        """List top survivors from the research pipeline for importing."""
        n = request.args.get("n", 20, type=int)

        # Proxy: GET /api/v1/import/survivors
        proxied = proxy_or_error(
            designer_proxy("GET", "/api/v1/import/survivors", params={"n": n})
        )
        if proxied is not None:
            return proxied

        return jsonify({"success": False, "error": "Designer service unavailable"}), 503


    @app.route("/api/designer/import", methods=["POST"])
    def api_designer_import():
        """Import a computation graph from the research pipeline by result_id."""
        body = request.get_json(silent=True) or {}
        result_id = body.get("result_id")
        if not result_id:
            return jsonify({"success": False, "error": "Missing result_id"}), 400

        # Proxy: POST /api/v1/import/survivors/{result_id}
        proxied = proxy_or_error(
            designer_proxy("POST", f"/api/v1/import/survivors/{result_id}")
        )
        if proxied is not None:
            return proxied

        return jsonify({"success": False, "error": "Designer service unavailable"}), 503


    @app.route("/api/designer/lineage/sync", methods=["POST"])
    def api_designer_lineage_sync():
        """Upsert Aria Designer run-lineage metadata into the research notebook."""
        body = request.get_json(silent=True) or {}
        run_id = str(body.get("run_id") or "").strip()
        workflow_id = str(body.get("workflow_id") or "").strip()
        if not run_id or not workflow_id:
            return jsonify({
                "success": False,
                "error": "run_id and workflow_id are required",
            }), 400

        workflow_version = body.get("workflow_version")
        try:
            workflow_version = int(workflow_version) if workflow_version is not None else None
        except Exception:
            workflow_version = None

        total_time_ms = body.get("total_time_ms")
        try:
            total_time_ms = float(total_time_ms) if total_time_ms is not None else None
        except Exception:
            total_time_ms = None

        created_at = body.get("created_at")
        try:
            created_at = float(created_at) if created_at is not None else None
        except Exception:
            created_at = None

        nb = LabNotebook(notebook_path)
        try:
            nb.save_designer_run_lineage(
                run_id=run_id,
                workflow_id=workflow_id,
                workflow_version=workflow_version,
                graph_fingerprint=body.get("graph_fingerprint"),
                status=str(body.get("status") or "unknown"),
                source=str(body.get("source") or "aria_designer"),
                total_time_ms=total_time_ms,
                metrics=body.get("metrics") if isinstance(body.get("metrics"), dict) else {},
                payload=body.get("payload") if isinstance(body.get("payload"), dict) else {},
                created_at=created_at,
            )
            row = nb.get_designer_run_lineage(run_id)
            return jsonify({
                "success": True,
                "run_id": run_id,
                "workflow_id": workflow_id,
                "stored": bool(row),
            })
        finally:
            nb.close()


    @app.route("/api/designer/lineage/<run_id>")
    def api_designer_lineage_get(run_id):
        """Get one designer run-lineage record."""
        nb = LabNotebook(notebook_path)
        try:
            row = nb.get_designer_run_lineage(run_id)
            if row is None:
                return jsonify({"error": "Lineage run not found"}), 404
            return jsonify(row)
        finally:
            nb.close()


    @app.route("/api/designer/lineage")
    def api_designer_lineage_list():
        """List designer run-lineage rows, optionally filtered by workflow_id."""
        workflow_id = request.args.get("workflow_id")
        limit = request.args.get("limit", 100, type=int)
        limit = max(1, min(int(limit or 100), 500))
        nb = LabNotebook(notebook_path)
        try:
            rows = nb.list_designer_run_lineage(workflow_id=workflow_id, limit=limit)
            return jsonify(rows)
        finally:
            nb.close()


    @app.route("/api/hypotheses/<hypothesis_id>/chain")
    def api_hypothesis_chain(hypothesis_id):
        """Hypothesis lineage chain."""
        nb = LabNotebook(notebook_path)
        try:
            chain = nb.get_hypothesis_chain(hypothesis_id)
            return jsonify(chain)
        except Exception as e:
            logger.error(f"Error in hypothesis chain: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()


    @app.route("/api/aria/autonomy")
    def api_aria_autonomy_get():
        """Get current autonomy trust level and per-decision-type settings."""
        try:
            autonomy, _ = get_autonomy(notebook_path)
            return jsonify(autonomy.get_config())
        except Exception as e:
            logger.error(f"Error in GET /api/aria/autonomy: {e}")
            return jsonify({"error": str(e)}), 500


    @app.route("/api/aria/autonomy", methods=["PUT"])
    def api_aria_autonomy_put():
        """Update autonomy trust level or per-decision-type overrides."""
        try:
            autonomy, _ = get_autonomy(notebook_path)
            body = request.get_json(force=True, silent=True) or {}
            config = autonomy.update_config(body)
            return jsonify(config)
        except Exception as e:
            logger.error(f"Error in PUT /api/aria/autonomy: {e}")
            return jsonify({"error": str(e)}), 500


    @app.route("/api/aria/activity")
    def api_aria_activity():
        """Get Aria's recent autonomous decisions and their outcomes."""
        try:
            autonomy, store = get_autonomy(notebook_path)
            limit = request.args.get("limit", 20, type=int)
            # Combine in-memory actions with persisted ones
            memory_actions = autonomy.get_recent_activity(limit)
            stored_actions = store.get_recent(limit)

            # Merge: prefer in-memory (fresher), fill with stored
            seen_ids = {a["action_id"] for a in memory_actions}
            merged = list(memory_actions)
            for sa in stored_actions:
                if sa["action_id"] not in seen_ids:
                    merged.append(sa)
                    seen_ids.add(sa["action_id"])

            merged.sort(key=lambda a: a.get("created_at", 0), reverse=True)
            return jsonify(merged[:limit])
        except Exception as e:
            logger.error(f"Error in /api/aria/activity: {e}")
            return jsonify([]), 500


    @app.route("/api/v1/components", methods=["GET"])
    def api_v1_components():
        """Return designer components — proxy to designer API or fallback to local DB."""
        proxied = proxy_or_error(
            designer_proxy("GET", "/api/v1/components", params=dict(request.args))
        )
        if proxied is not None:
            return proxied
        # Fallback: read directly from the designer component database
        try:
            import sys as _sys
            _designer_root = str(Path(__file__).resolve().parents[2] / "aria_designer")
            if _designer_root not in _sys.path:
                _sys.path.insert(0, _designer_root)
            from api.app import database as _designer_db
            comps = _designer_db.list_components(
                category=request.args.get("category"),
                status=request.args.get("status"),
            )
            if comps:
                return jsonify(comps)
        except Exception:
            logger.debug("Could not load components from designer DB, falling back to primitives")
        return jsonify(get_designer_components())


    @app.route("/api/v1/import/survivors", methods=["GET"])
    def api_v1_import_survivors():
        """List importable survivors — proxy to designer or local fallback."""
        n = request.args.get("n", 20, type=int)
        sort_by = request.args.get("sort_by", "loss_ratio")
        min_novelty = request.args.get("min_novelty", 0.0, type=float)

        proxied = proxy_or_error(
            designer_proxy("GET", "/api/v1/import/survivors",
                            params={"n": n, "sort_by": sort_by, "min_novelty": min_novelty})
        )
        if proxied is not None:
            return proxied

        # Local fallback: use importer directly
        try:
            import sys as _sys
            _designer_root = str(Path(__file__).resolve().parents[2] / "aria_designer")
            if _designer_root not in _sys.path:
                _sys.path.insert(0, _designer_root)
            from runtime.importer import import_survivors as _import_survivors
            return jsonify(_import_survivors(n=n, sort_by=sort_by, min_novelty=min_novelty))
        except ImportError:
            nb = LabNotebook(notebook_path)
            try:
                survivors = nb.get_top_programs(n, sort_by=sort_by)
                return jsonify(survivors)
            finally:
                nb.close()


    @app.route("/api/v1/import/survivors/<result_id>", methods=["POST"])
    def api_v1_import_single(result_id):
        """Import a single survivor — proxy to designer or local fallback."""
        proxied = proxy_or_error(
            designer_proxy("POST", f"/api/v1/import/survivors/{result_id}")
        )
        if proxied is not None:
            return proxied

        # Local fallback: use importer directly
        try:
            import sys as _sys
            _designer_root = str(Path(__file__).resolve().parents[2] / "aria_designer")
            if _designer_root not in _sys.path:
                _sys.path.insert(0, _designer_root)
            from runtime.importer import import_single as _import_single
            wf = _import_single(result_id)
            return jsonify(wf)
        except ImportError:
            return jsonify({"error": "Importer not available"}), 501
        except ValueError as e:
            return jsonify({"error": str(e)}), 404


    @app.route("/api/v1/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    def api_v1_catchall(subpath):
        """Proxy unhandled /api/v1/ requests to the aria_designer backend."""
        method = request.method
        json_body = request.get_json(silent=True)
        params = dict(request.args) if request.args else None

        # SSE streaming endpoints need special handling — don't buffer the response
        if "stream" in subpath:
            return proxy_stream(method, f"/api/v1/{subpath}", json_body=json_body, params=params)

        resp = designer_proxy(method, f"/api/v1/{subpath}", json_body=json_body, params=params)
        result = proxy_or_error(resp)
        if result is not None:
            return result
        return jsonify({"error": f"Designer backend unavailable for /api/v1/{subpath}"}), 502


    @app.route("/designer-proxy/")
    def designer_index():
        """Serve the built aria_designer index.html for the embedded iframe.

        Serving the designer from the same origin as the dashboard avoids
        cross-origin iframe restrictions in Brave and other browsers.
        """
        return send_from_directory(_designer_dist, "index.html")


    @app.route("/designer-proxy/<path:subpath>")
    def designer_assets(subpath):
        """Serve aria_designer static assets (JS, CSS, etc.)."""
        return send_from_directory(_designer_dist, subpath)


    @app.route("/")
    def index():
        if not _dashboard_index_path():
            return _dashboard_missing_response()
        return send_from_directory(app.static_folder, "index.html")


    @app.route("/favicon.ico")
    def favicon():
        if app.static_folder:
            icon = Path(app.static_folder) / "favicon.ico"
            if icon.is_file():
                return send_from_directory(app.static_folder, "favicon.ico")
        return "", 204


    @app.route("/<path:path>")
    def static_files(path):
        if app.static_folder:
            static_path = Path(app.static_folder) / path
            if static_path.is_file():
                return send_from_directory(app.static_folder, path)
        index_path = _dashboard_index_path()
        if index_path and not _is_asset_path(path):
            return send_from_directory(app.static_folder, "index.html")
        return "Not found", 404

