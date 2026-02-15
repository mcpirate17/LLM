"""
REST API Server for the AI Scientist Dashboard

Serves data from the lab notebook to the React dashboard.
Provides control endpoints for starting/stopping experiments.
Uses Flask for simplicity, SSE for real-time streaming.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Optional

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from .notebook import LabNotebook
from .persona import get_aria
from .runner import ExperimentRunner, RunConfig
from .llm.context import build_program_context

logger = logging.getLogger(__name__)

# Singleton runner shared across requests
_runner: Optional[ExperimentRunner] = None


def _get_sse_timeout_seconds() -> float:
    """Get SSE stream polling timeout from env with safe fallback."""
    raw = os.environ.get("ARIA_SSE_TIMEOUT_SECONDS", "30")
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid ARIA_SSE_TIMEOUT_SECONDS=%r; using 30s", raw)
        return 30.0
    if timeout <= 0:
        logger.warning("Non-positive ARIA_SSE_TIMEOUT_SECONDS=%r; using 30s", raw)
        return 30.0
    return timeout


def _get_runner(notebook_path: str) -> ExperimentRunner:
    global _runner
    if _runner is None:
        _runner = ExperimentRunner(notebook_path)
    return _runner


def create_app(
    notebook_path: str = "research/lab_notebook.db",
    static_folder: Optional[str] = None,
) -> Flask:
    """Create the Flask API app."""

    if static_folder is None:
        static_folder = str(Path(__file__).parent.parent / "dashboard" / "build")

    app = Flask(__name__, static_folder=static_folder, static_url_path="")
    CORS(app)

    # ── Global error handlers ──

    @app.errorhandler(404)
    def not_found(e):
        # Only return JSON for API routes; let static files 404 naturally
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not found"}), 404
        return send_from_directory(app.static_folder, "index.html")

    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"500 error on {request.method} {request.path}: {e}")
        return jsonify({"error": "Internal server error"}), 500

    @app.errorhandler(Exception)
    def unhandled_exception(e):
        logger.error(f"Unhandled exception on {request.method} {request.path}: "
                     f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500

    @app.after_request
    def log_response(response):
        if request.path.startswith("/api/") and response.status_code >= 400:
            logger.warning(f"{request.method} {request.path} -> {response.status_code}")
        return response

    # ── Dashboard routes ──

    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/<path:path>")
    def static_files(path):
        return send_from_directory(app.static_folder, path)

    # ── Read-only API routes ──

    @app.route("/api/status")
    def api_status():
        """Get Aria's current status and dashboard summary."""
        nb = LabNotebook(notebook_path)
        runner = _get_runner(notebook_path)
        aria = get_aria()
        try:
            return jsonify({
                "aria": aria.get_status(),
                "summary": nb.get_dashboard_summary(),
                "is_running": runner.is_running,
                "progress": runner.progress.to_dict(),
            })
        except Exception as e:
            logger.error(f"Error in /api/status: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments")
    def api_experiments():
        """List recent experiments."""
        n = request.args.get("n", 20, type=int)
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_recent_experiments(n))
        except Exception as e:
            logger.error(f"Error in /api/experiments: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>")
    def api_experiment_detail(experiment_id):
        """Get experiment details with entries and per-experiment programs."""
        nb = LabNotebook(notebook_path)
        try:
            exp = nb.get_experiment(experiment_id)
            if exp is None:
                return jsonify({"error": "Not found"}), 404
            entries = nb.get_entries(experiment_id=experiment_id)
            programs = nb.get_program_results(experiment_id)
            return jsonify({
                "experiment": exp,
                "entries": entries,
                "programs": programs,
            })
        except Exception as e:
            logger.error(f"Error in /api/experiments/{experiment_id}: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>/programs")
    def api_experiment_programs(experiment_id):
        """All programs for an experiment (not just S1 survivors)."""
        nb = LabNotebook(notebook_path)
        try:
            programs = nb.get_program_results(experiment_id)
            return jsonify(programs)
        except Exception as e:
            logger.error(f"Error in /api/experiments/{experiment_id}/programs: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/programs/<result_id>")
    def api_program_detail(result_id):
        """Full program detail with parsed graph JSON + fingerprint + all metrics."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            program = nb.get_program_detail(result_id)
            if program is None:
                return jsonify({"error": "Not found"}), 404

            # Include training curve availability flag
            try:
                curve = nb.get_training_curve(result_id)
                program["has_training_curve"] = len(curve) > 0
            except Exception:
                program["has_training_curve"] = False

            # Try LLM explanation of fingerprint (non-critical)
            try:
                ctx = build_program_context(program)
                explanation = aria.explain_fingerprint(ctx)
                if explanation:
                    program["llm_explanation"] = explanation
            except Exception as e:
                logger.debug(f"LLM fingerprint explanation failed for {result_id}: {e}")

            return jsonify(program)
        except Exception as e:
            logger.error(f"Error in /api/programs/{result_id}: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>/failures")
    def api_failure_analysis(experiment_id):
        """Failure analysis: error distribution, stage funnel."""
        nb = LabNotebook(notebook_path)
        try:
            analysis = nb.get_failure_analysis(experiment_id)
            return jsonify(analysis)
        except Exception as e:
            logger.error(f"Error in failure analysis: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>/analysis")
    def api_experiment_analysis(experiment_id):
        """LLM-generated analysis (stored or on-demand)."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            exp = nb.get_experiment(experiment_id)
            if exp is None:
                return jsonify({"error": "Not found"}), 404

            # Return stored analysis if available
            stored = exp.get("llm_analysis")
            if stored:
                return jsonify({"analysis": stored, "source": "stored"})

            # Try generating on-demand
            results = exp.get("results") or {}
            from .llm.context import build_experiment_context
            ctx = build_experiment_context(results)
            analysis = aria.analyze_results(results, context=ctx)

            if analysis:
                # Cache it
                try:
                    nb.conn.execute(
                        "UPDATE experiments SET llm_analysis = ? WHERE experiment_id = ?",
                        (analysis, experiment_id),
                    )
                    nb.conn.commit()
                except Exception as e:
                    logger.warning("Failed caching llm_analysis for %s: %s",
                                   experiment_id, e)
                return jsonify({"analysis": analysis, "source": "generated"})

            return jsonify({"analysis": None, "source": "unavailable",
                            "reason": "No LLM backend configured"})
        except Exception as e:
            logger.error(f"Error in experiment analysis: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/trends")
    def api_trends():
        """Cross-experiment trend data for charts."""
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_experiment_trends())
        except Exception as e:
            logger.error(f"Error in /api/trends: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/programs")
    def api_programs():
        """List top programs."""
        n = request.args.get("n", 20, type=int)
        sort_by = request.args.get("sort", "novelty_score")
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_top_programs(n, sort_by))
        except Exception as e:
            logger.error(f"Error in /api/programs: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/insights")
    def api_insights():
        """List active insights."""
        category = request.args.get("category")
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_insights(category=category))
        except Exception as e:
            logger.error(f"Error in /api/insights: {e}")
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
            return jsonify(nb.get_entries(
                experiment_id=exp_id, entry_type=entry_type, limit=n
            ))
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
        runner = _get_runner(notebook_path)
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

            data = {
                "aria": aria.get_status(),
                "summary": summary,
                "recent_experiments": nb.get_recent_experiments(10),
                "top_programs": nb.get_top_programs(10),
                "insights": nb.get_insights(limit=10),
                "recent_entries": nb.get_entries(limit=20),
                "is_running": runner.is_running,
                "progress": runner.progress.to_dict(),
            }

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

    # ── Report endpoint ──

    @app.route("/api/report")
    def api_report():
        """Consolidated research report with all data."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)

            data = {
                "summary": nb.get_dashboard_summary(),
                "top_programs": nb.get_top_programs(20, sort_by="loss_ratio"),
                "recent_experiments": nb.get_recent_experiments(100),
                "op_success_rates": analytics.op_success_rates(),
                "structural_correlations": analytics.structural_correlations(),
                "failure_patterns": analytics.failure_patterns(),
                "top_op_combinations": analytics.top_op_combinations(10),
                "efficiency_frontier": analytics.efficiency_frontier(),
                "experiment_clusters": analytics.experiment_clusters(),
                "grammar_weights": {
                    "learned": analytics.compute_grammar_weights(),
                    "default": analytics.get_current_grammar_weights(),
                    "control_comparison": analytics.control_experiment_comparison(),
                    "holdout_validation": analytics.holdout_validation(),
                },
                "learning_log": nb.get_learning_log(limit=50),
                "insights": nb.get_insights(),
            }

            # Generate narrative (optional, non-blocking)
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

    # ── Analytics endpoints ──

    @app.route("/api/analytics/op-success")
    def api_op_success():
        """Op success rate table."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.op_success_rates())
        except Exception as e:
            logger.error(f"Error in op-success: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/failure-patterns")
    def api_failure_patterns():
        """Failure analysis by error type and stage."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.failure_patterns())
        except Exception as e:
            logger.error(f"Error in failure-patterns: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/grammar-weights")
    def api_grammar_weights():
        """Current vs learned grammar weights."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            defaults = analytics.get_current_grammar_weights()
            learned = analytics.compute_grammar_weights()
            control_comparison = analytics.control_experiment_comparison()
            holdout = analytics.holdout_validation()
            return jsonify({
                "default": defaults,
                "learned": learned,
                "control_comparison": control_comparison,
                "holdout_validation": holdout,
            })
        except Exception as e:
            logger.error(f"Error in grammar-weights: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/efficiency-frontier")
    def api_efficiency_frontier():
        """Pareto-optimal programs on loss vs FLOPs."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.efficiency_frontier())
        except Exception as e:
            logger.error(f"Error in efficiency-frontier: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/experiment-clusters")
    def api_experiment_clusters():
        """Deterministic experiment clustering summary and stability signal."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.experiment_clusters())
        except Exception as e:
            logger.error(f"Error in experiment-clusters: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/routing-health")
    def api_routing_health():
        """Routing telemetry health summary grouped by routing mode."""
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.routing_health())
        except Exception as e:
            logger.error(f"Error in routing-health: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/learning-log")
    def api_learning_log():
        """Audit trail of grammar weight changes."""
        n = request.args.get("n", 100, type=int)
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_learning_log(limit=n))
        except Exception as e:
            logger.error(f"Error in learning-log: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/programs/<result_id>/training-curve")
    def api_training_curve(result_id):
        """Per-step training data for a program."""
        nb = LabNotebook(notebook_path)
        try:
            curve = nb.get_training_curve(result_id)
            return jsonify(curve)
        except Exception as e:
            logger.error(f"Error in training-curve: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Leaderboard endpoints ──

    @app.route("/api/leaderboard")
    def api_leaderboard():
        """Get leaderboard entries, optionally filtered by tier."""
        tier = request.args.get("tier")
        limit = request.args.get("limit", 50, type=int)
        sort_by = request.args.get("sort", "composite_score")
        nb = LabNotebook(notebook_path)
        try:
            entries = nb.get_leaderboard(tier=tier, limit=limit, sort_by=sort_by)
            # Group by tier for the dashboard
            tiers = {}
            for entry in entries:
                t = entry.get("tier", "screening")
                if t not in tiers:
                    tiers[t] = []
                tiers[t].append(entry)
            return jsonify({
                "entries": entries,
                "by_tier": tiers,
                "total": len(entries),
            })
        except Exception as e:
            logger.error(f"Error in /api/leaderboard: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Control endpoints ──

    @app.route("/api/experiments/start", methods=["POST"])
    def api_start_experiment():
        """Start a new experiment. Accepts RunConfig fields + optional hypothesis."""
        runner = _get_runner(notebook_path)
        if runner.is_running:
            return jsonify({"error": "An experiment is already running"}), 409

        body = request.get_json(silent=True) or {}
        hypothesis = body.pop("hypothesis", None)
        mode = body.pop("mode", "single")  # "single", "continuous", "evolve", "novelty"

        config = RunConfig.from_dict(body) if body else RunConfig()

        try:
            if mode == "continuous":
                config.continuous = True
                exp_id = runner.start_continuous(config)
            elif mode == "evolve":
                exp_id = runner.start_evolution(config, hypothesis=hypothesis)
            elif mode == "novelty":
                exp_id = runner.start_novelty_search(config, hypothesis=hypothesis)
            elif mode == "investigation":
                result_ids = body.get("result_ids", [])
                if not result_ids:
                    return jsonify({"error": "result_ids required for investigation mode"}), 400
                exp_id = runner.start_investigation(result_ids, config, hypothesis=hypothesis)
            elif mode == "validation":
                result_ids = body.get("result_ids", [])
                if not result_ids:
                    return jsonify({"error": "result_ids required for validation mode"}), 400
                exp_id = runner.start_validation(result_ids, config, hypothesis=hypothesis)
            elif mode == "scale_up":
                result_ids = body.get("result_ids", [])
                if not result_ids:
                    return jsonify({"error": "result_ids required for scale_up mode"}), 400
                config.scale_up = True
                config.scale_up_result_ids = ",".join(result_ids)
                exp_id = runner.start_scale_up(result_ids, config, hypothesis=hypothesis)
            else:
                exp_id = runner.start_experiment(config, hypothesis=hypothesis)

            return jsonify({
                "experiment_id": exp_id,
                "status": "started",
                "config": config.to_dict(),
                "aria_message": runner.progress.aria_message,
            })
        except Exception as e:
            logger.error(f"Error starting experiment: {e}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/experiments/stop", methods=["POST"])
    def api_stop_experiment():
        """Stop the currently running experiment."""
        runner = _get_runner(notebook_path)
        if not runner.is_running:
            return jsonify({"error": "No experiment is running"}), 409

        runner.stop()
        return jsonify({
            "status": "stopping",
            "aria_message": runner.progress.aria_message,
        })

    @app.route("/api/progress")
    def api_progress():
        """Get current experiment progress (poll-based alternative to SSE)."""
        runner = _get_runner(notebook_path)
        return jsonify({
            "is_running": runner.is_running,
            "progress": runner.progress.to_dict(),
        })

    @app.route("/api/events")
    def api_events():
        """SSE endpoint for real-time experiment events."""
        runner = _get_runner(notebook_path)
        sse_timeout = _get_sse_timeout_seconds()

        def event_stream():
            while True:
                for event in runner.get_events(timeout=sse_timeout):
                    data = json.dumps(event["data"])
                    yield f"event: {event['type']}\ndata: {data}\n\n"
                # After timeout, check if client is still connected
                yield f"event: keepalive\ndata: {{}}\n\n"

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.route("/api/config", methods=["GET"])
    def api_get_config():
        """Get the default RunConfig."""
        return jsonify(RunConfig().to_dict())

    # ── LLM Configuration endpoints ──

    @app.route("/api/llm/config")
    def api_llm_config():
        """Get current LLM backend configuration."""
        aria = get_aria()
        return jsonify(aria.get_llm_config())

    @app.route("/api/llm/config", methods=["POST"])
    def api_llm_configure():
        """Configure the LLM backend at runtime."""
        aria = get_aria()
        body = request.get_json(silent=True) or {}

        backend_name = body.get("backend", "")
        if not backend_name:
            return jsonify({"error": "backend is required (anthropic, openai, ollama)"}), 400

        success = aria.configure_llm(
            backend_name=backend_name,
            api_key=body.get("api_key", ""),
            model=body.get("model", ""),
            host=body.get("host", ""),
        )

        if success:
            return jsonify({
                "status": "configured",
                "config": aria.get_llm_config(),
            })
        else:
            return jsonify({"error": "Failed to configure LLM backend"}), 500

    # ── Aria Intelligence endpoints ──

    @app.route("/api/aria/recommendation")
    def api_aria_recommendation():
        """Get Aria's experiment recommendation based on all data."""
        runner = _get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            analytics_data = runner._gather_analytics_data(nb)
            history = nb.get_recent_experiments(10)
            past_hypotheses = runner._get_past_hypotheses(nb)
            from .llm.context import build_rich_context
            context = build_rich_context(
                results={"total": 0, "stage0_passed": 0, "stage05_passed": 0,
                         "stage1_passed": 0, "novel_count": 0},
                analytics_data=analytics_data,
                history=history,
                past_hypotheses=past_hypotheses,
            )
            suggestion = aria.suggest_experiment(context)
            return jsonify(suggestion)
        except Exception as e:
            logger.error(f"Error in /api/aria/recommendation: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/aria/strategy")
    def api_aria_strategy():
        """Get Aria's research strategy recommendation."""
        runner = _get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            analytics_data = runner._gather_analytics_data(nb)
            history = nb.get_recent_experiments(10)
            past_hypotheses = runner._get_past_hypotheses(nb)
            from .llm.context import build_rich_context
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

    @app.route("/api/system/status")
    def api_system_status():
        """Report system status: CUDA, LLM, database, runner state."""
        import torch
        runner = _get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()
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
            llm_info = {
                "available": llm is not None,
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
                "is_running": runner.is_running,
            })
        except Exception as e:
            logger.error(f"Error in /api/system/status: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/validate", methods=["POST"])
    def api_validate_pipeline():
        """Validate the synthesis pipeline by generating and testing programs."""
        body = request.get_json(silent=True) or {}
        n = body.get("n", 5)
        n = min(n, 20)  # cap at 20

        try:
            from ..synthesis.grammar import GrammarConfig, batch_generate
            from ..synthesis.compiler import compile_model
            from ..synthesis.validator import validate_graph
            from ..eval.sandbox import safe_eval

            grammar = GrammarConfig(model_dim=256, max_depth=8, max_ops=12)
            graphs = batch_generate(n, grammar)

            generated = len(graphs)
            compiled = 0
            passed_s0 = 0
            errors = []

            for graph in graphs:
                val = validate_graph(graph)
                if not val.valid:
                    errors.append(f"validation: {val.errors[0] if val.errors else 'unknown'}")
                    continue

                try:
                    model = compile_model(
                        [graph] * 2,
                        vocab_size=1000,
                        max_seq_len=128,
                    )
                    compiled += 1

                    result = safe_eval(model, batch_size=1, seq_len=64,
                                       vocab_size=1000, device="cpu")
                    if result.passed:
                        passed_s0 += 1
                    else:
                        errors.append(f"sandbox: {result.error or 'failed'}")
                    del model
                except Exception as e:
                    errors.append(f"compile: {str(e)[:60]}")

            healthy = compiled > 0 and passed_s0 > 0
            return jsonify({
                "generated": generated,
                "compiled": compiled,
                "passed_s0": passed_s0,
                "errors": errors[:5],
                "healthy": healthy,
            })
        except Exception as e:
            logger.error(f"Error in pipeline validation: {e}")
            return jsonify({
                "generated": 0,
                "compiled": 0,
                "passed_s0": 0,
                "errors": [str(e)],
                "healthy": False,
            })

    # ── Campaign endpoints ──

    @app.route("/api/campaigns")
    def api_campaigns():
        """List all campaigns with summary stats."""
        nb = LabNotebook(notebook_path)
        try:
            rows = nb.conn.execute(
                "SELECT * FROM campaigns ORDER BY timestamp DESC"
            ).fetchall()
            campaigns = []
            for r in rows:
                d = dict(r)
                # Add summary stats
                d["n_experiments"] = nb.conn.execute(
                    "SELECT COUNT(*) FROM experiments WHERE campaign_id = ?",
                    (d["campaign_id"],),
                ).fetchone()[0]
                d["n_hypotheses"] = nb.conn.execute(
                    "SELECT COUNT(*) FROM hypotheses WHERE campaign_id = ?",
                    (d["campaign_id"],),
                ).fetchone()[0]
                d["n_decisions"] = nb.conn.execute(
                    "SELECT COUNT(*) FROM decisions WHERE campaign_id = ?",
                    (d["campaign_id"],),
                ).fetchone()[0]
                campaigns.append(d)
            return jsonify(campaigns)
        except Exception as e:
            logger.error(f"Error in /api/campaigns: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>")
    def api_campaign_detail(campaign_id):
        """Full campaign detail with experiments, hypotheses, decisions."""
        nb = LabNotebook(notebook_path)
        try:
            campaign = nb.get_campaign(campaign_id)
            if campaign is None:
                return jsonify({"error": "Not found"}), 404
            return jsonify({
                "campaign": campaign,
                "experiments": nb.get_campaign_experiments(campaign_id),
                "hypotheses": nb.get_campaign_hypotheses(campaign_id),
                "decisions": nb.get_campaign_decisions(campaign_id),
            })
        except Exception as e:
            logger.error(f"Error in /api/campaigns/{campaign_id}: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/report")
    def api_campaign_report(campaign_id):
        """Compiled campaign report (LLM-generated narrative)."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            campaign = nb.get_campaign(campaign_id)
            if campaign is None:
                return jsonify({"error": "Not found"}), 404

            experiments = nb.get_campaign_experiments(campaign_id)
            hypotheses = nb.get_campaign_hypotheses(campaign_id)
            decisions = nb.get_campaign_decisions(campaign_id)
            knowledge = nb.get_knowledge()

            from .llm.context import build_campaign_report_context
            context = build_campaign_report_context(
                campaign, experiments, hypotheses, decisions, knowledge)
            report = aria.compile_campaign_report(
                campaign, experiments, hypotheses, decisions, knowledge,
                context=context)

            return jsonify({
                "campaign": campaign,
                "report": report,
                "stats": {
                    "n_experiments": len(experiments),
                    "n_hypotheses": len(hypotheses),
                    "n_confirmed": sum(1 for h in hypotheses if h.get("status") == "confirmed"),
                    "n_refuted": sum(1 for h in hypotheses if h.get("status") == "refuted"),
                    "n_decisions": len(decisions),
                },
            })
        except Exception as e:
            logger.error(f"Error in /api/campaigns/{campaign_id}/report: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/hypotheses")
    def api_campaign_hypotheses(campaign_id):
        """Hypothesis chain for a campaign."""
        nb = LabNotebook(notebook_path)
        try:
            hypotheses = nb.get_campaign_hypotheses(campaign_id)
            return jsonify(hypotheses)
        except Exception as e:
            logger.error(f"Error in campaign hypotheses: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/decisions")
    def api_campaign_decisions(campaign_id):
        """Decision log for a campaign."""
        nb = LabNotebook(notebook_path)
        try:
            decisions = nb.get_campaign_decisions(campaign_id)
            return jsonify(decisions)
        except Exception as e:
            logger.error(f"Error in campaign decisions: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns", methods=["POST"])
    def api_create_campaign():
        """Create a new campaign manually."""
        body = request.get_json(silent=True) or {}
        title = body.get("title", "")
        objective = body.get("objective", "")
        success_criteria = body.get("success_criteria", "")

        if not title or not objective or not success_criteria:
            return jsonify({"error": "title, objective, and success_criteria required"}), 400

        nb = LabNotebook(notebook_path)
        try:
            campaign_id = nb.create_campaign(
                title=title, objective=objective,
                success_criteria=success_criteria,
                parent_id=body.get("parent_campaign_id"),
            )
            return jsonify({
                "campaign_id": campaign_id,
                "status": "created",
            })
        except Exception as e:
            logger.error(f"Error creating campaign: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/pause", methods=["POST"])
    def api_pause_campaign(campaign_id):
        """Pause a campaign."""
        nb = LabNotebook(notebook_path)
        try:
            nb.update_campaign(campaign_id, status="paused")
            return jsonify({"status": "paused"})
        except Exception as e:
            logger.error(f"Error pausing campaign: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/complete", methods=["POST"])
    def api_complete_campaign(campaign_id):
        """Complete a campaign."""
        nb = LabNotebook(notebook_path)
        try:
            nb.update_campaign(campaign_id, status="completed",
                               completed_at=time.time())
            return jsonify({"status": "completed"})
        except Exception as e:
            logger.error(f"Error completing campaign: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    # ── Hypothesis endpoints ──

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

    # ── Knowledge base endpoints ──

    @app.route("/api/knowledge")
    def api_knowledge():
        """Knowledge base entries, optionally filtered by category."""
        category = request.args.get("category")
        nb = LabNotebook(notebook_path)
        try:
            entries = nb.get_knowledge(category=category)
            return jsonify(entries)
        except Exception as e:
            logger.error(f"Error in /api/knowledge: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/knowledge/search")
    def api_knowledge_search():
        """Search knowledge base."""
        q = request.args.get("q", "")
        if not q:
            return jsonify([])
        nb = LabNotebook(notebook_path)
        try:
            entries = nb.search_knowledge(q)
            return jsonify(entries)
        except Exception as e:
            logger.error(f"Error in knowledge search: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    return app


def _setup_logging(log_dir: Optional[str] = None):
    """Configure logging with console and file handlers."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler
    if log_dir is None:
        log_dir = str(Path(__file__).parent.parent)
    log_path = Path(log_dir) / "aria_dashboard.log"
    try:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=3,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        logger.info(f"Logging to {log_path}")
    except Exception as e:
        logger.warning(f"Could not create log file at {log_path}: {e}")


def run_server(
    notebook_path: str = "research/lab_notebook.db",
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
):
    """Run the API server."""
    _setup_logging()
    app = create_app(notebook_path)
    logger.info(f"Starting Aria's Dashboard API on http://{host}:{port}")
    print(f"Starting Aria's Dashboard API on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)
