"""
REST API Server for the AI Scientist Dashboard

Serves data from the lab notebook to the React dashboard.
Provides control endpoints for starting/stopping experiments.
Uses Flask for simplicity, SSE for real-time streaming.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from .notebook import LabNotebook
from .persona import get_aria
from .runner import ExperimentRunner, RunConfig
from .llm.context import build_program_context, build_history_context


# Singleton runner shared across requests
_runner: Optional[ExperimentRunner] = None


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
        finally:
            nb.close()

    @app.route("/api/experiments")
    def api_experiments():
        """List recent experiments."""
        n = request.args.get("n", 20, type=int)
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_recent_experiments(n))
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
            # Fixed: query programs for THIS experiment, not global top 50
            programs = nb.get_program_results(experiment_id)
            return jsonify({
                "experiment": exp,
                "entries": entries,
                "programs": programs,
            })
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>/programs")
    def api_experiment_programs(experiment_id):
        """All programs for an experiment (not just S1 survivors)."""
        nb = LabNotebook(notebook_path)
        try:
            programs = nb.get_program_results(experiment_id)
            return jsonify(programs)
        finally:
            nb.close()

    @app.route("/api/programs/<result_id>")
    def api_program_detail(result_id):
        """Full program detail with parsed graph JSON + fingerprint."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            program = nb.get_program_detail(result_id)
            if program is None:
                return jsonify({"error": "Not found"}), 404

            # Try LLM explanation of fingerprint
            ctx = build_program_context(program)
            explanation = aria.explain_fingerprint(ctx)
            if explanation:
                program["llm_explanation"] = explanation

            return jsonify(program)
        finally:
            nb.close()

    @app.route("/api/experiments/<experiment_id>/failures")
    def api_failure_analysis(experiment_id):
        """Failure analysis: error distribution, stage funnel."""
        nb = LabNotebook(notebook_path)
        try:
            analysis = nb.get_failure_analysis(experiment_id)
            return jsonify(analysis)
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
                nb.conn.execute(
                    "UPDATE experiments SET llm_analysis = ? WHERE experiment_id = ?",
                    (analysis, experiment_id),
                )
                nb.conn.commit()
                return jsonify({"analysis": analysis, "source": "generated"})

            return jsonify({"analysis": None, "source": "unavailable",
                            "reason": "No LLM backend configured"})
        finally:
            nb.close()

    @app.route("/api/trends")
    def api_trends():
        """Cross-experiment trend data for charts."""
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_experiment_trends())
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
        finally:
            nb.close()

    @app.route("/api/insights")
    def api_insights():
        """List active insights."""
        category = request.args.get("category")
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_insights(category=category))
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
        finally:
            nb.close()

    @app.route("/api/metrics/<metric_name>")
    def api_metrics(metric_name):
        """Get time-series metrics."""
        exp_id = request.args.get("experiment_id")
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_metrics(metric_name, experiment_id=exp_id))
        finally:
            nb.close()

    @app.route("/api/dashboard")
    def api_dashboard():
        """Get all dashboard data in one call."""
        runner = _get_runner(notebook_path)
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            return jsonify({
                "aria": aria.get_status(),
                "summary": nb.get_dashboard_summary(),
                "recent_experiments": nb.get_recent_experiments(10),
                "top_programs": nb.get_top_programs(10),
                "insights": nb.get_insights(limit=10),
                "recent_entries": nb.get_entries(limit=20),
                "is_running": runner.is_running,
                "progress": runner.progress.to_dict(),
            })
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
        mode = body.pop("mode", "single")  # "single" or "continuous"

        config = RunConfig.from_dict(body) if body else RunConfig()

        try:
            if mode == "continuous":
                config.continuous = True
                exp_id = runner.start_continuous(config)
            else:
                exp_id = runner.start_experiment(config, hypothesis=hypothesis)

            return jsonify({
                "experiment_id": exp_id,
                "status": "started",
                "config": config.to_dict(),
                "aria_message": runner.progress.aria_message,
            })
        except Exception as e:
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

        def event_stream():
            while True:
                for event in runner.get_events(timeout=30.0):
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

    return app


def run_server(
    notebook_path: str = "research/lab_notebook.db",
    host: str = "0.0.0.0",
    port: int = 5000,
    debug: bool = False,
):
    """Run the API server."""
    app = create_app(notebook_path)
    print(f"Starting Aria's Dashboard API on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)
