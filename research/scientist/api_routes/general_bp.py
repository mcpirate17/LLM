"""Aria control and strategy route registration."""

from __future__ import annotations

import csv
import io
import logging
from typing import Any, Dict, List

from flask import jsonify, request, Response
from ..runner import RunConfig
from ..persona import get_aria
from ..evidence import build_evidence_pack
from ._helpers import (
    get_runner,
    get_run_trigger_snapshot,
    record_run_trigger,
    resolve_runner_status,
)
from ._strategy_report import (
    normalize_entries,
)
from ._chat import (
    chat_guardrail_snapshot,
    local_ollama_helper_status,
    get_local_ollama_settings,
)
from .deps import ApiRouteContext
from ._utils import with_notebook_context

logger = logging.getLogger(__name__)


def register_general_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)
    _dashboard_index_path = context.dashboard_index_path
    _dashboard_missing_response = context.dashboard_missing_response
    _is_asset_path = context.is_asset_path

    @app.route("/api/template-names")
    def api_template_names():
        """Return sorted list of all available template names."""
        from ...synthesis.templates import TEMPLATES

        return jsonify({"names": sorted(TEMPLATES.keys())})

    @app.route("/api/aria/cycle-status")
    @wnb
    def api_aria_cycle_status(nb=None):
        """Get Aria continuous-cycle status (planning/running/analyzing)."""
        runner = get_runner(notebook_path)
        cycle = runner.get_aria_cycle_status()
        runner_state = resolve_runner_status(nb, runner)
        external = runner_state.get("external_snapshot")
        if external and not runner.is_running:
            cycle.update(
                {
                    "aria_message": runner_state["progress"].get("aria_message", ""),
                    "continuous_active": True,
                    "experiment_id": external["experiment_id"],
                    "is_running": True,
                    "last_note": (
                        f"External {external['mode']} experiment detected via notebook activity."
                    ),
                    "phase": "running",
                    "phase_label": "Running",
                    "progress_status": "running",
                    "selected_mode": external["mode"],
                    "external_process": True,
                }
            )
        return jsonify(cycle)

    @app.route("/api/aria/cycle-history")
    @wnb
    def api_aria_cycle_history(nb=None):
        """Get persisted Aria cycle summaries from notebook live-feed entries."""
        n = request.args.get("n", 100, type=int)
        mode_filter = str(request.args.get("mode") or "").strip().lower()
        status_filter = str(request.args.get("status") or "").strip().lower()
        query_text = str(request.args.get("q") or "").strip().lower()
        output_format = str(request.args.get("format") or "json").strip().lower()
        compact = str(request.args.get("compact", "1")).strip().lower() not in {
            "0",
            "false",
            "no",
        }
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
                searchable = " ".join(
                    [
                        str(row.get("mode") or ""),
                        str(row.get("status") or ""),
                        str(row.get("reasoning") or ""),
                        str(row.get("error") or ""),
                    ]
                ).lower()
                if query_text not in searchable:
                    continue

            if compact:
                reasoning = str(row.get("reasoning") or "").strip()
                error = str(row.get("error") or "").strip()
                row = {
                    "cycle_index": row.get("cycle_index"),
                    "mode": row.get("mode"),
                    "status": row.get("status"),
                    "timestamp": row.get("timestamp"),
                    "delta_programs": row.get("delta_programs"),
                    "delta_stage1_survivors": row.get("delta_stage1_survivors"),
                    "stage1_survivors": row.get("stage1_survivors"),
                    "confidence": row.get("confidence"),
                    "entry_id": row.get("entry_id"),
                    "experiment_id": row.get("experiment_id"),
                    "entry_timestamp": row.get("entry_timestamp"),
                    "reasoning": reasoning[:240],
                    "error": error[:180],
                }

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
            config_payload = (
                body.get("config") if isinstance(body.get("config"), dict) else body
            )
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
                return jsonify(
                    {
                        "ok": True,
                        "action": "start",
                        "experiment_id": exp_id,
                        "config": config.to_dict(),
                        "prescreen": prescreen,
                        "cycle": runner.get_aria_cycle_status(),
                    }
                )
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                logger.error(f"Error starting cycle control: {e}")
                return jsonify({"error": str(e)}), 500

        return jsonify({"error": "action must be one of: start, pause, resume"}), 400

    @app.route("/api/aria/recommendation")
    @wnb
    def api_aria_recommendation(nb=None):
        """Get Aria's experiment recommendation based on all data."""
        runner = get_runner(notebook_path)
        aria = get_aria()
        analytics_data = runner._gather_analytics_data(nb)
        history = nb.get_recent_experiments(10)
        past_hypotheses = runner._get_past_hypotheses(nb)
        from ..llm.context_experiment import build_rich_context

        context = build_rich_context(
            results={
                "total": 0,
                "stage0_passed": 0,
                "stage05_passed": 0,
                "stage1_passed": 0,
                "novel_count": 0,
            },
            analytics_data=analytics_data,
            history=history,
            past_hypotheses=past_hypotheses,
        )
        suggestion = aria.suggest_experiment(
            context,
            op_success_rates=analytics_data.get("op_success_rates"),
            compression_coverage=analytics_data.get("compression_coverage"),
        )
        if suggestion:
            suggestion["evidence_pack"] = build_evidence_pack(
                nb,
                analytics=None,
                recommendation=suggestion,
                decision_type="api_recommendation",
                recent_experiments=history,
            )
        return jsonify(suggestion)

    @app.route("/api/aria/strategy")
    @wnb
    def api_aria_strategy(nb=None):
        """Get Aria's research strategy recommendation."""
        runner = get_runner(notebook_path)
        aria = get_aria()
        analytics_data = runner._gather_analytics_data(nb)
        history = nb.get_recent_experiments(10)
        past_hypotheses = runner._get_past_hypotheses(nb)
        from ..llm.context_experiment import build_rich_context

        context = build_rich_context(
            results={
                "total": 0,
                "stage0_passed": 0,
                "stage05_passed": 0,
                "stage1_passed": 0,
                "novel_count": 0,
            },
            analytics_data=analytics_data,
            history=history,
            past_hypotheses=past_hypotheses,
        )
        strategy = aria.plan_strategy(context)
        return jsonify(
            {
                "strategy": strategy,
                "available": strategy is not None,
            }
        )

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
            except Exception as exc:
                logger.debug("LLM availability check failed: %s", exc)
                llm_available = False
                llm_reason = "unreachable"

        cycle_status = runner.get_aria_cycle_status()
        ollama_helper = local_ollama_helper_status(llm)
        return jsonify(
            {
                "codebase_agent": {
                    "spawn_endpoint": True,
                    "status_endpoint": True,
                    "workspace_scoped": True,
                    "allow_write_default": True,
                    "execution_first_for_fix_requests": True,
                    "small_model_swarm_enabled": True,
                    "small_model_swarm_max_workers": get_local_ollama_settings().get(
                        "max_small_workers", 3
                    ),
                    "simple_task_policy": "prefer_3b_swarm_then_7b",
                    "complex_task_policy": "prefer_7b_single",
                },
                "local_ollama_helper": ollama_helper,
                "chat_actions": [
                    "adjust_config",
                    "adjust_grammar",
                    "start_experiment",
                    "edit_file",
                    "spawn_agent",
                ],
                "chat_guardrails": chat_guardrail_snapshot(window=200),
                "local_context_tools": [
                    "runner.progress",
                    "notebook.get_recent_experiments",
                    "workspace.search",
                ],
                "llm": {
                    "available": llm_available,
                    "reason": llm_reason,
                },
                "runner": {
                    "is_running": bool(runner.is_running),
                    "progress_status": (runner.progress.to_dict() or {}).get("status"),
                },
                "run_trigger": get_run_trigger_snapshot(
                    (runner.progress.to_dict() or {}).get("experiment_id")
                ),
                "continuous": {
                    "active": bool(cycle_status.get("continuous_active")),
                    "phase": cycle_status.get("phase"),
                },
            }
        )
