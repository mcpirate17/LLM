"""System and validation API route registration."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from flask import jsonify, request

from ..native_runner import native_runner_capability_report
from ..persona import get_aria
from ..runner import RunConfig
from ...perf_contract import list_recent_perf_artifacts, summarize_perf_artifacts
from ._helpers import (
    get_runner,
    native_runner_canary_status_payload,
    resolve_runner_status,
)
from ._strategy_preflight import (
    normalize_start_mode,
    run_launch_preflight,
    run_pipeline_sample_check,
)
from ._strategy_report import parse_bool_query
from ._utils import with_notebook_context
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def register_system_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)

    @app.route("/api/system/status")
    @wnb
    def api_system_status(nb=None):
        """Report system status: CUDA, LLM, database, runner state."""
        import torch

        runner = get_runner(notebook_path)
        aria = get_aria()
        refresh_canary = parse_bool_query(
            request.args.get("refresh_canary"), default=False
        )

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

        llm = aria._get_llm()
        llm_reachable = False
        if llm is not None:
            try:
                llm_reachable = (
                    bool(llm.is_available()) if hasattr(llm, "is_available") else True
                )
            except Exception:
                llm_reachable = False
        llm_info = {
            "available": llm_reachable,
            "configured": llm is not None,
            "backend": llm.name if llm else None,
        }

        summary = nb.get_dashboard_summary()
        runner_state = resolve_runner_status(nb, runner)
        db_info = {
            "path": notebook_path,
            "total_experiments": summary.get("total_experiments", 0),
            "total_programs": summary.get("total_programs_evaluated", 0),
        }

        return jsonify(
            {
                "cuda": {"available": cuda_available, **cuda_info},
                "llm": llm_info,
                "database": db_info,
                "native_runner": native_runner_capability_report(),
                "native_runner_canary": native_runner_canary_status_payload(
                    force_refresh=refresh_canary
                ),
                "is_running": runner_state["is_running"],
            }
        )

    @app.route("/api/native-runner/capability")
    def api_native_runner_capability():
        """Report native-runner adapter capability and current mode flags."""
        try:
            return jsonify(native_runner_capability_report())
        except Exception as e:
            logger.error("Error in /api/native-runner/capability: %s", e)
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
            logger.error("Error in /api/native-runner/canary/refresh: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/native-runner/telemetry")
    def api_native_runner_telemetry():
        """Return native runner fallback metrics for dashboard consumption."""
        try:
            report = native_runner_capability_report()
            return jsonify(
                {
                    "status": "ok",
                    "metrics": report.get("fallback_metrics", {}),
                    "capability": {
                        "enabled": report.get("enabled"),
                        "strict": report.get("strict"),
                        "designer_runtime_available": report.get(
                            "designer_runtime_available"
                        ),
                        "status": report.get("status"),
                    },
                    "op_support": report.get("native_op_support", {}),
                }
            )
        except Exception as e:
            logger.error("Error in /api/native-runner/telemetry: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/perf/summary")
    def api_perf_summary():
        """Return recent research perf artifacts and aggregate budget state."""
        try:
            limit = max(1, min(100, int(request.args.get("limit", 20))))
        except Exception:
            limit = 20
        try:
            artifacts = list_recent_perf_artifacts(component="research", limit=limit)
            return jsonify(
                {
                    "status": "ok",
                    "summary": summarize_perf_artifacts(
                        artifacts, component="research"
                    ),
                    "artifacts": artifacts,
                }
            )
        except Exception as e:
            logger.error("Error in /api/perf/summary: %s", e)
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
            return jsonify(
                {
                    "generated": sample.get("generated", 0),
                    "compiled": sample.get("compiled", 0),
                    "passed_s0": sample.get("passed_s0", 0),
                    "errors": sample.get("errors", [])[:5],
                    "healthy": healthy,
                    "mode": mode,
                    "config": config.to_dict(),
                    "prescreen": prescreen,
                    "preflight": preflight,
                }
            )
        except Exception as e:
            logger.error("Error in pipeline validation: %s", e)
            return jsonify(
                {
                    "generated": 0,
                    "compiled": 0,
                    "passed_s0": 0,
                    "errors": [str(e)],
                    "healthy": False,
                    "mode": mode,
                    "config": config.to_dict(),
                    "prescreen": prescreen,
                }
            )
