"""System and validation API route registration."""

from __future__ import annotations

import logging
import shutil
import subprocess
from datetime import datetime, timezone

from flask import jsonify, request

from ..native.telemetry import native_runner_capability_report
from ..persona import get_aria
from ..runner._types import RunConfig
from ._helpers import (
    get_passive_llm_config,
    get_runner,
    native_runner_canary_status_payload,
    resolve_runner_status,
)
from ._ml_influence_status import build_ml_influence_status
from ._strategy_preflight import (
    normalize_start_mode,
    apply_live_screening_bias,
    run_launch_preflight,
    run_pipeline_sample_check,
)
from ._strategy_report import parse_bool_query
from ._utils import register_notebook_routes, register_routes, with_notebook_context
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def _probe_cuda_status() -> tuple[bool, dict]:
    """Return lightweight CUDA availability without importing torch."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False, {}
    try:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("CUDA probe via nvidia-smi failed: %s", exc)
        return False, {}

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return False, {}
    first = [part.strip() for part in lines[0].split(",")]
    info = {"device_count": len(lines)}
    if first:
        info["device_name"] = first[0]
    if len(first) >= 3:
        try:
            info["memory_free_gb"] = round(float(first[1]) / 1024.0, 1)
            info["memory_total_gb"] = round(float(first[2]) / 1024.0, 1)
        except ValueError:
            logger.debug("Failed parsing CUDA memory info from nvidia-smi: %r", first)
    return True, info


def register_system_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)

    def api_system_status(nb=None):
        """Report system status: CUDA, LLM, database, runner state."""
        runner = get_runner(notebook_path, create_if_missing=False)
        aria = get_aria()
        refresh_canary = parse_bool_query(
            request.args.get("refresh_canary"), default=False
        )

        cuda_available, cuda_info = _probe_cuda_status()

        llm_info = get_passive_llm_config(notebook_path, aria=aria)

        summary = nb.get_dashboard_headline_summary()
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
                "ml_influence": build_ml_influence_status(),
                "native_runner": native_runner_capability_report(deep=False),
                "native_runner_canary": native_runner_canary_status_payload(
                    force_refresh=refresh_canary
                ),
                "is_running": runner_state["is_running"],
            }
        )

    def api_native_runner_capability():
        """Report native-runner adapter capability and current mode flags."""
        try:
            return jsonify(native_runner_capability_report())
        except Exception as e:
            logger.error("Error in /api/native-runner/capability: %s", e)
            return jsonify({"error": str(e)}), 500

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

    def api_perf_summary():
        """Return recent research perf artifacts and aggregate budget state."""
        try:
            from ...perf_contract import (
                list_recent_perf_artifacts,
                summarize_perf_artifacts,
            )

            limit = max(1, min(100, int(request.args.get("limit", 20))))
        except (TypeError, ValueError):
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

    def api_validate_pipeline():
        """Validate the synthesis pipeline by generating and testing programs."""
        body = request.get_json(silent=True) or {}
        n = min(int(body.get("n", body.get("sample_n", 5)) or 5), 20)
        mode = normalize_start_mode(body.pop("mode", "single"))
        auto_harden = bool(body.pop("auto_harden", True))
        runner = get_runner(notebook_path)
        config = RunConfig.from_dict(body) if body else RunConfig()
        if mode == "live_screening":
            apply_live_screening_bias(config)
            mode = "single"
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

    register_notebook_routes(
        app,
        wnb,
        (("/api/system/status", "api_system_status", api_system_status),),
    )
    register_routes(
        app,
        (
            (
                "/api/native-runner/capability",
                "api_native_runner_capability",
                api_native_runner_capability,
            ),
            (
                "/api/native-runner/canary/refresh",
                "api_native_runner_canary_refresh",
                api_native_runner_canary_refresh,
                ("POST",),
            ),
            (
                "/api/native-runner/telemetry",
                "api_native_runner_telemetry",
                api_native_runner_telemetry,
            ),
            ("/api/perf/summary", "api_perf_summary", api_perf_summary),
            (
                "/api/validate",
                "api_validate_pipeline",
                api_validate_pipeline,
                ("POST",),
            ),
        ),
    )
