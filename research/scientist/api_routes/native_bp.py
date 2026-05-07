"""native API route registration."""

from __future__ import annotations

import logging
from flask import jsonify, request
from .deps import ApiRouteContext
from ._utils import register_routes

logger = logging.getLogger(__name__)


def register_native_routes(app, context: ApiRouteContext):

    def api_native_runner_profile():
        """Return per-node profiling data from the most recent native execution."""
        try:
            from .. import native_runner

            rust = native_runner._try_import_rust_scheduler()
            profiling_enabled = bool(
                rust is not None
                and hasattr(rust, "profiler_enabled")
                and rust.profiler_enabled()
            )

            profile = native_runner.get_native_profile()
            if profile is not None:
                node_profiles = list(profile.get("node_profiles", []))
                total_duration_us = sum(
                    float(p.get("duration_us", 0)) for p in node_profiles
                )
                return jsonify(
                    {
                        "status": "ok",
                        "enabled": profiling_enabled,
                        "node_profiles": node_profiles,
                        "peak_memory_bytes": int(profile.get("peak_memory_bytes", 0)),
                        "total_duration_us": total_duration_us,
                    }
                )
            else:
                return jsonify(
                    {
                        "status": "ok",
                        "enabled": profiling_enabled,
                        "node_profiles": [],
                        "peak_memory_bytes": 0,
                        "total_duration_us": 0.0,
                    }
                )
        except Exception as e:
            logger.error(f"Error in /api/native-profile/v2/data: {e}")
            return jsonify({"error": str(e)}), 500

    def api_native_runner_profile_enable():
        """Toggle native kernel profiling on or off."""
        try:
            from .. import native_runner

            body = request.get_json(silent=True) or {}
            enable = bool(body.get("enable", True))

            result = native_runner.enable_native_profiling(enable)

            rust = native_runner._try_import_rust_scheduler()
            now_enabled = bool(
                rust is not None
                and hasattr(rust, "profiler_enabled")
                and rust.profiler_enabled()
            )

            return jsonify(
                {
                    "status": "ok",
                    "requested": enable,
                    "enabled": now_enabled,
                    "accepted": result,
                }
            )
        except Exception as e:
            logger.error(f"Error in /api/native-profile/v2/enable: {e}")
            return jsonify({"error": str(e)}), 500

    register_routes(
        app,
        (
            (
                "/api/native-profile/v2/data",
                "api_native_runner_profile",
                api_native_runner_profile,
            ),
            (
                "/api/native-profile/v2/enable",
                "api_native_runner_profile_enable",
                api_native_runner_profile_enable,
                ("POST",),
            ),
        ),
    )
