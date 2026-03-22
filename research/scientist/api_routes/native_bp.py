"""native API route registration."""

from __future__ import annotations

import logging
from flask import jsonify, request
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def register_native_routes(app, context: ApiRouteContext):

    @app.route("/api/native-profile/v2/data")
    def api_native_runner_profile():
        """Return per-node profiling data from the most recent native execution."""
        try:
            from ..native_runner import get_native_profile, _try_import_rust_scheduler

            rust = _try_import_rust_scheduler()
            profiling_enabled = bool(
                rust is not None
                and hasattr(rust, "profiler_enabled")
                and rust.profiler_enabled()
            )

            profile = get_native_profile()
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

    @app.route("/api/native-profile/v2/enable", methods=["POST"])
    def api_native_runner_profile_enable():
        """Toggle native kernel profiling on or off."""
        try:
            from ..native_runner import (
                enable_native_profiling,
                _try_import_rust_scheduler,
            )

            body = request.get_json(silent=True) or {}
            enable = bool(body.get("enable", True))

            result = enable_native_profiling(enable)

            rust = _try_import_rust_scheduler()
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
