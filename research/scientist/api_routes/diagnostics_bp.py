"""diagnostics API route registration."""

from __future__ import annotations

import logging
import os
from flask import jsonify, request
from ._utils import register_notebook_routes, register_routes, with_notebook_context
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def register_diagnostics_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb_writer = with_notebook_context(notebook_path, read_only=False)

    def api_fingerprint_diagnostics():
        """Expose lightweight runtime diagnostics for fingerprint analysis."""
        reset = str(request.args.get("reset", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        try:
            from research.eval._sensitivity_skip_stats import get_sensitivity_skip_stats

            stats = get_sensitivity_skip_stats(reset=reset)
            return jsonify(
                {
                    "sensitivity_skips": stats,
                }
            )
        except Exception as e:
            logger.error(f"Error in /api/diagnostics/fingerprint: {e}")
            return jsonify(
                {
                    "sensitivity_skips": {
                        "total": 0,
                        "by_reason": {},
                    },
                    "error": str(e),
                }
            ), 500

    def api_report_cache_diagnostics(nb=None):
        """Expose report snapshot cache usage and retention diagnostics."""
        cleanup = str(request.args.get("cleanup", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        try:
            ttl_seconds = int(
                os.environ.get("ARIA_REPORT_SNAPSHOT_TTL_SECONDS", str(7 * 24 * 3600))
            )
        except (TypeError, ValueError):
            ttl_seconds = 7 * 24 * 3600
        try:
            max_rows_per_scope = int(
                os.environ.get("ARIA_REPORT_SNAPSHOT_MAX_ROWS_PER_SCOPE", "400")
            )
        except (TypeError, ValueError):
            max_rows_per_scope = 400

        cleanup_stats = None
        if cleanup:
            cleanup_stats = nb.cleanup_report_snapshots(
                ttl_seconds=max(60, ttl_seconds),
                max_rows_per_scope=max(20, max_rows_per_scope),
            )

        snapshot_stats = nb.get_report_snapshot_stats()
        return jsonify(
            {
                "snapshot_cache": snapshot_stats,
                "retention": {
                    "ttl_seconds": max(60, int(ttl_seconds or 0)),
                    "max_rows_per_scope": max(20, int(max_rows_per_scope or 0)),
                },
                "cleanup_triggered": bool(cleanup),
                "cleanup": cleanup_stats,
            }
        )

    register_routes(
        app,
        (
            (
                "/api/diagnostics/fingerprint",
                "api_fingerprint_diagnostics",
                api_fingerprint_diagnostics,
            ),
        ),
    )
    register_notebook_routes(
        app,
        wnb_writer,
        (
            (
                "/api/diagnostics/report-cache",
                "api_report_cache_diagnostics",
                api_report_cache_diagnostics,
            ),
        ),
    )
