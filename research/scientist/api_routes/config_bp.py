"""config API route registration."""

from __future__ import annotations

import logging
from flask import jsonify, request
from ..runner import RunConfig
from ..persona import get_aria
from ._helpers import save_llm_config
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def register_config_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path

    @app.route("/api/config", methods=["GET"])
    def api_get_config():
        """Get the default RunConfig."""
        return jsonify(RunConfig().to_dict())

    @app.route("/api/scoring/version", methods=["GET"])
    def api_get_scoring_version():
        """Return the active composite scoring version.

        ``v8`` (default) uses the original v8 weights. ``v8.1`` applies the
        capability-first rebalance (tighter binding penalty + boost for
        graphs that actually bind). See research/tasks/todo.md for the full
        rationale.
        """
        from ..leaderboard_scoring import (
            SUPPORTED_SCORING_VERSIONS,
            get_scoring_version,
        )

        return jsonify(
            {
                "version": get_scoring_version(),
                "supported": list(SUPPORTED_SCORING_VERSIONS),
            }
        )

    @app.route("/api/scoring/version", methods=["POST"])
    def api_set_scoring_version():
        """Switch the active composite scoring version at runtime.

        Historical rows scored under the previous version are not
        rescored — the switch only affects composites computed after the
        call returns.
        """
        from ..leaderboard_scoring import (
            SUPPORTED_SCORING_VERSIONS,
            set_scoring_version,
        )

        body = request.get_json(silent=True) or {}
        version = str(body.get("version", "")).strip()
        if not version:
            return jsonify({"error": "version is required"}), 400
        try:
            new_version = set_scoring_version(version)
        except ValueError as exc:
            return jsonify(
                {
                    "error": str(exc),
                    "supported": list(SUPPORTED_SCORING_VERSIONS),
                }
            ), 400
        logger.info("Scoring version changed to %s via API", new_version)
        return jsonify({"version": new_version})

    @app.route("/api/llm/config")
    def api_llm_config():
        """Get current LLM backend configuration."""
        aria = get_aria()
        return jsonify(aria.get_llm_config())

    @app.route("/api/llm/config", methods=["POST"])
    def api_llm_configure():
        """Configure the LLM backend at runtime and persist to disk."""
        aria = get_aria()
        body = request.get_json(silent=True) or {}

        backend_name = str(body.get("backend", "")).strip()
        if not backend_name:
            return jsonify(
                {"error": "backend is required (anthropic, openai, ollama)"}
            ), 400

        api_key = str(body.get("api_key", "")).strip()
        model = str(body.get("model", "")).strip()
        host = str(body.get("host", "")).strip()

        success = aria.configure_llm(
            backend_name=backend_name,
            api_key=api_key,
            model=model,
            host=host,
        )

        if success:
            health_ok = True
            health_error = None
            llm = aria._get_llm()
            if llm:
                try:
                    test_resp = llm.generate(
                        "Respond with exactly: OK",
                        max_tokens=10,
                        temperature=0,
                    )
                    if not (test_resp and test_resp.text):
                        health_ok = False
                        health_error = "LLM returned empty response"
                except Exception as e:
                    health_ok = False
                    health_error = f"{type(e).__name__}: {str(e)[:150]}"
                    logger.warning(f"LLM health check failed: {health_error}")

            save_llm_config(
                notebook_path,
                {
                    "backend": backend_name,
                    "api_key_env": "ANTHROPIC_API_KEY"
                    if backend_name == "anthropic"
                    else "",
                    "model": model,
                    "host": host,
                },
            )

            if hasattr(aria, "_briefing_cache"):
                aria._briefing_cache = None

            result = {
                "status": "configured",
                "config": aria.get_llm_config(),
            }
            if not health_ok:
                result["status"] = "configured_with_warning"
                result["warning"] = health_error
            return jsonify(result)
        else:
            return jsonify({"error": "Failed to configure LLM backend"}), 500
