"""actions API route registration."""

from __future__ import annotations

import logging
from flask import jsonify, request
from ._helpers import get_autonomy, _DISMISSED_ACTIONS
from ._strategy_recommendations import compute_action_queue
from ._utils import register_notebook_routes, register_routes, with_notebook_context
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def register_actions_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)

    def api_actions(nb=None):
        """Aggregated prioritized action list for the dashboard."""
        actions = compute_action_queue(nb)
        return jsonify(actions)

    def api_action_dismiss(action_id):
        """Dismiss an action card (ephemeral, resets on server restart)."""
        clean_id = str(action_id or "").strip()[:64]
        if not clean_id:
            return jsonify({"error": "Missing action_id"}), 400
        _DISMISSED_ACTIONS.add(clean_id)
        return jsonify(
            {"dismissed": clean_id, "total_dismissed": len(_DISMISSED_ACTIONS)}
        )

    def api_action_approve(action_id):
        """User approves a pending autonomous action."""
        try:
            autonomy, store = get_autonomy(notebook_path)
            action = autonomy.approve(action_id)
            if not action:
                return jsonify({"error": "Action not found or not pending"}), 404
            store.update_status(
                action_id,
                action.status,
                executed_at=action.executed_at,
                undo_snapshot=action.undo_snapshot,
            )
            return jsonify(action.to_dict())
        except Exception as e:
            logger.error(f"Error approving action {action_id}: {e}")
            return jsonify({"error": str(e)}), 500

    def api_action_undo(action_id):
        """Undo a recently executed autonomous action (within 5 min window)."""
        try:
            autonomy, store = get_autonomy(notebook_path)
            action = autonomy.undo(action_id)
            if not action:
                return jsonify(
                    {"error": "Action not found or undo window expired"}
                ), 404
            store.update_status(action_id, action.status)
            return jsonify(action.to_dict())
        except Exception as e:
            logger.error(f"Error undoing action {action_id}: {e}")
            return jsonify({"error": str(e)}), 500

    def api_aria_autonomy():
        """Get or update autonomy configuration (trust level, behaviors)."""
        try:
            autonomy, _store = get_autonomy(notebook_path)
            if request.method == "PUT":
                data = request.get_json(silent=True) or {}
                config = autonomy.update_config(data)
                return jsonify(config)
            return jsonify(autonomy.get_config())
        except Exception as e:
            logger.error(f"Error in /api/aria/autonomy: {e}")
            return jsonify({"error": str(e)}), 500

    def api_aria_activity():
        """Recent autonomous action history for the activity feed."""
        try:
            _autonomy, store = get_autonomy(notebook_path)
            limit = request.args.get("limit", 50, type=int)
            return jsonify(store.get_recent(limit=limit))
        except Exception as e:
            logger.error(f"Error in /api/aria/activity: {e}")
            return jsonify([]), 500

    register_notebook_routes(
        app,
        wnb,
        (("/api/actions", "api_actions", api_actions),),
    )
    register_routes(
        app,
        (
            (
                "/api/actions/<action_id>/dismiss",
                "api_action_dismiss",
                api_action_dismiss,
                ("POST",),
            ),
            (
                "/api/actions/<action_id>/approve",
                "api_action_approve",
                api_action_approve,
                ("POST",),
            ),
            (
                "/api/actions/<action_id>/undo",
                "api_action_undo",
                api_action_undo,
                ("POST",),
            ),
            (
                "/api/aria/autonomy",
                "api_aria_autonomy",
                api_aria_autonomy,
                ("GET", "PUT"),
            ),
            ("/api/aria/activity", "api_aria_activity", api_aria_activity),
        ),
    )
