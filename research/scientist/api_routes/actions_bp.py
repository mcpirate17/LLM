"""actions API route registration."""
from __future__ import annotations

import logging
from flask import jsonify
from ..notebook import LabNotebook
from ._helpers import get_autonomy, _DISMISSED_ACTIONS
from ._strategy_recommendations import compute_action_queue
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def register_actions_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path

    @app.route("/api/actions")
    def api_actions():
        """Aggregated prioritized action list for the dashboard."""
        nb = LabNotebook(notebook_path)
        try:
            actions = compute_action_queue(nb)
            return jsonify(actions)
        except Exception as e:
            logger.error(f"Error in /api/actions: {e}")
            return jsonify([]), 500
        finally:
            nb.close()

    @app.route("/api/actions/<action_id>/dismiss", methods=["POST"])
    def api_action_dismiss(action_id):
        """Dismiss an action card (ephemeral, resets on server restart)."""
        clean_id = str(action_id or "").strip()[:64]
        if not clean_id:
            return jsonify({"error": "Missing action_id"}), 400
        _DISMISSED_ACTIONS.add(clean_id)
        return jsonify({"dismissed": clean_id, "total_dismissed": len(_DISMISSED_ACTIONS)})

    @app.route("/api/actions/<action_id>/approve", methods=["POST"])
    def api_action_approve(action_id):
        """User approves a pending autonomous action."""
        try:
            autonomy, store = get_autonomy(notebook_path)
            action = autonomy.approve(action_id)
            if not action:
                return jsonify({"error": "Action not found or not pending"}), 404
            store.update_status(
                action_id, action.status,
                executed_at=action.executed_at,
                undo_snapshot=action.undo_snapshot,
            )
            return jsonify(action.to_dict())
        except Exception as e:
            logger.error(f"Error approving action {action_id}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/actions/<action_id>/undo", methods=["POST"])
    def api_action_undo(action_id):
        """Undo a recently executed autonomous action (within 5 min window)."""
        try:
            autonomy, store = get_autonomy(notebook_path)
            action = autonomy.undo(action_id)
            if not action:
                return jsonify({"error": "Action not found or undo window expired"}), 404
            store.update_status(action_id, action.status)
            return jsonify(action.to_dict())
        except Exception as e:
            logger.error(f"Error undoing action {action_id}: {e}")
            return jsonify({"error": str(e)}), 500
