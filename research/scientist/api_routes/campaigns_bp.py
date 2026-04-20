"""campaigns API route registration."""

from __future__ import annotations

import logging
import sqlite3
import time
from flask import jsonify, request
from ._helpers import get_aria_for_notebook, get_runner
from ._utils import is_malformed_db_error as _is_malformed_db_error
from ._utils import register_notebook_routes, with_notebook_context
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def _safe_campaigns_list(nb):
    rows = nb.conn.execute(
        """
        SELECT
            c.*,
            COALESCE(e.n_experiments, 0) AS n_experiments,
            COALESCE(h.n_hypotheses, 0) AS n_hypotheses,
            COALESCE(d.n_decisions, 0) AS n_decisions
        FROM campaigns
        AS c
        LEFT JOIN (
            SELECT campaign_id, COUNT(DISTINCT experiment_id) AS n_experiments
            FROM experiments
            GROUP BY campaign_id
        ) AS e ON e.campaign_id = c.campaign_id
        LEFT JOIN (
            SELECT campaign_id, COUNT(DISTINCT hypothesis_id) AS n_hypotheses
            FROM hypotheses
            GROUP BY campaign_id
        ) AS h ON h.campaign_id = c.campaign_id
        LEFT JOIN (
            SELECT campaign_id, COUNT(DISTINCT decision_id) AS n_decisions
            FROM decisions
            GROUP BY campaign_id
        ) AS d ON d.campaign_id = c.campaign_id
        ORDER BY c.timestamp DESC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def _safe_campaign_detail_payload(nb, campaign_id: str):
    campaign = nb.get_campaign(campaign_id)
    if campaign is None:
        return None

    experiments = nb.get_campaign_experiments(campaign_id)
    hypotheses = nb.get_campaign_hypotheses(campaign_id)
    decisions = nb.get_campaign_decisions(campaign_id)
    return {
        "campaign": campaign,
        "experiments": experiments,
        "hypotheses": hypotheses,
        "decisions": decisions,
    }


def register_campaigns_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)

    def api_campaigns(nb=None):
        """List all campaigns with summary stats."""
        campaigns = _safe_campaigns_list(nb)
        return jsonify(campaigns)

    def api_campaign_detail(campaign_id, nb=None):
        """Full campaign detail with experiments, hypotheses, decisions."""
        payload = _safe_campaign_detail_payload(nb, campaign_id)
        if payload is None:
            return jsonify({"error": "Not found"}), 404

        success_criteria_tracker = {"criteria": [], "summary": None}
        try:
            from ..analytics import ExperimentAnalytics

            analytics = ExperimentAnalytics(nb)
            success_criteria_tracker = analytics.campaign_success_criteria_tracker(
                campaign=payload["campaign"],
                experiments=payload["experiments"],
                hypotheses=payload["hypotheses"],
                decisions=payload["decisions"],
            )
        except sqlite3.OperationalError as exc:
            if not _is_malformed_db_error(exc):
                raise
            logger.warning(
                "Campaign analytics degraded for %s due to malformed DB pages: %s",
                campaign_id,
                exc,
            )

        payload["success_criteria_tracker"] = success_criteria_tracker
        return jsonify(payload)

    def api_campaign_report(campaign_id, nb=None):
        """Compiled campaign report (LLM-generated narrative)."""
        aria = get_aria_for_notebook(notebook_path)
        payload = _safe_campaign_detail_payload(nb, campaign_id)
        if payload is None:
            return jsonify({"error": "Not found"}), 404

        campaign = payload["campaign"]
        experiments = payload["experiments"]
        hypotheses = payload["hypotheses"]
        decisions = payload["decisions"]
        knowledge = nb.get_knowledge()
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        success_criteria_tracker = analytics.campaign_success_criteria_tracker(
            campaign=campaign,
            experiments=experiments,
            hypotheses=hypotheses,
            decisions=decisions,
        )

        from ..llm.context_hypothesis import build_campaign_report_context

        ctx = build_campaign_report_context(
            campaign, experiments, hypotheses, decisions, knowledge
        )
        report = aria.compile_campaign_report(
            campaign, experiments, hypotheses, decisions, knowledge, context=ctx
        )

        return jsonify(
            {
                "campaign": campaign,
                "report": report,
                "stats": {
                    "n_experiments": len(experiments),
                    "n_hypotheses": len(hypotheses),
                    "n_confirmed": sum(
                        1 for h in hypotheses if h.get("status") == "confirmed"
                    ),
                    "n_refuted": sum(
                        1 for h in hypotheses if h.get("status") == "refuted"
                    ),
                    "n_decisions": len(decisions),
                },
                "success_criteria_tracker": success_criteria_tracker,
            }
        )

    def api_campaign_hypotheses(campaign_id, nb=None):
        """Hypothesis chain for a campaign."""
        payload = _safe_campaign_detail_payload(nb, campaign_id)
        if payload is None:
            return jsonify({"error": "Not found"}), 404
        hypotheses = payload["hypotheses"]
        return jsonify(hypotheses)

    def api_campaign_decisions(campaign_id, nb=None):
        """Decision log for a campaign."""
        payload = _safe_campaign_detail_payload(nb, campaign_id)
        if payload is None:
            return jsonify({"error": "Not found"}), 404
        decisions = payload["decisions"]
        return jsonify(decisions)

    def api_create_campaign(nb=None):
        """Create a new campaign manually."""
        body = request.get_json(silent=True) or {}
        title = body.get("title", "")
        objective = body.get("objective", "")
        success_criteria = body.get("success_criteria", "")

        if not title or not objective or not success_criteria:
            return jsonify(
                {"error": "title, objective, and success_criteria required"}
            ), 400

        campaign_id = nb.create_campaign(
            title=title,
            objective=objective,
            success_criteria=success_criteria,
            parent_id=body.get("parent_campaign_id"),
        )
        return jsonify(
            {
                "campaign_id": campaign_id,
                "status": "created",
            }
        )

    def api_pause_campaign(campaign_id, nb=None):
        """Pause a campaign."""
        nb.update_campaign(campaign_id, status="paused")
        return jsonify({"status": "paused"})

    def api_complete_campaign(campaign_id, nb=None):
        """Complete a campaign."""
        campaign = nb.get_campaign(campaign_id)
        nb.update_campaign(campaign_id, status="completed", completed_at=time.time())
        runner = get_runner(notebook_path)
        runner._emit_event(
            "campaign_completed",
            {
                "campaign_id": campaign_id,
                "title": (campaign or {}).get("title", ""),
            },
        )
        return jsonify({"status": "completed"})

    register_notebook_routes(
        app,
        wnb,
        (
            ("/api/campaigns", "api_campaigns", api_campaigns),
            (
                "/api/campaigns/<campaign_id>",
                "api_campaign_detail",
                api_campaign_detail,
            ),
            (
                "/api/campaigns/<campaign_id>/report",
                "api_campaign_report",
                api_campaign_report,
            ),
            (
                "/api/campaigns/<campaign_id>/hypotheses",
                "api_campaign_hypotheses",
                api_campaign_hypotheses,
            ),
            (
                "/api/campaigns/<campaign_id>/decisions",
                "api_campaign_decisions",
                api_campaign_decisions,
            ),
            ("/api/campaigns", "api_create_campaign", api_create_campaign, ("POST",)),
            (
                "/api/campaigns/<campaign_id>/pause",
                "api_pause_campaign",
                api_pause_campaign,
                ("POST",),
            ),
            (
                "/api/campaigns/<campaign_id>/complete",
                "api_complete_campaign",
                api_complete_campaign,
                ("POST",),
            ),
        ),
    )
