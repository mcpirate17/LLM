"""campaigns API route registration."""
from __future__ import annotations

import logging
import time
from flask import jsonify, request
from ..notebook import LabNotebook
from ..persona import get_aria
from ._helpers import get_runner
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def register_campaigns_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path

    @app.route("/api/campaigns")
    def api_campaigns():
        """List all campaigns with summary stats."""
        nb = LabNotebook(notebook_path)
        try:
            rows = nb.conn.execute(
                "SELECT * FROM campaigns ORDER BY timestamp DESC"
            ).fetchall()
            campaigns = []
            for r in rows:
                d = dict(r)
                d["n_experiments"] = nb.conn.execute(
                    "SELECT COUNT(*) FROM experiments WHERE campaign_id = ?",
                    (d["campaign_id"],),
                ).fetchone()[0]
                d["n_hypotheses"] = nb.conn.execute(
                    "SELECT COUNT(*) FROM hypotheses WHERE campaign_id = ?",
                    (d["campaign_id"],),
                ).fetchone()[0]
                d["n_decisions"] = nb.conn.execute(
                    "SELECT COUNT(*) FROM decisions WHERE campaign_id = ?",
                    (d["campaign_id"],),
                ).fetchone()[0]
                campaigns.append(d)
            return jsonify(campaigns)
        except Exception as e:
            logger.error(f"Error in /api/campaigns: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>")
    def api_campaign_detail(campaign_id):
        """Full campaign detail with experiments, hypotheses, decisions."""
        nb = LabNotebook(notebook_path)
        try:
            campaign = nb.get_campaign(campaign_id)
            if campaign is None:
                return jsonify({"error": "Not found"}), 404
            experiments = nb.get_campaign_experiments(campaign_id)
            hypotheses = nb.get_campaign_hypotheses(campaign_id)
            decisions = nb.get_campaign_decisions(campaign_id)
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            success_criteria_tracker = analytics.campaign_success_criteria_tracker(
                campaign=campaign,
                experiments=experiments,
                hypotheses=hypotheses,
                decisions=decisions,
            )
            return jsonify({
                "campaign": campaign,
                "experiments": experiments,
                "hypotheses": hypotheses,
                "decisions": decisions,
                "success_criteria_tracker": success_criteria_tracker,
            })
        except Exception as e:
            logger.error(f"Error in /api/campaigns/{campaign_id}: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/report")
    def api_campaign_report(campaign_id):
        """Compiled campaign report (LLM-generated narrative)."""
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            campaign = nb.get_campaign(campaign_id)
            if campaign is None:
                return jsonify({"error": "Not found"}), 404

            experiments = nb.get_campaign_experiments(campaign_id)
            hypotheses = nb.get_campaign_hypotheses(campaign_id)
            decisions = nb.get_campaign_decisions(campaign_id)
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
                campaign, experiments, hypotheses, decisions, knowledge)
            report = aria.compile_campaign_report(
                campaign, experiments, hypotheses, decisions, knowledge,
                context=ctx)

            return jsonify({
                "campaign": campaign,
                "report": report,
                "stats": {
                    "n_experiments": len(experiments),
                    "n_hypotheses": len(hypotheses),
                    "n_confirmed": sum(1 for h in hypotheses if h.get("status") == "confirmed"),
                    "n_refuted": sum(1 for h in hypotheses if h.get("status") == "refuted"),
                    "n_decisions": len(decisions),
                },
                "success_criteria_tracker": success_criteria_tracker,
            })
        except Exception as e:
            logger.error(f"Error in /api/campaigns/{campaign_id}/report: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/hypotheses")
    def api_campaign_hypotheses(campaign_id):
        """Hypothesis chain for a campaign."""
        nb = LabNotebook(notebook_path)
        try:
            hypotheses = nb.get_campaign_hypotheses(campaign_id)
            return jsonify(hypotheses)
        except Exception as e:
            logger.error(f"Error in campaign hypotheses: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/decisions")
    def api_campaign_decisions(campaign_id):
        """Decision log for a campaign."""
        nb = LabNotebook(notebook_path)
        try:
            decisions = nb.get_campaign_decisions(campaign_id)
            return jsonify(decisions)
        except Exception as e:
            logger.error(f"Error in campaign decisions: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns", methods=["POST"])
    def api_create_campaign():
        """Create a new campaign manually."""
        body = request.get_json(silent=True) or {}
        title = body.get("title", "")
        objective = body.get("objective", "")
        success_criteria = body.get("success_criteria", "")

        if not title or not objective or not success_criteria:
            return jsonify({"error": "title, objective, and success_criteria required"}), 400

        nb = LabNotebook(notebook_path)
        try:
            campaign_id = nb.create_campaign(
                title=title, objective=objective,
                success_criteria=success_criteria,
                parent_id=body.get("parent_campaign_id"),
            )
            return jsonify({
                "campaign_id": campaign_id,
                "status": "created",
            })
        except Exception as e:
            logger.error(f"Error creating campaign: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/pause", methods=["POST"])
    def api_pause_campaign(campaign_id):
        """Pause a campaign."""
        nb = LabNotebook(notebook_path)
        try:
            nb.update_campaign(campaign_id, status="paused")
            return jsonify({"status": "paused"})
        except Exception as e:
            logger.error(f"Error pausing campaign: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/campaigns/<campaign_id>/complete", methods=["POST"])
    def api_complete_campaign(campaign_id):
        """Complete a campaign."""
        nb = LabNotebook(notebook_path)
        try:
            campaign = nb.get_campaign(campaign_id)
            nb.update_campaign(campaign_id, status="completed",
                               completed_at=time.time())
            runner = get_runner(notebook_path)
            runner._emit_event("campaign_completed", {
                "campaign_id": campaign_id,
                "title": (campaign or {}).get("title", ""),
            })
            return jsonify({"status": "completed"})
        except Exception as e:
            logger.error(f"Error completing campaign: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()
