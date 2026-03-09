"""leaderboard API route registration."""
from __future__ import annotations

import logging
from flask import jsonify, request
from ..json_utils import json_safe as _json_safe
from ..notebook import LabNotebook
from ._strategy import (
    annotate_qkv_usage, attach_long_context_breakdown,
    compute_cross_run_stability, infer_tier_for_program, count_discovery_tiers,
)
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def register_leaderboard_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path

    @app.route("/api/leaderboard")
    def api_leaderboard():
        """Get leaderboard entries, optionally filtered by tier."""
        tier = request.args.get("tier")
        limit = request.args.get("limit", 50, type=int)
        sort_by = request.args.get("sort", "composite_score")
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            entries = nb.get_leaderboard(tier=tier, limit=limit, sort_by=sort_by)
            attach_long_context_breakdown(nb, entries)
            stability = compute_cross_run_stability(
                nb, nb.get_top_programs(20, sort_by="loss_ratio")
            )
            stability_by_result = {
                c.get("result_id"): c
                for c in stability.get("candidates", [])
                if c.get("result_id")
            }
            for entry in entries:
                entry["cross_run_stability"] = stability_by_result.get(
                    entry.get("result_id"),
                    {"trend": "unknown", "seen_runs": 0,
                     "latest_rank": None, "previous_rank": None, "rank_delta": None},
                )
            annotate_qkv_usage(entries, analytics)
            tiers = {}
            for entry in entries:
                t = entry.get("tier", "screening")
                if t not in tiers:
                    tiers[t] = []
                tiers[t].append(entry)
            return jsonify({
                "entries": entries,
                "by_tier": tiers,
                "total": len(entries),
                "cross_run_stability_summary": stability.get("summary", {}),
                "cross_run_stability_window": stability.get("window_size", 0),
            })
        except Exception as e:
            logger.error(f"Error in /api/leaderboard: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/leaderboard/status", methods=["POST"])
    def api_leaderboard_update_status():
        body = request.get_json(silent=True) or {}
        tier = str(body.get("tier") or "").strip().lower()
        entry_id = str(body.get("entry_id") or "").strip()
        result_id = str(body.get("result_id") or "").strip()

        valid_tiers = {"screening", "investigation", "validation", "breakthrough"}
        if tier not in valid_tiers:
            return jsonify({"error": "tier must be one of screening, investigation, validation, breakthrough"}), 400
        if not entry_id and not result_id:
            return jsonify({"error": "entry_id or result_id is required"}), 400

        nb = LabNotebook(notebook_path)
        try:
            row = None
            if entry_id:
                row = nb.conn.execute(
                    "SELECT entry_id, result_id, tier FROM leaderboard WHERE entry_id = ?",
                    (entry_id,),
                ).fetchone()
            if row is None and result_id:
                row = nb.conn.execute(
                    "SELECT entry_id, result_id, tier FROM leaderboard WHERE result_id = ?",
                    (result_id,),
                ).fetchone()
            if row is None:
                return jsonify({"error": "Leaderboard entry not found"}), 404

            resolved_entry_id = row["entry_id"]
            nb.promote_to_tier(resolved_entry_id, tier)

            updated = nb.conn.execute(
                "SELECT entry_id, result_id, tier, timestamp FROM leaderboard WHERE entry_id = ?",
                (resolved_entry_id,),
            ).fetchone()

            return jsonify({
                "success": True,
                "entry": dict(updated) if updated else {"entry_id": resolved_entry_id, "tier": tier},
            })
        except Exception as e:
            logger.error(f"Error in /api/leaderboard/status: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/leaderboard/pin", methods=["POST"])
    def api_leaderboard_pin():
        body = request.get_json(silent=True) or {}
        entry_id = str(body.get("entry_id") or "").strip()
        result_id = str(body.get("result_id") or "").strip()
        pinned = bool(body.get("pinned", False))

        if not entry_id and not result_id:
            return jsonify({"error": "entry_id or result_id is required"}), 400

        nb = LabNotebook(notebook_path)
        try:
            resolved_entry_id = entry_id
            if not resolved_entry_id and result_id:
                row = nb.conn.execute(
                    "SELECT entry_id FROM leaderboard WHERE result_id = ?",
                    (result_id,),
                ).fetchone()
                if row:
                    resolved_entry_id = row["entry_id"]
            if not resolved_entry_id:
                return jsonify({"error": "Leaderboard entry not found"}), 404

            nb.set_leaderboard_pin(resolved_entry_id, pinned)
            return jsonify({"success": True, "entry_id": resolved_entry_id, "pinned": pinned})
        except Exception as e:
            logger.error(f"Error in /api/leaderboard/pin: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/discoveries")
    def api_discoveries():
        """Unified discoveries endpoint merging leaderboard + raw candidates."""
        from ..naming import annotate_display_names

        tier = request.args.get("tier")
        limit = request.args.get("limit", 100, type=int)
        sort_by = request.args.get("sort", "composite_score")
        view = request.args.get("view", "ranked")
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)

            if view == "all":
                programs = nb.get_top_programs(limit, sort_by="loss_ratio")
                attach_long_context_breakdown(nb, programs)
                annotate_qkv_usage(programs, analytics)
                for p in programs:
                    p["architecture_family"] = nb._classify_architecture_family(
                        graph_json=p.get("graph_json"),
                        routing_mode=p.get("routing_mode"),
                    )
                    p["tier"] = infer_tier_for_program(nb, p)
                annotate_display_names(programs)
                for p in programs:
                    p.pop("graph_json", None)
                    p.pop("_graph_json", None)
                    p.pop("loss_curve", None)

                tier_counts = count_discovery_tiers(nb)

                return jsonify({
                    "entries": _json_safe(programs),
                    "total": len(programs),
                    "tier_counts": tier_counts,
                    "view": "all",
                })

            entries = nb.get_leaderboard(tier=tier, limit=limit, sort_by=sort_by)
            attach_long_context_breakdown(nb, entries)
            stability = compute_cross_run_stability(
                nb, nb.get_top_programs(20, sort_by="loss_ratio")
            )
            stability_by_result = {
                c.get("result_id"): c
                for c in stability.get("candidates", [])
                if c.get("result_id")
            }
            for entry in entries:
                entry["cross_run_stability"] = stability_by_result.get(
                    entry.get("result_id"),
                    {"trend": "unknown", "seen_runs": 0,
                     "latest_rank": None, "previous_rank": None, "rank_delta": None},
                )
            annotate_qkv_usage(entries, analytics)
            annotate_display_names(entries)

            tier_counts = count_discovery_tiers(nb)

            return jsonify({
                "entries": _json_safe(entries),
                "total": len(entries),
                "tier_counts": tier_counts,
                "cross_run_stability_summary": stability.get("summary", {}),
                "cross_run_stability_window": stability.get("window_size", 0),
                "view": "ranked",
            })
        except Exception as e:
            logger.error(f"Error in /api/discoveries: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()
