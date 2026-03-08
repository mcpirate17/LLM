"""leaderboard API route registration."""
from __future__ import annotations

import functools
import time
import datetime
from flask import jsonify, request, Response
from ..json_utils import json_safe as _json_safe
from ..notebook import LabNotebook
from .deps import ApiRouteContext, install_legacy_symbols

def register_leaderboard_routes(app, context: ApiRouteContext):
    install_legacy_symbols(globals(), context)

    @app.route("/api/leaderboard")
    def api_leaderboard():
        """Get leaderboard entries, optionally filtered by tier."""
        tier = request.args.get("tier")
        limit = request.args.get("limit", 50, type=int)
        sort_by = request.args.get("sort", "composite_score")
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            entries = nb.get_leaderboard(tier=tier, limit=limit, sort_by=sort_by)
            _attach_long_context_breakdown(nb, entries)
            stability = _compute_cross_run_stability(
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
                    {
                        "trend": "unknown",
                        "seen_runs": 0,
                        "latest_rank": None,
                        "previous_rank": None,
                        "rank_delta": None,
                    },
                )
            _annotate_qkv_usage(entries, analytics)
            # Group by tier for the dashboard
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
        """Update status (tier) for an existing leaderboard record."""
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
        """Pin or unpin a leaderboard entry."""
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
        """Unified discoveries endpoint merging leaderboard + raw candidates.

        Query params:
          tier: filter by tier (screening/investigation/validation/breakthrough)
          limit: max results (default 100)
          sort: sort key (default composite_score)
          view: 'all' for raw candidates, 'ranked' for leaderboard (default ranked)
        """
        from .naming import annotate_display_names

        tier = request.args.get("tier")
        limit = request.args.get("limit", 100, type=int)
        sort_by = request.args.get("sort", "composite_score")
        view = request.args.get("view", "ranked")
        nb = LabNotebook(notebook_path)
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)

            if view == "all":
                # Raw S1 survivors from program_results
                programs = nb.get_top_programs(limit, sort_by="loss_ratio")
                _attach_long_context_breakdown(nb, programs)
                _annotate_qkv_usage(programs, analytics)
                # Add family classification + display names
                for p in programs:
                    p["architecture_family"] = nb._classify_architecture_family(
                        graph_json=p.get("graph_json"),
                        routing_mode=p.get("routing_mode"),
                    )
                    p["tier"] = _infer_tier_for_program(nb, p)
                annotate_display_names(programs)
                # Strip large fields from response
                for p in programs:
                    p.pop("graph_json", None)
                    p.pop("_graph_json", None)
                    p.pop("loss_curve", None)

                # Compute tier counts from all S1 survivors
                tier_counts = _count_discovery_tiers(nb)

                return jsonify({
                    "entries": _json_safe(programs),
                    "total": len(programs),
                    "tier_counts": tier_counts,
                    "view": "all",
                })

            # Default: ranked leaderboard view
            entries = nb.get_leaderboard(tier=tier, limit=limit, sort_by=sort_by)
            _attach_long_context_breakdown(nb, entries)
            stability = _compute_cross_run_stability(
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
            _annotate_qkv_usage(entries, analytics)
            annotate_display_names(entries)

            # Summary counts
            tier_counts = _count_discovery_tiers(nb)

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


