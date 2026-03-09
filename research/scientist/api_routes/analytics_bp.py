"""analytics API route registration."""
from __future__ import annotations

import importlib.util
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from flask import jsonify, request
from ..notebook import LabNotebook
from ..persona import get_aria
from ..shared_utils import safe_float as _to_safe_float
from ._helpers import deduplicate_insights, native_runner_canary_status_payload
from ._strategy import compute_compression_opportunities
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def register_analytics_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path

    @app.route("/api/trends")
    def api_trends():
        """Cross-experiment trend data for charts."""
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_experiment_trends())
        except Exception as e:
            logger.error(f"Error in /api/trends: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/trends/context")
    def api_trends_context():
        """Trend data plus adaptation-event deltas for inline linkage UI."""
        nb = LabNotebook(notebook_path)

        def _event_delta_payload(trends: List[Dict[str, Any]], event: Dict[str, Any]) -> Dict[str, Any]:
            timestamp = float(event.get("timestamp") or 0.0)
            previous = [row for row in trends if float(row.get("timestamp") or 0.0) < timestamp]
            following = [row for row in trends if float(row.get("timestamp") or 0.0) >= timestamp]

            before = previous[-3:]
            after = following[:3]

            before_ids = [str(row.get("experiment_id")) for row in before if row.get("experiment_id")]
            after_ids = [str(row.get("experiment_id")) for row in after if row.get("experiment_id")]

            def _avg(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
                values = [float(row[key]) for row in rows if row.get(key) is not None]
                if not values:
                    return None
                return sum(values) / len(values)

            before_adj_s1 = _avg(before, "adjusted_s1_pass_rate")
            after_adj_s1 = _avg(after, "adjusted_s1_pass_rate")
            before_novelty = _avg(before, "best_novelty_score")
            after_novelty = _avg(after, "best_novelty_score")
            before_loss = _avg(before, "best_loss_ratio")
            after_loss = _avg(after, "best_loss_ratio")

            return {
                "timestamp": timestamp,
                "event_type": event.get("event_type"),
                "description": event.get("description") or "Grammar weights adjusted",
                "before_window": {
                    "n_experiments": len(before),
                    "experiment_ids": before_ids,
                    "adjusted_s1_rate": before_adj_s1,
                    "best_novelty": before_novelty,
                    "best_loss_ratio": before_loss,
                },
                "after_window": {
                    "n_experiments": len(after),
                    "experiment_ids": after_ids,
                    "adjusted_s1_rate": after_adj_s1,
                    "best_novelty": after_novelty,
                    "best_loss_ratio": after_loss,
                },
                "delta": {
                    "adjusted_s1_rate": (
                        after_adj_s1 - before_adj_s1
                        if after_adj_s1 is not None and before_adj_s1 is not None
                        else None
                    ),
                    "best_novelty": (
                        after_novelty - before_novelty
                        if after_novelty is not None and before_novelty is not None
                        else None
                    ),
                    "best_loss_ratio": (
                        after_loss - before_loss
                        if after_loss is not None and before_loss is not None
                        else None
                    ),
                },
            }

        try:
            trends = nb.get_experiment_trends()
            learning_log = nb.get_learning_log(limit=300)
            adaptation_events = [
                _event_delta_payload(trends, event)
                for event in learning_log
                if event.get("event_type") == "grammar_weights_applied"
            ]
            return jsonify({
                "trends": trends,
                "adaptation_events": adaptation_events,
                "generated_at": time.time(),
            })
        except Exception as e:
            logger.error(f"Error in /api/trends/context: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/insights")
    def api_insights():
        """List active insights, deduplicated by content (keeps latest)."""
        category = request.args.get("category")
        nb = LabNotebook(notebook_path)
        try:
            raw = nb.get_insights(category=category, limit=200)
            return jsonify(deduplicate_insights(raw))
        except Exception as e:
            logger.error(f"Error in /api/insights: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/insights/boost", methods=["POST"])
    def api_insights_boost():
        """Record a request to boost an insight in future experiment selection."""
        payload = request.get_json(silent=True) or {}
        insight_id = str(payload.get("insight_id") or "").strip()
        content = str(payload.get("content") or "").strip()
        category = str(payload.get("category") or "").strip()
        confidence = payload.get("confidence")
        if not insight_id:
            return jsonify({"error": "insight_id required"}), 400
        nb = LabNotebook(notebook_path)
        try:
            evidence = json.dumps({
                "insight_id": insight_id,
                "category": category or None,
                "confidence": confidence,
                "content": content[:400] if content else None,
            }, sort_keys=True)
            desc = f"Boost requested for insight {insight_id}"
            if category:
                desc += f" ({category})"
            nb.log_learning_event(
                "insight_boost",
                desc,
                evidence=evidence,
            )
            return jsonify({"status": "ok", "insight_id": insight_id})
        except Exception as e:
            logger.error(f"Error in /api/insights/boost: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/op-success")
    def api_op_success():
        """Op success rate table."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.op_success_rates())
        except Exception as e:
            logger.error(f"Error in op-success: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/recommendation-signals")
    def api_recommendation_signals():
        """Aggregate compact, data-driven recommendation signals for Designer."""
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics

            analytics = ExperimentAnalytics(nb)
            op_rates = analytics.op_success_rates() or {}
            op_priors = []
            for op_name, stats in op_rates.items():
                n_used = int(stats.get("n_used") or 0)
                if n_used < 5:
                    continue
                op_priors.append({
                    "op_name": op_name,
                    "s1_rate": round(_to_safe_float(stats.get("s1_rate")), 6),
                    "n_used": n_used,
                })
            op_priors.sort(key=lambda r: (r.get("s1_rate", 0.0), r.get("n_used", 0)), reverse=True)

            toxic_pairs = nb.get_failure_signature_blocklist(min_seen=5, max_fail_rate=0.85)
            toxic_op_names: set = set()
            toxic_signatures = []
            for signature, penalty in toxic_pairs.items():
                toks = [tok.strip() for tok in str(signature).split("->") if tok.strip()]
                toxic_op_names.update(toks)
                toxic_signatures.append({
                    "signature": signature,
                    "penalty": round(_to_safe_float(penalty, 1.0), 6),
                })
            toxic_signatures.sort(key=lambda r: r.get("penalty", 1.0))

            compression_coverage = analytics.compression_coverage() or {}
            comp_opps = compute_compression_opportunities(compression_coverage)
            top_techniques = (comp_opps or {}).get("top_techniques") or []
            compression_techniques = [
                str(item.get("technique") or "").strip()
                for item in top_techniques
                if str(item.get("technique") or "").strip()
            ]

            interactions_raw = nb.get_selection_insight_interactions(limit=120)
            interactions = []
            for row in interactions_raw:
                n_trials = int(row.get("n_trials") or 0)
                if n_trials < 2:
                    continue
                interactions.append({
                    "insight_a": row.get("insight_a"),
                    "insight_b": row.get("insight_b"),
                    "mean_reward": round(_to_safe_float(row.get("mean_reward"), 0.0), 6),
                    "n_trials": n_trials,
                    "n_supported": int(row.get("n_supported") or 0),
                    "n_not_supported": int(row.get("n_not_supported") or 0),
                })
            interactions.sort(
                key=lambda r: (abs(r.get("mean_reward", 0.0) - 0.5), r.get("n_trials", 0)),
                reverse=True,
            )

            insights = deduplicate_insights(nb.get_insights(limit=120))
            compressed_insights = [
                {
                    "insight_id": row.get("insight_id"),
                    "category": row.get("category"),
                    "insight_type": row.get("insight_type"),
                    "subject_key": row.get("subject_key"),
                    "semantic_key": row.get("semantic_key"),
                    "content": row.get("content"),
                    "confidence": round(_to_safe_float(row.get("confidence"), 0.0), 6),
                }
                for row in insights
            ]

            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "research.analytics",
                "summary": nb.get_dashboard_summary(),
                "op_priors": op_priors[:80],
                "toxic_signatures": toxic_signatures[:80],
                "toxic_ops": sorted(toxic_op_names),
                "compression_opportunities": comp_opps,
                "compression_techniques": compression_techniques[:20],
                "insights": compressed_insights[:80],
                "insight_interactions": interactions[:60],
                "native_runner": native_runner_canary_status_payload(force_refresh=False),
                "aria_core": {
                    "available": bool(importlib.util.find_spec("aria_core")),
                    "proactive_gating_in_research_runner": True,
                },
            }
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in recommendation-signals: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/failure-patterns")
    def api_failure_patterns():
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.failure_patterns())
        except Exception as e:
            logger.error(f"Error in failure-patterns: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/grammar-weights")
    def api_grammar_weights():
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            defaults = analytics.get_current_grammar_weights()
            learned = analytics.compute_grammar_weights()
            control_comparison = analytics.control_experiment_comparison()
            holdout = analytics.holdout_validation()
            explanation = aria.explain_grammar_weights(defaults, learned)
            diagnostics = analytics.grammar_weight_learning_diagnostics()
            return jsonify({
                "default": defaults,
                "learned": learned,
                "control_comparison": control_comparison,
                "holdout_validation": holdout,
                "learning_diagnostics": diagnostics,
                "architecture_rerun_telemetry": {
                    "unique_fingerprint_count": int(diagnostics.get("unique_fingerprints") or 0),
                    "total_result_rows": int(diagnostics.get("total_rows") or 0),
                    "repeat_result_rows": int(diagnostics.get("repeat_rows") or 0),
                    "rerun_ratio": float(diagnostics.get("rerun_ratio") or 0.0),
                    "top_fingerprint_concentration": float(diagnostics.get("top_fingerprint_concentration") or 0.0),
                    "weighting_mode": str(diagnostics.get("mode") or "unknown"),
                },
                "explanation": explanation,
            })
        except Exception as e:
            logger.error(f"Error in grammar-weights: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/efficiency-frontier")
    def api_efficiency_frontier():
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.efficiency_frontier())
        except Exception as e:
            logger.error(f"Error in efficiency-frontier: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/efficiency-frontier-3d")
    def api_efficiency_frontier_3d():
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.efficiency_frontier_3d())
        except Exception as e:
            logger.error(f"Error in efficiency-frontier-3d: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/regression-vs-baseline")
    def api_regression_vs_baseline():
        limit = request.args.get("limit", 200, type=int)
        nb = LabNotebook(notebook_path)
        try:
            rows = nb.conn.execute(
                """
                SELECT result_id, experiment_id, timestamp, loss_ratio,
                       baseline_loss_ratio, throughput_tok_s, flops_per_token, novelty_score
                FROM program_results
                WHERE stage1_passed = 1
                  AND baseline_loss_ratio IS NOT NULL
                  AND throughput_tok_s IS NOT NULL
                  AND throughput_tok_s > 0
                ORDER BY timestamp DESC LIMIT ?
                """,
                (max(20, int(limit)),),
            ).fetchall()
            points = []
            for row in rows:
                item = dict(row)
                item["baseline_beats_reference"] = float(item.get("baseline_loss_ratio") or 0.0) < 1.0
                points.append(item)
            frontier = []
            best_ratio = float("inf")
            for item in sorted(points, key=lambda p: float(p.get("throughput_tok_s") or 0.0), reverse=True):
                ratio = float(item.get("baseline_loss_ratio") or float("inf"))
                if ratio <= best_ratio:
                    frontier.append(item)
                    best_ratio = ratio
            summary = {
                "n_points": len(points),
                "n_beating_baseline": sum(1 for p in points if p["baseline_beats_reference"]),
                "best_baseline_ratio": min(
                    (float(p.get("baseline_loss_ratio") or float("inf")) for p in points), default=None),
                "best_throughput_tok_s": max(
                    (float(p.get("throughput_tok_s") or 0.0) for p in points), default=0.0),
                "frontier_count": len(frontier),
            }
            return jsonify({"points": points, "pareto_frontier": frontier, "summary": summary})
        except Exception as e:
            logger.error(f"Error in regression-vs-baseline: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/experiment-clusters")
    def api_experiment_clusters():
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.experiment_clusters())
        except Exception as e:
            logger.error(f"Error in experiment-clusters: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/routing-health")
    def api_routing_health():
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            payload = analytics.routing_health() or {}
            payload.setdefault("available", False)
            payload.setdefault("by_mode", [])
            payload.setdefault("explanation", "Routing telemetry is unavailable.")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in routing-health: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/routing-comparison")
    def api_routing_comparison():
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            payload = analytics.routing_mode_comparison() or {}
            payload.setdefault("available", False)
            payload.setdefault("by_mode", [])
            payload.setdefault("n_modes", 0)
            payload.setdefault("total_programs", 0)
            payload.setdefault("routed_programs", 0)
            payload.setdefault("uniform_programs", 0)
            payload.setdefault("explanation", "Routing comparison data is unavailable.")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in routing-comparison: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/gating-diagnostics")
    def api_gating_diagnostics():
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            payload = analytics.gating_behavior_diagnostics() or {}
            payload.setdefault("available", False)
            payload.setdefault("total_routed_programs", 0)
            payload.setdefault("avg_gate_entropy", None)
            payload.setdefault("collapse_risk_counts", {"low": 0, "medium": 0, "high": 0, "unknown": 0})
            payload.setdefault("by_mode", [])
            payload.setdefault("token_retention_curve_overall", [])
            payload.setdefault("explanation", "Gating diagnostics are unavailable.")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in gating-diagnostics: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/gate-health")
    def api_gate_health():
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            n_days = request.args.get("days", 14, type=int)
            return jsonify(analytics.gate_health_daily(n_days=n_days))
        except Exception as e:
            logger.error(f"Error in gate-health: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/math-family-coverage")
    def api_math_family_coverage():
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.math_family_coverage())
        except Exception as e:
            logger.error(f"Error in math-family-coverage: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/mathspace-impact")
    def api_mathspace_impact():
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            payload = analytics.mathspace_operator_impact() or {}
            payload.setdefault("available", False)
            payload.setdefault("totals", {
                "n_programs_with_graph": 0,
                "n_programs_with_mathspace": 0,
                "n_mathspace_ops_observed": 0,
            })
            payload.setdefault("by_operator", [])
            payload.setdefault("by_family", [])
            payload.setdefault("top_trustworthy_operators", [])
            payload.setdefault("explanation", "Math-space impact data is unavailable.")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in mathspace-impact: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/compression-coverage")
    def api_compression_coverage():
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.compression_coverage())
        except Exception as e:
            logger.error(f"Error in compression-coverage: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/compression-opportunities")
    def api_compression_opportunities():
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            coverage = analytics.compression_coverage() or {}
            return jsonify(compute_compression_opportunities(coverage))
        except Exception as e:
            logger.error(f"Error in compression-opportunities: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/negative-results")
    def api_negative_results():
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.negative_results_synthesis())
        except Exception as e:
            logger.error(f"Error in negative-results: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/learning-trajectory")
    def api_learning_trajectory():
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.learning_trajectory())
        except Exception as e:
            logger.error(f"Error in learning-trajectory: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/strategy-backtest")
    def api_strategy_backtest():
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return jsonify(analytics.strategy_backtest())
        except Exception as e:
            logger.error(f"Error in strategy-backtest: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/control-comparison")
    def api_control_comparison():
        nb = LabNotebook(notebook_path)
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            result = analytics.control_experiment_comparison()
            if result is None:
                return jsonify({"status": "insufficient_data",
                                "message": "Need at least 2 control and 2 learned experiments"})
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error in control-comparison: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/learning-summary")
    def api_learning_summary():
        nb = LabNotebook(notebook_path)
        aria = get_aria()
        try:
            from ..analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            payload = aria.summarize_learning_bullets({
                "summary": nb.get_dashboard_summary(),
                "grammar_default": analytics.get_current_grammar_weights(),
                "grammar_learned": analytics.compute_grammar_weights(),
                "frontier": analytics.efficiency_frontier(),
                "clusters": analytics.experiment_clusters(),
                "recent_experiments": nb.get_recent_experiments(10),
                "trajectory": analytics.learning_trajectory(),
            })
            payload.setdefault("bullets", [])
            payload.setdefault("source", "rule-based")
            return jsonify(payload)
        except Exception as e:
            logger.error(f"Error in learning-summary: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/learning-log")
    def api_learning_log():
        n = request.args.get("n", 100, type=int)
        nb = LabNotebook(notebook_path)
        try:
            return jsonify(nb.get_learning_log(limit=n))
        except Exception as e:
            logger.error(f"Error in learning-log: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()

    @app.route("/api/analytics/insight-interactions")
    def api_insight_interactions():
        nb = LabNotebook(notebook_path)
        try:
            limit = request.args.get("limit", 80, type=int)
            min_trials = request.args.get("min_trials", 1, type=int)
            rows = nb.get_selection_insight_interactions(limit=max(1, min(limit, 500)))
            rows = [
                row for row in rows
                if int(row.get("n_trials") or 0) >= max(1, int(min_trials))
            ]

            insight_rows = nb.get_insights(limit=500)
            insight_by_id = {
                str(row.get("insight_id")): row for row in insight_rows if row.get("insight_id")
            }

            enriched: List[Dict[str, Any]] = []
            for row in rows:
                a_id = str(row.get("insight_a") or "")
                b_id = str(row.get("insight_b") or "")
                a = insight_by_id.get(a_id, {})
                b = insight_by_id.get(b_id, {})
                mean_reward = _to_safe_float(row.get("mean_reward"), 0.0)
                n_trials = int(row.get("n_trials") or 0)
                supported = int(row.get("n_supported") or 0)
                not_supported = int(row.get("n_not_supported") or 0)
                support_rate = (supported / n_trials) if n_trials > 0 else 0.0
                label = "synergistic" if mean_reward >= 0.55 else ("antagonistic" if mean_reward <= 0.45 else "mixed")
                confidence = "high" if n_trials >= 8 else ("medium" if n_trials >= 4 else "low")
                enriched.append({
                    **row,
                    "support_rate": round(support_rate, 6),
                    "interaction_label": label,
                    "confidence_label": confidence,
                    "insight_a_content": a.get("content"),
                    "insight_b_content": b.get("content"),
                    "insight_a_category": a.get("category"),
                    "insight_b_category": b.get("category"),
                    "is_singleton": a_id == b_id,
                })

            synergistic = [
                row for row in enriched
                if not row.get("is_singleton") and row.get("interaction_label") == "synergistic"
            ][:10]
            antagonistic = [
                row for row in enriched
                if not row.get("is_singleton") and row.get("interaction_label") == "antagonistic"
            ][:10]
            singleton = [
                row for row in enriched
                if row.get("is_singleton")
            ][:10]
            return jsonify({
                "available": len(enriched) > 0,
                "total_interactions": len(enriched),
                "synergistic_pairs": synergistic,
                "antagonistic_pairs": antagonistic,
                "singleton_insights": singleton,
                "interactions": enriched,
                "explanation": (
                    "Interaction score is learned from downstream outcomes of selection decisions "
                    "(supported/not_supported with reward aggregation)."
                ),
            })
        except Exception as e:
            logger.error(f"Error in insight-interactions: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            nb.close()
