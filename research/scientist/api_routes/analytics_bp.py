"""analytics API route registration."""

from __future__ import annotations

import importlib.util
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from flask import jsonify, request
from ..persona import get_aria
from ..shared_utils import safe_float as _to_safe_float
from ._helpers import deduplicate_insights, native_runner_canary_status_payload
from ._strategy_recommendations import compute_compression_opportunities
from ._utils import with_notebook_context
from .deps import ApiRouteContext

logger = logging.getLogger(__name__)


def register_analytics_routes(app, context: ApiRouteContext):
    notebook_path = context.notebook_path
    wnb = with_notebook_context(notebook_path)

    @app.route("/api/trends")
    @wnb
    def api_trends(nb=None):
        """Cross-experiment trend data for charts."""
        return jsonify(nb.get_experiment_trends())

    @app.route("/api/trends/context")
    @wnb
    def api_trends_context(nb=None):
        """Trend data plus adaptation-event deltas for inline linkage UI."""

        def _event_delta_payload(
            trends: List[Dict[str, Any]], event: Dict[str, Any]
        ) -> Dict[str, Any]:
            timestamp = float(event.get("timestamp") or 0.0)
            previous = [
                row for row in trends if float(row.get("timestamp") or 0.0) < timestamp
            ]
            following = [
                row for row in trends if float(row.get("timestamp") or 0.0) >= timestamp
            ]

            before = previous[-3:]
            after = following[:3]

            before_ids = [
                str(row.get("experiment_id"))
                for row in before
                if row.get("experiment_id")
            ]
            after_ids = [
                str(row.get("experiment_id"))
                for row in after
                if row.get("experiment_id")
            ]

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

        trends = nb.get_experiment_trends()
        learning_log = nb.get_learning_log(limit=300)
        adaptation_events = [
            _event_delta_payload(trends, event)
            for event in learning_log
            if event.get("event_type") == "grammar_weights_applied"
        ]
        return jsonify(
            {
                "trends": trends,
                "adaptation_events": adaptation_events,
                "generated_at": time.time(),
            }
        )

    @app.route("/api/insights")
    @wnb
    def api_insights(nb=None):
        """List active insights, deduplicated by content (keeps latest)."""
        category = request.args.get("category")
        raw = nb.get_insights(category=category, limit=200)
        return jsonify(deduplicate_insights(raw))

    @app.route("/api/insights/boost", methods=["POST"])
    @wnb
    def api_insights_boost(nb=None):
        """Record a request to boost an insight in future experiment selection."""
        payload = request.get_json(silent=True) or {}
        insight_id = str(payload.get("insight_id") or "").strip()
        content = str(payload.get("content") or "").strip()
        category = str(payload.get("category") or "").strip()
        confidence = payload.get("confidence")
        if not insight_id:
            return jsonify({"error": "insight_id required"}), 400
        evidence = json.dumps(
            {
                "insight_id": insight_id,
                "category": category or None,
                "confidence": confidence,
                "content": content[:400] if content else None,
            },
            sort_keys=True,
        )
        desc = f"Boost requested for insight {insight_id}"
        if category:
            desc += f" ({category})"
        nb.log_learning_event(
            "insight_boost",
            desc,
            evidence=evidence,
        )
        return jsonify({"status": "ok", "insight_id": insight_id})

    @app.route("/api/analytics/op-success")
    @wnb
    def api_op_success(nb=None):
        """Op success rate table."""
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        return jsonify(analytics.op_success_rates())

    @app.route("/api/analytics/recommendation-signals")
    @wnb
    def api_recommendation_signals(nb=None):
        """Aggregate compact, data-driven recommendation signals for Designer."""
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        op_rates = analytics.op_success_rates() or {}
        op_priors = []
        for op_name, stats in op_rates.items():
            n_used = int(stats.get("n_used") or 0)
            if n_used < 5:
                continue
            op_priors.append(
                {
                    "op_name": op_name,
                    "s1_rate": round(_to_safe_float(stats.get("s1_rate")), 6),
                    "n_used": n_used,
                }
            )
        op_priors.sort(
            key=lambda r: (r.get("s1_rate", 0.0), r.get("n_used", 0)), reverse=True
        )

        toxic_pairs = nb.get_failure_signature_blocklist(min_seen=5, max_fail_rate=0.85)
        toxic_op_names: set = set()
        toxic_signatures = []
        for signature, penalty in toxic_pairs.items():
            toks = [tok.strip() for tok in str(signature).split("->") if tok.strip()]
            toxic_op_names.update(toks)
            toxic_signatures.append(
                {
                    "signature": signature,
                    "penalty": round(_to_safe_float(penalty, 1.0), 6),
                }
            )
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
            interactions.append(
                {
                    "insight_a": row.get("insight_a"),
                    "insight_b": row.get("insight_b"),
                    "mean_reward": round(
                        _to_safe_float(row.get("mean_reward"), 0.0), 6
                    ),
                    "n_trials": n_trials,
                    "n_supported": int(row.get("n_supported") or 0),
                    "n_not_supported": int(row.get("n_not_supported") or 0),
                }
            )
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

        op_pair_priors = nb.get_op_pair_priors(min_support=5, limit=100)
        fingerprint_buckets = nb.get_fingerprint_buckets(limit=5)
        lineage_successors = nb.get_lineage_successor_stats(limit=50)
        failure_risks = nb.get_failure_risk_signatures(limit=100)

        # 1.5 Expand recommendation-signals: top leaderboard + grammar weights
        leaderboard = nb.get_leaderboard(limit=5)
        top_entries = [
            {
                "result_id": entry.get("result_id"),
                "composite_score": round(float(entry.get("composite_score") or 0.0), 4),
                "tier": entry.get("tier"),
                "fingerprint": entry.get("graph_fingerprint"),
            }
            for entry in leaderboard
        ]

        op_weights = {}
        try:
            weights = analytics.compute_grammar_weights()
            if weights:
                op_weights = {k: round(float(v), 3) for k, v in weights.items()}
        except Exception as e:
            logger.warning("Could not compute grammar weights: %s", e)

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "research.analytics",
            "summary": nb.get_dashboard_summary(),
            "op_priors": op_priors[:80],
            "op_pair_priors": op_pair_priors,
            "fingerprint_buckets": fingerprint_buckets,
            "lineage_successors": lineage_successors,
            "top_entries": top_entries,
            "op_weights": op_weights,
            "toxic_signatures": toxic_signatures[:80],
            "toxic_ops": sorted(toxic_op_names),
            "failure_risk_signatures": failure_risks.get("failure_risk_signatures", []),
            "critical_failures": failure_risks.get("critical_failures", []),
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

    @app.route("/api/analytics/failure-patterns")
    @wnb
    def api_failure_patterns(nb=None):
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        return jsonify(analytics.failure_patterns())

    @app.route("/api/analytics/grammar-weights")
    @wnb
    def api_grammar_weights(nb=None):
        aria = get_aria()
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        defaults = analytics.get_current_grammar_weights()
        learned = analytics.compute_grammar_weights()
        control_comparison = analytics.control_experiment_comparison()
        holdout = analytics.holdout_validation()
        explanation = aria.explain_grammar_weights(defaults, learned)
        diagnostics = analytics.grammar_weight_learning_diagnostics()
        return jsonify(
            {
                "default": defaults,
                "learned": learned,
                "control_comparison": control_comparison,
                "holdout_validation": holdout,
                "learning_diagnostics": diagnostics,
                "architecture_rerun_telemetry": {
                    "unique_fingerprint_count": int(
                        diagnostics.get("unique_fingerprints") or 0
                    ),
                    "total_result_rows": int(diagnostics.get("total_rows") or 0),
                    "repeat_result_rows": int(diagnostics.get("repeat_rows") or 0),
                    "rerun_ratio": float(diagnostics.get("rerun_ratio") or 0.0),
                    "top_fingerprint_concentration": float(
                        diagnostics.get("top_fingerprint_concentration") or 0.0
                    ),
                    "weighting_mode": str(diagnostics.get("mode") or "unknown"),
                },
                "explanation": explanation,
            }
        )

    @app.route("/api/analytics/efficiency-frontier")
    @wnb
    def api_efficiency_frontier(nb=None):
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        return jsonify(analytics.efficiency_frontier())

    @app.route("/api/analytics/efficiency-frontier-3d")
    @wnb
    def api_efficiency_frontier_3d(nb=None):
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        return jsonify(analytics.efficiency_frontier_3d())

    @app.route("/api/analytics/regression-vs-baseline")
    @wnb
    def api_regression_vs_baseline(nb=None):
        limit = request.args.get("limit", 200, type=int)
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
            item["baseline_beats_reference"] = (
                float(item.get("baseline_loss_ratio") or 0.0) < 1.0
            )
            points.append(item)
        frontier = []
        best_ratio = float("inf")
        for item in sorted(
            points, key=lambda p: float(p.get("throughput_tok_s") or 0.0), reverse=True
        ):
            ratio = float(item.get("baseline_loss_ratio") or float("inf"))
            if ratio <= best_ratio:
                frontier.append(item)
                best_ratio = ratio
        summary = {
            "n_points": len(points),
            "n_beating_baseline": sum(
                1 for p in points if p["baseline_beats_reference"]
            ),
            "best_baseline_ratio": min(
                (float(p.get("baseline_loss_ratio") or float("inf")) for p in points),
                default=None,
            ),
            "best_throughput_tok_s": max(
                (float(p.get("throughput_tok_s") or 0.0) for p in points), default=0.0
            ),
            "frontier_count": len(frontier),
        }
        return jsonify(
            {"points": points, "pareto_frontier": frontier, "summary": summary}
        )

    @app.route("/api/analytics/experiment-clusters")
    @wnb
    def api_experiment_clusters(nb=None):
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        return jsonify(analytics.experiment_clusters())

    @app.route("/api/analytics/routing-health")
    @wnb
    def api_routing_health(nb=None):
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        payload = analytics.routing_health() or {}
        payload.setdefault("available", False)
        payload.setdefault("by_mode", [])
        payload.setdefault("explanation", "Routing telemetry is unavailable.")
        return jsonify(payload)

    @app.route("/api/analytics/routing-comparison")
    @wnb
    def api_routing_comparison(nb=None):
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

    @app.route("/api/analytics/gating-diagnostics")
    @wnb
    def api_gating_diagnostics(nb=None):
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        payload = analytics.gating_behavior_diagnostics() or {}
        payload.setdefault("available", False)
        payload.setdefault("total_routed_programs", 0)
        payload.setdefault("avg_gate_entropy", None)
        payload.setdefault(
            "collapse_risk_counts", {"low": 0, "medium": 0, "high": 0, "unknown": 0}
        )
        payload.setdefault("by_mode", [])
        payload.setdefault("token_retention_curve_overall", [])
        payload.setdefault("explanation", "Gating diagnostics are unavailable.")
        return jsonify(payload)

    @app.route("/api/analytics/gate-health")
    @wnb
    def api_gate_health(nb=None):
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        n_days = request.args.get("days", 14, type=int)
        return jsonify(analytics.gate_health_daily(n_days=n_days))

    @app.route("/api/analytics/math-family-coverage")
    @wnb
    def api_math_family_coverage(nb=None):
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        return jsonify(analytics.math_family_coverage())

    @app.route("/api/analytics/mathspace-impact")
    @wnb
    def api_mathspace_impact(nb=None):
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        payload = analytics.mathspace_operator_impact() or {}
        payload.setdefault("available", False)
        payload.setdefault(
            "totals",
            {
                "n_programs_with_graph": 0,
                "n_programs_with_mathspace": 0,
                "n_mathspace_ops_observed": 0,
            },
        )
        payload.setdefault("by_operator", [])
        payload.setdefault("by_family", [])
        payload.setdefault("top_trustworthy_operators", [])
        payload.setdefault("explanation", "Math-space impact data is unavailable.")
        return jsonify(payload)

    @app.route("/api/analytics/compression-coverage")
    @wnb
    def api_compression_coverage(nb=None):
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        return jsonify(analytics.compression_coverage())

    @app.route("/api/analytics/compression-opportunities")
    @wnb
    def api_compression_opportunities(nb=None):
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        coverage = analytics.compression_coverage() or {}
        return jsonify(compute_compression_opportunities(coverage))

    @app.route("/api/analytics/negative-results")
    @wnb
    def api_negative_results(nb=None):
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        return jsonify(analytics.negative_results_synthesis())

    @app.route("/api/analytics/learning-trajectory")
    @wnb
    def api_learning_trajectory(nb=None):
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        return jsonify(analytics.learning_trajectory())

    @app.route("/api/analytics/strategy-backtest")
    @wnb
    def api_strategy_backtest(nb=None):
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        return jsonify(analytics.strategy_backtest())

    @app.route("/api/analytics/control-comparison")
    @wnb
    def api_control_comparison(nb=None):
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        result = analytics.control_experiment_comparison()
        if result is None:
            return jsonify(
                {
                    "status": "insufficient_data",
                    "message": "Need at least 2 control and 2 learned experiments",
                }
            )
        return jsonify(result)

    @app.route("/api/analytics/learning-summary")
    @wnb
    def api_learning_summary(nb=None):
        aria = get_aria()
        from ..analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics(nb)
        payload = aria.summarize_learning_bullets(
            {
                "summary": nb.get_dashboard_summary(),
                "grammar_default": analytics.get_current_grammar_weights(),
                "grammar_learned": analytics.compute_grammar_weights(),
                "frontier": analytics.efficiency_frontier(),
                "clusters": analytics.experiment_clusters(),
                "recent_experiments": nb.get_recent_experiments(10),
                "trajectory": analytics.learning_trajectory(),
            }
        )
        payload.setdefault("bullets", [])
        payload.setdefault("source", "rule-based")
        return jsonify(payload)

    @app.route("/api/analytics/learning-log")
    @wnb
    def api_learning_log(nb=None):
        n = request.args.get("n", 100, type=int)
        return jsonify(nb.get_learning_log(limit=n))

    @app.route("/api/analytics/insight-interactions")
    @wnb
    def api_insight_interactions(nb=None):
        limit = request.args.get("limit", 80, type=int)
        min_trials = request.args.get("min_trials", 1, type=int)
        rows = nb.get_selection_insight_interactions(limit=max(1, min(limit, 500)))
        rows = [
            row
            for row in rows
            if int(row.get("n_trials") or 0) >= max(1, int(min_trials))
        ]

        insight_rows = nb.get_insights(limit=500)
        insight_by_id = {
            str(row.get("insight_id")): row
            for row in insight_rows
            if row.get("insight_id")
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
            int(row.get("n_not_supported") or 0)
            support_rate = (supported / n_trials) if n_trials > 0 else 0.0
            label = (
                "synergistic"
                if mean_reward >= 0.55
                else ("antagonistic" if mean_reward <= 0.45 else "mixed")
            )
            confidence = (
                "high" if n_trials >= 8 else ("medium" if n_trials >= 4 else "low")
            )
            enriched.append(
                {
                    **row,
                    "support_rate": round(support_rate, 6),
                    "interaction_label": label,
                    "confidence_label": confidence,
                    "insight_a_content": a.get("content"),
                    "insight_b_content": b.get("content"),
                    "insight_a_category": a.get("category"),
                    "insight_b_category": b.get("category"),
                    "is_singleton": a_id == b_id,
                }
            )

        synergistic = [
            row
            for row in enriched
            if not row.get("is_singleton")
            and row.get("interaction_label") == "synergistic"
        ][:10]
        antagonistic = [
            row
            for row in enriched
            if not row.get("is_singleton")
            and row.get("interaction_label") == "antagonistic"
        ][:10]
        singleton = [row for row in enriched if row.get("is_singleton")][:10]
        return jsonify(
            {
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
            }
        )
