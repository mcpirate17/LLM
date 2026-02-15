"""
Experiment Analytics — Learning Feedback Engine

Analyzes experiment history to learn which operations, structures, and
combinations correlate with success. Feeds back into grammar weights
to improve synthesis over time.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..synthesis.grammar import GrammarConfig
from ..synthesis.primitives import get_primitive
from .notebook import LabNotebook


class ExperimentAnalytics:
    """Data-driven analytics over experiment history."""

    def __init__(self, notebook: LabNotebook):
        self.nb = notebook

    def op_success_rates(self) -> Dict[str, Dict]:
        """Get per-op success rates from the op_success_rates table."""
        rows = self.nb.get_op_success_rates()
        result = {}
        for row in rows:
            op = row["op_name"]
            n_used = row["n_used"] or 1
            result[op] = {
                "n_used": n_used,
                "s0_rate": (row.get("n_stage0_passed") or 0) / n_used,
                "s05_rate": (row.get("n_stage05_passed") or 0) / n_used,
                "s1_rate": (row.get("n_stage1_passed") or 0) / n_used,
                "avg_loss_ratio": row.get("avg_loss_ratio"),
                "avg_novelty": row.get("avg_novelty"),
            }
        return result

    def structural_correlations(self) -> Dict[str, float]:
        """Analyze which graph properties correlate with Stage 1 success.

        Returns correlation-like scores for graph metrics vs success.
        Single-pass accumulation for efficiency.
        """
        rows = self.nb.conn.execute("""
            SELECT stage1_passed, graph_n_ops, graph_depth,
                   graph_n_params_estimate, graph_n_unique_ops,
                   graph_uses_math_spaces, graph_uses_frequency_domain,
                   graph_has_gradient_path
            FROM program_results
            WHERE graph_n_ops IS NOT NULL
        """).fetchall()

        if len(rows) < 10:
            return {}

        metrics = ["graph_n_ops", "graph_depth", "graph_n_params_estimate",
                    "graph_n_unique_ops", "graph_uses_math_spaces",
                    "graph_uses_frequency_domain", "graph_has_gradient_path"]

        # Single-pass: accumulate sums/counts per metric per group
        acc = {m: {"s_sum": 0.0, "s_n": 0, "f_sum": 0.0, "f_n": 0,
                    "all_sum": 0.0, "all_sq": 0.0, "all_n": 0}
               for m in metrics}

        for r in rows:
            passed = r["stage1_passed"]
            for m in metrics:
                val = r[m]
                if val is None:
                    continue
                v = float(val)
                a = acc[m]
                a["all_sum"] += v
                a["all_sq"] += v * v
                a["all_n"] += 1
                if passed:
                    a["s_sum"] += v
                    a["s_n"] += 1
                else:
                    a["f_sum"] += v
                    a["f_n"] += 1

        correlations = {}
        for m in metrics:
            a = acc[m]
            if a["s_n"] == 0 or a["f_n"] == 0 or a["all_n"] == 0:
                continue
            avg_success = a["s_sum"] / a["s_n"]
            avg_fail = a["f_sum"] / a["f_n"]
            mean = a["all_sum"] / a["all_n"]
            variance = a["all_sq"] / a["all_n"] - mean * mean
            std = variance ** 0.5 if variance > 0 else 0.0
            if std > 0:
                correlations[m] = (avg_success - avg_fail) / std
            else:
                correlations[m] = 0.0

        return correlations

    def compute_grammar_weights(self) -> Optional[Dict[str, float]]:
        """Compute learned category weights from historical success data.

        Returns a dict of category -> weight, or None if insufficient data.
        """
        op_rates = self.op_success_rates()
        if len(op_rates) < 5:
            return None

        # Group by category
        cat_stats: Dict[str, Dict] = defaultdict(lambda: {
            "total": 0, "s1_total": 0, "novelty_sum": 0.0, "count": 0,
        })

        for op_name, stats in op_rates.items():
            try:
                op = get_primitive(op_name)
                cat = op.category.value
            except (KeyError, Exception):
                continue

            cat_stats[cat]["total"] += stats["n_used"]
            cat_stats[cat]["s1_total"] += int(stats["s1_rate"] * stats["n_used"])
            if stats.get("avg_novelty"):
                cat_stats[cat]["novelty_sum"] += stats["avg_novelty"] * stats["n_used"]
                cat_stats[cat]["count"] += stats["n_used"]

        if not cat_stats:
            return None

        # Get default weights to preserve designer intent as base
        default_weights = GrammarConfig().category_weights

        # Compute per-category s1 rates, then compare to mean
        cat_s1_rates = {}
        cat_novelties = {}
        for cat, stats in cat_stats.items():
            if stats["total"] < 2:
                continue  # not enough data
            cat_s1_rates[cat] = stats["s1_total"] / max(stats["total"], 1)
            cat_novelties[cat] = (stats["novelty_sum"] / stats["count"]
                                  if stats["count"] > 0 else 0.0)

        if not cat_s1_rates:
            return None

        mean_s1 = sum(cat_s1_rates.values()) / len(cat_s1_rates)

        # Multiplicative formula with contrast amplification
        weights = {}
        for cat, s1_rate in cat_s1_rates.items():
            # Relative performance vs mean (>1 = above avg, <1 = below)
            relative = s1_rate / max(mean_s1, 0.01)
            # Square to amplify differences (2x better → 4x weight boost)
            amplified = relative ** 2
            # Novelty bonus: multiplicative, range 1.0–2.0
            novelty_factor = 1.0 + cat_novelties.get(cat, 0.0)
            # Apply to default weight as base
            base = default_weights.get(cat, 1.0)
            weight = base * amplified * novelty_factor
            weights[cat] = round(max(0.1, min(8.0, weight)), 2)

        return weights if weights else None

    def failure_patterns(self) -> Dict[str, Dict]:
        """Analyze common failure modes by error type and stage."""
        rows = self.nb.conn.execute("""
            SELECT error_type, stage_at_death, COUNT(*) as count
            FROM program_results
            WHERE error_type IS NOT NULL
            GROUP BY error_type, stage_at_death
            ORDER BY count DESC
        """).fetchall()

        patterns: Dict[str, Dict] = {}
        for r in rows:
            error_type = r["error_type"] or "unknown"
            if error_type not in patterns:
                patterns[error_type] = {"total": 0, "by_stage": {}}
            patterns[error_type]["total"] += r["count"]
            stage = r["stage_at_death"] or "unknown"
            patterns[error_type]["by_stage"][stage] = r["count"]

        return patterns

    def top_op_combinations(self, n: int = 10) -> List[Dict]:
        """Find op combinations that co-occur in Stage 1 survivors."""
        rows = self.nb.conn.execute("""
            SELECT graph_json, novelty_score, loss_ratio
            FROM program_results
            WHERE stage1_passed = 1 AND graph_json IS NOT NULL
            ORDER BY novelty_score DESC NULLS LAST
            LIMIT 200
        """).fetchall()

        # Count op pair co-occurrences
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_novelty: Dict[Tuple[str, str], List[float]] = defaultdict(list)

        for r in rows:
            try:
                graph_data = json.loads(r["graph_json"])
                nodes = graph_data.get("nodes", {})
                ops = sorted(set(
                    nd["op_name"] for nd in nodes.values()
                    if nd.get("op_name") and nd["op_name"] != "input"
                ))
            except (json.JSONDecodeError, TypeError):
                continue

            for i in range(len(ops)):
                for j in range(i + 1, len(ops)):
                    pair = (ops[i], ops[j])
                    pair_counts[pair] += 1
                    if r["novelty_score"]:
                        pair_novelty[pair].append(r["novelty_score"])

        # Sort by frequency
        top_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:n]
        results = []
        for (op_a, op_b), count in top_pairs:
            novelties = pair_novelty.get((op_a, op_b), [])
            results.append({
                "ops": [op_a, op_b],
                "count": count,
                "avg_novelty": sum(novelties) / len(novelties) if novelties else 0,
            })

        return results

    def compute_insights(self) -> List[str]:
        """Generate data-driven insights from experiment history.

        Replaces the 4 hardcoded rules with actual data analysis.
        """
        insights = []

        # 1. Op success rate insights
        op_rates = self.op_success_rates()
        if op_rates:
            # Find best and worst ops
            rated_ops = [(op, s["s1_rate"], s["n_used"])
                         for op, s in op_rates.items() if s["n_used"] >= 5]
            if rated_ops:
                rated_ops.sort(key=lambda x: -x[1])
                best_ops = rated_ops[:3]
                worst_ops = rated_ops[-3:]

                if best_ops[0][1] > 0:
                    op_names = ", ".join(f"{op}({rate:.0%})" for op, rate, _ in best_ops)
                    insights.append(
                        f"Top-performing ops (S1 rate): {op_names}. "
                        f"These compose well into learnable architectures."
                    )

                if worst_ops and worst_ops[-1][1] == 0 and worst_ops[-1][2] >= 10:
                    op_names = ", ".join(op for op, _, _ in worst_ops if _ == 0)
                    if op_names:
                        insights.append(
                            f"Consistently failing ops: {op_names}. "
                            f"Consider reducing their grammar weight."
                        )

        # 2. Structural correlation insights
        correlations = self.structural_correlations()
        if correlations:
            for metric, effect in sorted(correlations.items(),
                                         key=lambda x: -abs(x[1])):
                if abs(effect) > 0.5:
                    direction = "positively" if effect > 0 else "negatively"
                    name = metric.replace("graph_", "").replace("_", " ")
                    insights.append(
                        f"Graph {name} is {direction} correlated with "
                        f"Stage 1 success (effect={effect:.2f})."
                    )
                    break  # just the strongest

        # 3. Failure pattern insights
        failures = self.failure_patterns()
        if failures:
            top_failure = max(failures.items(), key=lambda x: x[1]["total"])
            if top_failure[1]["total"] >= 10:
                insights.append(
                    f"Most common failure: {top_failure[0]} "
                    f"({top_failure[1]['total']} occurrences). "
                    f"Stages: {top_failure[1]['by_stage']}"
                )

        # 4. Op combination insights
        combos = self.top_op_combinations(5)
        if combos and combos[0]["count"] >= 3:
            top = combos[0]
            insights.append(
                f"Winning combination: {' + '.join(top['ops'])} "
                f"appears in {top['count']} survivors "
                f"(avg novelty {top['avg_novelty']:.3f})."
            )

        # 5. Overall progress insight
        summary = self.nb.get_dashboard_summary()
        total = summary.get("total_programs_evaluated", 0)
        survivors = summary.get("stage1_survivors", 0)
        if total > 0:
            rate = survivors / total
            insights.append(
                f"Overall survival rate: {rate:.1%} "
                f"({survivors}/{total} programs). "
                f"{'Grammar is productive.' if rate > 0.03 else 'Grammar needs tuning.'}"
            )

        return insights

    def efficiency_frontier(self) -> List[Dict]:
        """Find Pareto-optimal programs on loss vs FLOPs/params.

        Returns programs that are not dominated by any other program
        (lower loss AND lower FLOPs simultaneously).
        """
        rows = self.nb.conn.execute("""
            SELECT result_id, graph_fingerprint, final_loss,
                   flops_forward, param_count, novelty_score,
                   loss_ratio, baseline_loss_ratio
            FROM program_results
            WHERE stage1_passed = 1
              AND final_loss IS NOT NULL
              AND flops_forward IS NOT NULL
              AND flops_forward > 0
            ORDER BY final_loss ASC
        """).fetchall()

        if not rows:
            return []

        programs = [dict(r) for r in rows]

        # Find Pareto frontier: not dominated in (loss, flops)
        frontier = []
        for p in programs:
            dominated = False
            for q in programs:
                if (q["final_loss"] <= p["final_loss"]
                        and q["flops_forward"] <= p["flops_forward"]
                        and (q["final_loss"] < p["final_loss"]
                             or q["flops_forward"] < p["flops_forward"])):
                    dominated = True
                    break
            if not dominated:
                frontier.append(p)

        return frontier

    def get_current_grammar_weights(self) -> Dict[str, float]:
        """Get the default grammar weights for comparison."""
        return dict(GrammarConfig().category_weights)
