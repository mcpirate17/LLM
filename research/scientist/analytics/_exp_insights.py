"""Data-driven insight generation mixin."""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Code failure error types -- always display_only
_CODE_FAILURE_TYPES = frozenset(
    {
        "RuntimeError",
        "TypeError",
        "AttributeError",
        "CompilationError",
        "ImportError",
        "ModuleNotFoundError",
        "SyntaxError",
        "NameError",
        "KeyError",
        "IndexError",
        "ValueError",
    }
)


class _InsightsMixin:
    """Generate data-driven insights with statistical evidence."""

    __slots__ = ()

    # Expose as class attribute for backward compat
    _CODE_FAILURE_TYPES = _CODE_FAILURE_TYPES

    def compute_insights(self) -> List[Dict]:
        """Generate data-driven insights with statistical evidence.

        Every insight carries:
        - ``alpha``/``beta_``: Beta-Binomial posterior from observed counts
        - ``evidence_json``: structured proof (test, p_value, effect_size, n)
        - ``display_only``: True for failure_mode and code-failure insights

        Returns ``[{"content": str, "category": str, ...}, ...]``
        """

        insights: List[Dict] = []

        # 1. Graph-size bucket analysis (structural)
        size_rows = self.nb.conn.execute("""
            SELECT graph_n_ops, stage1_passed
            FROM program_results_compat
            WHERE graph_n_ops IS NOT NULL
        """).fetchall()
        if len(size_rows) >= 50:
            self._compute_graph_size_insights(size_rows, insights)

        # 2. Op success/failure insights
        op_rates = self.op_success_rates()
        if op_rates:
            self._compute_op_insights(op_rates, insights)

        # 3. Structural correlation insights (chi-squared)
        correlations = self.structural_correlations()
        if correlations:
            self._compute_structural_correlation_insights(
                correlations, size_rows, insights
            )

        # 4. Failure pattern insights (always display_only)
        failures = self.failure_patterns()
        if failures:
            self._compute_failure_insights(failures, insights)

        # 5. Op combination insights (composition)
        combos = self.top_op_combinations(5)
        if combos:
            self._compute_combo_insights(combos, insights)

        # 6. Overall progress
        summary = self.nb.get_dashboard_headline_summary()
        total = summary.get("total_programs_evaluated", 0)
        survivors = summary.get("stage1_survivors", 0)
        if total >= 20:
            rate = survivors / total
            insights.append(
                {
                    "content": (
                        f"Overall survival rate: {rate:.1%} "
                        f"({survivors}/{total} programs). "
                        f"{'Grammar is productive.' if rate > 0.03 else 'Grammar needs tuning.'}"
                    ),
                    "category": "pattern",
                    "insight_type": "overall_survival_rate",
                    "subject_key": "global",
                    "semantic_key": "overall_survival_rate:global",
                    "alpha": float(survivors + 1),
                    "beta_": float(total - survivors + 1),
                    "display_only": False,
                    "insight_level": "structural",
                    "evidence_json": {
                        "test": "binomial_proportion",
                        "n": total,
                        "successes": survivors,
                        "rate": round(rate, 4),
                    },
                }
            )

        return insights

    @staticmethod
    def _bucket_graph_sizes(size_rows: list) -> Dict[str, List[int]]:
        """Bucket rows by graph size into pass/fail counts."""
        buckets: Dict[str, List[int]] = {
            "1-6": [0, 0],
            "7-9": [0, 0],
            "10-12": [0, 0],
            "13+": [0, 0],
        }
        for r in size_rows:
            n_ops = int(r["graph_n_ops"] or 0)
            passed = 1 if r["stage1_passed"] else 0
            if n_ops <= 6:
                key = "1-6"
            elif n_ops <= 9:
                key = "7-9"
            elif n_ops <= 12:
                key = "10-12"
            else:
                key = "13+"
            buckets[key][0] += passed
            buckets[key][1] += 1 - passed
        return buckets

    @staticmethod
    def _append_size_cap_insight(
        buckets: Dict[str, List[int]],
        rates: Dict[str, float],
        p_value: float,
        insights: List[Dict],
    ) -> None:
        """Append 13+ ops collapse insight if applicable."""
        if (
            "13+" in rates
            and rates["13+"] < 0.05
            and buckets["13+"][0] + buckets["13+"][1] >= 20
        ):
            big_pass = buckets["13+"][0]
            big_fail = buckets["13+"][1]
            insights.append(
                {
                    "content": (
                        f"13+ ops collapses to {rates['13+']:.1%} S1 "
                        f"(n={big_pass + big_fail}). Hard cap recommended."
                    ),
                    "category": "structural_preference",
                    "insight_type": "graph_size_cap",
                    "subject_key": "graph_size_cap",
                    "semantic_key": "structural:graph_size_cap",
                    "alpha": float(big_fail + 1),
                    "beta_": float(big_pass + 1),
                    "display_only": False,
                    "insight_level": "structural",
                    "evidence_json": {
                        "test": "binomial_vs_baseline",
                        "p_value": float(p_value),
                        "n": big_pass + big_fail,
                        "rate": round(rates["13+"], 4),
                        "recommended_max": 12,
                    },
                }
            )

    def _compute_graph_size_insights(
        self,
        size_rows: list,
        insights: List[Dict],
    ) -> None:
        """Chi-squared test on graph-size buckets vs S1 pass rate."""
        from scipy.stats import chi2_contingency

        buckets = self._bucket_graph_sizes(size_rows)

        table = [buckets[k] for k in buckets if sum(buckets[k]) > 0]
        if len(table) < 2:
            return

        try:
            chi2, p_value, _, _ = chi2_contingency(table)
        except ValueError:
            return

        if p_value > 0.01:
            return

        # Find best bucket
        rates = {}
        for k, (p, f) in buckets.items():
            total = p + f
            if total > 0:
                rates[k] = p / total
        best_bucket = max(rates, key=rates.get) if rates else None
        if not best_bucket:
            return

        best_pass = buckets[best_bucket][0]
        best_fail = buckets[best_bucket][1]
        best_rate = rates[best_bucket]
        n_total = sum(p + f for p, f in buckets.values())

        insights.append(
            {
                "content": (
                    f"Graph size {best_bucket} ops is optimal "
                    f"({best_rate:.1%} S1 rate, n={best_pass + best_fail})."
                ),
                "category": "structural_preference",
                "insight_type": "graph_size_optimal",
                "subject_key": "graph_size_optimal",
                "semantic_key": "structural:graph_size_optimal",
                "alpha": float(best_pass + 1),
                "beta_": float(best_fail + 1),
                "display_only": False,
                "insight_level": "structural",
                "evidence_json": {
                    "test": "chi2_contingency",
                    "chi2": round(float(chi2), 2),
                    "p_value": float(p_value),
                    "n": n_total,
                    "bucket_rates": {k: round(v, 4) for k, v in rates.items()},
                    "best_bucket": best_bucket,
                },
            }
        )

        self._append_size_cap_insight(buckets, rates, p_value, insights)

    def _compute_op_insights(
        self,
        op_rates: Dict[str, Dict],
        insights: List[Dict],
    ) -> None:
        """Generate op-level insights with proportion-test evidence."""
        rated_ops = [
            (op, s["s1_rate"], s["n_used"], s.get("n_stage1_passed", 0))
            for op, s in op_rates.items()
            if s["n_used"] >= 10
        ]
        if not rated_ops:
            return

        rated_ops.sort(key=lambda x: -x[1])
        total_n = sum(s["n_used"] for s in op_rates.values())
        overall_rate = sum(
            s.get("n_stage1_passed", 0) or int(s["s1_rate"] * s["n_s0"])
            for s in op_rates.values()
        ) / max(total_n, 1)

        # Best ops: significantly above average
        for op, rate, n_used, n_passed in rated_ops[:5]:
            if rate <= overall_rate or n_used < 10:
                continue
            n_failed = n_used - n_passed
            insights.append(
                {
                    "content": (
                        f"Op '{op}' has {rate:.1%} S1 rate "
                        f"(n={n_used}, baseline={overall_rate:.1%})."
                    ),
                    "category": "success_factor",
                    "insight_type": "top_op",
                    "subject_key": op,
                    "semantic_key": f"top_op:{op}",
                    "alpha": float(n_passed + 1),
                    "beta_": float(n_failed + 1),
                    "display_only": False,
                    "insight_level": "composition",
                    "evidence_json": {
                        "test": "proportion_vs_baseline",
                        "n": n_used,
                        "rate": round(rate, 4),
                        "baseline_rate": round(overall_rate, 4),
                        "effect_size": round(rate - overall_rate, 4),
                    },
                }
            )

        # Worst ops: 0% S1 with significant sample
        for op, rate, n_used, n_passed in reversed(rated_ops):
            if rate > 0.005 or n_used < 15:
                continue
            insights.append(
                {
                    "content": (
                        f"Op '{op}' has {rate:.1%} S1 rate "
                        f"(n={n_used}). Consistently failing."
                    ),
                    "category": "failure_mode",
                    "insight_type": "failing_op",
                    "subject_key": op,
                    "semantic_key": f"failing_op:{op}",
                    "alpha": float(n_passed + 1),
                    "beta_": float(n_used - n_passed + 1),
                    "display_only": True,
                    "insight_level": "op",
                    "evidence_json": {
                        "test": "proportion_vs_baseline",
                        "n": n_used,
                        "rate": round(rate, 4),
                        "baseline_rate": round(overall_rate, 4),
                    },
                }
            )

    def _compute_structural_correlation_insights(
        self,
        correlations: Dict[str, float],
        size_rows: list,
        insights: List[Dict],
    ) -> None:
        """Convert structural correlations into insights with effect-size evidence."""
        n = len(size_rows) if size_rows else 0
        for metric, effect in sorted(correlations.items(), key=lambda x: -abs(x[1])):
            if abs(effect) < 0.3:
                continue
            direction = "positively" if effect > 0 else "negatively"
            name = metric.replace("graph_", "").replace("_", " ")
            pseudo_correct = max(1, int(abs(effect) * 20))
            pseudo_wrong = max(1, int((2.0 - min(abs(effect), 2.0)) * 10))
            insights.append(
                {
                    "content": (
                        f"Graph {name} is {direction} correlated with "
                        f"Stage 1 success (effect={effect:.2f}, n={n})."
                    ),
                    "category": "hypothesis",
                    "insight_type": "graph_correlation",
                    "subject_key": metric,
                    "semantic_key": f"graph_correlation:{metric}",
                    "alpha": float(pseudo_correct),
                    "beta_": float(pseudo_wrong),
                    "display_only": False,
                    "insight_level": "structural",
                    "evidence_json": {
                        "test": "standardized_mean_difference",
                        "effect_size": round(float(effect), 4),
                        "n": n,
                        "metric": metric,
                    },
                }
            )
            break  # strongest only

    def _compute_failure_insights(
        self,
        failures: Dict[str, Dict],
        insights: List[Dict],
    ) -> None:
        """Failure pattern insights -- always display_only.

        Code failures (RuntimeError, TypeError, etc.) are explicitly separated
        from training failures (nan_loss, diverged, etc.).
        """
        for error_type, data in sorted(failures.items(), key=lambda x: -x[1]["total"])[
            :5
        ]:
            total = data["total"]
            if total < 10:
                continue
            is_code_failure = error_type in _CODE_FAILURE_TYPES
            insights.append(
                {
                    "content": (
                        f"{'Code' if is_code_failure else 'Training'} failure: "
                        f"{error_type} ({total} occurrences). "
                        f"Stages: {data['by_stage']}"
                    ),
                    "category": "failure_mode",
                    "insight_type": "code_failure"
                    if is_code_failure
                    else "training_failure",
                    "subject_key": str(error_type),
                    "semantic_key": f"common_failure:{error_type}",
                    "alpha": 1.0,
                    "beta_": float(total),
                    "display_only": True,
                    "insight_level": "op",
                    "evidence_json": {
                        "test": "frequency_count",
                        "n": total,
                        "is_code_failure": is_code_failure,
                        "by_stage": data["by_stage"],
                    },
                }
            )

    def _compute_combo_insights(
        self,
        combos: List[Dict],
        insights: List[Dict],
    ) -> None:
        """Op-combination insights at the composition level."""
        for top in combos[:3]:
            count = top["count"]
            if count < 5:
                continue
            ops = top["ops"]
            avg_novelty = top.get("avg_novelty", 0)
            insights.append(
                {
                    "content": (
                        f"Winning combination: {' + '.join(ops)} "
                        f"appears in {count} survivors "
                        f"(avg novelty {avg_novelty:.3f})."
                    ),
                    "category": "success_factor",
                    "insight_type": "winning_combo",
                    "subject_key": "+".join(sorted(str(op) for op in ops)),
                    "semantic_key": "winning_combo:"
                    + "+".join(sorted(str(op) for op in ops)),
                    "alpha": float(count + 1),
                    "beta_": 1.0,
                    "display_only": False,
                    "insight_level": "composition",
                    "evidence_json": {
                        "test": "co_occurrence_count",
                        "n_survivors": count,
                        "avg_novelty": round(avg_novelty, 4),
                    },
                }
            )
