from __future__ import annotations
import hashlib
import logging
import math
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from ...synthesis.grammar import GrammarConfig
from ...synthesis.primitives import get_primitive

logger = logging.getLogger(__name__)


class _GrammarMixin:
    """Grammar weight computation, statistical helpers, and attribution."""

    __slots__ = ()

    def _gather_category_stats(
        self,
        op_rates: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """Group op success rates by category.

        Structural ops (identity, splits, masks, reduce ops) are excluded
        from S1 total aggregation — they have no learnable parameters and
        should not drag down category weights as standalone learners.
        """
        from ...synthesis.context_rules import S1_EXEMPT_OPS

        cat_stats: Dict[str, Dict] = defaultdict(
            lambda: {
                "total": 0,
                "s1_total": 0,
                "novelty_sum": 0.0,
                "count": 0,
                "conf_sum": 0.0,
                "conf_count": 0,
            }
        )
        for op_name, stats in op_rates.items():
            try:
                op = get_primitive(op_name)
                cat = op.category.value
            except (KeyError, Exception):
                continue
            if op_name in S1_EXEMPT_OPS:
                continue
            cat_stats[cat]["total"] += stats["n_used"]
            cat_stats[cat]["s1_total"] += int(stats["s1_rate"] * stats["n_used"])
            if stats.get("avg_novelty"):
                cat_stats[cat]["novelty_sum"] += stats["avg_novelty"] * stats["n_used"]
                cat_stats[cat]["count"] += stats["n_used"]
            if stats.get("avg_novelty_confidence"):
                cat_stats[cat]["conf_sum"] += (
                    stats["avg_novelty_confidence"] * stats["n_used"]
                )
                cat_stats[cat]["conf_count"] += stats["n_used"]
        return cat_stats

    def _compute_weights_from_stats(
        self,
        cat_stats: Dict[str, Dict],
    ) -> Optional[Dict[str, float]]:
        """Compute grammar weights from per-category statistics."""
        default_weights = GrammarConfig().category_weights

        cat_s1_rates = {}
        cat_novelties = {}
        cat_confidences = {}
        for cat, stats in cat_stats.items():
            if stats["total"] < 2:
                continue
            cat_s1_rates[cat] = stats["s1_total"] / max(stats["total"], 1)
            cat_novelties[cat] = (
                stats["novelty_sum"] / stats["count"] if stats["count"] > 0 else 0.0
            )
            cat_confidences[cat] = (
                stats["conf_sum"] / stats["conf_count"]
                if stats["conf_count"] > 0
                else 0.0
            )

        if not cat_s1_rates:
            return None

        mean_s1 = sum(cat_s1_rates.values()) / len(cat_s1_rates)

        learned = {}
        stability_multipliers = self.instability_attribution()

        for cat, s1_rate in cat_s1_rates.items():
            n = cat_stats[cat]["total"]
            relative = s1_rate / max(mean_s1, 0.01)

            # Statistical guard (#42): skip noisy differences
            se = (
                math.sqrt(s1_rate * (1 - s1_rate) / n)
                if n > 0 and 0 < s1_rate < 1
                else 0.0
            )
            effect = abs(s1_rate - mean_s1)
            if se > 0 and effect < se:
                default = default_weights.get(cat, 1.0)
                tentative = default * (relative**2)
                learned[cat] = round(0.5 * tentative + 0.5 * default, 2)
                continue

            amplified = relative**2
            # Discount novelty factor by average confidence for this category
            # Low-confidence novelty (e.g. structural-only at 0.2) contributes
            # much less than high-confidence (full behavioral at 0.9)
            raw_novelty = cat_novelties.get(cat, 0.0)
            confidence = cat_confidences.get(cat, 0.0)
            novelty_factor = 1.0 + raw_novelty * confidence
            base = default_weights.get(cat, 1.0)

            # Apply stability multiplier
            stab = stability_multipliers.get(cat, 1.0)
            weight = base * amplified * novelty_factor * stab

            if cat == "frequency_domain":
                learned[cat] = round(max(0.0, min(0.1, weight)), 2)
            else:
                learned[cat] = round(max(0.5, min(8.0, weight)), 2)

        # EMA blending if last_applied exists (moved from compute_grammar_weights)
        # This function should just return the raw learned weights from stats.
        # Actually, the caller handles EMA blending.

        # Hard cap: frequency_domain shows strong negative correlation with S1 (-0.33).
        # Cap at 0.1 to suppress generation of frequency-domain ops.
        _SUPPRESSED_CATEGORIES = {"frequency_domain": 0.1}
        for cat, cap in _SUPPRESSED_CATEGORIES.items():
            if cat in learned:
                learned[cat] = min(learned[cat], cap)

        return learned if learned else None

    def _collect_fingerprint_capped_op_rates(
        self,
        per_fingerprint_cap: float,
    ) -> Tuple[Dict[str, Dict], Dict[str, float]]:
        """Build op success rates with capped contribution per architecture fingerprint."""
        cursor = self.nb.conn.execute(
            """SELECT result_id, graph_fingerprint, graph_json, stage1_passed,
                      novelty_score, novelty_confidence
               FROM program_results
               WHERE graph_json IS NOT NULL"""
        )

        extracted_rows: List[Dict] = []
        fingerprint_counts: Dict[str, int] = defaultdict(int)

        # Z13: Identify Pareto winners for weighting boost
        pareto_ids = set(self.pareto_optimal_programs())

        for row in cursor:
            graph_json = row["graph_json"]
            if not graph_json:
                continue

            ops = self._extract_ops_fast(graph_json)
            if ops is None:
                ops = self._extract_ops_fallback(graph_json)
            if not ops:
                continue

            graph_fingerprint = str(row["graph_fingerprint"] or "").strip()
            fp_key = graph_fingerprint or f"result:{row['result_id']}"
            fingerprint_counts[fp_key] += 1

            # Pareto boost: 5x weight for non-dominated models
            is_pareto = row["result_id"] in pareto_ids

            extracted_rows.append(
                {
                    "fingerprint": fp_key,
                    "ops": set(ops),
                    "stage1_passed": bool(row["stage1_passed"]),
                    "novelty_score": self._as_float(row["novelty_score"]),
                    "novelty_confidence": self._as_float(row["novelty_confidence"]),
                    "weight_multiplier": 5.0 if is_pareto else 1.0,
                }
            )

        op_stats: Dict[str, Dict] = {}
        effective_rows = 0.0
        for extracted in extracted_rows:
            fp_key = extracted["fingerprint"]
            fp_count = max(fingerprint_counts.get(fp_key, 1), 1)
            row_weight = min(1.0, max(per_fingerprint_cap, 0.1) / float(fp_count))
            # Apply Pareto multiplier
            row_weight *= extracted["weight_multiplier"]

            effective_rows += row_weight

            for op_name in extracted["ops"]:
                stats = op_stats.setdefault(
                    op_name,
                    {
                        "n_used": 0.0,
                        "n_s1": 0.0,
                        "nov_sum": 0.0,
                        "nov_n": 0.0,
                        "conf_sum": 0.0,
                        "conf_n": 0.0,
                    },
                )
                stats["n_used"] += row_weight
                if extracted["stage1_passed"]:
                    stats["n_s1"] += row_weight
                novelty = extracted["novelty_score"]
                if novelty is not None:
                    stats["nov_sum"] += novelty * row_weight
                    stats["nov_n"] += row_weight
                confidence = extracted["novelty_confidence"]
                if confidence is not None:
                    stats["conf_sum"] += confidence * row_weight
                    stats["conf_n"] += row_weight

        op_rates: Dict[str, Dict] = {}
        for op_name, stats in op_stats.items():
            n_used = stats["n_used"]
            if n_used <= 0:
                continue
            op_rates[op_name] = {
                "n_used": n_used,
                "n_stage1_passed": stats["n_s1"],
                "s1_rate": stats["n_s1"] / n_used,
                "avg_novelty": (stats["nov_sum"] / stats["nov_n"])
                if stats["nov_n"] > 0
                else None,
                "avg_novelty_confidence": (stats["conf_sum"] / stats["conf_n"])
                if stats["conf_n"] > 0
                else None,
            }

        total_rows = len(extracted_rows)
        unique_fingerprints = len(fingerprint_counts)
        repeat_rows = sum(max(0, count - 1) for count in fingerprint_counts.values())
        top_fingerprint_count = (
            max(fingerprint_counts.values()) if fingerprint_counts else 0
        )
        diagnostics: Dict[str, float] = {
            "total_rows": float(total_rows),
            "effective_rows": float(round(effective_rows, 4)),
            "unique_fingerprints": float(unique_fingerprints),
            "repeat_rows": float(repeat_rows),
            "rerun_ratio": (repeat_rows / total_rows) if total_rows > 0 else 0.0,
            "top_fingerprint_concentration": (top_fingerprint_count / total_rows)
            if total_rows > 0
            else 0.0,
            "fingerprint_cap": float(per_fingerprint_cap),
        }
        return op_rates, diagnostics

    @staticmethod
    def _wilson_interval(
        successes: int, total: int, z: float = 1.96
    ) -> Tuple[float, float]:
        """Wilson score interval for Bernoulli proportion."""
        if total <= 0:
            return (0.0, 0.0)
        p = successes / total
        z2 = z * z
        denom = 1.0 + z2 / total
        center = (p + z2 / (2.0 * total)) / denom
        margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denom
        return (max(0.0, center - margin), min(1.0, center + margin))

    @staticmethod
    def _two_prop_pvalue(
        successes_a: int, total_a: int, successes_b: int, total_b: int
    ) -> float:
        """Two-sided z-test p-value for proportion difference."""
        if total_a <= 0 or total_b <= 0:
            return 1.0
        p_a = successes_a / total_a
        p_b = successes_b / total_b
        pooled = (successes_a + successes_b) / max(total_a + total_b, 1)
        if pooled <= 0.0 or pooled >= 1.0:
            return 1.0
        se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / total_a + 1.0 / total_b))
        if se <= 1e-12:
            return 1.0
        z = abs((p_a - p_b) / se)
        # 2 * (1 - Phi(|z|)) expressed via erfc for determinism without scipy
        return min(1.0, max(0.0, math.erfc(z / math.sqrt(2.0))))

    @staticmethod
    def _apply_fdr_bh(
        rows: List[Dict[str, Any]], p_key: str = "p_value", q_key: str = "q_value"
    ) -> List[Dict[str, Any]]:
        """Apply Benjamini-Hochberg FDR correction in-place and return rows."""
        indexed = []
        for idx, row in enumerate(rows):
            p = row.get(p_key)
            if isinstance(p, (int, float)):
                indexed.append((idx, float(p)))
        m = len(indexed)
        if m == 0:
            return rows
        ranked = sorted(indexed, key=lambda t: t[1])
        q_vals = [1.0] * m
        prev = 1.0
        for i in range(m - 1, -1, -1):
            rank = i + 1
            p = ranked[i][1]
            q = min(prev, p * m / rank)
            q_vals[i] = q
            prev = q
        for i, (orig_idx, _p) in enumerate(ranked):
            rows[orig_idx][q_key] = float(q_vals[i])
        for row in rows:
            row.setdefault(q_key, 1.0)
        return rows

    @staticmethod
    def _depth_bucket(depth: Optional[int]) -> str:
        if depth is None:
            return "unknown"
        d = int(depth)
        if d <= 3:
            return "shallow"
        if d <= 6:
            return "medium"
        return "deep"

    def pareto_optimal_programs(self) -> List[str]:
        """Find result_ids of non-dominated programs (Accuracy vs Efficiency).

        Uses vectorized NumPy dominance check with early pruning via
        sort on first objective.
        """
        rows = self.nb.conn.execute("""
            SELECT result_id, loss_ratio, validation_loss_ratio,
                   graph_n_params_estimate, param_count
            FROM program_results
            WHERE stage1_passed = 1
        """).fetchall()

        if not rows:
            return []

        ids = []
        data = []
        for r in rows:
            lr = (
                r["validation_loss_ratio"]
                if r["validation_loss_ratio"] is not None
                else r["loss_ratio"]
            )
            params = (
                r["param_count"]
                if r["param_count"] is not None
                else r["graph_n_params_estimate"]
            )
            if lr is not None and params is not None:
                data.append((1.0 - lr, 1.0 / max(1, params)))
                ids.append(r["result_id"])

        if not data:
            return []

        costs = np.array(data, dtype=np.float32)
        n = costs.shape[0]

        # Sort by objective 1 descending, then objective 2 descending so
        # equal-loss candidates keep the most efficient point first.
        order = np.lexsort((-costs[:, 1], -costs[:, 0]))
        costs_sorted = costs[order]

        is_pareto = np.ones(n, dtype=bool)
        max_obj2 = -np.inf
        for i in range(n):
            if costs_sorted[i, 1] <= max_obj2:
                is_pareto[i] = False
            else:
                max_obj2 = costs_sorted[i, 1]

        # Map back to original indices
        pareto_mask = np.zeros(n, dtype=bool)
        pareto_mask[order[is_pareto]] = True
        return [ids[i] for i in range(n) if pareto_mask[i]]

    def _load_program_factor_rows(self) -> List[Dict[str, Any]]:
        """Load per-program factors for attribution analysis."""
        cursor = self.nb.conn.execute(
            """SELECT result_id, experiment_id, graph_json, stage1_passed,
                      graph_depth, graph_uses_math_spaces
               FROM program_results
               WHERE graph_json IS NOT NULL"""
        )
        parsed: List[Dict[str, Any]] = []
        for row in cursor:
            graph_json = row["graph_json"]
            ops = self._extract_ops_fast(graph_json)
            if ops is None:
                ops = self._extract_ops_fallback(graph_json)
            if not ops:
                continue
            op_set = set(ops)
            families: Set[str] = set()
            for op_name in op_set:
                try:
                    families.add(get_primitive(op_name).category.value)
                except (KeyError, ValueError):
                    continue
            parsed.append(
                {
                    "result_id": row["result_id"],
                    "experiment_id": row["experiment_id"],
                    "stage1_passed": int(bool(row["stage1_passed"])),
                    "ops": op_set,
                    "families": families,
                    "math_space": bool(row["graph_uses_math_spaces"]),
                    "depth_bucket": self._depth_bucket(row["graph_depth"]),
                }
            )
        parsed.sort(key=lambda r: (str(r["experiment_id"]), str(r["result_id"])))
        return parsed

    def _factor_success_stats(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compute per-factor success rates, uncertainty, and p-values."""
        total_n = len(rows)
        if total_n <= 1:
            return []
        total_s = sum(r["stage1_passed"] for r in rows)

        factors: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            for op_name in row["ops"]:
                factors[("op", op_name)].append(idx)
            for fam in row["families"]:
                factors[("family", fam)].append(idx)
            factors[
                ("math_space", "enabled" if row["math_space"] else "disabled")
            ].append(idx)
            factors[("depth_bucket", row["depth_bucket"])].append(idx)

        out: List[Dict[str, Any]] = []
        for (factor_type, factor_name), indices in sorted(factors.items()):
            with_n = len(indices)
            without_n = total_n - with_n
            if with_n <= 0 or without_n <= 0:
                continue
            with_s = sum(rows[i]["stage1_passed"] for i in indices)
            without_s = total_s - with_s
            with_rate = with_s / with_n
            without_rate = without_s / without_n
            ci_low, ci_high = self._wilson_interval(with_s, with_n)
            p_val = self._two_prop_pvalue(with_s, with_n, without_s, without_n)
            out.append(
                {
                    "factor_type": factor_type,
                    "factor_name": factor_name,
                    "n_with": with_n,
                    "n_without": without_n,
                    "success_with": with_s,
                    "success_without": without_s,
                    "rate_with": with_rate,
                    "rate_without": without_rate,
                    "delta_rate": with_rate - without_rate,
                    "ci_with_low": ci_low,
                    "ci_with_high": ci_high,
                    "p_value": p_val,
                }
            )
        self._apply_fdr_bh(out, p_key="p_value", q_key="q_value")
        out.sort(key=lambda r: (r["factor_type"], r["factor_name"]))
        return out

    def _matched_control_stats(
        self, rows: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Matched-control comparisons for single-factor contrasts."""
        comparisons: List[Dict[str, Any]] = []
        if len(rows) < 4:
            return comparisons

        # Math-space effect, matched on depth bucket
        for bucket in sorted({r["depth_bucket"] for r in rows}):
            group_a = [
                r for r in rows if r["depth_bucket"] == bucket and r["math_space"]
            ]
            group_b = [
                r for r in rows if r["depth_bucket"] == bucket and not r["math_space"]
            ]
            if not group_a or not group_b:
                continue
            s_a = sum(r["stage1_passed"] for r in group_a)
            s_b = sum(r["stage1_passed"] for r in group_b)
            n_a = len(group_a)
            n_b = len(group_b)
            p_val = self._two_prop_pvalue(s_a, n_a, s_b, n_b)
            delta = (s_a / n_a) - (s_b / n_b)
            se = math.sqrt(
                (s_a / n_a) * (1 - (s_a / n_a)) / n_a
                + (s_b / n_b) * (1 - (s_b / n_b)) / n_b
            )
            comparisons.append(
                {
                    "factor": "math_space",
                    "match_on": f"depth_bucket={bucket}",
                    "n_a": n_a,
                    "n_b": n_b,
                    "delta_rate": delta,
                    "ci_low": delta - 1.96 * se,
                    "ci_high": delta + 1.96 * se,
                    "p_value": p_val,
                }
            )

        # Op family effect, matched on depth bucket and math-space
        families = sorted({f for r in rows for f in r["families"]})
        strata_keys = sorted({(r["depth_bucket"], r["math_space"]) for r in rows})
        for family in families:
            total_a = total_b = succ_a = succ_b = 0
            for depth_bucket, math_space in strata_keys:
                stratum = [
                    r
                    for r in rows
                    if r["depth_bucket"] == depth_bucket
                    and r["math_space"] == math_space
                ]
                if not stratum:
                    continue
                a = [r for r in stratum if family in r["families"]]
                b = [r for r in stratum if family not in r["families"]]
                if not a or not b:
                    continue
                total_a += len(a)
                total_b += len(b)
                succ_a += sum(r["stage1_passed"] for r in a)
                succ_b += sum(r["stage1_passed"] for r in b)
            if total_a <= 0 or total_b <= 0:
                continue
            rate_a = succ_a / total_a
            rate_b = succ_b / total_b
            delta = rate_a - rate_b
            p_val = self._two_prop_pvalue(succ_a, total_a, succ_b, total_b)
            se = math.sqrt(
                rate_a * (1 - rate_a) / total_a + rate_b * (1 - rate_b) / total_b
            )
            comparisons.append(
                {
                    "factor": f"family:{family}",
                    "match_on": "depth_bucket,math_space",
                    "n_a": total_a,
                    "n_b": total_b,
                    "delta_rate": delta,
                    "ci_low": delta - 1.96 * se,
                    "ci_high": delta + 1.96 * se,
                    "p_value": p_val,
                }
            )

        self._apply_fdr_bh(comparisons, p_key="p_value", q_key="q_value")
        comparisons.sort(key=lambda r: (r["factor"], r["match_on"]))
        return comparisons

    def grammar_weight_attribution_report(self) -> Dict[str, Any]:
        """Attribution report separating correlation from stronger evidence."""
        rows = self._load_program_factor_rows()
        factor_stats = self._factor_success_stats(rows)
        matched_controls = self._matched_control_stats(rows)

        def _is_interpretable_factor(signal: Dict[str, Any]) -> bool:
            factor_name = str(signal.get("factor_name") or "").strip().lower()
            return bool(
                factor_name and factor_name not in {"unknown", "none", "null", "nan"}
            )

        strong_correlational = [
            s
            for s in factor_stats
            if s["n_with"] >= 20
            and s["delta_rate"] > 0.05
            and s.get("q_value", 1.0) <= 0.10
            and s["ci_with_low"] > s["rate_without"]
        ]
        strong_correlational_interpretable = [
            s for s in strong_correlational if _is_interpretable_factor(s)
        ]
        matched_positive = [
            m
            for m in matched_controls
            if m["n_a"] >= 12
            and m["n_b"] >= 12
            and m["delta_rate"] > 0.05
            and m.get("q_value", 1.0) <= 0.10
            and m["ci_low"] > 0.0
        ]

        correlational_ok = bool(strong_correlational_interpretable or matched_positive)
        top_signal = None
        if strong_correlational_interpretable:
            top_signal = sorted(
                strong_correlational_interpretable,
                key=lambda s: (s["q_value"], -s["delta_rate"], -s["n_with"]),
            )[0]
        elif matched_positive:
            top_signal = sorted(
                matched_positive,
                key=lambda m: (m["q_value"], -m["delta_rate"], -(m["n_a"] + m["n_b"])),
            )[0]

        uncertainty = {
            "n_programs": len(rows),
            "n_factors_tested": len(factor_stats),
            "n_matched_tests": len(matched_controls),
            "fdr_method": "benjamini_hochberg",
            "correlational_signal_count": len(strong_correlational),
            "interpretable_correlational_signal_count": len(
                strong_correlational_interpretable
            ),
            "matched_signal_count": len(matched_positive),
        }

        return {
            "factors": factor_stats,
            "matched_controls": matched_controls,
            "strong_correlational_evidence": correlational_ok,
            "top_signal": top_signal,
            "uncertainty": uncertainty,
            "requires_ablation": correlational_ok,
        }

    def compute_grammar_weights(
        self,
        last_applied: Optional[Dict[str, float]] = None,
        alpha: float = 0.6,
    ) -> Optional[Dict[str, float]]:
        """Compute learned category weights from historical success data.

        Uses the aggregate op_success_rates table. For holdout validation
        of the learned weights, call ``holdout_validation()`` separately.

        Args:
            last_applied: Previous effective weights for EMA blending.
                When None (first run, tests), no blending is applied.
            alpha: EMA weight for new signal vs last_applied (default 0.6).

        Returns a dict of category -> weight, or None if insufficient data.
        """
        fingerprint_rates, fingerprint_diag = self._collect_fingerprint_capped_op_rates(
            per_fingerprint_cap=self.FINGERPRINT_WEIGHT_CAP,
        )
        weighting_mode = "fingerprint_capped"
        op_rates = fingerprint_rates
        if len(op_rates) < 5:
            op_rates = self.op_success_rates()
            weighting_mode = "legacy_op_aggregate"
        if len(op_rates) < 5:
            self._last_grammar_weight_diagnostics = {
                "mode": weighting_mode,
                "insufficient_op_coverage": True,
                "op_count": len(op_rates),
                **fingerprint_diag,
            }
            return None

        cat_stats = self._gather_category_stats(op_rates)
        if not cat_stats:
            self._last_grammar_weight_diagnostics = {
                "mode": weighting_mode,
                "insufficient_category_coverage": True,
                "op_count": len(op_rates),
                **fingerprint_diag,
            }
            return None

        learned = self._compute_weights_from_stats(cat_stats)

        # Z13: Search Health Guard (Step 3)
        # If discovery and validation are uncorrelated, we are likely 'reward hacking'.
        # In this case, we should revert towards uniform (default) weights to find
        # a new region of the search space.
        gate_stats = self.gate_performance_summary()
        correlation = gate_stats.get("discovery_validation_correlation")
        n_samples = gate_stats.get("n_correlation_samples", 0)
        default_weights = self.get_current_grammar_weights() or {}

        # If correlation is low (< 0.3) and we have enough data, dampen learned signal
        if learned and correlation is not None and n_samples > 10 and correlation < 0.3:
            logger.info(
                f"Low discovery-validation correlation detected ({correlation:.2f}); dampening learned weights to increase diversity."
            )
            for cat in learned:
                default = default_weights.get(cat, 1.0)
                # Blend 70% default, 30% learned
                learned[cat] = round(0.7 * default + 0.3 * learned[cat], 2)

        # ── P0.2: Op synergy → category weight adjustment ──
        if learned:
            learned = self._apply_synergy_adjustments(learned, op_rates)

        # ── P2.4: Survivorship bias correction — exploration bonus ──
        if learned:
            learned = self._apply_exploration_bonus(learned, op_rates)

        # ── P2.2: Family regression detection → penalize regressing categories ──
        if learned:
            learned = self._apply_family_regression_penalty(learned, op_rates)

        if learned and last_applied:
            for cat in learned:
                if cat in last_applied:
                    learned[cat] = round(
                        alpha * learned[cat] + (1 - alpha) * last_applied[cat], 2
                    )
        self._last_grammar_weight_diagnostics = {
            "mode": weighting_mode,
            "insufficient_op_coverage": False,
            "op_count": len(op_rates),
            "category_count": len(cat_stats),
            "used_fingerprint_capping": weighting_mode == "fingerprint_capped",
            **fingerprint_diag,
        }
        return learned

    def recent_hierarchy_fitness(self, lookback: int = 50) -> Optional[float]:
        """Query average hierarchy_fitness from recent program_results.

        Returns the mean fp_hierarchy_fitness across recent results, or None
        if insufficient data.
        """
        try:
            rows = self.nb.conn.execute(
                """SELECT fp_hierarchy_fitness FROM program_results
                   WHERE fp_hierarchy_fitness IS NOT NULL
                   ORDER BY timestamp DESC LIMIT ?""",
                (lookback,),
            ).fetchall()
            if len(rows) < 5:
                return None
            vals = [float(r[0]) for r in rows if r[0] is not None]
            if not vals:
                return None
            return sum(vals) / len(vals)
        except (TypeError, ValueError, sqlite3.OperationalError):
            return None

    def grammar_weight_learning_diagnostics(self) -> Dict:
        """Return diagnostics for grammar-weight learning robustness."""
        if self._last_grammar_weight_diagnostics is None:
            self.compute_grammar_weights()
        diagnostics = dict(self._last_grammar_weight_diagnostics or {})
        if "mode" not in diagnostics:
            diagnostics["mode"] = "unknown"
        if "used_fingerprint_capping" not in diagnostics:
            diagnostics["used_fingerprint_capping"] = False
        if "fingerprint_cap" not in diagnostics:
            diagnostics["fingerprint_cap"] = float(self.FINGERPRINT_WEIGHT_CAP)
        return diagnostics

    def grammar_weight_audit_info(self) -> Dict[str, Any]:
        """Return reproducible query info for grammar weight learning."""
        return {
            "query": (
                "SELECT result_id, graph_fingerprint, graph_json, stage1_passed, "
                "novelty_score, novelty_confidence FROM program_results "
                "WHERE graph_json IS NOT NULL"
            ),
            "params": [],
            "fingerprint_cap": float(self.FINGERPRINT_WEIGHT_CAP),
            "weighting_mode": "fingerprint_capped",
            "source": "program_results",
        }

    def holdout_validation(self, holdout_fraction: float = 0.2) -> Optional[Dict]:
        """Evaluate grammar quality on holdout experiments.

        Returns s1_rate and program count for holdout experiments.
        """
        experiments = self.nb.conn.execute(
            "SELECT experiment_id FROM experiments WHERE status = 'completed'"
        ).fetchall()

        if len(experiments) < 5:
            return None

        holdout_ids = []
        for row in experiments:
            eid = row["experiment_id"]
            h = int(hashlib.md5(eid.encode()).hexdigest()[:8], 16)
            if (h % 100) < int(holdout_fraction * 100):
                holdout_ids.append(eid)

        if not holdout_ids:
            return None

        placeholders = ",".join("?" * len(holdout_ids))
        row = self.nb.conn.execute(
            f"""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) as s1_passed
            FROM program_results
            WHERE experiment_id IN ({placeholders})
        """,
            tuple(holdout_ids),
        ).fetchone()

        total = row["total"] or 0
        s1 = row["s1_passed"] or 0
        return {
            "holdout_experiments": len(holdout_ids),
            "holdout_programs": total,
            "holdout_s1_passed": s1,
            "holdout_s1_rate": s1 / max(total, 1),
        }

    def instability_attribution(self) -> Dict[str, float]:
        """Correlate architectural categories with high Jacobian spectral norm.

        Returns a penalty multiplier [0.5, 1.0] for each category.
        Categories that frequently cause instability get lower multipliers.
        """
        count_row = self.nb.conn.execute(
            "SELECT COUNT(*) FROM program_results "
            "WHERE fp_jacobian_spectral_norm IS NOT NULL"
        ).fetchone()
        if (count_row[0] or 0) < 10:
            return {}

        cursor = self.nb.conn.execute("""
            SELECT graph_json, fp_jacobian_spectral_norm
            FROM program_results
            WHERE fp_jacobian_spectral_norm IS NOT NULL
        """)

        cat_norms = defaultdict(list)
        for r in cursor:
            ops = self._extract_ops_fast(r["graph_json"])
            norm = float(r["fp_jacobian_spectral_norm"])
            if not ops:
                continue

            seen_cats = set()
            for op in ops:
                try:
                    cat = get_primitive(op).category.value
                    if cat not in seen_cats:
                        cat_norms[cat].append(norm)
                        seen_cats.add(cat)
                except (KeyError, ValueError):
                    continue

        penalties = {}
        for cat, norms in cat_norms.items():
            if len(norms) < 5:
                continue
            avg_norm = np.mean(norms)
            # Threshold: > 15 is risky, > 50 is toxic
            if avg_norm > 15.0:
                # Scale penalty from 1.0 down to 0.5
                penalty = max(0.5, 1.0 - (avg_norm - 15.0) / 70.0)
                penalties[cat] = round(float(penalty), 2)
            else:
                penalties[cat] = 1.0

        return penalties

    def _apply_family_regression_penalty(
        self,
        learned: Dict[str, float],
        op_rates: Dict[str, Dict],
    ) -> Dict[str, float]:
        """Penalize categories associated with regressing architecture families (P2.2).

        Detects families whose composite_score MA has declined > 20% from peak,
        maps their ops to categories, and reduces category weights proportionally.
        """
        from ...scientist.intelligence.analyzer import detect_family_regression

        try:
            regressions = detect_family_regression(self.nb)
        except (KeyError, TypeError, ValueError, sqlite3.OperationalError) as e:
            logger.warning("Family regression detection failed: %s", e)
            return learned

        if not regressions:
            return learned

        # Map regressing family fingerprint prefixes → ops → categories
        regressing_prefixes = {r["family"] for r in regressions}
        decline_by_prefix = {r["family"]: r["decline_pct"] for r in regressions}

        # Query ops for regressing families
        cat_decline: Dict[str, List[float]] = defaultdict(list)
        try:
            rows = self.nb.conn.execute(
                """SELECT pr.graph_fingerprint, pr.graph_json
                   FROM program_results pr
                   WHERE pr.graph_json IS NOT NULL
                     AND pr.graph_fingerprint IS NOT NULL
                   ORDER BY pr.timestamp DESC LIMIT 500"""
            ).fetchall()

            for row in rows:
                fp = str(row["graph_fingerprint"] or "").strip()
                prefix = fp[:8] if len(fp) >= 8 else ""
                if prefix not in regressing_prefixes:
                    continue

                decline = decline_by_prefix[prefix]
                ops = self._extract_ops_fast(row["graph_json"])
                if not ops:
                    continue

                seen_cats: Set[str] = set()
                for op_name in ops:
                    try:
                        cat = get_primitive(op_name).category.value
                        if cat not in seen_cats:
                            cat_decline[cat].append(decline)
                            seen_cats.add(cat)
                    except (KeyError, ValueError):
                        continue
        except (KeyError, TypeError, ValueError, sqlite3.OperationalError) as e:
            logger.warning("Family regression op lookup failed: %s", e)
            return learned

        n_penalized = 0
        for cat in learned:
            if cat not in cat_decline:
                continue
            avg_decline = sum(cat_decline[cat]) / len(cat_decline[cat])
            # Penalty: max(0.5, 1.0 - decline_pct)
            penalty = max(0.5, 1.0 - avg_decline)
            learned[cat] = round(learned[cat] * penalty, 2)
            n_penalized += 1

        if n_penalized:
            logger.info(
                "Family regression penalty: %d categories penalized "
                "(from %d regressing families)",
                n_penalized,
                len(regressions),
            )

        return learned

    def _apply_exploration_bonus(
        self,
        learned: Dict[str, float],
        op_rates: Dict[str, Dict],
    ) -> Dict[str, float]:
        """Apply exploration bonus to under-tested categories (P2.4).

        Tracks n_tested per category (total programs, not just S1 passes)
        and applies: weight *= (1.0 + 0.5 / sqrt(n_tested + 1)).
        This prevents under-explored ops from staying at default weights.
        """
        from ...synthesis.context_rules import S1_EXEMPT_OPS

        # Aggregate n_tested per category
        cat_n_tested: Dict[str, float] = defaultdict(float)
        for op_name, stats in op_rates.items():
            try:
                op = get_primitive(op_name)
                cat = op.category.value
            except (KeyError, ValueError):
                continue
            if op_name in S1_EXEMPT_OPS:
                continue
            cat_n_tested[cat] += stats.get("n_used", 0)

        under_explored = []
        for cat in learned:
            n_tested = cat_n_tested.get(cat, 0)
            bonus = 1.0 + 0.5 / math.sqrt(n_tested + 1)
            learned[cat] = round(learned[cat] * bonus, 2)
            if n_tested < 10:
                under_explored.append(f"{cat}({n_tested:.0f})")

        if under_explored:
            logger.info(
                "Exploration bonus: %d under-explored categories (n_tested < 10): %s",
                len(under_explored),
                ", ".join(under_explored[:10]),
            )

        return learned

    def _apply_synergy_adjustments(
        self,
        learned: Dict[str, float],
        op_rates: Dict[str, Dict],
    ) -> Dict[str, float]:
        """Adjust category weights based on op synergy/anti-synergy signal.

        Aggregates per-op-pair lift into per-category bonuses/penalties.
        Synergistic pairs (lift > 1.5): boost categories containing those ops.
        Anti-synergistic pairs (lift < 0.5): penalize categories.
        """
        from ...scientist.intelligence.analyzer import analyze_op_synergies

        try:
            synergies = analyze_op_synergies(self.nb)
        except (KeyError, TypeError, ValueError, sqlite3.OperationalError) as e:
            logger.warning("Synergy analysis failed, skipping adjustments: %s", e)
            return learned

        if not synergies:
            return learned

        # Map ops → categories
        op_to_cat: Dict[str, str] = {}
        for op_name in op_rates:
            try:
                op = get_primitive(op_name)
                op_to_cat[op_name] = op.category.value
            except (KeyError, ValueError):
                continue

        # Accumulate per-category synergy/anti-synergy signal
        cat_syn_lifts: Dict[str, List[float]] = defaultdict(list)
        cat_anti_lifts: Dict[str, List[float]] = defaultdict(list)

        for syn in synergies:
            cats = set()
            for op in (syn.op_a, syn.op_b):
                cat = op_to_cat.get(op)
                if cat:
                    cats.add(cat)
            if syn.label == "synergistic" and syn.lift > 1.5:
                for cat in cats:
                    cat_syn_lifts[cat].append(syn.lift)
            elif syn.label == "anti_synergistic" and syn.lift < 0.5:
                for cat in cats:
                    cat_anti_lifts[cat].append(syn.lift)

        n_boosted = 0
        n_penalized = 0
        for cat in learned:
            # Synergy boost: use best lift for this category
            if cat in cat_syn_lifts:
                best_lift = max(cat_syn_lifts[cat])
                # Clamped bonus: 1.0 + 0.2 * min(lift - 1.0, 3.0), max 1.6x
                bonus = 1.0 + 0.2 * min(best_lift - 1.0, 3.0)
                learned[cat] = round(learned[cat] * bonus, 2)
                n_boosted += 1
            # Anti-synergy penalty: use worst lift for this category
            if cat in cat_anti_lifts:
                worst_lift = min(cat_anti_lifts[cat])
                # Penalty: max(0.3, 1.0 - 0.3 * (1.0 - lift))
                penalty = max(0.3, 1.0 - 0.3 * (1.0 - worst_lift))
                learned[cat] = round(learned[cat] * penalty, 2)
                n_penalized += 1

        if n_boosted or n_penalized:
            logger.info(
                "Synergy adjustments: %d categories boosted, %d penalized "
                "(from %d synergistic, %d anti-synergistic pairs)",
                n_boosted,
                n_penalized,
                sum(len(v) for v in cat_syn_lifts.values()),
                sum(len(v) for v in cat_anti_lifts.values()),
            )

        return learned

    def get_current_grammar_weights(self) -> Dict[str, float]:
        """Get the default grammar weights for comparison."""
        return dict(GrammarConfig().category_weights)
