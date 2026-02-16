"""
Experiment Analytics — Learning Feedback Engine

Analyzes experiment history to learn which operations, structures, and
combinations correlate with success. Feeds back into grammar weights
to improve synthesis over time.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from ..synthesis.grammar import GrammarConfig
from ..synthesis.primitives import get_primitive
from .notebook import LabNotebook


class ExperimentAnalytics:
    """Data-driven analytics over experiment history."""

    def __init__(self, notebook: LabNotebook):
        self.nb = notebook

    _OP_NAME_PATTERN = re.compile(r'"op_name"\s*:\s*"([^"]+)"')

    @classmethod
    def _extract_ops_fast(cls, graph_json: str) -> Optional[List[str]]:
        """Fast-path op extraction from JSON string without full decode."""
        if not graph_json or '"op_name"' not in graph_json:
            return []
        ops = sorted({
            op for op in cls._OP_NAME_PATTERN.findall(graph_json)
            if op and op != "input"
        })
        return ops

    @staticmethod
    def _extract_ops_fallback(graph_json: str) -> Optional[List[str]]:
        """Robust fallback extraction using JSON decode."""
        try:
            graph_data = json.loads(graph_json)
            nodes = graph_data.get("nodes", {}) if isinstance(graph_data, dict) else {}
            return sorted({
                nd["op_name"]
                for nd in nodes.values()
                if isinstance(nd, dict) and nd.get("op_name") and nd["op_name"] != "input"
            })
        except (json.JSONDecodeError, TypeError, AttributeError):
            return None

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
                "avg_novelty_confidence": row.get("avg_novelty_confidence"),
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

    def _gather_category_stats(
        self, op_rates: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """Group op success rates by category."""
        cat_stats: Dict[str, Dict] = defaultdict(lambda: {
            "total": 0, "s1_total": 0, "novelty_sum": 0.0, "count": 0,
            "conf_sum": 0.0, "conf_count": 0,
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
            if stats.get("avg_novelty_confidence"):
                cat_stats[cat]["conf_sum"] += stats["avg_novelty_confidence"] * stats["n_used"]
                cat_stats[cat]["conf_count"] += stats["n_used"]
        return cat_stats

    def _compute_weights_from_stats(
        self, cat_stats: Dict[str, Dict],
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
            cat_novelties[cat] = (stats["novelty_sum"] / stats["count"]
                                  if stats["count"] > 0 else 0.0)
            cat_confidences[cat] = (stats["conf_sum"] / stats["conf_count"]
                                    if stats["conf_count"] > 0 else 0.0)

        if not cat_s1_rates:
            return None

        mean_s1 = sum(cat_s1_rates.values()) / len(cat_s1_rates)

        weights = {}
        for cat, s1_rate in cat_s1_rates.items():
            n = cat_stats[cat]["total"]
            relative = s1_rate / max(mean_s1, 0.01)

            # Statistical guard (#42): skip noisy differences
            se = math.sqrt(s1_rate * (1 - s1_rate) / n) if n > 0 and 0 < s1_rate < 1 else 0.0
            effect = abs(s1_rate - mean_s1)
            if se > 0 and effect < se:
                weights[cat] = default_weights.get(cat, 1.0)
                continue

            amplified = relative ** 2
            # Discount novelty factor by average confidence for this category
            # Low-confidence novelty (e.g. structural-only at 0.2) contributes
            # much less than high-confidence (full behavioral at 0.9)
            raw_novelty = cat_novelties.get(cat, 0.0)
            confidence = cat_confidences.get(cat, 0.0)
            novelty_factor = 1.0 + raw_novelty * confidence
            base = default_weights.get(cat, 1.0)
            weight = base * amplified * novelty_factor
            weights[cat] = round(max(0.1, min(8.0, weight)), 2)

        return weights if weights else None

    def compute_grammar_weights(self) -> Optional[Dict[str, float]]:
        """Compute learned category weights from historical success data.

        Uses the aggregate op_success_rates table. For holdout validation
        of the learned weights, call ``holdout_validation()`` separately.

        Returns a dict of category -> weight, or None if insufficient data.
        """
        op_rates = self.op_success_rates()
        if len(op_rates) < 5:
            return None

        cat_stats = self._gather_category_stats(op_rates)
        if not cat_stats:
            return None

        return self._compute_weights_from_stats(cat_stats)

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
        row = self.nb.conn.execute(f"""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) as s1_passed
            FROM program_results
            WHERE experiment_id IN ({placeholders})
        """, tuple(holdout_ids)).fetchone()

        total = row["total"] or 0
        s1 = row["s1_passed"] or 0
        return {
            "holdout_experiments": len(holdout_ids),
            "holdout_programs": total,
            "holdout_s1_passed": s1,
            "holdout_s1_rate": s1 / max(total, 1),
        }

    def experiment_clusters(self, n_clusters: int = 3) -> Optional[Dict]:
        """Cluster completed experiments by outcome profile.

        Uses deterministic k-means style clustering over normalized experiment
        features (S1 pass rate, best novelty, best loss ratio, duration), with
        model selection across candidate k values and consensus-based stability.
        Returns cluster summaries with compact quality diagnostics.
        """
        rows = self.nb.conn.execute("""
            SELECT experiment_id,
                   n_programs_generated,
                   n_stage1_passed,
                   best_novelty_score,
                   best_loss_ratio,
                   duration_seconds
            FROM experiments
            WHERE status = 'completed'
              AND n_programs_generated > 0
        """).fetchall()

        if len(rows) < 3:
            return None

        experiments = []
        for row in rows:
            total = row["n_programs_generated"] or 0
            s1 = row["n_stage1_passed"] or 0
            if total <= 0:
                continue
            experiments.append({
                "experiment_id": row["experiment_id"],
                "s1_rate": s1 / max(total, 1),
                "best_novelty": float(row["best_novelty_score"] or 0.0),
                "best_loss_ratio": float(row["best_loss_ratio"] or 1.0),
                "duration_seconds": float(row["duration_seconds"] or 0.0),
            })

        if len(experiments) < 3:
            return None

        exp_ids = [e["experiment_id"] for e in experiments]
        signatures_by_exp: Dict[str, Dict[str, float]] = {
            exp_id: {
                "compile_fail_rate": 0.0,
                "train_fail_rate": 0.0,
                "stage1_fail_rate": 0.0,
                "error_diversity": 0.0,
            }
            for exp_id in exp_ids
        }

        if exp_ids:
            placeholders = ",".join("?" * len(exp_ids))
            failure_rows = self.nb.conn.execute(f"""
                SELECT experiment_id,
                       COUNT(*) as n_total,
                       SUM(CASE WHEN COALESCE(stage0_passed, 0) = 0 THEN 1 ELSE 0 END) as n_compile_fail,
                       SUM(CASE WHEN COALESCE(stage0_passed, 0) = 1 AND COALESCE(stage05_passed, 0) = 0 THEN 1 ELSE 0 END) as n_train_fail,
                       SUM(CASE WHEN COALESCE(stage05_passed, 0) = 1 AND COALESCE(stage1_passed, 0) = 0 THEN 1 ELSE 0 END) as n_stage1_fail
                FROM program_results
                WHERE experiment_id IN ({placeholders})
                GROUP BY experiment_id
            """, tuple(exp_ids)).fetchall()

            error_rows = self.nb.conn.execute(f"""
                SELECT experiment_id,
                       error_type,
                       COUNT(*) as n
                FROM program_results
                WHERE experiment_id IN ({placeholders})
                  AND error_type IS NOT NULL
                  AND TRIM(error_type) != ''
                GROUP BY experiment_id, error_type
            """, tuple(exp_ids)).fetchall()

            error_counts_by_exp: Dict[str, Dict[str, int]] = defaultdict(dict)
            for row in error_rows:
                error_counts_by_exp[row["experiment_id"]][row["error_type"]] = int(row["n"] or 0)

            for row in failure_rows:
                exp_id = row["experiment_id"]
                n_total = float(row["n_total"] or 0)
                if n_total <= 0:
                    continue
                signatures_by_exp[exp_id] = {
                    "compile_fail_rate": float(row["n_compile_fail"] or 0) / n_total,
                    "train_fail_rate": float(row["n_train_fail"] or 0) / n_total,
                    "stage1_fail_rate": float(row["n_stage1_fail"] or 0) / n_total,
                    "error_diversity": 0.0,
                }

                err_counts = error_counts_by_exp.get(exp_id, {})
                total_err = float(sum(err_counts.values()))
                if total_err > 0 and len(err_counts) > 1:
                    entropy = 0.0
                    for count in err_counts.values():
                        p = count / total_err
                        if p > 0:
                            entropy -= p * math.log(p)
                    max_entropy = math.log(len(err_counts))
                    if max_entropy > 0:
                        signatures_by_exp[exp_id]["error_diversity"] = entropy / max_entropy

        for e in experiments:
            sig = signatures_by_exp.get(e["experiment_id"], {})
            e["compile_fail_rate"] = float(sig.get("compile_fail_rate", 0.0))
            e["train_fail_rate"] = float(sig.get("train_fail_rate", 0.0))
            e["stage1_fail_rate"] = float(sig.get("stage1_fail_rate", 0.0))
            e["error_diversity"] = float(sig.get("error_diversity", 0.0))

        trajectory_by_exp: Dict[str, Dict[str, float]] = {
            exp_id: {
                "stage1_momentum": 0.0,
                "novelty_momentum": 0.0,
                "loss_improvement_momentum": 0.0,
                "outcome_volatility": 0.0,
                "outcome_peak_timing": 0.0,
                "recovery_lag": 0.0,
                "stage1_transition_timing": 0.0,
                "primary_change_point_timing": 0.0,
                "stage1_transition_density": 0.0,
                "change_point_confidence": 0.0,
                "windowed_change_dispersion": 0.0,
                "window_change_localization": 0.0,
                "transition_gap_entropy": 0.0,
            }
            for exp_id in exp_ids
        }

        if exp_ids:
            placeholders = ",".join("?" * len(exp_ids))
            seq_rows = self.nb.conn.execute(f"""
                SELECT experiment_id,
                       timestamp,
                       stage1_passed,
                       loss_ratio,
                       novelty_score
                FROM program_results
                WHERE experiment_id IN ({placeholders})
                ORDER BY experiment_id ASC, timestamp ASC
            """, tuple(exp_ids)).fetchall()

            per_exp_seq: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
            for row in seq_rows:
                stage1 = float(row["stage1_passed"] or 0.0)
                novelty = float(row["novelty_score"] or 0.0)
                loss_ratio = float(row["loss_ratio"] or 1.0)
                per_exp_seq[row["experiment_id"]].append((stage1, novelty, loss_ratio))

            def _window_means(values: List[float]) -> Tuple[float, float]:
                if not values:
                    return 0.0, 0.0
                window = max(1, len(values) // 3)
                early = values[:window]
                late = values[-window:]
                return (sum(early) / len(early), sum(late) / len(late))

            for exp_id, seq in per_exp_seq.items():
                if len(seq) < 2:
                    continue

                stage1_values = [item[0] for item in seq]
                novelty_values = [item[1] for item in seq]
                loss_values = [item[2] for item in seq]

                early_s1, late_s1 = _window_means(stage1_values)
                early_nov, late_nov = _window_means(novelty_values)
                early_loss, late_loss = _window_means(loss_values)

                outcome_proxy = [
                    (0.5 * s1) + (0.3 * nov) + (0.2 * (1.0 / (1.0 + max(lr, 1e-9))))
                    for s1, nov, lr in seq
                ]
                proxy_mean = sum(outcome_proxy) / len(outcome_proxy)
                proxy_var = sum((x - proxy_mean) ** 2 for x in outcome_proxy) / len(outcome_proxy)
                peak_idx = max(range(len(outcome_proxy)), key=lambda idx: outcome_proxy[idx])
                trough_idx = min(range(len(outcome_proxy)), key=lambda idx: outcome_proxy[idx])
                normalizer = max(len(outcome_proxy) - 1, 1)
                peak_timing = peak_idx / normalizer

                transition_positions: List[int] = []
                for idx in range(1, len(stage1_values)):
                    if stage1_values[idx] != stage1_values[idx - 1]:
                        transition_positions.append(idx)

                first_transition_idx = transition_positions[0] if transition_positions else None
                stage1_transition_timing = (
                    (first_transition_idx / normalizer) if first_transition_idx is not None else 0.0
                )
                stage1_transition_density = (
                    len(transition_positions) / normalizer if normalizer > 0 else 0.0
                )

                if len(transition_positions) >= 2:
                    transition_gaps = [
                        transition_positions[i] - transition_positions[i - 1]
                        for i in range(1, len(transition_positions))
                    ]
                    total_gap = float(sum(transition_gaps))
                    if total_gap > 0:
                        gap_entropy = 0.0
                        for gap in transition_gaps:
                            p = gap / total_gap
                            if p > 0:
                                gap_entropy -= p * math.log(p)
                        max_entropy = math.log(len(transition_gaps)) if len(transition_gaps) > 1 else 0.0
                        transition_gap_entropy = (gap_entropy / max_entropy) if max_entropy > 0 else 0.0
                    else:
                        transition_gap_entropy = 0.0
                else:
                    transition_gap_entropy = 0.0

                if len(outcome_proxy) > 1:
                    deltas = [
                        abs(outcome_proxy[i] - outcome_proxy[i - 1])
                        for i in range(1, len(outcome_proxy))
                    ]
                    cp_idx = 1 + max(range(len(deltas)), key=lambda idx: deltas[idx])
                    primary_change_point_timing = cp_idx / normalizer
                    max_delta = max(deltas) if deltas else 0.0
                    total_delta = sum(deltas)
                    change_point_confidence = max_delta / total_delta if total_delta > 1e-9 else 0.0
                else:
                    primary_change_point_timing = 0.0
                    change_point_confidence = 0.0

                if len(outcome_proxy) > 2:
                    deltas = [
                        abs(outcome_proxy[i] - outcome_proxy[i - 1])
                        for i in range(1, len(outcome_proxy))
                    ]
                    n_deltas = len(deltas)
                    seg = max(1, n_deltas // 3)
                    window_slices = [
                        deltas[:seg],
                        deltas[seg:2 * seg],
                        deltas[2 * seg:],
                    ]
                    window_means = [
                        (sum(chunk) / len(chunk)) if chunk else 0.0
                        for chunk in window_slices
                    ]
                    mean_change = sum(window_means) / len(window_means)
                    variance = sum((w - mean_change) ** 2 for w in window_means) / len(window_means)
                    windowed_change_dispersion = math.sqrt(max(variance, 0.0))
                    total_window_change = sum(window_means)
                    window_change_localization = (
                        (max(window_means) / total_window_change)
                        if total_window_change > 1e-9
                        else 0.0
                    )
                else:
                    windowed_change_dispersion = 0.0
                    window_change_localization = 0.0

                early_window = max(1, len(outcome_proxy) // 3)
                early_baseline = sum(outcome_proxy[:early_window]) / early_window
                recovery_idx = None
                for idx in range(trough_idx + 1, len(outcome_proxy)):
                    if outcome_proxy[idx] >= early_baseline:
                        recovery_idx = idx
                        break
                if recovery_idx is None:
                    recovery_lag = 1.0 if len(outcome_proxy) > 1 else 0.0
                else:
                    recovery_steps = recovery_idx - trough_idx
                    recovery_lag = recovery_steps / normalizer

                trajectory_by_exp[exp_id] = {
                    "stage1_momentum": late_s1 - early_s1,
                    "novelty_momentum": late_nov - early_nov,
                    "loss_improvement_momentum": early_loss - late_loss,
                    "outcome_volatility": math.sqrt(max(proxy_var, 0.0)),
                    "outcome_peak_timing": peak_timing,
                    "recovery_lag": recovery_lag,
                    "stage1_transition_timing": stage1_transition_timing,
                    "primary_change_point_timing": primary_change_point_timing,
                    "stage1_transition_density": stage1_transition_density,
                    "change_point_confidence": change_point_confidence,
                    "windowed_change_dispersion": windowed_change_dispersion,
                    "window_change_localization": window_change_localization,
                    "transition_gap_entropy": transition_gap_entropy,
                }

        for e in experiments:
            traj = trajectory_by_exp.get(e["experiment_id"], {})
            e["stage1_momentum"] = float(traj.get("stage1_momentum", 0.0))
            e["novelty_momentum"] = float(traj.get("novelty_momentum", 0.0))
            e["loss_improvement_momentum"] = float(traj.get("loss_improvement_momentum", 0.0))
            e["outcome_volatility"] = float(traj.get("outcome_volatility", 0.0))
            e["outcome_peak_timing"] = float(traj.get("outcome_peak_timing", 0.0))
            e["recovery_lag"] = float(traj.get("recovery_lag", 0.0))
            e["stage1_transition_timing"] = float(traj.get("stage1_transition_timing", 0.0))
            e["primary_change_point_timing"] = float(traj.get("primary_change_point_timing", 0.0))
            e["stage1_transition_density"] = float(traj.get("stage1_transition_density", 0.0))
            e["change_point_confidence"] = float(traj.get("change_point_confidence", 0.0))
            e["windowed_change_dispersion"] = float(traj.get("windowed_change_dispersion", 0.0))
            e["window_change_localization"] = float(traj.get("window_change_localization", 0.0))
            e["transition_gap_entropy"] = float(traj.get("transition_gap_entropy", 0.0))

        feature_keys = [
            "s1_rate",
            "best_novelty",
            "best_loss_ratio",
            "duration_seconds",
            "compile_fail_rate",
            "train_fail_rate",
            "stage1_fail_rate",
            "error_diversity",
            "stage1_momentum",
            "novelty_momentum",
            "loss_improvement_momentum",
            "outcome_volatility",
            "outcome_peak_timing",
            "recovery_lag",
            "stage1_transition_timing",
            "primary_change_point_timing",
            "stage1_transition_density",
            "change_point_confidence",
            "windowed_change_dispersion",
            "window_change_localization",
            "transition_gap_entropy",
        ]
        mins = {k: min(e[k] for e in experiments) for k in feature_keys}
        maxs = {k: max(e[k] for e in experiments) for k in feature_keys}

        def _norm(v: float, k: str) -> float:
            lo, hi = mins[k], maxs[k]
            if hi <= lo:
                return 0.0
            return (v - lo) / (hi - lo)

        points = []
        for e in experiments:
            # Lower loss_ratio is better, so invert after normalization.
            loss_norm = 1.0 - _norm(e["best_loss_ratio"], "best_loss_ratio")
            vec = [
                _norm(e["s1_rate"], "s1_rate"),
                _norm(e["best_novelty"], "best_novelty"),
                loss_norm,
                _norm(e["duration_seconds"], "duration_seconds"),
                _norm(e["compile_fail_rate"], "compile_fail_rate"),
                _norm(e["train_fail_rate"], "train_fail_rate"),
                _norm(e["stage1_fail_rate"], "stage1_fail_rate"),
                _norm(e["error_diversity"], "error_diversity"),
                _norm(e["stage1_momentum"], "stage1_momentum"),
                _norm(e["novelty_momentum"], "novelty_momentum"),
                _norm(e["loss_improvement_momentum"], "loss_improvement_momentum"),
                _norm(e["outcome_volatility"], "outcome_volatility"),
                _norm(e["outcome_peak_timing"], "outcome_peak_timing"),
                _norm(e["recovery_lag"], "recovery_lag"),
                _norm(e["stage1_transition_timing"], "stage1_transition_timing"),
                _norm(e["primary_change_point_timing"], "primary_change_point_timing"),
                _norm(e["stage1_transition_density"], "stage1_transition_density"),
                _norm(e["change_point_confidence"], "change_point_confidence"),
                _norm(e["windowed_change_dispersion"], "windowed_change_dispersion"),
                _norm(e["window_change_localization"], "window_change_localization"),
                _norm(e["transition_gap_entropy"], "transition_gap_entropy"),
            ]
            points.append((e, vec))

        def _sq_dist(a: List[float], b: List[float]) -> float:
            return sum((x - y) ** 2 for x, y in zip(a, b))

        n_points = len(points)
        max_k = min(max(2, n_clusters), min(6, n_points - 1))
        if max_k < 2:
            return None

        distance_matrix: List[List[float]] = [[0.0] * n_points for _ in range(n_points)]
        for i in range(n_points):
            for j in range(i + 1, n_points):
                d = math.sqrt(_sq_dist(points[i][1], points[j][1]))
                distance_matrix[i][j] = d
                distance_matrix[j][i] = d

        dataset_signature = "|".join(sorted(p[0]["experiment_id"] for p in points))

        def _init_centroids(k_value: int, salt: int) -> List[List[float]]:
            seed_hex = hashlib.md5(f"{dataset_signature}:{salt}".encode()).hexdigest()
            first_idx = int(seed_hex[:8], 16) % n_points
            chosen_idxs = [first_idx]
            centroids_local = [list(points[first_idx][1])]

            while len(centroids_local) < k_value:
                farthest_idx = max(
                    range(n_points),
                    key=lambda idx: min(_sq_dist(points[idx][1], c) for c in centroids_local),
                )
                if farthest_idx in chosen_idxs:
                    remaining = [idx for idx in range(n_points) if idx not in chosen_idxs]
                    if not remaining:
                        break
                    farthest_idx = remaining[0]
                chosen_idxs.append(farthest_idx)
                centroids_local.append(list(points[farthest_idx][1]))
            return centroids_local

        def _run_kmeans(k_value: int, salt: int) -> Dict:
            centroids_local = _init_centroids(k_value, salt)
            assignments_local: List[int] = [-1] * n_points

            for _ in range(30):
                changed = False
                for i, (_, vec) in enumerate(points):
                    nearest_idx = min(
                        range(k_value), key=lambda ci: _sq_dist(vec, centroids_local[ci])
                    )
                    if assignments_local[i] != nearest_idx:
                        assignments_local[i] = nearest_idx
                        changed = True

                new_centroids: List[List[float]] = []
                for ci in range(k_value):
                    members = [points[i][1] for i in range(n_points) if assignments_local[i] == ci]
                    if not members:
                        new_centroids.append(list(centroids_local[ci]))
                        continue
                    dim = len(members[0])
                    new_centroids.append([
                        sum(m[d] for m in members) / len(members) for d in range(dim)
                    ])
                centroids_local = new_centroids
                if not changed:
                    break

            inertia = sum(
                _sq_dist(points[i][1], centroids_local[assignments_local[i]])
                for i in range(n_points)
            )
            return {
                "assignments": assignments_local,
                "centroids": centroids_local,
                "inertia": inertia,
            }

        def _silhouette(assignments_local: List[int]) -> float:
            unique_clusters = sorted(set(assignments_local))
            if len(unique_clusters) < 2:
                return 0.0

            cluster_members = {
                c: [i for i, a in enumerate(assignments_local) if a == c]
                for c in unique_clusters
            }
            silhouettes: List[float] = []

            for i in range(n_points):
                c_i = assignments_local[i]
                same_cluster = [j for j in cluster_members[c_i] if j != i]
                if not same_cluster:
                    silhouettes.append(0.0)
                    continue

                a_i = sum(distance_matrix[i][j] for j in same_cluster) / len(same_cluster)
                b_i = float("inf")
                for c in unique_clusters:
                    if c == c_i or not cluster_members[c]:
                        continue
                    avg_dist = sum(distance_matrix[i][j] for j in cluster_members[c]) / len(cluster_members[c])
                    if avg_dist < b_i:
                        b_i = avg_dist

                if not math.isfinite(b_i):
                    silhouettes.append(0.0)
                    continue
                denom = max(a_i, b_i, 1e-9)
                silhouettes.append((b_i - a_i) / denom)

            return sum(silhouettes) / len(silhouettes) if silhouettes else 0.0

        def _imbalance(assignments_local: List[int], k_value: int) -> float:
            counts = [0] * k_value
            for a in assignments_local:
                counts[a] += 1
            ideal = n_points / max(k_value, 1)
            return sum(abs(c - ideal) for c in counts) / max(2.0 * n_points, 1.0)

        runs_per_k = 4
        candidates: List[Dict] = []
        for k_value in range(2, max_k + 1):
            runs = []
            for salt in range(runs_per_k):
                run = _run_kmeans(k_value, salt)
                silhouette = _silhouette(run["assignments"])
                imbalance = _imbalance(run["assignments"], k_value)
                quality = silhouette - (0.15 * imbalance)
                run.update({
                    "silhouette": silhouette,
                    "imbalance": imbalance,
                    "quality": quality,
                })
                runs.append(run)

            best_run = max(runs, key=lambda r: (r["quality"], -r["inertia"]))
            candidates.append({
                "k": k_value,
                "best": best_run,
                "runs": runs,
                "score": best_run["quality"],
            })

        selected = max(candidates, key=lambda c: (c["score"], -c["k"]))
        k = selected["k"]
        assignments = selected["best"]["assignments"]
        centroids = selected["best"]["centroids"]

        def _coassociation_agreement(a1: List[int], a2: List[int]) -> float:
            pair_total = n_points * (n_points - 1) // 2
            if pair_total <= 0:
                return 1.0
            agree = 0
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    same_1 = a1[i] == a1[j]
                    same_2 = a2[i] == a2[j]
                    if same_1 == same_2:
                        agree += 1
            return agree / pair_total

        selected_runs = selected["runs"]
        consensus_scores: List[float] = []
        for i in range(len(selected_runs)):
            for j in range(i + 1, len(selected_runs)):
                consensus_scores.append(
                    _coassociation_agreement(
                        selected_runs[i]["assignments"],
                        selected_runs[j]["assignments"],
                    )
                )
        consensus = sum(consensus_scores) / len(consensus_scores) if consensus_scores else 1.0

        clusters = []
        intra_dists: List[float] = []
        for ci in range(k):
            members = [points[i] for i in range(len(points)) if assignments[i] == ci]
            if not members:
                continue

            member_exps = [m[0] for m in members]
            centroid = centroids[ci]
            dists = [math.sqrt(_sq_dist(m[1], centroid)) for m in members]
            intra_dists.extend(dists)

            clusters.append({
                "cluster_id": ci,
                "size": len(member_exps),
                "avg_s1_rate": round(sum(m["s1_rate"] for m in member_exps) / len(member_exps), 4),
                "avg_best_novelty": round(sum(m["best_novelty"] for m in member_exps) / len(member_exps), 4),
                "avg_best_loss_ratio": round(sum(m["best_loss_ratio"] for m in member_exps) / len(member_exps), 4),
                "avg_duration_seconds": round(sum(m["duration_seconds"] for m in member_exps) / len(member_exps), 2),
                "avg_compile_fail_rate": round(sum(m["compile_fail_rate"] for m in member_exps) / len(member_exps), 4),
                "avg_train_fail_rate": round(sum(m["train_fail_rate"] for m in member_exps) / len(member_exps), 4),
                "avg_stage1_fail_rate": round(sum(m["stage1_fail_rate"] for m in member_exps) / len(member_exps), 4),
                "avg_error_diversity": round(sum(m["error_diversity"] for m in member_exps) / len(member_exps), 4),
                "avg_stage1_momentum": round(sum(m["stage1_momentum"] for m in member_exps) / len(member_exps), 4),
                "avg_novelty_momentum": round(sum(m["novelty_momentum"] for m in member_exps) / len(member_exps), 4),
                "avg_loss_improvement_momentum": round(sum(m["loss_improvement_momentum"] for m in member_exps) / len(member_exps), 4),
                "avg_outcome_volatility": round(sum(m["outcome_volatility"] for m in member_exps) / len(member_exps), 4),
                "avg_outcome_peak_timing": round(sum(m["outcome_peak_timing"] for m in member_exps) / len(member_exps), 4),
                "avg_recovery_lag": round(sum(m["recovery_lag"] for m in member_exps) / len(member_exps), 4),
                "avg_stage1_transition_timing": round(sum(m["stage1_transition_timing"] for m in member_exps) / len(member_exps), 4),
                "avg_primary_change_point_timing": round(sum(m["primary_change_point_timing"] for m in member_exps) / len(member_exps), 4),
                "avg_stage1_transition_density": round(sum(m["stage1_transition_density"] for m in member_exps) / len(member_exps), 4),
                "avg_change_point_confidence": round(sum(m["change_point_confidence"] for m in member_exps) / len(member_exps), 4),
                "avg_windowed_change_dispersion": round(sum(m["windowed_change_dispersion"] for m in member_exps) / len(member_exps), 4),
                "avg_window_change_localization": round(sum(m["window_change_localization"] for m in member_exps) / len(member_exps), 4),
                "avg_transition_gap_entropy": round(sum(m["transition_gap_entropy"] for m in member_exps) / len(member_exps), 4),
                "experiment_ids": [m["experiment_id"] for m in member_exps[:10]],
            })

        clusters.sort(key=lambda c: c["avg_s1_rate"], reverse=True)

        # Generate plain-language description for each cluster
        for c in clusters:
            c["description"] = self._describe_cluster(c)

        inter_centroid = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                inter_centroid.append(math.sqrt(_sq_dist(centroids[i], centroids[j])))

        avg_intra = sum(intra_dists) / len(intra_dists) if intra_dists else 0.0
        min_inter = min(inter_centroid) if inter_centroid else 0.0
        separation = min_inter / (min_inter + avg_intra + 1e-9)
        stability = (0.6 * separation) + (0.4 * consensus)

        ordered_candidate_scores = sorted(candidates, key=lambda c: c["score"], reverse=True)
        selected_margin = 0.0
        if len(ordered_candidate_scores) > 1:
            selected_margin = ordered_candidate_scores[0]["score"] - ordered_candidate_scores[1]["score"]

        return {
            "n_experiments": len(points),
            "n_clusters": len(clusters),
            "feature_keys": [
                "s1_rate",
                "best_novelty",
                "best_loss_inverse",
                "duration_seconds",
                "compile_fail_rate",
                "train_fail_rate",
                "stage1_fail_rate",
                "error_diversity",
                "stage1_momentum",
                "novelty_momentum",
                "loss_improvement_momentum",
                "outcome_volatility",
                "outcome_peak_timing",
                "recovery_lag",
                "stage1_transition_timing",
                "primary_change_point_timing",
                "stage1_transition_density",
                "change_point_confidence",
                "windowed_change_dispersion",
                "window_change_localization",
                "transition_gap_entropy",
            ],
            "stability_score": round(max(0.0, min(1.0, stability)), 4),
            "model_selection": {
                "candidate_ks": [c["k"] for c in candidates],
                "selected_k": k,
                "silhouette": round(selected["best"]["silhouette"], 4),
                "consensus": round(max(0.0, min(1.0, consensus)), 4),
                "selection_margin": round(selected_margin, 4),
            },
            "clusters": clusters,
        }

    @staticmethod
    def _describe_cluster(c: Dict) -> str:
        """Generate a plain-language summary for a cluster."""
        size = c.get("size", 0)
        s1_pct = (c.get("avg_s1_rate", 0) or 0) * 100
        novelty = c.get("avg_best_novelty", 0) or 0
        loss_ratio = c.get("avg_best_loss_ratio", 0) or 0
        compile_fail = (c.get("avg_compile_fail_rate", 0) or 0) * 100

        # Characterize S1 rate
        if s1_pct >= 30:
            s1_desc = f"high S1 pass rate ({s1_pct:.0f}%)"
        elif s1_pct >= 10:
            s1_desc = f"moderate S1 pass rate ({s1_pct:.0f}%)"
        elif s1_pct > 0:
            s1_desc = f"low S1 pass rate ({s1_pct:.0f}%)"
        else:
            s1_desc = "no S1 survivors"

        # Characterize novelty
        if novelty >= 0.7:
            nov_desc = "high novelty"
        elif novelty >= 0.3:
            nov_desc = "moderate novelty"
        else:
            nov_desc = "low novelty"

        # Characterize loss ratio
        if loss_ratio < 0.8:
            loss_desc = "strong loss improvement"
        elif loss_ratio < 1.0:
            loss_desc = "some loss improvement"
        else:
            loss_desc = "no loss improvement over baseline"

        # Determine cluster character
        if s1_pct >= 20 and loss_ratio < 0.9:
            character = "the productive cluster"
        elif s1_pct >= 10:
            character = "a moderately productive cluster"
        elif compile_fail >= 50:
            character = "a high-failure cluster"
        else:
            character = "an exploratory cluster"

        return (
            f"{size} experiments with {s1_desc}, {nov_desc}, "
            f"and {loss_desc} \u2014 {character}."
        )

    @staticmethod
    def _explain_routing_health(by_mode: List[Dict], total_programs: int,
                                overall_stage1_pass_rate: float) -> str:
        """Generate deterministic plain-language routing interpretation."""
        if not by_mode:
            return (
                "No routing telemetry available yet. Run routed architectures "
                "to estimate drop rate, utilization balance, and confidence."
            )

        def weighted_mean(key: str) -> Optional[float]:
            weighted_sum = 0.0
            weighted_n = 0
            for row in by_mode:
                value = row.get(key)
                n_programs = int(row.get("n_programs") or 0)
                if value is None or n_programs <= 0:
                    continue
                weighted_sum += float(value) * n_programs
                weighted_n += n_programs
            if weighted_n == 0:
                return None
            return weighted_sum / weighted_n

        avg_drop = weighted_mean("avg_drop_rate")
        avg_entropy = weighted_mean("avg_utilization_entropy")
        avg_confidence = weighted_mean("avg_confidence_mean")

        if avg_drop is None:
            drop_desc = "unknown"
            drop_text = "drop rate is unavailable"
        elif avg_drop <= 0.05:
            drop_desc = "low"
            drop_text = f"drop rate is low ({avg_drop * 100:.1f}%)"
        elif avg_drop <= 0.15:
            drop_desc = "moderate"
            drop_text = f"drop rate is moderate ({avg_drop * 100:.1f}%)"
        else:
            drop_desc = "high"
            drop_text = f"drop rate is high ({avg_drop * 100:.1f}%)"

        if avg_entropy is None:
            entropy_text = "utilization balance is unavailable"
        elif avg_entropy >= 1.2:
            entropy_text = f"utilization appears balanced (entropy {avg_entropy:.2f})"
        elif avg_entropy >= 0.8:
            entropy_text = f"utilization is moderately balanced (entropy {avg_entropy:.2f})"
        else:
            entropy_text = f"utilization looks concentrated (entropy {avg_entropy:.2f})"

        if avg_confidence is None:
            confidence_desc = "unknown"
            confidence_text = "confidence is unavailable"
        elif avg_confidence >= 0.7:
            confidence_desc = "strong"
            confidence_text = f"confidence is strong ({avg_confidence:.2f})"
        elif avg_confidence >= 0.5:
            confidence_desc = "moderate"
            confidence_text = f"confidence is moderate ({avg_confidence:.2f})"
        else:
            confidence_desc = "weak"
            confidence_text = f"confidence is weak ({avg_confidence:.2f})"

        best_mode = max(by_mode, key=lambda r: float(r.get("stage1_pass_rate") or 0.0))
        sentence_1 = (
            f"Across {total_programs} routed programs, overall Stage 1 pass rate is "
            f"{overall_stage1_pass_rate * 100:.1f}% and {drop_text}."
        )
        sentence_2 = (
            f"Routing {entropy_text}, while {confidence_text}. "
            f"Best-performing mode is '{best_mode.get('routing_mode', 'unknown')}' "
            f"at {float(best_mode.get('stage1_pass_rate') or 0.0) * 100:.1f}% S1 pass."
        )

        if drop_desc == "high":
            sentence_3 = "High drop suggests routing-capacity pressure; consider reducing overflow and token skipping."
        elif confidence_desc == "weak":
            sentence_3 = "Low confidence suggests uncertain routing choices; prioritize modes with steadier confidence and lower variance."
        elif avg_entropy is not None and avg_entropy < 0.8:
            sentence_3 = "Low entropy suggests expert over-concentration; improve utilization balance to reduce mode-collapse risk."
        else:
            sentence_3 = "Telemetry suggests routing is reasonably healthy and balanced across current modes."

        return f"{sentence_1} {sentence_2} {sentence_3}"

    def routing_health(self) -> Dict:
        """Aggregate routing telemetry by routing mode.

        Returns structured defaults when routing telemetry is not yet available,
        so dashboard/API consumers can safely render partial states.
        """
        rows = self.nb.conn.execute("""
            SELECT
                COALESCE(routing_mode, 'unknown') as routing_mode,
                COUNT(*) as n_programs,
                SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) as n_stage1_passed,
                AVG(loss_ratio) as avg_loss_ratio,
                AVG(routing_tokens_total) as avg_tokens_total,
                AVG(routing_tokens_processed) as avg_tokens_processed,
                AVG(routing_tokens_skipped) as avg_tokens_skipped,
                AVG(routing_drop_rate) as avg_drop_rate,
                AVG(routing_utilization_entropy) as avg_utilization_entropy,
                AVG(routing_capacity_overflow_count) as avg_capacity_overflow_count,
                AVG(routing_confidence_mean) as avg_confidence_mean,
                AVG(routing_confidence_std) as avg_confidence_std
            FROM program_results
            WHERE routing_mode IS NOT NULL
            GROUP BY COALESCE(routing_mode, 'unknown')
            ORDER BY n_programs DESC
        """).fetchall()

        if not rows:
            return {
                "available": False,
                "n_modes": 0,
                "total_programs": 0,
                "by_mode": [],
                "explanation": (
                    "No routing telemetry available yet. Run routed architectures "
                    "to estimate drop rate, utilization balance, and confidence."
                ),
            }

        by_mode = []
        total_programs = 0
        total_stage1 = 0

        for row in rows:
            n_programs = row["n_programs"] or 0
            n_stage1 = row["n_stage1_passed"] or 0
            total_programs += n_programs
            total_stage1 += n_stage1
            by_mode.append({
                "routing_mode": row["routing_mode"],
                "n_programs": n_programs,
                "stage1_pass_rate": n_stage1 / max(n_programs, 1),
                "avg_loss_ratio": row["avg_loss_ratio"],
                "avg_tokens_total": row["avg_tokens_total"],
                "avg_tokens_processed": row["avg_tokens_processed"],
                "avg_tokens_skipped": row["avg_tokens_skipped"],
                "avg_drop_rate": row["avg_drop_rate"],
                "avg_utilization_entropy": row["avg_utilization_entropy"],
                "avg_capacity_overflow_count": row["avg_capacity_overflow_count"],
                "avg_confidence_mean": row["avg_confidence_mean"],
                "avg_confidence_std": row["avg_confidence_std"],
            })

        overall_stage1_pass_rate = total_stage1 / max(total_programs, 1)

        return {
            "available": True,
            "n_modes": len(by_mode),
            "total_programs": total_programs,
            "overall_stage1_pass_rate": overall_stage1_pass_rate,
            "by_mode": by_mode,
            "explanation": self._explain_routing_health(
                by_mode=by_mode,
                total_programs=total_programs,
                overall_stage1_pass_rate=overall_stage1_pass_rate,
            ),
        }

    def control_experiment_comparison(self) -> Optional[Dict]:
        """Compare control experiments (default weights) vs learned-weight experiments.

        Control experiments are flagged with ``control_experiment = true`` in
        config_json.  This method computes S1 rates for each group and a
        z-test for the difference, providing evidence for or against the
        hypothesis that learned grammar weights improve synthesis quality.

        Returns None if fewer than 2 control experiments exist.
        """
        rows = self.nb.conn.execute("""
            SELECT experiment_id, config_json
            FROM experiments WHERE status = 'completed' AND config_json IS NOT NULL
        """).fetchall()

        control_ids: list[str] = []
        learned_ids: list[str] = []
        for row in rows:
            try:
                cfg = json.loads(row["config_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            if cfg.get("control_experiment"):
                control_ids.append(row["experiment_id"])
            else:
                learned_ids.append(row["experiment_id"])

        if len(control_ids) < 2 or len(learned_ids) < 2:
            return None

        def _s1_stats(exp_ids: list[str]) -> dict:
            ph = ",".join("?" * len(exp_ids))
            r = self.nb.conn.execute(f"""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) as s1
                FROM program_results WHERE experiment_id IN ({ph})
            """, tuple(exp_ids)).fetchone()
            total = r["total"] or 0
            s1 = r["s1"] or 0
            return {"experiments": len(exp_ids), "programs": total,
                    "s1_passed": s1, "s1_rate": s1 / max(total, 1)}

        control = _s1_stats(control_ids)
        learned = _s1_stats(learned_ids)

        # Two-proportion z-test
        p_c = control["s1_rate"]
        p_l = learned["s1_rate"]
        n_c = control["programs"]
        n_l = learned["programs"]
        pooled_p = (control["s1_passed"] + learned["s1_passed"]) / max(n_c + n_l, 1)
        se = math.sqrt(pooled_p * (1 - pooled_p) * (1 / max(n_c, 1) + 1 / max(n_l, 1))) if 0 < pooled_p < 1 else 0.0
        z_score = (p_l - p_c) / se if se > 0 else 0.0

        return {
            "control": control,
            "learned": learned,
            "s1_rate_difference": round(p_l - p_c, 4),
            "z_score": round(z_score, 3),
            "significant_at_p05": abs(z_score) > 1.96,
            "learned_is_better": p_l > p_c,
            "interpretation": (
                "Learned weights significantly outperform controls"
                if z_score > 1.96
                else "Learned weights significantly underperform controls"
                if z_score < -1.96
                else "No significant difference between learned and control weights"
            ),
        }

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
        ops_cache: Dict[str, Optional[List[str]]] = {}

        for r in rows:
            graph_json = r["graph_json"]
            if graph_json in ops_cache:
                ops = ops_cache[graph_json]
            else:
                ops = self._extract_ops_fast(graph_json)
                if ops is None:
                    ops = self._extract_ops_fallback(graph_json)
                ops_cache[graph_json] = ops

            if not ops:
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
                   loss_ratio, baseline_loss_ratio, graph_json
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

        # Extract ops list from graph_json
        for p in programs:
            ops = []
            try:
                if p.get("graph_json"):
                    graph = json.loads(p["graph_json"])
                    nodes = graph.get("nodes")
                    if isinstance(nodes, dict):
                        node_iter = nodes.values()
                    elif isinstance(nodes, list):
                        node_iter = nodes
                    else:
                        node_iter = []
                    for n in node_iter:
                        if isinstance(n, dict) and n.get("op"):
                            ops.append(n["op"])
            except (json.JSONDecodeError, TypeError):
                pass
            p["ops"] = ops
            p.pop("graph_json", None)

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
