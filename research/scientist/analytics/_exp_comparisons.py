"""Comparison, failure pattern, op combination, and learning trajectory mixin."""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from research.scientist.intelligence.ml_corpus import load_deduped_graph_analysis_rows

logger = logging.getLogger(__name__)


class _ComparisonsMixin:
    """Control experiment comparison, failure patterns, op combos, trajectory."""

    __slots__ = ()

    def control_experiment_comparison(self) -> Optional[Dict]:
        """Compare control experiments (default weights) vs learned-weight experiments.

        Control experiments are flagged with ``control_experiment = true`` in
        config_json.  Uses time-matched comparison: each control experiment is
        paired with temporally adjacent learned experiments (the nearest before
        and after) to avoid confounding from increasing search difficulty over
        a session.

        Returns None if fewer than 2 control experiments exist.
        """
        control_exps, learned_exps = self._load_control_learned_exps()
        if len(control_exps) < 2 or len(learned_exps) < 2:
            return None

        all_exp_ids = list(
            {eid for eid, _ in control_exps} | {eid for eid, _ in learned_exps}
        )
        s1_cache = self._s1_bulk(all_exp_ids)

        pair_diffs = self._time_matched_diffs(control_exps, learned_exps, s1_cache)

        all_control_ids = [eid for eid, _ in control_exps]
        all_learned_ids = [eid for eid, _ in learned_exps]
        control = self._s1_stats(all_control_ids, s1_cache)
        learned = self._s1_stats(all_learned_ids, s1_cache)

        matched_diff, matched_z = self._paired_z_test(pair_diffs)

        return {
            "control": control,
            "learned": learned,
            "s1_rate_difference": round(matched_diff, 4),
            "z_score": round(matched_z, 3),
            "significant_at_p05": abs(matched_z) > 1.96,
            "learned_is_better": matched_diff > 0,
            "time_matched": True,
            "matched_pairs": len(pair_diffs),
            "interpretation": (
                "Learned weights significantly outperform controls (time-matched)"
                if matched_z > 1.96
                else "Learned weights significantly underperform controls (time-matched)"
                if matched_z < -1.96
                else "No significant difference between learned and control weights (time-matched)"
            ),
            "caveat": (
                "Comparison is time-matched: each control is compared only with "
                "temporally adjacent learned experiments to account for increasing "
                "search difficulty over a session."
            ),
        }

    def _load_control_learned_exps(
        self,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Load and partition experiments into control vs learned groups."""
        rows = self.nb.conn.execute("""
            SELECT experiment_id, config_json, timestamp
            FROM experiments WHERE status = 'completed' AND config_json IS NOT NULL
            ORDER BY timestamp
        """).fetchall()

        control_exps: list[tuple[str, str]] = []
        learned_exps: list[tuple[str, str]] = []
        for row in rows:
            try:
                cfg = json.loads(row["config_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            ts = row["timestamp"] or ""
            if cfg.get("control_experiment"):
                control_exps.append((row["experiment_id"], ts))
            else:
                learned_exps.append((row["experiment_id"], ts))
        return control_exps, learned_exps

    @staticmethod
    def _time_matched_diffs(
        control_exps: list[tuple[str, str]],
        learned_exps: list[tuple[str, str]],
        s1_cache: dict[str, tuple[int, int]],
    ) -> list[float]:
        """Compute per-control time-matched S1 rate differences."""
        learned_ts = [(eid, ts) for eid, ts in learned_exps]
        pair_diffs: list[float] = []
        for ctrl_id, ctrl_ts in control_exps:
            before = [e for e in learned_ts if e[1] <= ctrl_ts]
            after = [e for e in learned_ts if e[1] > ctrl_ts]
            neighbors = []
            if before:
                neighbors.append(before[-1][0])
            if after:
                neighbors.append(after[0][0])
            if not neighbors:
                continue

            ctrl_total, ctrl_s1 = s1_cache.get(ctrl_id, (0, 0))
            if ctrl_total == 0:
                continue
            ctrl_rate = ctrl_s1 / ctrl_total

            nbr_total = nbr_s1 = 0
            for nid in neighbors:
                t, s = s1_cache.get(nid, (0, 0))
                nbr_total += t
                nbr_s1 += s
            if nbr_total == 0:
                continue

            pair_diffs.append(nbr_s1 / nbr_total - ctrl_rate)
        return pair_diffs

    @staticmethod
    def _paired_z_test(pair_diffs: list[float]) -> tuple[float, float]:
        """Approximate paired z-test. Returns (mean_diff, z_score)."""
        if not pair_diffs:
            return 0.0, 0.0
        matched_diff = sum(pair_diffs) / len(pair_diffs)
        if len(pair_diffs) > 1:
            diff_std = (
                sum((d - matched_diff) ** 2 for d in pair_diffs) / (len(pair_diffs) - 1)
            ) ** 0.5
            matched_se = diff_std / len(pair_diffs) ** 0.5 if diff_std > 0 else 0.0
            matched_z = matched_diff / matched_se if matched_se > 0 else 0.0
        else:
            matched_z = 0.0
        return matched_diff, matched_z

    def _s1_bulk(self, exp_ids: list[str]) -> dict[str, tuple[int, int]]:
        """Return {exp_id: (total, s1_passed)} for all exp_ids in one query."""
        if not exp_ids:
            return {}
        placeholders = ",".join("?" for _ in exp_ids)
        rows = self.nb.conn.execute(
            f"""
            SELECT experiment_id,
                   COUNT(*) as total,
                   SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) as s1
            FROM program_results
            WHERE experiment_id IN ({placeholders})
            GROUP BY experiment_id
        """,
            exp_ids,
        ).fetchall()
        return {r["experiment_id"]: (r["total"] or 0, r["s1"] or 0) for r in rows}

    @staticmethod
    def _s1_stats(exp_ids: list[str], s1_cache: dict[str, tuple[int, int]]) -> dict:
        """Aggregate S1 stats from cache for a list of experiment IDs."""
        total = s1 = 0
        for eid in exp_ids:
            t, s = s1_cache.get(eid, (0, 0))
            total += t
            s1 += s
        return {
            "experiments": len(exp_ids),
            "programs": total,
            "s1_passed": s1,
            "s1_rate": s1 / max(total, 1),
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
        rows = [
            row
            for row in load_deduped_graph_analysis_rows(self.nb.db_path)
            if row.get("stage1_any_passed")
        ]
        rows.sort(
            key=lambda row: (
                row.get("novelty_score") is None,
                -(float(row["novelty_score"]))
                if row.get("novelty_score") is not None
                else 0.0,
            )
        )
        rows = rows[:200]

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
                    if r.get("novelty_score"):
                        pair_novelty[pair].append(float(r["novelty_score"]))

        # Sort by frequency
        top_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])[:n]
        results = []
        for (op_a, op_b), count in top_pairs:
            novelties = pair_novelty.get((op_a, op_b), [])
            results.append(
                {
                    "ops": [op_a, op_b],
                    "count": count,
                    "avg_novelty": sum(novelties) / len(novelties) if novelties else 0,
                }
            )

        return results

    def learning_trajectory(self) -> Dict:
        """Compute learning trajectory: S1 rate trend with regression.

        Returns:
            {
                "points": [{experiment_id, timestamp, s1_rate, n_programs}, ...],
                "trend": "improving" | "plateaued" | "declining",
                "slope": float,           # S1-rate change per experiment
                "recent_s1_rate": float,   # avg of last 5
                "overall_s1_rate": float,
                "n_experiments": int,
                "min_experiments_required": int,
                "weight_adjustments": int, # count of grammar_weights_applied events
            }
        """
        experiments = self.nb.get_recent_experiments(100)
        experiments = list(reversed(experiments))

        points = self._build_trajectory_points(experiments)
        if not points:
            return self._empty_trajectory()

        self._apply_bayesian_shrinkage(points)
        slope, trend, trend_confidence = self._compute_trend(points)

        recent = [p.get("adjusted_s1_rate", p["s1_rate"]) for p in points[-5:]]
        recent_rate = sum(recent) / len(recent)
        mean_y = sum(p.get("adjusted_s1_rate", p["s1_rate"]) for p in points) / len(
            points
        )

        avg_conf_halfwidth = sum(
            p.get("s1_confidence_halfwidth", 0.0) for p in points
        ) / max(len(points), 1)

        weight_adjustments = self._count_weight_adjustments()

        return {
            "points": points,
            "trend": trend,
            "slope": round(slope, 6),
            "recent_s1_rate": round(recent_rate, 4),
            "overall_s1_rate": round(mean_y, 4),
            "n_experiments": len(points),
            "min_experiments_required": self.LEARNING_TRAJECTORY_MIN_EXPERIMENTS,
            "trend_confidence": trend_confidence,
            "overall_s1_confidence_halfwidth": round(avg_conf_halfwidth, 6),
            "weight_adjustments": weight_adjustments,
        }

    def _build_trajectory_points(self, experiments: list) -> List[Dict]:
        """Extract trajectory data points from experiment rows."""
        points = []
        for exp in experiments:
            n_gen = exp.get("n_programs_generated") or 0
            n_s1 = exp.get("n_stage1_passed") or 0
            if n_gen == 0:
                continue
            points.append(
                {
                    "experiment_id": exp.get("experiment_id", ""),
                    "timestamp": exp.get("timestamp", 0),
                    "s1_rate": n_s1 / n_gen,
                    "n_stage1_passed": n_s1,
                    "n_programs": n_gen,
                }
            )
        return points

    def _empty_trajectory(self) -> Dict:
        """Return empty trajectory result."""
        return {
            "points": [],
            "trend": "insufficient_data",
            "slope": 0.0,
            "recent_s1_rate": 0.0,
            "overall_s1_rate": 0.0,
            "n_experiments": 0,
            "min_experiments_required": self.LEARNING_TRAJECTORY_MIN_EXPERIMENTS,
            "trend_confidence": "low",
            "overall_s1_confidence_halfwidth": 0.0,
            "weight_adjustments": 0,
        }

    @staticmethod
    def _apply_bayesian_shrinkage(points: List[Dict]) -> None:
        """Apply Bayesian shrinkage to S1 rates in-place."""
        total_programs = sum(max(int(p.get("n_programs") or 0), 0) for p in points)
        total_stage1 = sum(max(int(p.get("n_stage1_passed") or 0), 0) for p in points)
        global_rate = total_stage1 / max(total_programs, 1)
        prior_strength = 12.0

        for point in points:
            n_programs = max(int(point.get("n_programs") or 0), 0)
            effective_n = max(1.0, float(n_programs))
            raw_rate = float(point.get("s1_rate") or 0.0)
            shrinkage = effective_n / (effective_n + prior_strength)
            adjusted_rate = global_rate + shrinkage * (raw_rate - global_rate)
            variance = max(adjusted_rate * (1.0 - adjusted_rate), 0.0)
            halfwidth = 1.96 * math.sqrt(variance / max(effective_n, 1.0))

            point["adjusted_s1_rate"] = adjusted_rate
            point["s1_confidence_lower"] = max(0.0, adjusted_rate - halfwidth)
            point["s1_confidence_upper"] = min(1.0, adjusted_rate + halfwidth)
            point["s1_confidence_halfwidth"] = halfwidth
            point["trend_weight"] = min(1.0, effective_n / 20.0)

    def _compute_trend(self, points: List[Dict]) -> tuple[float, str, str]:
        """Compute linear regression trend. Returns (slope, trend_label, confidence)."""
        n = len(points)
        rates = [p.get("adjusted_s1_rate", p["s1_rate"]) for p in points]
        mean_x = (n - 1) / 2.0
        mean_y = sum(rates) / n
        num = sum((i - mean_x) * (r - mean_y) for i, r in enumerate(rates))
        den = sum((i - mean_x) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0.0

        relative_slope = slope / max(mean_y, 0.01)
        if n < self.LEARNING_TRAJECTORY_MIN_EXPERIMENTS:
            trend = "insufficient_data"
        elif relative_slope > 0.05:
            trend = "improving"
        elif relative_slope < -0.05:
            trend = "declining"
        else:
            trend = "plateaued"

        avg_trend_weight = sum(p.get("trend_weight", 0.0) for p in points) / max(
            len(points), 1
        )
        if avg_trend_weight >= 0.75:
            trend_confidence = "high"
        elif avg_trend_weight >= 0.45:
            trend_confidence = "medium"
        else:
            trend_confidence = "low"

        return slope, trend, trend_confidence

    def _count_weight_adjustments(self) -> int:
        """Count grammar weight adjustment events from learning log."""
        try:
            log = self.nb.get_learning_log(limit=200)
            return sum(
                1
                for entry in log
                if entry.get("event_type") == "grammar_weights_applied"
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.debug("Grammar weight adjustment count failed: %s", e)
            return 0
