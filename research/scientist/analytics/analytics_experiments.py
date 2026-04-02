from __future__ import annotations
import json
import hashlib
import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class _ExperimentsMixin:
    """Experiment clustering, correlations, insights, and math coverage."""

    __slots__ = ()

    def reproducibility_packet_status(self, program: Dict) -> Dict:
        """Evaluate reproducibility packet completeness for a program."""
        arch_choices = self._extract_arch_choices(program.get("arch_spec_json"))
        checks = [
            ("result_id", bool(program.get("result_id"))),
            ("graph_fingerprint", bool(program.get("graph_fingerprint"))),
            ("arch_spec", bool(arch_choices)),
            (
                "baseline_ratio",
                program.get("validation_baseline_ratio") is not None
                or program.get("baseline_loss_ratio") is not None,
            ),
            (
                "multi_seed_std",
                program.get("validation_multi_seed_std") is not None,
            ),
            ("cka_artifact", program.get("cka_source") == "artifact"),
        ]
        ready_count = sum(1 for _, ok in checks if ok)
        total_checks = len(checks)
        if ready_count == total_checks:
            status = "ready"
        elif ready_count >= 4:
            status = "partial"
        else:
            status = "sparse"
        return {
            "status": status,
            "ready_count": ready_count,
            "total_checks": total_checks,
            "missing": [name for name, ok in checks if not ok],
        }

    def op_success_rates(self, since_ts: float = 0.0) -> Dict[str, Dict]:
        """Get per-op success rates.

        Args:
            since_ts: If > 0, compute rates from program_results within the
                time window (windowed view) instead of the accumulated table.
                This breaks the death spiral where fixed ops remain poisoned
                by stale lifetime data.
        """
        if since_ts > 0:
            rows = self.nb.get_op_success_rates_windowed(since_ts)
        else:
            rows = self.nb.get_op_success_rates()
        result = {}
        for row in rows:
            op = row["op_name"]
            n_used = row["n_used"] or 1
            n_s0 = row.get("n_stage0_passed") or 0

            # S1 success rate should be relative to things that actually
            # passed compilation. If it didn't compile, it's a code issue,
            # not a failure of the architecture's scientific utility.
            s1_rate = (row.get("n_stage1_passed") or 0) / n_s0 if n_s0 > 0 else 0.0

            result[op] = {
                "n_used": n_used,
                "n_s0": n_s0,
                "s0_rate": n_s0 / n_used,
                "s05_rate": (row.get("n_stage05_passed") or 0) / n_used,
                "s1_rate": s1_rate,
                "avg_loss_ratio": row.get("avg_loss_ratio"),
                "avg_novelty": row.get("avg_novelty"),
                "avg_novelty_confidence": row.get("avg_novelty_confidence"),
            }
        return result

    def compute_op_weights(
        self, since_ts: float = 0.0, min_used: int = 5
    ) -> Dict[str, float]:
        """Per-op weights via contrast amplification: (s1_rate/mean)^2, clamped [0.1, 8.0].

        Structural ops (no learnable params) are excluded from the mean
        calculation and get weight 1.0 — they should not be penalized or
        rewarded based on S1 attribution since they are scaffolding.
        """
        from research.synthesis.context_rules import S1_EXEMPT_OPS

        rates = self.op_success_rates(since_ts=since_ts)
        if not rates:
            return {}
        eligible = {
            op: info
            for op, info in rates.items()
            if info["n_used"] >= min_used and op not in S1_EXEMPT_OPS
        }
        if not eligible:
            return {}
        mean_s1 = sum(info["s1_rate"] for info in eligible.values()) / len(eligible)
        if mean_s1 < 1e-6:
            return {}
        weights: Dict[str, float] = {}
        for op, info in eligible.items():
            relative = info["s1_rate"] / mean_s1
            amplified = relative**2
            weights[op] = round(max(0.1, min(8.0, amplified)), 3)
        return weights

    def under_observed_ops(self, threshold: int = 20) -> Dict[str, int]:
        """Return ops with fewer than threshold observations.

        Returns dict of op_name → n_used. Also includes ops in
        PRIMITIVE_REGISTRY but not tracked in op_success_rates (count=0).
        """
        from research.synthesis.primitives import PRIMITIVE_REGISTRY

        rates = self.op_success_rates()
        result = {}
        for op, info in rates.items():
            if info["n_used"] < threshold:
                result[op] = info["n_used"]

        # Ops in registry but not tracked at all
        tracked = set(rates.keys())
        for name in PRIMITIVE_REGISTRY:
            if name not in tracked and name not in ("input", "output"):
                result[name] = 0

        return result

    def _compute_metadata_weights(
        self,
        metadata_key: str,
        since_ts: float,
        min_used: int,
    ) -> Dict[str, float]:
        """Compute contrast-amplified weights from graph metadata lists.

        Extracts ``metadata_key`` (e.g. ``templates_used``, ``motifs_used``)
        from ``graph_json.metadata`` and computes per-item S1 success rates.
        Returns ``{item_name: weight}`` clamped to ``[0.1, 8.0]``.
        """
        rows = self.nb.conn.execute(
            "SELECT graph_json, stage1_passed FROM program_results "
            "WHERE stage0_passed = 1 AND timestamp >= ? "
            "AND graph_json IS NOT NULL AND graph_json != '{}'",
            (since_ts,),
        ).fetchall()
        counts: Dict[str, int] = defaultdict(int)
        s1_counts: Dict[str, int] = defaultdict(int)
        for row in rows:
            try:
                meta = json.loads(row[0]).get("metadata", {})
            except (json.JSONDecodeError, TypeError):
                continue
            items = meta.get(metadata_key)
            if not isinstance(items, list):
                continue
            passed = bool(row[1])
            for item in items:
                if not isinstance(item, str):
                    continue
                counts[item] += 1
                if passed:
                    s1_counts[item] += 1
        stats = {
            name: {"n_used": n, "s1_rate": s1_counts.get(name, 0) / n}
            for name, n in counts.items()
            if n >= min_used
        }
        if not stats:
            return {}
        mean_s1 = sum(s["s1_rate"] for s in stats.values()) / len(stats)
        if mean_s1 < 1e-6:
            return {}
        weights: Dict[str, float] = {}
        for name, s in stats.items():
            relative = s["s1_rate"] / mean_s1
            # Moderate contrast: relative^1.5 (not ^2) to avoid collapsing
            # low-performers too aggressively — they still need search coverage.
            amplified = relative**1.5
            # Confidence discount: shrink toward 1.0 for small sample sizes.
            # At n=min_used the weight is 50% amplified + 50% neutral (1.0).
            # At n=30+ the weight is fully amplified.
            confidence = min(1.0, s["n_used"] / 30.0)
            blended = confidence * amplified + (1.0 - confidence) * 1.0
            weights[name] = round(max(0.3, min(5.0, blended)), 3)
        return weights

    def compute_template_weights(
        self, since_ts: float = 0.0, min_used: int = 3
    ) -> Dict[str, float]:
        """Per-template weights from S1 success rates via contrast amplification."""
        return self._compute_metadata_weights("templates_used", since_ts, min_used)

    def compute_motif_weights(
        self, since_ts: float = 0.0, min_used: int = 3
    ) -> Dict[str, float]:
        """Per-motif weights from S1 success rates via contrast amplification."""
        return self._compute_metadata_weights("motifs_used", since_ts, min_used)

    def compute_template_and_motif_weights(
        self, since_ts: float = 0.0, min_used: int = 3
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute template and motif weights in a single DB query pass."""
        rows = self.nb.conn.execute(
            "SELECT graph_json, stage1_passed FROM program_results "
            "WHERE stage0_passed = 1 AND timestamp >= ? "
            "AND graph_json IS NOT NULL AND graph_json != '{}'",
            (since_ts,),
        ).fetchall()

        results: Dict[str, Dict[str, float]] = {}
        for metadata_key in ("templates_used", "motifs_used"):
            counts: Dict[str, int] = defaultdict(int)
            s1_counts: Dict[str, int] = defaultdict(int)
            for row in rows:
                try:
                    meta = json.loads(row[0]).get("metadata", {})
                except (json.JSONDecodeError, TypeError):
                    continue
                items = meta.get(metadata_key)
                if not isinstance(items, list):
                    continue
                passed = bool(row[1])
                for item in items:
                    if not isinstance(item, str):
                        continue
                    counts[item] += 1
                    if passed:
                        s1_counts[item] += 1
            stats = {
                name: {"n_used": n, "s1_rate": s1_counts.get(name, 0) / n}
                for name, n in counts.items()
                if n >= min_used
            }
            if not stats:
                results[metadata_key] = {}
                continue
            mean_s1 = sum(s["s1_rate"] for s in stats.values()) / len(stats)
            if mean_s1 < 1e-6:
                results[metadata_key] = {}
                continue
            weights: Dict[str, float] = {}
            for name, s in stats.items():
                relative = s["s1_rate"] / mean_s1
                amplified = relative**1.5
                confidence = min(1.0, s["n_used"] / 30.0)
                blended = confidence * amplified + (1.0 - confidence) * 1.0
                weights[name] = round(max(0.3, min(5.0, blended)), 3)
            results[metadata_key] = weights

        return results.get("templates_used", {}), results.get("motifs_used", {})

    def compute_synergy_boosts(
        self,
        min_lift: float = 1.5,
        min_co_occurrences: int = 5,
        boost_cap: float = 3.0,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Boost motif/template weights for ops that are synergistic in S1 survivors.

        For each synergistic pair (A, B) with lift > min_lift:
          - Find motifs containing A → boost by sqrt(lift)
          - Find motifs containing B → boost by sqrt(lift)
          - Find templates mapped to A or B → boost by sqrt(lift)

        sqrt(lift) because both ops' motifs get boosted independently;
        the compound effect when both land in the same graph ≈ lift.

        Returns (motif_boosts, template_boosts) — multiplicative factors.
        """
        from research.scientist.intelligence.analyzer import analyze_op_synergies
        from research.synthesis.motifs import ALL_MOTIFS

        synergies = analyze_op_synergies(self.nb, min_co_occurrences=min_co_occurrences)
        if not synergies:
            return {}, {}

        # Build op → motif index
        op_to_motifs: Dict[str, List[str]] = defaultdict(list)
        for motif in ALL_MOTIFS:
            for step in motif.steps:
                op_to_motifs[step.op_name].append(motif.name)

        # Build op → template index from _OP_TO_TEMPLATE
        try:
            pass  # ensure loaded

            # _OP_TO_TEMPLATE is defined inside generate_layer_graph, but the
            # grammar module also exposes the mapping via GrammarConfig.exploration.
            # We'll use the static mapping from the grammar source.
            pass
        except ImportError:
            pass
        # Use the known _OP_TO_TEMPLATE mapping — it's a local dict inside
        # generate_layer_graph, so we reconstruct the subset we need.
        _OP_TO_TEMPLATE = {
            "lif_neuron": "spiking_moe_block",
            "sparse_threshold": "spiking_moe_block",
            "spike_rate_code": "spiking_moe_block",
            "split3": "three_way_split",
            "tropical_center": "tropical_center_block",
            "tropical_attention": "tropical_center_block",
            "state_space": "state_space_block",
            "conv_only": "conv_residual_block",
            "gated_delta": "recurrent_delta_block",
            "early_exit": "cascaded_early_exit",
            "n_way_sparse_router": "n_way_moe_block",
        }

        motif_boosts: Dict[str, float] = {}
        template_boosts: Dict[str, float] = {}

        for syn in synergies:
            if syn.label != "synergistic" or syn.lift < min_lift:
                continue
            boost = min(math.sqrt(syn.lift), boost_cap)

            for op in (syn.op_a, syn.op_b):
                # Boost motifs containing this op
                for motif_name in op_to_motifs.get(op, []):
                    motif_boosts[motif_name] = max(
                        motif_boosts.get(motif_name, 1.0), boost
                    )
                # Boost templates mapped to this op
                tpl = _OP_TO_TEMPLATE.get(op)
                if tpl:
                    template_boosts[tpl] = max(template_boosts.get(tpl, 1.0), boost)

        n_syn = sum(1 for s in synergies if s.label == "synergistic")
        if motif_boosts or template_boosts:
            logger.info(
                "Synergy boosts: %d synergistic pairs → %d motif boosts, %d template boosts",
                n_syn,
                len(motif_boosts),
                len(template_boosts),
            )
        return motif_boosts, template_boosts

    def gate_performance_summary(self) -> Dict:
        """Analyze Stage 0.5 (Causality Gate) vs Stage 1 (Micro-Corpus) efficiency.

        Tracks how many 'cheaters' (random-token hackers) are caught by the
        causality gate and how well discovery loss predicts validation loss.
        """
        rows = self.nb.conn.execute("""
            SELECT result_id, stage05_passed, stage1_passed, discovery_loss_ratio, 
                   validation_loss_ratio, error_type
            FROM program_results
            WHERE stage0_passed = 1
        """).fetchall()

        if not rows:
            return {}

        total = len(rows)
        s05_passed = sum(1 for r in rows if r["stage05_passed"])
        total - s05_passed

        causality_violations = sum(
            1 for r in rows if r["error_type"] == "causality_violation"
        )

        # Correlation between discovery and validation
        discovery = []
        validation = []
        for r in rows:
            if (
                r["discovery_loss_ratio"] is not None
                and r["validation_loss_ratio"] is not None
            ):
                discovery.append(r["discovery_loss_ratio"])
                validation.append(r["validation_loss_ratio"])

        correlation = None
        if len(discovery) > 5:
            try:
                import numpy as np

                correlation = float(np.corrcoef(discovery, validation)[0, 1])
            except Exception:
                pass

        return {
            "total_screened": total,
            "stage05_pass_rate": round(s05_passed / total, 4) if total > 0 else 0.0,
            "causality_violations": causality_violations,
            "discovery_validation_correlation": round(correlation, 4)
            if correlation is not None
            else None,
            "n_correlation_samples": len(discovery),
        }

    def gate_health_daily(self, n_days: int = 14) -> Dict:
        """Daily breakdown of causality gate metrics for monitoring dashboards.

        Returns per-day stats: models screened, gate pass rate, causality
        violations, and discovery-vs-validation correlation.
        """
        import time as _time

        cutoff = _time.time() - (n_days * 86400)
        rows = self.nb.conn.execute(
            """
            SELECT result_id, stage05_passed, stage1_passed,
                   discovery_loss_ratio, validation_loss_ratio,
                   error_type, timestamp
            FROM program_results
            WHERE stage0_passed = 1 AND timestamp > ?
            ORDER BY timestamp
        """,
            (cutoff,),
        ).fetchall()

        if not rows:
            return {"daily": [], "summary": self.gate_performance_summary()}

        from collections import defaultdict
        from datetime import datetime

        buckets: dict = defaultdict(list)
        for r in rows:
            day = datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d")
            buckets[day].append(r)

        daily = []
        for day in sorted(buckets):
            day_rows = buckets[day]
            n = len(day_rows)
            passed = sum(1 for r in day_rows if r["stage05_passed"])
            violations = sum(
                1 for r in day_rows if r["error_type"] == "causality_violation"
            )

            disc, val = [], []
            for r in day_rows:
                if (
                    r["discovery_loss_ratio"] is not None
                    and r["validation_loss_ratio"] is not None
                ):
                    disc.append(r["discovery_loss_ratio"])
                    val.append(r["validation_loss_ratio"])

            corr = None
            if len(disc) > 3:
                try:
                    import numpy as np

                    corr = round(float(np.corrcoef(disc, val)[0, 1]), 4)
                except Exception:
                    pass

            daily.append(
                {
                    "date": day,
                    "models_screened": n,
                    "gate_pass_rate": round(passed / n, 4) if n else 0.0,
                    "causality_violations": violations,
                    "gate_failure_rate": round((n - passed) / n, 4) if n else 0.0,
                    "discovery_validation_correlation": corr,
                    "n_correlation_samples": len(disc),
                }
            )

        return {"daily": daily, "summary": self.gate_performance_summary()}

    def structural_correlations(self) -> Dict[str, float]:
        """Analyze which graph properties correlate with Stage 1 success.

        Returns correlation-like scores for graph metrics vs success.
        Vectorized via NumPy for high-performance orchestration.
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

        metrics = [
            "graph_n_ops",
            "graph_depth",
            "graph_n_params_estimate",
            "graph_n_unique_ops",
            "graph_uses_math_spaces",
            "graph_uses_frequency_domain",
            "graph_has_gradient_path",
        ]

        data = np.array(
            [[float(r[m] or 0) for m in metrics] for r in rows], dtype=np.float32
        )
        passed = np.array([bool(r["stage1_passed"]) for r in rows], dtype=bool)

        if not np.any(passed) or np.all(passed):
            return {m: 0.0 for m in metrics}

        success_data = data[passed]
        fail_data = data[~passed]

        avg_success = np.mean(success_data, axis=0)
        avg_fail = np.mean(fail_data, axis=0)
        std_all = np.std(data, axis=0)

        correlations = {}
        for i, m in enumerate(metrics):
            if std_all[i] > 1e-9:
                correlations[m] = float((avg_success[i] - avg_fail[i]) / std_all[i])
            else:
                correlations[m] = 0.0

        return correlations

    def experiment_clusters(self, n_clusters: int = 3) -> Optional[Dict]:
        """Cluster completed experiments by outcome profile.

        Uses high-performance NumPy vectorization for k-means clustering,
        silhouette scores, and model selection.
        """
        self.nb.flush_writes()
        rows = self.nb.conn.execute("""
            SELECT experiment_id, n_programs_generated, n_stage1_passed,
                   best_novelty_score, best_loss_ratio, duration_seconds
            FROM experiments
            WHERE status = 'completed' AND n_programs_generated > 0
            ORDER BY timestamp DESC LIMIT 2000
        """).fetchall()

        if len(rows) < 3:
            return None

        experiments = []
        for row in rows:
            total = row["n_programs_generated"] or 0
            if total <= 0:
                continue
            experiments.append(
                {
                    "experiment_id": row["experiment_id"],
                    "s1_rate": (row["n_stage1_passed"] or 0) / total,
                    "best_novelty": float(row["best_novelty_score"] or 0.0),
                    "best_loss_ratio": float(row["best_loss_ratio"] or 1.0),
                    "duration_seconds": float(row["duration_seconds"] or 0.0),
                }
            )

        if len(experiments) < 3:
            return None

        exp_ids = [e["experiment_id"] for e in experiments]
        placeholders = ",".join("?" * len(exp_ids))

        # Load failures and errors
        failure_rows = self.nb.conn.execute(
            f"""
            SELECT experiment_id, COUNT(*) as n_total,
                   SUM(CASE WHEN COALESCE(stage0_passed, 0) = 0 THEN 1 ELSE 0 END) as n_compile_fail,
                   SUM(CASE WHEN COALESCE(stage0_passed, 0) = 1 AND COALESCE(stage05_passed, 0) = 0 THEN 1 ELSE 0 END) as n_train_fail,
                   SUM(CASE WHEN COALESCE(stage05_passed, 0) = 1 AND COALESCE(stage1_passed, 0) = 0 THEN 1 ELSE 0 END) as n_stage1_fail
            FROM program_results WHERE experiment_id IN ({placeholders}) GROUP BY experiment_id
        """,
            tuple(exp_ids),
        ).fetchall()

        fail_map = {r["experiment_id"]: r for r in failure_rows}

        error_rows = self.nb.conn.execute(
            f"""
            SELECT experiment_id, error_type, COUNT(*) as n
            FROM program_results WHERE experiment_id IN ({placeholders})
            AND error_type IS NOT NULL AND TRIM(error_type) != '' GROUP BY experiment_id, error_type
        """,
            tuple(exp_ids),
        ).fetchall()

        error_map = defaultdict(dict)
        for r in error_rows:
            error_map[r["experiment_id"]][r["error_type"]] = int(r["n"] or 0)

        for e in experiments:
            f = fail_map.get(
                e["experiment_id"],
                {
                    "n_total": 1,
                    "n_compile_fail": 0,
                    "n_train_fail": 0,
                    "n_stage1_fail": 0,
                },
            )
            n = float(f["n_total"] or 1)
            e.update(
                {
                    "compile_fail_rate": f["n_compile_fail"] / n,
                    "train_fail_rate": f["n_train_fail"] / n,
                    "stage1_fail_rate": f["n_stage1_fail"] / n,
                    "error_diversity": 0.0,
                }
            )
            errs = error_map.get(e["experiment_id"], {})
            total_err = float(sum(errs.values()))
            if total_err > 0 and len(errs) > 1:
                probs = np.array(list(errs.values())) / total_err
                e["error_diversity"] = -np.sum(probs * np.log(probs)) / np.log(
                    len(errs)
                )

        # Load trajectories
        seq_rows = self.nb.conn.execute(
            f"""
            SELECT experiment_id, stage1_passed, loss_ratio, novelty_score
            FROM program_results WHERE experiment_id IN ({placeholders})
            ORDER BY experiment_id ASC, timestamp ASC
        """,
            tuple(exp_ids),
        ).fetchall()

        per_exp_seq = defaultdict(list)
        for r in seq_rows:
            per_exp_seq[r["experiment_id"]].append(
                (
                    float(r["stage1_passed"] or 0),
                    float(r["novelty_score"] or 0),
                    float(r["loss_ratio"] or 1.0),
                )
            )

        for e in experiments:
            seq = np.array(per_exp_seq.get(e["experiment_id"], []), dtype=np.float32)
            if len(seq) < 2:
                e.update(
                    {
                        k: 0.0
                        for k in [
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
                    }
                )
                continue

            # Vectorized momentum and statistics
            window = max(1, len(seq) // 3)
            e["stage1_momentum"] = np.mean(seq[-window:, 0]) - np.mean(seq[:window, 0])
            e["novelty_momentum"] = np.mean(seq[-window:, 1]) - np.mean(seq[:window, 1])
            e["loss_improvement_momentum"] = np.mean(seq[:window, 2]) - np.mean(
                seq[-window:, 2]
            )

            proxy = (
                0.5 * seq[:, 0]
                + 0.3 * seq[:, 1]
                + 0.2 * (1.0 / (1.0 + np.maximum(seq[:, 2], 1e-9)))
            )
            e["outcome_volatility"] = np.std(proxy)
            e["outcome_peak_timing"] = np.argmax(proxy) / max(len(seq) - 1, 1)

            # Transitions
            transitions = np.where(seq[1:, 0] != seq[:-1, 0])[0] + 1
            e["stage1_transition_timing"] = (
                transitions[0] / (len(seq) - 1) if len(transitions) > 0 else 0.0
            )
            e["stage1_transition_density"] = len(transitions) / max(len(seq) - 1, 1)
            if len(transitions) >= 2:
                gaps = np.diff(transitions).astype(np.float32)
                p = gaps / gaps.sum()
                e["transition_gap_entropy"] = -np.sum(p * np.log(p + 1e-10)) / np.log(
                    len(transitions)
                )
            else:
                e["transition_gap_entropy"] = 0.0

            deltas = np.abs(np.diff(proxy))
            if len(deltas) > 0:
                e["primary_change_point_timing"] = (np.argmax(deltas) + 1) / max(
                    len(seq) - 1, 1
                )
                e["change_point_confidence"] = np.max(deltas) / (np.sum(deltas) + 1e-10)

                # Windowed change dispersion and localization
                n_deltas = len(deltas)
                seg = max(1, n_deltas // 3)
                window_means = [
                    np.mean(deltas[i * seg : (i + 1) * seg])
                    if i * seg < n_deltas
                    else 0.0
                    for i in range(3)
                ]
                e["windowed_change_dispersion"] = np.std(window_means)
                total_window_change = np.sum(window_means)
                e["window_change_localization"] = (
                    np.max(window_means) / total_window_change
                    if total_window_change > 1e-9
                    else 0.0
                )
            else:
                e.update(
                    {
                        "primary_change_point_timing": 0.0,
                        "change_point_confidence": 0.0,
                        "windowed_change_dispersion": 0.0,
                        "window_change_localization": 0.0,
                    }
                )

            # Recovery lag
            early_baseline = np.mean(proxy[:window])
            trough_idx = np.argmin(proxy)
            recovery_idx = np.where(proxy[trough_idx + 1 :] >= early_baseline)[0]
            e["recovery_lag"] = (
                (recovery_idx[0] + 1) / (len(seq) - 1)
                if len(recovery_idx) > 0
                else (1.0 if len(seq) > 1 else 0.0)
            )

        # Prepare for Vectorized K-Means
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
        X = np.array(
            [[e[k] for k in feature_keys] for e in experiments], dtype=np.float32
        )
        # Normalize and invert loss_ratio
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_range = X_max - X_min
        X_norm = np.zeros_like(X)
        mask = X_range > 1e-9
        X_norm[:, mask] = (X[:, mask] - X_min[mask]) / X_range[mask]
        X_norm[:, feature_keys.index("best_loss_ratio")] = (
            1.0 - X_norm[:, feature_keys.index("best_loss_ratio")]
        )

        def _vectorized_kmeans(k, salt):
            # Exact match of original deterministic init
            seed_hex = hashlib.md5(f"{dataset_signature}:{salt}".encode()).hexdigest()
            first_idx = int(seed_hex[:8], 16) % len(X_norm)
            centroids = [X_norm[first_idx]]
            chosen_idxs = {first_idx}

            for _ in range(1, k):
                # Farthest point initialization (K-Means++)
                dists = np.min(cdist(X_norm, np.array(centroids)), axis=1)
                # Filter out already chosen to be safe (though dist would be 0)
                next_idx = np.argmax(dists)
                centroids.append(X_norm[next_idx])
                chosen_idxs.add(next_idx)
            centroids = np.array(centroids)

            for _ in range(30):
                dists = cdist(X_norm, centroids)
                assignments = np.argmin(dists, axis=1)
                new_centroids = np.array(
                    [
                        X_norm[assignments == i].mean(axis=0)
                        if np.any(assignments == i)
                        else centroids[i]
                        for i in range(k)
                    ]
                )
                if np.allclose(centroids, new_centroids):
                    break
                centroids = new_centroids

            inertia = np.sum(np.min(cdist(X_norm, centroids), axis=1) ** 2)
            return assignments, centroids, inertia

        def _vectorized_silhouette(assignments, dist_matrix):
            unique = np.unique(assignments)
            if len(unique) < 2:
                return 0.0
            sil = []
            for i in range(len(X_norm)):
                c_i = assignments[i]
                mask_same = assignments == c_i
                mask_same[i] = False
                if not np.any(mask_same):
                    sil.append(0.0)
                    continue
                a_i = dist_matrix[i, mask_same].mean()
                b_i = min(
                    dist_matrix[i, assignments == c].mean() for c in unique if c != c_i
                )
                sil.append((b_i - a_i) / max(a_i, b_i, 1e-9))
            return np.mean(sil)

        dataset_signature = "|".join(sorted(exp_ids))
        dist_matrix = cdist(X_norm, X_norm)
        max_k = min(max(2, n_clusters), len(X_norm) - 1)
        if max_k < 2:
            return None

        candidates = []
        for k_val in range(2, max_k + 1):
            runs = []
            for salt in range(4):
                assign, cents, inertia = _vectorized_kmeans(k_val, salt)
                sil = _vectorized_silhouette(assign, dist_matrix)
                counts = np.bincount(assign, minlength=k_val)
                imbalance = np.sum(np.abs(counts - len(X_norm) / k_val)) / (
                    2.0 * len(X_norm)
                )
                runs.append(
                    {
                        "assignments": assign,
                        "centroids": cents,
                        "inertia": inertia,
                        "silhouette": sil,
                        "quality": sil - 0.15 * imbalance,
                    }
                )

            best = max(runs, key=lambda r: (r["quality"], -r["inertia"]))
            candidates.append(
                {"k": k_val, "best": best, "runs": runs, "score": best["quality"]}
            )

        selected = max(candidates, key=lambda c: (c["score"], -c["k"]))
        k, best_run = selected["k"], selected["best"]
        assign, cents = best_run["assignments"], best_run["centroids"]

        # Consensus and Stability
        def _agreement(a1, a2):
            m1 = a1[:, None] == a1[None, :]
            m2 = a2[:, None] == a2[None, :]
            return np.mean(m1 == m2)

        cons_scores = [
            _agreement(r1["assignments"], r2["assignments"])
            for i, r1 in enumerate(selected["runs"])
            for r2 in selected["runs"][i + 1 :]
        ]
        consensus = np.mean(cons_scores) if cons_scores else 1.0

        intra = np.mean(
            [np.mean(dist_matrix[i, assign == assign[i]]) for i in range(len(X_norm))]
        )
        inter = np.min(cdist(cents, cents) + np.eye(k) * 1e9)
        stability = 0.6 * (inter / (inter + intra + 1e-9)) + 0.4 * consensus

        # Summary and Description
        clusters = []
        for ci in range(k):
            members = [
                experiments[i] for i in range(len(experiments)) if assign[i] == ci
            ]
            if not members:
                continue
            summary = {
                k: round(float(np.mean([m[k] for m in members])), 4)
                for k in feature_keys
                if k != "duration_seconds"
            }
            summary["avg_duration_seconds"] = round(
                float(np.mean([m["duration_seconds"] for m in members])), 2
            )
            summary.update(
                {
                    "cluster_id": ci,
                    "size": len(members),
                    "experiment_ids": [m["experiment_id"] for m in members[:10]],
                }
            )
            # Map avg_s1_rate, etc back to required keys
            for fk in ["s1_rate", "best_novelty", "best_loss_ratio"]:
                summary[f"avg_{fk}"] = summary.pop(fk)
            for fk in feature_keys:
                if (
                    fk
                    not in [
                        "s1_rate",
                        "best_novelty",
                        "best_loss_ratio",
                        "duration_seconds",
                    ]
                    and fk in summary
                ):
                    summary[f"avg_{fk}"] = summary.pop(fk)
            clusters.append(summary)

        clusters.sort(key=lambda c: c["avg_s1_rate"], reverse=True)
        self._describe_clusters(clusters)

        return {
            "n_experiments": len(experiments),
            "n_clusters": len(clusters),
            "feature_keys": feature_keys,
            "stability_score": round(float(np.clip(stability, 0, 1)), 4),
            "model_selection": {
                "candidate_ks": [c["k"] for c in candidates],
                "selected_k": k,
                "silhouette": round(float(best_run["silhouette"]), 4),
                "consensus": round(float(consensus), 4),
                "selection_margin": round(
                    float(
                        sorted(candidates, key=lambda c: -c["score"])[0]["score"]
                        - sorted(candidates, key=lambda c: -c["score"])[1]["score"]
                    ),
                    4,
                )
                if len(candidates) > 1
                else 0.0,
            },
            "clusters": clusters,
        }

    @staticmethod
    def _describe_clusters(clusters: List[Dict]) -> None:
        """Generate contrastive plain-language descriptions for clusters.

        Ranks clusters against each other so labels are mutually exclusive
        (e.g., "the most productive", "moderate", "the least productive").
        """
        if not clusters:
            return

        # Rank by S1 rate descending to assign relative labels
        ranked = sorted(
            enumerate(clusters),
            key=lambda ic: ic[1].get("avg_s1_rate", 0) or 0,
            reverse=True,
        )

        for rank_idx, (orig_idx, c) in enumerate(ranked):
            size = c.get("size", 0)
            s1_pct = (c.get("avg_s1_rate", 0) or 0) * 100
            novelty = c.get("avg_best_novelty", 0) or 0
            c.get("avg_best_loss_ratio", 0) or 0
            compile_fail = (c.get("avg_compile_fail_rate", 0) or 0) * 100
            duration = c.get("avg_duration_seconds", 0) or 0

            # S1 description
            if s1_pct >= 30:
                s1_desc = f"high S1 pass rate ({s1_pct:.0f}%)"
            elif s1_pct >= 10:
                s1_desc = f"moderate S1 pass rate ({s1_pct:.0f}%)"
            elif s1_pct > 0:
                s1_desc = f"low S1 pass rate ({s1_pct:.0f}%)"
            else:
                s1_desc = "no S1 survivors"

            # Novelty description
            if novelty >= 0.7:
                nov_desc = "high novelty"
            elif novelty >= 0.3:
                nov_desc = "moderate novelty"
            else:
                nov_desc = "low novelty"

            # Find distinguishing feature for this cluster
            distinguisher = ""
            if compile_fail >= 50:
                distinguisher = f" High compile failure ({compile_fail:.0f}%) suggests grammar is exploring risky territory."
            elif novelty >= 0.5:
                distinguisher = f" High novelty ({novelty:.2f}) means these explore unfamiliar architecture space."
            elif duration > 600:
                distinguisher = f" Long average duration ({duration:.0f}s) indicates deeper investigation runs."

            # Relative character label
            n_clusters = len(clusters)
            if n_clusters == 1:
                character = "the only cluster"
            elif rank_idx == 0:
                character = "the most productive cluster"
            elif rank_idx == n_clusters - 1:
                if s1_pct == 0:
                    character = "the failing cluster"
                else:
                    character = "the least productive cluster"
            else:
                character = "a mid-tier cluster"

            clusters[orig_idx]["description"] = (
                f"{size} experiments with {s1_desc}, {nov_desc}."
                f" {character.capitalize()}.{distinguisher}"
            )

    def control_experiment_comparison(self) -> Optional[Dict]:
        """Compare control experiments (default weights) vs learned-weight experiments.

        Control experiments are flagged with ``control_experiment = true`` in
        config_json.  Uses time-matched comparison: each control experiment is
        paired with temporally adjacent learned experiments (the nearest before
        and after) to avoid confounding from increasing search difficulty over
        a session.

        Returns None if fewer than 2 control experiments exist.
        """
        rows = self.nb.conn.execute("""
            SELECT experiment_id, config_json, timestamp
            FROM experiments WHERE status = 'completed' AND config_json IS NOT NULL
            ORDER BY timestamp
        """).fetchall()

        control_exps: list[tuple[str, str]] = []  # (exp_id, created_at)
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

        if len(control_exps) < 2 or len(learned_exps) < 2:
            return None

        def _s1_bulk(exp_ids: list[str]) -> dict[str, tuple[int, int]]:
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

        all_exp_ids = list(
            {eid for eid, _ in control_exps} | {eid for eid, _ in learned_exps}
        )
        s1_cache = _s1_bulk(all_exp_ids)

        def _s1_for_exp(exp_id: str) -> tuple[int, int]:
            return s1_cache.get(exp_id, (0, 0))

        def _s1_stats(exp_ids: list[str]) -> dict:
            total = s1 = 0
            for eid in exp_ids:
                t, s = _s1_for_exp(eid)
                total += t
                s1 += s
            return {
                "experiments": len(exp_ids),
                "programs": total,
                "s1_passed": s1,
                "s1_rate": s1 / max(total, 1),
            }

        # Time-matched comparison: for each control, find nearest learned
        # experiments (one before, one after) and compare within that window
        learned_ts = [(eid, ts) for eid, ts in learned_exps]
        pair_diffs: list[float] = []
        matched_control_ids: list[str] = []
        matched_learned_ids: list[str] = []

        for ctrl_id, ctrl_ts in control_exps:
            # Find nearest learned experiments by timestamp
            before = [e for e in learned_ts if e[1] <= ctrl_ts]
            after = [e for e in learned_ts if e[1] > ctrl_ts]
            neighbors = []
            if before:
                neighbors.append(before[-1][0])  # latest before
            if after:
                neighbors.append(after[0][0])  # earliest after
            if not neighbors:
                continue

            ctrl_total, ctrl_s1 = _s1_for_exp(ctrl_id)
            if ctrl_total == 0:
                continue
            ctrl_rate = ctrl_s1 / ctrl_total

            nbr_total = nbr_s1 = 0
            for nid in neighbors:
                t, s = _s1_for_exp(nid)
                nbr_total += t
                nbr_s1 += s
            if nbr_total == 0:
                continue
            nbr_rate = nbr_s1 / nbr_total

            pair_diffs.append(nbr_rate - ctrl_rate)
            matched_control_ids.append(ctrl_id)
            matched_learned_ids.extend(neighbors)

        # Also compute overall stats for display
        all_control_ids = [eid for eid, _ in control_exps]
        all_learned_ids = [eid for eid, _ in learned_exps]
        control = _s1_stats(all_control_ids)
        learned = _s1_stats(all_learned_ids)

        # Time-matched effect: mean of per-window differences
        if pair_diffs:
            matched_diff = sum(pair_diffs) / len(pair_diffs)
            # Paired z-test (approximate)
            if len(pair_diffs) > 1:
                diff_std = (
                    sum((d - matched_diff) ** 2 for d in pair_diffs)
                    / (len(pair_diffs) - 1)
                ) ** 0.5
                matched_se = diff_std / len(pair_diffs) ** 0.5 if diff_std > 0 else 0.0
                matched_z = matched_diff / matched_se if matched_se > 0 else 0.0
            else:
                matched_z = 0.0
        else:
            matched_diff = 0.0
            matched_z = 0.0

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
        # Reverse to chronological order
        experiments = list(reversed(experiments))

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

        if not points:
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

        # Linear regression on S1 rate vs experiment index
        n = len(points)
        rates = [p.get("adjusted_s1_rate", p["s1_rate"]) for p in points]
        mean_x = (n - 1) / 2.0
        mean_y = sum(rates) / n
        num = sum((i - mean_x) * (r - mean_y) for i, r in enumerate(rates))
        den = sum((i - mean_x) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0.0

        # Trend classification
        # Use slope relative to mean to avoid noise at tiny scales
        relative_slope = slope / max(mean_y, 0.01)
        if n < self.LEARNING_TRAJECTORY_MIN_EXPERIMENTS:
            trend = "insufficient_data"
        elif relative_slope > 0.05:
            trend = "improving"
        elif relative_slope < -0.05:
            trend = "declining"
        else:
            trend = "plateaued"

        recent = rates[-5:] if len(rates) >= 5 else rates
        recent_rate = sum(recent) / len(recent)
        avg_trend_weight = sum(p.get("trend_weight", 0.0) for p in points) / max(
            len(points), 1
        )
        avg_conf_halfwidth = sum(
            p.get("s1_confidence_halfwidth", 0.0) for p in points
        ) / max(len(points), 1)

        if avg_trend_weight >= 0.75:
            trend_confidence = "high"
        elif avg_trend_weight >= 0.45:
            trend_confidence = "medium"
        else:
            trend_confidence = "low"

        # Count grammar weight adjustments
        try:
            log = self.nb.get_learning_log(limit=200)
            weight_adjustments = sum(
                1
                for entry in log
                if entry.get("event_type") == "grammar_weights_applied"
            )
        except Exception:
            weight_adjustments = 0

        return {
            "points": points,
            "trend": trend,
            "slope": round(slope, 6),
            "recent_s1_rate": round(recent_rate, 4),
            "overall_s1_rate": round(mean_y, 4),
            "n_experiments": n,
            "min_experiments_required": self.LEARNING_TRAJECTORY_MIN_EXPERIMENTS,
            "trend_confidence": trend_confidence,
            "overall_s1_confidence_halfwidth": round(avg_conf_halfwidth, 6),
            "weight_adjustments": weight_adjustments,
        }

    def math_family_coverage(self) -> Dict:
        """Summarize evaluated/surviving coverage by mathematical family."""
        rows = self.nb.conn.execute("""
            SELECT stage1_passed, graph_json, arch_spec_json
            FROM program_results
            WHERE graph_json IS NOT NULL OR arch_spec_json IS NOT NULL
            ORDER BY timestamp DESC LIMIT 5000
        """).fetchall()

        family_order = [
            "euclidean",
            "hyperbolic",
            "tropical",
            "p-adic",
            "clifford",
            "functional",
        ]
        stats = {
            fam: {"family": fam, "n_tested": 0, "n_survived": 0} for fam in family_order
        }

        hyperbolic_ops = {
            "poincare_add",
            "exp_map",
            "log_map",
            "hyp_linear",
            "hyp_distance",
            "hyp_tangent_nonlinear",
        }
        tropical_ops = {
            "tropical_matmul",
            "tropical_add",
            "tropical_attention",
            "tropical_center",
        }
        padic_ops = {"padic_expand", "ultrametric_attention", "padic_gate"}
        clifford_ops = {
            "geometric_product",
            "rotor_transform",
            "grade_select",
            "grade_mix",
        }
        functional_ops = {"basis_expansion", "integral_kernel", "fixed_point_iter"}

        def _family_from_row(
            graph_json: Optional[str], arch_spec_json: Optional[str]
        ) -> str:
            op_names: set[str] = set()
            if graph_json:
                try:
                    graph = json.loads(graph_json)
                    nodes = graph.get("nodes", {}) if isinstance(graph, dict) else {}
                    node_iter = (
                        nodes.values()
                        if isinstance(nodes, dict)
                        else nodes
                        if isinstance(nodes, list)
                        else []
                    )
                    for node in node_iter:
                        if isinstance(node, dict):
                            op_name = node.get("op_name") or node.get("op")
                            if op_name:
                                op_names.add(op_name)
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass

            token_mixing = None
            channel_mixing = None
            if arch_spec_json:
                try:
                    arch = json.loads(arch_spec_json)
                    choices = arch.get("choices", {}) if isinstance(arch, dict) else {}
                    if isinstance(choices, dict):
                        token_mixing = choices.get("token_mixing")
                        channel_mixing = choices.get("channel_mixing")
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass

            if (
                (op_names & functional_ops)
                or token_mixing == "integral_kernel_mixing"
                or channel_mixing in {"basis_expansion_layer", "implicit_fixed_point"}
            ):
                return "functional"
            if op_names & hyperbolic_ops:
                return "hyperbolic"
            if op_names & tropical_ops:
                return "tropical"
            if op_names & padic_ops:
                return "p-adic"
            if op_names & clifford_ops:
                return "clifford"
            return "euclidean"

        total_tested = 0
        total_survived = 0
        for row in rows:
            family = _family_from_row(row["graph_json"], row["arch_spec_json"])
            bucket = stats.get(family, stats["euclidean"])
            bucket["n_tested"] += 1
            total_tested += 1

            if row["stage1_passed"]:
                bucket["n_survived"] += 1
                total_survived += 1

        families = []
        for fam in family_order:
            entry = stats[fam]
            n_tested = entry["n_tested"]
            n_survived = entry["n_survived"]
            families.append(
                {
                    "family": fam,
                    "n_tested": n_tested,
                    "n_survived": n_survived,
                    "survival_rate": round(n_survived / n_tested, 4)
                    if n_tested > 0
                    else 0.0,
                    "tested_share": round(n_tested / total_tested, 4)
                    if total_tested > 0
                    else 0.0,
                    "survivor_share": round(n_survived / total_survived, 4)
                    if total_survived > 0
                    else 0.0,
                }
            )

        return {
            "families": families,
            "totals": {
                "n_tested": total_tested,
                "n_survived": total_survived,
            },
        }

    def mathspace_operator_impact(self) -> Dict:
        """Impact summary for math-space operators and families.

        Reports tested counts, S1/validation pass rates, novelty signal,
        and baseline-win rates for each math-space operator family.
        """
        cached = getattr(self.nb, "_program_results_columns", None)
        if cached is None:
            cached = {
                row["name"]
                for row in self.nb.conn.execute(
                    "PRAGMA table_info(program_results)"
                ).fetchall()
                if row and row["name"]
            }
            self.nb._program_results_columns = cached
        columns = cached
        validation_col = (
            "validation_passed"
            if "validation_passed" in columns
            else "NULL AS validation_passed"
        )
        baseline_col = (
            "validation_baseline_ratio"
            if "validation_baseline_ratio" in columns
            else "NULL AS validation_baseline_ratio"
        )

        rows = self.nb.conn.execute(f"""
            SELECT graph_json, stage1_passed, {validation_col}, novelty_score, {baseline_col}
            FROM program_results
            WHERE graph_json IS NOT NULL
        """).fetchall()

        op_family = {
            "poincare_add": "hyperbolic",
            "exp_map": "hyperbolic",
            "log_map": "hyperbolic",
            "hyp_linear": "hyperbolic",
            "hyp_distance": "hyperbolic",
            "hyp_tangent_nonlinear": "hyperbolic",
            "tropical_matmul": "tropical",
            "tropical_add": "tropical",
            "tropical_attention": "tropical",
            "tropical_center": "tropical",
            "padic_expand": "p-adic",
            "ultrametric_attention": "p-adic",
            "padic_gate": "p-adic",
            "geometric_product": "clifford",
            "rotor_transform": "clifford",
            "grade_select": "clifford",
            "grade_mix": "clifford",
        }
        tracked_ops = set(op_family.keys())

        if not rows:
            return {
                "available": False,
                "totals": {
                    "n_programs_with_graph": 0,
                    "n_programs_with_mathspace": 0,
                    "n_mathspace_ops_observed": 0,
                },
                "by_operator": [],
                "by_family": [],
                "explanation": "No graph-level program data available for math-space impact analysis.",
            }

        by_operator: Dict[str, Dict[str, float]] = {}
        by_family: Dict[str, Dict[str, float]] = {}
        programs_with_mathspace = 0

        def _ensure_bucket(
            store: Dict[str, Dict[str, float]], key: str
        ) -> Dict[str, float]:
            if key not in store:
                store[key] = {
                    "n_tested": 0,
                    "n_stage1_passed": 0,
                    "n_validation_passed": 0,
                    "n_baseline_wins": 0,
                    "novelty_sum": 0.0,
                    "novelty_count": 0,
                }
            return store[key]

        for row in rows:
            graph_json = row["graph_json"]
            if not graph_json:
                continue

            ops = self._extract_ops_fast(graph_json)
            if ops is None:
                ops = self._extract_ops_fallback(graph_json)
            if not ops:
                continue

            used_ops = sorted(tracked_ops.intersection(set(ops)))
            if not used_ops:
                continue

            programs_with_mathspace += 1
            used_families = sorted({op_family[op] for op in used_ops})
            stage1_passed = bool(row["stage1_passed"])
            validation_passed = bool(row["validation_passed"])
            novelty = self._as_float(row["novelty_score"])
            baseline_ratio = self._as_float(row["validation_baseline_ratio"])
            baseline_win = baseline_ratio is not None and baseline_ratio < 1.0

            for op_name in used_ops:
                bucket = _ensure_bucket(by_operator, op_name)
                bucket["n_tested"] += 1
                if stage1_passed:
                    bucket["n_stage1_passed"] += 1
                if validation_passed:
                    bucket["n_validation_passed"] += 1
                if baseline_win:
                    bucket["n_baseline_wins"] += 1
                if novelty is not None:
                    bucket["novelty_sum"] += novelty
                    bucket["novelty_count"] += 1

            for family in used_families:
                bucket = _ensure_bucket(by_family, family)
                bucket["n_tested"] += 1
                if stage1_passed:
                    bucket["n_stage1_passed"] += 1
                if validation_passed:
                    bucket["n_validation_passed"] += 1
                if baseline_win:
                    bucket["n_baseline_wins"] += 1
                if novelty is not None:
                    bucket["novelty_sum"] += novelty
                    bucket["novelty_count"] += 1

        def _finalize(
            rows_by_key: Dict[str, Dict[str, float]], label_key: str
        ) -> List[Dict[str, float]]:
            finalized: List[Dict[str, float]] = []
            for key, bucket in rows_by_key.items():
                n_tested = int(bucket["n_tested"])
                if n_tested <= 0:
                    continue
                novelty_count = int(bucket["novelty_count"])
                stage1_rate = float(bucket["n_stage1_passed"]) / n_tested
                validation_rate = float(bucket["n_validation_passed"]) / n_tested
                baseline_win_rate = float(bucket["n_baseline_wins"]) / n_tested
                sample_weight = min(1.0, n_tested / 25.0)
                trust_score = (
                    0.5 * stage1_rate + 0.3 * validation_rate + 0.2 * baseline_win_rate
                ) * sample_weight
                if trust_score >= 0.6 and n_tested >= 20:
                    trust_label = "high"
                elif trust_score >= 0.35 and n_tested >= 8:
                    trust_label = "medium"
                else:
                    trust_label = "low"
                finalized.append(
                    {
                        label_key: key,
                        "n_tested": n_tested,
                        "n_stage1_passed": int(bucket["n_stage1_passed"]),
                        "n_validation_passed": int(bucket["n_validation_passed"]),
                        "n_baseline_wins": int(bucket["n_baseline_wins"]),
                        "stage1_pass_rate": round(stage1_rate, 4),
                        "validation_pass_rate": round(validation_rate, 4),
                        "baseline_win_rate": round(baseline_win_rate, 4),
                        "trust_score": round(trust_score, 4),
                        "trust_label": trust_label,
                        "avg_novelty_score": (
                            round(float(bucket["novelty_sum"]) / novelty_count, 4)
                            if novelty_count > 0
                            else None
                        ),
                    }
                )
            return sorted(finalized, key=lambda row: (-row["n_tested"], row[label_key]))

        by_operator_rows = _finalize(by_operator, "op_name")
        by_family_rows = _finalize(by_family, "family")
        top_trustworthy_ops = sorted(
            by_operator_rows,
            key=lambda row: (
                -(row.get("trust_score") or 0.0),
                -(row.get("n_tested") or 0),
                row.get("op_name") or "",
            ),
        )[:3]

        top_op = by_operator_rows[0]["op_name"] if by_operator_rows else None
        explanation = (
            f"Observed {len(by_operator_rows)} math-space ops across {programs_with_mathspace}/{len(rows)} programs with graph traces. "
            f"Most common op: {top_op}."
            if top_op
            else "No math-space operators were observed in current graph traces."
        )

        return {
            "available": len(by_operator_rows) > 0,
            "totals": {
                "n_programs_with_graph": len(rows),
                "n_programs_with_mathspace": programs_with_mathspace,
                "n_mathspace_ops_observed": len(by_operator_rows),
            },
            "by_operator": by_operator_rows,
            "by_family": by_family_rows,
            "top_trustworthy_operators": top_trustworthy_ops,
            "explanation": explanation,
        }

    # ── Code failure error types — always display_only ──
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

    def compute_insights(self) -> List[Dict]:
        """Generate data-driven insights with statistical evidence.

        Every insight carries:
        - ``alpha``/``beta_``: Beta-Binomial posterior from observed counts
        - ``evidence_json``: structured proof (test, p_value, effect_size, n)
        - ``display_only``: True for failure_mode and code-failure insights

        Returns ``[{"content": str, "category": str, ...}, ...]``
        """

        insights: List[Dict] = []

        # ── 1. Graph-size bucket analysis (structural) ──
        size_rows = self.nb.conn.execute("""
            SELECT graph_n_ops, stage1_passed
            FROM program_results
            WHERE graph_n_ops IS NOT NULL
        """).fetchall()
        if len(size_rows) >= 50:
            self._compute_graph_size_insights(size_rows, insights)

        # ── 2. Op success/failure insights ──
        op_rates = self.op_success_rates()
        if op_rates:
            self._compute_op_insights(op_rates, insights)

        # ── 3. Structural correlation insights (chi-squared) ──
        correlations = self.structural_correlations()
        if correlations:
            self._compute_structural_correlation_insights(
                correlations, size_rows, insights
            )

        # ── 4. Failure pattern insights (always display_only) ──
        failures = self.failure_patterns()
        if failures:
            self._compute_failure_insights(failures, insights)

        # ── 5. Op combination insights (composition) ──
        combos = self.top_op_combinations(5)
        if combos:
            self._compute_combo_insights(combos, insights)

        # ── 6. Overall progress ──
        summary = self.nb.get_dashboard_summary()
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

    def _compute_graph_size_insights(
        self,
        size_rows: list,
        insights: List[Dict],
    ) -> None:
        """Chi-squared test on graph-size buckets vs S1 pass rate."""
        from scipy.stats import chi2_contingency

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

        # Check if 13+ ops collapses
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
                    "alpha": float(
                        big_fail + 1
                    ),  # correct = predicted fail, actually failed
                    "beta_": float(
                        big_pass + 1
                    ),  # wrong = predicted fail, actually passed
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
        overall_rate = sum(s["n_stage1_passed"] for s in op_rates.values()) / max(
            sum(s["n_used"] for s in op_rates.values()), 1
        )

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
            # Use effect size to set alpha/beta: stronger effect = higher confidence
            # Map |effect| ∈ [0.3, 2.0] to confidence ∈ [0.55, 0.85]
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
        """Failure pattern insights — always display_only.

        Code failures (RuntimeError, TypeError, etc.) are explicitly separated
        from training failures (nan_loss, diverged, etc.).
        """
        for error_type, data in sorted(failures.items(), key=lambda x: -x[1]["total"])[
            :5
        ]:
            total = data["total"]
            if total < 10:
                continue
            is_code_failure = error_type in self._CODE_FAILURE_TYPES
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
                    "display_only": True,  # Always display-only for failures
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
                    "beta_": 1.0,  # We only see survivors; beta starts weak
                    "display_only": False,
                    "insight_level": "composition",
                    "evidence_json": {
                        "test": "co_occurrence_count",
                        "n_survivors": count,
                        "avg_novelty": round(avg_novelty, 4),
                    },
                }
            )
