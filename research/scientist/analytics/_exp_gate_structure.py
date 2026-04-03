"""Gate performance analysis and structural correlation mixin."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class _GateStructureMixin:
    """Causality gate metrics and graph-structure correlations."""

    __slots__ = ()

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
                correlation = float(np.corrcoef(discovery, validation)[0, 1])
            except (TypeError, ValueError):
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
                    corr = round(float(np.corrcoef(disc, val)[0, 1]), 4)
                except (TypeError, ValueError):
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
