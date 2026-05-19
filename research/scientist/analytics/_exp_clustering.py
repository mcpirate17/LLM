"""Experiment clustering mixin — k-means with model selection."""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


def _vectorized_kmeans(
    X_norm: np.ndarray,
    k: int,
    dataset_signature: str,
    salt: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Deterministic K-Means++ init followed by Lloyd iteration."""
    seed_hex = hashlib.md5(
        f"{dataset_signature}:{salt}".encode(),
        usedforsecurity=False,
    ).hexdigest()
    first_idx = int(seed_hex[:8], 16) % len(X_norm)
    centroids = [X_norm[first_idx]]

    for _ in range(1, k):
        dists = np.min(cdist(X_norm, np.array(centroids)), axis=1)
        next_idx = np.argmax(dists)
        centroids.append(X_norm[next_idx])
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


def _vectorized_silhouette(
    X_norm: np.ndarray,
    assignments: np.ndarray,
    dist_matrix: np.ndarray,
) -> float:
    """Compute mean silhouette score."""
    unique = np.unique(assignments)
    if len(unique) < 2:
        return 0.0

    n_rows = len(X_norm)
    intra = np.zeros(n_rows, dtype=np.float32)
    nearest_other = np.full(n_rows, np.inf, dtype=np.float32)
    singleton = np.zeros(n_rows, dtype=bool)

    for cluster in unique:
        mask = assignments == cluster
        cluster_size = int(mask.sum())
        if cluster_size > 1:
            intra[mask] = dist_matrix[np.ix_(mask, mask)].sum(axis=1) / (
                cluster_size - 1
            )
        else:
            singleton[mask] = True
        mean_to_cluster = dist_matrix[:, mask].mean(axis=1)
        nearest_other[~mask] = np.minimum(nearest_other[~mask], mean_to_cluster[~mask])

    denom = np.maximum(np.maximum(intra, nearest_other), 1e-9)
    scores = (nearest_other - intra) / denom
    scores[singleton] = 0.0
    scores[~np.isfinite(scores)] = 0.0
    return float(scores.mean())


_TRAJECTORY_ZERO_KEYS = [
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


def _compute_momentum(e: Dict, seq: np.ndarray, window: int) -> np.ndarray:
    """Compute momentum features and outcome proxy. Returns proxy array."""
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
    return proxy


def _compute_transitions(e: Dict, seq: np.ndarray, proxy: np.ndarray) -> None:
    """Compute transition timing, density, entropy, and change-point features."""
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

        n_deltas = len(deltas)
        seg = max(1, n_deltas // 3)
        window_means = [
            np.mean(deltas[i * seg : (i + 1) * seg]) if i * seg < n_deltas else 0.0
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


def _compute_recovery_lag(
    e: Dict, seq: np.ndarray, proxy: np.ndarray, window: int
) -> None:
    """Compute recovery lag feature."""
    early_baseline = np.mean(proxy[:window])
    trough_idx = np.argmin(proxy)
    recovery_idx = np.where(proxy[trough_idx + 1 :] >= early_baseline)[0]
    e["recovery_lag"] = (
        (recovery_idx[0] + 1) / (len(seq) - 1)
        if len(recovery_idx) > 0
        else (1.0 if len(seq) > 1 else 0.0)
    )


def _compute_trajectory_features(experiments: List[Dict], seq_rows: list) -> None:
    """Compute momentum, volatility, and transition features in-place."""
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
            e.update({k: 0.0 for k in _TRAJECTORY_ZERO_KEYS})
            continue

        window = max(1, len(seq) // 3)
        proxy = _compute_momentum(e, seq, window)
        _compute_transitions(e, seq, proxy)
        _compute_recovery_lag(e, seq, proxy, window)


# Feature keys used for clustering — shared between build and summary steps
CLUSTER_FEATURE_KEYS = [
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


class _ClusteringMixin:
    """Experiment clustering with k-means and model selection."""

    __slots__ = ()

    def experiment_clusters(self, n_clusters: int = 3) -> Optional[Dict]:
        """Cluster completed experiments by outcome profile.

        Uses high-performance NumPy vectorization for k-means clustering,
        silhouette scores, and model selection.
        """
        self.nb.flush_writes()
        experiments, exp_ids = self._load_cluster_experiments()
        if experiments is None:
            return None

        self._enrich_failure_data(experiments, exp_ids)
        self._enrich_trajectory_data(experiments, exp_ids)

        # Build normalized feature matrix
        X = np.array(
            [[e[k] for k in CLUSTER_FEATURE_KEYS] for e in experiments],
            dtype=np.float32,
        )
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        X_range = X_max - X_min
        X_norm = np.zeros_like(X)
        mask = X_range > 1e-9
        X_norm[:, mask] = (X[:, mask] - X_min[mask]) / X_range[mask]
        X_norm[:, CLUSTER_FEATURE_KEYS.index("best_loss_ratio")] = (
            1.0 - X_norm[:, CLUSTER_FEATURE_KEYS.index("best_loss_ratio")]
        )

        dataset_signature = "|".join(sorted(exp_ids))
        dist_matrix = cdist(X_norm, X_norm)
        max_k = min(max(2, n_clusters), len(X_norm) - 1)
        if max_k < 2:
            return None

        # Model selection: try k=2..max_k with 4 restarts each
        candidates = self._run_model_selection(
            X_norm, dist_matrix, max_k, dataset_signature
        )
        selected = max(candidates, key=lambda c: (c["score"], -c["k"]))
        k, best_run = selected["k"], selected["best"]
        assign, cents = best_run["assignments"], best_run["centroids"]

        # Consensus and stability
        stability, consensus = self._compute_stability(
            selected, assign, dist_matrix, cents, k, X_norm
        )

        # Build cluster summaries
        clusters = self._build_cluster_summaries(experiments, assign, k)
        self._describe_clusters(clusters)

        return {
            "n_experiments": len(experiments),
            "n_clusters": len(clusters),
            "feature_keys": CLUSTER_FEATURE_KEYS,
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

    def _load_cluster_experiments(
        self,
    ) -> tuple[Optional[List[Dict]], Optional[List[str]]]:
        """Load and filter experiments for clustering. Returns (experiments, exp_ids) or (None, None)."""
        rows = self.nb.conn.execute("""
            SELECT experiment_id, n_programs_generated, n_stage1_passed,
                   best_novelty_score, best_loss_ratio, duration_seconds
            FROM experiments
            WHERE status = 'completed' AND n_programs_generated > 0
            ORDER BY timestamp DESC LIMIT 2000
        """).fetchall()

        if len(rows) < 3:
            return None, None

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
            return None, None

        exp_ids = [e["experiment_id"] for e in experiments]
        return experiments, exp_ids

    def _enrich_failure_data(self, experiments: List[Dict], exp_ids: List[str]) -> None:
        """Attach failure rates and error diversity to experiments in-place."""
        placeholders = ",".join("?" * len(exp_ids))

        failure_rows = self.nb.conn.execute(
            f"""
            SELECT experiment_id, COUNT(*) as n_total,
                   SUM(CASE WHEN COALESCE(stage0_passed, 0) = 0 THEN 1 ELSE 0 END) as n_compile_fail,
                   SUM(CASE WHEN COALESCE(stage0_passed, 0) = 1 AND COALESCE(stage05_passed, 0) = 0 THEN 1 ELSE 0 END) as n_train_fail,
                   SUM(CASE WHEN COALESCE(stage05_passed, 0) = 1 AND COALESCE(stage1_passed, 0) = 0 THEN 1 ELSE 0 END) as n_stage1_fail
            FROM program_results_compat WHERE experiment_id IN ({placeholders}) GROUP BY experiment_id
        """,
            tuple(exp_ids),
        ).fetchall()

        fail_map = {r["experiment_id"]: r for r in failure_rows}

        error_rows = self.nb.conn.execute(
            f"""
            SELECT experiment_id, error_type, COUNT(*) as n
            FROM program_results_compat WHERE experiment_id IN ({placeholders})
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

    def _enrich_trajectory_data(
        self, experiments: List[Dict], exp_ids: List[str]
    ) -> None:
        """Attach trajectory features to experiments in-place."""
        placeholders = ",".join("?" * len(exp_ids))
        seq_rows = self.nb.conn.execute(
            f"""
            SELECT experiment_id, stage1_passed, loss_ratio, novelty_score
            FROM program_results_compat WHERE experiment_id IN ({placeholders})
            ORDER BY experiment_id ASC, timestamp ASC
        """,
            tuple(exp_ids),
        ).fetchall()

        _compute_trajectory_features(experiments, seq_rows)

    def _run_model_selection(
        self,
        X_norm: np.ndarray,
        dist_matrix: np.ndarray,
        max_k: int,
        dataset_signature: str,
    ) -> List[Dict]:
        """Run k-means for k=2..max_k with 4 restarts, return candidate list."""
        candidates = []
        for k_val in range(2, max_k + 1):
            runs = []
            for salt in range(4):
                assign, cents, inertia = _vectorized_kmeans(
                    X_norm, k_val, dataset_signature, salt
                )
                sil = _vectorized_silhouette(X_norm, assign, dist_matrix)
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
        return candidates

    @staticmethod
    def _compute_stability(
        selected: Dict,
        assign: np.ndarray,
        dist_matrix: np.ndarray,
        cents: np.ndarray,
        k: int,
        X_norm: np.ndarray,
    ) -> tuple[float, float]:
        """Compute clustering stability and consensus. Returns (stability, consensus)."""

        def _agreement(a1: np.ndarray, a2: np.ndarray) -> float:
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
        return stability, consensus

    @staticmethod
    def _build_cluster_summaries(
        experiments: List[Dict],
        assign: np.ndarray,
        k: int,
    ) -> List[Dict]:
        """Build per-cluster summary dicts from assignments."""
        clusters = []
        for ci in range(k):
            members = [
                experiments[i] for i in range(len(experiments)) if assign[i] == ci
            ]
            if not members:
                continue
            summary = {
                fk: round(float(np.mean([m[fk] for m in members])), 4)
                for fk in CLUSTER_FEATURE_KEYS
                if fk != "duration_seconds"
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
            # Rename to avg_ prefix
            for fk in ["s1_rate", "best_novelty", "best_loss_ratio"]:
                summary[f"avg_{fk}"] = summary.pop(fk)
            for fk in CLUSTER_FEATURE_KEYS:
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
        return clusters

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
