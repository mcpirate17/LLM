"""Behavioral novelty fingerprinting (WS-5).

Novelty in the fab has been *structural* — an axis-tuple hash. Two ops with
different axes but identical behavior count as distinct, so the loop re-grades
behavioral clones forever. This module makes novelty *behavioral*: a fixed-length
fingerprint assembled from the scorecards a candidate already produces, and a
distance to its nearest catalog neighbor.

The fingerprint dims are stable scalar summaries (so they are comparable across
candidates and cheap to persist into the ledger): the learning / binding /
state-tracking subscores ``ranking`` already computes, plus the gate metrics
(ERF density + decay, nano-bind accuracy, induction accuracy, mean AR recall,
effective binding range). Missing fields default to 0.0, so a coarse fingerprint
is still computed from whatever a partially-graded or legacy ledger row carries.

Two consumers (per the plan):
  - dedupe: a candidate within ``clone_eps`` of an existing entry is a behavioral
    clone and can be skipped even if its axis hash differs.
  - ranking: ``novelty_distance`` becomes an objective in the WS-4 Pareto vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..improver.ranking import (
    binding_subscore,
    learning_subscore,
    state_tracking_subscore,
)

# Ordered fingerprint dimensions — the vector layout. Stable scalars only.
FINGERPRINT_KEYS: tuple[str, ...] = (
    "learning",
    "binding",
    "state_tracking",
    "erf_density",
    "erf_decay_slope",
    "nb_max_accuracy",
    "ind_max_accuracy",
    "mean_relative_recall",
    "range_effective_distance",
)
# Default state-degeneracy threshold in z-scored Euclidean space.
DEFAULT_DEGENERACY_EPS = 0.75


def _mean_relative_recall(capability_scorecard: dict[str, Any] | None) -> float:
    if not capability_scorecard:
        return 0.0
    recalls = capability_scorecard.get("relative_recall_per_probe") or {}
    if not recalls:
        return 0.0
    return sum(max(0.0, float(v)) for v in recalls.values()) / len(recalls)


def operational_spectrum(
    probe_scorecard: dict[str, Any] | None,
    capability_scorecard: dict[str, Any] | None,
) -> dict[str, float]:
    """Named fixed-key spectrum from the live in-context + capability scorecards."""
    cap = capability_scorecard or {}
    return {
        "learning": learning_subscore(probe_scorecard),
        "binding": binding_subscore(capability_scorecard),
        "state_tracking": state_tracking_subscore(probe_scorecard),
        "erf_density": float(cap.get("erf_density") or 0.0),
        "erf_decay_slope": float(cap.get("erf_decay_slope") or 0.0),
        "nb_max_accuracy": float(cap.get("nb_max_accuracy") or 0.0),
        "ind_max_accuracy": float(cap.get("ind_max_accuracy") or 0.0),
        "mean_relative_recall": _mean_relative_recall(capability_scorecard),
        "range_effective_distance": float(cap.get("range_effective_distance") or 0.0),
    }


def spectrum_from_metadata(metadata: dict[str, Any]) -> dict[str, float]:
    """Reconstruct an operational spectrum from persisted grade-metadata.

    Prefers a stored ``operational_spectrum`` (full); falls back to the coarse
    subset every grade record already carries.
    """
    stored = metadata.get("operational_spectrum")
    if isinstance(stored, dict) and stored:
        return {k: float(stored.get(k) or 0.0) for k in FINGERPRINT_KEYS}
    coarse = {k: 0.0 for k in FINGERPRINT_KEYS}
    coarse["erf_density"] = float(metadata.get("erf_density") or 0.0)
    coarse["nb_max_accuracy"] = float(metadata.get("nb_max_accuracy") or 0.0)
    coarse["range_effective_distance"] = float(
        metadata.get("range_effective_distance") or 0.0
    )
    if metadata.get("can_bind"):
        coarse["binding"] = 1.0
    return coarse


def to_vector(fp: dict[str, float]) -> np.ndarray:
    return np.array([float(fp.get(k, 0.0)) for k in FINGERPRINT_KEYS], dtype=float)


@dataclass(slots=True)
class Normalizer:
    """Per-dimension z-scoring over a catalog of spectra."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, spectra: Sequence[dict[str, float]]) -> "Normalizer":
        if not spectra:
            n = len(FINGERPRINT_KEYS)
            return cls(mean=np.zeros(n), std=np.ones(n))
        M = np.vstack([to_vector(fp) for fp in spectra])
        std = M.std(axis=0)
        std[std < 1e-9] = 1.0  # constant dims contribute nothing, never divide by 0
        return cls(mean=M.mean(axis=0), std=std)

    def transform(self, fp: dict[str, float]) -> np.ndarray:
        return (to_vector(fp) - self.mean) / self.std


def orthogonality_radius(
    fp: dict[str, float],
    catalog: Sequence[dict[str, float]],
    *,
    normalizer: Normalizer | None = None,
) -> float:
    """Min z-scored Euclidean distance from ``fp`` to any catalog spectrum.

    Returns ``inf`` for an empty catalog. A self-identical neighbor yields 0.0.
    """
    if not catalog:
        return float("inf")
    norm = normalizer or Normalizer.fit(catalog)
    target = norm.transform(fp)
    M = (np.vstack([to_vector(c) for c in catalog]) - norm.mean) / norm.std
    return float(np.linalg.norm(M - target, axis=1).min())


def is_degenerate(distance: float, *, eps: float = DEFAULT_DEGENERACY_EPS) -> bool:
    return distance <= eps


# Frontier baseline anchors (Calibrated 2026-06-07).
# These represent the 'softmax twin' and 'SOTA recurrence' behaviors we
# want to move away from. Orthogonality is anchored against these.
FRONTIER_SPECTRA: dict[str, dict[str, float]] = {
    "softmax_attention": {
        "learning": 0.42,
        "binding": 0.58,
        "state_tracking": 0.05,
        "erf_density": 0.12,
        "erf_decay_slope": -0.8,
        "nb_max_accuracy": 0.98,
        "ind_max_accuracy": 0.55,
        "mean_relative_recall": 0.85,
        "range_effective_distance": 1024,
    },
    "gpt2": {
        "learning": 0.45,
        "binding": 0.62,
        "state_tracking": 0.08,
        "erf_density": 0.15,
        "erf_decay_slope": -0.75,
        "nb_max_accuracy": 1.0,
        "ind_max_accuracy": 0.62,
        "mean_relative_recall": 0.92,
        "range_effective_distance": 1024,
    },
    "mamba2": {
        "learning": 0.38,
        "binding": 0.45,
        "state_tracking": 0.65,
        "erf_density": 0.08,
        "erf_decay_slope": -0.4,
        "nb_max_accuracy": 0.85,
        "ind_max_accuracy": 0.02,
        "mean_relative_recall": 0.42,
        "range_effective_distance": 256,
    },
}
