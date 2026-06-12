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
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..improver.ranking import (
    binding_subscore,
    learning_subscore,
    state_tracking_subscore,
)
from ..state.ledger import DEFAULT_LEDGER_PATH, iter_jsonl_records, latest_by_key

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
# Default behavioral-clone threshold in z-scored Euclidean space.
DEFAULT_CLONE_EPS = 0.75


def _mean_relative_recall(capability_scorecard: dict[str, Any] | None) -> float:
    if not capability_scorecard:
        return 0.0
    recalls = capability_scorecard.get("relative_recall_per_probe") or {}
    if not recalls:
        return 0.0
    return sum(max(0.0, float(v)) for v in recalls.values()) / len(recalls)


def behavior_fingerprint(
    probe_scorecard: dict[str, Any] | None,
    capability_scorecard: dict[str, Any] | None,
) -> dict[str, float]:
    """Named fixed-key fingerprint from the live in-context + capability scorecards."""
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


def fingerprint_from_metadata(metadata: dict[str, Any]) -> dict[str, float]:
    """Reconstruct a fingerprint from a persisted ledger grade-metadata dict.

    Prefers a stored ``behavior_fingerprint`` (full, written going forward); falls
    back to the coarse subset every grade record already carries (erf_density,
    nb_max_accuracy, range_effective_distance) so legacy rows still place in the
    behavioral space, just less precisely.
    """
    stored = metadata.get("behavior_fingerprint")
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
    """Per-dimension z-scoring over a catalog of fingerprints."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, fingerprints: Sequence[dict[str, float]]) -> "Normalizer":
        if not fingerprints:
            n = len(FINGERPRINT_KEYS)
            return cls(mean=np.zeros(n), std=np.ones(n))
        M = np.vstack([to_vector(fp) for fp in fingerprints])
        std = M.std(axis=0)
        std[std < 1e-9] = 1.0  # constant dims contribute nothing, never divide by 0
        return cls(mean=M.mean(axis=0), std=std)

    def transform(self, fp: dict[str, float]) -> np.ndarray:
        return (to_vector(fp) - self.mean) / self.std


def novelty_distance(
    fp: dict[str, float],
    catalog: Sequence[dict[str, float]],
    *,
    normalizer: Normalizer | None = None,
) -> float:
    """Min z-scored Euclidean distance from ``fp`` to any catalog fingerprint.

    Returns ``inf`` for an empty catalog (nothing to be a clone of). A
    self-identical neighbor yields 0.0.
    """
    if not catalog:
        return float("inf")
    norm = normalizer or Normalizer.fit(catalog)
    target = norm.transform(fp)
    M = (np.vstack([to_vector(c) for c in catalog]) - norm.mean) / norm.std
    return float(np.linalg.norm(M - target, axis=1).min())


def is_clone(distance: float, *, clone_eps: float = DEFAULT_CLONE_EPS) -> bool:
    return distance <= clone_eps


def catalog_from_ledger(
    ledger_path: Path | str = DEFAULT_LEDGER_PATH,
) -> list[dict[str, float]]:
    """Behavioral fingerprints for every graded ledger entry (latest per id)."""
    latest = latest_by_key(
        (r for r in iter_jsonl_records(Path(ledger_path)) if r.get("event") == "grade"),
        "proposal_id",
    )
    return [fingerprint_from_metadata(g.get("metadata") or {}) for g in latest.values()]
