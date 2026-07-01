"""Geometric-novelty MAP-Elites axis: distance from the softmax/attention basin (NM-10).

Under global top-K (or a capability-only archive) the fab population drifts back
toward the familiar softmax-shaped convex averagers — the mission's pathology.
This module turns the MEASURED softmax-twin signature (NM-11,
``algebraic_properties``) into a MAP-Elites behavior axis so the archive spreads
candidates across the *distance from the softmax basin*. A far-from-softmax
region that the population has not yet covered becomes an EMPTY niche, and the
archive's ``empty_niches`` / archive-guided exploration then steers fresh
sampling toward that novel geometry instead of piling onto the basin.

``softmax_basin_distance = 1 - softmax_twin_score`` ∈ [0, 1]: 0 = sits in the
softmax basin (a twin), 1 = maximally distant novel geometry.
"""

from __future__ import annotations

from typing import Callable, Mapping

from research.synthesis.quality_diversity import BehaviorAxis

from component_fab.proposer.algebraic_properties import (
    AlgebraicPropertyProbe,
    Operator,
)

#: Descriptor key the novelty axis bins on.
SOFTMAX_BASIN_DISTANCE = "softmax_basin_distance"

#: 3 bins: [0, 0.15) softmax basin (twin ≳ 0.85), [0.15, 0.5) transitional,
#: [0.5, 1] genuinely novel geometry. Coarse-first, matching the archive's other
#: default axes.
_NOVELTY_EDGES: tuple[float, ...] = (0.15, 0.5)


def softmax_basin_distance_axis(
    edges: tuple[float, ...] = _NOVELTY_EDGES,
) -> BehaviorAxis:
    """The MAP-Elites behavior axis binning on distance from the softmax basin."""
    return BehaviorAxis(SOFTMAX_BASIN_DISTANCE, edges)


def with_novelty_axis(
    base_axes: tuple[BehaviorAxis, ...],
    *,
    edges: tuple[float, ...] = _NOVELTY_EDGES,
) -> tuple[BehaviorAxis, ...]:
    """Append the softmax-basin-distance axis to an existing behavior-axis tuple.

    Idempotent: if the axis is already present, ``base_axes`` is returned
    unchanged so callers can wrap freely without duplicating the dimension.
    """
    if any(axis.name == SOFTMAX_BASIN_DISTANCE for axis in base_axes):
        return base_axes
    return (*base_axes, softmax_basin_distance_axis(edges))


def distance_from_twin_score(softmax_twin_score: float) -> float:
    """``1 - softmax_twin_score`` clamped to [0, 1]; higher = more novel geometry."""
    return max(0.0, min(1.0, 1.0 - float(softmax_twin_score)))


def measure_softmax_basin_distance(
    f: Operator,
    *,
    dim: int = 32,
    seq_len: int = 16,
    n_seeds: int = 2,
    device: str = "cpu",
) -> float:
    """Softmax-basin distance for a bare ``[B, L, D] -> [B, L, D]`` callable."""
    probe = AlgebraicPropertyProbe(
        seq_len=seq_len, dim=dim, n_seeds=n_seeds, device=device
    )
    props = probe.measure(f)
    return distance_from_twin_score(props.softmax_twin_score)


def measure_model_softmax_basin_distance(
    factory: Callable[[int], object],
    *,
    seq_len: int = 16,
    vocab: int = 64,
    n_seeds: int = 2,
    device: str = "cpu",
) -> float | None:
    """Softmax-basin distance for a model built by ``factory(seed)``.

    Uses the ``embed`` + ``_fingerprint_forward_from_embed`` contract, so it probes
    at the model's own embedding width. Returns ``None`` when no seed is probeable
    (the caller must decide whether to drop the candidate — never silently default
    an unmeasured op into a niche).
    """
    probe = AlgebraicPropertyProbe(
        seq_len=seq_len, vocab=vocab, n_seeds=n_seeds, device=device
    )
    props = probe.measure_model(factory)
    if props is None:
        return None
    return distance_from_twin_score(props.softmax_twin_score)


def add_novelty_descriptor(
    descriptors: Mapping[str, float], softmax_basin_distance: float
) -> dict[str, float]:
    """Return ``descriptors`` plus the novelty axis value (does not mutate input)."""
    out = dict(descriptors)
    out[SOFTMAX_BASIN_DISTANCE] = float(softmax_basin_distance)
    return out
