"""Geometric novelty: distance of an operator's physics signature from the
softmax / attention basin (NM-10).

The MAP-Elites archive in :mod:`quality_diversity` bins candidates by their
MEASURED behavior descriptors and keeps the best per niche. By default it bins on
the raw symmetry classes (:func:`physics_descriptors.physics_behavior_axes`):
position-aware vs set-like, translation-equivariant vs absolute, linear vs
nonlinear, contractive vs expansive. That spreads the population across
*qualitative* niches, but it does not by itself reward distance from the
softmax / attention behavior we are trying to beat — a niche can be qualitatively
distinct yet still sit inside the convex-averaging basin that makes a mechanism a
"softmax twin".

This module adds the missing coordinate. ``geometric_novelty`` is the
standardized distance from a candidate's physics fingerprint to the NEAREST
softmax-shaped basin signature — softmax-QK attention (the canonical
score-weighted blend) and uniform mean-pool (the row-stochastic blend limit), the
two convex-averaging tells. Folding it in as a MAP-Elites behavior axis makes the
archive keep an elite at EACH distance-from-softmax, so a geometrically far
mechanism survives even when its raw capability is modest — the search is driven
toward unexplored, non-softmax geometry instead of piling onto the global
maximum (which, under a capability fitness, is usually softmax-shaped).

Layering: this module imports only :mod:`physics_descriptors` and
:mod:`quality_diversity` (both inside ``research.synthesis``). It NEVER imports
``component_fab``. The NM-11 measured ``softmax_twin_score`` (component_fab side)
may be supplied as an optional ``twin_score`` float to sharpen novelty
(twin≈1 → low novelty) — the caller bridges that layer boundary, not this module.

All basin signatures are MEASURED (not hand-declared) on fixed-seed reference
operators, so the basin is reproducible: the same philosophy as the NM-11
detector, which replaced hand-declared ``softmax_twin_like`` with a measured
signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from .physics_descriptors import (
    PHYSICS_DESCRIPTOR_NAMES,
    PhysicsDescriptorProbe,
    physics_behavior_axes,
)
from .quality_diversity import BehaviorAxis

__all__ = [
    "BasinSignature",
    "DESCRIPTOR_SCALE",
    "SOFTMAX_BASIN_NAMES",
    "NOVELTY_AXIS_NAME",
    "softmax_basin_signatures",
    "geometric_novelty",
    "augment_with_novelty",
    "novelty_behavior_axis",
    "novelty_aware_axes",
]

# Curated per-axis standardization scale = the characteristic half-range a
# typical sequence operator spans on each physics descriptor (grounded in the
# documented descriptor semantics on physics_descriptors). The distance is
# reported in these units so no single axis (e.g. energy_gain's wider range)
# dominates. Curated, like archive_guided._OP_BEHAVIOR_SIGNATURE — not fit to
# data; the BASIN SIGNATURES are measured, these are stable normalization units.
DESCRIPTOR_SCALE: dict[str, float] = {
    "perm_equivariance": 0.35,  # (0,1]; softmax-attn low, pointwise 1.0
    "shift_equivariance": 0.35,  # (0,1]; causal/abs-pos low, conv/SSM 1.0
    "scale_homogeneity": 0.35,  # (0,1]; softmax/norm low, linear 1.0
    "spectral_radius": 0.30,  # ~[0.5,1.5]; rank-deficient ops <1
    "energy_gain": 0.50,  # ~[0.3,2]; mean-pool / pooling <1
}

# The softmax-shaped basins: the two convex-averaging / row-stochastic tells.
# softmax-QK attention = score-weighted blend (the mechanism to beat);
# mean_pool = the uniform-blend limit. A candidate near EITHER is geometrically
# close to softmax. Both are measured, not hand-declared.
SOFTMAX_BASIN_NAMES: tuple[str, ...] = ("softmax_attention", "mean_pool")

NOVELTY_AXIS_NAME = "geometric_novelty"

# Fixed seed for the reference basin operators so the measured basin is a
# reproducible reference point (the probe itself uses its own generators).
_BASIN_SEED = 20260701


class _SoftmaxAttentionBasin(nn.Module):
    """Canonical softmax-QK attention with random projections, no position info.

    ``y = softmax(Q K^T / sqrt(d)) V`` with bias-free random projections. This is
    the score-weighted convex-averaging basin — the frontier mechanism the
    project is funded to beat. Untrained + fixed-seed so its physics fingerprint
    is a stable reference.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self._scale = dim**-0.5

    def forward(self, x: Tensor) -> Tensor:
        scores = (self.q(x) @ self.k(x).transpose(-2, -1)) * self._scale
        attn = torch.softmax(scores, dim=-1)
        return attn @ self.v(x)


class _MeanPoolBasin(nn.Module):
    """Uniform mean-pool over the sequence — the row-stochastic blend limit.

    ``y[i] = mean_j x[j]``: every output is the same convex combination of inputs
    (weights all ``1/L``). Geometrically this is the degenerate end of the
    convex-averaging basin — no content routing at all, but still row-stochastic
    all-to-all blending, so it shares softmax's structural tell.
    """

    def forward(self, x: Tensor) -> Tensor:
        pooled = x.mean(dim=1, keepdim=True)
        return pooled.expand(-1, x.shape[1], -1)


@dataclass(frozen=True, slots=True)
class BasinSignature:
    """A measured physics fingerprint of one softmax-shaped reference basin."""

    name: str
    descriptors: Mapping[str, float]


# Per-process, per-dim cache: the basin is a fixed reference point (not
# data-dependent), so it is measured once per dim and reused. Keyed by dim so a
# caller probing candidates at a different dim gets a matching reference.
_BASIN_CACHE: dict[int, list[BasinSignature]] = {}


def _build_basin_operators(dim: int) -> dict[str, nn.Module]:
    """Construct the fixed-seed reference basin operators at ``dim``."""
    torch.manual_seed(_BASIN_SEED)
    return {
        "softmax_attention": _SoftmaxAttentionBasin(dim).eval(),
        "mean_pool": _MeanPoolBasin().eval(),
    }


def softmax_basin_signatures(
    *,
    dim: int = 32,
    probe: PhysicsDescriptorProbe | None = None,
) -> list[BasinSignature]:
    """MEASURED physics signatures of the softmax-shaped basins (cached per dim).

    Each reference basin operator is characterised by a fixed-seed
    ``PhysicsDescriptorProbe`` so the basin is reproducible across calls. The
    result is cached for the (dim) for the process; the basin is a fixed
    reference point, not something that depends on the candidates.
    """
    cached = _BASIN_CACHE.get(dim)
    if cached is not None:
        return cached
    p = probe or PhysicsDescriptorProbe(dim=dim, n_seeds=3)
    ops = _build_basin_operators(p.dim)
    sigs = [
        BasinSignature(name=name, descriptors=p.describe_operator(ops[name]))
        for name in SOFTMAX_BASIN_NAMES
    ]
    for sig in sigs:
        _check_basin_descriptors(sig.descriptors)
    _BASIN_CACHE[dim] = sigs
    return sigs


def _check_basin_descriptors(descriptors: Mapping[str, float]) -> None:
    """A basin signature must cover the full physics fingerprint — fail loud."""
    missing = [n for n in PHYSICS_DESCRIPTOR_NAMES if n not in descriptors]
    if missing:
        raise RuntimeError(
            f"basin signature missing physics descriptors {missing}; "
            f"got {sorted(descriptors)}"
        )
    for name in PHYSICS_DESCRIPTOR_NAMES:
        val = float(descriptors[name])
        if val != val or val in (float("inf"), float("-inf")):  # NaN/inf guard
            raise RuntimeError(f"basin descriptor {name!r} is non-finite: {val}")


def _standardized_distance(
    descriptors: Mapping[str, float],
    basin: Mapping[str, float],
) -> float:
    """Standardized Euclidean distance over the shared physics descriptor axes.

    Each axis is divided by its curated :data:`DESCRIPTOR_SCALE` so the distance
    is in 'typical-operator half-range' units and no axis dominates. Fails loud if
    a basin axis is missing from the candidate — a candidate measured on a
    different descriptor family is a caller bug, not a silent 0 distance.
    """
    total = 0.0
    for name in PHYSICS_DESCRIPTOR_NAMES:
        if name not in basin:
            continue
        if name not in descriptors:
            raise KeyError(
                f"basin axis {name!r} missing from candidate descriptors; "
                f"candidate has {sorted(descriptors)}"
            )
        delta = float(descriptors[name]) - float(basin[name])
        total += (delta / DESCRIPTOR_SCALE[name]) ** 2
    return total**0.5


def geometric_novelty(
    descriptors: Mapping[str, float],
    *,
    basins: Sequence[BasinSignature] | None = None,
    twin_score: float | None = None,
) -> float:
    """Geometric novelty of an operator = standardized distance to the NEAREST
    softmax-shaped basin, optionally sharpened by the NM-11 twin score.

    Args:
        descriptors: the candidate's measured physics fingerprint.
        basins: defaults to the MEASURED softmax basins. Pass a custom set to
            redefine the basin (e.g. an attention-only reference).
        twin_score: NM-11 ``softmax_twin_score`` in ``[0, 1]``. When provided,
            the novelty is multiplied by ``(0.25 + 0.75 * (1 - twin_score))`` so a
            measured softmax-twin (twin≈1) collapses toward 0 even if its physics
            fingerprint happened to land off-basin, and a confirmed non-twin
            (twin≈0) keeps the full distance. The 0.25 floor keeps the
            physics-distance audible — twin_score REFINES the geometry signal, it
            does not erase it.

    Returns:
        non-negative standardized distance to the nearest softmax-shaped basin.
    """
    sigs = basins if basins is not None else softmax_basin_signatures()
    if not sigs:
        raise ValueError("geometric_novelty needs at least one basin signature")
    dist = min(_standardized_distance(descriptors, sig.descriptors) for sig in sigs)
    if twin_score is None:
        return dist
    ts = float(twin_score)
    if not 0.0 <= ts <= 1.0:
        raise ValueError(f"twin_score must be in [0,1]; got {ts}")
    return dist * (0.25 + 0.75 * (1.0 - ts))


def augment_with_novelty(
    descriptors: Mapping[str, float],
    *,
    basins: Sequence[BasinSignature] | None = None,
    twin_score: float | None = None,
) -> dict[str, float]:
    """Return a copy of ``descriptors`` with ``geometric_novelty`` added.

    Convenience for the discovery loop: measure physics once, then call this so
    the archive's ``niche_for`` finds the novelty axis. Does not mutate the input.
    """
    out = dict(descriptors)
    out[NOVELTY_AXIS_NAME] = geometric_novelty(
        descriptors, basins=basins, twin_score=twin_score
    )
    return out


def novelty_behavior_axis(
    edges: tuple[float, ...] = (0.75, 1.75, 4.0, 16.0),
) -> BehaviorAxis:
    """MAP-Elites behavior axis over geometric novelty (distance from softmax).

    5-bin edges in standardized-distance units: ``<0.75`` ≈ inside the softmax
    basin, ``[0.75, 1.75)`` ≈ adjacent, ``[1.75, 4.0)`` ≈ the far-from-softmax
    bulk (the region the mission wants illuminated), ``[4.0, 16.0)`` ≈ very far,
    ``>=16.0`` ≈ a blow-up quarantine for numerically-expansive descriptors
    (saturating ``energy_gain``/``spectral_radius``) so they cannot squat on the
    far niches that *stable* novel mechanisms should own.

    The 3-bin default ``(0.75, 1.75)`` did zero selection work among far elites:
    the G3 within-archive audit (``research/tools/audit_novelty_axis_distribution.py``,
    commit ``bbec2a4a``) found all 19 "far" deep-registry elites collapsed into one
    bin spanning 1.805–175.589 (97×). The two extra edges sit in the natural data
    gaps 3.298→4.030 and 13.75→29.74. Edges are a parameter so a finer archive can
    refine.
    """
    return BehaviorAxis(NOVELTY_AXIS_NAME, edges)


def novelty_aware_axes(
    novelty_edges: tuple[float, ...] = (0.75, 1.75, 4.0, 16.0),
) -> tuple[BehaviorAxis, ...]:
    """Physics behavior axes + the geometric-novelty axis.

    The archive axis set for a novelty-aware MAP-Elites run: the qualitative
    symmetry classes (:func:`physics_behavior_axes`) crossed with
    distance-from-softmax, so an elite is kept per (symmetry class × novelty)
    niche. The far-from-softmax niches are exactly the ones the mission wants
    populated; without this axis a single capability-fitness MAP-Elites archive
    lets softmax-shaped mechanisms crowd them out.
    """
    return (*physics_behavior_axes(), novelty_behavior_axis(novelty_edges))
