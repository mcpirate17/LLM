"""Quality-Diversity (MAP-Elites) archive over measured-descriptor behavior space.

The anti-collapse selection primitive for the diversity generator
(``research/notes/diversity_generator_charter_2026-06-03.md``). Under global
top-K selection both the NAS grammar population and the ``component_fab``
population collapse to the familiar (grammar-favored / tropical). This archive
bins candidates by their MEASURED behavior descriptors and keeps the best
candidate PER NICHE, so selection rewards filling empty regions of the behavior
space instead of piling onto the global maximum.

- Behavior characterization = the label-free measured descriptors from
  ``research.tools.measured_descriptors`` (``long_range_reach``,
  ``content_dependence``, ``content_match_gating``; ``effective_rank`` optional).
- Fitness = ``capability_score`` (→ frontier margin once labeled).
- Empty niches (``empty_niches`` / ``niche_bounds``) feed archive-guided
  generation back to the grammar / fab proposer.

Pure and dependency-light: operates on already-extracted descriptor dicts — no
torch, no model build, no numpy. The expensive measurement happens upstream.
"""

from __future__ import annotations

import bisect
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    "BehaviorAxis",
    "Elite",
    "MapElitesArchive",
    "default_behavior_axes",
    "select_diverse",
]


@dataclass(frozen=True, slots=True)
class BehaviorAxis:
    """One descriptor axis of the behavior space, discretized by ``edges``.

    ``edges`` are the internal bin boundaries (ascending, exclusive of the open
    ends), so an axis with ``edges=(0.05, 0.25)`` has ``3`` bins:
    ``[-inf, 0.05) , [0.05, 0.25) , [0.25, +inf)``. A value falling on an edge
    goes to the upper bin (``bisect_right`` semantics).
    """

    name: str
    edges: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.edges:
            raise ValueError(f"axis {self.name!r} needs at least one edge")
        if list(self.edges) != sorted(self.edges):
            raise ValueError(
                f"axis {self.name!r} edges must be ascending: {self.edges}"
            )

    @property
    def n_bins(self) -> int:
        return len(self.edges) + 1

    def bin_of(self, value: float) -> int:
        return bisect.bisect_right(self.edges, float(value))

    def bounds_of(self, index: int) -> tuple[float | None, float | None]:
        """Half-open [lo, hi) descriptor range of bin ``index`` (None = open end)."""
        if not 0 <= index < self.n_bins:
            raise IndexError(f"bin {index} out of range for axis {self.name!r}")
        lo = None if index == 0 else self.edges[index - 1]
        hi = None if index == len(self.edges) else self.edges[index]
        return lo, hi


@dataclass(frozen=True, slots=True)
class Elite:
    """The current best candidate occupying one niche."""

    key: str
    fitness: float
    niche: tuple[int, ...]
    descriptors: Mapping[str, float]
    payload: Any = None


# Coarse default behavior space — the three normalized ([0, 1]) measured
# descriptors with the strongest single-feature ROC vs induction-capable
# (content_dependence 0.81, long_range_reach 0.79, content_match_gating 0.64;
# see measured_descriptors._CAPABILITY_WEIGHTS). 3 bins/axis = 27 niches, the
# "coarse bins first" the charter calls for. ``effective_rank`` is omitted from
# the default because its scale is hidden-dim dependent; add it explicitly with
# absolute soft-rank edges once ``dim`` is fixed.
_DEFAULT_AXES: tuple[BehaviorAxis, ...] = (
    # below ~0.05 ≈ non-binder (threshold 0.01); >0.25 = strong routing-back.
    BehaviorAxis("long_range_reach", (0.05, 0.25)),
    # fixed-routing (conv/SSM) low TV; attention-class high.
    BehaviorAxis("content_dependence", (0.10, 0.30)),
    # content-gated copy is usually small; >0.10 = strong binding signature.
    BehaviorAxis("content_match_gating", (0.02, 0.10)),
)


def default_behavior_axes() -> tuple[BehaviorAxis, ...]:
    """The coarse 3-axis (27-niche) default behavior space."""

    return _DEFAULT_AXES


@dataclass(slots=True)
class MapElitesArchive:
    """MAP-Elites archive: best candidate per behavior niche.

    ``add`` inserts a candidate and returns ``True`` when it becomes (or replaces)
    the niche elite. Selection reads ``elites``; ``coverage`` /
    ``empty_niches`` drive archive-guided generation.
    """

    axes: tuple[BehaviorAxis, ...] = field(default_factory=default_behavior_axes)
    _cells: dict[tuple[int, ...], Elite] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("archive needs at least one behavior axis")
        self.axes = tuple(self.axes)

    # ── geometry ────────────────────────────────────────────────────
    @property
    def total_cells(self) -> int:
        cells = 1
        for axis in self.axes:
            cells *= axis.n_bins
        return cells

    def niche_for(self, descriptors: Mapping[str, float]) -> tuple[int, ...]:
        """Behavior niche (one bin index per axis) for a descriptor vector.

        Fails loud if an axis descriptor is missing — a candidate with no
        measured behavior must not be silently dropped into bin 0.
        """

        niche: list[int] = []
        for axis in self.axes:
            if axis.name not in descriptors:
                raise KeyError(
                    f"descriptor {axis.name!r} required by behavior axis is missing"
                )
            niche.append(axis.bin_of(descriptors[axis.name]))
        return tuple(niche)

    def niche_bounds(
        self, niche: Sequence[int]
    ) -> dict[str, tuple[float | None, float | None]]:
        """Per-axis descriptor [lo, hi) range of ``niche`` — guidance for refilling it."""

        if len(niche) != len(self.axes):
            raise ValueError(f"niche has {len(niche)} dims, expected {len(self.axes)}")
        return {
            axis.name: axis.bounds_of(index) for axis, index in zip(self.axes, niche)
        }

    # ── population ──────────────────────────────────────────────────
    def add(
        self,
        key: str,
        descriptors: Mapping[str, float],
        fitness: float,
        payload: Any = None,
    ) -> bool:
        niche = self.niche_for(descriptors)
        incumbent = self._cells.get(niche)
        # Strict improvement only: ties keep the incumbent so insertion order is
        # the deterministic tie-break (reproducible archives).
        if incumbent is not None and float(fitness) <= incumbent.fitness:
            return False
        self._cells[niche] = Elite(
            key=key,
            fitness=float(fitness),
            niche=niche,
            descriptors=dict(descriptors),
            payload=payload,
        )
        return True

    @property
    def elites(self) -> list[Elite]:
        """All niche elites, best fitness first."""

        return sorted(self._cells.values(), key=lambda e: e.fitness, reverse=True)

    @property
    def filled(self) -> int:
        return len(self._cells)

    def best(self) -> Elite | None:
        return max(self._cells.values(), key=lambda e: e.fitness, default=None)

    def coverage(self) -> float:
        """Filled niches / total niches — THE diversity metric (in [0, 1])."""

        return self.filled / self.total_cells

    def empty_niches(self, *, limit: int = 100_000) -> list[tuple[int, ...]]:
        """Unfilled niches, for archive-guided generation.

        Enumerates the full grid, so guarded by ``limit`` — intended for the
        coarse default archive, not a fine-grained one.
        """

        if self.total_cells > limit:
            raise ValueError(
                f"{self.total_cells} cells exceeds enumerate limit {limit}; "
                "use a coarser archive or raise the limit deliberately"
            )
        ranges = [range(axis.n_bins) for axis in self.axes]
        return [
            niche for niche in itertools.product(*ranges) if niche not in self._cells
        ]


def select_diverse(
    records: Iterable[Any],
    *,
    k: int,
    descriptors: Callable[[Any], Mapping[str, float]],
    fitness: Callable[[Any], float],
    key: Callable[[Any], str],
    axes: Sequence[BehaviorAxis] | None = None,
    backfill: bool = True,
) -> list[Any]:
    """Diversity-preserving top-``k``: one best-per-niche, then optional backfill.

    Replacement for global top-K selection (e.g. ``cpu_screening_cascade._select``
    or the ``component_fab`` exploration budget). Returns the original ``records``
    (the payloads), not ``Elite`` wrappers, so callers keep their own objects.

    1. Build a MAP-Elites archive → niche elites (spread across behavior space).
    2. Take elites best-fitness-first up to ``k``.
    3. If ``backfill`` and ``k`` exceeds the niche count, fill the remaining slots
       from the highest-fitness non-elite records (never silently under-fill).
    """

    if k <= 0:
        return []
    archive = MapElitesArchive(axes=tuple(axes) if axes else default_behavior_axes())
    payload_by_key: dict[str, Any] = {}
    fitness_by_key: dict[str, float] = {}
    order: list[str] = []
    for record in records:
        rid = key(record)
        payload_by_key[rid] = record
        fitness_by_key[rid] = float(fitness(record))
        order.append(rid)
        archive.add(rid, descriptors(record), fitness(record), payload=record)

    elites = archive.elites
    chosen_keys: list[str] = [e.key for e in elites[:k]]
    chosen_set = set(chosen_keys)
    if backfill and len(chosen_keys) < k:
        remainder = sorted(
            (rid for rid in dict.fromkeys(order) if rid not in chosen_set),
            key=lambda rid: fitness_by_key[rid],
            reverse=True,
        )
        for rid in remainder:
            if len(chosen_keys) >= k:
                break
            chosen_keys.append(rid)
            chosen_set.add(rid)
    return [payload_by_key[rid] for rid in chosen_keys]
