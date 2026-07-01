"""Tests for the geometric-novelty MAP-Elites axis (NM-10).

The softmax-basin-distance axis must place softmax-shaped averagers in the basin
bin (distance ~0) and genuinely novel geometry in a distant bin, so the archive
spreads across the novelty dimension and empty far-from-softmax niches surface
for archive-guided exploration.
"""

from __future__ import annotations

import torch

from research.synthesis.quality_diversity import (
    MapElitesArchive,
    default_behavior_axes,
)

from component_fab.proposer.novelty_archive import (
    SOFTMAX_BASIN_DISTANCE,
    distance_from_twin_score,
    measure_softmax_basin_distance,
    softmax_basin_distance_axis,
    with_novelty_axis,
)

_DIM = 24


def _softmax_qk_mixer(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    gen = torch.Generator().manual_seed(3)
    wq = torch.randn(d, d, generator=gen) * d**-0.5
    wk = torch.randn(d, d, generator=gen) * d**-0.5
    scores = (x @ wq) @ (x @ wk).transpose(1, 2) / (d**0.5)
    return torch.softmax(scores, dim=-1) @ x


def _signed_token_mixer(x: torch.Tensor) -> torch.Tensor:
    return x - torch.roll(x, 1, dims=1)


def test_distance_from_twin_score_bounds() -> None:
    assert distance_from_twin_score(1.0) == 0.0
    assert distance_from_twin_score(0.0) == 1.0
    assert distance_from_twin_score(0.3) == 0.7
    # Clamped.
    assert distance_from_twin_score(1.5) == 0.0
    assert distance_from_twin_score(-0.2) == 1.0


def test_softmax_mixer_sits_in_basin() -> None:
    dist = measure_softmax_basin_distance(_softmax_qk_mixer, dim=_DIM, n_seeds=3)
    assert dist < 0.15  # basin bin


def test_novel_op_is_distant() -> None:
    dist = measure_softmax_basin_distance(_signed_token_mixer, dim=_DIM, n_seeds=3)
    assert dist > 0.15  # out of the basin bin


def test_with_novelty_axis_appends_once() -> None:
    base = default_behavior_axes()
    augmented = with_novelty_axis(base)
    assert len(augmented) == len(base) + 1
    assert augmented[-1].name == SOFTMAX_BASIN_DISTANCE
    # Idempotent.
    assert with_novelty_axis(augmented) is augmented


def test_axis_bins_basin_vs_novel_separately() -> None:
    axis = softmax_basin_distance_axis()
    assert axis.bin_of(0.05) == 0  # basin
    assert axis.bin_of(0.3) == 1  # transitional
    assert axis.bin_of(0.8) == 2  # novel


def test_archive_separates_basin_and_novel_niches() -> None:
    axes = with_novelty_axis(default_behavior_axes())
    archive = MapElitesArchive(axes=axes)
    base = {name: 0.2 for name in (a.name for a in default_behavior_axes())}
    twin = {**base, SOFTMAX_BASIN_DISTANCE: 0.02}
    novel = {**base, SOFTMAX_BASIN_DISTANCE: 0.9}
    archive.add("twin", twin, fitness=0.5)
    archive.add("novel", novel, fitness=0.4)
    # Distinct niches on the novelty axis -> both survive despite lower fitness.
    assert archive.filled == 2
    assert archive.niche_for(twin) != archive.niche_for(novel)
