"""The open-ended discovery loop must run name-free and illuminate niches."""

from __future__ import annotations

import pytest
import torch

from research.synthesis.open_discovery import (
    OpenDiscovery,
    ProgramSpec,
    _frontier_elites,
    _mutate,
    build_program,
    calibrate_proxy,
    sample_spec,
)
from research.synthesis.parametric_atoms import AtomSpec
from research.synthesis.parametric_ops import StageSpec
from research.synthesis.quality_diversity import Elite, MapElitesArchive
from research.synthesis.physics_descriptors import physics_behavior_axes


def _spec() -> ProgramSpec:
    return ProgramSpec(
        atom=AtomSpec(kinds=("norm",)), stage=StageSpec(), knob_scale=1.5
    )


def test_build_program_is_deterministic_and_finite() -> None:
    a = build_program(_spec(), dim=16, seed=3)
    b = build_program(_spec(), dim=16, seed=3)
    x = torch.randn(2, 8, 16)
    ya, yb = a(x), b(x)
    assert ya.shape == x.shape and torch.isfinite(ya).all()
    assert torch.allclose(ya, yb)  # same seed -> identical program


def test_evaluate_returns_physics_and_fitness() -> None:
    disc = OpenDiscovery(dim=16, vocab=32, n_seeds=1)
    graded = disc.evaluate(_spec())
    assert graded is not None
    phys, fitness = graded
    assert {a.name for a in physics_behavior_axes()} <= set(phys)
    assert isinstance(fitness, float)


def test_sample_spec_is_valid_with_and_without_archive() -> None:
    gen = torch.Generator().manual_seed(0)
    empty = MapElitesArchive(axes=physics_behavior_axes())
    spec = sample_spec(gen, empty)
    assert isinstance(spec, ProgramSpec)
    # build it to confirm the sampled choices are all dispatchable
    out = build_program(spec, dim=16, seed=0)(torch.randn(2, 8, 16))
    assert torch.isfinite(out).all()


def test_run_illuminates_multiple_niches() -> None:
    disc = OpenDiscovery(dim=16, vocab=32, n_seeds=1)
    result = disc.run(iters=12, seed=0)
    assert result.evaluated > 0
    assert result.inserted > 0
    assert len(result.archive.elites) >= 1
    # leaderboard is fitness-sorted and within the archive.
    board = result.leaderboard(top=5)
    assert board == sorted(board, key=lambda e: e.fitness, reverse=True)


# ── P3.3: empty-niche / frontier-biased illumination ──────────────────────────


def _elite_at(niche: tuple[int, ...], fitness: float, payload=None) -> Elite:
    return Elite(
        key=f"e{niche}",
        fitness=fitness,
        niche=niche,
        descriptors={},
        payload=payload if payload is not None else _spec(),
    )


def test_frontier_elites_flags_edge_of_explored_space() -> None:
    arch = MapElitesArchive(axes=physics_behavior_axes())
    # One lone elite -> all its Hamming-1 neighbours are empty -> it is frontier.
    lone = _elite_at((0, 0, 0, 0), 0.5)
    arch._cells[lone.niche] = lone
    front = _frontier_elites(arch)
    assert len(front) == 1 and front[0].niche == (0, 0, 0, 0)


def test_mutate_push_widens_knob_spread() -> None:
    gen = torch.Generator().manual_seed(0)
    base = _spec()
    pushed = [_mutate(base, gen, push=True).knob_scale for _ in range(20)]
    plain = [_mutate(base, gen, push=False).knob_scale for _ in range(20)]
    # push samples knob_scale from [2, 4]; plain from [0.5, 3].
    assert all(2.0 <= k <= 4.0 for k in pushed)
    assert min(plain) < 2.0  # plain can go below the push floor
    assert sum(pushed) / len(pushed) > sum(plain) / len(plain)


# ── P3.2: proxy calibration ───────────────────────────────────────────────────


def test_calibrate_proxy_monotone_passes() -> None:
    elites = [_elite_at((i, 0, 0, 0), float(i), payload=float(i)) for i in range(6)]
    report = calibrate_proxy(elites, real_scorer=lambda p: p, threshold=0.4)
    assert report.n == 6
    assert report.rho == 1.0
    assert report.ok is True


def test_calibrate_proxy_warns_when_proxy_does_not_track_real() -> None:
    elites = [_elite_at((i, 0, 0, 0), float(i), payload=float(i)) for i in range(6)]
    # real score anti-correlated with proxy fitness -> rho < threshold -> warn.
    with pytest.warns(UserWarning, match="calibration LOW"):
        report = calibrate_proxy(elites, real_scorer=lambda p: -p, threshold=0.4)
    assert report.ok is False
    assert report.rho < 0.4


# ── NM-10: novelty-aware archive (distance-from-softmax axis) ─────────────────


def test_run_novelty_aware_adds_distance_axis() -> None:
    """NM-10: a novelty-aware run bins on a geometric-novelty axis (distance from
    the softmax/attention basin) on top of the physics symmetry classes, so
    far-from-softmax niches are illuminated rather than crowded out by
    softmax-shaped mechanisms."""
    from research.synthesis.novelty_distance import NOVELTY_AXIS_NAME

    disc = OpenDiscovery(dim=16, vocab=32, n_seeds=1, novelty_aware=True)
    result = disc.run(iters=12, seed=0)
    assert result.evaluated > 0
    assert result.inserted > 0
    axis_names = [a.name for a in result.archive.axes]
    assert NOVELTY_AXIS_NAME in axis_names
    # physics axes are still present, so the novelty axis is additive
    assert "perm_equivariance" in axis_names
    # every archived elite carries a measured geometric-novelty coordinate
    assert all(NOVELTY_AXIS_NAME in e.descriptors for e in result.archive.elites)
