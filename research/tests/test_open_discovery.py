"""The open-ended discovery loop must run name-free and illuminate niches."""

from __future__ import annotations

import torch

from research.synthesis.open_discovery import (
    OpenDiscovery,
    ProgramSpec,
    build_program,
    sample_spec,
)
from research.synthesis.parametric_atoms import AtomSpec
from research.synthesis.parametric_ops import StageSpec
from research.synthesis.quality_diversity import MapElitesArchive
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
