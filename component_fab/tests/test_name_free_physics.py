from __future__ import annotations

from pathlib import Path

import torch

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.proposer.enumeration import enumerate_cycle_specs
from component_fab.proposer.name_free import enumerate_name_free_physics_experiments
from component_fab.proposer.quality import BUCKET_EXPLORATION, score_quality
from component_fab.state.ledger import Ledger


def test_name_free_physics_specs_are_measured_and_buildable(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")

    specs = enumerate_name_free_physics_experiments(
        ledger,
        cycle=1,
        dim=16,
        max_specs=3,
        max_candidates_per_experiment=4,
    )

    assert specs
    spec = specs[0]
    assert spec.math_axes["op_search_track"] == "physics_atom"
    assert spec.math_axes["op_physics_source"] == "name_free_experiment"
    assert spec.math_axes["op_physics_experiment"]
    assert spec.math_axes["op_physics_niche"]
    assert "op_physics_descriptor_perm_equivariance" in spec.math_axes
    assert "Name-free physics experiment" in spec.rationale

    module = generate_module_from_spec(spec, dim=16)
    y = module(torch.randn(2, 7, 16))
    assert y.shape == (2, 7, 16)
    assert torch.isfinite(y).all()


def test_cycle_includes_name_free_physics_specs(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")

    specs = enumerate_cycle_specs(
        ledger,
        [],
        cycle=1,
        dim=16,
        include_static_variants=False,
        include_frontier=False,
        include_nas=False,
        include_training_regimes=False,
        max_cross_pairs=0,
        max_knob_specs=0,
        max_dynamic_specs=0,
        max_name_free_specs=2,
    )

    assert len(specs) == 2
    assert all(
        spec.math_axes.get("op_physics_source") == "name_free_experiment"
        for spec in specs
    )


def test_name_free_physics_gets_exploration_priority(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    spec = enumerate_name_free_physics_experiments(
        ledger,
        cycle=1,
        dim=16,
        max_specs=1,
        max_candidates_per_experiment=4,
    )[0]

    score = score_quality(spec, entry=None)

    assert score.bucket == BUCKET_EXPLORATION
    assert any("name-free physics experiment" in r for r in score.evidence_reasons)
    assert score.quality_score > 0.0
