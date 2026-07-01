from __future__ import annotations

from pathlib import Path

import torch

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.proposer.cross_product import enumerate_cross_product_specs
from component_fab.proposer.enumeration import enumerate_cycle_specs
from component_fab.state.ledger import Ledger


def _cell_tuple(spec) -> tuple[str, str, str, str, str]:
    axes = spec.math_axes
    return (
        str(axes["op_physics_address_family"]),
        str(axes["op_physics_score_norm_family"]),
        str(axes["op_physics_aggregate_family"]),
        str(axes["op_algebraic_space"]),
        str(axes["op_spectral_preferred_basis"]),
    )


def test_cross_product_specs_span_unseen_cells_and_skip_softmax_twins(
    tmp_path: Path,
) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")

    specs = enumerate_cross_product_specs(ledger, cycle=1, max_specs=6)

    assert len(specs) == 6
    cells = {_cell_tuple(spec) for spec in specs}
    assert len(cells) == 6
    assert len(cells) >= 3
    assert any(cell[2] == "semiring" for cell in cells)
    assert any(cell[3] != "euclidean" for cell in cells)
    assert all(spec.math_axes["op_physics_source"] == "cross_product" for spec in specs)
    assert all(
        not (
            spec.math_axes["op_physics_aggregate_family"] == "mean"
            and spec.math_axes["op_physics_score_norm_family"]
            in {"softmax", "sharpen"}
        )
        for spec in specs
    )


def test_cross_product_specs_skip_cells_seen_in_ledger(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    first = enumerate_cross_product_specs(ledger, cycle=0, max_specs=1)[0]

    ledger.record_grade(
        first.proposal_id,
        name=first.name,
        category=first.category,
        synthesis_kind=first.synthesis_kind,
        cycle=0,
        composite_score=0.1,
        smoke_pass=True,
        learned_signal=False,
        metadata={"math_axes": first.math_axes},
    )
    next_specs = enumerate_cross_product_specs(ledger, cycle=0, max_specs=4)

    assert first.math_axes["op_cross_product_cell"] not in {
        spec.math_axes["op_cross_product_cell"] for spec in next_specs
    }


def test_cross_product_spec_is_buildable(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    spec = enumerate_cross_product_specs(ledger, cycle=2, max_specs=1)[0]

    module = generate_module_from_spec(spec, dim=16)
    y = module(torch.randn(2, 7, 16))

    assert y.shape == (2, 7, 16)
    assert torch.isfinite(y).all()


def test_cycle_specs_can_include_cross_product_track(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")

    specs = enumerate_cycle_specs(
        ledger,
        [],
        cycle=3,
        include_static_variants=False,
        include_frontier=False,
        include_nas=False,
        include_training_regimes=False,
        include_name_free_physics=False,
        max_cross_pairs=0,
        max_knob_specs=0,
        max_dynamic_specs=0,
        max_cross_product_specs=3,
    )

    assert len(specs) == 3
    assert all(
        spec.math_axes.get("op_physics_source") == "cross_product" for spec in specs
    )
