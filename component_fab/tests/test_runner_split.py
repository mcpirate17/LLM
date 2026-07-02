"""Contracts for the split autonomous runner helpers."""

from __future__ import annotations

from pathlib import Path

from component_fab.runner.grading import metadata_for_grade
from component_fab.runner.selection import select_active_specs
from component_fab.state.ledger import Ledger, PROMOTION_PROMOTED
from component_fab.tests.conftest import make_spec


def test_metadata_for_grade_persists_build_recipe_and_capability_fields() -> None:
    spec = make_spec(
        {
            "op_algebraic_space": "tropical",
            "op_math_knobs": "spectral_chebyshev+tensor_tucker",
        },
        pid="meta_contract",
    )
    meta = metadata_for_grade(
        spec,
        {
            "can_bind": True,
            "erf_density": 0.125,
            "nb_max_accuracy": 0.75,
            "range_effective_distance": 4,
            "range_ran": True,
        },
        None,
    )
    assert meta["math_knobs"] == ["spectral_chebyshev", "tensor_tucker"]
    assert meta["can_bind"] is True
    assert meta["range_effective_distance"] == 4
    assert meta["math_axes"] == spec.math_axes


def test_select_active_specs_skips_terminal_entries(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    promoted = make_spec({"op_algebraic_space": "tropical"}, pid="promoted")
    fresh = make_spec({"op_algebraic_space": "padic"}, pid="fresh")
    ledger.record_grade(
        promoted.proposal_id,
        name=promoted.name,
        category=promoted.category,
        synthesis_kind=promoted.synthesis_kind,
        cycle=1,
        composite_score=0.9,
        smoke_pass=True,
        learned_signal=True,
        metadata={"math_axes": promoted.math_axes},
    )
    ledger.record_promotion(promoted.proposal_id, PROMOTION_PROMOTED)

    active, nas, buckets, n_new, n_skipped = select_active_specs(
        [promoted, fresh],
        ledger,
        selection="legacy",
        acquisition_beta=1.0,
        use_nas_screen=False,
        use_quality_order=False,
        max_graded_per_cycle=0,
        tier2_feedback_by_id={},
    )
    assert [spec.proposal_id for spec in active] == [fresh.proposal_id]
    assert nas == {}
    assert buckets["exploit"] == 0
    assert n_new == 1
    assert n_skipped == 1
