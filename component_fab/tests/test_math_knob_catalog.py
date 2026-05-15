"""Tests for autonomous math-knob composition enumeration."""

from __future__ import annotations

import pytest

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.generator.primitive_templates import SparseBandedAdapterLane
from component_fab.improver.axis_variants import DEFAULT_META_DB
from component_fab.improver.math_knob_catalog import (
    DEFAULT_MATH_KNOBS,
    enumerate_adaptive_math_knob_compositions,
    enumerate_math_knob_compositions,
    score_knob_stack,
)
from component_fab.state.ledger import Ledger, PROMOTION_PROMOTED, PROMOTION_REJECTED


def test_default_math_knob_catalog_covers_requested_families() -> None:
    families = {knob.family for knob in DEFAULT_MATH_KNOBS}
    assert "calculus" in families
    assert "linear_algebra" in families
    assert "sparse_matrix" in families
    assert "kernel_methods" in families
    assert "multiscale" in families
    assert "graph_diffusion" in families


def test_enumerate_math_knob_compositions_against_real_db() -> None:
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    specs = enumerate_math_knob_compositions(
        ["tropical_attention"], min_depth=2, max_depth=3
    )
    # Six knobs at depth 2/3 => C(6,2) + C(6,3) = 35 specs per anchor.
    assert len(specs) == 35
    assert any(
        spec.math_axes["op_math_knobs"]
        == "calculus_finite_difference+linear_algebra_low_rank+sparse_matrix_banded"
        for spec in specs
    )
    assert any(
        spec.math_axes["op_math_knobs"]
        == "kernel_random_features+multiscale_wavelet+graph_laplacian_diffusion"
        for spec in specs
    )


def test_composed_math_knob_spec_dispatches_to_stack() -> None:
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    spec = enumerate_math_knob_compositions(
        ["tropical_attention"], min_depth=3, max_depth=3
    )[0]
    module = generate_module_from_spec(spec, dim=16)
    assert isinstance(module, SparseBandedAdapterLane)


def test_score_knob_stack_rejects_exact_failed_stack(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record_grade(
        "p1",
        name="compose_anchor_linear_algebra_low_rank__kernel_random_features",
        category="lane",
        synthesis_kind="semiring_swap",
        cycle=1,
        composite_score=0.0,
        smoke_pass=False,
        learned_signal=False,
    )
    ledger.record_promotion("p1", PROMOTION_REJECTED)

    score = score_knob_stack(
        ("linear_algebra_low_rank", "kernel_random_features"), ledger
    )

    assert score.rejected is True
    assert score.score < 0.0


def test_score_knob_stack_prefers_promoted_prior(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record_grade(
        "p1",
        name="compose_anchor_multiscale_wavelet",
        category="lane",
        synthesis_kind="semiring_swap",
        cycle=1,
        composite_score=0.7,
        smoke_pass=True,
        learned_signal=False,
    )
    ledger.record_promotion("p1", PROMOTION_PROMOTED)

    score = score_knob_stack(
        ("multiscale_wavelet", "graph_laplacian_diffusion"), ledger
    )

    assert score.rejected is False
    assert score.score > 0.0
    assert score.reason == "subset prior"


def test_score_knob_stack_uses_capability_metadata(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record_grade(
        "p1",
        name="compose_anchor_graph_laplacian_diffusion",
        category="lane",
        synthesis_kind="semiring_swap",
        cycle=1,
        composite_score=0.3,
        smoke_pass=True,
        learned_signal=False,
        metadata={
            "math_knobs": ["graph_laplacian_diffusion"],
            "eliminated_by": None,
            "can_bind": True,
            "erf_density": 0.2,
            "nb_max_accuracy": 0.5,
        },
    )

    score = score_knob_stack(("graph_laplacian_diffusion",), ledger)

    assert score.rejected is False
    assert score.score > 0.5


def test_score_knob_stack_penalizes_gate_elimination_metadata(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record_grade(
        "p1",
        name="compose_anchor_kernel_random_features",
        category="lane",
        synthesis_kind="semiring_swap",
        cycle=1,
        composite_score=0.0,
        smoke_pass=False,
        learned_signal=False,
        metadata={
            "math_knobs": ["kernel_random_features"],
            "eliminated_by": "s05_causality_stability",
            "can_bind": False,
            "erf_density": 0.0,
            "nb_max_accuracy": 0.0,
        },
    )
    ledger.record_promotion("p1", PROMOTION_REJECTED)

    score = score_knob_stack(("kernel_random_features",), ledger)

    assert score.rejected is True


def test_adaptive_math_knob_compositions_prunes_rejected_stack(tmp_path) -> None:
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.record_grade(
        "p1",
        name="compose_tropical_attention_linear_algebra_low_rank__kernel_random_features",
        category="lane",
        synthesis_kind="semiring_swap",
        cycle=1,
        composite_score=0.0,
        smoke_pass=False,
        learned_signal=False,
    )
    ledger.record_promotion("p1", PROMOTION_REJECTED)

    specs = enumerate_adaptive_math_knob_compositions(
        ["tropical_attention"], ledger, min_depth=2, max_depth=2, max_specs=100
    )
    stacks = {spec.math_axes["op_math_knobs"] for spec in specs}

    assert "linear_algebra_low_rank+kernel_random_features" not in stacks
    assert "calculus_finite_difference+linear_algebra_low_rank" in stacks
