"""Tests for autonomous math-knob composition enumeration."""

from __future__ import annotations

from math import comb

import pytest

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.generator.primitive_templates import (
    LambdaFunctionalAdapterLane,
    SparseBandedAdapterLane,
)
from component_fab.math_knobs import (
    AUTO_DEEPENING_MATH_KNOBS,
    MathKnob,
    math_knobs_from_axes,
)
from component_fab.improver.axis_variants import DEFAULT_META_DB, AnchorAxes
from component_fab.improver.math_knob_catalog import (
    DEFAULT_MATH_KNOBS,
    auto_deepen_math_knobs,
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
    assert "lambda_functional" in families
    assert "tropical" in families
    assert "padic" in families
    assert "clifford" in families
    assert "hyperbolic" in families


def test_math_knobs_from_axes_parses_explicit_stack() -> None:
    assert math_knobs_from_axes({"op_math_knobs": "a + b+c"}) == ("a", "b", "c")
    assert math_knobs_from_axes({"op_math_knobs": ("a", "b")}) == ("a", "b")


def test_math_knobs_from_axes_uses_family_fallbacks() -> None:
    assert math_knobs_from_axes({"op_math_family": "calculus"}) == (
        "calculus_finite_difference",
    )
    assert math_knobs_from_axes({"op_math_family": "spectral_graph"}) == (
        "spectral_chebyshev",
    )
    assert math_knobs_from_axes({"op_math_family": "lambda_functional"}) == (
        "lambda_functional_blend",
    )
    assert math_knobs_from_axes({"op_math_family": "tropical"}) == (
        "tropical_knob",
    )
    assert math_knobs_from_axes({"op_math_family": "padic"}) == ("padic_knob",)
    assert math_knobs_from_axes({"op_math_family": "clifford"}) == (
        "clifford_knob",
    )
    assert math_knobs_from_axes({"op_math_family": "hyperbolic"}) == (
        "hyperbolic_knob",
    )


def test_auto_deepen_math_knobs_adds_bounded_natural_siblings() -> None:
    expanded = auto_deepen_math_knobs(DEFAULT_MATH_KNOBS, max_new=4)
    ids = tuple(knob.knob_id for knob in expanded)

    assert ids[: len(DEFAULT_MATH_KNOBS)] == tuple(
        knob.knob_id for knob in DEFAULT_MATH_KNOBS
    )
    assert ids[len(DEFAULT_MATH_KNOBS) :] == tuple(
        knob.knob_id for knob in AUTO_DEEPENING_MATH_KNOBS[:4]
    )
    assert len(set(ids)) == len(ids)


def test_auto_deepen_math_knobs_skips_families_with_multiple_variants() -> None:
    base = next(
        knob for knob in DEFAULT_MATH_KNOBS if knob.knob_id == "calculus_finite_difference"
    )
    manual_sibling = MathKnob(
        knob_id="calculus_manual_sibling",
        family="calculus",
        axes={"op_calculus_operator": "manual_sibling"},
        cost_class="low",
        rationale="manual calculus sibling",
    )

    expanded = auto_deepen_math_knobs((base, manual_sibling), max_new=12)

    assert tuple(knob.knob_id for knob in expanded) == (
        "calculus_finite_difference",
        "calculus_manual_sibling",
    )


def test_enumerate_math_knob_compositions_against_real_db() -> None:
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    specs = enumerate_math_knob_compositions(
        ["tropical_attention"], min_depth=2, max_depth=3
    )
    assert len(specs) == comb(len(DEFAULT_MATH_KNOBS), 2) + comb(
        len(DEFAULT_MATH_KNOBS), 3
    )
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
    exotic = next(
        spec
        for spec in specs
        if spec.math_axes["op_math_knobs"] == "tropical_knob+hyperbolic_knob"
    )
    assert exotic.math_axes["op_tropical_adapter"] == "maxplus_read"
    assert exotic.math_axes["op_hyperbolic_adapter"] == "poincare_projection"
    assert exotic.math_axes["op_algebraic_space"] == "tropical"


def test_composed_math_knob_spec_dispatches_to_stack() -> None:
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    spec = enumerate_math_knob_compositions(
        ["tropical_attention"], min_depth=3, max_depth=3
    )[0]
    module = generate_module_from_spec(spec, dim=16)
    assert isinstance(module, SparseBandedAdapterLane)


def test_lambda_math_knob_spec_dispatches_to_stack() -> None:
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    spec = enumerate_math_knob_compositions(
        ["tropical_attention"],
        knobs=tuple(
            knob for knob in DEFAULT_MATH_KNOBS if knob.knob_id == "lambda_functional_blend"
        ),
        min_depth=1,
        max_depth=1,
    )[0]
    module = generate_module_from_spec(spec, dim=16)
    assert isinstance(module, LambdaFunctionalAdapterLane)


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
        ["tropical_attention"],
        ledger,
        min_depth=2,
        max_depth=2,
        max_specs=100,
        include_auto_deepening=False,
    )
    stacks = {spec.math_axes["op_math_knobs"] for spec in specs}

    assert "linear_algebra_low_rank+kernel_random_features" not in stacks
    assert "calculus_finite_difference+linear_algebra_low_rank" in stacks


def test_adaptive_math_knob_compositions_emits_auto_deepened_siblings(
    monkeypatch, tmp_path
) -> None:
    def fake_anchor_axes_for_op(name, db_path=DEFAULT_META_DB):
        return AnchorAxes(
            op_name=name,
            axes={
                "op_algebraic_space": "tropical",
                "op_dynamical_has_state": 0,
            },
            eval_count=3,
            pass_rate=0.5,
        )

    monkeypatch.setattr(
        "component_fab.improver.math_knob_catalog.anchor_axes_for_op",
        fake_anchor_axes_for_op,
    )
    base = next(
        knob for knob in DEFAULT_MATH_KNOBS if knob.knob_id == "calculus_finite_difference"
    )

    specs = enumerate_adaptive_math_knob_compositions(
        ["anchor_op"],
        Ledger(tmp_path / "ledger.jsonl"),
        knobs=(base,),
        min_depth=1,
        max_depth=1,
        max_specs=8,
        axis_lift=None,
    )
    stacks = {spec.math_axes["op_math_knobs"] for spec in specs}

    assert "calculus_finite_difference" in stacks
    assert "calculus_causal_gradient" in stacks
    assert "calculus_laplacian" in stacks
    assert "calculus_lie_derivative" in stacks
