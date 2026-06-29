"""Tests for component_fab.improver.axis_variants."""

from __future__ import annotations

import pytest

from component_fab.improver.axis_variants import (
    DATA_ROUTE_AXIS_VARIANTS,
    DEFAULT_AXIS_VARIANT_TEMPLATES,
    AnchorAxes,
    AxisVariant,
    DEFAULT_META_DB,
    anchor_axes_for_op,
    enumerate_axis_variants,
    spec_for_variant,
)
from research.synthesis.data_pipeline_grammar import data_route_from_axes


def test_data_route_variants_round_trip_to_valid_specs() -> None:
    # Each data-route variant's delta must rebuild a valid DataRouteSpec, so a
    # candidate can carry it as a genotype consumed by the LM A/B.
    assert DATA_ROUTE_AXIS_VARIANTS, "expected non-empty data-route variants"
    names = {v.delta_name for v in DATA_ROUTE_AXIS_VARIANTS}
    assert {"data_reverse", "data_doc_boundary", "data_surprisal_route"} <= names
    for variant in DATA_ROUTE_AXIS_VARIANTS:
        spec = data_route_from_axes(variant.delta)  # raises on an invalid axis value
        assert not spec.is_identity, f"{variant.delta_name} must change the data route"


def test_data_route_variants_kept_out_of_default_templates() -> None:
    # They must NOT pollute the synthetic-probe search (they are LM-only).
    default_names = {v.delta_name for v in DEFAULT_AXIS_VARIANT_TEMPLATES}
    route_names = {v.delta_name for v in DATA_ROUTE_AXIS_VARIANTS}
    assert default_names.isdisjoint(route_names)


def test_anchor_axes_returns_known_op() -> None:
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    anchor = anchor_axes_for_op("tropical_attention")
    assert anchor is not None
    assert anchor.op_name == "tropical_attention"
    assert anchor.axes.get("op_algebraic_space") == "tropical"
    assert anchor.eval_count > 0


def test_anchor_axes_missing_op_returns_none() -> None:
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    assert anchor_axes_for_op("definitely_not_a_real_op_xyz") is None


def test_spec_for_variant_changes_axes_per_delta() -> None:
    anchor = AnchorAxes(
        op_name="tropical_attention",
        axes={
            "op_algebraic_space": "tropical",
            "op_dynamical_has_state": 0,
            "op_dynamical_memory_length_class": "O(L^2)",
            "op_activation_sparsity_pattern": "dense",
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
        },
        eval_count=300,
        pass_rate=0.18,
    )
    state_variant = AxisVariant(
        delta_name="add_state",
        delta={"op_dynamical_has_state": 1, "op_dynamical_memory_length_class": "O(L)"},
        rationale="x",
    )
    spec = spec_for_variant(anchor, state_variant)
    assert spec.math_axes["op_dynamical_has_state"] == 1
    assert spec.math_axes["op_dynamical_memory_length_class"] == "O(L)"
    assert spec.math_axes["op_algebraic_space"] == "tropical"
    assert spec.anchor_witness_op == "tropical_attention"
    assert "tropical_attention" in spec.name
    assert "add_state" in spec.name
    assert spec.notes


def test_enumerate_axis_variants_full_count_against_real_db() -> None:
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    anchors = ["tropical_attention", "padic_gate"]
    specs = enumerate_axis_variants(anchors)
    assert len(specs) == len(anchors) * len(DEFAULT_AXIS_VARIANT_TEMPLATES)
    names = {s.name for s in specs}
    assert any("tropical_attention" in n and "add_state_OL" in n for n in names)
    assert any("padic_gate" in n and "fourier_basis" in n for n in names)


def test_default_axis_variant_templates_cover_core_and_math_knobs() -> None:
    kinds = {v.delta_name for v in DEFAULT_AXIS_VARIANT_TEMPLATES}
    assert "add_state_OL" in kinds
    assert "top_k_sparsity" in kinds
    assert "fourier_basis" in kinds
    assert "global_receptive" in kinds
    assert "calculus_finite_difference" in kinds
    assert "linear_algebra_low_rank" in kinds
    assert "sparse_matrix_banded" in kinds
    assert "route_site_recursion_mixer" in kinds
    assert "block_loss_monster_pair_hyper_mor" in kinds
    assert "block_loss_monster_pair_slot_dplr" in kinds
    assert "block_loss_monster_pair_native_semiring" in kinds


def test_math_knob_variant_adds_real_axes() -> None:
    anchor = AnchorAxes(
        op_name="tropical_attention",
        axes={
            "op_algebraic_space": "tropical",
            "op_dynamical_has_state": 0,
            "op_dynamical_memory_length_class": "O(L^2)",
            "op_activation_sparsity_pattern": "dense",
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
        },
        eval_count=300,
        pass_rate=0.18,
    )
    variant = AxisVariant(
        delta_name="calculus_finite_difference",
        delta={
            "op_math_family": "calculus",
            "op_calculus_operator": "causal_finite_difference_integral",
        },
        rationale="x",
    )
    spec = spec_for_variant(anchor, variant)
    assert spec.math_axes["op_math_family"] == "calculus"
    assert spec.math_axes["op_calculus_operator"] == "causal_finite_difference_integral"
    assert "calculus_finite_difference" in spec.name
