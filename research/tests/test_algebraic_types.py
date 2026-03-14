from __future__ import annotations

import random

import pytest

from research.mathspaces.registry import register_all_mathspaces
from research.synthesis.graph import ComputationGraph, ShapeInfo
from research.synthesis.grammar import _check_shape_compat
from research.synthesis.motifs import MOTIFS_BY_CLASS, Motif, MotifStep
from research.synthesis.op_roles import OpRole
from research.synthesis.primitives import (
    AlgebraicType,
    PRIMITIVE_REGISTRY,
    algebraic_types_compatible,
)
from research.synthesis.templates import _instantiate_motif

pytestmark = pytest.mark.unit
register_all_mathspaces()


def test_bridge_operators_have_expected_algebraic_types():
    assert PRIMITIVE_REGISTRY["exp_map"].algebraic_type == AlgebraicType("poincare", "real", "unit_ball")
    assert PRIMITIVE_REGISTRY["log_map"].algebraic_type == AlgebraicType("euclidean", "unit_ball", "real")


def test_algebraic_type_compatibility_enforces_bridges():
    real = AlgebraicType("euclidean", "real", "real")
    unit_ball = AlgebraicType("poincare", "unit_ball", "unit_ball")
    tropical = PRIMITIVE_REGISTRY["tropical_center"].algebraic_type
    assert algebraic_types_compatible(real, PRIMITIVE_REGISTRY["exp_map"].algebraic_type)
    assert algebraic_types_compatible(unit_ball, PRIMITIVE_REGISTRY["log_map"].algebraic_type)
    assert not algebraic_types_compatible(unit_ball, tropical)


def test_shape_compat_respects_algebraic_space_context():
    input_shape = [ShapeInfo(dim=32)]
    assert _check_shape_compat(PRIMITIVE_REGISTRY["exp_map"], input_shape, 32, current_space="euclidean")
    assert not _check_shape_compat(PRIMITIVE_REGISTRY["tropical_center"], input_shape, 32, current_space="poincare")


def test_space_entry_ops_accept_real_context():
    input_shape = [ShapeInfo(dim=32)]
    for op_name in (
        "tropical_attention",
        "tropical_gate",
        "tropical_center",
        "clifford_attention",
        "grade_mix",
        "padic_expand",
    ):
        assert _check_shape_compat(PRIMITIVE_REGISTRY[op_name], input_shape, 32, current_space="euclidean")


def test_instantiate_motif_rejects_incompatible_cross_space_sequence():
    graph = ComputationGraph(32)
    input_id = graph.add_input()
    exp_id = graph.add_op("exp_map", [input_id])
    motif = Motif(
        name="bad_cross_space",
        motif_class="test",
        steps=(MotifStep("tropical_center", OpRole.MIX),),
    )
    result = _instantiate_motif(graph, exp_id, motif, random.Random(0))
    assert result == exp_id


def test_typed_space_motifs_are_compatible_from_euclidean_input():
    graph = ComputationGraph(32)
    input_id = graph.add_input()
    motif_names = {
        motif.name
        for motifs in MOTIFS_BY_CLASS.values()
        for motif in motifs
        if motif.name in {
            "hyperbolic_residual_bridge",
            "tropical_attention_gate",
            "clifford_attention_mix",
            "padic_hierarchy_block",
        }
    }
    assert motif_names == {
        "hyperbolic_residual_bridge",
        "tropical_attention_gate",
        "clifford_attention_mix",
        "padic_hierarchy_block",
    }
    for motif_name in motif_names:
        motif = next(m for motifs in MOTIFS_BY_CLASS.values() for m in motifs if m.name == motif_name)
        out_id = _instantiate_motif(graph, input_id, motif, random.Random(0))
        assert out_id != input_id, motif_name
