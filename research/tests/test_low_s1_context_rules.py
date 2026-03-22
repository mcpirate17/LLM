from __future__ import annotations

import random

import pytest

from research.search.evolution import _mutate_graph
from research.synthesis.grammar import GrammarConfig, generate_layer_graph
from research.synthesis.graph import ComputationGraph
from research.synthesis.templates import apply_template
from research.synthesis.validator import validate_graph


pytestmark = pytest.mark.unit


def _ops_in_graph(graph: ComputationGraph) -> list[str]:
    return [node.op_name for node in graph.nodes.values() if not node.is_input]


def _build_valid_local_window_graph(model_dim: int = 64) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    input_id = graph.add_input()
    normed = graph.add_op("rmsnorm", [input_id])
    attn = graph.add_op("local_window_attn", [normed], {"window_size": 8})
    projected = graph.add_op("linear_proj", [attn], {"out_dim": model_dim})
    output_id = graph.add_op("add", [input_id, projected])
    graph.set_output(output_id)
    return graph


def _build_invalid_local_window_graph(model_dim: int = 64) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    input_id = graph.add_input()
    normed = graph.add_op("rmsnorm", [input_id])
    output_id = graph.add_op("local_window_attn", [normed], {"window_size": 8})
    graph.set_output(output_id)
    return graph


def test_local_window_attn_allowed_in_valid_residual_attention_context():
    graph = _build_valid_local_window_graph()

    result = validate_graph(graph)

    assert result.valid, result.errors
    assert "local_window_attn" in _ops_in_graph(graph)


def test_local_window_attn_rejected_in_invalid_standalone_context():
    graph = _build_invalid_local_window_graph()

    result = validate_graph(graph)

    assert not result.valid
    assert any("local_window_attn" in error for error in result.errors)


@pytest.mark.parametrize(
    ("builder", "expected_fragment"),
    [
        (
            lambda: _build_identity_standalone_graph(),
            "identity cannot be the primary learning carrier",
        ),
        (
            lambda: _build_split3_standalone_graph(),
            "split3 must rejoin through concat before output",
        ),
    ],
)
def test_structural_ops_rejected_as_standalone_learning_carriers(
    builder, expected_fragment
):
    graph = builder()

    result = validate_graph(graph)

    assert not result.valid
    assert any(expected_fragment in error for error in result.errors)


def _build_identity_standalone_graph(model_dim: int = 64) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    input_id = graph.add_input()
    output_id = graph.add_op("identity", [input_id])
    graph.set_output(output_id)
    return graph


def _build_split3_standalone_graph(model_dim: int = 96) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    input_id = graph.add_input()
    split = graph.add_op("split3", [input_id])
    projected = graph.add_op("linear_proj", [split], {"out_dim": model_dim})
    graph.set_output(projected)
    return graph


def test_fresh_generation_respects_context_policy():
    config = GrammarConfig(
        model_dim=64,
        composition_depth=1,
        template_weights={"sequential": 1_000.0},
        motif_weights={"attn_local_window": 1_000_000.0},
        residual_prob=0.0,
    )

    graph = generate_layer_graph(config, seed=7)

    assert "local_window_attn" not in _ops_in_graph(graph)


def test_mutation_generation_respects_context_policy():
    parent = _build_valid_local_window_graph()
    grammar = GrammarConfig(
        model_dim=64,
        composition_depth=1,
        template_weights={"sequential": 1_000.0},
        motif_weights={"attn_local_window": 1_000_000.0},
        residual_prob=0.0,
    )

    child = _mutate_graph(parent, grammar, random.Random(11))

    assert "local_window_attn" not in _ops_in_graph(child)


def test_residual_template_can_still_select_local_window_attn():
    graph = ComputationGraph(model_dim=64)
    input_id = graph.add_input()

    output_id = apply_template(
        graph,
        input_id,
        random.Random(5),
        template_name="residual_block",
        motif_weights={"attn_local_window": 1_000_000.0},
    )
    graph.set_output(output_id)

    result = validate_graph(graph)

    assert result.valid, result.errors
    assert "local_window_attn" in _ops_in_graph(graph)
