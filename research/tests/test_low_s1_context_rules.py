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
            "split3 must rejoin through concat or add before output",
        ),
        (
            lambda: _build_split2_standalone_graph(),
            "split2 must rejoin through concat or add before output",
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


def _build_split2_standalone_graph(model_dim: int = 64) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    input_id = graph.add_input()
    split = graph.add_op("split2", [input_id])
    projected = graph.add_op("linear_proj", [split], {"out_dim": model_dim})
    graph.set_output(projected)
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
        routing_mandatory=False,  # test targets context policy, not routing
    )

    # Context rules may reject the graph entirely (ValueError) or exclude
    # the forbidden op. Either outcome proves the policy is enforced.
    try:
        graph = generate_layer_graph(config, seed=7)
        assert "local_window_attn" not in _ops_in_graph(graph)
    except ValueError:
        pass  # Context rule rejection = policy enforced


def test_mutation_generation_respects_context_policy():
    parent = _build_valid_local_window_graph()
    grammar = GrammarConfig(
        model_dim=64,
        composition_depth=1,
        template_weights={"sequential": 1_000.0},
        motif_weights={"attn_local_window": 1_000_000.0},
        residual_prob=0.0,
        routing_mandatory=False,  # test targets context policy, not routing
    )

    child = _mutate_graph(parent, grammar, random.Random(11))

    assert "local_window_attn" not in _ops_in_graph(child)


def test_split2_with_explicit_parts_returns_different_halves():
    """split2 with part=0 and part=1 must return different tensor slices."""
    import torch
    from research.synthesis.compiler_ops_math import _op_split2

    x = torch.arange(8, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 8]
    part0 = _op_split2(None, (x,), {"part": 0})
    part1 = _op_split2(None, (x,), {"part": 1})

    assert part0.shape[-1] == 4
    assert part1.shape[-1] == 4
    assert not torch.equal(part0, part1), "part=0 and part=1 must differ"
    assert torch.equal(torch.cat([part0, part1], dim=-1), x)


def test_split3_with_explicit_parts_returns_different_thirds():
    """split3 with part=0/1/2 must return different tensor slices."""
    import torch
    from research.synthesis.compiler_ops_math import _op_split3

    x = torch.arange(9, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 9]
    part0 = _op_split3(None, (x,), {"part": 0})
    part1 = _op_split3(None, (x,), {"part": 1})
    part2 = _op_split3(None, (x,), {"part": 2})

    assert part0.shape[-1] == 3
    assert part1.shape[-1] == 3
    assert part2.shape[-1] == 3
    assert not torch.equal(part0, part1)
    assert not torch.equal(part1, part2)
    assert torch.equal(torch.cat([part0, part1, part2], dim=-1), x)


def test_split2_template_produces_two_parts_with_rejoin():
    """tpl_parallel_split must create two split2 nodes that rejoin via concat."""
    # Try multiple seeds — motif selection is random and some combos hit
    # context rule violations unrelated to split logic.
    for seed in range(50):
        graph = ComputationGraph(model_dim=64)
        input_id = graph.add_input()

        output_id = apply_template(
            graph,
            input_id,
            random.Random(seed),
            template_name="parallel_split",
        )
        graph.set_output(output_id)

        ops = _ops_in_graph(graph)
        if ops.count("split2") < 2:
            continue  # fallback to residual_block (dim too small, etc.)

        result = validate_graph(graph)
        if not result.valid:
            continue  # context rule violation from motif, not split

        # Found a valid graph with split2 — verify structure
        assert ops.count("split2") == 2
        assert "concat" in ops, "split2 branches must rejoin via concat"

        split_nodes = [n for n in graph.nodes.values() if n.op_name == "split2"]
        parts = sorted(n.config.get("part", 0) for n in split_nodes)
        assert parts == [0, 1], f"expected parts [0, 1], got {parts}"
        return

    pytest.fail("no seed produced a valid parallel_split graph in 50 attempts")


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
