from __future__ import annotations

from research.synthesis.graph import ComputationGraph


def _make_graph_with_dead_branch() -> ComputationGraph:
    graph = ComputationGraph(8)
    inp = graph.add_input()
    live = graph.add_op("relu", [inp])
    dead = graph.add_op("gelu", [inp])
    graph.set_output(live)
    assert dead in graph.nodes
    return graph


def test_lower_to_ir_strips_dead_branch():
    graph = _make_graph_with_dead_branch()

    ir = graph.lower_to_ir()

    assert ir.node_ids.tolist() == [0, 1]
    assert ir.output_node_idx == 1
    assert ir.input_indices.tolist() == [[-1, -1], [0, -1]]


def test_analysis_ir_keeps_dead_branch():
    graph = _make_graph_with_dead_branch()

    ir = graph._analysis_ir()

    assert ir.node_ids.tolist() == [0, 1, 2]
    assert ir.output_node_idx == 1
    assert ir.input_indices.tolist() == [[-1, -1], [0, -1], [0, -1]]
