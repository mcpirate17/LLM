import pytest
import random
from unittest.mock import patch

from research.search.evolution import _crossover_graphs, _mutate_graph
from research.synthesis.grammar import GrammarConfig
from research.synthesis.graph import ComputationGraph

pytestmark = pytest.mark.unit


def _make_parent_graph(model_dim: int = 64) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    node_input = graph.add_input()
    node_main = graph.add_op("relu", [node_input])
    graph.set_output(node_main)
    return graph


def _make_graph_with_dead_branch(model_dim: int = 64) -> ComputationGraph:
    graph = ComputationGraph(model_dim=model_dim)
    node_input = graph.add_input()
    node_main = graph.add_op("relu", [node_input])
    graph.set_output(node_main)
    graph.add_op("gelu", [node_input])
    return graph


def test_mutate_graph_prunes_dead_branches_immediately():
    parent = _make_parent_graph()
    child_with_dead_branch = _make_graph_with_dead_branch()
    grammar = GrammarConfig(model_dim=64)
    rng = random.Random(123)

    assert child_with_dead_branch.get_dead_nodes()

    with patch("research.search.evolution.generate_layer_graph", return_value=child_with_dead_branch):
        child = _mutate_graph(parent, grammar, rng)

    assert child is child_with_dead_branch
    assert child.get_dead_nodes() == set()


def test_crossover_graph_prunes_dead_branches_immediately():
    parent_a = _make_parent_graph()
    parent_b = _make_parent_graph()
    child_with_dead_branch = _make_graph_with_dead_branch()
    grammar = GrammarConfig(model_dim=64)
    rng = random.Random(456)

    assert child_with_dead_branch.get_dead_nodes()

    with patch("research.search.evolution.generate_layer_graph", return_value=child_with_dead_branch):
        child = _crossover_graphs(parent_a, parent_b, grammar, rng)

    assert child is child_with_dead_branch
    assert child.get_dead_nodes() == set()
