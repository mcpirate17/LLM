from __future__ import annotations

import random

import pytest

from research.search._mutation import (
    _native_local_mutation_trials,
    _python_local_mutation_trials,
)
from research.search.native_graph_mutation import load_native_graph_mutation_lib
from research.synthesis.graph import ComputationGraph


def _build_mutation_graph() -> ComputationGraph:
    graph = ComputationGraph(64)
    inp = graph.add_input()
    lhs = graph.add_op("rmsnorm", [inp])
    rhs = graph.add_op("linear_proj", [lhs], {"out_dim": 64})
    out = graph.add_op("add", [inp, rhs])
    graph.set_output(out)
    return graph


def test_native_mutation_planner_matches_python_candidate_space():
    if load_native_graph_mutation_lib() is None:
        pytest.skip("native graph mutation runtime not available")

    graph = _build_mutation_graph()
    native_trials = _native_local_mutation_trials(graph, random.Random(7))
    python_trials = _python_local_mutation_trials(graph, random.Random(7))

    assert native_trials is not None
    assert set(native_trials) == set(python_trials)
    assert native_trials
