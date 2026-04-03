from __future__ import annotations

import random

from research.search.evolution import _mutate_graph
from research.synthesis.context_rules import find_graph_context_violations
from research.synthesis.graph import ComputationGraph
from research.synthesis.grammar import GrammarConfig, generate_layer_graph
from research.synthesis.validator import validate_graph


def _valid_local_window_graph() -> ComputationGraph:
    g = ComputationGraph(64)
    inp = g.add_input()
    ln = g.add_op("rmsnorm", [inp])
    attn = g.add_op("local_window_attn", [ln], {"window_size": 16})
    proj = g.add_op("linear_proj", [attn], {"out_dim": 64})
    out = g.add_op("add", [inp, proj])
    g.set_output(out)
    return g


def test_local_window_attn_valid_context_passes():
    result = validate_graph(_valid_local_window_graph(), max_ops=8, max_depth=6)
    assert result.valid, result.errors


def test_local_window_attn_invalid_context_fails():
    g = ComputationGraph(64)
    inp = g.add_input()
    bad = g.add_op("local_window_attn", [inp], {"window_size": 16})
    g.set_output(bad)
    violations = find_graph_context_violations(g)
    assert any("local_window_attn" in msg for msg in violations), violations
    result = validate_graph(g, max_ops=4, max_depth=4)
    assert not result.valid


def test_structural_identity_cannot_be_primary_learning_carrier():
    g = ComputationGraph(64)
    inp = g.add_input()
    ident = g.add_op("identity", [inp])
    g.set_output(ident)
    violations = find_graph_context_violations(g)
    assert "identity cannot be the primary learning carrier" in violations


def test_fresh_generation_keeps_local_window_valid_when_sampled():
    cfg = GrammarConfig.exploration(
        frozenset({"local_window_attn"}),
        model_dim=64,
        boost_factor=12.0,
    )
    generated = 0
    for seed in range(40):
        try:
            graph = generate_layer_graph(cfg, seed=seed)
        except ValueError:
            continue
        generated += 1
        ops = {node.op_name for node in graph.nodes.values() if not node.is_input}
        if "local_window_attn" not in ops:
            continue
        assert not find_graph_context_violations(graph), graph.to_dict()
    assert generated > 0


def test_mutation_generation_keeps_local_window_valid_when_sampled():
    parent = _valid_local_window_graph()
    cfg = GrammarConfig.exploration(
        frozenset({"local_window_attn"}),
        model_dim=64,
        boost_factor=12.0,
    )
    generated = 0
    for seed in range(10):
        try:
            child = _mutate_graph(parent, cfg, random.Random(seed))
        except ValueError:
            continue
        generated += 1
        ops = {node.op_name for node in child.nodes.values() if not node.is_input}
        if "local_window_attn" not in ops:
            continue
        assert not find_graph_context_violations(child), child.to_dict()
    assert generated > 0


def test_n_way_sparse_router_valid_context_passes():
    g = ComputationGraph(64)
    inp = g.add_input()
    ln = g.add_op("rmsnorm", [inp])
    router = g.add_op("n_way_sparse_router", [ln], {"n_ways": 4, "top_k": 2})
    renorm = g.add_op("rmsnorm", [router])
    out = g.add_op("add", [inp, renorm])
    g.set_output(out)
    result = validate_graph(g, max_ops=8, max_depth=6)
    assert result.valid, result.errors


def test_n_way_sparse_router_without_redensifying_successor_fails():
    g = ComputationGraph(64)
    inp = g.add_input()
    ln = g.add_op("rmsnorm", [inp])
    router = g.add_op("n_way_sparse_router", [ln], {"n_ways": 4, "top_k": 2})
    out = g.add_op("add", [inp, router])
    g.set_output(out)
    violations = find_graph_context_violations(g)
    assert any("n_way_sparse_router" in msg for msg in violations), violations
    result = validate_graph(g, max_ops=6, max_depth=5)
    assert not result.valid
