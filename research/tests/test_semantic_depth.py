from __future__ import annotations

from research.synthesis.graph import ComputationGraph
from research.synthesis.validator import (
    compute_effective_depth,
    validate_graph,
    validate_ir,
)


def _split_scaffold_graph() -> ComputationGraph:
    g = ComputationGraph(64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    left = g.add_op("split2", [norm], {"part": 0})
    right = g.add_op("split2", [norm], {"part": 1})
    merged = g.add_op("concat", [left, right])
    proj = g.add_op("linear_proj", [merged], {"out_dim": 64})
    out = g.add_op("add", [inp, proj])
    g.set_output(out)
    return g


def test_effective_depth_downweights_structural_scaffolding():
    g = _split_scaffold_graph()
    assert g.depth() == 5
    assert abs(compute_effective_depth(g) - 1.05) < 1e-9


def test_validate_graph_uses_effective_depth_budget():
    g = _split_scaffold_graph()
    result = validate_graph(g, max_ops=8, max_depth=2)
    assert result.valid, result.errors
    assert result.depth == 5
    assert result.effective_depth < 2.0


def test_exp_mul_chain_is_context_valid_again():
    g = ComputationGraph(64)
    inp = g.add_input()
    norm = g.add_op("rmsnorm", [inp])
    exp = g.add_op("exp", [norm])
    scaled = g.add_op("mul", [exp, norm])
    proj = g.add_op("linear_proj", [scaled], {"out_dim": 64})
    out = g.add_op("add", [inp, proj])
    g.set_output(out)

    result = validate_graph(g, max_ops=8, max_depth=3)
    assert result.valid, result.errors


def test_mixture_of_recursions_is_not_penalized_above_full_cost():
    g = ComputationGraph(64)
    inp = g.add_input()
    mor = g.add_op("mixture_of_recursions", [inp], {"max_depth": 4})
    proj = g.add_op("linear_proj", [mor], {"out_dim": 64})
    out = g.add_op("add", [inp, proj])
    g.set_output(out)

    result = validate_graph(g, max_ops=6, max_depth=2)
    assert result.valid, result.errors
    assert result.depth == 3
    assert 1.6 <= result.effective_depth <= 1.8


def test_validate_ir_uses_effective_depth_budget():
    g = _split_scaffold_graph()
    ir = g.lower_to_ir()
    result = validate_ir(ir, max_ops=8, max_depth=2)
    assert result.valid, result.errors
    assert result.depth == 5
    assert abs(result.effective_depth - compute_effective_depth(ir)) < 1e-9
