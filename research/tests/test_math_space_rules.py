"""Tests for MATH_SPACE_RULES entries added by Volta audit (2026-03-21).

Covers: spiking ops (T2), tropical_matmul/tropical_center (T3),
        hyperbolic ops (T5).

Each test builds a minimal ComputationGraph and verifies that
grammar._validate_graph() rejects invalid placements and accepts
valid ones.
"""

from __future__ import annotations

import pytest

from research.synthesis.graph import ComputationGraph
from research.synthesis.grammar import GrammarConfig, _validate_graph


def _make_graph(dim: int = 64) -> tuple[ComputationGraph, int]:
    """Create a graph with an input node, return (graph, input_id)."""
    g = ComputationGraph(dim)
    inp = g.add_input()
    return g, inp


def _finalize(g: ComputationGraph, last_id: int) -> ComputationGraph:
    """Set output and return the graph."""
    g.set_output(last_id)
    return g


# ── Spiking: sparse_threshold must_follow {lif_neuron, spike_rate_code} ──


def test_sparse_threshold_after_lif_neuron_accepted():
    """Full valid spiking chain: lif_neuron → sparse_threshold → stdp_attention → proj."""
    g, inp = _make_graph()
    ln = g.add_op("rmsnorm", [inp])
    lif = g.add_op("lif_neuron", [ln])
    st = g.add_op("sparse_threshold", [lif])
    stdp = g.add_op("stdp_attention", [st])
    proj = g.add_op("linear_proj", [stdp])
    res = g.add_op("add", [inp, proj])
    _finalize(g, res)
    _validate_graph(g, GrammarConfig())  # should not raise


def test_sparse_threshold_after_linear_proj_rejected():
    g, inp = _make_graph()
    ln = g.add_op("rmsnorm", [inp])
    proj = g.add_op("linear_proj", [ln])
    st = g.add_op("sparse_threshold", [proj])
    proj2 = g.add_op("linear_proj", [st])
    res = g.add_op("add", [inp, proj2])
    _finalize(g, res)
    # Rejected by context_rules.py (requires stdp_attention successor) or
    # MATH_SPACE_RULES (must_follow spiking predecessor) — either fires
    with pytest.raises(ValueError, match="sparse_threshold"):
        _validate_graph(g, GrammarConfig())


# ── Spiking: stdp_attention must_follow {sparse_threshold, spike_rate_code, lif_neuron} ──


def test_stdp_attention_after_sparse_threshold_accepted():
    g, inp = _make_graph()
    ln = g.add_op("rmsnorm", [inp])
    lif = g.add_op("lif_neuron", [ln])
    st = g.add_op("sparse_threshold", [lif])
    stdp = g.add_op("stdp_attention", [st])
    proj = g.add_op("linear_proj", [stdp])
    res = g.add_op("add", [inp, proj])
    _finalize(g, res)
    _validate_graph(g, GrammarConfig())  # should not raise


def test_stdp_attention_after_gelu_rejected():
    g, inp = _make_graph()
    ln = g.add_op("rmsnorm", [inp])
    act = g.add_op("gelu", [ln])
    stdp = g.add_op("stdp_attention", [act])
    proj = g.add_op("linear_proj", [stdp])
    res = g.add_op("add", [inp, proj])
    _finalize(g, res)
    # Rejected by context_rules.py or MATH_SPACE_RULES
    with pytest.raises(ValueError, match="stdp_attention"):
        _validate_graph(g, GrammarConfig())


# ── Spiking: lif_neuron must_follow_with {spike_rate_code, sparse_threshold, linear_proj} ──


def test_lif_neuron_followed_by_sparse_threshold_accepted():
    g, inp = _make_graph()
    ln = g.add_op("rmsnorm", [inp])
    lif = g.add_op("lif_neuron", [ln])
    st = g.add_op("sparse_threshold", [lif])
    stdp = g.add_op("stdp_attention", [st])
    proj = g.add_op("linear_proj", [stdp])
    res = g.add_op("add", [inp, proj])
    _finalize(g, res)
    _validate_graph(g, GrammarConfig())  # should not raise


def test_lif_neuron_followed_by_spike_rate_code_accepted():
    g, inp = _make_graph()
    ln = g.add_op("rmsnorm", [inp])
    lif = g.add_op("lif_neuron", [ln])
    src = g.add_op("spike_rate_code", [lif])
    proj = g.add_op("linear_proj", [src])
    res = g.add_op("add", [inp, proj])
    _finalize(g, res)
    _validate_graph(g, GrammarConfig())  # should not raise


def test_lif_neuron_followed_only_by_gelu_rejected():
    g, inp = _make_graph()
    ln = g.add_op("rmsnorm", [inp])
    lif = g.add_op("lif_neuron", [ln])
    act = g.add_op("gelu", [lif])
    proj = g.add_op("linear_proj", [act])
    res = g.add_op("add", [inp, proj])
    _finalize(g, res)
    # Rejected by either MATH_SPACE_RULES (must_follow_with) or
    # context_rules.py (requires spiking successor context)
    with pytest.raises(ValueError, match="lif_neuron"):
        _validate_graph(g, GrammarConfig())


# ── Tropical: tropical_matmul must_precede {rmsnorm, layernorm} ──


def test_tropical_matmul_after_rmsnorm_accepted():
    """tropical_matmul is binary — must_precede checks direct parents."""
    g, inp = _make_graph()
    ln_a = g.add_op("rmsnorm", [inp])
    ln_b = g.add_op("layernorm", [inp])
    tm = g.add_op("tropical_matmul", [ln_a, ln_b])
    proj = g.add_op("linear_proj", [tm])
    res = g.add_op("add", [inp, proj])
    _finalize(g, res)
    _validate_graph(g, GrammarConfig())  # should not raise


def test_tropical_matmul_after_gelu_rejected():
    """tropical_matmul needs rmsnorm/layernorm predecessor, not gelu."""
    g, inp = _make_graph()
    act_a = g.add_op("gelu", [inp])
    act_b = g.add_op("gelu", [inp])
    tm = g.add_op("tropical_matmul", [act_a, act_b])
    proj = g.add_op("linear_proj", [tm])
    res = g.add_op("add", [inp, proj])
    _finalize(g, res)
    # Rejected by context_rules.py or MATH_SPACE_RULES
    with pytest.raises(ValueError, match="tropical_matmul"):
        _validate_graph(g, GrammarConfig())


# ── Tropical: tropical_matmul must_follow_with {linear_proj, linear_proj_down} ──


def test_tropical_matmul_without_proj_successor_rejected():
    """tropical_matmul must feed into a projection."""
    g, inp = _make_graph()
    ln_a = g.add_op("rmsnorm", [inp])
    ln_b = g.add_op("layernorm", [inp])
    tm = g.add_op("tropical_matmul", [ln_a, ln_b])
    # No linear_proj successor — goes directly to add
    res = g.add_op("add", [inp, tm])
    _finalize(g, res)
    # Rejected by MATH_SPACE_RULES (must_follow_with) or context_rules.py
    with pytest.raises(ValueError, match="tropical_matmul"):
        _validate_graph(g, GrammarConfig())


# ── Tropical: tropical_center must_follow {tropical_attention, tropical_gate} ──


def test_tropical_center_after_tropical_attention_accepted():
    g, inp = _make_graph()
    ln = g.add_op("rmsnorm", [inp])
    ta = g.add_op("tropical_attention", [ln])
    tc = g.add_op("tropical_center", [ta])
    proj = g.add_op("linear_proj", [tc])
    res = g.add_op("add", [inp, proj])
    _finalize(g, res)
    _validate_graph(g, GrammarConfig())  # should not raise


def test_tropical_center_after_linear_proj_rejected():
    g, inp = _make_graph()
    ln = g.add_op("rmsnorm", [inp])
    proj1 = g.add_op("linear_proj", [ln])
    tc = g.add_op("tropical_center", [proj1])
    proj2 = g.add_op("linear_proj", [tc])
    res = g.add_op("add", [inp, proj2])
    _finalize(g, res)
    with pytest.raises(ValueError, match="tropical_center"):
        _validate_graph(g, GrammarConfig())


# ── Hyperbolic: hyp_linear must_follow {exp_map} ──


def test_hyp_linear_after_exp_map_accepted():
    g, inp = _make_graph()
    ln = g.add_op("rmsnorm", [inp])
    em = g.add_op("exp_map", [ln])
    hl = g.add_op("hyp_linear", [em])
    ht = g.add_op("hyp_tangent_nonlinear", [hl])
    lm = g.add_op("log_map", [ht])
    proj = g.add_op("linear_proj", [lm])
    res = g.add_op("add", [inp, proj])
    _finalize(g, res)
    _validate_graph(g, GrammarConfig())  # should not raise


def test_hyp_linear_after_rmsnorm_rejected():
    """hyp_linear after rmsnorm is rejected — either by algebraic space
    conflict (euclidean→poincare) or by MATH_SPACE_RULES (must_follow exp_map)."""
    g, inp = _make_graph()
    ln = g.add_op("rmsnorm", [inp])
    hl = g.add_op("hyp_linear", [ln])
    proj = g.add_op("linear_proj", [hl])
    res = g.add_op("add", [inp, proj])
    _finalize(g, res)
    with pytest.raises(ValueError, match="hyp_linear|Space conflict"):
        _validate_graph(g, GrammarConfig())


# ── Hyperbolic: hyp_tangent_nonlinear must_follow {hyp_linear} ──


def test_hyp_tangent_nonlinear_after_gelu_rejected():
    """hyp_tangent_nonlinear after gelu is rejected — by algebraic space
    conflict or MATH_SPACE_RULES (must_follow hyp_linear)."""
    g, inp = _make_graph()
    ln = g.add_op("rmsnorm", [inp])
    act = g.add_op("gelu", [ln])
    ht = g.add_op("hyp_tangent_nonlinear", [act])
    proj = g.add_op("linear_proj", [ht])
    res = g.add_op("add", [inp, proj])
    _finalize(g, res)
    with pytest.raises(ValueError, match="hyp_tangent_nonlinear|Space conflict"):
        _validate_graph(g, GrammarConfig())


# ── Hyperbolic: hyp_tangent_nonlinear must_follow_with {log_map, linear_proj} ──


def test_hyp_tangent_nonlinear_without_log_map_rejected():
    """hyp_tangent_nonlinear must be followed by log_map or linear_proj."""
    g, inp = _make_graph()
    ln = g.add_op("rmsnorm", [inp])
    em = g.add_op("exp_map", [ln])
    hl = g.add_op("hyp_linear", [em])
    ht = g.add_op("hyp_tangent_nonlinear", [hl])
    # No log_map or linear_proj successor — goes directly to add
    res = g.add_op("add", [inp, ht])
    _finalize(g, res)
    with pytest.raises(ValueError, match="hyp_tangent_nonlinear"):
        _validate_graph(g, GrammarConfig())
