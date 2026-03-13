import pytest
import os
import tempfile
from unittest.mock import MagicMock

from research.scientist.runner import ExperimentRunner, RunConfig
from research.scientist.refinement_scoring import rank_synthesis_candidates_by_stability
from research.synthesis.graph import ComputationGraph

pytestmark = pytest.mark.unit


def _simple_graph() -> ComputationGraph:
    g = ComputationGraph(model_dim=64)
    x = g.add_input()
    y = g.add_op("linear_proj", [x], config={"out_dim": 64})
    g.set_output(y)
    return g


def _complex_graph() -> ComputationGraph:
    g = ComputationGraph(model_dim=64)
    x = g.add_input()
    a = g.add_op("linear_proj", [x], config={"out_dim": 64})
    b = g.add_op("conv1d_seq", [a], config={})
    c = g.add_op("gelu", [b], config={})
    d = g.add_op("linear_proj", [c], config={"out_dim": 64})
    g.set_output(d)
    return g


def _risky_graph() -> ComputationGraph:
    g = ComputationGraph(model_dim=64)
    x = g.add_input()
    a = g.add_op("moe_2expert", [x], config={})
    b = g.add_op("gelu", [a], config={})
    c = g.add_op("gated_linear", [b], config={"out_dim": 64})
    g.set_output(c)
    return g


def _stabilized_risky_graph() -> ComputationGraph:
    g = ComputationGraph(model_dim=64)
    x = g.add_input()
    n = g.add_op("layernorm", [x], config={})
    a = g.add_op("moe_2expert", [n], config={})
    b = g.add_op("gelu", [a], config={})
    c = g.add_op("gated_linear", [b], config={"out_dim": 64})
    y = g.add_op("add", [x, c], config={})
    g.set_output(y)
    return g


def test_refinement_compression_prefers_simpler_graph():
    runner = ExperimentRunner(os.path.join(tempfile.mkdtemp(), "refine_score.db"))
    simple = _simple_graph()
    complex_g = _complex_graph()
    op_success = {"linear_proj": 0.5, "conv1d_seq": 0.5, "gelu": 0.5}

    s_simple = runner._score_refinement_candidate(simple, op_success, "compression")
    s_complex = runner._score_refinement_candidate(complex_g, op_success, "compression")
    assert s_simple > s_complex


def test_refinement_quality_uses_learned_op_success():
    runner = ExperimentRunner(os.path.join(tempfile.mkdtemp(), "refine_quality.db"))
    simple = _simple_graph()
    complex_g = _complex_graph()
    op_success = {"linear_proj": 0.9, "conv1d_seq": 0.1, "gelu": 0.1}

    s_simple = runner._score_refinement_candidate(simple, op_success, "quality")
    s_complex = runner._score_refinement_candidate(complex_g, op_success, "quality")
    assert s_simple > s_complex


def test_refinement_breakdown_matches_score():
    runner = ExperimentRunner(os.path.join(tempfile.mkdtemp(), "refine_breakdown.db"))
    simple = _simple_graph()
    op_success = {"linear_proj": 0.7}

    score, breakdown = runner._score_refinement_candidate(
        simple,
        op_success,
        "balanced",
        include_breakdown=True,
    )
    weighted = breakdown.get("weighted_terms", {})
    assert weighted
    assert abs(score - sum(weighted.values())) < 1e-12
    assert breakdown.get("mode") == "balanced"


def test_refinement_balanced_penalizes_oscillation_risk():
    runner = ExperimentRunner(os.path.join(tempfile.mkdtemp(), "refine_stability.db"))
    risky = _risky_graph()
    stable = _stabilized_risky_graph()
    op_success = {"moe_2expert": 0.6, "gelu": 0.6, "gated_linear": 0.6, "layernorm": 0.6, "add": 0.6}

    risky_score, risky_breakdown = runner._score_refinement_candidate(
        risky, op_success, "balanced", include_breakdown=True
    )
    stable_score, stable_breakdown = runner._score_refinement_candidate(
        stable, op_success, "balanced", include_breakdown=True
    )

    assert risky_breakdown["components"]["oscillation_risk"] > stable_breakdown["components"]["oscillation_risk"]
    assert risky_breakdown["weighted_terms"]["oscillation_penalty"] < 0.0
    assert stable_score > risky_score


def test_synthesis_stability_rerank_prefers_stabilized_graph():
    ranked = rank_synthesis_candidates_by_stability([
        _risky_graph(),
        _stabilized_risky_graph(),
    ])
    assert ranked[0].has_residual_path()


def test_fingerprint_refinement_default_hypothesis_is_structured():
    runner = ExperimentRunner(os.path.join(tempfile.mkdtemp(), "refine_hyp.db"))
    runner._recent_synthesis_health = MagicMock(return_value={"s1_rate": 0.0})
    runner.start_experiment = MagicMock(return_value="exp-refine")

    exp_id = runner.start_fingerprint_refinement(
        result_ids=["abc123"],
        config=RunConfig(),
        hypothesis=None,
    )
    assert exp_id == "exp-refine"
    call = runner.start_experiment.call_args
    hyp = call.kwargs.get("hypothesis", "")
    assert "source_selection_rule=" in hyp
    assert "mutation_mechanism=" in hyp
    assert "intent=" in hyp and "weights=" in hyp and "score=" in hyp
    assert "success_criteria=" in hyp
    assert "fallback_plan=" in hyp
