"""Scoring integrity tests — prevent score inflation, fingerprint collisions,
and non-learning models reaching high tiers.

Tests the invariants introduced in the fingerprint collision fix:
  1. Loss_ratio > 0.9 → composite score capped
  2. Different op configs → different fingerprints
  3. Composite score reflects THIS result's loss, never inherited
"""

from __future__ import annotations

import pytest

from research.scientist.leaderboard_scoring import (
    compute_composite,
)
from research.synthesis.graph import ComputationGraph


# ── Loss ratio cap tests ───────────────────────────────────────────────


@pytest.mark.unit
def test_v4_nonlearning_capped_at_10():
    """A result with loss_ratio=0.9825 cannot score above 10 (v4)."""
    score = compute_composite(
        screening_lr=0.9825,
        screening_nov=0.9,
        novelty_confidence=1.0,
        routing_savings=0.5,
        loss_improvement_rate=0.01,
    )
    assert score <= 10.0, f"Non-learning model scored {score}, expected <= 10"


@pytest.mark.unit
def test_barely_learning_scores_low():
    """A result with loss_ratio=0.92 scores far below a well-performing model."""
    barely = compute_composite(
        screening_lr=0.92,
        screening_nov=0.9,
        novelty_confidence=1.0,
        routing_savings=0.5,
    )
    good = compute_composite(
        screening_lr=0.3,
        screening_nov=0.9,
        novelty_confidence=1.0,
        routing_savings=0.5,
    )
    assert barely < good, (
        f"Barely-learning model ({barely}) should score below good model ({good})"
    )


@pytest.mark.unit
def test_legitimate_model_not_capped():
    """A model with good loss ratio should not be capped by the insufficient-learning gate."""
    result = compute_composite(
        screening_lr=0.3,
        screening_nov=0.68,
        novelty_confidence=0.5,
        decompose=True,
    )
    score = result["composite_score"]
    # Must not be capped by the insufficient-learning path. Active v14 scoring
    # can still apply tokenizer/provenance penalties to sparse test inputs.
    assert "insufficient_learning_cap" not in result["breakdown"]
    # Should also beat a non-learning model
    bad = compute_composite(
        screening_lr=0.96, screening_nov=0.68, novelty_confidence=0.5
    )
    assert score > bad, f"Legitimate model ({score}) should beat non-learning ({bad})"


@pytest.mark.unit
def test_ar_gate_is_bounded_gate_not_full_rank_signal():
    """Saturated AR Gate alone should not max the AR capability bucket."""
    result = compute_composite(
        screening_lr=0.3,
        ar_gate_score=1.0,
        decompose=True,
    )
    bd = result["breakdown"]

    assert bd["cap_ar_nano_gate_fraction"] == pytest.approx(0.45)
    assert bd["cap_ar_signal_fraction"] == pytest.approx(0.45)
    assert bd["cap_ar"] < 45.0


@pytest.mark.unit
def test_ar_gate_below_gate_stays_low_but_monotonic():
    low = compute_composite(
        screening_lr=0.3,
        ar_gate_score=0.10,
        decompose=True,
    )["breakdown"]
    gated = compute_composite(
        screening_lr=0.3,
        ar_gate_score=0.50,
        decompose=True,
    )["breakdown"]

    assert 0.0 < low["cap_ar"] < gated["cap_ar"]
    assert gated["cap_ar_nano_gate_fraction"] >= 0.35


@pytest.mark.unit
def test_ar_validation_rank_signal_beats_saturated_nano_tie():
    weak_ar_validation = compute_composite(
        screening_lr=0.3,
        ar_gate_score=1.0,
        ar_validation_rank_score=2.0,
        decompose=True,
    )["breakdown"]
    strong_ar_validation = compute_composite(
        screening_lr=0.3,
        ar_gate_score=1.0,
        ar_validation_rank_score=6.0,
        decompose=True,
    )["breakdown"]

    assert weak_ar_validation["cap_ar_nano_gate_fraction"] == pytest.approx(0.45)
    assert strong_ar_validation["cap_ar_rank_signal"] == pytest.approx(0.60)
    assert strong_ar_validation["cap_ar"] > weak_ar_validation["cap_ar"]


@pytest.mark.unit
def test_nonlearning_cannot_reach_validated():
    """A result with loss_ratio=0.9825 produces a score too low for validation."""
    score_v4 = compute_composite(
        screening_lr=0.9825,
        screening_nov=0.99,
        novelty_confidence=1.0,
        routing_savings=0.9,
        compression_ratio=0.5,
        loss_improvement_rate=0.01,
        scaling_param_efficiency=5.0,
    )
    # Even with maxed-out novelty, routing, etc., score must stay below 20
    assert score_v4 <= 10.0, f"v4 non-learning {score_v4} > 10"


# ── Fingerprint integrity tests ────────────────────────────────────────


@pytest.mark.unit
def test_different_configs_different_fingerprints():
    """Two graphs with same ops but different configs produce different fingerprints."""
    g1 = ComputationGraph(model_dim=64)
    inp1 = g1.add_input()
    g1.add_op("token_type_classifier", [inp1], config={"n_classes": 4})
    g1.set_output(g1._next_id - 1)

    g2 = ComputationGraph(model_dim=64)
    inp2 = g2.add_input()
    g2.add_op("token_type_classifier", [inp2], config={"n_classes": 8})
    g2.set_output(g2._next_id - 1)

    assert g1.fingerprint() != g2.fingerprint(), (
        f"Different configs produced same fingerprint: {g1.fingerprint()}"
    )


@pytest.mark.unit
def test_different_model_dim_different_fingerprints():
    """Same ops but different model_dim produce different fingerprints."""
    g1 = ComputationGraph(model_dim=64)
    inp1 = g1.add_input()
    g1.add_op("gelu", [inp1])
    g1.set_output(g1._next_id - 1)

    g2 = ComputationGraph(model_dim=128)
    inp2 = g2.add_input()
    g2.add_op("gelu", [inp2])
    g2.set_output(g2._next_id - 1)

    assert g1.fingerprint() != g2.fingerprint(), (
        f"Different model_dim produced same fingerprint: {g1.fingerprint()}"
    )


@pytest.mark.unit
def test_same_graph_same_fingerprint():
    """Identical graphs produce identical fingerprints."""

    def _make_graph() -> ComputationGraph:
        g = ComputationGraph(model_dim=64)
        inp = g.add_input()
        h = g.add_op("linear_proj", [inp], config={"out_dim": 64})
        g.add_op("gelu", [h])
        g.set_output(g._next_id - 1)
        return g

    assert _make_graph().fingerprint() == _make_graph().fingerprint()


@pytest.mark.unit
def test_composite_score_uses_own_loss_ratio():
    """Composite score is always computed from THIS result's loss_ratio.

    A high-novelty, high-routing model with loss_ratio=0.98 cannot score
    higher than a boring model with loss_ratio=0.1.
    """
    good_score = compute_composite(screening_lr=0.1, screening_nov=0.3)
    bad_score = compute_composite(
        screening_lr=0.98,
        screening_nov=0.99,
        novelty_confidence=1.0,
        routing_savings=0.9,
    )
    assert good_score > bad_score, (
        f"Bad model (lr=0.98) score {bad_score} >= good model (lr=0.1) {good_score}"
    )


@pytest.mark.unit
def test_v4_decompose_shows_cap():
    """Decompose mode shows the insufficient_learning_cap when applied."""
    result = compute_composite(
        screening_lr=0.96,
        screening_nov=0.9,
        decompose=True,
    )
    assert isinstance(result, dict)
    assert result["breakdown"]["insufficient_learning_cap"] == 10.0
    assert result["composite_score"] <= 10.0
