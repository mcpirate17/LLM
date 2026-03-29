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
    compute_composite_score,
)
from research.synthesis.graph import ComputationGraph


# ── Loss ratio cap tests ───────────────────────────────────────────────


@pytest.mark.unit
def test_v4_nonlearning_capped_at_10():
    """A result with loss_ratio=0.9825 cannot score above 10 (v4)."""
    score = compute_composite_score(
        screening_lr=0.9825,
        screening_nov=0.9,
        novelty_confidence=1.0,
        routing_savings=0.5,
        loss_improvement_rate=0.01,
    )
    assert score <= 10.0, f"Non-learning model scored {score}, expected <= 10"


@pytest.mark.unit
def test_v4_barely_learning_capped_at_20():
    """A result with loss_ratio=0.92 cannot score above 20 (v4)."""
    score = compute_composite_score(
        screening_lr=0.92,
        screening_nov=0.9,
        novelty_confidence=1.0,
        routing_savings=0.5,
        loss_improvement_rate=0.05,
    )
    assert score <= 20.0, f"Barely-learning model scored {score}, expected <= 20"


@pytest.mark.unit
def test_v4_legitimate_model_not_capped():
    """Var H (loss_ratio=0.54) must not be capped."""
    score = compute_composite_score(
        screening_lr=0.54,
        screening_nov=0.68,
        novelty_confidence=0.5,
        loss_improvement_rate=0.5,
    )
    assert score > 20.0, f"Legitimate model scored {score}, should be > 20"


@pytest.mark.unit
def test_nonlearning_cannot_reach_validated():
    """A result with loss_ratio=0.9825 produces a score too low for validation."""
    score_v4 = compute_composite_score(
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
    good_score = compute_composite_score(screening_lr=0.1, screening_nov=0.3)
    bad_score = compute_composite_score(
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
    result = compute_composite_score(
        screening_lr=0.96,
        screening_nov=0.9,
        decompose=True,
    )
    assert isinstance(result, dict)
    assert result["breakdown"]["insufficient_learning_cap"] == 10.0
    assert result["composite_score"] <= 10.0
