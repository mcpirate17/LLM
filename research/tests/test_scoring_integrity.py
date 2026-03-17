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
    compute_composite_v5,
    compute_composite_v6,
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
def test_v5_nonlearning_capped_at_10():
    """A result with loss_ratio=0.9825 cannot score above 10 (v5)."""
    score = compute_composite_v5(
        screening_lr=0.9825,
        screening_nov=0.9,
        novelty_confidence=1.0,
        loss_improvement_rate=0.01,
        param_count=10_000_000,
    )
    assert score <= 10.0, f"Non-learning model scored {score}, expected <= 10"


@pytest.mark.unit
def test_v5_barely_learning_capped_at_20():
    """A result with loss_ratio=0.92 cannot score above 20 (v5)."""
    score = compute_composite_v5(
        screening_lr=0.92,
        screening_nov=0.9,
        novelty_confidence=1.0,
        loss_improvement_rate=0.05,
        param_count=10_000_000,
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
def test_v5_legitimate_model_not_capped():
    """Var H (loss_ratio=0.54) must not be capped (v5)."""
    score = compute_composite_v5(
        screening_lr=0.54,
        screening_nov=0.68,
        novelty_confidence=0.5,
        loss_improvement_rate=0.5,
        param_count=30_000_000,
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
    score_v5 = compute_composite_v5(
        screening_lr=0.9825,
        screening_nov=0.99,
        novelty_confidence=1.0,
        loss_improvement_rate=0.01,
        param_count=5_000_000,
    )
    # Even with maxed-out novelty, routing, etc., score must stay below 20
    assert score_v4 <= 10.0, f"v4 non-learning {score_v4} > 10"
    assert score_v5 <= 10.0, f"v5 non-learning {score_v5} > 10"


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


# ── v6 step gate tests ───────────────────────────────────────────────
# Bug: c05a (500-step screening run) scored 206.20 because
# validation_loss was measured on training corpus val split, not WikiText.
# Step gates prevent screening runs from competing with validated results.


@pytest.mark.unit
def test_v6_screening_run_cannot_outscore_validated():
    """A 500-step screening run cannot score above 40 (hard cap)."""
    score = compute_composite_v6(
        final_loss=4.07,
        loss_ratio=0.39,
        screening_lr=0.39,
        val_lr=0.04,
        n_train_steps=500,
        validation_passed=True,
        loss_improvement_rate=0.5,
    )
    assert score <= 40.0, f"500-step screening run scored {score} > 40 cap"


@pytest.mark.unit
def test_v6_validation_tier_requires_real_training():
    """Validation tier with <4000 steps is capped at investigation ceiling."""
    score = compute_composite_v6(
        final_loss=4.07,
        loss_ratio=0.39,
        val_lr=0.04,
        n_train_steps=2500,
        validation_passed=True,
        loss_improvement_rate=0.5,
    )
    assert score <= 85.0, f"Short validation run scored {score} > 85 ceiling"


@pytest.mark.unit
def test_v6_legitimate_10k_model_not_capped():
    """A 10K-step validated model is NOT penalized by step gates."""
    score = compute_composite_v6(
        final_loss=4.07,
        loss_ratio=0.39,
        val_lr=0.04,
        n_train_steps=10000,
        validation_passed=True,
        loss_improvement_rate=0.5,
    )
    # Should score well above the screening cap
    assert score > 40.0, f"10K-step model scored {score}, should be > 40"


@pytest.mark.unit
def test_v6_reference_bypasses_step_gate():
    """Reference architectures bypass step gates regardless of n_train_steps."""
    ref_score = compute_composite_v6(
        final_loss=2.63,
        loss_ratio=0.265,
        screening_lr=0.265,
        n_train_steps=500,
        is_reference=True,
    )
    # References must not be capped by step gate
    assert ref_score > 40.0, f"Reference scored {ref_score}, step gate should not apply"


@pytest.mark.unit
def test_v6_step_gate_decompose():
    """Decompose mode shows step gate was applied."""
    result = compute_composite_v6(
        final_loss=4.07,
        screening_lr=0.39,
        n_train_steps=500,
        decompose=True,
    )
    assert isinstance(result, dict)
    bd = result["breakdown"]
    assert bd.get("step_gate") is True
    assert bd["step_fraction"] == 500 / 2000
    assert result["composite_score"] <= 40.0


@pytest.mark.unit
def test_v6_zero_steps_no_crash():
    """n_train_steps=0 or None should not trigger step gate (no data)."""
    score_zero = compute_composite_v6(
        final_loss=4.07,
        screening_lr=0.39,
        n_train_steps=0,
    )
    score_none = compute_composite_v6(
        final_loss=4.07,
        screening_lr=0.39,
        n_train_steps=None,
    )
    # Both should produce the same score — no step gate applied
    assert abs(score_zero - score_none) < 0.01, (
        f"n_steps=0 ({score_zero}) != n_steps=None ({score_none})"
    )
