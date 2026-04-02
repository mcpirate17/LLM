"""Tests for v8 composite scoring — understanding-rebalanced scorer.

Validates:
- All new components degrade to 0 when input is None
- Understanding gate blocks pure-perplexity models
- v7 backward compatibility via dispatcher
- Point budget totals match expected maxima
- Perplexity reduction is effective
"""

import pytest

from research.scientist.leaderboard_scoring import (
    SCORING_VERSION,
    compute_composite,
    compute_composite_v7,
    compute_composite_v8,
)
from research.scientist.thresholds import (
    HELLASWAG_RANDOM_CHANCE_GATE,
    UNDERSTANDING_MIN_BINDING,
    UNDERSTANDING_MIN_DIAGNOSTIC,
)


@pytest.mark.unit
class TestV8ScoringBasics:
    """Core v8 scoring function tests."""

    def test_no_inputs_returns_zero(self):
        score = compute_composite_v8()
        assert isinstance(score, (int, float))
        assert score == 0.0

    def test_decompose_returns_dict(self):
        result = compute_composite_v8(decompose=True)
        assert isinstance(result, dict)
        assert "composite_score" in result
        assert "breakdown" in result

    def test_insufficient_learning_hard_gate(self):
        score = compute_composite_v8(screening_lr=0.98)
        assert score == 10.0

    def test_investigation_failed_zeros_perf(self):
        result = compute_composite_v8(
            ppl_screening=8.0,
            tier="investigation_failed",
            decompose=True,
        )
        bd = result["breakdown"]
        assert bd.get("perf_short", 0) == 0.0
        assert bd.get("perf_medium", 0) == 0.0
        assert bd.get("perf_long", 0) == 0.0


@pytest.mark.unit
class TestV8NewComponents:
    """Test the 5 new understanding components."""

    def test_tinystories_none_returns_zero(self):
        result = compute_composite_v8(decompose=True, tier="validation")
        assert result["breakdown"].get("tinystories", 0) == 0.0

    def test_tinystories_scored_at_validation(self):
        result = compute_composite_v8(
            tinystories_score=0.50,
            tier="validation",
            decompose=True,
        )
        pts = result["breakdown"].get("tinystories", 0)
        assert pts > 0, "TinyStories should score > 0 at validation"
        assert pts <= 30.0, "TinyStories max is 30pts"

    def test_tinystories_not_scored_at_screening(self):
        result = compute_composite_v8(
            tinystories_score=0.50,
            tier="screening",
            decompose=True,
        )
        assert result["breakdown"].get("tinystories", 0) == 0.0

    def test_cross_task_none_returns_zero(self):
        result = compute_composite_v8(decompose=True, tier="investigation")
        assert result["breakdown"].get("cross_task", 0) == 0.0

    def test_cross_task_scored_at_investigation(self):
        result = compute_composite_v8(
            cross_task_score=0.70,
            ppl_investigation=9.0,
            tier="investigation",
            decompose=True,
        )
        pts = result["breakdown"].get("cross_task", 0)
        assert pts > 0, "Cross-task should score > 0 at investigation"
        assert pts <= 30.0

    def test_diagnostic_none_returns_zero(self):
        result = compute_composite_v8(decompose=True, tier="validation")
        assert result["breakdown"].get("diagnostic", 0) == 0.0

    def test_diagnostic_scored_at_validation(self):
        result = compute_composite_v8(
            diagnostic_score=0.40,
            tier="validation",
            decompose=True,
        )
        pts = result["breakdown"].get("diagnostic", 0)
        assert pts > 0
        assert pts <= 45.0

    def test_hellaswag_below_noise_floor_zero(self):
        result = compute_composite_v8(
            hellaswag_acc_investigation=0.25,
            ppl_investigation=9.0,
            tier="investigation",
            decompose=True,
        )
        assert result["breakdown"].get("hellaswag", 0) == 0.0

    def test_hellaswag_above_noise_scored(self):
        result = compute_composite_v8(
            hellaswag_acc_investigation=0.32,
            ppl_investigation=9.0,
            tier="investigation",
            decompose=True,
        )
        pts = result["breakdown"].get("hellaswag", 0)
        assert pts > 0
        assert pts <= 30.0

    def test_hierarchy_none_returns_zero(self):
        result = compute_composite_v8(decompose=True, tier="investigation")
        assert result["breakdown"].get("hierarchy", 0) == 0.0

    def test_hierarchy_scored_at_investigation(self):
        result = compute_composite_v8(
            hierarchy_fitness=0.55,
            ppl_investigation=9.0,
            tier="investigation",
            decompose=True,
        )
        pts = result["breakdown"].get("hierarchy", 0)
        assert pts > 0
        assert pts <= 15.0


@pytest.mark.unit
class TestV8PerplexityReduction:
    """Verify perplexity weights are reduced from v7."""

    def _score_with_good_ppl(self, fn):
        return fn(
            ppl_screening=8.0,
            ppl_investigation=7.0,
            ppl_validation=5.0,
            param_count=5_000_000,
            ppl_at_100=15.0,
            ppl_at_500=10.0,
            ppl_at_1000=8.0,
            tier="validation",
            decompose=True,
        )

    def test_v8_perf_short_reduced(self):
        v7 = self._score_with_good_ppl(compute_composite_v7)
        v8 = self._score_with_good_ppl(compute_composite_v8)
        assert v8["breakdown"]["perf_short"] < v7["breakdown"]["perf_short"]

    def test_v8_perf_medium_reduced(self):
        v7 = self._score_with_good_ppl(compute_composite_v7)
        v8 = self._score_with_good_ppl(compute_composite_v8)
        assert v8["breakdown"]["perf_medium"] < v7["breakdown"]["perf_medium"]

    def test_v8_perf_long_reduced(self):
        v7 = self._score_with_good_ppl(compute_composite_v7)
        v8 = self._score_with_good_ppl(compute_composite_v8)
        assert v8["breakdown"]["perf_long"] < v7["breakdown"]["perf_long"]

    def test_v8_binding_reduced(self):
        v7 = compute_composite_v7(
            ar_auc=0.10, induction_auc=0.08, binding_auc=0.12, decompose=True
        )
        v8 = compute_composite_v8(
            ar_auc=0.10, induction_auc=0.08, binding_auc=0.12, decompose=True
        )
        assert v8["breakdown"]["binding"] < v7["breakdown"]["binding"]


@pytest.mark.unit
class TestV8PointBudget:
    """Verify maximum point allocations match spec."""

    def test_max_perplexity_budget(self):
        """v8 perplexity max = 35 + 50 + 65 + 30 + 20 + 10 = 210."""
        result = compute_composite_v8(
            ppl_screening=1.0,  # impossibly good → near max
            ppl_investigation=1.0,
            ppl_validation=1.0,
            param_count=100,  # tiny → huge param_eff
            ppl_at_100=100.0,
            ppl_at_500=10.0,
            ppl_at_1000=1.0,
            tier="validation",
            decompose=True,
        )
        bd = result["breakdown"]
        ppl_total = (
            bd.get("perf_short", 0)
            + bd.get("perf_medium", 0)
            + bd.get("perf_long", 0)
            + bd.get("param_efficiency", 0)
            + bd.get("learning_efficiency", 0)
            + bd.get("early_convergence", 0)
        )
        # Should be close to 210 (S-curve saturates near but not at max)
        assert ppl_total <= 210.0, f"Perplexity budget exceeded: {ppl_total}"

    def test_new_understanding_max(self):
        """New understanding components max = 30 + 30 + 45 + 30 + 15 = 150."""
        result = compute_composite_v8(
            tinystories_score=0.99,
            cross_task_score=0.99,
            diagnostic_score=0.99,
            hellaswag_acc_investigation=0.99,
            hierarchy_fitness=0.99,
            ppl_investigation=9.0,
            tier="validation",
            decompose=True,
        )
        bd = result["breakdown"]
        new_total = (
            bd.get("tinystories", 0)
            + bd.get("cross_task", 0)
            + bd.get("diagnostic", 0)
            + bd.get("hellaswag", 0)
            + bd.get("hierarchy", 0)
        )
        assert new_total <= 150.0, f"New understanding budget exceeded: {new_total}"
        assert new_total > 100.0, (
            f"With near-perfect inputs, should be well above 100: {new_total}"
        )


@pytest.mark.unit
class TestV8Dispatcher:
    """Test the version dispatcher routes correctly."""

    def test_dispatcher_returns_float_by_default(self):
        score = compute_composite()
        assert isinstance(score, (int, float))

    def test_dispatcher_uses_v8(self):
        """When SCORING_VERSION is v8, dispatcher should use v8."""
        assert SCORING_VERSION == "v8"
        # v8 with good understanding metrics should score differently than v7
        kw = dict(
            ppl_screening=8.0,
            tier="validation",
            tinystories_score=0.50,
            diagnostic_score=0.40,
        )
        v8_direct = compute_composite_v8(**kw)
        dispatched = compute_composite(**kw)
        assert dispatched == v8_direct


@pytest.mark.unit
class TestV8UnderstandingGateThresholds:
    """Test the threshold constants used by the understanding gate."""

    def test_thresholds_exist(self):
        assert UNDERSTANDING_MIN_DIAGNOSTIC == 0.15
        assert UNDERSTANDING_MIN_BINDING == 0.05
        assert HELLASWAG_RANDOM_CHANCE_GATE == 0.28

    def test_pure_perplexity_model_scores_lower(self):
        """A model with only perplexity should score much less in v8 than v7."""
        kw = dict(
            ppl_screening=8.0,
            ppl_investigation=7.5,
            ppl_validation=5.5,
            param_count=5_000_000,
            tier="validation",
        )
        v7 = compute_composite_v7(**kw)
        v8 = compute_composite_v8(**kw)
        # v8 should be meaningfully lower due to reduced perplexity weights
        assert v8 < v7, (
            f"v8 ({v8:.1f}) should be < v7 ({v7:.1f}) for perplexity-only model"
        )

    def test_understanding_rich_model_benefits(self):
        """A model with understanding metrics should benefit more in v8."""
        kw = dict(
            ppl_screening=9.0,
            ppl_investigation=8.0,
            tier="investigation",
            ar_auc=0.10,
            induction_auc=0.08,
            binding_auc=0.12,
            hellaswag_acc_investigation=0.30,
            cross_task_score=0.55,
            hierarchy_fitness=0.45,
        )
        v7 = compute_composite_v7(**kw)
        v8 = compute_composite_v8(**kw)
        # v8 should be higher because understanding metrics now contribute points
        assert v8 > v7, (
            f"v8 ({v8:.1f}) should be > v7 ({v7:.1f}) for understanding-rich model"
        )
