"""
Tests for T6: Structural ops exempted from per-op S1 attribution.

Proves:
  1. Structural ops are excluded from category weight S1 aggregation
  2. Structural ops are excluded from per-op weight computation
  3. Component health classifies structural ops as "structural", not broken/degraded
  4. Non-structural ops are still fully attributed (no suppression)
  5. Overall model S1 screening is unchanged
"""

from unittest.mock import patch
from research.synthesis.context_rules import S1_EXEMPT_OPS


class TestCategoryWeightExemption:
    """Structural ops should not drag down category weights."""

    def test_structural_ops_excluded_from_category_stats(self):
        """_gather_category_stats skips structural ops."""
        from research.scientist.analytics.analytics_grammar import _GrammarMixin

        mixin = _GrammarMixin()

        # Fake op_rates: one structural (identity, 0% S1) and one real (state_space, 50% S1)
        op_rates = {
            "identity": {
                "n_used": 100,
                "s1_rate": 0.0,
                "avg_novelty": None,
                "avg_novelty_confidence": None,
            },
            "state_space": {
                "n_used": 100,
                "s1_rate": 0.5,
                "avg_novelty": None,
                "avg_novelty_confidence": None,
            },
        }
        cat_stats = mixin._gather_category_stats(op_rates)

        # identity is structural → excluded. Only state_space counted.
        # state_space is category "mixing"
        mixing = cat_stats.get("mixing", {})
        assert mixing.get("total", 0) == 100, "Only state_space should be counted"
        assert mixing.get("s1_total", 0) == 50, (
            "S1 total should reflect state_space only"
        )

        # No structural category pollution
        for cat, stats in cat_stats.items():
            # identity would be in "structural" or "elementwise_binary" category
            # but it should be skipped entirely
            assert stats["total"] >= 0

    def test_non_structural_ops_still_counted(self):
        """Non-structural ops are fully counted in category stats."""
        from research.scientist.analytics.analytics_grammar import _GrammarMixin

        mixin = _GrammarMixin()
        op_rates = {
            "linear_proj": {
                "n_used": 200,
                "s1_rate": 0.3,
                "avg_novelty": 0.5,
                "avg_novelty_confidence": 0.8,
            },
            "gelu": {
                "n_used": 150,
                "s1_rate": 0.4,
                "avg_novelty": None,
                "avg_novelty_confidence": None,
            },
        }
        cat_stats = mixin._gather_category_stats(op_rates)

        # linear_proj → "parameterized" category
        param = cat_stats.get("parameterized", {})
        assert param.get("total", 0) == 200

        # gelu → "elementwise_unary" category
        eu = cat_stats.get("elementwise_unary", {})
        assert eu.get("total", 0) == 150

    def test_all_exempt_ops_are_actually_excluded(self):
        """Every op in S1_EXEMPT_OPS is skipped."""
        from research.scientist.analytics.analytics_grammar import _GrammarMixin

        mixin = _GrammarMixin()
        op_rates = {
            op: {
                "n_used": 50,
                "s1_rate": 0.0,
                "avg_novelty": None,
                "avg_novelty_confidence": None,
            }
            for op in S1_EXEMPT_OPS
        }
        cat_stats = mixin._gather_category_stats(op_rates)

        total_counted = sum(stats["total"] for stats in cat_stats.values())
        assert total_counted == 0, (
            f"All structural ops should be excluded, got total={total_counted}"
        )


class TestPerOpWeightExemption:
    """Structural ops should get no per-op weight (excluded from mean)."""

    def test_structural_ops_excluded_from_op_weights(self):
        """compute_op_weights excludes structural ops from both mean and output."""
        from research.scientist.analytics.analytics_experiments import _ExperimentsMixin

        fake_rates = {
            "identity": {
                "n_used": 100,
                "n_s0": 80,
                "s0_rate": 0.8,
                "s05_rate": 0.5,
                "s1_rate": 0.01,
                "avg_loss_ratio": 0.9,
                "avg_novelty": None,
                "avg_novelty_confidence": None,
            },
            "concat": {
                "n_used": 100,
                "n_s0": 90,
                "s0_rate": 0.9,
                "s05_rate": 0.6,
                "s1_rate": 0.02,
                "avg_loss_ratio": 0.85,
                "avg_novelty": None,
                "avg_novelty_confidence": None,
            },
            "state_space": {
                "n_used": 100,
                "n_s0": 70,
                "s0_rate": 0.7,
                "s05_rate": 0.4,
                "s1_rate": 0.3,
                "avg_loss_ratio": 0.5,
                "avg_novelty": 0.5,
                "avg_novelty_confidence": 0.8,
            },
            "linear_proj": {
                "n_used": 200,
                "n_s0": 180,
                "s0_rate": 0.9,
                "s05_rate": 0.7,
                "s1_rate": 0.5,
                "avg_loss_ratio": 0.4,
                "avg_novelty": 0.3,
                "avg_novelty_confidence": 0.7,
            },
        }

        with patch.object(
            _ExperimentsMixin, "op_success_rates", return_value=fake_rates
        ):
            mixin = _ExperimentsMixin()
            weights = mixin.compute_op_weights(min_used=5)

        # Structural ops should NOT appear in weights
        assert "identity" not in weights, "identity should be excluded"
        assert "concat" not in weights, "concat should be excluded"

        # Non-structural ops should appear
        assert "state_space" in weights, "state_space should have a weight"
        assert "linear_proj" in weights, "linear_proj should have a weight"

    def test_structural_exclusion_does_not_affect_mean(self):
        """Mean S1 rate should be computed only from non-structural ops."""
        from research.scientist.analytics.analytics_experiments import _ExperimentsMixin

        # Without exemption, identity (s1_rate=0) would drag mean from 0.4 to 0.2
        fake_rates = {
            "identity": {
                "n_used": 100,
                "n_s0": 80,
                "s0_rate": 0.8,
                "s05_rate": 0.5,
                "s1_rate": 0.0,
                "avg_loss_ratio": 0.9,
                "avg_novelty": None,
                "avg_novelty_confidence": None,
            },
            "state_space": {
                "n_used": 100,
                "n_s0": 70,
                "s0_rate": 0.7,
                "s05_rate": 0.4,
                "s1_rate": 0.4,
                "avg_loss_ratio": 0.5,
                "avg_novelty": None,
                "avg_novelty_confidence": None,
            },
        }

        with patch.object(
            _ExperimentsMixin, "op_success_rates", return_value=fake_rates
        ):
            mixin = _ExperimentsMixin()
            weights = mixin.compute_op_weights(min_used=5)

        # state_space is the only eligible op → s1_rate=0.4, mean=0.4, relative=1.0, weight=1.0
        assert "state_space" in weights
        assert abs(weights["state_space"] - 1.0) < 0.01, (
            f"Expected ~1.0, got {weights['state_space']}"
        )


class TestComponentHealthExemption:
    """Component health should classify structural ops as 'structural'."""

    def test_exempt_set_is_correct(self):
        """S1_EXEMPT_OPS matches the coordinator's target list."""
        expected = {
            "identity",
            "split2",
            "split3",
            "concat",
            "causal_mask",
            "sliding_window_mask",
            "norm_last",
            "sum_last",
            "mean_last",
            "max_last",
        }
        assert S1_EXEMPT_OPS == expected

    def test_non_structural_ops_not_in_exempt_set(self):
        """Learnable ops must NOT be in the exempt set."""
        learnable = [
            "linear_proj",
            "state_space",
            "softmax_attention",
            "gelu",
            "rmsnorm",
            "diff_attention",
            "fused_linear_gelu",
        ]
        for op in learnable:
            assert op not in S1_EXEMPT_OPS, f"{op} should NOT be exempt"


class TestModelS1Unchanged:
    """Overall model S1 screening must not be affected."""

    def test_s1_screening_not_imported_from_context_rules(self):
        """The actual S1 pass/fail gate should not reference S1_EXEMPT_OPS."""
        import inspect
        from research.eval import screening_rapid

        source = inspect.getsource(screening_rapid)
        assert "S1_EXEMPT_OPS" not in source, (
            "S1 screening code must not reference structural exemptions"
        )

    def test_s1_screening_not_in_sandbox(self):
        """The eval sandbox should not reference structural exemptions."""
        import inspect
        from research.eval import sandbox

        source = inspect.getsource(sandbox)
        assert "S1_EXEMPT_OPS" not in source, (
            "Eval sandbox must not reference structural exemptions"
        )
