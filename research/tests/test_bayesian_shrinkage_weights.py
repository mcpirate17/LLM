"""Tests for the Beta-Binomial shrinkage layer in analytics weight math.

The handoff (2026-05-11) identified that ``compute_template_weights`` /
``compute_op_weights`` excluded items with ``n < min_used`` from the output
dict entirely, leaving newly-added substrate absent from the downstream
multiplier chain. These tests pin the shrinkage behaviour that replaces
the hard exclusion.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from research.scientist.analytics._exp_weights import (
    _BAYES_PRIOR_STRENGTH,
    _WeightsMixin,
    _amplified_weights_from_counts,
    _bayesian_posterior_rate,
    _fit_prior_mean,
)


class _NotebookStub:
    def __init__(self, path: Path):
        self.db_path = path


class _AnalyticsStub(_WeightsMixin):
    __slots__ = ("nb",)

    def __init__(self, path: Path):
        self.nb = _NotebookStub(path)


def test_posterior_rate_with_zero_obs_returns_prior() -> None:
    rate = _bayesian_posterior_rate(s1_count=0, n_used=0, prior_mean=0.25)
    assert rate == pytest.approx(0.25)


def test_posterior_rate_converges_to_observed_with_large_n() -> None:
    rate = _bayesian_posterior_rate(s1_count=900, n_used=1000, prior_mean=0.1)
    # 1000 obs swamps a 6-pseudocount prior centred at 0.1.
    assert rate == pytest.approx(0.9, abs=0.01)


def test_posterior_rate_pulls_low_n_toward_prior() -> None:
    # n=1, s=0 — should NOT collapse to 0; should sit near prior.
    rate = _bayesian_posterior_rate(s1_count=0, n_used=1, prior_mean=0.3)
    # Beta(1.8, 4.2) with one failure observed → posterior mean = 1.8 / 7.
    assert rate == pytest.approx(1.8 / 7.0, abs=1e-6)
    assert rate > 0.2  # not collapsed to 0


def test_posterior_rate_clamps_pathological_prior() -> None:
    # Out-of-range prior_mean must not break the math.
    rate = _bayesian_posterior_rate(s1_count=1, n_used=2, prior_mean=1.5)
    assert 0.0 <= rate <= 1.0


def test_fit_prior_mean_is_median() -> None:
    assert _fit_prior_mean([0.1, 0.2, 0.9]) == pytest.approx(0.2)
    assert _fit_prior_mean([0.1, 0.2, 0.3, 0.9]) == pytest.approx(0.25)
    assert _fit_prior_mean([]) == 0.0


def test_amplified_weights_emit_low_n_items() -> None:
    """New substrate (n=1) appears in output instead of being excluded."""
    counts = {"established": 30, "fresh": 1}
    s1 = {"established": 15, "fresh": 0}
    weights = _amplified_weights_from_counts(counts=counts, s1_counts=s1, min_used=3)
    assert set(weights) == {"established", "fresh"}
    # fresh has 0 observed wins but should still be in the dict, near 1.0
    # (heavy confidence-shrink toward neutral at n=1).
    assert 0.5 < weights["fresh"] < 1.5
    # established should beat fresh on contrast.
    assert weights["established"] > weights["fresh"]


def test_amplified_weights_returns_empty_when_no_signal() -> None:
    """All-zero rates → return {} to preserve the legacy short-circuit."""
    counts = {"a": 10, "b": 10}
    s1 = {"a": 0, "b": 0}
    assert _amplified_weights_from_counts(counts=counts, s1_counts=s1, min_used=3) == {}


def test_amplified_weights_clamps_bounds() -> None:
    """Extreme contrast still respects the [lo, hi] clamp."""
    counts = {"hero": 100, "dud": 100}
    s1 = {"hero": 99, "dud": 1}
    weights = _amplified_weights_from_counts(
        counts=counts,
        s1_counts=s1,
        min_used=3,
        amp_exponent=2.0,
        weight_lo=0.1,
        weight_hi=8.0,
    )
    assert weights["hero"] <= 8.0
    assert weights["dud"] >= 0.1


def test_amplified_weights_shrinkage_strength() -> None:
    """At n == prior_strength, posterior is between prior and observed.

    With prior_mean ≈ 0.5 (median of [0, 1]) and strength=6, an item with
    n=6, s=0 should land between 0 and 0.5.
    """
    counts = {"alpha": 100, "beta": 100, "gamma": 6}
    s1 = {"alpha": 50, "beta": 50, "gamma": 0}
    # Establish prior_mean ≈ 0.5 from confident items.
    weights = _amplified_weights_from_counts(counts=counts, s1_counts=s1, min_used=3)
    # gamma had observed rate 0; with shrinkage it should NOT bottom out.
    assert weights["gamma"] > 0.3  # above clamp floor
    assert weights["gamma"] < 1.0  # but below mean (it has zero wins)


def _row(template: str, *, s1: bool) -> dict:
    return {
        "graph_json": ('{"metadata": {"templates_used": ["' + template + '"]}}'),
        "stage0_any_passed": 1,
        "stage1_any_passed": 1 if s1 else 0,
        "latest_timestamp": 1.0,
    }


def test_compute_template_weights_includes_new_templates(tmp_path: Path) -> None:
    """End-to-end: a brand-new template with n=1 lands in the output dict.

    Before the shrinkage refactor this template would be excluded because
    n < min_used=3, so the downstream multiplier chain (synergy / bayes /
    scheduler) never saw it.
    """
    # Established template: 8 attempts, 4 S1 passes.
    deduped = [_row("established", s1=(i < 4)) for i in range(8)]
    # Fresh template: 1 attempt, 1 S1 pass (the new-substrate case).
    deduped.append(_row("fresh", s1=True))

    analytics = _AnalyticsStub(tmp_path / "unused.sqlite3")
    with patch.object(_WeightsMixin, "_deduped_graph_rows", return_value=deduped):
        weights = analytics.compute_template_weights(since_ts=0.0, min_used=3)

    assert "fresh" in weights, (
        "n=1 templates must appear in the analytics dict so the multiplier "
        "chain can see them — that's the whole point of the shrinkage fix."
    )
    assert "established" in weights
    # Both should be in clamp range.
    assert 0.3 <= weights["fresh"] <= 5.0
    assert 0.3 <= weights["established"] <= 5.0


def test_compute_template_weights_legacy_excluded_low_n(tmp_path: Path) -> None:
    """Regression check: the legacy behaviour (excluding low-N) is gone.

    Pre-shrinkage code returned ``{"established": ...}`` only. This test
    pins the fix: ``fresh`` must now be present.
    """
    deduped = [_row("established", s1=(i < 4)) for i in range(8)]
    deduped.append(_row("fresh", s1=False))

    analytics = _AnalyticsStub(tmp_path / "unused.sqlite3")
    with patch.object(_WeightsMixin, "_deduped_graph_rows", return_value=deduped):
        weights = analytics.compute_template_weights(since_ts=0.0, min_used=3)

    assert "fresh" in weights
    # With observed rate 0 but a prior_mean=0.5 (from established's 4/8),
    # posterior pulls fresh's rate up — final weight is still below 1.0
    # because relative < 1.0, but the heavy confidence-shrink keeps it near 1.
    assert weights["fresh"] < weights["established"] or weights["fresh"] >= 0.3


def test_prior_strength_constant_is_documented() -> None:
    """Pin the prior strength so a stealth tune doesn't silently shift behaviour."""
    assert _BAYES_PRIOR_STRENGTH == 6.0
