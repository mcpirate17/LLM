"""Regression tests for the v10 scoring kwargs pipeline.

Bug shipped 2026-04-25: ``_pr_dict_to_score_kwargs`` did not forward any
of the v9 trajectory fields (``fp_jacobian_erf_density``,
``fp_id_collapse_rate``, ``fp_jacobian_erf_decay_slope``,
``fp_logit_margin_velocity``, ``fp_jacobian_erf_variance``,
``fp_icld_velocity``). v10's capability tier and aux trajectory tier read
those fields by exactly those keys via ``kw.get(...)``, so every
leaderboard rescore silently zeroed ~120 pts of capability + aux
components on every row. A frontier graph (erf_density=1.0,
logit_margin_velocity=0.25) lost ~75 points.

Pulled out of test_scoring_binding_safety.py because that file's
``importorskip("tasks.induction_native_probe...")`` skips every test in
it — but the kwargs plumbing is independent of the native induction
probe and must stay covered.
"""

from __future__ import annotations

import copy

import pytest

from research.scientist.leaderboard_scoring import (
    _pr_dict_to_score_kwargs,
    compute_composite,
)


pytestmark = pytest.mark.unit


def _make_pr_dict(**overrides):
    """Minimal program_results dict — only what the kwargs builder needs."""
    base = {
        "result_id": "rid",
        "wikitext_perplexity": 8.0,
        "wikitext_ppl_200": 12.0,
        "wikitext_ppl_500": 10.0,
        "wikitext_eval_steps": 750,
        "param_count": 1_000_000,
        "ar_auc": 0.30,
        "induction_auc": 0.40,
        "binding_auc": 0.50,
        "novelty_confidence": 0.7,
        "tinystories_score": 0.5,
        "blimp_overall_accuracy": 0.55,
        "fp_jacobian_spectral_norm": 1.5,
    }
    base.update(overrides)
    return base


def _make_lb_row(**overrides):
    base = {
        "tier": "investigation",
        "screening_loss_ratio": 0.75,
        "wikitext_perplexity": 8.0,
        "screening_novelty": 0.5,
        "novelty_confidence": 0.7,
    }
    base.update(overrides)
    return base


def test_v9_trajectory_metrics_plumbed_to_kwargs():
    """All six v9 trajectory fields must surface in score kwargs verbatim.

    The v10 capability + aux scorers index the kwargs dict by exactly
    these strings — no aliasing, no transformation.
    """
    pr = _make_pr_dict(
        fp_jacobian_erf_density=0.95,
        fp_jacobian_erf_variance=120000.0,
        fp_jacobian_erf_decay_slope=-0.08,
        fp_id_collapse_rate=-0.012,
        fp_logit_margin_velocity=0.05,
        fp_icld_velocity=-0.03,
    )
    d = _make_lb_row()
    kw = _pr_dict_to_score_kwargs(pr, d, is_reference=False)
    assert kw["fp_jacobian_erf_density"] == 0.95
    assert kw["fp_jacobian_erf_variance"] == 120000.0
    assert kw["fp_jacobian_erf_decay_slope"] == -0.08
    assert kw["fp_id_collapse_rate"] == -0.012
    assert kw["fp_logit_margin_velocity"] == 0.05
    assert kw["fp_icld_velocity"] == -0.03


def test_v9_trajectory_metrics_default_to_none_when_absent():
    """Missing v9 metrics surface as None (not KeyError) so the v10 capability
    scorers can short-circuit to 0 instead of crashing.
    """
    pr = _make_pr_dict()
    d = _make_lb_row()
    kw = _pr_dict_to_score_kwargs(pr, d, is_reference=False)
    for key in (
        "fp_jacobian_erf_density",
        "fp_jacobian_erf_variance",
        "fp_jacobian_erf_decay_slope",
        "fp_id_collapse_rate",
        "fp_logit_margin_velocity",
        "fp_icld_velocity",
    ):
        assert key in kw, f"missing v9 key in score kwargs: {key}"
        assert kw[key] is None, f"unexpected non-None value for {key}: {kw[key]}"


def test_v10_capability_tier_credits_v9_metrics_via_full_pipeline():
    """End-to-end: dense v9 trajectory metrics earn ≫30pts more than NULLs.

    Pre-fix the delta was 0 because the kwargs builder dropped fp_* keys
    before they reached compute_composite_v10. A frontier-density row
    (erf_density=1.0, logit_margin_velocity=0.25, etc.) should outscore
    the all-NULL row by roughly the v10 capability-tier weight (4×25pts).
    """
    pr_dense = _make_pr_dict(
        fp_jacobian_erf_density=1.0,
        fp_logit_margin_velocity=0.25,
        fp_jacobian_erf_decay_slope=-0.05,
        fp_jacobian_erf_variance=120000.0,
    )
    pr_empty = _make_pr_dict()
    d = _make_lb_row(tier="investigation")
    kw_dense = _pr_dict_to_score_kwargs(copy.deepcopy(pr_dense), d, is_reference=False)
    kw_empty = _pr_dict_to_score_kwargs(copy.deepcopy(pr_empty), d, is_reference=False)
    score_dense = float(compute_composite(**kw_dense))
    score_empty = float(compute_composite(**kw_empty))

    delta = score_dense - score_empty
    assert delta > 30.0, (
        f"v9 metrics earned only {delta:+.2f} pts in v10 capability tier — "
        f"pipeline likely dropping fp_* fields again."
    )


def test_pr_select_cols_includes_v9_trajectory_fields():
    """The single-row SQL select must also pull the v9 columns or the
    rescore-via-build_score_kwargs path will hit the bug from a different
    direction.
    """
    from research.scientist.leaderboard_scoring import _PR_SELECT_COLS

    for col in (
        "fp_jacobian_erf_density",
        "fp_jacobian_erf_variance",
        "fp_jacobian_erf_decay_slope",
        "fp_id_collapse_rate",
        "fp_logit_margin_velocity",
        "fp_icld_velocity",
    ):
        assert col in _PR_SELECT_COLS, f"_PR_SELECT_COLS missing {col}"
