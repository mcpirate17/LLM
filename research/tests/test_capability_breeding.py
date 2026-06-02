"""Capability-weighted breeding credit (_success_credit).

Foundation fix 2026-05-24: analytics breeding weights credit a graph by the
in-context capability it shows (induction AUC), not merely for clearing the
non-discriminative stage-1 (perplexity) gate. Soft tilt — keeps perplexity
passers and slow-capability SSMs in play. See
Obsidian note `capability_breeding_2026-05-24`.
"""

import importlib

import pytest

import research.scientist.analytics._exp_weights as W


@pytest.fixture
def mod(monkeypatch):
    monkeypatch.delenv("ARIA_DISABLE_CAPABILITY_BREEDING", raising=False)
    importlib.reload(W)
    return W


def test_non_s1_gets_zero_credit(mod):
    assert mod._success_credit({"stage1_any_passed": False}) == 0.0
    assert mod._success_credit({}) == 0.0


def test_high_induction_gets_full_credit(mod):
    c = mod._success_credit(
        {"stage1_any_passed": True, "induction_screening_auc_500": 0.6}
    )
    assert c == pytest.approx(1.0)


def test_low_induction_gets_baseline_only(mod):
    # AUC at/below floor -> only the base credit, NOT full credit.
    c = mod._success_credit(
        {"stage1_any_passed": True, "induction_screening_auc_500": 0.02}
    )
    assert c == pytest.approx(mod._CAP_BASE)


def test_capable_outranks_incapable(mod):
    """The whole point: a capable s1-pass must earn strictly more than an
    equally-passing capability-blind (e.g. position-independent) graph."""
    capable = mod._success_credit(
        {"stage1_any_passed": True, "induction_screening_auc_500": 0.5}
    )
    blind = mod._success_credit(
        {"stage1_any_passed": True, "induction_screening_auc_500": 0.0}
    )
    assert capable > blind
    # ...but the blind one is NOT zeroed (soft penalty, SSM-friendly floor).
    assert blind >= mod._CAP_BASE


def test_unmeasured_capability_is_neutral(mod):
    c = mod._success_credit({"stage1_any_passed": True})
    assert mod._CAP_BASE < c < 1.0


def test_env_flag_restores_legacy_signal(monkeypatch):
    monkeypatch.setenv("ARIA_DISABLE_CAPABILITY_BREEDING", "1")
    importlib.reload(W)
    try:
        # legacy: any stage-1 pass == full credit regardless of capability
        assert (
            W._success_credit(
                {"stage1_any_passed": True, "induction_screening_auc_500": 0.0}
            )
            == 1.0
        )
        assert W._success_credit({"stage1_any_passed": False}) == 0.0
    finally:
        monkeypatch.delenv("ARIA_DISABLE_CAPABILITY_BREEDING", raising=False)
        importlib.reload(W)
