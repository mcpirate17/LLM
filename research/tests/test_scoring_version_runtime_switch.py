"""Regression: scoring version is runtime-mutable via set_scoring_version.

The 2026-04-16 UI selector posts to ``/api/scoring/version`` and expects
the dispatcher to pick up the change without a server restart. Pins the
runtime mutator contract so a refactor doesn't silently freeze the
dispatcher back to import-time state.
"""

from __future__ import annotations

import pytest

from research.scientist import leaderboard_scoring as ls


@pytest.fixture(autouse=True)
def restore_version():
    original = ls.get_scoring_version()
    try:
        yield
    finally:
        ls.set_scoring_version(original)


def test_get_scoring_version_reflects_current_value() -> None:
    ls.set_scoring_version("v8")
    assert ls.get_scoring_version() == "v8"
    ls.set_scoring_version("v8.1")
    assert ls.get_scoring_version() == "v8.1"


def test_set_scoring_version_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        ls.set_scoring_version("v99")


def test_compute_composite_dispatches_to_live_version() -> None:
    """``compute_composite`` must read SCORING_VERSION at call time.

    If the dispatcher captured SCORING_VERSION at import time, the UI
    selector would appear to work but every subsequent composite score
    would silently keep using the old version.
    """
    kwargs = dict(
        ppl_screening=18.0,
        ppl_investigation=15.0,
        ppl_validation=13.0,
        param_count=4_000_000,
        ppl_at_100=20.0,
        ppl_at_500=15.0,
        induction_auc=0.01,
        binding_auc=0.01,
        ar_auc=0.02,
        ar_above_chance=False,
        hellaswag=0.30,
        diagnostic=0.32,
        tinystories=0.40,
        cross_task=0.55,
        hierarchy=0.45,
        inv_failed=False,
    )

    ls.set_scoring_version("v8")
    score_v8 = ls.compute_composite(decompose=False, **kwargs)

    ls.set_scoring_version("v8.1")
    score_v81 = ls.compute_composite(decompose=False, **kwargs)

    # Same inputs, different version — scores must differ because v8.1
    # applies the tighter all-below penalty for this zero-binding case.
    assert score_v8 != pytest.approx(score_v81), (
        f"compute_composite returned the same score ({score_v8}) under "
        "v8 and v8.1 for a zero-binding graph; dispatcher is not picking "
        "up the runtime version change."
    )
    assert score_v81 < score_v8  # v8.1 penalty is harsher
