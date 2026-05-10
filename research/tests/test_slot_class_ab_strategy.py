"""Tests for the slot-class A/B strategy resolver."""

from __future__ import annotations

from research.synthesis._slot_constraints_loader import (
    SLOT_CLASS_STRATEGIES,
    resolve_slot_class_strategy,
)


def test_explicit_use_derived_overrides_strategy():
    used, reason = resolve_slot_class_strategy(
        explicit_use_derived=True,
        strategy="static",
        experiment_id="exp-1",
    )
    assert used is True
    assert reason == "explicit_config"


def test_static_strategy_returns_false():
    used, reason = resolve_slot_class_strategy(
        explicit_use_derived=False,
        strategy="static",
        experiment_id="exp-1",
    )
    assert used is False
    assert reason == "strategy_static"


def test_derived_strategy_returns_true():
    used, reason = resolve_slot_class_strategy(
        explicit_use_derived=False,
        strategy="derived",
        experiment_id="exp-1",
    )
    assert used is True
    assert reason == "strategy_derived"


def test_ab_50_50_is_stable_per_exp_id():
    """Same exp_id always lands on the same arm — required for cohort analysis."""
    a1 = resolve_slot_class_strategy(
        explicit_use_derived=False, strategy="ab_50_50", experiment_id="exp-A"
    )
    a2 = resolve_slot_class_strategy(
        explicit_use_derived=False, strategy="ab_50_50", experiment_id="exp-A"
    )
    assert a1 == a2
    assert a1[1] == "strategy_ab_50_50"


def test_ab_50_50_splits_population_across_distinct_ids():
    """Across many exp_ids, both arms should be hit — not stuck on one side."""
    arms = {
        resolve_slot_class_strategy(
            explicit_use_derived=False,
            strategy="ab_50_50",
            experiment_id=f"exp-{i}",
        )[0]
        for i in range(64)
    }
    assert arms == {True, False}


def test_ab_50_50_without_exp_id_falls_back_to_static():
    used, reason = resolve_slot_class_strategy(
        explicit_use_derived=False,
        strategy="ab_50_50",
        experiment_id=None,
    )
    assert used is False
    assert reason == "strategy_static"


def test_unknown_strategy_falls_back_to_static():
    used, reason = resolve_slot_class_strategy(
        explicit_use_derived=False,
        strategy="not_a_strategy",
        experiment_id="exp-X",
    )
    assert used is False
    assert reason == "strategy_static"


def test_strategy_constants_exposed():
    assert "static" in SLOT_CLASS_STRATEGIES
    assert "derived" in SLOT_CLASS_STRATEGIES
    assert "ab_50_50" in SLOT_CLASS_STRATEGIES
