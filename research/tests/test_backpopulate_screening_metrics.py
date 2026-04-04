from __future__ import annotations

from research.tools.backpopulate_screening_metrics import (
    _needs_post_train,
    _needs_rapid,
    _select_updates,
)


class _Row(dict):
    def keys(self):
        return super().keys()


def test_needs_rapid_requires_stage05():
    row = _Row(
        stage0_passed=1,
        stage05_passed=0,
        rapid_screening_passed=None,
        rapid_screening_elapsed_ms=None,
        rapid_screening_steps_completed=None,
        rapid_screening_max_steps=None,
    )
    assert not _needs_rapid(row, force=False)


def test_needs_post_train_requires_train_steps():
    row = _Row(
        stage0_passed=1,
        stage05_passed=1,
        n_train_steps=None,
        wikitext_perplexity=None,
        hellaswag_acc=None,
        induction_auc=None,
        binding_auc=None,
        binding_composite=None,
    )
    assert not _needs_post_train(row, force=False)


def test_select_updates_only_fills_missing_without_force():
    row = _Row(induction_auc=0.1, binding_auc=None, rapid_screening_elapsed_ms=None)
    updates = {
        "induction_auc": 0.2,
        "binding_auc": 0.3,
        "rapid_screening_elapsed_ms": 1812.0,
    }
    assert _select_updates(row, updates, force=False) == {
        "binding_auc": 0.3,
        "rapid_screening_elapsed_ms": 1812.0,
    }


def test_select_updates_overwrites_with_force():
    row = _Row(induction_auc=0.1, binding_auc=None)
    updates = {"induction_auc": 0.2, "binding_auc": 0.3}
    assert _select_updates(row, updates, force=True) == updates
