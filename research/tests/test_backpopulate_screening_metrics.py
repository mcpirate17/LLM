from __future__ import annotations

from research.tools.backpopulate_screening_metrics import (
    _needs_post_train,
    _needs_rapid,
    _recover_hellaswag_after_gate_failure,
    _select_updates,
)
from research.scientist.runner import RunConfig


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
    assert not _needs_post_train(
        row,
        force=False,
        target_fields=("hellaswag_acc",),
    )


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


def test_recover_hellaswag_after_gate_failure_respects_skip(monkeypatch):
    called = {"count": 0}

    def _fake_eval(model, vocab_size, device):
        called["count"] += 1
        return {"hellaswag_acc": 0.31, "hellaswag_status": "ok", "hellaswag_total": 50}

    monkeypatch.setattr(
        "research.eval.hellaswag_eval.screening_hellaswag_eval",
        _fake_eval,
    )
    cfg = RunConfig(skip_screening_hellaswag=True)
    assert (
        _recover_hellaswag_after_gate_failure(model=object(), config=cfg, device="cuda")
        == {}
    )
    assert called["count"] == 0


def test_recover_hellaswag_after_gate_failure_returns_metrics(monkeypatch):
    def _fake_eval(model, vocab_size, device):
        assert vocab_size == 100277
        assert device == "cuda:0"
        return {"hellaswag_acc": 0.31, "hellaswag_status": "ok", "hellaswag_total": 50}

    monkeypatch.setattr(
        "research.eval.hellaswag_eval.screening_hellaswag_eval",
        _fake_eval,
    )
    cfg = RunConfig(vocab_size=100277, skip_screening_hellaswag=False)
    assert _recover_hellaswag_after_gate_failure(
        model=object(),
        config=cfg,
        device="cuda:0",
    ) == {
        "hellaswag_acc": 0.31,
        "hellaswag_status": "ok",
        "hellaswag_n_examples": 50,
    }
