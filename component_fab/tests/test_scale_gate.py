"""Tests for the scale-gate before promotion (run_autonomous._scale_gate_promotions).

A fresh promotion is re-verified at scale; a candidate that does not beat its
anchor at scale is REJECTED (terminal) rather than promoted, because the nano
paired-CI promoted candidates that lose to softmax at scale. The expensive
paired probe is monkeypatched here — these tests pin the gate's decision logic.
"""

from __future__ import annotations

from pathlib import Path

import component_fab.validator.paired as paired_mod
from component_fab.policies.promotion import PromotionDecision
from component_fab.state.ledger import (
    PROMOTION_PROMOTED,
    PROMOTION_REJECTED,
    Ledger,
)
from component_fab.tools import run_autonomous as ra


def _ledger_with_pending(tmp_path: Path) -> Ledger:
    led = Ledger(tmp_path / "l.jsonl")
    led.record_grade(
        proposal_id="p1",
        name="invent_x",
        category="lane",
        synthesis_kind="invent",
        cycle=1,
        composite_score=0.7,
        smoke_pass=True,
        learned_signal=False,
        metadata={"math_axes": {}},
    )
    return led


def _patch_probe(monkeypatch, *, excludes_zero: bool, ci_low: float) -> None:
    monkeypatch.setattr(ra, "spec_from_ledger_entry", lambda entry: object())
    monkeypatch.setattr(
        paired_mod,
        "paired_metadata_for_spec",
        lambda spec, **kw: {
            "paired_delta_ci_excludes_zero": excludes_zero,
            "paired_delta_ci_low": ci_low,
            "paired_anchor_op": "frontier:causal_attention",
        },
    )


def test_scale_gate_rejects_scale_loser(monkeypatch, tmp_path: Path):
    led = _ledger_with_pending(tmp_path)
    _patch_probe(monkeypatch, excludes_zero=False, ci_low=-1.42)
    decisions = [PromotionDecision("p1", PROMOTION_PROMOTED, "nano streak met")]
    out = ra._scale_gate_promotions(
        led, decisions, dim=96, steps=1500, seeds=2, seq_len=32
    )
    assert out[0].decision == PROMOTION_REJECTED
    assert "scale-gate" in out[0].reason


def test_scale_gate_keeps_scale_winner(monkeypatch, tmp_path: Path):
    led = _ledger_with_pending(tmp_path)
    _patch_probe(monkeypatch, excludes_zero=True, ci_low=0.04)
    decisions = [PromotionDecision("p1", PROMOTION_PROMOTED, "nano streak met")]
    out = ra._scale_gate_promotions(
        led, decisions, dim=96, steps=1500, seeds=2, seq_len=32
    )
    assert out[0].decision == PROMOTION_PROMOTED


def test_scale_gate_passthrough_non_promotion(monkeypatch, tmp_path: Path):
    # A non-promotion decision is never scale-probed (would raise if it were).
    def _boom(*a, **k):
        raise AssertionError("scale probe must not run on non-promotions")

    led = _ledger_with_pending(tmp_path)
    monkeypatch.setattr(paired_mod, "paired_metadata_for_spec", _boom)
    decisions = [PromotionDecision("p1", "pending", "streak not met")]
    out = ra._scale_gate_promotions(
        led, decisions, dim=96, steps=1500, seeds=2, seq_len=32
    )
    assert out[0].decision == "pending"
