"""Tests for WS-2: paired delta CI + the noise-guard promotion gate."""

from __future__ import annotations

from pathlib import Path

import pytest
from torch import nn

from component_fab.policies.promotion import (
    PROMOTION_PENDING,
    PROMOTION_PROMOTED,
    PromotionRules,
    decide_promotion,
)
from component_fab.improver.axis_variants import DEFAULT_META_DB
from component_fab.proposer.spec_generator import ProposalSpec
from component_fab.state.ledger import Ledger
from component_fab.validator.paired import (
    paired_delta_ci,
    paired_metadata_for_spec,
    run_paired_probe,
)
from component_fab.tests.conftest import make_spec


def _spec(axes: dict, anchor: str) -> ProposalSpec:
    return make_spec(axes, "cand", anchor_witness_op=anchor, rationale="t")


# --------------------------------------------------------------------------- #
# paired_delta_ci (pure stats)
# --------------------------------------------------------------------------- #
def test_ci_excludes_zero_for_consistent_positive_delta():
    ci = paired_delta_ci([0.30, 0.28, 0.35, 0.31])
    assert ci.excludes_zero
    assert ci.ci_low > 0.0
    assert ci.mean == pytest.approx(0.31, abs=0.02)


def test_ci_includes_zero_for_noisy_delta():
    # straddles zero -> not significant
    ci = paired_delta_ci([0.30, -0.25, 0.10, -0.18])
    assert not ci.excludes_zero
    assert ci.ci_low < 0.0 < ci.ci_high


def test_single_seed_never_significant():
    ci = paired_delta_ci([5.0])  # one lucky seed
    assert not ci.excludes_zero
    assert ci.ci_low == float("-inf")


def test_invalid_confidence_raises():
    with pytest.raises(ValueError):
        paired_delta_ci([0.1, 0.2, 0.3], confidence=0.0)
    with pytest.raises(ValueError):
        paired_delta_ci([0.1, 0.2, 0.3], confidence=1.0)


def test_non_default_confidence_supported():
    # scipy t.ppf replaced the 95%-only table: a 90% CI is strictly narrower.
    ci_95 = paired_delta_ci([0.1, 0.2, 0.3], confidence=0.95)
    ci_90 = paired_delta_ci([0.1, 0.2, 0.3], confidence=0.90)
    assert ci_90.confidence == 0.90
    assert ci_95.ci_low < ci_90.ci_low < ci_90.ci_high < ci_95.ci_high


def test_to_metadata_keys():
    md = paired_delta_ci([0.3, 0.31, 0.29]).to_metadata()
    assert md["paired_delta_ci_excludes_zero"] is True
    assert set(md) == {
        "paired_delta_n",
        "paired_delta_mean",
        "paired_delta_ci_low",
        "paired_delta_ci_high",
        "paired_delta_ci_excludes_zero",
    }


# --------------------------------------------------------------------------- #
# run_paired_probe (compute plumbing — light smoke)
# --------------------------------------------------------------------------- #
def test_run_paired_probe_returns_ci_per_seed():
    # nn.Linear is a valid [B,S,D]->[B,S,D] lane; identical factories.
    ci = run_paired_probe(
        lambda: nn.Linear(16, 16),
        lambda: nn.Linear(16, 16),
        seeds=(0, 1),
        dim=16,
        seq_len=16,
        n_steps=5,
    )
    assert ci.n == 2
    assert ci.ci_low <= ci.mean <= ci.ci_high


def test_run_paired_probe_requires_seeds():
    with pytest.raises(ValueError):
        run_paired_probe(lambda: nn.Linear(8, 8), lambda: nn.Linear(8, 8), seeds=())


# --------------------------------------------------------------------------- #
# paired_metadata_for_spec (anchor build + explicit skips — no silent fallback)
# --------------------------------------------------------------------------- #
def test_metadata_frontier_anchors_when_no_witness_op():
    # No anchor witness op -> fall back to the softmax causal-attention FRONTIER
    # baseline (greenlit 2026-06-16) so "beats frontier with CI>0" is the
    # promotion path, instead of skipping. Emits CI keys, not a skip reason.
    md = paired_metadata_for_spec(
        _spec({"op_invention_mechanism": "causal_fast_weight_memory"}, ""),
        seeds=(0, 1),
    )
    assert md.get("paired_anchor_op") == "frontier:causal_attention"
    assert "paired_delta_ci_excludes_zero" in md
    assert "paired_skipped_reason" not in md


def test_metadata_frontier_anchors_when_anchor_unbuildable():
    # An anchor that cannot be built (softmax_attention has no dispatch trigger)
    # also falls back to the frontier baseline rather than skipping.
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    md = paired_metadata_for_spec(
        _spec(
            {"op_invention_mechanism": "causal_fast_weight_memory"},
            "softmax_attention",
        ),
        seeds=(0, 1),
    )
    assert md.get("paired_anchor_op") == "frontier:causal_attention"
    assert "paired_delta_ci_excludes_zero" in md
    assert "paired_skipped_reason" not in md


def test_metadata_emits_ci_for_buildable_anchor():
    if not DEFAULT_META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    md = paired_metadata_for_spec(
        _spec(
            {"op_invention_mechanism": "causal_fast_weight_memory"},
            "tropical_attention",
        ),
        seeds=(0, 1, 2),
        dim=16,
        seq_len=16,
        n_steps=8,
    )
    assert md["paired_anchor_op"] == "tropical_attention"
    assert md["paired_delta_n"] == 3
    assert "paired_delta_ci_excludes_zero" in md


def test_loss_specialist_metadata_pairs_against_declared_partner_slot(monkeypatch):
    from component_fab.generator.memory_primitives import MultiHeadSlotTableMemoryLane

    def fake_run(candidate_factory, anchor_factory, **kwargs):  # noqa: ANN001
        candidate = candidate_factory()
        anchor = anchor_factory()
        assert candidate.__class__.__name__ == "LossMonsterPairedBlock"
        assert isinstance(anchor, MultiHeadSlotTableMemoryLane)
        assert kwargs["anchor_cache_key"] == (
            "loss_specialist_partner_slot",
            "slot_table_memory",
        )
        return paired_delta_ci([0.2, 0.21, 0.22])

    monkeypatch.setattr("component_fab.validator.paired.run_paired_probe", fake_run)
    md = paired_metadata_for_spec(
        _spec(
            {
                "op_block_template": "loss_monster_paired",
                "op_partner_kind": "slot_dplr",
                "op_block_slot_loss": "routed_bottleneck",
                "op_candidate_role": "loss_specialist_pair",
            },
            "",
        ),
        seeds=(0, 1, 2),
        dim=16,
    )

    assert md["paired_anchor_op"] == "loss_partner:slot_dplr"
    assert md["paired_delta_ci_excludes_zero"] is True


def test_loss_specialist_metadata_skips_when_partner_unbuildable(monkeypatch):
    def fail_if_called(*args, **kwargs):  # noqa: ANN001
        raise AssertionError("loss specialist without buildable partner must skip")

    monkeypatch.setattr(
        "component_fab.validator.paired.run_paired_probe", fail_if_called
    )
    md = paired_metadata_for_spec(
        _spec(
            {
                "op_block_template": "loss_monster_paired",
                "op_partner_kind": "unknown_partner",
                "op_candidate_role": "loss_specialist_pair",
            },
            "",
        ),
        dim=16,
    )

    assert md["paired_skipped_reason"].startswith(
        "loss_specialist_partner_unbuildable"
    )


# --------------------------------------------------------------------------- #
# Promotion gate (the WS-2 acceptance criterion)
# --------------------------------------------------------------------------- #
def _streak_ledger(tmp_path: Path, pid: str, metadata: dict, *, cycles: int) -> Ledger:
    ledger = Ledger(tmp_path / f"{pid}.jsonl")
    for cycle in range(1, cycles + 1):
        ledger.record_grade(
            proposal_id=pid,
            name=pid,
            category="lane",
            synthesis_kind="novel_hybrid",
            cycle=cycle,
            composite_score=0.8,  # streak-eligible every cycle
            smoke_pass=True,
            learned_signal=True,
            metadata=metadata,
        )
    return ledger


def test_noop_candidate_never_promoted_across_20_cycles(tmp_path: Path):
    """A no-op (paired CI vs anchor includes zero) is never promoted, ever."""
    md = {"paired_delta_ci_excludes_zero": False, "paired_delta_ci_low": -0.12}
    ledger = Ledger(tmp_path / "noop.jsonl")
    for cycle in range(1, 21):
        ledger.record_grade(
            proposal_id="noop",
            name="noop",
            category="lane",
            synthesis_kind="novel_hybrid",
            cycle=cycle,
            composite_score=0.8,
            smoke_pass=True,
            learned_signal=True,
            metadata=md,
        )
        decision = decide_promotion(ledger.entries["noop"], PromotionRules())
        assert decision.decision == PROMOTION_PENDING
        # Once the 2-cycle streak is satisfied, it is the CI guard (not the
        # streak) that holds promotion back.
        if cycle >= 2:
            assert "noise guard" in decision.reason


def test_real_winner_with_significant_delta_promotes(tmp_path: Path):
    md = {"paired_delta_ci_excludes_zero": True, "paired_delta_ci_low": 0.08}
    entry = _streak_ledger(tmp_path, "winner", md, cycles=2).entries["winner"]
    assert decide_promotion(entry, PromotionRules()).decision == PROMOTION_PROMOTED


def test_ci_low_fallback_when_flag_absent(tmp_path: Path):
    # Only ci_low present (no boolean) -> derive from ci_low > 0.
    blocked = _streak_ledger(
        tmp_path, "b", {"paired_delta_ci_low": -0.01}, cycles=2
    ).entries["b"]
    assert decide_promotion(blocked, PromotionRules()).decision == PROMOTION_PENDING
    ok = _streak_ledger(
        tmp_path, "ok", {"paired_delta_ci_low": 0.05}, cycles=2
    ).entries["ok"]
    assert decide_promotion(ok, PromotionRules()).decision == PROMOTION_PROMOTED


def test_entry_without_ci_stays_pending_by_default(tmp_path: Path):
    # Default P0 behavior is fail-closed: a modern promotion streak without
    # paired evidence stays pending instead of silently promoting.
    entry = _streak_ledger(tmp_path, "legacy", {}, cycles=2).entries["legacy"]
    decision = decide_promotion(entry, PromotionRules())
    assert decision.decision == PROMOTION_PENDING
    assert "paired promotion evidence missing" in decision.reason


def test_legacy_entry_without_ci_requires_explicit_grandfathering(tmp_path: Path):
    entry = _streak_ledger(tmp_path, "legacy", {}, cycles=2).entries["legacy"]
    rules = PromotionRules(grandfather_legacy_missing_evidence=True)
    assert decide_promotion(entry, rules).decision == PROMOTION_PROMOTED


def test_skipped_paired_evidence_blocks_promotion(tmp_path: Path):
    md = {"paired_skipped_reason": "anchor_unbuildable:softmax_attention"}
    entry = _streak_ledger(tmp_path, "skip", md, cycles=2).entries["skip"]
    decision = decide_promotion(entry, PromotionRules())
    assert decision.decision == PROMOTION_PENDING
    assert "anchor_unbuildable:softmax_attention" in decision.reason


def test_guard_off_ignores_ci(tmp_path: Path):
    md = {"paired_delta_ci_excludes_zero": False}
    entry = _streak_ledger(tmp_path, "off", md, cycles=2).entries["off"]
    rules = PromotionRules(require_ci_excludes_zero=False)
    assert decide_promotion(entry, rules).decision == PROMOTION_PROMOTED


def test_loss_specialist_cannot_promote_without_carrier(tmp_path: Path):
    md = {
        "candidate_role": "loss_specialist",
        "loss_specialist_paired": True,
        "paired_delta_ci_excludes_zero": True,
        "paired_delta_ci_low": 0.05,
    }
    entry = _streak_ledger(tmp_path, "loss_only", md, cycles=2).entries["loss_only"]
    decision = decide_promotion(entry, PromotionRules())
    assert decision.decision == PROMOTION_PENDING
    assert "missing long-range carrier" in decision.reason


def test_loss_specialist_cannot_promote_when_unpaired(tmp_path: Path):
    md = {
        "candidate_role": "loss_specialist",
        "loss_specialist_partner_op": "hyper_mor_b_145m",
        "paired_delta_ci_excludes_zero": True,
        "paired_delta_ci_low": 0.05,
    }
    entry = _streak_ledger(tmp_path, "solo_loss", md, cycles=2).entries["solo_loss"]
    decision = decide_promotion(entry, PromotionRules())
    assert decision.decision == PROMOTION_PENDING
    assert "must be paired" in decision.reason


def test_loss_specialist_pair_promotes_with_positive_carrier_delta(tmp_path: Path):
    md = {
        "candidate_role": "loss_specialist_pair",
        "loss_specialist_partner_op": "hyper_mor_b_145m",
        "paired_delta_ci_excludes_zero": True,
        "paired_delta_ci_low": 0.05,
    }
    entry = _streak_ledger(tmp_path, "paired_loss", md, cycles=2).entries["paired_loss"]
    assert decide_promotion(entry, PromotionRules()).decision == PROMOTION_PROMOTED


def test_loss_specialist_pair_guard_reads_nested_math_axes(tmp_path: Path):
    md = {
        "math_axes": {
            "op_candidate_role": "loss_specialist_pair",
            "op_loss_specialist_partner_op": "hyper_mor_b_145m",
        },
        "paired_delta_ci_excludes_zero": True,
        "paired_delta_ci_low": 0.05,
    }
    entry = _streak_ledger(tmp_path, "nested_pair", md, cycles=2).entries[
        "nested_pair"
    ]
    assert decide_promotion(entry, PromotionRules()).decision == PROMOTION_PROMOTED
