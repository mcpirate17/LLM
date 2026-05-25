"""Tests for the opt-in range-blind promotion veto."""

from __future__ import annotations

from pathlib import Path

from component_fab.policies.promotion import (
    PROMOTION_PENDING,
    PROMOTION_PROMOTED,
    PromotionRules,
    decide_promotion,
)
from component_fab.state.ledger import Ledger


def _promote_eligible_entry(tmp_path: Path, pid: str, metadata: dict):
    """Build a 2-cycle promote-eligible streak with the given grade metadata."""
    ledger = Ledger(tmp_path / f"{pid}.jsonl")
    for cycle in (1, 2):
        ledger.record_grade(
            proposal_id=pid,
            name=pid,
            category="lane",
            synthesis_kind="novel_hybrid",
            cycle=cycle,
            composite_score=0.8,
            smoke_pass=True,
            learned_signal=True,
            metadata=metadata,
        )
    return ledger.entries[pid]


def test_default_rules_ignore_range(tmp_path: Path) -> None:
    entry = _promote_eligible_entry(tmp_path, "blind", {"range_effective_distance": 0})
    # Default (veto off): a range-blind candidate still promotes on its streak.
    assert decide_promotion(entry, PromotionRules()).decision == PROMOTION_PROMOTED


def test_veto_blocks_confirmed_range_blind(tmp_path: Path) -> None:
    entry = _promote_eligible_entry(tmp_path, "blind", {"range_effective_distance": 0})
    rules = PromotionRules(veto_range_blind=True)
    assert decide_promotion(entry, rules).decision == PROMOTION_PENDING


def test_veto_allows_range_binder(tmp_path: Path) -> None:
    entry = _promote_eligible_entry(
        tmp_path, "binder", {"range_effective_distance": 256}
    )
    rules = PromotionRules(veto_range_blind=True)
    assert decide_promotion(entry, rules).decision == PROMOTION_PROMOTED


def test_veto_ignores_unmeasured_range(tmp_path: Path) -> None:
    # No range measurement -> veto must not block (it vetoes measured failure,
    # not absence of a measurement).
    entry = _promote_eligible_entry(tmp_path, "unmeasured", {})
    rules = PromotionRules(veto_range_blind=True)
    assert decide_promotion(entry, rules).decision == PROMOTION_PROMOTED
