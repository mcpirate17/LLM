"""Promotion + rejection policy for the autonomous fab loop.

Decides which proposals graduate from "graded" to "promoted" (shippable),
and which get rejected (don't try again). Reads off the cross-cycle
ledger and consults a small set of rules.

Default rules:
- **promote** when the last K cycles' composite scores all >= threshold
  AND smoke has passed each time AND in-context probe has shown learned
  signal at least once.
- **reject** when after N attempts the composite has never crossed a
  low floor AND smoke has failed at least once.
- **pending** otherwise — keep trying.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..state.ledger import (
    Ledger,
    LedgerEntry,
    PROMOTION_PENDING,
    PROMOTION_PROMOTED,
    PROMOTION_REJECTED,
)


@dataclass(frozen=True, slots=True)
class PromotionRules:
    promote_min_streak_cycles: int = 2
    promote_min_composite: float = 0.6
    # Day-6: don't require learned signal — research/'s runs.db has BLiMP
    # winners that failed loss-ratio metrics. User notes loss_ratio is
    # not a reliable model-strength indicator.
    promote_require_learned_signal: bool = False
    reject_after_n_cycles: int = 4
    # Day-6: lowered from 0.35 → 0.20 so smoke-passing specs whose
    # learning subscore is weak (but binding/cross OK) aren't auto-killed
    # before they can build a 2-cycle streak.
    reject_max_composite: float = 0.20


DEFAULT_PROMOTION_RULES = PromotionRules()


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    proposal_id: str
    decision: str  # one of pending/promoted/rejected
    reason: str
    composite_history: tuple[float, ...] = field(default_factory=tuple)


def _promote_streak_satisfied(entry: LedgerEntry, rules: PromotionRules) -> bool:
    if len(entry.composite_history) < rules.promote_min_streak_cycles:
        return False
    recent = entry.composite_history[-rules.promote_min_streak_cycles :]
    if not all(score >= rules.promote_min_composite for score in recent):
        return False
    if entry.smoke_pass_count < rules.promote_min_streak_cycles:
        return False
    if rules.promote_require_learned_signal and entry.learned_signal_count == 0:
        return False
    return True


def _reject_satisfied(entry: LedgerEntry, rules: PromotionRules) -> bool:
    if len(entry.composite_history) < rules.reject_after_n_cycles:
        return False
    return all(score <= rules.reject_max_composite for score in entry.composite_history)


def decide_promotion(
    entry: LedgerEntry, rules: PromotionRules = DEFAULT_PROMOTION_RULES
) -> PromotionDecision:
    if entry.promotion_status == PROMOTION_PROMOTED:
        return PromotionDecision(
            proposal_id=entry.proposal_id,
            decision=PROMOTION_PROMOTED,
            reason="already promoted",
            composite_history=tuple(entry.composite_history),
        )
    if entry.promotion_status == PROMOTION_REJECTED:
        return PromotionDecision(
            proposal_id=entry.proposal_id,
            decision=PROMOTION_REJECTED,
            reason="already rejected",
            composite_history=tuple(entry.composite_history),
        )
    if _promote_streak_satisfied(entry, rules):
        return PromotionDecision(
            proposal_id=entry.proposal_id,
            decision=PROMOTION_PROMOTED,
            reason=(
                f"composite >= {rules.promote_min_composite} for "
                f"{rules.promote_min_streak_cycles} consecutive cycles "
                f"with smoke + learned signal"
            ),
            composite_history=tuple(entry.composite_history),
        )
    if _reject_satisfied(entry, rules):
        return PromotionDecision(
            proposal_id=entry.proposal_id,
            decision=PROMOTION_REJECTED,
            reason=(
                f"after {rules.reject_after_n_cycles} cycles, composite "
                f"never exceeded {rules.reject_max_composite}"
            ),
            composite_history=tuple(entry.composite_history),
        )
    return PromotionDecision(
        proposal_id=entry.proposal_id,
        decision=PROMOTION_PENDING,
        reason="streak not yet satisfied",
        composite_history=tuple(entry.composite_history),
    )


def decide_promotions_for_ledger(
    ledger: Ledger,
    rules: PromotionRules = DEFAULT_PROMOTION_RULES,
) -> list[PromotionDecision]:
    return [decide_promotion(entry, rules) for entry in ledger.all_entries()]


def apply_decisions(
    ledger: Ledger, decisions: Iterable[PromotionDecision]
) -> dict[str, int]:
    counts = {PROMOTION_PROMOTED: 0, PROMOTION_REJECTED: 0, PROMOTION_PENDING: 0}
    for decision in decisions:
        entry = ledger.entries.get(decision.proposal_id)
        if entry is None:
            continue
        if entry.promotion_status == decision.decision:
            counts[decision.decision] = counts.get(decision.decision, 0) + 1
            continue
        if decision.decision in (PROMOTION_PROMOTED, PROMOTION_REJECTED):
            ledger.record_promotion(decision.proposal_id, decision.decision)
        counts[decision.decision] = counts.get(decision.decision, 0) + 1
    return counts
