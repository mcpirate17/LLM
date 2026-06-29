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
from typing import Any, Iterable, Mapping

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
    # 2026-05-25: optional veto on *confirmed* sparse-range blindness. Off by
    # default — range-blindness is legitimate for some specialists (e.g. the
    # AR-curriculum top_ar_block). When on, a candidate whose grade metadata
    # carries a measured ``range_effective_distance`` BELOW the threshold is
    # not promoted; candidates with no range measurement are unaffected (this
    # is a veto on measured failure, not a requirement that range was probed).
    veto_range_blind: bool = False
    min_range_effective_distance: int = 1
    # WS-2 (2026-06-10): require the candidate's paired delta vs its anchor
    # baseline to be significantly positive (95% CI excludes zero) before
    # promotion — kills single-seed noise promotions. LEGACY-SAFE: only blocks
    # when the grade metadata actually carries a paired-delta CI
    # (``paired_delta_ci_excludes_zero`` / ``paired_delta_ci_low``); entries
    # graded before the paired probe was wired in are unaffected, so this never
    # freezes the loop. Once validator/paired.py is wired into grading, every
    # candidate carries a CI and the guard is always live.
    require_ci_excludes_zero: bool = True
    # 2026-06-12 P0: fail closed for NEW promotion evidence. A candidate that
    # has a promotion streak but lacks paired evidence (or whose paired evidence
    # explicitly skipped because the anchor was absent/unbuildable) stays
    # pending. Historical ledgers can be reviewed with
    # ``grandfather_legacy_missing_evidence=True`` when intentionally needed.
    require_complete_promotion_evidence: bool = True
    grandfather_legacy_missing_evidence: bool = False
    # WS-4 (2026-06-10): niche/Pareto promotion. When on, a candidate that sits
    # on the first Pareto front (``metadata.on_pareto_front``) for a streak
    # promotes even if its scalar composite never clears the bar — so a
    # specialist (e.g. binding-only, low learning) graduates in its own niche
    # instead of being crushed by the scalar composite. A front member is also
    # shielded from the low-composite reject rule (it is still competing in its
    # niche — this supersedes ``veto_range_blind``). OFF by default: the grade
    # loop must emit ``on_pareto_front`` (run_autonomous --niche-promotion) first.
    promote_by_pareto: bool = False
    pareto_streak_cycles: int = 2


DEFAULT_PROMOTION_RULES = PromotionRules()

_LOSS_SPECIALIST_ROLES = frozenset(
    {"loss_specialist", "loss_monster", "loss_specialist_pair"}
)


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
    if rules.veto_range_blind and entry.metadata_history:
        latest = entry.metadata_history[-1]
        measured = latest.get("range_effective_distance")
        if measured is not None and measured < rules.min_range_effective_distance:
            return False
    return True


def _paired_ci_satisfied(entry: LedgerEntry, rules: PromotionRules) -> bool:
    """True unless a *present* paired-delta CI vs anchor fails to exclude zero.

    Legacy-safe: absent CI metadata → satisfied (the guard activates only once
    the paired probe is wired into grading and starts emitting the CI).
    """
    if not rules.require_ci_excludes_zero or not entry.metadata_history:
        return True
    latest = entry.metadata_history[-1]
    if "paired_delta_ci_excludes_zero" in latest:
        return bool(latest["paired_delta_ci_excludes_zero"])
    if "paired_delta_ci_low" in latest:
        return float(latest["paired_delta_ci_low"]) > 0.0
    return True


def _promotion_evidence_failure(
    entry: LedgerEntry,
    rules: PromotionRules,
    *,
    pareto_ok: bool,
) -> str | None:
    """Return a fail-closed reason when a promotion streak lacks evidence."""
    if not rules.require_complete_promotion_evidence:
        return None
    if not entry.metadata_history:
        if rules.grandfather_legacy_missing_evidence:
            return None
        return "promotion evidence metadata missing"

    latest = entry.metadata_history[-1]
    loss_specialist_failure = _loss_specialist_pairing_failure(latest)
    if loss_specialist_failure is not None:
        return loss_specialist_failure
    if rules.require_ci_excludes_zero:
        skipped = latest.get("paired_skipped_reason")
        if skipped:
            return f"paired promotion evidence incomplete: {skipped}"
        has_ci = (
            "paired_delta_ci_excludes_zero" in latest or "paired_delta_ci_low" in latest
        )
        if not has_ci:
            if rules.grandfather_legacy_missing_evidence:
                return None
            return "paired promotion evidence missing"

    if pareto_ok:
        recent = entry.metadata_history[-rules.pareto_streak_cycles :]
        if not all("pareto_objective_vector" in metadata for metadata in recent):
            return "niche promotion evidence missing pareto_objective_vector"
    return None


def _loss_specialist_pairing_failure(metadata: dict) -> str | None:
    """Fail closed when a local loss specialist tries to promote unpaired."""
    def value(*keys: str) -> Any:
        return _metadata_value(metadata, *keys)

    role = str(
        value("candidate_role")
        or value("component_role")
        or value("specialist_role")
        or ""
    )
    is_loss_specialist = role in _LOSS_SPECIALIST_ROLES or bool(
        value("loss_specialist")
    )
    if not is_loss_specialist:
        return None

    carrier = (
        value("loss_specialist_partner_op")
        or value("loss_specialist_carrier_op")
        or value("paired_anchor_op")
    )
    if not carrier:
        return "loss specialist missing long-range carrier evidence"

    if role != "loss_specialist_pair" and not bool(value("loss_specialist_paired")):
        return "loss specialist must be paired with a long-range carrier"

    ci_excludes_zero = value("paired_delta_ci_excludes_zero")
    ci_low = value("paired_delta_ci_low")
    if ci_excludes_zero is not None and not bool(ci_excludes_zero):
        return "loss specialist carrier delta is not significantly positive"
    if ci_low is not None and float(ci_low) <= 0.0:
        return "loss specialist carrier delta is not positive"
    if ci_excludes_zero is None and ci_low is None:
        return "loss specialist missing paired carrier delta"
    return None


def _metadata_value(metadata: Mapping[str, Any], *keys: str) -> Any:
    axes = metadata.get("math_axes")
    axis_map = axes if isinstance(axes, Mapping) else {}
    for key in keys:
        if key in metadata:
            return metadata[key]
        op_key = f"op_{key}"
        if op_key in metadata:
            return metadata[op_key]
        if key in axis_map:
            return axis_map[key]
        if op_key in axis_map:
            return axis_map[op_key]
    return None


def _pareto_streak_satisfied(entry: LedgerEntry, rules: PromotionRules) -> bool:
    """True when the entry held a first-Pareto-front slot for the recent streak."""
    if not rules.promote_by_pareto:
        return False
    if len(entry.metadata_history) < rules.pareto_streak_cycles:
        return False
    if entry.smoke_pass_count < rules.pareto_streak_cycles:
        return False
    recent = entry.metadata_history[-rules.pareto_streak_cycles :]
    return all(m.get("on_pareto_front") for m in recent)


def _on_pareto_front_now(entry: LedgerEntry, rules: PromotionRules) -> bool:
    if not rules.promote_by_pareto or not entry.metadata_history:
        return False
    return bool(entry.metadata_history[-1].get("on_pareto_front"))


def _reject_satisfied(entry: LedgerEntry, rules: PromotionRules) -> bool:
    if len(entry.composite_history) < rules.reject_after_n_cycles:
        return False
    # A current Pareto-front member is still competing in its niche — do not
    # reject it on low scalar composite (this supersedes veto_range_blind).
    if _on_pareto_front_now(entry, rules):
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
    scalar_ok = _promote_streak_satisfied(entry, rules)
    pareto_ok = _pareto_streak_satisfied(entry, rules)
    if scalar_ok or pareto_ok:
        evidence_failure = _promotion_evidence_failure(
            entry, rules, pareto_ok=pareto_ok
        )
        if evidence_failure is not None:
            return PromotionDecision(
                proposal_id=entry.proposal_id,
                decision=PROMOTION_PENDING,
                reason=f"streak met but {evidence_failure}",
                composite_history=tuple(entry.composite_history),
            )
    if (scalar_ok or pareto_ok) and not _paired_ci_satisfied(entry, rules):
        return PromotionDecision(
            proposal_id=entry.proposal_id,
            decision=PROMOTION_PENDING,
            reason=(
                "streak met but paired delta vs anchor is not significant "
                "(95% CI includes zero) — noise guard"
            ),
            composite_history=tuple(entry.composite_history),
        )
    if scalar_ok or pareto_ok:
        if scalar_ok:
            learned_clause = (
                " + learned signal" if rules.promote_require_learned_signal else ""
            )
            reason = (
                f"composite >= {rules.promote_min_composite} for "
                f"{rules.promote_min_streak_cycles} consecutive cycles "
                f"with smoke{learned_clause}"
            )
        else:
            reason = (
                f"on the first Pareto front for {rules.pareto_streak_cycles} "
                f"consecutive cycles (niche specialist)"
            )
        return PromotionDecision(
            proposal_id=entry.proposal_id,
            decision=PROMOTION_PROMOTED,
            reason=reason,
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
    """Apply decisions; return counts of THIS call's status transitions.

    ``promoted``/``rejected`` count only entries whose status changed now —
    not cumulative ledger totals. The quiescence halt in ``run_autonomous``
    reads these as "did anything move this cycle"; counting entries that
    already held the decided status (the old behavior) made every cycle look
    like movement, so ``--halt-quiescent`` could never fire. ``pending``
    counts entries still awaiting a terminal decision.
    """
    counts = {PROMOTION_PROMOTED: 0, PROMOTION_REJECTED: 0, PROMOTION_PENDING: 0}
    for decision in decisions:
        entry = ledger.entries.get(decision.proposal_id)
        if entry is None:
            continue
        if entry.promotion_status == decision.decision:
            if decision.decision == PROMOTION_PENDING:
                counts[PROMOTION_PENDING] += 1
            continue
        if decision.decision in (PROMOTION_PROMOTED, PROMOTION_REJECTED):
            ledger.record_promotion(decision.proposal_id, decision.decision)
            counts[decision.decision] += 1
        else:
            counts[PROMOTION_PENDING] += 1
    return counts
