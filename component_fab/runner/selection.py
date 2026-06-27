"""Candidate filtering, screening, and ordering for autonomous fab cycles."""

from __future__ import annotations

from component_fab.proposer.acquisition import select_by_acquisition
from component_fab.proposer.nas_screen import NasScreenResult, score_specs_with_nas
from component_fab.proposer.quality import (
    allocate_budget_buckets,
    bucket_counts,
    score_specs_quality,
)
from component_fab.proposer.spec_generator import ProposalSpec
from component_fab.proposer.tier2_feedback import Tier2Feedback
from component_fab.state.ledger import Ledger, PROMOTION_PROMOTED, PROMOTION_REJECTED
from component_fab.state.surrogate import Surrogate
from component_fab.validator.trust import axes_counts_for_specs


def order_active_specs_by_quality(
    active_specs: list[ProposalSpec],
    ledger: Ledger,
    *,
    tier2_feedback_by_id: dict[str, Tier2Feedback],
    nas_screen_by_id: dict[str, NasScreenResult],
    max_graded_per_cycle: int = 0,
) -> tuple[list[ProposalSpec], dict[str, int]]:
    """Rank active specs by fused quality, applying the budget split if capped."""

    if not active_specs:
        return active_specs, bucket_counts(())
    quality_by_id = score_specs_quality(
        active_specs,
        tier2_by_id=tier2_feedback_by_id,
        nas_by_id=nas_screen_by_id,
        entries_by_id=ledger.entries,
        axes_counts=axes_counts_for_specs(active_specs),
    )
    scores = list(quality_by_id.values())
    if max_graded_per_cycle > 0:
        chosen = allocate_budget_buckets(scores, total=max_graded_per_cycle)
    else:
        chosen = sorted(scores, key=lambda s: s.quality_score, reverse=True)
    chosen_ids = [s.proposal_id for s in chosen]
    spec_by_id = {s.proposal_id: s for s in active_specs}
    ordered = [spec_by_id[pid] for pid in chosen_ids if pid in spec_by_id]
    return ordered, bucket_counts(chosen)


def select_active_specs(
    specs: list[ProposalSpec],
    ledger: Ledger,
    *,
    selection: str,
    acquisition_beta: float,
    use_nas_screen: bool,
    use_quality_order: bool,
    max_graded_per_cycle: int,
    tier2_feedback_by_id: dict[str, Tier2Feedback],
) -> tuple[list[ProposalSpec], dict[str, NasScreenResult], dict[str, int], int, int]:
    """Filter terminal specs, screen candidates, and order the grading queue.

    Returns ``(active_specs, nas_screen_by_id, bucket_summary,
    n_new_proposals, n_terminal_skipped)``.
    """

    skippable = {
        pid
        for pid, entry in ledger.entries.items()
        if entry.promotion_status in (PROMOTION_PROMOTED, PROMOTION_REJECTED)
    }
    active_specs = [s for s in specs if s.proposal_id not in skippable]
    nas_screen_by_id = score_specs_with_nas(active_specs, enabled=use_nas_screen)
    n_new_proposals = sum(1 for s in active_specs if not ledger.has_seen(s.proposal_id))

    bucket_summary = bucket_counts(())
    if selection == "surrogate":
        active_specs = select_by_acquisition(
            active_specs,
            Surrogate.fit(),
            budget=max_graded_per_cycle,
            beta=acquisition_beta,
        )
    elif use_quality_order:
        active_specs, bucket_summary = order_active_specs_by_quality(
            active_specs,
            ledger,
            tier2_feedback_by_id=tier2_feedback_by_id,
            nas_screen_by_id=nas_screen_by_id,
            max_graded_per_cycle=max_graded_per_cycle,
        )
    return (
        active_specs,
        nas_screen_by_id,
        bucket_summary,
        n_new_proposals,
        len(skippable),
    )
