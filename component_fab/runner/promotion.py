"""Promotion-side effects for autonomous fab cycles."""

from __future__ import annotations

from component_fab.policies.promotion import (
    PromotionDecision,
    PromotionRules,
    apply_decisions,
    decide_promotions_for_ledger,
)
from component_fab.proposer.dynamic import spec_from_ledger_entry
from component_fab.state.aria_registration import register_promotion
from component_fab.state.ledger import (
    Ledger,
    PROMOTION_PROMOTED,
    PROMOTION_REJECTED,
)
from component_fab.validator.paired import paired_metadata_for_spec


def register_promoted(ledger: Ledger, decisions: list) -> None:
    """Emit an ARIA handoff row for each freshly promoted ledger entry."""

    for decision in decisions:
        if (
            decision.decision != PROMOTION_PROMOTED
            or decision.reason == "already promoted"
        ):
            continue
        entry = ledger.entries.get(decision.proposal_id)
        if entry is None:
            continue
        spec = spec_from_ledger_entry(entry)
        if spec is None:
            continue
        meta = entry.metadata_history[-1] if entry.metadata_history else {}
        evidence = {
            "composite": entry.composite_history[-1]
            if entry.composite_history
            else 0.0,
            "transplant_portability": meta.get("transplant_portability"),
            "on_pareto_front": meta.get("on_pareto_front"),
        }
        register_promotion(spec, evidence=evidence)


def scale_gate_promotions(
    ledger: Ledger,
    decisions: list,
    *,
    dim: int,
    steps: int,
    seeds: int,
    seq_len: int,
) -> list:
    """Final gate before promotion: re-verify each FRESH promotion beats its
    anchor at SCALE.

    The nano paired-CI is not scale-predictive — at dim32/100 steps inventions
    showed tiny positive margins that INVERT to large losses at dim96/1500
    (validated 2026-06-16). So before a candidate promotes, re-run the paired
    probe (vs its catalog anchor, or the softmax-frontier fallback) at a larger
    width + many more steps. A candidate that does not beat its anchor at scale
    is REJECTED — terminal, so it is not re-tested (and re-promoted) every cycle —
    rather than minted as a scale-losing artifact.
    """

    gated: list = []
    for decision in decisions:
        entry = ledger.entries.get(decision.proposal_id)
        # Only re-verify FRESH promotions (not already-promoted, not pending/reject).
        if (
            decision.decision != PROMOTION_PROMOTED
            or entry is None
            or entry.promotion_status == PROMOTION_PROMOTED
        ):
            gated.append(decision)
            continue
        spec = spec_from_ledger_entry(entry)
        if spec is None:
            gated.append(decision)
            continue
        md = paired_metadata_for_spec(
            spec, seeds=tuple(range(seeds)), dim=dim, seq_len=seq_len, n_steps=steps
        )
        beats = bool(md.get("paired_delta_ci_excludes_zero"))
        anchor = md.get("paired_anchor_op", "?")
        ci_low = md.get("paired_delta_ci_low")
        print(
            f"  scale-gate {decision.proposal_id[:24]} vs {anchor} "
            f"@dim{dim}/{steps}st: {'PASS' if beats else 'FAIL'} ci_low={ci_low}"
        )
        if beats:
            gated.append(decision)
        else:
            gated.append(
                PromotionDecision(
                    proposal_id=decision.proposal_id,
                    decision=PROMOTION_REJECTED,
                    reason=(
                        f"scale-gate: loses to {anchor} at dim{dim}/{steps}st "
                        f"(ci_low={ci_low})"
                    ),
                    composite_history=decision.composite_history,
                )
            )
    return gated


def resolve_promotions(
    ledger: Ledger,
    promotion_rules: PromotionRules,
    *,
    scale_gate: bool,
    scale_gate_dim: int,
    scale_gate_steps: int,
    scale_gate_seeds: int,
    seq_len: int,
) -> dict[str, int]:
    """Decide promotions, optionally scale-gate the fresh ones, apply + register."""

    decisions = decide_promotions_for_ledger(ledger, promotion_rules)
    if scale_gate:
        decisions = scale_gate_promotions(
            ledger,
            decisions,
            dim=scale_gate_dim,
            steps=scale_gate_steps,
            seeds=scale_gate_seeds,
            seq_len=seq_len,
        )
    counts = apply_decisions(ledger, decisions)
    register_promoted(ledger, decisions)
    return counts
