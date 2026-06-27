"""Promotion-side effects for autonomous fab cycles."""

from __future__ import annotations

from component_fab.proposer.dynamic import spec_from_ledger_entry
from component_fab.state.aria_registration import register_promotion
from component_fab.state.ledger import Ledger, PROMOTION_PROMOTED


def register_promoted(ledger: Ledger, decisions: list) -> None:
    """Emit an ARIA handoff row for each freshly promoted ledger entry."""

    for decision in decisions:
        if decision.decision != PROMOTION_PROMOTED or decision.reason == "already promoted":
            continue
        entry = ledger.entries.get(decision.proposal_id)
        if entry is None:
            continue
        spec = spec_from_ledger_entry(entry)
        if spec is None:
            continue
        meta = entry.metadata_history[-1] if entry.metadata_history else {}
        evidence = {
            "composite": entry.composite_history[-1] if entry.composite_history else 0.0,
            "transplant_portability": meta.get("transplant_portability"),
            "on_pareto_front": meta.get("on_pareto_front"),
        }
        register_promotion(spec, evidence=evidence)
