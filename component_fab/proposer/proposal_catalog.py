"""Load persisted ProposalSpecs from the proposals.jsonl catalog.

Moved from ``research/tools/run_tier2_binding_cohort`` (which keeps a
back-compat alias) so component_fab tools stop reaching across packages
for a private helper.
"""

from __future__ import annotations

from pathlib import Path

from component_fab.proposer.spec_generator import ProposalSpec
from component_fab.state.ledger import iter_jsonl_records, iter_rotated_jsonl_paths
from component_fab.validator.solo import DEFAULT_CATALOG


def load_proposals_by_id(path: Path = DEFAULT_CATALOG) -> dict[str, ProposalSpec]:
    """Build a {proposal_id: ProposalSpec} map from the catalog jsonl.

    Also scans rotated ``proposals.jsonl.N`` files since the autonomous
    loop rotates at 2 MB and promoted specs may live in older rotations.
    Last-wins when the same proposal_id appears in multiple files.
    """
    out: dict[str, ProposalSpec] = {}
    for p in iter_rotated_jsonl_paths(path):
        for row in iter_jsonl_records(p):
            pid = row.get("proposal_id")
            if not pid:
                continue
            out[str(pid)] = ProposalSpec(
                proposal_id=str(pid),
                name=str(row.get("name") or ""),
                category=str(row.get("category") or ""),
                synthesis_kind=str(row.get("synthesis_kind") or ""),
                math_axes=dict(row.get("math_axes") or {}),
                anchor_witness_op=str(row.get("anchor_witness_op") or ""),
                anchor_witnesses_all=tuple(row.get("anchor_witnesses_all") or ()),
                declared_property_row=dict(row.get("declared_property_row") or {}),
                predicted_lift=float(row.get("predicted_lift") or 0.0),
                rationale=str(row.get("rationale") or ""),
                notes=tuple(row.get("notes") or ()),
            )
    return out
