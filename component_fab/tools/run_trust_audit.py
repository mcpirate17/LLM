"""Audit fab candidates against downstream evidence tiers.

This command does not train models. It reads existing ledger/cohort artifacts and
classifies candidates as screened, promising, trusted, or rejected.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from component_fab.proposer.proposal_catalog import load_proposals_by_id
from component_fab.state.ledger import (
    PROMOTION_PROMOTED,
    Ledger,
    resolve_proposal_id,
)
from component_fab.tools._cli import add_common_args, open_ledger, write_report
from component_fab.validator.trust import TrustThresholds, build_trust_report

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_SAVED_WINNERS = _REPO / "component_fab" / "catalog" / "saved_winners.json"
_AUDIT_DIR = _REPO / "tasks" / "audit"


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"artifact not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_promoted(ledger: Ledger, limit: int) -> list[str]:
    rows: list[tuple[float, str]] = []
    for entry in ledger.all_entries():
        if entry.promotion_status != PROMOTION_PROMOTED:
            continue
        score = max(entry.composite_history or [0.0])
        rows.append((float(score), entry.proposal_id))
    rows.sort(reverse=True)
    return [proposal_id for _, proposal_id in rows[:limit]]


def _resolve_targets(args: argparse.Namespace, ledger: Ledger) -> list[str]:
    if args.proposal_id:
        try:
            return [resolve_proposal_id(ledger, str(args.proposal_id)).proposal_id]
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return []
    return _resolve_promoted(ledger, int(args.top_promoted))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="component_fab trust audit")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--proposal-id")
    target.add_argument("--top-promoted", type=int)
    parser.add_argument("--tier2", help="Tier-2 cohort JSON artifact")
    parser.add_argument("--blimp", help="BLiMP cohort JSON artifact")
    parser.add_argument("--saved-winners", default=str(_DEFAULT_SAVED_WINNERS))
    parser.add_argument("--min-seed-count", type=int, default=2)
    parser.add_argument("--min-blimp-delta", type=float, default=0.005)
    parser.add_argument("--min-tier2-mean-delta", type=float, default=0.0)
    parser.add_argument("--max-wikitext-ppl-regression", type=float, default=0.10)
    add_common_args(parser, dry_run=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    ledger = open_ledger(args)
    proposal_ids = _resolve_targets(args, ledger)
    if not proposal_ids:
        print("no proposals to audit", file=sys.stderr)
        return 2
    proposals = load_proposals_by_id()
    thresholds = TrustThresholds(
        min_seed_count=max(1, int(args.min_seed_count)),
        min_blimp_delta=float(args.min_blimp_delta),
        min_tier2_mean_delta=float(args.min_tier2_mean_delta),
        max_wikitext_ppl_regression=float(args.max_wikitext_ppl_regression),
    )
    saved_winners = _load_json(args.saved_winners)
    report = build_trust_report(
        proposal_ids,
        ledger=ledger,
        proposals_by_id=proposals,
        tier2_summary=_load_json(args.tier2),
        blimp_summary=_load_json(args.blimp),
        saved_winners=saved_winners,
        thresholds=thresholds,
    )
    report["proposal_ids"] = proposal_ids
    write_report(
        report,
        default_dir=_AUDIT_DIR,
        prefix="fab_trust_audit",
        output=args.output,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
