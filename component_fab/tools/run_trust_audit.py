"""Audit fab candidates against downstream evidence tiers.

This command does not train models. It reads existing ledger/cohort artifacts and
classifies candidates as screened, promising, trusted, or rejected.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any

from component_fab.state.ledger import (
    DEFAULT_LEDGER_PATH,
    PROMOTION_PROMOTED,
    Ledger,
)
from component_fab.validator.trust import TrustThresholds, build_trust_report
from research.tools.run_tier2_binding_cohort import _load_proposals_by_id

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
        needle = str(args.proposal_id)
        if needle in ledger.entries:
            return [needle]
        matches = [pid for pid in ledger.entries if pid.startswith(needle)]
        if len(matches) == 1:
            return matches
        if len(matches) > 1:
            print(
                f"proposal id prefix {needle!r} is ambiguous ({len(matches)} matches)",
                file=sys.stderr,
            )
            return []
        print(f"proposal id {needle!r} not found in ledger", file=sys.stderr)
        return []
    return _resolve_promoted(ledger, int(args.top_promoted))


def _default_output_path() -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return _AUDIT_DIR / f"fab_trust_audit_{stamp}.json"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="component_fab trust audit")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--proposal-id")
    target.add_argument("--top-promoted", type=int)
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--tier2", help="Tier-2 cohort JSON artifact")
    parser.add_argument("--blimp", help="BLiMP cohort JSON artifact")
    parser.add_argument("--saved-winners", default=str(_DEFAULT_SAVED_WINNERS))
    parser.add_argument("--min-seed-count", type=int, default=2)
    parser.add_argument("--min-blimp-delta", type=float, default=0.005)
    parser.add_argument("--min-tier2-mean-delta", type=float, default=0.0)
    parser.add_argument("--max-wikitext-ppl-regression", type=float, default=0.10)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print JSON to stdout and do not write an artifact",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    ledger = Ledger(args.ledger, include_rotated=True)
    proposal_ids = _resolve_targets(args, ledger)
    if not proposal_ids:
        print("no proposals to audit", file=sys.stderr)
        return 2
    proposals = _load_proposals_by_id()
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
    payload = json.dumps(report, indent=2, default=str)
    if args.dry_run:
        print(payload)
        return 0
    out = args.output or _default_output_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")
    print(f"wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
