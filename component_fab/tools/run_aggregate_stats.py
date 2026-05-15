"""CLI: aggregate stats across all autonomous runs + the live ledger.

Walks ``component_fab/catalog/`` for every ``autonomous_run_*.json`` file
and the active ``ledger.jsonl`` + rotated ``ledger.jsonl.N`` files, then
emits:

- total cycles run across all sessions
- total promotions / rejections (lifetime)
- all-time best leaderboard (top-N by max composite_score)
- promotion streaks: longest-running promoted components

Usage:
    python -m component_fab.tools.run_aggregate_stats
    python -m component_fab.tools.run_aggregate_stats --top 20 --out trend.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from component_fab.state.ledger import Ledger

_REPO = Path(__file__).resolve().parents[2]
_CATALOG_DIR = _REPO / "component_fab" / "catalog"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="component_fab aggregate stats")
    parser.add_argument("--catalog-dir", default=str(_CATALOG_DIR), type=str)
    parser.add_argument("--top", default=15, type=int)
    parser.add_argument("--out", default=None, type=str)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _load_run_files(catalog_dir: Path) -> list[dict]:
    out: list[dict] = []
    for path in sorted(catalog_dir.glob("autonomous_run_*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _replay_full_ledger_history(catalog_dir: Path) -> Ledger:
    """Build a ledger from the active log + all rotated logs in chronological order."""
    ledger_path = catalog_dir / "ledger.jsonl"
    ledger = Ledger(ledger_path)
    # Also replay any rotated logs that may not be in the active file's path.
    for rotated in sorted(catalog_dir.glob("ledger.jsonl.*")):
        with rotated.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ledger._apply_record(record)  # noqa: SLF001
    return ledger


def _all_time_best(ledger: Ledger, top: int) -> list[dict]:
    entries = sorted(
        ledger.all_entries(),
        key=lambda e: max(e.composite_history) if e.composite_history else 0.0,
        reverse=True,
    )[:top]
    return [
        {
            "name": entry.name,
            "category": entry.category,
            "synthesis_kind": entry.synthesis_kind,
            "max_composite": round(
                max(entry.composite_history) if entry.composite_history else 0.0, 4
            ),
            "mean_composite": round(
                sum(entry.composite_history) / max(1, len(entry.composite_history)), 4
            ),
            "n_cycles": len(entry.composite_history),
            "smoke_pass_count": entry.smoke_pass_count,
            "learned_signal_count": entry.learned_signal_count,
            "promotion_status": entry.promotion_status,
        }
        for entry in entries
    ]


def _trends(runs: list[dict], ledger: Ledger) -> dict:
    total_cycles = sum(r.get("cycles_run", 0) for r in runs)
    total_promoted = sum(
        1 for e in ledger.all_entries() if e.promotion_status == "promoted"
    )
    total_rejected = sum(
        1 for e in ledger.all_entries() if e.promotion_status == "rejected"
    )
    total_pending = sum(
        1 for e in ledger.all_entries() if e.promotion_status == "pending"
    )
    by_category: dict[str, int] = defaultdict(int)
    for entry in ledger.all_entries():
        if entry.promotion_status == "promoted":
            by_category[entry.category] += 1
    by_kind: dict[str, int] = defaultdict(int)
    for entry in ledger.all_entries():
        if entry.promotion_status == "promoted":
            by_kind[entry.synthesis_kind] += 1
    return {
        "n_runs": len(runs),
        "total_cycles_lifetime": total_cycles,
        "total_proposals_tracked": len(ledger.entries),
        "lifetime_promoted": total_promoted,
        "lifetime_rejected": total_rejected,
        "lifetime_pending": total_pending,
        "promoted_by_category": dict(by_category),
        "promoted_by_synthesis_kind": dict(by_kind),
    }


def _print_report(trends: dict, best: list[dict]) -> None:
    print("=== fab lifetime stats ===")
    for key, value in trends.items():
        print(f"  {key:<32} {value}")
    print()
    print(f"=== all-time best (top {len(best)}) ===")
    print(f"{'rank':<5} {'max':<7} {'mean':<7} {'cycles':<7} {'status':<10} {'name'}")
    print("-" * 100)
    for index, entry in enumerate(best, start=1):
        print(
            f"{index:<5} {entry['max_composite']:<7.3f} "
            f"{entry['mean_composite']:<7.3f} {entry['n_cycles']:<7} "
            f"{entry['promotion_status']:<10} {entry['name']}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    catalog_dir = Path(args.catalog_dir)
    runs = _load_run_files(catalog_dir)
    ledger = _replay_full_ledger_history(catalog_dir)
    trends = _trends(runs, ledger)
    best = _all_time_best(ledger, args.top)

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {"trends": trends, "all_time_best": best}, indent=2, default=str
            ),
            encoding="utf-8",
        )
    if not args.quiet:
        _print_report(trends, best)
        if args.out:
            print()
            print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
