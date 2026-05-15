"""CLI: run the property-tuple miner and print/persist results.

Usage:
    python -m component_fab.tools.run_property_miner [--db PATH] [--top-k N]
        [--max-candidates N] [--min-pass-rate F] [--out PATH]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from component_fab.proposer.property_miner import DEFAULT_META_DB, run

_REPO = Path(__file__).resolve().parents[2]
_CATALOG_DIR = _REPO / "component_fab" / "catalog"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="component_fab property-tuple miner")
    parser.add_argument("--db", default=str(DEFAULT_META_DB), type=str)
    parser.add_argument("--top-k", default=4, type=int)
    parser.add_argument("--max-candidates", default=50, type=int)
    parser.add_argument("--min-pass-rate", default=0.10, type=float)
    parser.add_argument("--min-axis-ops", default=2, type=int)
    parser.add_argument(
        "--out",
        default=None,
        type=str,
        help="JSON output path; defaults to component_fab/catalog/proposals_<ts>.json",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _resolve_out_path(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    _CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return _CATALOG_DIR / f"proposals_{timestamp}.json"


def _print_summary(report: dict) -> None:
    print(f"axes:              {', '.join(report['axes'])}")
    print(f"rows scanned:      {report['n_rows']}")
    print(f"extant tuples:     {report['n_extant_tuples']}")
    print(f"candidates found:  {report['n_candidates_returned']}")
    print()
    print("Top 10 unbuilt tuples by predicted lift:")
    print("-" * 78)
    for index, candidate in enumerate(report["candidates"][:10], start=1):
        tup = ", ".join(
            f"{cell['axis'].removeprefix('op_')}={cell['value']}"
            for cell in candidate["tuple"]
        )
        witnesses = ", ".join(filter(None, candidate["witness_ops"]))
        print(f"{index:>2}. lift={candidate['predicted_lift']:.3f}  {tup}")
        print(f"     witnesses: {witnesses}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    report = run(
        db_path=args.db,
        top_k_values_per_axis=args.top_k,
        max_candidates=args.max_candidates,
        min_axis_pass_rate=args.min_pass_rate,
        min_axis_n_ops=args.min_axis_ops,
    )
    out_path = _resolve_out_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.quiet:
        _print_summary(report)
        print()
        print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
