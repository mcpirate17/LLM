"""CLI: scope existing primitives + templates into a fab inventory file.

Usage:
    python -m component_fab.tools.run_scope_existing [--db PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from component_fab.intake.scope_existing import DEFAULT_META_DB, scope_all

_REPO = Path(__file__).resolve().parents[2]
_CATALOG_DIR = _REPO / "component_fab" / "catalog"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="component_fab existing-component scoper"
    )
    parser.add_argument("--db", default=str(DEFAULT_META_DB), type=str)
    parser.add_argument("--out", default=None, type=str)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _resolve_out_path(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    _CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return _CATALOG_DIR / f"existing_components_{timestamp}.json"


def _print_summary(report: dict) -> None:
    totals = report["totals"]
    print(f"source:         {report['source_db']}")
    print(f"ops:            {totals['ops']}")
    print(f"templates:      {totals['templates']}")
    print("by category:")
    for category, count in totals["by_category"].items():
        print(f"  {category:<15} {count}")
    print()
    print("Top 15 multilane routing templates by observed_count:")
    print("-" * 78)
    for index, record in enumerate(report["multilane_routing_templates"][:15], start=1):
        decl = record["declared_properties"]
        obs = record["performance"]["observed_count"]
        print(
            f"{index:>2}. {record['name']:<40} "
            f"paths={decl.get('template_est_parallel_paths')} "
            f"slots={decl.get('slot_count')} "
            f"obs={obs}"
        )
    print()
    print("Top 10 novel-axis underperforming ops (goal-b targets):")
    print("-" * 78)
    for index, record in enumerate(report["underperforming_novel_ops"][:10], start=1):
        decl = record["declared_properties"]
        perf = record["performance"]
        print(
            f"{index:>2}. {record['name']:<28} "
            f"space={decl.get('op_algebraic_space'):<10} "
            f"evals={perf['eval_count']:>4} "
            f"pass={perf['pass_rate']:.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    report = scope_all(db_path=args.db)
    out_path = _resolve_out_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if not args.quiet:
        _print_summary(report)
        print()
        print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
