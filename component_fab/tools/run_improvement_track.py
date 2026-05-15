"""CLI: improvement track on goal-(b) underperforming-novel anchor ops.

For each anchor (default list = top 5 underperforming-novel ops surfaced
by the scoper), enumerate axis-variants, generate runnable modules,
grade via solo validator, and rank by intrinsic scorecard.

Usage:
    python -m component_fab.tools.run_improvement_track
    python -m component_fab.tools.run_improvement_track --anchors tropical_attention,padic_gate
    python -m component_fab.tools.run_improvement_track --dim 64
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from dataclasses import asdict
from pathlib import Path

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.improver.axis_variants import (
    DEFAULT_AXIS_VARIANT_TEMPLATES,
    enumerate_axis_variants,
)
from component_fab.intake.scope_existing import scope_all
from component_fab.validator.solo import append_scorecard, validate_solo

_REPO = Path(__file__).resolve().parents[2]
_CATALOG_DIR = _REPO / "component_fab" / "catalog"
_DEFAULT_TOP_N_ANCHORS = 5


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="component_fab improvement track")
    parser.add_argument("--db", default=None, type=str)
    parser.add_argument(
        "--anchors",
        default=None,
        type=str,
        help="comma-separated op_names; defaults to top-5 underperforming-novel",
    )
    parser.add_argument("--dim", default=32, type=int)
    parser.add_argument("--seq-len", default=32, type=int)
    parser.add_argument("--catalog-out", default=None, type=str)
    parser.add_argument("--summary-out", default=None, type=str)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _default_anchors(db_path: str | None) -> list[str]:
    report = scope_all() if db_path is None else scope_all(db_path=db_path)
    targets = report["underperforming_novel_ops"][:_DEFAULT_TOP_N_ANCHORS]
    return [t["name"] for t in targets]


def _resolve_summary_path(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    _CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return _CATALOG_DIR / f"improvement_track_{timestamp}.json"


def _print_summary(results: list[dict]) -> None:
    print(f"graded {len(results)} variants:")
    print("-" * 90)
    promoted = sum(1 for r in results if r["promoted"])
    print(f"  smoke+cross-check PASSED: {promoted}/{len(results)}")
    print()
    print(f"{'anchor → variant':<55} {'kind':<22} {'promoted'}")
    print("-" * 90)
    for r in results:
        if not r["promoted"]:
            continue
        print(f"{r['name']:<55} {r['synthesis_kind']:<22} {'YES'}")
    fails = [r for r in results if not r["promoted"]]
    if fails:
        print()
        print("Failed/rejected variants:")
        print("-" * 90)
        for r in fails:
            reason = r["smoke"].get("error") or "cross-check failed"
            print(f"  {r['name']:<55} {reason}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    anchors = (
        [a.strip() for a in args.anchors.split(",") if a.strip()]
        if args.anchors
        else _default_anchors(args.db)
    )
    if not anchors:
        print("no anchor ops found")
        return 1

    specs = enumerate_axis_variants(anchors)
    catalog_path = Path(args.catalog_out) if args.catalog_out else None
    results: list[dict] = []
    for spec in specs:
        module = generate_module_from_spec(spec, dim=args.dim)
        scorecard = validate_solo(spec, module, dim=args.dim, seq_len=args.seq_len)
        if catalog_path:
            append_scorecard(scorecard, catalog_path)
        else:
            append_scorecard(scorecard)
        results.append(asdict(scorecard))

    summary_path = _resolve_summary_path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "anchors": anchors,
                "n_variant_templates": len(DEFAULT_AXIS_VARIANT_TEMPLATES),
                "n_results": len(results),
                "results": results,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    if not args.quiet:
        _print_summary(results)
        print()
        print(f"wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
