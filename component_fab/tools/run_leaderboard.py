"""CLI: full sprint-4 pipeline — improve, solo, in-context, leaderboard.

For each goal-(b) anchor op, enumerate axis-variants, generate the
runnable lane via code_generator, run solo + in-context probes, and
emit a composite-scored leaderboard. Writes all scorecards to JSONL and
prints the ranked table.

Usage:
    python -m component_fab.tools.run_leaderboard
    python -m component_fab.tools.run_leaderboard --dim 32 --probe-steps 60
    python -m component_fab.tools.run_leaderboard --anchors tropical_attention,clifford_attention
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from dataclasses import asdict
from pathlib import Path

from component_fab.generator.code_generator import generate_module_from_spec
from component_fab.improver.axis_variants import enumerate_axis_variants
from component_fab.improver.ranking import leaderboard_to_json, rank_proposals
from component_fab.intake.scope_existing import scope_all
from component_fab.validator.in_context import validate_in_context
from component_fab.validator.solo import append_scorecard, validate_solo

_REPO = Path(__file__).resolve().parents[2]
_CATALOG_DIR = _REPO / "component_fab" / "catalog"
_DEFAULT_TOP_N_ANCHORS = 5


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="component_fab sprint-4 leaderboard")
    parser.add_argument("--anchors", default=None, type=str)
    parser.add_argument("--dim", default=32, type=int)
    parser.add_argument("--seq-len", default=32, type=int)
    parser.add_argument("--probe-steps", default=80, type=int)
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="solo-only ranking, no in-context training",
    )
    parser.add_argument("--out", default=None, type=str)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _default_anchors() -> list[str]:
    report = scope_all()
    targets = report["underperforming_novel_ops"][:_DEFAULT_TOP_N_ANCHORS]
    return [t["name"] for t in targets]


def _resolve_out_path(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    _CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return _CATALOG_DIR / f"leaderboard_{timestamp}.json"


def _print_leaderboard(rows: list[dict]) -> None:
    print(f"{'rank':<5} {'score':<7} {'smoke':<7} {'cross':<7} {'learn':<7} {'name'}")
    print("-" * 110)
    for row in rows:
        c = row["components"]
        print(
            f"{row['rank']:<5} {row['composite_score']:<7.3f} "
            f"{c['smoke']:<7.2f} {c['cross_check']:<7.2f} {c['learning']:<7.2f} "
            f"{row['name']}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    anchors = (
        [a.strip() for a in args.anchors.split(",") if a.strip()]
        if args.anchors
        else _default_anchors()
    )
    if not anchors:
        print("no anchor ops found")
        return 1

    specs = enumerate_axis_variants(anchors)
    solo_scorecards: list[dict] = []
    probe_scorecards_by_id: dict[str, dict] = {}
    for spec in specs:
        module = generate_module_from_spec(spec, dim=args.dim)
        solo_card = validate_solo(spec, module, dim=args.dim, seq_len=args.seq_len)
        append_scorecard(solo_card)
        solo_scorecards.append(asdict(solo_card))
        if not args.skip_probe and solo_card.promoted:
            probe_card = validate_in_context(
                spec,
                module,
                dim=args.dim,
                seq_len=args.seq_len,
                n_steps=args.probe_steps,
            )
            probe_scorecards_by_id[spec.proposal_id] = asdict(probe_card)

    ranked = rank_proposals(solo_scorecards, probe_scorecards_by_id)
    leaderboard = leaderboard_to_json(ranked)

    out_path = _resolve_out_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "anchors": anchors,
                "n_proposals": len(solo_scorecards),
                "n_probed": len(probe_scorecards_by_id),
                "leaderboard": leaderboard,
                "probe_scorecards": probe_scorecards_by_id,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    if not args.quiet:
        _print_leaderboard(leaderboard)
        print()
        print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
