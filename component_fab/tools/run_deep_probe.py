"""Deep-probe tier CLI: train the top-K nano survivors to depth vs the frontier.

The autonomous loop grades at nano scale (dim=32, short), where the binding and
learning subscores saturate toward the floor — so its composite cannot tell a
genuine frontier-beater from a plausible non-learner, and a 200-step Tier-2
cohort ties baselines. This command takes the ledger's top-K candidates by
*relative* nano-composite rank and trains each for many more steps against the
real GPT-2 / Mamba / Mamba2 baselines, reporting which actually beat frontier.

It is opt-in and CPU-only (the Tier-2 micro-models are ~6K params), so it never
disturbs a live GPU training run. ``--promote`` writes promotion decisions back
to the ledger; without it the run is a dry-run report.

Example::

    python -m component_fab.tools.run_deep_probe --top-k 12 --steps 3000 \
        --statuses promoted --promote
"""

from __future__ import annotations

import argparse
from pathlib import Path

from component_fab.improver.deep_probe import run_deep_probe
from component_fab.tools._cli import open_ledger, write_report
from component_fab.state.ledger import (
    DEFAULT_LEDGER_PATH,
    PROMOTION_PENDING,
    PROMOTION_PROMOTED,
    PROMOTION_REJECTED,
)

_REPO = Path(__file__).resolve().parents[2]
_REPORT_DIR = _REPO / "research" / "reports"

_STATUS_PRESETS: dict[str, frozenset[str] | None] = {
    "any": None,
    "promoted": frozenset({PROMOTION_PROMOTED}),
    "pending": frozenset({PROMOTION_PENDING}),
    "promoted+pending": frozenset({PROMOTION_PROMOTED, PROMOTION_PENDING}),
    "rejected": frozenset({PROMOTION_REJECTED}),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="training steps per task (nano grading uses far fewer; binding "
        "only separates after thousands)",
    )
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument(
        "--window",
        type=int,
        default=2,
        help="cycles of composite history averaged for the relative rank",
    )
    parser.add_argument(
        "--statuses",
        choices=sorted(_STATUS_PRESETS),
        default="promoted+pending",
        help="which ledger promotion_status values are eligible for selection",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="record frontier-beaters as promoted in the ledger (default: dry run)",
    )
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="report JSON path (default: research/reports/deep_probe_<ts>.json)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def _print_summary(report: dict) -> None:
    print(
        f"\ndeep-probe: selected {report['n_selected']} candidates, "
        f"{report['n_beats_frontier']} beat frontier "
        f"({report['n_promoted']} promoted) "
        f"@ {report['n_train_steps']} steps, dim={report['dim']}"
    )
    print(f"baselines: {', '.join(report['baseline_names'])}")
    print("results (sorted by beats-frontier, mean Δ):")
    for outcome in report["outcomes"]:
        flag = "BEATS" if outcome["beats_frontier"] else "     "
        if outcome["status"] != "ok":
            flag = outcome["status"][:5].upper()
        print(
            f"  [{flag}] Δ={outcome['mean_delta_vs_frontier']:+.4f} "
            f"pass={outcome['pass_count']}/{outcome['n_tasks']}  "
            f"{outcome['name'][:64]}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ledger = open_ledger(args)

    report = run_deep_probe(
        ledger,
        top_k=args.top_k,
        n_train_steps=args.steps,
        dim=args.dim,
        n_blocks=args.n_blocks,
        seed=args.seed,
        seed_count=args.seed_count,
        window=args.window,
        statuses=_STATUS_PRESETS[args.statuses],
        promote=args.promote,
        quiet=args.quiet,
    )

    out_path = write_report(
        report,
        default_dir=_REPORT_DIR,
        prefix="deep_probe",
        output=args.output,
        quiet=True,  # summary block below prints the path in its own format
    )

    _print_summary(report)
    print(f"\n[report → {out_path}]")
    if args.promote and report["n_promoted"]:
        print(f"[ledger updated: {report['n_promoted']} promotions recorded]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
