"""CLI: replay the fab ledger and write per-gate kill rates + anchor pool.

Usage:
    python -m component_fab.tools.run_failure_attribution [--ledger PATH] [--out PATH]
        [--over-eager F] [--min-n N] [--anchor-min-composite F] [--anchor-min-erf F]
        [--anchor-pool-size N] [--quiet]

Output goes to ``component_fab/catalog/failure_attribution.json`` by default.
The proposer reads that file (if present) to fold the anchor pool back as
re-seeded candidates and to surface over-eager gates for calibration review.
"""

from __future__ import annotations

import argparse
import sys

from component_fab.state.failure_attribution import (
    DEFAULT_LEDGER_PATH,
    DEFAULT_OUTPUT_PATH,
    compute_failure_attribution,
    write_failure_attribution,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="component_fab failure attribution analyzer"
    )
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH), type=str)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH), type=str)
    parser.add_argument("--over-eager", default=0.85, type=float)
    parser.add_argument("--min-n", default=20, type=int)
    parser.add_argument("--anchor-min-composite", default=0.4, type=float)
    parser.add_argument("--anchor-min-erf", default=0.10, type=float)
    parser.add_argument("--anchor-pool-size", default=25, type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _print_report(report) -> None:
    print(
        f"graded: {report.total_graded}  promoted: {report.total_promoted}  "
        f"rejected: {report.total_rejected}  pending: {report.total_pending}"
    )
    print()
    print("Gate funnel (canonical order):")
    print(f"  {'gate':28s}  {'reached':>7s}  {'killed':>6s}  {'rate':>6s}  flag")
    for g in report.gate_stats:
        flag = "OVER-EAGER" if g.over_eager else ""
        print(
            f"  {g.gate:28s}  {g.reached:>7d}  {g.killed:>6d}  "
            f"{g.kill_rate:>6.1%}  {flag}"
        )
    if report.over_eager_gates:
        print()
        print(
            "Over-eager gates flagged for calibration review: "
            + ", ".join(report.over_eager_gates)
        )
    if report.anchor_pool:
        print()
        print(f"Anchor pool (top {len(report.anchor_pool)} rejected-but-promising):")
        for c in report.anchor_pool[:15]:
            erf = f"{c.erf_density:.3f}" if c.erf_density is not None else "  -  "
            nb = (
                f"{c.nb_max_accuracy:.3f}" if c.nb_max_accuracy is not None else "  -  "
            )
            print(
                f"  comp={c.composite_score:.3f}  erf={erf}  nb={nb}  "
                f"killed_by={c.eliminated_by:20s}  {c.name}"
            )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    report = compute_failure_attribution(
        ledger_path=args.ledger,
        over_eager_threshold=args.over_eager,
        min_n_for_over_eager=args.min_n,
        anchor_min_composite=args.anchor_min_composite,
        anchor_min_erf=args.anchor_min_erf,
        anchor_pool_size=args.anchor_pool_size,
    )
    out_path = write_failure_attribution(report, output_path=args.out)
    if not args.quiet:
        _print_report(report)
        print()
        print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
