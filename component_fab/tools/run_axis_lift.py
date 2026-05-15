"""CLI: replay the fab ledger and write per-(axis, value) shrunken lifts.

Usage:
    python -m component_fab.tools.run_axis_lift [--ledger PATH] [--out PATH]
        [--prior-strength F] [--min-n N] [--top-k N] [--quiet]

Output goes to ``component_fab/catalog/axis_lift.json`` by default. The
proposer reads that file (if present) to bias knob sampling toward axes
with empirical lift over the global promotion rate.
"""

from __future__ import annotations

import argparse
import sys

from component_fab.state.axis_lift import (
    DEFAULT_LEDGER_PATH,
    DEFAULT_OUTPUT_PATH,
    compute_axis_lift,
    write_axis_lift,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="component_fab axis-lift analyzer")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH), type=str)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH), type=str)
    parser.add_argument("--prior-strength", default=5.0, type=float)
    parser.add_argument("--min-n", default=2, type=int)
    parser.add_argument(
        "--top-k", default=10, type=int, help="rows printed per axis (full set in JSON)"
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _print_report(report, top_k: int) -> None:
    print(
        f"global: {report.global_promoted}/{report.global_total} promoted "
        f"({report.global_pass_rate:.3%}), prior={report.prior_strength}, "
        f"min_n={report.min_n}"
    )
    for axis in sorted(report.by_axis.keys()):
        rows = report.by_axis[axis]
        if not rows:
            continue
        print()
        print(f"=== axis: {axis} ({len(rows)} values) ===")
        for row in rows[:top_k]:
            print(
                f"  lift={row.lift:5.2f}  n={row.n:4d} k={row.k_promoted:3d}  "
                f"raw={row.pass_rate_raw:.3f} shrunk={row.pass_rate_shrunk:.3f}  "
                f"{row.value}"
            )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    report = compute_axis_lift(
        ledger_path=args.ledger,
        prior_strength=args.prior_strength,
        min_n=args.min_n,
    )
    out_path = write_axis_lift(report, output_path=args.out)
    if not args.quiet:
        _print_report(report, args.top_k)
        print()
        print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
