"""CLI: calibrate the fab validator gate stack against ledger outcomes (WS-1).

Replays ``catalog/ledger.jsonl`` and writes per-gate ROC/AUC + threshold
sweeps to ``catalog/gate_calibration.json``. Measures whether each gate's
recorded signal actually separates eventually-good candidates from bad —
the discriminative power ``failure_attribution.py`` does not check.

Usage:
    python -m component_fab.tools.run_gate_calibration [--ledger PATH] [--out PATH]
        [--db PATH] [--label {learned_signal,promoted}] [--min-class-n N]
        [--sweep-points N] [--survey] [--quiet]
"""

from __future__ import annotations

import argparse
import sys

from component_fab.state.gate_calibration import (
    DEFAULT_DB_PATH,
    DEFAULT_LEDGER_PATH,
    DEFAULT_OUTPUT_PATH,
    compute_gate_calibration,
    write_gate_calibration,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="component_fab gate calibration")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH), type=str)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH), type=str)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), type=str)
    parser.add_argument(
        "--label",
        default="learned_signal",
        choices=("learned_signal", "promoted"),
        help="Primary outcome label for AUC + threshold recommendation.",
    )
    parser.add_argument("--min-class-n", default=10, type=int)
    parser.add_argument("--sweep-points", default=12, type=int)
    parser.add_argument(
        "--survey",
        action="store_true",
        help=(
            "Opt back in to the op_property_catalog buildability survey (imports "
            "torch + the full generator stack to re-prove the settled "
            "corpus-degeneracy finding; off by default)."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _fmt_auc(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "  -  "


def _print_report(report) -> None:
    print(
        f"graded: {report.total_graded}  primary_label: {report.primary_label}  "
        f"(pos={report.labels[report.primary_label]['pos']} "
        f"neg={report.labels[report.primary_label]['neg']})"
    )
    print()
    print(f"Per-gate discriminative power (vs {report.primary_label}):")
    print(
        f"  {'gate':16s} {'signal':16s} {'auc_reach':>9s} {'auc_pass':>9s} "
        f"{'n_pass':>6s}  verdict"
    )
    for g in report.gate_aucs[report.primary_label]:
        flag = "  [circular]" if g.circular_inflation else ""
        print(
            f"  {g.gate:16s} {g.signal:16s} {_fmt_auc(g.auc_reached):>9s} "
            f"{_fmt_auc(g.auc_passed):>9s} {g.n_passed:>6d}  {g.verdict}{flag}"
        )
    if report.buildability is not None:
        b = report.buildability
        print()
        print(
            f"op_property_catalog buildability: {b.buildable}/{b.total_ops} "
            f"buildable, {b.fallback_linear} nn.Linear fallback, "
            f"{b.distinct_module_classes} distinct module classes"
        )
    print()
    print("Findings:")
    for f in report.findings:
        print(f"  - {f}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    report = compute_gate_calibration(
        ledger_path=args.ledger,
        db_path=args.db,
        primary_label=args.label,
        min_class_n=args.min_class_n,
        sweep_points=args.sweep_points,
        run_buildability_survey=args.survey,
    )
    out_path = write_gate_calibration(report, output_path=args.out)
    if not args.quiet:
        _print_report(report)
        print()
        print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
