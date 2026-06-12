"""CLI: fit the ledger surrogate and write its K-fold report (WS-3).

Replays ``catalog/ledger.jsonl`` and writes ``catalog/surrogate_report.json`` with
the out-of-fold composite ranking quality, the promotion AUC head-to-head vs the
marginal/independence baseline (the ``predicted_lift`` analogue), and the WS-3
acceptance verdict. When acceptance passes, ``run_autonomous --selection surrogate``
is safe to flip on.

Usage:
    python -m component_fab.tools.run_surrogate [--ledger PATH] [--out PATH]
        [--folds N] [--quiet]
"""

from __future__ import annotations

import argparse
import sys

from component_fab.state.surrogate import (
    DEFAULT_LEDGER_PATH,
    DEFAULT_OUTPUT_PATH,
    compute_surrogate_report,
    write_surrogate_report,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="component_fab ledger surrogate")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH), type=str)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH), type=str)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    report = compute_surrogate_report(ledger_path=args.ledger, n_folds=args.folds)
    out_path = write_surrogate_report(report, output_path=args.out)
    if not args.quiet:
        print(
            f"rows={report.n_rows} promoted={report.n_promoted} "
            f"features={report.n_features}"
        )
        print(
            f"promotion AUC: surrogate={report.promoted_auc_surrogate} "
            f"marginal={report.promoted_auc_marginal} | "
            f"composite Spearman={report.composite_spearman_oof}"
        )
        print(f"acceptance_passed={report.acceptance_passed}")
        for finding in report.findings:
            print(f"  - {finding}")
        print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
