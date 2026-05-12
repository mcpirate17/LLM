#!/usr/bin/env python
"""Summarize routing-decision choices against historical outcomes.

Usage:
    python -m research.tools.routing_decision_report --db research/runs.db
    python -m research.tools.routing_decision_report --template intelligent_multilane_router
    python -m research.tools.routing_decision_report --output research/reports/routing_decisions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.defaults import RUNS_DB
from research.meta_analysis.routing_decision_analytics import (
    iter_routing_decision_outcomes,
    summarize_routing_decisions,
)

DEFAULT_OUTCOMES: tuple[str, ...] = (
    "stage1_passed",
    "loss_ratio",
    "validation_loss_ratio",
    "ar_gate_score",
    "binding_intermediate_auc",
    "binding_screening_auc",
    "routing_utilization_entropy",
    "routing_drop_rate",
    "routing_savings_ratio",
    "routing_collapse_score",
)


def build_routing_decision_report(
    db_path: str | Path = RUNS_DB,
    *,
    template_filter: str | None = None,
    limit: int | None = None,
    min_support: int = 1,
) -> dict[str, Any]:
    rows = list(
        iter_routing_decision_outcomes(
            db_path,
            outcome_columns=DEFAULT_OUTCOMES,
            template_filter=template_filter,
            limit=limit,
        )
    )
    records = summarize_routing_decisions(
        rows,
        primary_outcome="stage1_passed",
        secondary_outcomes=DEFAULT_OUTCOMES[1:],
    )
    if min_support > 1:
        records = [row for row in records if int(row.get("n") or 0) >= min_support]
    records.sort(
        key=lambda row: (
            -int(row.get("n") or 0),
            row.get("template_name") or "",
            row.get("decision_key") or "",
        )
    )
    return {
        "db_path": str(db_path),
        "template_filter": template_filter,
        "input_rows": len(rows),
        "decision_groups": len(records),
        "min_support": int(min_support),
        "records": records,
    }


def _format_markdown(report: dict[str, Any], *, max_rows: int) -> str:
    lines = [
        "# Routing Decision Report",
        "",
        f"- DB: `{report['db_path']}`",
        f"- Template filter: `{report.get('template_filter') or 'all'}`",
        f"- Input rows: {report['input_rows']}",
        f"- Decision groups: {report['decision_groups']}",
        "",
        "| template | decision | value | n | pass_rate | loss | ar_gate | binding |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["records"][:max_rows]:
        lines.append(
            "| {template} | {decision} | `{value}` | {n} | {pass_rate:.3f} | "
            "{loss} | {ar} | {binding} |".format(
                template=row.get("template_name") or "",
                decision=row.get("decision_key") or "",
                value=json.dumps(row.get("value"), sort_keys=True),
                n=int(row.get("n") or 0),
                pass_rate=float(row.get("pass_rate") or 0.0),
                loss=_fmt(row.get("mean_loss_ratio")),
                ar=_fmt(row.get("mean_ar_gate_score")),
                binding=_fmt(row.get("mean_binding_intermediate_auc")),
            )
        )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(RUNS_DB), help="runs/lab notebook DB path")
    parser.add_argument("--template", default=None, help="optional template filter")
    parser.add_argument(
        "--limit", type=int, default=None, help="optional source-row cap"
    )
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=80)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--format", choices=("json", "md"), default="json")
    args = parser.parse_args(argv)

    report = build_routing_decision_report(
        args.db,
        template_filter=args.template,
        limit=args.limit,
        min_support=max(1, args.min_support),
    )
    payload = (
        _format_markdown(report, max_rows=max(1, args.max_rows))
        if args.format == "md"
        else json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
