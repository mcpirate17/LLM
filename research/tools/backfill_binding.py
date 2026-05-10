#!/usr/bin/env python3
"""Legacy compatibility wrapper for the unified backfill runner."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Sequence

from research.tools._legacy_backfill_cli import (
    add_common_backfill_args,
    run_legacy_backfill,
)

if TYPE_CHECKING:
    from research.scientist.notebook import LabNotebook

_VALID_METRICS = ("binding", "induction", "ar")


def _parse_metrics(raw: str) -> tuple[str, ...]:
    metrics = tuple(metric.strip() for metric in raw.split(",") if metric.strip())
    if not metrics:
        raise ValueError("At least one metric must be requested")
    invalid = [metric for metric in metrics if metric not in _VALID_METRICS]
    if invalid:
        raise ValueError(
            f"Unsupported metrics: {invalid}. Valid metrics: {list(_VALID_METRICS)}"
        )
    probe_order = []
    if "binding" in metrics or "induction" in metrics or "ar" in metrics:
        probe_order.append("binding")
    return tuple(probe_order)


def _query_candidates(
    nb: "LabNotebook",
    tiers: Sequence[str],
    *,
    top: int,
    force: bool,
    metrics: Sequence[str] = ("binding",),
):
    tier_params = tuple(str(tier) for tier in tiers)
    tier_placeholders = ",".join("?" for _ in tier_params)
    missing = ""
    if not force:
        missing_fields = []
        if "binding" in metrics:
            missing_fields.append("pr.binding_screening_auc IS NULL")
        if "induction" in metrics:
            missing_fields.append("pr.induction_screening_auc IS NULL")
        if "ar" in metrics:
            missing_fields.append("pr.ar_legacy_auc IS NULL")
        if missing_fields:
            missing = " AND (" + " OR ".join(missing_fields) + ")"
    rows = [
        dict(row)
        for row in nb.conn.execute(
            f"""
            SELECT
                l.entry_id,
                l.result_id,
                l.tier,
                l.composite_score,
                l.is_reference,
                l.model_source,
                pr.graph_json,
                pr.graph_fingerprint
            FROM leaderboard l
            JOIN program_results_compat pr ON pr.result_id = l.result_id
            WHERE l.tier IN ({tier_placeholders})
              AND pr.stage1_passed = 1
              {missing}
            ORDER BY l.tier ASC, l.composite_score DESC, l.result_id ASC
            LIMIT ?
            """,
            tier_params + (int(top),),
        ).fetchall()
    ]
    return rows, {"rows": len(rows), "metrics": tuple(metrics)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill binding-family metrics via research.tools.backfill"
    )
    parser.add_argument("--metrics", default="binding")
    add_common_backfill_args(
        parser,
        default_top=50,
        default_tier="validation,investigation",
    )
    args = parser.parse_args()
    probes = _parse_metrics(args.metrics)
    run_legacy_backfill(
        probes=probes,
        tier_csv=str(args.tier),
        top_per_tier=int(args.top),
        device=str(args.device),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
