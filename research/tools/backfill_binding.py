#!/usr/bin/env python3
"""Legacy compatibility wrapper for the unified backfill runner."""

from __future__ import annotations

import argparse

from research.tools._legacy_backfill_cli import (
    add_common_backfill_args,
    run_legacy_backfill,
)

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


def _requested_metric_is_missing(row, metrics: tuple[str, ...]) -> bool:
    for metric in metrics:
        if metric == "binding" and row["binding_auc"] is None:
            return True
        if metric == "induction" and row["induction_auc"] is None:
            return True
        if metric == "ar" and row["ar_auc"] is None:
            return True
    return False


def _query_candidates(
    nb, tiers: list[str], top: int, force: bool, metrics: tuple[str, ...]
):
    tier_ph = ",".join("?" for _ in tiers)
    rows = nb.conn.execute(
        f"SELECT l.entry_id, l.result_id, l.tier, l.composite_score, "
        f"l.is_reference, pr.graph_json, pr.binding_auc, pr.induction_auc, "
        f"pr.ar_auc, pr.graph_fingerprint, pr.stage1_passed "
        f"FROM leaderboard l "
        f"LEFT JOIN program_results pr ON l.result_id = pr.result_id "
        f"WHERE l.tier IN ({tier_ph}) AND COALESCE(pr.stage1_passed, 0) = 1 "
        f"ORDER BY l.composite_score DESC",
        tuple(tiers),
    ).fetchall()
    if not force:
        rows = [row for row in rows if _requested_metric_is_missing(row, metrics)]
    by_tier: dict[str, list] = {}
    for row in rows:
        tier_rows = by_tier.setdefault(row["tier"], [])
        if len(tier_rows) < top:
            tier_rows.append(row)
    ordered = []
    for tier in tiers:
        ordered.extend(by_tier.get(tier, []))
    return ordered, by_tier


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
