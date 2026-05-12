#!/usr/bin/env python
"""Build advisory priors from historical routing-decision outcomes.

The artifact is intentionally offline and read-only for generation. It scores
observed routing knob values against historical outcomes, shrinks low-support
signals toward neutral, and writes both a timestamped JSON artifact and
``latest.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from research.defaults import RUNS_DB
from research.meta_analysis.priors import _connect_readonly
from research.meta_analysis.routing_decision_analytics import (
    RoutingDecisionRow,
    _program_results_read_table,
    iter_routing_decision_outcomes,
)
from research.scientist.shared_utils import clamp, coerce_finite_float
from research.synthesis.routing_decision_priors import (
    ROUTING_DECISION_PRIOR_SCHEMA_VERSION,
    build_routing_decision_prior_index,
    canonical_routing_decision_value,
)


DEFAULT_OUTPUT_DIR = Path("research/artifacts/routing_decision_priors")
DEFAULT_MIN_SUPPORT = 8
DEFAULT_RETENTION = 12

REQUESTED_OUTCOMES: tuple[str, ...] = (
    "stage1_passed",
    "loss_ratio",
    "validation_loss_ratio",
    "ar_gate_score",
    "binding_intermediate_auc",
    "binding_screening_auc",
    "binding_screening_composite",
    "routing_utilization_entropy",
    "routing_drop_rate",
    "routing_savings_ratio",
    "routing_collapse_score",
)

HIGHER_IS_BETTER: tuple[str, ...] = (
    "stage1_passed",
    "ar_gate_score",
    "binding_intermediate_auc",
    "binding_screening_auc",
    "binding_screening_composite",
    "routing_utilization_entropy",
    "routing_savings_ratio",
)

LOWER_IS_BETTER: tuple[str, ...] = (
    "loss_ratio",
    "validation_loss_ratio",
)


def build_routing_decision_prior(
    db_path: str | Path = RUNS_DB,
    *,
    template_filter: str | None = None,
    limit: int | None = None,
    min_support: int = DEFAULT_MIN_SUPPORT,
    created_at: float | None = None,
) -> dict[str, Any]:
    """Build a support-shrunk advisory prior from runs DB routing telemetry."""

    created = float(time.time() if created_at is None else created_at)
    outcomes = _available_outcome_columns(db_path, REQUESTED_OUTCOMES)
    rows = list(
        iter_routing_decision_outcomes(
            db_path,
            outcome_columns=outcomes,
            template_filter=template_filter,
            limit=limit,
        )
    )
    global_stats = _global_stats(rows, outcomes)
    records = _score_records(rows, outcomes, global_stats, min_support=min_support)
    priors = build_routing_decision_prior_index(records)
    version = f"routing_prior_{time.strftime('%Y%m%dT%H%M%S', time.gmtime(created))}"
    return {
        "schema_version": ROUTING_DECISION_PRIOR_SCHEMA_VERSION,
        "version": version,
        "created_at": created,
        "source_db": str(db_path),
        "template_filter": template_filter,
        "limit": limit,
        "min_support": int(min_support),
        "outcome_columns": list(outcomes),
        "input_rows": len(rows),
        "decision_groups": len(records),
        "global_stats": global_stats,
        "records": records,
        "priors": priors,
        "rationale": [
            "Stage-1 pass rate, loss, validation loss, AR, and binding signals lift routing values above neutral.",
            "Routing drop and collapse signals penalize otherwise promising values.",
            "Support confidence shrinks low-observation groups toward neutral advisory weights.",
            "Missing optional AR/binding/routing metrics are ignored per signal, not interpreted as zero quality.",
        ],
    }


def write_routing_decision_prior(
    prior: dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    retention: int = DEFAULT_RETENTION,
) -> Path:
    """Write timestamped and latest routing-decision prior artifacts."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    version = str(prior.get("version") or f"routing_prior_{int(time.time())}")
    payload = json.dumps(prior, sort_keys=True, separators=(",", ":"))
    path = out_dir / f"{version}.json"
    path.write_text(payload + "\n", encoding="utf-8")
    (out_dir / "latest.json").write_text(payload + "\n", encoding="utf-8")

    if retention > 0:
        versions = sorted(
            out_dir.glob("routing_prior_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in versions[int(retention) :]:
            stale.unlink(missing_ok=True)
    return path


def _available_outcome_columns(
    db_path: str | Path, requested: Iterable[str]
) -> tuple[str, ...]:
    conn = _connect_readonly(db_path)
    try:
        table = _program_results_read_table(conn)
        columns = _table_columns(conn, table)
    finally:
        conn.close()
    available = tuple(col for col in requested if col in columns)
    if not available:
        raise ValueError(f"no requested outcome columns found in {db_path}")
    return available


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}
    except sqlite3.DatabaseError:
        return set()


def _global_stats(
    rows: list[RoutingDecisionRow], outcomes: tuple[str, ...]
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for metric in outcomes:
        values = [
            value
            for row in rows
            if (value := coerce_finite_float(row.outcomes.get(metric))) is not None
        ]
        stats[f"mean_{metric}"] = _mean(values)
        stats[f"n_{metric}"] = len(values)
    return stats


def _score_records(
    rows: list[RoutingDecisionRow],
    outcomes: tuple[str, ...],
    global_stats: dict[str, Any],
    *,
    min_support: int,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        value_key = canonical_routing_decision_value(row.value)
        key = (row.template_name, row.decision_key, value_key)
        bucket = buckets.setdefault(
            key,
            {
                "template_name": row.template_name,
                "decision_key": row.decision_key,
                "value": row.value,
                "value_key": value_key,
                "source": row.source,
                "n": 0,
                "_metrics": {metric: [] for metric in outcomes},
                "_sources": {},
            },
        )
        bucket["n"] += 1
        bucket["_sources"][row.source] = int(bucket["_sources"].get(row.source, 0)) + 1
        for metric in outcomes:
            value = coerce_finite_float(row.outcomes.get(metric))
            if value is not None:
                bucket["_metrics"][metric].append(value)

    records: list[dict[str, Any]] = []
    for bucket in buckets.values():
        metrics = _metric_summary(bucket["_metrics"])
        contributions = _score_contributions(metrics, global_stats)
        raw_score = sum(float(v) for v in contributions.values())
        confidence = _support_confidence(int(bucket["n"]), min_support=min_support)
        score = raw_score * confidence
        advisory_weight = clamp(1.0 + score, 0.25, 2.5)
        record = {
            "template_name": bucket["template_name"],
            "decision_key": bucket["decision_key"],
            "value": bucket["value"],
            "value_key": bucket["value_key"],
            "source": bucket["source"],
            "source_counts": dict(sorted(bucket["_sources"].items())),
            "n": int(bucket["n"]),
            "support_confidence": round(confidence, 6),
            "raw_score": round(raw_score, 6),
            "score": round(score, 6),
            "advisory_weight": round(advisory_weight, 6),
            "metrics": metrics,
            "contributions": {k: round(float(v), 6) for k, v in contributions.items()},
        }
        records.append(record)

    records.sort(
        key=lambda r: (
            -float(r["advisory_weight"]),
            -int(r["n"]),
            str(r["template_name"]),
            str(r["decision_key"]),
            str(r["value_key"]),
        )
    )
    return records


def _metric_summary(metric_values: dict[str, list[float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric, values in metric_values.items():
        summary[f"mean_{metric}"] = _mean(values)
        summary[f"n_{metric}"] = len(values)
    return summary


def _score_contributions(
    metrics: dict[str, Any],
    global_stats: dict[str, Any],
) -> dict[str, float]:
    s1 = _higher_lift(metrics, global_stats, "stage1_passed", floor=0.05)
    loss = _lower_lift(metrics, global_stats, "loss_ratio", floor=0.05)
    validation_loss = _lower_lift(
        metrics, global_stats, "validation_loss_ratio", floor=0.05
    )
    ar = _higher_lift(metrics, global_stats, "ar_gate_score", floor=0.05)
    binding = (
        _mean(
            [
                _higher_lift(
                    metrics, global_stats, "binding_intermediate_auc", floor=0.05
                ),
                _higher_lift(
                    metrics, global_stats, "binding_screening_auc", floor=0.05
                ),
                _higher_lift(
                    metrics, global_stats, "binding_screening_composite", floor=0.05
                ),
            ],
            skip_zero_missing=True,
        )
        or 0.0
    )
    entropy = _higher_lift(
        metrics, global_stats, "routing_utilization_entropy", floor=0.05
    )
    savings = _higher_lift(metrics, global_stats, "routing_savings_ratio", floor=0.05)
    drop_penalty = _rate_penalty(
        metrics, global_stats, "routing_drop_rate", tolerated=0.15
    )
    collapse_penalty = _rate_penalty(
        metrics, global_stats, "routing_collapse_score", tolerated=0.05
    )
    return {
        "s1_pass_lift": 0.35 * clamp(s1, -1.5, 1.5),
        "loss_improvement": 0.25 * clamp(loss, -1.5, 1.5),
        "validation_loss_improvement": 0.20 * clamp(validation_loss, -1.5, 1.5),
        "ar_lift": 0.10 * clamp(ar, -1.5, 1.5),
        "binding_lift": 0.10 * clamp(binding, -1.5, 1.5),
        "routing_entropy_lift": 0.05 * clamp(entropy, -1.5, 1.5),
        "routing_savings_lift": 0.05 * clamp(savings, -1.5, 1.5),
        "routing_drop_penalty": -0.20 * clamp(drop_penalty, 0.0, 2.0),
        "routing_collapse_penalty": -0.25 * clamp(collapse_penalty, 0.0, 2.0),
    }


def _higher_lift(
    metrics: dict[str, Any], global_stats: dict[str, Any], metric: str, *, floor: float
) -> float:
    value = _metric_mean(metrics, metric)
    baseline = coerce_finite_float(global_stats.get(f"mean_{metric}"))
    if value is None or baseline is None:
        return 0.0
    return (value - baseline) / max(abs(baseline), floor)


def _lower_lift(
    metrics: dict[str, Any], global_stats: dict[str, Any], metric: str, *, floor: float
) -> float:
    value = _metric_mean(metrics, metric)
    baseline = coerce_finite_float(global_stats.get(f"mean_{metric}"))
    if value is None or baseline is None:
        return 0.0
    return (baseline - value) / max(abs(baseline), floor)


def _rate_penalty(
    metrics: dict[str, Any],
    global_stats: dict[str, Any],
    metric: str,
    *,
    tolerated: float,
) -> float:
    value = _metric_mean(metrics, metric)
    if value is None:
        return 0.0
    baseline = coerce_finite_float(global_stats.get(f"mean_{metric}")) or 0.0
    absolute = max(0.0, value - tolerated) / max(tolerated, 0.01)
    relative = max(0.0, value - baseline) / max(abs(baseline), tolerated, 0.01)
    return max(absolute, relative)


def _metric_mean(metrics: dict[str, Any], metric: str) -> float | None:
    if int(metrics.get(f"n_{metric}") or 0) <= 0:
        return None
    return coerce_finite_float(metrics.get(f"mean_{metric}"))


def _support_confidence(n: int, *, min_support: int) -> float:
    return min(1.0, math.sqrt(max(0, int(n)) / max(1.0, float(min_support) * 4.0)))


def _mean(values: Iterable[float], *, skip_zero_missing: bool = False) -> float | None:
    clean = []
    for value in values:
        converted = coerce_finite_float(value)
        if converted is None:
            continue
        if skip_zero_missing and converted == 0.0:
            continue
        clean.append(converted)
    if not clean:
        return None if skip_zero_missing else 0.0
    return float(sum(clean) / len(clean))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(RUNS_DB), help="runs/lab notebook DB path")
    parser.add_argument("--template", default=None, help="optional template filter")
    parser.add_argument(
        "--limit", type=int, default=None, help="optional source-row cap"
    )
    parser.add_argument("--min-support", type=int, default=DEFAULT_MIN_SUPPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--retention", type=int, default=DEFAULT_RETENTION)
    args = parser.parse_args(argv)

    prior = build_routing_decision_prior(
        args.db,
        template_filter=args.template,
        limit=args.limit,
        min_support=max(1, args.min_support),
    )
    path = write_routing_decision_prior(
        prior,
        output_dir=args.output_dir,
        retention=max(0, args.retention),
    )
    summary = {
        "path": str(path),
        "version": prior["version"],
        "input_rows": prior["input_rows"],
        "decision_groups": prior["decision_groups"],
        "min_support": prior["min_support"],
        "outcome_columns": prior["outcome_columns"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
