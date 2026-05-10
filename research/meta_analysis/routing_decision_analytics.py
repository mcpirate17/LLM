"""Join routing-knob decisions with program outcomes for policy learning.

Move #2 records every random routing knob (gate_threshold, lane_count,
hard_classes, etc.) on ``graph.metadata['routing_decisions']``. Because
``graph.to_dict()`` emits metadata into the graph_json column on
``program_results``, the decisions are already queryable via SQLite's
``json_extract`` — no schema migration required.

This module provides the analytics layer: pull decisions × outcomes, group
by (template_name, decision_key, value), and emit pass-rate / mean-loss /
mean-capability statistics. The output is the input a Thompson-sampling or
UCB policy needs to replace the rng.choice calls.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .priors import _connect_readonly


_DEFAULT_OUTCOME_COLUMNS: tuple[str, ...] = (
    "stage1_passed",
    "loss_ratio",
    "validation_loss_ratio",
    "ar_gate_score",
    "binding_intermediate_auc",
    "binding_screening_composite",
)


@dataclass(frozen=True)
class RoutingDecisionRow:
    """A single (decision, outcome) join row."""

    template_name: str
    decision_key: str
    value: Any
    source: str
    outcomes: Dict[str, Any]


def iter_routing_decision_outcomes(
    runs_db_path: str | Path,
    *,
    outcome_columns: Iterable[str] = _DEFAULT_OUTCOME_COLUMNS,
    template_filter: Optional[str] = None,
    limit: Optional[int] = None,
) -> Iterator[RoutingDecisionRow]:
    """Yield (decision, outcome) rows from program_results.

    Reads each program_result row, parses graph_json metadata for
    routing_decisions, and emits one RoutingDecisionRow per recorded
    decision. Only programs whose graph metadata contains routing_decisions
    contribute rows.

    Outcome columns are pulled from program_results directly; missing or
    NULL values pass through as None.
    """
    cols = tuple(outcome_columns)
    select_outcomes = ", ".join(f"pr.{col}" for col in cols)
    conn = _connect_readonly(runs_db_path)
    try:
        pr_table = _program_results_read_table(conn)
        sql = f"""
        SELECT pr.graph_json, {select_outcomes}
        FROM {pr_table} pr
        WHERE pr.graph_json LIKE '%routing_decisions%'
        """
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        for row in conn.execute(sql):
            graph_json = row[0]
            outcomes = dict(zip(cols, row[1:]))
            decisions = _parse_decisions(graph_json)
            if not decisions:
                continue
            for d in decisions:
                template_name = str(d.get("template_name") or "")
                if template_filter and template_name != template_filter:
                    continue
                yield RoutingDecisionRow(
                    template_name=template_name,
                    decision_key=str(d.get("decision_key") or ""),
                    value=d.get("value"),
                    source=str(d.get("source") or ""),
                    outcomes=outcomes,
                )
    finally:
        conn.close()


def _program_results_read_table(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type IN ('table', 'view') AND name = 'program_results_compat'
        """
    ).fetchone()
    return "program_results_compat" if row else "program_results"


def _parse_decisions(graph_json: Any) -> List[Dict[str, Any]]:
    if not isinstance(graph_json, str) or not graph_json:
        return []
    try:
        graph = json.loads(graph_json)
    except (TypeError, ValueError):
        return []
    metadata = graph.get("metadata") if isinstance(graph, dict) else None
    if not isinstance(metadata, dict):
        return []
    decisions = metadata.get("routing_decisions")
    if not isinstance(decisions, list):
        return []
    return [d for d in decisions if isinstance(d, dict)]


def summarize_routing_decisions(
    rows: Iterable[RoutingDecisionRow],
    *,
    primary_outcome: str = "stage1_passed",
    secondary_outcomes: Iterable[str] = (
        "ar_gate_score",
        "binding_intermediate_auc",
        "loss_ratio",
    ),
) -> List[Dict[str, Any]]:
    """Aggregate per (template_name, decision_key, value).

    Returns one record per unique (template, decision_key, value) tuple with:
        n: number of programs that hit this knob value
        pass_rate: fraction with primary_outcome=1 (Bernoulli proxy)
        mean_<col>: mean of secondary outcomes among programs with non-null values

    Missing/None outcomes are excluded from each individual mean — partial
    coverage is normal because not every program runs every probe.
    """
    secondary = tuple(secondary_outcomes)
    buckets: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for row in rows:
        value_repr = json.dumps(row.value, sort_keys=True)
        key = (row.template_name, row.decision_key, value_repr)
        bucket = buckets.setdefault(
            key,
            {
                "template_name": row.template_name,
                "decision_key": row.decision_key,
                "value": row.value,
                "source": row.source,
                "n": 0,
                "pass_count": 0,
                "_secondary": {col: [] for col in secondary},
            },
        )
        bucket["n"] += 1
        primary = row.outcomes.get(primary_outcome)
        if primary is not None and bool(primary):
            bucket["pass_count"] += 1
        for col in secondary:
            value = row.outcomes.get(col)
            if value is None:
                continue
            try:
                bucket["_secondary"][col].append(float(value))
            except (TypeError, ValueError):
                continue

    records: List[Dict[str, Any]] = []
    for bucket in buckets.values():
        n = int(bucket["n"])
        pass_rate = bucket["pass_count"] / n if n else 0.0
        record = {
            "template_name": bucket["template_name"],
            "decision_key": bucket["decision_key"],
            "value": bucket["value"],
            "source": bucket["source"],
            "n": n,
            "pass_rate": pass_rate,
        }
        for col in secondary:
            samples = bucket["_secondary"][col]
            record[f"mean_{col}"] = statistics.fmean(samples) if samples else None
            record[f"n_{col}"] = len(samples)
        records.append(record)

    records.sort(key=lambda r: (r["template_name"], r["decision_key"], -r["n"]))
    return records
