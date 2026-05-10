"""Read-only AR/binding outcome overlays for proposal surfaces.

The overlay is deliberately advisory. It joins candidate structures against
already-materialized meta-analysis observations and returns support-aware
relative signals without changing generation policy, rank models, or audit
metadata.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from research.defaults import RUNS_DB
from research.scientist.shared_utils import coerce_finite_float as _finite

from .metadata_db import DEFAULT_META_ANALYSIS_DB
from .priors import _connect_readonly


OverlayPayload = dict[str, Any]

_MIN_SUPPORT = 25
_AR_EXPR = (
    "COALESCE(ar_curriculum_auc_pair_final, ar_validation_rank_score, "
    "ar_intermediate_auc, ar_gate_score, ar_legacy_auc)"
)
_BINDING_EXPR = (
    "COALESCE(binding_multislot_auc, binding_intermediate_auc, "
    "binding_curriculum_auc, binding_screening_auc)"
)


def empty_overlay(*, holdout_required: bool = True) -> OverlayPayload:
    return {
        "expected_ar_gain": None,
        "ar_gain_n": 0,
        "expected_binding_gain": None,
        "binding_gain_n": 0,
        "retention_risk": None,
        "collapse_risk": None,
        "holdout_required": bool(holdout_required),
    }


def overlay_for_pair(
    op_a: str,
    op_b: str,
    *,
    meta_db_path: str | Path = DEFAULT_META_ANALYSIS_DB,
) -> OverlayPayload:
    """Return overlay for programs containing both pair ops."""
    return _overlay_for_ops((op_a, op_b), meta_db_path=meta_db_path)


def overlay_for_chain(
    chain: Iterable[str],
    *,
    meta_db_path: str | Path = DEFAULT_META_ANALYSIS_DB,
) -> OverlayPayload:
    """Return overlay for programs containing all ops in ``chain``.

    The V1 overlay uses set membership, not exact order. Exact ordered-chain
    evidence can replace this implementation later without changing callers.
    """
    return _overlay_for_ops(tuple(str(op) for op in chain), meta_db_path=meta_db_path)


def overlay_for_graph(
    graph_json: str | dict[str, Any],
    *,
    meta_db_path: str | Path = DEFAULT_META_ANALYSIS_DB,
) -> OverlayPayload:
    """Return overlay for the op set parsed from a graph JSON payload."""
    return overlay_for_chain(_graph_ops(graph_json), meta_db_path=meta_db_path)


def overlay_for_routing_decision(
    template_name: str,
    decision_key: str,
    value: Any,
    *,
    runs_db_path: str | Path = RUNS_DB,
) -> OverlayPayload:
    """Return overlay for matching routing-decision audit rows in runs.db."""
    try:
        conn = _connect_readonly(runs_db_path)
    except sqlite3.Error:
        return empty_overlay()
    try:
        pr_table = _program_results_read_table(conn)
        global_stats = _runs_global_stats(conn, pr_table)
        rows = conn.execute(
            f"""
            SELECT graph_json,
                   {_runs_ar_expr()} AS ar_signal,
                   {_runs_binding_expr()} AS binding_signal,
                   ar_curriculum_s0_retention,
                   routing_collapse_score
            FROM {pr_table}
            WHERE graph_json LIKE '%routing_decisions%'
            """
        ).fetchall()
    except sqlite3.Error:
        return empty_overlay()
    finally:
        conn.close()

    wanted_value = _canonical_json(value)
    matches: list[dict[str, Any]] = []
    for row in rows:
        for decision in _parse_routing_decisions(row["graph_json"]):
            if str(decision.get("template_name") or "") != str(template_name):
                continue
            if str(decision.get("decision_key") or "") != str(decision_key):
                continue
            if _canonical_json(decision.get("value")) != wanted_value:
                continue
            matches.append(dict(row))
            break
    return _payload_from_rows(matches, global_stats=global_stats)


def _overlay_for_ops(
    ops: Iterable[str],
    *,
    meta_db_path: str | Path,
) -> OverlayPayload:
    op_names = tuple(dict.fromkeys(str(op).strip() for op in ops if str(op).strip()))
    if not op_names:
        return empty_overlay()
    try:
        conn = _connect_readonly(meta_db_path)
    except sqlite3.Error:
        return empty_overlay()
    try:
        if not _table_exists(conn, "op_observations"):
            return empty_overlay()
        global_stats = _meta_global_stats(conn)
        rows = _meta_rows_for_ops(conn, op_names)
    except sqlite3.Error:
        return empty_overlay()
    finally:
        conn.close()
    return _payload_from_rows(rows, global_stats=global_stats)


def _meta_rows_for_ops(
    conn: sqlite3.Connection, op_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in op_names)
    rows = conn.execute(
        f"""
        WITH hits AS (
            SELECT result_id
            FROM op_observations
            WHERE op_name IN ({placeholders})
            GROUP BY result_id
            HAVING COUNT(DISTINCT op_name) = ?
        )
        SELECT
            MAX({_AR_EXPR}) AS ar_signal,
            MAX({_BINDING_EXPR}) AS binding_signal,
            MAX(ar_curriculum_s0_retention) AS ar_curriculum_s0_retention,
            AVG(frequency_collapse_risk) AS collapse_signal
        FROM op_observations
        WHERE result_id IN (SELECT result_id FROM hits)
        GROUP BY result_id
        """,
        (*op_names, len(op_names)),
    ).fetchall()
    return [dict(row) for row in rows]


def _meta_global_stats(conn: sqlite3.Connection) -> dict[str, float | None]:
    row = conn.execute(
        f"""
        SELECT
            AVG(ar_signal) AS mean_ar,
            AVG(binding_signal) AS mean_binding,
            AVG(ar_curriculum_s0_retention) AS mean_retention,
            AVG(collapse_signal) AS mean_collapse
        FROM (
            SELECT
                result_id,
                MAX({_AR_EXPR}) AS ar_signal,
                MAX({_BINDING_EXPR}) AS binding_signal,
                MAX(ar_curriculum_s0_retention) AS ar_curriculum_s0_retention,
                AVG(frequency_collapse_risk) AS collapse_signal
            FROM op_observations
            GROUP BY result_id
        )
        """
    ).fetchone()
    return _stats_from_row(row)


def _runs_global_stats(
    conn: sqlite3.Connection, pr_table: str
) -> dict[str, float | None]:
    row = conn.execute(
        f"""
        SELECT
            AVG({_runs_ar_expr()}) AS mean_ar,
            AVG({_runs_binding_expr()}) AS mean_binding,
            AVG(ar_curriculum_s0_retention) AS mean_retention,
            AVG(routing_collapse_score) AS mean_collapse
        FROM {pr_table}
        """
    ).fetchone()
    return _stats_from_row(row)


def _payload_from_rows(
    rows: list[dict[str, Any]],
    *,
    global_stats: dict[str, float | None],
) -> OverlayPayload:
    ar_values = [_finite(row.get("ar_signal")) for row in rows]
    ar_values = [value for value in ar_values if value is not None]
    binding_values = [_finite(row.get("binding_signal")) for row in rows]
    binding_values = [value for value in binding_values if value is not None]
    retention_values = [_finite(row.get("ar_curriculum_s0_retention")) for row in rows]
    retention_values = [value for value in retention_values if value is not None]
    collapse_values = [
        _finite(row.get("collapse_signal", row.get("routing_collapse_score")))
        for row in rows
    ]
    collapse_values = [value for value in collapse_values if value is not None]

    ar_mean = _mean(ar_values)
    binding_mean = _mean(binding_values)
    retention_mean = _mean(retention_values)
    collapse_mean = _mean(collapse_values)

    retention_risk = None if retention_mean is None else _clamp01(1.0 - retention_mean)
    collapse_risk = None if collapse_mean is None else _clamp01(collapse_mean)
    ar_n = len(ar_values)
    binding_n = len(binding_values)
    holdout_required = (
        max(ar_n, binding_n, len(retention_values), len(collapse_values)) < _MIN_SUPPORT
        or ar_n < _MIN_SUPPORT
        or binding_n < _MIN_SUPPORT
        or (retention_risk is not None and retention_risk >= 0.50)
        or (collapse_risk is not None and collapse_risk >= 0.60)
    )

    return {
        "expected_ar_gain": _gain(ar_mean, global_stats.get("mean_ar")),
        "ar_gain_n": ar_n,
        "expected_binding_gain": _gain(binding_mean, global_stats.get("mean_binding")),
        "binding_gain_n": binding_n,
        "retention_risk": _round_or_none(retention_risk),
        "collapse_risk": _round_or_none(collapse_risk),
        "holdout_required": bool(holdout_required),
    }


def _stats_from_row(row: sqlite3.Row | None) -> dict[str, float | None]:
    if row is None:
        return {
            "mean_ar": None,
            "mean_binding": None,
            "mean_retention": None,
            "mean_collapse": None,
        }
    return {
        "mean_ar": _finite(row["mean_ar"]),
        "mean_binding": _finite(row["mean_binding"]),
        "mean_retention": _finite(row["mean_retention"]),
        "mean_collapse": _finite(row["mean_collapse"]),
    }


def _graph_ops(graph_json: str | dict[str, Any]) -> list[str]:
    graph: Any = graph_json
    if isinstance(graph_json, str):
        try:
            graph = json.loads(graph_json)
        except (TypeError, ValueError):
            return []
    if not isinstance(graph, dict):
        return []
    raw_nodes = graph.get("nodes")
    if isinstance(raw_nodes, dict):
        nodes = raw_nodes.values()
    elif isinstance(raw_nodes, list):
        nodes = raw_nodes
    else:
        nodes = []
    ops: list[str] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        op_name = str(node.get("op_name") or node.get("component_type") or "").strip()
        if op_name:
            ops.append(op_name)
    return ops


def _parse_routing_decisions(graph_json: Any) -> list[dict[str, Any]]:
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
    return [decision for decision in decisions if isinstance(decision, dict)]


def _program_results_read_table(conn: sqlite3.Connection) -> str:
    return (
        "program_results_compat"
        if _table_exists(conn, "program_results_compat")
        else "program_results"
    )


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _runs_ar_expr() -> str:
    return (
        "COALESCE(ar_curriculum_auc_pair_final, ar_validation_rank_score, "
        "ar_intermediate_auc, ar_gate_score, ar_legacy_auc)"
    )


def _runs_binding_expr() -> str:
    return (
        "COALESCE(binding_multislot_auc, binding_intermediate_auc, "
        "binding_curriculum_auc, binding_screening_auc)"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _gain(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return round(value - baseline, 6)


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
