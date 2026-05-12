"""Mine component-rule evidence from historical graph runs.

Read-only tool. It summarizes observed op/role transitions, multi-mixer
patterns, recursive contexts, and >=N-op candidate windows from
``program_results_compat`` so dynamic component rules can be data-backed
instead of hardcoded into templates.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.synthesis.component_rules import (
    component_role_counts,
    estimated_chain_lowered_op_count,
    validate_component_op_chain,
)
from research.synthesis.op_roles import OpRole, get_role


DEFAULT_DB = Path("research/runs.db")
DEFAULT_OUTPUT_DIR = Path("research/reports")
RECURSION_OPS = frozenset(
    {
        "fixed_point_iter",
        "mixture_of_recursions",
        "depth_gated_transform",
        "score_depth_blend",
        "depth_weighted_proj",
    }
)


def mine_component_rules(
    *,
    db_path: str | Path = DEFAULT_DB,
    limit: int = 5000,
    min_window_ops: int = 8,
    min_support: int = 8,
) -> dict[str, Any]:
    """Return a compact read-only component-rule mining report."""
    db = Path(db_path)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    rows = _fetch_rows(conn, limit=limit)
    pair_stats: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0, 0.0])
    role_pair_stats: dict[tuple[str, str], list[float]] = defaultdict(
        lambda: [0, 0, 0.0]
    )
    window_stats: dict[tuple[str, ...], list[float]] = defaultdict(lambda: [0, 0, 0.0])
    mixer_stats: dict[tuple[str, ...], list[float]] = defaultdict(lambda: [0, 0, 0.0])
    recursion_contexts: Counter[tuple[str, str, str]] = Counter()
    template_counts: Counter[str] = Counter()
    dynamic_template_rows = 0
    parsed_graphs = 0
    stage1_graphs = 0

    for row in rows:
        graph = _parse_graph(conn, db, row["graph_json"])
        if not graph:
            continue
        parsed_graphs += 1
        passed = bool(row["stage1_passed"])
        if passed:
            stage1_graphs += 1
        loss_ratio = _safe_float(row["loss_ratio"])

        metadata = graph.get("metadata") if isinstance(graph, dict) else {}
        if isinstance(metadata, dict):
            template_counts.update(str(t) for t in metadata.get("templates_used") or ())
            if metadata.get("dynamic_templates_used") or metadata.get(
                "dynamic_template_attempts"
            ):
                dynamic_template_rows += 1

        ops = _topological_ops(graph)
        if not ops:
            continue
        _record_pairs(pair_stats, ops, passed, loss_ratio)
        _record_pairs(
            role_pair_stats,
            tuple(get_role(op).value for op in ops),
            passed,
            loss_ratio,
        )
        _record_windows(window_stats, ops, min_window_ops, passed, loss_ratio)
        mixers = tuple(op for op in ops if get_role(op) is OpRole.MIX)
        if len(mixers) >= 2:
            _increment(mixer_stats, mixers, passed, loss_ratio)
        _record_recursion_contexts(recursion_contexts, ops)

    global_stage1_rate = stage1_graphs / parsed_graphs if parsed_graphs else 0.0
    return {
        "schema_version": "component_rule_mining_v1",
        "created_at": time.time(),
        "db_path": str(db),
        "limit": int(limit),
        "min_window_ops": int(min_window_ops),
        "min_support": int(min_support),
        "summary": {
            "rows_scanned": len(rows),
            "graphs_parsed": parsed_graphs,
            "stage1_graphs": stage1_graphs,
            "stage1_rate": global_stage1_rate,
            "dynamic_template_rows": dynamic_template_rows,
        },
        "template_counts": dict(template_counts.most_common(80)),
        "op_pair_rules": _rank_rules(pair_stats, min_support, global_stage1_rate),
        "role_pair_rules": _rank_rules(
            role_pair_stats, min_support, global_stage1_rate
        ),
        "multi_mixer_patterns": _rank_rules(
            mixer_stats,
            min_support=max(2, min_support // 2),
            baseline=global_stage1_rate,
        ),
        "candidate_windows": _rank_windows(
            window_stats,
            min_support=min_support,
            baseline=global_stage1_rate,
            min_window_ops=min_window_ops,
        ),
        "recursion_contexts": [
            {"prev": p, "op": op, "next": n, "count": count}
            for (p, op, n), count in recursion_contexts.most_common(80)
        ],
    }


def _fetch_rows(conn: sqlite3.Connection, *, limit: int) -> list[sqlite3.Row]:
    table = _program_results_read_table(conn)
    return list(
        conn.execute(
            f"""
            SELECT result_id, graph_json, stage0_passed, stage1_passed, loss_ratio
            FROM {table}
            WHERE COALESCE(graph_json, '') NOT IN ('', '{{}}')
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        )
    )


def _program_results_read_table(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='program_results_compat' LIMIT 1"
    ).fetchone()
    return "program_results_compat" if row else "program_results"


def _parse_graph(
    conn: sqlite3.Connection,
    db_path: Path,
    graph_json: Any,
) -> dict[str, Any] | None:
    try:
        payload = resolve_graph_json_value(conn, db_path, graph_json)
        graph = json.loads(payload)
    except Exception:
        return None
    return graph if isinstance(graph, dict) else None


def _topological_ops(graph: dict[str, Any]) -> tuple[str, ...]:
    nodes = graph.get("nodes")
    if not isinstance(nodes, dict):
        return ()
    ordered = sorted(
        (node for node in nodes.values() if isinstance(node, dict)),
        key=lambda node: int(node.get("id", 0)),
    )
    ops: list[str] = []
    for node in ordered:
        if node.get("is_input") or node.get("is_output"):
            continue
        op_name = str(node.get("op_name") or "")
        if op_name and op_name not in {"input", "output"}:
            ops.append(op_name)
    return tuple(ops)


def _record_pairs(
    stats: dict[tuple[str, str], list[float]],
    values: Iterable[str],
    passed: bool,
    loss_ratio: float | None,
) -> None:
    seq = tuple(values)
    for left, right in zip(seq, seq[1:]):
        _increment(stats, (left, right), passed, loss_ratio)


def _record_windows(
    stats: dict[tuple[str, ...], list[float]],
    ops: tuple[str, ...],
    min_window_ops: int,
    passed: bool,
    loss_ratio: float | None,
) -> None:
    size = max(1, int(min_window_ops))
    if len(ops) < size:
        return
    for idx in range(0, len(ops) - size + 1):
        _increment(stats, ops[idx : idx + size], passed, loss_ratio)


def _record_recursion_contexts(
    stats: Counter[tuple[str, str, str]],
    ops: tuple[str, ...],
) -> None:
    for idx, op_name in enumerate(ops):
        if op_name not in RECURSION_OPS:
            continue
        prev_op = ops[idx - 1] if idx > 0 else "<start>"
        next_op = ops[idx + 1] if idx + 1 < len(ops) else "<end>"
        stats[(prev_op, op_name, next_op)] += 1


def _increment(
    stats: dict[tuple[str, ...], list[float]],
    key: tuple[str, ...],
    passed: bool,
    loss_ratio: float | None,
) -> None:
    record = stats[key]
    record[0] += 1
    if passed:
        record[1] += 1
    if loss_ratio is not None:
        record[2] += float(loss_ratio)


def _rank_rules(
    stats: dict[tuple[str, ...], list[float]],
    min_support: int,
    baseline: float,
) -> dict[str, list[dict[str, Any]]]:
    rows = [_format_rule(key, values, baseline) for key, values in stats.items()]
    supported = [row for row in rows if row["n"] >= min_support]
    positive = sorted(
        supported,
        key=lambda row: (row["pass_rate_lift"], row["n"]),
        reverse=True,
    )[:80]
    negative = sorted(
        supported,
        key=lambda row: (row["pass_rate_lift"], -row["n"]),
    )[:80]
    return {"positive": positive, "negative": negative}


def _rank_windows(
    stats: dict[tuple[str, ...], list[float]],
    *,
    min_support: int,
    baseline: float,
    min_window_ops: int,
) -> list[dict[str, Any]]:
    rows = []
    for key, values in stats.items():
        row = _format_rule(key, values, baseline)
        if row["n"] < min_support:
            continue
        row["lowered_op_count"] = estimated_chain_lowered_op_count(key)
        row["role_counts"] = component_role_counts(key)
        row["violations"] = list(validate_component_op_chain(key))
        if row["lowered_op_count"] >= min_window_ops:
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (row["pass_rate_lift"], row["n"]),
        reverse=True,
    )[:120]


def _format_rule(
    key: tuple[str, ...],
    values: list[float],
    baseline: float,
) -> dict[str, Any]:
    n = int(values[0])
    passed = int(values[1])
    pass_rate = passed / n if n else 0.0
    loss_sum = float(values[2])
    return {
        "pattern": list(key),
        "n": n,
        "stage1_passed": passed,
        "pass_rate": round(pass_rate, 4),
        "pass_rate_lift": round(pass_rate - baseline, 4),
        "mean_loss_ratio": round(loss_sum / n, 4) if loss_sum else None,
    }


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _write_report(report: dict[str, Any], output: str | Path | None) -> Path:
    if output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"component_rule_mining_{stamp}.json"
    else:
        output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--min-window-ops", type=int, default=8)
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    report = mine_component_rules(
        db_path=args.db,
        limit=args.limit,
        min_window_ops=args.min_window_ops,
        min_support=args.min_support,
    )
    path = _write_report(report, args.output)
    summary = report["summary"]
    print(
        "component_rule_mining "
        f"graphs={summary['graphs_parsed']} "
        f"stage1_rate={summary['stage1_rate']:.3f} "
        f"dynamic_rows={summary['dynamic_template_rows']} "
        f"report={path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
