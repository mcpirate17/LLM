"""Backfill template_stats, op_stats, motif_stats from existing program_results.

Usage:
    python -m research.tools.backfill_stats [--db research/lab_notebook.db]

Reads graph_json from program_results, extracts templates_used/motifs_used/op
names, and populates the three analytics tables with aggregated statistics.
Idempotent — safe to re-run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sqlite3
import time
from collections import Counter
from typing import Dict, List, Tuple


def _extract_graph_info(
    graph_json: str,
) -> Tuple[List[str], List[str], List[str]]:
    """Extract template names, motif names, and op names from graph JSON."""
    try:
        g = json.loads(graph_json)
    except (json.JSONDecodeError, TypeError):
        return [], [], []

    metadata = g.get("metadata", {})
    templates = metadata.get("templates_used", [])
    motifs = metadata.get("motifs_used", [])

    ops = []
    nodes = g.get("nodes", {})
    node_iter = nodes.values() if isinstance(nodes, dict) else nodes
    for node in node_iter:
        if isinstance(node, dict):
            op = node.get("op_name")
            if op and op != "input":
                ops.append(op)

    return (
        templates if isinstance(templates, list) else [],
        motifs if isinstance(motifs, list) else [],
        ops,
    )


def _safe_std(values: List[float]) -> float:
    """Standard deviation, or 0.0 if fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _mean_or_none(values: List[float]):
    return sum(values) / len(values) if values else None


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Ensure analytics tables exist via LabNotebook schema."""
    from research.scientist.notebook._shared import NOTEBOOK_SCHEMA

    conn.executescript(NOTEBOOK_SCHEMA)


def backfill(db_path: str = "research/lab_notebook.db") -> Dict[str, int]:
    """Backfill analytics tables. Returns row counts inserted."""
    conn = sqlite3.connect(db_path, timeout=15.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    _ensure_tables(conn)

    now = time.time()

    rows = conn.execute(
        """SELECT graph_json, stage0_passed, stage1_passed, loss_ratio, novelty_score
           FROM program_results
           WHERE graph_json IS NOT NULL"""
    ).fetchall()

    # Accumulators — using lists for losses/novelties, counters for co-occurrence
    tpl_data: Dict[str, list] = {}  # [eval, s0, s1, [losses], [novelties]]
    op_data: Dict[str, list] = {}  # [eval, s0, s1, [losses], [novelties], Counter]
    motif_data: Dict[
        str, list
    ] = {}  # [eval, s0, s1, [losses], [novelties], best_tpl, best_loss]

    for graph_json, s0, s1, loss_ratio, novelty in rows:
        templates, motifs, ops = _extract_graph_info(graph_json)
        s0_pass = 1 if s0 else 0
        s1_pass = 1 if s1 else 0
        valid_loss = loss_ratio is not None and math.isfinite(loss_ratio)
        valid_nov = novelty is not None and math.isfinite(novelty)

        for tpl in templates:
            if tpl not in tpl_data:
                tpl_data[tpl] = [0, 0, 0, [], []]
            d = tpl_data[tpl]
            d[0] += 1
            d[1] += s0_pass
            d[2] += s1_pass
            if valid_loss:
                d[3].append(loss_ratio)
            if valid_nov:
                d[4].append(novelty)

        op_set = set(ops)
        for op in op_set:
            if op not in op_data:
                op_data[op] = [0, 0, 0, [], [], Counter()]
            d = op_data[op]
            d[0] += 1
            d[1] += s0_pass
            d[2] += s1_pass
            if valid_loss:
                d[3].append(loss_ratio)
            if valid_nov:
                d[4].append(novelty)

        # Co-occurrence: iterate pairs once via combinations (not O(n²) nested loop)
        for a, b in itertools.combinations(op_set, 2):
            op_data[a][5][b] += 1
            op_data[b][5][a] += 1

        for motif in motifs:
            if motif not in motif_data:
                motif_data[motif] = [0, 0, 0, [], [], None, float("inf")]
            d = motif_data[motif]
            d[0] += 1
            d[1] += s0_pass
            d[2] += s1_pass
            if valid_loss:
                d[3].append(loss_ratio)
                if loss_ratio < d[6]:
                    d[6] = loss_ratio
                    d[5] = templates[0] if templates else None
            if valid_nov:
                d[4].append(novelty)

    # Write template_stats
    conn.execute("DELETE FROM template_stats")
    for tpl, (ev, s0, s1, losses, novs) in tpl_data.items():
        conn.execute(
            """INSERT INTO template_stats
               (template_name, eval_count, s0_pass_count, s1_pass_count,
                mean_loss, min_loss, std_loss, mean_novelty, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tpl,
                ev,
                s0,
                s1,
                _mean_or_none(losses),
                min(losses) if losses else None,
                _safe_std(losses) if losses else None,
                _mean_or_none(novs),
                now,
            ),
        )

    # Write op_stats
    conn.execute("DELETE FROM op_stats")
    for op, (ev, s0, s1, losses, novs, co_counter) in op_data.items():
        top20 = dict(co_counter.most_common(20))
        conn.execute(
            """INSERT INTO op_stats
               (op_name, eval_count, s0_pass_count, s1_pass_count,
                mean_loss, min_loss, std_loss, mean_novelty,
                co_occurrence_json, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                op,
                ev,
                s0,
                s1,
                _mean_or_none(losses),
                min(losses) if losses else None,
                _safe_std(losses) if losses else None,
                _mean_or_none(novs),
                json.dumps(top20) if top20 else None,
                now,
            ),
        )

    # Write motif_stats
    conn.execute("DELETE FROM motif_stats")
    for motif, (ev, s0, s1, losses, novs, best_tpl, _) in motif_data.items():
        conn.execute(
            """INSERT INTO motif_stats
               (motif_name, eval_count, s0_pass_count, s1_pass_count,
                mean_loss, min_loss, std_loss, mean_novelty,
                best_template, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                motif,
                ev,
                s0,
                s1,
                _mean_or_none(losses),
                min(losses) if losses else None,
                _safe_std(losses) if losses else None,
                _mean_or_none(novs),
                best_tpl,
                now,
            ),
        )

    conn.commit()
    conn.close()

    counts = {
        "template_stats": len(tpl_data),
        "op_stats": len(op_data),
        "motif_stats": len(motif_data),
    }
    print(f"Backfilled: {counts}")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill analytics stats tables")
    parser.add_argument("--db", default="research/lab_notebook.db")
    args = parser.parse_args()
    backfill(args.db)
