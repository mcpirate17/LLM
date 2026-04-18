#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DB_PATH = "research/lab_notebook.db"
DEFAULT_OUT_DIR = Path("research/reports/pass_vs_induction")


def _bucket(auc: float) -> str:
    if auc >= 0.05:
        return "learner"
    if auc >= 0.02:
        return "weak_learner"
    if auc > 0.0:
        return "trace_learner"
    return "zero"


def _load_graph(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _iter_ops(graph: dict[str, Any]) -> list[str]:
    nodes = graph.get("nodes") or {}
    values = (
        nodes.values()
        if isinstance(nodes, dict)
        else nodes
        if isinstance(nodes, list)
        else []
    )
    ops: list[str] = []
    for node in values:
        if not isinstance(node, dict):
            continue
        op = str(node.get("op_name") or "").strip()
        if op and op not in {"input", "output"}:
            ops.append(op)
    return ops


def _primary_template(graph: dict[str, Any]) -> str:
    md = graph.get("metadata") or {}
    primary = md.get("primary_template")
    if primary:
        return str(primary)
    used = md.get("templates_used") or []
    if isinstance(used, list) and used:
        return str(used[0])
    return ""


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x <= 0.0 or den_y <= 0.0:
        return 0.0
    return num / (den_x * den_y)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze which graphs pass stage 1 versus actually learn induction."
    )
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        WITH latest AS (
            SELECT pr.*,
                   ROW_NUMBER() OVER (
                       PARTITION BY pr.graph_fingerprint
                       ORDER BY pr.timestamp DESC
                   ) AS rn
            FROM program_results pr
            WHERE COALESCE(pr.graph_json, '') NOT IN ('', '{}')
              AND COALESCE(pr.graph_fingerprint, '') <> ''
        )
        SELECT
            latest.graph_fingerprint,
            latest.result_id,
            latest.model_source,
            latest.stage0_passed,
            latest.stage05_passed,
            latest.stage1_passed,
            latest.loss_ratio,
            latest.graph_json,
            latest.graph_category_histogram,
            im.auc,
            im.gap_4,
            im.gap_8,
            im.gap_16,
            im.gap_32,
            im.gap_64,
            im.wall_ms
        FROM latest
        JOIN induction_metrics_v2 im ON im.graph_fingerprint = latest.graph_fingerprint
        WHERE latest.rn = 1
        """
    ).fetchall()
    conn.close()

    quadrant_counts = Counter()
    template_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "s1": 0, "learner": 0, "auc_sum": 0.0}
    )
    op_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "s1": 0, "learner": 0, "auc_sum": 0.0}
    )
    category_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "s1": 0, "learner": 0, "auc_sum": 0.0}
    )

    gap_names = ("gap_4", "gap_8", "gap_16", "gap_32", "gap_64")
    gap_values: dict[str, list[float]] = {name: [] for name in gap_names}
    auc_values: list[float] = []
    per_graph_rows: list[dict[str, Any]] = []

    for row in rows:
        record = dict(row)
        auc = float(record.get("auc") or 0.0)
        s1 = int(bool(record.get("stage1_passed")))
        learner = int(auc >= 0.02)
        quadrant = f"s1_{'pass' if s1 else 'fail'}__ind_{_bucket(auc)}"
        quadrant_counts[quadrant] += 1

        graph = _load_graph(record.get("graph_json"))
        template = _primary_template(graph)
        ops = sorted(set(_iter_ops(graph)))

        if template:
            bucket = template_stats[template]
            bucket["n"] += 1
            bucket["s1"] += s1
            bucket["learner"] += learner
            bucket["auc_sum"] += auc

        for op in ops:
            bucket = op_stats[op]
            bucket["n"] += 1
            bucket["s1"] += s1
            bucket["learner"] += learner
            bucket["auc_sum"] += auc

        cats = record.get("graph_category_histogram")
        if cats:
            try:
                parsed = json.loads(cats)
            except Exception:
                parsed = {}
            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    if not value:
                        continue
                    bucket = category_stats[str(key)]
                    bucket["n"] += 1
                    bucket["s1"] += s1
                    bucket["learner"] += learner
                    bucket["auc_sum"] += auc

        auc_values.append(auc)
        for gap_name in gap_names:
            gap_values[gap_name].append(float(record.get(gap_name) or 0.0))

        per_graph_rows.append(
            {
                "graph_fingerprint": record["graph_fingerprint"],
                "result_id": record["result_id"],
                "model_source": record["model_source"],
                "stage0_passed": int(bool(record["stage0_passed"])),
                "stage05_passed": int(bool(record["stage05_passed"])),
                "stage1_passed": s1,
                "loss_ratio": record["loss_ratio"],
                "auc": auc,
                "learner_bucket": _bucket(auc),
                "is_induction_learner": learner,
                "primary_template": template,
                "op_names": "|".join(ops),
                "graph_category_histogram": record["graph_category_histogram"],
                "gap_4": record["gap_4"],
                "gap_8": record["gap_8"],
                "gap_16": record["gap_16"],
                "gap_32": record["gap_32"],
                "gap_64": record["gap_64"],
                "wall_ms": record["wall_ms"],
            }
        )

    template_rows = [
        {
            "primary_template": name,
            "n": stats["n"],
            "s1_rate": round(stats["s1"] / stats["n"], 4),
            "induction_learner_rate": round(stats["learner"] / stats["n"], 4),
            "avg_auc": round(stats["auc_sum"] / stats["n"], 4),
        }
        for name, stats in template_stats.items()
        if stats["n"] >= 10
    ]
    template_rows.sort(
        key=lambda row: (row["avg_auc"], row["induction_learner_rate"]), reverse=True
    )

    op_rows = [
        {
            "op_name": name,
            "n": stats["n"],
            "s1_rate": round(stats["s1"] / stats["n"], 4),
            "induction_learner_rate": round(stats["learner"] / stats["n"], 4),
            "avg_auc": round(stats["auc_sum"] / stats["n"], 4),
        }
        for name, stats in op_stats.items()
        if stats["n"] >= 20
    ]
    op_rows.sort(
        key=lambda row: (row["avg_auc"], row["induction_learner_rate"]), reverse=True
    )

    category_rows = [
        {
            "category": name,
            "n": stats["n"],
            "s1_rate": round(stats["s1"] / stats["n"], 4),
            "induction_learner_rate": round(stats["learner"] / stats["n"], 4),
            "avg_auc": round(stats["auc_sum"] / stats["n"], 4),
        }
        for name, stats in category_stats.items()
        if stats["n"] >= 20
    ]
    category_rows.sort(
        key=lambda row: (row["avg_auc"], row["induction_learner_rate"]), reverse=True
    )

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "per_graph.csv", per_graph_rows)
    _write_csv(out_dir / "by_template.csv", template_rows)
    _write_csv(out_dir / "by_op.csv", op_rows)
    _write_csv(out_dir / "by_category.csv", category_rows)

    print("quadrants")
    for key in sorted(quadrant_counts):
        print(f"{key}\t{quadrant_counts[key]}")
    print("\ngap_auc_correlation")
    for gap_name in gap_names:
        print(f"{gap_name}\t{_pearson(gap_values[gap_name], auc_values):.4f}")
    print(f"\nout_dir\t{out_dir}")


if __name__ == "__main__":
    main()
