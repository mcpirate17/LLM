#!/usr/bin/env python3
"""Analyze backpopulate timing sweep structure versus graph features.

Produces:
- per-row feature TSV
- summary JSON with percentile cutoffs
- fast/slow template and op enrichment TSVs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any


DB_PATH = Path("research/lab_notebook.db")
TIMING_PATH = Path(
    "research/reports/backpopulate_timing_sweeps/timing_sweep_100.full.tsv"
)
OUT_DIR = Path("research/reports/backpopulate_timing_clusters")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze backpopulate timing clusters")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--timing-tsv", type=Path, default=TIMING_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--prefix", default="timing_sweep_100")
    return parser.parse_args()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return values[lo]
    weight = rank - lo
    return values[lo] * (1.0 - weight) + values[hi] * weight


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows]
    return round(fmean(vals), 4) if vals else float("nan")


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    timing_rows = list(csv.DictReader(args.timing_tsv.open(), delimiter="\t"))
    by_id = {row["result_id"]: row for row in timing_rows}
    ids = list(by_id)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    query = (
        "SELECT result_id, graph_json, graph_n_ops, graph_n_unique_ops, graph_depth, "
        "param_count, avg_step_time_ms, total_train_time_ms "
        f"FROM program_results WHERE result_id IN ({','.join('?' for _ in ids)})"
    )
    db_rows = {row["result_id"]: row for row in conn.execute(query, ids)}

    feature_rows: list[dict[str, Any]] = []
    op_presence_all: Counter[str] = Counter()
    template_all: Counter[str] = Counter()

    for result_id, timing in by_id.items():
        db_row = db_rows[result_id]
        graph = json.loads(db_row["graph_json"])
        metadata = graph.get("metadata") or {}
        templates = metadata.get("templates_used") or []
        slot_usage = metadata.get("template_slot_usage") or {}
        motifs = metadata.get("motifs_used") or []
        nodes = graph.get("nodes") or {}
        node_values = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
        ops = [
            (node.get("op_name") or node.get("op"))
            for node in node_values
            if not node.get("is_input", False)
        ]
        op_counts = Counter(ops)
        primary_template = templates[0] if templates else ""
        template_all[primary_template] += 1
        for op in set(ops):
            op_presence_all[op] += 1

        feature_rows.append(
            {
                "result_id": result_id,
                "graph_fingerprint": timing["graph_fingerprint"],
                "elapsed_s": float(timing["elapsed_s"]),
                "status": timing["status"],
                "primary_template": primary_template,
                "template_count": len(templates),
                "slot_count": len(slot_usage) if isinstance(slot_usage, dict) else 0,
                "motif_count": len(motifs),
                "graph_n_ops": int(db_row["graph_n_ops"] or len(ops) or 0),
                "graph_n_unique_ops": int(
                    db_row["graph_n_unique_ops"] or len(set(ops)) or 0
                ),
                "graph_depth": int(db_row["graph_depth"] or 0),
                "param_count": int(db_row["param_count"] or 0),
                "avg_step_time_ms": float(db_row["avg_step_time_ms"] or 0.0),
                "total_train_time_ms": float(db_row["total_train_time_ms"] or 0.0),
                "routing_ops": sum(
                    op_counts[op]
                    for op in op_counts
                    if any(
                        token in op
                        for token in (
                            "route",
                            "gate",
                            "moe",
                            "expert",
                            "merge",
                            "sparse_bottleneck_moe",
                        )
                    )
                ),
                "sparse_ops": sum(
                    op_counts[op]
                    for op in op_counts
                    if "sparse" in op or "topk" in op or "ternary" in op
                ),
                "attention_ops": sum(
                    op_counts[op] for op in op_counts if "attention" in op
                ),
                "conv_ops": sum(op_counts[op] for op in op_counts if "conv" in op),
                "norm_ops": sum(op_counts[op] for op in op_counts if "norm" in op),
                "state_ops": sum(
                    op_counts[op]
                    for op in op_counts
                    if "state" in op or "rwkv" in op or "scan" in op
                ),
                "ops_json": json.dumps(ops),
            }
        )

    elapsed_values = sorted(row["elapsed_s"] for row in feature_rows)
    q25 = _percentile(elapsed_values, 0.25)
    q75 = _percentile(elapsed_values, 0.75)
    q90 = _percentile(elapsed_values, 0.90)
    fast = [row for row in feature_rows if row["elapsed_s"] <= q25]
    slow = [row for row in feature_rows if row["elapsed_s"] >= q75]

    feature_path = args.out_dir / f"{args.prefix}.features.tsv"
    with feature_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(feature_rows[0].keys()), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(feature_rows)

    summary = {
        "n_rows": len(feature_rows),
        "q25_s": round(q25, 4),
        "median_s": round(_percentile(elapsed_values, 0.50), 4),
        "q75_s": round(q75, 4),
        "q90_s": round(q90, 4),
        "fast_n": len(fast),
        "slow_n": len(slow),
        "feature_means": {
            key: {
                "fast": _mean(fast, key),
                "slow": _mean(slow, key),
            }
            for key in (
                "elapsed_s",
                "graph_n_ops",
                "graph_n_unique_ops",
                "graph_depth",
                "param_count",
                "avg_step_time_ms",
                "template_count",
                "slot_count",
                "motif_count",
                "routing_ops",
                "sparse_ops",
                "attention_ops",
                "conv_ops",
                "norm_ops",
                "state_ops",
            )
        },
    }
    summary_path = args.out_dir / f"{args.prefix}.summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    def write_enrichment(
        path: Path,
        all_counts: Counter[str],
        subset_counts: Counter[str],
        subset_n: int,
        total_n: int,
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["name", "subset_count", "all_count", "enrichment_ratio"])
            for name, subset_count in subset_counts.most_common():
                all_count = all_counts[name]
                ratio = (
                    ((subset_count / subset_n) / (all_count / total_n))
                    if all_count
                    else 0.0
                )
                writer.writerow([name, subset_count, all_count, round(ratio, 4)])

    fast_templates = Counter(row["primary_template"] for row in fast)
    slow_templates = Counter(row["primary_template"] for row in slow)
    write_enrichment(
        args.out_dir / f"{args.prefix}.fast_templates.tsv",
        template_all,
        fast_templates,
        len(fast),
        len(feature_rows),
    )
    write_enrichment(
        args.out_dir / f"{args.prefix}.slow_templates.tsv",
        template_all,
        slow_templates,
        len(slow),
        len(feature_rows),
    )

    fast_op_presence: Counter[str] = Counter()
    slow_op_presence: Counter[str] = Counter()
    for row in fast:
        for op in set(json.loads(row["ops_json"])):
            fast_op_presence[op] += 1
    for row in slow:
        for op in set(json.loads(row["ops_json"])):
            slow_op_presence[op] += 1
    write_enrichment(
        args.out_dir / f"{args.prefix}.fast_ops.tsv",
        op_presence_all,
        fast_op_presence,
        len(fast),
        len(feature_rows),
    )
    write_enrichment(
        args.out_dir / f"{args.prefix}.slow_ops.tsv",
        op_presence_all,
        slow_op_presence,
        len(slow),
        len(feature_rows),
    )

    print(f"feature_path={feature_path}")
    print(f"summary_path={summary_path}")
    print(f"q25_s={q25:.2f}")
    print(f"q75_s={q75:.2f}")
    print(f"q90_s={q90:.2f}")


if __name__ == "__main__":
    main()
