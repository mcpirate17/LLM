#!/usr/bin/env python3
"""Run guarded backpopulate on structural lanes with live progress files."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any


DB_PATH = Path("research/lab_notebook.db")
OUT_DIR = Path("research/reports/backpopulate_lanes")

FAST_TEMPLATES = {
    "latent_attn_sparse_ffn",
    "latent_attn_moe",
    "latent_attn_ffn_block",
    "diff_attn_ffn_block",
    "latent_attn_conv_hybrid",
    "cross_dim_mixer",
    "dual_axis_block",
}

MEDIUM_TEMPLATES = FAST_TEMPLATES | {
    "local_attn_ffn_block",
    "gated_residual",
    "residual_block",
    "token_merge_block",
    "latent_attn_ssm_hybrid",
}

SLOW_OP_BLOCKLIST = {
    "local_window_attn",
    "selective_scan",
    "hetero_moe",
    "arch_router",
    "moe_topk",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run guarded structural-lane backpopulate"
    )
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--prefix", default="lane_run")
    parser.add_argument("--lane", choices=("fast", "medium"), default="fast")
    parser.add_argument(
        "--stage1-cohort",
        choices=("passed", "semantic_failure"),
        default="passed",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker-timeout-seconds", type=int, default=180)
    parser.add_argument("--post-train-stability-runs", type=int, default=2)
    return parser.parse_args()


def _load_candidates(conn: sqlite3.Connection, stage1_cohort: str) -> list[sqlite3.Row]:
    if stage1_cohort == "passed":
        stage1_clause = "pr.stage1_passed = 1"
    else:
        stage1_clause = "(pr.stage1_passed = 0 OR pr.stage1_passed IS NULL)"
    query = """
        SELECT
            pr.result_id,
            pr.graph_fingerprint,
            pr.graph_json,
            pr.graph_n_ops,
            pr.graph_depth,
            pr.param_count,
            e.timestamp
        FROM program_results pr
        JOIN experiments e ON e.experiment_id = pr.experiment_id
        WHERE e.experiment_type = 'backfill'
          AND pr.stage0_passed = 1
          AND pr.stage05_passed = 1
          AND {stage1_clause}
          AND pr.n_train_steps IS NOT NULL
          AND (
            pr.wikitext_perplexity IS NULL OR
            pr.hellaswag_acc IS NULL OR
            pr.induction_auc IS NULL OR
            pr.binding_auc IS NULL OR
            pr.binding_composite IS NULL
          )
        ORDER BY e.timestamp DESC, pr.result_id DESC
    """.format(stage1_clause=stage1_clause)
    return conn.execute(query).fetchall()


def _classify_lane(row: sqlite3.Row, lane: str) -> tuple[bool, str]:
    graph = json.loads(row["graph_json"])
    metadata = graph.get("metadata") or {}
    templates = metadata.get("templates_used") or []
    primary_template = templates[0] if templates else ""
    nodes = graph.get("nodes") or {}
    node_values = list(nodes.values()) if isinstance(nodes, dict) else list(nodes)
    ops = [
        (node.get("op_name") or node.get("op"))
        for node in node_values
        if not node.get("is_input", False)
    ]
    op_set = set(ops)

    allowed_templates = FAST_TEMPLATES if lane == "fast" else MEDIUM_TEMPLATES
    if primary_template not in allowed_templates:
        return False, f"template={primary_template or 'none'}"
    blocked = sorted(op for op in op_set if op in SLOW_OP_BLOCKLIST)
    if blocked:
        return False, f"blocked_ops={','.join(blocked)}"
    graph_n_ops = int(row["graph_n_ops"] or len(ops) or 0)
    graph_depth = int(row["graph_depth"] or 0)
    param_count = int(row["param_count"] or 0)
    if lane == "fast":
        if graph_n_ops > 11:
            return False, f"graph_n_ops={graph_n_ops}"
        if graph_depth > 11:
            return False, f"graph_depth={graph_depth}"
        if param_count > 12_000_000:
            return False, f"param_count={param_count}"
    else:
        if graph_n_ops > 14:
            return False, f"graph_n_ops={graph_n_ops}"
        if graph_depth > 13:
            return False, f"graph_depth={graph_depth}"
        if param_count > 16_000_000:
            return False, f"param_count={param_count}"
    return True, primary_template


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_row_report(report_path: Path) -> tuple[str, str]:
    if not report_path.exists():
        return "missing_report", ""
    lines = report_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return "missing_row", ""
    parts = lines[1].split("\t")
    if len(parts) < 8:
        return "malformed_row", ""
    return parts[6], parts[7]


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    all_rows = _load_candidates(conn, args.stage1_cohort)
    selected: list[sqlite3.Row] = []
    skipped: list[tuple[str, str, str]] = []
    for row in all_rows:
        ok, reason = _classify_lane(row, args.lane)
        if ok:
            selected.append(row)
        else:
            skipped.append(
                (str(row["result_id"]), str(row["graph_fingerprint"]), reason)
            )
    if args.limit > 0:
        selected = selected[: args.limit]

    summary_path = args.out_dir / f"{args.prefix}.summary.tsv"
    status_path = args.out_dir / f"{args.prefix}.status.json"
    selected_path = args.out_dir / f"{args.prefix}.selected.tsv"
    skipped_path = args.out_dir / f"{args.prefix}.skipped.tsv"

    with selected_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["result_id", "graph_fingerprint"])
        for row in selected:
            writer.writerow([row["result_id"], row["graph_fingerprint"]])

    with skipped_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["result_id", "graph_fingerprint", "reason"])
        writer.writerows(skipped)

    counts = Counter()
    t0 = time.time()

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "result_id",
                "graph_fingerprint",
                "elapsed_s",
                "status",
                "error",
            ]
        )
        for idx, row in enumerate(selected, start=1):
            rid = str(row["result_id"])
            fp = str(row["graph_fingerprint"])
            row_report = args.out_dir / f"{args.prefix}.{rid}.tsv"
            row_log = args.out_dir / f"{args.prefix}.{rid}.log"
            _write_status(
                status_path,
                {
                    "phase": "running",
                    "total_selected": len(selected),
                    "completed": idx - 1,
                    "updated": counts["updated"],
                    "error": counts["error"],
                    "current_index": idx,
                    "current_result_id": rid,
                    "current_graph_fingerprint": fp,
                    "selected_file": str(selected_path),
                    "skipped_file": str(skipped_path),
                    "lane": args.lane,
                    "stage1_cohort": args.stage1_cohort,
                    "summary_file": str(summary_path),
                    "started_at_epoch_s": t0,
                    "elapsed_total_s": round(time.time() - t0, 2),
                },
            )
            cmd = [
                "python",
                "-m",
                "research.tools.backpopulate_screening_metrics",
                "--result-id",
                rid,
                "--device",
                str(args.device),
                "--fallback-device",
                "none",
                "--batch-commit",
                "1",
                "--post-train-stability-runs",
                str(args.post_train_stability_runs),
                "--allow-insufficient-learning-metrics",
                "--worker-timeout-seconds",
                str(args.worker_timeout_seconds),
                "--report",
                str(row_report),
            ]
            row_t0 = time.time()
            with row_log.open("w", encoding="utf-8") as log_f:
                proc = subprocess.run(
                    cmd,
                    cwd=str(Path.cwd()),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            elapsed_s = round(time.time() - row_t0, 2)
            status, error = _read_row_report(row_report)
            if proc.returncode != 0 and not error:
                error = (
                    row_log.read_text(encoding="utf-8").strip().replace("\n", " ")[:500]
                )
            counts[status] += 1
            writer.writerow([rid, fp, elapsed_s, status, error])
            f.flush()
            print(f"[{idx}/{len(selected)}] {rid} {elapsed_s}s {status} {error}")

    _write_status(
        status_path,
        {
            "phase": "completed",
            "total_selected": len(selected),
            "completed": len(selected),
            "updated": counts["updated"],
            "error": counts["error"],
            "selected_file": str(selected_path),
            "skipped_file": str(skipped_path),
            "lane": args.lane,
            "stage1_cohort": args.stage1_cohort,
            "summary_file": str(summary_path),
            "elapsed_total_s": round(time.time() - t0, 2),
        },
    )
    print(f"selected={len(selected)} skipped={len(skipped)}")
    print(f"summary={summary_path}")
    print(f"status={status_path}")


if __name__ == "__main__":
    main()
