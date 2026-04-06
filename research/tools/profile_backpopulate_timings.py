#!/usr/bin/env python3
"""Profile per-row backpopulate replay timings on a serial CUDA sweep.

Runs the backpopulate tool one row at a time for a selected cohort and writes:
- a full timing TSV for every attempted row
- a slow-row TSV for rows above the chosen percentile or timed out

This is for timeout tuning and operational planning. It does not modify the
training pipeline. Use ``--dry-run`` when you only want timing data.
"""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any


DB_PATH = Path("research/lab_notebook.db")
OUT_DIR = Path("research/reports/backpopulate_timing_sweeps")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile backpopulate row timings")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker-timeout-seconds", type=int, default=600)
    parser.add_argument("--post-train-stability-runs", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-stage1-passed", action="store_true", default=True)
    parser.add_argument("--prefix", default="timing_sweep")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def _select_rows(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    query = """
        SELECT
            pr.result_id,
            pr.graph_fingerprint,
            e.timestamp
        FROM program_results pr
        JOIN experiments e ON e.experiment_id = pr.experiment_id
        WHERE e.experiment_type = 'backfill'
          AND pr.stage0_passed = 1
          AND pr.stage05_passed = 1
          AND pr.stage1_passed = 1
          AND pr.n_train_steps IS NOT NULL
          AND (
            pr.wikitext_perplexity IS NULL OR
            pr.hellaswag_acc IS NULL OR
            pr.induction_auc IS NULL OR
            pr.binding_auc IS NULL OR
            pr.binding_composite IS NULL
          )
        ORDER BY e.timestamp DESC, pr.result_id DESC
        LIMIT ?
    """
    return conn.execute(query, (int(limit),)).fetchall()


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * pct
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(sorted_values[lo])
    weight = rank - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def _read_row_result(report_path: Path) -> tuple[str, str]:
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
    rows = _select_rows(conn, args.limit)
    if not rows:
        print("No candidate rows found.")
        return

    full_path = args.out_dir / f"{args.prefix}.full.tsv"
    slow_path = args.out_dir / f"{args.prefix}.slow.tsv"

    results: list[dict[str, Any]] = []

    with full_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "result_id",
                "graph_fingerprint",
                "elapsed_s",
                "status",
                "error",
                "timed_out",
            ]
        )

        for idx, row in enumerate(rows, start=1):
            result_id = str(row["result_id"])
            report_path = args.out_dir / f"{args.prefix}.{result_id}.tsv"
            cmd = [
                "python",
                "-m",
                "research.tools.backpopulate_screening_metrics",
                "--result-id",
                result_id,
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
                str(report_path),
            ]
            if args.dry_run:
                cmd.append("--dry-run")

            t0 = time.time()
            proc = subprocess.run(
                cmd,
                cwd=str(Path.cwd()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            elapsed_s = round(time.time() - t0, 2)
            status, error = _read_row_result(report_path)
            if proc.returncode != 0 and not error:
                error = (proc.stdout or "").strip().replace("\n", " ")[:500]
            timed_out = int("worker_timeout_after_" in error)
            record = {
                "result_id": result_id,
                "graph_fingerprint": str(row["graph_fingerprint"]),
                "elapsed_s": elapsed_s,
                "status": status,
                "error": error,
                "timed_out": timed_out,
            }
            results.append(record)
            writer.writerow(
                [
                    record["result_id"],
                    record["graph_fingerprint"],
                    record["elapsed_s"],
                    record["status"],
                    record["error"],
                    record["timed_out"],
                ]
            )
            f.flush()
            print(
                f"[{idx}/{len(rows)}] {record['result_id']} "
                f"{record['elapsed_s']}s {record['status']} {record['error']}"
            )

    elapsed_values = sorted(float(r["elapsed_s"]) for r in results)
    median_s = _percentile(elapsed_values, 0.50)
    p75_s = _percentile(elapsed_values, 0.75)
    p90_s = _percentile(elapsed_values, 0.90)

    with slow_path.open("w", newline="", encoding="utf-8") as f:
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
        for record in results:
            if float(record["elapsed_s"]) >= p90_s or int(record["timed_out"]):
                writer.writerow(
                    [
                        record["result_id"],
                        record["graph_fingerprint"],
                        record["elapsed_s"],
                        record["status"],
                        record["error"],
                    ]
                )

    print(f"rows={len(results)}")
    print(f"median_s={median_s:.2f}")
    print(f"p75_s={p75_s:.2f}")
    print(f"p90_s={p90_s:.2f}")
    print(f"full_report={full_path}")
    print(f"slow_report={slow_path}")


if __name__ == "__main__":
    main()
