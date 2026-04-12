from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from research.scientist.notebook import LabNotebook

BASE = Path("/home/tim/Projects/LLM")
DB_PATH = BASE / "research/lab_notebook.db"
OUT_DIR = BASE / "research/reports/backpopulate_lanes"


@dataclass(frozen=True)
class RowCandidate:
    result_id: str
    graph_fingerprint: str
    timestamp: float
    avg_step_time_ms: float | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sequential S1 backfill runner with heartbeat and fingerprint defer."
    )
    p.add_argument("--prefix", default="s1_newest_to_oldest_run_2")
    p.add_argument("--timeout-seconds", default="null")
    p.add_argument("--heartbeat-seconds", type=float, default=5.0)
    p.add_argument("--defer-fingerprint-after-timeouts", type=int, default=2)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument(
        "--post-train-target",
        default="full",
        choices=["full", "one", "two", "all", "hellaswag", "binding", "induction"],
    )
    p.add_argument("--distinct-fingerprint", action="store_true")
    p.add_argument(
        "--order",
        default="newest_to_oldest",
        choices=["newest_to_oldest", "fastest_known"],
    )
    return p.parse_args()


def parse_optional_timeout(raw: str) -> int | None:
    text = str(raw).strip().lower()
    if text in {"", "none", "null"}:
        return None
    parsed = int(text)
    if parsed <= 0:
        return None
    return parsed


def write_tsv(path: Path, rows: list[RowCandidate]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["result_id", "graph_fingerprint", "timestamp", "avg_step_time_ms"])
        for r in rows:
            w.writerow(
                [r.result_id, r.graph_fingerprint, r.timestamp, r.avg_step_time_ms]
            )


def write_status(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def target_missing_clause(target: str) -> str:
    normalized = {"one": "hellaswag", "two": "binding"}.get(target, target)
    if normalized == "hellaswag":
        return "pr.hellaswag_acc IS NULL"
    if normalized == "induction":
        return "pr.induction_auc IS NULL"
    if normalized == "binding":
        return "pr.binding_auc IS NULL"
    if normalized == "all":
        return "(pr.hellaswag_acc IS NULL OR pr.induction_auc IS NULL OR pr.binding_auc IS NULL OR pr.binding_composite IS NULL)"
    return (
        "("
        "pr.wikitext_perplexity IS NULL OR "
        "pr.hellaswag_acc IS NULL OR "
        "pr.induction_auc IS NULL OR "
        "pr.binding_auc IS NULL OR "
        "pr.binding_composite IS NULL"
        ")"
    )


def fetch_candidates(
    max_rows: int,
    post_train_target: str,
    distinct_fingerprint: bool,
    order: str,
) -> list[RowCandidate]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        select_prefix = """
        SELECT pr.result_id, pr.graph_fingerprint, e.timestamp AS row_timestamp, pr.avg_step_time_ms
        """
        if distinct_fingerprint:
            select_prefix = """
            SELECT result_id, graph_fingerprint, row_timestamp, avg_step_time_ms FROM (
              SELECT
                pr.result_id,
                pr.graph_fingerprint,
                e.timestamp AS row_timestamp,
                pr.avg_step_time_ms,
                ROW_NUMBER() OVER (
                  PARTITION BY pr.graph_fingerprint
                  ORDER BY
                    COALESCE(pr.avg_step_time_ms, 1e18) ASC,
                    e.timestamp DESC,
                    pr.result_id DESC
                ) AS fp_rank
            """
        sql = (
            select_prefix
            + """
            FROM program_results pr
            JOIN experiments e ON e.experiment_id = pr.experiment_id
            WHERE pr.stage0_passed = 1
              AND pr.stage05_passed = 1
              AND pr.n_train_steps IS NOT NULL
              AND pr.stage1_passed = 1
              AND """
            + target_missing_clause(post_train_target)
        )
        if distinct_fingerprint:
            sql += """
            ) WHERE fp_rank = 1
            """
        if order == "fastest_known":
            sql += """
            ORDER BY COALESCE(avg_step_time_ms, 1e18) ASC, row_timestamp DESC, result_id DESC
            """
        else:
            sql += """
            ORDER BY row_timestamp DESC, result_id DESC
            """
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    out = [
        RowCandidate(
            result_id=str(r["result_id"]),
            graph_fingerprint=str(r["graph_fingerprint"]),
            timestamp=float(r["row_timestamp"]),
            avg_step_time_ms=(
                None if r["avg_step_time_ms"] is None else float(r["avg_step_time_ms"])
            ),
        )
        for r in rows
    ]
    if max_rows > 0:
        out = out[:max_rows]
    return out


def parse_row_report(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "error", "missing_row_report"
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rec = next(reader, None)
    if rec is None:
        return "error", "empty_row_report"
    status = rec.get("status") or "error"
    err = rec.get("error") or ""
    return status, err


def is_timeout_error(status: str, error: str) -> bool:
    return status == "error" and error.startswith("worker_timeout_after_")


def main() -> int:
    args = parse_args()
    timeout_seconds = parse_optional_timeout(args.timeout_seconds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix
    status_path = OUT_DIR / f"{prefix}.status.json"
    summary_path = OUT_DIR / f"{prefix}.summary.tsv"
    selected_path = OUT_DIR / f"{prefix}.selected.tsv"
    deferred_path = OUT_DIR / f"{prefix}.deferred.tsv"
    launch_log = OUT_DIR / f"{prefix}.launch.log"

    initial_rows = fetch_candidates(
        args.max_rows,
        post_train_target=args.post_train_target,
        distinct_fingerprint=bool(args.distinct_fingerprint),
        order=args.order,
    )
    active_queue = list(initial_rows)
    deferred_queue: list[RowCandidate] = []
    deferred_ids: set[str] = set()
    fp_timeout_counts: dict[str, int] = {}

    write_tsv(selected_path, initial_rows)
    write_tsv(deferred_path, [])
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["result_id", "graph_fingerprint", "elapsed_s", "status", "error"])

    started = time.time()
    completed = 0
    updated = 0
    errors = 0
    timeouts = 0
    global_index = 0
    nb = LabNotebook(str(DB_PATH))
    audit_exp_id = nb.start_experiment(
        experiment_type="screening_backpopulate",
        config={
            "prefix": prefix,
            "timeout_seconds": timeout_seconds,
            "heartbeat_seconds": float(args.heartbeat_seconds),
            "defer_fingerprint_after_timeouts": int(
                args.defer_fingerprint_after_timeouts
            ),
            "max_rows": int(args.max_rows),
            "post_train_target": str(args.post_train_target),
            "distinct_fingerprint": bool(args.distinct_fingerprint),
            "order": str(args.order),
            "device": "cuda",
            "source_script": "run_s1_backpopulate",
        },
        hypothesis=f"S1 backpopulate run {prefix} target={args.post_train_target}",
    )

    try:
        with launch_log.open("a", encoding="utf-8") as log:
            while active_queue:
                row = active_queue.pop(0)
                global_index += 1
                rid = row.result_id
                fp = row.graph_fingerprint
                row_report = OUT_DIR / f"{prefix}.{rid}.tsv"
                row_log = OUT_DIR / f"{prefix}.{rid}.log"

                cmd = [
                    "python",
                    "-m",
                    "research.tools.backpopulate_screening_metrics",
                    "--result-id",
                    rid,
                    "--device",
                    "cuda",
                    "--fallback-device",
                    "none",
                    "--batch-commit",
                    "1",
                    "--post-train-stability-runs",
                    "2",
                    "--post-train-target",
                    str(args.post_train_target),
                    "--allow-insufficient-learning-metrics",
                    "--audit-prefix",
                    prefix,
                    "--audit-experiment-id",
                    audit_exp_id,
                    "--audit-source-script",
                    "run_s1_backpopulate",
                    "--report",
                    str(row_report.relative_to(BASE)),
                ]
                if timeout_seconds is not None:
                    cmd.extend(["--worker-timeout-seconds", str(timeout_seconds)])
                else:
                    cmd.extend(["--worker-timeout-seconds", "none"])

                t0 = time.time()
                with row_log.open("w", encoding="utf-8") as row_out:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=BASE,
                        stdout=row_out,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    while True:
                        ret = proc.poll()
                        now = time.time()
                        write_status(
                            status_path,
                            {
                                "phase": "running",
                                "experiment_id": audit_exp_id,
                                "cohort": "stage1_passed",
                                "order": "newest_to_oldest",
                                "post_train_target": args.post_train_target,
                                "total_selected": len(initial_rows),
                                "completed": completed,
                                "updated": updated,
                                "error": errors,
                                "timeout": timeouts,
                                "current_index": global_index,
                                "current_result_id": rid,
                                "current_graph_fingerprint": fp,
                                "current_worker_pid": proc.pid,
                                "current_row_started_at_epoch_s": t0,
                                "current_row_elapsed_s": round(now - t0, 2),
                                "current_row_log_size_bytes": row_log.stat().st_size
                                if row_log.exists()
                                else 0,
                                "deferred_remaining": len(deferred_queue),
                                "active_remaining": len(active_queue),
                                "selected_file": str(selected_path.relative_to(BASE)),
                                "summary_file": str(summary_path.relative_to(BASE)),
                                "deferred_file": str(deferred_path.relative_to(BASE)),
                                "started_at_epoch_s": started,
                                "elapsed_total_s": round(now - started, 2),
                                "last_heartbeat_epoch_s": now,
                            },
                        )
                        if ret is not None:
                            break
                        time.sleep(args.heartbeat_seconds)

                elapsed = round(time.time() - t0, 2)
                status, err = parse_row_report(row_report)
                summary_status = status
                if is_timeout_error(status, err):
                    summary_status = "timeout"

                if status == "updated":
                    updated += 1
                    err = ""
                elif summary_status == "timeout":
                    timeouts += 1
                else:
                    errors += 1
                completed += 1

                with summary_path.open("a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow([rid, fp, elapsed, summary_status, err])
                log.write(
                    f"[{completed}/{len(initial_rows)}] {rid} {fp} {elapsed}s {summary_status} {err}\n"
                )
                log.flush()

                if is_timeout_error(status, err):
                    fp_timeout_counts[fp] = fp_timeout_counts.get(fp, 0) + 1
                    if fp_timeout_counts[fp] >= args.defer_fingerprint_after_timeouts:
                        keep: list[RowCandidate] = []
                        moved = False
                        for candidate in active_queue:
                            if candidate.graph_fingerprint == fp:
                                deferred_queue.append(candidate)
                                deferred_ids.add(candidate.result_id)
                                moved = True
                            else:
                                keep.append(candidate)
                        if moved:
                            active_queue = keep
                            write_tsv(deferred_path, deferred_queue)
                            log.write(
                                f"deferred fingerprint {fp} after {fp_timeout_counts[fp]} timeouts; "
                                f"moved {sum(1 for c in deferred_queue if c.graph_fingerprint == fp)} rows\n"
                            )
                            log.flush()

                if not active_queue and deferred_queue:
                    active_queue = deferred_queue
                    deferred_queue = []
                    write_tsv(deferred_path, deferred_queue)
                    log.write("resuming deferred queue\n")
                    log.flush()

            write_status(
                status_path,
                {
                    "phase": "completed",
                    "experiment_id": audit_exp_id,
                    "cohort": "stage1_passed",
                    "order": "newest_to_oldest",
                    "post_train_target": args.post_train_target,
                    "total_selected": len(initial_rows),
                    "completed": completed,
                    "updated": updated,
                    "error": errors,
                    "timeout": timeouts,
                    "selected_file": str(selected_path.relative_to(BASE)),
                    "summary_file": str(summary_path.relative_to(BASE)),
                    "deferred_file": str(deferred_path.relative_to(BASE)),
                    "started_at_epoch_s": started,
                    "elapsed_total_s": round(time.time() - started, 2),
                },
            )
        nb.complete_experiment(
            audit_exp_id,
            results={
                "total_selected": len(initial_rows),
                "completed": completed,
                "updated": updated,
                "errors": errors,
                "timeouts": timeouts,
                "prefix": prefix,
                "post_train_target": str(args.post_train_target),
            },
            aria_summary=(
                f"S1 backpopulate {prefix}: updated={updated} "
                f"errors={errors} timeouts={timeouts}"
            ),
        )
    except Exception as exc:
        write_status(
            status_path,
            {
                "phase": "failed",
                "experiment_id": audit_exp_id,
                "cohort": "stage1_passed",
                "order": "newest_to_oldest",
                "post_train_target": args.post_train_target,
                "total_selected": len(initial_rows),
                "completed": completed,
                "updated": updated,
                "error": errors + 1,
                "timeout": timeouts,
                "selected_file": str(selected_path.relative_to(BASE)),
                "summary_file": str(summary_path.relative_to(BASE)),
                "deferred_file": str(deferred_path.relative_to(BASE)),
                "started_at_epoch_s": started,
                "elapsed_total_s": round(time.time() - started, 2),
                "failure_reason": str(exc),
            },
        )
        nb.fail_experiment(audit_exp_id, error=str(exc))
        raise
    finally:
        nb.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
