#!/usr/bin/env python3
"""Concurrent hellaswag backfill using subprocess workers.

Reads a manifest, launches up to --concurrency workers via
backpopulate_screening_metrics --post-train-target hellaswag,
monitors for errors, and writes per-row results + summary.

Usage:
    python -m research.tools.run_hellaswag_backfill \
        --run-dir research/reports/hellaswag_fastpath_missing_c14_2026-04-11 \
        --concurrency 14 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

BASE = Path("/home/tim/Projects/LLM")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Concurrent hellaswag backfill")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--concurrency", type=int, default=14)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sample-seconds", type=float, default=2.0)
    return p.parse_args()


def _launch_worker(run_dir: Path, result_id: str, device: str) -> dict:
    per_row_report = run_dir / "rows" / f"{result_id}.tsv"
    per_row_report.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "research.tools.backpopulate_screening_metrics",
        "--result-id",
        result_id,
        "--device",
        device,
        "--fallback-device",
        "none",
        "--batch-commit",
        "1",
        "--post-train-target",
        "hellaswag",
        "--skip-rapid",
        "--selection-slice",
        "backfill",
        "--report",
        str(per_row_report),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {
        "proc": proc,
        "result_id": result_id,
        "started_at": time.time(),
        "report": str(per_row_report),
    }


def _read_row_status(report_path: str) -> str:
    path = Path(report_path)
    if not path.exists():
        return "no_report"
    try:
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            row = next(reader, None)
        return str(row.get("status", "unknown")).strip() if row else "empty_report"
    except Exception:
        return "read_error"


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}")
        return 1

    manifest = json.loads(manifest_path.read_text())["rows"]
    total = len(manifest)
    print(
        f"Hellaswag backfill: {total} entries, concurrency={args.concurrency}, device={args.device}"
    )

    results_path = run_dir / "results.tsv"
    status_path = run_dir / "status.json"

    # Load already-done
    done: dict[str, dict] = {}
    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                done[row["result_id"]] = row
        print(f"  Resuming: {len(done)} already done")

    # Write results header if new
    if not results_path.exists():
        with results_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter="\t").writerow(
                ["result_id", "graph_fingerprint", "status", "elapsed_s", "worker_pid"]
            )

    pending = deque(r for r in manifest if r["result_id"] not in done)
    active: dict[str, dict] = {}
    errors: list[dict] = []
    updated = 0
    t0 = time.time()

    try:
        while pending or active:
            # Launch workers up to concurrency
            while len(active) < args.concurrency and pending:
                row = pending.popleft()
                rid = row["result_id"]
                info = _launch_worker(run_dir, rid, args.device)
                info["graph_fingerprint"] = row.get("graph_fingerprint", "")
                active[rid] = info

            # Poll for completions
            finished: list[str] = []
            for rid, info in active.items():
                rc = info["proc"].poll()
                if rc is None:
                    continue
                finished.append(rid)
                elapsed = round(time.time() - info["started_at"], 2)
                status = _read_row_status(info["report"])
                if status == "error" or (
                    rc != 0 and status not in ("updated", "no_missing_fields")
                ):
                    status = (
                        f"error_rc{rc}"
                        if status in ("no_report", "empty_report")
                        else status
                    )
                    errors.append(
                        {"result_id": rid, "status": status, "elapsed_s": elapsed}
                    )

                if "updated" in status:
                    updated += 1

                result_row = {
                    "result_id": rid,
                    "graph_fingerprint": info["graph_fingerprint"],
                    "status": status,
                    "elapsed_s": elapsed,
                    "worker_pid": info["proc"].pid,
                }
                done[rid] = result_row

                # Append to results file
                with results_path.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f, delimiter="\t").writerow(
                        [
                            result_row["result_id"],
                            result_row["graph_fingerprint"],
                            result_row["status"],
                            result_row["elapsed_s"],
                            result_row["worker_pid"],
                        ]
                    )

            for rid in finished:
                del active[rid]

            # Progress
            if finished:
                n_done = len(done)
                elapsed_total = time.time() - t0
                rate = n_done / elapsed_total if elapsed_total > 0 else 0
                remaining = total - n_done
                eta = remaining / rate if rate > 0 else 0
                print(
                    f"  [{n_done}/{total}] updated={updated} errors={len(errors)} "
                    f"active={len(active)} rate={rate:.1f}/s ETA={eta:.0f}s",
                    flush=True,
                )

            # Write status
            if finished:
                status_payload = {
                    "updated_at": time.time(),
                    "started_at": t0,
                    "total": total,
                    "completed": len(done),
                    "updated": updated,
                    "errors": len(errors),
                    "active": len(active),
                    "remaining": total - len(done),
                    "error_rows": errors[-20:],  # last 20 errors
                }
                status_path.write_text(json.dumps(status_payload, indent=2))

            if not finished and active:
                time.sleep(0.5)

    except KeyboardInterrupt:
        print(f"\nInterrupted. Killing {len(active)} workers...")
        for info in active.values():
            try:
                info["proc"].kill()
            except ProcessLookupError:
                pass
        return 1

    # Summary
    elapsed_total = time.time() - t0
    summary = {
        "started_at": t0,
        "finished_at": time.time(),
        "wall_time_s": round(elapsed_total, 1),
        "total": total,
        "updated": updated,
        "errors": len(errors),
        "error_rows": errors,
        "mean_row_time_s": round(
            sum(float(d.get("elapsed_s", 0)) for d in done.values())
            / max(len(done), 1),
            3,
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"\nDone: {updated} updated, {len(errors)} errors, {elapsed_total:.1f}s wall time"
    )
    if errors:
        print("Errors:")
        for e in errors[:10]:
            print(f"  {e['result_id']}: {e['status']} ({e['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
