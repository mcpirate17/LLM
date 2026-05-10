from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from research.scientist.notebook import LabNotebook
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.synthesis.context_rules import find_byte_safety_violations
from research.synthesis.serializer import graph_from_json
from research.tools._candidate_selection import fetch_latest_unique_fingerprint_rows
from research.tools._fingerprint_selection import dedupe_records_by_fingerprint

BASE = Path(__file__).resolve().parents[2]
DB_PATH = BASE / "research/runs.db"
OUT_DIR = BASE / "research/reports/binding_pilot"


def _dedupe_manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return dedupe_records_by_fingerprint(
        rows,
        result_id_key="result_id",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a restartable binding-only S1 pilot with fixed concurrency and VRAM sampling."
    )
    p.add_argument("--prefix", required=True)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sample-seconds", type=float, default=1.0)
    p.add_argument(
        "--order",
        choices=("leaderboard", "newest"),
        default="leaderboard",
        help="How to choose the fixed 20-row manifest.",
    )
    p.add_argument("--reset", action="store_true")
    return p.parse_args()


def _run_dir(prefix: str) -> Path:
    return OUT_DIR / prefix


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _choose_manifest(limit: int, order: str) -> list[dict[str, Any]]:
    nb = LabNotebook(str(DB_PATH))
    try:
        fetch_limit = max(int(limit) * 10, int(limit) + 50)
        if order == "newest":
            rows = fetch_latest_unique_fingerprint_rows(
                nb.conn,
                select_sql="""
                    pr.result_id,
                    pr.graph_fingerprint,
                    pr.graph_json,
                    e.timestamp AS ts,
                    COALESCE(l.composite_score, 0) AS composite_score
                """,
                extra_where_sql="""
                    AND COALESCE(pr.stage1_passed, 0) = 1
                    AND pr.binding_screening_auc IS NULL
                """,
                limit=fetch_limit,
            )
        else:
            # Mask byte-era PPL inline: a row whose
            # screening_wikitext_metric_version != 'bpe_eval_v1' has PPL
            # in different units (typical byte-era values 20–50 sort to
            # the top of an ASC ordering and pollute the pilot pick).
            sql = """
            SELECT pr.result_id, pr.graph_fingerprint, pr.graph_json, e.timestamp AS ts,
                   COALESCE(l.composite_score, 0) AS composite_score,
                   CASE WHEN COALESCE(pr.screening_wikitext_metric_version, '')
                              = 'bpe_eval_v1'
                        THEN pr.wikitext_perplexity END AS bpe_ppl
            FROM leaderboard l
            JOIN program_results_compat pr ON pr.result_id = l.result_id
            JOIN experiments e ON e.experiment_id = pr.experiment_id
            WHERE COALESCE(pr.stage1_passed, 0) = 1
              AND COALESCE(pr.stage0_passed, 0) = 1
              AND COALESCE(pr.stage05_passed, 0) = 1
              AND TRIM(COALESCE(pr.graph_json, '')) <> ''
              AND pr.graph_json <> '{}'
              AND pr.binding_screening_auc IS NULL
            ORDER BY l.composite_score DESC,
                     CASE WHEN bpe_ppl IS NULL THEN 1 ELSE 0 END ASC,
                     bpe_ppl ASC,
                     COALESCE(pr.induction_screening_auc, -1.0) DESC,
                     e.timestamp DESC,
                     pr.result_id DESC
            LIMIT ?
            """
            rows = nb.conn.execute(sql, (fetch_limit,)).fetchall()
        manifest = [
            {
                "result_id": str(row["result_id"]),
                "graph_fingerprint": str(row["graph_fingerprint"] or ""),
                "graph_json": resolve_graph_json_value(
                    nb.conn,
                    nb.db_path,
                    row["graph_json"],
                ),
                "timestamp": float(row["ts"]),
                "composite_score": float(row["composite_score"] or 0.0),
            }
            for row in rows
        ]
        deduped = _dedupe_manifest_rows(manifest)
        eligible = [
            row
            for row in deduped
            if _is_native_binding_eligible(row.get("graph_json", ""))
        ]
        trimmed = [
            {
                "result_id": row["result_id"],
                "graph_fingerprint": row["graph_fingerprint"],
                "timestamp": row["timestamp"],
                "composite_score": row["composite_score"],
            }
            for row in eligible[:limit]
        ]
        return trimmed
    finally:
        nb.close()


def _is_native_binding_eligible(graph_json: str) -> bool:
    payload = str(graph_json or "").strip()
    if not payload or payload == "{}":
        return False
    try:
        graph = graph_from_json(payload)
    except Exception:
        return False
    return not find_byte_safety_violations(graph)


def _ensure_manifest(
    run_dir: Path, *, limit: int, order: str, reset: bool
) -> list[dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    if reset and manifest_path.exists():
        manifest_path.unlink()
    if manifest_path.exists():
        return _read_json(manifest_path)["rows"]
    rows = _choose_manifest(limit, order)
    payload = {
        "created_at": time.time(),
        "db_path": str(DB_PATH),
        "limit": limit,
        "order": order,
        "rows": rows,
    }
    _write_json(manifest_path, payload)
    return rows


def _ensure_results_header(results_path: Path) -> None:
    if results_path.exists():
        return
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "result_id",
                "graph_fingerprint",
                "status",
                "elapsed_s",
                "started_at",
                "finished_at",
                "worker_pid",
                "report_path",
            ]
        )


def _load_done(results_path: Path) -> dict[str, dict[str, str]]:
    if not results_path.exists():
        return {}
    with results_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        done: dict[str, dict[str, str]] = {}
        for row in reader:
            result_id = str(row.get("result_id") or "").strip()
            if not result_id:
                continue
            done[result_id] = row
        return done


def _write_results(results_path: Path, rows: list[dict[str, Any]]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "result_id",
                "graph_fingerprint",
                "status",
                "elapsed_s",
                "started_at",
                "finished_at",
                "worker_pid",
                "report_path",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    row["result_id"],
                    row["graph_fingerprint"],
                    row["status"],
                    row["elapsed_s"],
                    row["started_at"],
                    row["finished_at"],
                    row["worker_pid"],
                    row["report_path"],
                ]
            )


def _store_result(
    results_path: Path,
    done: dict[str, dict[str, str]],
    row: dict[str, Any],
) -> dict[str, str]:
    stored = {k: str(v) for k, v in row.items()}
    done[str(row["result_id"])] = stored
    ordered_rows = sorted(done.values(), key=lambda item: float(item["finished_at"]))
    _write_results(results_path, ordered_rows)
    return stored


def _load_completed_binding_result_ids(result_ids: set[str]) -> set[str]:
    if not result_ids:
        return set()
    nb = LabNotebook(str(DB_PATH))
    try:
        placeholders = ",".join("?" for _ in result_ids)
        rows = nb.conn.execute(
            f"""
            SELECT result_id
            FROM program_results_compat
            WHERE result_id IN ({placeholders})
              AND binding_screening_auc IS NOT NULL
            """,
            tuple(sorted(result_ids)),
        ).fetchall()
        return {str(row["result_id"]) for row in rows}
    finally:
        nb.close()


def _read_worker_report_status(report_path: str) -> str | None:
    path = Path(report_path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        row = next(reader, None)
    if not row:
        return None
    status = str(row.get("status") or "").strip()
    return status or None


def _sample_vram() -> dict[str, Any]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    line = proc.stdout.strip().splitlines()[0]
    ts, name, total, used, util = [part.strip() for part in line.split(",")]
    return {
        "timestamp": ts,
        "gpu_name": name,
        "memory_total_mb": int(float(total)),
        "memory_used_mb": int(float(used)),
        "utilization_gpu": int(float(util)),
    }


def _write_status(
    status_path: Path,
    *,
    manifest: list[dict[str, Any]],
    done: dict[str, dict[str, str]],
    active: dict[str, dict[str, Any]],
    vram_samples: int,
) -> None:
    manifest_by_id = {row["result_id"]: row for row in manifest}
    all_ids = [row["result_id"] for row in manifest]
    remaining = [rid for rid in all_ids if rid not in done and rid not in active]
    payload = {
        "updated_at": time.time(),
        "total": len(all_ids),
        "completed": len(done),
        "active": len(active),
        "remaining": len(remaining),
        "completed_result_ids": sorted(done),
        "active_result_ids": sorted(active),
        "remaining_result_ids": remaining,
        "completed_rows": [
            {
                "result_id": rid,
                "graph_fingerprint": manifest_by_id.get(rid, {}).get(
                    "graph_fingerprint", ""
                ),
                "status": done[rid].get("status", ""),
            }
            for rid in sorted(done)
        ],
        "active_rows": [
            {
                "result_id": rid,
                "graph_fingerprint": active[rid]["result_row"].get(
                    "graph_fingerprint", ""
                ),
                "worker_pid": active[rid]["proc"].pid,
                "started_at": round(float(active[rid]["started_at"]), 3),
            }
            for rid in sorted(active)
        ],
        "remaining_rows": [
            {
                "result_id": rid,
                "graph_fingerprint": manifest_by_id.get(rid, {}).get(
                    "graph_fingerprint", ""
                ),
            }
            for rid in remaining
        ],
        "vram_samples": vram_samples,
    }
    _write_json(status_path, payload)


def _launch_worker(
    run_dir: Path, result_row: dict[str, Any], device: str
) -> dict[str, Any]:
    result_id = result_row["result_id"]
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
        "binding",
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
        "result_row": result_row,
        "started_at": time.time(),
        "report_path": str(per_row_report),
    }


def main() -> int:
    args = _parse_args()
    run_dir = _run_dir(args.prefix)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _ensure_manifest(
        run_dir, limit=args.limit, order=args.order, reset=args.reset
    )
    manifest_ids = {row["result_id"] for row in manifest}
    if not manifest:
        raise SystemExit("No binding candidates found.")

    results_path = run_dir / "results.tsv"
    vram_path = run_dir / "vram.tsv"
    status_path = run_dir / "status.json"
    _ensure_results_header(results_path)
    if args.reset and vram_path.exists():
        vram_path.unlink()
    if not vram_path.exists():
        with vram_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(
                [
                    "sample_time",
                    "gpu_name",
                    "memory_total_mb",
                    "memory_used_mb",
                    "utilization_gpu",
                    "active_workers",
                ]
            )

    done = {
        rid: row for rid, row in _load_done(results_path).items() if rid in manifest_ids
    }
    if done:
        _write_results(
            results_path,
            sorted(done.values(), key=lambda item: float(item["finished_at"])),
        )
    db_completed = _load_completed_binding_result_ids(manifest_ids)
    for result_id in sorted(db_completed):
        if result_id in done:
            continue
        row = next(
            (entry for entry in manifest if entry["result_id"] == result_id), None
        )
        if row is None:
            continue
        _store_result(
            results_path,
            done,
            {
                "result_id": result_id,
                "graph_fingerprint": row["graph_fingerprint"],
                "status": "db_done",
                "elapsed_s": 0.0,
                "started_at": 0.0,
                "finished_at": round(time.time(), 3),
                "worker_pid": "",
                "report_path": "",
            },
        )
    done = {
        rid: row for rid, row in _load_done(results_path).items() if rid in manifest_ids
    }
    pending = deque(row for row in manifest if row["result_id"] not in done)
    active: dict[str, dict[str, Any]] = {}
    vram_samples = 0
    last_sample = 0.0

    try:
        while True:
            while len(active) < args.concurrency:
                while pending and pending[0]["result_id"] in done:
                    pending.popleft()
                if not pending:
                    break
                next_row = pending.popleft()
                active[next_row["result_id"]] = _launch_worker(
                    run_dir, next_row, args.device
                )

            now = time.time()
            if now - last_sample >= args.sample_seconds:
                sample = _sample_vram()
                with vram_path.open("a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow(
                        [
                            sample["timestamp"],
                            sample["gpu_name"],
                            sample["memory_total_mb"],
                            sample["memory_used_mb"],
                            sample["utilization_gpu"],
                            len(active),
                        ]
                    )
                vram_samples += 1
                last_sample = now

            finished: list[str] = []
            for result_id, info in active.items():
                proc = info["proc"]
                rc = proc.poll()
                if rc is None:
                    continue
                finished.append(result_id)
                elapsed_s = round(time.time() - info["started_at"], 3)
                report_status = _read_worker_report_status(info["report_path"])
                status = report_status or ("ok" if rc == 0 else f"exit_{rc}")
                row = {
                    "result_id": result_id,
                    "graph_fingerprint": info["result_row"]["graph_fingerprint"],
                    "status": status,
                    "elapsed_s": elapsed_s,
                    "started_at": round(info["started_at"], 3),
                    "finished_at": round(time.time(), 3),
                    "worker_pid": proc.pid,
                    "report_path": info["report_path"],
                }
                done[result_id] = _store_result(results_path, done, row)

            for result_id in finished:
                active.pop(result_id, None)

            _write_status(
                status_path,
                manifest=manifest,
                done=done,
                active=active,
                vram_samples=vram_samples,
            )

            if len(done) >= len(manifest) and not active:
                break

            time.sleep(0.25)
    except KeyboardInterrupt:
        for info in active.values():
            proc = info["proc"]
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except Exception:
                proc.terminate()
        _write_status(
            status_path,
            manifest=manifest,
            done=done,
            active=active,
            vram_samples=vram_samples,
        )
        raise

    _write_status(
        status_path,
        manifest=manifest,
        done=done,
        active={},
        vram_samples=vram_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
