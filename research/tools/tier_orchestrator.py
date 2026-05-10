#!/usr/bin/env python3
"""Autonomously re-runs s1/investigation/capability_ranking/validation tiers
for a list of program result_ids, monitoring each via the dashboard API.

Usage:
    python -m research.tools.tier_orchestrator \
        --target 54b0557c-472 \
        --backfill <r1> <r2> ...
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "http://localhost:5000"
DB_PATH = Path("/home/tim/Projects/LLM/research/runs.db")
LOG_PATH = Path("/home/tim/Projects/LLM/research/reports/orchestrator/orchestrator.log")
STATUS_PATH = Path(
    "/home/tim/Projects/LLM/research/reports/orchestrator/orchestrator.status.json"
)

STAGES = ("screening", "investigation", "capability_ranking", "validation")
PER_RUN_TIMEOUT_S = 3600  # 1 hour cap per single tier run
START_WAIT_TIMEOUT_S = 90  # how long to wait for the runner to *start* after drain
DRAIN_RETRY_S = 30  # if drain returns busy, retry after this delay
POLL_S = 8


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def write_status(state: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(state, indent=2))


def http(method: str, path: str, body=None, timeout: int = 30):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(
        f"{API}{path}", data=data, method=method, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode() or "{}"
            return resp.status, json.loads(raw) if raw.strip().startswith(
                ("{", "[")
            ) else {"raw": raw}
    except urllib.error.HTTPError as e:
        body_raw = ""
        try:
            body_raw = e.read().decode()
        except Exception:
            pass
        try:
            return e.code, json.loads(body_raw or "{}")
        except Exception:
            return e.code, {"error": str(e), "body": body_raw[:500]}
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        return -1, {"error": f"transport: {e}"}


def is_runner_busy() -> bool:
    code, payload = http("GET", "/api/system/status")
    if code != 200:
        return False  # treat unreachable as not busy; caller will retry queue
    return bool(payload.get("is_running"))


def wait_for_state(want_busy: bool, timeout_s: int) -> bool:
    t0 = time.time()
    while True:
        if is_runner_busy() == want_busy:
            return True
        if time.time() - t0 > timeout_s:
            return False
        time.sleep(POLL_S)


def queue_rerun(result_id: str, stage: str):
    return http(
        "POST",
        f"/api/programs/{result_id}/queue-validation-rerun",
        {"stage": stage, "n": 1, "reason": f"orchestrator:{stage}"},
    )


def drain(result_id: str):
    # Drain synchronously initializes the run before returning, which can
    # take >30s. Use a long timeout; on transport timeout the run is often
    # already started and check_task_status() will confirm.
    return http(
        "POST",
        "/api/runner/drain-pending-validation-rerun",
        {"result_id": result_id},
        timeout=180,
    )


def check_task_status(task_id: str) -> str | None:
    """Return current status of a followup_task, or None if not found."""
    if not task_id:
        return None
    with sqlite3.connect(DB_PATH) as c:
        row = c.execute(
            "SELECT status FROM followup_tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
    return row[0] if row else None


def get_fp_for_result(result_id: str) -> str | None:
    with sqlite3.connect(DB_PATH) as c:
        row = c.execute(
            "SELECT graph_fingerprint FROM program_results_compat WHERE result_id = ?",
            (result_id,),
        ).fetchone()
    return row[0] if row else None


def count_program_results(fp: str) -> int:
    with sqlite3.connect(DB_PATH) as c:
        n = c.execute(
            "SELECT COUNT(*) FROM program_results_compat WHERE graph_fingerprint = ?",
            (fp,),
        ).fetchone()[0]
    return int(n)


def write_live_feed_entry(
    *,
    fp: str,
    result_id: str,
    stage: str,
    elapsed_s: int,
    new_rows: int,
    snap: dict,
) -> None:
    """Write a 'live_feed' entry into the notebook so dashboard's LiveFeed
    panel surfaces orchestrator-driven tier reruns alongside continuous-mode
    cycle entries.
    """
    import uuid

    entry_id = uuid.uuid4().hex[:24]
    payload = {
        "live_feed_type": "tier_rerun",
        "stage": stage,
        "fingerprint": fp,
        "result_id": result_id,
        "elapsed_s": elapsed_s,
        "new_rows": new_rows,
        "tier": snap.get("tier"),
        "composite_score": snap.get("composite_score"),
        "induction_intermediate_auc": snap.get("induction_intermediate_auc"),
        "binding_intermediate_auc": snap.get("binding_intermediate_auc"),
    }
    title = f"Orchestrator: {stage} on fp={fp[:12]}"
    score = snap.get("composite_score")
    score_text = f" composite={score:.1f}" if isinstance(score, (int, float)) else ""
    content = (
        f"{stage} rerun completed for result_id={result_id} "
        f"in {elapsed_s}s, +{new_rows} program_results row(s);"
        f"{score_text} tier={snap.get('tier')}."
    )
    try:
        with sqlite3.connect(DB_PATH, timeout=10) as c:
            c.execute(
                "INSERT INTO entries (entry_id, timestamp, entry_type, title, "
                "content, metadata_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    entry_id,
                    time.time(),
                    "live_feed",
                    title,
                    content,
                    json.dumps({"payload": payload}, sort_keys=True),
                ),
            )
    except sqlite3.OperationalError as exc:
        log(f"     WARN: live_feed insert failed: {exc}")


def leaderboard_snapshot(fp: str) -> dict:
    with sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        r = c.execute(
            "SELECT result_id, tier, composite_score, induction_intermediate_auc, "
            "binding_intermediate_auc FROM leaderboard "
            "WHERE graph_fingerprint = ? LIMIT 1",
            (fp,),
        ).fetchone()
    return dict(r) if r else {}


def run_tier(result_id: str, stage: str, fp: str) -> dict:
    """Queue and execute one tier rerun, return outcome."""
    log(f"  -> {stage} on result_id={result_id} (fp={fp[:16]})")
    write_status({"phase": "wait_idle_pre", "result_id": result_id, "stage": stage})
    # Wait for runner idle before queueing.
    if not wait_for_state(False, timeout_s=PER_RUN_TIMEOUT_S):
        return {"ok": False, "stage": stage, "reason": "runner_never_idle"}

    pre_n = count_program_results(fp)

    code, q = queue_rerun(result_id, stage)
    if code >= 400:
        log(f"     QUEUE FAIL {code}: {json.dumps(q)[:300]}")
        return {"ok": False, "stage": stage, "reason": f"queue_{code}", "body": q}
    task_ids = q.get("task_ids") or []
    log(f"     queued task_ids={task_ids}")

    # drain (with retries on transient busy / transport-timeout). The drain
    # endpoint synchronously starts the run before responding, so an HTTP
    # timeout does not mean failure — check the task's DB status instead.
    drain_attempts = 0
    d: dict = {}
    while True:
        drain_attempts += 1
        code, d = drain(result_id)
        if code == 200 and d.get("status") == "launched":
            log(
                f"     launched stage={d.get('stage')} exp={d.get('running_experiment_id')}"
            )
            break
        # Transport timeout / 5xx / idle: peek at the queued task's DB status.
        # If it has moved out of 'queued', the run is in flight.
        my_task = task_ids[0] if task_ids else None
        ts = check_task_status(my_task) if my_task else None
        if ts and ts != "queued":
            log(
                f"     drain {code} status={d.get('status') or d.get('error')!r} "
                f"BUT task {my_task} is now status={ts!r} — treat as launched"
            )
            d.setdefault("running_experiment_id", None)
            break
        if code == 200 and d.get("status") == "idle":
            log("     drain idle (no task matched) — retry queue/drain")
            return {"ok": False, "stage": stage, "reason": "drain_idle"}
        log(
            f"     drain {code} status={d.get('status') or d.get('error')!r} "
            f"task_status={ts!r} attempt={drain_attempts}"
        )
        if drain_attempts >= 5:
            return {"ok": False, "stage": stage, "reason": "drain_failed", "body": d}
        time.sleep(DRAIN_RETRY_S)

    # Wait for run to actually start
    if not wait_for_state(True, START_WAIT_TIMEOUT_S):
        log("     runner never went busy after drain; possibly already finished")

    # Wait for completion
    write_status(
        {
            "phase": "running",
            "result_id": result_id,
            "stage": stage,
            "started_at": time.time(),
            "running_experiment_id": d.get("running_experiment_id"),
        }
    )
    t0 = time.time()
    if not wait_for_state(False, PER_RUN_TIMEOUT_S):
        return {"ok": False, "stage": stage, "reason": "run_timeout"}
    elapsed = int(time.time() - t0)

    post_n = count_program_results(fp)
    new_rows = post_n - pre_n
    snap = leaderboard_snapshot(fp)
    log(
        f"     <- {stage} done in {elapsed}s; new rows={new_rows}; "
        f"tier={snap.get('tier')} composite={snap.get('composite_score')}"
    )
    write_live_feed_entry(
        fp=fp,
        result_id=result_id,
        stage=stage,
        elapsed_s=elapsed,
        new_rows=new_rows,
        snap=snap,
    )
    return {
        "ok": True,
        "stage": stage,
        "elapsed_s": elapsed,
        "new_rows": new_rows,
        "lb": snap,
    }


def process_fp(result_id: str, label: str = "") -> dict:
    fp = get_fp_for_result(result_id)
    if not fp:
        log(f"=== SKIP {result_id} {label}: not in DB ===")
        return {"result_id": result_id, "ok": False, "reason": "not_in_db"}
    log(f"=== START {result_id} {label} fp={fp[:16]} ===")
    pre = leaderboard_snapshot(fp)
    log(f"  pre lb: tier={pre.get('tier')} composite={pre.get('composite_score')}")
    outcomes = []
    for stage in STAGES:
        out = run_tier(result_id, stage, fp)
        outcomes.append(out)
        if not out["ok"]:
            log(f"=== ABORT {result_id} at {stage}: {out.get('reason')} ===")
            return {"result_id": result_id, "fp": fp, "ok": False, "outcomes": outcomes}
    post = leaderboard_snapshot(fp)
    log(
        f"=== DONE {result_id} {label}: tier {pre.get('tier')} -> {post.get('tier')} "
        f"composite {pre.get('composite_score')} -> {post.get('composite_score')} ==="
    )
    return {
        "result_id": result_id,
        "fp": fp,
        "ok": True,
        "outcomes": outcomes,
        "pre_lb": pre,
        "post_lb": post,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--backfill", nargs="*", default=[])
    args = parser.parse_args()

    log(f"orchestrator start: target={args.target} backfill_n={len(args.backfill)}")
    overall = {"target": None, "backfill": []}
    target = process_fp(args.target, label="(target)")
    overall["target"] = target
    write_status({"phase": "target_done", "result": target})
    if not target["ok"]:
        log("TARGET FAILED — stopping before backfill batch")
        write_status({"phase": "stopped_target_failed", "result": target})
        return 1

    for i, rid in enumerate(args.backfill, 1):
        log(f"--- backfill {i}/{len(args.backfill)}: {rid} ---")
        out = process_fp(rid, label=f"(backfill {i}/{len(args.backfill)})")
        overall["backfill"].append(out)
        write_status(
            {
                "phase": "backfill_progress",
                "i": i,
                "n": len(args.backfill),
                "result": out,
            }
        )

    write_status({"phase": "done", "overall": overall})
    log("orchestrator complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
