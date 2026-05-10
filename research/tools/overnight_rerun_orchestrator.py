"""Overnight orchestrator: queue 2x (screening + investigation + validation)
reruns for the top-10 leaderboard fps, drain sequentially, monitor outcomes,
document findings, then call happy_times.py to shut down.

Designed to run unattended.  Logs to research/perf_artifacts/overnight_*.log
and writes a findings markdown to the same directory.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

DB = "/home/tim/Projects/LLM/research/runs.db"
DASHBOARD_BASE = "http://localhost:5000"
ARTIFACT_DIR = Path("/home/tim/Projects/LLM/research/perf_artifacts")
HAPPY_TIMES = "/home/tim/Projects/LLM/happy_times.py"
TOP_N = 10
RERUNS_PER_STAGE = 2
# Screening (replay path) is DISABLED for stability reruns: it calls
# nb.merge_program_result_patch on the source row instead of adding a
# new sample, which CORRUPTS the original data point rather than
# growing the CV pool.  Investigation and validation paths use
# nb.record_program_result, which correctly adds new rows.  See
# dashboard_orchestrator.py:439 for the merge code, _helpers_benchmark
# .py:754 for the correct record path.
STAGES = ("investigation", "validation")
STAGE_RUNNER = {"investigation": "investigation", "validation": "validation"}

# Phase 2: scaling probe — does training longer at validation tier
# improve the score for these fingerprints?  Tim's hypothesis: some
# archs (SSM-like, conv-based) need more steps to form induction
# heads; others (attention-rich) saturate earlier.  Run one validation
# rerun at SCALING_STEPS for each top-N fp and compare.
SCALING_STEPS = 20000  # 2× the default 10K validation budget
SCALING_TIMEOUT_SEC = 60 * 90  # 90 min cap; 20K × 1 seed runs ~45-60
# Per-stage timeouts (max wall-clock for a single task before we move on).
STAGE_TIMEOUT_SEC = {
    "screening": 60 * 15,  # S1 at 750 steps: ~5 min, give 15
    "investigation": 60 * 30,  # 2500 steps: ~10-15 min, give 30
    "validation": 60 * 60,  # 10k × 1 seed: ~30-45 min, give 60
}
# How long to wait before declaring a drain "didn't claim anything"
# vs the runner just took a moment.
DRAIN_RETRY_SEC = 30
# Backoff when no eligible task in queue.
IDLE_SLEEP_SEC = 10

# Logging
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
LOG_PATH = ARTIFACT_DIR / f"overnight_{stamp}.log"
FINDINGS_PATH = ARTIFACT_DIR / f"overnight_findings_{stamp}.md"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("overnight")


def _ro_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_top_n(n: int) -> list[dict]:
    conn = _ro_conn()
    rows = conn.execute(
        """SELECT graph_fingerprint AS fp, result_id, tier,
                  composite_score, n_runs
           FROM leaderboard
           WHERE COALESCE(is_reference, 0) = 0
             AND tier IN ('screening','investigation','validation','breakthrough')
             AND graph_fingerprint IS NOT NULL
           ORDER BY composite_score DESC
           LIMIT ?""",
        (n,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def queue_one(
    result_id: str, stage: str, n_steps: Optional[int] = None
) -> Optional[str]:
    body: dict[str, Any] = {"stage": stage, "n": 1, "reason": "overnight_orchestrator"}
    if n_steps is not None:
        body["n_steps"] = int(n_steps)
    try:
        r = requests.post(
            f"{DASHBOARD_BASE}/api/programs/{result_id}/queue-validation-rerun",
            json=body,
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        ids = data.get("task_ids") or []
        return ids[0] if ids else None
    except Exception as exc:
        logger.error("queue_one(%s, %s) failed: %s", result_id[:12], stage, exc)
        return None


def queue_all(top: list[dict]) -> list[dict]:
    """Queue 2 reruns per stage for each fp.  Returns task records."""
    queued: list[dict] = []
    for entry in top:
        rid = str(entry["result_id"])
        fp = str(entry["fp"])
        for stage in STAGES:
            for i in range(RERUNS_PER_STAGE):
                tid = queue_one(rid, stage)
                if tid:
                    queued.append(
                        {
                            "task_id": tid,
                            "fp": fp,
                            "result_id": rid,
                            "stage": stage,
                            "rerun_index": i + 1,
                        }
                    )
                else:
                    logger.warning(
                        "failed to queue %s rerun %d for %s", stage, i + 1, fp
                    )
    logger.info(
        "queued %d total tasks across %d fps × %d stages × %d reruns",
        len(queued),
        len(top),
        len(STAGES),
        RERUNS_PER_STAGE,
    )
    return queued


def runner_busy() -> bool:
    try:
        r = requests.get(f"{DASHBOARD_BASE}/api/aria/cycle-status", timeout=5)
        if r.status_code == 200:
            return bool(r.json().get("is_running"))
    except Exception:
        pass
    return False


def drain_once() -> Optional[dict]:
    """Try to drain one task.  Returns response dict or None on error.

    Timeout is generous (180s) because replay drains run the
    exact_graph_replay synchronously inside the request handler — they
    don't return until the replay thread joins, which can take up to a
    minute for non-fast replays.  Investigation / validation drains
    return immediately (their experiments run in background threads).
    """
    try:
        r = requests.post(
            f"{DASHBOARD_BASE}/api/runner/drain-pending-validation-rerun",
            timeout=180,
        )
        if r.status_code == 200:
            return r.json()
        if r.status_code == 409:
            return {"status": "busy"}
        logger.error("drain returned %d: %s", r.status_code, r.text[:200])
    except Exception as exc:
        logger.error("drain request failed: %s", exc)
    return None


def queued_count_by_stage() -> dict[str, int]:
    conn = _ro_conn()
    rows = conn.execute(
        """SELECT stage, COUNT(*) AS n FROM followup_tasks
           WHERE status = 'queued'
             AND source_context = 'program_detail_rerun'
           GROUP BY stage"""
    ).fetchall()
    conn.close()
    return {r["stage"]: r["n"] for r in rows}


def task_status(task_id: str) -> dict[str, Any]:
    conn = _ro_conn()
    row = conn.execute(
        """SELECT status, started_timestamp, completed_timestamp,
                  outcome, evidence_pack_json
           FROM followup_tasks WHERE task_id = ?""",
        (task_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else {}


def _snapshot_composites(fps: list[str]) -> dict[str, dict]:
    """Snapshot leaderboard composite + tier-breakdown for these fps."""
    if not fps:
        return {}
    conn = _ro_conn()
    placeholders = ",".join("?" * len(fps))
    rows = conn.execute(
        f"""SELECT graph_fingerprint AS fp, composite_score, tier,
                   n_runs, cv_loss, cv_understanding, cv_capability,
                   score_stability_penalty, wikitext_perplexity
            FROM leaderboard
            WHERE graph_fingerprint IN ({placeholders})""",
        fps,
    ).fetchall()
    conn.close()
    return {r["fp"]: dict(r) for r in rows}


def latest_pr_row(fp: str, since_ts: float) -> Optional[dict]:
    """Return the most recent program_results row for this fp newer than since_ts."""
    conn = _ro_conn()
    row = conn.execute(
        """SELECT result_id, timestamp, stage0_passed, stage05_passed,
                  stage1_passed, wikitext_perplexity, n_train_steps,
                  screening_wikitext_metric_version
           FROM program_results_compat
           WHERE graph_fingerprint = ?
             AND timestamp > ?
           ORDER BY timestamp DESC LIMIT 1""",
        (fp, since_ts),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def wait_for_completion(
    task_id: str,
    fp: str,
    stage: str,
    max_sec: int,
    since_ts: float,
) -> dict[str, Any]:
    """Poll until the experiment actually finishes, not just task launch.

    The followup_task gets marked 'completed' the moment start_*
    returns (which is at experiment LAUNCH for investigation /
    validation — they kick off a background experiment and return).
    The actual training takes minutes-to-hours afterward.  So we
    wait on a stronger signal: the runner becoming idle AGAIN after
    being busy.  We also wait a small grace period for the
    program_results row to be written by the experiment's completion
    callback.

    Returns {final_status, outcome, duration_sec, new_pr_row}.
    """
    t0 = time.time()
    seen_busy = False
    task_state: dict[str, Any] = {}
    while time.time() - t0 < max_sec:
        st = task_status(task_id)
        s = st.get("status")
        task_state = st
        # Stuck-queued: drain didn't actually fire.
        if s == "queued" and time.time() - t0 > DRAIN_RETRY_SEC * 2:
            return {
                "task_id": task_id,
                "stage": stage,
                "fp": fp,
                "final_status": "stuck_queued",
                "outcome": st.get("outcome"),
                "duration_sec": round(time.time() - t0, 1),
                "new_pr_row": None,
            }
        # Hard failure / cancellation.
        if s in ("failed", "cancelled"):
            return {
                "task_id": task_id,
                "stage": stage,
                "fp": fp,
                "final_status": s,
                "outcome": st.get("outcome"),
                "duration_sec": round(time.time() - t0, 1),
                "new_pr_row": latest_pr_row(fp, since_ts),
            }
        # Runner state determines real completion.
        busy = runner_busy()
        if busy:
            seen_busy = True
        elif s == "completed" and seen_busy:
            # Experiment was running and is now idle: experiment finished.
            # Give the completion callback up to 15s to write the row.
            for _ in range(15):
                row = latest_pr_row(fp, since_ts)
                if row is not None:
                    break
                time.sleep(1)
            else:
                row = latest_pr_row(fp, since_ts)
            return {
                "task_id": task_id,
                "stage": stage,
                "fp": fp,
                "final_status": "completed",
                "outcome": st.get("outcome"),
                "duration_sec": round(time.time() - t0, 1),
                "new_pr_row": row,
            }
        elif s == "completed" and not seen_busy:
            # Some replay paths run synchronously inside the API
            # request and never flip the runner-busy flag.  Give the
            # row-write a brief grace then check.
            for _ in range(15):
                row = latest_pr_row(fp, since_ts)
                if row is not None:
                    break
                time.sleep(1)
            return {
                "task_id": task_id,
                "stage": stage,
                "fp": fp,
                "final_status": "completed",
                "outcome": st.get("outcome"),
                "duration_sec": round(time.time() - t0, 1),
                "new_pr_row": latest_pr_row(fp, since_ts),
            }
        time.sleep(8)
    # Timeout
    return {
        "task_id": task_id,
        "stage": stage,
        "fp": fp,
        "final_status": "timeout",
        "outcome": task_state.get("outcome"),
        "duration_sec": round(max_sec, 1),
        "new_pr_row": latest_pr_row(fp, since_ts),
    }


def drain_loop(queued: list[dict]) -> list[dict]:
    """Drive the drain endpoint until all queued tasks are processed."""
    results: list[dict] = []
    by_id = {q["task_id"]: q for q in queued}
    seen: set[str] = set()
    no_progress_streak = 0
    max_no_progress = 10  # 10 idle drains in a row → bail

    while len(seen) < len(queued):
        if runner_busy():
            logger.info("runner busy, waiting %ds", IDLE_SLEEP_SEC)
            time.sleep(IDLE_SLEEP_SEC)
            continue

        drain_resp = drain_once()
        if drain_resp is None:
            no_progress_streak += 1
            time.sleep(IDLE_SLEEP_SEC)
            if no_progress_streak >= max_no_progress:
                logger.error("too many consecutive drain failures, abandoning queue")
                break
            continue

        status = drain_resp.get("status")
        if status == "idle":
            # Maybe the runner cleared a backlog from a different source —
            # check for residual queued among ours.
            resid = [
                tid
                for tid in by_id
                if tid not in seen and task_status(tid).get("status") == "queued"
            ]
            if not resid:
                logger.info("drain idle and no residual queued of ours — done")
                break
            no_progress_streak += 1
            if no_progress_streak >= max_no_progress:
                logger.warning(
                    "queued tasks remain but drain says idle %dx; abandoning",
                    max_no_progress,
                )
                break
            time.sleep(IDLE_SLEEP_SEC)
            continue
        if status == "busy":
            time.sleep(IDLE_SLEEP_SEC)
            continue
        if status not in ("launched", "no_op"):
            logger.warning("unexpected drain status %r: %s", status, drain_resp)
            time.sleep(IDLE_SLEEP_SEC)
            continue

        no_progress_streak = 0
        launched = drain_resp.get("task_ids") or []
        if not launched:
            logger.info("drain reported %s but no task_ids; %s", status, drain_resp)
            time.sleep(IDLE_SLEEP_SEC)
            continue

        tid = launched[0]
        meta = by_id.get(tid)
        if meta is None:
            logger.info("drained an unrelated task %s — skipping", tid[:12])
            seen.add(tid)
            continue

        stage = meta["stage"]
        fp = meta["fp"]
        timeout = STAGE_TIMEOUT_SEC[stage]
        since_ts = time.time() - 5  # small backstep so we catch rows written ~now
        logger.info(
            "[drain] %s | %s | task=%s timeout=%ds", stage, fp, tid[:12], timeout
        )
        result = wait_for_completion(tid, fp, stage, timeout, since_ts)
        results.append(result)
        seen.add(tid)
        new_row = result.get("new_pr_row")
        wrote = "OK row" if new_row else "NO ROW WRITTEN"
        logger.info(
            "[done]  %s | %s | %ds | status=%s outcome=%s | %s",
            stage,
            fp,
            result["duration_sec"],
            result["final_status"],
            result.get("outcome"),
            wrote,
        )

    # Mark any unprocessed
    for q in queued:
        if q["task_id"] not in seen:
            results.append(
                {
                    "task_id": q["task_id"],
                    "stage": q["stage"],
                    "fp": q["fp"],
                    "final_status": "not_drained",
                    "duration_sec": 0.0,
                    "new_pr_row": None,
                }
            )
    return results


def write_findings(
    top: list[dict],
    queued: list[dict],
    results: list[dict],
    *,
    scaling_queued: Optional[list[dict]] = None,
    scaling_results: Optional[list[dict]] = None,
    score_after_reruns: Optional[dict[str, dict]] = None,
    score_after_scaling: Optional[dict[str, dict]] = None,
) -> None:
    scaling_queued = scaling_queued or []
    scaling_results = scaling_results or []
    score_after_reruns = score_after_reruns or {}
    score_after_scaling = score_after_scaling or {}
    by_fp_stage: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        by_fp_stage.setdefault((r["fp"], r["stage"]), []).append(r)

    with FINDINGS_PATH.open("w") as f:
        f.write("# Overnight rerun orchestrator findings\n\n")
        f.write(f"**Started:** {stamp}\n")
        f.write(
            f"**Top {TOP_N} fingerprints, {RERUNS_PER_STAGE}× per stage, "
            f"stages = {', '.join(STAGES)}**\n\n"
        )

        f.write("## Summary\n\n")
        n_total = len(results)
        n_with_row = sum(1 for r in results if r.get("new_pr_row"))
        n_completed = sum(1 for r in results if r["final_status"] == "completed")
        n_timeout = sum(1 for r in results if r["final_status"] == "timeout")
        n_failed = sum(
            1 for r in results if r["final_status"] in ("failed", "cancelled")
        )
        n_undrained = sum(
            1 for r in results if r["final_status"] in ("not_drained", "stuck_queued")
        )
        f.write(f"- Tasks queued: {len(queued)}\n")
        f.write(f"- Tasks processed: {n_total}\n")
        f.write(f"- Produced new program_results row: **{n_with_row}**\n")
        f.write(f"- Completed cleanly: {n_completed}\n")
        f.write(f"- Timed out: {n_timeout}\n")
        f.write(f"- Failed/cancelled: {n_failed}\n")
        f.write(f"- Never drained: {n_undrained}\n\n")

        # Per-stage breakdown
        f.write("## Per-stage breakdown\n\n")
        f.write("| Stage | Queued | Completed | Produced row | Timeout | Failed |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for s in STAGES:
            sub = [r for r in results if r["stage"] == s]
            f.write(
                f"| {s} | {len(sub)} "
                f"| {sum(1 for r in sub if r['final_status'] == 'completed')} "
                f"| {sum(1 for r in sub if r.get('new_pr_row'))} "
                f"| {sum(1 for r in sub if r['final_status'] == 'timeout')} "
                f"| {sum(1 for r in sub if r['final_status'] in ('failed', 'cancelled'))} |\n"
            )
        f.write("\n")

        f.write("## Per-fingerprint breakdown\n\n")
        for entry in top:
            fp = entry["fp"]
            f.write(
                f"### {fp} ({entry['tier']}, score {entry['composite_score']:.1f})\n\n"
            )
            f.write(f"- result_id: `{entry['result_id']}`\n")
            f.write(f"- starting n_runs: {entry.get('n_runs') or 1}\n")
            for s in STAGES:
                sub = by_fp_stage.get((fp, s), [])
                if not sub:
                    f.write(f"  - **{s}**: no tasks recorded\n")
                    continue
                wrote = sum(1 for r in sub if r.get("new_pr_row"))
                f.write(f"  - **{s}**: {len(sub)} runs, {wrote} produced rows; ")
                outcomes = ", ".join(
                    f"{r['final_status']}({r.get('outcome') or '-'})" for r in sub
                )
                f.write(f"{outcomes}\n")
            f.write("\n")

        f.write("## Notes / known issues\n\n")
        if any(r["stage"] == "screening" and not r.get("new_pr_row") for r in results):
            f.write(
                "- Some screening reruns produced no program_results row.  "
                "If `outcome` is `replay_no_rows` or similar, the architecture "
                "failed S0/S05 quality gates at the chosen step budget — "
                "expected for some archs at low budgets.\n"
            )
        if any(r["final_status"] == "timeout" for r in results):
            f.write(
                "- Some tasks hit the per-stage timeout.  Check the runner logs "
                "(`research/aria_dashboard.log`) for failures or hung threads.\n"
            )
        if any(r["final_status"] in ("not_drained", "stuck_queued") for r in results):
            f.write(
                "- Some tasks never drained.  Likely the orchestrator gave up "
                "after consecutive idle drain responses.  Re-run "
                "`python -m research.tools.queue_rerun --auto` or click "
                "▶ Run next pending in the UI to clear residuals.\n"
            )
        # Phase-2 scaling probe: did extending validation steps help?
        if scaling_queued or score_after_scaling:
            f.write("## Phase 2 — scaling probe\n\n")
            f.write(
                f"Each top-{TOP_N} fp got 1 validation rerun at "
                f"**{SCALING_STEPS} steps × 1 seed** (vs the default "
                f"10 000 steps used by the regular reruns).  This "
                f"tests the hypothesis that some architectures need "
                f"more steps to form induction heads / saturate.\n\n"
            )
            f.write(
                "| fp | tier | post-reruns | post-scaling | Δ | "
                "PPL post-scaling | n_runs | CV(loss) |\n"
            )
            f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
            for entry in top:
                fp = entry["fp"]
                a = score_after_reruns.get(fp, {})
                b = score_after_scaling.get(fp, {})
                a_score = a.get("composite_score")
                b_score = b.get("composite_score")
                delta = (
                    f"{(b_score - a_score):+.1f}"
                    if a_score is not None and b_score is not None
                    else "-"
                )
                a_str = f"{a_score:.1f}" if a_score is not None else "-"
                b_str = f"{b_score:.1f}" if b_score is not None else "-"
                ppl = b.get("wikitext_perplexity")
                ppl_str = f"{ppl:.1f}" if ppl is not None else "-"
                n_runs = b.get("n_runs") or "-"
                cv = b.get("cv_loss")
                cv_str = f"{cv:.3f}" if cv is not None else "-"
                f.write(
                    f"| `{fp}` | {b.get('tier') or a.get('tier')} | "
                    f"{a_str} | {b_str} | **{delta}** | "
                    f"{ppl_str} | {n_runs} | {cv_str} |\n"
                )
            f.write("\n")
            n_scale_with_row = sum(1 for r in scaling_results if r.get("new_pr_row"))
            f.write(
                f"- Scaling probe: queued {len(scaling_queued)}, "
                f"produced {n_scale_with_row} new program_results rows.\n\n"
            )

        f.write("## Raw results\n\n")
        f.write("```json\n")
        f.write(json.dumps(results, indent=2, default=str))
        f.write("\n```\n\n")
        if scaling_results:
            f.write("## Raw scaling-probe results\n\n```json\n")
            f.write(json.dumps(scaling_results, indent=2, default=str))
            f.write("\n```\n")

    logger.info("findings written to %s", FINDINGS_PATH)


def main() -> None:
    logger.info("=== overnight orchestrator starting ===")
    logger.info("log: %s", LOG_PATH)
    logger.info("findings: %s", FINDINGS_PATH)

    # Sanity: dashboard reachable
    try:
        r = requests.get(f"{DASHBOARD_BASE}/api/leaderboard?n=1", timeout=10)
        r.raise_for_status()
    except Exception as exc:
        logger.error("dashboard not reachable on %s: %s", DASHBOARD_BASE, exc)
        sys.exit(2)

    top = fetch_top_n(TOP_N)
    if not top:
        logger.error("no top-N rows returned")
        sys.exit(2)
    logger.info("top-%d fingerprints:", TOP_N)
    for i, entry in enumerate(top, start=1):
        logger.info(
            "  %2d. %s %-13s %.1f",
            i,
            entry["fp"],
            entry["tier"],
            entry["composite_score"],
        )

    # Tim's rule: if any fp needs validation first, do it before the 2x.
    # All top-10 are already at validation/breakthrough so the pre-step is a no-op.
    not_validated = [e for e in top if e["tier"] not in ("validation", "breakthrough")]
    if not_validated:
        logger.info(
            "pre-step: %d fps need an initial validation before 2x reruns",
            len(not_validated),
        )
        pre = []
        for entry in not_validated:
            tid = queue_one(entry["result_id"], "validation")
            if tid:
                pre.append(
                    {
                        "task_id": tid,
                        "fp": entry["fp"],
                        "result_id": entry["result_id"],
                        "stage": "validation_pre",
                        "rerun_index": 0,
                    }
                )
        if pre:
            drain_loop(pre)

    queued = queue_all(top)

    # Snapshot queue counts before drain
    pre_drain = queued_count_by_stage()
    logger.info("queue snapshot pre-drain: %s", pre_drain)

    results = drain_loop(queued)

    # Snapshot composite scores AFTER reruns (post-rescore happens
    # naturally via upserts during the experiments).  Used to compare
    # against the scaling-probe results.
    score_after_reruns = _snapshot_composites([t["fp"] for t in top])

    # Phase 2: scaling probe — does training longer help?
    logger.info("=== phase 2: scaling probe at %d steps ===", SCALING_STEPS)
    scaling_queued = []
    for entry in top:
        rid = str(entry["result_id"])
        body = {
            "stage": "validation",
            "n": 1,
            "n_steps": SCALING_STEPS,
            "n_seeds": 1,
            "reason": f"scaling_probe_{SCALING_STEPS}",
        }
        try:
            r = requests.post(
                f"{DASHBOARD_BASE}/api/programs/{rid}/queue-validation-rerun",
                json=body,
                timeout=15,
            )
            r.raise_for_status()
            ids = r.json().get("task_ids") or []
            if ids:
                scaling_queued.append(
                    {
                        "task_id": ids[0],
                        "fp": entry["fp"],
                        "result_id": rid,
                        "stage": "validation_scaling",
                        "rerun_index": 1,
                    }
                )
        except Exception as exc:
            logger.error("scaling-queue failed for %s: %s", entry["fp"], exc)
    logger.info("scaling probe queued %d tasks", len(scaling_queued))
    # Bump validation timeout for the scaling probe.
    saved_timeout = STAGE_TIMEOUT_SEC.get("validation_scaling")
    STAGE_TIMEOUT_SEC["validation_scaling"] = SCALING_TIMEOUT_SEC
    # Re-tag stage so wait_for_completion uses the longer timeout.
    for q in scaling_queued:
        q["_runner_stage"] = "validation"
    scaling_results = drain_loop(scaling_queued) if scaling_queued else []
    if saved_timeout is None:
        STAGE_TIMEOUT_SEC.pop("validation_scaling", None)

    score_after_scaling = _snapshot_composites([t["fp"] for t in top])

    write_findings(
        top,
        queued,
        results,
        scaling_queued=scaling_queued,
        scaling_results=scaling_results,
        score_after_reruns=score_after_reruns,
        score_after_scaling=score_after_scaling,
    )

    logger.info("=== orchestrator complete; calling happy_times.py ===")
    # Hand off to shutdown.  happy_times.py turns off Aria + the host.
    try:
        subprocess.run(
            [sys.executable, HAPPY_TIMES],
            cwd="/home/tim/Projects/LLM",
            check=False,
        )
    except Exception as exc:
        logger.error("happy_times invocation failed: %s", exc)


if __name__ == "__main__":
    main()
