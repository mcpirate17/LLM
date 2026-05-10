#!/usr/bin/env python3
"""Run champion-focused causal ablations sequentially through the dashboard API."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from research.defaults import RUNS_DB
from research.tools._db_maintenance import connect_readonly


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / RUNS_DB
LOG_PATH = PROJECT_ROOT / "research/runtime/targeted_champion_ablation.log"
API_BASE = "http://127.0.0.1:5000"


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def api_json(path: str, payload: dict | None = None) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{API_BASE}{path}", data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def top_champions(limit: int) -> list[dict]:
    with connect_readonly(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT l.result_id,
                   pr.graph_fingerprint,
                   l.composite_score,
                   l.tier,
                   pr.loss_ratio
            FROM leaderboard l
            JOIN program_results_compat pr ON pr.result_id = l.result_id
            WHERE COALESCE(l.is_reference, 0) = 0
              AND l.result_id NOT LIKE 'ref_%'
              AND pr.graph_json IS NOT NULL
              AND TRIM(CAST(pr.graph_json AS TEXT)) <> ''
            ORDER BY l.composite_score DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def parent_evidence_count(result_id: str) -> int:
    with connect_readonly(DB_PATH) as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM causal_rule_evidence WHERE parent_result_id = ?",
                (result_id,),
            ).fetchone()[0]
            or 0
        )


def concrete_running_count() -> int:
    with connect_readonly(DB_PATH) as conn:
        return int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM experiments
                WHERE status = 'running'
                  AND experiment_type != 'unknown'
                """
            ).fetchone()[0]
            or 0
        )


def clear_stale_wrapper(run_id: str) -> None:
    encoded = urllib.parse.quote(run_id, safe="")
    try:
        api_json(f"/api/experiments/{encoded}/cancel", {})
        log(f"cleared stale wrapper run_id={run_id}")
    except urllib.error.URLError as exc:
        log(f"stale wrapper cleanup failed run_id={run_id}: {exc}")


def wait_until_idle(
    *,
    active_result_id: str | None = None,
    active_run_id: str | None = None,
    pre_evidence_count: int | None = None,
    timeout_s: int = 7200,
) -> None:
    deadline = time.time() + timeout_s
    no_child_running_since: float | None = None
    while time.time() < deadline:
        try:
            status = api_json("/api/aria/cycle-status")
            if not status.get("is_running"):
                return
            concrete_running = concrete_running_count()
            evidence_delta = 0
            if active_result_id is not None and pre_evidence_count is not None:
                evidence_delta = (
                    parent_evidence_count(active_result_id) - pre_evidence_count
                )
            if concrete_running == 0:
                if no_child_running_since is None:
                    no_child_running_since = time.time()
                elif (
                    active_run_id
                    and evidence_delta > 0
                    and time.time() - no_child_running_since >= 90
                ):
                    clear_stale_wrapper(active_run_id)
                    return
            else:
                no_child_running_since = None
            log(
                "waiting: "
                f"phase={status.get('phase')} progress={status.get('progress_status')} "
                f"experiment={status.get('experiment_id')} "
                f"child_running={concrete_running} evidence_delta={evidence_delta}"
            )
        except Exception as exc:  # noqa: BLE001 - operational script
            log(f"waiting: status check failed: {exc}")
        time.sleep(20)
    raise TimeoutError("runner did not become idle")


def start_ablation(result_id: str) -> dict:
    payload = {
        "top_k": 200,
        "max_signals": 6,
        "max_graphs": 6,
        "causal_ablation_top_k": 200,
        "causal_ablation_max_signals": 6,
        "causal_ablation_max_graphs": 6,
    }
    return api_json(f"/api/programs/{result_id}/causal-ablation", payload)


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    champions = top_champions(limit)
    log(f"targeted champion ablation campaign starting: targets={len(champions)}")
    for idx, row in enumerate(champions, start=1):
        result_id = str(row["result_id"])
        before = parent_evidence_count(result_id)
        log(
            f"target {idx}/{len(champions)} result={result_id} "
            f"fp={str(row['graph_fingerprint'])[:14]} "
            f"score={float(row['composite_score']):.1f} tier={row['tier']} "
            f"pre_evidence={before}"
        )
        if before > 0:
            log(f"skipping result={result_id} existing_evidence={before}")
            continue
        wait_until_idle()
        try:
            response = start_ablation(result_id)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            log(f"start failed result={result_id} status={exc.code} body={body}")
            continue
        log(
            f"started result={result_id} response={json.dumps(response, sort_keys=True)}"
        )
        wait_until_idle(
            active_result_id=result_id,
            active_run_id=str(response.get("run_id") or ""),
            pre_evidence_count=before,
        )
        after = parent_evidence_count(result_id)
        log(
            f"finished result={result_id} added_evidence={after - before} "
            f"post_evidence={after}"
        )
    log("targeted champion ablation campaign complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
