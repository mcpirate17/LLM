#!/usr/bin/env python3
"""Start and supervise continuous mode on a fixed overnight cadence.

Behavior:
- Ensure the dashboard is running on port 5000.
- Start a fresh continuous session through the dashboard API if none is active.
- Monitor every 2 minutes for the first 14 minutes, then every 10 minutes
  for the next hour.
- If the supervisor intervenes (restart, DB repair, or relaunch), the cadence
  resets from that point.

This script is designed for unattended use. It logs every action to both stdout
and a timestamped file under research/logs/.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT / "research"
DB_PATH = ROOT / "research" / "lab_notebook.db"
DASHBOARD_LOG = ROOT / "research" / "aria_dashboard.log"
RUNTIME_EVENTS_DIR = ROOT / "research" / "runtime_events"
LOG_DIR = ROOT / "research" / "logs"
API_ROOT = "http://127.0.0.1:5000"
DASHBOARD_CMD = [sys.executable, "-m", "research", "--mode=dashboard", "--port", "5000"]
DASHBOARD_PATTERN = "python -m research --mode=dashboard --port 5000"
CHECK_MINUTES = [2, 4, 6, 8, 10, 12, 14, 24, 34, 44, 54, 64, 74]
CORE_ENDPOINTS = [
    "/api/aria/cycle-status",
    "/api/diagnostics/fingerprint",
    "/api/live-loss-curve",
    "/api/discoveries?sort=composite_score&limit=25&view=ranked&trusted_only=0",
]
LOG_ERROR_MARKERS = (
    " ERROR ",
    " CRITICAL ",
    "Traceback",
    "continuous_session_failed",
    "database is locked",
    "writer lock",
    "cannot import name",
    " -> 500",
)


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_http_json(
    path: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 20.0,
) -> Tuple[int, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(f"{API_ROOT}{path}", data=data, method=method, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), json.loads(body) if body else {}
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body else {"error": body}
        except json.JSONDecodeError:
            parsed = {"error": body}
        return int(exc.code), parsed
    except (URLError, ConnectionResetError, TimeoutError, OSError) as exc:
        return 0, {"error": str(exc)}


def latest_runtime_event_mtime() -> float:
    latest = 0.0
    if not RUNTIME_EVENTS_DIR.exists():
        return latest
    for path in RUNTIME_EVENTS_DIR.glob("segment-*.ndjson"):
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            continue
    return latest


@dataclass
class Supervisor:
    log_path: Path
    dashboard_output_path: Path
    log_offset: int = 0
    sequence_started_at: float = 0.0
    reset_count: int = 0
    last_runtime_event_mtime: float = field(default_factory=latest_runtime_event_mtime)
    last_cycle_transition_ts: float = 0.0
    last_cycle_experiment_id: str = ""
    check_index: int = 0

    def log(self, message: str) -> None:
        line = f"[{now_iso()}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def find_dashboard_pid(self) -> Optional[int]:
        result = subprocess.run(
            ["ps", "-eo", "pid=,args="],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        for raw in result.stdout.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(None, 1)
            if len(parts) != 2:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            args = parts[1]
            if pid == os.getpid():
                continue
            try:
                argv = shlex.split(args)
            except ValueError:
                continue
            if len(argv) >= 6 and argv[1:6] == [
                "-m",
                "research",
                "--mode=dashboard",
                "--port",
                "5000",
            ]:
                return pid
        return None

    def wait_for_dashboard(self, timeout_s: float = 45.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            status, _ = read_http_json("/api/aria/cycle-status", timeout=5.0)
            if status == 200:
                return True
            time.sleep(1.0)
        return False

    def start_dashboard(self) -> bool:
        pid = self.find_dashboard_pid()
        if pid is not None:
            self.log(f"Dashboard already running (pid={pid}).")
            return True

        self.log("Starting dashboard on port 5000.")
        with self.dashboard_output_path.open("a", encoding="utf-8") as out:
            proc = subprocess.Popen(
                DASHBOARD_CMD,
                cwd=ROOT,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        ok = self.wait_for_dashboard()
        if ok:
            self.log(f"Dashboard started (pid={proc.pid}).")
        else:
            self.log(
                f"Dashboard failed to become healthy after start (pid={proc.pid})."
            )
        return ok

    def stop_dashboard(self, reason: str) -> None:
        pid = self.find_dashboard_pid()
        if pid is None:
            self.log(f"Dashboard stop skipped; no pid found ({reason}).")
            return
        self.log(f"Stopping dashboard pid={pid} ({reason}).")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.time() + 25.0
        while time.time() < deadline:
            if self.find_dashboard_pid() is None:
                self.log(f"Dashboard pid={pid} stopped cleanly.")
                return
            time.sleep(0.5)
        self.log(f"Dashboard pid={pid} did not stop on SIGTERM; sending SIGKILL.")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(1.0)

    def ensure_continuous_running(self) -> bool:
        status, payload = read_http_json("/api/aria/cycle-status")
        if status != 200:
            self.log(
                f"Cannot read cycle status before launch: status={status} payload={payload}"
            )
            return False
        if (
            payload.get("is_running")
            and payload.get("continuous_active")
            and not payload.get("external_process")
        ):
            self.log(
                "Continuous session already active "
                f"(experiment_id={payload.get('experiment_id')}, phase={payload.get('phase')})."
            )
            self.last_cycle_transition_ts = float(
                payload.get("last_transition_ts") or 0.0
            )
            self.last_cycle_experiment_id = str(payload.get("experiment_id") or "")
            return True

        self.log("Starting continuous session through /api/aria/cycle-control.")
        start_status, start_payload = read_http_json(
            "/api/aria/cycle-control",
            method="POST",
            payload={"action": "start", "auto_harden": True},
            timeout=30.0,
        )
        self.log(
            f"Continuous start response: status={start_status} payload={start_payload}"
        )
        if start_status != 200:
            return False
        cycle = start_payload.get("cycle") or {}
        self.last_cycle_transition_ts = float(cycle.get("last_transition_ts") or 0.0)
        self.last_cycle_experiment_id = str(start_payload.get("experiment_id") or "")
        return True

    def tail_new_dashboard_log(self) -> List[str]:
        if not DASHBOARD_LOG.exists():
            return []
        try:
            with DASHBOARD_LOG.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self.log_offset)
                chunk = fh.read()
                self.log_offset = fh.tell()
        except OSError as exc:
            self.log(f"Failed to read dashboard log: {exc}")
            return []
        if not chunk:
            return []
        return [line for line in chunk.splitlines() if line.strip()]

    def db_health_snapshot(self) -> Dict[str, int]:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT
                    SUM(
                        CASE WHEN pr.fingerprint_json IS NOT NULL
                               AND json_valid(pr.fingerprint_json) = 1
                               AND abs(coalesce(pr.novelty_score,-9999) -
                                       coalesce(json_extract(pr.fingerprint_json,'$.novelty_score'),-9999)) > 1e-9
                             THEN 1 ELSE 0 END
                    ) AS novelty_mismatch,
                    SUM(
                        CASE WHEN pr.fingerprint_json IS NOT NULL
                               AND json_valid(pr.fingerprint_json) = 1
                               AND coalesce(pr.cka_source,'') !=
                                   coalesce(json_extract(pr.fingerprint_json,'$.cka_source'),'')
                             THEN 1 ELSE 0 END
                    ) AS cka_source_mismatch,
                    SUM(
                        CASE WHEN pr.fingerprint_json IS NOT NULL
                               AND json_valid(pr.fingerprint_json) = 1
                               AND abs(coalesce(pr.fp_jacobian_spectral_norm,-9999) -
                                       coalesce(json_extract(pr.fingerprint_json,'$.jacobian_spectral_norm'),-9999)) > 1e-9
                             THEN 1 ELSE 0 END
                    ) AS spectral_mismatch,
                    SUM(
                        CASE WHEN pr.fingerprint_json IS NOT NULL
                               AND json_valid(pr.fingerprint_json) = 1
                               AND coalesce(pr.novelty_validity_reason,'') !=
                                   coalesce(json_extract(pr.fingerprint_json,'$.novelty_validity_reason'),'')
                             THEN 1 ELSE 0 END
                    ) AS reason_mismatch,
                    (
                        SELECT COUNT(*)
                        FROM leaderboard l
                        LEFT JOIN program_results pr ON pr.result_id = l.result_id
                        WHERE pr.result_id IS NULL
                    ) AS orphan_leaderboard_rows,
                    (
                        SELECT COUNT(*)
                        FROM program_results p
                        JOIN experiments e ON e.experiment_id = p.experiment_id
                        WHERE p.stage1_passed = 1
                          AND e.experiment_type IN ('synthesis', 'novelty', 'evolution', 'reference')
                          AND NOT EXISTS (SELECT 1 FROM leaderboard l WHERE l.result_id = p.result_id)
                          AND NOT EXISTS (
                                SELECT 1
                                FROM leaderboard l
                                JOIN program_results pr2 ON pr2.result_id = l.result_id
                                WHERE pr2.graph_fingerprint = p.graph_fingerprint
                          )
                    ) AS missing_screening_leaderboard_rows
                FROM program_results pr
                """
            ).fetchone()
            return {key: int(row[key] or 0) for key in row.keys()}
        finally:
            conn.close()

    def repair_fingerprint_mismatches(self) -> Dict[str, int]:
        from research.scientist.notebook import LabNotebook

        nb = LabNotebook(str(DB_PATH))
        try:
            rows = nb.conn.execute(
                """
                SELECT result_id, graph_fingerprint, fingerprint_json
                FROM program_results
                WHERE fingerprint_json IS NOT NULL
                  AND json_valid(fingerprint_json) = 1
                  AND (
                        abs(coalesce(novelty_score,-9999)-coalesce(json_extract(fingerprint_json,'$.novelty_score'),-9999)) > 1e-9
                     OR coalesce(cka_source,'') != coalesce(json_extract(fingerprint_json,'$.cka_source'),'')
                     OR abs(coalesce(fp_jacobian_spectral_norm,-9999)-coalesce(json_extract(fingerprint_json,'$.jacobian_spectral_norm'),-9999)) > 1e-9
                     OR coalesce(novelty_validity_reason,'') != coalesce(json_extract(fingerprint_json,'$.novelty_validity_reason'),'')
                     OR coalesce(novelty_valid_for_promotion,0) != coalesce(json_extract(fingerprint_json,'$.novelty_valid_for_promotion'),0)
                     OR (
                          coalesce(json_extract(fingerprint_json,'$.novelty_validity_reason'),'') = 'cka_degenerate_zeros'
                      AND coalesce(json_extract(fingerprint_json,'$.cka_source'),'') != 'artifact'
                     )
                  )
                """
            ).fetchall()
            relabelled = 0
            synced = 0
            fingerprint_anchor: Dict[str, str] = {}
            for row in rows:
                fp_payload = json.loads(row["fingerprint_json"])
                cka_source = str(fp_payload.get("cka_source") or "").strip().lower()
                reason = str(fp_payload.get("novelty_validity_reason") or "").strip()
                if reason == "cka_degenerate_zeros" and cka_source != "artifact":
                    fp_payload["novelty_validity_reason"] = (
                        "cka_deferred_post_investigation"
                        if cka_source == "deferred"
                        else "no_reference_available"
                    )
                    fp_payload["novelty_valid_for_promotion"] = False
                    relabelled += 1
                if nb.sync_behavioral_fingerprint_result(
                    result_id=row["result_id"],
                    fp_payload=fp_payload,
                    sync_leaderboard=False,
                ):
                    synced += 1
                    fingerprint = str(row["graph_fingerprint"] or "").strip()
                    if fingerprint and fingerprint not in fingerprint_anchor:
                        fingerprint_anchor[fingerprint] = str(row["result_id"])
            nb.flush_writes()
            for result_id in fingerprint_anchor.values():
                nb._sync_fingerprint_leaderboard(result_id)
            nb._maybe_commit()
            return {
                "rows_scanned": len(rows),
                "rows_synced": synced,
                "rows_relabelled": relabelled,
                "fingerprints_resynced": len(fingerprint_anchor),
            }
        finally:
            nb.close()

    def reset_sequence(self, reason: str) -> None:
        self.reset_count += 1
        self.sequence_started_at = time.time()
        self.check_index = 0
        self.last_runtime_event_mtime = latest_runtime_event_mtime()
        if DASHBOARD_LOG.exists():
            try:
                self.log_offset = DASHBOARD_LOG.stat().st_size
            except OSError:
                self.log_offset = 0
        self.log(f"Sequence reset #{self.reset_count}: {reason}")

    def intervene_and_reset(self, reason: str) -> None:
        self.log(f"Intervention triggered: {reason}")
        self.stop_dashboard(reason)
        repaired = self.repair_fingerprint_mismatches()
        self.log(f"Repair result: {repaired}")
        if not self.start_dashboard():
            raise RuntimeError("Dashboard failed to restart after intervention")
        if not self.ensure_continuous_running():
            raise RuntimeError(
                "Continuous session failed to restart after intervention"
            )
        self.reset_sequence(reason)

    def check_once(self) -> bool:
        """Return True when the sequence should reset."""
        anomalies: List[str] = []

        pid = self.find_dashboard_pid()
        if pid is None:
            anomalies.append("dashboard_process_missing")
        else:
            self.log(f"Dashboard pid={pid} is present.")

        cycle_status_code, cycle_status = read_http_json("/api/aria/cycle-status")
        if cycle_status_code != 200:
            anomalies.append(f"cycle_status_http_{cycle_status_code}")
        else:
            self.log(
                "Cycle status: "
                f"is_running={cycle_status.get('is_running')} "
                f"continuous_active={cycle_status.get('continuous_active')} "
                f"phase={cycle_status.get('phase')} "
                f"experiment_id={cycle_status.get('experiment_id')}"
            )
            if not cycle_status.get("is_running") or not cycle_status.get(
                "continuous_active"
            ):
                anomalies.append("continuous_not_running")

        for path in CORE_ENDPOINTS:
            status, payload = read_http_json(path)
            if status != 200:
                anomalies.append(f"endpoint_failed:{path}:{status}:{payload}")
            else:
                self.log(f"Endpoint OK: {path} (status=200)")

        db_health = self.db_health_snapshot()
        self.log(f"DB health: {db_health}")
        if any(db_health[key] > 0 for key in db_health):
            anomalies.append(f"db_health_nonzero:{db_health}")

        new_log_lines = self.tail_new_dashboard_log()
        error_lines = [
            line
            for line in new_log_lines
            if any(marker in line for marker in LOG_ERROR_MARKERS)
        ]
        if error_lines:
            sample = error_lines[-5:]
            anomalies.append(f"dashboard_errors:{sample}")

        runtime_mtime = latest_runtime_event_mtime()
        if (
            cycle_status_code == 200
            and cycle_status.get("is_running")
            and self.last_runtime_event_mtime > 0
            and runtime_mtime <= self.last_runtime_event_mtime
            and (time.time() - runtime_mtime) > 12 * 60
        ):
            anomalies.append(f"runtime_events_stalled:last_mtime={runtime_mtime:.3f}")
        self.last_runtime_event_mtime = max(
            self.last_runtime_event_mtime, runtime_mtime
        )

        if anomalies:
            self.log(f"Anomalies detected: {anomalies}")
            self.intervene_and_reset("; ".join(anomalies))
            return True

        self.last_cycle_transition_ts = float(
            cycle_status.get("last_transition_ts") or 0.0
        )
        self.last_cycle_experiment_id = str(cycle_status.get("experiment_id") or "")
        self.log("Check completed cleanly.")
        return False

    def run(self) -> int:
        self.log(f"Supervisor starting. Root={ROOT} DB={DB_PATH}")
        self.log(f"Dashboard log={DASHBOARD_LOG}")
        if DASHBOARD_LOG.exists():
            try:
                self.log_offset = DASHBOARD_LOG.stat().st_size
            except OSError:
                self.log_offset = 0
        if not self.start_dashboard():
            self.log("Supervisor aborting: dashboard is unavailable.")
            return 1
        if not self.ensure_continuous_running():
            self.log("Supervisor aborting: failed to start continuous session.")
            return 1
        self.reset_sequence("initial continuous launch")

        while self.check_index < len(CHECK_MINUTES):
            target_elapsed = CHECK_MINUTES[self.check_index] * 60
            deadline = self.sequence_started_at + target_elapsed
            sleep_s = max(0.0, deadline - time.time())
            self.log(
                f"Sleeping {sleep_s:.1f}s until check {self.check_index + 1}/{len(CHECK_MINUTES)} "
                f"at +{CHECK_MINUTES[self.check_index]}m."
            )
            time.sleep(sleep_s)
            reset = self.check_once()
            if not reset:
                self.check_index += 1

        self.log(
            "Supervisor completed full overnight watch window without further resets."
        )
        return 0


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"overnight_continuous_supervisor_{ts}.log"
    dashboard_output_path = LOG_DIR / f"overnight_dashboard_{ts}.log"
    sup = Supervisor(
        log_path=log_path,
        dashboard_output_path=dashboard_output_path,
    )
    try:
        return sup.run()
    except KeyboardInterrupt:
        sup.log("Supervisor interrupted by user.")
        return 130
    except Exception as exc:
        sup.log(f"Supervisor failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
