#!/usr/bin/env python3
"""Start and supervise continuous mode on a fixed overnight cadence.

Behavior:
- Ensure the dashboard is running on port 5000.
- Start a fresh continuous session through the dashboard API if none is active.
- Monitor every 2 minutes for the first 20 minutes, then every 10 minutes
  for the remainder of the watch window.
- If the run is clean through the full watch window, optionally extend once
  before shutdown.
- If the supervisor intervenes (restart, DB repair, or relaunch), the cadence
  resets from that point.

This script is designed for unattended use. It logs every action to both stdout
and a timestamped file under research/logs/.
"""

from __future__ import annotations

import argparse
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
DB_PATH = ROOT / "research" / "runs.db"
DASHBOARD_LOG = ROOT / "research" / "aria_dashboard.log"
RUNTIME_EVENTS_DIR = ROOT / "research" / "runtime_events"
LOG_DIR = ROOT / "research" / "logs"
API_ROOT = "http://127.0.0.1:5000"
DASHBOARD_CMD = [sys.executable, "-m", "research", "--mode=dashboard", "--port", "5000"]
DASHBOARD_PATTERN = "python -m research --mode=dashboard --port 5000"
DEFAULT_DENSE_MINUTES = 20
DEFAULT_DENSE_INTERVAL_MINUTES = 2
DEFAULT_SPARSE_INTERVAL_MINUTES = 10
DEFAULT_STELLAR_EXTENSION_MINUTES = 60
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
PROMOTABLE_EXPERIMENT_TYPES = (
    "synthesis",
    "novelty",
    "evolution",
    "reference",
    "backfill",
    "forced_exploration",
    "ablation",
)
HTTP_LOG_MARKERS = (
    '"GET /api/',
    '"POST /api/',
    '"OPTIONS /api/',
)


def build_check_minutes(
    total_minutes: int,
    *,
    dense_minutes: int = DEFAULT_DENSE_MINUTES,
    dense_interval_minutes: int = DEFAULT_DENSE_INTERVAL_MINUTES,
    sparse_interval_minutes: int = DEFAULT_SPARSE_INTERVAL_MINUTES,
) -> List[int]:
    total = max(2, int(total_minutes))
    dense_limit = min(total, max(2, int(dense_minutes)))
    dense_step = max(1, int(dense_interval_minutes))
    sparse_step = max(1, int(sparse_interval_minutes))
    checks = [minute for minute in range(dense_step, dense_limit + 1, dense_step)]
    if total > dense_limit:
        sparse_start = dense_limit + sparse_step
        checks.extend(range(sparse_start, total + 1, sparse_step))
    if checks[-1] != total:
        checks.append(total)
    return sorted(set(checks))


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
    check_minutes: List[int] = field(
        default_factory=lambda: build_check_minutes(10 * 60)
    )
    max_wall_minutes: Optional[int] = None
    shutdown_command: Optional[str] = None
    stellar_extension_minutes: int = DEFAULT_STELLAR_EXTENSION_MINUTES
    activity_stall_minutes: int = 25
    dashboard_start_retries: int = 3
    log_offset: int = 0
    sequence_started_at: float = 0.0
    overall_started_at: float = 0.0
    reset_count: int = 0
    last_runtime_event_mtime: float = field(default_factory=latest_runtime_event_mtime)
    last_cycle_transition_ts: float = 0.0
    last_cycle_experiment_id: str = ""
    check_index: int = 0
    last_meaningful_log_ts: float = 0.0
    extended_once: bool = False

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

    def wait_for_dashboard(self, timeout_s: float = 60.0) -> bool:
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

        for attempt in range(1, self.dashboard_start_retries + 1):
            self.log(
                f"Starting dashboard on port 5000 (attempt {attempt}/{self.dashboard_start_retries})."
            )
            with self.dashboard_output_path.open("a", encoding="utf-8") as out:
                proc = subprocess.Popen(
                    DASHBOARD_CMD,
                    cwd=ROOT,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            ok = self.wait_for_dashboard(timeout_s=60.0 if attempt == 1 else 90.0)
            if ok:
                self.log(f"Dashboard started (pid={proc.pid}).")
                return True
            self.log(
                f"Dashboard failed to become healthy after start (pid={proc.pid})."
            )
            self.stop_dashboard(f"failed_start_attempt_{attempt}")
        return False

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
        lines = [line for line in chunk.splitlines() if line.strip()]
        if any(
            not any(marker in line for marker in HTTP_LOG_MARKERS) for line in lines
        ):
            self.last_meaningful_log_ts = time.time()
        return lines

    def current_experiment_activity_mtime(self, experiment_id: str) -> float:
        exp_id = str(experiment_id or "").strip()
        if not exp_id:
            return 0.0
        latest = 0.0
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    """
                    SELECT MAX(timestamp) AS latest_program_ts
                    FROM program_results
                    WHERE experiment_id = ?
                    """,
                    (exp_id,),
                ).fetchone()
                latest = max(latest, float(row["latest_program_ts"] or 0.0))
            finally:
                conn.close()
        except (sqlite3.OperationalError, OSError, TypeError, ValueError):
            pass

        ckpt_dir = ROOT / "checkpoints" / exp_id
        if ckpt_dir.exists():
            try:
                latest = max(
                    latest,
                    max(
                        (
                            path.stat().st_mtime
                            for path in ckpt_dir.rglob("*")
                            if path.is_file()
                        ),
                        default=0.0,
                    ),
                )
            except OSError:
                pass
        return latest

    def db_health_snapshot(self) -> Dict[str, int]:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            promotable_sql = ",".join(repr(x) for x in PROMOTABLE_EXPERIMENT_TYPES)
            row = conn.execute(
                f"""
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
                          AND e.experiment_type IN ({promotable_sql})
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

    def repair_leaderboard_coverage(self) -> Dict[str, Any]:
        from research.scientist.notebook import LabNotebook

        nb = LabNotebook(str(DB_PATH))
        try:
            result = nb.backfill_missing_screening_leaderboard_entries(
                experiment_types=list(PROMOTABLE_EXPERIMENT_TYPES)
            )
            return result
        finally:
            nb.close()

    def repair_stranded_experiments(
        self,
        *,
        active_experiment_id: str = "",
        min_age_minutes: int = 2,
        reason: str,
    ) -> Dict[str, Any]:
        from research.scientist.notebook import LabNotebook

        nb = LabNotebook(str(DB_PATH))
        try:
            cutoff = time.time() - (max(0, int(min_age_minutes)) * 60)
            rows = nb.conn.execute(
                """
                SELECT experiment_id
                FROM experiments
                WHERE status = 'running'
                  AND started_at <= ?
                ORDER BY started_at ASC
                """,
                (cutoff,),
            ).fetchall()
            active = str(active_experiment_id or "").strip()
            repaired: List[str] = []
            skipped: List[str] = []
            for row in rows:
                experiment_id = str(row["experiment_id"] or "").strip()
                if not experiment_id:
                    continue
                if active and experiment_id == active:
                    skipped.append(experiment_id)
                    continue
                nb.interrupt_experiment(
                    experiment_id,
                    f"INTERRUPTED: supervisor repair ({reason})",
                )
                repaired.append(experiment_id)
            return {"repaired": repaired, "skipped": skipped}
        finally:
            nb.close()

    def reset_sequence(self, reason: str) -> None:
        self.reset_count += 1
        if self.overall_started_at <= 0:
            self.overall_started_at = time.time()
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
        previous_active_experiment = self.last_cycle_experiment_id
        self.stop_dashboard(reason)
        stranded = self.repair_stranded_experiments(
            active_experiment_id="",
            reason=reason,
        )
        self.log(f"Stranded experiment repair: {stranded}")
        repaired = self.repair_fingerprint_mismatches()
        self.log(f"Repair result: {repaired}")
        coverage = self.repair_leaderboard_coverage()
        self.log(f"Coverage repair result: {coverage}")
        if not self.start_dashboard():
            raise RuntimeError("Dashboard failed to restart after intervention")
        if not self.ensure_continuous_running():
            raise RuntimeError(
                "Continuous session failed to restart after intervention"
            )
        if previous_active_experiment:
            self.log(
                f"Previous active experiment before intervention was {previous_active_experiment[:12]}."
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
        activity_stall_s = max(5, int(self.activity_stall_minutes)) * 60
        exp_activity_mtime = self.current_experiment_activity_mtime(
            str(cycle_status.get("experiment_id") or "")
        )
        latest_activity_mtime = max(
            runtime_mtime,
            self.last_meaningful_log_ts,
            exp_activity_mtime,
        )
        if (
            cycle_status_code == 200
            and cycle_status.get("is_running")
            and self.last_runtime_event_mtime > 0
            and runtime_mtime <= self.last_runtime_event_mtime
            and (time.time() - latest_activity_mtime) > activity_stall_s
        ):
            anomalies.append(
                "runtime_events_stalled:"
                f"runtime={runtime_mtime:.3f}:"
                f"log={self.last_meaningful_log_ts:.3f}:"
                f"exp={exp_activity_mtime:.3f}"
            )
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
        if self.overall_started_at <= 0:
            self.log(f"Supervisor starting. Root={ROOT} DB={DB_PATH}")
            self.log(f"Dashboard log={DASHBOARD_LOG}")
            self.overall_started_at = time.time()
            if DASHBOARD_LOG.exists():
                try:
                    self.log_offset = DASHBOARD_LOG.stat().st_size
                except OSError:
                    self.log_offset = 0
            status, payload = read_http_json("/api/aria/cycle-status")
            active_exp_id = ""
            if status == 200 and payload.get("is_running"):
                active_exp_id = str(payload.get("experiment_id") or "")
            stranded = self.repair_stranded_experiments(
                active_experiment_id=active_exp_id,
                reason="startup_repair",
            )
            if stranded.get("repaired"):
                self.log(f"Startup stranded experiment repair: {stranded}")
            if not self.start_dashboard():
                self.log("Supervisor aborting: dashboard is unavailable.")
                return 1
            if not self.ensure_continuous_running():
                self.log("Supervisor aborting: failed to start continuous session.")
                return 1
            self.reset_sequence("initial continuous launch")

        while True:
            while self.check_index < len(self.check_minutes):
                if self.max_wall_minutes is not None:
                    overall_deadline = self.overall_started_at + (
                        self.max_wall_minutes * 60
                    )
                    if time.time() >= overall_deadline:
                        break
                target_elapsed = self.check_minutes[self.check_index] * 60
                deadline = self.sequence_started_at + target_elapsed
                if self.max_wall_minutes is not None:
                    deadline = min(deadline, overall_deadline)
                sleep_s = max(0.0, deadline - time.time())
                self.log(
                    f"Sleeping {sleep_s:.1f}s until check {self.check_index + 1}/{len(self.check_minutes)} "
                    f"at +{self.check_minutes[self.check_index]}m."
                )
                time.sleep(sleep_s)
                if (
                    self.max_wall_minutes is not None
                    and time.time() >= overall_deadline
                ):
                    break
                reset = self.check_once()
                if not reset:
                    self.check_index += 1
            if not self.maybe_extend_watch_window():
                break
        self.log("Supervisor completed requested watch window without further resets.")
        return 0

    def maybe_extend_watch_window(self) -> bool:
        if self.extended_once:
            return False
        extension_minutes = max(0, int(self.stellar_extension_minutes))
        if extension_minutes <= 0:
            return False
        if self.max_wall_minutes is None:
            return False
        if self.reset_count > 1:
            self.log(
                "Skipping watch-window extension because the supervisor already intervened."
            )
            return False
        status, payload = read_http_json("/api/aria/cycle-status")
        if (
            status != 200
            or not payload.get("is_running")
            or not payload.get("continuous_active")
        ):
            self.log(
                f"Skipping watch-window extension because cycle status is not clean: status={status} payload={payload}"
            )
            return False
        db_health = self.db_health_snapshot()
        if any(db_health.values()):
            self.log(
                f"Skipping watch-window extension because DB health is non-zero: {db_health}"
            )
            return False
        self.extended_once = True
        self.max_wall_minutes += extension_minutes
        self.check_minutes = build_check_minutes(self.max_wall_minutes)
        self.log(
            f"Clean run detected; extending watch window by {extension_minutes} minutes "
            f"to total {self.max_wall_minutes} minutes."
        )
        return True

    def run_shutdown_command(self) -> int:
        command = str(self.shutdown_command or "").strip()
        if not command:
            return 0
        self.log(f"Running shutdown command: {command}")
        proc = subprocess.run(
            command,
            cwd=ROOT,
            shell=True,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.stdout.strip():
            self.log(f"Shutdown command stdout:\n{proc.stdout.strip()}")
        if proc.stderr.strip():
            self.log(f"Shutdown command stderr:\n{proc.stderr.strip()}")
        self.log(f"Shutdown command exit code: {proc.returncode}")
        return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Supervise Aria continuous mode on a bounded cadence."
    )
    parser.add_argument(
        "--total-minutes",
        type=int,
        default=10 * 60,
        help="Total wall-clock watch window in minutes.",
    )
    parser.add_argument(
        "--activity-stall-minutes",
        type=int,
        default=25,
        help="Minutes of no meaningful log or experiment activity before declaring a stall.",
    )
    parser.add_argument(
        "--shutdown-command",
        type=str,
        default="",
        help="Shell command to run after the watch window completes cleanly.",
    )
    parser.add_argument(
        "--stellar-extension-minutes",
        type=int,
        default=DEFAULT_STELLAR_EXTENSION_MINUTES,
        help="One-time extension added after a clean run with no interventions.",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"overnight_continuous_supervisor_{ts}.log"
    dashboard_output_path = LOG_DIR / f"overnight_dashboard_{ts}.log"
    sup = Supervisor(
        log_path=log_path,
        dashboard_output_path=dashboard_output_path,
        check_minutes=build_check_minutes(args.total_minutes),
        max_wall_minutes=int(args.total_minutes),
        shutdown_command=args.shutdown_command or None,
        stellar_extension_minutes=int(args.stellar_extension_minutes),
        activity_stall_minutes=int(args.activity_stall_minutes),
    )
    try:
        rc = sup.run()
        if rc == 0 and sup.shutdown_command:
            shutdown_rc = sup.run_shutdown_command()
            if shutdown_rc != 0:
                return shutdown_rc
        return rc
    except KeyboardInterrupt:
        sup.log("Supervisor interrupted by user.")
        return 130
    except Exception as exc:
        sup.log(f"Supervisor failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
