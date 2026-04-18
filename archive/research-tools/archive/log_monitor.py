#!/usr/bin/env python3
"""Aria Log Monitor — watches aria_dashboard.log and writes structured alerts.

Tails the dashboard log, classifies events, tracks experiment lifecycle,
and writes alerts + summaries to a JSON file that can be read by Claude
or any other agent.

Usage:
    python -m research.tools.log_monitor                    # watch default log
    python -m research.tools.log_monitor --log path/to.log  # custom log
    python -m research.tools.log_monitor --alert-file alerts.json

The monitor writes to research/monitor_alerts.json with:
  - current_experiment: what's running, how many programs, S1 rate
  - recent_errors: last N errors with timestamps and context
  - experiment_transitions: when experiments start/stop/fail
  - health_summary: is the pipeline healthy or broken?
  - action_needed: list of issues that need human/agent intervention

Kill with Ctrl+C. Safe to restart — reads from current log position.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentState:
    experiment_id: str = ""
    hypothesis: str = ""
    mode: str = ""
    started_at: float = 0.0
    programs_generated: int = 0
    s1_survivors: int = 0
    errors: int = 0
    last_activity: float = 0.0


@dataclass
class MonitorState:
    current_experiment: ExperimentState = field(default_factory=ExperimentState)
    experiments_seen: int = 0
    total_programs: int = 0
    total_s1: int = 0
    total_errors: int = 0
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=20))
    experiment_transitions: deque = field(default_factory=lambda: deque(maxlen=50))
    action_needed: List[str] = field(default_factory=list)
    health: str = "unknown"
    started_at: float = field(default_factory=time.time)
    last_write: float = 0.0
    consecutive_failures: int = 0
    db_lock_count: int = 0
    investigation_failure_count: int = 0
    triage_count: int = 0


# ── Pattern matchers ─────────────────────────────────────────────────

_PATTERNS = {
    "s1_survivor": re.compile(r"S1 SURVIVOR \[(\d+)\] (\w+) .* loss_ratio=([\d.]+)"),
    "experiment_start": re.compile(r"Cycle (\d+): mode=(\w+)"),
    "experiment_done": re.compile(r"Cycle (\d+) done: S0=(\d+) S0\.5=(\d+) S1=(\d+)"),
    "screening_pass": re.compile(r"Rapid screening PASSED"),
    "screening_kill": re.compile(r"Rapid screening KILLED.*: (.+)"),
    "triage": re.compile(r"Triage: (\d+) fields"),
    "wikitext": re.compile(r"Screening WikiText ppl=([\d.]+) score=([\d.]+)"),
    "investigation_fail": re.compile(r"Inline investigation failed: (.+)"),
    "investigation_start": re.compile(r"Investigation plan"),
    "db_locked": re.compile(r"database is locked"),
    "runtime_error": re.compile(r"RuntimeError"),
    "error_line": re.compile(r"ERROR\s+\[([^\]]+)\]\s+(.+)"),
    "warning_fail": re.compile(
        r"WARNING.*(?:failed|crash|locked|error).*", re.IGNORECASE
    ),
    "quality_gate": re.compile(r"Quality gate.*s0=(\w+) s1=(\w+) lr=(\S+)"),
    "auto_escalate": re.compile(r"Auto-escalat"),
    "experiment_failed": re.compile(r"Deleted zero-value failed experiment (\w+)"),
    "recurring_error": re.compile(r"RECURRING ERROR.*: (.+)"),
}


def _classify_line(line: str, state: MonitorState) -> Optional[Dict[str, Any]]:
    """Classify a log line and update state. Returns event dict or None."""
    ts_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
    timestamp = ts_match.group(1) if ts_match else ""

    # S1 survivor
    m = _PATTERNS["s1_survivor"].search(line)
    if m:
        state.current_experiment.s1_survivors += 1
        state.current_experiment.last_activity = time.time()
        state.total_s1 += 1
        return {
            "type": "s1_survivor",
            "ts": timestamp,
            "count": int(m.group(1)),
            "result_id": m.group(2),
            "loss_ratio": float(m.group(3)),
        }

    # Experiment cycle start
    m = _PATTERNS["experiment_start"].search(line)
    if m:
        cycle = int(m.group(1))
        mode = m.group(2)
        if cycle == 1 or mode != state.current_experiment.mode:
            # New experiment
            if state.current_experiment.experiment_id:
                state.experiment_transitions.append(
                    {
                        "type": "experiment_end",
                        "ts": timestamp,
                        "experiment_id": state.current_experiment.experiment_id,
                        "programs": state.current_experiment.programs_generated,
                        "s1": state.current_experiment.s1_survivors,
                        "errors": state.current_experiment.errors,
                    }
                )
            state.current_experiment = ExperimentState(
                mode=mode,
                started_at=time.time(),
                last_activity=time.time(),
            )
            state.experiments_seen += 1
            state.experiment_transitions.append(
                {
                    "type": "experiment_start",
                    "ts": timestamp,
                    "cycle": cycle,
                    "mode": mode,
                    "experiment_number": state.experiments_seen,
                }
            )
        state.current_experiment.last_activity = time.time()
        return None

    # Experiment cycle done
    m = _PATTERNS["experiment_done"].search(line)
    if m:
        s1 = int(m.group(4))
        state.current_experiment.programs_generated += int(m.group(2))
        state.current_experiment.s1_survivors += s1
        state.current_experiment.last_activity = time.time()
        state.total_programs += int(m.group(2))
        if s1 == 0:
            state.consecutive_failures += 1
        else:
            state.consecutive_failures = 0
        return {
            "type": "cycle_done",
            "ts": timestamp,
            "s0": int(m.group(2)),
            "s1": s1,
        }

    # Triage
    if _PATTERNS["triage"].search(line):
        state.triage_count += 1
        state.current_experiment.last_activity = time.time()
        return None

    # Investigation failure
    m = _PATTERNS["investigation_fail"].search(line)
    if m:
        state.investigation_failure_count += 1
        state.total_errors += 1
        state.current_experiment.errors += 1
        error_msg = m.group(1)
        state.recent_errors.append(
            {
                "ts": timestamp,
                "type": "investigation_failure",
                "message": error_msg,
            }
        )
        if state.investigation_failure_count == 3:
            state.action_needed.append(
                f"Investigation failing repeatedly: {error_msg[:80]}"
            )
        return {"type": "investigation_failure", "ts": timestamp, "message": error_msg}

    # DB locked
    if _PATTERNS["db_locked"].search(line):
        state.db_lock_count += 1
        state.recent_errors.append(
            {
                "ts": timestamp,
                "type": "db_locked",
            }
        )
        if state.db_lock_count == 5:
            state.action_needed.append(
                "Database lock contention detected — add PRAGMA busy_timeout"
            )
        return {"type": "db_locked", "ts": timestamp}

    # ERROR log line
    m = _PATTERNS["error_line"].search(line)
    if m:
        module = m.group(1)
        message = m.group(2)
        state.total_errors += 1
        state.current_experiment.errors += 1
        state.recent_errors.append(
            {
                "ts": timestamp,
                "type": "error",
                "module": module,
                "message": message[:200],
            }
        )
        return {
            "type": "error",
            "ts": timestamp,
            "module": module,
            "message": message[:100],
        }

    # Recurring error
    m = _PATTERNS["recurring_error"].search(line)
    if m:
        state.action_needed.append(f"Recurring error: {m.group(1)[:80]}")
        return {"type": "recurring_error", "ts": timestamp, "message": m.group(1)}

    # Experiment failed/deleted
    m = _PATTERNS["experiment_failed"].search(line)
    if m:
        state.consecutive_failures += 1
        if state.consecutive_failures >= 5:
            state.action_needed.append(
                f"{state.consecutive_failures} consecutive failed experiments — pipeline may be stuck"
            )
        return {"type": "experiment_deleted", "ts": timestamp, "id": m.group(1)}

    return None


def _compute_health(state: MonitorState) -> str:
    """Compute overall health status."""
    if state.consecutive_failures >= 5:
        return "critical"
    if state.db_lock_count >= 5:
        return "degraded"
    if state.investigation_failure_count >= 3:
        return "degraded"
    if state.total_errors > state.total_programs * 0.5 and state.total_programs > 10:
        return "degraded"
    if state.total_programs == 0 and time.time() - state.started_at > 300:
        return "stalled"
    if state.total_s1 > 0:
        return "healthy"
    if state.total_programs > 0:
        return "running"
    return "starting"


def _write_alert_file(state: MonitorState, path: Path) -> None:
    """Write current state to JSON alert file."""
    state.health = _compute_health(state)

    uptime = time.time() - state.started_at
    s1_rate = state.total_s1 / max(state.total_programs, 1) * 100

    alert = {
        "timestamp": time.time(),
        "uptime_seconds": round(uptime),
        "health": state.health,
        "current_experiment": {
            "mode": state.current_experiment.mode,
            "programs": state.current_experiment.programs_generated,
            "s1": state.current_experiment.s1_survivors,
            "errors": state.current_experiment.errors,
            "idle_seconds": round(time.time() - state.current_experiment.last_activity)
            if state.current_experiment.last_activity > 0
            else 0,
        },
        "totals": {
            "experiments": state.experiments_seen,
            "programs": state.total_programs,
            "s1_passers": state.total_s1,
            "s1_rate": round(s1_rate, 1),
            "errors": state.total_errors,
            "triage_runs": state.triage_count,
            "db_locks": state.db_lock_count,
            "investigation_failures": state.investigation_failure_count,
            "consecutive_failures": state.consecutive_failures,
        },
        "recent_errors": list(state.recent_errors),
        "experiment_transitions": list(state.experiment_transitions),
        "action_needed": state.action_needed,
    }

    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(alert, f, indent=2)
    tmp.rename(path)
    state.last_write = time.time()


def monitor(log_path: str, alert_path: str, write_interval: float = 10.0) -> None:
    """Main monitor loop — tail log, classify, write alerts."""
    log = Path(log_path)
    alerts = Path(alert_path)
    state = MonitorState()

    if not log.exists():
        print(f"Log file not found: {log}", file=sys.stderr)
        sys.exit(1)

    # Start from end of file
    fh = open(log, "r")
    fh.seek(0, 2)  # seek to end
    print(f"Monitoring {log} → {alerts}", file=sys.stderr)
    print("Ctrl+C to stop", file=sys.stderr)

    try:
        while True:
            line = fh.readline()
            if not line:
                # No new data — check if we should write summary
                if time.time() - state.last_write >= write_interval:
                    _write_alert_file(state, alerts)

                # Check for stall
                idle = time.time() - state.current_experiment.last_activity
                if idle > 600 and state.current_experiment.last_activity > 0:
                    if "Pipeline stalled" not in str(state.action_needed):
                        state.action_needed.append(
                            f"Pipeline stalled — no activity for {idle:.0f}s"
                        )

                time.sleep(0.5)
                continue

            line = line.strip()
            if not line:
                continue

            event = _classify_line(line, state)

            # Print significant events to stderr
            if event and event["type"] in (
                "s1_survivor",
                "error",
                "investigation_failure",
                "db_locked",
                "recurring_error",
                "experiment_deleted",
            ):
                print(
                    f"[{event.get('ts', '?')}] {event['type']}: "
                    f"{event.get('message', event.get('result_id', event.get('id', '')))[:80]}",
                    file=sys.stderr,
                )

            # Write alerts periodically
            if time.time() - state.last_write >= write_interval:
                _write_alert_file(state, alerts)

    except KeyboardInterrupt:
        _write_alert_file(state, alerts)
        print(f"\nFinal summary written to {alerts}", file=sys.stderr)
        print(
            f"Health: {state.health} | "
            f"Experiments: {state.experiments_seen} | "
            f"Programs: {state.total_programs} | "
            f"S1: {state.total_s1} ({state.total_s1 / max(state.total_programs, 1) * 100:.0f}%) | "
            f"Errors: {state.total_errors}",
            file=sys.stderr,
        )
    finally:
        fh.close()


def main():
    parser = argparse.ArgumentParser(description="Monitor Aria dashboard log")
    parser.add_argument(
        "--log",
        default="research/aria_dashboard.log",
        help="Log file to monitor",
    )
    parser.add_argument(
        "--alert-file",
        default="research/monitor_alerts.json",
        help="Output alert file",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Write interval in seconds",
    )
    args = parser.parse_args()
    monitor(args.log, args.alert_file, args.interval)


if __name__ == "__main__":
    main()
