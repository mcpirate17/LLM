#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:5000"
DEFAULT_DASHBOARD_CMD = [
    "python",
    "-m",
    "research",
    "--mode=dashboard",
    "--port",
    "5000",
]


def _get_json(url: str) -> dict | None:
    try:
        with urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError, URLError):
        return None


def _latest_continuous_config(db_path: Path) -> dict | None:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    row = cur.execute(
        """
        SELECT config_json
        FROM experiments
        WHERE json_extract(config_json, '$.continuous') = 1
        ORDER BY timestamp DESC
        LIMIT 1
        """
    ).fetchone()
    con.close()
    if not row or not row[0]:
        return None
    try:
        payload = json.loads(row[0])
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _restart_dashboard(project_root: Path, dashboard_cmd: list[str]) -> None:
    subprocess.run(
        ["pkill", "-f", "python -m research --mode=dashboard --port 5000"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    log_path = project_root / "research" / "aria_dashboard.log"
    with log_path.open("ab") as log_file:
        subprocess.Popen(
            ["setsid", *dashboard_cmd],
            cwd=str(project_root),
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )


def _start_continuous(base_url: str, config: dict) -> bool:
    payload = dict(config)
    payload["action"] = "start"
    req = Request(
        f"{base_url}/api/aria/cycle-control",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=30) as resp:
            return 200 <= resp.status < 300
    except (OSError, URLError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch and restart continuous mode")
    parser.add_argument("--project-root", default="/home/tim/Projects/LLM")
    parser.add_argument("--db", default="research/lab_notebook.db")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--duration-seconds", type=int, default=80 * 60)
    parser.add_argument("--interval-seconds", type=int, default=30)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    db_path = (project_root / args.db).resolve()
    deadline = time.time() + max(args.duration_seconds, args.interval_seconds)

    while time.time() < deadline:
        status = _get_json(f"{args.base_url}/api/aria/cycle-status")
        if status is None:
            print(
                f"{time.strftime('%Y-%m-%dT%H:%M:%S')} dashboard unavailable; restarting"
            )
            _restart_dashboard(project_root, DEFAULT_DASHBOARD_CMD)
            time.sleep(8)
            status = _get_json(f"{args.base_url}/api/aria/cycle-status")

        need_start = status is not None and (
            not status.get("is_running")
            or not status.get("continuous_active")
            or status.get("phase") in {"failed", "completed"}
        )
        if need_start:
            config = _latest_continuous_config(db_path)
            if config:
                print(
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S')} continuous inactive; restarting"
                )
                _start_continuous(args.base_url, config)

        time.sleep(args.interval_seconds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
