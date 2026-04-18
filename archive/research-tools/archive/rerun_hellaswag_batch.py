#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


BASE = Path(__file__).resolve().parents[2]
DB = BASE / "research/lab_notebook.db"


def _ts() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sequentially rerun HellaSwag backpopulate rows with flushed progress logs."
    )
    p.add_argument(
        "--list",
        type=Path,
        required=True,
        help="Text file of result_ids, one per line.",
    )
    p.add_argument("--log", type=Path, required=True, help="Append-only progress log.")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--max-rows", type=int, default=0, help="Optional cap for smoke tests."
    )
    p.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip rows already marked hellaswag_status='ok'.",
    )
    return p.parse_args()


def _db_status(result_id: str) -> dict[str, object] | None:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT hellaswag_acc, hellaswag_status, hellaswag_n_examples "
            "FROM program_results WHERE result_id=?",
            (result_id,),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def _log_line(handle, text: str) -> None:
    print(text, file=handle, flush=True)


def main() -> int:
    args = _parse_args()
    if not args.list.exists():
        raise SystemExit(f"Missing list file: {args.list}")

    ids = [
        line.strip()
        for line in args.list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.max_rows > 0:
        ids = ids[: int(args.max_rows)]

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as log:
        _log_line(log, f"started_at {_ts()}")
        _log_line(log, f"list {args.list}")
        _log_line(log, f"count {len(ids)}")

        for i, rid in enumerate(ids, 1):
            before = _db_status(rid)
            if (
                args.skip_completed
                and before
                and before.get("hellaswag_status") == "ok"
            ):
                _log_line(
                    log,
                    f"SKIP {i}/{len(ids)} {rid} already_ok acc={before.get('hellaswag_acc')} n={before.get('hellaswag_n_examples')} {_ts()}",
                )
                continue

            _log_line(log, f"START {i}/{len(ids)} {rid} {_ts()}")
            t0 = time.time()
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "research.tools.backpopulate_screening_metrics",
                    "--result-id",
                    rid,
                    "--force",
                    "--skip-rapid",
                    "--post-train-target",
                    "hellaswag",
                    "--device",
                    str(args.device),
                    "--batch-commit",
                    "1",
                ],
                cwd=str(BASE),
                capture_output=True,
                text=True,
            )
            elapsed = round(time.time() - t0, 1)
            after = _db_status(rid)
            _log_line(
                log,
                "END "
                f"{i}/{len(ids)} {rid} rc={proc.returncode} elapsed={elapsed}s "
                f"acc={(after or {}).get('hellaswag_acc')} "
                f"status={(after or {}).get('hellaswag_status')} "
                f"n={(after or {}).get('hellaswag_n_examples')} {_ts()}",
            )

            stdout_lines = (proc.stdout or "").strip().splitlines()
            stderr_lines = (proc.stderr or "").strip().splitlines()
            for line in stdout_lines[-3:]:
                if line.strip():
                    _log_line(log, f"STDOUT {rid} {line}")
            for line in stderr_lines[-3:]:
                if line.strip():
                    _log_line(log, f"STDERR {rid} {line}")

        _log_line(log, f"finished_at {_ts()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
