"""SQLite health and backup helpers for lab-notebook maintenance tools."""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path
from typing import Iterable


DEFAULT_CHECKS = ("quick_check",)


class HealthCheckError(RuntimeError):
    """Raised when SQLite reports anything other than ``ok``."""


def run_sqlite_health_check(
    db_path: str | Path,
    *,
    checks: Iterable[str] = DEFAULT_CHECKS,
) -> dict[str, list[str]]:
    """Run SQLite integrity pragmas and return their raw result lines."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(path)

    results: dict[str, list[str]] = {}
    with sqlite3.connect(str(path)) as conn:
        for check in checks:
            normalized = str(check).strip().lower()
            if normalized not in {"quick_check", "integrity_check"}:
                raise ValueError(f"unsupported sqlite health check: {check}")
            rows = conn.execute(f"PRAGMA {normalized};").fetchall()
            results[normalized] = [str(row[0]) for row in rows]
    return results


def assert_sqlite_health(
    db_path: str | Path,
    *,
    checks: Iterable[str] = DEFAULT_CHECKS,
    label: str | None = None,
) -> dict[str, list[str]]:
    """Run SQLite health checks and raise if any check fails."""
    results = run_sqlite_health_check(db_path, checks=checks)
    failures = {
        check: lines
        for check, lines in results.items()
        if lines != ["ok"]
    }
    if failures:
        prefix = f"{label}: " if label else ""
        detail = "; ".join(f"{check}={lines!r}" for check, lines in failures.items())
        raise HealthCheckError(f"{prefix}sqlite health check failed for {db_path}: {detail}")
    return results


def backup_sqlite_db(db_path: str | Path, *, suffix: str) -> Path:
    """Create a consistent SQLite backup using the backup API."""
    path = Path(db_path)
    ts = time.strftime("%Y%m%dT%H%M%S")
    backup_path = path.with_name(f"{path.name}.{suffix}_{ts}")
    with sqlite3.connect(str(path)) as src, sqlite3.connect(str(backup_path)) as dst:
        src.backup(dst)
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="research/lab_notebook.db")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run integrity_check as well as quick_check.",
    )
    args = parser.parse_args()

    checks = ("quick_check", "integrity_check") if args.full else DEFAULT_CHECKS
    results = assert_sqlite_health(args.db, checks=checks)
    for check, lines in results.items():
        print(f"{check}: {'; '.join(lines)}")


if __name__ == "__main__":
    main()
