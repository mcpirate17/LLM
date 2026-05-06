"""Hourly auto-snapshot rotation for ``lab_notebook.db``.

Runs as a daemon thread inside the dashboard process. Every
``SNAPSHOT_INTERVAL_S`` seconds it copies the live notebook to
``<db>.snap_<UTC ISO>`` using SQLite's online backup API (safe against
concurrent writers), checks snapshot health, then prunes any snapshots
beyond ``KEEP_LAST``. The default retention keeps the last six hourly
snapshots.

This is the redundancy baseline that lets us drop the process-wide
aria-db writer flock: if the main DB corrupts, we fall back to the
most recent snapshot.

The SQLite online backup is performed against a separate read-only
sqlite3 connection. Snapshots that fail ``PRAGMA quick_check`` are
renamed with a ``.bad`` suffix and excluded from healthy retention.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from research.tools.db_health import HealthCheckError, assert_sqlite_health

logger = logging.getLogger(__name__)

SNAPSHOT_INTERVAL_S = 3600.0
KEEP_LAST = 6
SNAPSHOT_SUFFIX_PREFIX = ".snap_"
SNAPSHOT_TS_RE = re.compile(r"\.snap_(\d{8}T\d{6})$")

_ROTATOR_LOCK = threading.Lock()
_STARTED_FOR_PATHS: set[str] = set()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _list_snapshots(db_path: Path) -> List[Path]:
    parent = db_path.parent
    base = db_path.name
    matches: List[Path] = []
    for entry in parent.iterdir():
        if not entry.name.startswith(base + SNAPSHOT_SUFFIX_PREFIX):
            continue
        if SNAPSHOT_TS_RE.search(entry.name):
            matches.append(entry)
    matches.sort(key=lambda p: p.name)
    return matches


def _prune_old_snapshots(db_path: Path, keep_last: int) -> int:
    snaps = _list_snapshots(db_path)
    if len(snaps) <= keep_last:
        return 0
    removed = 0
    for stale in snaps[: len(snaps) - keep_last]:
        try:
            stale.unlink()
            removed += 1
        except OSError as exc:
            logger.warning("snapshot prune failed for %s: %s", stale, exc)
    return removed


def _mark_bad_snapshot(path: Path, reason: BaseException) -> None:
    bad = path.with_name(f"{path.name}.bad")
    try:
        path.replace(bad)
        logger.warning("snapshot health failed; moved %s -> %s: %s", path, bad, reason)
    except OSError as exc:
        logger.warning(
            "snapshot health failed and quarantine failed for %s: %s",
            path,
            exc,
        )


def _snapshot_is_healthy(path: Path) -> bool:
    try:
        assert_sqlite_health(path, label="notebook snapshot")
        return True
    except (HealthCheckError, sqlite3.Error, OSError) as exc:
        _mark_bad_snapshot(path, exc)
        return False


def take_snapshot(db_path: str | os.PathLike[str]) -> Optional[Path]:
    """Take one snapshot of ``db_path`` using SQLite online backup.

    Returns the snapshot ``Path`` on success, ``None`` on failure. Safe
    against concurrent aria-db writers.
    """
    src = Path(db_path)
    if not src.is_file():
        logger.debug("snapshot skipped: %s does not exist", src)
        return None
    dst = src.with_name(f"{src.name}{SNAPSHOT_SUFFIX_PREFIX}{_utc_stamp()}")
    src_uri = f"file:{src}?mode=ro"
    try:
        src_conn = sqlite3.connect(src_uri, uri=True, timeout=30.0)
    except sqlite3.OperationalError as exc:
        logger.warning("snapshot source open failed for %s: %s", src, exc)
        return None
    try:
        dst_conn = sqlite3.connect(str(dst), timeout=30.0)
    except sqlite3.OperationalError as exc:
        src_conn.close()
        logger.warning("snapshot dest open failed for %s: %s", dst, exc)
        return None
    try:
        with dst_conn:
            src_conn.backup(dst_conn, pages=2000, sleep=0.05)
    except sqlite3.Error as exc:
        logger.warning("snapshot backup failed for %s: %s", dst, exc)
        try:
            dst.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    finally:
        dst_conn.close()
        src_conn.close()
    return dst if _snapshot_is_healthy(dst) else None


def _rotator_loop(
    db_path: Path,
    interval_s: float,
    keep_last: int,
    stop_event: threading.Event,
) -> None:
    if stop_event.wait(interval_s):
        return
    while not stop_event.is_set():
        t0 = time.monotonic()
        try:
            snap = take_snapshot(db_path)
            if snap is not None:
                pruned = _prune_old_snapshots(db_path, keep_last)
                logger.info(
                    "notebook snapshot taken: %s (pruned %d older)",
                    snap.name,
                    pruned,
                )
        except Exception:
            logger.exception("snapshot rotator iteration failed; will retry")
        elapsed = time.monotonic() - t0
        wait_for = max(60.0, interval_s - elapsed)
        if stop_event.wait(wait_for):
            return


def ensure_snapshot_rotator(
    notebook_path: str | os.PathLike[str],
    *,
    interval_s: float = SNAPSHOT_INTERVAL_S,
    keep_last: int = KEEP_LAST,
) -> None:
    """Start the daemon rotator thread for ``notebook_path``, idempotent."""
    db_path = Path(notebook_path).resolve()
    key = str(db_path)
    with _ROTATOR_LOCK:
        if key in _STARTED_FOR_PATHS:
            return
        _STARTED_FOR_PATHS.add(key)
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_rotator_loop,
        args=(db_path, interval_s, keep_last, stop_event),
        name="notebook-snapshot-rotator",
        daemon=True,
    )
    thread.start()
    logger.info(
        "notebook snapshot rotator started: interval=%.0fs keep_last=%d db=%s",
        interval_s,
        keep_last,
        db_path,
    )
