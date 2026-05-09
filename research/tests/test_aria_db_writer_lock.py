"""Regression: aria-db writer flock prevents the orphan-WAL failure mode.

Background (2026-04-16, second order):

Before this hardening, any second Python process that opened the
production ``lab_notebook.db`` for writing could, on its exit, trigger
SQLite's close-time teardown — which sometimes unlinks the WAL file.
The long-running dashboard process still held an open FD pointing at
the now-deleted inode, so its subsequent writes went to a WAL that no
reader could ever see. Hours of program_result rows vanished while the
main .db file's newest timestamp stayed frozen.

The fix: the ``aria_db::ConnectionManager::new`` constructor now
acquires an exclusive advisory flock on ``<db_path>.writer-lock``. A
second writer fails fast with a clear error, and read-only callers can
still open via ``get_manager_readonly`` without taking the lock.

These tests exercise both paths against a throwaway DB so the
production notebook stays untouched.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import aria_db


@pytest.fixture
def tmp_db(tmp_path):
    """Yield a fresh DB path. Tear down by closing the manager cache."""
    db = tmp_path / "wlock_test.db"
    # Touch the file so the parent dir exists / perms are normal.
    db.parent.mkdir(parents=True, exist_ok=True)
    yield str(db)


def _spawn_second_writer(db_path: str) -> subprocess.CompletedProcess:
    """Run ``aria_db.get_manager(db_path)`` in a fresh subprocess.

    Returns the completed process. When the parent already holds the
    writer lock on ``db_path`` the child must exit non-zero.
    """
    script = textwrap.dedent(
        f"""
        import aria_db, sys
        try:
            aria_db.get_manager({db_path!r})
        except RuntimeError as e:
            print('EXPECTED:', str(e))
            sys.exit(17)
        print('UNEXPECTED: second writer acquired the lock')
        sys.exit(0)
        """
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_writer_manager_holds_flock(tmp_db) -> None:
    mgr = aria_db.get_manager(tmp_db)
    assert mgr.holds_writer_lock is True
    assert mgr.read_only is False


def test_readonly_manager_does_not_hold_flock(tmp_db) -> None:
    # Touch the db via a writer so there's something to read.
    writer = aria_db.get_manager(tmp_db)
    writer.execute("CREATE TABLE IF NOT EXISTS t (x INTEGER)", ())
    writer.submit_write("INSERT INTO t VALUES (?)", (42,))
    writer.flush_writes(5.0)
    # Checkpoint so immutable reader can see the data (immutable mode
    # reads the main .db file only, not the WAL).
    writer.checkpoint()

    ro = aria_db.get_manager_readonly(tmp_db)
    assert ro.holds_writer_lock is False
    assert ro.read_only is True

    rows = ro.fetchall("SELECT x FROM t", ())
    assert rows and rows[0].get("x") == 42


def test_readonly_manager_rejects_writes(tmp_db) -> None:
    aria_db.get_manager(tmp_db)  # ensure db exists
    ro = aria_db.get_manager_readonly(tmp_db)
    with pytest.raises(RuntimeError, match="read-only"):
        ro.submit_write("INSERT INTO t VALUES (?)", (1,))


def test_second_writer_is_rejected_with_clear_error(tmp_db) -> None:
    """A second Python process must fail to take the writer lock.

    This is the regression for the 2026-04-16 orphan-WAL incident.
    Before the fix, the second writer would succeed and go on to
    corrupt the WAL on exit. Now it must refuse.
    """
    _primary = aria_db.get_manager(tmp_db)
    assert _primary.holds_writer_lock is True

    result = _spawn_second_writer(tmp_db)
    assert result.returncode == 17, (
        f"Second writer should have been rejected with exit code 17, "
        f"got {result.returncode}. stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    # Error message must mention the lockfile and that the caller should
    # stop the other process.
    assert "already holds the writer lock" in result.stdout
    assert ".writer-lock" in result.stdout


def test_labnotebook_read_only_opens_without_lock(tmp_db) -> None:
    """LabNotebook(read_only=True) must not take the writer lock."""
    from research.scientist.notebook import LabNotebook

    nb_rw = LabNotebook(db_path=tmp_db)
    nb_rw.flush_writes()

    nb_ro = LabNotebook(db_path=tmp_db, read_only=True)
    # Both reads should work; writes from the RO handle must raise.
    assert nb_ro.conn.execute("PRAGMA query_only").fetchall() is not None
    with pytest.raises(RuntimeError, match="read-only"):
        nb_ro.conn._mgr.submit_write("CREATE TABLE foo (x INT)", ())


def test_api_write_notebook_uses_native_single_writer(tmp_db) -> None:
    """Writable API notebooks must not bypass aria-db with raw sqlite3."""
    from research.scientist.api_routes.deps import get_notebook

    nb = get_notebook(tmp_db, read_only=False)

    assert nb._use_native is True
    assert nb.conn._mgr.holds_writer_lock is True
