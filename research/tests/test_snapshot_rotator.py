from __future__ import annotations

import sqlite3

from research.scientist import snapshot_rotator
from research.tools.db_health import HealthCheckError


def test_take_snapshot_uses_backup_api_and_health_checks(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO t (value) VALUES ('ok')")

    snap = snapshot_rotator.take_snapshot(db_path)

    assert snap is not None
    assert snap.exists()
    with sqlite3.connect(str(snap)) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT value FROM t").fetchone()[0] == "ok"


def test_take_snapshot_quarantines_unhealthy_snapshot(tmp_path, monkeypatch):
    db_path = tmp_path / "lab_notebook.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")

    def _fail_health(*args, **kwargs):
        raise HealthCheckError("forced bad snapshot")

    monkeypatch.setattr(snapshot_rotator, "assert_sqlite_health", _fail_health)

    snap = snapshot_rotator.take_snapshot(db_path)

    assert snap is None
    assert not [
        path
        for path in tmp_path.glob("lab_notebook.db.snap_*")
        if not path.name.endswith(".bad")
    ]
    assert list(tmp_path.glob("lab_notebook.db.snap_*.bad"))


def test_prune_old_snapshots_keeps_six_newest_by_default(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    db_path.touch()
    for hour in range(8):
        (tmp_path / f"lab_notebook.db.snap_20260505T{hour:02d}0543").touch()

    removed = snapshot_rotator._prune_old_snapshots(
        db_path,
        snapshot_rotator.KEEP_LAST,
    )

    assert snapshot_rotator.KEEP_LAST == 6
    assert removed == 2
    assert [path.name for path in snapshot_rotator._list_snapshots(db_path)] == [
        "lab_notebook.db.snap_20260505T020543",
        "lab_notebook.db.snap_20260505T030543",
        "lab_notebook.db.snap_20260505T040543",
        "lab_notebook.db.snap_20260505T050543",
        "lab_notebook.db.snap_20260505T060543",
        "lab_notebook.db.snap_20260505T070543",
    ]
