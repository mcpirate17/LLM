from __future__ import annotations

import sqlite3

from research.tools.restore_lab_notebook import restore_lab_notebook


def _write_db(path, value: str) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE t (value TEXT)")
        conn.execute("INSERT INTO t VALUES (?)", (value,))


def _read_value(path) -> str:
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT value FROM t").fetchone()
        return str(row[0])


def test_restore_lab_notebook_dry_run_does_not_replace_destination(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    source = tmp_path / "snapshot.db"
    _write_db(db_path, "current")
    _write_db(source, "snapshot")

    plan = restore_lab_notebook(source=source, db_path=db_path, apply=False)

    assert plan["source"] == str(source.resolve())
    assert _read_value(db_path) == "current"


def test_restore_lab_notebook_moves_current_db_aside(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    source = tmp_path / "snapshot.db"
    _write_db(db_path, "current")
    _write_db(source, "snapshot")

    plan = restore_lab_notebook(source=source, db_path=db_path, apply=True)

    assert _read_value(db_path) == "snapshot"
    moved = plan["moved_current_db"]
    assert "corrupt_" in moved
    assert _read_value(moved) == "current"
