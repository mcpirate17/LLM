import sqlite3

import pytest

from research.tools.db_health import HealthCheckError, assert_sqlite_health


def test_assert_sqlite_health_accepts_clean_db(tmp_path):
    db_path = tmp_path / "clean.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE example(id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO example(value) VALUES ('ok')")

    result = assert_sqlite_health(db_path)

    assert result == {"quick_check": ["ok"]}


def test_assert_sqlite_health_rejects_missing_db(tmp_path):
    with pytest.raises(FileNotFoundError):
        assert_sqlite_health(tmp_path / "missing.db")


def test_assert_sqlite_health_rejects_unsupported_check(tmp_path):
    db_path = tmp_path / "clean.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE example(id INTEGER PRIMARY KEY)")

    with pytest.raises(ValueError):
        assert_sqlite_health(db_path, checks=("foreign_key_check",))


def test_health_error_type_is_runtime_error():
    assert issubclass(HealthCheckError, RuntimeError)
