from __future__ import annotations

import json
import sqlite3

from research.scientist.notebook import LabNotebook
from research.scientist.notebook.knowledge_digest_store import default_cache_path
from research.tools.migrate_knowledge_digests_cache import migrate


def _digest(ts: float, narrative: str) -> dict:
    return {
        "timestamp": ts,
        "cycle_number": 1,
        "narrative": narrative,
        "n_experiments_analyzed": 3,
        "n_curves_analyzed": 2,
    }


def test_store_digest_uses_side_cache_not_main_notebook(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(str(db_path), use_native=False)
    digest_id = nb.store_digest(_digest(10.0, "side cache"))
    latest = nb.get_latest_digest()
    nb.close()

    assert digest_id
    assert latest["narrative"] == "side cache"
    cache_path = default_cache_path(db_path)
    assert cache_path.exists()
    with sqlite3.connect(str(db_path)) as conn:
        table = conn.execute(
            """
            SELECT 1
            FROM sqlite_schema
            WHERE type='table' AND name='knowledge_digests'
            """
        ).fetchone()
    assert table is None


def test_get_latest_digest_falls_back_to_legacy_main_table(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(str(db_path), use_native=False)
    nb.conn.execute(
        """
        CREATE TABLE knowledge_digests (
            digest_id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            cycle_number INTEGER,
            digest_json TEXT NOT NULL,
            narrative_summary TEXT,
            n_experiments_analyzed INTEGER,
            n_curves_analyzed INTEGER
        )
        """
    )
    payload = _digest(20.0, "legacy fallback")
    nb.conn.execute(
        """
        INSERT INTO knowledge_digests
        (digest_id, timestamp, cycle_number, digest_json, narrative_summary,
         n_experiments_analyzed, n_curves_analyzed)
        VALUES ('d1', 20.0, 1, ?, 'legacy fallback', 3, 2)
        """,
        (json.dumps(payload),),
    )

    latest = nb.get_latest_digest()
    nb.close()

    assert latest["narrative"] == "legacy fallback"


def test_malformed_side_cache_is_quarantined(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    cache_path = default_cache_path(db_path)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("not sqlite", encoding="utf-8")

    nb = LabNotebook(str(db_path), use_native=False)
    assert nb.get_latest_digest() is None
    nb.store_digest(_digest(30.0, "after quarantine"))
    latest = nb.get_latest_digest()
    nb.close()

    assert latest["narrative"] == "after quarantine"
    assert list(cache_path.parent.glob("knowledge_digests.db.corrupt_*"))


def test_migrate_knowledge_digests_cache_best_effort(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    cache_path = tmp_path / "cache.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE knowledge_digests (
                digest_id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                cycle_number INTEGER,
                digest_json TEXT NOT NULL,
                narrative_summary TEXT,
                n_experiments_analyzed INTEGER,
                n_curves_analyzed INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO knowledge_digests
            (digest_id, timestamp, cycle_number, digest_json, narrative_summary,
             n_experiments_analyzed, n_curves_analyzed)
            VALUES ('d1', 40.0, 1, ?, 'migrated', 3, 2)
            """,
            (json.dumps(_digest(40.0, "migrated")),),
        )

    result = migrate(db_path, cache_path)

    assert result["status"] == "ok"
    assert result["copied"] == 1
    with sqlite3.connect(str(cache_path)) as conn:
        row = conn.execute("SELECT digest_json FROM knowledge_digests").fetchone()
    assert json.loads(row[0])["narrative"] == "migrated"


def test_migrate_creates_empty_side_cache_when_legacy_table_missing(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    cache_path = tmp_path / "cache.db"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE source_data (id INTEGER PRIMARY KEY)")

    result = migrate(db_path, cache_path)

    assert result["status"] == "ok"
    assert result["copied"] == 0
    assert result["legacy_table"] is False
    with sqlite3.connect(str(cache_path)) as conn:
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        count = conn.execute("SELECT COUNT(*) FROM knowledge_digests").fetchone()[0]
    assert count == 0
