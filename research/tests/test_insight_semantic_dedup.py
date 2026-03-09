from __future__ import annotations

import pytest

import sqlite3
import time

from research.scientist.notebook import LabNotebook

pytestmark = pytest.mark.unit


def test_record_insight_supersedes_semantic_duplicate(tmp_path):
    db_path = tmp_path / "notebook.db"
    nb = LabNotebook(db_path)
    try:
        exp_id = "exp-semantic"
        nb.record_insight(
            "success_factor",
            "Winning combination: mul + tanh appears in 7 survivors (avg novelty 0.697).",
            exp_id,
            confidence=0.8,
        )
        nb.record_insight(
            "success_factor",
            "Winning combination: mul + tanh appears in 8 survivors (avg novelty 0.712).",
            exp_id,
            confidence=0.82,
        )

        active = nb.conn.execute(
            "SELECT * FROM insights WHERE status = 'active' AND semantic_key LIKE 'winning_combo:%'"
        ).fetchall()
        superseded = nb.conn.execute(
            "SELECT * FROM insights WHERE status = 'superseded' AND semantic_key LIKE 'winning_combo:%'"
        ).fetchall()

        assert len(active) == 1
        assert len(superseded) == 1
        assert "8 survivors" in active[0]["content"]
    finally:
        nb.close()


def test_migrate_backfills_semantic_key_and_supersedes_existing_duplicates(tmp_path):
    db_path = tmp_path / "legacy_notebook.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE insights (
            insight_id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            experiment_id TEXT,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            supporting_evidence TEXT,
            status TEXT DEFAULT 'active',
            confirmation_strength REAL DEFAULT 0.0,
            independent_validations INTEGER DEFAULT 0
        )
        """
    )
    now = time.time()
    conn.execute(
        """INSERT INTO insights
           (insight_id, timestamp, experiment_id, category, content, confidence, status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            "old1",
            now - 5,
            "exp-old",
            "success_factor",
            "Winning combination: gelu + linear_proj_down appears in 4 survivors (avg novelty 0.646).",
            0.75,
            "active",
        ),
    )
    conn.execute(
        """INSERT INTO insights
           (insight_id, timestamp, experiment_id, category, content, confidence, status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            "old2",
            now,
            "exp-old",
            "success_factor",
            "Winning combination: gelu + linear_proj_down appears in 5 survivors (avg novelty 0.676).",
            0.78,
            "active",
        ),
    )
    conn.commit()
    conn.close()

    nb = LabNotebook(db_path)
    try:
        cols = {
            row[1]
            for row in nb.conn.execute("PRAGMA table_info(insights)").fetchall()
        }
        assert "semantic_key" in cols
        assert "insight_type" in cols
        assert "subject_key" in cols

        rows = nb.conn.execute(
            "SELECT insight_id, status, semantic_key FROM insights ORDER BY timestamp ASC"
        ).fetchall()
        assert rows[0]["semantic_key"] == rows[1]["semantic_key"]
        assert rows[0]["status"] == "superseded"
        assert rows[1]["status"] == "active"
    finally:
        nb.close()
