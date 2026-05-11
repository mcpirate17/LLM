from __future__ import annotations

import argparse
import io
import sqlite3
from pathlib import Path

from research.tools import rescore_champion_tiny_model as tool


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE graph_runs (
            result_id TEXT PRIMARY KEY,
            experiment_id TEXT,
            final_loss REAL,
            induction_intermediate_auc REAL,
            induction_intermediate_gap_accuracies_json TEXT,
            binding_intermediate_auc REAL,
            robustness_long_ctx_combined_score REAL,
            ar_gate_held_pair_acc REAL,
            ar_gate_held_class_acc REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE training_curves (
            result_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            loss REAL,
            grad_norm REAL,
            step_time_ms REAL,
            PRIMARY KEY (result_id, step)
        )
        """
    )
    rows = [
        (
            "gpt2cal490d5",
            tool.GPT2_EXPERIMENT_ID,
            5.0,
            0.94,
            '{"4": 0.91, "8": 0.95}',
            0.90,
            0.02,
            None,
            None,
        ),
        (
            "gpt2cal87a29",
            tool.GPT2_EXPERIMENT_ID,
            4.9,
            0.84,
            '{"4": 0.81, "8": 0.86}',
            0.95,
            0.01,
            None,
            None,
        ),
        (
            "mamba-test",
            "mamba-exp",
            4.8,
            0.50,
            '{"4": 0.45, "8": 0.55}',
            0.20,
            0.03,
            0.70,
            0.60,
        ),
    ]
    conn.executemany(
        """
        INSERT INTO graph_runs (
            result_id, experiment_id, final_loss,
            induction_intermediate_auc,
            induction_intermediate_gap_accuracies_json,
            binding_intermediate_auc,
            robustness_long_ctx_combined_score,
            ar_gate_held_pair_acc,
            ar_gate_held_class_acc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    curves = {
        "gpt2cal490d5": [(0, 9.0), (500, 5.2), (1000, 5.0), (1500, 5.0), (2000, 5.0)],
        "gpt2cal87a29": [(0, 9.0), (500, 5.1), (1000, 4.9), (1500, 4.9), (2000, 4.9)],
        "mamba-test": [(0, 9.0), (500, 4.8), (1000, 4.8), (1500, 4.8)],
    }
    for result_id, points in curves.items():
        conn.executemany(
            """
            INSERT INTO training_curves(result_id, step, loss, grad_norm, step_time_ms)
            VALUES (?, ?, ?, NULL, 1.0)
            """,
            [(result_id, step, loss) for step, loss in points],
        )
    conn.commit()
    conn.close()


def _args(db_path: Path, checkpoint_root: Path, *, write: bool) -> argparse.Namespace:
    return argparse.Namespace(
        db=str(db_path),
        checkpoint_root=str(checkpoint_root),
        mamba_result_id="mamba-test",
        write=write,
        make_backup=False,
        json=False,
    )


def _columns(path: Path) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(graph_runs)")}
    finally:
        conn.close()


def test_dry_run_does_not_write_or_check_backup(tmp_path, monkeypatch):
    db_path = tmp_path / "lab.db"
    checkpoint_root = tmp_path / "artifacts"
    _make_db(db_path)

    def fail_if_called(argv):
        raise AssertionError("backup check should not run in dry-run mode")

    monkeypatch.setattr(tool, "check_backup_freshness_main", fail_if_called)
    before = _columns(db_path)
    output = io.StringIO()

    rc = tool.run(_args(db_path, checkpoint_root, write=False), output)

    assert rc == 0
    assert "mode=DRY-RUN" in output.getvalue()
    assert "mamba-test" in output.getvalue()
    assert _columns(db_path) == before


def test_write_requires_fresh_backup_before_schema_mutation(tmp_path, monkeypatch):
    db_path = tmp_path / "lab.db"
    checkpoint_root = tmp_path / "artifacts"
    _make_db(db_path)
    calls = []

    def stale_backup(argv):
        calls.append(argv)
        return 1

    monkeypatch.setattr(tool, "check_backup_freshness_main", stale_backup)
    before = _columns(db_path)

    rc = tool.run(_args(db_path, checkpoint_root, write=True), io.StringIO())

    assert rc == 1
    assert calls == [[]]
    assert _columns(db_path) == before
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT result_id FROM graph_runs WHERE result_id = 'mamba-test'"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("mamba-test",)]


def test_write_after_fresh_backup_check_persists_champion_fields(tmp_path, monkeypatch):
    db_path = tmp_path / "lab.db"
    checkpoint_root = tmp_path / "artifacts"
    _make_db(db_path)
    calls = []

    def fresh_backup(argv):
        calls.append(argv)
        return 0

    monkeypatch.setattr(tool, "check_backup_freshness_main", fresh_backup)

    rc = tool.run(_args(db_path, checkpoint_root, write=True), io.StringIO())

    assert rc == 0
    assert calls == [[]]
    columns = _columns(db_path)
    assert "champion_tiny_model_score" in columns
    assert "champion_steps_to_floor" in columns
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT champion_tiny_model_score,
                   champion_steps_to_floor,
                   champion_tiny_model_protocol_version
            FROM graph_runs
            WHERE result_id = 'mamba-test'
            """
        ).fetchone()
    finally:
        conn.close()
    assert row["champion_tiny_model_score"] > 0.0
    assert row["champion_steps_to_floor"] == 500
    assert row["champion_tiny_model_protocol_version"] == tool.SCORE_PROTOCOL_VERSION
