from __future__ import annotations

import sqlite3

from research.scientist.controlled_lang_gates import CONTROLLED_LANG_SCORE_GATES
from research.tools.controlled_lang_backfill import (
    _apply_first_controlled_lang_failure,
    _select_targets,
)


def _seed_backfill_db(path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE leaderboard (
            entry_id TEXT PRIMARY KEY,
            result_id TEXT,
            composite_score REAL,
            tier TEXT,
            is_reference INTEGER DEFAULT 0,
            validation_passed INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE program_results (
            result_id TEXT PRIMARY KEY,
            graph_fingerprint TEXT,
            graph_json TEXT,
            controlled_lang_metric_version TEXT,
            controlled_lang_s05_sa_score REAL,
            controlled_lang_s10_sa_score REAL,
            controlled_lang_inv_sa_score REAL,
            controlled_lang_s05_nb_order_acc REAL,
            controlled_lang_s05_nb_score REAL,
            controlled_lang_s10_nb_order_acc REAL,
            controlled_lang_s10_nb_score REAL,
            controlled_lang_inv_nb_order_acc REAL,
            controlled_lang_inv_nb_score REAL,
            controlled_lang_s10_checkpoints_json TEXT,
            controlled_lang_inv_checkpoints_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE program_graph_features (
            result_id TEXT PRIMARY KEY,
            template_name TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO leaderboard(entry_id, result_id, composite_score, tier)
        VALUES ('e1', 'r1', 100.0, 'validation')
        """
    )
    conn.execute(
        """
        INSERT INTO program_results(
            result_id,
            graph_fingerprint,
            graph_json,
            controlled_lang_metric_version,
            controlled_lang_s05_sa_score,
            controlled_lang_s10_sa_score,
            controlled_lang_inv_sa_score,
            controlled_lang_s05_nb_order_acc,
            controlled_lang_s05_nb_score,
            controlled_lang_s10_nb_order_acc,
            controlled_lang_s10_nb_score,
            controlled_lang_inv_nb_order_acc,
            controlled_lang_inv_nb_score,
            controlled_lang_s10_checkpoints_json,
            controlled_lang_inv_checkpoints_json
        )
        VALUES (
            'r1',
            'fp1',
            '{"nodes":[]}',
            'controlled_lang_v2',
            1.0,
            1.0,
            1.0,
            0.80,
            0.64,
            0.80,
            0.80,
            0.80,
            0.80,
            '[]',
            '[]'
        )
        """
    )
    conn.commit()
    conn.close()


def test_select_targets_returns_gate_only_for_existing_s05_no_go(tmp_path) -> None:
    db_path = tmp_path / "backfill.db"
    _seed_backfill_db(db_path)

    targets = _select_targets(
        db_path,
        top_n=10,
        force=False,
        required_tiers=("s05", "s10", "inv"),
    )

    assert len(targets) == 1
    assert targets[0]["result_id"] == "r1"
    assert targets[0]["s05_nb"] == 0.64
    assert targets[0]["_gate_only_label"] == "controlled_lang_s05_nb"
    assert targets[0]["_s05_gate_only"] is True


def test_select_targets_returns_gate_only_for_existing_s10_no_go(tmp_path) -> None:
    db_path = tmp_path / "backfill.db"
    _seed_backfill_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        UPDATE program_results
        SET controlled_lang_s05_nb_score = 0.80,
            controlled_lang_s10_nb_score = 0.62
        WHERE result_id = 'r1'
        """
    )
    conn.commit()
    conn.close()

    targets = _select_targets(
        db_path,
        top_n=10,
        force=False,
        required_tiers=("s10",),
        target_cohorts=("validation_pending",),
        missing_before_limit=True,
    )

    assert len(targets) == 1
    assert targets[0]["result_id"] == "r1"
    assert targets[0]["_gate_only_label"] == "controlled_lang_s10_nb"
    assert targets[0]["_s05_gate_only"] is False


def test_select_targets_returns_gate_only_for_existing_s10_sa_no_go(tmp_path) -> None:
    db_path = tmp_path / "backfill.db"
    _seed_backfill_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        UPDATE program_results
        SET controlled_lang_s05_nb_score = 0.80,
            controlled_lang_s10_sa_score = 0.24,
            controlled_lang_s10_nb_score = 0.80
        WHERE result_id = 'r1'
        """
    )
    conn.commit()
    conn.close()

    targets = _select_targets(
        db_path,
        top_n=10,
        force=False,
        required_tiers=("s10",),
        target_cohorts=("validation_pending",),
        missing_before_limit=True,
    )

    assert len(targets) == 1
    assert targets[0]["result_id"] == "r1"
    assert targets[0]["_gate_only_label"] == "controlled_lang_s10_sa"
    assert targets[0]["_s05_gate_only"] is False


def test_backfill_gate_application_screens_out_existing_s10_sa_no_go(tmp_path) -> None:
    db_path = tmp_path / "backfill.db"
    _seed_backfill_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("ALTER TABLE leaderboard ADD COLUMN notes TEXT")
    conn.execute("ALTER TABLE program_results ADD COLUMN failure_op TEXT")
    conn.execute("ALTER TABLE program_results ADD COLUMN failure_details_json TEXT")
    conn.execute(
        """
        UPDATE program_results
        SET controlled_lang_s05_nb_score = 0.80,
            controlled_lang_s10_sa_score = 0.24,
            controlled_lang_s10_nb_score = 0.80
        WHERE result_id = 'r1'
        """
    )
    gate = next(
        gate
        for gate in CONTROLLED_LANG_SCORE_GATES
        if gate["failure_op"] == "controlled_lang_s10_sa"
    )

    failure_op = _apply_first_controlled_lang_failure(
        conn,
        result_id="r1",
        updates={
            gate["score_key"]: 0.24,
        },
        source="test",
    )
    conn.commit()

    assert failure_op == "controlled_lang_s10_sa"
    assert conn.execute("SELECT tier FROM leaderboard").fetchone()[0] == "screened_out"
    pr = conn.execute("SELECT failure_op FROM program_results").fetchone()
    assert pr[0] == "controlled_lang_s10_sa"
    conn.close()
