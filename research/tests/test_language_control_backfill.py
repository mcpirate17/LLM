from __future__ import annotations

import sqlite3

from research.scientist.language_control_gates import LANGUAGE_CONTROL_NB_GATES
from research.tools.language_control_backfill import (
    _apply_first_language_control_failure,
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
            language_control_metric_version TEXT,
            language_control_s05_sentence_assoc_score REAL,
            language_control_s10_sentence_assoc_score REAL,
            language_control_investigation_sentence_assoc_score REAL,
            language_control_s05_binding_order_acc REAL,
            language_control_s05_binding_score REAL,
            language_control_s10_binding_order_acc REAL,
            language_control_s10_binding_score REAL,
            language_control_investigation_binding_order_acc REAL,
            language_control_investigation_binding_score REAL,
            language_control_s10_checkpoints_json TEXT,
            language_control_investigation_checkpoints_json TEXT,
            fp_jacobian_erf_density REAL,
            fp_jacobian_erf_decay_slope REAL,
            graph_category_histogram TEXT,
            failure_op TEXT,
            failure_details_json TEXT
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
            language_control_metric_version,
            language_control_s05_sentence_assoc_score,
            language_control_s10_sentence_assoc_score,
            language_control_investigation_sentence_assoc_score,
            language_control_s05_binding_order_acc,
            language_control_s05_binding_score,
            language_control_s10_binding_order_acc,
            language_control_s10_binding_score,
            language_control_investigation_binding_order_acc,
            language_control_investigation_binding_score,
            language_control_s10_checkpoints_json,
            language_control_investigation_checkpoints_json,
            fp_jacobian_erf_density,
            fp_jacobian_erf_decay_slope,
            graph_category_histogram
        )
        VALUES (
            'r1',
            'fp1',
            '{"nodes":[]}',
            'language_control_v2',
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
            '[]',
            0.015625,
            -0.04,
            '{}'
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
    assert targets[0]["_gate_only_tier"] == "s05_nb"
    assert targets[0]["_s05_gate_only"] is True


def test_select_targets_returns_gate_only_for_existing_s05_sa_no_escape(
    tmp_path,
) -> None:
    db_path = tmp_path / "backfill.db"
    _seed_backfill_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        UPDATE program_results
        SET language_control_s05_binding_score = 0.80,
            language_control_s05_sentence_assoc_score = 0.40,
            fp_jacobian_erf_density = 0.015625,
            fp_jacobian_erf_decay_slope = -0.04,
            graph_category_histogram = '{}'
        WHERE result_id = 'r1'
        """
    )
    conn.commit()
    conn.close()

    targets = _select_targets(
        db_path,
        top_n=10,
        force=False,
        required_tiers=("s05", "s10", "inv"),
    )

    assert len(targets) == 1
    assert targets[0]["result_id"] == "r1"
    assert targets[0]["_gate_only_tier"] == "s05_sa"
    assert targets[0]["_s05_gate_only"] is True


def test_select_targets_allows_existing_s05_sa_mixing_escape(tmp_path) -> None:
    db_path = tmp_path / "backfill.db"
    _seed_backfill_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        UPDATE program_results
        SET language_control_s05_binding_score = 0.80,
            language_control_s05_sentence_assoc_score = 0.40,
            graph_category_histogram = '{"mixing": 1}'
        WHERE result_id = 'r1'
        """
    )
    conn.commit()
    conn.close()

    targets = _select_targets(
        db_path,
        top_n=10,
        force=False,
        required_tiers=("s05", "s10", "inv"),
    )

    assert targets == []


def test_select_targets_returns_gate_only_for_existing_s10_no_go(tmp_path) -> None:
    db_path = tmp_path / "backfill.db"
    _seed_backfill_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        UPDATE program_results
        SET language_control_s05_binding_score = 0.80,
            language_control_s10_binding_score = 0.62
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
    assert targets[0]["_gate_only_tier"] == "s10_nb"
    assert targets[0]["_s05_gate_only"] is False


def test_select_targets_returns_gate_only_for_existing_s10_nb_sa_no_go(
    tmp_path,
) -> None:
    db_path = tmp_path / "backfill.db"
    _seed_backfill_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        UPDATE program_results
        SET language_control_s05_binding_score = 0.80,
            language_control_s10_sentence_assoc_score = 0.25,
            language_control_s10_binding_score = 0.75
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
    assert targets[0]["_gate_only_tier"] == "s10_nb_sa"
    assert targets[0]["_s05_gate_only"] is False


def test_backfill_gate_application_screens_out_existing_s10_no_go(tmp_path) -> None:
    db_path = tmp_path / "backfill.db"
    _seed_backfill_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("ALTER TABLE leaderboard ADD COLUMN notes TEXT")
    conn.execute(
        """
        UPDATE program_results
        SET language_control_s05_binding_score = 0.80,
            language_control_s10_binding_score = 0.62
        WHERE result_id = 'r1'
        """
    )

    failure_op = _apply_first_language_control_failure(
        conn,
        result_id="r1",
        updates={
            LANGUAGE_CONTROL_NB_GATES["s10"]["score_key"]: 0.62,
        },
        context={},
        source="test",
    )
    conn.commit()

    assert failure_op == "language_control_s10_nb"
    assert conn.execute("SELECT tier FROM leaderboard").fetchone()[0] == "screened_out"
    pr = conn.execute("SELECT failure_op FROM program_results").fetchone()
    assert pr[0] == "language_control_s10_nb"
    conn.close()


def test_backfill_gate_application_screens_out_existing_s10_nb_sa_no_go(
    tmp_path,
) -> None:
    db_path = tmp_path / "backfill.db"
    _seed_backfill_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("ALTER TABLE leaderboard ADD COLUMN notes TEXT")

    failure_op = _apply_first_language_control_failure(
        conn,
        result_id="r1",
        updates={
            "language_control_s10_sentence_assoc_score": 0.25,
            "language_control_s10_binding_score": 0.75,
        },
        context={},
        source="test",
    )
    conn.commit()

    assert failure_op == "language_control_s10_nb_sa"
    assert conn.execute("SELECT tier FROM leaderboard").fetchone()[0] == "screened_out"
    pr = conn.execute("SELECT failure_op FROM program_results").fetchone()
    assert pr[0] == "language_control_s10_nb_sa"
    conn.close()


def test_backfill_gate_application_screens_out_existing_inv_no_go(tmp_path) -> None:
    db_path = tmp_path / "backfill.db"
    _seed_backfill_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("ALTER TABLE leaderboard ADD COLUMN notes TEXT")

    failure_op = _apply_first_language_control_failure(
        conn,
        result_id="r1",
        updates={
            LANGUAGE_CONTROL_NB_GATES["inv"]["score_key"]: 0.64,
        },
        context={},
        source="test",
    )
    conn.commit()

    assert failure_op == "language_control_investigation_binding"
    assert conn.execute("SELECT tier FROM leaderboard").fetchone()[0] == "screened_out"
    pr = conn.execute("SELECT failure_op FROM program_results").fetchone()
    assert pr[0] == "language_control_investigation_binding"
    conn.close()


def test_backfill_gate_application_allows_existing_inv_pass(tmp_path) -> None:
    db_path = tmp_path / "backfill.db"
    _seed_backfill_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("ALTER TABLE leaderboard ADD COLUMN notes TEXT")

    failure_op = _apply_first_language_control_failure(
        conn,
        result_id="r1",
        updates={
            LANGUAGE_CONTROL_NB_GATES["inv"]["score_key"]: 0.65,
        },
        context={},
        source="test",
    )
    conn.commit()

    assert failure_op is None
    assert conn.execute("SELECT tier FROM leaderboard").fetchone()[0] == "validation"
    pr = conn.execute("SELECT failure_op FROM program_results").fetchone()
    assert pr[0] is None
    conn.close()
