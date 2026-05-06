from __future__ import annotations

import json
import sqlite3

from research.scientist.controlled_lang_gates import (
    S05_SA_FAILURE_OP,
    S05_NB_FAILURE_OP,
    S10_NB_SA_FAILURE_OP,
    apply_controlled_lang_nb_screening_failure,
    apply_s05_sa_screening_failure,
    apply_s10_nb_sa_screening_failure,
    allows_controlled_lang_advanced_tiers,
    apply_s05_nb_screening_failure,
    controlled_lang_gate_manual_override,
    is_controlled_lang_nb_screening_failure,
    is_s05_sa_screening_failure,
    is_s05_nb_screening_failure,
    is_s10_nb_sa_screening_failure,
)
from research.tools.apply_controlled_lang_gates import (
    _candidate_rows,
    _manual_override_for_row,
)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE leaderboard (
            result_id TEXT PRIMARY KEY,
            tier TEXT,
            validation_passed INTEGER DEFAULT 0,
            is_reference INTEGER DEFAULT 0,
            notes TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE program_results (
            result_id TEXT PRIMARY KEY,
            failure_op TEXT,
            failure_details_json TEXT
        )
        """
    )
    return conn


def test_s05_nb_screening_failure_threshold_is_conservative() -> None:
    assert is_s05_nb_screening_failure(0.6499)
    assert not is_s05_nb_screening_failure(0.65)
    assert not is_s05_nb_screening_failure(None)
    assert is_controlled_lang_nb_screening_failure(0.6499)
    assert not is_controlled_lang_nb_screening_failure(0.65)


def test_s05_nb_gate_blocks_advanced_tiers_until_passed() -> None:
    assert not allows_controlled_lang_advanced_tiers(None)
    assert not allows_controlled_lang_advanced_tiers(0.6499)
    assert allows_controlled_lang_advanced_tiers(0.65)


def test_s05_sa_gate_requires_low_score_without_escape() -> None:
    assert is_s05_sa_screening_failure(
        0.64,
        erf_density=0.015625,
        erf_decay_slope=-0.04,
        graph_category_histogram='{"elementwise_binary": 1}',
    )
    assert not is_s05_sa_screening_failure(
        0.65,
        erf_density=0.015625,
        erf_decay_slope=-0.04,
        graph_category_histogram='{"elementwise_binary": 1}',
    )
    assert not is_s05_sa_screening_failure(
        0.20,
        erf_density=0.0625,
        erf_decay_slope=-0.103282,
        graph_category_histogram='{"elementwise_binary": 1}',
    )
    assert not is_s05_sa_screening_failure(
        0.20,
        erf_density=0.0,
        erf_decay_slope=0.0,
        graph_category_histogram='{"mixing": 1}',
    )


def test_s05_sa_gate_blocks_advanced_tiers_without_escape() -> None:
    assert not allows_controlled_lang_advanced_tiers(
        0.80,
        sa_score=0.30,
        erf_density=0.015625,
        erf_decay_slope=-0.04,
        graph_category_histogram="{}",
    )
    assert allows_controlled_lang_advanced_tiers(
        0.80,
        sa_score=0.30,
        erf_density=0.0625,
        erf_decay_slope=-0.103282,
        graph_category_histogram="{}",
    )
    assert allows_controlled_lang_advanced_tiers(
        0.80,
        sa_score=0.30,
        erf_density=0.015625,
        erf_decay_slope=-0.04,
        graph_category_histogram='{"mixing": 1}',
    )


def test_s10_nb_sa_gate_requires_both_scores_low() -> None:
    assert is_s10_nb_sa_screening_failure(nb_score=0.7999, sa_score=0.6499)
    assert not is_s10_nb_sa_screening_failure(nb_score=0.80, sa_score=0.20)
    assert not is_s10_nb_sa_screening_failure(nb_score=0.70, sa_score=0.65)
    assert not is_s10_nb_sa_screening_failure(nb_score=None, sa_score=0.20)
    assert not is_s10_nb_sa_screening_failure(nb_score=0.70, sa_score=None)


def test_apply_s05_nb_failure_screens_out_candidate() -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO leaderboard(result_id, tier, validation_passed) VALUES (?, ?, ?)",
        ("r1", "validation", 1),
    )
    conn.execute("INSERT INTO program_results(result_id) VALUES (?)", ("r1",))

    applied = apply_s05_nb_screening_failure(
        conn, result_id="r1", score=0.61, source="test"
    )

    assert applied
    lb = conn.execute(
        "SELECT tier, validation_passed, notes FROM leaderboard"
    ).fetchone()
    assert lb == (
        "screened_out",
        0,
        "controlled_lang_s05_nb: score 0.6100 below screening threshold 0.65",
    )
    pr = conn.execute(
        "SELECT failure_op, failure_details_json FROM program_results"
    ).fetchone()
    assert pr[0] == S05_NB_FAILURE_OP
    details = json.loads(pr[1])
    assert details["reason"] == "controlled_lang_s05_nb_below_threshold"
    assert details["stage"] == "s0.5"


def test_apply_s05_nb_failure_preserves_references() -> None:
    conn = _conn()
    conn.execute(
        """
        INSERT INTO leaderboard(result_id, tier, validation_passed, is_reference)
        VALUES (?, ?, ?, ?)
        """,
        ("ref", "validation", 1, 1),
    )
    conn.execute("INSERT INTO program_results(result_id) VALUES (?)", ("ref",))

    applied = apply_s05_nb_screening_failure(
        conn, result_id="ref", score=0.50, source="test"
    )

    assert not applied
    assert conn.execute("SELECT tier FROM leaderboard").fetchone()[0] == "validation"
    assert conn.execute("SELECT failure_op FROM program_results").fetchone()[0] is None


def test_apply_s05_sa_failure_screens_out_candidate_without_escape() -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO leaderboard(result_id, tier, validation_passed) VALUES (?, ?, ?)",
        ("r-sa", "validation", 1),
    )
    conn.execute("INSERT INTO program_results(result_id) VALUES (?)", ("r-sa",))

    applied = apply_s05_sa_screening_failure(
        conn,
        result_id="r-sa",
        score=0.40,
        erf_density=0.015625,
        erf_decay_slope=-0.04,
        graph_category_histogram="{}",
        source="test",
    )

    assert applied
    lb = conn.execute(
        "SELECT tier, validation_passed, notes FROM leaderboard"
    ).fetchone()
    assert lb == (
        "screened_out",
        0,
        "controlled_lang_s05_sa: score 0.4000 below screening threshold 0.65 "
        "without ERF/mixing escape",
    )
    pr = conn.execute(
        "SELECT failure_op, failure_details_json FROM program_results"
    ).fetchone()
    assert pr[0] == S05_SA_FAILURE_OP
    details = json.loads(pr[1])
    assert details["reason"] == "controlled_lang_s05_sa_below_threshold_without_escape"
    assert details["controlled_lang_s05_sa_score"] == 0.40
    assert details["graph_has_mixing"] is False


def test_manual_override_preserves_known_s05_sa_candidate() -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO leaderboard(result_id, tier, validation_passed) VALUES (?, ?, ?)",
        ("1b70ab74-c98", "validation", 1),
    )
    conn.execute(
        "INSERT INTO program_results(result_id) VALUES (?)",
        ("1b70ab74-c98",),
    )

    applied = apply_s05_sa_screening_failure(
        conn,
        result_id="1b70ab74-c98",
        score=0.5658,
        erf_density=1.0,
        erf_decay_slope=0.019458947703242302,
        graph_category_histogram='{"elementwise_binary": 4}',
        source="test",
    )

    assert not applied
    assert conn.execute(
        "SELECT tier, validation_passed, notes FROM leaderboard"
    ).fetchone() == ("validation", 1, None)
    assert conn.execute("SELECT failure_op FROM program_results").fetchone()[0] is None
    override = controlled_lang_gate_manual_override(
        entry_id="c03cdadf-af7",
        result_id="1b70ab74-c98",
        failure_op=S05_SA_FAILURE_OP,
    )
    assert override is not None
    assert override["reason"] == "manual_pass_harder_nano_ar_good"


def test_manual_override_does_not_apply_to_other_gate_ops() -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO leaderboard(result_id, tier, validation_passed) VALUES (?, ?, ?)",
        ("1b70ab74-c98", "validation", 1),
    )
    conn.execute(
        "INSERT INTO program_results(result_id) VALUES (?)",
        ("1b70ab74-c98",),
    )

    applied = apply_controlled_lang_nb_screening_failure(
        conn,
        result_id="1b70ab74-c98",
        tier="inv",
        score=0.64,
        source="test",
    )

    assert applied
    assert conn.execute("SELECT tier FROM leaderboard").fetchone()[0] == "screened_out"
    assert conn.execute("SELECT failure_op FROM program_results").fetchone()[0] == (
        "controlled_lang_inv_nb"
    )


def test_apply_s10_nb_failure_screens_out_candidate() -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO leaderboard(result_id, tier, validation_passed) VALUES (?, ?, ?)",
        ("r-s10", "validation", 0),
    )
    conn.execute("INSERT INTO program_results(result_id) VALUES (?)", ("r-s10",))

    applied = apply_controlled_lang_nb_screening_failure(
        conn,
        result_id="r-s10",
        tier="s10",
        score=0.62,
        source="test",
    )

    assert applied
    lb = conn.execute(
        "SELECT tier, validation_passed, notes FROM leaderboard"
    ).fetchone()
    assert lb == (
        "screened_out",
        0,
        "controlled_lang_s10_nb: score 0.6200 below screening threshold 0.65",
    )
    pr = conn.execute(
        "SELECT failure_op, failure_details_json FROM program_results"
    ).fetchone()
    assert pr[0] == "controlled_lang_s10_nb"
    details = json.loads(pr[1])
    assert details["reason"] == "controlled_lang_s10_nb_below_threshold"
    assert details["stage"] == "s1.0"
    assert details["controlled_lang_s10_nb_score"] == 0.62


def test_apply_inv_nb_failure_screens_out_candidate() -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO leaderboard(result_id, tier, validation_passed) VALUES (?, ?, ?)",
        ("r-inv", "validation", 1),
    )
    conn.execute("INSERT INTO program_results(result_id) VALUES (?)", ("r-inv",))

    applied = apply_controlled_lang_nb_screening_failure(
        conn,
        result_id="r-inv",
        tier="inv",
        score=0.64,
        source="test",
    )

    assert applied
    lb = conn.execute(
        "SELECT tier, validation_passed, notes FROM leaderboard"
    ).fetchone()
    assert lb == (
        "screened_out",
        0,
        "controlled_lang_inv_nb: score 0.6400 below screening threshold 0.65",
    )
    pr = conn.execute(
        "SELECT failure_op, failure_details_json FROM program_results"
    ).fetchone()
    assert pr[0] == "controlled_lang_inv_nb"
    details = json.loads(pr[1])
    assert details["reason"] == "controlled_lang_inv_nb_below_threshold"
    assert details["stage"] == "investigation"
    assert details["controlled_lang_inv_nb_score"] == 0.64


def test_apply_inv_nb_pass_does_not_screen_out_candidate() -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO leaderboard(result_id, tier, validation_passed) VALUES (?, ?, ?)",
        ("r-inv-pass", "validation", 1),
    )
    conn.execute(
        "INSERT INTO program_results(result_id) VALUES (?)",
        ("r-inv-pass",),
    )

    applied = apply_controlled_lang_nb_screening_failure(
        conn,
        result_id="r-inv-pass",
        tier="inv",
        score=0.65,
        source="test",
    )

    assert not applied
    assert conn.execute(
        "SELECT tier, validation_passed, notes FROM leaderboard"
    ).fetchone() == ("validation", 1, None)
    assert conn.execute("SELECT failure_op FROM program_results").fetchone()[0] is None


def test_apply_s10_nb_sa_failure_screens_out_candidate() -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO leaderboard(result_id, tier, validation_passed) VALUES (?, ?, ?)",
        ("r-s10-combo", "validation", 0),
    )
    conn.execute("INSERT INTO program_results(result_id) VALUES (?)", ("r-s10-combo",))

    applied = apply_s10_nb_sa_screening_failure(
        conn,
        result_id="r-s10-combo",
        nb_score=0.75,
        sa_score=0.25,
        source="test",
    )

    assert applied
    lb = conn.execute(
        "SELECT tier, validation_passed, notes FROM leaderboard"
    ).fetchone()
    assert lb == (
        "screened_out",
        0,
        "controlled_lang_s10_nb_sa: nb 0.7500 below 0.80 and sa 0.2500 below 0.65",
    )
    pr = conn.execute(
        "SELECT failure_op, failure_details_json FROM program_results"
    ).fetchone()
    assert pr[0] == S10_NB_SA_FAILURE_OP
    details = json.loads(pr[1])
    assert details["reason"] == "controlled_lang_s10_nb_sa_below_threshold"
    assert details["controlled_lang_s10_nb_score"] == 0.75
    assert details["controlled_lang_s10_sa_score"] == 0.25


def test_apply_tool_classifies_known_candidate_as_manual_override() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE leaderboard (
            entry_id TEXT PRIMARY KEY,
            result_id TEXT,
            tier TEXT,
            composite_score REAL,
            validation_passed INTEGER DEFAULT 0,
            is_reference INTEGER DEFAULT 0,
            notes TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE program_results (
            result_id TEXT PRIMARY KEY,
            graph_fingerprint TEXT,
            controlled_lang_s05_sa_score REAL,
            controlled_lang_s05_nb_score REAL,
            controlled_lang_s10_sa_score REAL,
            controlled_lang_s10_nb_score REAL,
            controlled_lang_inv_nb_score REAL,
            fp_jacobian_erf_density REAL,
            fp_jacobian_erf_decay_slope REAL,
            graph_category_histogram TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO leaderboard(
            entry_id, result_id, tier, composite_score, validation_passed
        )
        VALUES ('c03cdadf-af7', '1b70ab74-c98', 'validation', 413.6, 0)
        """
    )
    conn.execute(
        """
        INSERT INTO program_results(
            result_id,
            controlled_lang_s05_sa_score,
            controlled_lang_s05_nb_score,
            fp_jacobian_erf_density,
            fp_jacobian_erf_decay_slope,
            graph_category_histogram
        )
        VALUES (
            '1b70ab74-c98',
            0.5658,
            0.7281,
            1.0,
            0.019458947703242302,
            '{"elementwise_binary": 4}'
        )
        """
    )

    rows = _candidate_rows(conn)

    assert len(rows) == 1
    override = _manual_override_for_row(rows[0])
    assert override is not None
    assert override["reason"] == "manual_pass_harder_nano_ar_good"
