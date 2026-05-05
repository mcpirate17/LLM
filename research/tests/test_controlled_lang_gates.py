from __future__ import annotations

import json
import sqlite3

from research.scientist.controlled_lang_gates import (
    CONTROLLED_LANG_SCORE_GATES,
    S05_NB_FAILURE_OP,
    apply_controlled_lang_nb_screening_failure,
    apply_controlled_lang_screening_failure,
    allows_controlled_lang_advanced_tiers,
    apply_s05_nb_screening_failure,
    is_controlled_lang_nb_screening_failure,
    is_controlled_lang_screening_failure,
    is_s05_nb_screening_failure,
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
    assert is_controlled_lang_screening_failure(0.6499)
    assert not is_controlled_lang_screening_failure(0.65)


def test_s05_nb_gate_blocks_advanced_tiers_until_passed() -> None:
    assert not allows_controlled_lang_advanced_tiers(None)
    assert not allows_controlled_lang_advanced_tiers(0.6499)
    assert allows_controlled_lang_advanced_tiers(0.65)


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


def test_apply_s10_sa_failure_screens_out_candidate() -> None:
    conn = _conn()
    conn.execute(
        "INSERT INTO leaderboard(result_id, tier, validation_passed) VALUES (?, ?, ?)",
        ("r-s10-sa", "validation", 0),
    )
    conn.execute("INSERT INTO program_results(result_id) VALUES (?)", ("r-s10-sa",))
    gate = next(
        gate
        for gate in CONTROLLED_LANG_SCORE_GATES
        if gate["failure_op"] == "controlled_lang_s10_sa"
    )

    applied = apply_controlled_lang_screening_failure(
        conn,
        result_id="r-s10-sa",
        gate=gate,
        score=0.24,
        source="test",
    )

    assert applied
    lb = conn.execute(
        "SELECT tier, validation_passed, notes FROM leaderboard"
    ).fetchone()
    assert lb == (
        "screened_out",
        0,
        "controlled_lang_s10_sa: score 0.2400 below screening threshold 0.65",
    )
    pr = conn.execute(
        "SELECT failure_op, failure_details_json FROM program_results"
    ).fetchone()
    assert pr[0] == "controlled_lang_s10_sa"
    details = json.loads(pr[1])
    assert details["reason"] == "controlled_lang_s10_sa_below_threshold"
    assert details["stage"] == "s1.0"
    assert details["controlled_lang_s10_sa_score"] == 0.24
