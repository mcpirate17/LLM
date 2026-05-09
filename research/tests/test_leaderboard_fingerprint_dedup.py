from __future__ import annotations

import json

import pytest

from research.scientist.notebook import LabNotebook
from research.scientist.notebook.notebook_leaderboard import (
    DuplicateLeaderboardFingerprintError,
)


def _seed(nb: LabNotebook, fp: str, rid: str, *, reason: str | None = None) -> None:
    exp = nb.start_experiment("evolution", {"tag": rid}, f"seed for {rid}")
    kwargs = dict(
        experiment_id=exp,
        graph_fingerprint=fp,
        graph_json=json.dumps({"nodes": [], "id": rid}),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.5,
        result_id=rid,
        trust_label="test_fixture",
    )
    if reason is not None:
        kwargs["intentional_rerun_reason"] = reason
    nb.record_program_result(**kwargs)
    nb.flush_writes()


def test_upsert_blocks_duplicate_fingerprint(tmp_path):
    db = tmp_path / "lab.db"
    nb = LabNotebook(db)
    _seed(nb, fp="sharedfp0001aaaa", rid="rid-A")
    _seed(nb, fp="sharedfp0001aaaa", rid="rid-B", reason="historical_dup")
    first = nb.upsert_leaderboard(
        result_id="rid-A",
        model_source="test",
        tier="screening",
    )
    assert first
    with pytest.raises(DuplicateLeaderboardFingerprintError) as exc_info:
        nb.upsert_leaderboard(
            result_id="rid-B",
            model_source="test",
            tier="screening",
        )
    err = exc_info.value
    assert err.graph_fingerprint == "sharedfp0001aaaa"
    assert err.existing_entry_id == first
    assert err.attempted_result_id == "rid-B"
    nb.close()


def test_allow_flag_bypasses_python_gate_but_schema_still_blocks(tmp_path):
    """Bypass flag skips the Python-level check, but the schema-level
    `idx_leaderboard_fp` UNIQUE index is the backstop and still blocks.
    Defense-in-depth: both layers must be satisfied to create a dup.
    """
    import sqlite3 as _sqlite3

    db = tmp_path / "lab.db"
    nb = LabNotebook(db)
    _seed(nb, fp="sharedfp0002bbbb", rid="rid-A")
    _seed(nb, fp="sharedfp0002bbbb", rid="rid-B", reason="historical_dup")
    nb.upsert_leaderboard(
        result_id="rid-A",
        model_source="test",
        tier="screening",
    )
    # Bypass flag silences the Python-layer DuplicateLeaderboardFingerprintError.
    # But the schema-level UNIQUE index catches it with an IntegrityError
    # / OperationalError (depending on the sqlite path). Both outcomes are
    # acceptable — the point is that a duplicate cannot land in the DB.
    raised = (
        _sqlite3.IntegrityError,
        _sqlite3.OperationalError,
        RuntimeError,
    )
    with pytest.raises(raised):
        nb.upsert_leaderboard(
            result_id="rid-B",
            model_source="test",
            tier="screening",
            allow_fingerprint_duplicate=True,
        )
    rows = nb.conn.execute(
        "SELECT COUNT(*) c FROM leaderboard l JOIN program_results pr "
        "ON l.result_id = pr.result_id WHERE pr.graph_fingerprint = ?",
        ("sharedfp0002bbbb",),
    ).fetchone()
    # Schema kept the table clean
    assert rows["c"] == 1
    nb.close()


def test_upsert_allows_update_of_existing_entry(tmp_path):
    db = tmp_path / "lab.db"
    nb = LabNotebook(db)
    _seed(nb, fp="sharedfp0003cccc", rid="rid-A")
    first = nb.upsert_leaderboard(
        result_id="rid-A",
        model_source="test",
        tier="screening",
    )
    # Re-upsert same result_id — gate must NOT fire; updates are fine
    second = nb.upsert_leaderboard(
        result_id="rid-A",
        model_source="test",
        tier="screening",
        notes="updated",
    )
    assert first == second
    nb.close()


def test_lookup_by_fingerprint_helper(tmp_path):
    db = tmp_path / "lab.db"
    nb = LabNotebook(db)
    _seed(nb, fp="sharedfp0004dddd", rid="rid-A")
    nb.upsert_leaderboard(
        result_id="rid-A",
        model_source="test",
        tier="screening",
    )
    row = nb.get_leaderboard_entry_by_fingerprint("sharedfp0004dddd")
    assert row is not None
    assert row["result_id"] == "rid-A"
    assert nb.get_leaderboard_entry_by_fingerprint("doesnotexist0000") is None
    assert nb.get_leaderboard_entry_by_fingerprint("") is None
    nb.close()


def test_reference_upsert_reuses_existing_fingerprint_parent(tmp_path):
    db = tmp_path / "lab.db"
    nb = LabNotebook(db)
    _seed(nb, fp="sharedfp0005eeee", rid="rid-A")
    _seed(nb, fp="sharedfp0005eeee", rid="rid-B", reason="historical_dup")
    first = nb.upsert_leaderboard(
        result_id="rid-A",
        model_source="test",
        tier="screening",
    )
    nb.conn.execute(
        "UPDATE leaderboard SET composite_score = ? WHERE entry_id = ?",
        (500.0, first),
    )
    second = nb.upsert_leaderboard(
        result_id="rid-B",
        model_source="reference_calibration",
        tier="screening",
        is_reference=True,
        reference_name="gpt2_test",
    )
    assert second == first
    row = nb.conn.execute(
        "SELECT result_id, is_reference, reference_name, composite_score FROM leaderboard "
        "WHERE entry_id = ?",
        (first,),
    ).fetchone()
    assert row["result_id"] == "rid-A"
    assert int(row["is_reference"]) == 1
    assert row["reference_name"] == "gpt2_test"
    assert row["composite_score"] == pytest.approx(500.0)
    nb.close()
