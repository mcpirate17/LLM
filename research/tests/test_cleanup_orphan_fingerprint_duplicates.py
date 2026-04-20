from __future__ import annotations

import sqlite3

from research.scientist.notebook import LabNotebook
from research.tools import cleanup_orphan_fingerprint_duplicates as cleanup


def test_cleanup_merges_orphan_duplicates_and_relabels_backfill(tmp_path):
    db_path = tmp_path / "cleanup.db"
    nb = LabNotebook(db_path)
    exp_old = nb.start_experiment("evolution", {"tag": "old"})
    exp_new = nb.start_experiment("novelty", {"tag": "new"})

    keeper_id = nb.record_program_result(
        experiment_id=exp_new,
        result_id="dup-new",
        graph_fingerprint="dup-fp-001",
        graph_json="{}",
        bypass_quality_gate=True,
        intentional_rerun_reason="test_fixture_orphan_dup",
        stage0_passed=True,
        stage1_passed=False,
        timestamp=200.0,
        hellaswag_acc=0.30,
        trust_label="runtime_observation",
        result_cohort="search",
    )
    older_id = nb.record_program_result(
        experiment_id=exp_old,
        result_id="dup-old",
        graph_fingerprint="dup-fp-001",
        graph_json="{}",
        bypass_quality_gate=True,
        intentional_rerun_reason="test_fixture_orphan_dup",
        stage0_passed=True,
        stage1_passed=False,
        timestamp=100.0,
        induction_auc=0.04,
        binding_auc=0.08,
        hellaswag_acc=0.26,
        trust_label="runtime_observation",
        result_cohort="search",
    )
    nb.flush_writes()
    nb.conn.execute(
        """
        INSERT INTO training_curves(result_id, step, loss, grad_norm, step_time_ms)
        VALUES (?, ?, ?, ?, ?)
        """,
        (older_id, 10, 1.23, 0.4, 5.0),
    )
    nb.conn.commit()
    nb.close()

    exit_code = cleanup.run(
        db_path,
        apply=True,
        fingerprint=None,
        limit_groups=None,
    )
    assert exit_code == 0

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    kept = conn.execute(
        """
        SELECT result_id, induction_auc, binding_auc, hellaswag_acc,
               result_cohort, trust_label, comparability_label,
               evaluation_protocol_version, init_regime
        FROM program_results
        WHERE graph_fingerprint = 'dup-fp-001'
        """
    ).fetchall()
    assert len(kept) == 1
    row = kept[0]
    assert row["induction_auc"] == 0.04
    assert row["binding_auc"] == 0.08
    assert row["hellaswag_acc"] == 0.30
    assert row["result_cohort"] == "backfill"
    assert row["trust_label"] == "backfill_observation"
    assert row["comparability_label"] == "reconstructed_init_variant"
    assert row["evaluation_protocol_version"] == "backfill_replay_v1"
    assert row["init_regime"] == "reconstructed_fresh_init"
    survivor_id = row["result_id"]

    training_curve = conn.execute(
        "SELECT result_id, step FROM training_curves WHERE result_id = ?",
        (survivor_id,),
    ).fetchone()
    assert training_curve is not None
    deleted_id = older_id if survivor_id != older_id else keeper_id
    deleted = conn.execute(
        "SELECT result_id FROM program_results WHERE result_id = ?",
        (deleted_id,),
    ).fetchone()
    assert deleted is None

    backup_rows = conn.execute(
        f"SELECT backup_kind FROM {cleanup.BACKUP_TABLE} WHERE canonical_result_id = ?",
        (survivor_id,),
    ).fetchall()
    assert {row["backup_kind"] for row in backup_rows} == {
        "keeper_premerge",
        "deleted_duplicate",
    }
    conn.close()


def test_cleanup_skips_groups_with_leaderboard_rows(tmp_path):
    db_path = tmp_path / "cleanup_leaderboard.db"
    nb = LabNotebook(db_path)
    exp_a = nb.start_experiment("evolution", {})
    exp_b = nb.start_experiment("novelty", {})
    rid_a = nb.record_program_result(
        experiment_id=exp_a,
        result_id="lb-a",
        graph_fingerprint="dup-fp-lb",
        graph_json="{}",
        intentional_rerun_reason="test_fixture_lb_dup",
        stage0_passed=True,
        stage1_passed=True,
        loss_ratio=0.8,
        timestamp=100.0,
    )
    nb.record_program_result(
        experiment_id=exp_b,
        result_id="lb-b",
        graph_fingerprint="dup-fp-lb",
        graph_json="{}",
        intentional_rerun_reason="test_fixture_lb_dup",
        stage0_passed=True,
        stage1_passed=True,
        loss_ratio=0.7,
        timestamp=200.0,
        induction_auc=0.05,
    )
    nb.flush_writes()
    nb.upsert_leaderboard(
        result_id=rid_a,
        model_source="graph_synthesis",
        screening_loss_ratio=0.8,
        screening_novelty=0.3,
        tier="screening",
    )
    nb.close()

    exit_code = cleanup.run(
        db_path,
        apply=True,
        fingerprint=None,
        limit_groups=None,
        mode="orphan",
    )
    assert exit_code == 0

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM program_results WHERE graph_fingerprint = 'dup-fp-lb'"
    ).fetchone()
    assert count["n"] == 2
    conn.close()


def test_single_lb_cleanup_merges_into_leaderboard_row(tmp_path):
    db_path = tmp_path / "cleanup_single_lb.db"
    nb = LabNotebook(db_path)
    exp_screen = nb.start_experiment("synthesis", {})
    exp_val = nb.start_experiment("validation", {})
    lb_id = nb.record_program_result(
        experiment_id=exp_screen,
        result_id="single-lb-keeper",
        graph_fingerprint="dup-fp-single-lb",
        graph_json="{}",
        intentional_rerun_reason="test_fixture_single_lb_dup",
        stage0_passed=True,
        stage1_passed=True,
        loss_ratio=0.8,
        timestamp=100.0,
    )
    dup_id = nb.record_program_result(
        experiment_id=exp_val,
        result_id="single-lb-dup",
        graph_fingerprint="dup-fp-single-lb",
        graph_json="{}",
        intentional_rerun_reason="test_fixture_single_lb_dup",
        stage0_passed=True,
        stage1_passed=True,
        loss_ratio=0.7,
        validation_loss_ratio=0.45,
        induction_auc=0.05,
        timestamp=200.0,
    )
    nb.flush_writes()
    nb.upsert_leaderboard(
        result_id=lb_id,
        model_source="graph_synthesis",
        screening_loss_ratio=0.8,
        screening_novelty=0.3,
        tier="screening",
    )
    nb.conn.execute(
        """
        INSERT INTO training_curves(result_id, step, loss, grad_norm, step_time_ms)
        VALUES (?, ?, ?, ?, ?)
        """,
        (dup_id, 10, 1.11, 0.4, 5.0),
    )
    nb.conn.commit()
    nb.close()

    exit_code = cleanup.run(
        db_path,
        apply=True,
        fingerprint=None,
        limit_groups=None,
        mode="single-lb",
    )
    assert exit_code == 0

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT result_id, loss_ratio, validation_loss_ratio, induction_auc
        FROM program_results
        WHERE graph_fingerprint = 'dup-fp-single-lb'
        """
    ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["result_id"] == lb_id
    assert row["loss_ratio"] == 0.7
    assert row["validation_loss_ratio"] == 0.45
    assert row["induction_auc"] == 0.05
    curve = conn.execute(
        "SELECT COUNT(*) AS n FROM training_curves WHERE result_id = ?",
        (lb_id,),
    ).fetchone()
    assert curve["n"] == 1
    deleted = conn.execute(
        "SELECT result_id FROM program_results WHERE result_id = ?",
        (dup_id,),
    ).fetchone()
    assert deleted is None
    backup_rows = conn.execute(
        f"SELECT backup_kind FROM {cleanup.BACKUP_TABLE} WHERE canonical_result_id = ?",
        (lb_id,),
    ).fetchall()
    assert {row["backup_kind"] for row in backup_rows} == {
        "keeper_premerge",
        "deleted_duplicate",
    }
    conn.close()


def test_single_lb_cleanup_handles_missing_experiment_row(tmp_path):
    db_path = tmp_path / "cleanup_single_lb_missing_exp.db"
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("synthesis", {})
    keeper_id = nb.record_program_result(
        experiment_id=exp_id,
        result_id="single-lb-missing-exp-keeper",
        graph_fingerprint="dup-fp-missing-exp",
        graph_json="{}",
        intentional_rerun_reason="test_fixture_single_lb_dup",
        stage0_passed=True,
        stage1_passed=True,
        loss_ratio=0.5,
    )
    missing_id = nb.record_program_result(
        experiment_id="missing-exp-id",
        result_id="single-lb-missing-exp-dup",
        graph_fingerprint="dup-fp-missing-exp",
        graph_json="{}",
        intentional_rerun_reason="test_fixture_single_lb_dup",
        stage0_passed=True,
        stage1_passed=True,
        loss_ratio=0.4,
        induction_auc=0.07,
    )
    nb.flush_writes()
    nb.upsert_leaderboard(
        result_id=keeper_id,
        model_source="graph_synthesis",
        screening_loss_ratio=0.5,
        screening_novelty=0.2,
        tier="screening",
    )
    nb.close()

    exit_code = cleanup.run(
        db_path,
        apply=True,
        fingerprint=None,
        limit_groups=None,
        mode="single-lb",
    )
    assert exit_code == 0

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT result_id, loss_ratio, induction_auc FROM program_results WHERE graph_fingerprint = 'dup-fp-missing-exp'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["result_id"] == keeper_id
    assert rows[0]["loss_ratio"] == 0.4
    assert rows[0]["induction_auc"] == 0.07
    deleted = conn.execute(
        "SELECT result_id FROM program_results WHERE result_id = ?",
        (missing_id,),
    ).fetchone()
    assert deleted is None
    conn.close()


def test_cleanup_relabels_standalone_orphan_backfill_rows(tmp_path):
    db_path = tmp_path / "cleanup_orphan_relabel.db"
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment("evolution", {})
    rid = nb.record_program_result(
        experiment_id=exp_id,
        result_id="orphan-single",
        graph_fingerprint="single-fp-001",
        graph_json="{}",
        bypass_quality_gate=True,
        stage0_passed=True,
        stage1_passed=False,
        hellaswag_acc=0.29,
        trust_label="runtime_observation",
        result_cohort="search",
    )
    nb.flush_writes()
    nb.close()

    exit_code = cleanup.run(
        db_path,
        apply=True,
        fingerprint=None,
        limit_groups=None,
        mode="orphan",
    )
    assert exit_code == 0

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT result_cohort, trust_label, comparability_label
        FROM program_results
        WHERE result_id = ?
        """,
        (rid,),
    ).fetchone()
    assert row["result_cohort"] == "backfill"
    assert row["trust_label"] == "backfill_observation"
    assert row["comparability_label"] == "reconstructed_init_variant"
    backup = conn.execute(
        f"""
        SELECT COUNT(*) AS n
        FROM {cleanup.BACKUP_TABLE}
        WHERE result_id = ? AND backup_kind = 'orphan_backfill_relabel_preupdate'
        """,
        (rid,),
    ).fetchone()
    assert backup["n"] == 1
    conn.close()
