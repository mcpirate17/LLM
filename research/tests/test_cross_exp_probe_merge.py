from __future__ import annotations

import json
import sqlite3

from research.scientist.notebook import LabNotebook
from research.tools import cross_exp_probe_merge as cxpm


def _seed_two_experiments(nb: LabNotebook) -> dict[str, str]:
    """One fingerprint, two experiments, three program_results rows.

    Canonical = latest timestamp + stage1_passed + best loss.
    Probe values live on the oldest sibling — the merge should promote them.
    """
    exp_invest = nb.start_experiment(
        "investigation", {"tag": "latest"}, "latest investigation"
    )
    exp_synth = nb.start_experiment(
        "synthesis", {"tag": "probe-carrier"}, "probe carrier"
    )
    # Canonical candidate — latest + best loss + stage1
    nb.record_program_result(
        experiment_id=exp_invest,
        graph_fingerprint="sharedfp00000001",
        graph_json=json.dumps({"nodes": [], "v": 1}),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.50,
        result_id="canon-rid",
        timestamp=2000.0,
        trust_label="test_fixture",
        hellaswag_acc=0.32,
    )
    # Different experiment, worse loss — stays non-canonical by loss ordering.
    # The cross-experiment dedup gate requires intentional_rerun_reason; these
    # rows represent pre-gate historical data we're cleaning up.
    exp_mid = nb.start_experiment(
        "evolution", {"tag": "mid-loss"}, "mid-loss contender"
    )
    nb.record_program_result(
        experiment_id=exp_mid,
        graph_fingerprint="sharedfp00000001",
        graph_json=json.dumps({"nodes": [], "v": 1}),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.78,
        result_id="mid-rid",
        timestamp=1500.0,
        trust_label="test_fixture",
        intentional_rerun_reason="test_fixture_historical_dup",
    )
    # Oldest but has the probe measurements
    nb.record_program_result(
        experiment_id=exp_synth,
        graph_fingerprint="sharedfp00000001",
        graph_json=json.dumps({"nodes": [], "v": 1}),
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=True,
        loss_ratio=0.62,
        result_id="probe-rid",
        timestamp=1000.0,
        trust_label="test_fixture",
        induction_auc=0.04,
        binding_auc=0.08,
        blimp_overall_accuracy=0.55,
        intentional_rerun_reason="test_fixture_historical_dup",
    )
    # Cross-exp dup that is INTENTIONAL (exact_graph_replay) — should be excluded
    exp_replay = nb.start_experiment("exact_graph_replay", {"tag": "replay"}, "replay")
    nb.record_program_result(
        experiment_id=exp_replay,
        graph_fingerprint="sharedfp00000001",
        graph_json=json.dumps({"nodes": [], "v": 1}),
        stage0_passed=True,
        stage1_passed=True,
        loss_ratio=0.55,
        result_id="replay-rid",
        timestamp=2500.0,
        trust_label="test_fixture",
        intentional_rerun_reason="exact_graph_replay",
    )
    nb.flush_writes()
    return {"canon": "canon-rid", "mid": "mid-rid", "probe": "probe-rid"}


def test_dry_run_identifies_merge_plan(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(db_path)
    _seed_two_experiments(nb)
    exit_code = cxpm.run(
        db_path,
        apply=False,
        fingerprint=None,
        families=None,
        limit_groups=None,
    )
    assert exit_code == 0
    # Canonical row remains untouched on dry-run
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT induction_auc, binding_auc, blimp_overall_accuracy "
        "FROM program_results WHERE result_id = 'canon-rid'"
    ).fetchone()
    assert row["induction_auc"] is None
    assert row["binding_auc"] is None
    assert row["blimp_overall_accuracy"] is None
    conn.close()


def test_apply_merges_probes_onto_canonical(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(db_path)
    _seed_two_experiments(nb)
    nb.close()
    exit_code = cxpm.run(
        db_path,
        apply=True,
        fingerprint=None,
        families=None,
        limit_groups=None,
    )
    assert exit_code == 0
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    canon = conn.execute(
        "SELECT induction_auc, binding_auc, blimp_overall_accuracy, loss_ratio "
        "FROM program_results WHERE result_id = 'canon-rid'"
    ).fetchone()
    assert canon["induction_auc"] == 0.04
    assert canon["binding_auc"] == 0.08
    assert canon["blimp_overall_accuracy"] == 0.55
    # loss_ratio is NOT a merge column → stays at canonical's own value
    assert canon["loss_ratio"] == 0.50
    # Sibling rows remain intact (no deletion)
    probe = conn.execute(
        "SELECT induction_auc FROM program_results WHERE result_id = 'probe-rid'"
    ).fetchone()
    assert probe["induction_auc"] == 0.04
    mid = conn.execute(
        "SELECT result_id FROM program_results WHERE result_id = 'mid-rid'"
    ).fetchone()
    assert mid is not None
    # Backup row records the pre-merge canonical state
    backup = conn.execute(
        f"SELECT induction_auc, merged_columns, merged_from_result_ids "
        f"FROM {cxpm.BACKUP_TABLE} WHERE result_id = 'canon-rid'"
    ).fetchone()
    assert backup["induction_auc"] is None
    merged_cols = json.loads(backup["merged_columns"])
    assert "induction_auc" in merged_cols
    assert "blimp_overall_accuracy" in merged_cols
    contributors = json.loads(backup["merged_from_result_ids"])
    assert "probe-rid" in contributors
    conn.close()


def test_intentional_experiments_are_excluded(tmp_path):
    """exact_graph_replay should not pull rows into the merge pool."""
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(db_path)
    # Two rows in exact_graph_replay only — should NOT produce a merge plan
    exp_a = nb.start_experiment("exact_graph_replay", {}, "")
    exp_b = nb.start_experiment("exact_graph_replay", {}, "")
    for i, eid in enumerate((exp_a, exp_b)):
        nb.record_program_result(
            experiment_id=eid,
            graph_fingerprint="onlyreplayfp0001",
            graph_json=json.dumps({"n": i}),
            stage0_passed=True,
            stage1_passed=True,
            loss_ratio=0.5 + i * 0.01,
            result_id=f"replay-{i}",
            timestamp=1000.0 + i,
            induction_auc=0.1 if i == 0 else None,
            trust_label="test_fixture",
            intentional_rerun_reason="exact_graph_replay",
        )
    nb.flush_writes()
    nb.close()
    exit_code = cxpm.run(
        db_path,
        apply=True,
        fingerprint=None,
        families=None,
        limit_groups=None,
    )
    assert exit_code == 0
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    # Neither row mutated — intentional cross-exp is left alone
    row = conn.execute(
        "SELECT induction_auc FROM program_results WHERE result_id = 'replay-1'"
    ).fetchone()
    assert row["induction_auc"] is None
    conn.close()


def test_column_family_filter_restricts_merge(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(db_path)
    _seed_two_experiments(nb)
    nb.close()
    exit_code = cxpm.run(
        db_path,
        apply=True,
        fingerprint=None,
        families=["language"],
        limit_groups=None,
    )
    assert exit_code == 0
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    canon = conn.execute(
        "SELECT induction_auc, binding_auc, blimp_overall_accuracy "
        "FROM program_results WHERE result_id = 'canon-rid'"
    ).fetchone()
    # language family only → blimp merged, v1 probes skipped
    assert canon["blimp_overall_accuracy"] == 0.55
    assert canon["induction_auc"] is None
    assert canon["binding_auc"] is None
    conn.close()
