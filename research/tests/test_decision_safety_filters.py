from __future__ import annotations

import time

from research.scientist.notebook import LabNotebook
from research.scientist.runner import ExperimentRunner


def test_get_recent_experiments_excludes_invalid_by_default(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(db_path)
    try:
        now = time.time()
        nb.conn.execute(
            "INSERT INTO experiments (experiment_id, timestamp, experiment_type, status, config_json) VALUES (?, ?, 'synthesis', 'completed', '{}')",
            ("exp_completed", now - 3),
        )
        nb.conn.execute(
            "INSERT INTO experiments (experiment_id, timestamp, experiment_type, status, config_json) VALUES (?, ?, 'synthesis', 'invalid', '{}')",
            ("exp_invalid", now - 2),
        )
        nb.conn.execute(
            "INSERT INTO experiments (experiment_id, timestamp, experiment_type, status, config_json) VALUES (?, ?, 'synthesis', 'failed', '{}')",
            ("exp_failed", now - 1),
        )
        nb.conn.commit()

        recent_default = nb.get_recent_experiments(10)
        recent_all = nb.get_recent_experiments(10, include_invalid=True)

        assert all(r.get("status") != "invalid" for r in recent_default)
        assert any(r.get("status") == "invalid" for r in recent_all)
    finally:
        nb.close()


def test_filter_decision_candidates_drops_invalid_and_non_s1(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(db_path)
    try:
        now = time.time()
        nb.conn.execute(
            "INSERT INTO experiments (experiment_id, timestamp, experiment_type, status, config_json) VALUES (?, ?, 'synthesis', 'invalid', '{}')",
            ("exp_invalid", now - 2),
        )
        nb.conn.execute(
            "INSERT INTO experiments (experiment_id, timestamp, experiment_type, status, config_json) VALUES (?, ?, 'synthesis', 'completed', '{}')",
            ("exp_valid", now - 1),
        )
        nb.conn.commit()

        candidates = [
            {"result_id": "r_invalid", "experiment_id": "exp_invalid", "stage1_passed": 1},
            {"result_id": "r_not_s1", "experiment_id": "exp_valid", "stage1_passed": 0},
            {"result_id": "r_keep", "experiment_id": "exp_valid", "stage1_passed": 1},
        ]
        filtered = ExperimentRunner._filter_decision_candidates(nb, candidates)
        ids = [r.get("result_id") for r in filtered]
        assert ids == ["r_keep"]
    finally:
        nb.close()


def test_get_top_programs_excludes_invalid_experiment_rows(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(db_path)
    try:
        exp_good = nb.start_experiment("synthesis", {"n_programs": 1}, "good")
        good_result = nb.record_program_result(
            experiment_id=exp_good,
            graph_fingerprint="fp_good",
            graph_json='{"nodes": {}}',
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.4,
            novelty_score=0.6,
        )
        nb.complete_experiment(exp_good, {"total": 1, "stage0_passed": 1, "stage05_passed": 1, "stage1_passed": 1})

        exp_bad = nb.start_experiment("synthesis", {"n_programs": 1}, "bad")
        bad_result = nb.record_program_result(
            experiment_id=exp_bad,
            graph_fingerprint="fp_bad",
            graph_json='{"nodes": {}}',
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.2,
            novelty_score=0.9,
        )
        nb.complete_experiment(exp_bad, {"total": 1, "stage0_passed": 1, "stage05_passed": 1, "stage1_passed": 1})
        nb.conn.execute("UPDATE experiments SET status = 'invalid' WHERE experiment_id = ?", (exp_bad,))
        nb.conn.commit()

        top = nb.get_top_programs(20, sort_by="loss_ratio")
        ids = {row.get("result_id") for row in top}
        assert good_result in ids
        assert bad_result not in ids
    finally:
        nb.close()


def test_rebuild_result_lineage_index_populates_normalized_rows(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(db_path)
    try:
        exp_id = nb.start_experiment("synthesis", {"n_programs": 2}, "lineage")
        parent_id = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_parent",
            graph_json='{"nodes": {"0": {"id": 0, "op_name": "input", "input_ids": []}}}',
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.5,
            novelty_score=0.3,
        )
        child_id = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_child",
            graph_json=(
                '{"nodes": {}, "metadata": {"refinement": {"source_result_id": "' + parent_id + '"}, '
                '"lineage": {"type": "mutation", "parent": "fp_parent"}}}'
            ),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.4,
            novelty_score=0.5,
        )
        nb.flush_writes()
        nb.conn.execute("DELETE FROM result_lineage WHERE result_id = ?", (child_id,))
        nb.conn.commit()

        rebuilt = nb.rebuild_result_lineage_index()
        row = nb.get_result_lineage(child_id)

        assert rebuilt >= 1
        assert row is not None
        assert row.get("parent_result_id") == parent_id
        assert row.get("parent_fingerprint") == "fp_parent"
    finally:
        nb.close()
