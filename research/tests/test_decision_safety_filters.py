from __future__ import annotations

import pytest


from research.scientist.notebook import LabNotebook

pytestmark = pytest.mark.unit


def test_get_top_programs_returns_stage1_survivors(tmp_path):
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
            trust_label="test_fixture",
        )
        nb.complete_experiment(
            exp_good,
            {"total": 1, "stage0_passed": 1, "stage05_passed": 1, "stage1_passed": 1},
        )
        nb.flush_writes()

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
            trust_label="test_fixture",
        )
        nb.complete_experiment(
            exp_bad,
            {"total": 1, "stage0_passed": 1, "stage05_passed": 1, "stage1_passed": 1},
        )
        nb.flush_writes()
        nb.conn.execute(
            "UPDATE experiments SET status = 'invalid' WHERE experiment_id = ?",
            (exp_bad,),
        )
        nb.conn.commit()

        top = nb.get_top_programs(20, sort_by="loss_ratio")
        ids = {row.get("result_id") for row in top}
        assert good_result in ids
        assert bad_result in ids
    finally:
        nb.close()
