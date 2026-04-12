from research.tools._script_audit import (
    build_metric_backfill_context,
    complete_script_experiment,
    fail_script_experiment,
    start_script_experiment,
)


def test_complete_script_experiment_supports_zero_total(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb, exp_id = start_script_experiment(
        db_path=db_path,
        experiment_type="predictor_training",
        config={"component": "graph"},
        source_script="train_predictors",
        hypothesis="test run",
    )
    try:
        complete_script_experiment(
            nb,
            exp_id,
            results={"components_trained": 2, "elapsed_s": 1.5},
            summary="predictors complete",
        )
        row = nb.conn.execute(
            "SELECT status, aria_summary, n_programs_generated FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        assert row["status"] == "completed"
        assert row["aria_summary"] == "predictors complete"
        assert row["n_programs_generated"] == 2
    finally:
        nb.close()


def test_fail_script_experiment_preserves_partial_counts(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb, exp_id = start_script_experiment(
        db_path=db_path,
        experiment_type="probe_backfill",
        config={"probe": "hellaswag"},
        source_script="backfill",
        hypothesis="test run",
    )
    try:
        fail_script_experiment(
            nb,
            exp_id,
            error="KeyboardInterrupt",
            results={"evaluated": 14, "failed": 2},
        )
        row = nb.conn.execute(
            "SELECT status, aria_summary, n_programs_generated FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        assert row["status"] == "failed"
        assert "KeyboardInterrupt" in row["aria_summary"]
        assert row["n_programs_generated"] == 14
    finally:
        nb.close()


def test_build_metric_backfill_context_serializes_paths(tmp_path):
    ctx = build_metric_backfill_context(
        kind="probe_backfill",
        source_script="backfill",
        experiment_id="exp123",
        device="cuda",
        report_path=tmp_path / "report.json",
    )
    assert ctx["kind"] == "probe_backfill"
    assert ctx["experiment_id"] == "exp123"
    assert ctx["report_path"].endswith("report.json")
