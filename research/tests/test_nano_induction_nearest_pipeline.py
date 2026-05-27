from __future__ import annotations

import json
import sqlite3

import pytest
from torch import nn

from component_fab.harness.nano_induction_probe import nano_induction_nearest
from research.tools.backfill_nano_induction_nearest import (
    apply_backfill,
    ensure_nano_induction_nearest_columns,
    updates_from_record,
)

pytestmark = pytest.mark.unit


def test_nano_induction_nearest_output_shape_and_status():
    body = nn.Sequential(nn.LayerNorm(16), nn.Linear(16, 16))

    result = nano_induction_nearest(
        body,
        dim=16,
        seq_len=12,
        n_keys=4,
        n_values=4,
        n_train_steps=2,
        checkpoint_at_steps=(1, 2),
        batch_size=2,
        eval_batch=4,
        seed=123,
    )

    payload = result.to_dict()
    assert payload["status"] == "ok"
    assert len(payload["accuracies"]) == 2
    assert 0.0 <= result.max_accuracy <= 1.0
    assert 0.0 <= result.final_accuracy <= 1.0
    assert payload["train_steps"] == 2


def test_backfill_maps_jsonl_fields_to_db_columns(tmp_path):
    db = tmp_path / "runs.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE program_results (result_id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE graph_runs (result_id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO graph_runs(result_id) VALUES ('rid-ok')")
    conn.commit()

    added = ensure_nano_induction_nearest_columns(conn)
    assert "graph_runs.nano_induction_nearest_max_accuracy" in added

    report = tmp_path / "nearest.jsonl"
    report.write_text(
        json.dumps(
            {
                "result_id": "rid-ok",
                "nearest_status": "ok",
                "nearest_max_accuracy": 0.625,
                "nearest_final_accuracy": 0.5,
                "nearest_elapsed_s": 0.25,
                "nearest_accuracies": [0.25, 0.625],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = apply_backfill(conn, report, dry_run=False)
    row = conn.execute("""
        SELECT nano_induction_nearest_max_accuracy,
               nano_induction_nearest_final_accuracy,
               nano_induction_nearest_status,
               nano_induction_nearest_elapsed_ms,
               nano_induction_nearest_accuracies_json
        FROM graph_runs WHERE result_id = 'rid-ok'
        """).fetchone()

    assert summary["updated_rows"] == 1
    assert row["nano_induction_nearest_max_accuracy"] == 0.625
    assert row["nano_induction_nearest_final_accuracy"] == 0.5
    assert row["nano_induction_nearest_status"] == "ok"
    assert row["nano_induction_nearest_elapsed_ms"] == 250.0
    assert json.loads(row["nano_induction_nearest_accuracies_json"]) == [0.25, 0.625]


def test_failed_probe_rows_do_not_create_metric_values():
    updates = updates_from_record(
        {
            "result_id": "rid-fail",
            "nearest_status": "error",
            "nearest_error": "RuntimeError: shape mismatch",
            "nearest_max_accuracy": 1.0,
            "nearest_final_accuracy": 1.0,
        }
    )

    assert updates["nano_induction_nearest_status"] == "error"
    assert updates["nano_induction_nearest_error"] == "RuntimeError: shape mismatch"
    assert "nano_induction_nearest_max_accuracy" not in updates
    assert "nano_induction_nearest_final_accuracy" not in updates


def test_gbm_feature_registry_keeps_ar_gate_and_nearest_induction():
    from research.scientist.intelligence import predictor_gbm

    features = set(predictor_gbm._POST_EVAL_FEATURE_NAMES)
    assert "ar_gate_score_best" in features
    assert "nano_induction_nearest_max_accuracy_best" in features
    assert "language_control_s05_binding_score_best" in features
    assert "language_control_s10_binding_score_best" in features
    assert not hasattr(predictor_gbm, "_PROBE_FEATURE_NAMES")
