from __future__ import annotations

import json
import sqlite3

from research.scientist.analytics import ExperimentAnalytics
from research.scientist.notebook import LabNotebook
from research.scientist.notebook._shared import ExperimentEntry
from research.scientist.notebook.artifact_store import (
    NotebookArtifactStore,
    parse_artifact_pointer,
)
from research.scientist.notebook.graph_artifacts import resolve_graph_json_value
from research.tools.externalize_notebook_artifacts import run as externalize_artifacts
from research.tools.restore_inline_notebook_artifacts import (
    run as restore_inline_artifacts,
)


def test_record_program_result_externalizes_large_json_field(tmp_path, monkeypatch):
    monkeypatch.setenv("ARIA_NOTEBOOK_ARTIFACT_MIN_BYTES", "32")
    nb = LabNotebook(tmp_path / "lab_notebook.db", use_native=False)

    rid = nb.record_program_result(
        experiment_id="exp-artifact",
        graph_fingerprint="fp-artifact",
        graph_json='{"nodes":[]}',
        result_id="rid-artifact",
        stage0_passed=1,
        stage1_passed=0,
        loss_ratio=0.9,
        rapid_screening_metrics_json=json.dumps({"values": list(range(20))}),
        bypass_quality_gate=True,
    )
    nb.flush_writes()

    row = nb.conn.execute(
        "SELECT rapid_screening_metrics_json FROM program_results_compat WHERE result_id = ?",
        (rid,),
    ).fetchone()
    pointer = parse_artifact_pointer(row["rapid_screening_metrics_json"])
    assert pointer is not None
    detail = nb.get_program_detail(rid)
    assert detail["rapid_screening_metrics_json_parsed"]["values"] == list(range(20))


def test_training_curve_round_trips_from_artifact(tmp_path):
    nb = LabNotebook(tmp_path / "lab_notebook.db", use_native=False)
    rid = nb.record_program_result(
        experiment_id="exp-curve",
        graph_fingerprint="fp-curve",
        graph_json='{"nodes":[]}',
        result_id="rid-curve",
        stage0_passed=1,
        stage1_passed=1,
        loss_ratio=0.5,
        trust_label="test_fixture",
        bypass_quality_gate=True,
    )
    nb.flush_writes()

    curve = [
        {"step": 0, "loss": 2.0, "grad_norm": 1.0, "step_time_ms": 3.0},
        {"step": 1, "loss": 1.5, "grad_norm": 0.8, "step_time_ms": 2.5},
    ]
    nb.store_training_curve(rid, curve)

    assert nb.get_training_curve(rid) == curve
    assert (
        nb.conn.execute(
            "SELECT COUNT(*) FROM training_curves WHERE result_id = ?",
            (rid,),
        ).fetchone()[0]
        == 0
    )


def test_externalize_tool_moves_existing_payloads_and_curves(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(db_path, use_native=False)
    nb.record_program_result(
        experiment_id="exp-existing",
        graph_fingerprint="fp-existing",
        graph_json='{"nodes":[]}',
        result_id="rid",
        stage0_passed=1,
        stage1_passed=1,
        loss_ratio=0.7,
        trust_label="test_fixture",
        rapid_screening_metrics_json=json.dumps({"values": list(range(30))}),
        bypass_quality_gate=True,
    )
    nb.flush_writes()
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO training_curves VALUES ('rid', 0, 1.0, 2.0, 3.0)")
    conn.commit()
    conn.close()
    nb.close()

    report = externalize_artifacts(
        db_path=db_path,
        min_bytes=16,
        apply=True,
        limit=None,
        vacuum=False,
    )

    assert report["training_curves_applied"]["rows"] == 1
    nb = LabNotebook(db_path, use_native=False)
    row = nb.conn.execute(
        "SELECT rapid_screening_metrics_json FROM program_results_compat WHERE result_id = 'rid'"
    ).fetchone()
    assert parse_artifact_pointer(row[0]) is not None
    assert nb.get_training_curve("rid")[0]["loss"] == 1.0


def test_externalize_tool_can_move_graph_json_when_explicitly_enabled(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    graph_json = json.dumps(
        {
            "nodes": {
                "0": {"id": 0, "op_name": "input", "input_ids": []},
                "1": {"id": 1, "op_name": "softmax_attention", "input_ids": [0]},
            }
        },
        sort_keys=True,
    )
    nb = LabNotebook(db_path, use_native=False)
    rid = nb.record_program_result(
        experiment_id="exp-graph",
        graph_fingerprint="fp-graph",
        graph_json=graph_json,
        result_id="rid-graph",
        stage0_passed=1,
        stage1_passed=1,
        loss_ratio=0.7,
        trust_label="test_fixture",
        bypass_quality_gate=True,
    )
    nb.flush_writes()
    nb.close()

    report = externalize_artifacts(
        db_path=db_path,
        min_bytes=16,
        apply=True,
        limit=None,
        vacuum=False,
        include_graph_json=True,
        graph_json_cold_only=False,
    )

    applied = {
        (item["table"], item["column"]): item["rows"] for item in report["applied"]
    }
    assert applied[("graphs", "graph_json")] == 1
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT graph_json FROM program_results_compat WHERE result_id = ?",
        (rid,),
    ).fetchone()
    assert parse_artifact_pointer(row["graph_json"]) is not None
    assert resolve_graph_json_value(conn, db_path, row["graph_json"]) == graph_json
    conn.close()


def test_math_family_coverage_skips_missing_graph_artifact(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    graph_json = json.dumps(
        {
            "nodes": {
                "0": {"id": 0, "op_name": "input", "input_ids": []},
                "1": {"id": 1, "op_name": "poincare_add", "input_ids": [0]},
            }
        },
        sort_keys=True,
    )
    nb = LabNotebook(db_path, use_native=False)
    rid = nb.record_program_result(
        experiment_id="exp-missing-graph-artifact",
        graph_fingerprint="fp-missing-graph-artifact",
        graph_json=graph_json,
        result_id="rid-missing-graph-artifact",
        stage0_passed=1,
        stage1_passed=1,
        loss_ratio=0.7,
        trust_label="test_fixture",
        bypass_quality_gate=True,
    )
    nb.flush_writes()
    nb.close()

    externalize_artifacts(
        db_path=db_path,
        min_bytes=16,
        apply=True,
        limit=None,
        vacuum=False,
        include_graph_json=True,
        graph_json_cold_only=False,
    )

    nb = LabNotebook(db_path, use_native=False)
    row = nb.conn.execute(
        "SELECT graph_json FROM program_results_compat WHERE result_id = ?",
        (rid,),
    ).fetchone()
    pointer = parse_artifact_pointer(row["graph_json"])
    assert pointer is not None
    (NotebookArtifactStore(db_path).root / pointer["path"]).unlink()

    coverage = ExperimentAnalytics(nb).math_family_coverage()

    assert coverage["totals"] == {"n_tested": 0, "n_survived": 0}
    assert all(family["n_tested"] == 0 for family in coverage["families"])
    nb.close()


def test_healer_task_payload_round_trips_from_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("ARIA_NOTEBOOK_ARTIFACT_MIN_BYTES", "32")
    nb = LabNotebook(tmp_path / "lab_notebook.db", use_native=False)

    task_id = nb.create_healer_task(
        experiment_id="exp-heal",
        trigger_type="test",
        scope="unit",
        reproduction_steps=["run pytest"],
        acceptance_tests=["passes"],
        model_endpoint=None,
        sandbox_policy={},
        trigger_payload={"values": list(range(30))},
    )

    row = nb.conn.execute(
        "SELECT trigger_payload_json FROM healer_tasks WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    assert parse_artifact_pointer(row["trigger_payload_json"]) is not None
    task = nb.get_healer_task(task_id)
    assert task is not None
    assert task["trigger_payload_json"]["values"] == list(range(30))


def test_entry_metadata_round_trips_from_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("ARIA_NOTEBOOK_ARTIFACT_MIN_BYTES", "32")
    nb = LabNotebook(tmp_path / "lab_notebook.db", use_native=False)

    entry_id = nb.add_entry(
        ExperimentEntry(
            experiment_id="exp-entry",
            entry_type="note",
            title="large metadata",
            content="body",
            metadata={"values": list(range(30))},
        )
    )

    row = nb.conn.execute(
        "SELECT metadata_json FROM entries WHERE entry_id = ?",
        (entry_id,),
    ).fetchone()
    assert parse_artifact_pointer(row["metadata_json"]) is not None
    entries = nb.get_entries(experiment_id="exp-entry")
    assert entries
    assert '"values"' in entries[0]["metadata_json"]


def test_externalize_tool_moves_healer_and_entry_payloads(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(db_path, use_native=False)
    nb.create_healer_task(
        experiment_id="exp-tool-heal",
        trigger_type="test",
        scope="unit",
        reproduction_steps=[],
        acceptance_tests=[],
        model_endpoint=None,
        sandbox_policy={},
        trigger_payload={"values": list(range(30))},
    )
    nb.add_entry(
        ExperimentEntry(
            experiment_id="exp-tool-entry",
            entry_type="note",
            title="existing metadata",
            content="body",
            metadata={"values": list(range(30))},
        )
    )
    nb.close()

    report = externalize_artifacts(
        db_path=db_path,
        min_bytes=16,
        apply=True,
        limit=None,
        vacuum=False,
    )

    applied = {
        (item["table"], item["column"]): item["rows"] for item in report["applied"]
    }
    assert applied[("healer_tasks", "trigger_payload_json")] == 1
    assert applied[("entries", "metadata_json")] == 1

    nb = LabNotebook(db_path, use_native=False)
    task = nb.get_recent_healer_tasks(limit=1)[0]
    assert task["trigger_payload_json"]["values"] == list(range(30))
    entry = nb.get_entries(experiment_id="exp-tool-entry")[0]
    assert '"values"' in entry["metadata_json"]


def test_restore_inline_tool_rehydrates_raw_sql_sensitive_columns(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(db_path, use_native=False)
    rid = nb.record_program_result(
        experiment_id="exp-inline",
        graph_fingerprint="fp-inline",
        graph_json='{"nodes":[]}',
        result_id="rid-inline",
        stage0_passed=1,
        stage1_passed=1,
        loss_ratio=0.7,
        trust_label="test_fixture",
        data_provenance_json=json.dumps({"values": list(range(30))}),
        bypass_quality_gate=True,
    )
    nb.flush_writes()
    pointer = nb._store_artifact_payload(
        table_name="graph_runs",
        row_pk=rid,
        column_name="data_provenance_json",
        payload=json.dumps({"values": list(range(30))}),
    )
    nb.conn.execute(
        "UPDATE graph_runs SET data_provenance_json = ? WHERE result_id = ?",
        (pointer, rid),
    )
    nb.conn.commit()
    nb.close()

    report = restore_inline_artifacts(
        db_path=db_path,
        targets=(("graph_runs", "result_id", "data_provenance_json"),),
        apply=True,
        limit=None,
        vacuum=False,
    )

    assert report["restored"][0]["rows"] == 1
    conn = sqlite3.connect(db_path)
    raw = conn.execute(
        "SELECT data_provenance_json FROM program_results_compat WHERE result_id = ?",
        (rid,),
    ).fetchone()[0]
    conn.close()
    assert parse_artifact_pointer(raw) is None
    assert json.loads(raw)["values"] == list(range(30))
