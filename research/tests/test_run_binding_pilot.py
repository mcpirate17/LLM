from __future__ import annotations

import json

from research.scientist.notebook import LabNotebook
from research.tools import run_binding_pilot as mod


def test_dedupe_manifest_rows_prefers_first_unique_fingerprint() -> None:
    rows = [
        {
            "result_id": "r1",
            "graph_fingerprint": "fp_a",
            "timestamp": 1.0,
            "composite_score": 0.9,
        },
        {
            "result_id": "r2",
            "graph_fingerprint": "fp_a",
            "timestamp": 2.0,
            "composite_score": 0.8,
        },
        {
            "result_id": "r1",
            "graph_fingerprint": "fp_b",
            "timestamp": 3.0,
            "composite_score": 0.7,
        },
        {
            "result_id": "r3",
            "graph_fingerprint": "",
            "timestamp": 4.0,
            "composite_score": 0.6,
        },
        {
            "result_id": "r4",
            "graph_fingerprint": "",
            "timestamp": 5.0,
            "composite_score": 0.5,
        },
    ]

    deduped = mod._dedupe_manifest_rows(rows)

    assert [row["result_id"] for row in deduped] == ["r1", "r3", "r4"]


def test_store_result_rewrites_existing_row_without_duplicate_entries(tmp_path) -> None:
    results_path = tmp_path / "results.tsv"
    mod._ensure_results_header(results_path)
    done = mod._load_done(results_path)

    mod._store_result(
        results_path,
        done,
        {
            "result_id": "r1",
            "graph_fingerprint": "fp1",
            "status": "exit_1",
            "elapsed_s": 1.0,
            "started_at": 10.0,
            "finished_at": 11.0,
            "worker_pid": 100,
            "report_path": "/tmp/old.tsv",
        },
    )
    mod._store_result(
        results_path,
        done,
        {
            "result_id": "r1",
            "graph_fingerprint": "fp1",
            "status": "ok",
            "elapsed_s": 2.0,
            "started_at": 12.0,
            "finished_at": 13.0,
            "worker_pid": 101,
            "report_path": "/tmp/new.tsv",
        },
    )

    loaded = mod._load_done(results_path)
    assert list(loaded) == ["r1"]
    assert loaded["r1"]["status"] == "ok"
    assert loaded["r1"]["worker_pid"] == "101"


def test_load_completed_binding_result_ids_detects_existing_binding_rows(
    tmp_path, monkeypatch
) -> None:
    db_path = tmp_path / "lab_notebook.db"
    monkeypatch.setattr(mod, "DB_PATH", db_path)
    nb = LabNotebook(str(db_path))
    try:
        exp_id = nb.start_experiment("backfill", {}, "binding-pilot-test")
        rid_done = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_done",
            graph_json='{"nodes":[],"edges":[]}',
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            binding_auc=0.2,
        )
        rid_pending = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_pending",
            graph_json='{"nodes":[],"edges":[]}',
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            binding_auc=None,
        )
        nb.flush_writes()
    finally:
        nb.close()

    completed = mod._load_completed_binding_result_ids({rid_done, rid_pending})

    assert completed == {rid_done}


def test_write_status_records_active_completed_and_remaining_rows(tmp_path) -> None:
    status_path = tmp_path / "status.json"
    manifest = [
        {"result_id": "r1", "graph_fingerprint": "fp1"},
        {"result_id": "r2", "graph_fingerprint": "fp2"},
        {"result_id": "r3", "graph_fingerprint": "fp3"},
    ]
    done = {"r1": {"status": "db_done"}}

    class _Proc:
        pid = 321

    active = {
        "r2": {
            "proc": _Proc(),
            "result_row": {"graph_fingerprint": "fp2"},
            "started_at": 123.456,
        }
    }

    mod._write_status(
        status_path, manifest=manifest, done=done, active=active, vram_samples=7
    )
    payload = json.loads(status_path.read_text(encoding="utf-8"))

    assert payload["completed_rows"] == [
        {"result_id": "r1", "graph_fingerprint": "fp1", "status": "db_done"}
    ]
    assert payload["active_rows"] == [
        {
            "result_id": "r2",
            "graph_fingerprint": "fp2",
            "worker_pid": 321,
            "started_at": 123.456,
        }
    ]
    assert payload["remaining_rows"] == [
        {"result_id": "r3", "graph_fingerprint": "fp3"}
    ]
