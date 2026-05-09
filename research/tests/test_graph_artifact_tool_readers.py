from __future__ import annotations

import json
import sqlite3

from research.scientist.notebook import LabNotebook
from research.scientist.notebook.artifact_store import parse_artifact_pointer
from research.tools.backfill import query_fingerprint_file_candidates
from research.tools.backpopulate_screening_metrics import _row_to_payload
from research.tools.externalize_notebook_artifacts import run as externalize_artifacts


def _artifact_backed_program_db(tmp_path):
    db_path = tmp_path / "runs.db"
    graph_json = json.dumps(
        {
            "nodes": {
                "0": {"id": 0, "op_name": "input", "input_ids": []},
                "1": {"id": 1, "op_name": "linear_proj", "input_ids": [0]},
            },
            "metadata": {"templates_used": ["unit_test"]},
        },
        sort_keys=True,
    )
    nb = LabNotebook(db_path, use_native=False)
    nb.record_program_result(
        experiment_id="exp-tool-artifact",
        graph_fingerprint="fp-tool-artifact",
        graph_json=graph_json,
        result_id="rid-tool-artifact",
        stage0_passed=1,
        stage05_passed=1,
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
    return db_path, graph_json


def test_backpopulate_payload_resolves_artifact_backed_graph_json(tmp_path):
    db_path, graph_json = _artifact_backed_program_db(tmp_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM program_results WHERE result_id = 'rid-tool-artifact'"
        ).fetchone()
        assert parse_artifact_pointer(row["graph_json"]) is not None
        payload = _row_to_payload(row, conn=conn, db_path=db_path)
    finally:
        conn.close()

    assert payload["graph_json"] == graph_json


def test_unified_backfill_candidate_resolves_artifact_backed_graph_json(tmp_path):
    db_path, graph_json = _artifact_backed_program_db(tmp_path)
    priority_path = tmp_path / "priority.jsonl"
    priority_path.write_text(
        json.dumps(
            {
                "result_id": "rid-tool-artifact",
                "fp": "fp-tool-artifact",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    nb = LabNotebook(db_path, use_native=False)
    try:
        candidates = query_fingerprint_file_candidates(
            nb,
            str(priority_path),
            null_column=None,
            force=True,
        )
    finally:
        nb.close()

    assert len(candidates) == 1
    assert candidates[0].graph_json == graph_json
