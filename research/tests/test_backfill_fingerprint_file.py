from __future__ import annotations

import json

from research.scientist.notebook import LabNotebook
from research.tools import backfill


def _seed_rows(nb: LabNotebook) -> list[tuple[str, str]]:
    exp_id = nb.start_experiment("synthesis", {"test": True}, "fingerprint-file test")
    rows: list[tuple[str, str]] = []
    for i in range(4):
        rid = f"rid-{i:02d}"
        fp = f"fp{i:02d}cafebabe"
        nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=fp,
            graph_json=json.dumps({"nodes": [], "i": i}),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.5,
            result_id=rid,
            trust_label="test_fixture",
        )
        rows.append((rid, fp))
    nb.flush_writes()
    return rows


def test_fingerprint_file_preserves_priority_order(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(db_path)
    rows = _seed_rows(nb)

    priority_path = tmp_path / "priority.jsonl"
    # Intentionally reverse the rows so ordering is non-trivial.
    with priority_path.open("w") as fh:
        for rid, fp in reversed(rows):
            fh.write(json.dumps({"result_id": rid, "fp": fp, "priority": 1.0}) + "\n")

    candidates = backfill.query_fingerprint_file_candidates(
        nb,
        str(priority_path),
        null_column=None,
        force=False,
    )

    assert [c.graph_fingerprint for c in candidates] == [fp for _, fp in reversed(rows)]
    assert all(c.tier == "screening" for c in candidates)
    assert all(c.graph_json is not None for c in candidates)


def test_fingerprint_file_dedupes_and_honors_limit(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(db_path)
    rows = _seed_rows(nb)

    priority_path = tmp_path / "priority.jsonl"
    with priority_path.open("w") as fh:
        for rid, fp in rows:
            fh.write(json.dumps({"result_id": rid, "fp": fp}) + "\n")
        # Duplicate the first row — should be deduped.
        dup_rid, dup_fp = rows[0]
        fh.write(json.dumps({"result_id": dup_rid, "fp": dup_fp}) + "\n")

    candidates = backfill.query_fingerprint_file_candidates(
        nb,
        str(priority_path),
        null_column=None,
        force=False,
        limit=2,
    )

    assert len(candidates) == 2
    assert candidates[0].graph_fingerprint == rows[0][1]
    assert candidates[1].graph_fingerprint == rows[1][1]


def test_fingerprint_file_skips_missing_result_ids(tmp_path):
    db_path = tmp_path / "lab_notebook.db"
    nb = LabNotebook(db_path)
    rows = _seed_rows(nb)

    priority_path = tmp_path / "priority.jsonl"
    with priority_path.open("w") as fh:
        fh.write(json.dumps({"result_id": "rid-does-not-exist", "fp": "ghost"}) + "\n")
        for rid, fp in rows[:1]:
            fh.write(json.dumps({"result_id": rid, "fp": fp}) + "\n")

    candidates = backfill.query_fingerprint_file_candidates(
        nb,
        str(priority_path),
        null_column=None,
        force=False,
    )

    assert [c.graph_fingerprint for c in candidates] == [rows[0][1]]
