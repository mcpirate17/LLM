from __future__ import annotations

import csv
import json
import sqlite3
from types import SimpleNamespace

import pytest

from research.tools import calibrate_intermediate_probes as tool

pytestmark = pytest.mark.unit


def test_load_targets_from_json_filters_missing_by_default(tmp_path):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"placeholder")
    targets_path = tmp_path / "targets.json"
    targets_path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "name": "present",
                        "checkpoint_path": str(ckpt),
                        "fingerprint": "fp-present",
                        "branch": "branch-a",
                    },
                    {
                        "name": "missing",
                        "checkpoint_path": str(tmp_path / "missing.pt"),
                        "fingerprint": "fp-missing",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    targets = tool.load_targets(
        preset="none",
        targets_json=targets_path,
        include_missing=False,
    )

    assert [target.name for target in targets] == ["present"]
    assert targets[0].fingerprint == "fp-present"
    assert targets[0].branch == "branch-a"


def test_lookup_graph_json_uses_read_only_connection(tmp_path):
    db_path = tmp_path / "lab.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE program_results(result_id TEXT, graph_fingerprint TEXT, graph_json TEXT)"
    )
    conn.execute(
        "INSERT INTO program_results VALUES (?, ?, ?)",
        ("rid", "fp", '{"nodes":{"0":{"op_name":"input"}}}'),
    )
    conn.commit()
    conn.close()

    ro = tool.connect_ro(db_path)
    try:
        target = tool.CalibrationTarget(
            name="target",
            checkpoint_path=tmp_path / "checkpoint.pt",
            fingerprint="fp",
        )
        assert tool.lookup_graph_json(ro, target).startswith('{"nodes"')
        with pytest.raises(sqlite3.OperationalError):
            ro.execute("CREATE TABLE should_fail(x)")
    finally:
        ro.close()


def test_result_row_and_artifact_writer_round_trip(tmp_path):
    target = tool.CalibrationTarget(
        name="target",
        checkpoint_path=tmp_path / "checkpoint.pt",
        model="m",
        branch="b",
    )
    result = SimpleNamespace(
        status="ok",
        to_dict=lambda: {
            "ar_intermediate_status": "ok",
            "ar_intermediate_elapsed_ms": 12.5,
            "ar_intermediate_diagnostic_score": 1.25,
            "ar_intermediate_held_pair_acc": 0.125,
        },
    )

    row = tool.result_row(
        run_id="run",
        created_unix=1.0,
        target=target,
        checkpoint_step=100,
        probe="ar_intermediate",
        seed=7,
        wall_seconds=0.25,
        result=result,
    )
    csv_path = tmp_path / "out.csv"
    jsonl_path = tmp_path / "out.jsonl"
    tool.append_artifact_rows(csv_path=csv_path, jsonl_path=jsonl_path, rows=[row])

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["target_name"] == "target"
    assert rows[0]["ar_intermediate_diagnostic_score"] == "1.25"
    assert rows[0]["seed"] == "7"
    assert json.loads(jsonl_path.read_text(encoding="utf-8"))["checkpoint_step"] == 100


def test_cli_dry_run_prints_selected_targets(tmp_path, capsys):
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"placeholder")
    targets_path = tmp_path / "targets.json"
    targets_path.write_text(
        json.dumps([{"name": "dry", "checkpoint_path": str(ckpt)}]),
        encoding="utf-8",
    )

    rc = tool.main(
        [
            "--preset",
            "none",
            "--targets-json",
            str(targets_path),
            "--dry-run",
            "--out-dir",
            str(tmp_path),
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr().out
    assert rc == 0
    assert '"event": "selected"' in captured
    assert '"name": "dry"' in captured
