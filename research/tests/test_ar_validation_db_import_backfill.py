from __future__ import annotations

import csv
import json
import sqlite3

from research.tools import backfill_ar_validation as backfill_tool
from research.tools import import_ar_validation_fingerprint_sweep as import_tool


def _make_import_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE graph_runs (
            result_id TEXT PRIMARY KEY,
            graph_fingerprint TEXT,
            data_provenance_json TEXT
        )
        """,
    )
    import_tool.ensure_ar_validation_columns(conn)
    return conn


def _csv_metric_row(**overrides):
    row = {
        "run_id": "ar_validation_fp_sweep_test",
        "result_id": "rid-1",
        "graph_fingerprint": "fp-1",
        "ar_validation_metric_version": "ar_validation_v2_easy25",
        "ar_validation_status": "ok",
        "ar_validation_final_acc": "0.1797",
        "ar_validation_held_pair_acc": "0.0859",
        "ar_validation_held_class_acc": "0.1562",
        "ar_validation_steps_to_floor": "1500",
        "ar_validation_rank_score": "1.7344",
        "ar_validation_elapsed_ms": "1234.5",
        "ar_validation_size_bucket": "20m",
        "ar_validation_param_count": "20000000",
        "ar_validation_seed_count": "3",
        "ar_validation_seed_scores_json": '[{"seed":0,"score":1.0}]',
        "ar_validation_rank_score_mean": "1.7344",
        "ar_validation_rank_score_std": "0.25",
        "ar_validation_rank_score_stable": "1.4844",
        "ar_validation_held_pair_acc_mean": "0.0859",
        "ar_validation_held_pair_acc_std": "0.01",
        "ar_validation_held_class_acc_mean": "0.1562",
        "ar_validation_held_class_acc_std": "0.02",
        "ar_validation_budget_json": '{"size_bucket":"20m","train_steps":7500}',
        "ar_validation_checkpoint_path": "/tmp/stage.pt",
        "ar_validation_stage_status": "ok",
        "ar_validation_stage_elapsed_ms": "55.5",
        "learning_curve_json": '[{"step": 500, "final_acc": 0.10}]',
    }
    row.update(overrides)
    return row


def _indexes(conn: sqlite3.Connection):
    return import_tool._load_db_indexes(conn)


def test_csv_row_parsing_normalizes_meaningful_ar_validation_fields(tmp_path):
    parsed = import_tool.parse_csv_metric_row(
        _csv_metric_row(),
        source_csv=tmp_path / "sweep.csv",
        source_line=2,
    )

    assert parsed.result_id == "rid-1"
    assert parsed.graph_fingerprint == "fp-1"
    assert parsed.values["ar_validation_metric_version"] == "ar_validation_v2_easy25"
    assert parsed.values["ar_validation_final_acc"] == 0.1797
    assert parsed.values["ar_validation_held_pair_acc"] == 0.0859
    assert parsed.values["ar_validation_held_class_acc"] == 0.1562
    assert parsed.values["ar_validation_steps_to_floor"] == 1500
    assert parsed.values["ar_validation_rank_score"] == 1.7344
    assert parsed.values["ar_validation_size_bucket"] == "20m"
    assert parsed.values["ar_validation_param_count"] == 20_000_000
    assert parsed.values["ar_validation_seed_count"] == 3
    assert parsed.values["ar_validation_rank_score_std"] == 0.25
    assert json.loads(parsed.values["ar_validation_seed_scores_json"]) == [
        {"score": 1.0, "seed": 0}
    ]
    assert json.loads(parsed.values["ar_validation_budget_json"]) == {
        "size_bucket": "20m",
        "train_steps": 7500,
    }
    assert json.loads(parsed.values["ar_validation_learning_curve_json"]) == [
        {"final_acc": 0.10, "step": 500},
    ]


def test_import_updates_once_and_then_becomes_idempotent(tmp_path):
    conn = _make_import_db()
    conn.execute(
        "INSERT INTO graph_runs (result_id, graph_fingerprint, data_provenance_json) VALUES (?, ?, ?)",
        ("rid-1", "fp-1", "{}"),
    )
    csv_row = import_tool.parse_csv_metric_row(
        _csv_metric_row(),
        source_csv=tmp_path / "sweep.csv",
        source_line=2,
    )

    by_result_id, by_fingerprint = _indexes(conn)
    first_plan = import_tool.plan_import(
        [csv_row],
        by_result_id=by_result_id,
        by_fingerprint=by_fingerprint,
        overwrite=False,
    )
    assert [(d.action, d.reason) for d in first_plan] == [
        ("update", "missing_ar_validation")
    ]
    assert import_tool.apply_import_decisions(conn, first_plan, overwrite=False) == 1

    row = conn.execute(
        "SELECT ar_validation_rank_score, data_provenance_json FROM graph_runs WHERE result_id = ?",
        ("rid-1",),
    ).fetchone()
    assert row["ar_validation_rank_score"] == 1.7344
    provenance = json.loads(row["data_provenance_json"])
    assert (
        provenance["last_metric_backfill"]["source"]
        == "ar_validation_fingerprint_sweep_csv_import"
    )

    by_result_id, by_fingerprint = _indexes(conn)
    second_plan = import_tool.plan_import(
        [csv_row],
        by_result_id=by_result_id,
        by_fingerprint=by_fingerprint,
        overwrite=False,
    )
    assert [(d.action, d.reason) for d in second_plan] == [
        ("skip", "existing_ar_validation_values"),
    ]


def test_import_does_not_overwrite_without_flag_and_overwrites_with_flag(tmp_path):
    conn = _make_import_db()
    conn.execute(
        """
        INSERT INTO graph_runs (
            result_id, graph_fingerprint, ar_validation_rank_score
        ) VALUES (?, ?, ?)
        """,
        ("rid-1", "fp-1", 9.9),
    )
    csv_row = import_tool.parse_csv_metric_row(
        _csv_metric_row(ar_validation_rank_score="1.7344"),
        source_csv=tmp_path / "sweep.csv",
        source_line=2,
    )

    by_result_id, by_fingerprint = _indexes(conn)
    no_overwrite = import_tool.plan_import(
        [csv_row],
        by_result_id=by_result_id,
        by_fingerprint=by_fingerprint,
        overwrite=False,
    )
    assert no_overwrite[0].action == "skip"
    assert no_overwrite[0].reason == "existing_ar_validation_values"

    overwrite = import_tool.plan_import(
        [csv_row],
        by_result_id=by_result_id,
        by_fingerprint=by_fingerprint,
        overwrite=True,
    )
    assert overwrite[0].action == "update"
    assert overwrite[0].values["ar_validation_rank_score"] == 1.7344
    import_tool.apply_import_decisions(conn, overwrite, overwrite=True)
    score = conn.execute(
        """
        SELECT ar_validation_rank_score, data_provenance_json
        FROM graph_runs
        WHERE result_id = ?
        """,
        ("rid-1",),
    ).fetchone()
    assert score["ar_validation_rank_score"] == 1.7344
    previous = json.loads(score["data_provenance_json"])["last_metric_backfill"][
        "previous_ar_validation_values"
    ]
    assert previous["ar_validation_rank_score"] == 9.9


def test_ambiguous_fingerprint_fallback_is_skipped_and_reported(tmp_path):
    conn = _make_import_db()
    conn.execute(
        "INSERT INTO graph_runs (result_id, graph_fingerprint) VALUES (?, ?)",
        ("rid-a", "fp-shared"),
    )
    conn.execute(
        "INSERT INTO graph_runs (result_id, graph_fingerprint) VALUES (?, ?)",
        ("rid-b", "fp-shared"),
    )
    csv_row = import_tool.parse_csv_metric_row(
        _csv_metric_row(result_id="", graph_fingerprint="fp-shared"),
        source_csv=tmp_path / "sweep.csv",
        source_line=2,
    )

    by_result_id, by_fingerprint = _indexes(conn)
    plan = import_tool.plan_import(
        [csv_row],
        by_result_id=by_result_id,
        by_fingerprint=by_fingerprint,
        overwrite=False,
    )

    assert plan[0].action == "skip"
    assert plan[0].reason == "ambiguous_fingerprint"
    assert plan[0].match_mode == "fingerprint"


def test_load_csv_metric_rows_reads_csv_file(tmp_path):
    csv_path = tmp_path / "sweep.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_csv_metric_row().keys()))
        writer.writeheader()
        writer.writerow(_csv_metric_row())

    rows = import_tool.load_csv_metric_rows([csv_path])

    assert len(rows) == 1
    assert rows[0].values["ar_validation_rank_score"] == 1.7344


def test_backfill_update_payload_and_sql_use_ar_validation_fields():
    conn = _make_import_db()
    conn.execute(
        "INSERT INTO graph_runs (result_id, graph_fingerprint, data_provenance_json) VALUES (?, ?, ?)",
        ("rid-1", "fp-1", "{}"),
    )
    result_row = {
        "ar_validation_metric_version": "ar_validation_v2_easy25",
        "ar_validation_final_acc": 0.1797,
        "ar_validation_held_pair_acc": 0.0859,
        "ar_validation_held_class_acc": 0.1562,
        "ar_validation_learning_curve_json": '[{"step":500,"final_acc":0.10}]',
        "ar_validation_steps_to_floor": 1500,
        "ar_validation_rank_score": 1.7344,
        "ar_validation_status": "ok",
        "ar_validation_elapsed_ms": 1234.5,
        "ar_validation_size_bucket": "20m",
        "ar_validation_param_count": 20_000_000,
        "ar_validation_seed_count": 3,
        "ar_validation_seed_scores_json": '[{"seed":0,"score":1.7344}]',
        "ar_validation_rank_score_mean": 1.7344,
        "ar_validation_rank_score_std": 0.25,
        "ar_validation_rank_score_stable": 1.4844,
        "ar_validation_held_pair_acc_mean": 0.0859,
        "ar_validation_held_pair_acc_std": 0.01,
        "ar_validation_held_class_acc_mean": 0.1562,
        "ar_validation_held_class_acc_std": 0.02,
        "ar_validation_budget_json": '{"size_bucket":"20m","train_steps":7500}',
        "ar_validation_checkpoint_path": "/tmp/stage.pt",
        "ar_validation_stage_status": "ok",
        "ar_validation_stage_elapsed_ms": 55.5,
    }
    values = backfill_tool._ar_validation_values_from_result_row(result_row)

    assert set(values) == set(import_tool.AR_VALIDATION_COLUMNS)
    assert backfill_tool.persist_ar_validation_result(
        conn,
        result_id="rid-1",
        values=values,
        provenance={"source": "test_backfill"},
        overwrite=False,
    )
    row = conn.execute(
        """
        SELECT ar_validation_metric_version,
               ar_validation_held_pair_acc,
               ar_validation_held_class_acc,
               ar_validation_rank_score,
               ar_validation_size_bucket,
               ar_validation_seed_count,
               ar_validation_status,
               data_provenance_json
        FROM graph_runs
        WHERE result_id = ?
        """,
        ("rid-1",),
    ).fetchone()

    assert row["ar_validation_metric_version"] == "ar_validation_v2_easy25"
    assert row["ar_validation_held_pair_acc"] == 0.0859
    assert row["ar_validation_held_class_acc"] == 0.1562
    assert row["ar_validation_rank_score"] == 1.7344
    assert row["ar_validation_size_bucket"] == "20m"
    assert row["ar_validation_seed_count"] == 3
    assert row["ar_validation_status"] == "ok"
    assert (
        json.loads(row["data_provenance_json"])["last_metric_backfill"]["source"]
        == "test_backfill"
    )


def test_backfill_persist_does_not_overwrite_existing_without_flag_and_overwrites_with_flag():
    conn = _make_import_db()
    conn.execute(
        """
        INSERT INTO graph_runs (
            result_id, graph_fingerprint, ar_validation_rank_score
        ) VALUES (?, ?, ?)
        """,
        ("rid-1", "fp-1", 9.9),
    )
    values = {
        "ar_validation_metric_version": "ar_validation_v2_easy25",
        "ar_validation_rank_score": 1.7344,
        "ar_validation_status": "ok",
    }

    assert not backfill_tool.persist_ar_validation_result(
        conn,
        result_id="rid-1",
        values=values,
        provenance={"source": "test_backfill"},
        overwrite=False,
    )
    score = conn.execute(
        "SELECT ar_validation_rank_score FROM graph_runs WHERE result_id = ?",
        ("rid-1",),
    ).fetchone()[0]
    assert score == 9.9

    assert backfill_tool.persist_ar_validation_result(
        conn,
        result_id="rid-1",
        values=values,
        provenance={"source": "test_backfill"},
        overwrite=True,
    )
    score = conn.execute(
        "SELECT ar_validation_rank_score FROM graph_runs WHERE result_id = ?",
        ("rid-1",),
    ).fetchone()[0]
    assert score == 1.7344


def test_backfill_selection_skips_rows_with_existing_ar_validation_unless_overwrite():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE graph_runs (
            result_id TEXT PRIMARY KEY,
            experiment_id TEXT,
            timestamp REAL,
            graph_fingerprint TEXT,
            graph_json TEXT,
            model_source TEXT,
            loss_ratio REAL
        );
        CREATE TABLE leaderboard (
            result_id TEXT,
            graph_fingerprint TEXT,
            model_source TEXT,
            tier TEXT,
            is_reference INTEGER,
            reference_name TEXT,
            composite_score REAL,
            validation_loss_ratio REAL
        );
        """
    )
    import_tool.ensure_ar_validation_columns(conn)
    for rid, score, status in (
        ("missing", None, None),
        ("existing-score", 2.0, None),
        ("existing-status", None, "exception"),
    ):
        conn.execute(
            """
            INSERT INTO graph_runs (
                result_id, experiment_id, timestamp, graph_fingerprint, graph_json,
                model_source, loss_ratio, ar_validation_rank_score, ar_validation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                "exp",
                1.0,
                f"fp-{rid}",
                '{"nodes":{}}',
                "graph_synthesis",
                0.5,
                score,
                status,
            ),
        )
        conn.execute(
            "INSERT INTO leaderboard VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (rid, f"fp-{rid}", "graph_synthesis", "validation", 0, "", 10.0, 0.5),
        )

    rows = backfill_tool.select_backfill_rows(
        conn,
        tiers=("validation",),
        result_ids=(),
        fingerprints=(),
        limit=10,
        offset=0,
        overwrite=False,
    )

    assert [row["result_id"] for row in rows] == ["missing"]

    overwrite_rows = backfill_tool.select_backfill_rows(
        conn,
        tiers=("validation",),
        result_ids=(),
        fingerprints=(),
        limit=10,
        offset=0,
        overwrite=True,
    )

    assert {row["result_id"] for row in overwrite_rows} == {
        "missing",
        "existing-score",
        "existing-status",
    }


def test_backfill_selection_preserves_explicit_result_id_order():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE graph_runs (
            result_id TEXT PRIMARY KEY,
            experiment_id TEXT,
            timestamp REAL,
            graph_fingerprint TEXT,
            graph_json TEXT,
            model_source TEXT,
            loss_ratio REAL
        );
        CREATE TABLE leaderboard (
            result_id TEXT,
            graph_fingerprint TEXT,
            model_source TEXT,
            tier TEXT,
            is_reference INTEGER,
            reference_name TEXT,
            composite_score REAL,
            validation_loss_ratio REAL
        );
        """
    )
    import_tool.ensure_ar_validation_columns(conn)
    for rid, score in (
        ("wanted-third", 300.0),
        ("wanted-first", 100.0),
        ("wanted-second", 200.0),
    ):
        conn.execute(
            """
            INSERT INTO graph_runs (
                result_id, experiment_id, timestamp, graph_fingerprint, graph_json,
                model_source, loss_ratio
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                "exp",
                1.0,
                f"fp-{rid}",
                '{"nodes":{}}',
                "graph_synthesis",
                0.5,
            ),
        )
        conn.execute(
            "INSERT INTO leaderboard VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (rid, f"fp-{rid}", "graph_synthesis", "validation", 0, "", score, 0.5),
        )

    rows = backfill_tool.select_backfill_rows(
        conn,
        tiers=(),
        result_ids=("wanted-first", "wanted-second", "wanted-third"),
        fingerprints=(),
        limit=10,
        offset=0,
        overwrite=False,
    )

    assert [row["result_id"] for row in rows] == [
        "wanted-first",
        "wanted-second",
        "wanted-third",
    ]
