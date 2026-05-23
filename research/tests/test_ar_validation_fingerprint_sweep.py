from __future__ import annotations

import csv
import sqlite3
import warnings

import numpy as np
import pytest
import torch

from research.tools.run_ar_validation_fingerprint_sweep import (
    CSV_FIELDS,
    _append_row,
    _load_done_fingerprints,
    _load_projected_corpus,
    _query_top_fingerprints,
    main,
)


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE leaderboard (
            result_id TEXT,
            graph_fingerprint TEXT,
            model_source TEXT,
            tier TEXT,
            is_reference INTEGER,
            reference_name TEXT,
            composite_score REAL,
            validation_loss_ratio REAL,
            timestamp REAL
        );
        CREATE TABLE program_results (
            result_id TEXT,
            experiment_id TEXT,
            graph_fingerprint TEXT,
            graph_json TEXT,
            loss_ratio REAL
        );
        """
    )
    graph_json = '{"model_dim": 8, "nodes": {}, "metadata": {}}'
    rows = [
        ("low", "exp-low", "fp-low", graph_json, 0.7, 10.0, 1.0, 0),
        ("top-a-old", "exp-a1", "fp-a", graph_json, 0.5, 90.0, 2.0, 0),
        ("top-a", "exp-a2", "fp-a", graph_json, 0.4, 100.0, 3.0, 0),
        ("ref", "exp-ref", "fp-ref", graph_json, 0.3, 1000.0, 4.0, 1),
        ("top-b", "exp-b", "fp-b", graph_json, 0.6, 80.0, 5.0, 0),
    ]
    for rid, exp, fp, graph, loss, score, ts, is_ref in rows:
        conn.execute(
            "INSERT INTO program_results VALUES (?, ?, ?, ?, ?)",
            (rid, exp, fp, graph, loss),
        )
        conn.execute(
            "INSERT INTO leaderboard VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rid,
                fp,
                "test",
                "validation",
                is_ref,
                "ref" if is_ref else None,
                score,
                loss,
                ts,
            ),
        )
    return conn


def test_query_top_fingerprints_dedupes_by_best_score_and_excludes_references():
    rows = _query_top_fingerprints(
        _make_db(),
        limit=3,
        offset=0,
        include_references=False,
    )

    assert [row["result_id"] for row in rows] == ["top-a", "top-b", "low"]
    assert [row["graph_fingerprint"] for row in rows] == ["fp-a", "fp-b", "fp-low"]


def test_query_top_fingerprints_can_include_references():
    rows = _query_top_fingerprints(
        _make_db(),
        limit=2,
        offset=0,
        include_references=True,
    )

    assert [row["result_id"] for row in rows] == ["ref", "top-a"]


def test_append_row_and_done_fingerprints_round_trip(tmp_path):
    csv_path = tmp_path / "sweep.csv"

    _append_row(
        csv_path,
        {
            "run_id": "run",
            "graph_fingerprint": "fp-ok",
            "ar_validation_status": "ok",
            "ar_validation_rank_score": 0.5,
        },
    )
    _append_row(
        csv_path,
        {
            "run_id": "run",
            "graph_fingerprint": "fp-error",
            "ar_validation_status": "exception",
        },
    )

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["ar_validation_rank_score"] == "0.5"
    assert _load_done_fingerprints(csv_path) == {"fp-ok"}


def test_sweep_csv_fields_include_stable_v3_metadata():
    expected = {
        "ar_validation_size_bucket",
        "ar_validation_param_count",
        "ar_validation_seed_count",
        "ar_validation_seed_scores_json",
        "ar_validation_rank_score_mean",
        "ar_validation_rank_score_std",
        "ar_validation_rank_score_stable",
        "ar_validation_budget_json",
        "ar_validation_checkpoint_path",
        "ar_validation_stage_status",
        "ar_validation_stage_elapsed_ms",
    }

    assert expected.issubset(set(CSV_FIELDS))


def test_load_projected_corpus_copies_read_only_mmap_before_projection(tmp_path):
    corpus_path = tmp_path / "tokens.npy"
    np.save(corpus_path, np.arange(12, dtype=np.int64))
    mmap = np.load(str(corpus_path), mmap_mode="r")
    assert not mmap.flags.writeable

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tokens = _load_projected_corpus(
            corpus_path,
            5,
            device=torch.device("cpu"),
        )

    assert tokens.tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]
    assert tokens.is_contiguous()


def test_sweep_refuses_cpu_device(tmp_path):
    with pytest.raises(
        SystemExit, match="ar_validation_fingerprint_sweep_requires_cuda"
    ):
        main(["--device", "cpu", "--dry-run", "--out", str(tmp_path / "out.csv")])
