from __future__ import annotations

import sqlite3
from pathlib import Path

from research.scientist.analytics._exp_weights import _WeightsMixin
from research.scientist.analytics.analytics_routing import _RoutingMixin
from research.scientist.intelligence.ml_corpus import load_deduped_graph_training_rows
from research.tests._ml_corpus_test_support import graph_json


def _create_boundary_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE program_results (
            result_id TEXT,
            graph_json TEXT,
            graph_fingerprint TEXT,
            loss_ratio REAL,
            wikitext_perplexity REAL,
            stage0_passed INTEGER,
            stage05_passed INTEGER,
            stage1_passed INTEGER,
            timestamp REAL,
            routing_mode TEXT,
            routing_tokens_total REAL,
            routing_tokens_processed REAL,
            routing_tokens_skipped REAL,
            routing_drop_rate REAL,
            routing_utilization_entropy REAL,
            routing_capacity_overflow_count REAL,
            routing_confidence_mean REAL,
            routing_confidence_std REAL
        );
        CREATE VIEW program_results_compat AS
            SELECT * FROM program_results;
        """
    )
    conn.close()


class _NotebookStub:
    def __init__(self, path: Path):
        self.db_path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row


class _AnalyticsStub(_WeightsMixin, _RoutingMixin):
    __slots__ = ("nb",)

    def __init__(self, path: Path):
        self.nb = _NotebookStub(path)


def test_duplicate_graph_counts_in_raw_routing_but_once_in_architecture_weights(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "dedup_boundary.sqlite3"
    _create_boundary_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.executemany(
        """
        INSERT INTO program_results
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "dup_fail",
                graph_json('{"templates_used":["template_a"]}'),
                "stale_a1",
                0.9,
                10.0,
                1,
                0,
                0,
                1.0,
                "routed",
                100.0,
                70.0,
                30.0,
                0.3,
                0.6,
                1.0,
                0.8,
                0.1,
            ),
            (
                "dup_pass",
                graph_json(
                    '{"templates_used":["template_a"],"lineage":{"parent":"x"}}'
                ),
                "stale_a2",
                0.5,
                7.0,
                1,
                1,
                1,
                2.0,
                "routed",
                100.0,
                72.0,
                28.0,
                0.28,
                0.62,
                1.0,
                0.82,
                0.1,
            ),
            (
                "unique_pass",
                graph_json('{"templates_used":["template_b"]}', middle_op="relu"),
                "stale_b1",
                0.7,
                8.0,
                1,
                1,
                1,
                3.0,
                "uniform",
                100.0,
                100.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ),
        ],
    )
    conn.commit()
    conn.close()

    analytics = _AnalyticsStub(db_path)

    routing = analytics.routing_mode_comparison()
    by_mode = {row["routing_mode"]: row for row in routing["by_mode"]}
    assert routing["total_programs"] == 3
    assert by_mode["routed"]["n_programs"] == 2

    deduped_rows = load_deduped_graph_training_rows(db_path)
    template_a_rows = [
        row
        for row in deduped_rows
        if '"template_a"' in str(row.get("graph_json") or "")
    ]
    assert len(template_a_rows) == 1
    assert template_a_rows[0]["n_rows"] == 2

    template_weights = analytics.compute_template_weights(min_used=1)
    assert template_weights == {"template_a": 1.0, "template_b": 1.0}
