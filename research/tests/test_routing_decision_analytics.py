"""Tests for the routing-decision analytics layer.

Verifies the join between routing knobs (recorded by Move #2 on graph
metadata) and program outcomes (loss, pass, ar_gate, binding) flows
correctly through json_extract on graph_json.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from research.meta_analysis.routing_decision_analytics import (
    RoutingDecisionRow,
    iter_routing_decision_outcomes,
    summarize_routing_decisions,
)


def _build_runs_db_with_routing(path: Path, programs: list[dict]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE program_results (
            result_id TEXT PRIMARY KEY,
            graph_json TEXT,
            stage1_passed INTEGER,
            loss_ratio REAL,
            validation_loss_ratio REAL,
            ar_gate_score REAL,
            binding_intermediate_auc REAL,
            binding_screening_composite REAL
        )
        """
    )
    for p in programs:
        conn.execute(
            """
            INSERT INTO program_results (
                result_id, graph_json, stage1_passed, loss_ratio,
                validation_loss_ratio, ar_gate_score,
                binding_intermediate_auc, binding_screening_composite
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                p["result_id"],
                p["graph_json"],
                p.get("stage1_passed"),
                p.get("loss_ratio"),
                p.get("validation_loss_ratio"),
                p.get("ar_gate_score"),
                p.get("binding_intermediate_auc"),
                p.get("binding_screening_composite"),
            ),
        )
    conn.commit()
    conn.close()


def _make_graph_json(decisions: list[dict]) -> str:
    return json.dumps({"metadata": {"routing_decisions": decisions}})


def test_iter_emits_one_row_per_decision(tmp_path: Path):
    runs = tmp_path / "runs.db"
    _build_runs_db_with_routing(
        runs,
        programs=[
            {
                "result_id": "r1",
                "graph_json": _make_graph_json(
                    [
                        {
                            "template_name": "hybrid_sparse_triplet_router",
                            "decision_key": "gate_threshold",
                            "value": 0.5,
                            "source": "rng_choice",
                        },
                        {
                            "template_name": "hybrid_sparse_triplet_router",
                            "decision_key": "lane_id",
                            "value": 1,
                            "source": "rng_randrange",
                        },
                    ]
                ),
                "stage1_passed": 1,
                "ar_gate_score": 0.42,
            },
        ],
    )
    rows = list(iter_routing_decision_outcomes(runs))
    assert len(rows) == 2
    assert {r.decision_key for r in rows} == {"gate_threshold", "lane_id"}
    gate = next(r for r in rows if r.decision_key == "gate_threshold")
    assert gate.value == 0.5
    assert gate.outcomes["stage1_passed"] == 1
    assert gate.outcomes["ar_gate_score"] == 0.42


def test_iter_prefers_program_results_compat_when_available(tmp_path: Path):
    runs = tmp_path / "runs.db"
    _build_runs_db_with_routing(
        runs,
        programs=[
            {
                "result_id": "r1",
                "graph_json": _make_graph_json(
                    [
                        {
                            "template_name": "router",
                            "decision_key": "gate_threshold",
                            "value": 0.5,
                            "source": "rng_choice",
                        }
                    ]
                ),
                "stage1_passed": 0,
                "ar_gate_score": 0.1,
            },
        ],
    )
    conn = sqlite3.connect(runs)
    conn.execute(
        """
        CREATE VIEW program_results_compat AS
        SELECT result_id, graph_json, 1 AS stage1_passed, loss_ratio,
               validation_loss_ratio, 0.9 AS ar_gate_score,
               binding_intermediate_auc, binding_screening_composite
        FROM program_results
        """
    )
    conn.close()

    rows = list(iter_routing_decision_outcomes(runs))
    assert len(rows) == 1
    assert rows[0].outcomes["stage1_passed"] == 1
    assert rows[0].outcomes["ar_gate_score"] == 0.9


def test_iter_skips_programs_without_routing_decisions(tmp_path: Path):
    runs = tmp_path / "runs.db"
    _build_runs_db_with_routing(
        runs,
        programs=[
            {
                "result_id": "r1",
                "graph_json": _make_graph_json(
                    [
                        {
                            "template_name": "x",
                            "decision_key": "k",
                            "value": 1,
                            "source": "rng_choice",
                        }
                    ]
                ),
            },
            {
                "result_id": "r2",
                "graph_json": json.dumps({"metadata": {"template_slot_usage": []}}),
            },
            {
                "result_id": "r3",
                "graph_json": "not even json",
            },
        ],
    )
    rows = list(iter_routing_decision_outcomes(runs))
    assert len(rows) == 1


def test_summarize_groups_by_template_decision_value(tmp_path: Path):
    rows = [
        RoutingDecisionRow(
            template_name="t1",
            decision_key="gate_threshold",
            value=0.5,
            source="rng_choice",
            outcomes={"stage1_passed": 1, "ar_gate_score": 0.4},
        ),
        RoutingDecisionRow(
            template_name="t1",
            decision_key="gate_threshold",
            value=0.5,
            source="rng_choice",
            outcomes={"stage1_passed": 0, "ar_gate_score": 0.2},
        ),
        RoutingDecisionRow(
            template_name="t1",
            decision_key="gate_threshold",
            value=0.6,
            source="rng_choice",
            outcomes={"stage1_passed": 1, "ar_gate_score": None},
        ),
    ]
    summary = summarize_routing_decisions(rows)
    by_value = {r["value"]: r for r in summary}

    assert by_value[0.5]["n"] == 2
    assert by_value[0.5]["pass_rate"] == 0.5
    assert by_value[0.5]["mean_ar_gate_score"] == 0.30000000000000004
    assert by_value[0.5]["n_ar_gate_score"] == 2

    assert by_value[0.6]["n"] == 1
    assert by_value[0.6]["pass_rate"] == 1.0
    assert by_value[0.6]["mean_ar_gate_score"] is None
    assert by_value[0.6]["n_ar_gate_score"] == 0


def test_template_filter_narrows_iteration(tmp_path: Path):
    runs = tmp_path / "runs.db"
    _build_runs_db_with_routing(
        runs,
        programs=[
            {
                "result_id": "r1",
                "graph_json": _make_graph_json(
                    [
                        {
                            "template_name": "A",
                            "decision_key": "k",
                            "value": 1,
                            "source": "rng_choice",
                        },
                        {
                            "template_name": "B",
                            "decision_key": "k",
                            "value": 2,
                            "source": "rng_choice",
                        },
                    ]
                ),
            }
        ],
    )
    rows_a = list(iter_routing_decision_outcomes(runs, template_filter="A"))
    assert [r.template_name for r in rows_a] == ["A"]
