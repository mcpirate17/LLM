import json
import random
import sqlite3

from research.scientist.notebook import LabNotebook
from research.synthesis._template_helpers import record_routing_decision
from research.synthesis.graph import ComputationGraph
from research.synthesis.serializer import graph_to_json
from research.synthesis.templates import TEMPLATES
from research.tools.routing_decision_report import build_routing_decision_report


def _template_graph(template_name: str, seed: int = 42) -> ComputationGraph:
    graph = ComputationGraph(model_dim=128)
    graph.metadata["_active_template"] = template_name
    graph.metadata["_active_template_instance"] = 0
    graph.metadata["_active_template_slot_counter"] = 0
    input_id = graph.add_input()
    output_id = TEMPLATES[template_name](graph, input_id, random.Random(seed))
    graph.set_output(output_id)
    return graph


def test_routing_decision_invalidates_cached_graph_json() -> None:
    graph = ComputationGraph(model_dim=32)
    graph.add_input()

    cached_before = graph_to_json(graph)
    assert "routing_decisions" not in cached_before

    record_routing_decision(
        graph,
        template_name="unit_template",
        decision_key="gate_threshold",
        value=0.5,
        choices=(0.5,),
        source="static_config",
        context="unit",
    )

    payload = json.loads(graph_to_json(graph))
    decisions = payload["metadata"]["routing_decisions"]
    assert decisions[0]["decision_key"] == "gate_threshold"
    assert decisions[0]["value"] == 0.5


def test_routing_templates_emit_static_config_decisions() -> None:
    hybrid = _template_graph("hybrid_sparse_triplet_router")
    hybrid_keys = {
        entry["decision_key"] for entry in hybrid.metadata.get("routing_decisions", [])
    }
    assert {
        "gate_threshold",
        "confidence_threshold",
        "span_width",
        "lane_count",
    } <= hybrid_keys

    multilane = _template_graph("intelligent_multilane_router")
    multilane_keys = {
        entry["decision_key"]
        for entry in multilane.metadata.get("routing_decisions", [])
    }
    assert {
        "gate_threshold",
        "span2_span_width",
        "span2_lane_count",
        "span2_confidence_threshold",
        "span3_span_width",
        "span3_lane_count",
        "span3_confidence_threshold",
    } <= multilane_keys

    multiscale = _template_graph("multiscale_difficulty_router")
    multiscale_keys = {
        entry["decision_key"]
        for entry in multiscale.metadata.get("routing_decisions", [])
    }
    assert {
        "gate_threshold",
        "span2_span_width",
        "span3_span_width",
        "span4_span_width",
        "merge_medium_min_secondary_share",
        "merge_hard_min_secondary_share",
    } <= multiscale_keys


def test_routing_decisions_persist_through_notebook_graph_json(tmp_path) -> None:
    db_path = tmp_path / "lab_notebook.db"
    graph = _template_graph("hybrid_sparse_triplet_router")
    graph_json = graph_to_json(graph)

    nb = LabNotebook(db_path, use_native=False)
    try:
        result_id = nb.record_program_result(
            experiment_id="routing-decision-test",
            graph_fingerprint=graph.fingerprint(),
            graph_json=graph_json,
            stage0_passed=1,
            stage1_passed=0,
            loss_ratio=0.42,
            error_type="test_fixture",
            bypass_quality_gate=True,
        )
        nb.flush_writes(timeout=10.0)

        row = nb.conn.execute(
            "SELECT graph_json FROM program_results_compat WHERE result_id = ?",
            (result_id,),
        ).fetchone()
        assert row is not None
        stored = json.loads(row["graph_json"])
        assert stored["metadata"]["routing_decisions"]
    finally:
        nb.close()


def test_routing_decision_report_groups_outcomes(tmp_path) -> None:
    db_path = tmp_path / "runs.db"
    graph = {
        "metadata": {
            "routing_decisions": [
                {
                    "template_name": "unit_router",
                    "decision_key": "gate_threshold",
                    "value": 0.5,
                    "source": "static_config",
                }
            ]
        }
    }
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE program_results (
                graph_json TEXT,
                stage1_passed INTEGER,
                loss_ratio REAL,
                validation_loss_ratio REAL,
                ar_gate_score REAL,
                binding_intermediate_auc REAL,
                binding_screening_auc REAL,
                routing_utilization_entropy REAL,
                routing_drop_rate REAL,
                routing_savings_ratio REAL,
                routing_collapse_score REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO program_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (json.dumps(graph), 1, 0.4, 0.45, 0.7, 0.6, 0.55, 0.9, 0.1, 0.2, 0.0),
        )
        conn.commit()
    finally:
        conn.close()

    report = build_routing_decision_report(db_path)
    assert report["input_rows"] == 1
    assert report["decision_groups"] == 1
    row = report["records"][0]
    assert row["template_name"] == "unit_router"
    assert row["decision_key"] == "gate_threshold"
    assert row["n"] == 1
    assert row["pass_rate"] == 1.0
    assert row["mean_binding_intermediate_auc"] == 0.6
