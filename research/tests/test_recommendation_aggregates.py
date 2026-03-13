from __future__ import annotations

import json

import pytest

from research.scientist.notebook import LabNotebook

pytestmark = pytest.mark.unit


def _program_graph(ops: list[str]) -> str:
    nodes = {"0": {"op_name": "input", "input_ids": []}}
    for index, op_name in enumerate(ops, start=1):
        nodes[str(index)] = {"op_name": op_name, "input_ids": [str(index - 1)]}
    return json.dumps({"nodes": nodes})


def _workflow_payload(workflow_id: str, parent_fingerprint: str | None, node_type: str) -> dict:
    payload = {
        "workflow_id": workflow_id,
        "nodes": [{"id": "n1", "component_type": node_type}],
        "metadata": {},
    }
    if parent_fingerprint:
        payload["metadata"]["parent_fingerprint"] = parent_fingerprint
    return payload


@pytest.fixture
def nb(tmp_path):
    notebook = LabNotebook(str(tmp_path / "aggregates.db"))
    yield notebook
    notebook.close()


def test_op_pair_priors_and_failure_risks_require_positive_evidence(nb):
    exp_id = nb.start_experiment("synthesis", {})
    for idx in range(5):
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=f"top-{idx}",
            graph_json=_program_graph(["gelu", "linear_proj"]),
            stage1_passed=True,
            loss_ratio=0.2 + (idx * 0.01),
            novelty_score=0.8,
        )
        nb.upsert_leaderboard(
            result_id=rid,
            model_source="graph_synthesis",
            tier="investigation",
            screening_loss_ratio=0.2,
            screening_novelty=0.8,
            composite_score=120.0 + idx,
        )
    for idx in range(20):
        nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=f"fail-{idx}",
            graph_json=_program_graph(["gelu", "linear_proj"]),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=False,
            error_type="loss_diverged",
            loss_ratio=0.99,
        )
    nb.flush_writes()
    nb.recompute_failure_signatures()

    priors = nb.get_op_pair_priors(min_support=5, limit=10)
    assert priors[0]["signature"] == "gelu->linear_proj"
    assert priors[0]["support"] >= 25
    assert priors[0]["success_rate"] == 0.2

    risks = nb.get_failure_risk_signatures(limit=10)["failure_risk_signatures"]
    assert risks[0]["signature"] == "gelu->linear_proj"
    assert risks[0]["weight"] == 0.05


def test_fingerprint_buckets_and_lineage_successors_are_deterministic(nb):
    exp_id = nb.start_experiment("synthesis", {})
    parent_rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-parent",
        graph_json=_program_graph(["softmax_attention", "linear_proj"]),
        stage1_passed=True,
        loss_ratio=0.4,
        novelty_score=0.5,
        fp_cka_vs_transformer=0.9,
        graph_category_histogram=json.dumps({"attention": 3}),
    )
    child_rid = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint="fp-child",
        graph_json=_program_graph(["state_space", "linear_proj"]),
        stage1_passed=True,
        loss_ratio=0.2,
        novelty_score=0.7,
        fp_cka_vs_ssm=0.8,
        graph_category_histogram=json.dumps({"mixing": 4}),
    )
    nb.upsert_leaderboard(parent_rid, "graph_synthesis", composite_score=90.0, screening_loss_ratio=0.4, screening_novelty=0.5)
    nb.upsert_leaderboard(child_rid, "graph_synthesis", composite_score=130.0, screening_loss_ratio=0.2, screening_novelty=0.7)
    nb.save_designer_run_lineage(
        run_id="run-parent",
        workflow_id="wf-lineage",
        workflow_version=1,
        graph_fingerprint="fp-parent",
        payload=_workflow_payload("wf-lineage", None, "mixing/softmax_attention"),
    )
    nb.save_designer_run_lineage(
        run_id="run-child",
        workflow_id="wf-lineage",
        workflow_version=2,
        graph_fingerprint="fp-child",
        payload=_workflow_payload("wf-lineage", "fp-parent", "mixing/state_space"),
    )

    buckets = {row["bucket"]: row for row in nb.get_fingerprint_buckets(limit=10)}
    assert buckets["attention-heavy"]["top_ops"][0]["op_name"] == "linear_proj"
    assert buckets["mixing-heavy"]["n_graphs"] == 1

    lineage = nb.get_lineage_successor_stats(limit=10)
    assert lineage[0]["parent_fingerprint"] == "fp-parent"
    assert lineage[0]["child_fingerprint"] == "fp-child"
    assert lineage[0]["improved_rate"] == 1.0
