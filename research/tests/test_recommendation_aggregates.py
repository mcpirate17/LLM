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


def _workflow_payload(
    workflow_id: str, parent_fingerprint: str | None, node_type: str
) -> dict:
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


def test_failure_blocklist_skips_audited_false_positive_pairs(nb):
    now = 1_700_000_000.0
    nb.conn.execute(
        """INSERT INTO failure_signatures
           (signature, n_failures, n_successes, error_types, last_updated)
           VALUES (?, ?, ?, ?, ?)""",
        ("layernorm->rope_rotate", 19, 0, "insufficient_learning", now),
    )
    nb.conn.execute(
        """INSERT INTO failure_signatures
           (signature, n_failures, n_successes, error_types, last_updated)
           VALUES (?, ?, ?, ?, ?)""",
        ("bad_op->worse_op", 19, 0, "insufficient_learning", now),
    )
    nb.conn.commit()

    blocklist = nb.get_failure_signature_blocklist(min_seen=5, max_fail_rate=0.85)
    assert "layernorm->rope_rotate" not in blocklist
    assert blocklist["bad_op->worse_op"] == 0.05
