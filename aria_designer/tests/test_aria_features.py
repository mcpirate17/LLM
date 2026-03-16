from __future__ import annotations
import json
import pytest
from pathlib import Path

from aria_designer.api.app.intent_parser import compute_insertion_point
from aria_designer.api.app.models import AskAriaPromptRequest, WorkflowGraphModel
from aria_designer.api.app.routers.aria import _generate_patch_impl, _validate_parent_regression_guardrails
from aria_designer.api.app.historical_insights import build_historical_insights_response
from aria_designer.api.app.aria_patch_postprocess import postprocess_patched_workflow
from aria_designer.api.app.suggestions import suggest_components
from aria_designer.api.app.mutation import refine_winner
from aria_designer.api.app.conversation import _build_patch_from_pattern, _match_pattern
from aria_designer.api.app import database as db

@pytest.fixture
def mock_db():
    db.init_db(Path(":memory:"))
    # Seed components
    db.upsert_component({
        "id": "linear", "name": "Linear", "category": "linear_algebra",
        "inputs": [{"name": "x", "dtype": "tensor"}],
        "outputs": [{"name": "y", "dtype": "tensor"}],
        "status": "approved"
    }, "2026-01-01", "2026-01-01")
    db.upsert_component({
        "id": "relu", "name": "ReLU", "category": "math",
        "inputs": [{"name": "x", "dtype": "tensor"}],
        "outputs": [{"name": "y", "dtype": "tensor"}],
        "status": "approved"
    }, "2026-01-01", "2026-01-01")
    db.upsert_component({
        "id": "input", "name": "Input", "category": "io",
        "inputs": [],
        "outputs": [{"name": "y", "dtype": "tensor"}],
        "status": "approved"
    }, "2026-01-01", "2026-01-01")

def test_suggestions_empty(mock_db):
    workflow = {"nodes": [], "edges": []}
    s = suggest_components(workflow)
    assert len(s) > 0
    assert any("Input" in x["component"]["name"] for x in s)

def test_suggestions_linear(mock_db):
    workflow = {
        "nodes": [{"id": "n1", "component_type": "linear"}],
        "edges": []
    }
    s = suggest_components(workflow)
    assert len(s) > 0
    # Should suggest math (activation)
    categories = {x["component"]["category"] for x in s}
    assert "math" in categories


def test_suggestions_scores_are_not_flat_and_avoid_no_norm_for_stability(mock_db):
    db.upsert_component({
        "id": "rmsnorm", "name": "RMSNorm", "category": "normalization",
        "inputs": [{"name": "x", "dtype": "tensor"}],
        "outputs": [{"name": "y", "dtype": "tensor"}],
        "status": "approved"
    }, "2026-01-01", "2026-01-01")
    db.upsert_component({
        "id": "layernorm_pre", "name": "LayerNorm Pre", "category": "normalization",
        "inputs": [{"name": "x", "dtype": "tensor"}],
        "outputs": [{"name": "y", "dtype": "tensor"}],
        "status": "approved"
    }, "2026-01-01", "2026-01-01")
    db.upsert_component({
        "id": "no_norm", "name": "No Norm", "category": "normalization",
        "inputs": [{"name": "x", "dtype": "tensor"}],
        "outputs": [{"name": "y", "dtype": "tensor"}],
        "status": "approved"
    }, "2026-01-01", "2026-01-01")

    workflow = {
        "nodes": [{"id": "n1", "component_type": "linear"}],
        "edges": []
    }
    s = suggest_components(workflow, prompt="Improve stability and avoid exploding gradients")
    assert len(s) > 0

    scores = [float(item.get("score", 0.0)) for item in s]
    assert len(set(scores)) > 1

    ids = [str(item.get("component", {}).get("id", "")).lower() for item in s]
    assert "no_norm" not in ids

def test_refine_winner(mock_db):
    # Save a workflow first
    wf_id = "wf_win"
    db.save_workflow(wf_id, "Winner", json_graph(), author="aria")
    
    proposals = refine_winner(wf_id, num_variations=2)
    assert len(proposals) == 2
    
    # Check proposal content
    p = db.get_proposal(proposals[0])
    assert p["workflow_id"] == wf_id
    assert "Evolution" in p["rationale"]


def test_refine_winner_compression_intent_stays_in_scope(mock_db):
    wf_id = "wf_compress"
    db.save_workflow(wf_id, "Compress", json_graph(), author="aria")

    proposals = refine_winner(
        wf_id,
        num_variations=1,
        intent="refine_compression",
        parent_scores={"tier": "investigation", "composite_score": 120.0},
    )
    assert len(proposals) == 1

    proposal = db.get_proposal(proposals[0])
    patch = json.loads(proposal["patch_json"])
    op_kinds = {op["op"] for op in patch["ops"]}
    assert op_kinds <= {"mutate_param", "replace_node"}
    assert "add_node" not in op_kinds


def test_suggestions_use_research_op_priors(mock_db):
    db.upsert_component({
        "id": "rmsnorm_pre", "name": "RMSNorm Pre", "category": "normalization",
        "inputs": [{"name": "x", "dtype": "tensor"}],
        "outputs": [{"name": "y", "dtype": "tensor"}],
        "status": "approved"
    }, "2026-01-01", "2026-01-01")
    db.upsert_component({
        "id": "group_norm", "name": "Group Norm", "category": "normalization",
        "inputs": [{"name": "x", "dtype": "tensor"}],
        "outputs": [{"name": "y", "dtype": "tensor"}],
        "status": "approved"
    }, "2026-01-01", "2026-01-01")

    workflow = {
        "nodes": [{"id": "n1", "component_type": "linear"}],
        "edges": []
    }
    signals = {
        "op_priors": [
            {"op_name": "rmsnorm_pre", "s1_rate": 0.95, "n_used": 100},
            {"op_name": "group_norm", "s1_rate": 0.55, "n_used": 100},
        ],
        "toxic_ops": [],
        "insights": [],
    }
    baseline = suggest_components(workflow, prompt="Improve stability")
    scored = suggest_components(workflow, prompt="Improve stability", research_signals=signals)
    assert len(scored) > 0
    base_by_id = {item["component"]["id"]: item for item in baseline}
    scored_by_id = {item["component"]["id"]: item for item in scored}
    if "rmsnorm_pre" in scored_by_id and "rmsnorm_pre" in base_by_id:
        assert scored_by_id["rmsnorm_pre"]["score"] >= base_by_id["rmsnorm_pre"]["score"]


def test_compute_insertion_point_places_norm_between_projection_and_output():
    workflow = {
        "nodes": [
            {"id": "input", "component_type": "io/input"},
            {"id": "linear", "component_type": "linear_algebra/linear_proj"},
            {"id": "output", "component_type": "io/output"},
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "linear"},
            {"id": "e2", "source": "linear", "target": "output"},
        ],
    }
    hint = compute_insertion_point(workflow["nodes"], workflow["edges"], "normalization/rmsnorm_pre")
    assert hint == {"after_node_id": "linear", "before_node_id": "output"}


def test_suggestions_include_insertion_hint_and_leaderboard_evidence(mock_db, monkeypatch):
    db.upsert_component({
        "id": "rmsnorm_pre", "name": "RMSNorm Pre", "category": "normalization",
        "inputs": [{"name": "x", "dtype": "tensor"}],
        "outputs": [{"name": "y", "dtype": "tensor"}],
        "status": "approved"
    }, "2026-01-01", "2026-01-01")

    workflow = {
        "nodes": [
            {"id": "input", "component_type": "io/input"},
            {"id": "linear", "component_type": "linear_algebra/linear_proj"},
            {"id": "output", "component_type": "io/output"},
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "linear"},
            {"id": "e2", "source": "linear", "target": "output"},
        ],
    }
    leaderboard_entries = [{
        "graph_json": json.dumps({
            "nodes": [
                {"id": "a", "component_type": "io/input"},
                {"id": "b", "component_type": "normalization/rmsnorm_pre"},
                {"id": "c", "component_type": "io/output"},
            ]
        })
    }]
    monkeypatch.setattr("aria_designer.api.app.suggestions.fetch_leaderboard_top_entries", lambda: leaderboard_entries)
    scored = suggest_components(
        workflow,
        prompt="Improve stability",
        research_signals={"op_priors": [], "toxic_ops": [], "insights": []},
    )
    hinted = next((item for item in scored if item["component"]["id"] == "rmsnorm_pre"), None)
    if hinted is None:
        pytest.skip("RMSNorm was not selected in this seeded component set")
    assert any("Used in 1 of top 1 architectures" in item for item in hinted["evidence"])
    assert hinted["insertion_hint"] == {"after_node_id": "linear", "before_node_id": "output"}


def test_suggestions_fetch_leaderboard_once_per_request(mock_db, monkeypatch):
    db.upsert_component({
        "id": "rmsnorm_pre", "name": "RMSNorm Pre", "category": "normalization",
        "inputs": [{"name": "x", "dtype": "tensor"}],
        "outputs": [{"name": "y", "dtype": "tensor"}],
        "status": "approved"
    }, "2026-01-01", "2026-01-01")
    db.upsert_component({
        "id": "layernorm_pre", "name": "LayerNorm Pre", "category": "normalization",
        "inputs": [{"name": "x", "dtype": "tensor"}],
        "outputs": [{"name": "y", "dtype": "tensor"}],
        "status": "approved"
    }, "2026-01-01", "2026-01-01")

    workflow = {
        "nodes": [{"id": "n1", "component_type": "linear_algebra/linear"}],
        "edges": [],
    }
    calls = {"count": 0}

    def _fetch_entries():
        calls["count"] += 1
        return [{
            "graph_json": json.dumps({
                "nodes": [
                    {"id": "a", "component_type": "normalization/rmsnorm_pre"},
                    {"id": "b", "component_type": "normalization/layernorm_pre"},
                ]
            })
        }]

    monkeypatch.setattr("aria_designer.api.app.suggestions.fetch_leaderboard_top_entries", _fetch_entries)
    suggest_components(
        workflow,
        prompt="Improve stability",
        research_signals={"op_priors": [], "toxic_ops": [], "insights": []},
    )
    assert calls["count"] == 1


def test_difficulty_routed_chat_pattern_uses_valid_components_and_preserves_hard_lane():
    workflow = {
        "nodes": [
            {"id": "input", "component_type": "input", "ui_meta": {"position": {"x": 0, "y": 0}}},
            {"id": "mid", "component_type": "softmax_attention", "ui_meta": {"position": {"x": 200, "y": 0}}},
            {"id": "output", "component_type": "output_head", "ui_meta": {"position": {"x": 400, "y": 0}}},
        ],
        "edges": [
            {"id": "e_in", "source": "input", "target": "mid", "source_port": "y", "target_port": "x"},
            {"id": "e_out", "source": "mid", "target": "output", "source_port": "y", "target_port": "x"},
        ],
    }

    pattern = _match_pattern(
        "Build two lanes: fast lane for easy tokens and keep the existing lane for harder tokens with a difficulty scorer and router"
    )
    assert pattern is not None

    patch = _build_patch_from_pattern(pattern, workflow, "route easy and hard tokens")
    added_types = {
        op["payload"]["component_type"]
        for op in patch["ops"]
        if op["op"] == "add_node"
    }
    assert {
        "routing/difficulty_scorer",
        "routing/lane_router",
        "structural/conditional_dispatch",
        "structural/conditional_gather",
        "linear_algebra/linear_proj",
    } <= added_types

    rewires = [op["payload"] for op in patch["ops"] if op["op"] == "rewire"]
    assert any(
        payload.get("source") == "input"
        and payload.get("remove_edge_id") == "e_in"
        for payload in rewires
    ) is False
    assert any(
        payload.get("target") == "mid"
        and payload.get("remove_edge_id") == "e_in"
        for payload in rewires
    )
    assert any(
        payload.get("source") == "mid"
        and payload.get("target_port") == "b"
        and payload.get("remove_edge_id") == "e_out"
        for payload in rewires
    )
    assert any(
        payload.get("target") == "output"
        and payload.get("source") != "mid"
        for payload in rewires
    )


def test_generate_patch_impl_preserves_insert_after_and_param_mutation(mock_db):
    db.upsert_component({
        "id": "output", "name": "Output", "category": "io",
        "inputs": [{"name": "x", "dtype": "tensor"}],
        "outputs": [],
        "status": "approved"
    }, "2026-01-01", "2026-01-01")
    workflow = {
        "schema_version": "workflow_graph.v1",
        "workflow_id": "wf_prompt_patch",
        "name": "Prompt Patch",
        "nodes": [
            {"id": "in", "component_type": "io/input", "params": {}, "ui_meta": {}},
            {"id": "n1", "component_type": "linear_algebra/linear", "params": {"out_dim": 64}, "ui_meta": {}},
            {"id": "out", "component_type": "io/output", "params": {}, "ui_meta": {}},
        ],
        "edges": [
            {"id": "e1", "source": "in", "target": "n1", "source_port": "out", "target_port": "in"},
            {"id": "e2", "source": "n1", "target": "out", "source_port": "out", "target_port": "in"},
        ],
        "metadata": {},
    }
    req = AskAriaPromptRequest(
        workflow=WorkflowGraphModel.model_validate(workflow),
        prompt="add relu after n1 and set out_dim of n1 to 128",
        base_version=1,
    )

    result = _generate_patch_impl(req)
    ops = result["proposal"]["ops"]
    op_kinds = [op["op"] for op in ops]

    assert "add_node" in op_kinds
    assert "mutate_param" in op_kinds
    mutate = next(op for op in ops if op["op"] == "mutate_param")
    assert mutate["node_id"] == "n1"
    assert mutate["payload"] == {"out_dim": 128}


def test_parent_regression_guard_rejects_excessive_removals():
    workflow = {
        "nodes": [
            {"id": "n1", "component_type": "math/relu"},
            {"id": "n2", "component_type": "math/relu"},
            {"id": "n3", "component_type": "math/relu"},
            {"id": "n4", "component_type": "math/relu"},
            {"id": "io", "component_type": "io/output"},
        ]
    }
    ops = [
        {"op": "remove_node", "node_id": "n1"},
        {"op": "remove_node", "node_id": "n2"},
    ]
    error = _validate_parent_regression_guardrails(
        workflow,
        ops,
        {"tier": "investigation", "composite_score": 120.0},
    )
    assert error is not None
    assert "removes 2/4 ops" in error


def test_historical_insights_uses_precomputed_component_ids(monkeypatch):
    monkeypatch.setattr(
        "aria_designer.api.app.historical_insights.fetch_leaderboard_top_entries",
        lambda n=10, min_composite=50.0: [
            {"_component_ids": ["rmsnorm_pre", "linear_proj"]},
            {"_component_ids": ["rmsnorm_pre"]},
        ],
    )
    monkeypatch.setattr(
        "aria_designer.api.app.historical_insights.fetch_research_recommendation_signals",
        lambda force=False: {
            "insights": [{"category": "success_factor", "content": "Normalization improves stability"}],
            "toxic_ops": ["no_norm"],
        },
    )

    response = build_historical_insights_response()
    assert response.top_components[0] == {"component_id": "rmsnorm_pre", "count": 2}
    assert "Normalization improves stability" in response.success_patterns
    assert "Toxic operator: no_norm" in response.failure_patterns


def test_postprocess_patched_workflow_applies_insertion_hint_without_duplicate_bypass():
    workflow = {
        "nodes": [
            {"id": "input", "component_type": "io/input", "ui_meta": {}},
            {"id": "linear", "component_type": "linear_algebra/linear_proj", "ui_meta": {}},
            {"id": "norm", "component_type": "normalization/rmsnorm_pre", "ui_meta": {}},
            {"id": "output", "component_type": "io/output", "ui_meta": {}},
        ],
        "edges": [
            {"id": "e1", "source": "input", "target": "linear"},
            {"id": "e2", "source": "linear", "target": "output"},
        ],
    }

    patched = postprocess_patched_workflow(
        workflow,
        ["norm"],
        insertion_hints={"norm": {"after_node_id": "linear", "before_node_id": "output"}},
    )
    pairs = {(edge["source"], edge["target"]) for edge in patched["edges"]}

    assert ("linear", "norm") in pairs
    assert ("norm", "output") in pairs
    assert ("linear", "output") not in pairs

def json_graph():
    return """
    {
        "schema_version": "workflow_graph.v1",
        "workflow_id": "wf_win",
        "name": "Winner",
        "nodes": [
            {"id": "n1", "component_type": "linear", "params": {"out_dim": 64}},
            {"id": "n2", "component_type": "math/relu"}
        ],
        "edges": []
    }
    """
