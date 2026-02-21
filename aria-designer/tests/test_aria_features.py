from __future__ import annotations
import sys
import os
import pytest
from pathlib import Path

# Add api/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

from app.suggestions import suggest_components
from app.mutation import refine_winner
from app import database as db

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
