"""Stress tests for Aria Designer API."""
import sys
import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Add api/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

import tempfile

@pytest.fixture
def client():
    from app.main import app
    from app import database as db
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_stress.db"
        db.init_db(db_path)
        from app.loader import scan_and_load
        scan_and_load()
        with TestClient(app) as c:
            yield c

def test_large_chain_validation(client):
    """Test validation of a 100-node linear chain."""
    nodes = [{"id": "in", "component_type": "io/input", "params": {}, "ui_meta": {}}]
    edges = []
    
    for i in range(100):
        nodes.append({
            "id": f"n{i}",
            "component_type": "math/relu",
            "params": {},
            "ui_meta": {}
        })
        src = "in" if i == 0 else f"n{i-1}"
        edges.append({
            "id": f"e{i}",
            "source": src,
            "source_port": "y",
            "target": f"n{i}",
            "target_port": "x"
        })
        
    workflow = {
        "workflow": {
            "schema_version": "workflow_graph.v1",
            "workflow_id": "stress_test",
            "name": "Stress Test",
            "nodes": nodes,
            "edges": edges
        }
    }
    
    import time
    t0 = time.perf_counter()
    r = client.post("/api/v1/workflows/validate", json=workflow)
    duration = time.perf_counter() - t0
    
    assert r.status_code == 200
    assert r.json()["valid"] is True
    print(f"Validated 100-node chain in {duration:.4f}s")

def test_disconnected_islands(client):
    """Test validation with multiple disconnected components."""
    workflow = {
        "workflow": {
            "schema_version": "workflow_graph.v1",
            "workflow_id": "islands",
            "name": "Islands",
            "nodes": [
                {"id": "a1", "component_type": "math/relu", "params": {}, "ui_meta": {}},
                {"id": "a2", "component_type": "math/relu", "params": {}, "ui_meta": {}},
                {"id": "b1", "component_type": "math/relu", "params": {}, "ui_meta": {}},
                {"id": "b2", "component_type": "math/relu", "params": {}, "ui_meta": {}},
            ],
            "edges": [
                {"id": "e1", "source": "a1", "source_port": "y", "target": "a2", "target_port": "x"},
                {"id": "e2", "source": "b1", "source_port": "y", "target": "b2", "target_port": "x"},
            ]
        }
    }
    r = client.post("/api/v1/workflows/validate", json=workflow)
    assert r.status_code == 200
    # Current validator might mark this valid if no cycles, 
    # but technically it has no graph_input. 
    # Let's check what the API returns.
def test_import_and_validate_survivor(client):
    """Test importing a survivor from research and re-validating it."""
    # 1. Get survivors
    r_list = client.get("/api/v1/import/survivors")
    assert r_list.status_code == 200
    survivors = r_list.json()["survivors"]
    if not survivors:
        pytest.skip("No survivors in notebook to test import")
        
    res_id = survivors[0]["result_id"]
    
    # 2. Import
    r_imp = client.post(f"/api/v1/import/survivors/{res_id}")
    assert r_imp.status_code == 200
    workflow = r_imp.json()
    
    # 3. Validate
    r_val = client.post("/api/v1/workflows/validate", json={"workflow": workflow})
    assert r_val.status_code == 200
    assert r_val.json()["valid"] is True
