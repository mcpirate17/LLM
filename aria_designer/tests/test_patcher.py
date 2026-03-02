"""Tests for the Aria patch engine."""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.app.patcher import apply_patch_ops, PatchError


def _base_workflow():
    return {
        "schema_version": "workflow_graph.v1",
        "workflow_id": "test_wf",
        "name": "Test Workflow",
        "nodes": [
            {"id": "n0", "component_type": "graph_input", "params": {}},
            {"id": "n1", "component_type": "linear_proj", "params": {"out_dim": 256}},
            {"id": "n2", "component_type": "relu", "params": {}},
            {"id": "n3", "component_type": "graph_output", "params": {}},
        ],
        "edges": [
            {"id": "e0", "source": "n0", "target": "n1", "source_port": "out", "target_port": "in"},
            {"id": "e1", "source": "n1", "target": "n2", "source_port": "out", "target_port": "in"},
            {"id": "e2", "source": "n2", "target": "n3", "source_port": "out", "target_port": "in"},
        ],
    }


# ── add_node ─────────────────────────────────────────────────────────

def test_add_node_basic():
    wf = _base_workflow()
    result = apply_patch_ops(wf, [
        {
            "op": "add_node",
            "payload": {
                "id": "n_new",
                "component_type": "gelu",
                "params": {},
            }
        }
    ])
    node_ids = {n["id"] for n in result["nodes"]}
    assert "n_new" in node_ids
    assert len(result["nodes"]) == 5


def test_add_node_with_edges():
    wf = _base_workflow()
    result = apply_patch_ops(wf, [
        {
            "op": "add_node",
            "payload": {
                "id": "n_new",
                "component_type": "rmsnorm",
                "edges": [
                    {"id": "e_new", "source": "n1", "target": "n_new"},
                ],
            }
        }
    ])
    edge_ids = {e["id"] for e in result["edges"]}
    assert "e_new" in edge_ids


def test_add_node_duplicate_raises():
    wf = _base_workflow()
    with pytest.raises(PatchError, match="already exists"):
        apply_patch_ops(wf, [
            {"op": "add_node", "payload": {"id": "n1", "component_type": "gelu"}}
        ])


def test_add_node_missing_type_raises():
    wf = _base_workflow()
    with pytest.raises(PatchError, match="component_type"):
        apply_patch_ops(wf, [
            {"op": "add_node", "payload": {"id": "n_new"}}
        ])


# ── remove_node ──────────────────────────────────────────────────────

def test_remove_node_basic():
    wf = _base_workflow()
    result = apply_patch_ops(wf, [
        {"op": "remove_node", "node_id": "n2"}
    ])
    node_ids = {n["id"] for n in result["nodes"]}
    assert "n2" not in node_ids
    # Edges connected to n2 should be removed
    for e in result["edges"]:
        assert e["source"] != "n2" and e["target"] != "n2"


def test_remove_node_not_found():
    wf = _base_workflow()
    with pytest.raises(PatchError, match="not found"):
        apply_patch_ops(wf, [
            {"op": "remove_node", "node_id": "nonexistent"}
        ])


# ── replace_node ─────────────────────────────────────────────────────

def test_replace_node_type():
    wf = _base_workflow()
    result = apply_patch_ops(wf, [
        {
            "op": "replace_node",
            "node_id": "n2",
            "payload": {"component_type": "gelu"},
        }
    ])
    node_map = {n["id"]: n for n in result["nodes"]}
    assert node_map["n2"]["component_type"] == "gelu"


def test_replace_node_with_params():
    wf = _base_workflow()
    result = apply_patch_ops(wf, [
        {
            "op": "replace_node",
            "node_id": "n1",
            "payload": {"component_type": "linear_proj_down", "params": {"out_dim": 128}},
        }
    ])
    node_map = {n["id"]: n for n in result["nodes"]}
    assert node_map["n1"]["component_type"] == "linear_proj_down"
    assert node_map["n1"]["params"]["out_dim"] == 128


def test_replace_node_not_found():
    wf = _base_workflow()
    with pytest.raises(PatchError, match="not found"):
        apply_patch_ops(wf, [
            {"op": "replace_node", "node_id": "missing", "payload": {"component_type": "gelu"}}
        ])


# ── rewire ───────────────────────────────────────────────────────────

def test_rewire_add_edge():
    wf = _base_workflow()
    result = apply_patch_ops(wf, [
        {
            "op": "rewire",
            "edge_id": "e_skip",
            "payload": {"action": "add", "source": "n0", "target": "n2"},
        }
    ])
    edge_ids = {e["id"] for e in result["edges"]}
    assert "e_skip" in edge_ids


def test_rewire_remove_edge():
    wf = _base_workflow()
    result = apply_patch_ops(wf, [
        {"op": "rewire", "edge_id": "e1", "payload": {"action": "remove"}}
    ])
    edge_ids = {e["id"] for e in result["edges"]}
    assert "e1" not in edge_ids
    assert len(result["edges"]) == 2


def test_rewire_modify_edge():
    wf = _base_workflow()
    result = apply_patch_ops(wf, [
        {"op": "rewire", "edge_id": "e1", "payload": {"action": "modify", "source": "n0"}}
    ])
    edge_map = {e["id"]: e for e in result["edges"]}
    assert edge_map["e1"]["source"] == "n0"


def test_rewire_invalid_source():
    wf = _base_workflow()
    with pytest.raises(PatchError, match="not found"):
        apply_patch_ops(wf, [
            {"op": "rewire", "edge_id": "e_new", "payload": {"action": "add", "source": "ghost", "target": "n1"}}
        ])


# ── mutate_param ─────────────────────────────────────────────────────

def test_mutate_param_set():
    wf = _base_workflow()
    result = apply_patch_ops(wf, [
        {"op": "mutate_param", "node_id": "n1", "payload": {"out_dim": 512}}
    ])
    node_map = {n["id"]: n for n in result["nodes"]}
    assert node_map["n1"]["params"]["out_dim"] == 512


def test_mutate_param_delete():
    wf = _base_workflow()
    result = apply_patch_ops(wf, [
        {"op": "mutate_param", "node_id": "n1", "payload": {"out_dim": None}}
    ])
    node_map = {n["id"]: n for n in result["nodes"]}
    assert "out_dim" not in node_map["n1"]["params"]


def test_mutate_param_node_not_found():
    wf = _base_workflow()
    with pytest.raises(PatchError, match="not found"):
        apply_patch_ops(wf, [
            {"op": "mutate_param", "node_id": "ghost", "payload": {"x": 1}}
        ])


# ── Multi-op patches ────────────────────────────────────────────────

def test_multi_op_patch():
    """Apply multiple ops in sequence."""
    wf = _base_workflow()
    result = apply_patch_ops(wf, [
        # Add a normalization node
        {"op": "add_node", "payload": {"id": "norm", "component_type": "rmsnorm"}},
        # Wire it between n1 and n2
        {"op": "rewire", "edge_id": "e1", "payload": {"action": "modify", "target": "norm"}},
        {"op": "rewire", "edge_id": "e_norm_out", "payload": {"action": "add", "source": "norm", "target": "n2"}},
    ])
    node_ids = {n["id"] for n in result["nodes"]}
    assert "norm" in node_ids
    edge_map = {e["id"]: e for e in result["edges"]}
    assert edge_map["e1"]["target"] == "norm"
    assert edge_map["e_norm_out"]["source"] == "norm"


def test_immutability():
    """Original workflow should not be mutated."""
    wf = _base_workflow()
    original_node_count = len(wf["nodes"])
    apply_patch_ops(wf, [
        {"op": "add_node", "payload": {"id": "new", "component_type": "gelu"}}
    ])
    assert len(wf["nodes"]) == original_node_count


def test_unknown_op_raises():
    wf = _base_workflow()
    with pytest.raises(PatchError, match="Unknown operation"):
        apply_patch_ops(wf, [{"op": "teleport_node", "payload": {}}])
