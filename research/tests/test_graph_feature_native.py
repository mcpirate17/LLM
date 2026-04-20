from __future__ import annotations

import json

import pytest


def test_native_graph_feature_payload_extracts_expected_fields():
    from research.scientist.native.core import _try_import_rust_scheduler

    aria_scheduler = _try_import_rust_scheduler()
    if aria_scheduler is None:
        pytest.skip("aria_scheduler Rust module not available")
    if not hasattr(aria_scheduler, "extract_graph_feature_payload"):
        pytest.skip("graph feature payload extraction not exported by aria_scheduler")

    graph_json = json.dumps(
        {
            "nodes": {
                "input_0": {"op_name": "input", "input_ids": []},
                "n0": {"op_name": "linear_proj", "input_ids": ["input_0"]},
                "n1": {"op_name": "moe_topk", "input_ids": ["n0"]},
                "n2": {"op_name": "swiglu_mlp", "input_ids": ["n1"]},
            },
            "metadata": {
                "template": "moe_router",
                "templates_used": ["moe_router", "dense_adapter"],
                "motifs_used": ["router_chain"],
                "template_slot_usage": [{"slot": "router", "template": "moe_router"}],
            },
        }
    )

    native_payload = aria_scheduler.extract_graph_feature_payload(graph_json)

    assert native_payload[0] == "moe_router"
    assert tuple(native_payload[1]) == (
        "linear_proj",
        "moe_topk",
        "swiglu_mlp",
    )
    assert tuple(native_payload[2]) == (
        "linear_proj->moe_topk",
        "moe_topk->swiglu_mlp",
    )
    assert native_payload[3] == '["moe_router","dense_adapter"]'
    assert native_payload[4] == '["router_chain"]'
    assert native_payload[5] == '[{"slot":"router","template":"moe_router"}]'


def test_native_graph_segments_extracts_expected_counts():
    from research.scientist.native.core import _try_import_rust_scheduler

    aria_scheduler = _try_import_rust_scheduler()
    if aria_scheduler is None:
        pytest.skip("aria_scheduler Rust module not available")
    if not hasattr(aria_scheduler, "extract_graph_segments_native"):
        pytest.skip("graph segment extraction not exported by aria_scheduler")

    graph_json = json.dumps(
        {
            "nodes": {
                "0": {"id": 0, "op_name": "input", "input_ids": []},
                "1": {"id": 1, "op_name": "layernorm", "input_ids": [0]},
                "2": {"id": 2, "op_name": "gelu", "input_ids": [1]},
                "3": {"id": 3, "op_name": "linear_proj", "input_ids": [2]},
                "4": {"id": 4, "op_name": "add", "input_ids": [3]},
            }
        }
    )

    raw = aria_scheduler.extract_graph_segments_native(graph_json, 3, 6)
    payload = json.loads(raw)

    assert payload == {
        "seg_p3:gelu>linear_proj>add": 1,
        "seg_p3:layernorm>gelu>linear_proj": 1,
        "seg_p4:layernorm>gelu>linear_proj>add": 1,
    }


def test_native_graph_provenance_extracts_ops_and_upstream_source():
    from research.scientist.native.core import _try_import_rust_scheduler

    aria_scheduler = _try_import_rust_scheduler()
    if aria_scheduler is None:
        pytest.skip("aria_scheduler Rust module not available")
    if not hasattr(aria_scheduler, "analyze_graph_provenance_native"):
        pytest.skip("graph provenance analysis not exported by aria_scheduler")

    graph_json = json.dumps(
        {
            "nodes": {
                "0": {"id": 0, "op_name": "input", "input_ids": []},
                "1": {"id": 1, "op_name": "layernorm", "input_ids": [0]},
                "2": {"id": 2, "op_name": "identity", "input_ids": [1]},
                "3": {"id": 3, "op_name": "hybrid_sparse_router", "input_ids": [2]},
                "4": {"id": 4, "op_name": "add", "input_ids": [3]},
            }
        }
    )

    raw = aria_scheduler.analyze_graph_provenance_native(
        graph_json,
        ["add", "identity", "layernorm", "linear_proj", "rmsnorm"],
        "add",
    )
    payload = json.loads(raw)

    assert payload["op_names"] == [
        "layernorm",
        "identity",
        "hybrid_sparse_router",
        "add",
    ]
    assert payload["source_op"] == "hybrid_sparse_router"


def test_native_graph_structure_features_extracts_topology_and_aliases():
    from research.scientist.native.core import _try_import_rust_scheduler

    aria_scheduler = _try_import_rust_scheduler()
    if aria_scheduler is None:
        pytest.skip("aria_scheduler Rust module not available")
    if not hasattr(aria_scheduler, "extract_graph_structure_features_native"):
        pytest.skip("graph structure extraction not exported by aria_scheduler")

    graph_json = json.dumps(
        {
            "model_dim": 64,
            "nodes": {
                "0": {"id": 0, "op_name": "input", "input_ids": []},
                "1": {"id": 1, "op_name": "route_lanes", "input_ids": [0]},
                "2": {"id": 2, "op_name": "rope_rotate", "input_ids": [1]},
                "3": {"id": 3, "op_name": "add", "input_ids": [0, 2]},
            },
            "metadata": {"templates_used": ["routing_block", "tail_block"]},
        }
    )

    raw = aria_scheduler.extract_graph_structure_features_native(graph_json)
    payload = json.loads(raw)

    assert payload["op_names"] == ["gated_lane_blend", "rope_rotate", "add"]
    assert payload["n_nodes"] == 4.0
    assert payload["n_edges"] == 4.0
    assert payload["n_ops"] == 3.0
    assert payload["depth"] == 3.0
    assert payload["width"] == 2.0
    assert payload["n_unique_ops"] == 3.0
    assert payload["n_skip_connections"] == 1.0
    assert payload["edge_density"] == 1.0
    assert payload["n_templates_used"] == 2.0
    assert payload["model_dim"] == 64.0
