from __future__ import annotations

import json

import pytest

aria_scheduler = pytest.importorskip(
    "aria_scheduler",
    reason="aria_scheduler Rust module not available",
)


def test_native_graph_feature_payload_extracts_expected_fields():
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
