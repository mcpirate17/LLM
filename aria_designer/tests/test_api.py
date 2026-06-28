"""Integration tests for Aria Designer API."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "components" in data


def test_list_components(client):
    r = client.get("/api/v1/components")
    assert r.status_code == 200
    data = r.json()
    assert len(data) > 50  # Should have 135+ components


def test_list_components_by_category(client):
    r = client.get("/api/v1/components?category=math")
    assert r.status_code == 200
    data = r.json()
    assert len(data) > 0
    assert all(c["category"] == "math" for c in data)


def test_get_component(client):
    r = client.get("/api/v1/components/relu")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == "relu"
    assert data["category"] == "math"
    assert len(data["outputs"]) >= 1


def test_get_hybrid_sparse_router_component_exposes_slots_and_templates(client):
    r = client.get("/api/v1/components/hybrid_sparse_router")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == "hybrid_sparse_router"
    assert {slot["name"] for slot in data.get("slots", [])} == {
        "pre_router",
        "default_path",
        "easy_router",
        "medium_router",
        "routed_lane",
        "sparse_spans",
        "difficulty_signal",
        "compression_router",
        "hard_router",
        "token_merge",
        "post_merge",
    }
    template_ids = {template["id"] for template in data.get("templates", [])}
    assert "hybrid_sparse_triplet_router" in template_ids
    assert "multiscale_rich_lane_router" in template_ids
    assert "intelligent_multilane_router" in template_ids


def test_get_component_not_found(client):
    r = client.get("/api/v1/components/nonexistent_xyz")
    assert r.status_code == 404


def test_get_component_execution_capability(client):
    r = client.get("/api/v1/components/relu/execution-capability")
    assert r.status_code == 200
    data = r.json()
    assert data["component_id"] == "relu"
    assert "bridge" in data
    assert data["bridge"]["bridge_supported"] is True
    assert data["bridge"]["primitive_name"] == "relu"
    assert data["bridge"]["semantic_fidelity"] == "exact"
    assert data["has_semantic_warnings"] is False


@pytest.mark.skip(
    reason="sequential and u_net component dirs were removed in prior cleanup"
)
def test_get_component_execution_capability_unmapped(client):
    r = client.get("/api/v1/components/sequential/execution-capability")
    assert r.status_code == 200
    data = r.json()
    assert data["component_id"] == "sequential"
    assert data["bridge"]["bridge_supported"] is True
    assert data["bridge"]["execution_class"] == "composite"
    r_block = client.get("/api/v1/components/u_net/execution-capability")
    assert r_block.status_code == 200
    d_block = r_block.json()
    assert d_block["bridge"]["bridge_supported"] is True
    assert "template lowering" in d_block["bridge"]["reason"].lower()


def test_get_component_execution_capability_routing_passthrough(client):
    r = client.get("/api/v1/components/adjacent_token_merge/execution-capability")
    assert r.status_code == 200
    data = r.json()
    assert data["bridge"]["bridge_supported"] is True
    assert data["bridge"]["primitive_name"] == "adjacent_token_merge"
    r2 = client.get("/api/v1/components/cheap_verify_blend/execution-capability")
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["bridge"]["bridge_supported"] is True
    assert data2["bridge"]["primitive_name"] == "cheap_verify_blend"
    r3 = client.get("/api/v1/components/random_data_source/execution-capability")
    assert r3.status_code == 200
    data3 = r3.json()
    assert data3["bridge"]["bridge_supported"] is True
    assert data3["bridge"]["primitive_name"] is None
    r4 = client.get("/api/v1/components/loop/execution-capability")
    assert r4.status_code == 200
    data4 = r4.json()
    assert data4["bridge"]["bridge_supported"] is True


def test_get_component_execution_capability_data_plane(client):
    r = client.get("/api/v1/components/random_data_source/execution-capability")
    assert r.status_code == 200
    data = r.json()
    assert data["bridge"]["bridge_supported"] is True
    assert data["bridge"]["primitive_name"] is None

    r2 = client.get("/api/v1/components/filter/execution-capability")
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["bridge"]["bridge_supported"] is True
    assert data2["bridge"]["primitive_name"] is None

    r3 = client.get("/api/v1/components/csv_reader/execution-capability")
    assert r3.status_code == 200
    data3 = r3.json()
    assert data3["bridge"]["bridge_supported"] is True
    assert data3["bridge"]["primitive_name"] is None

    r4 = client.get("/api/v1/components/split_train_val_test/execution-capability")
    assert r4.status_code == 200
    data4 = r4.json()
    assert data4["bridge"]["bridge_supported"] is True
    assert data4["bridge"]["primitive_name"] is None

    r5 = client.get("/api/v1/components/select_columns/execution-capability")
    assert r5.status_code == 200
    data5 = r5.json()
    assert data5["bridge"]["bridge_supported"] is True
    assert data5["bridge"]["primitive_name"] is None


def test_compile_workflow_reports_semantic_warnings(client):
    workflow = {
        "workflow": {
            "schema_version": "workflow_graph.v1",
            "workflow_id": "wf_passthrough_compile",
            "name": "Passthrough Compile Warning",
            "nodes": [
                {"id": "n_in", "component_type": "input", "params": {}, "ui_meta": {}},
                {
                    "id": "n_seq",
                    "component_type": "blocks/sequential",
                    "params": {},
                    "ui_meta": {},
                },
                {"id": "n_relu", "component_type": "relu", "params": {}, "ui_meta": {}},
                {
                    "id": "n_out",
                    "component_type": "output_head",
                    "params": {},
                    "ui_meta": {},
                },
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "n_in",
                    "source_port": "y",
                    "target": "n_seq",
                    "target_port": "x",
                },
                {
                    "id": "e2",
                    "source": "n_seq",
                    "source_port": "y",
                    "target": "n_relu",
                    "target_port": "x",
                },
                {
                    "id": "e3",
                    "source": "n_relu",
                    "source_port": "y",
                    "target": "n_out",
                    "target_port": "x",
                },
            ],
        }
    }
    r = client.post("/api/v1/workflows/compile", json=workflow)
    assert r.status_code == 200
    data = r.json()
    assert "semantic_warnings" in data
    assert "semantic_warning_count" in data
    # Passthrough components have approximate fidelity
    assert isinstance(data["semantic_warning_count"], int)


def test_compile_workflow_returns_structured_error_details(client):
    workflow = {
        "workflow": {
            "schema_version": "workflow_graph.v1",
            "workflow_id": "wf_compile_error_details",
            "name": "Compile Error Details",
            "nodes": [
                {"id": "n_in", "component_type": "input", "params": {}, "ui_meta": {}},
                {
                    "id": "n_bad",
                    "component_type": "math/definitely_missing_component",
                    "params": {},
                    "ui_meta": {},
                },
                {
                    "id": "n_out",
                    "component_type": "output_head",
                    "params": {},
                    "ui_meta": {},
                },
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "n_in",
                    "source_port": "y",
                    "target": "n_bad",
                    "target_port": "x",
                },
                {
                    "id": "e2",
                    "source": "n_bad",
                    "source_port": "y",
                    "target": "n_out",
                    "target_port": "x",
                },
            ],
        }
    }
    r = client.post("/api/v1/workflows/compile", json=workflow)
    assert r.status_code == 200
    data = r.json()
    assert data["compiled"] is False
    assert "error_details" in data
    assert data["error_details"]["stage"] == "compilation"
    assert data["error_details"]["error_type"]


def test_eval_run_store_persists_to_database():
    from aria_designer.api.app import database as db
    from aria_designer.api.app.shared_api import _store_run

    with tempfile.TemporaryDirectory() as tmpdir:
        db.init_db(Path(tmpdir) / "test_runs.db")
        _store_run(
            "eval_persisted_1",
            {
                "run_id": "eval_persisted_1",
                "workflow_id": "wf_persisted",
                "status": "failed_sandbox",
                "created_at": "2026-03-29T12:00:00Z",
                "stages": {
                    "sandbox": {
                        "status": "done",
                        "elapsed_ms": 12.5,
                        "metrics": {"stability_score": 0.1},
                    }
                },
                "result": {"status": "failed_sandbox", "total_time_ms": 12.5},
                "error_details": {
                    "stage": "sandbox",
                    "error_type": "failed_sandbox",
                    "error_message": "nan detected",
                },
                "semantic_warnings": [
                    {"code": "approximate_lowering", "message": "passthrough lowering"}
                ],
            },
        )
        row = db.get_workflow_run("eval_persisted_1")
        assert row is not None
        assert row["workflow_id"] == "wf_persisted"
        assert row["stages"]["sandbox"]["metrics"]["stability_score"] == 0.1
        assert row["error_details"]["stage"] == "sandbox"


def test_bridge_gap_report(client):
    r = client.get("/api/v1/integration/bridge-gap-report")
    assert r.status_code == 200
    data = r.json()
    assert "total_components" in data
    assert "unsupported_components" in data
    assert "gaps" in data
    assert data["unsupported_components"] >= 0
    assert isinstance(data["gaps"], list)
    unsupported_ids = {g["component_id"] for g in data["gaps"]}
    assert "random_data_source" not in unsupported_ids
    assert "dataset_map" not in unsupported_ids
    assert "dataset_filter" not in unsupported_ids
    assert "split_train_val_test" not in unsupported_ids
    assert "select_columns" not in unsupported_ids
    assert "csv_reader" not in unsupported_ids


def test_validate_config_split_train_val_test(client):
    invalid = {
        "config": {
            "train_ratio": 0.8,
            "val_ratio": 0.3,
            "test_ratio": 0.1,
            "schema_validation": "strict",
            "expected_feature_dim": 4,
        }
    }
    r_bad = client.post(
        "/api/v1/components/split_train_val_test/validate-config", json=invalid
    )
    assert r_bad.status_code == 200
    bad_data = r_bad.json()
    assert bad_data["valid"] is False
    assert any("must equal 1.0" in e.get("message", "") for e in bad_data["errors"])

    valid = {
        "config": {
            "train_ratio": 0.7,
            "val_ratio": 0.2,
            "test_ratio": 0.1,
            "stratify": True,
            "stratify_col": 0,
            "stratify_bins": 8,
            "seed": 123,
        }
    }
    r_ok = client.post(
        "/api/v1/components/split_train_val_test/validate-config", json=valid
    )
    assert r_ok.status_code == 200
    ok_data = r_ok.json()
    assert ok_data["valid"] is True


def test_validate_config_select_columns(client):
    invalid = {
        "config": {
            "selection_mode": "indices",
            "selected_indices": "1,a,3",
            "schema_validation": "strict",
        }
    }
    r_bad = client.post(
        "/api/v1/components/select_columns/validate-config", json=invalid
    )
    assert r_bad.status_code == 200
    bad_data = r_bad.json()
    assert bad_data["valid"] is False
    assert any("selected_indices" in e.get("message", "") for e in bad_data["errors"])

    valid = {
        "config": {
            "selection_mode": "names",
            "schema_columns": "id,age,score",
            "selected_columns": "score,id",
            "schema_validation": "strict",
            "drop_invalid": True,
        }
    }
    r_ok = client.post("/api/v1/components/select_columns/validate-config", json=valid)
    assert r_ok.status_code == 200
    ok_data = r_ok.json()
    assert ok_data["valid"] is True


def test_validate_config_multi_select_enum_support(client):
    comp = {
        "id": "test_multi_enum_comp",
        "name": "Test Multi Enum",
        "category": "math",
        "version": "1.0.0",
        "inputs": [{"name": "x", "dtype": "tensor"}],
        "outputs": [{"name": "y", "dtype": "tensor"}],
        "params_schema": {
            "modes": {
                "type": "enum",
                "multi_select": True,
                "options": ["a", "b", "c"],
                "default": ["a"],
            }
        },
    }
    r_create = client.post("/api/v1/components", json=comp)
    assert r_create.status_code == 200

    r_ok = client.post(
        "/api/v1/components/test_multi_enum_comp/validate-config",
        json={"config": {"modes": ["a", "c"]}},
    )
    assert r_ok.status_code == 200
    data_ok = r_ok.json()
    assert data_ok["valid"] is True

    r_bad = client.post(
        "/api/v1/components/test_multi_enum_comp/validate-config",
        json={"config": {"modes": ["a", "z"]}},
    )
    assert r_bad.status_code == 200
    data_bad = r_bad.json()
    assert data_bad["valid"] is False
    assert any("Invalid options" in e.get("message", "") for e in data_bad["errors"])


def test_create_component(client):
    comp = {
        "id": "test_custom_op",
        "name": "Test Custom Op",
        "category": "math",
        "version": "1.0.0",
        "inputs": [{"name": "x", "dtype": "tensor"}],
        "outputs": [{"name": "y", "dtype": "tensor"}],
    }
    r = client.post("/api/v1/components", json=comp)
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == "test_custom_op"
    assert data["status"] == "draft"


def test_approve_component(client):
    # Create a draft component first
    comp = {
        "id": "test_approve_me",
        "name": "Approve Me",
        "category": "math",
        "version": "1.0.0",
        "inputs": [{"name": "x", "dtype": "tensor"}],
        "outputs": [{"name": "y", "dtype": "tensor"}],
    }
    client.post("/api/v1/components", json=comp)
    r = client.post("/api/v1/components/test_approve_me/approve")
    assert r.status_code == 200
    assert r.json()["status"] == "approved"


def test_validate_workflow_valid(client):
    workflow = {
        "workflow": {
            "schema_version": "workflow_graph.v1",
            "workflow_id": "wf_test1",
            "name": "Test Workflow",
            "nodes": [
                {"id": "n1", "component_type": "relu", "params": {}, "ui_meta": {}},
                {"id": "n2", "component_type": "gelu", "params": {}, "ui_meta": {}},
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "n1",
                    "source_port": "y",
                    "target": "n2",
                    "target_port": "x",
                }
            ],
        }
    }
    r = client.post("/api/v1/workflows/validate", json=workflow)
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is True


def test_validate_workflow_normalizes_legacy_component_ids(client):
    workflow = {
        "workflow": {
            "schema_version": "workflow_graph.v1",
            "workflow_id": "wf_legacy_aliases",
            "name": "Legacy Alias Workflow",
            "nodes": [
                {"id": "n_in", "component_type": "input", "params": {}, "ui_meta": {}},
                {"id": "n_mid", "component_type": "relu", "params": {}, "ui_meta": {}},
                {
                    "id": "n_out",
                    "component_type": "output_head",
                    "params": {},
                    "ui_meta": {},
                },
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "n_in",
                    "source_port": "y",
                    "target": "n_mid",
                    "target_port": "x",
                },
                {
                    "id": "e2",
                    "source": "n_mid",
                    "source_port": "y",
                    "target": "n_out",
                    "target_port": "x",
                },
            ],
        }
    }
    r = client.post("/api/v1/workflows/validate", json=workflow)
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is True
    assert not [
        issue for issue in data["issues"] if issue["code"] == "unknown_component"
    ]


def test_validate_workflow_rejects_unresolved_component_ids(client):
    workflow = {
        "workflow": {
            "schema_version": "workflow_graph.v1",
            "workflow_id": "wf_unknown_component",
            "name": "Unknown Component Workflow",
            "nodes": [
                {
                    "id": "n1",
                    "component_type": "math_space/not_real_component",
                    "params": {},
                    "ui_meta": {},
                },
            ],
            "edges": [],
        }
    }
    r = client.post("/api/v1/workflows/validate", json=workflow)
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is False
    issues = [issue for issue in data["issues"] if issue["code"] == "unknown_component"]
    assert issues
    assert issues[0]["node_id"] == "n1"


def test_validate_workflow_cycle(client):
    workflow = {
        "workflow": {
            "schema_version": "workflow_graph.v1",
            "workflow_id": "wf_cycle",
            "name": "Cycle Test",
            "nodes": [
                {"id": "n1", "component_type": "relu", "params": {}, "ui_meta": {}},
                {"id": "n2", "component_type": "gelu", "params": {}, "ui_meta": {}},
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "n1",
                    "source_port": "y",
                    "target": "n2",
                    "target_port": "x",
                },
                {
                    "id": "e2",
                    "source": "n2",
                    "source_port": "y",
                    "target": "n1",
                    "target_port": "x",
                },
            ],
        }
    }
    r = client.post("/api/v1/workflows/validate", json=workflow)
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is False
    assert any("cycle" in i["code"] for i in data["issues"])


def test_validate_workflow_dangling_edge(client):
    workflow = {
        "workflow": {
            "schema_version": "workflow_graph.v1",
            "workflow_id": "wf_dangle",
            "name": "Dangling Test",
            "nodes": [
                {"id": "n1", "component_type": "relu", "params": {}, "ui_meta": {}},
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "n1",
                    "source_port": "y",
                    "target": "n999",
                    "target_port": "x",
                }
            ],
        }
    }
    r = client.post("/api/v1/workflows/validate", json=workflow)
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is False


def test_validate_workflow_unsupported_edge_dtype_pairing(client):
    workflow = {
        "workflow": {
            "schema_version": "workflow_graph.v1",
            "workflow_id": "wf_dtype_pairing",
            "name": "Dtype Pairing Test",
            "nodes": [
                {
                    "id": "n1",
                    "component_type": "dataset_filter",
                    "params": {},
                    "ui_meta": {},
                },
                {"id": "n2", "component_type": "relu", "params": {}, "ui_meta": {}},
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "n1",
                    "source_port": "filtered",
                    "target": "n2",
                    "target_port": "x",
                }
            ],
        }
    }
    r = client.post("/api/v1/workflows/validate", json=workflow)
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is False
    mismatch_issues = [
        i for i in data["issues"] if i.get("code") == "unsupported_edge_dtype_pairing"
    ]
    assert mismatch_issues
    assert "Unsupported edge dtype pairing" in mismatch_issues[0]["message"]


def test_validate_workflow_dead_branch_detected(client):
    workflow = {
        "workflow": {
            "schema_version": "workflow_graph.v1",
            "workflow_id": "wf_dead_branch",
            "name": "Dead Branch Test",
            "nodes": [
                {"id": "n1", "component_type": "relu", "params": {}, "ui_meta": {}},
                {
                    "id": "n2",
                    "component_type": "io/output",
                    "params": {},
                    "ui_meta": {},
                },
                {"id": "n3", "component_type": "gelu", "params": {}, "ui_meta": {}},
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "n1",
                    "source_port": "y",
                    "target": "n2",
                    "target_port": "x",
                }
            ],
        }
    }
    r = client.post("/api/v1/workflows/validate", json=workflow)
    assert r.status_code == 200
    data = r.json()
    assert data["valid"] is False

    dead_branch_issues = [i for i in data["issues"] if i.get("code") == "dead_branch"]
    assert dead_branch_issues
    assert dead_branch_issues[0].get("node_id") == "n3"


def test_aria_suggest_components_data_control_prompt(client):
    workflow = {
        "workflow": {
            "schema_version": "workflow_graph.v1",
            "workflow_id": "wf_data_control_suggest",
            "name": "Data Control Suggest",
            "nodes": [
                {
                    "id": "src",
                    "component_type": "io/input",
                    "params": {},
                    "ui_meta": {},
                },
            ],
            "edges": [],
        },
        "prompt": "Optimize data/control workflow with better join/filter behavior and schema hygiene",
    }

    r = client.post("/api/v1/aria/suggest-components", json=workflow)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) > 0

    reasons = [str(item.get("reason", "")).lower() for item in data]
    assert any(
        "schema" in reason or "join" in reason or "filter" in reason
        for reason in reasons
    )


def test_router_suggest_components_forwards_research_signals(monkeypatch):
    from aria_designer.api.app.models import (
        SuggestComponentsRequest,
        WorkflowGraphModel,
    )
    from aria_designer.api.app.routers import aria as aria_router

    captured = {}

    monkeypatch.setattr(
        aria_router,
        "fetch_research_recommendation_signals",
        lambda force=False: {"op_priors": [{"op_name": "layernorm", "s1_rate": 0.8}]},
    )

    def _fake_suggest_components(workflow, prompt=None, research_signals=None):
        captured["workflow"] = workflow
        captured["prompt"] = prompt
        captured["research_signals"] = research_signals
        return [{"component_type": "normalization/layernorm", "reason": "test"}]

    monkeypatch.setattr(aria_router, "suggest_components", _fake_suggest_components)
    monkeypatch.setattr(aria_router, "HAS_SUGGESTIONS", True)

    req = SuggestComponentsRequest(
        workflow=WorkflowGraphModel(
            schema_version="workflow_graph.v1",
            workflow_id="wf_router_signals",
            name="Router Signals",
            nodes=[],
            edges=[],
        ),
        prompt="Improve stability",
    )

    resp = aria_router.post_suggest_components(req)
    assert resp[0]["component_type"] == "normalization/layernorm"
    assert captured["prompt"] == "Improve stability"
    assert captured["research_signals"] == {
        "op_priors": [{"op_name": "layernorm", "s1_rate": 0.8}]
    }


def test_save_and_get_workflow(client):
    workflow = {
        "schema_version": "workflow_graph.v1",
        "workflow_id": "wf_save_test",
        "name": "Save Test",
        "nodes": [
            {"id": "n1", "component_type": "relu", "params": {}, "ui_meta": {}},
        ],
        "edges": [],
    }
    r = client.put("/api/v1/workflows/wf_save_test", json=workflow)
    assert r.status_code == 200
    assert r.json()["version"] == 1

    r = client.get("/api/v1/workflows/wf_save_test")
    assert r.status_code == 200
    assert r.json()["name"] == "Save Test"
    assert r.json()["graph"]["nodes"][0]["component_type"] == "math/relu"

    # Update should increment version
    r = client.put(
        "/api/v1/workflows/wf_save_test", json={**workflow, "name": "Updated"}
    )
    assert r.json()["version"] == 2


def test_save_workflow_rejects_unresolved_component_ids(client):
    workflow = {
        "schema_version": "workflow_graph.v1",
        "workflow_id": "wf_save_invalid_component",
        "name": "Invalid Save",
        "nodes": [
            {
                "id": "n1",
                "component_type": "math/not_a_real_kernel",
                "params": {},
                "ui_meta": {},
            },
        ],
        "edges": [],
    }
    r = client.put("/api/v1/workflows/wf_save_invalid_component", json=workflow)
    assert r.status_code == 422
    data = r.json()
    assert "issues" in data["detail"]


def test_propose_and_apply_patch(client):
    # Create the workflow first
    workflow = {
        "schema_version": "workflow_graph.v1",
        "workflow_id": "wf_patch_test",
        "name": "Patch Test",
        "nodes": [
            {"id": "in", "component_type": "graph_input", "params": {}, "ui_meta": {}},
            {
                "id": "n1",
                "component_type": "linear_proj",
                "params": {"out_dim": 256},
                "ui_meta": {},
            },
            {
                "id": "out",
                "component_type": "graph_output",
                "params": {},
                "ui_meta": {},
            },
        ],
        "edges": [
            {
                "id": "e0",
                "source": "in",
                "source_port": "out",
                "target": "n1",
                "target_port": "in",
            },
            {
                "id": "e1",
                "source": "n1",
                "source_port": "out",
                "target": "out",
                "target_port": "in",
            },
        ],
    }
    r_save = client.put("/api/v1/workflows/wf_patch_test", json=workflow)
    assert r_save.status_code == 200, f"save failed: {r_save.json()}"

    patch = {
        "workflow_id": "wf_patch_test",
        "base_version": 1,
        "author": "aria",
        "rationale": "Test proposal",
        "ops": [
            {
                "op": "mutate_param",
                "node_id": "n1",
                "payload": {"param_name": "out_dim", "new_value": 512},
            }
        ],
    }
    r = client.post("/api/v1/aria/propose-patch", json=patch)
    assert r.status_code == 200
    proposal_id = r.json()["proposal_id"]

    # Apply
    r = client.post(
        "/api/v1/aria/apply-patch",
        json={
            "proposal_id": proposal_id,
            "approved_by": "test_user",
        },
    )
    assert r.status_code == 200, f"apply-patch failed: {r.json()}"
    assert r.json()["applied"] is True


def test_apply_patch_repairs_partial_add_node_wiring(client):
    workflow = {
        "schema_version": "workflow_graph.v1",
        "workflow_id": "wf_partial_insert_fix",
        "name": "Partial Insert Fix",
        "nodes": [
            {"id": "in", "component_type": "graph_input", "params": {}, "ui_meta": {}},
            {
                "id": "n1",
                "component_type": "linear_proj",
                "params": {"out_dim": 256},
                "ui_meta": {},
            },
            {
                "id": "out",
                "component_type": "graph_output",
                "params": {},
                "ui_meta": {},
            },
        ],
        "edges": [
            {
                "id": "e0",
                "source": "in",
                "source_port": "out",
                "target": "n1",
                "target_port": "in",
            },
            {
                "id": "e1",
                "source": "n1",
                "source_port": "out",
                "target": "out",
                "target_port": "in",
            },
        ],
    }
    r_save = client.put("/api/v1/workflows/wf_partial_insert_fix", json=workflow)
    assert r_save.status_code == 200, f"save failed: {r_save.json()}"

    patch = {
        "workflow_id": "wf_partial_insert_fix",
        "base_version": 1,
        "author": "aria",
        "rationale": "Insert norm before output (partial wiring)",
        "ops": [
            {
                "op": "add_node",
                "payload": {
                    "id": "aria_norm_insert",
                    "component_type": "linear_algebra/rmsnorm",
                    "params": {},
                    # Deliberately partial: only new -> output
                    "edges": [
                        {
                            "source": "aria_norm_insert",
                            "source_port": "out",
                            "target": "out",
                            "target_port": "in",
                        }
                    ],
                },
            }
        ],
    }
    r_prop = client.post("/api/v1/aria/propose-patch", json=patch)
    assert r_prop.status_code == 200, f"propose failed: {r_prop.json()}"
    proposal_id = r_prop.json()["proposal_id"]

    r_apply = client.post(
        "/api/v1/aria/apply-patch",
        json={
            "proposal_id": proposal_id,
            "approved_by": "test_user",
        },
    )
    assert r_apply.status_code == 200, f"apply failed: {r_apply.json()}"
    patched = r_apply.json()["patched_workflow"]
    edges = patched.get("edges", [])

    assert any(
        e.get("source") == "n1" and e.get("target") == "aria_norm_insert" for e in edges
    )
    assert any(
        e.get("source") == "aria_norm_insert" and e.get("target") == "out"
        for e in edges
    )
    assert not any(e.get("source") == "n1" and e.get("target") == "out" for e in edges)


def test_list_proposals(client):
    r = client.get("/api/v1/aria/proposals")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_reload_components(client):
    r = client.post("/api/v1/components/reload")
    assert r.status_code == 200
    assert "reloaded" in r.json()


def test_preview_workflow_multi_input_ports(client):
    workflow = {
        "workflow": {
            "schema_version": "workflow_graph.v1",
            "workflow_id": "wf_preview_multi_input",
            "name": "Preview Multi Input",
            "nodes": [
                {
                    "id": "input_a",
                    "component_type": "io/input",
                    "params": {},
                    "ui_meta": {},
                },
                {
                    "id": "input_b",
                    "component_type": "io/input",
                    "params": {},
                    "ui_meta": {},
                },
                {
                    "id": "add1",
                    "component_type": "math/add",
                    "params": {},
                    "ui_meta": {},
                },
                {
                    "id": "out",
                    "component_type": "io/output_head",
                    "params": {},
                    "ui_meta": {},
                },
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "input_a",
                    "source_port": "y",
                    "target": "add1",
                    "target_port": "a",
                },
                {
                    "id": "e2",
                    "source": "input_b",
                    "source_port": "y",
                    "target": "add1",
                    "target_port": "b",
                },
                {
                    "id": "e3",
                    "source": "add1",
                    "source_port": "y",
                    "target": "out",
                    "target_port": "x",
                },
            ],
        },
        "target": "auto",
    }

    r = client.post("/api/v1/workflows/preview", json=workflow)
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert "out" in data["results"]
    assert "shape" in data["results"]["out"]


def _ai_learning_workflow():
    return {
        "schema_version": "workflow_graph.v1",
        "workflow_id": "wf_ai_learning_loop",
        "name": "AI Learning Loop",
        "metadata": {"model_dim": 128},
        "nodes": [
            {"id": "in", "component_type": "graph_input", "params": {}, "ui_meta": {}},
            {
                "id": "proj",
                "component_type": "linear_proj",
                "params": {"out_dim": 128},
                "ui_meta": {},
            },
            {
                "id": "out",
                "component_type": "graph_output",
                "params": {},
                "ui_meta": {},
            },
        ],
        "edges": [
            {
                "id": "e0",
                "source": "in",
                "source_port": "out",
                "target": "proj",
                "target_port": "in",
            },
            {
                "id": "e1",
                "source": "proj",
                "source_port": "out",
                "target": "out",
                "target_port": "in",
            },
        ],
    }


class _FakeAiLearningBridgeResult:
    def to_dict(self):
        return {
            "status": "success",
            "sandbox_passed": True,
            "graph_fingerprint": "fp_ai_learning_loop_v2",
            "overall_novelty": 0.73,
            "efficiency_score": 0.62,
            "total_time_ms": 12.5,
        }


def _apply_ai_learning_patch(client, workflow):
    r_save = client.put("/api/v1/workflows/wf_ai_learning_loop", json=workflow)
    assert r_save.status_code == 200
    assert r_save.json()["version"] == 1
    r_patch = client.post(
        "/api/v1/aria/generate-patch",
        json={
            "workflow": workflow,
            "prompt": "add relu after projection",
            "base_version": 1,
        },
    )
    assert r_patch.status_code == 200
    proposal_id = r_patch.json()["proposal_id"]

    r_apply = client.post(
        "/api/v1/aria/apply-patch",
        json={"proposal_id": proposal_id, "approved_by": "test_user"},
    )
    assert r_apply.status_code == 200
    assert r_apply.json()["applied"] is True
    r_get = client.get("/api/v1/workflows/wf_ai_learning_loop")
    assert r_get.status_code == 200
    updated = r_get.json()
    assert updated["version"] == 2
    updated_graph = updated["graph"]
    updated_graph["version"] = updated["version"]
    return updated_graph


def _patch_ai_learning_eval(monkeypatch, captured):
    from aria_designer.api.app.routers import eval as eval_mod
    from aria_designer.api.app import shared_api as shared_mod

    monkeypatch.setattr(eval_mod, "HAS_BRIDGE", True)
    # eval router resolves bridge_evaluate lazily via runtime_features (_rf), so
    # patch it there rather than on the router module.
    monkeypatch.setattr(
        eval_mod._rf,
        "bridge_evaluate",
        lambda *args, **kwargs: _FakeAiLearningBridgeResult(),
    )
    monkeypatch.setattr(shared_mod, "HAS_BRIDGE", True)
    monkeypatch.setattr(shared_mod.settings, "LINEAGE_SYNC_ENABLED", True)

    def _capture_sync(payload):
        captured["payload"] = payload
        return True

    monkeypatch.setattr(eval_mod, "_sync_lineage_to_research", _capture_sync)


def _evaluate_ai_learning_workflow(client, updated_graph):
    r_eval = client.post(
        "/api/v1/workflows/evaluate",
        json={
            "workflow": updated_graph,
            "budget": {
                "model_dim": 128,
                "device": "cpu",
                "run_fingerprint": True,
                "run_novelty": True,
            },
        },
    )
    assert r_eval.status_code == 200
    eval_data = r_eval.json()
    assert eval_data["status"] == "success"
    assert eval_data["graph_fingerprint"] == "fp_ai_learning_loop_v2"
    assert "benchmarking" in eval_data
    assert "summary" in eval_data["benchmarking"]
    assert "targets" in eval_data["benchmarking"]
    assert eval_data["lineage_sync"]["attempted"] is True
    assert eval_data["lineage_sync"]["synced"] is True
    return eval_data


def test_ai_design_refine_evaluate_records_lineage(client, monkeypatch):
    updated_graph = _apply_ai_learning_patch(client, _ai_learning_workflow())
    captured = {}
    _patch_ai_learning_eval(monkeypatch, captured)
    _evaluate_ai_learning_workflow(client, updated_graph)
    lineage_payload = captured.get("payload")
    assert lineage_payload is not None
    assert lineage_payload["workflow_id"] == "wf_ai_learning_loop"
    assert lineage_payload["graph_fingerprint"] == "fp_ai_learning_loop_v2"
    assert lineage_payload["status"] == "success"
    assert lineage_payload["metrics"]["overall_novelty"] == 0.73


def test_benchmark_target_catalog_endpoint(client):
    r = client.get("/api/v1/benchmarks/targets")
    assert r.status_code == 200
    data = r.json()
    assert data["version"] == "benchmark_targets.v1"
    assert isinstance(data.get("targets"), list)
    assert len(data["targets"]) >= 8
    ids = {t["id"] for t in data["targets"]}
    assert "param_count" in ids
    assert "total_flops_per_token" in ids
    assert "mmlu_5shot" in ids
