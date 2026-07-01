from __future__ import annotations

import json

import pandas as pd

from research.scientist.analytics.model_strength import (
    _counter_feature_map,
    _graph_features,
    build_model_strength_report,
)
from research.scientist.api import create_app
from research.scientist.notebook import LabNotebook

_REPRESENTATIVE_PARAM_COUNT = 1_200_000


def _graph(template: str, ops: list[str], *, slot_motif: str | None = None) -> str:
    nodes = {
        str(index): {
            "id": str(index),
            "op_name": op,
            "input_ids": [str(index - 1)] if index > 0 else [],
        }
        for index, op in enumerate(ops)
    }
    metadata: dict[str, object] = {
        "primary_template": template,
        "templates_used": [template],
        "motifs_used": [f"{template}_motif"],
    }
    if slot_motif:
        metadata["template_slot_usage"] = [
            {
                "template_name": template,
                "slot_index": 0,
                "slot_key": f"{template}.slot0",
                "slot_classes": ["core"],
                "selected_motif": slot_motif,
                "selected_motif_class": "core",
            }
        ]
    return json.dumps({"nodes": nodes, "metadata": metadata})


def _seed_strength_db(db_path: str) -> None:
    nb = LabNotebook(db_path)
    exp_id = nb.start_experiment(
        "synthesis",
        {
            "stage1_steps": 400,
            "model_dim": 128,
            "n_layers": 3,
            "category_weights": {"mixing": 1.4, "math_space": 0.6},
            "template_weights": {"tpl_good": 1.3, "tpl_bad": 0.8},
        },
    )
    rows = [
        {
            "graph_fingerprint": "fp-good-1",
            "graph_json": _graph(
                "tpl_good",
                ["linear_proj", "relu_gate_routing", "moe_topk", "layernorm"],
                slot_motif="strong_core",
            ),
            "stage0_passed": True,
            "stage05_passed": True,
            "stage1_passed": True,
            "loss_ratio": 0.32,
            "validation_loss_ratio": 0.34,
            "induction_screening_auc": 0.19,
            "binding_screening_auc": 0.17,
            "binding_screening_composite": 0.18,
            "ar_legacy_auc": 0.21,
            "hellaswag_acc": 0.36,
            "blimp_overall_accuracy": 0.74,
            "wikitext_perplexity": 18.0,
            "wikitext_score": 0.68,
            "stability_score": 0.91,
            "validation_robustness_score": 0.88,
            "efficiency_multiple": 1.7,
            "param_count": _REPRESENTATIVE_PARAM_COUNT,
            "graph_depth": 5,
            "graph_n_ops": 4,
            "graph_n_unique_ops": 4,
            "train_budget_steps": 400,
            "result_cohort": "search",
            "trust_label": "candidate_grade",
            "comparability_label": "candidate_comparable",
            "evaluation_protocol_version": "candidate_grade_v1",
            "model_source": "graph_synthesis",
        },
        {
            "graph_fingerprint": "fp-good-2",
            "graph_json": _graph(
                "tpl_good",
                ["linear_proj", "relu_gate_routing", "moe_topk", "layernorm"],
                slot_motif="strong_core",
            ),
            "stage0_passed": True,
            "stage05_passed": True,
            "stage1_passed": True,
            "loss_ratio": 0.35,
            "validation_loss_ratio": 0.37,
            "induction_screening_auc": 0.15,
            "binding_screening_auc": 0.14,
            "binding_screening_composite": 0.145,
            "ar_legacy_auc": 0.18,
            "hellaswag_acc": 0.33,
            "blimp_overall_accuracy": 0.70,
            "wikitext_perplexity": 20.0,
            "wikitext_score": 0.61,
            "stability_score": 0.86,
            "validation_robustness_score": 0.82,
            "efficiency_multiple": 1.4,
            "param_count": _REPRESENTATIVE_PARAM_COUNT,
            "graph_depth": 5,
            "graph_n_ops": 4,
            "graph_n_unique_ops": 4,
            "train_budget_steps": 400,
            "result_cohort": "search",
            "trust_label": "candidate_grade",
            "comparability_label": "candidate_comparable",
            "evaluation_protocol_version": "candidate_grade_v1",
            "model_source": "graph_synthesis",
        },
        {
            "graph_fingerprint": "fp-bad-1",
            "graph_json": _graph(
                "tpl_bad",
                ["linear_proj", "add", "residual_scale", "layernorm"],
                slot_motif="weak_core",
            ),
            "stage0_passed": True,
            "stage05_passed": True,
            "stage1_passed": False,
            "loss_ratio": 0.82,
            "validation_loss_ratio": 0.84,
            "induction_screening_auc": 0.01,
            "binding_screening_auc": 0.02,
            "hellaswag_acc": 0.24,
            "wikitext_perplexity": 44.0,
            "wikitext_score": 0.18,
            "stability_score": 0.49,
            "validation_robustness_score": 0.31,
            "efficiency_multiple": 0.5,
            "param_count": _REPRESENTATIVE_PARAM_COUNT,
            "graph_depth": 5,
            "graph_n_ops": 4,
            "graph_n_unique_ops": 4,
            "train_budget_steps": 400,
            "result_cohort": "search",
            "trust_label": "candidate_grade",
            "comparability_label": "candidate_comparable",
            "evaluation_protocol_version": "candidate_grade_v1",
            "model_source": "graph_synthesis",
        },
        {
            "graph_fingerprint": "fp-bad-2",
            "graph_json": _graph(
                "tpl_bad",
                ["linear_proj", "add", "residual_scale", "layernorm"],
                slot_motif="weak_core",
            ),
            "stage0_passed": True,
            "stage05_passed": True,
            "stage1_passed": False,
            "loss_ratio": 0.78,
            "validation_loss_ratio": 0.79,
            "induction_screening_auc": 0.02,
            "binding_screening_auc": 0.01,
            "hellaswag_acc": 0.25,
            "wikitext_perplexity": 39.0,
            "wikitext_score": 0.21,
            "stability_score": 0.52,
            "validation_robustness_score": 0.34,
            "efficiency_multiple": 0.6,
            "param_count": _REPRESENTATIVE_PARAM_COUNT,
            "graph_depth": 5,
            "graph_n_ops": 4,
            "graph_n_unique_ops": 4,
            "train_budget_steps": 400,
            "result_cohort": "search",
            "trust_label": "candidate_grade",
            "comparability_label": "candidate_comparable",
            "evaluation_protocol_version": "candidate_grade_v1",
            "model_source": "graph_synthesis",
        },
    ]
    for row in rows:
        nb.record_program_result(experiment_id=exp_id, **row)
    nb.flush_writes()
    nb.complete_experiment(exp_id, results={"n_programs": len(rows)})
    nb.flush_writes()
    nb.close()


def test_build_model_strength_report_returns_rankings(tmp_path):
    db_path = str(tmp_path / "strength.sqlite3")
    _seed_strength_db(db_path)

    report = build_model_strength_report(db_path, min_support=1, top_k=10)

    assert report["support"]["dedup_trusted"]["runs"] >= 4
    assert report["support"]["dedup_promotable"]["runs"] >= 2
    assert report["rankings"]["best_components_overall"]
    assert report["rankings"]["best_templates_overall"]
    assert "models" in report["drift_analysis"]
    top_component = report["rankings"]["best_components_overall"][0]
    assert top_component["confidence_tier"] in {"low", "medium", "high"}
    assert "template_count" in top_component
    assert "protocol_count" in top_component
    assert "matched_template_controls" in top_component
    assert isinstance(top_component["artifact_flags"], list)

    top_template = report["rankings"]["best_templates_overall"][0]["name"]
    assert top_template == "tpl_good"


def test_graph_features_supports_list_nodes_and_preserves_raw_ops():
    graph_json = json.dumps(
        {
            "nodes": [
                {"id": 0, "op_name": "input", "input_ids": []},
                {"id": 1, "op_name": "state_space", "input_ids": [0]},
                {"id": 2, "op_name": "route_topk", "input_ids": [1]},
                {"id": 3, "op_name": "output", "input_ids": [2]},
            ],
            "metadata": {
                "primary_template": "router_block",
                "templates_used": ["router_block"],
                "motifs_used": ["router_chain"],
                "dynamic_components_used": [
                    {
                        "component_id": "component_branch",
                        "lowering": "mixer_sidecar_restore_v1",
                    }
                ],
                "template_slot_usage": [
                    {
                        "template_name": "router_block",
                        "slot_index": 0,
                        "selected_motif": "router_chain",
                    }
                ],
            },
        }
    )

    features = _graph_features(graph_json)

    assert features["primary_template"] == "router_block"
    assert features["ops"] == ["route_topk", "state_space"]
    assert features["op_pairs"] == ["route_topk+state_space"]
    assert features["depth_ops"] == ["late:route_topk", "middle:state_space"]
    assert features["slot_components"] == ["router_block.slot0:router_chain"]
    assert features["dynamic_components"] == [
        "component_branch",
        "lowering:mixer_sidecar_restore_v1",
    ]
    assert features["pattern_has_routing"] == 1
    assert features["pattern_has_ssm"] == 1
    assert features["pattern_dynamic_component"] == 1
    assert features["pattern_dynamic_branch_component"] == 1


def test_counter_feature_map_dedupes_within_row_and_preserves_index():
    series = pd.Series(
        [
            ["relu", "relu", "norm"],
            None,
            ["norm", ""],
            "relu",
        ],
        index=["a", "b", "c", "d"],
    )

    features = _counter_feature_map(series)

    assert sorted(features) == ["norm", "relu"]
    assert features["relu"].index.tolist() == ["a", "b", "c", "d"]
    assert features["relu"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert features["norm"].tolist() == [1.0, 0.0, 1.0, 0.0]


def test_model_strength_endpoint_returns_payload(tmp_path):
    db_path = str(tmp_path / "strength_api.sqlite3")
    _seed_strength_db(db_path)

    app = create_app(notebook_path=db_path)
    client = app.test_client()
    response = client.get("/api/reporting/model-strength?min_support=1&top_k=5")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    assert "rankings" in payload
    assert payload["rankings"]["best_components_overall"]
    assert payload["great_model_definition"]["score_name"] == "great_score"


def test_slot_compatibility_endpoint_returns_generated_rules(tmp_path):
    db_path = str(tmp_path / "slot_rules_api.sqlite3")
    _seed_strength_db(db_path)

    app = create_app(notebook_path=db_path)
    client = app.test_client()
    response = client.get("/api/reporting/slot-compatibility")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload is not None
    rules = {row["slot_key"]: row for row in payload["slot_rules"]}
    assert set(rules) == {"depth_token_mask_block.slot1"}
    assert (
        "route_lanes_block" in rules["depth_token_mask_block.slot1"]["blocked_motifs"]
    )
