import json
from pathlib import Path

import research.tools.build_dynamic_component_candidates as builder_mod
from research.tools.build_dynamic_component_candidates import (
    build_dynamic_component_candidates,
)


def _write_report(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "component_rule_mining_v1",
                "op_pair_rules": {"negative": []},
                "candidate_windows": [
                    {
                        "pattern": [
                            "rmsnorm",
                            "linear_proj",
                            "gelu",
                            "selective_scan",
                            "add",
                            "rmsnorm",
                            "linear_proj",
                            "add",
                        ],
                        "n": 12,
                        "stage1_passed": 11,
                        "pass_rate": 0.9167,
                        "pass_rate_lift": 0.25,
                        "mean_loss_ratio": 0.6,
                    },
                    {
                        "pattern": ["relu", "add", "rmsnorm"],
                        "n": 50,
                        "stage1_passed": 50,
                        "pass_rate": 1.0,
                        "pass_rate_lift": 0.5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _assert_math_physics_review_passed(validation: dict) -> None:
    assert validation["math_physics_review_passed"] is True
    assert validation["math_physics_review_requires_gpu"] is False
    assert validation["math_physics_review_reason"] in {
        "selected_variant",
        "parent_retained_after_sweep",
    }
    assert validation["math_sweep_passed"] is True
    assert validation["math_sweep_required_for_ready"] is True
    assert validation["math_sweep_variant_count"] >= 1
    assert validation["math_sweep_records"]
    assert validation["math_sweep_selected_variant_id"]
    assert "physics_descriptors" in validation
    assert "spectral_radius" in validation["physics_descriptors"]
    assert "measured_descriptors" in validation
    assert "long_range_reach" in validation["measured_descriptors"]


def test_build_dynamic_component_candidates_keeps_only_structural_windows(
    tmp_path: Path,
) -> None:
    report = _write_report(tmp_path / "mining.json")
    output = tmp_path / "dynamic_components.json"

    payload = build_dynamic_component_candidates(
        mining_report_path=report,
        output_path=output,
        validate_candidates=False,
        min_lowered_ops=8,
        min_support=8,
        min_pass_rate=0.7,
    )

    assert output.exists()
    assert payload["metadata"]["n_candidates"] == 1
    assert payload["metadata"]["component_rule_schema_versions"]
    assert payload["ready_for_registration"] == []
    candidate = payload["candidates"][0]
    assert candidate["lowered_op_count"] == 9
    assert candidate["component_descriptor"]["has_multi_mixer"] is False
    assert candidate["component_descriptor"]["lowering"] == "mixer_sidecar_restore_v1"
    assert candidate["component_descriptor"]["slot_plan"][3]["slot_classes"] == [
        "dynamic_role:mix",
        "dynamic_step",
        "dynamic_mixer",
    ]


def test_build_dynamic_component_candidates_blocks_negative_pairs(
    tmp_path: Path,
) -> None:
    report = _write_report(tmp_path / "mining.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["op_pair_rules"] = {
        "negative": [
            {
                "pattern": ["linear_proj", "gelu"],
                "pass_rate": 0.2,
                "pass_rate_lift": -0.6,
            }
        ]
    }
    report.write_text(json.dumps(payload), encoding="utf-8")

    built = build_dynamic_component_candidates(
        mining_report_path=report,
        output_path=None,
        validate_candidates=False,
    )

    assert built["metadata"]["negative_pairs_blocked"] == 1
    assert built["candidates"] == []


def test_build_dynamic_component_candidates_allows_preferred_negative_pairs(
    tmp_path: Path,
) -> None:
    report = _write_report(tmp_path / "mining.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["op_pair_rules"] = {
        "negative": [
            {
                "pattern": ["linear_proj", "latent_attention_compressor"],
                "pass_rate": 0.2,
                "pass_rate_lift": -0.6,
            }
        ]
    }
    payload["candidate_windows"] = [
        {
            "pattern": [
                "layernorm",
                "linear_proj",
                "latent_attention_compressor",
                "add",
                "semi_structured_2_4_linear",
                "relu",
                "add",
                "layernorm",
            ],
            "n": 9,
            "stage1_passed": 6,
            "pass_rate": 0.6667,
            "pass_rate_lift": -0.1493,
            "mean_loss_ratio": 0.67,
        }
    ]
    report.write_text(json.dumps(payload), encoding="utf-8")

    built = build_dynamic_component_candidates(
        mining_report_path=report,
        output_path=None,
        validate_candidates=False,
    )

    assert built["metadata"]["negative_pairs_blocked"] == 0
    assert len(built["candidates"]) == 1
    assert built["candidates"][0]["component_descriptor"]["has_multi_mixer"] is False


def test_build_dynamic_component_candidates_marks_multi_mixer_branch_lowering(
    tmp_path: Path,
) -> None:
    report = _write_report(tmp_path / "mining.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["candidate_windows"] = [
        {
            "pattern": [
                "latent_attention_compressor",
                "linear_proj",
                "conv1d_seq",
                "silu",
                "rmsnorm",
                "selective_scan",
                "add",
                "add",
            ],
            "n": 12,
            "stage1_passed": 11,
            "pass_rate": 0.9167,
            "pass_rate_lift": 0.25,
            "mean_loss_ratio": 0.6,
        }
    ]
    report.write_text(json.dumps(payload), encoding="utf-8")

    built = build_dynamic_component_candidates(
        mining_report_path=report,
        output_path=None,
        validate_candidates=False,
    )

    descriptor = built["candidates"][0]["component_descriptor"]
    assert descriptor["has_multi_mixer"] is True
    assert descriptor["lowering"] == "trunk_sidecar_merge_v1"
    assert descriptor["branch_plan"] == {
        "trunk_indices": [0, 1],
        "sidecar_indices": [2, 3, 4, 5],
        "merge_op": "add",
        "post_merge_norm": True,
        "residual_output": True,
    }


def test_build_dynamic_component_candidates_marks_single_mixer_restore_sidecar(
    tmp_path: Path,
) -> None:
    report = _write_report(tmp_path / "mining.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["candidate_windows"] = [
        {
            "pattern": [
                "rmsnorm",
                "latent_attention_compressor",
                "linear_proj",
                "add",
                "layernorm",
                "conv1d_seq",
                "swiglu_mlp",
                "add",
            ],
            "n": 12,
            "stage1_passed": 11,
            "pass_rate": 0.9167,
            "pass_rate_lift": 0.25,
            "mean_loss_ratio": 0.6,
        }
    ]
    report.write_text(json.dumps(payload), encoding="utf-8")

    built = build_dynamic_component_candidates(
        mining_report_path=report,
        output_path=None,
        validate_candidates=False,
    )

    descriptor = built["candidates"][0]["component_descriptor"]
    assert descriptor["has_multi_mixer"] is False
    assert descriptor["lowering"] == "mixer_sidecar_restore_v1"
    assert descriptor["branch_plan"] == {
        "trunk_indices": [1, 2],
        "sidecar_indices": [4, 5, 6],
        "merge_op": "add",
        "post_merge_norm": True,
        "residual_output": True,
    }


def test_build_dynamic_component_candidates_marks_router_lane_blend(
    tmp_path: Path,
) -> None:
    report = _write_report(tmp_path / "mining.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["candidate_windows"] = [
        {
            "pattern": [
                "add",
                "rmsnorm",
                "linear_proj",
                "matmul",
                "linear_proj",
                "gather_topk",
                "swiglu_mlp",
                "add",
            ],
            "n": 12,
            "stage1_passed": 11,
            "pass_rate": 0.9167,
            "pass_rate_lift": 0.25,
            "mean_loss_ratio": 0.6,
        }
    ]
    report.write_text(json.dumps(payload), encoding="utf-8")

    built = build_dynamic_component_candidates(
        mining_report_path=report,
        output_path=None,
        validate_candidates=False,
    )

    descriptor = built["candidates"][0]["component_descriptor"]
    assert descriptor["lowering"] == "router_lane_blend_v1"
    assert descriptor["branch_plan"] == {
        "value_project_index": 2,
        "matmul_index": 3,
        "score_project_index": 4,
        "route_index": 5,
        "gate_index": 6,
        "blend_op": "gated_lane_blend",
        "post_merge_norm": True,
        "residual_output": True,
    }


def test_branch_candidates_validate_lowered_topology(tmp_path: Path) -> None:
    report = _write_report(tmp_path / "mining.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["candidate_windows"] = [
        {
            "pattern": [
                "latent_attention_compressor",
                "linear_proj",
                "conv1d_seq",
                "silu",
                "rmsnorm",
                "selective_scan",
                "add",
                "add",
            ],
            "n": 12,
            "stage1_passed": 11,
            "pass_rate": 0.9167,
            "pass_rate_lift": 0.25,
            "mean_loss_ratio": 0.6,
        }
    ]
    report.write_text(json.dumps(payload), encoding="utf-8")

    built = build_dynamic_component_candidates(
        mining_report_path=report,
        output_path=None,
        validate_candidates=True,
    )

    validation = built["candidates"][0]["validation"]
    assert validation["lowering_validated"] == "trunk_sidecar_merge_v1"
    assert validation["validate_passed"] is True
    assert validation["compile_passed"] is True
    assert validation["forward_passed"] is True
    assert validation["backward_passed"] is True
    _assert_math_physics_review_passed(validation)


def test_router_lane_blend_candidates_validate_lowered_topology(
    tmp_path: Path,
) -> None:
    report = _write_report(tmp_path / "mining.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["candidate_windows"] = [
        {
            "pattern": [
                "add",
                "rmsnorm",
                "linear_proj",
                "matmul",
                "linear_proj",
                "gather_topk",
                "swiglu_mlp",
                "add",
            ],
            "n": 12,
            "stage1_passed": 11,
            "pass_rate": 0.9167,
            "pass_rate_lift": 0.25,
            "mean_loss_ratio": 0.6,
        }
    ]
    report.write_text(json.dumps(payload), encoding="utf-8")

    built = build_dynamic_component_candidates(
        mining_report_path=report,
        output_path=None,
        validate_candidates=True,
    )

    validation = built["candidates"][0]["validation"]
    assert validation["lowering_validated"] == "router_lane_blend_v1"
    assert validation["validate_passed"] is True
    assert validation["compile_passed"] is True
    assert validation["forward_passed"] is True
    assert validation["backward_passed"] is True
    _assert_math_physics_review_passed(validation)
    assert validation["static_route_count"] > 0


def test_restore_sidecar_candidates_validate_lowered_topology(tmp_path: Path) -> None:
    report = _write_report(tmp_path / "mining.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["candidate_windows"] = [
        {
            "pattern": [
                "rmsnorm",
                "latent_attention_compressor",
                "linear_proj",
                "add",
                "layernorm",
                "conv1d_seq",
                "swiglu_mlp",
                "add",
            ],
            "n": 12,
            "stage1_passed": 11,
            "pass_rate": 0.9167,
            "pass_rate_lift": 0.25,
            "mean_loss_ratio": 0.6,
        }
    ]
    report.write_text(json.dumps(payload), encoding="utf-8")

    built = build_dynamic_component_candidates(
        mining_report_path=report,
        output_path=None,
        validate_candidates=True,
    )

    validation = built["candidates"][0]["validation"]
    assert validation["lowering_validated"] == "mixer_sidecar_restore_v1"
    assert validation["validate_passed"] is True
    assert validation["compile_passed"] is True
    assert validation["forward_passed"] is True
    assert validation["backward_passed"] is True
    _assert_math_physics_review_passed(validation)


def test_candidate_validation_stops_positionwise_chain_before_smoke() -> None:
    validation = builder_mod._validate_candidate_chain(
        ("linear_proj", "relu", "add"),
        model_dim=16,
        run_smoke=True,
    )

    assert validation["compile_passed"] is True
    assert validation["math_physics_review_passed"] is False
    assert validation["math_physics_review_reason"] == "no_sequence_mixer_or_router_on_path"
    assert validation["failure_mode"] == "math_physics_review"
    assert validation["forward_passed"] is False
    assert validation["backward_passed"] is False


def test_build_dynamic_component_candidates_ready_requires_math_and_backward_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = _write_report(tmp_path / "mining.json")

    def fake_candidate_validation(candidate, **kwargs):
        return {
            "compile_passed": True,
            "validate_passed": True,
            "forward_passed": True,
            "backward_passed": True,
            "math_physics_review_passed": True,
            "math_sweep_passed": True,
            "n_ops": len(candidate["chain"]) + 1,
        }

    monkeypatch.setattr(
        "research.tools.build_dynamic_component_candidates._candidate_validation",
        fake_candidate_validation,
    )
    payload = build_dynamic_component_candidates(
        mining_report_path=report,
        output_path=None,
        validate_candidates=True,
    )

    assert len(payload["ready_for_registration"]) == 1
    assert payload["ready_for_registration"][0]["validation"]["backward_passed"] is True
    assert (
        payload["ready_for_registration"][0]["validation"]["math_physics_review_passed"]
        is True
    )

    not_reviewed = {
        "validate_passed": True,
        "compile_passed": True,
        "forward_passed": True,
        "backward_passed": True,
    }
    reviewed = dict(
        not_reviewed, math_physics_review_passed=True, math_sweep_passed=True
    )
    only_reviewed = dict(not_reviewed, math_physics_review_passed=True)
    assert builder_mod._is_ready(not_reviewed) is False
    assert builder_mod._is_ready(only_reviewed) is False
    assert builder_mod._is_ready(reviewed) is True
