import json
import os
import random
from pathlib import Path

from research.scientist.runner._types import RunConfig
from research.synthesis.dynamic_template_registry import (
    _candidate_selection_weights,
    apply_dynamic_template_candidate,
    choose_dynamic_template_candidate,
    load_dynamic_template_candidates,
)
from research.synthesis.generation_runtime import runtime_context_for_config
from research.synthesis.grammar import GrammarConfig, generate_layer_graph
from research.synthesis.graph import ComputationGraph


def _candidate(
    name: str,
    chain: list[str],
    *,
    promotion_score: float = 4.0,
    validated: bool = True,
    component_descriptor: dict | None = None,
) -> dict:
    out = {
        "proposed_template_name": name,
        "chain": chain,
        "n_total": 16,
        "pass_rate": 0.75,
        "lift_vs_cohort": 2.0,
        "promotion_score": promotion_score,
        "validation": {
            "validate_passed": validated,
            "compile_passed": validated,
            "backward_passed": validated,
        },
    }
    if component_descriptor is not None:
        out["component_descriptor"] = component_descriptor
    return out


def _mark_math_sweep_validation(
    validation: dict,
    *,
    passed: bool,
    variant_selected: bool = False,
) -> None:
    validation.update(
        {
            "math_physics_review_passed": True,
            "math_sweep_version": "dynamic_math_sweep_v1",
            "math_sweep_required_for_ready": True,
            "math_sweep_passed": passed,
            "math_sweep_run_id": "unit:sweep",
            "math_sweep_variant_count": 3,
            "math_sweep_selected_variant_id": "unit:parent",
            "math_sweep_selected_family": "parent",
            "math_sweep_selected_transform": "identity",
            "math_sweep_selected_axes": {"op_math_family": "parent"},
            "math_sweep_score": 0.0,
            "math_sweep_selection_reason": (
                "selected_variant"
                if variant_selected
                else "parent_retained_after_sweep"
            ),
            "math_sweep_descriptor_delta": {
                "long_range_reach": 0.0,
                "content_dependence": 0.0,
            },
            "math_variant_selected": variant_selected,
            "math_variant_family": "parent",
            "math_variant_transform": "identity",
            "math_variant_axes": {"op_math_family": "parent"},
            "math_variant_score": 0.0,
            "math_variant_failure_reason": None,
            "math_variant_delta_long_range_reach": 0.0,
            "math_variant_delta_content_dependence": 0.0,
        }
    )


def _write_candidates(path: Path, candidates: list[dict]) -> Path:
    path.write_text(
        json.dumps({"ready_for_registration": candidates}),
        encoding="utf-8",
    )
    return path


def test_dynamic_template_loader_filters_and_unique_ids(tmp_path: Path) -> None:
    artifact = _write_candidates(
        tmp_path / "validated.json",
        [
            _candidate(
                "mined_relu_variant_block",
                ["linear_proj", "relu", "add"],
                promotion_score=9,
            ),
            _candidate(
                "mined_relu_variant_block",
                ["linear_proj", "relu", "mul"],
                promotion_score=5,
                component_descriptor={"has_multi_mixer": False},
            ),
            _candidate(
                "bad_terminal_signal",
                ["linear_proj", "relu", "token_class_proj"],
                promotion_score=50,
            ),
            _candidate("bad_unknown", ["missing_op"], promotion_score=99),
            _candidate("bad_unvalidated", ["linear_proj", "relu"], validated=False),
        ],
    )

    candidates = load_dynamic_template_candidates(
        artifact, max_candidates=8, min_lowered_ops=1
    )

    assert len(candidates) == 2
    assert candidates[0].weight == 9
    assert candidates[0].template_id.startswith("dynamic_mined_relu_variant_block_")
    assert candidates[0].template_id != candidates[1].template_id
    assert candidates[0].chain == ("linear_proj", "relu", "add")
    assert candidates[0].lowered_op_count == 4
    assert candidates[1].component_descriptor == {"has_multi_mixer": False}


def test_dynamic_template_loader_rejects_failed_math_physics_review(
    tmp_path: Path,
) -> None:
    passed = _candidate(
        "reviewed_good",
        ["linear_proj", "selective_scan", "add"],
        promotion_score=4,
    )
    passed["validation"]["math_physics_review_passed"] = True
    failed = _candidate(
        "reviewed_bad",
        ["linear_proj", "selective_scan", "add"],
        promotion_score=9,
    )
    failed["validation"]["math_physics_review_passed"] = False
    artifact = _write_candidates(tmp_path / "reviewed.json", [failed, passed])

    candidates = load_dynamic_template_candidates(
        artifact, max_candidates=8, min_lowered_ops=1
    )

    assert len(candidates) == 1
    assert candidates[0].display_name == "reviewed_good"


def test_dynamic_template_loader_rejects_failed_math_sweep(
    tmp_path: Path,
) -> None:
    passed = _candidate(
        "sweep_parent_retained",
        ["linear_proj", "selective_scan", "add"],
        promotion_score=4,
    )
    _mark_math_sweep_validation(passed["validation"], passed=True)
    failed = _candidate(
        "sweep_failed",
        ["linear_proj", "selective_scan", "add"],
        promotion_score=9,
    )
    _mark_math_sweep_validation(failed["validation"], passed=False)
    artifact = _write_candidates(tmp_path / "sweep.json", [failed, passed])

    candidates = load_dynamic_template_candidates(
        artifact, max_candidates=8, min_lowered_ops=1
    )

    assert len(candidates) == 1
    assert candidates[0].display_name == "sweep_parent_retained"
    assert candidates[0].validation["math_variant_selected"] is False


def test_dynamic_template_loader_accepts_legacy_without_math_sweep(
    tmp_path: Path,
) -> None:
    artifact = _write_candidates(
        tmp_path / "legacy.json",
        [
            _candidate(
                "legacy_good",
                ["linear_proj", "selective_scan", "add"],
                promotion_score=4,
            )
        ],
    )

    candidates = load_dynamic_template_candidates(
        artifact, max_candidates=8, min_lowered_ops=1
    )

    assert len(candidates) == 1
    assert candidates[0].display_name == "legacy_good"
    assert "math_sweep_passed" not in candidates[0].validation


def test_dynamic_template_loader_rejects_micro_chains_by_default(
    tmp_path: Path,
) -> None:
    artifact = _write_candidates(
        tmp_path / "validated.json",
        [_candidate("micro_chain", ["linear_proj", "relu", "add"])],
    )

    assert load_dynamic_template_candidates(artifact) == ()


def test_dynamic_template_selection_weights_include_lowering_multipliers(
    tmp_path: Path,
) -> None:
    artifact = _write_candidates(
        tmp_path / "validated.json",
        [
            _candidate(
                "restore",
                ["rmsnorm", "latent_attention_compressor", "linear_proj", "add"],
                component_descriptor={"lowering": "mixer_sidecar_restore_v1"},
            ),
            _candidate(
                "router",
                ["linear_proj", "matmul", "linear_proj", "gather_topk"],
                component_descriptor={"lowering": "router_lane_blend_v1"},
            ),
        ],
    )
    candidates = load_dynamic_template_candidates(artifact, min_lowered_ops=1)

    weights = _candidate_selection_weights(candidates, strength=1.0)

    assert weights[0] > weights[1]


def test_dynamic_template_lowering_records_metadata(tmp_path: Path) -> None:
    artifact = _write_candidates(
        tmp_path / "validated.json",
        [
            _candidate(
                "chain_block",
                ["linear_proj", "relu", "add", "layernorm"],
                component_descriptor={
                    "has_multi_mixer": False,
                    "slot_plan": [
                        {
                            "slot_index": 0,
                            "slot_classes": ["dynamic_role:project", "dynamic_step"],
                        },
                        {
                            "slot_index": 1,
                            "slot_classes": ["dynamic_role:activate", "dynamic_step"],
                        },
                        {
                            "slot_index": 2,
                            "slot_classes": ["dynamic_role:residual", "dynamic_step"],
                        },
                        {
                            "slot_index": 3,
                            "slot_classes": ["dynamic_role:normalize", "dynamic_step"],
                        },
                    ],
                },
            )
        ],
    )
    candidate = load_dynamic_template_candidates(artifact, min_lowered_ops=1)[0]
    graph = ComputationGraph(model_dim=64)
    input_id = graph.add_input()

    tail = apply_dynamic_template_candidate(
        graph, input_id, random.Random(1), candidate
    )

    assert tail in graph.nodes
    assert graph.metadata["templates_used"] == [candidate.template_id]
    assert graph.metadata["dynamic_templates_used"][0]["chain"] == [
        "linear_proj",
        "relu",
        "add",
        "layernorm",
    ]
    assert "component_descriptor" not in graph.metadata["dynamic_templates_used"][0]
    assert graph.metadata["dynamic_components_used"][0]["chain"] == [
        "linear_proj",
        "relu",
        "add",
        "layernorm",
    ]
    assert (
        graph.metadata["dynamic_components_used"][0]["component_descriptor"][
            "has_multi_mixer"
        ]
        is False
    )
    slots = graph.metadata["template_slot_usage"]
    assert [slot["selected_motif"] for slot in slots] == [
        "linear_proj",
        "relu",
        "add",
        "layernorm",
    ]
    assert slots[0]["slot_classes"] == ["dynamic_role:project", "dynamic_step"]
    assert slots[2]["selected_motif_class"] == "dynamic_op_arity2"


def test_dynamic_template_usage_records_math_sweep_summary(tmp_path: Path) -> None:
    raw = _candidate(
        "sweep_chain_block",
        ["linear_proj", "relu", "add", "layernorm"],
        component_descriptor={"has_multi_mixer": False},
    )
    _mark_math_sweep_validation(raw["validation"], passed=True)
    artifact = _write_candidates(tmp_path / "sweep_validated.json", [raw])
    candidate = load_dynamic_template_candidates(artifact, min_lowered_ops=1)[0]
    graph = ComputationGraph(model_dim=64)
    input_id = graph.add_input()

    apply_dynamic_template_candidate(graph, input_id, random.Random(1), candidate)

    template_sweep = graph.metadata["dynamic_templates_used"][0]["math_sweep"]
    component_sweep = graph.metadata["dynamic_components_used"][0]["math_sweep"]
    assert template_sweep == component_sweep
    assert template_sweep["version"] == "dynamic_math_sweep_v1"
    assert template_sweep["passed"] is True
    assert template_sweep["variant_selected"] is False
    assert template_sweep["selected_family"] == "parent"
    assert template_sweep["selected_axes"] == {"op_math_family": "parent"}
    assert template_sweep["descriptor_delta"]["long_range_reach"] == 0.0
    assert "component_descriptor" not in graph.metadata["dynamic_templates_used"][0]


def test_dynamic_template_can_lower_trunk_sidecar_component(tmp_path: Path) -> None:
    chain = [
        "latent_attention_compressor",
        "linear_proj",
        "conv1d_seq",
        "silu",
        "rmsnorm",
        "selective_scan",
        "add",
        "add",
    ]
    artifact = _write_candidates(
        tmp_path / "validated.json",
        [
            _candidate(
                "branch_block",
                chain,
                component_descriptor={
                    "has_multi_mixer": True,
                    "lowering": "trunk_sidecar_merge_v1",
                    "branch_plan": {
                        "trunk_indices": [0, 1],
                        "sidecar_indices": [2, 3, 4, 5],
                        "merge_op": "add",
                        "post_merge_norm": True,
                        "residual_output": True,
                    },
                    "slot_plan": [
                        {
                            "slot_index": idx,
                            "slot_classes": ["dynamic_role:mix", "dynamic_step"],
                        }
                        for idx in range(len(chain))
                    ],
                },
            )
        ],
    )
    candidate = load_dynamic_template_candidates(artifact, min_lowered_ops=1)[0]
    graph = ComputationGraph(model_dim=64)
    input_id = graph.add_input()

    tail = apply_dynamic_template_candidate(
        graph, input_id, random.Random(1), candidate
    )

    assert graph.nodes[tail].op_name == "add"
    norm_id = next(
        node.id
        for node in graph.nodes.values()
        if node.op_name == "rmsnorm" and node.input_ids == [input_id]
    )
    branch_children = [
        node.op_name for node in graph.nodes.values() if node.input_ids == [norm_id]
    ]
    assert "latent_attention_compressor" in branch_children
    assert "conv1d_seq" in branch_children
    branch_merge = [
        node
        for node in graph.nodes.values()
        if node.op_name == "add" and set(node.input_ids) != {input_id}
    ]
    assert any(len(node.input_ids) == 2 for node in branch_merge)
    slots = graph.metadata["template_slot_usage"]
    assert {slot["slot_index"] for slot in slots} == {0, 1, 2, 3, 4, 5}
    assert any(".trunk.step0" in slot["slot_key"] for slot in slots)
    assert any(".sidecar.step2" in slot["slot_key"] for slot in slots)
    used_component = graph.metadata["dynamic_components_used"][0]
    assert used_component["lowering"] == "trunk_sidecar_merge_v1"
    assert used_component["component_descriptor"]["branch_plan"]["trunk_indices"] == [
        0,
        1,
    ]


def test_dynamic_template_can_lower_mixer_restore_sidecar_component(
    tmp_path: Path,
) -> None:
    chain = [
        "rmsnorm",
        "latent_attention_compressor",
        "linear_proj",
        "add",
        "layernorm",
        "conv1d_seq",
        "swiglu_mlp",
        "add",
    ]
    artifact = _write_candidates(
        tmp_path / "validated.json",
        [
            _candidate(
                "restore_branch_block",
                chain,
                component_descriptor={
                    "has_multi_mixer": False,
                    "lowering": "mixer_sidecar_restore_v1",
                    "branch_plan": {
                        "trunk_indices": [1, 2],
                        "sidecar_indices": [4, 5, 6],
                        "merge_op": "add",
                        "post_merge_norm": True,
                        "residual_output": True,
                    },
                    "slot_plan": [
                        {
                            "slot_index": idx,
                            "slot_classes": ["dynamic_role:mix", "dynamic_step"],
                        }
                        for idx in range(len(chain))
                    ],
                },
            )
        ],
    )
    candidate = load_dynamic_template_candidates(artifact, min_lowered_ops=1)[0]
    graph = ComputationGraph(model_dim=64)
    input_id = graph.add_input()

    tail = apply_dynamic_template_candidate(
        graph, input_id, random.Random(1), candidate
    )

    assert graph.nodes[tail].op_name == "add"
    slots = graph.metadata["template_slot_usage"]
    assert {slot["slot_index"] for slot in slots} == {1, 2, 4, 5, 6}
    assert any(".trunk.step1" in slot["slot_key"] for slot in slots)
    assert any(".sidecar.step4" in slot["slot_key"] for slot in slots)
    used_component = graph.metadata["dynamic_components_used"][0]
    assert used_component["lowering"] == "mixer_sidecar_restore_v1"


def test_dynamic_template_can_lower_router_lane_blend_component(
    tmp_path: Path,
) -> None:
    chain = [
        "add",
        "rmsnorm",
        "linear_proj",
        "matmul",
        "linear_proj",
        "gather_topk",
        "swiglu_mlp",
        "add",
    ]
    artifact = _write_candidates(
        tmp_path / "validated.json",
        [
            _candidate(
                "router_lane_block",
                chain,
                component_descriptor={
                    "has_multi_mixer": False,
                    "lowering": "router_lane_blend_v1",
                    "branch_plan": {
                        "value_project_index": 2,
                        "matmul_index": 3,
                        "score_project_index": 4,
                        "route_index": 5,
                        "gate_index": 6,
                        "blend_op": "gated_lane_blend",
                        "post_merge_norm": True,
                        "residual_output": True,
                    },
                    "slot_plan": [
                        {
                            "slot_index": idx,
                            "slot_classes": ["dynamic_role:route", "dynamic_step"],
                        }
                        for idx in range(len(chain))
                    ],
                },
            )
        ],
    )
    candidate = load_dynamic_template_candidates(artifact, min_lowered_ops=1)[0]
    graph = ComputationGraph(model_dim=64)
    input_id = graph.add_input()

    tail = apply_dynamic_template_candidate(
        graph, input_id, random.Random(1), candidate
    )

    assert graph.nodes[tail].op_name == "add"
    op_names = [node.op_name for node in graph.nodes.values()]
    assert "matmul" in op_names
    assert "gather_topk" in op_names
    assert "gated_lane_blend" in op_names
    slots = graph.metadata["template_slot_usage"]
    assert {slot["slot_index"] for slot in slots} == {2, 3, 4, 5, 6}
    assert all(".router." in slot["slot_key"] for slot in slots)
    used_component = graph.metadata["dynamic_components_used"][0]
    assert used_component["lowering"] == "router_lane_blend_v1"


def test_generate_layer_graph_can_use_dynamic_candidates(tmp_path: Path) -> None:
    artifact = _write_candidates(
        tmp_path / "validated.json",
        [_candidate("chain_block", ["linear_proj", "relu", "add", "layernorm"])],
    )
    config = GrammarConfig(
        model_dim=64,
        max_ops=16,
        composition_depth=1,
        routing_mandatory=False,
        use_db_weights=False,
        use_dynamic_template_candidates=True,
        dynamic_template_candidate_path=str(artifact),
        dynamic_template_candidate_prob=1.0,
        dynamic_template_min_lowered_ops=1,
        residual_prob=0.0,
        freq_domain_prob=0.0,
    )

    graph = generate_layer_graph(config, seed=3)

    used = graph.metadata["dynamic_templates_used"]
    assert used[0]["display_name"] == "chain_block"
    assert graph.metadata["templates_used"][0] == used[0]["template_id"]
    components_used = graph.metadata["dynamic_components_used"]
    assert components_used[0]["template_id"] == used[0]["template_id"]
    assert "component_descriptor" in components_used[0]
    assert graph.metadata["dynamic_template_candidates"]["count"] == 1
    assert graph.metadata["dynamic_template_candidates"]["min_lowered_ops"] == 1


def test_dynamic_template_config_fields_round_trip() -> None:
    config = RunConfig(
        use_dynamic_template_candidates=True,
        dynamic_template_candidate_path="/tmp/dynamic.json",
        dynamic_template_candidate_prob=0.35,
        dynamic_template_candidate_strength=0.75,
        dynamic_template_max_candidates=7,
        dynamic_template_min_lowered_ops=9,
    )

    reconstructed = RunConfig.from_dict(config.to_dict())

    assert reconstructed.use_dynamic_template_candidates is True
    assert reconstructed.dynamic_template_candidate_path == "/tmp/dynamic.json"
    assert reconstructed.dynamic_template_candidate_prob == 0.35
    assert reconstructed.dynamic_template_candidate_strength == 0.75
    assert reconstructed.dynamic_template_max_candidates == 7
    assert reconstructed.dynamic_template_min_lowered_ops == 9


def test_runtime_context_reloads_dynamic_candidates_when_artifact_changes(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "validated.json"

    def write_candidates(name: str, chain: list[str], mtime_ns: int) -> None:
        _write_candidates(artifact, [_candidate(name, chain, promotion_score=1)])
        os.utime(artifact, ns=(mtime_ns, mtime_ns))

    write_candidates("chain_a", ["linear_proj", "relu"], 1_000_000_000)
    config = GrammarConfig(
        use_db_weights=False,
        use_dynamic_template_candidates=True,
        dynamic_template_candidate_path=str(artifact),
        dynamic_template_candidate_prob=1.0,
        dynamic_template_min_lowered_ops=1,
    )

    first = runtime_context_for_config(config)
    assert first.dynamic_template_candidates[0].display_name == "chain_a"

    write_candidates("chain_b", ["linear_proj", "gelu"], 2_000_000_000)
    second = runtime_context_for_config(config)
    assert second.dynamic_template_candidates[0].display_name == "chain_b"


def test_dynamic_candidate_choice_can_be_uniform(tmp_path: Path) -> None:
    candidates = load_dynamic_template_candidates(
        _write_candidates(
            tmp_path / "validated.json",
            [
                _candidate("high", ["linear_proj", "relu"], promotion_score=100),
                _candidate("low", ["linear_proj", "gelu"], promotion_score=1),
            ],
        ),
        min_lowered_ops=1,
    )

    draws = {
        choose_dynamic_template_candidate(
            random.Random(seed), candidates, strength=0.0
        ).display_name
        for seed in range(20)
    }
    assert draws == {"high", "low"}
