import json
import os
import random
from pathlib import Path

from research.scientist.runner._types import RunConfig
from research.synthesis.dynamic_template_registry import (
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


def test_dynamic_template_loader_rejects_micro_chains_by_default(
    tmp_path: Path,
) -> None:
    artifact = _write_candidates(
        tmp_path / "validated.json",
        [_candidate("micro_chain", ["linear_proj", "relu", "add"])],
    )

    assert load_dynamic_template_candidates(artifact) == ()


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
    assert (
        graph.metadata["dynamic_templates_used"][0]["component_descriptor"][
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
