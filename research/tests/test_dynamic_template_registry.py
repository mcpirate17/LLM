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
) -> dict:
    return {
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
            _candidate("mined_relu_variant_block", ["relu", "add"], promotion_score=9),
            _candidate("mined_relu_variant_block", ["relu", "mul"], promotion_score=5),
            _candidate(
                "bad_terminal_signal",
                ["relu", "token_class_proj"],
                promotion_score=50,
            ),
            _candidate("bad_unknown", ["missing_op"], promotion_score=99),
            _candidate("bad_unvalidated", ["relu"], validated=False),
        ],
    )

    candidates = load_dynamic_template_candidates(artifact, max_candidates=8)

    assert len(candidates) == 2
    assert candidates[0].weight == 9
    assert candidates[0].template_id.startswith("dynamic_mined_relu_variant_block_")
    assert candidates[0].template_id != candidates[1].template_id
    assert candidates[0].chain == ("relu", "add")


def test_dynamic_template_lowering_records_metadata(tmp_path: Path) -> None:
    artifact = _write_candidates(
        tmp_path / "validated.json",
        [_candidate("chain_block", ["relu", "add", "layernorm"])],
    )
    candidate = load_dynamic_template_candidates(artifact)[0]
    graph = ComputationGraph(model_dim=64)
    input_id = graph.add_input()

    tail = apply_dynamic_template_candidate(
        graph, input_id, random.Random(1), candidate
    )

    assert tail in graph.nodes
    assert graph.metadata["templates_used"] == [candidate.template_id]
    assert graph.metadata["dynamic_templates_used"][0]["chain"] == [
        "relu",
        "add",
        "layernorm",
    ]
    slots = graph.metadata["template_slot_usage"]
    assert [slot["selected_motif"] for slot in slots] == ["relu", "add", "layernorm"]
    assert slots[1]["selected_motif_class"] == "dynamic_op_arity2"


def test_generate_layer_graph_can_use_dynamic_candidates(tmp_path: Path) -> None:
    artifact = _write_candidates(
        tmp_path / "validated.json",
        [_candidate("chain_block", ["relu", "add", "layernorm"])],
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
        residual_prob=0.0,
        freq_domain_prob=0.0,
    )

    graph = generate_layer_graph(config, seed=3)

    used = graph.metadata["dynamic_templates_used"]
    assert used[0]["display_name"] == "chain_block"
    assert graph.metadata["templates_used"][0] == used[0]["template_id"]
    assert graph.metadata["dynamic_template_candidates"]["count"] == 1


def test_dynamic_template_config_fields_round_trip() -> None:
    config = RunConfig(
        use_dynamic_template_candidates=True,
        dynamic_template_candidate_path="/tmp/dynamic.json",
        dynamic_template_candidate_prob=0.35,
        dynamic_template_candidate_strength=0.75,
        dynamic_template_max_candidates=7,
    )

    reconstructed = RunConfig.from_dict(config.to_dict())

    assert reconstructed.use_dynamic_template_candidates is True
    assert reconstructed.dynamic_template_candidate_path == "/tmp/dynamic.json"
    assert reconstructed.dynamic_template_candidate_prob == 0.35
    assert reconstructed.dynamic_template_candidate_strength == 0.75
    assert reconstructed.dynamic_template_max_candidates == 7


def test_runtime_context_reloads_dynamic_candidates_when_artifact_changes(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "validated.json"

    def write_candidates(name: str, chain: list[str], mtime_ns: int) -> None:
        _write_candidates(artifact, [_candidate(name, chain, promotion_score=1)])
        os.utime(artifact, ns=(mtime_ns, mtime_ns))

    write_candidates("chain_a", ["relu"], 1_000_000_000)
    config = GrammarConfig(
        use_db_weights=False,
        use_dynamic_template_candidates=True,
        dynamic_template_candidate_path=str(artifact),
        dynamic_template_candidate_prob=1.0,
    )

    first = runtime_context_for_config(config)
    assert first.dynamic_template_candidates[0].display_name == "chain_a"

    write_candidates("chain_b", ["gelu"], 2_000_000_000)
    second = runtime_context_for_config(config)
    assert second.dynamic_template_candidates[0].display_name == "chain_b"


def test_dynamic_candidate_choice_can_be_uniform(tmp_path: Path) -> None:
    candidates = load_dynamic_template_candidates(
        _write_candidates(
            tmp_path / "validated.json",
            [
                _candidate("high", ["relu"], promotion_score=100),
                _candidate("low", ["gelu"], promotion_score=1),
            ],
        )
    )

    draws = {
        choose_dynamic_template_candidate(
            random.Random(seed), candidates, strength=0.0
        ).display_name
        for seed in range(20)
    }
    assert draws == {"high", "low"}
