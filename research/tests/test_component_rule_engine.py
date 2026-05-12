import json
from pathlib import Path

from research.synthesis.component_rule_engine import (
    component_slot_plan,
    evaluate_component_chain_rules,
    load_component_rule_set,
)


def test_component_rule_engine_loads_file_backed_rules() -> None:
    rules = load_component_rule_set()

    assert rules.min_lowered_ops == 8
    assert ("clifford_attention", "linear_proj") in rules.blocked_op_pairs
    assert ("softmax_attention", "linear_proj") not in rules.blocked_op_pairs
    assert ("rope_rotate", "softmax_attention") not in rules.blocked_op_pairs
    assert ("linear_proj", "latent_attention_compressor") not in rules.blocked_op_pairs
    assert ("rmsnorm", "add") not in rules.blocked_op_pairs
    assert ("rope_rotate", "softmax_attention") in rules.preferred_op_pairs
    assert ("linear_proj", "latent_attention_compressor") in rules.preferred_op_pairs
    assert (
        "latent_attention_compressor",
        "softmax_attention",
    ) in rules.preferred_multi_mixer_pairs
    assert ("softmax_attention", "state_space") in rules.preferred_multi_mixer_pairs
    assert "swiglu_mlp" in rules.restore_ops
    assert "gated_proj_up" in rules.restore_ops
    assert "bottleneck_proj_up" in rules.restore_ops
    assert "mix" in rules.recursion_required_predecessor_roles
    assert "confidence_token_gate" in rules.terminal_blocked_ops
    assert "latent_attention_compressor" in rules.compression_ops
    assert "depth_weighted_proj" in rules.recursion_ops


def test_component_rule_engine_blocks_file_backed_pairs() -> None:
    violations = evaluate_component_chain_rules(
        [
            "rmsnorm",
            "clifford_attention",
            "linear_proj",
            "gelu",
            "linear_proj",
            "add",
            "rmsnorm",
        ],
        lowered_op_count=8,
    )

    assert "blocked_pair:clifford_attention->linear_proj" in violations


def test_component_rule_engine_allows_winner_shaped_softened_pairs() -> None:
    violations = evaluate_component_chain_rules(
        [
            "rmsnorm",
            "linear_proj",
            "rope_rotate",
            "softmax_attention",
            "rmsnorm",
            "linear_proj",
            "latent_attention_compressor",
            "add",
        ],
        lowered_op_count=9,
    )

    assert not any(v.startswith("blocked_pair:") for v in violations)


def test_component_rule_engine_allows_recursion_after_mixer() -> None:
    violations = evaluate_component_chain_rules(
        [
            "rmsnorm",
            "softmax_attention",
            "depth_weighted_proj",
            "rmsnorm",
            "linear_proj",
            "gelu",
            "linear_proj",
            "add",
        ],
        lowered_op_count=9,
    )

    assert (
        "bad_recursion_predecessor:softmax_attention->depth_weighted_proj"
        not in violations
    )


def test_component_rule_engine_builds_role_aware_slot_plan() -> None:
    plan = component_slot_plan(
        ["rmsnorm", "latent_attention_compressor", "depth_weighted_proj", "add"]
    )

    assert plan[1]["slot_classes"] == (
        "dynamic_role:mix",
        "dynamic_step",
        "dynamic_mixer",
        "dynamic_compressor",
    )
    assert "dynamic_recursion" in plan[2]["slot_classes"]


def test_component_rule_engine_reloads_when_rule_file_changes(tmp_path: Path) -> None:
    rule_dir = tmp_path / "rules"
    rule_dir.mkdir()
    (rule_dir / "component_rules_v1.json").write_text(
        json.dumps(
            {
                "schema_version": "component_rules_v1",
                "defaults": {"min_lowered_ops": 11, "min_distinct_roles": 1},
                "blocked_op_pairs": [],
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "mixer_rules_v1.json",
        "compression_rules_v1.json",
        "recursion_rules_v1.json",
    ):
        (rule_dir / name).write_text(
            json.dumps({"schema_version": name.replace(".json", "")}),
            encoding="utf-8",
        )

    assert load_component_rule_set(rule_dir).min_lowered_ops == 11
