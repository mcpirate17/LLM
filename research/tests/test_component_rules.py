from research.synthesis.component_rules import (
    ComponentRuleConfig,
    component_role_counts,
    estimated_chain_lowered_op_count,
    validate_component_op_chain,
)


def test_component_rules_enforce_min_lowered_ops() -> None:
    violations = validate_component_op_chain(
        ["rmsnorm", "linear_proj", "gelu"],
        config=ComponentRuleConfig(min_lowered_ops=8, min_distinct_roles=1),
    )

    assert "too_small:4<min8" in violations


def test_component_rules_reject_restricted_terminal_signal() -> None:
    violations = validate_component_op_chain(
        [
            "rmsnorm",
            "linear_proj",
            "gelu",
            "linear_proj",
            "relu",
            "add",
            "token_class_proj",
        ],
        config=ComponentRuleConfig(min_lowered_ops=1, min_distinct_roles=1),
    )

    assert "restricted_terminal:token_class_proj" in violations


def test_component_rules_accept_larger_mixed_chain() -> None:
    ops = [
        "rmsnorm",
        "selective_scan",
        "linear_proj",
        "gelu",
        "linear_proj",
        "add",
        "rmsnorm",
    ]

    assert estimated_chain_lowered_op_count(ops) == 8
    assert validate_component_op_chain(ops) == ()
    assert component_role_counts(ops)["mix"] == 1


def test_component_rules_flag_unknown_and_arity() -> None:
    violations = validate_component_op_chain(
        ["relu", "missing_op"],
        config=ComponentRuleConfig(min_lowered_ops=1, min_distinct_roles=1),
    )

    assert "unknown_op:missing_op" in violations
