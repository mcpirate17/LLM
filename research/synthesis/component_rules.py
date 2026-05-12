"""Component-level structural rules for dynamic graph generation.

This module is the shared rule boundary for descriptor-backed components.
It intentionally starts small: rules here are cheap, deterministic checks that
can run in hot generation paths before expensive graph construction. Richer
rules should be mined into compact data artifacts and consumed here rather
than scattered through template callables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .component_rule_engine import (
    ComponentRuleSet,
    evaluate_component_chain_rules,
    load_component_rule_set,
)
from .op_roles import OpRole, get_role
from .primitives import PRIMITIVE_REGISTRY, get_wiring_rule


DEFAULT_MIN_LOWERED_OPS = 8


@dataclass(frozen=True, slots=True)
class ComponentRuleConfig:
    """Config for descriptor/component structural admission checks."""

    min_lowered_ops: int = DEFAULT_MIN_LOWERED_OPS
    min_distinct_roles: int | None = None
    allow_terminal_restricted_consumer: bool | None = None
    max_consecutive_mixers: int | None = None
    rule_set: ComponentRuleSet | None = None


@dataclass(frozen=True, slots=True)
class ComponentRuleViolation:
    """A failed component rule."""

    code: str
    message: str


def estimated_chain_lowered_op_count(ops: Sequence[str]) -> int:
    """Minimum graph ops emitted by dynamic-chain lowering."""
    return 1 + len(ops)


def validate_component_op_chain(
    ops: Iterable[str],
    *,
    config: ComponentRuleConfig | None = None,
) -> tuple[str, ...]:
    """Return violations for a descriptor chain as compact error codes."""
    cfg = config or ComponentRuleConfig()
    rule_set = cfg.rule_set or load_component_rule_set()
    chain = tuple(str(op) for op in ops)
    violations: list[str] = []

    if not chain:
        return ("empty_chain",)

    for op_name in chain:
        prim = PRIMITIVE_REGISTRY.get(op_name)
        if prim is None:
            violations.append(f"unknown_op:{op_name}")
            continue
        if prim.n_inputs not in (1, 2):
            violations.append(f"unsupported_arity:{op_name}:{prim.n_inputs}")

    violations.extend(
        evaluate_component_chain_rules(
            chain,
            lowered_op_count=estimated_chain_lowered_op_count(chain),
            rule_set=rule_set,
            min_lowered_ops_override=int(cfg.min_lowered_ops),
            min_distinct_roles_override=cfg.min_distinct_roles,
            allow_terminal_restricted_consumer_override=(
                cfg.allow_terminal_restricted_consumer
            ),
            max_consecutive_mixers_override=cfg.max_consecutive_mixers,
            terminal_restricted=op_requires_restricted_consumer(chain[-1]),
        )
    )

    return tuple(violations)


def op_requires_restricted_consumer(op_name: str) -> bool:
    """Return True when an op cannot safely be a reusable component tail."""
    rule = get_wiring_rule(op_name)
    return bool(rule and rule.get("valid_consumers"))


def component_role_counts(ops: Iterable[str]) -> dict[str, int]:
    """Return role histogram for report/mining code."""
    counts: dict[str, int] = {}
    for op_name in ops:
        role = get_role(str(op_name)).value
        counts[role] = counts.get(role, 0) + 1
    return counts


def _max_consecutive_role_run(chain: Sequence[str], role: OpRole) -> int:
    max_run = 0
    current = 0
    for op_name in chain:
        if get_role(op_name) is role:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run
