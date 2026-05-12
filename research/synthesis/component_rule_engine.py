"""File-backed component rule engine for dynamic architecture generation.

Rules are intentionally data files under ``research/synthesis/rules``. This
module owns fast loading, normalization, and deterministic chain evaluation so
templates, slots, and dynamic component builders do not grow separate rule
implementations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .op_roles import OpRole, get_role


DEFAULT_RULE_DIR = Path(__file__).resolve().parent / "rules"


@dataclass(frozen=True, slots=True)
class ComponentRuleSet:
    """Normalized rule files used by the dynamic component engine."""

    schema_versions: tuple[str, ...]
    min_lowered_ops: int
    min_distinct_roles: int
    allow_terminal_restricted_consumer: bool
    max_consecutive_mixers: int
    blocked_op_pairs: frozenset[tuple[str, str]]
    preferred_op_pairs: frozenset[tuple[str, str]]
    terminal_blocked_roles: frozenset[str]
    terminal_blocked_ops: frozenset[str]
    required_roles: frozenset[str]
    preferred_multi_mixer_pairs: frozenset[tuple[str, str]]
    blocked_multi_mixer_pairs: frozenset[tuple[str, str]]
    compression_ops: frozenset[str]
    restore_ops: frozenset[str]
    compression_must_restore_before_terminal: bool
    compression_max_unrestored_tail_distance: int
    recursion_ops: frozenset[str]
    recursion_required_predecessor_roles: frozenset[str]
    recursion_required_successor_roles: frozenset[str]
    max_recursion_ops_per_component: int
    allow_terminal_recursion: bool


def load_component_rule_set(
    rule_dir: str | Path = DEFAULT_RULE_DIR,
) -> ComponentRuleSet:
    """Load and cache normalized component rules from JSON files."""
    directory = Path(rule_dir)
    token = _rule_dir_token(directory)
    return _load_component_rule_set_cached(str(directory), token)


def evaluate_component_chain_rules(
    ops: Iterable[str],
    *,
    lowered_op_count: int,
    rule_set: ComponentRuleSet | None = None,
    min_lowered_ops_override: int | None = None,
    min_distinct_roles_override: int | None = None,
    allow_terminal_restricted_consumer_override: bool | None = None,
    max_consecutive_mixers_override: int | None = None,
    terminal_restricted: bool = False,
) -> tuple[str, ...]:
    """Evaluate file-backed component rules and return compact violations."""
    rules = rule_set or load_component_rule_set()
    chain = tuple(str(op) for op in ops)
    if not chain:
        return ("empty_chain",)

    min_lowered_ops = (
        int(min_lowered_ops_override)
        if min_lowered_ops_override is not None
        else rules.min_lowered_ops
    )
    min_distinct_roles = (
        int(min_distinct_roles_override)
        if min_distinct_roles_override is not None
        else rules.min_distinct_roles
    )
    allow_terminal_restricted_consumer = (
        bool(allow_terminal_restricted_consumer_override)
        if allow_terminal_restricted_consumer_override is not None
        else rules.allow_terminal_restricted_consumer
    )
    max_consecutive_mixers = (
        int(max_consecutive_mixers_override)
        if max_consecutive_mixers_override is not None
        else rules.max_consecutive_mixers
    )

    violations: list[str] = []
    if lowered_op_count < max(1, min_lowered_ops):
        violations.append(f"too_small:{lowered_op_count}<min{min_lowered_ops}")

    roles = tuple(get_role(op).value for op in chain)
    role_set = frozenset(roles)
    if len(role_set) < max(1, min_distinct_roles):
        violations.append(f"too_few_roles:{len(role_set)}<min{min_distinct_roles}")

    for required_role in sorted(rules.required_roles):
        if required_role not in role_set:
            violations.append(f"missing_required_role:{required_role}")

    for left, right in zip(chain, chain[1:]):
        if (left, right) in rules.blocked_op_pairs:
            violations.append(f"blocked_pair:{left}->{right}")

    tail = chain[-1]
    tail_role = roles[-1]
    if tail in rules.terminal_blocked_ops or tail_role in rules.terminal_blocked_roles:
        violations.append(f"blocked_terminal:{tail}")
    if terminal_restricted and not allow_terminal_restricted_consumer:
        violations.append(f"restricted_terminal:{tail}")

    max_mixer_run = _max_role_run(roles, OpRole.MIX.value)
    if max_mixer_run > max(1, max_consecutive_mixers):
        violations.append(
            f"too_many_consecutive_mixers:{max_mixer_run}>max{max_consecutive_mixers}"
        )

    mixer_pairs = tuple(
        (left, right)
        for left, right, left_role, right_role in zip(
            chain, chain[1:], roles, roles[1:]
        )
        if left_role == OpRole.MIX.value and right_role == OpRole.MIX.value
    )
    for left, right in mixer_pairs:
        if (left, right) in rules.blocked_multi_mixer_pairs:
            violations.append(f"blocked_multi_mixer_pair:{left}->{right}")

    violations.extend(_compression_violations(chain, rules))
    violations.extend(_recursion_violations(chain, roles, rules))
    return tuple(violations)


def component_slot_plan(
    ops: Iterable[str],
    *,
    rule_set: ComponentRuleSet | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return deterministic role-aware dynamic slot descriptors for a chain."""
    rules = rule_set or load_component_rule_set()
    plan: list[dict[str, Any]] = []
    for index, raw_op in enumerate(ops):
        op_name = str(raw_op)
        role = get_role(op_name).value
        classes = [f"dynamic_role:{role}", "dynamic_step"]
        if role == OpRole.MIX.value:
            classes.append("dynamic_mixer")
        if role == OpRole.ROUTE.value:
            classes.append("dynamic_router")
        if role == OpRole.GATE.value:
            classes.append("dynamic_gate")
        if op_name in rules.compression_ops:
            classes.append("dynamic_compressor")
        if op_name in rules.recursion_ops:
            classes.append("dynamic_recursion")
        if op_name in rules.restore_ops:
            classes.append("dynamic_restore")
        plan.append(
            {
                "slot_index": index,
                "op_name": op_name,
                "role": role,
                "slot_classes": tuple(dict.fromkeys(classes)),
            }
        )
    return tuple(plan)


def _compression_violations(
    chain: Sequence[str],
    rules: ComponentRuleSet,
) -> list[str]:
    if not rules.compression_must_restore_before_terminal:
        return []
    last_compression_idx = -1
    last_restore_idx = -1
    for idx, op_name in enumerate(chain):
        if op_name in rules.compression_ops:
            last_compression_idx = idx
        if op_name in rules.restore_ops:
            last_restore_idx = idx
    if last_compression_idx < 0:
        return []
    if last_restore_idx < last_compression_idx:
        return [f"compression_unrestored:{chain[last_compression_idx]}"]
    tail_distance = len(chain) - 1 - last_restore_idx
    max_distance = max(0, int(rules.compression_max_unrestored_tail_distance))
    if tail_distance > max_distance:
        return [f"compression_restore_too_far:{tail_distance}>max{max_distance}"]
    return []


def _recursion_violations(
    chain: Sequence[str],
    roles: Sequence[str],
    rules: ComponentRuleSet,
) -> list[str]:
    # guardrail: allow-complexity - bounded rule scan over <=8-op component chains.
    indices = [
        idx for idx, op_name in enumerate(chain) if op_name in rules.recursion_ops
    ]
    if not indices:
        return []
    violations: list[str] = []
    if len(indices) > max(1, int(rules.max_recursion_ops_per_component)):
        violations.append(
            f"too_many_recursion_ops:{len(indices)}>max{int(rules.max_recursion_ops_per_component)}"
        )
    for idx in indices:
        op_name = chain[idx]
        if idx == len(chain) - 1 and not rules.allow_terminal_recursion:
            violations.append(f"terminal_recursion:{op_name}")
        if idx > 0:
            prev_role = roles[idx - 1]
            if prev_role not in rules.recursion_required_predecessor_roles:
                violations.append(
                    f"bad_recursion_predecessor:{chain[idx - 1]}->{op_name}"
                )
        if idx + 1 < len(chain):
            next_role = roles[idx + 1]
            if next_role not in rules.recursion_required_successor_roles:
                violations.append(
                    f"bad_recursion_successor:{op_name}->{chain[idx + 1]}"
                )
    return violations


def _max_role_run(roles: Sequence[str], role: str) -> int:
    max_run = 0
    current = 0
    for item in roles:
        if item == role:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _rule_dir_token(directory: Path) -> tuple[tuple[str, int | None], ...]:
    # guardrail: allow-complexity - four small JSON files, cached by mtime token.
    names = (
        "component_rules_v1.json",
        "mixer_rules_v1.json",
        "compression_rules_v1.json",
        "recursion_rules_v1.json",
    )
    token: list[tuple[str, int | None]] = []
    for name in names:
        path = directory / name
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            mtime_ns = None
        token.append((name, mtime_ns))
    return tuple(token)


@lru_cache(maxsize=16)
def _load_component_rule_set_cached(
    directory: str,
    token: tuple[tuple[str, int | None], ...],
) -> ComponentRuleSet:
    del token
    rule_dir = Path(directory)
    component = _read_rule_file(rule_dir / "component_rules_v1.json")
    mixer = _read_rule_file(rule_dir / "mixer_rules_v1.json")
    compression = _read_rule_file(rule_dir / "compression_rules_v1.json")
    recursion = _read_rule_file(rule_dir / "recursion_rules_v1.json")

    defaults = _mapping(component.get("defaults"))
    return ComponentRuleSet(
        schema_versions=tuple(
            str(doc.get("schema_version") or "")
            for doc in (component, mixer, compression, recursion)
        ),
        min_lowered_ops=int(defaults.get("min_lowered_ops", 8)),
        min_distinct_roles=int(defaults.get("min_distinct_roles", 2)),
        allow_terminal_restricted_consumer=bool(
            defaults.get("allow_terminal_restricted_consumer", False)
        ),
        max_consecutive_mixers=int(
            mixer.get("max_consecutive_mixers")
            or defaults.get("max_consecutive_mixers", 2)
        ),
        blocked_op_pairs=_pair_set_from_records(component.get("blocked_op_pairs")),
        preferred_op_pairs=_pair_set_from_records(component.get("preferred_op_pairs")),
        terminal_blocked_roles=frozenset(
            str(item) for item in _sequence(component.get("terminal_blocked_roles"))
        ),
        terminal_blocked_ops=frozenset(
            str(item) for item in _sequence(component.get("terminal_blocked_ops"))
        ),
        required_roles=frozenset(
            str(item) for item in _sequence(component.get("required_roles"))
        ),
        preferred_multi_mixer_pairs=_pair_set(mixer.get("preferred_multi_mixer_pairs")),
        blocked_multi_mixer_pairs=_pair_set(mixer.get("blocked_multi_mixer_pairs")),
        compression_ops=frozenset(
            str(item) for item in _sequence(compression.get("compression_ops"))
        ),
        restore_ops=frozenset(
            str(item) for item in _sequence(compression.get("restore_ops"))
        ),
        compression_must_restore_before_terminal=bool(
            compression.get("must_restore_before_terminal", True)
        ),
        compression_max_unrestored_tail_distance=int(
            compression.get("max_unrestored_tail_distance", 4)
        ),
        recursion_ops=frozenset(
            str(item) for item in _sequence(recursion.get("recursion_ops"))
        ),
        recursion_required_predecessor_roles=frozenset(
            str(item) for item in _sequence(recursion.get("required_predecessor_roles"))
        ),
        recursion_required_successor_roles=frozenset(
            str(item) for item in _sequence(recursion.get("required_successor_roles"))
        ),
        max_recursion_ops_per_component=int(
            recursion.get("max_recursion_ops_per_component", 1)
        ),
        allow_terminal_recursion=bool(recursion.get("allow_terminal_recursion", False)),
    )


def _read_rule_file(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        return ()
    if isinstance(value, Sequence):
        return tuple(value)
    return ()


def _pair_set(value: Any) -> frozenset[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for item in _sequence(value):
        if (
            isinstance(item, Sequence)
            and not isinstance(item, (str, bytes))
            and len(item) == 2
        ):
            out.add((str(item[0]), str(item[1])))
    return frozenset(out)


def _pair_set_from_records(value: Any) -> frozenset[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for item in _sequence(value):
        if isinstance(item, Mapping):
            pair = item.get("pair")
        else:
            pair = item
        if (
            isinstance(pair, Sequence)
            and not isinstance(pair, (str, bytes))
            and len(pair) == 2
        ):
            out.add((str(pair[0]), str(pair[1])))
    return frozenset(out)
