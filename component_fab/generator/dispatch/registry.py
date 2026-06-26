"""Typed dispatch-rule helpers for component_fab generation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from torch import nn


@dataclass(frozen=True, slots=True)
class DispatchRule:
    """One ordered code-generation rule.

    ``handler`` returns a module when the rule matches and ``None`` when the
    generator should fall through to the next rule. Rule ordering is load-bearing;
    keep order changes reviewable by editing the rule table instead of burying
    dispatch order inside ad hoc loops.
    """

    name: str
    handler: Callable[[dict[str, Any]], nn.Module | None]


def dispatch_first(rules: Iterable[DispatchRule], math_axes: dict[str, Any]) -> nn.Module | None:
    """Return the first module produced by an ordered dispatch registry."""

    for rule in rules:
        result = rule.handler(math_axes)
        if result is not None:
            return result
    return None
