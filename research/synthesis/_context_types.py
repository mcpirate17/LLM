"""Context rules — type definitions and constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet


class SearchMode(Enum):
    __slots__ = ()
    GENERAL = "general"


@dataclass(slots=True, frozen=True)
class ContextRule:
    """Machine-actionable placement constraint for a single op."""

    search_mode: SearchMode
    # Concrete op names that are forbidden as direct predecessors.
    forbidden_predecessors: FrozenSet[str] = field(default_factory=frozenset)
    # Concrete op names that are forbidden as direct successors.
    forbidden_successors: FrozenSet[str] = field(default_factory=frozenset)
    # If True, the op must sit inside a residual bypass (add consuming same input).
    requires_residual_context: bool = False


CONTEXT_CLASS_GENERAL = "general-use"
CONTEXT_CLASS_RESTRICTED = "restricted-use"
CONTEXT_CLASS_STRUCTURAL = "structural"
CONTEXT_CLASS_REHAB = "rehab"
