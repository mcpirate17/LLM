"""Shared primitive-registry helpers for LLM context builders."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple


@lru_cache(maxsize=1)
def grouped_primitive_registry() -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """Return primitive names grouped by category in a stable order."""
    try:
        from ...synthesis.primitives import PRIMITIVE_REGISTRY
    except (ImportError, SystemError):
        from synthesis.primitives import PRIMITIVE_REGISTRY
    by_cat: Dict[str, List[str]] = {}
    for name, op in sorted(PRIMITIVE_REGISTRY.items()):
        category = (
            op.category.value if hasattr(op.category, "value") else str(op.category)
        )
        by_cat.setdefault(category, []).append(name)
    return tuple((category, tuple(names)) for category, names in sorted(by_cat.items()))


def primitive_registry_size() -> int:
    """Return the number of registered primitives."""
    return sum(len(names) for _, names in grouped_primitive_registry())
