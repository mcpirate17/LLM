"""Canonical validator gate-chain vocabulary + chained-population helpers.

Single source of truth for the gate names ``validator/capability.py`` emits
and the ledger analyzers (``gate_calibration``, ``failure_attribution``)
replay. The chained semantics live here once: a candidate *reached* gate G
iff it was not eliminated by any gate strictly before G in the canonical
order, and *passed* G iff it reached G and was not killed by it.
"""

from __future__ import annotations

from typing import Any, Mapping

GATE_SMOKE = "smoke"
GATE_S05_CAUSALITY_STABILITY = "s05_causality_stability"
GATE_ERF_DENSITY = "erf_density"
GATE_NANO_BIND = "nano_bind"

CANONICAL_GATE_ORDER: tuple[str, ...] = (
    GATE_SMOKE,
    GATE_S05_CAUSALITY_STABILITY,
    GATE_ERF_DENSITY,
    GATE_NANO_BIND,
    "ar_easy",
    "ar_medium",
    "ar_hard",
)
SURVIVED = "survived"


def gate_index(gate: str, order: tuple[str, ...] = CANONICAL_GATE_ORDER) -> int:
    """Position in the canonical order; unknown gates sort after the last gate."""
    try:
        return order.index(gate)
    except ValueError:
        return len(order)


def eliminated_by(grade: Mapping[str, Any]) -> str:
    """First-killer gate recorded on a ledger grade record, or ``SURVIVED``."""
    meta = grade.get("metadata") or {}
    e = meta.get("eliminated_by")
    if isinstance(e, str) and e:
        return e
    if grade.get("smoke_pass") is False:
        return GATE_SMOKE
    return SURVIVED


def reached(
    eliminated: str, gate: str, order: tuple[str, ...] = CANONICAL_GATE_ORDER
) -> bool:
    """True if a candidate eliminated at ``eliminated`` passed every gate before ``gate``."""
    if eliminated == SURVIVED:
        return True
    return gate_index(eliminated, order) >= gate_index(gate, order)


def passed(
    eliminated: str, gate: str, order: tuple[str, ...] = CANONICAL_GATE_ORDER
) -> bool:
    """True if the candidate reached ``gate`` and was not killed by it."""
    return reached(eliminated, gate, order) and eliminated != gate
