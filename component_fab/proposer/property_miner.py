"""Property-tuple dataclasses for proposal assembly.

Historically this module mined ``meta_analysis.db`` for unbuilt property
tuples (per-axis empirical lift, extant-tuple enumeration, candidate
emission). That mining path and its CLI were retired — the surrogate
(``state.surrogate``) replaced ``predicted_lift`` ranking — leaving only
the ``AxisLift`` / ``CandidateTuple`` contracts that ``spec_generator``
and the improver/proposer assembly paths still build on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_META_DB = _REPO / "research" / "meta_analysis.db"


@dataclass(frozen=True, slots=True)
class AxisLift:
    axis: str
    value: Any
    n_ops: int
    total_evals: int
    total_s1_pass: int
    pass_rate: float
    representative_ops: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CandidateTuple:
    tuple_values: tuple[tuple[str, Any], ...]
    predicted_lift: float
    per_axis_lift: tuple[AxisLift, ...]
    witness_ops: tuple[str, ...]
    # Axes of the witness/host the candidate is derived from. Lets
    # ``synthesis_kind_for_axes`` compute true diffs instead of treating
    # every axis as a diff against an empty anchor — without this, the
    # algebra rule fires on every novel-algebra anchor and the label
    # collapses to ``semiring_swap`` regardless of what actually changed.
    anchor_axes: tuple[tuple[str, Any], ...] = ()
