"""Mine meta_analysis.db for unbuilt component property tuples.

Reads ``op_property_catalog`` (60+ declared/empirical columns per op)
and the rollup outcome columns (``eval_count``, ``s1_pass_count``,
``mean_loss``) to:

1. Compute per-axis-value empirical lift (weighted by eval_count).
2. Enumerate the extant tuples — property combinations realized by an
   existing op.
3. Emit candidate tuples — combinations of high-lift axis values that
   have NEVER been realized as a single op. Each emission carries the
   closest-existing-op witness per axis for proposal grounding.

Read-only. No mutation of meta_analysis.db. No coupling to runtime
synthesis registry.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Sequence

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_META_DB = _REPO / "research" / "meta_analysis.db"

DEFAULT_AXES: tuple[str, ...] = (
    "op_algebraic_space",
    "op_spectral_preferred_basis",
    "op_dynamical_memory_length_class",
    "op_dynamical_has_state",
    "op_activation_sparsity_pattern",
    "op_geometric_receptive_field",
)


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


def load_rows(db_path: Path | str = DEFAULT_META_DB) -> list[dict[str, Any]]:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"meta_analysis.db not found at {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM op_property_catalog").fetchall()
    finally:
        conn.close()
    return [{k: r[k] for k in r.keys()} for r in rows]


def compute_axis_lifts(
    rows: Sequence[dict[str, Any]], axes: Sequence[str]
) -> dict[str, dict[Any, AxisLift]]:
    out: dict[str, dict[Any, AxisLift]] = {}
    for axis in axes:
        bucket: dict[Any, list[dict[str, Any]]] = {}
        for row in rows:
            bucket.setdefault(row.get(axis), []).append(row)
        out[axis] = {
            value: _axis_lift(axis, value, members) for value, members in bucket.items()
        }
    return out


def _axis_lift(axis: str, value: Any, members: Sequence[dict[str, Any]]) -> AxisLift:
    total_evals = sum(int(m.get("eval_count") or 0) for m in members)
    total_pass = sum(int(m.get("s1_pass_count") or 0) for m in members)
    pass_rate = (total_pass / total_evals) if total_evals else 0.0
    reps = sorted(members, key=lambda m: int(m.get("eval_count") or 0), reverse=True)[
        :3
    ]
    return AxisLift(
        axis=axis,
        value=value,
        n_ops=len(members),
        total_evals=total_evals,
        total_s1_pass=total_pass,
        pass_rate=pass_rate,
        representative_ops=tuple(str(m["op_name"]) for m in reps),
    )


def extant_tuples(
    rows: Sequence[dict[str, Any]], axes: Sequence[str]
) -> set[tuple[Any, ...]]:
    return {tuple(r.get(a) for a in axes) for r in rows}


def enumerate_candidates(
    axis_lifts: dict[str, dict[Any, AxisLift]],
    extant: set[tuple[Any, ...]],
    axes: Sequence[str],
    *,
    min_axis_n_ops: int = 2,
    min_axis_pass_rate: float = 0.10,
    top_k_values_per_axis: int = 4,
) -> list[CandidateTuple]:
    per_axis_keepers = _keepers_per_axis(
        axis_lifts,
        axes,
        min_axis_n_ops=min_axis_n_ops,
        min_axis_pass_rate=min_axis_pass_rate,
        top_k_values_per_axis=top_k_values_per_axis,
    )
    if not all(per_axis_keepers.get(a) for a in axes):
        return []

    out: list[CandidateTuple] = []
    for combo in product(*[per_axis_keepers[a] for a in axes]):
        tup = tuple(lift.value for lift in combo)
        if tup in extant:
            continue
        out.append(_candidate_from_combo(combo, axes))
    out.sort(key=lambda c: c.predicted_lift, reverse=True)
    return out


def _keepers_per_axis(
    axis_lifts: dict[str, dict[Any, AxisLift]],
    axes: Sequence[str],
    *,
    min_axis_n_ops: int,
    min_axis_pass_rate: float,
    top_k_values_per_axis: int,
) -> dict[str, list[AxisLift]]:
    per_axis: dict[str, list[AxisLift]] = {}
    for axis in axes:
        keepers = [
            lift
            for lift in axis_lifts.get(axis, {}).values()
            if lift.n_ops >= min_axis_n_ops
            and lift.pass_rate >= min_axis_pass_rate
            and lift.value is not None
        ]
        keepers.sort(key=lambda x: x.pass_rate * x.total_evals, reverse=True)
        per_axis[axis] = keepers[:top_k_values_per_axis]
    return per_axis


def _candidate_from_combo(
    combo: tuple[AxisLift, ...], axes: Sequence[str]
) -> CandidateTuple:
    pred = 1.0
    for lift in combo:
        pred *= max(1e-6, lift.pass_rate)
    pred = pred ** (1.0 / len(combo)) if combo else 0.0
    witnesses = tuple(
        lift.representative_ops[0] if lift.representative_ops else "" for lift in combo
    )
    return CandidateTuple(
        tuple_values=tuple((a, lift.value) for a, lift in zip(axes, combo)),
        predicted_lift=pred,
        per_axis_lift=tuple(combo),
        witness_ops=witnesses,
    )


def candidate_to_json(c: CandidateTuple) -> dict[str, Any]:
    return {
        "tuple": [{"axis": a, "value": v} for a, v in c.tuple_values],
        "predicted_lift": round(c.predicted_lift, 4),
        "witness_ops": list(c.witness_ops),
        "per_axis": [
            {
                "axis": lift.axis,
                "value": lift.value,
                "n_ops": lift.n_ops,
                "total_evals": lift.total_evals,
                "pass_rate": round(lift.pass_rate, 4),
                "representative_ops": list(lift.representative_ops),
            }
            for lift in c.per_axis_lift
        ],
    }


def run(
    db_path: Path | str = DEFAULT_META_DB,
    *,
    axes: Sequence[str] = DEFAULT_AXES,
    top_k_values_per_axis: int = 4,
    max_candidates: int = 50,
    min_axis_n_ops: int = 2,
    min_axis_pass_rate: float = 0.10,
) -> dict[str, Any]:
    rows = load_rows(db_path)
    lifts = compute_axis_lifts(rows, axes)
    extant = extant_tuples(rows, axes)
    candidates = enumerate_candidates(
        lifts,
        extant,
        axes,
        min_axis_n_ops=min_axis_n_ops,
        min_axis_pass_rate=min_axis_pass_rate,
        top_k_values_per_axis=top_k_values_per_axis,
    )[:max_candidates]
    return {
        "axes": list(axes),
        "n_rows": len(rows),
        "n_extant_tuples": len(extant),
        "n_candidates_returned": len(candidates),
        "candidates": [candidate_to_json(c) for c in candidates],
    }
