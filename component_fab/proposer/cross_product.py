"""Bounded cross-product proposer for parametric physics cells (NM-3).

The dynamic proposer repairs one measured failure at a time; name-free physics
experiments search descriptor targets. This module fills the gap between them:
systematically sample cells of

``address x score_norm x aggregate x algebra x basis``

without enumerating the full product every cycle. Each emitted spec is buildable
through the existing ``physics_atom`` dispatcher and carries the five product
coordinates explicitly for ledger dedupe and downstream analysis.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass
from typing import Any, Mapping

from component_fab.proposer.spec_generator import ProposalSpec, build_spec_from_axes
from component_fab.state.ledger import Ledger
from research.synthesis.parametric_ops import (
    ADDRESS_FAMILIES,
    AGGREGATE_FAMILIES,
    SCORE_NORM_FAMILIES,
)

ALGEBRA_FAMILIES: tuple[str, ...] = (
    "euclidean",
    "parametric_semiring",
    "tropical",
    "padic",
    "clifford",
    "hyperbolic",
)
BASIS_FAMILIES: tuple[str, ...] = (
    "identity",
    "channel",
    "token",
    "content",
    "frequency",
)
_SOFTMAX_SHAPED_SCORE_NORMS = frozenset({"softmax", "sharpen"})


@dataclass(frozen=True, slots=True, order=True)
class CrossProductCell:
    address: str
    score_norm: str
    aggregate: str
    algebra: str
    basis: str

    @property
    def key(self) -> str:
        return "|".join(
            (self.address, self.score_norm, self.aggregate, self.algebra, self.basis)
        )

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-zA-Z0-9_]+", "_", self.key).strip("_").lower()


def _stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def _is_softmax_twin_cell(cell: CrossProductCell) -> bool:
    """Reject cells whose runtime stage is a convex softmax-shaped average."""
    return (
        cell.aggregate == "mean"
        and cell.score_norm in _SOFTMAX_SHAPED_SCORE_NORMS
    )


def _cell_priority(cell: CrossProductCell) -> float:
    priority = 0.0
    if cell.algebra != "euclidean":
        priority += 0.30
    if cell.aggregate == "semiring":
        priority += 0.25
    if cell.score_norm not in _SOFTMAX_SHAPED_SCORE_NORMS:
        priority += 0.20
    if cell.address != "dot":
        priority += 0.10
    if cell.basis in {"token", "content", "frequency"}:
        priority += 0.10
    return priority


def _all_cells() -> list[CrossProductCell]:
    cells = [
        CrossProductCell(
            address=str(address),
            score_norm=str(score_norm),
            aggregate=str(aggregate),
            algebra=str(algebra),
            basis=str(basis),
        )
        for address, score_norm, aggregate, algebra, basis in itertools.product(
            ADDRESS_FAMILIES,
            SCORE_NORM_FAMILIES,
            AGGREGATE_FAMILIES,
            ALGEBRA_FAMILIES,
            BASIS_FAMILIES,
        )
    ]
    return sorted(
        (cell for cell in cells if not _is_softmax_twin_cell(cell)),
        key=lambda cell: (
            -_cell_priority(cell),
            _stable_int("cross_product_order", cell.key),
        ),
    )


def _cell_from_axes(axes: Mapping[str, Any]) -> CrossProductCell | None:
    required = (
        "op_physics_address_family",
        "op_physics_score_norm_family",
        "op_physics_aggregate_family",
        "op_algebraic_space",
        "op_spectral_preferred_basis",
    )
    if not all(axes.get(key) is not None for key in required):
        return None
    return CrossProductCell(
        address=str(axes["op_physics_address_family"]),
        score_norm=str(axes["op_physics_score_norm_family"]),
        aggregate=str(axes["op_physics_aggregate_family"]),
        algebra=str(axes["op_algebraic_space"]),
        basis=str(axes["op_spectral_preferred_basis"]),
    )


def _seen_cells(ledger: Ledger) -> set[str]:
    seen: set[str] = set()
    for entry in ledger.all_entries():
        for metadata in entry.metadata_history:
            axes = metadata.get("math_axes") or {}
            if not isinstance(axes, Mapping):
                continue
            explicit = axes.get("op_cross_product_cell")
            if explicit:
                seen.add(str(explicit))
                continue
            cell = _cell_from_axes(axes)
            if cell is not None:
                seen.add(cell.key)
    return seen


def _basis_axis_for_cell(cell: CrossProductCell) -> str:
    return "token" if cell.basis in {"token", "content", "frequency"} else "channel"


def _atom_kinds_for_cell(cell: CrossProductCell) -> str:
    if cell.basis == "identity":
        return "scan"
    if cell.basis == "channel":
        return "norm+basis+scan"
    return "basis+scan"


def _sparsity_for_cell(cell: CrossProductCell) -> str:
    if cell.score_norm in {"sparsemax", "entmax", "entmax_alpha"}:
        return "top_k"
    if cell.aggregate == "semiring" or cell.score_norm != "softmax":
        return "learned_structured"
    return "dense"


def _axes_for_cell(cell: CrossProductCell, *, cycle: int, rank: int) -> dict[str, Any]:
    seed = _stable_int("physics_cross_product", cycle, cell.key) % 1_000_000
    basis_axis = _basis_axis_for_cell(cell)
    atom_kinds = _atom_kinds_for_cell(cell)
    knob_scale = 0.85 + 0.15 * (rank % 4)
    return {
        "op_search_track": "physics_atom",
        "op_physics_source": "cross_product",
        "op_physics_target": "cross_product_novel_geometry",
        "op_physics_variant": f"xprod{rank:02d}",
        "op_cross_product_cell": cell.key,
        "op_cross_product_priority": round(_cell_priority(cell), 4),
        "op_physics_seed": seed,
        "op_physics_atom_kinds": atom_kinds,
        "op_physics_norm_axis": "token" if basis_axis == "token" else "channel",
        "op_physics_basis_axis": basis_axis,
        "op_physics_address_family": cell.address,
        "op_physics_score_norm_family": cell.score_norm,
        "op_physics_aggregate_family": cell.aggregate,
        "op_physics_knob_scale": round(knob_scale, 4),
        "op_algebraic_space": cell.algebra,
        "op_spectral_preferred_basis": cell.basis,
        "op_dynamical_has_state": 1,
        "op_dynamical_memory_length_class": "O(L)",
        "op_geometric_receptive_field": "global",
        "op_activation_sparsity_pattern": _sparsity_for_cell(cell),
    }


def _spec_for_cell(cell: CrossProductCell, *, cycle: int, rank: int) -> ProposalSpec:
    axes = _axes_for_cell(cell, cycle=cycle, rank=rank)
    rationale = (
        "Bounded physics cross-product cell over "
        "address x score_norm x aggregate x algebra x basis. "
        f"Cell={cell.key}; priority={axes['op_cross_product_priority']:.4f}. "
        "Softmax-shaped mean cells are excluded so the budget targets novel geometry."
    )
    return build_spec_from_axes(
        f"xprod_{cell.slug}",
        axes,
        witness_ops=("physics_cross_product",),
        anchor_axes={},
        notes=(
            "source=physics_cross_product",
            f"cell={cell.key}",
            f"cycle={cycle}",
            rationale,
        ),
        fingerprint_dispatched_axes=True,
        rationale=rationale,
    )


def enumerate_cross_product_specs(
    ledger: Ledger,
    *,
    cycle: int = 0,
    max_specs: int = 12,
) -> list[ProposalSpec]:
    """Emit unseen, buildable cross-product cells under a small cycle budget."""
    if max_specs <= 0:
        return []
    cells = _all_cells()
    if not cells:
        return []
    offset = (max(0, int(cycle)) * max(1, int(max_specs))) % len(cells)
    ordered = cells[offset:] + cells[:offset]
    seen = _seen_cells(ledger)
    specs: list[ProposalSpec] = []
    emitted: set[str] = set()
    for cell in ordered:
        if cell.key in seen or cell.key in emitted:
            continue
        emitted.add(cell.key)
        specs.append(_spec_for_cell(cell, cycle=cycle, rank=len(specs) + 1))
        if len(specs) >= max_specs:
            break
    return specs
