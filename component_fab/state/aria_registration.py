"""ARIA registration handoff (WS-7).

Promotion has dead-ended in ``catalog/``: a promoted fab component never re-enters
the ARIA / synthesis search space, so the loop is open. This module closes it the
decoupled way — on promotion it appends an ``op_property_catalog``-compatible row
plus the build recipe to a handoff file (``catalog/aria_handoff.jsonl``). The
synthesis runtime *reads* this file to consume promoted components; nothing here
imports the runtime, preserving the no-runtime-coupling rule (the runtime imports
from the handoff, not vice versa).

Each record carries enough to (a) register the op in the search space (the
declared property axes, op_category) and (b) re-instantiate it
(``generate_module`` is a pure function of ``math_axes``), plus the promotion
evidence (composite, transplant portability) for provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .ledger import iter_jsonl_records

if TYPE_CHECKING:
    from ..proposer.spec_generator import ProposalSpec

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_HANDOFF_PATH = _REPO / "component_fab" / "catalog" / "aria_handoff.jsonl"

# op_property_catalog declared columns we can populate from a spec's math_axes.
_DECLARED_AXIS_KEYS: tuple[str, ...] = (
    "op_algebraic_space",
    "op_spectral_preferred_basis",
    "op_dynamical_memory_length_class",
    "op_dynamical_has_state",
    "op_activation_sparsity_pattern",
    "op_geometric_receptive_field",
    "op_routing_kind",
    "op_block_template",
    "op_math_family",
)


def aria_registration_row(
    spec: "ProposalSpec", *, evidence: dict[str, Any] | None = None
) -> dict[str, Any]:
    """An op_property_catalog-compatible registration row for ``spec``."""
    axes = dict(spec.math_axes or {})
    declared = {k: axes.get(k) for k in _DECLARED_AXIS_KEYS if axes.get(k) is not None}
    return {
        "op_name": spec.name or spec.proposal_id,
        "proposal_id": spec.proposal_id,
        "op_category": spec.category,
        "synthesis_kind": spec.synthesis_kind,
        "source": "component_fab",
        # Declared property axes — the search-space registration payload.
        "declared_axes": declared,
        # Build recipe — generate_module(math_axes) re-instantiates the lane.
        "math_axes": axes,
        # Provenance / evidence for the runtime to weigh.
        "evidence": dict(evidence or {}),
    }


def register_promotion(
    spec: "ProposalSpec",
    *,
    evidence: dict[str, Any] | None = None,
    handoff_path: Path | str = DEFAULT_HANDOFF_PATH,
) -> Path:
    """Append a registration row for a promoted spec to the ARIA handoff file."""
    out = Path(handoff_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    row = aria_registration_row(spec, evidence=evidence)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return out


def read_handoff(
    handoff_path: Path | str = DEFAULT_HANDOFF_PATH,
) -> list[dict[str, Any]]:
    """Read registered components (latest row per proposal_id). Runtime-facing."""
    latest = {
        str(record.get("proposal_id") or record.get("op_name")): record
        for record in iter_jsonl_records(handoff_path)
        if record.get("proposal_id") or record.get("op_name")
    }
    return list(latest.values())
