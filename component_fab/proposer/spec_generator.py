"""Turn property-miner CandidateTuples into structured ProposalSpecs.

A ``CandidateTuple`` is a math-axis recipe ("padic + content + O(L^2) +
state + dense + local"). A ``ProposalSpec`` is the buildable thing — a
target fab category (lane/routing/compression), a synthesis_kind hint
for the eventual code generator (semiring_swap / state_kernel_swap /
projection_swap / basis_swap), an anchor witness op to start from, and
a declared ``op_property_catalog`` row for the new op.

Read-only. Stateless. No DB access.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from .property_miner import CandidateTuple


SYNTHESIS_KIND_SEMIRING_SWAP = "semiring_swap"
SYNTHESIS_KIND_STATE_KERNEL_SWAP = "state_kernel_swap"
SYNTHESIS_KIND_PROJECTION_SWAP = "projection_swap"
SYNTHESIS_KIND_BASIS_SWAP = "basis_swap"
SYNTHESIS_KIND_NOVEL_HYBRID = "novel_hybrid"

CATEGORY_LANE = "lane"
CATEGORY_ROUTING = "routing"
CATEGORY_COMPRESSION = "compression"

_NOVEL_SPACES = frozenset(
    {
        "tropical",
        "clifford",
        "padic",
        "spiking",
        "hyperbolic",
        "hyperbolic_poincare",
        "complex",
        "quaternion",
    }
)


@dataclass(frozen=True, slots=True)
class ProposalSpec:
    proposal_id: str
    name: str
    category: str
    synthesis_kind: str
    math_axes: dict[str, Any]
    anchor_witness_op: str
    anchor_witnesses_all: tuple[str, ...]
    declared_property_row: dict[str, Any]
    predicted_lift: float
    rationale: str
    notes: tuple[str, ...] = field(default_factory=tuple)


def _axes_to_dict(candidate: CandidateTuple) -> dict[str, Any]:
    return {axis: value for axis, value in candidate.tuple_values}


def category_from_axes(axes: dict[str, Any]) -> str:
    sparsity = str(axes.get("op_activation_sparsity_pattern") or "")
    receptive = str(axes.get("op_geometric_receptive_field") or "")
    has_state = int(axes.get("op_dynamical_has_state") or 0)
    memory = str(axes.get("op_dynamical_memory_length_class") or "")

    if sparsity == "top_k":
        if receptive in ("global", "hybrid_local_global"):
            return CATEGORY_ROUTING
        if has_state == 0:
            return CATEGORY_ROUTING
    if has_state == 1 and sparsity in ("structured", "learned_structured", "top_k"):
        return CATEGORY_COMPRESSION
    if has_state == 0 and memory == "O(L^2)" and sparsity == "dense":
        return CATEGORY_LANE
    return CATEGORY_LANE


def synthesis_kind_for_axes(axes: dict[str, Any], anchor_axes: dict[str, Any]) -> str:
    """Decide which code-gen template fits the axis delta from the anchor."""
    diffs = {axis for axis, value in axes.items() if anchor_axes.get(axis) != value}
    if (
        "op_algebraic_space" in diffs
        and axes.get("op_algebraic_space") in _NOVEL_SPACES
    ):
        return SYNTHESIS_KIND_SEMIRING_SWAP
    if "op_dynamical_has_state" in diffs or "op_dynamical_memory_length_class" in diffs:
        return SYNTHESIS_KIND_STATE_KERNEL_SWAP
    if "op_activation_sparsity_pattern" in diffs:
        return SYNTHESIS_KIND_PROJECTION_SWAP
    if "op_spectral_preferred_basis" in diffs:
        return SYNTHESIS_KIND_BASIS_SWAP
    return SYNTHESIS_KIND_NOVEL_HYBRID


def _pick_anchor(candidate: CandidateTuple) -> tuple[str, dict[str, Any]]:
    """Return ``(anchor_op_name, anchor_axes_dict)`` for synthesis-kind diffing.

    Uses ``candidate.anchor_axes`` when the caller supplied it (the only way
    ``synthesis_kind_for_axes`` can compute meaningful diffs — otherwise it
    sees every axis as a diff against an empty dict and the algebra rule
    short-circuits to ``semiring_swap`` for every novel-algebra host).
    """
    if not candidate.witness_ops:
        return "", dict(candidate.anchor_axes)
    return candidate.witness_ops[0], dict(candidate.anchor_axes)


def _declared_property_row(axes: dict[str, Any]) -> dict[str, Any]:
    row = dict(axes)
    row.setdefault("op_n_inputs", 1)
    row.setdefault("op_is_parameterized", 1)
    row.setdefault("op_is_stateless", 0 if axes.get("op_dynamical_has_state") else 1)
    return row


def _build_name(category: str, axes: dict[str, Any]) -> str:
    space = str(axes.get("op_algebraic_space") or "euclidean")
    memory = str(axes.get("op_dynamical_memory_length_class") or "O(1)")
    state = "stateful" if int(axes.get("op_dynamical_has_state") or 0) else "stateless"
    sparsity = str(axes.get("op_activation_sparsity_pattern") or "dense")
    raw = f"{category}_{space}_{state}_{memory}_{sparsity}"
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_").lower()
    return slug or f"{category}_proposal"


def axes_fingerprint(axes: Mapping[str, Any]) -> str:
    """Stable hash of an axes dict. Two specs with identical merged axes
    share a fingerprint — they generate the same module under
    ``code_generator.generate_module``, so grading them twice is waste.

    Sorted-key blake2b(digest_size=5) keeps the hash stable across runs
    and identical to the digest historically used by ``_proposal_id``.
    """
    return hashlib.blake2b(
        ("|".join(f"{k}={v}" for k, v in sorted(axes.items()))).encode("utf-8"),
        digest_size=5,
    ).hexdigest()


def make_proposal_id(name: str, axes: Mapping[str, Any]) -> str:
    """Stable proposal id = ``<name>_<axes_fingerprint>``."""
    return f"{name}_{axes_fingerprint(axes)}"


def _proposal_id(name: str, axes: dict[str, Any]) -> str:
    return make_proposal_id(name, axes)


def _rationale(axes: dict[str, Any], category: str, anchor: str, kind: str) -> str:
    summary = ", ".join(f"{k.removeprefix('op_')}={v}" for k, v in axes.items())
    return (
        f"Unrealized property tuple anchored on {anchor or 'no witness'} via "
        f"{kind}. Target category {category}. Tuple: {summary}."
    )


def spec_from_candidate(candidate: CandidateTuple) -> ProposalSpec:
    axes = _axes_to_dict(candidate)
    category = category_from_axes(axes)
    anchor, anchor_axes = _pick_anchor(candidate)
    kind = synthesis_kind_for_axes(axes, anchor_axes)
    # Day-3 (2026-05-15): make synthesis_kind visible to code_generator's
    # _dispatch_synthesis_hint by mirroring it into math_axes. Without this
    # the dispatcher can't see the kind and basis_swap / projection_swap /
    # state_kernel_swap labels stay inert.
    axes["synthesis_kind"] = kind
    name = _build_name(category, axes)
    return ProposalSpec(
        proposal_id=_proposal_id(name, axes),
        name=name,
        category=category,
        synthesis_kind=kind,
        math_axes=axes,
        anchor_witness_op=anchor,
        anchor_witnesses_all=tuple(w for w in candidate.witness_ops if w),
        declared_property_row=_declared_property_row(axes),
        predicted_lift=candidate.predicted_lift,
        rationale=_rationale(axes, category, anchor, kind),
    )


def specs_from_candidates(
    candidates: list[CandidateTuple],
) -> list[ProposalSpec]:
    return [spec_from_candidate(c) for c in candidates]


def dedupe_specs_by_axes(specs: list[ProposalSpec]) -> list[ProposalSpec]:
    """Drop specs whose ``math_axes`` already appeared earlier in ``specs``.

    Cross-anchor nesting can produce many proposal_ids with identical merged
    axes (e.g. ``cross_X_x_cross_Y_x_Z`` and ``cross_X_x_cross_W_x_Z`` when
    Y and W contribute the same inherited axes). They generate the same
    module and grade to the same numbers, so keeping both is waste. When two
    specs share an axes fingerprint we keep the one with the **shorter
    name** — the simplest canonical representation.
    """
    seen: dict[str, ProposalSpec] = {}
    order: list[str] = []
    for spec in specs:
        fp = axes_fingerprint(spec.math_axes)
        existing = seen.get(fp)
        if existing is None:
            seen[fp] = spec
            order.append(fp)
        elif len(spec.name) < len(existing.name):
            seen[fp] = spec
    return [seen[fp] for fp in order]


def spec_to_json(spec: ProposalSpec) -> dict[str, Any]:
    return {
        "proposal_id": spec.proposal_id,
        "name": spec.name,
        "category": spec.category,
        "synthesis_kind": spec.synthesis_kind,
        "math_axes": dict(spec.math_axes),
        "anchor_witness_op": spec.anchor_witness_op,
        "anchor_witnesses_all": list(spec.anchor_witnesses_all),
        "declared_property_row": dict(spec.declared_property_row),
        "predicted_lift": spec.predicted_lift,
        "rationale": spec.rationale,
        "notes": list(spec.notes),
    }
