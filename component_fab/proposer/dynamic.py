"""Evidence-driven proposal synthesis for autonomous fab runs.

This module turns ledger failures and near-misses into new ``ProposalSpec``
objects. It deliberately emits only schema-level axes that the existing
``code_generator`` already understands; it does not generate arbitrary Python
or CUDA code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from component_fab.improver.axis_variants import AnchorAxes, anchor_axes_for_op
from component_fab.proposer.property_miner import DEFAULT_META_DB
from component_fab.proposer.spec_generator import (
    ProposalSpec,
    build_spec_from_axes,
    category_from_axes,
    synthesis_kind_for_axes,
)
from component_fab.state.ledger import Ledger, LedgerEntry
from component_fab.proposer.tier2_feedback import (
    Tier2Feedback,
    WEAK_FAIL_BROAD_KV,
    WEAK_FAIL_COMPOSITIONAL,
    WEAK_FAIL_LONG_GAP,
    WEAK_NARROW_DISTRACTOR_ONLY,
    WEAK_NEAR_SURVIVOR,
    WEAK_REJECTED,
    load_tier2_feedback,
)

_CODEGEN_AXIS_PREFIXES: tuple[str, ...] = (
    "op_",
    "synthesis_kind",
)


@dataclass(frozen=True, slots=True)
class DynamicEvidenceCase:
    """A ledger-derived candidate base plus the measured weakness to repair."""

    source_id: str
    name: str
    base_axes: dict[str, Any]
    anchor_axes: dict[str, Any]
    score: float
    weaknesses: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Repair:
    name: str
    delta: dict[str, Any]
    rationale: str


def _latest_metadata(entry: LedgerEntry) -> dict[str, Any]:
    return dict(entry.metadata_history[-1]) if entry.metadata_history else {}


def _clean_axes(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in raw.items()
        if any(str(key).startswith(prefix) for prefix in _CODEGEN_AXIS_PREFIXES)
    }


def _weaknesses_from_metadata(
    metadata: Mapping[str, Any], score: float
) -> tuple[str, ...]:
    weaknesses: list[str] = []
    eliminated_by = str(metadata.get("eliminated_by") or "")
    if eliminated_by:
        weaknesses.append(f"eliminated_{eliminated_by}")
    erf_density = float(metadata.get("erf_density") or 0.0)
    if 0.0 < erf_density < 0.035:
        weaknesses.append("low_erf_density")
    nb_max_accuracy = float(metadata.get("nb_max_accuracy") or 0.0)
    if nb_max_accuracy and nb_max_accuracy < 0.62:
        weaknesses.append("weak_nano_bind")
    if (
        metadata.get("range_ran")
        and int(metadata.get("range_effective_distance") or 0) <= 0
    ):
        weaknesses.append("range_blind")
    if not bool(metadata.get("can_bind")) and score >= 0.15:
        weaknesses.append("cannot_bind")
    if not weaknesses and score >= 0.55:
        weaknesses.append("promising_needs_novelty")
    return tuple(dict.fromkeys(weaknesses))


def _weaknesses_with_tier2_feedback(
    metadata: Mapping[str, Any],
    score: float,
    feedback: Tier2Feedback | None,
) -> tuple[str, ...]:
    weaknesses = list(_weaknesses_from_metadata(metadata, score))
    if feedback is not None:
        weaknesses.extend(feedback.signatures)
    return tuple(dict.fromkeys(weaknesses))


def collect_dynamic_evidence_cases(
    ledger: Ledger,
    *,
    max_cases: int = 64,
    tier2_feedback_by_id: Mapping[str, Tier2Feedback] | None = None,
) -> list[DynamicEvidenceCase]:
    """Collect proposal bases whose recorded metrics suggest an actionable repair."""

    cases: list[DynamicEvidenceCase] = []
    for entry in ledger.all_entries():
        metadata = _latest_metadata(entry)
        axes = _clean_axes(metadata.get("math_axes") or {})
        if not axes:
            continue
        score = entry.mean_composite()
        feedback = (tier2_feedback_by_id or {}).get(entry.proposal_id)
        weaknesses = _weaknesses_with_tier2_feedback(metadata, score, feedback)
        if not weaknesses:
            continue
        if feedback is not None:
            score = max(0.05, score + feedback.mean_delta)
        cases.append(
            DynamicEvidenceCase(
                source_id=entry.proposal_id,
                name=entry.name or entry.proposal_id,
                base_axes=axes,
                anchor_axes=axes,
                score=score,
                weaknesses=weaknesses,
                notes=(
                    f"ledger_status={entry.promotion_status}",
                    *(
                        (
                            f"tier2_pass_count={feedback.pass_count}/{feedback.n_tasks}",
                            f"tier2_mean_delta={feedback.mean_delta:.4f}",
                        )
                        if feedback is not None
                        else ()
                    ),
                ),
            )
        )
    cases.sort(key=lambda c: (len(c.weaknesses), c.score), reverse=True)
    return cases[:max_cases]


def _axis_value_pool(cases: Iterable[DynamicEvidenceCase]) -> dict[str, list[Any]]:
    values: dict[str, dict[Any, float]] = {}
    for case in cases:
        for axis, value in case.base_axes.items():
            if not axis.startswith("op_"):
                continue
            values.setdefault(axis, {})
            values[axis][value] = max(values[axis].get(value, -1.0), case.score)
    return {
        axis: [value for value, _ in sorted(bucket.items(), key=lambda r: -r[1])]
        for axis, bucket in values.items()
    }


def _dynamic_alternative(
    value_pool: Mapping[str, Sequence[Any]],
    axis: str,
    current: Any,
    fallback: Any,
) -> Any:
    for value in value_pool.get(axis, ()):
        if value not in (None, "", current):
            return value
    return fallback


def _delta_bind_sparse_content(
    value_pool: Mapping[str, Sequence[Any]], axes: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "op_activation_sparsity_pattern": _dynamic_alternative(
            value_pool,
            "op_activation_sparsity_pattern",
            axes.get("op_activation_sparsity_pattern"),
            "learned_structured",
        ),
        "op_geometric_receptive_field": "global",
        "op_spectral_preferred_basis": "content",
        "op_routing_kind": _dynamic_alternative(
            value_pool,
            "op_routing_kind",
            axes.get("op_routing_kind"),
            "difficulty",
        ),
    }


def _delta_ledger_novel_composite(
    value_pool: Mapping[str, Sequence[Any]], axes: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "op_block_template": _dynamic_alternative(
            value_pool,
            "op_block_template",
            axes.get("op_block_template"),
            "gated_parallel",
        ),
        "op_block_slot_b": _dynamic_alternative(
            value_pool,
            "op_block_slot_b",
            axes.get("op_block_slot_b"),
            "chebyshev_spectral",
        ),
        "op_math_knobs": _dynamic_alternative(
            value_pool, "op_math_knobs", axes.get("op_math_knobs"), "spectral_chebyshev"
        ),
    }


@dataclass(frozen=True, slots=True)
class _RepairRule:
    """A declarative weakness -> repair mapping; fires if any trigger intersects.

    Exactly one of ``static_delta`` / ``dynamic_delta`` is set. ``only_if_no_prior``
    models the ``WEAK_REJECTED`` block, which fired only when no earlier rule matched.
    """

    triggers: frozenset[str]
    name: str
    rationale: str
    static_delta: dict[str, Any] | None = None
    dynamic_delta: (
        Callable[[Mapping[str, Sequence[Any]], Mapping[str, Any]], dict[str, Any]]
        | None
    ) = None
    only_if_no_prior: bool = False


# Order matters: rules fire (and append) in this sequence, matching the original
# longhand dispatch. WEAK_FAIL_LONG_GAP intentionally triggers two rules.
_REPAIR_RULES: tuple[_RepairRule, ...] = (
    _RepairRule(
        frozenset(
            {
                "range_blind",
                "low_erf_density",
                "eliminated_erf_density",
                WEAK_FAIL_LONG_GAP,
            }
        ),
        "extend_receptive_state",
        "repair measured range/ERF weakness with causal state and global content mixing",
        static_delta={
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
        },
    ),
    _RepairRule(
        frozenset({WEAK_FAIL_LONG_GAP}),
        "repair_long_gap_memory",
        "repair Tier-2 long_gap_recall failure with explicit causal state and recursive depth",
        static_delta={
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
            "op_block_template": "recursive_depth_router",
            "op_max_depth": 8,
        },
    ),
    _RepairRule(
        frozenset({WEAK_FAIL_COMPOSITIONAL}),
        "repair_compositional_tensor",
        "repair Tier-2 compositional_binding failure with content-aware tensor-factorized interactions",
        static_delta={
            "op_activation_sparsity_pattern": "learned_structured",
            "op_spectral_preferred_basis": "content",
            "op_math_family": "tensor_decomp",
            "op_tensor_decomp_kind": "tucker",
            "op_math_knobs": "tensor_tucker",
        },
    ),
    _RepairRule(
        frozenset({WEAK_FAIL_BROAD_KV}),
        "repair_broad_kv_lookup",
        "repair Tier-2 broad key/value recall failure with global content lookup and sparse expert routing",
        static_delta={
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
            "op_activation_sparsity_pattern": "top_k",
            "op_routing_kind": "top_k_moe",
            "op_top_k": 2,
        },
    ),
    _RepairRule(
        frozenset({WEAK_NARROW_DISTRACTOR_ONLY}),
        "escape_distractor_only",
        "escape narrow distractor-only Tier-2 wins by adding long-range state and harder content routing",
        static_delta={
            "op_geometric_receptive_field": "hybrid_local_global",
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_routing_kind": "difficulty",
            "op_math_knobs": "spectral_chebyshev",
        },
    ),
    _RepairRule(
        frozenset({"weak_nano_bind", "cannot_bind", "eliminated_nano_bind"}),
        "bind_sparse_content",
        "repair binding failure using content basis plus sparse/dynamic routing values mined from the ledger",
        dynamic_delta=_delta_bind_sparse_content,
    ),
    _RepairRule(
        frozenset({"eliminated_s05_causality_stability"}),
        "stabilize_causal_lane",
        "repair causality/stability elimination with bounded state and soft low-info routing",
        static_delta={
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_geometric_receptive_field": "hybrid_local_global",
            "op_activation_sparsity_pattern": "learned_structured",
            "op_routing_kind": "low_info_skip",
            "op_skip_hard": 0,
        },
    ),
    _RepairRule(
        frozenset({"promising_needs_novelty", WEAK_NEAR_SURVIVOR}),
        "ledger_novel_composite",
        "compose a promising near-miss with the strongest distinct block/knob values observed in the ledger",
        dynamic_delta=_delta_ledger_novel_composite,
    ),
    _RepairRule(
        frozenset({WEAK_REJECTED}),
        "rejected_to_memory_lookup",
        "Tier-2 rejected candidate needs a broader memory lookup repair before reuse",
        static_delta={
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
            "op_activation_sparsity_pattern": "learned_structured",
        },
        only_if_no_prior=True,
    ),
)

# Fired only when no rule matched, so the case always yields at least one repair.
_FALLBACK_REPAIR = _Repair(
    name="feedback_depth_router",
    delta={
        "op_routing_kind": "depth_router",
        "op_max_depth": 4,
        "op_geometric_receptive_field": "global",
    },
    rationale="generic ledger-feedback repair when no narrower weakness matched",
)


def _repairs_for_case(
    case: DynamicEvidenceCase,
    value_pool: Mapping[str, Sequence[Any]],
) -> list[_Repair]:
    weakness_set = set(case.weaknesses)
    repairs: list[_Repair] = []
    for rule in _REPAIR_RULES:
        if rule.only_if_no_prior and repairs:
            continue
        if not (rule.triggers & weakness_set):
            continue
        delta = (
            rule.dynamic_delta(value_pool, case.base_axes)
            if rule.dynamic_delta is not None
            else dict(rule.static_delta or {})
        )
        repairs.append(_Repair(rule.name, delta, rule.rationale))
    if not repairs:
        repairs.append(_FALLBACK_REPAIR)
    return repairs


def _slug(raw: str, *, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_").lower()
    return (slug or "case")[:max_len].strip("_") or "case"


def _spec_from_case_and_repair(
    case: DynamicEvidenceCase, repair: _Repair
) -> ProposalSpec:
    axes = {**case.base_axes, **repair.delta}
    axes.pop("synthesis_kind", None)
    source = _slug(case.name)
    weakness = _slug("_".join(case.weaknesses), max_len=36)
    # This path historically fingerprints the dispatched axes (including
    # the mirrored synthesis_kind) — preserved for proposal_id stability.
    return build_spec_from_axes(
        f"dynamic_{source}_{repair.name}_{weakness}",
        axes,
        witness_ops=(case.name,),
        anchor_axes=case.anchor_axes,
        notes=(
            f"source_id={case.source_id}",
            f"source_score={case.score:.4f}",
            f"repair={repair.name}",
            *case.notes,
        ),
        predicted_lift=max(0.1, min(1.0, case.score + 0.08)),
        lift_pass_rate=min(1.0, max(0.05, case.score)),
        fingerprint_dispatched_axes=True,
        rationale=(
            f"Dynamic proposal derived from ledger evidence for {case.source_id}. "
            f"Weaknesses={', '.join(case.weaknesses)}. {repair.rationale}."
        ),
    )


def spec_from_ledger_entry(entry: LedgerEntry) -> ProposalSpec | None:
    """Reconstruct a buildable spec from a ledger row with persisted axes."""

    metadata = _latest_metadata(entry)
    axes = _clean_axes(metadata.get("math_axes") or {})
    if not axes:
        return None
    kind = str(axes.get("synthesis_kind") or synthesis_kind_for_axes(axes, axes))
    axes["synthesis_kind"] = kind
    row = dict(axes)
    row.setdefault("op_n_inputs", 1)
    row.setdefault("op_is_parameterized", 1)
    row.setdefault("op_is_stateless", 0 if axes.get("op_dynamical_has_state") else 1)
    score = entry.mean_composite()
    return ProposalSpec(
        proposal_id=entry.proposal_id,
        name=entry.name or entry.proposal_id,
        category=entry.category or category_from_axes(axes),
        synthesis_kind=entry.synthesis_kind or kind,
        math_axes=axes,
        anchor_witness_op=entry.name or entry.proposal_id,
        anchor_witnesses_all=(entry.name or entry.proposal_id,),
        declared_property_row=row,
        predicted_lift=max(0.1, min(1.0, score)),
        rationale=f"Reconstructed from persisted ledger math_axes for {entry.proposal_id}.",
        notes=(f"ledger_status={entry.promotion_status}", f"source_score={score:.4f}"),
    )


def specs_from_ledger_entries(ledger: Ledger) -> list[ProposalSpec]:
    """Return all directly rebuildable ledger specs with persisted math axes."""

    specs: list[ProposalSpec] = []
    for entry in ledger.all_entries():
        spec = spec_from_ledger_entry(entry)
        if spec is not None:
            specs.append(spec)
    return specs


def _anchor_cases(
    anchor_op_names: Sequence[str],
    *,
    db_path: Path | str = DEFAULT_META_DB,
) -> list[DynamicEvidenceCase]:
    cases: list[DynamicEvidenceCase] = []
    for name in anchor_op_names:
        anchor: AnchorAxes | None = anchor_axes_for_op(name, db_path=db_path)
        if anchor is None:
            continue
        score = anchor.pass_rate
        weaknesses = ("promising_needs_novelty",) if score >= 0.2 else ("cannot_bind",)
        cases.append(
            DynamicEvidenceCase(
                source_id=name,
                name=name,
                base_axes=dict(anchor.axes),
                anchor_axes=dict(anchor.axes),
                score=score,
                weaknesses=weaknesses,
                notes=("source=anchor_catalog",),
            )
        )
    return cases


def enumerate_dynamic_proposals(
    anchor_op_names: Sequence[str],
    ledger: Ledger,
    *,
    max_specs: int = 32,
    max_cases: int = 64,
    db_path: Path | str = DEFAULT_META_DB,
    include_anchor_fallback: bool = True,
    tier2_feedback_by_id: Mapping[str, Tier2Feedback] | None = None,
) -> list[ProposalSpec]:
    """Generate proposals from current evidence rather than static templates."""

    if max_specs <= 0:
        return []

    if tier2_feedback_by_id is None:
        tier2_feedback_by_id = load_tier2_feedback()
    evidence_cases = collect_dynamic_evidence_cases(
        ledger,
        max_cases=max_cases,
        tier2_feedback_by_id=tier2_feedback_by_id,
    )
    if include_anchor_fallback and not evidence_cases:
        evidence_cases = _anchor_cases(anchor_op_names, db_path=db_path)
    if not evidence_cases:
        return []

    value_pool = _axis_value_pool(evidence_cases)
    specs: list[ProposalSpec] = []
    seen_axes: set[tuple[tuple[str, str], ...]] = set()
    for case in evidence_cases:
        for repair in _repairs_for_case(case, value_pool):
            spec = _spec_from_case_and_repair(case, repair)
            key = tuple(
                sorted((key, str(value)) for key, value in spec.math_axes.items())
            )
            if key in seen_axes:
                continue
            seen_axes.add(key)
            specs.append(spec)
            if len(specs) >= max_specs:
                return specs
    return specs
