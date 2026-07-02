"""Evidence-driven proposal synthesis for autonomous fab runs.

This module turns ledger failures and near-misses into new ``ProposalSpec``
objects. It deliberately emits only schema-level axes that the existing
``code_generator`` already understands; it does not generate arbitrary Python
or CUDA code.
"""

from __future__ import annotations

import hashlib
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
from component_fab.state.ledger import (
    Ledger,
    LedgerEntry,
    PROMOTION_PROMOTED,
    PROMOTION_REJECTED,
)
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
_LOSS_SPECIALIST_ROLES = frozenset(
    {"loss_specialist", "loss_monster", "loss_specialist_pair"}
)
_SOFTMAX_TWIN_REPAIR_THRESHOLD = 0.85
_SOFTMAX_SHAPED_SCORE_NORMS = frozenset({"", "softmax", "sharpen"})
_SCORE_NORM_SPECTRUM = ("tsallis_q", "renyi", "entmax_alpha")
_WEAK_SCORE_NORM_SOFTMAX_BASIN = "score_norm_softmax_basin"


@dataclass(frozen=True, slots=True)
class DynamicEvidenceCase:
    """A ledger-derived candidate base plus the measured weakness to repair."""

    source_id: str
    root_source_id: str
    name: str
    base_axes: dict[str, Any]
    anchor_axes: dict[str, Any]
    score: float
    weaknesses: tuple[str, ...]
    repair_depth: int = 0
    repair_history: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Repair:
    name: str
    delta: dict[str, Any]
    rationale: str


def _latest_metadata(entry: LedgerEntry) -> dict[str, Any]:
    return dict(entry.metadata_history[-1]) if entry.metadata_history else {}


def _dynamic_repair_depth(name: str) -> int:
    depth = 0
    remaining = name
    while remaining.startswith("dynamic_"):
        depth += 1
        remaining = remaining[len("dynamic_") :]
    return depth


def _positive_tier2_repair_evidence(feedback: Tier2Feedback | None) -> bool:
    return (
        feedback is not None and feedback.mean_delta > 0.0 and feedback.pass_count > 0
    )


def _clean_axes(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in raw.items()
        if any(str(key).startswith(prefix) for prefix in _CODEGEN_AXIS_PREFIXES)
    }


def _metadata_value(metadata: Mapping[str, Any], *keys: str) -> Any:
    axes = metadata.get("math_axes")
    axis_map = axes if isinstance(axes, Mapping) else {}
    for key in keys:
        if key in metadata:
            return metadata[key]
        op_key = f"op_{key}"
        if op_key in metadata:
            return metadata[op_key]
        if key in axis_map:
            return axis_map[key]
        if op_key in axis_map:
            return axis_map[op_key]
    return None


def _float_metadata(metadata: Mapping[str, Any], *keys: str) -> float | None:
    raw = _metadata_value(metadata, *keys)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _truthy_metadata(metadata: Mapping[str, Any], *keys: str) -> bool:
    raw = _metadata_value(metadata, *keys)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return False
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(raw)


# A compression candidate is weak when its declared latent budget is
# under-utilized (low effective rank) or it reconstructs poorly. These keys are
# written by the grading path when the compiled graph contains a real
# compression op (see component_fab.metrics.compression_quality); when no
# compression op is present they are absent and this returns False (no fake).
_COMPRESSION_MIN_RANK_RATIO = 0.5
_COMPRESSION_MAX_RECONSTRUCT_MSE = 0.5


def _compression_is_weak(metadata: Mapping[str, Any]) -> bool:
    declared = _metadata_value(
        metadata,
        "compression_declared",
        "compression_target",
        "compression_target_op",
    )
    if not declared:
        return False
    rank_ratio = _float_metadata(
        metadata,
        "compression_effective_rank_ratio",
        "effective_rank_ratio",
    )
    reconstruct_mse = _float_metadata(
        metadata,
        "compression_reconstruct_mse",
        "reconstruct_mse",
        "reconstruction_mse",
    )
    if rank_ratio is None and reconstruct_mse is None:
        return False
    under_utilized = rank_ratio is not None and rank_ratio < _COMPRESSION_MIN_RANK_RATIO
    poor_reconstruct = (
        reconstruct_mse is not None
        and reconstruct_mse > _COMPRESSION_MAX_RECONSTRUCT_MSE
    )
    return under_utilized or poor_reconstruct


def _score_norm_spectrum_is_weak(metadata: Mapping[str, Any]) -> bool:
    """Measured basin collapse that should be repaired in score-norm space."""

    reason = str(
        _metadata_value(
            metadata,
            "math_variant_failure_reason",
            "variant_failure_reason",
            "failure_reason",
        )
        or ""
    )
    if reason == "softmax_twin_regression" or _truthy_metadata(
        metadata, "math_variant_softmax_twin_regression"
    ):
        return True

    twin_score = _float_metadata(
        metadata, "softmax_twin_score", "math_variant_softmax_twin_score"
    )
    if twin_score is None or twin_score < _SOFTMAX_TWIN_REPAIR_THRESHOLD:
        return False

    score_norm = str(
        _metadata_value(metadata, "physics_score_norm_family", "score_norm_family")
        or ""
    ).strip()
    search_track = str(_metadata_value(metadata, "search_track") or "").strip()
    return (
        not score_norm
        or score_norm in _SOFTMAX_SHAPED_SCORE_NORMS
        or score_norm in _SCORE_NORM_SPECTRUM
        or search_track == "physics_atom"
    )


def _weaknesses_from_metadata(
    metadata: Mapping[str, Any], score: float
) -> tuple[str, ...]:
    weaknesses: list[str] = []
    role = str(
        _metadata_value(
            metadata,
            "candidate_role",
            "component_role",
            "specialist_role",
        )
        or ""
    )
    is_loss_specialist = role in _LOSS_SPECIALIST_ROLES or bool(
        _metadata_value(metadata, "loss_specialist")
    )
    carrier = _metadata_value(
        metadata,
        "loss_specialist_partner_op",
        "loss_specialist_carrier_op",
        "paired_anchor_op",
    )
    if is_loss_specialist and not carrier:
        weaknesses.append("loss_monster_unpaired")
    loss_ratio = _float_metadata(
        metadata,
        "screening_loss_ratio",
        "loss_ratio",
        "next_token_loss_ratio",
    )
    if (
        loss_ratio is not None
        and loss_ratio < 0.1
        and not bool(metadata.get("can_bind"))
    ):
        weaknesses.append("strong_loss_floor_reasoning")
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
    if _compression_is_weak(metadata):
        weaknesses.append("weak_compression")
    if _score_norm_spectrum_is_weak(metadata):
        weaknesses.append(_WEAK_SCORE_NORM_SOFTMAX_BASIN)
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
        repair_depth = _dynamic_repair_depth(entry.name or entry.proposal_id)
        if repair_depth >= 2 and not _positive_tier2_repair_evidence(feedback):
            continue
        weaknesses = _weaknesses_with_tier2_feedback(metadata, score, feedback)
        if not weaknesses:
            continue
        if feedback is not None:
            score = max(0.05, score + feedback.mean_delta)
        cases.append(
            DynamicEvidenceCase(
                source_id=entry.proposal_id,
                root_source_id=str(
                    metadata.get("root_source_id")
                    or metadata.get("source_id")
                    or entry.proposal_id
                ),
                name=entry.name or entry.proposal_id,
                base_axes=axes,
                anchor_axes=axes,
                score=score,
                weaknesses=weaknesses,
                repair_depth=repair_depth,
                repair_history=tuple(metadata.get("repair_history") or ()),
                notes=(
                    f"ledger_status={entry.promotion_status}",
                    f"repair_depth={repair_depth}",
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
        "op_search_track": "physics_atom",
        "op_physics_atom_kinds": "scan+basis",
        "op_physics_basis_axis": "token",
        "op_physics_address_family": "cosine",
        "op_physics_score_norm_family": "sharpen",
        "op_physics_aggregate_family": "semiring",
        "op_physics_knob_scale": 2.0,
        "op_physics_target": "binding_content_addressed_state",
        "op_dynamical_has_state": 1,
        "op_dynamical_memory_length_class": "O(L)",
        "op_activation_sparsity_pattern": _dynamic_alternative(
            value_pool,
            "op_activation_sparsity_pattern",
            axes.get("op_activation_sparsity_pattern"),
            "learned_structured",
        ),
        "op_geometric_receptive_field": "global",
        "op_spectral_preferred_basis": "content",
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


def _score_norm_spectrum_delta(score_norm: str, knob_scale: float) -> dict[str, Any]:
    sparsity = "top_k" if score_norm == "entmax_alpha" else "learned_structured"
    return {
        "op_search_track": "physics_atom",
        "op_physics_atom_kinds": "scan+basis+norm",
        "op_physics_norm_axis": "token",
        "op_physics_basis_axis": "token",
        "op_physics_address_family": "reciprocal",
        "op_physics_score_norm_family": score_norm,
        "op_physics_aggregate_family": "semiring",
        "op_physics_knob_scale": knob_scale,
        "op_physics_target": "score_norm_spectrum_escape",
        "op_dynamical_has_state": 1,
        "op_dynamical_memory_length_class": "O(L)",
        "op_geometric_receptive_field": "global",
        "op_spectral_preferred_basis": "content",
        "op_activation_sparsity_pattern": sparsity,
    }


def _delta_score_norm_spectrum(
    value_pool: Mapping[str, Sequence[Any]], axes: Mapping[str, Any]
) -> dict[str, Any]:
    del value_pool
    current = str(axes.get("op_physics_score_norm_family") or "")
    score_norm = next(
        (candidate for candidate in _SCORE_NORM_SPECTRUM if candidate != current),
        _SCORE_NORM_SPECTRUM[0],
    )
    return _score_norm_spectrum_delta(score_norm, 1.75)


def _physics_variant_patches(delta: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Small grammar-bounded fan-out over atomic/physics coordinates.

    The repair rules pick a target region; this expands that target into a few
    nearby valid ``AtomSpec``/``StageSpec`` coordinates instead of treating a
    human-chosen coordinate as the only candidate.
    """

    if delta.get("op_search_track") != "physics_atom":
        return ()

    target = str(delta.get("op_physics_target") or "")
    base_scale = float(delta.get("op_physics_knob_scale") or 1.0)
    if target.startswith("long_gap"):
        curated = (
            {
                "op_physics_variant": "physv01",
                "op_physics_seed": 1,
                "op_physics_atom_kinds": "scan+basis+norm",
                "op_physics_norm_axis": "channel",
                "op_physics_address_family": "dot",
                "op_physics_score_norm_family": "sharpen",
                "op_physics_aggregate_family": "semiring",
                "op_physics_knob_scale": round(base_scale * 0.85, 4),
            },
            {
                "op_physics_variant": "physv02",
                "op_physics_seed": 2,
                "op_physics_atom_kinds": "basis+scan",
                "op_physics_address_family": "reciprocal",
                "op_physics_score_norm_family": "softmax",
                "op_physics_aggregate_family": "mean",
                "op_physics_knob_scale": round(base_scale * 1.15, 4),
            },
            {
                "op_physics_variant": "physv03",
                "op_physics_seed": 3,
                "op_physics_atom_kinds": "scan+norm+basis",
                "op_physics_norm_axis": "token",
                "op_physics_address_family": "cosine",
                "op_physics_score_norm_family": "sharpen",
                "op_physics_aggregate_family": "mean",
                "op_physics_knob_scale": round(base_scale * 0.7, 4),
            },
        )
        return (*curated, *_open_discovery_variant_patches(delta))
    if target in {"binding_content_addressed_state", "broad_kv_content_lookup"}:
        curated = (
            {
                "op_physics_variant": "physv01",
                "op_physics_seed": 1,
                "op_physics_atom_kinds": "basis+scan",
                "op_physics_address_family": "dot",
                "op_physics_score_norm_family": "sharpen",
                "op_physics_aggregate_family": "semiring",
                "op_physics_knob_scale": round(base_scale * 0.75, 4),
            },
            {
                "op_physics_variant": "physv02",
                "op_physics_seed": 2,
                "op_physics_atom_kinds": "norm+basis+scan",
                "op_physics_norm_axis": "channel",
                "op_physics_address_family": "cosine",
                "op_physics_score_norm_family": "softmax",
                "op_physics_aggregate_family": "mean",
                "op_physics_knob_scale": round(base_scale * 1.25, 4),
            },
            {
                "op_physics_variant": "physv03",
                "op_physics_seed": 3,
                "op_physics_atom_kinds": "scan+basis+norm",
                "op_physics_norm_axis": "token",
                "op_physics_address_family": "reciprocal",
                "op_physics_score_norm_family": "sharpen",
                "op_physics_aggregate_family": "semiring",
                "op_physics_knob_scale": round(base_scale * 1.5, 4),
            },
        )
        return (*curated, *_open_discovery_variant_patches(delta))
    if target == "score_norm_spectrum_escape":
        return (
            {
                "op_physics_variant": "physv01",
                "op_physics_seed": 1,
                "op_physics_atom_kinds": "scan+basis",
                "op_physics_norm_axis": "token",
                "op_physics_basis_axis": "token",
                "op_physics_address_family": "reciprocal",
                "op_physics_score_norm_family": "tsallis_q",
                "op_physics_aggregate_family": "semiring",
                "op_physics_knob_scale": 1.75,
            },
            {
                "op_physics_variant": "physv02",
                "op_physics_seed": 2,
                "op_physics_atom_kinds": "norm+scan+basis",
                "op_physics_norm_axis": "channel",
                "op_physics_basis_axis": "token",
                "op_physics_address_family": "cosine",
                "op_physics_score_norm_family": "renyi",
                "op_physics_aggregate_family": "semiring",
                "op_physics_knob_scale": 2.0,
            },
            {
                "op_physics_variant": "physv03",
                "op_physics_seed": 3,
                "op_physics_atom_kinds": "basis+scan+norm",
                "op_physics_norm_axis": "token",
                "op_physics_basis_axis": "channel",
                "op_physics_address_family": "dot",
                "op_physics_score_norm_family": "entmax_alpha",
                "op_physics_aggregate_family": "semiring",
                "op_physics_knob_scale": 2.25,
                "op_activation_sparsity_pattern": "top_k",
            },
        )
    curated = (
        {
            "op_physics_variant": "physv01",
            "op_physics_seed": 1,
            "op_physics_atom_kinds": "norm+basis+scan",
            "op_physics_norm_axis": "channel",
            "op_physics_score_norm_family": "sharpen",
            "op_physics_knob_scale": round(base_scale * 1.1, 4),
        },
    )
    return (*curated, *_open_discovery_variant_patches(delta, max_variants=1))


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def _open_discovery_variant_patches(
    delta: Mapping[str, Any], *, max_variants: int = 3
) -> tuple[dict[str, Any], ...]:
    """Generate grammar-sampled physics variants from ``open_discovery``.

    This is a mini, deterministic prepass over the existing atom/stage search
    grammar. It rejects identity/default softmax-shaped samples so repair budget
    keeps moving toward novel physics coordinates.
    """

    try:
        import torch
        from research.synthesis.open_discovery import sample_spec
    except Exception:  # pragma: no cover - torch/synthesis unavailable in tiny envs
        return ()

    target = str(delta.get("op_physics_target") or "generic")
    base_seed = _stable_seed(target, delta.get("op_physics_atom_kinds"), max_variants)
    gen = torch.Generator().manual_seed(base_seed)
    patches: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for _ in range(max_variants * 16):
        sampled = sample_spec(gen, None, max_atom_depth=3, mutate_prob=0.0)
        kinds = sampled.atom.kinds
        if not kinds:
            continue
        if target.startswith("long_gap") and "scan" not in kinds:
            continue
        if target in {"binding_content_addressed_state", "broad_kv_content_lookup"}:
            if "basis" not in kinds and sampled.stage.address == "dot":
                continue
        key = (
            "+".join(kinds),
            sampled.atom.norm_axis,
            sampled.atom.basis_axis,
            sampled.stage.key,
            f"{sampled.knob_scale:.2f}",
        )
        if key in seen:
            continue
        seen.add(key)
        index = len(patches) + 1
        patches.append(
            {
                "op_physics_variant": f"physod{index:02d}",
                "op_physics_seed": 100 + index,
                "op_physics_atom_kinds": "+".join(kinds),
                "op_physics_norm_axis": sampled.atom.norm_axis,
                "op_physics_basis_axis": sampled.atom.basis_axis,
                "op_physics_address_family": sampled.stage.address,
                "op_physics_score_norm_family": sampled.stage.score_norm,
                "op_physics_aggregate_family": sampled.stage.aggregate,
                "op_physics_knob_scale": round(sampled.knob_scale, 4),
            }
        )
        if len(patches) >= max_variants:
            break
    return tuple(patches)


def _expand_physics_repairs(repairs: Sequence[_Repair]) -> list[_Repair]:
    expanded = list(repairs)
    seen: set[tuple[tuple[str, str], ...]] = {
        tuple(sorted((key, str(value)) for key, value in repair.delta.items()))
        for repair in repairs
    }
    for repair in repairs:
        for patch in _physics_variant_patches(repair.delta):
            delta = {**repair.delta, **patch}
            key = tuple(sorted((axis, str(value)) for axis, value in delta.items()))
            if key in seen:
                continue
            seen.add(key)
            variant = str(patch["op_physics_variant"])
            expanded.append(
                _Repair(
                    name=f"{repair.name}_{variant}",
                    delta=delta,
                    rationale=(
                        f"{repair.rationale} Descriptor variant {variant} "
                        "samples a nearby atom/stage coordinate."
                    ),
                )
            )
    return expanded


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
            "op_search_track": "physics_atom",
            "op_physics_atom_kinds": "scan+basis",
            "op_physics_basis_axis": "token",
            "op_physics_address_family": "reciprocal",
            "op_physics_score_norm_family": "sharpen",
            "op_physics_aggregate_family": "semiring",
            "op_physics_knob_scale": 2.25,
            "op_physics_target": "long_gap_ordered_memory",
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
            "op_search_track": "physics_atom",
            "op_physics_atom_kinds": "scan+basis",
            "op_physics_basis_axis": "token",
            "op_physics_address_family": "reciprocal",
            "op_physics_score_norm_family": "sharpen",
            "op_physics_aggregate_family": "semiring",
            "op_physics_knob_scale": 2.5,
            "op_physics_target": "long_gap_recursive_memory",
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
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
            "op_search_track": "physics_atom",
            "op_physics_atom_kinds": "scan+basis",
            "op_physics_basis_axis": "token",
            "op_physics_address_family": "cosine",
            "op_physics_score_norm_family": "sharpen",
            "op_physics_aggregate_family": "semiring",
            "op_physics_knob_scale": 2.0,
            "op_physics_target": "broad_kv_content_lookup",
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
            "op_activation_sparsity_pattern": "top_k",
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
        frozenset({"loss_monster_unpaired", "strong_loss_floor_reasoning"}),
        "pair_loss_monster_with_carrier",
        (
            "repair local loss-specialist evidence by pairing it with a "
            "long-range carrier and grading only partner-relative capability"
        ),
        static_delta={
            "op_block_template": "loss_monster_paired",
            "op_partner_kind": "hyper_mor",
            "op_block_slot_loss": "routed_bottleneck",
            "op_partner_floor": 0.5,
            "op_candidate_role": "loss_specialist_pair",
            "op_loss_specialist_paired": 1,
            "op_loss_specialist_partner_op": "hyper_mor_b_145m",
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_geometric_receptive_field": "global",
        },
    ),
    _RepairRule(
        frozenset({"weak_nano_bind", "cannot_bind", "eliminated_nano_bind"}),
        "bind_sparse_content",
        "repair binding failure using content basis plus sparse/dynamic routing values mined from the ledger",
        dynamic_delta=_delta_bind_sparse_content,
    ),
    _RepairRule(
        frozenset({"weak_compression", "eliminated_compression"}),
        "compress_content_bottleneck",
        (
            "repair under-utilized/low-fidelity compression with a content-addressed "
            "bounded-state bottleneck — the non-QKV state-compression specialty"
        ),
        static_delta={
            "op_search_track": "physics_atom",
            "op_physics_atom_kinds": "scan+basis",
            "op_physics_basis_axis": "channel",
            "op_physics_address_family": "cosine",
            "op_physics_score_norm_family": "sharpen",
            "op_physics_aggregate_family": "semiring",
            "op_physics_knob_scale": 1.5,
            "op_physics_target": "compression_bottleneck_state",
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
            "op_activation_sparsity_pattern": "learned_structured",
        },
    ),
    _RepairRule(
        frozenset({_WEAK_SCORE_NORM_SOFTMAX_BASIN}),
        "repair_score_norm_spectrum",
        (
            "repair measured softmax-basin score-normalization collapse by "
            "searching Tsallis, Renyi, and entmax-alpha non-softmax spectra"
        ),
        dynamic_delta=_delta_score_norm_spectrum,
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
    return _expand_physics_repairs(repairs)


def _slug(raw: str, *, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", raw).strip("_").lower()
    return (slug or "case")[:max_len].strip("_") or "case"


def _strip_named_composition_axes_for_physics(axes: dict[str, Any]) -> None:
    if axes.get("op_search_track") != "physics_atom":
        return
    for key in (
        "op_algebraic_space",
        "op_math_family",
        "op_math_knobs",
        "op_sparse_matrix_pattern",
        "op_kernel_feature_map",
        "op_graph_topology",
        "op_tensor_decomp_kind",
        "op_tensor_rank",
        "op_activation_sparsity_pattern",
        "op_block_template",
        "op_block_inner_template",
        "op_block_slot_a",
        "op_block_slot_b",
        "op_block_slot_c",
        "op_routing_kind",
        "op_max_depth",
        "op_skip_hard",
        "op_top_k",
    ):
        axes.pop(key, None)


def _spec_from_case_and_repair(
    case: DynamicEvidenceCase, repair: _Repair
) -> ProposalSpec:
    axes = {**case.base_axes, **repair.delta}
    _strip_named_composition_axes_for_physics(axes)
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
            f"root_source_id={case.root_source_id}",
            f"source_score={case.score:.4f}",
            f"repair={repair.name}",
            f"repair_depth={case.repair_depth + 1}",
            f"repair_history={'+'.join((*case.repair_history, repair.name))}",
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
                root_source_id=name,
                name=name,
                base_axes=dict(anchor.axes),
                anchor_axes=dict(anchor.axes),
                score=score,
                weaknesses=weaknesses,
                repair_depth=0,
                repair_history=(),
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
    if include_anchor_fallback:
        # Anchor cases are a recovery pool, not just an empty-ledger fallback:
        # mature ledgers can have abundant evidence whose deterministic repairs
        # are all already terminal. Keep them after ledger cases so measured
        # failures still drive the first proposals.
        evidence_cases = [
            *evidence_cases,
            *_anchor_cases(anchor_op_names, db_path=db_path),
        ]
    if not evidence_cases:
        return []

    terminal_ids = {
        proposal_id
        for proposal_id, entry in ledger.entries.items()
        if entry.promotion_status in (PROMOTION_PROMOTED, PROMOTION_REJECTED)
    }
    value_pool = _axis_value_pool(evidence_cases)
    unseen_specs: list[ProposalSpec] = []
    pending_specs: list[ProposalSpec] = []
    seen_axes: set[tuple[tuple[str, str], ...]] = set()
    for case in evidence_cases:
        for repair in _repairs_for_case(case, value_pool):
            spec = _spec_from_case_and_repair(case, repair)
            if spec.proposal_id in terminal_ids:
                continue
            key = tuple(
                sorted((key, str(value)) for key, value in spec.math_axes.items())
            )
            if key in seen_axes:
                continue
            seen_axes.add(key)
            if ledger.has_seen(spec.proposal_id):
                pending_specs.append(spec)
            else:
                unseen_specs.append(spec)
    return [*unseen_specs, *pending_specs][:max_specs]
