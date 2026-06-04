"""Curated external research priors for fab proposal quality.

These priors encode published architecture families that are relevant to the
component_fab search (long-context memory, recursive/sparse structure, SSM/MoE
hybrids, structured-matrix attention, mixture-of-memory, structure-preserving
operators). Each prior maps a paper family onto the *local* axis vocabulary and
validation tasks so a candidate spec can be scored for affinity.

Hard rule (from the plan): priors only influence proposal generation and
exploration budget. They never directly promote a candidate — downstream Tier-2
/ BLiMP evidence remains the only promotion gate.

Source of truth is the ``RESEARCH_PRIORS`` table below. ``to_catalog_rows``
emits rows shaped for ``external_component_prior_catalog`` so they can be
imported by a separate tool; this module performs no DB access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from component_fab.proposer.spec_generator import ProposalSpec


@dataclass(frozen=True, slots=True)
class ResearchPrior:
    """A curated published-architecture prior mapped to local search axes."""

    family: str
    summary: str
    mapped_ops: tuple[str, ...]
    mapped_templates: tuple[str, ...]
    # axis -> accepted values; a spec matches the axis if its value is in the set.
    mapped_axes: Mapping[str, tuple[Any, ...]]
    mapped_knobs: tuple[str, ...]
    expected_strength: str
    expected_risk: str
    validation_tasks: tuple[str, ...]
    hardware_note: str
    confidence: float
    source_url: str
    tags: tuple[str, ...] = field(default_factory=tuple)


# fmt: off
RESEARCH_PRIORS: tuple[ResearchPrior, ...] = (
    ResearchPrior(
        family="chunked_attention_gated_fifo_memory",
        summary="Chunked/local attention with a gated FIFO long-context memory bank "
        "that carries key/value state across chunks.",
        mapped_ops=("route_lanes", "softmax_attention"),
        mapped_templates=("recursive_depth_router", "three_lane_adaptive"),
        mapped_axes={
            "op_dynamical_has_state": (1,),
            "op_dynamical_memory_length_class": ("O(L)",),
            "op_geometric_receptive_field": ("global", "hybrid_local_global"),
        },
        mapped_knobs=(),
        expected_strength="long-range key/value recall at sub-quadratic cost via "
        "carried gated memory",
        expected_risk="gate saturation / stale memory if write gate is untuned",
        validation_tasks=("long_gap_recall", "multi_query_kv_recall"),
        hardware_note="native scan kernel preferred for the FIFO state update "
        "(see native_surprise_memory_cuda.cu pattern)",
        confidence=0.72,
        source_url="https://arxiv.org/abs/2507.00453",
        tags=("memory_augmented", "long_context"),
    ),
    ResearchPrior(
        family="ressformer_sparse_recursive_structured",
        summary="ReSSFormer: recursive depth with sparse structured routing for "
        "compositional generalization.",
        mapped_ops=("route_recursion", "moe_topk"),
        mapped_templates=("recursive_depth_router", "sparse_moe_block"),
        mapped_axes={
            "op_routing_kind": ("depth_router", "sparse_depth", "top_k_moe"),
            "op_activation_sparsity_pattern": ("top_k", "learned_structured"),
        },
        mapped_knobs=(),
        expected_strength="compositional binding via reusable recursive structure",
        expected_risk="router collapse / non-differentiable hard skips",
        validation_tasks=("compositional_binding", "heldout_pair_recall"),
        hardware_note="recursion router is Python-light; keep hard-skip gate "
        "differentiable (multiplicative g in [0,1])",
        confidence=0.68,
        source_url="https://arxiv.org/abs/2510.01585",
        tags=("recursive", "sparse_routing"),
    ),
    ResearchPrior(
        family="hydra_ssm_sparse_attn_moe_memory",
        summary="Hydra-style hybrid: SSM + sparse attention + MoE + memory in one "
        "block stack.",
        mapped_ops=("route_lanes", "moe_topk", "softmax_attention"),
        mapped_templates=("three_lane_adaptive", "hetero_moe_block"),
        mapped_axes={
            "op_dynamical_has_state": (1,),
            "op_dynamical_memory_length_class": ("O(L)",),
            "op_routing_kind": ("difficulty", "top_k_moe", "low_info_skip"),
        },
        mapped_knobs=(),
        expected_strength="broad capability coverage (recall + binding) from "
        "heterogeneous lanes",
        expected_risk="lane imbalance; expensive to train; hard to attribute wins",
        validation_tasks=("long_gap_recall", "variable_layout_recall",
                          "multi_query_kv_recall"),
        hardware_note="SSM lane needs a native selective-scan; MoE routing stays "
        "Python orchestration",
        confidence=0.65,
        source_url="https://arxiv.org/abs/2508.15099",
        tags=("hybrid", "ssm", "moe"),
    ),
    ResearchPrior(
        family="log_linear_structured_matrix_attention",
        summary="Log-linear / structured-matrix attention and Gated-DeltaNet style "
        "linear-time mixing with structured state transitions.",
        mapped_ops=("route_lanes",),
        mapped_templates=("attn_spectral_filter", "latent_compress"),
        mapped_axes={
            "op_dynamical_has_state": (1,),
            "op_dynamical_memory_length_class": ("O(L)",),
            "op_activation_sparsity_pattern": ("structured", "learned_structured"),
        },
        mapped_knobs=("linear_algebra_low_rank", "sparse_matrix_banded",
                      "low_rank_factorized"),
        expected_strength="linear-time long-range recall with structured decay",
        expected_risk="capacity loss at small width; structured matrix brittleness",
        validation_tasks=("long_gap_recall", "distractor_kv_recall"),
        hardware_note="structured-matrix scan must be native-backed; no Python "
        "inner loop over sequence",
        confidence=0.7,
        source_url="https://www.sciencestack.ai/paper/2506.04761v2",
        tags=("linear_attention", "structured_matrix"),
    ),
    ResearchPrior(
        family="mixture_of_memory_state",
        summary="Mixture-of-Memory / mixture-of-state: multiple specialized memory "
        "slots selected per token.",
        mapped_ops=("moe_topk", "route_lanes"),
        mapped_templates=("hetero_moe_block", "sparse_moe_block"),
        mapped_axes={
            "op_dynamical_has_state": (1,),
            "op_routing_kind": ("top_k_moe", "difficulty", "hash"),
            "op_activation_sparsity_pattern": ("top_k",),
        },
        mapped_knobs=(),
        expected_strength="specialized recall channels for multi-query / "
        "variable-layout binding",
        expected_risk="memory-slot collapse; load imbalance across slots",
        validation_tasks=("multi_query_kv_recall", "compositional_binding"),
        hardware_note="per-slot state update should be native; gating is "
        "Python-light",
        confidence=0.66,
        source_url="https://www.aimodels.fyi/papers/arxiv/mom-linear-sequence-modeling-mixture-memories",
        tags=("mixture_of_memory", "routing"),
    ),
    ResearchPrior(
        family="symplectic_hamiltonian_operator",
        summary="Symplectic / Hamiltonian neural operators: structure-preserving "
        "dynamics for stable long-horizon recurrence.",
        mapped_ops=("route_lanes",),
        mapped_templates=("attn_spectral_filter", "recursive_depth"),
        mapped_axes={
            "op_dynamical_has_state": (1,),
            "op_dynamical_memory_length_class": ("O(L)",),
            "op_spectral_preferred_basis": ("content",),
        },
        mapped_knobs=("graph_laplacian_diffusion", "causal_path_laplacian",
                      "calculus_finite_difference"),
        expected_strength="stable long-horizon dynamics (no blow-up) — repairs "
        "stability eliminations",
        expected_risk="immature kernels; numerical integration cost",
        validation_tasks=("long_gap_recall",),
        hardware_note="symplectic integrator step must be native; energy-conserving "
        "update is the hot path",
        confidence=0.55,
        source_url="https://arxiv.org/abs/2605.15881",
        tags=("structure_preserving", "stability"),
    ),
)
# fmt: on


def load_research_priors() -> tuple[ResearchPrior, ...]:
    """Return the curated research-prior table (source of truth)."""

    return RESEARCH_PRIORS


@dataclass(frozen=True, slots=True)
class PriorAffinity:
    """How well a spec aligns with the closest curated research prior."""

    family: str
    affinity: float
    matched_axes: tuple[str, ...]
    reasons: tuple[str, ...]
    confidence: float
    validation_tasks: tuple[str, ...]


def _prior_signal_count(prior: ResearchPrior) -> int:
    return (
        len(prior.mapped_axes)
        + (1 if prior.mapped_templates else 0)
        + (1 if prior.mapped_knobs else 0)
    )


def _spec_knobs(spec: ProposalSpec) -> set[str]:
    raw = str(spec.math_axes.get("op_math_knobs") or "")
    return {part for part in raw.split("+") if part}


def _score_prior(spec: ProposalSpec, prior: ResearchPrior) -> PriorAffinity:
    axes = spec.math_axes
    matched_axes: list[str] = []
    reasons: list[str] = []
    hits = 0
    for axis, accepted in prior.mapped_axes.items():
        if axis in axes and axes[axis] in accepted:
            matched_axes.append(axis)
            reasons.append(f"{axis}={axes[axis]} matches {prior.family}")
            hits += 1
    template = str(axes.get("op_block_template") or "")
    if template and template in prior.mapped_templates:
        reasons.append(f"block_template={template} matches {prior.family}")
        hits += 1
    knobs = _spec_knobs(spec)
    if knobs and knobs.intersection(prior.mapped_knobs):
        shared = sorted(knobs.intersection(prior.mapped_knobs))
        reasons.append(f"knobs {shared} match {prior.family}")
        hits += 1
    total = max(1, _prior_signal_count(prior))
    affinity = min(1.0, hits / total)
    return PriorAffinity(
        family=prior.family,
        affinity=affinity,
        matched_axes=tuple(matched_axes),
        reasons=tuple(reasons),
        confidence=prior.confidence,
        validation_tasks=prior.validation_tasks,
    )


def prior_affinity_for_spec(
    spec: ProposalSpec,
    priors: Sequence[ResearchPrior] | None = None,
) -> PriorAffinity:
    """Return the best-matching curated prior for a spec (affinity in [0, 1]).

    Returns a zero-affinity ``unknown`` result when no prior shares any signal,
    so callers always get a well-formed record.
    """

    candidates = priors if priors is not None else RESEARCH_PRIORS
    best = PriorAffinity(
        family="unknown",
        affinity=0.0,
        matched_axes=(),
        reasons=("no curated research prior matched this spec",),
        confidence=0.0,
        validation_tasks=(),
    )
    for prior in candidates:
        scored = _score_prior(spec, prior)
        if scored.affinity <= 0.0:
            continue  # a prior that shares no signal must not displace "unknown"
        # Tie-break by confidence so a more-trusted family wins equal affinity.
        if (scored.affinity, scored.confidence) > (best.affinity, best.confidence):
            best = scored
    return best


def to_catalog_rows() -> list[dict[str, Any]]:
    """Emit rows shaped for ``external_component_prior_catalog`` (no DB write).

    The catalog schema has no dedicated validation-task column, so the validation
    task mapping and source URL are carried in ``tags_json`` / ``source_ref``.
    """

    rows: list[dict[str, Any]] = []
    for prior in RESEARCH_PRIORS:
        tags = list(prior.tags) + [
            f"validation:{task}" for task in prior.validation_tasks
        ]
        rows.append(
            {
                "external_family": prior.family,
                "mapped_ops_json": json.dumps(list(prior.mapped_ops)),
                "mapped_templates_json": json.dumps(list(prior.mapped_templates)),
                "expected_strength": prior.expected_strength,
                "expected_risk": prior.expected_risk,
                "hardware_note": prior.hardware_note,
                "tags_json": json.dumps(tags),
                "confidence": prior.confidence,
                "source_ref": prior.source_url,
            }
        )
    return rows
