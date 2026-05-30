"""Mechanism-first invention specs for component_fab.

This is deliberately separate from the rehab improvers. Rehab starts from an
underperforming existing op and mutates axes. Invention starts from a concrete
mechanism contract and only then emits a ``ProposalSpec`` so the existing fab
codegen, validators, ledger, and hard-binding LM probes can grade it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from component_fab.proposer.spec_generator import (
    CATEGORY_COMPRESSION,
    CATEGORY_LANE,
    CATEGORY_ROUTING,
    ProposalSpec,
    SYNTHESIS_KIND_NOVEL_HYBRID,
    make_proposal_id,
)

INVENTION_TRACK = "invention"


@dataclass(frozen=True, slots=True)
class InventionBlueprint:
    mechanism_id: str
    category: str
    axes: dict[str, Any]
    information_flow: str
    forgetting_rule: str
    causality_argument: str
    target_failure_mode: str
    expected_baseline: str
    complexity: str
    prior_art_label: str = "mechanistically_new_candidate"


DEFAULT_INVENTION_BLUEPRINTS: tuple[InventionBlueprint, ...] = (
    InventionBlueprint(
        mechanism_id="causal_fast_weight_memory",
        category=CATEGORY_LANE,
        axes={
            "op_invention_mechanism": "causal_fast_weight_memory",
            "op_algebraic_space": "fast_weight_memory",
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_activation_sparsity_pattern": "dense",
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
        },
        information_flow="current token writes key/value outer product into causal fast-weight memory; current query reads memory",
        forgetting_rule="learned scalar decay plus gated current write",
        causality_argument="memory is updated left-to-right and never reads future tokens",
        target_failure_mode="binding failure when exact key/value associations must persist across distractors",
        expected_baseline="causal_conv",
        complexity="O(L * D * M) with M<=D memory projection",
    ),
    InventionBlueprint(
        mechanism_id="causal_slot_router_memory",
        category=CATEGORY_ROUTING,
        axes={
            "op_invention_mechanism": "causal_slot_router_memory",
            "op_algebraic_space": "slot_memory",
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_activation_sparsity_pattern": "learned_structured",
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
        },
        information_flow="token routes to persistent slots, writes gated candidate state, reads route-weighted slot mixture",
        forgetting_rule="per-slot write gate overwrites only selected slot dimensions",
        causality_argument="slots are recurrent state updated strictly in token order",
        target_failure_mode="routing collapse where one global summary erases key-specific state",
        expected_baseline="softmax_attention",
        complexity="O(L * S * D) for S memory slots",
    ),
    InventionBlueprint(
        mechanism_id="hierarchical_residual_compressor",
        category=CATEGORY_COMPRESSION,
        axes={
            "op_invention_mechanism": "hierarchical_residual_compressor",
            "op_algebraic_space": "hierarchical_residual_state",
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(log L)",
            "op_activation_sparsity_pattern": "structured",
            "op_geometric_receptive_field": "hybrid_local_global",
            "op_spectral_preferred_basis": "content",
        },
        information_flow="fixed hierarchy of summaries updates at powers-of-two periods and emits a gated readout",
        forgetting_rule="level-wise gated replacement compresses old residual state",
        causality_argument="summary levels only update from prior summaries and current token",
        target_failure_mode="long-gap recall under a fixed small state budget",
        expected_baseline="causal_conv",
        complexity="O(L * K * D) for K summary levels",
    ),
    InventionBlueprint(
        mechanism_id="symplectic_residual_mixer",
        category=CATEGORY_LANE,
        axes={
            "op_invention_mechanism": "symplectic_residual_mixer",
            "op_algebraic_space": "symplectic_residual",
            "op_dynamical_has_state": 0,
            "op_dynamical_memory_length_class": "O(L)",
            "op_activation_sparsity_pattern": "dense",
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
        },
        information_flow="causal running context is split into q/p halves and mixed by alternating symplectic-style updates",
        forgetting_rule="running context averages prior tokens, dampening stale details",
        causality_argument="context is a causal cumulative statistic",
        target_failure_mode="unstable dense mixing that loses gradient structure across long sequences",
        expected_baseline="softmax_attention",
        complexity="O(L * D^2)",
    ),
    InventionBlueprint(
        mechanism_id="tropical_surprise_memory",
        category=CATEGORY_LANE,
        axes={
            "op_invention_mechanism": "tropical_surprise_memory",
            "op_algebraic_space": "tropical_surprise_memory",
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_activation_sparsity_pattern": "winner_take_all",
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
        },
        information_flow="each token reads memory by max-plus retrieval, writes the surprise (associative prediction error) outer-product as one online gradient step, with momentum + data-dependent forgetting",
        forgetting_rule="data-dependent per-key decay gate plus momentum on the surprise stream (Titans adaptive weight decay)",
        causality_argument="memory is a strict left-to-right scan; output at t reads only memory built from tokens <= t",
        target_failure_mode="cross-key interference in dense associative recall where sum-based linear memory blurs distinct key/value bindings",
        expected_baseline="causal_fast_weight_memory",
        complexity="O(L * M^2) with M<=D memory projection",
    ),
    InventionBlueprint(
        mechanism_id="semiring_surprise_memory",
        category=CATEGORY_LANE,
        axes={
            "op_invention_mechanism": "semiring_surprise_memory",
            "op_algebraic_space": "semiring_surprise_memory",
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_activation_sparsity_pattern": "learned_semiring",
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
        },
        information_flow="each token reads memory by a LEARNABLE tempered-semiring retrieval (1/β)logsumexp_i(β(M[i,j]+addr_i)) whose inverse-temperature β slides the read from arithmetic mean (β->0) to max-plus winner-take-all (β->inf), writes the surprise (associative prediction error) outer-product as one online gradient step with momentum + data-dependent forgetting",
        forgetting_rule="data-dependent per-key decay gate plus momentum on the surprise stream (Titans adaptive weight decay); retrieval sharpness is itself a learned parameter",
        causality_argument="memory is a strict left-to-right scan; output at t reads only memory built from tokens <= t",
        target_failure_mode="cross-key interference in dense associative recall: a fixed max-plus read can be too hard (drops useful soft evidence) and a fixed mean read too soft (blurs distinct bindings); learning beta lets the same op adapt retrieval sharpness per data",
        expected_baseline="tropical_surprise_memory",
        complexity="O(L * M^2) with M<=D memory projection",
    ),
    InventionBlueprint(
        mechanism_id="semiring_surprise_memory_rope",
        category=CATEGORY_LANE,
        axes={
            "op_invention_mechanism": "semiring_surprise_memory_rope",
            "op_algebraic_space": "semiring_surprise_memory",
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_activation_sparsity_pattern": "learned_semiring",
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
        },
        information_flow="semiring_surprise_memory with RoPE applied to the addressing q/k: rotating query and key by absolute position injects the relative phase (t-s) into the memory retrieval score q_t·k_s, giving the delta-rule memory an explicit notion of how far back an association was written; read is the learnable tempered semiring (1/β)logsumexp_i(β(M[i,j]+addr_i))",
        forgetting_rule="data-dependent per-key decay gate plus momentum on the surprise stream; retrieval sharpness β is learned; addressing carries rotary relative position",
        causality_argument="memory is a strict left-to-right scan and RoPE is a per-position rotation of the current token's q/k only; output at t depends only on tokens <= t",
        target_failure_mode="position-agnostic associative recall: a content-only memory address cannot prefer a recent vs distant match; rotary addressing lets retrieval weight by relative distance",
        expected_baseline="semiring_surprise_memory",
        complexity="O(L * M^2) with M<=D memory projection",
    ),
    InventionBlueprint(
        mechanism_id="padic_surprise_memory",
        category=CATEGORY_LANE,
        axes={
            "op_invention_mechanism": "padic_surprise_memory",
            "op_algebraic_space": "padic_surprise_memory",
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_activation_sparsity_pattern": "structured",
            "op_geometric_receptive_field": "global",
            "op_spectral_preferred_basis": "content",
        },
        information_flow="token writes the surprise (associative prediction error) into a hierarchy of ultrametric p-adic memory levels via the delta rule, reads a gated sum across coarse-to-fine levels",
        forgetting_rule="per-level data-dependent decay gate plus momentum; coarse levels share capacity across p-adically-near keys, fine levels isolate exact associations",
        causality_argument="every level is a strict left-to-right recurrent scan; output at t depends only on tokens <= t",
        target_failure_mode="long-gap hierarchical recall where related keys must generalize while exact keys stay isolated under a fixed memory budget",
        expected_baseline="causal_fast_weight_memory",
        complexity="O(L * K * M^2) for K ultrametric levels",
    ),
)


def _axes_for_blueprint(blueprint: InventionBlueprint) -> dict[str, Any]:
    axes = dict(blueprint.axes)
    axes.update(
        {
            "op_search_track": INVENTION_TRACK,
            "op_prior_art_label": blueprint.prior_art_label,
            "op_information_flow": blueprint.information_flow,
            "op_forgetting_rule": blueprint.forgetting_rule,
            "op_causality_argument": blueprint.causality_argument,
            "op_target_failure_mode": blueprint.target_failure_mode,
            "op_expected_baseline": blueprint.expected_baseline,
            "op_complexity": blueprint.complexity,
        }
    )
    return axes


def spec_from_blueprint(blueprint: InventionBlueprint) -> ProposalSpec:
    axes = _axes_for_blueprint(blueprint)
    name = f"invent_{blueprint.mechanism_id}"
    return ProposalSpec(
        proposal_id=make_proposal_id(name, axes),
        name=name,
        category=blueprint.category,
        synthesis_kind=SYNTHESIS_KIND_NOVEL_HYBRID,
        math_axes=axes,
        anchor_witness_op="",
        anchor_witnesses_all=(),
        declared_property_row={
            **axes,
            "op_n_inputs": 1,
            "op_is_parameterized": 1,
            "op_is_stateless": 0 if axes.get("op_dynamical_has_state") else 1,
        },
        predicted_lift=0.5,
        rationale=(
            f"Invention-track mechanism {blueprint.mechanism_id}: "
            f"{blueprint.information_flow}. Targets {blueprint.target_failure_mode}."
        ),
        notes=(
            f"track={INVENTION_TRACK}",
            f"complexity={blueprint.complexity}",
            f"expected_baseline={blueprint.expected_baseline}",
            f"prior_art_label={blueprint.prior_art_label}",
        ),
    )


def enumerate_invention_specs(
    blueprints: Iterable[InventionBlueprint] = DEFAULT_INVENTION_BLUEPRINTS,
) -> list[ProposalSpec]:
    return [spec_from_blueprint(blueprint) for blueprint in blueprints]


def invention_gate_reasons(spec: ProposalSpec) -> tuple[str, ...]:
    """Return reasons a spec is not acceptable for the invention track."""
    reasons: list[str] = []
    axes = spec.math_axes
    if axes.get("op_search_track") != INVENTION_TRACK:
        reasons.append("missing invention search track")
    if not axes.get("op_invention_mechanism"):
        reasons.append("missing mechanism id")
    if spec.anchor_witness_op or spec.anchor_witnesses_all:
        reasons.append("anchored rehab/cross-anchor spec")
    if axes.get("op_math_knobs"):
        reasons.append("adapter composition belongs in rehab track")
    for key in (
        "op_information_flow",
        "op_forgetting_rule",
        "op_causality_argument",
        "op_target_failure_mode",
        "op_expected_baseline",
        "op_complexity",
    ):
        if not axes.get(key):
            reasons.append(f"missing contract field {key}")
    return tuple(reasons)


def is_invention_spec(spec: ProposalSpec) -> bool:
    return not invention_gate_reasons(spec)
