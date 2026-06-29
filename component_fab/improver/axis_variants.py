"""Axis-variant generator for goal-(b) anchor ops.

For each anchor op (a "novel-looking but underperforming" op surfaced by
the scoper — tropical_attention, clifford_attention, padic_gate, etc.)
this module enumerates a small set of axis-delta variants and emits
``ProposalSpec`` objects ready for ``code_generator`` + ``solo`` validator.

A variant is an axis-tuple change relative to the anchor — e.g. "+state"
flips ``op_dynamical_has_state=1`` and ``op_dynamical_memory_length_class=O(L)``.
The idea: the anchor's math is novel but underperforms; one axis change
may unlock it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ..proposer.spec_generator import ProposalSpec, build_spec_from_axes
from .math_knobs import DEFAULT_MATH_KNOBS, MathKnob

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_META_DB = _REPO / "research" / "meta_analysis.db"

_AXES_OF_INTEREST: tuple[str, ...] = (
    "op_algebraic_space",
    "op_spectral_preferred_basis",
    "op_dynamical_memory_length_class",
    "op_dynamical_has_state",
    "op_activation_sparsity_pattern",
    "op_geometric_receptive_field",
)


@dataclass(frozen=True, slots=True)
class AxisVariant:
    delta_name: str
    delta: dict[str, Any]
    rationale: str


@dataclass(frozen=True, slots=True)
class AnchorAxes:
    op_name: str
    axes: dict[str, Any]
    eval_count: int
    pass_rate: float


def _variant_from_math_knob(knob: MathKnob) -> AxisVariant:
    """Derive a single-knob ``AxisVariant`` from the canonical knob definition.

    Keeps the low-cost knob axes (``DEFAULT_MATH_KNOBS``) as the single
    source of truth instead of retyping the same dicts here.
    """
    return AxisVariant(
        delta_name=knob.knob_id,
        delta={"op_math_family": knob.family, **knob.axes},
        rationale=knob.rationale,
    )


_KNOB_BY_ID: dict[str, MathKnob] = {knob.knob_id: knob for knob in DEFAULT_MATH_KNOBS}


DEFAULT_AXIS_VARIANT_TEMPLATES: tuple[AxisVariant, ...] = (
    AxisVariant(
        delta_name="add_state_OL",
        delta={
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
        },
        rationale="add SSM-style running state on the sequence dim",
    ),
    AxisVariant(
        delta_name="top_k_sparsity",
        delta={"op_activation_sparsity_pattern": "top_k"},
        rationale="replace dense activation with top-k sparsity",
    ),
    AxisVariant(
        delta_name="fourier_basis",
        delta={"op_spectral_preferred_basis": "frequency"},
        rationale="apply the op in the frequency basis along sequence",
    ),
    AxisVariant(
        delta_name="global_receptive",
        delta={"op_geometric_receptive_field": "global"},
        rationale="widen to global receptive field",
    ),
    _variant_from_math_knob(_KNOB_BY_ID["calculus_finite_difference"]),
    _variant_from_math_knob(_KNOB_BY_ID["linear_algebra_low_rank"]),
    _variant_from_math_knob(_KNOB_BY_ID["sparse_matrix_banded"]),
    # Routing variants — wrap the anchor's primitive in a per-token
    # compute-allocation router. Added 2026-05-15 (Phase 4.5).
    AxisVariant(
        delta_name="route_depth_router",
        delta={"op_routing_kind": "depth_router", "op_max_depth": 4},
        rationale="ACT-style mixture-of-recursions: per-token learned halt",
    ),
    AxisVariant(
        delta_name="route_site_recursion_mixer",
        delta={
            "op_routing_kind": "site_recursion",
            "op_recursion_sites": "mixer",
            "op_max_depth": 4,
        },
        rationale=(
            "general RecursionSite wrapper over the anchor mixer; first slice of "
            "recursion as a weighted-site search axis"
        ),
    ),
    AxisVariant(
        delta_name="route_site_recursion_ffn",
        delta={
            "op_routing_kind": "site_recursion",
            "op_recursion_sites": "ffn",
            "op_max_depth_ffn": 4,
        },
        rationale=(
            "recurse the position-wise FFN site (mixer kept single-pass): give "
            "channel mixing learned per-token depth"
        ),
    ),
    AxisVariant(
        delta_name="route_site_recursion_router",
        delta={
            "op_routing_kind": "site_recursion",
            "op_recursion_sites": "router",
            "op_max_depth_router": 4,
        },
        rationale="recurse a top-k router/gate site over the anchor's expert pool",
    ),
    AxisVariant(
        delta_name="route_site_recursion_embedding",
        delta={
            "op_routing_kind": "site_recursion",
            "op_recursion_sites": "embedding",
            "op_max_depth_embedding": 4,
        },
        rationale=(
            "recurse a low-rank refinement of the embedded representation at the "
            "lane input"
        ),
    ),
    AxisVariant(
        delta_name="route_site_recursion_mixer_ffn",
        delta={
            "op_routing_kind": "site_recursion",
            "op_recursion_sites": "mixer+ffn",
            "op_max_depth_mixer": 4,
            "op_max_depth_ffn": 4,
        },
        rationale=(
            "recurse both the mixer and FFN sites with independent per-token "
            "depth: recursion anywhere there are weights"
        ),
    ),
    AxisVariant(
        delta_name="route_sparse_depth",
        delta={"op_routing_kind": "sparse_depth", "op_max_depth": 4},
        rationale="top-25% tokens get extra recursion passes; others pin at depth=1",
    ),
    AxisVariant(
        delta_name="route_low_info_skip_soft",
        delta={"op_routing_kind": "low_info_skip", "op_skip_hard": 0},
        rationale="low-content tokens route through low-rank cheap path",
    ),
    AxisVariant(
        delta_name="route_low_info_skip_hard",
        delta={"op_routing_kind": "low_info_skip", "op_skip_hard": 1},
        rationale="low-content tokens bypass mixer entirely (residual only)",
    ),
    AxisVariant(
        delta_name="route_difficulty",
        delta={"op_routing_kind": "difficulty"},
        rationale="per-token 1-bit router: easy SSM lane vs hard anchor lane",
    ),
    AxisVariant(
        delta_name="route_hash_moe",
        delta={"op_routing_kind": "hash"},
        rationale="hash-routed 3-expert MoE (attn + ssm + topk), no aux loss",
    ),
    AxisVariant(
        delta_name="route_top_k_moe",
        delta={"op_routing_kind": "top_k_moe", "op_top_k": 2},
        rationale="switch-style top-2 MoE with load-balancing aux loss",
    ),
    # Block-template variants — compose the anchor's primitive inside a
    # multi-lane block (cf3e6bc6-class winners use these). Added
    # 2026-05-15 (Phase 4).
    AxisVariant(
        delta_name="block_latent_compress",
        delta={"op_block_template": "latent_compress", "op_block_compress": 2},
        rationale="anchor mixer runs at compressed inner dim (d -> d/2)",
    ),
    AxisVariant(
        delta_name="block_three_lane_adaptive",
        delta={"op_block_template": "three_lane_adaptive"},
        rationale="3 parallel lanes (anchor + tropical attn + ssm) via softmax gate",
    ),
    AxisVariant(
        delta_name="block_recursive_depth",
        delta={"op_block_template": "recursive_depth", "op_max_depth": 3},
        rationale="per-sequence content-conditional re-application of anchor mixer",
    ),
    AxisVariant(
        delta_name="block_gated_parallel",
        delta={"op_block_template": "gated_parallel"},
        rationale="2-lane gate (anchor + wavelet multiscale)",
    ),
    AxisVariant(
        delta_name="block_loss_monster_pair_hyper_mor",
        delta={
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
        rationale=(
            "pair a local loss-specialist lane with the HyperMoR long-range "
            "carrier and floor the carrier gate so loss cannot starve it"
        ),
    ),
    AxisVariant(
        delta_name="block_loss_monster_pair_slot_dplr",
        delta={
            "op_block_template": "loss_monster_paired",
            "op_partner_kind": "slot_dplr",
            "op_block_slot_loss": "routed_bottleneck",
            "op_partner_floor": 0.5,
            "op_candidate_role": "loss_specialist_pair",
            "op_loss_specialist_paired": 1,
            "op_loss_specialist_partner_op": "slot_dplr",
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_geometric_receptive_field": "global",
        },
        rationale=(
            "pair a local loss-specialist lane with slot/DPLR-style memory "
            "so the pair is graded against a recall-capable carrier"
        ),
    ),
    AxisVariant(
        delta_name="block_loss_monster_pair_native_semiring",
        delta={
            "op_block_template": "loss_monster_paired",
            "op_partner_kind": "native_semiring",
            "op_block_slot_loss": "routed_bottleneck",
            "op_partner_floor": 0.5,
            "op_candidate_role": "loss_specialist_pair",
            "op_loss_specialist_paired": 1,
            "op_loss_specialist_partner_op": "native_semiring",
            "op_dynamical_has_state": 1,
            "op_dynamical_memory_length_class": "O(L)",
            "op_geometric_receptive_field": "global",
        },
        rationale=(
            "pair a local loss-specialist lane with native semiring surprise "
            "memory and grade only the carrier-relative capability delta"
        ),
    ),
    # Day-3 (2026-05-15): block_gated_parallel was the day-2 BLiMP winner
    # (+0.0071 vs softmax). Adding slot_b variants to explore which
    # second-lane mixer pairs best with the anchor.
    AxisVariant(
        delta_name="block_gated_parallel_slot_fisher",
        delta={
            "op_block_template": "gated_parallel",
            "op_block_slot_b": "fisher_attention",
        },
        rationale="gated_parallel with Fisher-info attention as second lane",
    ),
    AxisVariant(
        delta_name="block_gated_parallel_slot_chebyshev",
        delta={
            "op_block_template": "gated_parallel",
            "op_block_slot_b": "chebyshev_spectral",
        },
        rationale="gated_parallel with Chebyshev positional basis as second lane",
    ),
    AxisVariant(
        delta_name="block_gated_parallel_slot_fourier",
        delta={
            "op_block_template": "gated_parallel",
            "op_block_slot_b": "fourier_basis",
        },
        rationale="gated_parallel with rFFT basis as second lane (frequency mixing)",
    ),
    AxisVariant(
        delta_name="block_gated_parallel_slot_poincare",
        delta={"op_block_template": "gated_parallel", "op_block_slot_b": "poincare"},
        rationale="gated_parallel with Poincaré-ball hyperbolic attention as second lane",
    ),
    AxisVariant(
        delta_name="block_three_lane_fisher_chebyshev",
        delta={
            "op_block_template": "three_lane_adaptive",
            "op_block_slot_b": "fisher_attention",
            "op_block_slot_c": "chebyshev_spectral",
        },
        rationale="3-lane adaptive: anchor + Fisher attn + Chebyshev spectral",
    ),
    # New algebraic spaces — quaternion (Hamilton-product attention) and
    # hyperbolic (Poincaré-ball attention). Phase 1 from the expansion
    # plan; 2026-05-15.
    AxisVariant(
        delta_name="space_quaternion",
        delta={"op_algebraic_space": "quaternion"},
        rationale="quaternion-valued attention via Hamilton product affinity + composition",
    ),
    AxisVariant(
        delta_name="space_hyperbolic_poincare",
        delta={"op_algebraic_space": "hyperbolic_poincare"},
        rationale="Poincaré-ball hyperbolic attention (negative squared hyp distance)",
    ),
    # Phase 2 math knob families (2026-05-15): information_geometry,
    # spectral_graph, tensor_decomp. Each variant turns the anchor's
    # mixer into base + this-knob adapter via _apply_math_knobs.
    AxisVariant(
        delta_name="knob_fisher_attention",
        delta={
            "op_math_family": "information_geometry",
            "op_info_geom_operator": "fisher_attention",
            "op_math_knobs": "info_geom_fisher",
        },
        rationale="Fisher-information metric affinity (distribution-space attention)",
    ),
    AxisVariant(
        delta_name="knob_chebyshev_spectral",
        delta={
            "op_math_family": "spectral_graph",
            "op_spectral_graph_operator": "chebyshev_polynomial",
            "op_math_knobs": "spectral_chebyshev",
        },
        rationale="Chebyshev polynomial position basis (conditioning-optimal on aperiodic signals)",
    ),
    AxisVariant(
        delta_name="knob_tucker_decomp",
        delta={
            "op_math_family": "tensor_decomp",
            "op_tensor_decomp_kind": "tucker",
            "op_math_knobs": "tensor_tucker",
        },
        rationale="Tucker-decomposed channel mix (core tensor × mode matrices)",
    ),
    # Day-6 (2026-05-15): "turn more knobs, search harder, search longer".
    # Deeper recursion + nested block-of-block + wider slot Cartesian.
    AxisVariant(
        delta_name="route_mor_depth8",
        delta={"op_routing_kind": "depth_router", "op_max_depth": 8},
        rationale="MoR with deeper recursion budget (max_depth=8)",
    ),
    AxisVariant(
        delta_name="route_site_recursion_mixer_depth8",
        delta={
            "op_routing_kind": "site_recursion",
            "op_recursion_sites": "mixer",
            "op_max_depth": 8,
        },
        rationale="RecursionSite over the anchor mixer with a deeper recursion budget",
    ),
    AxisVariant(
        delta_name="route_mor_depth16",
        delta={"op_routing_kind": "depth_router", "op_max_depth": 16},
        rationale="MoR with maximum recursion budget (max_depth=16)",
    ),
    AxisVariant(
        delta_name="route_sparse_depth8",
        delta={"op_routing_kind": "sparse_depth", "op_max_depth": 8},
        rationale="SparseMoR with deeper recursion (max_depth=8)",
    ),
    AxisVariant(
        delta_name="block_recursive_depth8",
        delta={"op_block_template": "recursive_depth", "op_max_depth": 8},
        rationale="RecursiveDepthBlock with max_depth=8 (anchor re-applied 8 times causally)",
    ),
    # Block-of-block: outer block wraps an inner block as its anchor slot.
    AxisVariant(
        delta_name="block_gated_nest_latent_compress",
        delta={
            "op_block_template": "gated_parallel",
            "op_block_inner_template": "latent_compress",
        },
        rationale="gated_parallel whose anchor lane is itself a latent_compress block",
    ),
    AxisVariant(
        delta_name="block_gated_nest_recursive_depth",
        delta={
            "op_block_template": "gated_parallel",
            "op_block_inner_template": "recursive_depth",
            "op_max_depth": 4,
        },
        rationale="gated_parallel + inner recursive_depth (per-token gate over recursively-applied anchor)",
    ),
    AxisVariant(
        delta_name="block_three_lane_nest_gated",
        delta={
            "op_block_template": "three_lane_adaptive",
            "op_block_inner_template": "gated_parallel",
        },
        rationale="3-lane adaptive whose first lane is a gated_parallel block",
    ),
    AxisVariant(
        delta_name="block_latent_compress_nest_gated",
        delta={
            "op_block_template": "latent_compress",
            "op_block_inner_template": "gated_parallel",
            "op_block_compress": 2,
        },
        rationale="compress-then-gated: gated_parallel mixer runs at compressed inner dim",
    ),
    # Block templates with explicit knob slots (turn the block into a knob switch).
    AxisVariant(
        delta_name="block_gated_slot_tropical",
        delta={
            "op_block_template": "gated_parallel",
            "op_block_slot_b": "tropical_attention",
        },
        rationale="gated_parallel anchor + tropical attention",
    ),
    AxisVariant(
        delta_name="block_gated_slot_quaternion",
        delta={"op_block_template": "gated_parallel", "op_block_slot_b": "quaternion"},
        rationale="gated_parallel anchor + quaternion attention",
    ),
    AxisVariant(
        delta_name="block_gated_slot_ssm",
        delta={
            "op_block_template": "gated_parallel",
            "op_block_slot_b": "linear_state_space",
        },
        rationale="gated_parallel anchor + state-space lane",
    ),
    AxisVariant(
        delta_name="block_gated_slot_graph",
        delta={
            "op_block_template": "gated_parallel",
            "op_block_slot_b": "graph_diffusion",
        },
        rationale="gated_parallel anchor + graph diffusion",
    ),
    AxisVariant(
        delta_name="block_three_lane_attn_graph",
        delta={
            "op_block_template": "three_lane_adaptive",
            "op_block_slot_b": "tropical_attention",
            "op_block_slot_c": "graph_diffusion",
        },
        rationale="3-lane: anchor + tropical attn + graph diffusion",
    ),
    AxisVariant(
        delta_name="block_three_lane_wavelet_quaternion",
        delta={
            "op_block_template": "three_lane_adaptive",
            "op_block_slot_b": "multiscale_wavelet",
            "op_block_slot_c": "quaternion",
        },
        rationale="3-lane: anchor + multiscale wavelet + quaternion",
    ),
    AxisVariant(
        delta_name="block_three_lane_random_features_poincare",
        delta={
            "op_block_template": "three_lane_adaptive",
            "op_block_slot_b": "random_features",
            "op_block_slot_c": "poincare",
        },
        rationale="3-lane: anchor + random-feature kernel + Poincaré hyperbolic",
    ),
    # Day-6 (2026-05-15): 6 missing block templates mined from runs.db
    # top-25 BLiMP winners. Each was used by ≥1 architecture scoring
    # ≥ 0.565 on BLiMP in research/'s leaderboard.
    AxisVariant(
        delta_name="block_recursive_depth_router",
        delta={"op_block_template": "recursive_depth_router", "op_max_depth": 4},
        rationale="full-block recursion with per-token halt (most common cf3e6bc6-class winner)",
    ),
    AxisVariant(
        delta_name="block_recursive_depth_router_d8",
        delta={"op_block_template": "recursive_depth_router", "op_max_depth": 8},
        rationale="recursive_depth_router with max_depth=8",
    ),
    AxisVariant(
        delta_name="block_sparse_moe",
        delta={"op_block_template": "sparse_moe_block", "op_top_k": 2},
        rationale="top-2 sparse MoE block (anchor + 3 experts) with load-balancing loss",
    ),
    AxisVariant(
        delta_name="block_hetero_moe",
        delta={"op_block_template": "hetero_moe_block"},
        rationale="heterogeneous-expert MoE (anchor + 4 diverse-class lanes via softmax gate)",
    ),
    AxisVariant(
        delta_name="block_hyperbolic_bridge",
        delta={"op_block_template": "hyperbolic_bridge"},
        rationale="euclidean ↔ Poincaré chart bridge with per-token gate",
    ),
    AxisVariant(
        delta_name="block_attn_spectral_filter",
        delta={"op_block_template": "attn_spectral_filter"},
        rationale="anchor attention piped through learned spectral filter",
    ),
    AxisVariant(
        delta_name="block_graph_attention",
        delta={"op_block_template": "graph_attention"},
        rationale="edge-conditioned attention with learned low-rank adjacency",
    ),
)


def anchor_axes_for_op(
    op_name: str, db_path: Path | str = DEFAULT_META_DB
) -> AnchorAxes | None:
    """Look up the declared math axes for ``op_name`` from op_property_catalog."""
    path = Path(db_path)
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM op_property_catalog WHERE op_name = ?", (op_name,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    axes = {key: row[key] for key in _AXES_OF_INTEREST if key in row.keys()}
    evals = int(row["eval_count"] or 0)
    s1 = int(row["s1_pass_count"] or 0)
    return AnchorAxes(
        op_name=op_name,
        axes=axes,
        eval_count=evals,
        pass_rate=(s1 / evals) if evals else 0.0,
    )


def spec_for_variant(anchor: AnchorAxes, variant: AxisVariant) -> ProposalSpec:
    merged: dict[str, Any] = {**anchor.axes, **variant.delta}
    notes = (
        f"anchor={anchor.op_name} "
        f"(pass_rate={anchor.pass_rate:.2f} on {anchor.eval_count} evals)",
        variant.rationale,
    )
    # This path historically fingerprints the dispatched axes (including
    # the mirrored synthesis_kind) — preserved for proposal_id stability.
    return build_spec_from_axes(
        f"improve_{anchor.op_name}_{variant.delta_name}",
        merged,
        witness_ops=(anchor.op_name,),
        anchor_axes=anchor.axes,
        notes=notes,
        fingerprint_dispatched_axes=True,
    )


def enumerate_axis_variants(
    anchor_op_names: Sequence[str],
    *,
    variants: Sequence[AxisVariant] = DEFAULT_AXIS_VARIANT_TEMPLATES,
    db_path: Path | str = DEFAULT_META_DB,
) -> list[ProposalSpec]:
    out: list[ProposalSpec] = []
    for name in anchor_op_names:
        anchor = anchor_axes_for_op(name, db_path=db_path)
        if anchor is None:
            continue
        for variant in variants:
            out.append(spec_for_variant(anchor, variant))
    return out
