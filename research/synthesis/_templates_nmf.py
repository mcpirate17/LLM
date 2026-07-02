"""NM-F operator-family templates — forced-structure mixers as proposable blocks.

Same shape as the NM-C family (``_templates_compaction.py``): one general block
sampling all wired NM-F ops, plus a routing-capable variant whose primary is
always an op in ``ROUTING_COMPRESSION_MOE_OPS`` (CDMA hard top-1 slot
addressing = token-conditional routing; the integral-control anti-windup gate
= gating), so the family is slot-0 eligible under ``routing_mandatory``. Every
NM-F op is identity-at-init with an internal residual/zero-init lift, so the
scaffold stays minimal.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph
from ._template_helpers import (
    MotifWeights,
    _fix_dim,
    template_add_op as _add,
    template_add_residual as _residual,
)
from ._templates_attention_advanced import (
    _pick_ffn_or_swiglu,
    _pick_norm_or_default,
)

# Ops with a routing/gating mechanism (members of ROUTING_COMPRESSION_MOE_OPS).
_NMF_ROUTING_OPS = (
    "cdma_slot_binding",
    "integral_control_mixer",
)
_NMF_STATE_OPS = (
    "idempotent_oblique_memory",
    "nilpotent_lie_scan",
    "port_hamiltonian_mix",
)
NMF_OPS = _NMF_ROUTING_OPS + _NMF_STATE_OPS


def _build_nmf_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights,
    *,
    primary_pool: tuple[str, ...],
    template_ctx: str,
) -> int:
    """norm → <primary NM-F mixer> → [second NM-F mixer] → residual → [FFN]."""
    normed = _pick_norm_or_default(
        graph, input_id, rng, weights, fallback_context=f"{template_ctx}.norm1"
    )

    primary = rng.choice(primary_pool)
    mixed = _add(graph, primary, [normed], context=f"{template_ctx}.{primary}")

    if rng.random() < 0.5:
        secondary_pool = tuple(op for op in NMF_OPS if op != primary)
        secondary = rng.choice(secondary_pool)
        mixed = _add(graph, secondary, [mixed], context=f"{template_ctx}.{secondary}")

    mixed = _fix_dim(graph, mixed)
    out = _residual(graph, input_id, mixed, context=f"{template_ctx}.residual")

    if rng.random() < 0.5:
        normed2 = _pick_norm_or_default(
            graph, out, rng, weights, fallback_context=f"{template_ctx}.norm2"
        )
        ffn = _pick_ffn_or_swiglu(
            graph, normed2, rng, weights, fallback_context=f"{template_ctx}.ffn"
        )
        ffn = _fix_dim(graph, ffn)
        out = _residual(graph, out, ffn, context=f"{template_ctx}.ffn_residual")

    return out


def tpl_nmf_mixer_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → <any NM-F operator> → residual → [norm → FFN → residual]."""
    return _build_nmf_block(
        graph,
        input_id,
        rng,
        weights,
        primary_pool=NMF_OPS,
        template_ctx="nmf_mixer_block",
    )


def tpl_nmf_routing_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Routing-capable variant: primary is ALWAYS a routing/gating NM-F op
    (CDMA slot binding / integral-control gate), satisfying the
    routing-mandatory slot-0 gate while carrying an NM-F mechanism."""
    return _build_nmf_block(
        graph,
        input_id,
        rng,
        weights,
        primary_pool=_NMF_ROUTING_OPS,
        template_ctx="nmf_routing_block",
    )
