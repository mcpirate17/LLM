"""Compaction mixer templates — NM-C Tier-D ops as proposable blocks.

One parametric block template makes all 11 shipped NM-C compaction mixers
(monarch/butterfly/recurrent-depth/weight-dictionary/hypernet/persistent-
memory/block-sparse/token-merge/ternary/p-adic-lowprec/subspace-mixture) plus
the two validated collapse-proof p-adic RDR gates (padic-gated-mixer /
padic-depth-route) reachable by the grammar; the sampled op name is recorded in
the node context so per-op attribution survives into the ledger. Every op is
identity-at-init or highway-gated with its own internal residual/ReZero, so the
block scaffold stays minimal: norm → mixer → residual, with an optional FFN tail.
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
    weighted_op_choice,
)
from ._templates_attention_advanced import (
    _pick_norm_or_default,
    _pick_ffn_or_swiglu,
)

# Channel-factorization mixers (pointwise over tokens).
_CHANNEL_COMPACTION_OPS = (
    "monarch_mix",
    "butterfly_mix",
    "weight_dictionary_mix",
    "hypernet_layer_mix",
    "block_sparse_mix",
    "ternary_sign_mix",
    "padic_lowprec_mix",
    "subspace_mixture_mix",
    # Validated collapse-proof p-adic RDR gates (2026-06-29; underpin the RDR
    # scale result). Both are pointwise-over-tokens (no S-axis mixing), so they
    # belong in the channel pool where the block ALWAYS forces a sequence
    # secondary — that pairing keeps them gate-6 safe. They were registered +
    # dispatchable but reachable by no template; wiring them here keeps two
    # validated novel mechanisms in the proposable space (GLM handoff).
    "padic_gated_mixer",
    "padic_depth_route",
)
# Sequence/memory/depth mixers (cross-token or virtual depth).
_SEQUENCE_COMPACTION_OPS = (
    "recurrent_depth_refine",
    "persistent_memory_refine",
    "token_merge_mix",
    "lowrank_state_memory",
)
COMPACTION_OPS = _CHANNEL_COMPACTION_OPS + _SEQUENCE_COMPACTION_OPS


def _build_compaction_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights,
    *,
    primary_pool: tuple[str, ...],
    template_ctx: str,
) -> int:
    """norm → <primary mixer> → [secondary mixer] → residual → [norm→FFN→res]."""
    normed = _pick_norm_or_default(
        graph, input_id, rng, weights, fallback_context=f"{template_ctx}.norm1"
    )

    primary = weighted_op_choice(graph, rng, primary_pool)
    mixed = _add(graph, primary, [normed], context=f"{template_ctx}.{primary}")

    if primary in _SEQUENCE_COMPACTION_OPS:
        if rng.random() < 0.5:
            secondary = weighted_op_choice(graph, rng, _CHANNEL_COMPACTION_OPS)
            mixed = _add(
                graph, secondary, [mixed], context=f"{template_ctx}.{secondary}"
            )
    else:
        # A channel-only block has no cross-token path and dies at the
        # structural no-mixing gate after burning rapid-screening compute
        # (measured: overnight campaign 2026-07-02). ALWAYS pair a channel
        # primary with a sequence mixer.
        secondary = weighted_op_choice(graph, rng, _SEQUENCE_COMPACTION_OPS)
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


def tpl_compaction_mixer_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → <any NM-C compaction mixer> → residual → [norm → FFN → residual].

    Samples across all 11 NM-C ops and (50%) pairs a sequence mixer with a
    channel mixer — the composition the compaction program predicts (e.g.
    token_merge shrinking L for a cheap factored channel mix).
    """
    return _build_compaction_block(
        graph,
        input_id,
        rng,
        weights,
        primary_pool=COMPACTION_OPS,
        template_ctx="compaction_mixer_block",
    )


def tpl_compaction_routing_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Routing-capable variant: the primary is ALWAYS a sequence-compaction op
    (token merge / conditional depth / routed retrieval — all in
    ROUTING_COMPRESSION_MOE_OPS), so this block reliably satisfies the
    routing-mandatory slot-0 gate and the post-build routing check while
    carrying an NM-C mechanism.
    """
    return _build_compaction_block(
        graph,
        input_id,
        rng,
        weights,
        primary_pool=_SEQUENCE_COMPACTION_OPS,
        template_ctx="compaction_routing_block",
    )
