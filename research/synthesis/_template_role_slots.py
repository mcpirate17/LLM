"""Capability-role slot taxonomy for additive template expansion.

These role slots sit above the existing motif-class taxonomy. They let new
templates reason about *what a lane is for* rather than just *which low-level
motif class it samples from*.

Design goals:
- Additive only: existing templates keep using the legacy slot API.
- Backward compatible: role-slot telemetry still writes into the existing
  ``template_slot_usage`` metadata channel.
- Explicitly non-attention / non-SSM by default: the initial role menus are
  biased toward sparse, algebraic, convolutional, and control motifs.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from ._template_helpers import (
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_CHANNEL,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_EFFICIENT_PROJ,
    MOTIF_CLASS_FFN,
    MOTIF_CLASS_GATE,
    MOTIF_CLASS_GUARDED_ACT,
    MOTIF_CLASS_MATH_SPACE,
    MOTIF_CLASS_NORM,
    MOTIF_CLASS_REDUCE,
    MOTIF_CLASS_SPARSE,
    Motif,
    MotifWeights,
    _compatible_from_classes,
    _filter_slot_candidates,
    _record_slot_usage,
    _select_from_candidates,
    record_template_slot_binding,
)

if TYPE_CHECKING:
    from .graph import ComputationGraph
else:
    ComputationGraph = object

ROLE_SLOT_CLASS_GROUPS: Dict[str, Tuple[str, ...]] = {
    "trunk_compression": (
        MOTIF_CLASS_CONV,
        MOTIF_CLASS_FFN,
        MOTIF_CLASS_SPARSE,
        MOTIF_CLASS_EFFICIENT_PROJ,
        MOTIF_CLASS_GATE,
        MOTIF_CLASS_REDUCE,
    ),
    "local_mixing": (
        MOTIF_CLASS_CONV,
        MOTIF_CLASS_CHANNEL,
        MOTIF_CLASS_GATE,
        MOTIF_CLASS_SPARSE,
        MOTIF_CLASS_EFFICIENT_PROJ,
    ),
    "global_retrieval": (
        MOTIF_CLASS_MATH_SPACE,
        MOTIF_CLASS_SPARSE,
        MOTIF_CLASS_EFFICIENT_PROJ,
        MOTIF_CLASS_GATE,
        MOTIF_CLASS_CHANNEL,
        MOTIF_CLASS_ATTENTION,
    ),
    "binding_write": (
        MOTIF_CLASS_SPARSE,
        MOTIF_CLASS_EFFICIENT_PROJ,
        MOTIF_CLASS_GATE,
        MOTIF_CLASS_REDUCE,
        MOTIF_CLASS_CHANNEL,
        MOTIF_CLASS_ATTENTION,
    ),
    "binding_read": (
        MOTIF_CLASS_MATH_SPACE,
        MOTIF_CLASS_EFFICIENT_PROJ,
        MOTIF_CLASS_GATE,
        MOTIF_CLASS_CHANNEL,
        MOTIF_CLASS_SPARSE,
        MOTIF_CLASS_ATTENTION,
    ),
    "neural_symbolic": (
        MOTIF_CLASS_ATTENTION,
        MOTIF_CLASS_MATH_SPACE,
        MOTIF_CLASS_SPARSE,
    ),
    "controller": (
        MOTIF_CLASS_GATE,
        MOTIF_CLASS_REDUCE,
        MOTIF_CLASS_EFFICIENT_PROJ,
        MOTIF_CLASS_SPARSE,
    ),
    "merge_policy": (
        MOTIF_CLASS_GATE,
        MOTIF_CLASS_EFFICIENT_PROJ,
        MOTIF_CLASS_SPARSE,
        MOTIF_CLASS_REDUCE,
    ),
    "stabilizer": (
        MOTIF_CLASS_NORM,
        MOTIF_CLASS_GATE,
        MOTIF_CLASS_GUARDED_ACT,
    ),
}


def get_role_slot_classes(role_name: str) -> Tuple[str, ...]:
    """Return the motif classes associated with a semantic role slot."""
    try:
        return ROLE_SLOT_CLASS_GROUPS[role_name]
    except KeyError as exc:
        raise ValueError(f"Unknown role slot '{role_name}'") from exc


# Role slots where retrieval-pair biasing applies: boost motifs whose name
# contains one of the retrieval-family ops (matmul, outer_product,
# cosine_similarity, gather_topk). This does not force the pair — the
# templates themselves already wire the paired consumer — but it tilts the
# motif distribution toward building blocks that include real bilinear /
# similarity / outer-product structure instead of bare linear projections.
_RETRIEVAL_BIASED_ROLES: frozenset = frozenset(
    {"global_retrieval", "binding_read", "binding_write", "neural_symbolic"}
)
_RETRIEVAL_MOTIF_BOOST: float = 2.0

# Binding-capable role slots: in addition to the soft retrieval bias, these
# roles enforce a *legality* check — the chosen motif's emitted op chain must
# contain at least one content-addressed op. Without this, the role's contract
# (e.g. "binding_write" = produces an addressable key) is purely nominal:
# nothing prevents the slot from being filled with a plain linear projection.
# Audit fix 2026-04-17 (slots.csv role:binding_* flagged "under-observed,
# broad class bucket without per-role legality checks").
_BINDING_LEGAL_ROLES: frozenset = frozenset(
    {"global_retrieval", "binding_read", "binding_write", "neural_symbolic"}
)
# Ops that constitute real content-addressed access — superset of
# CONTENT_ADDRESSED_OPS in execution_screening_graphs.py, also accepting
# bilinear / similarity ops that the retrieval-bias keyword list already
# rewards. Listing inline avoids cross-package import cycles at module load.
_BINDING_LEGAL_OPS: frozenset = frozenset(
    {
        "softmax_attention",
        "linear_attention",
        "diff_attention",
        "graph_attention",
        "local_window_attn",
        "gated_linear_attention",
        "latent_attention_compressor",
        "role_slot_attention",
        "matmul",
        "outer_product",
        "cosine_similarity",
        "gather_topk",
        "associative_memory",
    }
)


def _motif_is_binding_capable(motif: "Motif") -> bool:
    """Return True if a motif emits at least one content-addressed op.

    Used to gate binding-related role slots so the contract isn't purely
    nominal. A motif whose chain is all linear projections cannot fulfill
    a "binding_write" role no matter what class bucket it lives in.
    """
    steps = getattr(motif, "steps", None) or ()
    for step in steps:
        op_name = getattr(step, "op_name", None)
        if op_name is not None and op_name in _BINDING_LEGAL_OPS:
            return True
    return False


def _retrieval_biased_weights(
    candidates: list,
    role_name: str,
    weights: MotifWeights,
) -> MotifWeights:
    """Return a weights dict that boosts motifs containing retrieval ops.

    When ``role_name`` is in ``_RETRIEVAL_BIASED_ROLES``, any motif whose
    name hints at a retrieval-family op (matmul, outer, cosine, gather_topk)
    has its weight multiplied by ``_RETRIEVAL_MOTIF_BOOST``. This is applied
    on top of whatever base weights the caller supplied.
    """
    if role_name not in _RETRIEVAL_BIASED_ROLES:
        return weights
    # Avoid importing from primitives at module load time — inline the set.
    retrieval_tokens = ("matmul", "outer", "cosine", "gather_topk", "bilinear")
    biased: Dict[str, float] = dict(weights) if weights else {}
    for motif in candidates:
        name = getattr(motif, "name", "")
        if any(tok in name for tok in retrieval_tokens):
            prior = biased.get(name, getattr(motif, "lift", 1.0))
            biased[name] = prior * _RETRIEVAL_MOTIF_BOOST
    return biased


def pick_role_motif(
    graph: ComputationGraph,
    node_id: int,
    rng: random.Random,
    role_name: str,
    weights: MotifWeights = None,
    *,
    wildcard_prob: Optional[float] = None,
) -> Optional[Motif]:
    """Pick a compatible motif while recording semantic role-slot telemetry."""
    classes = get_role_slot_classes(role_name)
    # Semantic role slots are contracts, not exploration hints. Do not widen
    # them to the global motif pool when the prescribed classes yield zero hits.
    is_wildcard = False
    candidates = _compatible_from_classes(graph, node_id, classes)
    candidates = _filter_slot_candidates(graph, candidates)
    # Per-role legality check: binding-capable roles must produce a motif
    # that actually emits a content-addressed op. Falling back to the broad
    # class menu when this filter empties the pool would defeat the purpose,
    # so the slot returns None instead and the caller decides how to recover.
    if role_name in _BINDING_LEGAL_ROLES and candidates:
        binding_capable = [m for m in candidates if _motif_is_binding_capable(m)]
        if binding_capable:
            candidates = binding_capable
        else:
            # No binding-capable candidate — record the empty pick so telemetry
            # surfaces the gap instead of silently filling with a non-binding op.
            _record_slot_usage(
                graph,
                node_id=node_id,
                slot_classes=[f"role:{role_name}"],
                candidates=[],
                selected=None,
                wildcard=False,
            )
            return None
    biased_weights = _retrieval_biased_weights(candidates, role_name, weights)
    selected = _select_from_candidates(graph, candidates, rng, biased_weights)
    _record_slot_usage(
        graph,
        node_id=node_id,
        slot_classes=[f"role:{role_name}"],
        candidates=candidates,
        selected=selected,
        wildcard=is_wildcard,
    )
    return selected


def record_role_slot_binding(
    graph: ComputationGraph,
    *,
    role_name: str,
    selected_name: str,
    input_node_id: int,
    selected_class: str | None = None,
) -> None:
    """Record a structural role-slot choice using existing slot telemetry."""
    template_name = str(graph.metadata.get("_active_template") or "unknown")
    template_instance = int(graph.metadata.get("_active_template_instance", 0) or 0)
    slot_index = int(graph.metadata.get("_active_template_slot_counter", 0) or 0)
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=slot_index,
        slot_key=f"{template_name}[{template_instance}].slot{slot_index}",
        slot_classes=[f"role:{role_name}"],
        selected_name=selected_name,
        selected_class=selected_class or f"role_{role_name}",
        input_node_id=input_node_id,
    )
    graph.metadata["_active_template_slot_counter"] = slot_index + 1
