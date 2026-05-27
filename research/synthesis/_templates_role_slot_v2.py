"""Role-slot v2 variants of strong legacy templates.

The 2026-04-16 capability-first analysis showed the frontier splits into
two disjoint families: conv/SSM/sparse graphs that win perplexity, and
matmul/gather_topk/outer_product graphs that win binding. No legacy
template forces both capabilities into a single graph.

These v2 templates keep the proven trunks from the strongest legacy
templates (``conv_residual_block``, ``state_space_block``,
``latent_attn_ffn_block``) and bolt on an explicit retrieval sidecar
that uses the role-slot taxonomy from ``_template_role_slots.py``
(binding_write → global_retrieval → binding_read merged via a typed-
entropy gate). The sidecar can be configured off by the grammar at
sampling time for ablation, but the default wiring always produces a
content-addressed retrieval path.

Registered in ``templates.py`` as ``*_retrieval_v2``. Legacy templates
are untouched — existing leaderboard comparability stays intact.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph

from ._template_helpers import (
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_NORM,
    MotifWeights,
    _instantiate_motif,
    _pick_compatible_motif,
    template_add_op as _add,
    template_add_residual as _residual,
)
from ._template_role_slots import record_role_slot_binding


def _retrieval_sidecar(
    graph: ComputationGraph,
    trunk_out: int,
    *,
    context: str,
) -> int:
    """Emit a shape-preserving bilinear retrieval sidecar.

    Uses the proven ``induction_matmul_block`` pattern (``matmul(q, q)``
    for bilinear self-relation on projected activations). The retrieval
    output stays at (B, S, D) so it composes trivially with the trunk
    via ``add``.

    Flow:

        trunk_out ── linear_proj ─┬─ matmul(q, q) ── linear_proj ── retrieved
                                  └─ linear_proj ───────── (bind_write anchor)

    Why no explicit gate here: the controller-gated variants
    (``typed_slot_memory_block``, ``token_program_interpreter_block``)
    already cover that topology. v2 deliberately stays minimal —
    preserve the trunk's ppl strength, add the smallest retrieval
    structure that can carry binding. A future iteration can thread a
    shape-safe gate through ``mul`` once ``linear_proj_up`` from (B,S,1)
    signals is repaired at the op level.
    """
    d = graph.model_dim

    # --- Binding write: project trunk into retrieval key/query space ------
    query = _add(
        graph,
        "linear_proj",
        [trunk_out],
        {"out_dim": d},
        context=f"{context}.query",
    )
    record_role_slot_binding(
        graph,
        role_name="binding_write",
        selected_name="linear_query_write_v2",
        input_node_id=trunk_out,
    )

    # --- Global retrieval: bilinear self-relation via matmul(q, q) --------
    # Matches induction_matmul_block — the runtime treats matmul(x, x)
    # as a shape-preserving bilinear when x is (B, S, D).
    scores = _add(graph, "matmul", [query, query], context=f"{context}.retrieve.scores")
    record_role_slot_binding(
        graph,
        role_name="global_retrieval",
        selected_name="bilinear_self_matmul_v2",
        input_node_id=query,
    )

    # --- Binding read: project retrieval scores back into trunk space -----
    retrieved = _add(
        graph,
        "linear_proj",
        [scores],
        {"out_dim": d},
        context=f"{context}.retrieve.project",
    )
    record_role_slot_binding(
        graph,
        role_name="binding_read",
        selected_name="linear_read_v2",
        input_node_id=scores,
    )

    record_role_slot_binding(
        graph,
        role_name="merge_policy",
        selected_name="bilinear_additive_merge_v2",
        input_node_id=trunk_out,
    )
    return retrieved


def _safe_local_mixing_trunk(
    graph: ComputationGraph,
    node_id: int,
    *,
    context: str,
) -> int:
    """Use a fixed LM-safe local mixer instead of the broad legacy FFN slot.

    The generic `_FFN_CLASSES` pool admits routing/gating motifs that are legal
    in other templates but invalid when immediately followed by the retrieval
    sidecar's `linear_proj` query path. A pinned SwiGLU trunk preserves the
    intended capability-first design without reintroducing alias-era wiring
    failures.
    """
    return _add(graph, "swiglu_mlp", [node_id], {"mlp_ratio": 4.0}, context=context)


def tpl_conv_residual_retrieval_v2(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """conv_residual_block trunk + role-slot retrieval sidecar.

    Trunk: norm → conv_only → FFN motif → residual_add (original
    conv_residual_block behavior).
    Sidecar: binding_write/read through matmul+gather_topk, gated by a
    typed-entropy controller, merged back via add.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    D = graph.model_dim
    try:
        conved = _add(
            graph,
            "conv_only",
            [normed],
            context="conv_residual_retrieval_v2.trunk",
        )
    except (ValueError, KeyError):
        # If conv_only is unavailable, fall back to conv1d_seq.
        conved = _add(
            graph, "conv1d_seq", [normed], context="conv_residual_retrieval_v2.trunk"
        )
    conved = _add(
        graph,
        "linear_proj",
        [conved],
        {"out_dim": D},
        context="conv_residual_retrieval_v2.trunk_project",
    )
    record_role_slot_binding(
        graph,
        role_name="trunk_compression",
        selected_name="conv_only_trunk_v2",
        input_node_id=normed,
    )

    processed = _safe_local_mixing_trunk(
        graph, conved, context="conv_residual_retrieval_v2.local_mixing"
    )
    record_role_slot_binding(
        graph,
        role_name="local_mixing",
        selected_name="swiglu_local_v2",
        input_node_id=conved,
    )

    # Retrieval sidecar branches off the post-FFN trunk output.
    retrieved = _retrieval_sidecar(
        graph, processed, context="conv_residual_retrieval_v2"
    )

    merged = _add(
        graph,
        "add",
        [processed, retrieved],
        context="conv_residual_retrieval_v2.merge",
    )
    record_role_slot_binding(
        graph,
        role_name="stabilizer",
        selected_name="identity_stabilizer_v2",
        input_node_id=merged,
    )
    return _residual(
        graph, input_id, merged, context="conv_residual_retrieval_v2.output"
    )


def tpl_state_space_retrieval_v2(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """state_space_block trunk + role-slot retrieval sidecar.

    Trunk: norm → state_space → proj → FFN motif → residual.
    Sidecar: same pattern as conv_residual_retrieval_v2.
    """
    d = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        ssm = graph.add_op("state_space", [normed])
    except (ValueError, KeyError):
        ssm = _add(
            graph,
            "selective_scan",
            [normed],
            context="state_space_retrieval_v2.trunk",
        )
    record_role_slot_binding(
        graph,
        role_name="trunk_compression",
        selected_name="ssm_trunk_v2",
        input_node_id=normed,
    )
    projected = _add(
        graph,
        "linear_proj",
        [ssm],
        {"out_dim": d},
        context="state_space_retrieval_v2.proj",
    )

    processed = _safe_local_mixing_trunk(
        graph, projected, context="state_space_retrieval_v2.local_mixing"
    )
    record_role_slot_binding(
        graph,
        role_name="local_mixing",
        selected_name="swiglu_local_v2",
        input_node_id=projected,
    )

    retrieved = _retrieval_sidecar(graph, processed, context="state_space_retrieval_v2")

    merged = _add(
        graph,
        "add",
        [processed, retrieved],
        context="state_space_retrieval_v2.merge",
    )
    record_role_slot_binding(
        graph,
        role_name="stabilizer",
        selected_name="identity_stabilizer_v2",
        input_node_id=merged,
    )
    return _residual(graph, input_id, merged, context="state_space_retrieval_v2.output")


def tpl_latent_attn_retrieval_v2(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """latent_attn_ffn_block trunk + role-slot retrieval sidecar.

    Tests whether a template that already has content-addressed ops
    (latent attention) benefits from an explicit secondary retrieval
    path. The v2 sidecar uses matmul+gather_topk rather than softmax
    attention, so it is complementary to the attention lane.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    attn = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_ATTENTION, weights)
    if attn is None:
        # Guaranteed fallback path — latent_attention_compressor is the strongest.
        attended = _add(
            graph,
            "latent_attention_compressor",
            [normed],
            context="latent_attn_retrieval_v2.trunk",
        )
    else:
        attended = _instantiate_motif(graph, normed, attn, rng)
    record_role_slot_binding(
        graph,
        role_name="trunk_compression",
        selected_name="latent_attn_trunk_v2",
        input_node_id=normed,
    )

    processed = _safe_local_mixing_trunk(
        graph, attended, context="latent_attn_retrieval_v2.local_mixing"
    )
    record_role_slot_binding(
        graph,
        role_name="local_mixing",
        selected_name="swiglu_local_v2",
        input_node_id=attended,
    )

    retrieved = _retrieval_sidecar(graph, processed, context="latent_attn_retrieval_v2")

    merged = _add(
        graph,
        "add",
        [processed, retrieved],
        context="latent_attn_retrieval_v2.merge",
    )
    record_role_slot_binding(
        graph,
        role_name="stabilizer",
        selected_name="identity_stabilizer_v2",
        input_node_id=merged,
    )
    return _residual(graph, input_id, merged, context="latent_attn_retrieval_v2.output")


def tpl_neural_symbolic_retrieval_v2(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → state_space_block ──┬── residual_add
                                 └── neural_symbolic sidecar

    A v2 hybrid lane that combines the perplexity strength of SSM with
    a dedicated 'Neural Symbolic' slot for persistent register-based retrieval.
    """
    from ._templates_research import tpl_state_space_block
    from ._template_role_slots import pick_role_motif

    # 1. Primary Trunk (SSM)
    # We use state_space_block as the default high-ppl carrier.
    trunk_out = tpl_state_space_block(graph, input_id, rng, weights)

    # 2. Neural Symbolic Sidecar
    # This semantic slot samples from MOTIF_CLASS_ATTENTION but requires
    # content-addressed behavior (binding capability).
    symbolic_motif = pick_role_motif(graph, trunk_out, rng, "neural_symbolic", weights)
    if symbolic_motif:
        symbolic_out = _instantiate_motif(graph, trunk_out, symbolic_motif, rng)
        # Residual merge: trunk + symbolic workspace
        return _add(
            graph, "add", [trunk_out, symbolic_out], context="ns_retrieval_v2.merge"
        )

    return trunk_out
