"""Attention template tail — private split. Re-exported from _templates_attention_tail."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph
from ._template_helpers import (
    MOTIF_CLASS_NORM,
    MOTIF_CLASS_SSM,
    MotifWeights,
    _FFN_CLASSES,
    _fix_dim,
    _instantiate_motif,
    _pick_compatible_motif,
    _pick_compatible_motif_from_classes,
    template_add_op as _add,
    template_add_residual as _residual,
)
from ._templates_attention_tail import (
    _pick_with_local_wildcard,
    _tpl_controlled_attn_matmul_ablation,
    _tpl_softmax_matmul_tail,
)


def tpl_attn_softmax_normalized_matmul(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Softmax attention with bilinear refinement and a dense recovery head."""
    return _tpl_softmax_matmul_tail(
        graph, input_id, name="attn_softmax_normalized_matmul", ffn_ratio=3.0
    )


def tpl_attn_softmax_normalized_matmul_v2(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {softmax_attention || SSM} → merge → residual → norm → FFN → residual.

    Parallel softmax attention + SSM hybrid. Replaces the original
    sequential matmul-bridge-FFN chain with the proven parallel mixing
    pattern that drives top S1 rates.
    """
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    # Path A: softmax attention
    pa = _add(
        graph,
        "softmax_attention",
        [normed],
        context="attn_softmax_normalized_matmul_v2.softmax_attn",
    )
    pa = _add(
        graph,
        "rmsnorm",
        [pa],
        context="attn_softmax_normalized_matmul_v2.attn_norm",
    )
    pa = _add(
        graph,
        "linear_proj",
        [pa],
        {"out_dim": D},
        context="attn_softmax_normalized_matmul_v2.attn_proj",
    )

    # Path B: SSM
    ssm = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_SSM, weights)
    pb = _instantiate_motif(graph, normed, ssm, rng) if ssm else normed
    pb = _fix_dim(graph, pb)

    # Merge parallel paths
    merged = _residual(graph, pa, pb, context="attn_softmax_normalized_matmul_v2.merge")
    merged = _fix_dim(graph, merged)
    mid = _residual(
        graph, input_id, merged, context="attn_softmax_normalized_matmul_v2.mid"
    )

    # FFN sub-block
    norm2 = _pick_with_local_wildcard(
        graph, mid, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffned = _add(
        graph,
        "swiglu_mlp",
        [normed2],
        {"mlp_ratio": 3.0},
        context="attn_softmax_normalized_matmul_v2.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(
        graph, mid, ffned, context="attn_softmax_normalized_matmul_v2.output"
    )


def tpl_attn_softmax_normalized_matmul_compact_ffn(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Winner-derived variant that changes only the final FFN width."""
    return _tpl_softmax_matmul_tail(
        graph,
        input_id,
        name="attn_softmax_normalized_matmul_compact_ffn",
        ffn_ratio=2.0,
    )


def tpl_attn_softmax_normalized_matmul_fixed_tail_norm(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Winner-derived variant that fixes the tail norm placement."""
    return _tpl_softmax_matmul_tail(
        graph,
        input_id,
        name="attn_softmax_normalized_matmul_fixed_tail_norm",
        ffn_ratio=3.0,
    )


def tpl_attn_linear_normalized_matmul_control(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Linear-attention control anchored directly to the successful FFN scaffold."""
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id
    attended1 = _add(
        graph,
        "linear_attention",
        [normed1],
        context="attn_linear_normalized_matmul_control.attn1",
    )
    attended1 = _add(
        graph,
        "linear_proj",
        [attended1],
        {"out_dim": D},
        context="attn_linear_normalized_matmul_control.attn1_proj",
    )
    attended1 = _fix_dim(graph, attended1)
    mid1 = _residual(
        graph, input_id, attended1, context="attn_linear_normalized_matmul_control.mid1"
    )

    norm2 = _pick_with_local_wildcard(
        graph, mid1, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid1, norm2, rng) if norm2 else mid1
    refine_in = _add(
        graph,
        "rmsnorm",
        [normed2],
        context="attn_linear_normalized_matmul_control.refine_norm",
    )
    proj_a = _add(
        graph,
        "linear_proj",
        [refine_in],
        {"out_dim": D},
        context="attn_linear_normalized_matmul_control.proj_a",
    )
    proj_b = _add(
        graph,
        "linear_proj",
        [refine_in],
        {"out_dim": D},
        context="attn_linear_normalized_matmul_control.proj_b",
    )
    refined = _add(
        graph,
        "matmul",
        [proj_a, proj_b],
        context="attn_linear_normalized_matmul_control.refined",
    )
    refined = _add(
        graph,
        "linear_proj",
        [refined],
        {"out_dim": D},
        context="attn_linear_normalized_matmul_control.refined_proj",
    )
    refined = _fix_dim(graph, refined)
    mid2 = _residual(
        graph, mid1, refined, context="attn_linear_normalized_matmul_control.mid2"
    )

    norm3 = _pick_with_local_wildcard(
        graph, mid2, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed3 = _instantiate_motif(graph, mid2, norm3, rng) if norm3 else mid2
    ffned = _add(
        graph,
        "swiglu_mlp",
        [normed3],
        {"mlp_ratio": 3.0},
        context="attn_linear_normalized_matmul_control.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(
        graph, mid2, ffned, context="attn_linear_normalized_matmul_control.output"
    )


def tpl_attn_linear_softmax_recovery_control(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {linear_attention || SSM} → merge → residual → norm → FFN → residual.

    Parallel hybrid: linear attention captures token interactions while SSM
    provides complementary long-range decay. Follows the winning
    latent_attn_ssm_hybrid pattern.
    """
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    # Path A: linear attention
    pa = _add(
        graph,
        "linear_attention",
        [normed],
        context="attn_linear_softmax_recovery_control.linear_attn",
    )
    pa = _add(
        graph,
        "linear_proj",
        [pa],
        {"out_dim": D},
        context="attn_linear_softmax_recovery_control.linear_attn_proj",
    )

    # Path B: SSM
    ssm = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_SSM, weights)
    pb = _instantiate_motif(graph, normed, ssm, rng) if ssm else normed
    pb = _fix_dim(graph, pb)

    # Merge parallel paths
    merged = _residual(
        graph, pa, pb, context="attn_linear_softmax_recovery_control.merge"
    )
    merged = _fix_dim(graph, merged)
    mid = _residual(
        graph, input_id, merged, context="attn_linear_softmax_recovery_control.mid"
    )

    # FFN sub-block
    norm2 = _pick_with_local_wildcard(
        graph, mid, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffned = _add(
        graph,
        "swiglu_mlp",
        [normed2],
        {"mlp_ratio": 3.0},
        context="attn_linear_softmax_recovery_control.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(
        graph, mid, ffned, context="attn_linear_softmax_recovery_control.output"
    )


def tpl_attn_linear_no_matmul_ffn(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Linear-attention stack without matmul, using a softmax recovery pass."""
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id
    attended1 = _add(
        graph,
        "linear_attention",
        [normed1],
        context="attn_linear_no_matmul_ffn.attn1",
    )
    attended1 = _add(
        graph,
        "linear_proj",
        [attended1],
        {"out_dim": D},
        context="attn_linear_no_matmul_ffn.attn1_proj",
    )
    attended1 = _fix_dim(graph, attended1)
    mid1 = _residual(
        graph, input_id, attended1, context="attn_linear_no_matmul_ffn.mid1"
    )

    norm2 = _pick_with_local_wildcard(
        graph, mid1, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid1, norm2, rng) if norm2 else mid1
    refine_in = _add(
        graph,
        "rmsnorm",
        [normed2],
        context="attn_linear_no_matmul_ffn.refine_norm",
    )
    attended2 = _add(
        graph,
        "softmax_attention",
        [refine_in],
        context="attn_linear_no_matmul_ffn.attn2",
    )
    attended2 = _add(
        graph,
        "rmsnorm",
        [attended2],
        context="attn_linear_no_matmul_ffn.attn2_norm",
    )
    attended2 = _add(
        graph,
        "linear_proj",
        [attended2],
        {"out_dim": D},
        context="attn_linear_no_matmul_ffn.attn2_proj",
    )
    attended2 = _fix_dim(graph, attended2)
    mid2 = _residual(graph, mid1, attended2, context="attn_linear_no_matmul_ffn.mid2")

    norm3 = _pick_with_local_wildcard(
        graph, mid2, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed3 = _instantiate_motif(graph, mid2, norm3, rng) if norm3 else mid2
    ffned = _add(
        graph,
        "swiglu_mlp",
        [normed3],
        {"mlp_ratio": 3.0},
        context="attn_linear_no_matmul_ffn.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(graph, mid2, ffned, context="attn_linear_no_matmul_ffn.output")


def tpl_attn_linear_no_matmul_ffn_dense_tail(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Winner-derived variant that swaps only the tail MLP family."""
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id
    attended1 = _add(
        graph,
        "linear_attention",
        [normed1],
        context="attn_linear_no_matmul_ffn_dense_tail.attn1",
    )
    attended1 = _add(
        graph,
        "linear_proj",
        [attended1],
        {"out_dim": D},
        context="attn_linear_no_matmul_ffn_dense_tail.attn1_proj",
    )
    attended1 = _fix_dim(graph, attended1)
    mid1 = _residual(
        graph, input_id, attended1, context="attn_linear_no_matmul_ffn_dense_tail.mid1"
    )

    norm2 = _pick_with_local_wildcard(
        graph, mid1, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid1, norm2, rng) if norm2 else mid1
    refine_in = _add(
        graph,
        "rmsnorm",
        [normed2],
        context="attn_linear_no_matmul_ffn_dense_tail.refine_norm",
    )
    attended2 = _add(
        graph,
        "softmax_attention",
        [refine_in],
        context="attn_linear_no_matmul_ffn_dense_tail.attn2",
    )
    attended2 = _add(
        graph,
        "rmsnorm",
        [attended2],
        context="attn_linear_no_matmul_ffn_dense_tail.attn2_norm",
    )
    attended2 = _add(
        graph,
        "linear_proj",
        [attended2],
        {"out_dim": D},
        context="attn_linear_no_matmul_ffn_dense_tail.attn2_proj",
    )
    attended2 = _fix_dim(graph, attended2)
    mid2 = _residual(
        graph, mid1, attended2, context="attn_linear_no_matmul_ffn_dense_tail.mid2"
    )

    norm3 = _pick_with_local_wildcard(
        graph, mid2, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed3 = _instantiate_motif(graph, mid2, norm3, rng) if norm3 else mid2
    ffned = _add(
        graph,
        "fused_linear_gelu",
        [normed3],
        {"out_dim": D},
        context="attn_linear_no_matmul_ffn_dense_tail.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(
        graph,
        mid2,
        ffned,
        context="attn_linear_no_matmul_ffn_dense_tail.output",
    )


def tpl_attn_linear_no_matmul_ffn_direct_recovery(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Winner-derived variant that removes only the explicit recovery norm."""
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id
    attended1 = _add(
        graph,
        "linear_attention",
        [normed1],
        context="attn_linear_no_matmul_ffn_direct_recovery.attn1",
    )
    attended1 = _add(
        graph,
        "linear_proj",
        [attended1],
        {"out_dim": D},
        context="attn_linear_no_matmul_ffn_direct_recovery.attn1_proj",
    )
    attended1 = _fix_dim(graph, attended1)
    mid1 = _residual(
        graph,
        input_id,
        attended1,
        context="attn_linear_no_matmul_ffn_direct_recovery.mid1",
    )

    norm2 = _pick_with_local_wildcard(
        graph, mid1, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid1, norm2, rng) if norm2 else mid1
    attended2 = _add(
        graph,
        "softmax_attention",
        [normed2],
        context="attn_linear_no_matmul_ffn_direct_recovery.attn2",
    )
    attended2 = _add(
        graph,
        "rmsnorm",
        [attended2],
        context="attn_linear_no_matmul_ffn_direct_recovery.attn2_norm",
    )
    attended2 = _add(
        graph,
        "linear_proj",
        [attended2],
        {"out_dim": D},
        context="attn_linear_no_matmul_ffn_direct_recovery.attn2_proj",
    )
    attended2 = _fix_dim(graph, attended2)
    mid2 = _residual(
        graph,
        mid1,
        attended2,
        context="attn_linear_no_matmul_ffn_direct_recovery.mid2",
    )

    norm3 = _pick_with_local_wildcard(
        graph, mid2, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed3 = _instantiate_motif(graph, mid2, norm3, rng) if norm3 else mid2
    ffned = _add(
        graph,
        "swiglu_mlp",
        [normed3],
        {"mlp_ratio": 3.0},
        context="attn_linear_no_matmul_ffn_direct_recovery.ffn",
    )
    ffned = _fix_dim(graph, ffned)
    return _residual(
        graph,
        mid2,
        ffned,
        context="attn_linear_no_matmul_ffn_direct_recovery.output",
    )


def tpl_attn_softmax_matmul_sparse_tail(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → {softmax_attention || SSM} → merge → residual → norm → FFN → residual.

    Parallel softmax attention + SSM hybrid. Replaces the original
    matmul+sparse_tail dead-gradient path with the proven parallel mixing
    pattern followed by a full-width FFN.
    """
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    # Path A: softmax attention + proj
    pa = _add(
        graph,
        "softmax_attention",
        [normed],
        context="attn_softmax_matmul_sparse_tail.softmax_attn",
    )
    pa = _add(
        graph,
        "rmsnorm",
        [pa],
        context="attn_softmax_matmul_sparse_tail.attn_norm",
    )
    pa = _add(
        graph,
        "linear_proj",
        [pa],
        {"out_dim": D},
        context="attn_softmax_matmul_sparse_tail.attn_proj",
    )

    # Path B: SSM
    ssm = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_SSM, weights)
    pb = _instantiate_motif(graph, normed, ssm, rng) if ssm else normed
    pb = _fix_dim(graph, pb)

    # Merge parallel paths
    merged = _residual(graph, pa, pb, context="attn_softmax_matmul_sparse_tail.merge")
    merged = _fix_dim(graph, merged)
    mid = _residual(
        graph, input_id, merged, context="attn_softmax_matmul_sparse_tail.mid"
    )

    # FFN sub-block
    norm2 = _pick_with_local_wildcard(
        graph, mid, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid
    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, _FFN_CLASSES, weights
    )
    if ffn:
        ffned = _instantiate_motif(graph, normed2, ffn, rng)
    else:
        ffned = _add(
            graph,
            "swiglu_mlp",
            [normed2],
            {"mlp_ratio": 3.0},
            context="attn_softmax_matmul_sparse_tail.ffn",
        )
    ffned = _fix_dim(graph, ffned)
    return _residual(
        graph, mid, ffned, context="attn_softmax_matmul_sparse_tail.output"
    )


def tpl_attn_linear_matmul_sparse_tail(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Linear attention with matmul refinement, then a sparse output head."""
    D = graph.model_dim
    norm1 = _pick_with_local_wildcard(
        graph, input_id, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id
    attended1 = _add(
        graph,
        "linear_attention",
        [normed1],
        context="attn_linear_matmul_sparse_tail.attn1",
    )
    attended1 = _add(
        graph,
        "linear_proj",
        [attended1],
        {"out_dim": D},
        context="attn_linear_matmul_sparse_tail.attn1_proj",
    )
    attended1 = _fix_dim(graph, attended1)
    mid1 = _residual(
        graph, input_id, attended1, context="attn_linear_matmul_sparse_tail.mid1"
    )

    norm2 = _pick_with_local_wildcard(
        graph, mid1, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed2 = _instantiate_motif(graph, mid1, norm2, rng) if norm2 else mid1
    refine_in = _add(
        graph,
        "rmsnorm",
        [normed2],
        context="attn_linear_matmul_sparse_tail.refine_norm",
    )
    proj_a = _add(
        graph,
        "linear_proj",
        [refine_in],
        {"out_dim": D},
        context="attn_linear_matmul_sparse_tail.proj_a",
    )
    proj_b = _add(
        graph,
        "linear_proj",
        [refine_in],
        {"out_dim": D},
        context="attn_linear_matmul_sparse_tail.proj_b",
    )
    refined = _add(
        graph,
        "matmul",
        [proj_a, proj_b],
        context="attn_linear_matmul_sparse_tail.refined",
    )
    refined = _add(
        graph,
        "linear_proj",
        [refined],
        {"out_dim": D},
        context="attn_linear_matmul_sparse_tail.refined_proj",
    )
    refined = _fix_dim(graph, refined)
    mid2 = _residual(
        graph, mid1, refined, context="attn_linear_matmul_sparse_tail.mid2"
    )

    norm3 = _pick_with_local_wildcard(
        graph, mid2, rng, MOTIF_CLASS_NORM, weights, wildcard_prob=0.0
    )
    normed3 = _instantiate_motif(graph, mid2, norm3, rng) if norm3 else mid2
    sparse = _add(
        graph,
        "block_sparse_linear",
        [normed3],
        {"block_size": 16, "block_density": 0.25},
        context="attn_linear_matmul_sparse_tail.sparse_tail",
    )
    sparse = _fix_dim(graph, sparse)
    return _residual(
        graph, mid2, sparse, context="attn_linear_matmul_sparse_tail.output"
    )


def tpl_attn_linear_matmul_router_sidecar(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Linear-attention control scaffold with routing as a side branch, not main path."""
    return _tpl_controlled_attn_matmul_ablation(
        graph,
        input_id,
        rng,
        weights,
        name="attn_linear_matmul_router_sidecar",
        attn_op="linear_attention",
        use_matmul_refine=True,
        tail_kind="router_sidecar",
    )
