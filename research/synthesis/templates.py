"""
Structural Templates for Motif-Based Grammar

A template is an abstract DAG pattern where nodes are motif slots.
Templates define the skeleton; motifs fill the slots.

Each template is a callable:
  (graph, input_id, rng, motif_picker, config) → output_node_id

Templates compose recursively — a parallel template can have a
residual template in one branch.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, Optional, Tuple

from .graph import ComputationGraph
from .motifs import (
    MOTIF_CLASS_ATTENTION, MOTIF_CLASS_CHANNEL, MOTIF_CLASS_CONV,
    MOTIF_CLASS_FFN, MOTIF_CLASS_GATE, MOTIF_CLASS_MOE,
    MOTIF_CLASS_NORM, MOTIF_CLASS_SPARSE, MOTIF_CLASS_SSM,
    Motif, pick_motif, pick_motif_from_classes, resolve_step,
)

# Type alias for motif weight dicts passed from judgment engine
MotifWeights = Optional[Dict[str, float]]

# ── Motif class groupings for slot constraints ──────────────────────

# Slots that accept any sequence mixer
_MIXER_CLASSES: Tuple[str, ...] = (
    MOTIF_CLASS_ATTENTION, MOTIF_CLASS_SSM, MOTIF_CLASS_CONV,
    MOTIF_CLASS_CHANNEL,
)

# Slots that accept any FFN-like transform
_FFN_CLASSES: Tuple[str, ...] = (
    MOTIF_CLASS_FFN, MOTIF_CLASS_GATE, MOTIF_CLASS_SPARSE,
)

# All motif classes
_ALL_CLASSES: Tuple[str, ...] = (
    MOTIF_CLASS_FFN, MOTIF_CLASS_ATTENTION, MOTIF_CLASS_SSM,
    MOTIF_CLASS_CONV, MOTIF_CLASS_GATE, MOTIF_CLASS_SPARSE,
    MOTIF_CLASS_MOE, MOTIF_CLASS_CHANNEL,
)


# ── Helper: instantiate a motif into a graph ────────────────────────

def _instantiate_motif(
    graph: ComputationGraph,
    node_id: int,
    motif: Motif,
    rng: random.Random,
) -> int:
    """Add a motif's ops to the graph starting from node_id.

    Returns the output node ID. On shape mismatch, returns input node_id.
    """
    current = node_id
    D = graph.model_dim
    for step in motif.steps:
        op_name, config = resolve_step(step, rng)
        # Use actual current dimension for config (not always model_dim)
        cur_dim = graph.nodes[current].output_shape.dim
        if op_name in ("linear_proj", "fused_linear_gelu", "gated_linear"):
            config.setdefault("out_dim", D)
        elif op_name == "linear_proj_down":
            config.setdefault("out_dim", cur_dim // 2)
        elif op_name == "linear_proj_up":
            config.setdefault("out_dim", cur_dim * 2)
        elif op_name in ("nm_sparse_linear", "block_sparse_linear",
                         "semi_structured_2_4_linear", "ternary_projection"):
            config.setdefault("out_dim", cur_dim)
            if op_name == "nm_sparse_linear":
                config.setdefault("n", 2)
                config.setdefault("m", 4)
            elif op_name == "block_sparse_linear":
                config.setdefault("block_size", rng.choice([8, 16, 32]))
                config.setdefault("block_density", rng.uniform(0.05, 0.5))
        elif op_name == "local_window_attn":
            config.setdefault("window_size", rng.choice([8, 16, 32]))
        elif op_name in ("swiglu_mlp", "rwkv_channel", "moe_topk",
                         "rwkv_time_mixing"):
            config.setdefault("mlp_ratio", rng.choice([2.0, 3.0, 4.0]))
        try:
            current = graph.add_op(op_name, [current], config=config)
        except (ValueError, KeyError):
            return node_id  # Bail on shape error
    return current


def _fix_dim(graph: ComputationGraph, node_id: int) -> int:
    """Add linear_proj to fix dimension back to model_dim if needed."""
    if graph.nodes[node_id].output_shape.dim != graph.model_dim:
        try:
            return graph.add_op("linear_proj", [node_id],
                                config={"out_dim": graph.model_dim})
        except ValueError:
            pass
    return node_id


# ── Template implementations ────────────────────────────────────────

def tpl_residual_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → motif → residual_add.

    The workhorse template: pre-norm + any functional motif + skip.
    """
    # Pre-norm
    norm_motif = pick_motif(rng, MOTIF_CLASS_NORM, weights)
    if norm_motif:
        normed = _instantiate_motif(graph, input_id, norm_motif, rng)
    else:
        normed = input_id

    # Core motif (mixer or FFN)
    core_classes = list(_MIXER_CLASSES + _FFN_CLASSES)
    core_motif = pick_motif_from_classes(rng, core_classes, weights)
    if core_motif:
        processed = _instantiate_motif(graph, normed, core_motif, rng)
    else:
        processed = normed

    # Fix dim and add residual
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_sequential(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """motif_a → motif_b → ... → motif_n.

    Stack 2-3 different functional motifs in sequence.
    """
    n_motifs = rng.choice([2, 3])
    current = input_id
    for _ in range(n_motifs):
        motif = pick_motif_from_classes(rng, _ALL_CLASSES, weights)
        if motif:
            current = _instantiate_motif(graph, current, motif, rng)
            current = _fix_dim(graph, current)
    return current


def tpl_transformer_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → mixer → add → norm → ffn → add.

    Classic pre-norm transformer block with any mixer + any FFN.
    """
    D = graph.model_dim

    # Attention sub-block
    norm1 = pick_motif(rng, MOTIF_CLASS_NORM, weights)
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    mixer = pick_motif_from_classes(rng, _MIXER_CLASSES, weights)
    if mixer:
        mixed = _instantiate_motif(graph, normed1, mixer, rng)
    else:
        mixed = normed1
    mixed = _fix_dim(graph, mixed)

    try:
        mid = graph.add_op("add", [input_id, mixed])
    except ValueError:
        mid = mixed

    # FFN sub-block
    norm2 = pick_motif(rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid

    ffn = pick_motif_from_classes(rng, _FFN_CLASSES, weights)
    if ffn:
        ffned = _instantiate_motif(graph, normed2, ffn, rng)
    else:
        ffned = normed2
    ffned = _fix_dim(graph, ffned)

    try:
        return graph.add_op("add", [mid, ffned])
    except ValueError:
        return ffned


def tpl_parallel_split(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """split → {motif_a | motif_b} → concat → project.

    Width: parallel processing paths with different motifs.
    """
    shape = graph.nodes[input_id].output_shape
    if shape.dim < 16:
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        split_id = graph.add_op("split2", [input_id])
    except ValueError:
        return tpl_residual_block(graph, input_id, rng, weights)

    # Path A: mixer
    motif_a = pick_motif_from_classes(rng, _MIXER_CLASSES, weights)
    if motif_a:
        path_a = _instantiate_motif(graph, split_id, motif_a, rng)
    else:
        path_a = split_id

    # Path B: FFN or gate
    motif_b = pick_motif_from_classes(rng, _FFN_CLASSES, weights)
    if motif_b:
        path_b = _instantiate_motif(graph, split_id, motif_b, rng)
    else:
        path_b = split_id

    try:
        merged = graph.add_op("concat", [path_a, path_b])
    except ValueError:
        return path_a

    return _fix_dim(graph, merged)


def tpl_bottleneck(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """project_down → motif → project_up → residual_add.

    Compression: information bottleneck with any core motif.
    """
    D = graph.model_dim
    try:
        down = graph.add_op("linear_proj_down", [input_id],
                            config={"out_dim": D // 2})
    except ValueError:
        return tpl_residual_block(graph, input_id, rng, weights)

    core = pick_motif_from_classes(rng, _ALL_CLASSES, weights)
    if core:
        processed = _instantiate_motif(graph, down, core, rng)
    else:
        processed = down

    try:
        up = graph.add_op("linear_proj_up", [processed],
                          config={"out_dim": D})
    except ValueError:
        return _fix_dim(graph, processed)

    try:
        return graph.add_op("add", [input_id, up])
    except ValueError:
        return up


def tpl_moe(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → moe_motif → residual_add.

    Sparsity: conditional computation via MoE.
    """
    norm = pick_motif(rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    moe = pick_motif(rng, MOTIF_CLASS_MOE, weights)
    if moe:
        routed = _instantiate_motif(graph, normed, moe, rng)
    else:
        # Fallback to a gate motif
        gate = pick_motif(rng, MOTIF_CLASS_GATE, weights)
        routed = _instantiate_motif(graph, normed, gate, rng) if gate else normed

    routed = _fix_dim(graph, routed)
    try:
        return graph.add_op("add", [input_id, routed])
    except ValueError:
        return routed


def tpl_hybrid_parallel(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """split → {attention_motif | ssm_motif} → concat → project.

    Hybrid: combining attention and SSM in parallel.
    """
    shape = graph.nodes[input_id].output_shape
    if shape.dim < 16:
        return tpl_transformer_block(graph, input_id, rng, weights)

    try:
        split_id = graph.add_op("split2", [input_id])
    except ValueError:
        return tpl_transformer_block(graph, input_id, rng, weights)

    # Attention path
    attn = pick_motif_from_classes(
        rng, (MOTIF_CLASS_ATTENTION,), weights)
    path_attn = _instantiate_motif(graph, split_id, attn, rng) if attn else split_id

    # SSM/conv path
    ssm = pick_motif_from_classes(
        rng, (MOTIF_CLASS_SSM, MOTIF_CLASS_CONV), weights)
    path_ssm = _instantiate_motif(graph, split_id, ssm, rng) if ssm else split_id

    try:
        merged = graph.add_op("concat", [path_attn, path_ssm])
    except ValueError:
        return path_attn

    merged = _fix_dim(graph, merged)
    try:
        return graph.add_op("add", [input_id, merged])
    except ValueError:
        return merged


def tpl_gated_residual(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → motif → gate → residual_add.

    Learned residual: adaptive skip weighting.
    """
    norm = pick_motif(rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    core = pick_motif_from_classes(rng, list(_MIXER_CLASSES + _FFN_CLASSES), weights)
    processed = _instantiate_motif(graph, normed, core, rng) if core else normed
    processed = _fix_dim(graph, processed)

    # Gate
    gate = pick_motif(rng, MOTIF_CLASS_GATE, weights)
    if gate:
        gated = _instantiate_motif(graph, processed, gate, rng)
        gated = _fix_dim(graph, gated)
    else:
        gated = processed

    try:
        return graph.add_op("add", [input_id, gated])
    except ValueError:
        return gated


def tpl_dense_cascade(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """motif_1 → motif_2 → motif_3 with dense skip connections.

    DenseNet-style: each motif receives all prior outputs.
    """
    outputs = [input_id]

    for i in range(3):
        # Pick from output of last motif
        prev = outputs[-1]
        motif = pick_motif_from_classes(rng, _ALL_CLASSES, weights)
        if motif:
            processed = _instantiate_motif(graph, prev, motif, rng)
            processed = _fix_dim(graph, processed)
        else:
            processed = prev

        # Dense skip: add to first available prior output
        if i > 0 and processed != outputs[0]:
            try:
                processed = graph.add_op("add", [outputs[0], processed])
            except ValueError:
                pass
        outputs.append(processed)

    return outputs[-1]


def tpl_sparse_ffn(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → sparse_linear → activate → project → residual_add.

    Uses sparse linear ops (N:M, block, ternary) as the main projection.
    """
    norm = pick_motif(rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    sparse = pick_motif(rng, MOTIF_CLASS_SPARSE, weights)
    if sparse:
        processed = _instantiate_motif(graph, normed, sparse, rng)
    else:
        processed = normed
    processed = _fix_dim(graph, processed)

    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_sparse_moe_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → sparse_motif → moe_motif → residual_add.

    Compound efficiency: forces both sparse AND MoE ops structurally.
    """
    norm = pick_motif(rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Sparse path
    sparse = pick_motif(rng, MOTIF_CLASS_SPARSE, weights)
    if sparse:
        processed = _instantiate_motif(graph, normed, sparse, rng)
    else:
        processed = normed

    # MoE routing
    moe = pick_motif(rng, MOTIF_CLASS_MOE, weights)
    if moe:
        processed = _instantiate_motif(graph, processed, moe, rng)
    else:
        # Fallback to gate
        gate = pick_motif(rng, MOTIF_CLASS_GATE, weights)
        if gate:
            processed = _instantiate_motif(graph, processed, gate, rng)

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_routed_bottleneck(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """project_down(D/4) → route → sparse_core → project_up → residual_add.

    4x bottleneck + routing + sparse = compound efficiency.
    """
    D = graph.model_dim
    try:
        down = graph.add_op("linear_proj_down", [input_id],
                            config={"out_dim": max(4, D // 4)})
    except ValueError:
        return tpl_bottleneck(graph, input_id, rng, weights)

    # Route op from gate class
    gate = pick_motif(rng, MOTIF_CLASS_GATE, weights)
    if gate:
        routed = _instantiate_motif(graph, down, gate, rng)
    else:
        routed = down

    # Sparse core
    sparse = pick_motif(rng, MOTIF_CLASS_SPARSE, weights)
    if sparse:
        processed = _instantiate_motif(graph, routed, sparse, rng)
    else:
        processed = routed

    try:
        up = graph.add_op("linear_proj_up", [processed],
                          config={"out_dim": D})
    except ValueError:
        return _fix_dim(graph, processed)

    try:
        return graph.add_op("add", [input_id, up])
    except ValueError:
        return up


def tpl_token_merge_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → token_merge → mixer → sparse_ffn → project → residual_add.

    Token merging reduces sequence length for proportional FLOP savings.
    """
    norm = pick_motif(rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Token merge
    try:
        merged = graph.add_op("token_merging", [normed])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Mixer
    mixer = pick_motif_from_classes(rng, _MIXER_CLASSES, weights)
    if mixer:
        mixed = _instantiate_motif(graph, merged, mixer, rng)
    else:
        mixed = merged

    # Sparse FFN
    sparse = pick_motif(rng, MOTIF_CLASS_SPARSE, weights)
    if sparse:
        processed = _instantiate_motif(graph, mixed, sparse, rng)
    else:
        processed = mixed

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_conditional_compute(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → entropy_router → sparse_core → gate → residual_add.

    Entropy-based gating: low-entropy tokens get attenuated.
    """
    norm = pick_motif(rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Entropy router
    try:
        routed = graph.add_op("entropy_router", [normed])
    except (ValueError, KeyError):
        return tpl_gated_residual(graph, input_id, rng, weights)

    # Sparse core
    sparse = pick_motif(rng, MOTIF_CLASS_SPARSE, weights)
    if sparse:
        processed = _instantiate_motif(graph, routed, sparse, rng)
    else:
        processed = routed

    # Gate
    gate = pick_motif(rng, MOTIF_CLASS_GATE, weights)
    if gate:
        gated = _instantiate_motif(graph, processed, gate, rng)
        gated = _fix_dim(graph, gated)
    else:
        gated = _fix_dim(graph, processed)

    try:
        return graph.add_op("add", [input_id, gated])
    except ValueError:
        return gated


# ── Template Registry ───────────────────────────────────────────────

TemplateFn = Callable[
    [ComputationGraph, int, random.Random, MotifWeights],
    int,
]

TEMPLATES: Dict[str, TemplateFn] = {
    "residual_block": tpl_residual_block,
    "sequential": tpl_sequential,
    "transformer_block": tpl_transformer_block,
    "parallel_split": tpl_parallel_split,
    "bottleneck": tpl_bottleneck,
    "moe": tpl_moe,
    "hybrid_parallel": tpl_hybrid_parallel,
    "gated_residual": tpl_gated_residual,
    "dense_cascade": tpl_dense_cascade,
    "sparse_ffn": tpl_sparse_ffn,
    "sparse_moe_block": tpl_sparse_moe_block,
    "routed_bottleneck": tpl_routed_bottleneck,
    "token_merge_block": tpl_token_merge_block,
    "conditional_compute": tpl_conditional_compute,
}

# Default weights — uniform, can be overridden by judgment engine priors
DEFAULT_TEMPLATE_WEIGHTS: Dict[str, float] = {
    "residual_block": 3.0,       # Most common, reliable
    "transformer_block": 3.0,    # Classic, well-validated
    "sequential": 2.0,           # Simple stacking
    "parallel_split": 1.5,       # Width exploration
    "bottleneck": 1.5,           # Compression
    "moe": 2.0,                  # High lift (3x)
    "hybrid_parallel": 1.0,      # Hybrid attention+SSM
    "gated_residual": 1.5,       # Learned skip
    "dense_cascade": 0.8,        # Complex, DenseNet-style
    "sparse_ffn": 2.0,           # Sparse ops have 2x lift
    "sparse_moe_block": 4.0,
    "routed_bottleneck": 4.0,
    "token_merge_block": 3.5,
    "conditional_compute": 3.5,
}


def pick_template(
    rng: random.Random,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[str, TemplateFn]:
    """Pick a template weighted by success priors."""
    names = list(TEMPLATES.keys())
    template_weights = [
        (weights or {}).get(n, DEFAULT_TEMPLATE_WEIGHTS.get(n, 1.0))
        for n in names
    ]
    name = rng.choices(names, weights=template_weights, k=1)[0]
    return name, TEMPLATES[name]


def apply_template(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    template_name: Optional[str] = None,
    template_weights: Optional[Dict[str, float]] = None,
    motif_weights: MotifWeights = None,
) -> int:
    """Apply a template to the graph. Main entry point for grammar.

    If template_name is None, picks one randomly weighted by priors.
    """
    if template_name and template_name in TEMPLATES:
        fn = TEMPLATES[template_name]
    else:
        _, fn = pick_template(rng, template_weights)
    return fn(graph, input_id, rng, motif_weights)


# ── Legacy compatibility ────────────────────────────────────────────
# The old grammar called apply_random_template(graph, node_id, rng, excluded_ops)
# Keep the signature for any external callers during transition.

def apply_random_template(
    graph: ComputationGraph,
    node_id: int,
    rng: random.Random,
    excluded_ops: set = None,
) -> int:
    """Legacy wrapper. Delegates to the new template system."""
    return apply_template(graph, node_id, rng)
