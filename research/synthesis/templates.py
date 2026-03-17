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

import contextvars
import random
from typing import Callable, Dict, Optional, Tuple

from .graph import ComputationGraph
from .motifs import (
    MOTIFS_BY_CLASS,
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_CHANNEL,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_EFFICIENT_PROJ,
    MOTIF_CLASS_FFN,
    MOTIF_CLASS_GATE,
    MOTIF_CLASS_GUARDED_ACT,
    MOTIF_CLASS_MATH_SPACE,
    MOTIF_CLASS_MOE,
    MOTIF_CLASS_NORM,
    MOTIF_CLASS_REDUCE,
    MOTIF_CLASS_SPARSE,
    MOTIF_CLASS_SSM,
    Motif,
    resolve_step,
)
from .primitives import (
    AlgebraicType,
    PRIMITIVE_REGISTRY,
    REQUIRES_RESIDUAL_BYPASS,
    algebraic_types_compatible,
)

# Type alias for motif weight dicts passed from judgment engine
MotifWeights = Optional[Dict[str, float]]

# Thread-safe context for excluded ops — set by apply_template, read by motif pickers
_excluded_ops_ctx: contextvars.ContextVar[frozenset] = contextvars.ContextVar(
    "_excluded_ops_ctx",
    default=frozenset(),
)

# ── Motif class groupings for slot constraints ──────────────────────

# Slots that accept any sequence mixer
_MIXER_CLASSES: Tuple[str, ...] = (
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_SSM,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_CHANNEL,
)

# Slots that accept any FFN-like transform
_FFN_CLASSES: Tuple[str, ...] = (
    MOTIF_CLASS_FFN,
    MOTIF_CLASS_GATE,
    MOTIF_CLASS_SPARSE,
    MOTIF_CLASS_EFFICIENT_PROJ,
)

# All motif classes
_ALL_CLASSES: Tuple[str, ...] = (
    MOTIF_CLASS_FFN,
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_SSM,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_GATE,
    MOTIF_CLASS_SPARSE,
    MOTIF_CLASS_MOE,
    MOTIF_CLASS_CHANNEL,
    MOTIF_CLASS_EFFICIENT_PROJ,
    MOTIF_CLASS_REDUCE,
    MOTIF_CLASS_GUARDED_ACT,
    MOTIF_CLASS_MATH_SPACE,
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
        if not _step_is_compatible(graph, current, op_name):
            return node_id
        # Auto-fix dim=1 outputs (from reduce_last ops like entropy_router):
        # if the current node reduced to dim=1 and the next op is parameterized,
        # insert a linear_proj to restore model_dim before the next op.
        cur_dim = graph.nodes[current].output_shape.dim
        if cur_dim == 1 and cur_dim != D:
            prim = PRIMITIVE_REGISTRY.get(op_name)
            if prim and prim.has_params:
                try:
                    current = graph.add_op(
                        "linear_proj", [current], config={"out_dim": D}
                    )
                except ValueError:
                    return node_id
                cur_dim = D
        if op_name in ("linear_proj", "fused_linear_gelu", "gated_linear"):
            config.setdefault("out_dim", D)
        elif op_name == "linear_proj_down":
            config.setdefault("out_dim", cur_dim // 2)
        elif op_name == "linear_proj_up":
            config.setdefault("out_dim", cur_dim * 2)
        elif op_name in (
            "nm_sparse_linear",
            "block_sparse_linear",
            "semi_structured_2_4_linear",
            "ternary_projection",
        ):
            config.setdefault("out_dim", cur_dim)
            if op_name == "nm_sparse_linear":
                config.setdefault("n", 2)
                config.setdefault("m", 4)
            elif op_name == "block_sparse_linear":
                config.setdefault("block_size", rng.choice([8, 16, 32]))
                config.setdefault("block_density", rng.uniform(0.05, 0.5))
        elif op_name in (
            "bottleneck_proj",
            "low_rank_proj",
            "grouped_linear",
            "shared_basis_proj",
            "tied_proj",
        ):
            config.setdefault("out_dim", cur_dim)
        elif op_name == "multi_head_mix":
            config.setdefault("n_heads", rng.choice([2, 4, 8]))
        elif op_name == "local_window_attn":
            config.setdefault("window_size", rng.choice([8, 16, 32]))
        elif op_name == "sliding_window_mask":
            config.setdefault("window_size", rng.choice([8, 16, 32]))
        elif op_name == "tropical_moe":
            config.setdefault("num_experts", rng.choice([2, 4]))
        elif op_name == "gather_topk":
            config.setdefault("k", rng.choice([4, 8, 16]))
        elif op_name in ("swiglu_mlp", "rwkv_channel", "moe_topk", "rwkv_time_mixing"):
            config.setdefault("mlp_ratio", rng.choice([2.0, 3.0, 4.0]))
        try:
            current = graph.add_op(op_name, [current], config=config)
        except (ValueError, KeyError):
            return node_id  # Bail on shape error
    return current


_INPUT_TYPE = AlgebraicType("euclidean", "real", "real")


def _node_output_type(graph: ComputationGraph, node_id: int) -> AlgebraicType:
    node = graph.nodes[node_id]
    if node.is_input:
        return _INPUT_TYPE
    return PRIMITIVE_REGISTRY[node.op_name].algebraic_type


def _step_is_compatible(graph: ComputationGraph, node_id: int, op_name: str) -> bool:
    current_type = _node_output_type(graph, node_id)
    next_op = PRIMITIVE_REGISTRY.get(op_name)
    if next_op is None:
        return False
    return algebraic_types_compatible(current_type, next_op.algebraic_type)


def _motif_is_compatible(graph: ComputationGraph, node_id: int, motif: Motif) -> bool:
    current_type = _node_output_type(graph, node_id)
    for step in motif.steps:
        step_op = PRIMITIVE_REGISTRY.get(step.op_name)
        if step_op is None or not algebraic_types_compatible(
            current_type, step_op.algebraic_type
        ):
            return False
        current_type = step_op.algebraic_type
    return True


def _motif_has_bypass_op(motif: Motif) -> bool:
    """Check if any step in motif uses an op that requires residual bypass."""
    return any(s.op_name in REQUIRES_RESIDUAL_BYPASS for s in motif.steps)


def _pick_compatible_motif(
    graph: ComputationGraph,
    node_id: int,
    rng: random.Random,
    motif_class: str,
    weights: MotifWeights = None,
) -> Optional[Motif]:
    excluded = _excluded_ops_ctx.get()
    candidates = [
        m
        for m in MOTIFS_BY_CLASS.get(motif_class, [])
        if _motif_is_compatible(graph, node_id, m)
        and not (excluded and any(s.op_name in excluded for s in m.steps))
        and not _motif_has_bypass_op(m)
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    candidate_weights = [
        weights.get(m.name, m.lift) if weights else m.lift for m in candidates
    ]
    return rng.choices(candidates, weights=candidate_weights, k=1)[0]


def _pick_compatible_motif_from_classes(
    graph: ComputationGraph,
    node_id: int,
    rng: random.Random,
    classes: Tuple[str, ...] | list[str],
    weights: MotifWeights = None,
) -> Optional[Motif]:
    excluded = _excluded_ops_ctx.get()
    pool = []
    for cls in classes:
        pool.extend(MOTIFS_BY_CLASS.get(cls, []))
    candidates = [
        m
        for m in pool
        if _motif_is_compatible(graph, node_id, m)
        and not (excluded and any(s.op_name in excluded for s in m.steps))
        and not _motif_has_bypass_op(m)
    ]
    if not candidates:
        return None
    candidate_weights = [
        weights.get(m.name, m.lift) if weights else m.lift for m in candidates
    ]
    return rng.choices(candidates, weights=candidate_weights, k=1)[0]


def _fix_dim(graph: ComputationGraph, node_id: int) -> int:
    """Add linear_proj to fix dimension back to model_dim if needed."""
    if graph.nodes[node_id].output_shape.dim != graph.model_dim:
        try:
            return graph.add_op(
                "linear_proj", [node_id], config={"out_dim": graph.model_dim}
            )
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
    norm_motif = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    if norm_motif:
        normed = _instantiate_motif(graph, input_id, norm_motif, rng)
    else:
        normed = input_id

    # Core motif (mixer or FFN)
    core_classes = list(_MIXER_CLASSES + _FFN_CLASSES)
    core_motif = _pick_compatible_motif_from_classes(
        graph, normed, rng, core_classes, weights
    )
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
        motif = _pick_compatible_motif_from_classes(
            graph, current, rng, _ALL_CLASSES, weights
        )
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
    # Attention sub-block
    norm1 = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    mixer = _pick_compatible_motif_from_classes(
        graph, normed1, rng, _MIXER_CLASSES, weights
    )
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
    norm2 = _pick_compatible_motif(graph, mid, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, mid, norm2, rng) if norm2 else mid

    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, _FFN_CLASSES, weights
    )
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
    motif_a = _pick_compatible_motif_from_classes(
        graph, split_id, rng, _MIXER_CLASSES, weights
    )
    if motif_a:
        path_a = _instantiate_motif(graph, split_id, motif_a, rng)
    else:
        path_a = split_id

    # Path B: FFN or gate
    motif_b = _pick_compatible_motif_from_classes(
        graph, split_id, rng, _FFN_CLASSES, weights
    )
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
        down = graph.add_op("linear_proj_down", [input_id], config={"out_dim": D // 2})
    except ValueError:
        return tpl_residual_block(graph, input_id, rng, weights)

    core = _pick_compatible_motif_from_classes(graph, down, rng, _ALL_CLASSES, weights)
    if core:
        processed = _instantiate_motif(graph, down, core, rng)
    else:
        processed = down

    try:
        up = graph.add_op("linear_proj_up", [processed], config={"out_dim": D})
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
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    moe = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_MOE, weights)
    if moe:
        routed = _instantiate_motif(graph, normed, moe, rng)
    else:
        # Fallback to a gate motif
        gate = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_GATE, weights)
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
    attn = _pick_compatible_motif_from_classes(
        graph, split_id, rng, (MOTIF_CLASS_ATTENTION,), weights
    )
    path_attn = _instantiate_motif(graph, split_id, attn, rng) if attn else split_id

    # SSM/conv path
    ssm = _pick_compatible_motif_from_classes(
        graph, split_id, rng, (MOTIF_CLASS_SSM, MOTIF_CLASS_CONV), weights
    )
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
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    core = _pick_compatible_motif_from_classes(
        graph, normed, rng, list(_MIXER_CLASSES + _FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, normed, core, rng) if core else normed
    processed = _fix_dim(graph, processed)

    # Gate
    gate = _pick_compatible_motif(graph, processed, rng, MOTIF_CLASS_GATE, weights)
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
        motif = _pick_compatible_motif_from_classes(
            graph, prev, rng, _ALL_CLASSES, weights
        )
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
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    sparse = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_SPARSE, weights)
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
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Sparse path
    sparse = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_SPARSE, weights)
    if sparse:
        processed = _instantiate_motif(graph, normed, sparse, rng)
    else:
        processed = normed

    # MoE routing
    moe = _pick_compatible_motif(graph, processed, rng, MOTIF_CLASS_MOE, weights)
    if moe:
        processed = _instantiate_motif(graph, processed, moe, rng)
    else:
        # Fallback to gate
        gate = _pick_compatible_motif(graph, processed, rng, MOTIF_CLASS_GATE, weights)
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
        down = graph.add_op(
            "linear_proj_down", [input_id], config={"out_dim": max(4, D // 4)}
        )
    except ValueError:
        return tpl_bottleneck(graph, input_id, rng, weights)

    # Route op from gate class
    gate = _pick_compatible_motif(graph, down, rng, MOTIF_CLASS_GATE, weights)
    if gate:
        routed = _instantiate_motif(graph, down, gate, rng)
    else:
        routed = down

    # Sparse core
    sparse = _pick_compatible_motif(graph, routed, rng, MOTIF_CLASS_SPARSE, weights)
    if sparse:
        processed = _instantiate_motif(graph, routed, sparse, rng)
    else:
        processed = routed

    try:
        up = graph.add_op("linear_proj_up", [processed], config={"out_dim": D})
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
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Token merge
    try:
        merged = graph.add_op("token_merge", [normed])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Mixer
    mixer = _pick_compatible_motif_from_classes(
        graph, merged, rng, _MIXER_CLASSES, weights
    )
    if mixer:
        mixed = _instantiate_motif(graph, merged, mixer, rng)
    else:
        mixed = merged

    # Sparse FFN
    sparse = _pick_compatible_motif(graph, mixed, rng, MOTIF_CLASS_SPARSE, weights)
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
    """norm → classifier → entropy_score → gate(sparse_core) → residual_add.

    token_type_classifier produces class logits, entropy_score measures their
    uncertainty as a (B,S,1) difficulty signal. Sparse core gated by entropy.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Classify → entropy: token_type_classifier (B,S,D)→(B,S,D) logits,
    # then entropy_score (B,S,D)→(B,S,1) difficulty signal.
    try:
        class_logits = graph.add_op(
            "token_type_classifier", [normed], config={"n_classes": 4}
        )
        difficulty = graph.add_op("entropy_score", [class_logits])
    except (ValueError, KeyError):
        return tpl_gated_residual(graph, input_id, rng, weights)

    # Sparse core: operates on full-dim normed input (NOT entropy output)
    sparse = _pick_compatible_motif(graph, normed, rng, MOTIF_CLASS_SPARSE, weights)
    if sparse:
        processed = _instantiate_motif(graph, normed, sparse, rng)
    else:
        processed = normed
    processed = _fix_dim(graph, processed)

    # Gate by difficulty: mul broadcasts (B,S,D) * (B,S,1) → (B,S,D)
    try:
        gated = graph.add_op("mul", [processed, difficulty])
    except ValueError:
        gated = processed

    try:
        return graph.add_op("add", [input_id, gated])
    except ValueError:
        return gated


# ── Routing-First Templates (Phase 2) ──────────────────────────────
#
# These templates MANDATE routing structure: every graph produced by
# these templates has a difficulty scorer and differential compute paths.
# The grammar fills motif slots, but the routing skeleton is fixed.

# Set of all routing-first template names for grammar filtering.
ROUTING_TEMPLATES: frozenset = frozenset(
    {
        "difficulty_routed_block",
        "three_lane_adaptive",
        "cascaded_early_exit",
        "recursive_depth_router",
        "conditional_compute",
        "token_merge_block",
        "routed_bottleneck",
        "sparse_moe_block",
    }
)


def tpl_difficulty_routed_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → classifier → entropy_score → {fast_path, slow_path} → gated_merge → residual.

    2-lane routing: token_type_classifier produces class logits, entropy_score
    measures their uncertainty as a (B,S,1) difficulty signal.
    Easy tokens (low entropy) get mostly the fast path (cheap linear).
    Hard tokens (high entropy) get fast + slow path (expensive motif).
    Uses mul broadcasting: (B,S,D) * (B,S,1) for differentiable gating.
    """
    # Pre-norm
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Classify → entropy: token_type_classifier (B,S,D)→(B,S,D) logits,
    # then entropy_score (B,S,D)→(B,S,1) difficulty signal.
    try:
        class_logits = graph.add_op(
            "token_type_classifier", [normed], config={"n_classes": 4}
        )
        difficulty = graph.add_op("entropy_score", [class_logits])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Fast path: cheap linear projection (always runs on all tokens)
    try:
        fast_out = graph.add_op(
            "linear_proj", [normed], config={"out_dim": graph.model_dim}
        )
    except ValueError:
        fast_out = normed

    # Slow path: expensive motif (attention/SSM/MoE + FFN)
    slow_motif = _pick_compatible_motif_from_classes(
        graph,
        normed,
        rng,
        list(_MIXER_CLASSES + _FFN_CLASSES),
        weights,
    )
    if slow_motif:
        slow_out = _instantiate_motif(graph, normed, slow_motif, rng)
    else:
        slow_out = normed
    slow_out = _fix_dim(graph, slow_out)

    # Gate slow path by difficulty: hard tokens get more slow-path signal
    try:
        slow_weighted = graph.add_op("mul", [slow_out, difficulty])
    except ValueError:
        slow_weighted = slow_out

    # Merge: fast + difficulty-weighted slow
    try:
        merged = graph.add_op("add", [fast_out, slow_weighted])
    except ValueError:
        merged = slow_weighted

    # Residual
    try:
        return graph.add_op("add", [input_id, merged])
    except ValueError:
        return merged


def tpl_three_lane_adaptive(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → adaptive_lane_mixer(3-way) → residual.

    Built-in 3-lane router: fast (identity), medium (low-rank), hard (MLP).
    The adaptive_lane_mixer op handles all lane logic internally with a
    learned gate that softly assigns tokens to difficulty lanes.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # adaptive_lane_mixer: self-contained 3-way routing
    try:
        routed = graph.add_op("adaptive_lane_mixer", [normed, normed])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    routed = _fix_dim(graph, routed)

    # Optional post-routing FFN for capacity
    ffn = _pick_compatible_motif_from_classes(graph, routed, rng, _FFN_CLASSES, weights)
    if ffn and rng.random() < 0.5:
        processed = _instantiate_motif(graph, routed, ffn, rng)
        processed = _fix_dim(graph, processed)
    else:
        processed = routed

    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_cascaded_early_exit(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """motif_1 → early_exit → motif_2 → cascade → residual.

    Progressive depth: easy tokens exit after first motif, medium tokens
    exit after second, hard tokens go through both. Uses early_exit and
    cascade ops which apply learned threshold gating.
    """
    # Stage 1: first motif block
    norm1 = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed1 = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    mixer1 = _pick_compatible_motif_from_classes(
        graph, normed1, rng, _MIXER_CLASSES, weights
    )
    if mixer1:
        stage1 = _instantiate_motif(graph, normed1, mixer1, rng)
    else:
        stage1 = normed1
    stage1 = _fix_dim(graph, stage1)

    # Early exit gate: easy tokens attenuated after stage 1
    # Residual bypass required: add(stage1, early_exit(stage1)) satisfies
    # the REQUIRES_RESIDUAL_BYPASS constraint for early_exit.
    try:
        exit_out = graph.add_op(
            "early_exit", [stage1], config={"threshold": rng.uniform(0.3, 0.6)}
        )
        gated1 = graph.add_op("add", [stage1, exit_out])
    except (ValueError, KeyError):
        gated1 = stage1

    # Stage 2: second (deeper) motif block
    norm2 = _pick_compatible_motif(graph, gated1, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, gated1, norm2, rng) if norm2 else gated1

    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, _FFN_CLASSES, weights
    )
    if ffn:
        stage2 = _instantiate_motif(graph, normed2, ffn, rng)
    else:
        stage2 = normed2
    stage2 = _fix_dim(graph, stage2)

    # Cascade gate: medium tokens attenuated after stage 2
    # Residual bypass required: add(stage2, cascade(stage2))
    try:
        cascade_out = graph.add_op(
            "cascade", [stage2], config={"threshold": rng.uniform(0.4, 0.7)}
        )
        gated2 = graph.add_op("add", [stage2, cascade_out])
    except (ValueError, KeyError):
        gated2 = stage2

    # Outer residual
    try:
        return graph.add_op("add", [input_id, gated2])
    except ValueError:
        return gated2


def tpl_recursive_depth_router(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → adaptive_recursion(depth-conditional) → motif → residual.

    Depth-adaptive: tokens re-enter the block with different parameters
    each iteration. Depth is conditional on input difficulty. Easy tokens
    get 1 pass, hard tokens get up to max_depth passes.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Depth-adaptive routing
    max_depth = rng.choice([2, 3, 4])
    try:
        depth_routed = graph.add_op(
            "adaptive_recursion", [normed], config={"max_depth": max_depth}
        )
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Post-routing motif (operates on depth-scaled tokens)
    core = _pick_compatible_motif_from_classes(
        graph,
        depth_routed,
        rng,
        list(_MIXER_CLASSES + _FFN_CLASSES),
        weights,
    )
    if core:
        processed = _instantiate_motif(graph, depth_routed, core, rng)
    else:
        processed = depth_routed
    processed = _fix_dim(graph, processed)

    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


# ── Latent Compression Templates ──────────────────────────────────
#
# Dedicated template for latent_attention_compressor — the single best-
# performing op in the leaderboard (lr=0.0061) but severely underexplored
# because it has no template forcing its selection.


def tpl_latent_compress_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → linear_proj → latent_attention_compressor → add →
    sparse_linear → act → residual_add.

    Based on the best-ever architecture pattern (5bc26a03, lr=0.0061):
    linear_proj → latent_attention_compressor → add → nm_sparse_linear →
    progressive_compression_gate → rmsnorm → rwkv_channel → add
    """
    D = graph.model_dim
    # Pre-norm
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Projection → latent attention compressor
    try:
        proj = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        compressed = graph.add_op("latent_attention_compressor", [proj])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Inner residual (normed + compressed)
    try:
        inner_res = graph.add_op("add", [normed, compressed])
    except ValueError:
        inner_res = compressed

    # Sparse linear (nm_sparse or semi_structured)
    sparse_op = rng.choice(["nm_sparse_linear", "semi_structured_2_4_linear"])
    sparse_config: dict = {"out_dim": D}
    if sparse_op == "nm_sparse_linear":
        sparse_config.update({"n": 2, "m": 4})
    try:
        sparse = graph.add_op(sparse_op, [inner_res], config=sparse_config)
    except (ValueError, KeyError):
        sparse = inner_res

    # Activation
    act_op = rng.choice(["silu", "gelu", "relu"])
    try:
        activated = graph.add_op(act_op, [sparse])
    except ValueError:
        activated = sparse

    activated = _fix_dim(graph, activated)

    # Outer residual
    try:
        return graph.add_op("add", [input_id, activated])
    except ValueError:
        return activated


def tpl_latent_compress_rwkv(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → latent_attention_compressor → add → sparse_linear →
    progressive_compression_gate → norm → rwkv_channel → residual.

    Exact replica of the best-ever graph pattern with randomized
    sparse op choice.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        proj = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        compressed = graph.add_op("latent_attention_compressor", [proj])
        inner_res = graph.add_op("add", [normed, compressed])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    sparse_op = rng.choice(
        ["nm_sparse_linear", "semi_structured_2_4_linear", "block_sparse_linear"]
    )
    sparse_cfg: dict = {"out_dim": D}
    if sparse_op == "nm_sparse_linear":
        sparse_cfg.update({"n": 2, "m": 4})
    elif sparse_op == "block_sparse_linear":
        sparse_cfg.update(
            {
                "block_size": rng.choice([8, 16, 32]),
                "block_density": rng.uniform(0.1, 0.4),
            }
        )
    try:
        sparse = graph.add_op(sparse_op, [inner_res], config=sparse_cfg)
    except (ValueError, KeyError):
        sparse = inner_res

    # Progressive compression gate (if available)
    try:
        gated = graph.add_op("progressive_compression_gate", [sparse])
    except (ValueError, KeyError):
        gated = sparse

    # Post-norm + RWKV channel mixing
    norm2 = _pick_compatible_motif(graph, gated, rng, MOTIF_CLASS_NORM, weights)
    post_normed = _instantiate_motif(graph, gated, norm2, rng) if norm2 else gated

    try:
        mixed = graph.add_op(
            "rwkv_channel",
            [post_normed],
            config={"mlp_ratio": rng.choice([2.0, 3.0, 4.0])},
        )
    except (ValueError, KeyError):
        mixed = post_normed

    mixed = _fix_dim(graph, mixed)

    try:
        return graph.add_op("add", [input_id, mixed])
    except ValueError:
        return mixed


# ── 2-Input Routing Templates ─────────────────────────────────────
#
# These templates wire routing ops that require a signal producer as
# input[1], matching OP_WIRING_RULES input_signals constraints.


def tpl_signal_routed_compression(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → classifier → {compression_mixture_experts | routing_conditioned_compression} → residual.

    2-input routing: token_type_classifier produces routing signal,
    which drives per-token compression method selection.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Produce routing signal
    try:
        signal = graph.add_op(
            "token_type_classifier",
            [normed],
            config={"n_classes": rng.choice([2, 3, 4])},
        )
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Route through compression op (2-input: data + signal)
    comp_op = rng.choice(
        ["compression_mixture_experts", "routing_conditioned_compression"]
    )
    try:
        compressed = graph.add_op(comp_op, [normed, signal])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    compressed = _fix_dim(graph, compressed)

    try:
        return graph.add_op("add", [input_id, compressed])
    except ValueError:
        return compressed


def tpl_mixed_recursion(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → classifier → mixed_recursion_gate(x, scores) → motif → residual.

    Depth-conditional: token_type_classifier produces depth scores,
    mixed_recursion_gate applies per-step transforms masked by depth.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Depth scores from classifier
    try:
        scores = graph.add_op(
            "token_type_classifier",
            [normed],
            config={"n_classes": rng.choice([3, 4, 5])},
        )
        gated = graph.add_op(
            "mixed_recursion_gate",
            [normed, scores],
            config={"max_depth": rng.choice([2, 3, 4])},
        )
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Post-routing motif
    core = _pick_compatible_motif_from_classes(
        graph,
        gated,
        rng,
        list(_MIXER_CLASSES + _FFN_CLASSES),
        weights,
    )
    if core:
        processed = _instantiate_motif(graph, gated, core, rng)
    else:
        processed = gated
    processed = _fix_dim(graph, processed)

    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_topk_retrieval(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj → cosine_similarity(proj, proj) → gather_topk → motif → residual.

    Retrieval-style: compute self-similarity scores, gather top-k
    vectors, process selected subset. Inspired by RAG reference arch.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        proj = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        scores = graph.add_op("cosine_similarity", [normed, proj])
        gathered = graph.add_op(
            "gather_topk", [normed, scores], config={"k": rng.choice([4, 8, 16])}
        )
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Process gathered subset
    core = _pick_compatible_motif_from_classes(
        graph,
        gathered,
        rng,
        list(_FFN_CLASSES),
        weights,
    )
    if core:
        processed = _instantiate_motif(graph, gathered, core, rng)
    else:
        processed = gathered
    processed = _fix_dim(graph, processed)

    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


# ── Binary-Op Safety Templates ───────────────────────────────────
#
# These templates make binary UNSAFE ops reachable by providing the
# structural context that guarantees numerical safety.


def tpl_normalized_matmul(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm_a → norm_b → matmul(a,b) → linear_proj → residual.

    Both inputs normalized ⇒ bounded spectral norm for matmul.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        proj_a = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        proj_b = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        product = graph.add_op("matmul", [proj_a, proj_b])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    product = _fix_dim(graph, product)
    try:
        return graph.add_op("add", [input_id, product])
    except ValueError:
        return product


def tpl_gated_product(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """feature → sigmoid(gate) → outer_product(feature, gate) → norm → residual.

    One input bounded by sigmoid ⇒ product bounded.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        feature = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        gate = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        gate_sig = graph.add_op("sigmoid", [gate])
        product = graph.add_op("outer_product", [feature, gate_sig])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    product = _fix_dim(graph, product)
    try:
        return graph.add_op("add", [input_id, product])
    except ValueError:
        return product


def tpl_safe_division(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """numerator → softmax(denom) → div_safe(num, denom) → proj → residual.

    Denominator from softmax ⇒ always > 0, sums to 1.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        numerator = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        denom_raw = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        denom = graph.add_op("softmax_last", [denom_raw])
        divided = graph.add_op("div_safe", [numerator, denom])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    divided = _fix_dim(graph, divided)
    try:
        return graph.add_op("add", [input_id, divided])
    except ValueError:
        return divided


def tpl_cosine_scoring(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm_a → norm_b → cosine_similarity(a,b) → linear_proj_up → residual.

    Both from norm layers ⇒ non-zero vectors guaranteed.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        proj_a = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        proj_b = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        scores = graph.add_op("cosine_similarity", [proj_a, proj_b])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    scores = _fix_dim(graph, scores)
    try:
        return graph.add_op("add", [input_id, scores])
    except ValueError:
        return scores


def tpl_decay_sequence(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """sigmoid → cumprod_safe → linear_proj → residual.

    sigmoid ∈ (0,1) ⇒ cumprod produces monotonically decaying sequence.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        proj = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        gated = graph.add_op("sigmoid", [proj])
        decayed = graph.add_op("cumprod_safe", [gated])
        out = graph.add_op("linear_proj", [decayed], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, out])
    except ValueError:
        return out


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
    # Phase 2: Routing-first templates (mandatory routing structure)
    "difficulty_routed_block": tpl_difficulty_routed_block,
    "three_lane_adaptive": tpl_three_lane_adaptive,
    "cascaded_early_exit": tpl_cascaded_early_exit,
    "recursive_depth_router": tpl_recursive_depth_router,
    # Latent compression templates (based on best-ever pattern, lr=0.0061)
    "latent_compress_block": tpl_latent_compress_block,
    "latent_compress_rwkv": tpl_latent_compress_rwkv,
    # 2-input routing templates (signal producer → routing consumer)
    "signal_routed_compression": tpl_signal_routed_compression,
    "mixed_recursion": tpl_mixed_recursion,
    "topk_retrieval": tpl_topk_retrieval,
    # Binary-op safety templates (context-aware composition)
    "normalized_matmul": tpl_normalized_matmul,
    "gated_product": tpl_gated_product,
    "safe_division": tpl_safe_division,
    "cosine_scoring": tpl_cosine_scoring,
    "decay_sequence": tpl_decay_sequence,
}

# Default weights — uniform, can be overridden by judgment engine priors
DEFAULT_TEMPLATE_WEIGHTS: Dict[str, float] = {
    "residual_block": 3.0,  # Most common, reliable
    "transformer_block": 3.0,  # Classic, well-validated
    "sequential": 2.0,  # Simple stacking
    "parallel_split": 1.5,  # Width exploration
    "bottleneck": 1.5,  # Compression
    "moe": 2.0,  # High lift (3x)
    "hybrid_parallel": 1.0,  # Hybrid attention+SSM
    "gated_residual": 1.5,  # Learned skip
    "dense_cascade": 0.8,  # Complex, DenseNet-style
    "sparse_ffn": 2.0,  # Sparse ops have 2x lift
    "sparse_moe_block": 4.0,
    "routed_bottleneck": 4.0,
    "token_merge_block": 3.5,
    "conditional_compute": 3.5,
    # Routing-first templates (Phase 2)
    "difficulty_routed_block": 5.0,  # 2-lane entropy-gated routing
    "three_lane_adaptive": 5.0,  # 3-lane adaptive mixer
    "cascaded_early_exit": 4.5,  # Progressive depth with exit gates
    "recursive_depth_router": 4.5,  # Depth-adaptive recursion
    # Latent compression (best-ever pattern, high priority)
    "latent_compress_block": 6.0,  # Latent attn compressor + sparse
    "latent_compress_rwkv": 6.0,  # Full best-ever pattern with RWKV
    # 2-input routing templates
    "signal_routed_compression": 4.0,  # Classifier-driven compression MoE
    "mixed_recursion": 4.0,  # Depth-conditional recursion gate
    "topk_retrieval": 3.5,  # Retrieval-style gather_topk
    # Binary-op safety templates
    "normalized_matmul": 2.0,  # Normalized matmul (attention-like)
    "gated_product": 2.0,  # Sigmoid-bounded outer product
    "safe_division": 1.5,  # Softmax-denominator division
    "cosine_scoring": 2.0,  # Cosine similarity scoring
    "decay_sequence": 2.0,  # Sigmoid cumprod decay
}


def pick_template(
    rng: random.Random,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[str, TemplateFn]:
    """Pick a template weighted by success priors."""
    names = list(TEMPLATES.keys())
    template_weights = [
        (weights or {}).get(n, DEFAULT_TEMPLATE_WEIGHTS.get(n, 1.0)) for n in names
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
    excluded_ops: frozenset = frozenset(),
) -> int:
    """Apply a template to the graph. Main entry point for grammar.

    If template_name is None, picks one randomly weighted by priors.
    excluded_ops: ops to filter out of motif selection (e.g. byte-unsafe ops).
    """
    token = _excluded_ops_ctx.set(excluded_ops)
    try:
        if template_name and template_name in TEMPLATES:
            fn = TEMPLATES[template_name]
        else:
            _, fn = pick_template(rng, template_weights)
        return fn(graph, input_id, rng, motif_weights)
    finally:
        _excluded_ops_ctx.reset(token)


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
