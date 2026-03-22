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
    MATH_SPACE_RULES,
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

# ── Motif class groupings for slot constraints ──────────────────────

# Slots that accept any sequence mixer
_MIXER_CLASSES: Tuple[str, ...] = (
    MOTIF_CLASS_ATTENTION,
    MOTIF_CLASS_SSM,
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_CHANNEL,
    MOTIF_CLASS_MATH_SPACE,
)

# Slots that accept any FFN-like transform
_FFN_CLASSES: Tuple[str, ...] = (
    MOTIF_CLASS_FFN,
    MOTIF_CLASS_GATE,
    MOTIF_CLASS_SPARSE,
    MOTIF_CLASS_EFFICIENT_PROJ,
    MOTIF_CLASS_MOE,
    MOTIF_CLASS_GUARDED_ACT,
    MOTIF_CLASS_REDUCE,
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


# ── Context-policy imports (canonical owner: context_rules.py) ────
from .context_rules import (
    motif_allowed_in_template as _motif_allowed_in_template,
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
    Reads op_weights from graph.metadata["_op_weights"] if present.
    """
    current = node_id
    D = graph.model_dim
    _op_weights = graph.metadata.get("_op_weights")
    prev_op = (
        graph.nodes[current].op_name if not graph.nodes[current].is_input else None
    )
    for i, step in enumerate(motif.steps):
        # Peek at next step's op to inform "before" constraint
        next_step_op = motif.steps[i + 1].op_name if i + 1 < len(motif.steps) else None
        op_name, config = resolve_step(
            step, rng, prev_op=prev_op, next_op=next_step_op, op_weights=_op_weights
        )
        if not _step_is_compatible(graph, current, op_name):
            return node_id
        # Math-space safety: auto-insert rmsnorm if must_precede is unsatisfied
        ms_rules = MATH_SPACE_RULES.get(op_name)
        if ms_rules and "must_precede" in ms_rules:
            if prev_op not in ms_rules["must_precede"]:
                try:
                    current = graph.add_op("rmsnorm", [current])
                    prev_op = "rmsnorm"
                except (ValueError, KeyError):
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
            # Cap window_size to avoid Triton shared memory overflow.
            # At D>=256, W=32 exceeds GPU shared memory (151KB > 100KB).
            choices = [8, 16] if cur_dim >= 256 else [8, 16, 32]
            config.setdefault("window_size", rng.choice(choices))
        elif op_name == "sliding_window_mask":
            config.setdefault("window_size", rng.choice([8, 16, 32]))
        elif op_name == "tropical_moe":
            config.setdefault("num_experts", rng.choice([2, 4]))
        elif op_name == "gather_topk":
            config.setdefault("k", rng.choice([4, 8, 16]))
        elif op_name in ("swiglu_mlp", "rwkv_channel", "moe_topk", "rwkv_time_mixing"):
            config.setdefault("mlp_ratio", rng.choice([2.0, 3.0, 4.0]))
        pre_op = current
        try:
            current = graph.add_op(op_name, [current], config=config)
        except (ValueError, KeyError):
            return node_id  # Bail on shape error
        # Auto-wrap REQUIRES_RESIDUAL_BYPASS ops with add(input, gated)
        if op_name in REQUIRES_RESIDUAL_BYPASS:
            current = graph.add_op("add", [pre_op, current])
        prev_op = op_name
    # Record motif usage for analytics feedback loop
    if current != node_id:
        graph.metadata.setdefault("motifs_used", []).append(motif.name)
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
    # Approximate depth of node_id for min_layer_depth check
    depth = 0
    nid = node_id
    while nid in graph.nodes:
        node = graph.nodes[nid]
        if node.is_input or not node.input_ids:
            break
        nid = node.input_ids[0]
        depth += 1
    for step in motif.steps:
        step_op = PRIMITIVE_REGISTRY.get(step.op_name)
        if step_op is None or not algebraic_types_compatible(
            current_type, step_op.algebraic_type
        ):
            return False
        # Reject motif if op requires deeper placement than current position
        if step_op.min_layer_depth > 0 and depth < step_op.min_layer_depth:
            return False
        current_type = step_op.algebraic_type
    return True


def _pick_compatible_motif(
    graph: ComputationGraph,
    node_id: int,
    rng: random.Random,
    motif_class: str,
    weights: MotifWeights = None,
) -> Optional[Motif]:
    # _instantiate_motif auto-wraps REQUIRES_RESIDUAL_BYPASS ops with add,
    # so bypass motifs are safe — no need to filter them out.
    candidates = [
        m
        for m in MOTIFS_BY_CLASS.get(motif_class, [])
        if _motif_is_compatible(graph, node_id, m)
        and _motif_allowed_in_template(m, graph.metadata.get("_active_template"))
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
    pool = []
    for cls in classes:
        pool.extend(MOTIFS_BY_CLASS.get(cls, []))
    candidates = [
        m
        for m in pool
        if _motif_is_compatible(graph, node_id, m)
        and _motif_allowed_in_template(m, graph.metadata.get("_active_template"))
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
            return node_id
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


def tpl_gated_maximum(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """proj_a → proj_b → maximum(a,b) → linear_proj → residual.

    Element-wise maximum for winner-take-all feature selection.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        proj_a = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        proj_b = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        maxed = graph.add_op("maximum", [proj_a, proj_b])
        out = graph.add_op("linear_proj", [maxed], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, out])
    except ValueError:
        return out


def tpl_three_way_split(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → route_lanes(3) → split3 → proj_up → residual_add.

    Routed feature split: route_lanes provides 3-lane difficulty-based
    routing (learned gate → per-lane transforms), then split3 selects a
    feature slice from the routed output, and proj_up re-expands to D.
    The routing decision gives split3 meaningful structure to partition.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        # 3-lane routing: learned gate scores tokens, per-lane transforms
        routed = graph.add_op("route_lanes", [normed], config={"n_lanes": 3})

        # Ensure 3-divisible for split3
        shape = graph.nodes[routed].output_shape
        if shape.dim % 3 != 0:
            target_dim = max(24, (shape.dim // 3) * 3)
            routed = graph.add_op(
                "linear_proj", [routed], config={"out_dim": target_dim}
            )

        # split3 selects a feature slice — each slice captures one lane's
        # contribution since route_lanes organizes features by lane
        split_out = graph.add_op("split3", [routed])

        # Project slice back to full dim and merge with full routed signal
        expanded = graph.add_op("linear_proj_up", [split_out], config={"out_dim": D})
        merged = graph.add_op("add", [normed, expanded])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, merged])
    except ValueError:
        return merged


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
                processed = processed
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
    """norm → difficulty_scorer → mixer → early_exit(difficulty-weighted) → FFN → residual.

    Difficulty-aware early exit following the proven routing pattern:
    1. token_type_classifier → entropy_score produces per-token difficulty
    2. Mixer processes the input (attention/linear)
    3. Difficulty signal is multiplied into mixer output so early_exit's
       confidence_proj can detect easy vs hard tokens from the signal itself
    4. early_exit gates easy tokens, FFN processes survivors

    Follows the architecture of difficulty_routed_block (14% S1) rather than
    the previous 2-stage cascade pattern (0% S1).
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        # Difficulty scoring: token_type_classifier → entropy_score
        classified = graph.add_op(
            "token_type_classifier", [normed], config={"n_classes": 4}
        )
        difficulty = graph.add_op("entropy_score", [classified])

        # Mixer: process input with difficulty weighting
        proj = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        weighted = graph.add_op("mul", [proj, difficulty])
        mixed = graph.add_op("linear_proj", [weighted], config={"out_dim": D})

        # Early exit gate with residual bypass: add(mixed, early_exit(mixed))
        # Both add and early_exit share input 'mixed' to satisfy
        # REQUIRES_RESIDUAL_BYPASS check.
        exited = graph.add_op("early_exit", [mixed], config={"threshold": 0.5})
        gated = graph.add_op("add", [mixed, exited])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # FFN to process the gated output
    ffn = _pick_compatible_motif_from_classes(
        graph, gated, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, gated, ffn, rng) if ffn else gated
    processed = _fix_dim(graph, processed)

    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


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
    """norm → value_proj → decay_weights(sigmoid→cumprod) → mul(value, decay) → proj → [FFN] → residual.

    Exponential decay weighting: cumprod(sigmoid(x)) produces monotonically
    decaying weights along the sequence. These weights are applied to projected
    values via element-wise multiply — similar to RWKV time-decay mixing.
    Previous pattern fed signal directly through cumprod, which decayed to zero.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        # Value branch: what to weight
        value = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        # Decay branch: how to weight (sigmoid bounds to (0,1), cumprod decays)
        decay_proj = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        decay_gate = graph.add_op("sigmoid", [decay_proj])
        decay_weights = graph.add_op("cumprod_safe", [decay_gate])
        # Apply decay weighting to values
        weighted = graph.add_op("mul", [value, decay_weights])
        projected = graph.add_op("linear_proj", [weighted], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, projected, ffn, rng) if ffn else projected
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_hyp_distance_scoring(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """exp_map(a) → exp_map(b) → hyp_distance(a,b) → linear_proj_up → residual.

    Hyperbolic distance scoring: two projections into Poincaré ball, distance
    reduces to dim=1, linear_proj_up restores to model_dim.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        proj_a = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        proj_b = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        map_a = graph.add_op("exp_map", [proj_a])
        map_b = graph.add_op("exp_map", [proj_b])
        dist = graph.add_op("hyp_distance", [map_a, map_b])
        out = graph.add_op("linear_proj_up", [dist], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, out])
    except ValueError:
        return out


def tpl_tropical_residual(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """rmsnorm → proj_a → proj_b → tropical_add(a,b) → linear_proj → residual.

    Tropical semiring addition (element-wise min) over two projections.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        proj_a = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        proj_b = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        tadd = graph.add_op("tropical_add", [proj_a, proj_b])
        out = graph.add_op("linear_proj", [tadd], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, out])
    except ValueError:
        return out


def tpl_tropical_center_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """layernorm → tropical_attention → tropical_gate → tropical_center → residual.

    Full tropical stack: the proven architecture from 11 S1-passing programs
    (best lr=0.079). tropical_attention provides min-plus sequence mixing,
    tropical_gate does shortest-path routing, tropical_center removes the
    running minimum baseline. All three are needed for learning.

    Satisfies MATH_SPACE_RULES:
      tropical_attention.must_precede = {rmsnorm, layernorm}  ✓
      tropical_attention.must_follow_with includes tropical_center  ✓
      tropical_gate.must_precede = {rmsnorm, layernorm}  ✓ (via tropical_attention)
      tropical_gate.must_follow_with includes tropical_center  ✓
      tropical_center.must_follow = {tropical_attention, tropical_gate}  ✓
      tropical_center.must_follow_with includes linear_proj  ✓
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        attended = graph.add_op("tropical_attention", [normed])
        gated = graph.add_op("tropical_gate", [attended])
        centered = graph.add_op("tropical_center", [gated])
        out = graph.add_op("linear_proj", [centered], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, out])
    except ValueError:
        return out


def tpl_geometric_product_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → rotor_a → rotor_b → geometric_product(a,b) → grade_select → proj → [FFN] → residual.

    Clifford geometric product: rotor_transform bridges euclidean→multivector.
    FFN motif provides capacity to interpret the multivector output.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        proj_a = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        proj_b = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        rotor_a = graph.add_op("rotor_transform", [proj_a])
        rotor_b = graph.add_op("rotor_transform", [proj_b])
        product = graph.add_op("geometric_product", [rotor_a, rotor_b])
        selected = graph.add_op("grade_select", [product])
        projected = graph.add_op("linear_proj", [selected], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, projected, ffn, rng) if ffn else projected
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_residual_difference(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj_a → proj_b → sub(a,b) → proj → [FFN] → residual.

    Difference-based feature extraction (contrastive representation).
    FFN motif provides capacity to learn from the contrastive signal.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        proj_a = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        proj_b = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        diff = graph.add_op("sub", [proj_a, proj_b])
        projected = graph.add_op("linear_proj", [diff], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, projected, ffn, rng) if ffn else projected
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_tropical_matmul(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj_a → proj_b → tropical_matmul(a,b) → proj → [FFN] → residual.

    Tropical (min,+) matmul computes shortest-path distances.
    FFN motif re-densifies gradients after sparse min-plus operation.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        proj_a = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        proj_b = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        result = graph.add_op("tropical_matmul", [proj_a, proj_b])
        projected = graph.add_op("linear_proj", [result], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, projected, ffn, rng) if ffn else projected
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_gated_minimum(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj_a → proj_b → minimum(a,b) → proj → [FFN] → residual.

    Element-wise minimum for competitive feature selection.
    FFN motif provides downstream capacity to interpret min-selected features.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        proj_a = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        proj_b = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        minned = graph.add_op("minimum", [proj_a, proj_b])
        projected = graph.add_op("linear_proj", [minned], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, projected, ffn, rng) if ffn else projected
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


# ── Phase 3: Component research templates ────────────────────────────
# Dedicated templates for 24 underperforming ops. Each template
# provides a guaranteed path for ops that otherwise never get selected.
# Built from debug harness profiling (2026-03-21): all ops verified
# compile+forward+backward in these contexts.

_SPIKING_OPS = frozenset(
    {"lif_neuron", "spike_rate_code", "sparse_threshold", "stdp_attention"}
)
_HYPERBOLIC_OPS = frozenset(
    {"exp_map", "hyp_linear", "hyp_tangent_nonlinear", "log_map"}
)


def tpl_spiking_residual_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → [spiking motif] → linear_proj → residual_add.

    Forces selection of spiking motifs (lif_neuron → spike_rate_code or
    lif_neuron → sparse_threshold → stdp_attention chains).
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Filter to spiking motifs only
    spiking_motifs = [
        m
        for m in MOTIFS_BY_CLASS.get(MOTIF_CLASS_MATH_SPACE, [])
        if any(s.op_name in _SPIKING_OPS for s in m.steps)
        and _motif_is_compatible(graph, normed, m)
    ]
    if spiking_motifs:
        motif = rng.choice(spiking_motifs)
        processed = _instantiate_motif(graph, normed, motif, rng)
    else:
        processed = normed

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_spiking_moe_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """[spiking_op] → tropical_gate → linear_proj → residual_add.

    Proven pattern: spiking encoding → tropical routing → projection.
    spike_rate_code + tropical_moe achieved lr=0.007 (best spiking result).
    Spiking ops normalize internally via firing rate, so no norm needed.
    """
    # Pick a spiking+tropical motif
    spiking_tropical_motifs = [
        m
        for m in MOTIFS_BY_CLASS.get(MOTIF_CLASS_MATH_SPACE, [])
        if m.name
        in {
            "spiking_tropical_gate",
            "spiking_rate_tropical_gate",
            "spiking_threshold_tropical_gate",
        }
        and _motif_is_compatible(graph, input_id, m)
    ]
    if spiking_tropical_motifs:
        motif = rng.choice(spiking_tropical_motifs)
        processed = _instantiate_motif(graph, input_id, motif, rng)
    else:
        # Fallback: manual construction
        spiking_ops = ["lif_neuron", "spike_rate_code"]
        spike_op = rng.choice(spiking_ops)
        try:
            spiked = graph.add_op(spike_op, [input_id])
            gated = graph.add_op("tropical_gate", [spiked])
            D = graph.model_dim
            processed = graph.add_op("linear_proj", [gated], config={"out_dim": D})
        except (ValueError, KeyError):
            return tpl_residual_block(graph, input_id, rng, weights)

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_hyperbolic_bridge_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → exp_map → hyp_linear → hyp_tangent_nonlinear → log_map → proj → residual.

    Forces the full Poincaré ball round-trip via hyperbolic_residual_bridge motif.
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Filter to hyperbolic motifs
    hyp_motifs = [
        m
        for m in MOTIFS_BY_CLASS.get(MOTIF_CLASS_SSM, [])
        if any(s.op_name in _HYPERBOLIC_OPS for s in m.steps)
        and _motif_is_compatible(graph, normed, m)
    ]
    if not hyp_motifs:
        # Also check math_space class
        hyp_motifs = [
            m
            for m in MOTIFS_BY_CLASS.get(MOTIF_CLASS_MATH_SPACE, [])
            if any(s.op_name in _HYPERBOLIC_OPS for s in m.steps)
            and _motif_is_compatible(graph, normed, m)
        ]
    if hyp_motifs:
        motif = rng.choice(hyp_motifs)
        processed = _instantiate_motif(graph, normed, motif, rng)
    else:
        processed = normed

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_n_way_moe_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → n_way_sparse_router → norm → [FFN motif] → residual_add.

    N-way sparse routing with bottleneck experts. Ensures n_ways divides
    model_dim by choosing from safe divisors.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Pick n_ways that divides D
    safe_n_ways = [n for n in [2, 4, 8] if D % n == 0]
    if not safe_n_ways:
        safe_n_ways = [2]
    n_ways = rng.choice(safe_n_ways)

    try:
        routed = graph.add_op(
            "n_way_sparse_router",
            [normed],
            config={"n_ways": n_ways, "top_k": min(2, n_ways)},
        )
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Post-routing norm + FFN
    norm2 = _pick_compatible_motif(graph, routed, rng, MOTIF_CLASS_NORM, weights)
    normed2 = _instantiate_motif(graph, routed, norm2, rng) if norm2 else routed

    ffn = _pick_compatible_motif_from_classes(
        graph, normed2, rng, list(_FFN_CLASSES), weights
    )
    if ffn:
        processed = _instantiate_motif(graph, normed2, ffn, rng)
    else:
        processed = normed2

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_conv_residual_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → conv_only → [FFN motif] → residual_add.

    Local depthwise conv for short-range mixing + FFN for learning.
    Satisfies conv_only MATH_SPACE_RULES (must_precede norm, must_follow_with proj).
    """
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        conved = graph.add_op("conv_only", [normed])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, conved, rng, list(_FFN_CLASSES), weights
    )
    if ffn:
        processed = _instantiate_motif(graph, conved, ffn, rng)
    else:
        processed = conved

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_causal_mix_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → causal_mask → proj → [FFN motif] → residual_add.

    Causal cumulative average as a lightweight O(S*D) causal mixer.
    Each token becomes the running mean of itself and all predecessors.
    FFN after the mixer provides the actual learning capacity.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        mixed = graph.add_op("causal_mask", [normed])
        projected = graph.add_op("linear_proj", [mixed], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    if ffn:
        processed = _instantiate_motif(graph, projected, ffn, rng)
    else:
        processed = projected

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_iterative_refinement(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → fixed_point_iter → linear_proj → residual_add.

    Damped fixed-point iteration: z = (1-d)*z + d*tanh(z@W+b), repeated n_iters times.
    Inherently stable due to tanh bounding and damping ∈ [0,1].
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        refined = graph.add_op(
            "fixed_point_iter",
            [normed],
            config={
                "n_iters": rng.choice([2, 3]),
                "damping": round(rng.uniform(0.3, 0.7), 2),
            },
        )
        projected = graph.add_op("linear_proj", [refined], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, projected])
    except ValueError:
        return projected


def tpl_recurrent_delta_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → gated_delta → proj → [FFN motif] → residual_add.

    GatedDeltaNet: recurrent outer-product state with gated decay/update.
    6D² params — high capacity recurrent mixing + feedforward.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        recurrent = graph.add_op("gated_delta", [normed])
        projected = graph.add_op("linear_proj", [recurrent], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Optional FFN after recurrence
    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    if ffn:
        processed = _instantiate_motif(graph, projected, ffn, rng)
    else:
        processed = projected

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


# ── 0% S1 fix: dedicated templates for ops that work in isolation ────
# but fail in production search. Each template provides the specific
# architectural context where these ops contribute to learning.


def tpl_cumulative_sequence(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → cumsum → norm → proj → residual_add.

    Cumulative sum creates position-aware running statistics. Must be
    followed by normalization to prevent unbounded growth, then projection
    to learn from the accumulated signal.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        accumulated = graph.add_op("cumsum", [normed])
        # Norm after cumsum to bound the growing sum
        renormed = graph.add_op("rmsnorm", [accumulated])
        projected = graph.add_op("linear_proj", [renormed], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, projected])
    except ValueError:
        return projected


def tpl_sqrt_gated_ffn(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj_up → sqrt → gate(sigmoid) → proj_down → residual_add.

    Sqrt as a bounded activation: compresses positive values (via abs→sqrt)
    while sigmoid provides a learned gate. The combination acts as a
    soft-magnitude attention over features.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        up = graph.add_op("linear_proj_up", [normed], config={"out_dim": D * 2})
        # abs ensures non-negative input for sqrt
        abs_val = graph.add_op("abs", [up])
        sqrted = graph.add_op("sqrt", [abs_val])
        # Gate branch
        gate = graph.add_op("sigmoid", [up])
        gated = graph.add_op("mul", [sqrted, gate])
        down = graph.add_op("linear_proj_down", [gated], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, down])
    except ValueError:
        return down


def tpl_reduce_attend(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → FFN_path ⊗ reduce_gate → proj → residual_add.

    Squeeze-and-excite style: reduce ops compute per-token feature summary
    as a side-channel gate, applied to a proper FFN path that carries the
    actual learning. Without the FFN, the reduce-gate alone has insufficient
    capacity (scalar → broadcast creates uniform per-feature scaling).
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    reduce_op = rng.choice(["norm_last", "mean_last", "max_last", "sum_last"])

    try:
        # Primary path: FFN with learned transform (the actual learner)
        up = graph.add_op("linear_proj_up", [normed], config={"out_dim": D * 4})
        activated = graph.add_op("gelu", [up])
        down = graph.add_op("linear_proj_down", [activated], config={"out_dim": D})

        # Side channel: reduce → expand → sigmoid gate
        reduced = graph.add_op(reduce_op, [normed])  # (B, S, 1)
        gate_proj = graph.add_op("linear_proj_up", [reduced], config={"out_dim": D})
        gate = graph.add_op("sigmoid", [gate_proj])

        # Gate modulates FFN output: features scaled by their summary
        gated = graph.add_op("mul", [down, gate])
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, gated])
    except ValueError:
        return gated


# ── 0% S1 fix round 2: attention, SSM, activation, and structural ops ──


def tpl_fused_gelu_ffn(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → fused_linear_gelu ⊗ sigmoid(linear_proj) → proj_down → residual_add.

    Gated FFN: fused_linear_gelu provides GELU-activated up-projection,
    a parallel linear_proj with sigmoid provides the gate. The element-wise
    product lets the network selectively suppress features (SwiGLU-style).
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        fused = graph.add_op("fused_linear_gelu", [normed], config={"out_dim": D})
        gate = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        gate = graph.add_op("sigmoid", [gate])
        gated = graph.add_op("mul", [fused, gate])
        down = graph.add_op("linear_proj_down", [gated], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, down])
    except ValueError:
        return down


def tpl_exp_gated_residual(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → exp → rmsnorm → sigmoid_gate → proj → residual_add.

    Exp as a soft attention mechanism: exp amplifies positive features,
    rmsnorm after exp controls magnitude (prevents explosion during training),
    sigmoid gate selects which amplified features pass through.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        exped = graph.add_op("exp", [normed])
        # rmsnorm after exp prevents magnitude explosion
        stabilized = graph.add_op("rmsnorm", [exped])
        gate = graph.add_op("sigmoid", [normed])
        gated = graph.add_op("mul", [stabilized, gate])
        projected = graph.add_op("linear_proj", [gated], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, projected])
    except ValueError:
        return projected


def tpl_integral_kernel_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → integral_kernel → proj → [FFN motif] → residual_add.

    Integral transform kernel: continuous-domain convolution via learned
    kernel functions. Similar to SSM but with integral operator semantics.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        mixed = graph.add_op("integral_kernel", [normed])
        projected = graph.add_op("linear_proj", [mixed], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # Optional FFN after
    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    if ffn:
        processed = _instantiate_motif(graph, projected, ffn, rng)
    else:
        processed = projected

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_windowed_attention(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → sliding_window_mask → proj → [FFN motif] → residual_add.

    Sliding window applies local context mixing via exponential-decay
    weighted sum. The mask has zero learnable params, so an FFN motif
    provides the downstream learning capacity (same pattern as
    local_attention_block).
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        windowed = graph.add_op(
            "sliding_window_mask",
            [normed],
            config={"window_size": rng.choice([8, 16, 32])},
        )
        projected = graph.add_op("linear_proj", [windowed], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    if ffn:
        processed = _instantiate_motif(graph, projected, ffn, rng)
    else:
        processed = projected

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_local_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → local_window_attn → proj → [FFN motif] → residual_add.

    Local window attention: efficient O(n*w) attention within sliding windows.
    Caps window_size to avoid Triton shared memory overflow.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    # Cap window_size based on dim to avoid shared memory overflow
    choices = [8, 16] if D >= 256 else [8, 16, 32]
    try:
        attended = graph.add_op(
            "local_window_attn",
            [normed],
            config={"window_size": rng.choice(choices)},
        )
        projected = graph.add_op("linear_proj", [attended], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    if ffn:
        processed = _instantiate_motif(graph, projected, ffn, rng)
    else:
        processed = projected

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_state_space_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → state_space → proj → [FFN motif] → residual_add.

    State-space model (selective scan): linear recurrence with gated updates.
    Efficient O(n log n) via parallel scan. Norm predecessor bounds scan input.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        ssm_out = graph.add_op("state_space", [normed])
        projected = graph.add_op("linear_proj", [ssm_out], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    if ffn:
        processed = _instantiate_motif(graph, projected, ffn, rng)
    else:
        processed = projected

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_rwkv_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → rwkv_time_mixing → proj → [FFN motif] → residual_add.

    RWKV time-mixing: WKV attention with exponential decay and learned bonus.
    4D² + 2D params (W_k, W_v, W_r, W_o, w_decay, u_bonus).
    Similar to state_space but with receptance gating.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        rwkv_out = graph.add_op("rwkv_time_mixing", [normed])
        projected = graph.add_op("linear_proj", [rwkv_out], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    if ffn:
        processed = _instantiate_motif(graph, projected, ffn, rng)
    else:
        processed = projected

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_reciprocal_gated(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → sigmoid → reciprocal → gate(mul) → proj → residual_add.

    Reciprocal as inverse-attention: 1/(1+exp(-x)) → 1/sigmoid(x) maps
    confident features to ~1.0 and uncertain features to ~2.0, inverting
    the attention distribution. Sigmoid predecessor bounds reciprocal input.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        sig = graph.add_op("sigmoid", [normed])
        recip = graph.add_op("reciprocal", [sig])
        gated = graph.add_op("mul", [normed, recip])
        projected = graph.add_op("linear_proj", [gated], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, projected])
    except ValueError:
        return projected


def tpl_log_gated(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → sigmoid → log → gate(mul) → proj → residual_add.

    Log-compression: sigmoid guarantees positive input (range (0,1)),
    log compresses to (-inf, 0). Gate controls which log-compressed
    features pass through, preventing unbounded negative values from
    dominating the representation.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        bounded = graph.add_op("sigmoid", [normed])
        logged = graph.add_op("log", [bounded])
        gate = graph.add_op("sigmoid", [normed])
        gated = graph.add_op("mul", [logged, gate])
        projected = graph.add_op("linear_proj", [gated], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, projected])
    except ValueError:
        return projected


def tpl_sign_ste_gated(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → proj → sign_ste → gate(sigmoid) → mul → proj → residual_add.

    Binary quantization via STE: sign binarizes activations while
    straight-through estimator passes gradients. Sigmoid gate controls
    which binarized features contribute, preventing sign from zeroing
    gradient signal entirely.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        projected = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        signed = graph.add_op("sign_ste", [projected])
        gate = graph.add_op("sigmoid", [projected])
        gated = graph.add_op("mul", [signed, gate])
        out = graph.add_op("linear_proj", [gated], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    try:
        return graph.add_op("add", [input_id, out])
    except ValueError:
        return out


def tpl_diff_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → diff_attention → proj → [FFN motif] → residual_add.

    Differential attention (Microsoft, ICLR 2025): two softmax maps
    subtracted to cancel noise. The op handles Q/K/V projection and
    dual-softmax internally. Template provides normalized input, FFN
    capacity, and residual connection.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        attended = graph.add_op("diff_attention", [normed])
        projected = graph.add_op("linear_proj", [attended], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, projected, ffn, rng) if ffn else projected
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_graph_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → graph_attention → proj → [FFN motif] → residual_add.

    Graph attention: attention where edge weights are learned per node pair.
    Higher capacity than softmax attention (2.19x lift in top performers).
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        attended = graph.add_op("graph_attention", [normed])
        projected = graph.add_op("linear_proj", [attended], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    if ffn:
        processed = _instantiate_motif(graph, projected, ffn, rng)
    else:
        processed = projected

    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


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
    "residual_difference": tpl_residual_difference,
    "gated_minimum": tpl_gated_minimum,
    "hyp_distance_scoring": tpl_hyp_distance_scoring,
    "tropical_residual": tpl_tropical_residual,
    "tropical_matmul_block": tpl_tropical_matmul,
    "geometric_product_block": tpl_geometric_product_block,
    "gated_maximum": tpl_gated_maximum,
    "three_way_split": tpl_three_way_split,
    # Phase 3: Component research templates (dedicated paths for underperforming ops)
    "cumulative_sequence": tpl_cumulative_sequence,
    "sqrt_gated_ffn": tpl_sqrt_gated_ffn,
    "reduce_attend": tpl_reduce_attend,
    # 0% S1 fix round 2: attention, SSM, activation, structural
    "fused_gelu_ffn": tpl_fused_gelu_ffn,
    "exp_gated_residual": tpl_exp_gated_residual,
    "integral_kernel_block": tpl_integral_kernel_block,
    "windowed_attention": tpl_windowed_attention,
    "local_attention_block": tpl_local_attention_block,
    "state_space_block": tpl_state_space_block,
    "rwkv_block": tpl_rwkv_block,
    "reciprocal_gated": tpl_reciprocal_gated,
    "diff_attention_block": tpl_diff_attention_block,
    "graph_attention_block": tpl_graph_attention_block,
    "spiking_residual_block": tpl_spiking_residual_block,
    "spiking_moe_block": tpl_spiking_moe_block,
    "hyperbolic_bridge_block": tpl_hyperbolic_bridge_block,
    "n_way_moe_block": tpl_n_way_moe_block,
    "conv_residual_block": tpl_conv_residual_block,
    "causal_mix_block": tpl_causal_mix_block,
    "iterative_refinement": tpl_iterative_refinement,
    "recurrent_delta_block": tpl_recurrent_delta_block,
    "sign_ste_gated": tpl_sign_ste_gated,
    "log_gated": tpl_log_gated,
    "tropical_center_block": tpl_tropical_center_block,
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
    "decay_sequence": 3.0,  # Sigmoid cumprod decay (only path to cumprod_safe)
    "residual_difference": 2.5,  # Contrastive sub(a,b)
    "gated_minimum": 2.5,  # Competitive minimum(a,b)
    "hyp_distance_scoring": 1.5,  # Hyperbolic distance scoring
    "tropical_residual": 2.5,  # Tropical semiring add
    "tropical_matmul_block": 2.5,  # Tropical (min,+) matmul
    "geometric_product_block": 1.5,  # Clifford geometric product
    "gated_maximum": 1.5,  # Winner-take-all maximum(a,b)
    "three_way_split": 2.5,  # 3-way parallel (split3, now auto-projects to 3-divisible dim)
    # 0% S1 fix: dedicated contexts for ops that need specific surroundings
    "cumulative_sequence": 2.5,  # cumsum → norm → proj (position-aware running stats)
    "sqrt_gated_ffn": 2.5,  # abs → sqrt gated by sigmoid (soft magnitude attention)
    "reduce_attend": 3.0,  # reduce → expand → gate input (feature summary attention)
    # Phase 3: Component research templates
    # 0% S1 fix round 2
    "fused_gelu_ffn": 3.0,  # Fused proj+GELU as FFN (faster than separate)
    "exp_gated_residual": 2.5,  # Exp amplification gated by sigmoid
    "integral_kernel_block": 3.0,  # Integral transform kernel (continuous conv)
    "windowed_attention": 3.0,  # Sliding window mask + attention
    "local_attention_block": 3.0,  # Local window attention (O(n*w))
    "state_space_block": 3.5,  # State-space model (parallel scan)
    "rwkv_block": 3.5,  # RWKV WKV attention (receptance gating, 4D²+2D params)
    "reciprocal_gated": 2.5,  # Inverse-attention via 1/sigmoid
    "sign_ste_gated": 2.5,  # Binary quantization via STE + sigmoid gate
    "log_gated": 2.5,  # Log-compression with sigmoid bounding
    "tropical_center_block": 2.5,  # Tropical gate → center → proj
    "diff_attention_block": 3.5,  # Differential attention (Microsoft ICLR 2025)
    "graph_attention_block": 3.5,  # Graph attention (2.19x lift in top performers)
    "spiking_residual_block": 3.0,  # Spiking neuromorphic (LIF + STDP chains)
    "spiking_moe_block": 4.0,  # Spiking + tropical routing (proven lr=0.007)
    "hyperbolic_bridge_block": 3.0,  # Poincaré ball round-trip (exp→hyp→log)
    "n_way_moe_block": 3.5,  # N-way sparse routing with bottleneck experts
    "conv_residual_block": 3.0,  # Depthwise causal conv + FFN
    "causal_mix_block": 2.5,  # Causal cumulative average + FFN (O(S*D) mixer)
    "iterative_refinement": 2.5,  # Damped fixed-point iteration
    "recurrent_delta_block": 3.5,  # GatedDeltaNet recurrence (6D² params)
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
    op_weights: Optional[Dict[str, float]] = None,
) -> int:
    """Apply a template to the graph. Main entry point for grammar.

    If template_name is None, picks one randomly weighted by priors.
    op_weights biases activation substitution in resolve_step.
    """
    if template_name and template_name in TEMPLATES:
        name = template_name
        fn = TEMPLATES[name]
    else:
        name, fn = pick_template(rng, template_weights)
    # Stash op_weights for _instantiate_motif to read
    if op_weights:
        graph.metadata["_op_weights"] = op_weights
    # Tag template usage for analytics feedback loop
    graph.metadata.setdefault("templates_used", []).append(name)
    prev_template = graph.metadata.get("_active_template")
    graph.metadata["_active_template"] = name
    try:
        return fn(graph, input_id, rng, motif_weights)
    finally:
        if prev_template is None:
            graph.metadata.pop("_active_template", None)
        else:
            graph.metadata["_active_template"] = prev_template
