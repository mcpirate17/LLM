
"""
Discovered Graph Templates

Canonical subgraphs mined from successful experiments. 
These act as "opinionated" seeds for the grammar to increase the
density of high-quality candidates in the search space.
"""

import random
from typing import List, Dict, Optional
from .graph import ComputationGraph

def apply_gated_split_template(graph: ComputationGraph, node_id: int, rng: random.Random) -> int:
    """Pattern: split2 -> {unary_op, identity} -> concat.
    Common in GLU-like and gated architectures.
    """
    try:
        split_id = graph.add_op("split2", [node_id])
        
        # Mined successful ops for the gate path
        gate_ops = ["sigmoid", "tanh", "silu", "gelu", "square"]
        op_name = rng.choice(gate_ops)
        
        gate_path = graph.add_op(op_name, [split_id])
        
        # Merge back
        merged = graph.add_op("concat", [gate_path, split_id])
        return merged
    except Exception:
        return node_id

def apply_fft_filter_template(graph: ComputationGraph, node_id: int, rng: random.Random) -> int:
    """Pattern: rfft_seq -> learnable_scale -> irfft_seq.
    Discovered in dba351f2 as a high-novelty survivor.
    """
    try:
        freq_id = graph.add_op("rfft_seq", [node_id])
        scaled_id = graph.add_op("learnable_scale", [freq_id])
        time_id = graph.add_op("irfft_seq", [scaled_id])
        return time_id
    except Exception:
        return node_id

def apply_bottleneck_template(graph: ComputationGraph, node_id: int, rng: random.Random) -> int:
    """Pattern: linear_proj_down -> bottleneck_proj -> linear_proj_up.
    Efficient parameter usage.
    """
    D = graph.model_dim
    try:
        down = graph.add_op("linear_proj_down", [node_id], config={"out_dim": D // 2})
        mid = graph.add_op("bottleneck_proj", [down])
        up = graph.add_op("linear_proj_up", [mid], config={"out_dim": D})
        return up
    except Exception:
        return node_id

def apply_mlp_template(graph: ComputationGraph, node_id: int, rng: random.Random) -> int:
    """Pattern: linear_proj_up -> {activation} -> linear_proj_down.
    Standard MLP expansion/contraction block.
    """
    D = graph.model_dim
    try:
        # Expansion
        up = graph.add_op("linear_proj_up", [node_id], config={"out_dim": D * 2})
        
        # Activation
        act_op = rng.choice(["gelu", "silu", "relu", "tanh"])
        act = graph.add_op(act_op, [up])
        
        # Contraction
        down = graph.add_op("linear_proj_down", [act], config={"out_dim": D})
        return down
    except Exception:
        return node_id

def apply_mod_multilane_template(graph: ComputationGraph, node_id: int, rng: random.Random) -> int:
    """Pattern: token_type_classifier → route_lanes → {easy: linear_proj, hard: moe_topk} → concat.
    Multi-lane Mixture-of-Depths routing with difficulty-aware dispatch.
    """
    try:
        classify = graph.add_op("token_type_classifier", [node_id])
        lanes = graph.add_op("route_lanes", [classify])
        # Easy path: lightweight linear projection
        easy = graph.add_op("linear_proj", [lanes])
        # Hard path: full MoE processing
        hard = graph.add_op("moe_topk", [lanes])
        merged = graph.add_op("concat", [easy, hard])
        return merged
    except Exception:
        return node_id


def apply_compression_routing_template(graph: ComputationGraph, node_id: int, rng: random.Random) -> int:
    """Pattern: entropy_router → route_topk → {compress: latent_attention_compressor, pass: identity} → add.
    Entropy-based routing decides which tokens get compressed.
    """
    try:
        router = graph.add_op("entropy_router", [node_id])
        topk = graph.add_op("route_topk", [router])
        compressed = graph.add_op("latent_attention_compressor", [topk])
        merged = graph.add_op("add", [compressed, node_id])
        return merged
    except Exception:
        return node_id


def apply_sparse_moe_block_template(graph: ComputationGraph, node_id: int, rng: random.Random) -> int:
    """Pattern: linear_proj → moe_topk → rmsnorm → residual add.
    Standard sparse MoE block with normalization.
    """
    try:
        proj = graph.add_op("linear_proj", [node_id])
        moe = graph.add_op("moe_topk", [proj])
        normed = graph.add_op("rmsnorm", [moe])
        out = graph.add_op("add", [normed, node_id])
        return out
    except Exception:
        return node_id


def apply_adaptive_depth_template(graph: ComputationGraph, node_id: int, rng: random.Random) -> int:
    """Pattern: mod_topk → adaptive_recursion → mixed_recursion_gate → add.
    Adaptive depth block: tokens choose their own processing depth.
    """
    try:
        mod = graph.add_op("mod_topk", [node_id])
        recurse = graph.add_op("adaptive_recursion", [mod])
        gate = graph.add_op("mixed_recursion_gate", [recurse, node_id])
        out = graph.add_op("add", [gate, node_id])
        return out
    except Exception:
        return node_id


TEMPLATES = [
    apply_gated_split_template,
    apply_fft_filter_template,
    apply_bottleneck_template,
    apply_mlp_template,
]

EXOTIC_TEMPLATES = [
    apply_mod_multilane_template,
    apply_compression_routing_template,
    apply_sparse_moe_block_template,
    apply_adaptive_depth_template,
]

def apply_random_template(graph: ComputationGraph, node_id: int, rng: random.Random,
                          excluded_ops: set = None) -> int:
    pool = TEMPLATES
    if excluded_ops and ("rfft_seq" in excluded_ops or "irfft_seq" in excluded_ops):
        pool = [t for t in pool if t is not apply_fft_filter_template]
    if not pool:
        return node_id
    template_fn = rng.choice(pool)
    return template_fn(graph, node_id, rng)
