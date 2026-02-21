
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

TEMPLATES = [
    apply_gated_split_template,
    apply_fft_filter_template,
    apply_bottleneck_template,
    apply_mlp_template,
]

def apply_random_template(graph: ComputationGraph, node_id: int, rng: random.Random) -> int:
    template_fn = rng.choice(TEMPLATES)
    return template_fn(graph, node_id, rng)
