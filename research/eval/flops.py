"""
FLOP Estimation

Walks a computation graph and estimates FLOPs per operation type.
Used for efficiency analysis and Pareto frontier computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from ..synthesis.graph import ComputationGraph
from ..synthesis.primitives import get_primitive, OpCategory


@dataclass
class FLOPEstimate:
    """FLOP estimation result."""
    flops_forward: int = 0
    flops_per_param: float = 0.0
    flops_per_token: float = 0.0
    breakdown: Dict[str, int] = None

    def __post_init__(self):
        if self.breakdown is None:
            self.breakdown = {}

    def to_dict(self) -> Dict:
        return {
            "flops_forward": self.flops_forward,
            "flops_per_param": self.flops_per_param,
            "flops_per_token": self.flops_per_token,
        }


# FLOP estimates per op category/type
# Expressed as functions of (seq_len S, dim D, input_dim)
def _estimate_op_flops(op_name: str, seq_len: int, d_model: int,
                       input_dim: int, config: dict = None) -> int:
    """Estimate FLOPs for a single op invocation on (B=1, S, D) tensor."""
    S, D = seq_len, input_dim
    if config is None:
        config = {}

    try:
        op = get_primitive(op_name)
    except KeyError:
        return S * D  # fallback: elementwise

    cat = op.category

    if cat == OpCategory.ELEMENTWISE_UNARY:
        # One op per element: relu, sigmoid, tanh, sin, etc.
        return S * D

    elif cat == OpCategory.ELEMENTWISE_BINARY:
        # One op per element: add, mul, etc.
        return S * D

    elif cat == OpCategory.REDUCTION:
        # Sum/mean/max over one dimension
        return S * D

    elif cat == OpCategory.LINEAR_ALGEBRA:
        if op_name == "matmul":
            # (S, D) x (D, D) => 2*S*D*D
            return 2 * S * D * D
        elif op_name == "outer_product":
            return S * D * D
        return S * D * D

    elif cat == OpCategory.PARAMETERIZED:
        if "linear" in op_name:
            # Linear projection: 2*S*D_in*D_out
            # Use actual out_dim from config if available
            out_dim = config.get("out_dim", D)
            return 2 * S * D * out_dim
        elif op_name == "conv1d_seq":
            # Depthwise conv1d: S * D * kernel_size
            return S * D * 3
        elif op_name == "selective_scan":
            # Sequential scan: ~6 ops per (S, D) element
            return 6 * S * D
        elif op_name == "topk_gate":
            # Gate projection D->2 plus gated multiply
            return S * (2 * D + 2 * D)
        elif "conv" in op_name:
            # Conv1d: 2 * S * channels * kernel_size
            k = 3  # typical kernel size
            return 2 * S * D * k
        elif "scale" in op_name or "bias" in op_name:
            return S * D
        return S * D

    elif cat == OpCategory.SEQUENCE:
        if op_name == "local_window_attn":
            # Windowed self-attention: S * W * D (W = window size)
            W = min(config.get("window_size", 32), S)
            return S * W * D + S * W * D  # scores + matmul
        elif op_name == "sliding_window_mask":
            # Build S*S mask + apply: S*S + S*S*D
            return S * S + S * S
        elif op_name == "token_pool_restore":
            # Pool S/2 elements + repeat
            return S * D
        elif "sort" in op_name:
            return S * D * int(math.log2(max(S, 2)))
        elif "cumsum" in op_name or "cumprod" in op_name:
            return S * D
        elif "roll" in op_name:
            return S * D  # just memory movement
        return S * D

    elif cat == OpCategory.FREQUENCY:
        if "rfft" in op_name or "irfft" in op_name:
            # FFT: S * log(S) * D
            return S * int(math.log2(max(S, 2))) * D
        return S * D

    elif cat == OpCategory.MATH_SPACE:
        # Exotic ops tend to be more expensive
        # Tropical/hyperbolic/clifford: ~2-5x elementwise
        return 3 * S * D

    elif cat == OpCategory.FUNCTIONAL:
        if op_name == "integral_kernel":
            # Kernel mixing: S*S + S*D*D
            return S * S * D + S * D * D
        if op_name == "fixed_point_iter":
            # 3 iterations of D*D linear + tanh
            return 3 * S * D * D
        # basis_expansion: ~4 * S * D (sin/cos)
        return 4 * S * D

    elif cat == OpCategory.STRUCTURAL:
        if op_name == "multi_head_mix":
            # L2 normalize per head: ~3 ops per element
            return 3 * S * D
        # split, concat: mostly memory movement
        return S * D

    # Default fallback
    return S * D


def estimate_flops(
    graph: ComputationGraph,
    seq_len: int = 128,
    d_model: int = 256,
) -> FLOPEstimate:
    """Estimate total forward-pass FLOPs for a computation graph.

    Args:
        graph: The computation graph to analyze
        seq_len: Sequence length
        d_model: Model dimension

    Returns:
        FLOPEstimate with total and per-param/per-token breakdowns
    """
    result = FLOPEstimate()
    total_flops = 0
    breakdown: Dict[str, int] = {}

    for nid in graph.topological_order():
        node = graph.nodes[nid]
        if node.is_input:
            continue

        input_dim = node.output_shape.dim or d_model
        flops = _estimate_op_flops(node.op_name, seq_len, d_model, input_dim,
                                    config=node.config)
        total_flops += flops
        breakdown[node.op_name] = breakdown.get(node.op_name, 0) + flops

    result.flops_forward = total_flops
    result.breakdown = breakdown

    n_params = graph.n_params_estimate()
    if n_params > 0:
        result.flops_per_param = total_flops / n_params

    if seq_len > 0:
        result.flops_per_token = total_flops / seq_len

    return result
