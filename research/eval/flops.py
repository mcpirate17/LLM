"""
FLOP Estimation

Walks a computation graph and estimates FLOPs per operation type.
Used for efficiency analysis and Pareto frontier computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional

from ..synthesis.graph import ComputationGraph
from ..synthesis.primitives import get_primitive, OpCategory


@dataclass
class FLOPEstimate:
    """FLOP estimation result."""

    flops_forward: int = 0
    flops_per_param: float = 0.0
    flops_per_token: float = 0.0
    breakdown: Dict[str, int] = field(default_factory=dict)
    estimate_method: str = "heuristic_static_graph"
    measured: bool = False
    performance_claim_valid: bool = False
    estimate_warning: str = "Heuristic FLOP estimate only. Not kernel-level measurement and not a runtime performance claim."

    def to_dict(self) -> Dict:
        return {
            "flops_forward": self.flops_forward,
            "flops_per_param": self.flops_per_param,
            "flops_per_token": self.flops_per_token,
            "estimate_method": self.estimate_method,
            "measured": self.measured,
            "performance_claim_valid": self.performance_claim_valid,
            "estimate_warning": self.estimate_warning,
        }


@lru_cache(maxsize=512)
def _cached_op_category(op_name: str) -> Optional[OpCategory]:
    try:
        return get_primitive(op_name).category
    except KeyError:
        return None


def _estimate_linear_algebra_flops(op_name: str, seq_len: int, width: int) -> int:
    if op_name == "matmul":
        return 2 * seq_len * width * width
    return seq_len * width * width


def _estimate_parameterized_flops(
    op_name: str, seq_len: int, width: int, config: Dict[str, object]
) -> int:
    out_dim = int(config.get("out_dim", width))
    if op_name == "nm_sparse_linear":
        n = max(1, int(config.get("n", 2)))
        m = max(n, int(config.get("m", 4)))
        density = min(1.0, float(n) / float(max(m, 1)))
        return int(2 * seq_len * width * out_dim * density)
    if op_name == "semi_structured_2_4_linear":
        density = 0.5 if width % 4 == 0 and out_dim % 4 == 0 else 1.0
        return int(2 * seq_len * width * out_dim * density)
    if op_name == "block_sparse_linear":
        density = float(max(0.05, min(1.0, config.get("block_density", 0.25))))
        return int(2 * seq_len * width * out_dim * density)
    if "linear" in op_name:
        return 2 * seq_len * width * out_dim
    if op_name == "conv1d_seq":
        return seq_len * width * 3
    if op_name == "selective_scan":
        return 6 * seq_len * width
    if op_name == "topk_gate":
        return seq_len * (4 * width)
    if "conv" in op_name:
        return 2 * seq_len * width * 3
    if "scale" in op_name or "bias" in op_name:
        return seq_len * width
    return seq_len * width


def _estimate_sequence_flops(
    op_name: str, seq_len: int, width: int, config: Dict[str, object]
) -> int:
    if op_name == "local_window_attn":
        window = min(int(config.get("window_size", 32)), seq_len)
        return 2 * seq_len * window * width
    if op_name == "sliding_window_mask":
        return 2 * seq_len * seq_len
    if op_name == "token_pool_restore":
        return seq_len * width
    if "sort" in op_name:
        return seq_len * width * int(math.log2(max(seq_len, 2)))
    if "cumsum" in op_name or "cumprod" in op_name or "roll" in op_name:
        return seq_len * width
    return seq_len * width


def _estimate_math_space_flops(op_name: str, seq_len: int, width: int) -> int:
    bottleneck = max(1, width // 4)
    if op_name in {"low_rank_proj", "bottleneck_proj", "tied_proj"}:
        return 2 * seq_len * width * bottleneck
    if op_name == "grouped_linear":
        return seq_len * width * bottleneck
    if op_name == "shared_basis_proj":
        return seq_len * 16 * width
    return 3 * seq_len * width


def _estimate_category_flops(
    category: OpCategory,
    op_name: str,
    seq_len: int,
    width: int,
    config: Dict[str, object],
) -> int:
    if category in {
        OpCategory.ELEMENTWISE_UNARY,
        OpCategory.ELEMENTWISE_BINARY,
        OpCategory.REDUCTION,
        OpCategory.STRUCTURAL,
    }:
        if category == OpCategory.STRUCTURAL and op_name == "multi_head_mix":
            return 3 * seq_len * width
        return seq_len * width
    if category == OpCategory.LINEAR_ALGEBRA:
        return _estimate_linear_algebra_flops(op_name, seq_len, width)
    if category == OpCategory.PARAMETERIZED:
        return _estimate_parameterized_flops(op_name, seq_len, width, config)
    if category == OpCategory.SEQUENCE:
        return _estimate_sequence_flops(op_name, seq_len, width, config)
    if category == OpCategory.FREQUENCY:
        if "rfft" in op_name or "irfft" in op_name:
            return seq_len * int(math.log2(max(seq_len, 2))) * width
        return seq_len * width
    if category == OpCategory.MATH_SPACE:
        return _estimate_math_space_flops(op_name, seq_len, width)
    if category == OpCategory.FUNCTIONAL:
        if op_name == "integral_kernel":
            return seq_len * seq_len * width + seq_len * width * width
        if op_name == "fixed_point_iter":
            return 3 * seq_len * width * width
        return 4 * seq_len * width
    return seq_len * width


def _estimate_op_flops(
    op_name: str, seq_len: int, d_model: int, input_dim: int, config: dict = None
) -> int:
    """Estimate FLOPs for a single op invocation on (B=1, S, D) tensor."""
    del d_model
    config_dict = {} if config is None else config
    width = input_dim
    category = _cached_op_category(op_name)
    if category is None:
        return seq_len * width
    return _estimate_category_flops(category, op_name, seq_len, width, config_dict)


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
        flops = _estimate_op_flops(
            node.op_name, seq_len, d_model, input_dim, config=node.config
        )
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
