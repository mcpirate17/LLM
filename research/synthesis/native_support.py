from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph

BOUND_POINTWISE_OPS = frozenset(
    {
        "relu",
        "gelu",
        "silu",
        "sigmoid",
        "tanh",
        "exp",
        "square",
        "abs",
        "neg",
        "sin",
        "cos",
        "log",
        "sqrt",
        "reciprocal",
        "softmax",
    }
)

BOUND_BINARY_OPS = frozenset({"add", "sub", "mul"})
BOUND_STRUCTURAL_OPS = frozenset({"input", "output", "identity", "noop"})
BOUND_PARAM_OPS = frozenset(
    {
        "linear_proj",
        "linear_proj_down",
        "linear_proj_up",
        "rmsnorm",
        "layernorm",
        "conv1d_seq",
        "gated_linear",
        "rwkv_time_mixing",
        "rwkv_channel",
        "swiglu_mlp",
        "conv_only",
        "softmax_attention",
        "linear_attention",
        "gated_lane_blend",
        "route_lanes",
        "depth_gated_transform",
        "route_recursion",
        "depth_weighted_proj",
        "adaptive_recursion",
        "score_depth_blend",
        "mixed_recursion_gate",
        "selective_scan",
        "state_space",
        "gated_delta",
    }
)
BOUND_SUPPORTED_OPS = (
    BOUND_POINTWISE_OPS | BOUND_BINARY_OPS | BOUND_STRUCTURAL_OPS | BOUND_PARAM_OPS
)
BOUND_BACKWARD_OPS = frozenset(
    {
        "relu",
        "gelu",
        "silu",
        "sigmoid",
        "tanh",
        "add",
        "sub",
        "mul",
        "linear_proj",
        "linear_proj_down",
        "linear_proj_up",
        "rmsnorm",
        "layernorm",
        "gated_linear",
        "rwkv_time_mixing",
        "rwkv_channel",
        "swiglu_mlp",
        "conv1d_seq",
        "conv_only",
        "softmax_attention",
        "selective_scan",
        "state_space",
        "gated_delta",
    }
)
BOUND_SUPPORTED_INPUT_RANKS = frozenset({2, 3})


def graph_has_bound_params(graph: ComputationGraph) -> bool:
    nodes = getattr(graph, "nodes", None)
    if not isinstance(nodes, dict):
        return False
    return any(
        getattr(node, "op_name", "") in BOUND_PARAM_OPS
        for node in nodes.values()
        if not getattr(node, "is_input", False)
    )
