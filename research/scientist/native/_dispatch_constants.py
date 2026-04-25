"""Op-set and alias constants shared by native dispatch + compiler paths.

Extracted from ``dispatch.py`` so the constant tables are easy to find and
diff. Importers should continue to pull these names from ``dispatch`` for
compatibility; this module is the single source of truth.
"""

from __future__ import annotations

from typing import Dict, Set

NATIVE_STRUCTURAL_OPS: Set[str] = {
    "input",
    "output",
    "identity",
    "noop",
    "reshape",
    "view",
    "concat",
    "split2",
    "split",
}
_NON_KERNEL_STRUCTURAL_OPS: Set[str] = NATIVE_STRUCTURAL_OPS
_NATIVE_OP_ALIASES: Dict[str, str] = {
    "linear_proj": "linear",
    "softmax_last": "softmax",
    "transpose": "transpose2d",
    "swiglu_mlp": "swiglu",
    "adaptive_recursion": "depth_weighted_proj",
    "gated_lane_blend": "depth_weighted_proj",
    "route_lanes": "depth_weighted_proj",
    "depth_gated_transform": "depth_weighted_proj",
    "route_recursion": "depth_weighted_proj",
}
_NATIVE_C_KERNEL_OPS: Set[str] = {
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
    "add",
    "sub",
    "mul",
    "matmul",
    "linear",
    "rmsnorm",
    "layernorm",
    "softmax",
    "transpose2d",
    "softmax_attention",
    "linear_attention",
    "selective_scan",
    "state_space",
    "gated_delta",
    "gated_linear",
    "rwkv_time_mixing",
    "rwkv_channel",
    "swiglu",
    "conv1d_seq",
    # depth_weighted_proj deliberately excluded (2026-04-16): the C kernel
    # has forward but no backward, so native dispatch crashes on backward
    # with "no backward kernel for op: depth_weighted_proj". 5 aliased ops
    # (gated_lane_blend, route_lanes, depth_gated_transform, route_recursion,
    # adaptive_recursion) inherit the same state. Forcing the PyTorch path
    # for all six until the Rust backward kernel lands.
}
_STANDALONE_NATIVE_DISABLED_OPS: Set[str] = {
    # These kernels are available for fused/native subgraph execution, but the
    # Python tensor -> native array -> tensor bridge loses to PyTorch/NumPy for
    # standalone calls on the current CPU path.
    "add",
    "sub",
    "mul",
    "matmul",
    "linear",
    "linear_proj",
    "linear_proj_down",
    "linear_proj_up",
    "relu",
    "gelu",
    "silu",
    "square",
    "abs",
    "neg",
    "sin",
    "cos",
    "log",
    "sqrt",
    "reciprocal",
    "softmax",
    "layernorm",
}
_PER_OP_BRIDGE_ONLY_OPS: Set[str] = set()
_CYTHON_WRAPPER_OPS: Set[str] = set(_NATIVE_C_KERNEL_OPS)
_SOFT_BRIDGE_OPS: Set[str] = {"causal_mask", "argsort_seq", "topk_gate"}
_CYTHON_UNARY_OPS: Set[str] = {
    "relu",
    "gelu",
    "silu",
    "square",
    "abs",
    "neg",
    "reciprocal",
    "log",
    "sqrt",
    "sin",
    "cos",
    "sigmoid",
    "tanh",
    "exp",
}
_CYTHON_BINARY_OPS: Set[str] = {"add", "mul", "sub"}
_CYTHON_UNARY_BACKWARD_OPS: Set[str] = {"relu", "gelu", "silu", "sigmoid", "tanh"}
_CYTHON_BINARY_BACKWARD_OPS: Set[str] = {"add", "mul", "sub"}
_RUST_SCHEDULER_UNSUPPORTED_OPS: Set[str] = {"grade_mix", "layernorm"}
