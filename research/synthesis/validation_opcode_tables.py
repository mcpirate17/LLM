from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .context_rules import S1_EXEMPT_OPS, STRUCTURAL_OPS
from .primitives import OPCODE_MAP, OpCategory, PRIMITIVE_REGISTRY


_NORM_OPS = frozenset({"rmsnorm", "layernorm", "batchnorm"})
_ZERO_DEPTH_OPS = STRUCTURAL_OPS | frozenset({"causal_mask", "sliding_window_mask"})
_BRIDGE_OPS = frozenset(
    {
        "add",
        "batchnorm",
        "calibrated_branch_merge",
        "default_path",
        "layernorm",
        "learnable_bias",
        "learnable_scale",
        "linear_proj",
        "linear_proj_down",
        "linear_proj_up",
        "rmsnorm",
        "rope_rotate",
        "shared_basis_proj",
        "tied_proj",
        "token_class_proj",
        "token_entropy",
    }
)
_CONTROL_FLOW_OPS = frozenset(
    {
        "adaptive_rank_gate",
        "arch_router",
        "cheap_verify_blend",
        "compute_budget_router",
        "confidence_token_gate",
        "depth_gated_transform",
        "depth_token_mask",
        "depth_weighted_proj",
        "difficulty_blend_3way",
        "dual_compression_blend",
        "feature_sparsity",
        "gated_lane_blend",
        "hybrid_sparse_router",
        "hybrid_token_gate",
        "lane_conditioned_block",
        "learned_token_gate",
        "score_depth_blend",
        "signal_conditioned_compression",
        "sparse_span_builder",
    }
)
_CATEGORY_DEPTH_WEIGHTS = {
    OpCategory.ELEMENTWISE_UNARY: 0.65,
    OpCategory.ELEMENTWISE_BINARY: 0.70,
    OpCategory.REDUCTION: 0.15,
    OpCategory.LINEAR_ALGEBRA: 1.00,
    OpCategory.STRUCTURAL: 0.00,
    OpCategory.PARAMETERIZED: 1.00,
    OpCategory.MIXING: 1.00,
    OpCategory.SEQUENCE: 1.00,
    OpCategory.FREQUENCY: 0.80,
    OpCategory.MATH_SPACE: 1.00,
    OpCategory.FUNCTIONAL: 0.75,
}
_REQUIRED_SUCCESSOR_DISCOUNTS = {
    "cumsum": frozenset({"rmsnorm", "layernorm"}),
    "div_safe": frozenset(
        {
            "add",
            "batchnorm",
            "layernorm",
            "linear_proj",
            "linear_proj_down",
            "linear_proj_up",
            "mul",
            "rmsnorm",
            "shared_basis_proj",
            "tied_proj",
        }
    ),
    "exp": frozenset({"batchnorm", "layernorm", "mul", "rmsnorm", "sigmoid", "tanh"}),
    "log": frozenset(
        {
            "batchnorm",
            "layernorm",
            "linear_proj",
            "mul",
            "rmsnorm",
            "sigmoid",
            "tanh",
        }
    ),
    "matmul": frozenset(
        {
            "add",
            "batchnorm",
            "layernorm",
            "linear_proj",
            "linear_proj_down",
            "linear_proj_up",
            "mul",
            "rmsnorm",
            "shared_basis_proj",
            "tied_proj",
        }
    ),
    "outer_product": frozenset(
        {
            "add",
            "batchnorm",
            "layernorm",
            "linear_proj",
            "linear_proj_down",
            "linear_proj_up",
            "mul",
            "rmsnorm",
            "shared_basis_proj",
            "tied_proj",
        }
    ),
    "sign_ste": frozenset({"linear_proj", "linear_proj_down", "linear_proj_up", "mul"}),
    "sub": frozenset(
        {
            "batchnorm",
            "layernorm",
            "linear_proj",
            "linear_proj_down",
            "linear_proj_up",
            "mul",
            "rmsnorm",
            "shared_basis_proj",
            "tied_proj",
        }
    ),
}


@dataclass(frozen=True, slots=True)
class ValidationOpcodeTables:
    known: np.ndarray
    risky: np.ndarray
    parameterized: np.ndarray
    norm: np.ndarray
    linear: np.ndarray
    effective_depth_weight: np.ndarray
    discount_successor: np.ndarray


def _base_effective_op_weight(op_name: str, category: OpCategory) -> float:
    if op_name in _ZERO_DEPTH_OPS:
        return 0.0
    if op_name in _BRIDGE_OPS:
        return 0.35
    if op_name in _CONTROL_FLOW_OPS:
        return 0.50
    if op_name in S1_EXEMPT_OPS:
        return 0.15
    return _CATEGORY_DEPTH_WEIGHTS.get(category, 1.0)


@lru_cache(maxsize=1)
def validation_opcode_tables() -> ValidationOpcodeTables:
    n_opcodes = max(OPCODE_MAP.values()) + 1
    known = np.zeros(n_opcodes, dtype=np.int32)
    risky = np.zeros(n_opcodes, dtype=np.int32)
    parameterized = np.zeros(n_opcodes, dtype=np.int32)
    norm = np.zeros(n_opcodes, dtype=np.int32)
    linear = np.zeros(n_opcodes, dtype=np.int32)
    effective_depth_weight = np.zeros(n_opcodes, dtype=np.float32)
    discount_successor = np.zeros((n_opcodes, n_opcodes), dtype=bool)

    for op_name, opcode in OPCODE_MAP.items():
        op = PRIMITIVE_REGISTRY.get(op_name)
        if op is None:
            continue
        known[opcode] = 1
        risky[opcode] = int(op.numerically_risky)
        parameterized[opcode] = int(op.has_params)
        norm[opcode] = int(op_name in _NORM_OPS)
        linear[opcode] = int(op.has_params and op.shape_rule == "linear")
        effective_depth_weight[opcode] = _base_effective_op_weight(op_name, op.category)

    for parent_name, child_names in _REQUIRED_SUCCESSOR_DISCOUNTS.items():
        parent_opcode = OPCODE_MAP.get(parent_name)
        if parent_opcode is None:
            continue
        for child_name in child_names:
            child_opcode = OPCODE_MAP.get(child_name)
            if child_opcode is not None:
                discount_successor[parent_opcode, child_opcode] = True

    return ValidationOpcodeTables(
        known=known,
        risky=risky,
        parameterized=parameterized,
        norm=norm,
        linear=linear,
        effective_depth_weight=effective_depth_weight,
        discount_successor=discount_successor,
    )
