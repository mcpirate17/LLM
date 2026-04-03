from __future__ import annotations

from functools import lru_cache

import numpy as np

from .primitives import OPCODE_MAP, PRIMITIVE_REGISTRY

FULL_DIM_OPS = frozenset(
    {
        "softmax_attention",
        "linear_attention",
        "graph_attention",
        "diff_attention",
        "state_space",
        "selective_scan",
        "rwkv_channel",
        "rwkv_time_mixing",
        "moe_topk",
        "moe_2expert",
        "swiglu_mlp",
        "gated_linear",
        "gated_delta",
    }
)
IDENTITY_LIKE_OPS = frozenset({"identity", "rmsnorm", "layernorm"})
KV_CACHE_BREAKING_OPS = frozenset(
    {
        "adjacent_token_merge",
        "depth_token_mask",
        "spectral_filter",
        "rfft",
        "irfft",
        "sort_seq",
        "unsort_seq",
        "cumsum",
        "cumprod_safe",
    }
)


@lru_cache(maxsize=None)
def build_dim_flow_opcode_tables(
    *,
    op_kind_default: int,
    op_kind_irfft: int,
    op_kind_identity: int,
    op_kind_binary_broadcast: int,
) -> dict[str, np.ndarray]:
    n_opcodes = max(OPCODE_MAP.values()) + 1
    opcode_has_params = np.zeros(n_opcodes, dtype=np.int32)
    opcode_nontrivial = np.zeros(n_opcodes, dtype=np.int32)
    opcode_kv_breaking = np.zeros(n_opcodes, dtype=np.int32)
    opcode_kind = np.full(n_opcodes, op_kind_default, dtype=np.int32)
    opcode_full_dim = np.zeros(n_opcodes, dtype=np.int32)

    for op_name, opcode in OPCODE_MAP.items():
        op = PRIMITIVE_REGISTRY.get(op_name)
        if op is None:
            continue
        if op.has_params:
            opcode_has_params[opcode] = 1
        if op_name not in IDENTITY_LIKE_OPS:
            opcode_nontrivial[opcode] = 1
        if op_name in KV_CACHE_BREAKING_OPS:
            opcode_kv_breaking[opcode] = 1
        if op_name in FULL_DIM_OPS:
            opcode_full_dim[opcode] = 1
        if op.shape_rule == "binary_broadcast":
            opcode_kind[opcode] = op_kind_binary_broadcast
        elif op.shape_rule == "irfft":
            opcode_kind[opcode] = op_kind_irfft
        elif op.shape_rule == "identity":
            opcode_kind[opcode] = op_kind_identity

    return {
        "opcode_has_params": opcode_has_params,
        "opcode_nontrivial": opcode_nontrivial,
        "opcode_kv_breaking": opcode_kv_breaking,
        "opcode_kind": opcode_kind,
        "opcode_full_dim": opcode_full_dim,
    }
