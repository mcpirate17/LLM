from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import ComputationGraph
from .primitives import PRIMITIVE_REGISTRY, estimate_op_params


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


@dataclass(slots=True)
class DimFlowInputs:
    analysis_ir: object
    analysis: object
    analysis_node_ids: np.ndarray
    node_id_to_analysis_idx: dict[int, int]
    has_params_flags: np.ndarray
    nontrivial_flags: np.ndarray
    kv_breaking_flags: np.ndarray
    param_estimates: np.ndarray
    node_dims: np.ndarray
    node_seq_flags: np.ndarray
    op_kind_flags: np.ndarray
    full_dim_flags: np.ndarray


def build_dim_flow_inputs(
    graph: ComputationGraph,
    *,
    op_kind_default: int,
    op_kind_irfft: int,
    op_kind_identity: int,
    op_kind_binary_broadcast: int,
) -> DimFlowInputs:
    analysis_ir = graph._analysis_ir()
    analysis = analysis_ir.analyze_structure(include_reachable=True)
    analysis_node_ids = (
        analysis_ir.node_ids
        if analysis_ir.node_ids is not None
        else np.arange(analysis_ir.n_nodes(), dtype=np.int32)
    )
    node_id_to_analysis_idx = {
        int(node_id): idx for idx, node_id in enumerate(analysis_node_ids.tolist())
    }

    has_params_flags = np.zeros(analysis_ir.n_nodes(), dtype=np.int32)
    nontrivial_flags = np.zeros(analysis_ir.n_nodes(), dtype=np.int32)
    kv_breaking_flags = np.zeros(analysis_ir.n_nodes(), dtype=np.int32)
    param_estimates = np.zeros(analysis_ir.n_nodes(), dtype=np.int64)
    node_dims = np.zeros(analysis_ir.n_nodes(), dtype=np.int32)
    node_seq_flags = np.zeros(analysis_ir.n_nodes(), dtype=np.int32)
    op_kind_flags = np.full(analysis_ir.n_nodes(), op_kind_default, dtype=np.int32)
    full_dim_flags = np.zeros(analysis_ir.n_nodes(), dtype=np.int32)

    for node_id, idx in node_id_to_analysis_idx.items():
        node = graph.nodes[node_id]
        node_dims[idx] = int(node.output_shape.dim)
        node_seq_flags[idx] = int(node.output_shape.is_freq_domain)
        if node.is_input:
            continue
        op = PRIMITIVE_REGISTRY.get(node.op_name)
        if op is not None:
            if op.shape_rule == "binary_broadcast":
                op_kind_flags[idx] = op_kind_binary_broadcast
            elif op.shape_rule == "irfft":
                op_kind_flags[idx] = op_kind_irfft
            elif op.shape_rule == "identity":
                op_kind_flags[idx] = op_kind_identity
            if node.op_name in FULL_DIM_OPS:
                full_dim_flags[idx] = 1
            if op.has_params:
                has_params_flags[idx] = 1
                d_in = node.output_shape.dim or graph.model_dim
                param_estimates[idx] = estimate_op_params(op, d_in)
        if node.op_name not in IDENTITY_LIKE_OPS:
            nontrivial_flags[idx] = 1
        if node.op_name in KV_CACHE_BREAKING_OPS:
            kv_breaking_flags[idx] = 1

    return DimFlowInputs(
        analysis_ir=analysis_ir,
        analysis=analysis,
        analysis_node_ids=analysis_node_ids,
        node_id_to_analysis_idx=node_id_to_analysis_idx,
        has_params_flags=has_params_flags,
        nontrivial_flags=nontrivial_flags,
        kv_breaking_flags=kv_breaking_flags,
        param_estimates=param_estimates,
        node_dims=node_dims,
        node_seq_flags=node_seq_flags,
        op_kind_flags=op_kind_flags,
        full_dim_flags=full_dim_flags,
    )
