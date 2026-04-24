from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dim_flow_opcode_tables import (
    build_dim_flow_opcode_tables,
)
from .graph import ComputationGraph, ComputationGraphIR
from .graph_ir_builder import build_graph_ir
from .native_analysis import analyze_ir_runtime_first


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
    analysis_source_ir = graph._analysis_ir()
    analysis = analyze_ir_runtime_first(analysis_source_ir, include_reachable=True)
    analysis_ir = (
        analysis_source_ir
        if hasattr(analysis_source_ir, "op_codes")
        and hasattr(analysis_source_ir, "input_indices")
        else build_graph_ir(
            graph,
            node_ids=sorted(graph.nodes.keys()),
            ir_cls=ComputationGraphIR,
        )
    )
    analysis_node_ids = (
        analysis_ir.node_ids
        if analysis_ir.node_ids is not None
        else np.arange(analysis_ir.n_nodes(), dtype=np.int32)
    )
    node_dims = np.zeros(analysis_ir.n_nodes(), dtype=np.int32)
    node_seq_flags = np.zeros(analysis_ir.n_nodes(), dtype=np.int32)
    param_estimates = np.ascontiguousarray(
        analysis_ir.param_estimates
        if analysis_ir.param_estimates is not None
        else np.zeros(analysis_ir.n_nodes(), dtype=np.int64),
        dtype=np.int64,
    )

    node_id_to_analysis_idx: dict[int, int] = {}
    for idx, raw_node_id in enumerate(analysis_node_ids):
        node_id = int(raw_node_id)
        node_id_to_analysis_idx[node_id] = idx
        node = graph.nodes[node_id]
        node_dims[idx] = int(node.output_shape.dim)
        node_seq_flags[idx] = int(node.output_shape.is_freq_domain)

    opcode_tables = build_dim_flow_opcode_tables(
        op_kind_default=op_kind_default,
        op_kind_irfft=op_kind_irfft,
        op_kind_identity=op_kind_identity,
        op_kind_binary_broadcast=op_kind_binary_broadcast,
    )
    op_codes = analysis_ir.op_codes
    has_params_flags = opcode_tables["opcode_has_params"][op_codes] * (
        param_estimates > 0
    ).astype(np.int32, copy=False)
    nontrivial_flags = opcode_tables["opcode_nontrivial"][op_codes].copy()
    kv_breaking_flags = opcode_tables["opcode_kv_breaking"][op_codes].copy()
    op_kind_flags = opcode_tables["opcode_kind"][op_codes].copy()
    full_dim_flags = opcode_tables["opcode_full_dim"][op_codes].copy()
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
