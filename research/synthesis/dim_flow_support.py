from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .dim_flow_opcode_tables import (
    FULL_DIM_OPS,
    KV_CACHE_BREAKING_OPS,
    build_dim_flow_opcode_tables,
)
from .graph import ComputationGraph, ComputationGraphIR
from .graph_ir_builder import build_graph_ir
from .native_analysis import analyze_ir_runtime_first

__all__ = [
    "DimFlowInputs",
    "FULL_DIM_OPS",
    "KV_CACHE_BREAKING_OPS",
    "build_dim_flow_inputs",
    "ensure_dim_flow_flags",
]


@dataclass(slots=True)
class DimFlowInputs:
    analysis_ir: object
    analysis: object | None
    analysis_node_ids: np.ndarray
    node_id_to_analysis_idx: Mapping[int, int]
    has_params_flags: np.ndarray
    nontrivial_flags: np.ndarray
    kv_breaking_flags: np.ndarray
    param_estimates: np.ndarray
    node_dims: np.ndarray
    node_seq_flags: np.ndarray
    op_kind_flags: np.ndarray
    full_dim_flags: np.ndarray
    flags_ready: bool = True


class _ContiguousNodeIndex:
    __slots__ = ("_n_nodes",)

    def __init__(self, n_nodes: int) -> None:
        self._n_nodes = int(n_nodes)

    def __contains__(self, node_id: object) -> bool:
        return (
            isinstance(node_id, (int, np.integer)) and 0 <= int(node_id) < self._n_nodes
        )

    def __getitem__(self, node_id: int) -> int:
        if 0 <= int(node_id) < self._n_nodes:
            return int(node_id)
        raise KeyError(node_id)

    def get(self, node_id: object, default: int = -1) -> int:
        return int(node_id) if node_id in self else default


def build_dim_flow_inputs(
    graph: ComputationGraph,
    *,
    op_kind_default: int,
    op_kind_irfft: int,
    op_kind_identity: int,
    op_kind_binary_broadcast: int,
    analysis_ir: Any | None = None,
    analysis: Any | None = None,
    compute_analysis: bool = True,
    build_flags: bool = True,
) -> DimFlowInputs:
    analysis_source_ir = (
        analysis_ir if analysis_ir is not None else graph._analysis_ir()
    )
    if analysis is None and compute_analysis:
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
    param_estimates = np.ascontiguousarray(
        analysis_ir.param_estimates
        if analysis_ir.param_estimates is not None
        else np.zeros(analysis_ir.n_nodes(), dtype=np.int64),
        dtype=np.int64,
    )

    if (
        getattr(analysis_ir, "node_ids_are_contiguous", False)
        and analysis_ir.node_dims is not None
        and analysis_ir.node_seq_flags is not None
    ):
        node_id_to_analysis_idx = _ContiguousNodeIndex(analysis_ir.n_nodes())
        node_dims = np.ascontiguousarray(analysis_ir.node_dims, dtype=np.int32)
        node_seq_flags = np.ascontiguousarray(
            analysis_ir.node_seq_flags, dtype=np.int32
        )
    else:
        node_dims = np.zeros(analysis_ir.n_nodes(), dtype=np.int32)
        node_seq_flags = np.zeros(analysis_ir.n_nodes(), dtype=np.int32)
        node_id_to_analysis_idx = {}
        for idx, raw_node_id in enumerate(analysis_node_ids):
            node_id = int(raw_node_id)
            node_id_to_analysis_idx[node_id] = idx
            node = graph.nodes[node_id]
            node_dims[idx] = int(node.output_shape.dim)
            node_seq_flags[idx] = int(node.output_shape.is_freq_domain)

    if build_flags:
        flag_arrays = _build_dim_flow_flag_arrays(
            op_codes=analysis_ir.op_codes,
            param_estimates=param_estimates,
            op_kind_default=op_kind_default,
            op_kind_irfft=op_kind_irfft,
            op_kind_identity=op_kind_identity,
            op_kind_binary_broadcast=op_kind_binary_broadcast,
        )
        flags_ready = True
    else:
        empty = np.empty(0, dtype=np.int32)
        flag_arrays = (empty, empty, empty, empty, empty)
        flags_ready = False

    return DimFlowInputs(
        analysis_ir=analysis_ir,
        analysis=analysis,
        analysis_node_ids=analysis_node_ids,
        node_id_to_analysis_idx=node_id_to_analysis_idx,
        has_params_flags=flag_arrays[0],
        nontrivial_flags=flag_arrays[1],
        kv_breaking_flags=flag_arrays[2],
        param_estimates=param_estimates,
        node_dims=node_dims,
        node_seq_flags=node_seq_flags,
        op_kind_flags=flag_arrays[3],
        full_dim_flags=flag_arrays[4],
        flags_ready=flags_ready,
    )


def _build_dim_flow_flag_arrays(
    *,
    op_codes: np.ndarray,
    param_estimates: np.ndarray,
    op_kind_default: int,
    op_kind_irfft: int,
    op_kind_identity: int,
    op_kind_binary_broadcast: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    opcode_tables = build_dim_flow_opcode_tables(
        op_kind_default=op_kind_default,
        op_kind_irfft=op_kind_irfft,
        op_kind_identity=op_kind_identity,
        op_kind_binary_broadcast=op_kind_binary_broadcast,
    )
    max_opcode = int(op_codes.max()) if op_codes.size else 0
    if max_opcode >= int(opcode_tables["opcode_has_params"].shape[0]):
        build_dim_flow_opcode_tables.cache_clear()
        opcode_tables = build_dim_flow_opcode_tables(
            op_kind_default=op_kind_default,
            op_kind_irfft=op_kind_irfft,
            op_kind_identity=op_kind_identity,
            op_kind_binary_broadcast=op_kind_binary_broadcast,
        )
        if max_opcode >= int(opcode_tables["opcode_has_params"].shape[0]):
            raise ValueError(
                "dim-flow opcode table is stale or incomplete: "
                f"max op code {max_opcode}, "
                f"table size {opcode_tables['opcode_has_params'].shape[0]}"
            )
    has_params_flags = opcode_tables["opcode_has_params"][op_codes] * (
        param_estimates > 0
    ).astype(np.int32, copy=False)
    return (
        has_params_flags,
        opcode_tables["opcode_nontrivial"][op_codes].copy(),
        opcode_tables["opcode_kv_breaking"][op_codes].copy(),
        opcode_tables["opcode_kind"][op_codes].copy(),
        opcode_tables["opcode_full_dim"][op_codes].copy(),
    )


def ensure_dim_flow_flags(
    inputs: DimFlowInputs,
    *,
    op_kind_default: int,
    op_kind_irfft: int,
    op_kind_identity: int,
    op_kind_binary_broadcast: int,
) -> DimFlowInputs:
    if inputs.flags_ready:
        return inputs
    (
        inputs.has_params_flags,
        inputs.nontrivial_flags,
        inputs.kv_breaking_flags,
        inputs.op_kind_flags,
        inputs.full_dim_flags,
    ) = _build_dim_flow_flag_arrays(
        op_codes=inputs.analysis_ir.op_codes,
        param_estimates=inputs.param_estimates,
        op_kind_default=op_kind_default,
        op_kind_irfft=op_kind_irfft,
        op_kind_identity=op_kind_identity,
        op_kind_binary_broadcast=op_kind_binary_broadcast,
    )
    inputs.flags_ready = True
    return inputs
