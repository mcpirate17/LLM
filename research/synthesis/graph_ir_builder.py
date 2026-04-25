from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Iterable, List

import numpy as np

from .primitives import (
    OPCODE_MAP,
    PRIMITIVE_REGISTRY,
    estimate_op_params,
)

if TYPE_CHECKING:
    from .graph import ComputationGraph, ComputationGraphIR


@dataclass(slots=True)
class PackedIRInputs:
    op_codes: np.ndarray
    input_indices: np.ndarray
    node_ids: np.ndarray
    param_estimates: np.ndarray
    node_dims: np.ndarray
    node_seq_flags: np.ndarray
    configs: List[dict]
    output_idx: int
    node_ids_are_contiguous: bool


def resolve_reachable_node_ids(graph: "ComputationGraph") -> list[int]:
    analysis_ir = graph._analysis_ir()
    analysis = analysis_ir.analyze_structure(include_reachable=True)
    all_node_ids = (
        analysis_ir.node_ids
        if analysis_ir.node_ids is not None
        else np.arange(analysis_ir.n_nodes(), dtype=np.int32)
    )
    return [int(all_node_ids[idx]) for idx in np.flatnonzero(analysis.reachable_mask)]


def pack_ir_inputs(
    graph: "ComputationGraph",
    *,
    node_ids: Iterable[int],
    assume_contiguous_ids: bool = False,
) -> PackedIRInputs:
    resolved_node_ids = [int(node_id) for node_id in node_ids]
    n_nodes = len(resolved_node_ids)
    contiguous_node_ids = assume_contiguous_ids or all(
        node_id == idx for idx, node_id in enumerate(resolved_node_ids)
    )
    id_to_idx = (
        None
        if contiguous_node_ids
        else {node_id: idx for idx, node_id in enumerate(resolved_node_ids)}
    )
    op_pack_info = _op_pack_info_for_dim(graph.model_dim)

    op_codes = np.zeros(n_nodes, dtype=np.int32)
    input_indices = np.full((n_nodes, 2), -1, dtype=np.int32)
    param_estimates = np.zeros(n_nodes, dtype=np.int64)
    node_dims = np.zeros(n_nodes, dtype=np.int32)
    node_seq_flags = np.zeros(n_nodes, dtype=np.int32)
    configs: List[dict] = []

    for idx, node_id in enumerate(resolved_node_ids):
        node = graph.nodes[node_id]
        opcode, params = op_pack_info.get(node.op_name, (0, 0))
        op_codes[idx] = opcode
        if not node.is_input:
            param_estimates[idx] = params
        node_dims[idx] = int(node.output_shape.dim)
        node_seq_flags[idx] = int(node.output_shape.is_freq_domain)
        for input_slot, input_id in enumerate(node.input_ids[:2]):
            if contiguous_node_ids:
                input_indices[idx, input_slot] = (
                    int(input_id) if 0 <= int(input_id) < n_nodes else -1
                )
            else:
                input_indices[idx, input_slot] = id_to_idx.get(input_id, -1)
        configs.append(node.config)

    output_node_id = graph._output_node_id
    if output_node_id is None:
        output_idx = -1
    elif contiguous_node_ids:
        output_idx = int(output_node_id) if 0 <= int(output_node_id) < n_nodes else -1
    else:
        output_idx = id_to_idx.get(output_node_id, -1)
    return PackedIRInputs(
        op_codes=op_codes,
        input_indices=input_indices,
        node_ids=np.asarray(resolved_node_ids, dtype=np.int32),
        param_estimates=param_estimates,
        node_dims=node_dims,
        node_seq_flags=node_seq_flags,
        configs=configs,
        output_idx=output_idx,
        node_ids_are_contiguous=contiguous_node_ids,
    )


def estimate_reachable_params(
    graph: "ComputationGraph", node_ids: Iterable[int]
) -> int:
    op_pack_info = _op_pack_info_for_dim(graph.model_dim)
    total = 0
    for node_id in node_ids:
        node = graph.nodes[int(node_id)]
        if not node.is_input:
            total += op_pack_info.get(node.op_name, (0, 0))[1]
    return int(total)


@lru_cache(maxsize=1024)
def _estimate_op_params_cached(op_name: str, model_dim: int) -> int:
    return estimate_op_params(PRIMITIVE_REGISTRY[op_name], int(model_dim))


@lru_cache(maxsize=32)
def _op_pack_info_for_dim(model_dim: int) -> dict[str, tuple[int, int]]:
    dim = int(model_dim)
    info: dict[str, tuple[int, int]] = {"input": (0, 0)}
    for op_name, opcode in OPCODE_MAP.items():
        op = PRIMITIVE_REGISTRY.get(op_name)
        if op is None:
            info[op_name] = (int(opcode), 0)
        else:
            info[op_name] = (int(opcode), _estimate_op_params_cached(op_name, dim))
    return info


def build_graph_ir(
    graph: "ComputationGraph",
    *,
    node_ids: Iterable[int],
    ir_cls: type["ComputationGraphIR"],
    assume_contiguous_ids: bool = False,
) -> "ComputationGraphIR":
    packed = pack_ir_inputs(
        graph,
        node_ids=node_ids,
        assume_contiguous_ids=assume_contiguous_ids,
    )
    return ir_cls(
        model_dim=graph.model_dim,
        op_codes=packed.op_codes,
        input_indices=packed.input_indices,
        output_node_idx=packed.output_idx,
        configs=packed.configs,
        node_ids=packed.node_ids,
        param_estimates=packed.param_estimates,
        node_dims=packed.node_dims,
        node_seq_flags=packed.node_seq_flags,
        node_ids_are_contiguous=packed.node_ids_are_contiguous,
        source_version=graph._ir_version,
    )
