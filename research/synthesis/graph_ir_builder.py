from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List

import numpy as np

from .primitives import OPCODE_MAP, PRIMITIVE_REGISTRY, estimate_op_params, get_primitive

if TYPE_CHECKING:
    from .graph import ComputationGraph, ComputationGraphIR


@dataclass(slots=True)
class PackedIRInputs:
    op_codes: np.ndarray
    input_indices: np.ndarray
    node_ids: np.ndarray
    param_estimates: np.ndarray
    configs: List[dict]
    output_idx: int


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
) -> PackedIRInputs:
    resolved_node_ids = [int(node_id) for node_id in node_ids]
    id_to_idx = {node_id: idx for idx, node_id in enumerate(resolved_node_ids)}
    n_nodes = len(resolved_node_ids)

    op_codes = np.zeros(n_nodes, dtype=np.int32)
    input_indices = np.full((n_nodes, 2), -1, dtype=np.int32)
    param_estimates = np.zeros(n_nodes, dtype=np.int64)
    configs: List[dict] = []

    for node_id in resolved_node_ids:
        node = graph.nodes[node_id]
        idx = id_to_idx[node_id]
        op_codes[idx] = OPCODE_MAP.get(node.op_name, 0)
        if not node.is_input and node.op_name in PRIMITIVE_REGISTRY:
            param_estimates[idx] = estimate_op_params(
                get_primitive(node.op_name), graph.model_dim
            )
        for input_slot, input_id in enumerate(node.input_ids[:2]):
            input_indices[idx, input_slot] = id_to_idx.get(input_id, -1)
        configs.append(node.config)

    output_idx = (
        id_to_idx[self_output_id]
        if (self_output_id := graph._output_node_id) in id_to_idx
        else -1
    )
    return PackedIRInputs(
        op_codes=op_codes,
        input_indices=input_indices,
        node_ids=np.asarray(resolved_node_ids, dtype=np.int32),
        param_estimates=param_estimates,
        configs=configs,
        output_idx=output_idx,
    )


def build_graph_ir(
    graph: "ComputationGraph",
    *,
    node_ids: Iterable[int],
    ir_cls: type["ComputationGraphIR"],
) -> "ComputationGraphIR":
    packed = pack_ir_inputs(graph, node_ids=node_ids)
    return ir_cls(
        model_dim=graph.model_dim,
        op_codes=packed.op_codes,
        input_indices=packed.input_indices,
        output_node_idx=packed.output_idx,
        configs=packed.configs,
        node_ids=packed.node_ids,
        param_estimates=packed.param_estimates,
        source_version=graph._ir_version,
    )
