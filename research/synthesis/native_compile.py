from __future__ import annotations

from typing import Optional, Set

import torch.nn as nn

from ..scientist.native.capability import probe_supported_native_ops
from .graph import ComputationGraph
from .native_support import graph_has_bound_params


def get_supported_native_ops(graph: ComputationGraph) -> Set[str]:
    try:
        return set(probe_supported_native_ops(graph))
    except Exception:
        return set()


def try_compile_native_subgraph_layer(
    layer_factory: type[nn.Module], graph: ComputationGraph
) -> Optional[nn.Module]:
    try:
        from .native_bound_graph import BoundNativeSubgraphDispatcher
        from ..scientist.native.autograd import SubgraphDispatcher
    except Exception:
        return None

    if not graph_has_bound_params(graph):
        return None

    supported_ops = get_supported_native_ops(graph)
    if not supported_ops:
        return None

    layer = layer_factory(graph)
    try:
        flat_ops = []
        ir_node_ids = []
        if graph_has_bound_params(graph):
            flat_ops, ir_node_ids = _bound_dispatcher_inputs_from_layer(layer, graph)
        dispatcher = _select_native_subgraph_dispatcher(
            graph,
            supported_ops=supported_ops,
            flat_ops=flat_ops,
            ir_node_ids=ir_node_ids,
            bound_dispatcher_cls=BoundNativeSubgraphDispatcher,
            plain_dispatcher_cls=SubgraphDispatcher,
        )
    except Exception:
        return None
    if dispatcher is None or not dispatcher.all_native:
        return None

    layer._subgraph_dispatcher = dispatcher
    attach_partial_native_wrapper(layer, graph)
    return layer


def attach_partial_native_wrapper(layer: nn.Module, graph: ComputationGraph) -> None:
    try:
        from ..scientist.native.autograd import NativeForwardWrapper
    except Exception:
        return

    supported_ops = get_supported_native_ops(graph)
    if not supported_ops:
        return

    ops = getattr(layer, "ops", None)
    if ops is None:
        return

    op_values = ops.values() if hasattr(ops, "values") else ops
    wrapper = NativeForwardWrapper(layer, supported_ops)
    propagated = False
    for op in op_values:
        if hasattr(op, "forward"):
            op._native_wrapper = wrapper
            propagated = True
    if propagated:
        layer._native_forward_wrapper = wrapper


def _select_native_subgraph_dispatcher(
    graph: ComputationGraph,
    *,
    supported_ops: Set[str],
    flat_ops,
    ir_node_ids,
    bound_dispatcher_cls,
    plain_dispatcher_cls,
):
    if graph_has_bound_params(graph):
        bound_dispatcher = bound_dispatcher_cls(
            graph,
            flat_ops=flat_ops,
            ir_node_ids=ir_node_ids,
            supported_ops=supported_ops,
        )
        if bound_dispatcher.all_native:
            return bound_dispatcher
        return None

    plain_dispatcher = plain_dispatcher_cls(graph, supported_ops)
    if plain_dispatcher.all_native:
        return plain_dispatcher
    return None


def _bound_dispatcher_inputs_from_layer(
    layer: nn.Module,
    graph: ComputationGraph,
) -> tuple[list[object | None], list[int]]:
    ir = graph.lower_to_ir()
    ir_node_ids = (
        ir.node_ids.tolist()
        if ir.node_ids is not None
        else list(range(len(ir.op_codes)))
    )
    ops = getattr(layer, "ops", None)
    flat_ops: list[object | None] = []
    for node_id in ir_node_ids:
        if getattr(graph.nodes[node_id], "is_input", False):
            flat_ops.append(None)
            continue
        flat_ops.append(ops[str(node_id)])
    return flat_ops, ir_node_ids
