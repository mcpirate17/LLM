from __future__ import annotations

from typing import Any, Iterable

import torch

_BOUND_SINGLE_OPS: frozenset[str] | None = None


def _single_op_bound_native_ops() -> frozenset[str]:
    global _BOUND_SINGLE_OPS
    if _BOUND_SINGLE_OPS is not None:
        return _BOUND_SINGLE_OPS
    from ...synthesis.native_support import BOUND_BACKWARD_OPS, BOUND_PARAM_OPS

    _BOUND_SINGLE_OPS = frozenset(BOUND_BACKWARD_OPS & BOUND_PARAM_OPS)
    return _BOUND_SINGLE_OPS


def dispatch_single_op_bound_native(
    op_name: str,
    module: Any,
    x: torch.Tensor,
    *,
    supported_ops: Iterable[str],
) -> torch.Tensor | None:
    if op_name not in _single_op_bound_native_ops():
        return None
    cache = getattr(module, "_native_single_op_dispatchers", None)
    if cache is not None:
        dispatcher = cache.get(op_name)
        if dispatcher is not None:
            return dispatcher.try_dispatch(x)
    dispatcher = _get_or_create_single_op_dispatcher(
        op_name,
        module,
        supported_ops=frozenset(supported_ops),
    )
    if dispatcher is None:
        return None
    return dispatcher.try_dispatch(x)


def _get_or_create_single_op_dispatcher(
    op_name: str,
    module: Any,
    *,
    supported_ops: frozenset[str],
) -> Any:
    if op_name not in supported_ops:
        return None

    cache = getattr(module, "_native_single_op_dispatchers", None)
    if cache is None:
        cache = {}
        setattr(module, "_native_single_op_dispatchers", cache)

    dispatcher = cache.get(op_name)
    if dispatcher is not None:
        return dispatcher

    dispatcher = _build_single_op_dispatcher(
        op_name,
        module,
        supported_ops=supported_ops,
    )
    if dispatcher is not None:
        cache[op_name] = dispatcher
    return dispatcher


def _build_single_op_dispatcher(
    op_name: str,
    module: Any,
    *,
    supported_ops: frozenset[str],
) -> Any:
    if not hasattr(module, "model_dim"):
        return None

    from ...synthesis.graph import ComputationGraph
    from ...synthesis.native_bound_graph import BoundNativeSubgraphDispatcher

    graph = ComputationGraph(int(module.model_dim))
    input_id = graph.add_input()
    config = dict(getattr(module, "config", {}))
    output_id = graph.add_op(op_name, [input_id], config=config)
    graph.set_output(output_id)

    ir = graph.lower_to_ir()
    ir_node_ids = (
        ir.node_ids.tolist()
        if ir.node_ids is not None
        else list(range(len(ir.op_codes)))
    )
    flat_ops: list[torch.nn.Module | None] = []
    for node_id in ir_node_ids:
        flat_ops.append(None if graph.nodes[node_id].is_input else module)

    dispatcher = BoundNativeSubgraphDispatcher(
        graph,
        flat_ops=flat_ops,
        ir_node_ids=ir_node_ids,
        supported_ops=set(supported_ops),
    )
    if not dispatcher.all_native:
        return None
    return dispatcher
