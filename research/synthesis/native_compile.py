from __future__ import annotations

from typing import Optional, Set

import torch.nn as nn

from .graph import ComputationGraph


def get_supported_native_ops(graph: ComputationGraph) -> Set[str]:
    try:
        from ..scientist.native.dispatch import _check_native_op_support
    except Exception:
        return set()

    try:
        op_support = _check_native_op_support([graph], native_lib=None)
    except Exception:
        return set()
    return set(op_support.get("supported") or [])


def try_compile_native_subgraph_layer(
    layer_factory: type[nn.Module], graph: ComputationGraph
) -> Optional[nn.Module]:
    try:
        from ..scientist.native.autograd import SubgraphDispatcher
    except Exception:
        return None

    supported_ops = get_supported_native_ops(graph)
    if not supported_ops:
        return None

    try:
        dispatcher = SubgraphDispatcher(graph, supported_ops)
    except Exception:
        return None
    if not dispatcher.all_native:
        return None

    layer = layer_factory(graph)
    layer._subgraph_dispatcher = dispatcher
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
