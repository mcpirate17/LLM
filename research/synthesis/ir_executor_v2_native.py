from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .graph import ComputationGraph
from .native_compile import get_supported_native_ops
from .native_support import graph_has_bound_params
from .native_bound_graph import BoundNativeSubgraphDispatcher
from ..scientist.native.autograd import SubgraphDispatcher

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IRExecutorV2NativeConfig:
    dispatcher: object | None = None
    setup_reason: str | None = None
    setup_detail: str | None = None


def configure_ir_executor_v2_native(
    source_graph: ComputationGraph | None,
    *,
    flat_ops: list[object | None] | None = None,
    ir_node_ids: list[int] | None = None,
) -> IRExecutorV2NativeConfig:
    if source_graph is None:
        return IRExecutorV2NativeConfig(setup_reason="missing_source_graph")
    try:
        supported_ops = get_supported_native_ops(source_graph)
        if not supported_ops:
            return IRExecutorV2NativeConfig(setup_reason="no_supported_native_ops")
        if graph_has_bound_params(source_graph):
            if flat_ops is None or ir_node_ids is None:
                return IRExecutorV2NativeConfig(
                    setup_reason="missing_bound_native_inputs"
                )
            dispatcher = BoundNativeSubgraphDispatcher(
                source_graph,
                flat_ops=flat_ops,
                ir_node_ids=ir_node_ids,
                supported_ops=supported_ops,
            )
            if not dispatcher.all_native:
                return IRExecutorV2NativeConfig(
                    dispatcher=dispatcher,
                    setup_reason="graph_not_fully_bound_native",
                )
            return IRExecutorV2NativeConfig(
                dispatcher=dispatcher,
                setup_reason="bound_subgraph_v2",
            )
        dispatcher = SubgraphDispatcher(source_graph, supported_ops)
        if not dispatcher.all_native:
            return IRExecutorV2NativeConfig(
                dispatcher=dispatcher,
                setup_reason="graph_not_fully_native",
            )
        return IRExecutorV2NativeConfig(
            dispatcher=dispatcher,
            setup_reason="native_subgraph_v2",
        )
    except Exception as exc:
        logger.debug("IRExecutorV2 native setup unavailable: %s", exc)
        return IRExecutorV2NativeConfig(
            setup_reason="native_setup_error",
            setup_detail=str(exc),
        )
