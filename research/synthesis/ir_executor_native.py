from __future__ import annotations

import logging
from dataclasses import dataclass

from .native_segments import NativeChainSegment, build_native_chain_segments

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NativeExecutionConfig:
    subgraph_dispatcher: object | None = None
    native_chain_segments: tuple[NativeChainSegment, ...] = ()
    native_chain_segments_by_plan_index: dict[int, NativeChainSegment] | None = None
    native_forward_wrapper: object | None = None


def configure_native_execution(
    layer,
    source_graph,
    *,
    op_codes_list: list[int],
    exec_plan_node_indices: tuple[int, ...],
) -> NativeExecutionConfig:
    if source_graph is None:
        return NativeExecutionConfig(native_chain_segments_by_plan_index={})

    try:
        from . import native_compile
        from ..scientist.native.autograd import SubgraphDispatcher

        supported_ops = native_compile.get_supported_native_ops(source_graph)
        if not supported_ops:
            return NativeExecutionConfig(native_chain_segments_by_plan_index={})

        dispatcher = SubgraphDispatcher(source_graph, supported_ops)
        if dispatcher.all_native:
            return NativeExecutionConfig(
                subgraph_dispatcher=dispatcher,
                native_chain_segments_by_plan_index={},
            )

        ir = source_graph.lower_to_ir()
        ir_node_ids = (
            ir.node_ids.tolist()
            if ir.node_ids is not None
            else list(range(len(op_codes_list)))
        )
        chain_segments = tuple(
            build_native_chain_segments(
                source_graph,
                ir_node_ids=ir_node_ids,
                exec_plan_node_indices=list(exec_plan_node_indices),
                supported_ops=supported_ops,
                subgraph_dispatcher_cls=SubgraphDispatcher,
            )
        )
        native_compile.attach_partial_native_wrapper(layer, source_graph)
        return NativeExecutionConfig(
            native_chain_segments=chain_segments,
            native_chain_segments_by_plan_index={
                segment.start_plan_index: segment for segment in chain_segments
            },
            native_forward_wrapper=getattr(layer, "_native_forward_wrapper", None),
        )
    except Exception as exc:
        logger.debug("IRExecutor native execution setup unavailable: %s", exc)
        return NativeExecutionConfig(native_chain_segments_by_plan_index={})
