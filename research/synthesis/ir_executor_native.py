from __future__ import annotations

import logging
from dataclasses import dataclass

from .native_bound_graph import BoundNativeSubgraphDispatcher
from .native_bound_segments import build_bound_native_chain_segments
from .native_segments import NativeChainSegment, build_native_chain_segments

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NativeExecutionConfig:
    subgraph_dispatcher: object | None = None
    native_chain_segments: tuple[NativeChainSegment, ...] = ()
    native_chain_segment_slots: tuple[NativeChainSegment | None, ...] = ()
    native_forward_wrapper: object | None = None
    setup_reason: str | None = None
    setup_detail: str | None = None


def configure_native_execution(
    layer,
    source_graph,
    *,
    flat_ops: list[object | None],
    n_nodes: int,
    exec_plan_node_indices: tuple[int, ...],
) -> NativeExecutionConfig:
    if source_graph is None:
        return NativeExecutionConfig(setup_reason="missing_source_graph")

    try:
        from . import native_compile
        from .native_support import graph_has_bound_params
        from ..scientist.native.autograd import SubgraphDispatcher

        supported_ops = native_compile.get_supported_native_ops(source_graph)
        if not supported_ops:
            return NativeExecutionConfig(setup_reason="no_supported_native_ops")

        ir = source_graph.lower_to_ir()
        ir_node_ids = (
            ir.node_ids.tolist() if ir.node_ids is not None else list(range(n_nodes))
        )
        bound_dispatcher = BoundNativeSubgraphDispatcher(
            source_graph,
            flat_ops=flat_ops,
            ir_node_ids=ir_node_ids,
            supported_ops=supported_ops,
        )
        has_bound_params = graph_has_bound_params(source_graph)
        if bound_dispatcher.all_native:
            return NativeExecutionConfig(
                subgraph_dispatcher=bound_dispatcher,
                setup_reason="bound_subgraph_native",
            )

        if not has_bound_params:
            dispatcher = SubgraphDispatcher(source_graph, supported_ops)
            if dispatcher.all_native:
                return NativeExecutionConfig(
                    subgraph_dispatcher=dispatcher,
                    setup_reason="subgraph_native",
                )

        plain_segments = build_native_chain_segments(
            source_graph,
            ir_node_ids=ir_node_ids,
            exec_plan_node_indices=list(exec_plan_node_indices),
            supported_ops=supported_ops,
            subgraph_dispatcher_cls=SubgraphDispatcher,
        )
        bound_segments = build_bound_native_chain_segments(
            source_graph,
            flat_ops=flat_ops,
            ir_node_ids=ir_node_ids,
            exec_plan_node_indices=list(exec_plan_node_indices),
            supported_ops=supported_ops,
        )
        chain_segments = tuple(_merge_chain_segments(bound_segments, plain_segments))
        chain_segment_slots: list[NativeChainSegment | None] = [None] * len(
            exec_plan_node_indices
        )
        for segment in chain_segments:
            chain_segment_slots[segment.start_plan_index] = segment
        native_compile.attach_partial_native_wrapper(layer, source_graph)
        return NativeExecutionConfig(
            native_chain_segments=chain_segments,
            native_chain_segment_slots=tuple(chain_segment_slots),
            native_forward_wrapper=getattr(layer, "_native_forward_wrapper", None),
            setup_reason=(
                "partial_native_segments" if chain_segments else "per_op_native_wrapper"
            ),
        )
    except Exception as exc:
        logger.debug("IRExecutor native execution setup unavailable: %s", exc)
        return NativeExecutionConfig(
            setup_reason="native_setup_error",
            setup_detail=str(exc),
        )


def _merge_chain_segments(
    preferred_segments: list[NativeChainSegment],
    fallback_segments: list[NativeChainSegment],
) -> list[NativeChainSegment]:
    merged: list[NativeChainSegment] = []
    occupied: set[int] = set()
    for segment in sorted(
        [*preferred_segments, *fallback_segments],
        key=lambda seg: (
            seg.start_plan_index,
            -(seg.end_plan_index - seg.start_plan_index),
        ),
    ):
        segment_range = range(segment.start_plan_index, segment.end_plan_index + 1)
        if any(index in occupied for index in segment_range):
            continue
        merged.append(segment)
        occupied.update(segment_range)
    return merged
