from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Set

from .graph import ComputationGraph
from .primitives import PRIMITIVE_REGISTRY

if TYPE_CHECKING:
    from ..scientist.native.autograd import SubgraphDispatcher


@dataclass(slots=True)
class NativeChainSegment:
    start_plan_index: int
    end_plan_index: int
    input_ir_idx: int
    input_consume_count: int
    output_ir_idx: int
    release_ir_counts: tuple[tuple[int, int], ...]
    dispatcher: "SubgraphDispatcher"


def _unique_consumers(graph: ComputationGraph) -> Dict[int, list[int]]:
    consumers: Dict[int, list[int]] = {node_id: [] for node_id in graph.nodes}
    for node_id, node in graph.nodes.items():
        for parent_id in node.input_ids:
            if parent_id in consumers:
                consumers[parent_id].append(node_id)
    return consumers


def _is_segment_eligible_node(
    graph: ComputationGraph,
    node_id: int,
    *,
    supported_ops: set[str],
) -> bool:
    node = graph.nodes[node_id]
    if node.is_input or node.op_name not in supported_ops:
        return False
    primitive = PRIMITIVE_REGISTRY.get(node.op_name)
    if primitive is None or primitive.has_params or not node.output_shape.is_standard:
        return False

    for parent_id in node.input_ids:
        parent = graph.nodes.get(parent_id)
        if parent is None or not parent.output_shape.is_standard:
            return False
        if parent.output_shape.dim != node.output_shape.dim:
            return False
    return True


def _eligible_unary_chain_node(
    graph: ComputationGraph,
    node_id: int,
    *,
    supported_ops: set[str],
    consumers: Dict[int, list[int]],
) -> bool:
    node = graph.nodes[node_id]
    if len(node.input_ids) != 1:
        return False
    if not _is_segment_eligible_node(graph, node_id, supported_ops=supported_ops):
        return False
    if len(consumers.get(node_id, ())) > 1:
        return False
    return True


def _build_chain_subgraph(
    graph: ComputationGraph,
    chain_node_ids: Iterable[int],
) -> ComputationGraph:
    chain_node_ids = list(chain_node_ids)
    first_node = graph.nodes[chain_node_ids[0]]
    subgraph = ComputationGraph(graph.nodes[first_node.input_ids[0]].output_shape.dim)
    input_id = subgraph.add_input()
    current_id = input_id
    for node_id in chain_node_ids:
        source_node = graph.nodes[node_id]
        current_id = subgraph.add_op(
            source_node.op_name,
            [current_id],
            config=dict(source_node.config),
        )
    subgraph.set_output(current_id)
    return subgraph


def _seed_segment_nodes(
    *,
    boundary_id: int,
    consumers: Dict[int, list[int]],
    reachable_node_ids: Set[int],
    visited: Set[int],
    graph: ComputationGraph,
    supported_ops: set[str],
) -> list[int]:
    return [
        node_id
        for node_id in consumers.get(boundary_id, [])
        if node_id in reachable_node_ids
        and node_id not in visited
        and _is_segment_eligible_node(graph, node_id, supported_ops=supported_ops)
        and all(
            parent_id == boundary_id for parent_id in graph.nodes[node_id].input_ids
        )
    ]


def _collect_single_input_segment(
    *,
    boundary_id: int,
    topo: list[int],
    graph: ComputationGraph,
    consumers: Dict[int, list[int]],
    reachable_node_ids: Set[int],
    visited: Set[int],
    supported_ops: set[str],
) -> list[int]:
    segment_node_ids = _seed_segment_nodes(
        boundary_id=boundary_id,
        consumers=consumers,
        reachable_node_ids=reachable_node_ids,
        visited=visited,
        graph=graph,
        supported_ops=supported_ops,
    )
    if not segment_node_ids:
        return []

    segment_set = set(segment_node_ids)
    changed = True
    while changed:
        changed = False
        for node_id in topo:
            if (
                node_id in segment_set
                or node_id not in reachable_node_ids
                or node_id in visited
                or not _is_segment_eligible_node(
                    graph, node_id, supported_ops=supported_ops
                )
            ):
                continue
            parents = graph.nodes[node_id].input_ids
            if parents and all(
                parent_id == boundary_id or parent_id in segment_set
                for parent_id in parents
            ):
                segment_node_ids.append(node_id)
                segment_set.add(node_id)
                changed = True
    return segment_node_ids


def _segment_sink_nodes(
    *,
    segment_set: Set[int],
    consumers: Dict[int, list[int]],
    output_node_id: int | None,
) -> list[int]:
    sinks: list[int] = []
    for node_id in segment_set:
        node_consumers = consumers.get(node_id, [])
        if output_node_id == node_id or any(
            consumer_id not in segment_set for consumer_id in node_consumers
        ):
            sinks.append(node_id)
    return sinks


def _build_single_input_subgraph(
    graph: ComputationGraph,
    *,
    boundary_id: int,
    segment_node_ids: Iterable[int],
    sink_node_id: int,
) -> ComputationGraph:
    subgraph = ComputationGraph(graph.nodes[boundary_id].output_shape.dim)
    input_id = subgraph.add_input()
    subgraph_node_ids = {boundary_id: input_id}
    for node_id in segment_node_ids:
        source_node = graph.nodes[node_id]
        subgraph_node_ids[node_id] = subgraph.add_op(
            source_node.op_name,
            [subgraph_node_ids[parent_id] for parent_id in source_node.input_ids],
            config=dict(source_node.config),
        )
    subgraph.set_output(subgraph_node_ids[sink_node_id])
    return subgraph


def _build_release_counts(
    *,
    segment_node_ids: Iterable[int],
    segment_set: Set[int],
    consumers: Dict[int, list[int]],
    node_id_to_ir_idx: Dict[int, int],
    sink_node_id: int,
) -> tuple[tuple[int, int], ...]:
    release_counts: list[tuple[int, int]] = []
    for node_id in segment_node_ids:
        if node_id == sink_node_id:
            continue
        internal_consume_count = sum(
            1 for child_id in consumers.get(node_id, []) if child_id in segment_set
        )
        if internal_consume_count > 0:
            release_counts.append((node_id_to_ir_idx[node_id], internal_consume_count))
    return tuple(release_counts)


def _single_input_segment_from_boundary(
    *,
    boundary_id: int,
    topo: list[int],
    graph: ComputationGraph,
    consumers: Dict[int, list[int]],
    reachable_node_ids: Set[int],
    node_id_to_ir_idx: Dict[int, int],
    node_id_to_plan_idx: Dict[int, int],
    visited: Set[int],
    supported_ops: set[str],
    subgraph_dispatcher_cls,
) -> NativeChainSegment | None:
    segment_node_ids = _collect_single_input_segment(
        boundary_id=boundary_id,
        topo=topo,
        graph=graph,
        consumers=consumers,
        reachable_node_ids=reachable_node_ids,
        visited=visited,
        supported_ops=supported_ops,
    )
    if len(segment_node_ids) < 2:
        return None

    segment_set = set(segment_node_ids)
    sink_node_ids = _segment_sink_nodes(
        segment_set=segment_set,
        consumers=consumers,
        output_node_id=graph._output_node_id,
    )
    if len(sink_node_ids) != 1:
        return None
    sink_node_id = sink_node_ids[0]

    plan_indices = sorted(node_id_to_plan_idx[node_id] for node_id in segment_node_ids)
    if plan_indices != list(range(plan_indices[0], plan_indices[-1] + 1)):
        return None

    dispatcher = subgraph_dispatcher_cls(
        _build_single_input_subgraph(
            graph,
            boundary_id=boundary_id,
            segment_node_ids=segment_node_ids,
            sink_node_id=sink_node_id,
        ),
        supported_ops,
    )
    if not dispatcher.all_native:
        return None

    for node_id in segment_node_ids:
        visited.add(node_id)

    input_consume_count = sum(
        1
        for node_id in segment_node_ids
        for parent_id in graph.nodes[node_id].input_ids
        if parent_id == boundary_id
    )
    return NativeChainSegment(
        start_plan_index=plan_indices[0],
        end_plan_index=plan_indices[-1],
        input_ir_idx=node_id_to_ir_idx[boundary_id],
        input_consume_count=input_consume_count,
        output_ir_idx=node_id_to_ir_idx[sink_node_id],
        release_ir_counts=_build_release_counts(
            segment_node_ids=segment_node_ids,
            segment_set=segment_set,
            consumers=consumers,
            node_id_to_ir_idx=node_id_to_ir_idx,
            sink_node_id=sink_node_id,
        ),
        dispatcher=dispatcher,
    )


def _unary_chain_segment_from_node(
    *,
    node_id: int,
    graph: ComputationGraph,
    consumers: Dict[int, list[int]],
    reachable_node_ids: Set[int],
    node_id_to_ir_idx: Dict[int, int],
    node_id_to_plan_idx: Dict[int, int],
    visited: Set[int],
    supported_ops: set[str],
    subgraph_dispatcher_cls,
) -> NativeChainSegment | None:
    if not _eligible_unary_chain_node(
        graph, node_id, supported_ops=supported_ops, consumers=consumers
    ):
        return None

    parent_id = graph.nodes[node_id].input_ids[0]
    if parent_id in reachable_node_ids and _eligible_unary_chain_node(
        graph, parent_id, supported_ops=supported_ops, consumers=consumers
    ):
        return None

    chain_node_ids = [node_id]
    current_id = node_id
    while True:
        next_ids = consumers.get(current_id, [])
        if len(next_ids) != 1:
            break
        next_id = next_ids[0]
        if next_id in visited or next_id not in reachable_node_ids:
            break
        next_node = graph.nodes[next_id]
        if len(next_node.input_ids) != 1 or next_node.input_ids[0] != current_id:
            break
        if not _eligible_unary_chain_node(
            graph, next_id, supported_ops=supported_ops, consumers=consumers
        ):
            break
        chain_node_ids.append(next_id)
        current_id = next_id

    if len(chain_node_ids) < 2:
        return None

    plan_indices = [
        node_id_to_plan_idx[current_node_id] for current_node_id in chain_node_ids
    ]
    if plan_indices != list(range(plan_indices[0], plan_indices[-1] + 1)):
        return None

    dispatcher = subgraph_dispatcher_cls(
        _build_chain_subgraph(graph, chain_node_ids),
        supported_ops,
    )
    if not dispatcher.all_native:
        return None

    for current_node_id in chain_node_ids:
        visited.add(current_node_id)

    return NativeChainSegment(
        start_plan_index=plan_indices[0],
        end_plan_index=plan_indices[-1],
        input_ir_idx=node_id_to_ir_idx[parent_id],
        input_consume_count=1,
        output_ir_idx=node_id_to_ir_idx[chain_node_ids[-1]],
        release_ir_counts=tuple(
            (node_id_to_ir_idx[current_node_id], 1)
            for current_node_id in chain_node_ids[:-1]
        ),
        dispatcher=dispatcher,
    )


def build_native_chain_segments(
    graph: ComputationGraph,
    *,
    ir_node_ids: List[int],
    exec_plan_node_indices: List[int],
    supported_ops: set[str],
    subgraph_dispatcher_cls,
) -> list[NativeChainSegment]:
    if not supported_ops:
        return []

    consumers = _unique_consumers(graph)
    topo = graph.topological_order()
    reachable_node_ids = set(ir_node_ids)
    node_id_to_ir_idx = {int(node_id): idx for idx, node_id in enumerate(ir_node_ids)}
    node_id_to_plan_idx = {
        int(ir_node_ids[node_idx]): plan_idx
        for plan_idx, node_idx in enumerate(exec_plan_node_indices)
    }
    visited: set[int] = set()
    segments: list[NativeChainSegment] = []

    for node_id in topo:
        if node_id in visited or node_id not in reachable_node_ids:
            continue

        segment = _single_input_segment_from_boundary(
            boundary_id=node_id,
            topo=topo,
            graph=graph,
            consumers=consumers,
            reachable_node_ids=reachable_node_ids,
            node_id_to_ir_idx=node_id_to_ir_idx,
            node_id_to_plan_idx=node_id_to_plan_idx,
            visited=visited,
            supported_ops=supported_ops,
            subgraph_dispatcher_cls=subgraph_dispatcher_cls,
        )
        if segment is not None:
            segments.append(segment)
            continue

        segment = _unary_chain_segment_from_node(
            node_id=node_id,
            graph=graph,
            consumers=consumers,
            reachable_node_ids=reachable_node_ids,
            node_id_to_ir_idx=node_id_to_ir_idx,
            node_id_to_plan_idx=node_id_to_plan_idx,
            visited=visited,
            supported_ops=supported_ops,
            subgraph_dispatcher_cls=subgraph_dispatcher_cls,
        )
        if segment is not None:
            segments.append(segment)

    return segments
