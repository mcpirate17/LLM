from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List

from .graph import ComputationGraph
from .primitives import PRIMITIVE_REGISTRY

if TYPE_CHECKING:
    from ..scientist.native.autograd import SubgraphDispatcher


@dataclass(slots=True)
class NativeChainSegment:
    start_plan_index: int
    end_plan_index: int
    input_ir_idx: int
    output_ir_idx: int
    release_ir_indices: tuple[int, ...]
    dispatcher: "SubgraphDispatcher"


def _unique_consumers(graph: ComputationGraph) -> Dict[int, list[int]]:
    consumers: Dict[int, list[int]] = {node_id: [] for node_id in graph.nodes}
    for node_id, node in graph.nodes.items():
        for parent_id in node.input_ids:
            if parent_id in consumers:
                consumers[parent_id].append(node_id)
    return consumers


def _eligible_unary_chain_node(
    graph: ComputationGraph,
    node_id: int,
    *,
    supported_ops: set[str],
    consumers: Dict[int, list[int]],
) -> bool:
    node = graph.nodes[node_id]
    if node.is_input or node.op_name not in supported_ops or len(node.input_ids) != 1:
        return False
    primitive = PRIMITIVE_REGISTRY.get(node.op_name)
    if primitive is None or primitive.has_params:
        return False
    if not node.output_shape.is_standard:
        return False
    parent = graph.nodes.get(node.input_ids[0])
    if parent is None or not parent.output_shape.is_standard:
        return False
    if parent.output_shape.dim != node.output_shape.dim:
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
    input_dim = graph.nodes[first_node.input_ids[0]].output_shape.dim
    subgraph = ComputationGraph(input_dim)
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
        if not _eligible_unary_chain_node(
            graph, node_id, supported_ops=supported_ops, consumers=consumers
        ):
            continue

        parent_id = graph.nodes[node_id].input_ids[0]
        if parent_id in reachable_node_ids and _eligible_unary_chain_node(
            graph, parent_id, supported_ops=supported_ops, consumers=consumers
        ):
            continue

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
            continue

        plan_indices = [node_id_to_plan_idx[node_id] for node_id in chain_node_ids]
        if plan_indices != list(range(plan_indices[0], plan_indices[-1] + 1)):
            continue

        dispatcher = subgraph_dispatcher_cls(
            _build_chain_subgraph(graph, chain_node_ids),
            supported_ops,
        )
        if not dispatcher.all_native:
            continue

        for current_id in chain_node_ids:
            visited.add(current_id)

        segments.append(
            NativeChainSegment(
                start_plan_index=plan_indices[0],
                end_plan_index=plan_indices[-1],
                input_ir_idx=node_id_to_ir_idx[
                    graph.nodes[chain_node_ids[0]].input_ids[0]
                ],
                output_ir_idx=node_id_to_ir_idx[chain_node_ids[-1]],
                release_ir_indices=tuple(
                    node_id_to_ir_idx[node_id] for node_id in chain_node_ids[:-1]
                ),
                dispatcher=dispatcher,
            )
        )

    return segments
