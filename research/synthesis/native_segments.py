from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

from .graph import ComputationGraph

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
