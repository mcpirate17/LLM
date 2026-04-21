from __future__ import annotations

from research.scientist.runner._helpers_metrics import (
    graph_observed_routing_ops,
    graph_routing_ops,
)
from research.scientist.runner._routing_ops import (
    ROUTING_FAST_LANE_OPS,
    ROUTING_OBSERVED_OPS,
)


def test_shared_routing_policy_sets_remain_consistent():
    assert ROUTING_FAST_LANE_OPS <= ROUTING_OBSERVED_OPS
    assert "moe_topk" in ROUTING_FAST_LANE_OPS
    assert "hybrid_sparse_router" in ROUTING_OBSERVED_OPS


def test_graph_routing_ops_uses_shared_fast_lane_policy():
    graph = {
        "nodes": [
            {"op_name": "signal_conditioned_compression"},
            {"op_name": "relu"},
            {"op_name": "route_recursion"},
        ]
    }

    assert graph_routing_ops(graph) == ["signal_conditioned_compression"]
    assert graph_observed_routing_ops(graph) == [
        "route_recursion",
        "signal_conditioned_compression",
    ]
