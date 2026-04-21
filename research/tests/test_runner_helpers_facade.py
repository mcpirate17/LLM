from __future__ import annotations

import research.scientist.runner._helpers as helpers_facade
from research.scientist.runner._helpers_benchmark import (
    SSELogHandler,
    run_baseline_comparison,
)
from research.scientist.runner._helpers_gate import (
    InflightState,
    clear_gpu_memory,
)
from research.scientist.runner._helpers_metrics import (
    graph_observed_routing_ops,
    graph_routing_ops,
)


def test_runner_helpers_facade_reexports_primary_symbols():
    assert helpers_facade.InflightState is InflightState
    assert helpers_facade.clear_gpu_memory is clear_gpu_memory
    assert helpers_facade.graph_routing_ops is graph_routing_ops
    assert helpers_facade.graph_observed_routing_ops is graph_observed_routing_ops
    assert helpers_facade.SSELogHandler is SSELogHandler
    assert helpers_facade.run_baseline_comparison is run_baseline_comparison


def test_runner_helpers_facade_declares_stable_export_surface():
    exported = set(helpers_facade.__all__)

    assert "InflightState" in exported
    assert "clear_gpu_memory" in exported
    assert "graph_routing_ops" in exported
    assert "graph_observed_routing_ops" in exported
    assert "SSELogHandler" in exported
    assert "run_baseline_comparison" in exported
