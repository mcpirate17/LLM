from __future__ import annotations

import torch
import pytest

from research.synthesis.graph import ComputationGraph
from research.synthesis.ir_executor_v2 import IRExecutorV2


def _make_relu_graph(model_dim: int = 8) -> ComputationGraph:
    graph = ComputationGraph(model_dim)
    inp = graph.add_input()
    out = graph.add_op("relu", [inp])
    graph.set_output(out)
    return graph


def _make_linear_graph(model_dim: int = 8) -> ComputationGraph:
    graph = ComputationGraph(model_dim)
    inp = graph.add_input()
    mid = graph.add_op("linear_proj", [inp], {"out_dim": model_dim})
    out = graph.add_op("relu", [mid])
    graph.set_output(out)
    return graph


def _make_moe_graph(model_dim: int = 8) -> ComputationGraph:
    graph = ComputationGraph(model_dim)
    inp = graph.add_input()
    out = graph.add_op("moe_2expert", [inp])
    graph.set_output(out)
    return graph


def test_ir_executor_v2_uses_native_dispatch_when_available(monkeypatch):
    import research.synthesis.ir_executor_v2_native as native_cfg

    class FakeDispatcher:
        def __init__(self):
            self.stats = {"last_refusal_reason": None}

        def try_dispatch(self, x):
            return x + 3

    monkeypatch.setattr(
        native_cfg,
        "configure_ir_executor_v2_native",
        lambda graph: native_cfg.IRExecutorV2NativeConfig(
            dispatcher=FakeDispatcher(),
            setup_reason="native_subgraph_v2",
        ),
    )

    graph = _make_relu_graph()
    executor = IRExecutorV2(graph.lower_to_ir(), source_graph=graph)
    x = torch.zeros(1, 3, 8)
    out = executor(x)

    assert torch.equal(out, x + 3)
    assert executor.execution_stats["last_execution_path"] == "v2_native_subgraph"
    assert executor.execution_stats["plan_initialized"] is False
    assert executor.execution_stats["v2_native_dispatches"] == 1


def test_ir_executor_v2_falls_back_to_ir_executor(monkeypatch):
    import research.synthesis.ir_executor_v2_native as native_cfg

    monkeypatch.setattr(
        native_cfg,
        "configure_ir_executor_v2_native",
        lambda graph: native_cfg.IRExecutorV2NativeConfig(
            dispatcher=None,
            setup_reason="graph_not_fully_native",
        ),
    )

    graph = _make_relu_graph()
    executor = IRExecutorV2(graph.lower_to_ir(), source_graph=graph)
    x = torch.tensor([[[-1.0] * 8, [2.0] * 8]], dtype=torch.float32)
    out = executor(x)

    assert torch.equal(out, torch.relu(x))
    assert executor.execution_stats["last_execution_path"] == "v2_fallback"
    assert executor.execution_stats["plan_initialized"] is True
    assert executor.execution_stats["v2_fallback_dispatches"] == 1


def test_ir_executor_v2_uses_bound_native_dispatch_for_param_graph():
    graph = _make_linear_graph()
    executor = IRExecutorV2(graph.lower_to_ir(), source_graph=graph)
    x = torch.randn(2, 3, 8)
    out = executor(x)

    assert out.shape == x.shape
    assert executor.execution_stats["last_execution_path"] == "v2_native_subgraph"
    assert executor.execution_stats["native_setup_reason"] == "bound_subgraph_v2"
    assert executor.execution_stats["plan_initialized"] is True
    assert executor.execution_stats["v2_native_dispatches"] == 1


def test_ir_executor_v2_defers_fallback_init_for_non_param_native_graph():
    graph = _make_relu_graph()
    executor = IRExecutorV2(graph.lower_to_ir(), source_graph=graph)

    assert executor.execution_stats["plan_initialized"] is False

    x = torch.randn(1, 2, 8)
    _ = executor(x)

    assert executor.execution_stats["last_execution_path"] == "v2_native_subgraph"
    assert executor.execution_stats["plan_initialized"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ir_executor_v2_to_cuda_moves_lazy_fallback_ops():
    graph = _make_moe_graph()
    executor = IRExecutorV2(graph.lower_to_ir(), source_graph=graph).to("cuda")

    x = torch.randn(2, 3, 8, device="cuda")
    out = executor(x)

    assert out.device.type == "cuda"
    assert executor.execution_stats["plan_initialized"] is True
    assert all(param.device.type == "cuda" for param in executor.parameters())
