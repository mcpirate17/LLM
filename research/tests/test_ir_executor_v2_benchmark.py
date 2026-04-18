from __future__ import annotations

import statistics
import time

import torch

from research.synthesis.graph import ComputationGraph
from research.synthesis.ir_executor import IRExecutor
from research.synthesis.ir_executor_v2 import IRExecutorV2


def _make_relu_chain_graph(model_dim: int = 256, depth: int = 32) -> ComputationGraph:
    graph = ComputationGraph(model_dim)
    inp = graph.add_input()
    cur = inp
    for _ in range(depth):
        cur = graph.add_op("relu", [cur])
    graph.set_output(cur)
    return graph


def _make_bound_linear_graph(model_dim: int = 128) -> ComputationGraph:
    graph = ComputationGraph(model_dim)
    inp = graph.add_input()
    hidden = graph.add_op("linear_proj", [inp], {"out_dim": model_dim})
    activated = graph.add_op("relu", [hidden])
    out = graph.add_op("linear_proj", [activated], {"out_dim": model_dim})
    graph.set_output(out)
    return graph


def _median_ctor_ms(factory, *, iterations: int = 120) -> float:
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        factory()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1_000_000.0)
    return statistics.median(samples)


def _median_run_ms(module, x: torch.Tensor, *, warmup: int = 20, iterations: int = 80):
    with torch.no_grad():
        for _ in range(warmup):
            module(x)
        samples = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            module(x)
            t1 = time.perf_counter_ns()
            samples.append((t1 - t0) / 1_000_000.0)
    return statistics.median(samples)


def test_ir_executor_v2_benchmark_smoke():
    relu_graph = _make_relu_chain_graph()
    relu_ir = relu_graph.lower_to_ir()
    relu_ctor_default = _median_ctor_ms(
        lambda: IRExecutor(relu_ir, source_graph=relu_graph)
    )
    relu_ctor_v2 = _median_ctor_ms(lambda: IRExecutorV2(relu_ir, source_graph=relu_graph))

    bound_graph = _make_bound_linear_graph()
    bound_ir = bound_graph.lower_to_ir()
    bound_ctor_default = _median_ctor_ms(
        lambda: IRExecutor(bound_ir, source_graph=bound_graph)
    )
    bound_ctor_v2 = _median_ctor_ms(
        lambda: IRExecutorV2(bound_ir, source_graph=bound_graph)
    )

    x_relu = torch.randn(8, 64, relu_graph.model_dim)
    x_bound = torch.randn(16, 32, bound_graph.model_dim)
    relu_default = IRExecutor(relu_ir, source_graph=relu_graph).eval()
    relu_v2 = IRExecutorV2(relu_ir, source_graph=relu_graph).eval()
    bound_default = IRExecutor(bound_ir, source_graph=bound_graph).eval()
    bound_v2 = IRExecutorV2(bound_ir, source_graph=bound_graph).eval()

    relu_default_ms = _median_run_ms(relu_default, x_relu)
    relu_v2_ms = _median_run_ms(relu_v2, x_relu)
    bound_default_ms = _median_run_ms(bound_default, x_bound)
    bound_v2_ms = _median_run_ms(bound_v2, x_bound)

    print(
        "\nIRExecutorV2 benchmark"
        f"\n  relu_ctor_default_ms={relu_ctor_default:.4f}"
        f"\n  relu_ctor_v2_ms={relu_ctor_v2:.4f}"
        f"\n  relu_ctor_speedup={relu_ctor_default / relu_ctor_v2:.3f}x"
        f"\n  relu_run_default_ms={relu_default_ms:.4f}"
        f"\n  relu_run_v2_ms={relu_v2_ms:.4f}"
        f"\n  bound_ctor_default_ms={bound_ctor_default:.4f}"
        f"\n  bound_ctor_v2_ms={bound_ctor_v2:.4f}"
        f"\n  bound_ctor_speedup={bound_ctor_default / bound_ctor_v2:.3f}x"
        f"\n  bound_run_default_ms={bound_default_ms:.4f}"
        f"\n  bound_run_v2_ms={bound_v2_ms:.4f}"
    )

    assert relu_ctor_v2 > 0.0
    assert bound_ctor_v2 > 0.0
    assert relu_v2_ms > 0.0
    assert bound_v2_ms > 0.0
    assert relu_ctor_v2 <= relu_ctor_default
    assert bound_ctor_v2 <= (bound_ctor_default * 1.25)
