from __future__ import annotations

import random
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from research.scientist.native_runner import compile_model_native_first, reset_native_runner_telemetry


@dataclass
class SelectiveCanaryBenchmarkResult:
    iterations: int
    seed: int
    probe_avg_latency_ms: float
    selective_avg_latency_ms: float
    latency_delta_ms: float
    latency_ratio: float
    probe_execution_paths: Dict[str, int]
    selective_execution_paths: Dict[str, int]
    selective_applied_layers_avg: float


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _make_fake_graph(op_names: List[str]) -> Any:
    nodes = {}
    for i, name in enumerate(op_names):
        nodes[f"n{i}"] = SimpleNamespace(op_name=name)
    return SimpleNamespace(nodes=nodes)


def run_selective_canary_latency_benchmark(
    iterations: int = 25,
    seed: int = 1337,
) -> SelectiveCanaryBenchmarkResult:
    """Benchmark probe vs selective-layer compile latency under fixed seeds.

    This benchmark is deterministic and lightweight by patching heavyweight compile
    dependencies, while preserving native-runner mode selection and telemetry paths.
    """
    iterations = max(1, int(iterations))
    seed = int(seed)

    # Phase D: NATIVE_RUNNER_ABI_MODEL_ONLY removed. Use NATIVE_RUNNER_ENABLED=0
    # since this benchmark exercises legacy compile paths.
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_EXECUTION_MODE": "probe",
        "NATIVE_RUNNER_SELECTIVE_LAYER_EXEC": "0",
    }

    class _DummyModel:
        def __init__(self):
            self.layers = [object()]
            self.model_dim = 32

    class _FakeWorkflowModule:
        def __call__(self, inputs):
            return {"y": next(iter(inputs.values()))}

    class _FakeNativeLib:
        def __init__(self):
            self.nk_is_registered = MagicMock(return_value=1)

        @staticmethod
        def aria_relu_f32(x, y, n):
            for i in range(int(n)):
                y[i] = x[i] if x[i] > 0 else 0.0

        @staticmethod
        def aria_add_f32(a, b, y, n):
            for i in range(int(n)):
                y[i] = a[i] + b[i]

    probe_payload: Dict[str, Any] = {
        "attempted": True,
        "succeeded": True,
        "parity_ok": True,
        "reason": "ok",
    }

    replacement_payload: Dict[str, Any] = {
        "attempted": True,
        "compiled_layers": 1,
        "failed_layers": 0,
        "total_layers": 1,
        "errors": [],
        "replacements": {
            0: {"module": _FakeWorkflowModule(), "input_node_id": "input_0", "workflow_id": "wf_0"},
        },
    }

    fake_graphs = [_make_fake_graph(["relu", "add", "matmul"])]

    def _run_mode(mode: str, *, layer_exec: bool) -> Dict[str, Any]:
        env["NATIVE_RUNNER_EXECUTION_MODE"] = str(mode)
        env["NATIVE_RUNNER_SELECTIVE_LAYER_EXEC"] = "1" if layer_exec else "0"
        reset_native_runner_telemetry()
        latencies_ms: List[float] = []
        execution_paths: Dict[str, int] = {}
        applied_layers: List[int] = []

        for idx in range(iterations):
            _set_seed(seed + idx)
            start = time.perf_counter()
            model = compile_model_native_first(fake_graphs, vocab_size=256, max_seq_len=16)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(float(elapsed_ms))

            report = getattr(model, "_native_runner_report", {}) or {}
            execution_path = str(report.get("execution_path") or "unknown")
            execution_paths[execution_path] = int(execution_paths.get(execution_path) or 0) + 1

            layer_build = ((report.get("selective_execution") or {}).get("layer_build") or {})
            applied_layers.append(int(layer_build.get("applied_layers") or 0))

        avg_latency = sum(latencies_ms) / float(len(latencies_ms))
        avg_applied_layers = sum(applied_layers) / float(len(applied_layers)) if applied_layers else 0.0
        return {
            "avg_latency_ms": float(avg_latency),
            "execution_paths": execution_paths,
            "avg_applied_layers": float(avg_applied_layers),
        }

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner_adapter.Path.exists", return_value=True
    ), patch(
        "research.scientist.native_runner._try_load_native_lib", return_value=_FakeNativeLib()
    ), patch(
        "research.scientist.native_runner.try_designer_runtime_probe", return_value=probe_payload
    ), patch(
        "research.scientist.native_runner._legacy_compile_model", side_effect=lambda *args, **kwargs: _DummyModel()
    ), patch(
        "research.scientist.native_runner.build_designer_layer_modules", return_value=replacement_payload
    ), patch(
        "research.scientist.native_runner._validate_designer_layer_adapter_contract", return_value=None
    ):
        probe = _run_mode("probe", layer_exec=False)
        selective = _run_mode("selective", layer_exec=True)

    probe_avg = float(probe["avg_latency_ms"])
    selective_avg = float(selective["avg_latency_ms"])
    delta = selective_avg - probe_avg
    ratio = (selective_avg / probe_avg) if probe_avg > 0 else 0.0

    return SelectiveCanaryBenchmarkResult(
        iterations=iterations,
        seed=seed,
        probe_avg_latency_ms=probe_avg,
        selective_avg_latency_ms=selective_avg,
        latency_delta_ms=float(delta),
        latency_ratio=float(ratio),
        probe_execution_paths=dict(probe["execution_paths"]),
        selective_execution_paths=dict(selective["execution_paths"]),
        selective_applied_layers_avg=float(selective["avg_applied_layers"]),
    )
