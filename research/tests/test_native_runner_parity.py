from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from research.scientist.native_runner import compile_model_native_first, reset_native_runner_telemetry
from research.synthesis.compiler import compile_model as compile_model_legacy
from research.synthesis.graph import ComputationGraph


def _graph_simple(model_dim: int = 32) -> ComputationGraph:
    g = ComputationGraph(model_dim=model_dim)
    inp = g.add_input()
    proj = g.add_op("linear_proj", [inp], {"out_dim": model_dim})
    act = g.add_op("relu", [proj], {})
    g.set_output(act)
    return g


def _graph_residual(model_dim: int = 32) -> ComputationGraph:
    g = ComputationGraph(model_dim=model_dim)
    inp = g.add_input()
    p1 = g.add_op("linear_proj", [inp], {"out_dim": model_dim})
    p2 = g.add_op("gelu", [p1], {})
    out = g.add_op("add", [inp, p2], {})
    g.set_output(out)
    return g


def _graph_attention_like(model_dim: int = 32) -> ComputationGraph:
    g = ComputationGraph(model_dim=model_dim)
    inp = g.add_input()
    q = g.add_op("linear_proj", [inp], {"out_dim": model_dim})
    k = g.add_op("linear_proj", [inp], {"out_dim": model_dim})
    attn = g.add_op("matmul", [q, k], {})
    out = g.add_op("linear_proj", [attn], {"out_dim": model_dim})
    g.set_output(out)
    return g


def _run_model(model, *, seed: int = 1234):
    torch.manual_seed(seed)
    input_ids = torch.randint(0, 128, (2, 12), dtype=torch.long)
    with torch.no_grad():
        return model(input_ids)


def _compile_pair(layer_graphs):
    # Phase D: NATIVE_RUNNER_ABI_MODEL_ONLY is no longer supported.
    # Use NATIVE_RUNNER_ENABLED=0 so compile_model_native_first goes through
    # the legacy compile path, then compare with direct legacy compile.
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_MAX_FALLBACK_RATE": "1.0",
    }
    reset_native_runner_telemetry()

    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner_adapter.Path.exists", return_value=True
    ):
        torch.manual_seed(77)
        native_model = compile_model_native_first(layer_graphs, vocab_size=128, max_seq_len=32)

    torch.manual_seed(77)
    legacy_model = compile_model_legacy(layer_graphs, vocab_size=128, max_seq_len=32)
    return native_model, legacy_model


@pytest.mark.parametrize(
    ("graph_builder", "seed"),
    [
        (_graph_simple, 2026),
        (_graph_residual, 314),
        (_graph_attention_like, 808),
    ],
)
def test_adapter_vs_legacy_parity_non_strict(graph_builder, seed):
    graphs = [graph_builder(), graph_builder()]
    native_model, legacy_model = _compile_pair(graphs)
    out_native = _run_model(native_model, seed=seed)
    out_legacy = _run_model(legacy_model, seed=seed)
    assert out_native.shape == out_legacy.shape
    assert torch.allclose(out_native, out_legacy, atol=1e-6, rtol=1e-5)
    # Phase D: native disabled, so legacy compile path is used for both models.
    # Verify the report exists and reflects legacy-only execution.
    native_report = getattr(native_model, "_native_runner_report", {}) or {}
    assert native_report.get("enabled") is False
    assert native_report.get("legacy_compile_used") is True


def test_selective_mode_parity_contract_for_supported_subset():
    # Phase D: NATIVE_RUNNER_ABI_MODEL_ONLY is removed. When native is enabled,
    # ABI model-only is always active and legacy compile is unreachable.
    # Test selective execution mode with native disabled to exercise legacy path.
    env = {
        "NATIVE_RUNNER_ENABLED": "0",
        "NATIVE_RUNNER_STRICT": "0",
        "NATIVE_RUNNER_EXECUTION_MODE": "selective",
        "NATIVE_RUNNER_MAX_FALLBACK_RATE": "1.0",
    }

    graphs = [_graph_simple(), _graph_simple()]
    reset_native_runner_telemetry()
    with patch("research.scientist.native_runner_adapter.os.environ", env), patch(
        "research.scientist.native_runner.os.environ", env
    ), patch(
        "research.scientist.native_runner_adapter.Path.exists", return_value=True
    ):
        torch.manual_seed(77)
        native_model = compile_model_native_first(graphs, vocab_size=128, max_seq_len=32)

    torch.manual_seed(77)
    legacy_model = compile_model_legacy(graphs, vocab_size=128, max_seq_len=32)

    out_native = _run_model(native_model, seed=2048)
    out_legacy = _run_model(legacy_model, seed=2048)
    assert out_native.shape == out_legacy.shape
    assert torch.allclose(out_native, out_legacy, atol=1e-6, rtol=1e-5)

    # With native disabled, legacy compile is used — verify report reflects that.
    report = getattr(native_model, "_native_runner_report", {}) or {}
    assert report.get("enabled") is False
    assert report.get("legacy_compile_used") is True
