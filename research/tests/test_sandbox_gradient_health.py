from __future__ import annotations

import pytest
import torch

from research.eval import sandbox


pytestmark = pytest.mark.unit


def _model_with_grads() -> torch.nn.Module:
    model = torch.nn.Linear(3, 2)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    return model


def test_gradient_health_uses_native_fused_stats(monkeypatch):
    captured = {}

    class _FakeRunnerNative:
        def grad_stats_fused(self, grads, names):
            captured["grad_shapes"] = [tuple(grad.shape) for grad in grads]
            captured["names"] = names
            return {
                "total_norm": 3.5,
                "has_nonfinite": False,
                "has_zero": False,
            }

    monkeypatch.setattr(
        "research.eval._runner_native.load_runner_native",
        lambda: _FakeRunnerNative(),
    )

    assert sandbox._gradient_health(_model_with_grads()) == (3.5, False, False, 2)
    assert captured["grad_shapes"] == [(2, 3), (2,)]
    assert captured["names"] == ["p0", "p1"]


def test_gradient_health_propagates_native_failure(monkeypatch):
    def _raise_native():
        raise RuntimeError("native grad stats unavailable")

    monkeypatch.setattr(
        "research.eval._runner_native.load_runner_native", _raise_native
    )

    with pytest.raises(RuntimeError, match="native grad stats unavailable"):
        sandbox._gradient_health(_model_with_grads())
