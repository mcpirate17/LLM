from __future__ import annotations

import torch

from aria_designer.components import base


def test_make_unary_handler_rejects_nonfinite_native_result(monkeypatch) -> None:
    def fake_native(op_name, x):
        return torch.full_like(x, float("nan"))

    monkeypatch.setattr(base, "_try_native", fake_native)
    handler = base.make_unary_handler(
        lambda x: torch.sqrt(torch.clamp(x, min=0.0)),
        native_op_name="sqrt",
    )()
    x = torch.tensor([[-1.0, 0.0, 4.0]])
    result = handler.forward({"x": x}, {})
    assert torch.equal(result["y"], torch.tensor([[0.0, 0.0, 2.0]]))


def test_make_binary_handler_rejects_nonfinite_native_result(monkeypatch) -> None:
    def fake_native(op_name, a, b):
        return torch.full_like(a, float("inf"))

    monkeypatch.setattr(base, "_try_native", fake_native)
    handler = base.make_binary_handler(
        lambda a, b: a / (b + 1e-6 * torch.where(b >= 0, 1.0, -1.0)),
        native_op_name="div_safe",
    )()
    a = torch.tensor([[2.0, 6.0]])
    b = torch.tensor([[1.0, 0.0]])
    result = handler.forward({"a": a, "b": b}, {})
    assert torch.all(torch.isfinite(result["y"]))
