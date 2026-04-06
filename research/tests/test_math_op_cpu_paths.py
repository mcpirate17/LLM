from __future__ import annotations

import torch

from research.synthesis.compiled_op import CompiledOp
from research.synthesis.graph import ShapeInfo
from research.synthesis.compiler_ops_math import _op_linear_common
from research.synthesis.compiler_op_utils import _safe_linear


def test_linear_common_cpu_path_matches_dense_pytorch(monkeypatch):
    module = CompiledOp(
        "linear_proj", {}, ShapeInfo(dim=16), ShapeInfo(dim=16), 16
    ).eval()
    x = torch.randn(2, 5, 16)

    def _fail(*args, **kwargs):
        raise AssertionError("aria_core.linear_f32 should not be used on CPU")

    monkeypatch.setattr(
        "research.synthesis.compiler_ops_math.aria_core.linear_f32",
        _fail,
        raising=False,
    )

    expected = _safe_linear(x, module.weight, getattr(module, "bias", None))
    result = _op_linear_common(module, [x], {})
    assert torch.allclose(result, expected, atol=1e-6, rtol=1e-5)
