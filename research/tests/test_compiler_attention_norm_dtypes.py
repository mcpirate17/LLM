from __future__ import annotations

import torch

from research.synthesis.compiler_ops_attention import _op_layernorm, _op_rmsnorm


class _NormModule:
    def __init__(self, dim: int) -> None:
        self.weight = torch.randn(dim, dtype=torch.float32)
        self.bias = torch.randn(dim, dtype=torch.float32)


def test_attention_norm_ops_preserve_low_precision_input_dtype():
    module = _NormModule(8)
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)

    layernorm_out = _op_layernorm(module, (x,), {})
    rmsnorm_out = _op_rmsnorm(module, (x,), {})

    assert layernorm_out.dtype == torch.bfloat16
    assert rmsnorm_out.dtype == torch.bfloat16
    assert torch.isfinite(layernorm_out).all()
    assert torch.isfinite(rmsnorm_out).all()
