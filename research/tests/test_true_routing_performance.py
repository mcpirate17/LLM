from __future__ import annotations

import statistics

import pytest
import torch
import torch.nn as nn

from research.synthesis.true_routing_ops import (
    _dispatch_to_experts,
    _mini_attention,
    _mini_cheap_linear,
    _mini_conv,
    _mini_mamba_block,
    _mini_mlp,
    _mini_ssm,
    _mini_transformer_block,
    _op_arch_router,
    _op_compute_budget_router,
    _op_hetero_moe,
)


class _TrueRoutingBenchModule(nn.Module):
    def __init__(self, dim: int, op_name: str):
        super().__init__()
        self.op_name = op_name
        self.gate_weight = nn.Parameter(torch.randn(3, dim, device="cuda") * 0.02)
        if op_name in {"hetero_moe", "arch_router", "compute_budget_router"}:
            self.attn_qkv = nn.Parameter(
                torch.randn(3 * dim, dim, device="cuda") * 0.02
            )
            self.attn_out = nn.Parameter(torch.randn(dim, dim, device="cuda") * 0.02)
            self.conv_proj = nn.Parameter(torch.randn(dim, dim, device="cuda") * 0.02)
        if op_name in {"hetero_moe", "arch_router"}:
            self.ssm_B_proj = nn.Parameter(torch.randn(dim, dim, device="cuda") * 0.02)
            self.ssm_C_proj = nn.Parameter(torch.randn(dim, dim, device="cuda") * 0.02)
            self.ssm_D = nn.Parameter(torch.randn(dim, device="cuda") * 0.02)
        if op_name == "arch_router":
            self.arch_ffn = nn.Parameter(torch.randn(dim, dim, device="cuda") * 0.02)
            self.arch_proj = nn.Parameter(torch.randn(dim, dim, device="cuda") * 0.02)
            self.mlp_up = nn.Parameter(torch.randn(4 * dim, dim, device="cuda") * 0.02)
            self.mlp_down = nn.Parameter(
                torch.randn(dim, 4 * dim, device="cuda") * 0.02
            )
        if op_name == "compute_budget_router":
            self.cheap_proj = nn.Parameter(torch.randn(dim, dim, device="cuda") * 0.02)


def _cuda_median_ms(fn, *, warmup: int = 20, iterations: int = 80) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples: list[float] = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("op_name", "fast_op", "grouped_experts"),
    [
        ("hetero_moe", _op_hetero_moe, [_mini_attention, _mini_conv, _mini_ssm]),
        (
            "arch_router",
            _op_arch_router,
            [_mini_transformer_block, _mini_mamba_block, _mini_mlp],
        ),
        (
            "compute_budget_router",
            _op_compute_budget_router,
            [_mini_cheap_linear, _mini_conv, _mini_attention],
        ),
    ],
)
def test_true_routing_cuda_fast_path_beats_grouped_python_dispatch(
    op_name,
    fast_op,
    grouped_experts,
):
    torch.manual_seed(1234)
    batch, seq, dim = 16, 256, 64
    module = _TrueRoutingBenchModule(dim, op_name).eval()
    x = torch.randn(batch, seq, dim, device="cuda")

    with torch.no_grad():
        fast = fast_op(module, [x], {})
        grouped = _dispatch_to_experts(x, module, 3, grouped_experts)
    torch.testing.assert_close(fast, grouped, rtol=1e-4, atol=1e-5)

    fast_ms = _cuda_median_ms(lambda: fast_op(module, [x], {}))
    grouped_ms = _cuda_median_ms(
        lambda: _dispatch_to_experts(x, module, 3, grouped_experts)
    )

    print(
        f"\ntrue-routing {op_name}: fast_ms={fast_ms:.4f} "
        f"grouped_ms={grouped_ms:.4f} speedup={grouped_ms / fast_ms:.3f}x"
    )
    assert fast_ms < grouped_ms * 0.85
