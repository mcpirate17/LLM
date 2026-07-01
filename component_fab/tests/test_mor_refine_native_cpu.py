"""CPU C++ MoR refine scan == the torch reference, exactly.

``native_mor_refine.cpp`` ports the CUDA refine kernel to CPU so fab grading
(CPU) stops running the per-token torch loop. These tests pin the CPU native
path against ``MoRRefineLaneA._scan`` (the torch reference the CUDA kernel was
validated against) — forward, ponder telemetry, and gradients through every
parameter — plus a float64 gradcheck of the autograd Function itself.
"""

from __future__ import annotations

import pytest
import torch

from component_fab.generator.mor_bilane import (
    MoRRefineLaneA,
    MoRRefineMLPLaneA,
    MoRSurpriseRefineMLPLaneA,
    _NativeMoRRefineScan,
)


def _lane_case(lane_cls, seed: int, **kwargs):
    torch.manual_seed(seed)
    lane = lane_cls(
        16, memory_dim=6, max_recursive_steps=4, router_hidden=8, **kwargs
    ).double()
    # Non-degenerate router: the deep-start init zeroes/level's the halt path,
    # which would hide W2/feature gradient bugs — randomize it for the test.
    with torch.no_grad():
        for p in lane.halt_head.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    x = torch.randn(2, 5, 16, dtype=torch.double)
    return lane, x


def _run(lane, x, native: bool):
    lane.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    if native:
        out = lane._scan(x)
        assert lane.last_ponder_cost is not None
        ponder = lane.last_ponder_cost
    else:
        out = MoRRefineLaneA._scan(lane, x)
        ponder = lane.last_ponder_cost
    (out.square().sum() + ponder).backward()
    grads = {
        name: p.grad.clone()
        for name, p in lane.named_parameters()
        if p.grad is not None
    }
    return (
        out.detach(),
        float(ponder.detach()),
        lane.last_mean_depth,
        grads,
        x.grad.clone(),
    )


@pytest.mark.parametrize("lane_cls", [MoRRefineMLPLaneA, MoRSurpriseRefineMLPLaneA])
def test_cpu_native_matches_torch_reference(lane_cls) -> None:
    lane, x = _lane_case(lane_cls, seed=0)
    out_n, ponder_n, depth_n, grads_n, gx_n = _run(lane, x, native=True)
    out_r, ponder_r, depth_r, grads_r, gx_r = _run(lane, x, native=False)

    assert torch.allclose(out_n, out_r, atol=1e-10)
    assert ponder_n == pytest.approx(ponder_r, abs=1e-10)
    assert depth_n == pytest.approx(depth_r, abs=1e-8)
    assert torch.allclose(gx_n, gx_r, atol=1e-8)
    assert grads_n.keys() == grads_r.keys()
    for name in grads_r:
        assert torch.allclose(grads_n[name], grads_r[name], atol=1e-8), (
            f"grad mismatch for {name}"
        )


def test_force_max_depth_uses_reference_and_agrees() -> None:
    lane, x = _lane_case(MoRRefineMLPLaneA, seed=1)
    out_native = lane._scan(x)
    lane.force_max_depth = True
    out_forced = lane._scan(x)  # torch reference path (ablation)
    lane.force_max_depth = False
    # Not equal in general (router halts early), but shapes/finiteness hold and
    # forcing max depth must change depth telemetry to the max.
    assert out_forced.shape == out_native.shape
    assert torch.isfinite(out_forced).all()
    assert lane.last_mean_depth == pytest.approx(4.0)


def test_native_mor_refine_scan_gradcheck() -> None:
    torch.manual_seed(2)
    bsz, seq_len, m, h, steps = 1, 3, 3, 4, 3
    q = torch.randn(bsz, seq_len, m, dtype=torch.double, requires_grad=True)
    k = torch.randn(bsz, seq_len, m, dtype=torch.double, requires_grad=True)
    v = torch.randn(bsz, seq_len, m, dtype=torch.double, requires_grad=True)
    write = torch.sigmoid(
        torch.randn(bsz, seq_len, dtype=torch.double)
    ).requires_grad_()
    forget = (
        (torch.sigmoid(torch.randn(bsz, seq_len, m, dtype=torch.double)) * 0.1)
        .detach()
        .requires_grad_()
    )
    momentum = torch.tensor(0.4, dtype=torch.double, requires_grad=True)
    beta = torch.tensor(3.0, dtype=torch.double, requires_grad=True)
    balance = torch.tensor(0.75, dtype=torch.double, requires_grad=True)
    W1 = (torch.randn(h, 3, dtype=torch.double) * 0.3).requires_grad_()
    b1 = (torch.randn(h, dtype=torch.double) * 0.3).requires_grad_()
    W2 = (torch.randn(h, dtype=torch.double) * 0.3).requires_grad_()
    b2 = torch.tensor(-0.5, dtype=torch.double, requires_grad=True)
    a_coupling = torch.tensor(0.3, dtype=torch.double, requires_grad=True)

    def fn(*args):
        y, depth, _hist = _NativeMoRRefineScan.apply(*args, steps)
        return y, depth

    assert torch.autograd.gradcheck(
        fn,
        (q, k, v, write, forget, momentum, beta, balance, W1, b1, W2, b2, a_coupling),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
