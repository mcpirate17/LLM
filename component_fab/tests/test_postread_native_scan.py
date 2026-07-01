"""Native post-write-read scan == the Python delta-rule loop, exactly.

The CPU C++ scans in ``native_postread_surprise.cpp`` are perf ports of
``_SurpriseMemoryBase._scan_python``; these tests pin forward AND gradient
parity in float64 so any algebra drift (read timing, retrieval semiring,
delta write) fails loudly. The Hebbian lanes' Kogge-Stone rewrites are pinned
against the original per-token recurrence the same way.
"""

from __future__ import annotations

import pytest
import torch

from component_fab.generator._postread_scan import (
    SemiringPostreadScan,
    TropicalPostreadScan,
)
from component_fab.generator.memory_primitives import (
    CausalFastWeightMemoryLane,
    DataDependentDecayMemoryLane,
    PadicSurpriseMemoryLane,
    SemiringSurpriseMemoryLane,
    TropicalSurpriseMemoryLane,
)


def _scan_case(seed: int, bsz: int = 2, seq_len: int = 7, memory_dim: int = 4):
    torch.manual_seed(seed)
    q = torch.randn(bsz, seq_len, memory_dim, dtype=torch.double, requires_grad=True)
    k = torch.randn(bsz, seq_len, memory_dim, dtype=torch.double, requires_grad=True)
    v = torch.randn(bsz, seq_len, memory_dim, dtype=torch.double, requires_grad=True)
    write = torch.sigmoid(
        torch.randn(bsz, seq_len, dtype=torch.double)
    ).requires_grad_()
    forget = (
        torch.sigmoid(torch.randn(bsz, seq_len, memory_dim, dtype=torch.double)) * 0.1
    )
    forget = forget.detach().requires_grad_()
    momentum = torch.tensor(0.4, dtype=torch.double, requires_grad=True)
    return q, k, v, write, forget, momentum


def test_tropical_postread_gradcheck() -> None:
    inputs = _scan_case(0, bsz=1, seq_len=3, memory_dim=2)
    assert torch.autograd.gradcheck(
        lambda *args: TropicalPostreadScan.apply(*args),
        inputs,
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


def test_semiring_postread_gradcheck() -> None:
    inputs = _scan_case(1, bsz=1, seq_len=3, memory_dim=2)
    beta = torch.tensor(3.0, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda *args: SemiringPostreadScan.apply(*args),
        (*inputs, beta),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "lane_cls", [TropicalSurpriseMemoryLane, SemiringSurpriseMemoryLane]
)
def test_native_scan_matches_python_loop(lane_cls) -> None:
    torch.manual_seed(2)
    lane = lane_cls(12, memory_dim=6).double()
    x = torch.randn(2, 9, 12, dtype=torch.double)

    scan_inputs = lane._scan_inputs(x)
    native = lane._scan_native(*scan_inputs)
    assert native is not None, "CPU float64 must dispatch to the native scan"
    reference = lane._scan_python(*scan_inputs)
    assert torch.allclose(native, reference, atol=1e-12)

    # Gradient parity through the FULL lane on both paths.
    def loss_for(path_native: bool) -> dict[str, torch.Tensor]:
        lane.zero_grad(set_to_none=True)
        inputs = lane._scan_inputs(x)
        read = lane._scan_native(*inputs) if path_native else lane._scan_python(*inputs)
        lane.out(read).square().sum().backward()
        return {
            name: p.grad.clone()
            for name, p in lane.named_parameters()
            if p.grad is not None
        }

    grads_native = loss_for(True)
    grads_python = loss_for(False)
    assert grads_native.keys() == grads_python.keys()
    for name in grads_python:
        assert torch.allclose(grads_native[name], grads_python[name], atol=1e-10), (
            f"grad mismatch for {name}"
        )


def test_padic_forward_native_matches_python(monkeypatch) -> None:
    torch.manual_seed(3)
    lane = PadicSurpriseMemoryLane(16, memory_dim=8, p=2, n_levels=3).double()
    x = torch.randn(2, 6, 16, dtype=torch.double)
    out_native = lane(x)

    monkeypatch.setattr(
        "component_fab.generator.memory_primitives.native_postread_supported",
        lambda _t: False,
    )
    out_python = lane(x)
    assert torch.allclose(out_native, out_python, atol=1e-12)


def _fast_weight_reference(lane: CausalFastWeightMemoryLane, x: torch.Tensor):
    batch_size, seq_len, _ = x.shape
    q = torch.tanh(lane.q(x))
    k = torch.tanh(lane.k(x))
    v = torch.tanh(lane.v(x))
    gates = torch.sigmoid(lane.write_gate(x)).squeeze(-1)
    decay = torch.sigmoid(lane.decay_logit)
    memory = torch.zeros(batch_size, lane.memory_dim, lane.memory_dim, dtype=x.dtype)
    outputs = []
    scale = float(lane.memory_dim) ** -0.5
    for t in range(seq_len):
        write = torch.einsum("bi,bj->bij", k[:, t], v[:, t]) * scale
        memory = decay * memory + gates[:, t].view(batch_size, 1, 1) * write
        outputs.append(torch.einsum("bi,bij->bj", q[:, t], memory))
    return lane.out(torch.stack(outputs, dim=1))


def _ddd_reference(lane: DataDependentDecayMemoryLane, x: torch.Tensor):
    batch_size, seq_len, _ = x.shape
    q = torch.tanh(lane.q(x))
    k = torch.tanh(lane.k(x))
    v = torch.tanh(lane.v(x))
    write_strength = torch.sigmoid(lane.write_gate(x))
    decay = torch.sigmoid(lane.decay_gate(x))
    memory = torch.zeros(batch_size, lane.memory_dim, lane.memory_dim, dtype=x.dtype)
    outputs = []
    scale = float(lane.memory_dim) ** -0.5
    for t in range(seq_len):
        write = torch.einsum("bi,bj->bij", k[:, t], v[:, t]) * scale
        memory = (
            decay[:, t].unsqueeze(-1) * memory
            + write_strength[:, t].unsqueeze(-1) * write
        )
        outputs.append(torch.einsum("bi,bij->bj", q[:, t], memory))
    return lane.out(torch.stack(outputs, dim=1))


@pytest.mark.parametrize(
    ("lane_cls", "reference"),
    [
        (CausalFastWeightMemoryLane, _fast_weight_reference),
        (DataDependentDecayMemoryLane, _ddd_reference),
    ],
)
def test_hebbian_scan_matches_loop_reference(lane_cls, reference) -> None:
    torch.manual_seed(4)
    lane = lane_cls(10, memory_dim=6).double()
    x = torch.randn(2, 11, 10, dtype=torch.double, requires_grad=True)

    out_scan = lane(x)
    out_scan.square().sum().backward()
    grad_scan = {name: p.grad.clone() for name, p in lane.named_parameters()}
    grad_x_scan = x.grad.clone()

    lane.zero_grad(set_to_none=True)
    x.grad = None
    out_ref = reference(lane, x)
    out_ref.square().sum().backward()

    assert torch.allclose(out_scan, out_ref, atol=1e-10)
    assert torch.allclose(grad_x_scan, x.grad, atol=1e-8)
    for name, p in lane.named_parameters():
        assert torch.allclose(grad_scan[name], p.grad, atol=1e-8), name


def test_native_scan_stays_causal() -> None:
    torch.manual_seed(5)
    lane = TropicalSurpriseMemoryLane(8, memory_dim=4).double()
    x = torch.randn(1, 6, 8, dtype=torch.double)
    base = lane(x)
    perturbed = x.clone()
    perturbed[:, 4:] += 1.0
    out = lane(perturbed)
    assert torch.allclose(base[:, :4], out[:, :4], atol=1e-12)
    assert not torch.allclose(base[:, 4:], out[:, 4:], atol=1e-6)
