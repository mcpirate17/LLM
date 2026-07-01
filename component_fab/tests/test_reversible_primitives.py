"""Tests for the reversible activation-free coupling stack (VRAM reduction).

Two decisive tests:
- gradient equivalence: the reverse-sweep backward must produce identical input-
  and parameter-gradients to a plain autograd stack of the same blocks;
- activation saving: a deep ``ReversibleSequential`` must save dramatically fewer
  activations than the plain stack (the whole point — O(1) in depth).

Also covers exact single-block invertibility, causality, and the
anti-softmax-twin structural claim.
"""

from __future__ import annotations

import pytest
import torch

from component_fab.generator.reversible_primitives import (
    ReversibleCouplingMixerLane,
    ReversibleSequential,
    _CausalDecayMLP,
)
from component_fab.proposer.algebraic_properties import measure_algebraic_properties


def _plain_stack_forward(stack: ReversibleSequential, x: torch.Tensor) -> torch.Tensor:
    """Reference: apply the same blocks under ordinary autograd (stores activations)."""
    y = x
    for block in stack.blocks:
        y = block.coupling_forward(y)
    return y


def _count_forward_saved_activations(build_forward, x: torch.Tensor) -> int:
    """Count activations persisted by the FORWARD pass (the peak-memory measure).

    Hooks only the forward: tensors saved during the backward recompute are
    transient and freed immediately, so counting them would unfairly inflate the
    reversible path. Peak training memory is set by what forward keeps alive.
    """
    saved: list[int] = []

    def pack(t: torch.Tensor) -> torch.Tensor:
        saved.append(t.numel())
        return t

    def unpack(t: torch.Tensor) -> torch.Tensor:
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        out = build_forward(x)
    out.sum().backward()
    return sum(saved)


def test_reversible_stack_backward_matches_plain_autograd() -> None:
    # float64: the reverse-sweep reconstructs inputs, so gradients match plain
    # autograd exactly up to numerical precision (in float32 the reconstruction
    # rounding is ~1e-3, expected for reversible nets; the logic is exact here).
    torch.manual_seed(0)
    stack = ReversibleSequential.build(dim=16, depth=4).double()

    x = torch.randn(3, 12, 16, dtype=torch.float64, requires_grad=True)
    stack(x).square().sum().backward()
    rev_x_grad = x.grad.clone()
    rev_p_grads = [p.grad.clone() for p in stack.parameters()]

    stack.zero_grad(set_to_none=True)
    x_ref = x.detach().clone().requires_grad_(True)
    _plain_stack_forward(stack, x_ref).square().sum().backward()

    assert torch.allclose(rev_x_grad, x_ref.grad, atol=1e-9)
    for rev_g, p in zip(rev_p_grads, stack.parameters()):
        assert p.grad is not None
        assert torch.allclose(rev_g, p.grad, atol=1e-9)


def test_reversible_stack_saves_far_fewer_activations() -> None:
    torch.manual_seed(0)
    depth = 16
    stack = ReversibleSequential.build(dim=32, depth=depth)
    x = torch.randn(4, 24, 32, requires_grad=True)

    rev_saved = _count_forward_saved_activations(stack, x)
    x.grad = None
    plain_saved = _count_forward_saved_activations(
        lambda z: _plain_stack_forward(stack, z), x
    )

    # The reversible stack persists only the final activation; the plain stack
    # keeps every block's internals, so the ratio grows with depth (~300x here).
    assert plain_saved > 20 * rev_saved


def test_single_block_is_exactly_invertible() -> None:
    torch.manual_seed(1)
    lane = ReversibleCouplingMixerLane(16)
    x = torch.randn(2, 10, 16)
    with torch.no_grad():
        x_rec = lane.inverse(lane(x))
    assert torch.allclose(x, x_rec, atol=1e-5)


def test_reversible_stack_shape_finite_and_grad() -> None:
    torch.manual_seed(0)
    stack = ReversibleSequential.build(dim=24, depth=3)
    x = torch.randn(2, 8, 24, requires_grad=True)
    y = stack(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_reversible_requires_even_dim() -> None:
    with pytest.raises(ValueError):
        ReversibleCouplingMixerLane(15)


def test_coupling_is_causal() -> None:
    torch.manual_seed(2)
    mlp = _CausalDecayMLP(8)
    x_a = torch.randn(1, 16, 8)
    x_b = x_a.clone()
    x_b[:, 9:] += torch.randn(1, 7, 8)
    with torch.no_grad():
        assert torch.allclose(mlp(x_a)[:, :9], mlp(x_b)[:, :9], atol=1e-5)


def test_reversible_is_not_a_softmax_twin() -> None:
    torch.manual_seed(0)
    lane = ReversibleCouplingMixerLane(24)
    props = measure_algebraic_properties(lane, dim=24, n_seeds=3)
    assert not props.is_softmax_twin(), props.to_dict()


def test_streaming_decode_matches_batched_forward() -> None:
    """Token-by-token O(1)-state decode reproduces the batched forward (KV-free)."""
    torch.manual_seed(0)
    lane = ReversibleCouplingMixerLane(16)
    x = torch.randn(2, 18, 16)
    with torch.no_grad():
        batched = lane(x)
        state = lane.stream_init(2)
        for t in range(x.shape[1]):
            y_t, state = lane.stream_step(x[:, t, :], state)
            assert torch.allclose(y_t, batched[:, t, :], atol=1e-5)
        # State is two [B, half] tensors — independent of sequence length.
        f_state, g_state = state
        assert f_state.context.shape == (2, 8)
        assert g_state.context.shape == (2, 8)
