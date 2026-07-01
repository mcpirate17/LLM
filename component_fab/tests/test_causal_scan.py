"""Tests for the chunked causal power-law decay scan (speed + inference VRAM)."""

from __future__ import annotations

import pytest
import torch

from component_fab.generator._causal_scan import (
    causal_decay_context,
    causal_decay_context_streaming,
    decay_scan_step,
    init_decay_scan_state,
)


def _naive_decay_context(x: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    """O(L^2) reference: the dense decay-matrix contraction."""
    length = x.shape[1]
    idx = torch.arange(length)
    exps = (idx[:, None] - idx[None, :]).clamp(min=0).to(x.dtype)
    causal = (idx[:, None] >= idx[None, :]).to(x.dtype)
    powmat = torch.exp(exps[None] * torch.log(decay)[:, None, None]) * causal[None]
    return torch.einsum("cts,bsc->btc", powmat, x)


@pytest.mark.parametrize(
    "length,chunk", [(20, 64), (64, 64), (100, 64), (129, 32), (200, 7)]
)
def test_chunked_matches_naive(length: int, chunk: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(3, length, 8)
    decay = torch.sigmoid(torch.randn(8))
    ref = _naive_decay_context(x, decay)
    got = causal_decay_context(x, decay, chunk=chunk)
    assert torch.allclose(ref, got, atol=1e-5)


def test_is_causal_and_recurrent() -> None:
    torch.manual_seed(1)
    x = torch.randn(1, 24, 4)
    decay = torch.sigmoid(torch.randn(4))
    c = causal_decay_context(x, decay, chunk=8)
    # c_t = decay * c_{t-1} + x_t  (the underlying linear recurrence)
    for t in range(1, 24):
        assert torch.allclose(c[:, t], decay * c[:, t - 1] + x[:, t], atol=1e-5)


def test_gradients_finite() -> None:
    x = torch.randn(2, 50, 8, requires_grad=True)
    decay = torch.sigmoid(torch.randn(8))
    causal_decay_context(x, decay, chunk=16).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        causal_decay_context(torch.randn(3, 8), torch.rand(8))
    with pytest.raises(ValueError):
        causal_decay_context(torch.randn(2, 8, 4), torch.rand(5))


def test_streaming_matches_batched() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 40, 8)
    decay = torch.sigmoid(torch.randn(8))
    assert torch.allclose(
        causal_decay_context_streaming(x, decay),
        causal_decay_context(x, decay),
        atol=1e-5,
    )


def test_streaming_state_is_constant_size() -> None:
    """The decode state is [B, C] regardless of how many tokens have been fed."""
    torch.manual_seed(1)
    x = torch.randn(2, 30, 4)
    decay = torch.sigmoid(torch.randn(4))
    batched = causal_decay_context(x, decay)
    state = init_decay_scan_state(2, 4)
    for t in range(x.shape[1]):
        c_t, state = decay_scan_step(state, x[:, t, :], decay)
        assert state.context.shape == (2, 4)  # never grows with t
        assert torch.allclose(c_t, batched[:, t, :], atol=1e-5)


def test_streaming_prefill_then_continue() -> None:
    """Prefill a prefix batched, then stream the rest — same as full batched."""
    torch.manual_seed(2)
    x = torch.randn(2, 20, 6)
    decay = torch.sigmoid(torch.randn(6))
    full = causal_decay_context(x, decay)
    split = 12
    prefix = causal_decay_context(x[:, :split, :], decay)
    state = init_decay_scan_state(2, 6)
    state.context = prefix[:, -1, :]  # carry the prefill's last state
    for t in range(split, x.shape[1]):
        c_t, state = decay_scan_step(state, x[:, t, :], decay)
        assert torch.allclose(c_t, full[:, t, :], atol=1e-5)
