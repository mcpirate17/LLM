"""Tests for the chunked causal power-law decay scan (speed + inference VRAM)."""

from __future__ import annotations

import pytest
import torch

from component_fab.generator._causal_scan import causal_decay_context


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
