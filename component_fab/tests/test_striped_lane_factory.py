"""Tests for the striped hybrid lane factory (attention interleaved with candidate)."""

from __future__ import annotations

import torch
from torch import nn

from component_fab.harness.tiny_lm import (
    SoftmaxCausalAttention,
    striped_lane_factory,
)


class _Marker(nn.Module):
    """A trivial candidate lane that is identity but type-distinguishable."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def _kinds(attn_every: int, n_blocks: int) -> list[str]:
    factory = striped_lane_factory(_Marker, attn_every=attn_every)
    out = []
    for _ in range(n_blocks):
        m = factory(8)
        out.append("attn" if isinstance(m, SoftmaxCausalAttention) else "cand")
    return out


def test_one_to_one_stripe() -> None:
    assert _kinds(2, 4) == ["attn", "cand", "attn", "cand"]


def test_three_to_one_stripe() -> None:
    # attn_every=4 → 1 full-attention block per 3 candidate blocks (Lahoti regime)
    assert _kinds(4, 4) == ["attn", "cand", "cand", "cand"]


def test_block_zero_is_always_attention() -> None:
    assert _kinds(2, 1) == ["attn"]
    assert _kinds(8, 1) == ["attn"]


def test_fresh_factory_resets_position() -> None:
    f1 = striped_lane_factory(_Marker, attn_every=2)
    f2 = striped_lane_factory(_Marker, attn_every=2)
    assert isinstance(f1(8), SoftmaxCausalAttention)  # f1 block 0 = attn
    # f2 is independent — its own block 0 is attn, not continuing f1's counter.
    assert isinstance(f2(8), SoftmaxCausalAttention)
