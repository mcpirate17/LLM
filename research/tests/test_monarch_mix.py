"""Tests for NM-C3 Monarch-structured feature mixer.

The blockwise forward is verified against the dense matrix reconstruction
``M = blkdiag_b(R)·P·blkdiag_m(L)·P`` for randomized blocks — that is the spec
that pins the reshape/einsum derivation. Identity-at-init falls out for free
since ``L_i = I_b, R_j = I_m ⟹ M = P·P = I`` (P involuted).
"""

from __future__ import annotations

import math

import pytest
import torch

from research.synthesis.monarch_mix import MonarchMix, monarch_param_count
from research.synthesis.physics_descriptors import PhysicsDescriptorProbe


def test_forward_preserves_shape_and_is_finite() -> None:
    mix = MonarchMix(dim=8)
    x = torch.randn(2, 10, 8)
    y = mix(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("d", [1, 2, 3, 5, 7, 8, 9, 16, 17, 32])
def test_identity_at_init(d: int) -> None:
    """At init the Monarch matrix is exactly identity, so the layer is a no-op
    drop-in — even for non-square d (padded internally)."""
    mix = MonarchMix(dim=d)
    x = torch.randn(3, 5, d)
    assert torch.allclose(mix(x), x, atol=1e-6), f"dim={d} not identity at init"


def test_random_blocks_match_dense_reconstruction_square() -> None:
    """Blockwise forward == x @ M.T for the dense Monarch matrix (no padding)."""
    torch.manual_seed(0)
    d = 16  # m = b = 4, n = 16 -> no padding
    mix = MonarchMix(dim=d)
    with torch.no_grad():
        mix.L.normal_(0.0, 0.3)
        mix.R.normal_(0.0, 0.3)
    x = torch.randn(4, 7, d)
    y_fast = mix(x)
    y_dense = x @ mix.dense_matrix().T
    assert torch.allclose(y_fast, y_dense, atol=1e-5)


def test_random_blocks_match_dense_reconstruction_padded() -> None:
    """Non-square d (padded to n) still matches the dense path on the active dim."""
    torch.manual_seed(1)
    d = 8  # b = 3, m = 3, n = 9 padded
    mix = MonarchMix(dim=d)
    with torch.no_grad():
        mix.L.normal_(0.0, 0.3)
        mix.R.normal_(0.0, 0.3)
    x = torch.randn(4, 7, d)
    y_fast = mix(x)
    pad = x.new_zeros(*x.shape[:-1], mix.n - d)
    y_dense = (torch.cat([x, pad], dim=-1) @ mix.dense_matrix().T)[..., :d]
    assert torch.allclose(y_fast, y_dense, atol=1e-5)


def test_random_blocks_match_dense_large_block() -> None:
    """A larger block_size (more padding, more capacity) still matches the dense
    reconstruction on the active dim. Square factors m == b throughout."""
    torch.manual_seed(2)
    d = 64
    mix = MonarchMix(dim=d, block_size=16)  # b = m = 16, n = 256 (padded 64 -> 256)
    assert (mix.b, mix.m, mix.n) == (16, 16, 256)
    with torch.no_grad():
        mix.L.normal_(0.0, 0.2)
        mix.R.normal_(0.0, 0.2)
    x = torch.randn(3, 11, d)
    pad = x.new_zeros(*x.shape[:-1], mix.n - d)
    y_dense = (torch.cat([x, pad], dim=-1) @ mix.dense_matrix().T)[..., :d]
    assert torch.allclose(mix(x), y_dense, atol=1e-4)


def test_param_count_is_subquadratic_and_exact() -> None:
    d = 384
    n = monarch_param_count(d)
    assert n == sum(p.numel() for p in MonarchMix(d).parameters())
    assert n < d * d  # strictly subquadratic
    # O(d·sqrt(d)) headroom: 2·d·(sqrt(d)+2) covers m·b² + b·m² with b≈m≈√d.
    assert n <= 2 * d * (math.isqrt(d) + 2)
    # the headline compaction claim: ~9× under dense at cl100k model dim.
    assert n * 9 <= d * d


def test_backward_flows_to_both_block_banks() -> None:
    mix = MonarchMix(dim=16)
    with torch.no_grad():  # move off identity so grads are nonzero
        mix.L.add_(0.1 * torch.randn_like(mix.L))
        mix.R.add_(0.1 * torch.randn_like(mix.R))
    x = torch.randn(2, 6, 16, requires_grad=True)
    mix(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert (
        mix.L.grad is not None
        and torch.isfinite(mix.L.grad).all()
        and mix.L.grad.abs().sum() > 0
    )
    assert (
        mix.R.grad is not None
        and torch.isfinite(mix.R.grad).all()
        and mix.R.grad.abs().sum() > 0
    )


def test_block_size_knob_changes_param_count_and_stays_identity() -> None:
    d = 64
    default = monarch_param_count(d)  # b = m = 8, n = 64
    wider = monarch_param_count(d, block_size=16)  # b = m = 16, n = 256
    assert wider != default
    mix = MonarchMix(d, block_size=16)
    assert (mix.b, mix.m) == (16, 16)
    x = torch.randn(1, 5, d)
    assert torch.allclose(
        mix(x), x, atol=1e-6
    )  # identity-at-init holds for any block size


def test_measurable_by_physics_descriptor_probe() -> None:
    """NM-10: the mixer exposes a finite physics fingerprint so it can be scored
    on the geometric-novelty axis alongside other primitives."""
    probe = PhysicsDescriptorProbe(batch=2, seq_len=8, dim=16, n_seeds=2)
    mix = MonarchMix(dim=16)
    with torch.no_grad():  # nudge off identity for a non-trivial fingerprint
        mix.L.add_(0.2 * torch.randn_like(mix.L))
        mix.R.add_(0.2 * torch.randn_like(mix.R))
    desc = probe.describe_operator(mix)
    assert desc, "probe returned no descriptors"
    for key, value in desc.items():
        assert isinstance(value, float) and math.isfinite(value), f"{key}={value}"
