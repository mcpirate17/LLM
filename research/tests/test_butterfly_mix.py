"""Tests for NM-C5 butterfly / orthogonal-flow feature mixer.

Two specs pin correctness: (1) the blockwise forward exactly equals the dense
matrix reconstruction ``M = B_{L-1}·…·B_0`` (all passes), and (2) the product is
exactly orthogonal (``M @ M.T == I``) for arbitrary angles — the defining
property of a Givens-block butterfly. Identity-at-init falls out since angles = 0
⟹ every Givens = ``I_2`` ⟹ ``M = I``.
"""

from __future__ import annotations

import math

import pytest
import torch

from research.synthesis.butterfly_mix import ButterflyMix, butterfly_param_count
from research.synthesis.physics_descriptors import PhysicsDescriptorProbe


def test_forward_preserves_shape_and_is_finite() -> None:
    mix = ButterflyMix(dim=8)
    x = torch.randn(2, 10, 8)
    y = mix(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("d", [1, 2, 3, 5, 7, 8, 9, 16, 17, 32])
def test_identity_at_init(d: int) -> None:
    """At init all angles are 0 ⟹ M = I (safe drop-in for any d, padded to pow2)."""
    mix = ButterflyMix(dim=d)
    x = torch.randn(3, 5, d)
    assert torch.allclose(mix(x), x, atol=1e-6), f"dim={d} not identity at init"


def test_random_angles_match_dense_reconstruction_pow2() -> None:
    """Blockwise forward == x @ M.T for the dense butterfly product (no padding)."""
    torch.manual_seed(0)
    d = 8  # n = 8, no padding
    mix = ButterflyMix(dim=d)
    with torch.no_grad():
        mix.angles.uniform_(-1.5, 1.5)
    x = torch.randn(4, 7, d)
    assert torch.allclose(mix(x), x @ mix.dense_matrix().T, atol=1e-5)


def test_random_angles_match_dense_reconstruction_padded() -> None:
    """Non-pow2 d (padded to next pow2) still matches the dense path on active dim."""
    torch.manual_seed(1)
    d = 6  # n = 8 padded
    mix = ButterflyMix(dim=d)
    with torch.no_grad():
        mix.angles.uniform_(-1.5, 1.5)
    x = torch.randn(4, 7, d)
    pad = x.new_zeros(*x.shape[:-1], mix.n - d)
    y_dense = (torch.cat([x, pad], dim=-1) @ mix.dense_matrix().T)[..., :d]
    assert torch.allclose(mix(x), y_dense, atol=1e-5)


def test_product_is_exactly_orthogonal() -> None:
    """Defining property: a Givens-block butterfly product is orthogonal for any
    angles ⟹ M @ M.T == I (spectral_radius = 1, energy_gain = 1)."""
    torch.manual_seed(2)
    d = 16
    mix = ButterflyMix(dim=d, n_passes=3)
    with torch.no_grad():
        mix.angles.uniform_(-3.0, 3.0)
    m = mix.dense_matrix()
    assert torch.allclose(m @ m.T, torch.eye(mix.n), atol=1e-5)


def test_param_count_is_subquadratic_and_exact() -> None:
    d = 384
    n = butterfly_param_count(d)  # n_passes=2 -> n=512, L=9
    assert n == sum(p.numel() for p in ButterflyMix(d).parameters())
    assert n < d * d  # strictly subquadratic
    # O(d log d): 2 passes * (n/2) * log2(n); headroom over 3 * d * log2(2d).
    assert n <= 3 * d * math.log2(2 * d)
    # headline compaction claim: ~32x under dense at cl100k model dim.
    assert n * 30 <= d * d


def test_backward_flows_to_angle_params() -> None:
    mix = ButterflyMix(dim=8)
    with torch.no_grad():
        mix.angles.add_(0.1 * torch.randn_like(mix.angles))  # off identity
    x = torch.randn(2, 6, 8, requires_grad=True)
    mix(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert mix.angles.grad is not None
    assert torch.isfinite(mix.angles.grad).all()
    assert mix.angles.grad.abs().sum() > 0


def test_n_passes_knob_changes_param_count_and_stays_identity() -> None:
    d = 16
    thin = butterfly_param_count(d, n_passes=1)
    wide = butterfly_param_count(d, n_passes=4)
    assert wide == 4 * thin
    assert wide != thin
    mix = ButterflyMix(d, n_passes=4)
    x = torch.randn(1, 5, d)
    assert torch.allclose(mix(x), x, atol=1e-6)  # identity-at-init for any n_passes


def test_measurable_by_physics_descriptor_probe_and_orthogonal_fingerprint() -> None:
    """NM-10: finite physics fingerprint; the orthogonal structure shows up as a
    norm-preserving (energy_gain ≈ 1), unit-spectral-radius linear map."""
    probe = PhysicsDescriptorProbe(batch=2, seq_len=8, dim=16, n_seeds=2)
    mix = ButterflyMix(dim=16)
    with torch.no_grad():  # off identity for a non-trivial fingerprint
        mix.angles.uniform_(-2.0, 2.0)
    desc = probe.describe_operator(mix)
    assert desc, "probe returned no descriptors"
    for key, value in desc.items():
        assert isinstance(value, float) and math.isfinite(value), f"{key}={value}"
    # orthogonal ⟹ stable / norm-preserving: loose bounds (probe is empirical).
    if "spectral_radius" in desc:
        assert 0.5 < desc["spectral_radius"] < 1.5, desc["spectral_radius"]
    if "energy_gain" in desc:
        assert 0.5 < desc["energy_gain"] < 1.5, desc["energy_gain"]
