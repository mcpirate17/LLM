"""Tests for NM-F3 nilpotent-Lie signature scan.

Pins the spec: the scan is the EXACT group product (Chen's identity — composing
the two halves' elements equals the whole sequence's element, to numerical
precision), the mixing law carries zero learned parameters, order sensitivity is
structural (anagrams share level 1 but differ at level 2), the mixer is causal,
identity-at-init, and exposes a finite NM-10 physics fingerprint.
"""

from __future__ import annotations

import math

import pytest
import torch

from research.synthesis.nilpotent_lie_scan import (
    NilpotentLieScan,
    compose,
    lie_scan_param_count,
)
from research.synthesis.physics_descriptors import PhysicsDescriptorProbe


def _nudged(dim: int = 16, lift_dim: int = 4) -> NilpotentLieScan:
    mix = NilpotentLieScan(dim=dim, lift_dim=lift_dim)
    with torch.no_grad():  # move off identity so the signature reaches the output
        mix.readout.weight.add_(0.3 * torch.randn_like(mix.readout.weight))
    return mix


def test_forward_preserves_shape_and_is_finite() -> None:
    mix = _nudged()
    x = torch.randn(2, 10, 16)
    y = mix(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("d", [1, 2, 8, 16, 33])
def test_identity_at_init(d: int) -> None:
    """Zero-init readout ⟹ the mixer is an exact no-op drop-in at init."""
    mix = NilpotentLieScan(dim=d, lift_dim=4)
    x = torch.randn(3, 6, d)
    assert torch.allclose(mix(x), x, atol=1e-6), f"dim={d} not identity at init"


def test_chen_identity_exact_semigroup() -> None:
    """THE structural claim: the group element of a concatenated sequence equals
    the composition of the halves' elements — associativity is algebra, so the
    equality holds to numerical precision, not approximately."""
    torch.manual_seed(0)
    mix = NilpotentLieScan(dim=16, lift_dim=4)
    x1 = torch.randn(2, 7, 16)
    x2 = torch.randn(2, 5, 16)
    whole = mix.group_element(torch.cat([x1, x2], dim=1))
    composed = compose(mix.group_element(x1), mix.group_element(x2))
    for got, expected in zip(whole, composed):
        assert torch.allclose(got, expected, atol=1e-4)


def test_anagram_discrimination_is_structural() -> None:
    """Permuting the tokens leaves level 1 (A, B) invariant — a sum/EMA pooler
    sees NOTHING — while the ordered level-2 term C differs. Order sensitivity
    costs zero parameters; it is the algebra."""
    torch.manual_seed(1)
    mix = NilpotentLieScan(dim=16, lift_dim=4)
    x = torch.randn(1, 12, 16)
    perm = torch.randperm(12)
    assert not torch.equal(perm, torch.arange(12))
    a1, b1, c1 = mix.group_element(x)
    a2, b2, c2 = mix.group_element(x[:, perm])
    assert torch.allclose(a1, a2, atol=1e-5)
    assert torch.allclose(b1, b2, atol=1e-5)
    assert not torch.allclose(c1, c2, atol=1e-3), "level-2 term blind to order"


def test_level2_is_strictly_causal_ordered_pairs() -> None:
    """C pairs a strictly-earlier ``a`` with a later ``b``: for a two-token
    sequence, C = a₁ ⊗ b₂ exactly (no self-pair, no reversed pair)."""
    mix = NilpotentLieScan(dim=8, lift_dim=3)
    x = torch.randn(1, 2, 8)
    a = mix.a_lift(x)
    b = mix.b_lift(x)
    _, _, c = mix.group_element(x)
    expected = a[0, 0].unsqueeze(-1) * b[0, 1].unsqueeze(-2)
    assert torch.allclose(c[0], expected, atol=1e-6)


def test_causality_future_tokens_do_not_leak() -> None:
    mix = _nudged()
    x = torch.randn(1, 10, 16)
    y = mix(x)
    x_perturbed = x.clone()
    x_perturbed[0, 7:] = torch.randn(3, 16)
    y_perturbed = mix(x_perturbed)
    assert torch.allclose(y[0, :7], y_perturbed[0, :7], atol=1e-6)
    assert not torch.allclose(y[0, 7:], y_perturbed[0, 7:], atol=1e-3)


def test_mixing_law_has_zero_learned_parameters() -> None:
    """Every trainable parameter lives in the lifts/readout; the group law and
    scan carry none."""
    mix = NilpotentLieScan(dim=16, lift_dim=4)
    names = {name.split(".")[0] for name, _ in mix.named_parameters()}
    assert names == {"a_lift", "b_lift", "readout"}


def test_readout_normalization_keeps_long_sequences_bounded() -> None:
    """Raw signatures grow polynomially; the normalized readout features stay
    O(1) at 32× lengths so the op is trainable without decay hacks."""
    mix = _nudged()
    x = torch.randn(1, 1024, 16)
    y = mix(x)
    assert torch.isfinite(y).all()
    assert y.abs().max() < 1e2


def test_backward_flows_to_all_parameters() -> None:
    mix = _nudged()
    x = torch.randn(2, 9, 16, requires_grad=True)
    mix(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in mix.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert param.grad.abs().sum() > 0, name


def test_param_count() -> None:
    d, k = 32, 8
    mix = NilpotentLieScan(dim=d, lift_dim=k)
    expected = 2 * k * d + (k * k + 2 * k) * d
    assert lie_scan_param_count(d, k) == expected
    assert mix.num_parameters == expected
    assert expected == sum(p.numel() for p in mix.parameters())


def test_invalid_configs_fail_fast() -> None:
    with pytest.raises(ValueError):
        NilpotentLieScan(dim=0)
    with pytest.raises(ValueError):
        NilpotentLieScan(dim=8, lift_dim=0)


def test_measurable_by_physics_descriptor_probe() -> None:
    """NM-10: finite physics fingerprint so the mixer is scorable on the
    geometric-novelty axis alongside the other synthesis operators."""
    probe = PhysicsDescriptorProbe(batch=2, seq_len=8, dim=16, n_seeds=2)
    mix = _nudged()
    desc = probe.describe_operator(mix)
    assert desc, "probe returned no descriptors"
    for key, value in desc.items():
        assert isinstance(value, float) and math.isfinite(value), f"{key}={value}"
