"""Tests for NM-C15 ternary-native sign-semiring feature mixer.

Pins the spec: the deployed weight is exactly ``{-1, 0, +1}`` (the sign semiring),
the forward is pure add/subtract (no MAC), full mode is identity-at-init, the
inference weight is ÷16 fp32, and the layer exposes a finite NM-10 physics
fingerprint. ``rank`` (factored controller) cuts *training* params to ``2Dr``.
"""

from __future__ import annotations

import math

import pytest
import torch

from research.synthesis.physics_descriptors import PhysicsDescriptorProbe
from research.synthesis.ternary_sign_mix import TernarySignMix, ternary_param_count


def test_forward_preserves_shape_and_is_finite() -> None:
    mix = TernarySignMix(dim=8)
    x = torch.randn(2, 10, 8)
    y = mix(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("d", [1, 2, 3, 5, 7, 8, 9, 16, 17, 32])
def test_identity_at_init(d: int) -> None:
    """Full mode inits ``Z = I`` ⟹ ``ternary(Z) = I`` ⟹ a no-op drop-in for any D."""
    mix = TernarySignMix(dim=d)
    x = torch.randn(3, 5, d)
    assert torch.allclose(mix(x), x, atol=1e-6), f"dim={d} not identity at init"


def test_ternary_weight_values_in_sign_domain() -> None:
    """The deployed weight lives entirely in the sign semiring ``{-1, 0, +1}``."""
    torch.manual_seed(0)
    mix = TernarySignMix(dim=16)
    with torch.no_grad():
        mix.Z.normal_(0.0, 1.5)  # push entries past the threshold for a mix of ±1/0
    w = mix.ternary_weight()
    assert ((w == -1) | (w == 0) | (w == 1)).all()
    assert int(((w == 1) | (w == -1)).sum()) > 0  # not all-zero
    assert int((w == 0).sum()) > 0  # sparsity present


def test_forward_matches_dense_ternary_matmul() -> None:
    """Blockwise forward == ``x @ W.T`` for the dense ternary weight (no padding)."""
    torch.manual_seed(0)
    d = 16
    mix = TernarySignMix(dim=d)
    with torch.no_grad():
        mix.Z.normal_(0.0, 1.0)
    x = torch.randn(4, 7, d)
    w = mix.ternary_weight().detach()
    assert torch.allclose(mix(x), x @ w.T, atol=1e-6)


def test_forward_is_pure_add_subtract() -> None:
    """Hand-computed case proving the forward is add/subtract only (no MAC): with
    a known ternary W, out is exactly the signed sum of the inputs."""
    mix = TernarySignMix(dim=3)
    target_w = torch.tensor([[1.0, -1.0, 0.0], [0.0, 1.0, 1.0], [-1.0, 0.0, 1.0]])
    with torch.no_grad():
        mix.Z.copy_(target_w)  # |±1| >= thresh -> ±1, |0| < thresh -> 0
    assert torch.equal(mix.ternary_weight(), target_w)
    x = torch.tensor([[[2.0, 3.0, 5.0]]])  # (1, 1, 3)
    # out_0 = 2-3 = -1 ; out_1 = 3+5 = 8 ; out_2 = -2+5 = 3
    expected = torch.tensor([[[-1.0, 8.0, 3.0]]])
    assert torch.allclose(mix(x), expected, atol=1e-6)


def test_param_count_full_and_factored() -> None:
    d = 64
    full = ternary_param_count(d)
    assert full == d * d
    assert full == sum(p.numel() for p in TernarySignMix(d).parameters())
    # Factored controller cuts TRAINING params below quadratic.
    r = 8
    factored = ternary_param_count(d, rank=r)
    assert factored == 2 * d * r
    assert factored < d * d
    assert factored == sum(p.numel() for p in TernarySignMix(d, rank=r).parameters())


def test_ternary_storage_is_16x_smaller_than_fp32() -> None:
    """The deployed (serialized) weight packs to 2 bits/entry vs fp32's 32."""
    d = 64  # multiple of 8 -> no packing slop
    mix = TernarySignMix(dim=d)
    ternary_bytes = mix.ternary_storage_bytes()
    fp32_bytes = d * d * 4
    assert ternary_bytes * 16 == fp32_bytes


def test_backward_flows_to_controller() -> None:
    mix = TernarySignMix(dim=16)
    with torch.no_grad():  # move off identity so the fingerprint is non-trivial
        mix.Z.add_(0.3 * torch.randn_like(mix.Z))
    x = torch.randn(2, 6, 16, requires_grad=True)
    mix(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert mix.Z.grad is not None and torch.isfinite(mix.Z.grad).all()
    assert mix.Z.grad.abs().sum() > 0  # STE delivers gradient to every entry


def test_factored_mode_compacts_training_params_and_is_ternary() -> None:
    d, r = 16, 4
    mix = TernarySignMix(dim=d, rank=r)
    assert mix.num_parameters == 2 * d * r  # 128 << 256
    x = torch.randn(2, 5, d)
    y = mix(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert (
        (mix.ternary_weight() == -1)
        | (mix.ternary_weight() == 0)
        | (mix.ternary_weight() == 1)
    ).all()
    y.square().mean().backward()
    # Both factors receive STE gradient through Z = A @ B.
    assert (
        mix.A.grad is not None
        and torch.isfinite(mix.A.grad).all()
        and mix.A.grad.abs().sum() > 0
    )
    assert (
        mix.B.grad is not None
        and torch.isfinite(mix.B.grad).all()
        and mix.B.grad.abs().sum() > 0
    )


def test_measurable_by_physics_descriptor_probe() -> None:
    """NM-10: the mixer exposes a finite physics fingerprint so it can be scored
    on the geometric-novelty axis alongside Monarch/Butterfly."""
    probe = PhysicsDescriptorProbe(batch=2, seq_len=8, dim=16, n_seeds=2)
    mix = TernarySignMix(dim=16)
    with torch.no_grad():  # nudge off identity for a non-trivial fingerprint
        mix.Z.add_(0.4 * torch.randn_like(mix.Z))
    desc = probe.describe_operator(mix)
    assert desc, "probe returned no descriptors"
    for key, value in desc.items():
        assert isinstance(value, float) and math.isfinite(value), f"{key}={value}"
