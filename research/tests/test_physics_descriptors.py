"""Physics descriptors must separate operators by their symmetry class.

The point of the descriptors is name-free discovery: a pointwise op, a
shift-equivariant mixer, and a position-aware mixer must land in different
regions of the fingerprint regardless of what they are called.
"""

from __future__ import annotations

import pytest
import torch

from research.synthesis.physics_descriptors import (
    PHYSICS_DESCRIPTOR_NAMES,
    PhysicsDescriptorProbe,
    energy_gain,
    perm_equivariance,
    physics_behavior_axes,
    scale_homogeneity,
    shift_equivariance,
    spectral_radius,
)


def _x(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(4, 16, 8, generator=g)


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _pointwise(x: torch.Tensor) -> torch.Tensor:
    # Channel/pointwise: commutes with any token permutation or shift.
    return torch.tanh(x)


def _neighbor_mix(x: torch.Tensor) -> torch.Tensor:
    # Linear, translation-equivariant (circular), but NOT permutation-equivariant.
    return x + torch.roll(x, shifts=1, dims=1)


def _position_aware(x: torch.Tensor) -> torch.Tensor:
    # Adds an absolute-position ramp: breaks BOTH permutation and shift equivariance.
    length = x.shape[1]
    ramp = torch.linspace(0, 1, length).reshape(1, length, 1)
    return x + ramp


# ── permutation equivariance separates set-like from position-aware ──
def test_pointwise_is_permutation_equivariant() -> None:
    x = _x()
    perm = torch.randperm(x.shape[1])
    assert perm_equivariance(_pointwise, x, perm) > 0.97
    assert perm_equivariance(_identity, x, perm) > 0.99


def test_token_mixer_breaks_permutation_equivariance() -> None:
    x = _x()
    perm = torch.randperm(x.shape[1])
    assert perm_equivariance(_neighbor_mix, x, perm) < 0.9
    assert perm_equivariance(_pointwise, x, perm) > perm_equivariance(
        _neighbor_mix, x, perm
    )


# ── shift equivariance separates convolutional from absolute-position ──
def test_circular_mixer_is_shift_equivariant() -> None:
    x = _x()
    # neighbor_mix uses circular roll -> commutes with circular shift.
    assert shift_equivariance(_neighbor_mix, x, 3) > 0.97
    # an absolute-position op does not.
    assert shift_equivariance(_position_aware, x, 3) < 0.95


# ── scale homogeneity separates linear from nonlinear ──
def test_scale_homogeneity_flags_nonlinearity() -> None:
    x = _x()
    assert scale_homogeneity(_neighbor_mix, x) > 0.97  # linear
    assert scale_homogeneity(_pointwise, x) < 0.95  # saturating nonlinearity


# ── energy / spectral radius are physically sane on knowns ──
def test_energy_gain_and_spectral_radius_on_identity() -> None:
    x = _x()
    assert energy_gain(_identity, x) == pytest.approx(1.0, abs=1e-4)
    assert spectral_radius(_identity, x) == pytest.approx(1.0, rel=0.1)


def test_scaled_operator_has_matching_spectral_radius() -> None:
    x = _x()
    assert energy_gain(lambda t: 0.5 * t, x) == pytest.approx(0.5, abs=1e-4)
    assert spectral_radius(lambda t: 0.5 * t, x) == pytest.approx(0.5, rel=0.15)


def test_fail_loud_on_wrong_rank() -> None:
    with pytest.raises(ValueError, match=r"\[B, L, D\]"):
        energy_gain(_identity, torch.randn(4, 8))


# ── probe + QD axes integration ──
def test_probe_returns_all_descriptors_and_separates_operators() -> None:
    probe = PhysicsDescriptorProbe(batch=4, seq_len=16, dim=8, n_seeds=2)
    point = probe.describe_operator(_pointwise)
    mixer = probe.describe_operator(_neighbor_mix)
    assert set(point) == set(PHYSICS_DESCRIPTOR_NAMES)
    # The two operators occupy different symmetry regions.
    assert point["perm_equivariance"] > mixer["perm_equivariance"]
    assert mixer["scale_homogeneity"] > point["scale_homogeneity"]


def test_physics_axes_bin_the_fingerprint() -> None:
    axes = physics_behavior_axes()
    names = {a.name for a in axes}
    assert {"perm_equivariance", "shift_equivariance", "spectral_radius"} <= names
    # Every axis name a descriptor the probe actually produces.
    assert names <= set(PHYSICS_DESCRIPTOR_NAMES)
