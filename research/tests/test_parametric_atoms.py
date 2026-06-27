"""Parametric atoms must be (1) pass-through at init and (2) steer a physics axis.

Part (2) is the whole point: each atom's knob is the steering wheel that moves a
specific coordinate of the physics fingerprint, so the discovery loop can aim at
an empty niche instead of picking a named mechanism.
"""

from __future__ import annotations

import pytest
import torch

from research.synthesis.parametric_atoms import (
    ATOM_KINDS,
    AtomSpec,
    ParametricBasis,
    ParametricNorm,
    ParametricScan,
    build_atom_stack,
    enumerate_atom_specs,
)
from research.synthesis.physics_descriptors import (
    perm_equivariance,
    scale_homogeneity,
)


def _x(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(4, 16, 8, generator=g)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b).norm() / (b.norm() + 1e-9)).detach())


# ── identity at init ─────────────────────────────────────────────────
@pytest.mark.parametrize("make", [ParametricNorm, ParametricBasis, ParametricScan])
def test_atom_is_passthrough_at_init(make) -> None:
    x = _x()
    atom = make(x.shape[-1])
    assert _rel(atom(x), x) < 1e-3


def test_atom_stack_is_passthrough_at_init() -> None:
    x = _x()
    for spec in enumerate_atom_specs(max_depth=2):
        stack = build_atom_stack(x.shape[-1], spec)
        assert _rel(stack(x), x) < 1e-3, f"stack {spec.key} not identity at init"


@pytest.mark.parametrize("make", [ParametricNorm, ParametricBasis, ParametricScan])
def test_atom_is_finite_and_trainable(make) -> None:
    x = _x().requires_grad_(True)
    atom = make(x.shape[-1])
    y = atom(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.square().mean().backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in atom.parameters()
    )


# ── knob steers the intended physics coordinate ──────────────────────
def test_norm_knob_lowers_scale_homogeneity() -> None:
    x = _x()
    norm = ParametricNorm(x.shape[-1])
    base = scale_homogeneity(norm, x)
    with torch.no_grad():
        norm.blend_logit.fill_(6.0)  # open the normalization
    opened = scale_homogeneity(norm, x)
    assert base > 0.99  # pass-through is linear
    assert opened < base - 0.05  # normalization divides out scale


def test_token_basis_knob_lowers_perm_equivariance() -> None:
    x = _x()
    perm = torch.randperm(x.shape[1])
    basis = ParametricBasis(x.shape[-1], axis="token")
    base = perm_equivariance(basis, x, perm)
    with torch.no_grad():
        basis.mix_logit.fill_(6.0)  # rotate tokens into the fixed basis
    opened = perm_equivariance(basis, x, perm)
    assert base > 0.99  # pass-through commutes with permutation
    assert opened < base - 0.05  # fixed token mixing breaks it


def test_scan_knob_introduces_order_dependence() -> None:
    x = _x()
    perm = torch.randperm(x.shape[1])
    scan = ParametricScan(x.shape[-1])
    base = perm_equivariance(scan, x, perm)
    with torch.no_grad():
        scan.gate_logit.fill_(6.0)  # open the causal scan
        scan.log_decay.fill_(2.0)  # long memory
    opened = perm_equivariance(scan, x, perm)
    assert base > 0.99  # pass-through is order-free
    assert opened < base - 0.05  # causal state is order-dependent


# ── spec validation fails loud ───────────────────────────────────────
def test_bad_atom_kind_fails_loud() -> None:
    with pytest.raises(ValueError, match="unknown atom kind"):
        AtomSpec(kinds=("norm", "not_an_atom"))


def test_enumerate_covers_identity_and_singletons() -> None:
    specs = enumerate_atom_specs(max_depth=1)
    keys = {s.key for s in specs}
    assert "identity" in keys
    assert {k for k in ATOM_KINDS} <= keys
