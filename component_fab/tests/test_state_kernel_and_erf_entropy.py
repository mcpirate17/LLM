"""Tests for the state-kernel dispatcher and ERF entropy path."""

from __future__ import annotations

import torch
from torch import nn

from component_fab.generator.code_generator import generate_module
from component_fab.generator.primitive_templates import LinearStateSpaceLane
from component_fab.harness.erf_probe import measure_erf


def _state_kernel_axes() -> dict:
    """Canonical state-kernel-swap proposal axes (non-tropical)."""
    return {
        "op_algebraic_space": "euclidean",
        "op_dynamical_has_state": 1,
        "op_dynamical_memory_length_class": "O(L)",
        "op_activation_sparsity_pattern": "dense",
        "op_geometric_receptive_field": "global",
    }


def test_generate_module_dispatches_state_kernel_for_euclidean_state() -> None:
    module = generate_module(_state_kernel_axes(), dim=32)
    assert isinstance(module, LinearStateSpaceLane)


def test_generate_module_keeps_tropical_state_routing() -> None:
    """Tropical state-bearing axes must still hit TropicalStateSpace, not the
    generic LinearStateSpaceLane added in the same dispatcher."""
    from component_fab.generator.primitive_templates import TropicalStateSpace

    axes = _state_kernel_axes()
    axes["op_algebraic_space"] = "tropical"
    module = generate_module(axes, dim=32)
    assert isinstance(module, TropicalStateSpace)


def test_linear_state_space_lane_forward_finite() -> None:
    module = LinearStateSpaceLane(dim=16)
    x = torch.randn(2, 12, 16, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all().item()
    g = torch.autograd.grad(y.sum(), x)[0]
    assert torch.isfinite(g).all().item()


def test_erf_entropy_separates_real_mixer_from_per_position_op() -> None:
    """Entropy must give an order-of-magnitude signal between a true mixer
    and a per-position lane — the discriminating axis density alone misses
    because both fail the residual-peak-biased density at small init."""

    class PerPositionWithResidual(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.lin = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x) + x

    per_pos = measure_erf(PerPositionWithResidual(32), seq_len=32, dim=32)
    mixer = measure_erf(LinearStateSpaceLane(32), seq_len=32, dim=32)
    # Per-position residual lane: entropy is essentially zero (one peak).
    assert abs(per_pos.density_entropy) < 0.05
    # State mixer: entropy clearly above the 0.10 threshold.
    assert mixer.density_entropy > 0.10
    # Density-only gate kills both; entropy path admits the mixer.
    assert per_pos.passed is False
    assert mixer.passed is True


def test_measure_erf_entropy_in_zero_path() -> None:
    """When forward fails or gradient is zero, density_entropy=0 is returned."""

    class Zero(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    result = measure_erf(Zero(), seq_len=8, dim=8)
    assert result.density == 0.0
    assert result.density_entropy == 0.0
    assert result.passed is False
