"""Tests for NM-F5 port-Hamiltonian mixer.

Pins the spec: exact skewness of J, the contraction CERTIFICATE (energy-norm
spectral radius ≤ 1 for arbitrary parameter values — the theorem, checked
numerically), monotone energy dissipation after an impulse, EXACT energy
conservation in the R→0 conservative limit (state circulates without decaying —
the anti-EMA claim), unconditional boundedness at 32× probe length,
identity-at-init, and a finite NM-10 physics fingerprint.
"""

from __future__ import annotations

import math

import pytest
import torch

from research.synthesis.physics_descriptors import PhysicsDescriptorProbe
from research.synthesis.port_hamiltonian_mix import (
    PortHamiltonianMixer,
    ph_param_count,
)


def _randomized(seed: int, dim: int = 16, band: int = 4) -> PortHamiltonianMixer:
    """A mixer with aggressively randomized dynamics parameters — the
    certificate must hold for ANY values the optimizer could reach."""
    torch.manual_seed(seed)
    mix = PortHamiltonianMixer(dim=dim, band=band)
    with torch.no_grad():
        mix.skew_flat.normal_(0.0, 2.0)
        mix.raw_r.normal_(0.0, 2.0)
        mix.raw_lam.normal_(0.0, 2.0)
    return mix


def test_forward_preserves_shape_and_is_finite() -> None:
    mix = PortHamiltonianMixer(dim=16)
    x = torch.randn(2, 10, 16)
    y = mix(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("d", [2, 8, 16, 33])
def test_identity_at_init(d: int) -> None:
    """Zero-init output lift ⟹ the mixer is an exact no-op drop-in at init."""
    mix = PortHamiltonianMixer(dim=d, band=1)
    x = torch.randn(3, 6, d)
    assert torch.allclose(mix(x), x, atol=1e-6), f"dim={d} not identity at init"


def test_j_is_exactly_skew_symmetric() -> None:
    mix = _randomized(0)
    j = mix.skew_matrix()
    assert torch.allclose(j, -j.T, atol=0.0)  # exact by construction
    assert torch.equal(torch.diagonal(j), torch.zeros(16))


def test_metric_and_damping_positive() -> None:
    mix = _randomized(1)
    assert (mix.energy_metric() > 0).all()
    assert (mix.damping() >= 0).all()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_contraction_certificate_for_arbitrary_parameters(seed: int) -> None:
    """THE theorem test: the Cayley transition is a contraction in the energy
    norm no matter what the parameters are — stability is not learned, it is
    parameterization."""
    mix = _randomized(seed)
    assert float(mix.energy_contraction_norm().detach()) <= 1.0 + 1e-5


def test_energy_dissipates_monotonically_after_impulse() -> None:
    """Passivity: with the input silent after an initial impulse, the storage
    function H(h_t) never increases."""
    mix = _randomized(2)
    with torch.no_grad():
        mix.gate_bias.fill_(20.0)  # gate open: the impulse actually enters
    x = torch.zeros(1, 64, 16)
    x[0, 0] = torch.randn(16)
    states = mix.evolve(x)
    energies = mix.energy(states[0])
    assert energies[0] > 0
    diffs = energies[1:] - energies[:-1]
    assert (diffs <= 1e-5).all(), "energy increased on a silent input"


def test_conservative_limit_circulates_without_decay() -> None:
    """R → 0: the Cayley map is orthogonal in the energy metric — after 512
    zero-input steps the energy is CONSERVED to numerical precision and the
    state has not decayed. An EMA/decay recurrence has lost essentially
    everything by then; here transport is rotation, not leakage."""
    mix = _randomized(3)
    with torch.no_grad():
        mix.raw_r.fill_(-30.0)  # softplus(-30) ≈ 1e-13: dissipation off
        mix.gate_bias.fill_(20.0)
    x = torch.zeros(1, 513, 16)
    x[0, 0] = torch.randn(16)
    states = mix.evolve(x)
    energies = mix.energy(states[0])
    assert torch.allclose(energies[-1], energies[0], rtol=1e-4)
    assert states[0, -1].abs().max() > 1e-3  # the state is still alive
    # The decay-by-parameterization baseline over the same horizon: gone.
    assert 0.99**512 < 0.01


def test_bounded_at_32x_length_under_persistent_input() -> None:
    """The collapse-proofing consequence: 1024 steps of persistent random input
    (32× the probe length) cannot blow the state up — bounded by contraction +
    dissipation, not by luck."""
    mix = _randomized(4)
    x = torch.randn(1, 1024, 16)
    states = mix.evolve(x)
    assert torch.isfinite(states).all()
    assert states.abs().max() < 1e3


def test_backward_flows_to_all_parameters() -> None:
    mix = PortHamiltonianMixer(dim=16)
    with torch.no_grad():  # move off identity so gradients are non-trivial
        mix.out_lift.weight.add_(0.3 * torch.randn_like(mix.out_lift.weight))
    x = torch.randn(2, 12, 16, requires_grad=True)
    mix(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in mix.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    assert mix.skew_flat.grad.abs().sum() > 0
    assert mix.raw_lam.grad.abs().sum() > 0


def test_param_count() -> None:
    d, band = 32, 4
    mix = PortHamiltonianMixer(dim=d, band=band)
    expected = (band * d - band * (band + 1) // 2) + 2 * d + d + 1 + 2 * d * d
    assert ph_param_count(d, band) == expected
    assert mix.num_parameters == expected
    assert expected == sum(p.numel() for p in mix.parameters())


def test_invalid_configs_fail_fast() -> None:
    with pytest.raises(ValueError):
        PortHamiltonianMixer(dim=1)
    with pytest.raises(ValueError):
        PortHamiltonianMixer(dim=8, band=8)  # band must be < dim
    with pytest.raises(ValueError):
        PortHamiltonianMixer(dim=8, tau=0.0)


def test_measurable_by_physics_descriptor_probe() -> None:
    """NM-10: finite physics fingerprint so the mixer is scorable on the
    geometric-novelty axis alongside the other synthesis operators."""
    probe = PhysicsDescriptorProbe(batch=2, seq_len=8, dim=16, n_seeds=2)
    mix = PortHamiltonianMixer(dim=16)
    with torch.no_grad():  # nudge off identity for a non-trivial fingerprint
        mix.out_lift.weight.add_(0.4 * torch.randn_like(mix.out_lift.weight))
    desc = probe.describe_operator(mix)
    assert desc, "probe returned no descriptors"
    for key, value in desc.items():
        assert isinstance(value, float) and math.isfinite(value), f"{key}={value}"
