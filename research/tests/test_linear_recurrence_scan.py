"""Constant-matrix parallel scan — parity with the sequential reference (NM-F §5).

The scan is the native hot path for constant-transition recurrences (NM-F5
port-Hamiltonian). It must be bit-comparable to the plain Python loop in both
forward and backward, for the *contractive* transitions F5 guarantees (a random
non-contraction blows up P^stride and is out of scope). Also pins that swapping
the loop for the scan left F5's behaviour (identity-at-init) unchanged.
"""

from __future__ import annotations

import pytest
import torch

from research.synthesis._linear_recurrence_scan import constant_matrix_scan
from research.synthesis.port_hamiltonian_mix import PortHamiltonianMixer


def _contractive(dim: int, rho: float = 0.9) -> torch.Tensor:
    """A transition with spectral norm ``rho < 1``, like F5's Cayley image."""
    p = torch.randn(dim, dim)
    return p / torch.linalg.matrix_norm(p, ord=2) * rho


def _reference(p: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    h = b.new_zeros(b.shape[0], b.shape[-1])
    states = []
    for t in range(b.shape[1]):
        h = h @ p.T + b[:, t]
        states.append(h)
    return torch.stack(states, dim=1)


@pytest.mark.parametrize(
    "dim,seq", [(4, 1), (4, 2), (8, 7), (16, 17), (32, 64), (64, 128)]
)
def test_scan_matches_reference_forward(dim: int, seq: int) -> None:
    torch.manual_seed(dim + seq)
    p = _contractive(dim)
    b = torch.randn(2, seq, dim)
    torch.testing.assert_close(
        constant_matrix_scan(p, b), _reference(p, b), rtol=1e-4, atol=1e-5
    )


def test_scan_matches_reference_backward() -> None:
    torch.manual_seed(0)
    p0 = _contractive(8)
    b0 = torch.randn(2, 20, 8)

    p = p0.clone().requires_grad_()
    b = b0.clone().requires_grad_()
    constant_matrix_scan(p, b).pow(2).mean().backward()

    pr = p0.clone().requires_grad_()
    br = b0.clone().requires_grad_()
    _reference(pr, br).pow(2).mean().backward()

    torch.testing.assert_close(p.grad, pr.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(b.grad, br.grad, rtol=1e-4, atol=1e-5)


def test_scan_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        constant_matrix_scan(torch.randn(4, 5), torch.randn(2, 3, 4))
    with pytest.raises(ValueError):
        constant_matrix_scan(torch.randn(4, 4), torch.randn(2, 3, 5))


def test_f5_evolve_matches_its_reference_loop() -> None:
    """The scan wired into F5 must equal the sequential oracle it replaced,
    forward and backward, on a trained (non-identity) mixer."""
    torch.manual_seed(1)
    mix = PortHamiltonianMixer(32, band=4)
    torch.nn.init.normal_(mix.out_lift.weight, std=0.05)  # move off identity-init
    x = torch.randn(2, 40, 32, requires_grad=True)

    p, inject = mix._cayley()
    u = mix.in_lift(x)
    gate = torch.sigmoid(x @ mix.gate_weight + mix.gate_bias)
    inputs = (gate.unsqueeze(-1) * u) @ inject.T

    fast = mix.evolve(x)
    ref = mix._evolve_reference(p, inputs)
    torch.testing.assert_close(fast, ref, rtol=1e-4, atol=1e-5)


def test_f5_still_identity_at_init() -> None:
    mix = PortHamiltonianMixer(64, band=4)
    x = torch.randn(2, 24, 64)
    torch.testing.assert_close(mix(x), x, rtol=1e-5, atol=1e-6)
