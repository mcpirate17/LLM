"""NM-F5 — Port-Hamiltonian mixer: collapse-proof state transport as a theorem.

A causal ``[B, L, D] -> [B, L, D]`` sequence mixer whose recurrence is constrained
to the **port-Hamiltonian normal form** from dissipativity theory:

    ḣ = (J − R) ∇H(h) + B·u ,   J = −Jᵀ (banded skew),  R ⪰ 0 (diagonal),
    H(h) = ½ Σ λ_i h_i²  with λ_i > 0 (learned convex energy metric)

discretized by the **Cayley / implicit-midpoint transform**. Because the matrix
``M̃ = Λ^{1/2}(J − R)Λ^{1/2}`` has symmetric part ``−Λ^{1/2}RΛ^{1/2} ⪯ 0``, its
Cayley image ``(I − τ/2·M̃)^{-1}(I + τ/2·M̃)`` is a contraction in spectral norm —
a THEOREM, for every parameter value the optimizer can ever reach. Consequences,
all certified by construction rather than learned:

  * **No blow-up at any length**: the state transition is a contraction in the
    energy norm ``H``; activations are bounded at 32× train length
    unconditionally. This is the principled successor to the p-adic gate's
    empirical collapse-proofing — collapse-proof as algebra, not as a fixed bug.
  * **Conservative transport**: the skew part ``J`` circulates state without
    decaying it (at ``R → 0`` the Cayley map is exactly orthogonal in the energy
    metric — energy is conserved to machine precision). Information is carried by
    rotation, not by fighting an EMA's forgetting — the structural complement of
    NM-F4's integrator.
  * **Dissipation is a separate, inspectable knob**: ``R`` (diagonal, softplus ≥
    0) is the only forgetting pathway, structurally separated from transport.

Honest baseline note (the mission's currency): diagonal-stable SSMs (S4/LRU-style
``|λ| ≤ 1`` parameterizations) are the *baseline to beat*, not the design. This
op is structurally distinct: NON-diagonal banded skew coupling (genuinely rotating
state across channels — non-normal dynamics a diagonal recurrence cannot express),
a learned energy metric ``Λ`` defining *which* norm contracts, and a passivity
certificate that is testable in-suite (``energy_contraction_norm() ≤ 1``). Probe:
length-extrapolated state tracking (train parity/mod-counter at 128, eval at
4096) — a lane where non-QKV mechanisms already beat attention. NON-softmax
throughout: the input gate is a sigmoid highway (validated non-twin form); there
is no normalization across positions or slots anywhere.

Since ``J, R, Λ`` are time-invariant, the Cayley transition ``P`` and input
injection are computed ONCE per forward (one banded-structure solve, O(D³) once,
amortized over ``B·L``); the recurrence itself is a plain matrix loop — the
probe-scale reference, with a native scan as the production hot path. Learned
parameters: banded skew (~``band·D``), gains/metric (``2·D``), gate (``D + 1``),
plus the two ``D×D`` lifts (input identity-init, output zero-init ⟹
**identity-at-init**). Self-contained on purpose — imports only ``torch`` so it
is measurable by ``PhysicsDescriptorProbe`` (NM-10-scorable). Registry wiring
deferred per the NM-C3/C5/C15 convention.
Lane: ``tasks/nm_f_operator_families_2026-07-01.md``.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def ph_param_count(dim: int, band: int) -> int:
    """Banded skew (``Σ_{j=1..band} (D−j)``) + ``r``/``λ`` (``2D``) + gate
    (``D+1``) + input/output lifts (``2D²``)."""
    if dim < 2 or band < 1 or band >= dim:
        raise ValueError(
            f"need dim >= 2 and 1 <= band < dim, got dim={dim}, band={band}"
        )
    skew = band * dim - band * (band + 1) // 2
    return skew + 2 * dim + dim + 1 + 2 * dim * dim


class PortHamiltonianMixer(nn.Module):
    """Cayley-discretized port-Hamiltonian recurrence with certified contraction."""

    def __init__(self, dim: int, *, band: int = 4, tau: float = 0.5) -> None:
        super().__init__()
        if dim < 2 or band < 1 or band >= dim:
            raise ValueError(
                f"need dim >= 2 and 1 <= band < dim, got dim={dim}, band={band}"
            )
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {tau}")
        self.d = dim
        self.band = band
        self.tau = float(tau)
        skew_size = band * dim - band * (band + 1) // 2
        self.skew_flat = nn.Parameter(0.1 * torch.randn(skew_size))
        self.raw_r = nn.Parameter(torch.full((dim,), math.log(math.expm1(0.1))))
        self.raw_lam = nn.Parameter(torch.zeros(dim))  # softplus(0) ≈ 0.693
        self.in_lift = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.in_lift.weight.copy_(torch.eye(dim))
        # Zero-init output lift ⟹ forward(x) == x at init (identity-at-init).
        self.out_lift = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.out_lift.weight)
        # Sigmoid-highway input gate (validated non-twin form), O(D) params.
        self.gate_weight = nn.Parameter(torch.zeros(dim))
        self.gate_bias = nn.Parameter(torch.zeros(1))

    @property
    def num_parameters(self) -> int:
        return ph_param_count(self.d, self.band)

    def damping(self) -> torch.Tensor:
        """``r = softplus(raw) ≥ 0`` — the only forgetting pathway (R ⪰ 0)."""
        return F.softplus(self.raw_r)

    def energy_metric(self) -> torch.Tensor:
        """``λ = softplus(raw) > 0`` — the learned convex energy metric."""
        return F.softplus(self.raw_lam)

    def skew_matrix(self) -> torch.Tensor:
        """``J = S − Sᵀ`` from the banded superdiagonal parameters — exactly skew
        for every parameter value."""
        pieces = torch.split(
            self.skew_flat, [self.d - j for j in range(1, self.band + 1)]
        )
        s = self.skew_flat.new_zeros(self.d, self.d)
        for j, diag in enumerate(pieces, start=1):
            s = s + torch.diag_embed(diag, offset=j)
        return s - s.T

    def energy(self, h: torch.Tensor) -> torch.Tensor:
        """``H(h) = ½ Σ λ_i h_i²`` — the certified-nonincreasing storage function."""
        return 0.5 * (self.energy_metric() * h * h).sum(dim=-1)

    def _cayley(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transition ``P = Cayley(τ·M)`` and input injection ``τ(I − τ/2·M)^{-1}``
        for ``M = (J − R)Λ``. ``I − τ/2·M`` is provably invertible (the field of
        values of ``M̃`` lies in the closed left half-plane)."""
        m = (self.skew_matrix() - torch.diag(self.damping())) * self.energy_metric()
        eye = torch.eye(self.d, device=m.device, dtype=m.dtype)
        lhs = eye - (self.tau / 2.0) * m
        rhs = torch.cat([eye + (self.tau / 2.0) * m, self.tau * eye], dim=1)
        solved = torch.linalg.solve(lhs, rhs)
        return solved[:, : self.d], solved[:, self.d :]

    def energy_contraction_norm(self) -> torch.Tensor:
        """Spectral norm of the transition in energy coordinates
        ``Λ^{1/2} P Λ^{-1/2}`` — the certificate; ≤ 1 for ALL parameter values."""
        p, _ = self._cayley()
        root = self.energy_metric().sqrt()
        return torch.linalg.matrix_norm(root.unsqueeze(1) * p / root, ord=2)

    def evolve(self, x: torch.Tensor) -> torch.Tensor:
        """Run the recurrence; returns states ``(B, L, D)`` (the verification
        surface for the dissipation/conservation/boundedness claims)."""
        p, inject = self._cayley()
        u = self.in_lift(x)
        gate = torch.sigmoid(x @ self.gate_weight + self.gate_bias)  # (B, L)
        h = x.new_zeros(x.shape[0], self.d)
        states = []
        for t in range(x.shape[1]):
            h = h @ p.T + (gate[:, t : t + 1] * u[:, t]) @ inject.T
            states.append(h)
        return torch.stack(states, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, L, D) -> (B, L, D)``: residual + lifted port-Hamiltonian state."""
        return x + self.out_lift(self.evolve(x))
