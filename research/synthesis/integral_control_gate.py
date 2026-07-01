"""NM-F4 — Internal-model integral controller: retention with zero steady-state error.

A causal ``[B, L, D] -> [B, L, D]`` sequence mixer whose state-holding law is the
**internal model principle** from control theory: a loop can track a reference
class forever only if it embeds a model of that class — for constant references
(a bound fact that must be HELD) the required internal model is an integrator.
So the recurrence is an error-driven diagonal integral controller:

    e_t = u_t − d̂ ⊙ s_{t−1}            (readout prediction error)
    s_t = s_{t−1} + k_i ⊙ g_t · e_t     (integral action, anti-windup clamped)
    y_t = s_t + k_p ⊙ e_t               (+ proportional path for fast transients)

with ``u_t`` a lifted input, ``d̂`` the diagonal readout model, ``k_i > 0``
(softplus-parameterized) the integral gains, ``k_p`` the proportional gains, and
``g_t`` a sigmoid-highway write-relevance gate (the validated non-twin form — no
normalization across positions or slots anywhere; NON-softmax by construction).

Why this beats decay-by-parameterization: every EMA/decay recurrence forgets
geometrically *because of its parameterization* — holding a fact for 1024 steps
costs it precision or parameters. Here the integrator holds bound state
**indefinitely** once the gate closes (``g ≈ 0 ⟹ s_t = s_{t−1}`` exactly), and
while the gate is open the integral action drives the readout error to zero
geometrically (zero steady-state error for ``0 < k_i·g·d̂ < 2``). This is the
cheapest structural fix for the retrieval gap the p-adic 100k scale run exposed
(capability floored while ppl was fine — the single-pass gate had no pathway that
*holds* associations): retention is a theorem of the loop shape, not a learned
behavior. Probe: retention flatness — bind, insert 16→1024 distractors, query;
accuracy-vs-gap must be FLAT where decay recurrences fall off exponentially, with
randomized query positions and distractor content (no positional shortcut).

Parameter efficiency. The control law is **O(D)**: gains ``k_i, k_p``, readout
``d̂``, gate weights — ``4·D + 1`` params. The ``D×D`` input/output lifts wrap it
(identity-init and zero-init respectively ⟹ **identity-at-init** overall);
``control_param_count`` reports the O(D) core separately so
capability-per-non-embedding-param is measured against the mechanism, not the
plumbing. Anti-windup: the classical integrator failure is windup on long noisy
streams — a per-channel clamp bounds ``|s|`` (and doubles as the fail-safe when
learned ``k_i·d̂`` leaves the stable band); ``anti_windup=False`` exists ONLY so
the ablation proving the clamp is real stays runnable.

The recurrence is sequential over ``L`` (the clamp is deliberately non-associative
— that is what makes windup impossible mid-sequence); this torch loop is the
probe-scale reference and a native scan is the production hot path. Self-contained
on purpose — imports only ``torch`` so it is measurable by
``PhysicsDescriptorProbe`` (NM-10-scorable). Registry wiring deferred per the
NM-C3/C5/C15 convention. Lane: ``tasks/nm_f_operator_families_2026-07-01.md``.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_KI_INIT = 0.1  # inside the stable band 0 < k_i·g·d̂ < 2 at init (d̂ = 1, g ≤ 1)


def control_param_count(dim: int) -> int:
    """The O(D) control core: ``k_i`` + ``k_p`` + ``d̂`` + gate weights + gate bias."""
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    return 4 * dim + 1


def integral_param_count(dim: int) -> int:
    """Total trainable params: O(D) control core + the two ``D×D`` lifts."""
    return control_param_count(dim) + 2 * dim * dim


class IntegralControlMixer(nn.Module):
    """Error-driven diagonal integral controller with anti-windup."""

    def __init__(
        self,
        dim: int,
        *,
        anti_windup: bool = True,
        s_max: float = 10.0,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if s_max <= 0.0:
            raise ValueError(f"s_max must be > 0, got {s_max}")
        self.d = dim
        self.anti_windup = anti_windup
        self.s_max = float(s_max)
        # Identity-init input lift: u = x at init, so the control loop sees the
        # token stream directly before training reshapes it.
        self.in_lift = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.in_lift.weight.copy_(torch.eye(dim))
        # Zero-init output lift ⟹ forward(x) == x at init (identity-at-init).
        self.out_lift = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.out_lift.weight)
        # O(D) control core. k_i = softplus(raw) > 0 always.
        self.raw_ki = nn.Parameter(torch.full((dim,), math.log(math.expm1(_KI_INIT))))
        self.k_p = nn.Parameter(torch.zeros(dim))
        self.d_hat = nn.Parameter(torch.ones(dim))
        # Sigmoid-highway write-relevance gate (validated non-twin form).
        self.gate_weight = nn.Parameter(torch.zeros(dim))
        self.gate_bias = nn.Parameter(torch.zeros(1))

    @property
    def num_parameters(self) -> int:
        return integral_param_count(self.d)

    @property
    def control_parameters(self) -> int:
        return control_param_count(self.d)

    def integral_gain(self) -> torch.Tensor:
        """``k_i = softplus(raw) > 0`` — integral action never changes sign."""
        return F.softplus(self.raw_ki)

    def integrate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the control loop; returns ``(states, errors)`` each ``(B, L, D)``.

        The verification surface for the retention/steady-state claims: tests
        assert on the raw trajectories, not on the lifted output.
        """
        u = self.in_lift(x)
        gate = torch.sigmoid(x @ self.gate_weight + self.gate_bias)  # (B, L)
        k_i = self.integral_gain()
        s = x.new_zeros(x.shape[0], self.d)
        states = []
        errors = []
        for t in range(x.shape[1]):
            e = u[:, t] - self.d_hat * s
            s = s + k_i * gate[:, t : t + 1] * e
            if self.anti_windup:
                s = s.clamp(-self.s_max, self.s_max)
            states.append(s)
            errors.append(e)
        return torch.stack(states, dim=1), torch.stack(errors, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, L, D) -> (B, L, D)``: residual + lifted integral + proportional paths."""
        states, errors = self.integrate(x)
        return x + self.out_lift(states + self.k_p * errors)
