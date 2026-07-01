"""NM-F3 — Nilpotent-Lie scan: sequence mixing by a truncated path signature.

A causal ``[B, L, D] -> [B, L, D]`` sequence mixer whose mixing law is the group
operation of a **step-2 nilpotent Lie group** (Heisenberg type). Each token is
lifted to a step increment ``(a_t, b_t) ∈ ℝ^k × ℝ^k``; the sequence state is the
running group product, whose composition is Chen's identity from rough-path
theory:

    (A₁, B₁, C₁) · (A₂, B₂, C₂) = (A₁+A₂, B₁+B₂, C₁+C₂ + A₁ ⊗ B₂)

so the per-token state is exactly the truncated **path signature**

    A_t = Σ_{s≤t} a_s ,   B_t = Σ_{s≤t} b_s ,   C_t = Σ_{s'<s≤t} a_{s'} ⊗ b_s .

Associativity is a THEOREM (Chen), not an approximation — the scan is an exact
semigroup, computable causally in linear time with plain cumulative sums, and the
level-2 cross term ``C_t`` accumulates **ordered** second-order statistics: a
strictly-earlier ``a`` paired with a later ``b`` — i.e. a key–value covariance
accumulator that knows which came first.

The mixing law carries **zero learned parameters.** What other architectures
spend parameter budget learning — order sensitivity — is given by the algebra:
two sequences that are permutations of each other (identical bag-of-tokens) have
identical ``A, B`` but different ``C``, so any sum/EMA-pooling mixer is at chance
on anagram discrimination *by construction* while the signature separates them
for free. Learned parameters are only the two lifts ``D→k`` and the zero-init
readout ``(k²+2k)→D`` (⟹ **identity-at-init**): ``2·k·D + (k²+2k)·D`` params —
at D=256/k=16 that is ~82K, with the entire sequence-mixing structure costing
none of it. Non-QKV and NON-softmax by construction: no pairwise scores, no
normalization over positions — state is one ``(k, k)`` matrix per stream.

Signature magnitude grows polynomially with length; the readout therefore
normalizes by the token count (level 1) and pair count (level 2). The
normalization is strictly readout-side — the scan itself stays the exact group
product, and the associativity test operates on the raw group elements. A
windowed/decayed variant is deliberately NOT provided here: decay would quietly
reduce the level-2 term toward an EMA, which is the failure mode the probe
(anagram discrimination + gMQAR with randomized queries) exists to catch.

Self-contained on purpose — imports only ``torch`` so it is measurable by
``PhysicsDescriptorProbe`` (NM-10-scorable). ``C`` is materialized per token
(``B·L·k²``) — fine at probe scale; a chunked native scan is the production hot
path. Registry wiring deferred per the NM-C3/C5/C15 convention.
Lane: ``tasks/nm_f_operator_families_2026-07-01.md``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

GroupElement = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def lie_scan_param_count(dim: int, lift_dim: int) -> int:
    """Two lifts (``2·k·D``) + readout (``(k²+2k)·D``). The group law costs zero."""
    if dim < 1 or lift_dim < 1:
        raise ValueError(f"dim and lift_dim must be >= 1, got {dim}, {lift_dim}")
    return 2 * lift_dim * dim + (lift_dim * lift_dim + 2 * lift_dim) * dim


def compose(g1: GroupElement, g2: GroupElement) -> GroupElement:
    """Chen's identity — the exact group law of the step-2 nilpotent group."""
    a1, b1, c1 = g1
    a2, b2, c2 = g2
    return a1 + a2, b1 + b2, c1 + c2 + a1.unsqueeze(-1) * b2.unsqueeze(-2)


class NilpotentLieScan(nn.Module):
    """Truncated-path-signature sequence mixer over a step-2 nilpotent Lie group."""

    def __init__(self, dim: int, *, lift_dim: int = 16) -> None:
        super().__init__()
        if dim < 1 or lift_dim < 1:
            raise ValueError(f"dim and lift_dim must be >= 1, got {dim}, {lift_dim}")
        self.d = dim
        self.k = lift_dim
        self.a_lift = nn.Linear(dim, lift_dim, bias=False)
        self.b_lift = nn.Linear(dim, lift_dim, bias=False)
        # Zero-init readout ⟹ forward(x) == x at init (identity-at-init).
        self.readout = nn.Linear(lift_dim * lift_dim + 2 * lift_dim, dim, bias=False)
        nn.init.zeros_(self.readout.weight)

    @property
    def num_parameters(self) -> int:
        return lie_scan_param_count(self.d, self.k)

    def _signature(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-token raw signature ``(A, B, C)``: ``(B,L,k)``, ``(B,L,k)``,
        ``(B,L,k,k)`` — exact cumulative group product, no approximation."""
        a = self.a_lift(x)
        b = self.b_lift(x)
        a_sum = torch.cumsum(a, dim=1)
        b_sum = torch.cumsum(b, dim=1)
        a_past = a_sum - a  # strictly-earlier prefix of a
        c = torch.cumsum(a_past.unsqueeze(-1) * b.unsqueeze(-2), dim=1)
        return a_sum, b_sum, c

    def group_element(self, x: torch.Tensor) -> GroupElement:
        """The whole sequence's group element (verification surface for the
        Chen-associativity test): ``(A, B, C)`` at the final position."""
        a_sum, b_sum, c = self._signature(x)
        return a_sum[:, -1], b_sum[:, -1], c[:, -1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, L, D) -> (B, L, D)``: residual + readout of the length-normalized
        signature (normalization is readout-side; the scan itself is exact)."""
        a_sum, b_sum, c = self._signature(x)
        n = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)
        level1 = n.view(1, -1, 1)
        pairs = (n * (n - 1) / 2).clamp(min=1.0).view(1, -1, 1, 1)
        features = torch.cat(
            [
                a_sum / level1,
                b_sum / level1,
                (c / pairs).flatten(-2),
            ],
            dim=-1,
        )
        return x + self.readout(features)
