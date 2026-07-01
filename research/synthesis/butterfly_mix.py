"""NM-C5 — Butterfly / orthogonal-flow feature mixer.

A ``[B, L, D] -> [B, L, D]`` feature mixer whose per-token weight is a product of
*butterfly factors* — the structured-orthogonal family (Parker 1995; Dao, Sohl,
Gupta, Gouws, "Learning Fast Algorithms for Linear Transforms", 2019). Each of
the ``log2(n)`` stages is block-diagonal with ``n/2`` learned ``2×2`` Givens
rotation blocks at XOR stride ``2^k``; stacking ``n_passes`` full butterfly sets
lifts expressivity while staying ``O(n_passes · d · log d)`` params.

    M = (butterfly_set_{P}) · … · (butterfly_set_1),
    butterfly_set = B_{L-1} · … · B_0,   B_k = blockdiag of n/2 Givens blocks,
    Givens(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]

Parameter cost: ``n_passes · (n/2) · log2(n) = O(n_passes · d · log d)`` vs ``d²``
(~32× cut at d=384 → n=512, ``n_passes=2``: 4 608 vs 147 456). Because every
Givens block is orthogonal, the full product is **exactly orthogonal** for any
angle values ⟹ ``spectral_radius = 1`` and ``energy_gain = 1`` — a distinctive
*stable* physics fingerprint that scores far from softmax on the NM-10
geometric-novelty axis (norm-preserving structured linear map, not convex
averaging). Butterfly + Monarch (NM-C3) are the two factorization gaps left after
codex's NM-1 shipped Kronecker / TT / tensor-ring / CP.

Identity-at-init: all angles 0 ⟹ each Givens = ``I_2`` ⟹ ``M = I_n`` (safe
drop-in for any ``d``, padded to the next power of two). Forward is ``O(d log d)``
gathers + 2×2 rotations; the dense ``n × n`` matrix is never materialized on the
hot path (``dense_matrix`` reconstructs it for tests only).

Self-contained on purpose — imports only ``torch`` so it is measurable by
``PhysicsDescriptorProbe.describe_operator`` (hence scored on the NM-10 axis)
without touching the synthesis registries. Registry wiring (``_init_butterfly_mix``
on ``CompiledOpParamInitMixin`` + ``OP_DISPATCH`` + ``estimate_op_params``) is
deferred until codex's in-flight NM-1 factorization work commits.

NM-C5 lane: ``research/notes/component_fab_compaction_lanes_2026-07-01.md``.
Plan mirror: ``tasks/fab_novel_math_expansion_plan.md`` Tier D.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def _next_pow2(d: int) -> int:
    if d < 1:
        raise ValueError(f"butterfly dim must be >= 1, got {d}")
    n = 1
    while n < d:
        n <<= 1
    return n


def _butterfly_shape(d: int) -> Tuple[int, int]:
    """``(n, L)``: padded size ``n = next pow2 >= d`` and stage count ``L = log2(n)``."""
    n = _next_pow2(d)
    return n, 0 if n == 1 else int(round(math.log2(n)))


def butterfly_param_count(d: int, n_passes: int = 2) -> int:
    """Angle params: ``n_passes · (n/2) · log2(n)``. Equals ``sum(p.numel())``."""
    n, L = _butterfly_shape(d)
    if L == 0:
        return 0
    return n_passes * (n // 2) * L


class ButterflyMix(nn.Module):
    """Butterfly / orthogonal-flow feature mixer (NM-C5). See module docstring."""

    def __init__(self, dim: int, *, n_passes: int = 2) -> None:
        super().__init__()
        if n_passes < 1:
            raise ValueError(f"n_passes must be >= 1, got {n_passes}")
        self.d = int(dim)
        self.n, self.L = _butterfly_shape(self.d)
        self.n_passes = int(n_passes)
        # Identity-at-init: all angles 0. Shape (n_passes, L, n/2); empty if L == 0.
        init = torch.zeros(self.n_passes, max(self.L, 0), self.n // 2)
        self.angles = nn.Parameter(init)

    @property
    def num_parameters(self) -> int:
        return butterfly_param_count(self.d, self.n_passes)

    def _stage_pairs(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lower/upper indices of the n/2 disjoint XOR-stride-2^k pairs (stage k)."""
        bit = 1 << k
        idx = torch.arange(self.n)
        lo = idx[idx & bit == 0]
        return lo, lo ^ bit

    def _apply_stage(
        self, x: torch.Tensor, theta: torch.Tensor, k: int
    ) -> torch.Tensor:
        """Apply butterfly stage k: n/2 Givens rotations over the XOR-stride-2^k pairs."""
        lo, hi = self._stage_pairs(k)
        c, s = torch.cos(theta), torch.sin(theta)  # (n/2,)
        a = x[..., lo]
        b = x[..., hi]
        out = torch.empty_like(x)
        out[..., lo] = c * a - s * b
        out[..., hi] = s * a + c * b
        return out

    def dense_matrix(self) -> torch.Tensor:
        """Full ``n×n`` product across all passes/stages. Tests only — ``O(n²)``."""
        if self.L == 0:
            return torch.eye(self.n)
        eye = torch.eye(self.n)
        m = eye
        for p in range(self.n_passes):
            for k in range(self.L):
                block = eye.clone()
                lo, hi = self._stage_pairs(k)
                c, s = torch.cos(self.angles[p, k]), torch.sin(self.angles[p, k])
                block[lo, lo] = c
                block[lo, hi] = -s
                block[hi, lo] = s
                block[hi, hi] = c
                m = block @ m  # B applied on the left, matching forward order
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the butterfly mixer over the last dim: ``(..., d) -> (..., d)``.
        Pads ``d -> n`` (pad dim only, never the prefix), applies
        ``B_{L-1}·…·B_0`` per pass, unpads. ``O(d·log d)`` FLOP, no ``d²``."""
        d, n = self.d, self.n
        lead = x.shape[:-1]
        if d != n:
            pad = x.new_zeros(*lead, n - d)
            x = torch.cat([x, pad], dim=-1)
        for p in range(self.n_passes):
            for k in range(self.L):
                x = self._apply_stage(x, self.angles[p, k], k)
        return x[..., :d] if d != n else x
