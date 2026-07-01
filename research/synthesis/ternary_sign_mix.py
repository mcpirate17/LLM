"""NM-C15 — Ternary-native sign-semiring feature mixer.

A ``[B, L, D] -> [B, L, D]`` feature-mixing layer whose weight is **native to the
sign semiring**: every entry is in ``{-1, 0, +1}``, so the forward is pure
add/subtract — multiply-by-``+1`` is a copy, by-``-1`` a sign flip, by-``0`` a drop.
There is no multiply-accumulate:

    out_j = ⊕_i  W_{ji} ⊗ x_i ,   W_{ji} ∈ {-1, 0, +1}
          = Σ_{i: W_{ji}=+1} x_i  −  Σ_{i: W_{ji}=−1} x_i

The weights live in the **sign semiring** ``({-1, 0, +1}, +, ×)`` — the
multiplicative sign monoid acting over the additive group of the feature space.
The magnitude is fixed at unity *by construction*: the layer never learns or
stores a per-weight magnitude. That is the novelty vs the baseline. A binarized /
ternarized dense net (TBN/BNN) trains a full fp32 matrix and discards the
magnitude at deploy time, so the dense magnitude shaped training; here the forward
*mathematics* operates entirely in the ternary sign domain at every step, and the
magnitude carries no information. This is a mixer designed FOR ternary precision,
not a dense mixer quantized after the fact.

Parameter cost. The trainable controller is a real-valued pre-sign tensor ``Z``;
the deployed weight is ``W = ternary(Z)`` via a straight-through estimator (hard
ternary forward, identity backward). Two modes:

  * **full** (``rank=None``, default): ``Z ∈ ℝ^{D×D}``, initialised to the identity
    ⟹ ``ternary(Z₀) = I`` ⟹ an identity-at-init safe drop-in for any ``D``. The
    deployed weight is a packed 2-bit ternary matrix ⟹ **÷16 inference weight
    VRAM vs fp32** (the controller is fp32, but the serialized/deployed weight is
    ternary — the figure that decides whether the mechanism fits on the GPU).
  * **factored** (``rank=r``): ``Z = A·B``, ``A ∈ ℝ^{D×r}``, ``B ∈ ℝ^{r×D}`` ⟹
    trainable params ``2·D·r ≪ D²`` (cuts *training* VRAM too), at the cost of the
    identity-at-init guarantee. The deployed weight ``ternary(A·B)`` is still
    ternary ⟹ same ÷16 inference storage.

Why it serves the mission (compaction as amplifier, not throttle): at cl100k the
tied embedding is ~75% of params and per-layer ``O(d²)`` weights dominate the
rest; a mixer whose stored weight is 2-bit ternary frees ``d²·30/8`` bits per
layer of VRAM — depth/width that a novel non-QKV mechanism can spend to reach the
scale where it beats softmax. Post-training quantization (GPTQ/AWQ) and BNN-style
post-hoc binarization are the **baselines to beat**; the differentiator here is
that the ternary constraint is the *algebra* of the mixer (the sign semiring), not
a projection applied to a learned dense matrix.

Self-contained on purpose — imports only ``torch`` so it is measurable by
``PhysicsDescriptorProbe.describe_operator`` (hence NM-10-scorable): a ternary sign
matrix has a distinctive fingerprint — integer entries bounded in ``[-1, 1]``,
zeros-induced sparsity, spectral radius ``≤ √(max row nnz)``. Registry wiring
(``_init_ternary_sign_mix`` + ``OP_DISPATCH`` + ``estimate_op_params``) is deferred
until codex's in-flight factorization/embedding work commits — see NM-C3/C5.

NM-C15 lane: ``research/notes/component_fab_compaction_lanes_2026-07-01.md``.
Plan mirror: ``tasks/fab_novel_math_expansion_plan.md`` Tier D.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

_DEFAULT_THRESH = (
    0.5  # |Z| < thresh -> 0; else sign(Z). In (0, 1) so eye init maps to I.
)


def ternary_param_count(dim: int, rank: int | None = None) -> int:
    """Trainable controller params for the sign-semiring mixer.

    ``rank=None`` (full): ``D²``. ``rank=r`` (factored): ``2·D·r``.
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if rank is None:
        return dim * dim
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    return 2 * dim * rank


def _ternarize(z: torch.Tensor, thresh: float) -> torch.Tensor:
    """Straight-through ternary rounding: forward ``{-1, 0, +1}``, backward identity.

    ``|z| < thresh -> 0``; otherwise ``sign(z)``. The backward pass passes the
    gradient through ``z`` unchanged (STE) so the continuous controller learns even
    though the deployed weight is discrete.
    """
    sign = torch.sign(z)
    w_hard = torch.where(z.abs() >= thresh, sign, torch.zeros_like(z))
    # STE: forward is the hard ternary value, backward is the identity on z.
    return z + (w_hard - z).detach()


class TernarySignMix(nn.Module):
    """Sign-semiring feature mixer; deployed weight native to ``{-1, 0, +1}``."""

    def __init__(
        self,
        dim: int,
        *,
        rank: int | None = None,
        thresh: float = _DEFAULT_THRESH,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if thresh <= 0.0:
            raise ValueError(f"thresh must be > 0, got {thresh}")
        self.d = dim
        self.rank = rank
        self.thresh = float(thresh)
        if rank is None:
            # Identity init: diag 1.0 (>= thresh -> +1), off-diag 0.0 (< thresh -> 0).
            self.Z = nn.Parameter(torch.eye(dim))
        else:
            # Factored controller Z = A @ B. Small random init; NOT identity-at-init.
            scale = 1.0 / math.sqrt(rank)
            self.A = nn.Parameter(torch.randn(dim, rank) * scale)
            self.B = nn.Parameter(torch.randn(rank, dim) * scale)

    @property
    def num_parameters(self) -> int:
        return ternary_param_count(self.d, self.rank)

    def _controller(self) -> torch.Tensor:
        """The real-valued pre-sign tensor ``Z`` (``D×D``)."""
        return self.Z if self.rank is None else self.A @ self.B

    def ternary_weight(self) -> torch.Tensor:
        """The deployed ``{-1, 0, +1}`` weight matrix ``W`` (``D×D``)."""
        return _ternarize(self._controller(), self.thresh)

    def ternary_storage_bytes(self) -> int:
        """Packed 2-bit storage of the ternary weight — the inference-VRAM figure.

        Two trits fit in one byte-friendly packing (ceil(D²·2/8)); vs fp32 dense
        (D²·4 bytes) this is the ÷16 storage win.
        """
        return math.ceil(self.d * self.d * 2 / 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., D) -> (..., D)`` sign-semiring mix: ``out = W · x`` with ``W``
        ternary. Mathematically pure add/subtract (no MAC); a fused sign-scatter
        kernel is the production hot path — here ``einsum`` for portability and so
        the layer is measurable by ``PhysicsDescriptorProbe``.
        """
        w = self.ternary_weight()  # (D, D) ternary, STE-linked to the controller
        return torch.einsum("ij,...j->...i", w, x)

    # ── verification helpers (tests only; never the hot path) ──────────────────

    def dense_weight(self) -> torch.Tensor:
        """Materialised ``D×D`` ternary weight (alias of ``ternary_weight`` for
        symmetry with the Monarch/Butterfly ``dense_matrix`` convention)."""
        return self.ternary_weight()

    def shape(self) -> Tuple[int, int]:
        return (self.d, self.d)
