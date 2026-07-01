"""Fingerprint-guided lottery-ticket mask (MiniMax-M3-align M2) — W5 + W1.

Classic lottery-ticket pruning (Frankle & Carbin 2019) finds a sparse binary mask
by magnitude/loss-gradient. This is the same structural-sparsity idea, but the
mask is *certified by the novel-mechanism signature* instead: a winning channel
mask ``m ∈ {0,1}^D`` is one whose masked lane STILL measures far from the softmax
basin (``softmax_twin_score < threshold``). A gate that collapses toward softmax
under the mask fails certification — so the mask cannot buy sparsity by
reconverging on the attention basin (the mission's pathology).

Sparsity is a top-k channel mask at a fixed target density (the knob, e.g. 0.3 →
30% of the lane's channel ops), so the realized density is exact. WHICH channels
are kept is learned through a straight-through estimator: the hard top-k mask is
applied in the forward pass, and gradients flow to the per-channel importance
logits, so the task loss (and the non-twin certification) select the winning
channels. Per-layer cost is ``D`` importance logits. Reuses the NM-11 detector in
``component_fab.proposer.algebraic_properties`` for certification.

This module is the reusable core; wiring it as an ``op_compression_kind =
"novelty_mask"`` parametric atom (``research/synthesis``) is deferred to the
compaction coordination.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from component_fab.proposer.algebraic_properties import (
    Operator,
    measure_algebraic_properties,
)

#: A masked lane at/above this measured twin score is rejected — it has drifted
#: into the softmax basin, so the sparsity would come at the mechanism's expense.
NOVELTY_MASK_TWIN_THRESHOLD = 0.4


@dataclass(frozen=True, slots=True)
class MaskCertification:
    """Result of certifying a masked lane against the softmax-twin detector."""

    softmax_twin_score: float
    density: float
    certified: bool


class NoveltyConstrainedMask(nn.Module):
    """Learnable top-k channel mask at a fixed density (lottery-ticket style).

    ``forward(x)`` multiplies ``x`` by a hard 0/1 mask that keeps the top
    ``round(density * dim)`` channels by learned importance. The realized density
    is exact; gradients flow to the importance logits via a straight-through
    estimator, so the kept-channel set is learned. The mask is the compressed
    artifact (``dim`` logits); :meth:`certify` checks a masked lane stays out of
    the softmax basin.
    """

    def __init__(self, dim: int, *, target_density: float = 0.3) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if not 0.0 < target_density <= 1.0:
            raise ValueError("target_density must be in (0, 1]")
        self.dim = dim
        self.target_density = float(target_density)
        self.k = max(1, round(target_density * dim))
        # Small random per-channel importance; the task/cert learns which to keep.
        self.logits = nn.Parameter(torch.randn(dim) * 0.01)

    def _hard_topk(self) -> torch.Tensor:
        threshold = torch.topk(self.logits, self.k).values.min()
        return (self.logits >= threshold).to(self.logits.dtype)

    def hard_mask(self) -> torch.Tensor:
        """Binary top-k keep-mask; forward is 0/1, backward flows to the logits.

        Straight-through: ``hard + (soft - soft.detach())`` equals ``hard`` in the
        forward pass and carries ``d sigmoid / d logits`` in the backward pass.
        """
        hard = self._hard_topk()
        soft = torch.sigmoid(self.logits)
        return hard + (soft - soft.detach())

    def density(self) -> float:
        """Realized fraction of channels kept (exact: ``k / dim``)."""
        return self.k / self.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dim {self.dim}; got {tuple(x.shape)}")
        return x * self.hard_mask()

    @torch.no_grad()
    def certify(
        self,
        lane: Operator,
        *,
        dim: int | None = None,
        seq_len: int = 16,
        n_seeds: int = 3,
        threshold: float = NOVELTY_MASK_TWIN_THRESHOLD,
    ) -> MaskCertification:
        """Certify that ``lane`` masked by this mask stays out of the softmax basin.

        Wraps ``lane`` so its output channels are pruned by the hard mask, measures
        the masked op's ``softmax_twin_score``, and accepts iff it is below
        ``threshold``. A lane that collapses toward softmax under the mask fails.
        """
        probe_dim = int(dim) if dim is not None else self.dim
        mask = self._hard_topk()

        def masked_lane(x: torch.Tensor) -> torch.Tensor:
            return lane(x) * mask

        props = measure_algebraic_properties(
            masked_lane, dim=probe_dim, seq_len=seq_len, n_seeds=n_seeds
        )
        score = float(props.softmax_twin_score)
        return MaskCertification(
            softmax_twin_score=score,
            density=self.density(),
            certified=score < threshold,
        )


class NoveltyMaskedLane(nn.Module):
    """Wrap a base lane with a top-k :class:`NoveltyConstrainedMask` on its output.

    ``forward`` prunes the base lane's output channels; the mask learns which to
    keep under the density and non-twin constraints. Exposes the held mask's
    ``density``/``certify`` surface.
    """

    def __init__(
        self, base: nn.Module, dim: int, *, target_density: float = 0.3
    ) -> None:
        super().__init__()
        self.base = base
        self.mask = NoveltyConstrainedMask(dim, target_density=target_density)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask(self.base(x))

    def density(self) -> float:
        return self.mask.density()

    def certify(self, **kwargs) -> MaskCertification:
        return self.mask.certify(self.base, **kwargs)
