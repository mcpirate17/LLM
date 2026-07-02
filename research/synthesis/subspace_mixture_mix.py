"""NM-C20 — mixture-of-subspaces mixer (Lever 6: conditional structure, O(r·d)).

Every channel of ``d`` is HARD-assigned to exactly ONE of ``m`` subspaces by a
LEARNED non-convex partition; subspace ``j`` reads only its channels, mixes them
through its own cheap bottleneck (down ``s×d`` → mix ``s×s`` → up ``d×s`` with
``s ≪ d``), and the subspace outputs recombine additively:

    W = Σ_j U_j M_j P_j diag(mask_j) ;   x ← x + α · (W x)

Params = ``d·m`` (assignment logits) + ``m·s·d`` (down) + ``m·s²`` (mix) +
``m·d·s`` (up) + 1 = **O(2·m·s·d) = O(r·d)** for total subspace width
``r = m·s ≪ d`` — never O(d²). The compaction ratio GROWS with ``d`` at fixed
``r``.

The novel structure vs the baselines-to-beat:

- **LoRA / single low-rank** (the note's explicit baseline): one subspace, no
  grouping, all channels feed one bottleneck. Here ``m`` subspaces own DISJOINT
  learned channel sets — the partition itself is trained (a non-convex,
  combinatorial grouping relaxed only in the backward pass).
- **MoE**: no router over tokens anywhere — the partition is static over
  CHANNELS, identical for every token and position; there is no per-token
  dispatch and no softmax (assignment is hard argmax with the validated
  Lorentzian bounded-reciprocal STE backward, the NM-C11/F9.1 non-softmax
  convention).
- **Block-diagonal mixing**: groups here are learned and non-contiguous, and
  each group's mix is a low-rank bottleneck with a FULL-``d`` write span, not a
  within-block square.

Collapse mode for this lane: the partition piles every channel onto one
subspace ⟹ the other bottlenecks starve, the operator degenerates to a single
rank-``s`` LoRA, and the effective rank collapses. Gates shipped (the note's
"subspace orthogonality + effective rank"):

- ``assignment_counts()`` / ``assignment_balance()`` — channels per subspace;
  balance 1.0 = perfectly even partition, 0.0 = at least one dead subspace.
- ``assignment_balance_loss()`` — differentiable, 0 at a perfectly balanced
  partition and exactly 1 when ALL channels pile onto one subspace.
- ``subspace_overlap()`` — mean pairwise overlap of the orthonormalized WRITE
  spans (the ``U_j`` column spaces): identical spans ⟹ 1, independent random
  spans ⟹ ≈ s/d. Keeps the subspaces writing into distinct output directions.
- ``assembled_rank()`` — numerical rank of the assembled operator (≤ m·s by
  construction; pile-up collapses it toward s).

Identity-at-init: ReZero ``α = 0`` ⟹ the module is exactly ``x``. Pointwise per
token (channels mixed, never tokens) ⟹ ``cross_token_mixing ≈ 0`` ⟹ passes the
NM-11 softmax-twin detector and is NM-10-measurable. Registry wiring DEFERRED
(NM-C3/C5/C8/C9/C11 convention).
"""

from __future__ import annotations

import torch
from torch import nn

_OVERLAP_EPS = 1e-12


def subspace_mixture_param_count(dim: int, n_subspaces: int, subspace_dim: int) -> int:
    """Exact trainable parameter count.

    ``d·m`` assignment logits + ``m·s·d`` down + ``m·s²`` mix + ``m·d·s`` up
    + 1 ReZero scale — O(r·d) for r = m·s, never O(d²).
    """
    _validate(dim, n_subspaces, subspace_dim)
    m, s, d = n_subspaces, subspace_dim, dim
    return d * m + m * s * d + m * s * s + m * d * s + 1


def _validate(dim: int, n_subspaces: int, subspace_dim: int) -> None:
    if dim < 1 or n_subspaces < 1 or subspace_dim < 1:
        raise ValueError(
            f"need dim>=1, n_subspaces>=1, subspace_dim>=1; "
            f"got {dim=}, {n_subspaces=}, {subspace_dim=}"
        )
    if n_subspaces * subspace_dim >= dim:
        raise ValueError(
            "total subspace width m·s must be < dim (the compaction claim); "
            f"got m·s={n_subspaces * subspace_dim} >= {dim=}"
        )


class SubspaceMixtureMix(nn.Module):
    """NM-C20 — mixer that partitions channels into m learned subspaces, mixes
    each through its own s≪d bottleneck, and recombines.

    ``forward(x)`` masks the input by the hard channel partition, runs each
    subspace's down→mix→up, sums, and applies the ReZero-scaled residual.
    ``assignment_*`` / ``subspace_overlap`` / ``assembled_rank`` are the
    anti-collapse gates.
    """

    def __init__(
        self,
        dim: int,
        *,
        n_subspaces: int = 4,
        subspace_dim: int = 4,
        lorentz_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        _validate(dim, n_subspaces, subspace_dim)
        if lorentz_gamma <= 0:
            raise ValueError(f"lorentz_gamma must be > 0, got {lorentz_gamma}")
        self.dim = int(dim)
        self.n_subspaces = int(n_subspaces)
        self.subspace_dim = int(subspace_dim)
        self.lorentz_gamma = float(lorentz_gamma)
        d, m, s = self.dim, self.n_subspaces, self.subspace_dim

        # Learned non-convex partition: per-channel logits over subspaces.
        self.assign_logits = nn.Parameter(torch.randn(d, m))
        # Per-subspace bottleneck. down/up std 1/sqrt(d) and mix std 1/sqrt(s)
        # keep the composed update well-scaled once the ReZero scale opens.
        self.down = nn.Parameter(torch.randn(m, s, d) / d**0.5)
        self.mix = nn.Parameter(torch.randn(m, s, s) / s**0.5)
        self.up = nn.Parameter(torch.randn(m, d, s) / d**0.5)
        # ReZero: 0 at init ⟹ the module is exactly the identity.
        self.scale = nn.Parameter(torch.zeros(()))

    @property
    def num_parameters(self) -> int:
        return subspace_mixture_param_count(
            self.dim, self.n_subspaces, self.subspace_dim
        )

    def _soft_assignment(self) -> torch.Tensor:
        """Lorentzian bounded-reciprocal assignment weights (NON-softmax),
        per channel over subspaces: ``1/(1+(max−s_j)²/γ)`` normalized."""
        gap = self.assign_logits.max(dim=-1, keepdim=True).values - self.assign_logits
        weights = 1.0 / (1.0 + gap * gap / self.lorentz_gamma)
        return weights / weights.sum(dim=-1, keepdim=True)

    def _ste_assignment(self) -> torch.Tensor:
        """Hard one-hot channel→subspace partition ``(d, m)`` with the
        Lorentzian soft backward path."""
        soft = self._soft_assignment()
        hard = torch.zeros_like(soft)
        hard.scatter_(-1, self.assign_logits.argmax(dim=-1, keepdim=True), 1.0)
        return soft + (hard - soft).detach()

    def assignment(self) -> torch.Tensor:
        """The hard subspace index of every channel: ``(d,)``."""
        with torch.no_grad():
            return self.assign_logits.argmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self._ste_assignment()  # (d, m) hard one-hot, STE backward
        # Each subspace reads ONLY its channels: masked down-projection.
        masked_down = self.down * mask.t().unsqueeze(1)  # (m, s, d)
        low = torch.einsum("msd,...d->...ms", masked_down, x)
        mixed = torch.einsum("mts,...ms->...mt", self.mix, low)
        update = torch.einsum("mdt,...mt->...d", self.up, mixed)
        return x + self.scale * update

    def assemble_operator(self) -> torch.Tensor:
        """The equivalent dense ``(d, d)`` operator ``Σ_j U_j M_j P_j diag(mask_j)``
        — INSPECTION ONLY (the forward never builds it)."""
        mask = torch.zeros_like(self.assign_logits)
        mask.scatter_(-1, self.assign_logits.argmax(dim=-1, keepdim=True), 1.0)
        masked_down = self.down * mask.t().unsqueeze(1)  # (m, s, d)
        return torch.einsum("mdt,mts,mse->de", self.up, self.mix, masked_down)

    def assignment_counts(self) -> torch.Tensor:
        """Channels owned by each subspace: ``(m,)`` integer counts."""
        return torch.bincount(self.assignment(), minlength=self.n_subspaces)

    def assignment_balance(self) -> float:
        """``min_j count_j · m / d``: 1.0 = perfectly even partition, 0.0 = at
        least one subspace owns no channels (starved bottleneck)."""
        counts = self.assignment_counts()
        return float(counts.min().item()) * self.n_subspaces / self.dim

    def assignment_balance_loss(self) -> torch.Tensor:
        """Differentiable anti-pile-up guard on the soft assignment mass.

        ``p_j`` = mean soft assignment of the channels to subspace ``j``;
        returns ``m/(m−1) · Σ_j (p_j − 1/m)²`` — 0 at a perfectly balanced
        partition, exactly 1 when ALL channels pile onto one subspace. The fab
        adds this to the training loss so the partition cannot silently
        degenerate to a single-subspace LoRA.
        """
        if self.n_subspaces < 2:
            return self.assign_logits.new_zeros(())
        p = self._soft_assignment().mean(dim=0)  # (m,)
        dev = p - 1.0 / self.n_subspaces
        return (dev * dev).sum() * self.n_subspaces / (self.n_subspaces - 1)

    def subspace_overlap(self) -> float:
        """Mean pairwise overlap of the orthonormalized WRITE spans ``span(U_j)``.

        ``‖Q_i^T Q_j‖_F² / s`` averaged over pairs: 1.0 ⟹ the subspaces write
        into the SAME output directions (the recombination degenerates to one
        subspace); independent random s-dim spans in R^d ⟹ ≈ s/d. The note's
        subspace-orthogonality gate.
        """
        if self.n_subspaces < 2:
            return 0.0
        with torch.no_grad():
            q, _ = torch.linalg.qr(self.up.float())  # (m, d, s), orthonormal cols
            total = 0.0
            n_pairs = 0
            for i in range(self.n_subspaces):
                for j in range(i + 1, self.n_subspaces):
                    cross = q[i].t() @ q[j]
                    total += float((cross * cross).sum().item()) / self.subspace_dim
                    n_pairs += 1
            return total / max(n_pairs, _OVERLAP_EPS)

    def assembled_rank(self) -> int:
        """Numerical rank of the assembled operator — ≤ ``m·s`` by construction;
        a piled-up partition collapses it toward ``s`` (the LoRA degeneracy)."""
        with torch.no_grad():
            return int(
                torch.linalg.matrix_rank(self.assemble_operator().float()).item()
            )
