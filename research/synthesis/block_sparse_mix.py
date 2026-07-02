"""NM-C11 — native block-sparse mixer (sparsity IS the mechanism).

The ``d×d`` mixing weight exists ONLY as ``n_blocks`` learned nonzero blocks on
a coarse ``(d/b)×(d/b)`` block grid. Block ``k`` stores a dense ``b×b`` value
``B_k`` and a LEARNED bipartite address — hard-argmax row/column placement over
the grid — so the module realizes

    W = Σ_k place(k) ⊗ B_k ;   x ← x + α · (W x)

with params, VRAM, and FLOP all ∝ ``n_blocks·b²`` (≪ d² when
``n_blocks ≪ (d/b)²``). There is NO dense ``d×d`` tensor anywhere: the forward
gathers each block's input-block, applies ``B_k``, and scatter-adds into its
output-block. This is Lever 4's "learned bipartite sparsity" lane.

Distinct from the repo's existing ``block_sparse_linear`` primitive
(``compiler_op_utils._build_block_sparse_mask``): that op stores a FULL dense
``d×d`` weight and masks it by block magnitude — post-hoc pruning, the baseline
this lane must beat (params/VRAM stay O(d²) there; here storage never exceeds
the K blocks). Distinct from the NM-C3/C5 factorizations (dense structured
products): here the compaction comes from learned STRUCTURAL sparsity.

Block placement is NON-softmax by construction — the documented softmax-router
collapse (`recursive_depth_router`) cannot occur structurally. Selection is a
hard per-block argmax over row/column logits; the backward path is a
straight-through estimator whose soft surrogate is the validated Lorentzian
bounded-reciprocal shaping (the NM-11-clean form used by the F9.1 annealed
selector and `recurrent_depth_refine`), NOT a softmax: weights
``1/(1 + (max−score)²/γ)`` normalized by their sum.

Collapse mode for this lane: all blocks pile onto one grid address ⟹ the
assembled weight has a single nonzero block ⟹ mixing dies while the loss can
look fine. Gates shipped:

- ``placement_utilization()`` — unique (row, col) addresses / n_blocks; 1.0 =
  every block occupies its own grid cell, ``1/n_blocks`` = total pile-up.
- ``placement_overlap_loss()`` — differentiable anti-pile-up guard: mean
  pairwise Bhattacharyya-style overlap of the blocks' soft address
  distributions (joint row⊗col). Identical addresses ⟹ 1; spread placement ⟹
  ≈0. The fab adds this to the training loss.
- ``assembled_rank()`` — numerical rank of the assembled weight (inspection;
  rank ≤ n_blocks·b by construction, and pile-up collapses it further).

Identity-at-init: ReZero scale ``α = 0`` ⟹ the module is exactly ``x``.
Pointwise per token (channels mixed, never tokens) ⟹ ``cross_token_mixing ≈ 0``
⟹ passes the NM-11 softmax-twin detector and is NM-10-measurable. Registry
wiring DEFERRED (NM-C3/C5/C8/C9/C15/C16 convention).
"""

from __future__ import annotations

import torch
from torch import nn

_OVERLAP_EPS = 1e-12


def block_sparse_param_count(dim: int, block_size: int, n_blocks: int) -> int:
    """Exact trainable parameter count.

    ``n_blocks·b²`` (the ONLY weight storage) + ``2·n_blocks·(d/b)`` (row and
    column address logits) + 1 (ReZero scale). The dense baseline is ``d²``;
    the repo's ``block_sparse_linear`` pruning baseline ALSO stores ``d²``.
    """
    _validate(dim, block_size, n_blocks)
    grid = dim // block_size
    return n_blocks * block_size * block_size + 2 * n_blocks * grid + 1


def _validate(dim: int, block_size: int, n_blocks: int) -> None:
    if dim < 1 or block_size < 1 or n_blocks < 1:
        raise ValueError(
            f"need dim>=1, block_size>=1, n_blocks>=1; got {dim=}, {block_size=}, {n_blocks=}"
        )
    if dim % block_size != 0:
        raise ValueError(
            f"dim must be divisible by block_size; got {dim=}, {block_size=}"
        )


class BlockSparseMix(nn.Module):
    """NM-C11 — mixer whose weight exists only as K learned blocks with learned
    hard (non-softmax) bipartite grid placement.

    ``forward(x)`` gathers each block's input-block, applies its ``b×b`` value,
    and scatter-adds into its output-block (residual, ReZero-scaled).
    ``assemble_weight()`` materializes the equivalent dense weight for
    inspection only; ``placement_utilization`` / ``placement_overlap_loss`` /
    ``assembled_rank`` are the anti-pile-up and capacity gates.
    """

    def __init__(
        self,
        dim: int,
        *,
        block_size: int = 8,
        n_blocks: int = 8,
        lorentz_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        _validate(dim, block_size, n_blocks)
        if lorentz_gamma <= 0:
            raise ValueError(f"lorentz_gamma must be > 0, got {lorentz_gamma}")
        self.dim = int(dim)
        self.block_size = int(block_size)
        self.n_blocks = int(n_blocks)
        self.grid = self.dim // self.block_size
        self.lorentz_gamma = float(lorentz_gamma)

        # The ONLY weight storage: K dense b×b blocks. std 1/sqrt(b) so each
        # block's output-block contribution is well-scaled once the scale opens.
        self.block_values = nn.Parameter(
            torch.randn(self.n_blocks, self.block_size, self.block_size)
            / self.block_size**0.5
        )
        # Learned bipartite addresses: per-block logits over output-block rows
        # and input-block columns of the grid. Random init spreads the argmax
        # placements across the grid.
        self.row_logits = nn.Parameter(torch.randn(self.n_blocks, self.grid))
        self.col_logits = nn.Parameter(torch.randn(self.n_blocks, self.grid))
        # ReZero: 0 at init ⟹ the module is exactly the identity.
        self.scale = nn.Parameter(torch.zeros(()))

    @property
    def num_parameters(self) -> int:
        return block_sparse_param_count(self.dim, self.block_size, self.n_blocks)

    def _soft_address(self, logits: torch.Tensor) -> torch.Tensor:
        """Lorentzian bounded-reciprocal address weights (NON-softmax), per block.

        ``w_g = 1 / (1 + (max − s_g)² / γ)`` normalized by the sum — the
        validated NM-11-clean shaping (F9.1 annealed selector convention).
        Peaks at the argmax, no exponential anywhere.
        """
        gap = logits.max(dim=-1, keepdim=True).values - logits
        weights = 1.0 / (1.0 + gap * gap / self.lorentz_gamma)
        return weights / weights.sum(dim=-1, keepdim=True)

    def _ste_address(self, logits: torch.Tensor) -> torch.Tensor:
        """Hard one-hot argmax placement with a Lorentzian soft backward path."""
        soft = self._soft_address(logits)
        hard = torch.zeros_like(soft)
        hard.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return soft + (hard - soft).detach()

    def placements(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The hard (row, col) grid address of every block: two ``(n_blocks,)``
        index tensors."""
        with torch.no_grad():
            return self.row_logits.argmax(dim=-1), self.col_logits.argmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        row = self._ste_address(self.row_logits)  # (K, G) hard one-hot, STE
        col = self._ste_address(self.col_logits)  # (K, G)
        blocks_in = x.reshape(*x.shape[:-1], self.grid, self.block_size)
        # Gather each block's input-block (exact one-hot gather in forward;
        # the einsum-with-one-hot form keeps the STE backward path to the
        # address logits — a deployment kernel uses a real gather/scatter).
        gathered = torch.einsum("kg,...gb->...kb", col, blocks_in)
        mixed = torch.einsum("kij,...kj->...ki", self.block_values, gathered)
        # Scatter-add every block's output into its output-block row.
        blocks_out = torch.einsum("kg,...kb->...gb", row, mixed)
        update = blocks_out.reshape(*x.shape[:-1], self.dim)
        return x + self.scale * update

    def assemble_weight(self) -> torch.Tensor:
        """The equivalent dense ``(d, d)`` weight — INSPECTION ONLY (the forward
        never materializes it). Blocks sharing an address sum, exactly matching
        the forward's scatter-add."""
        row_idx, col_idx = self.placements()
        w = self.block_values.new_zeros(self.dim, self.dim)
        b = self.block_size
        for k in range(self.n_blocks):
            r, c = int(row_idx[k]) * b, int(col_idx[k]) * b
            w[r : r + b, c : c + b] += self.block_values[k]
        return w

    def placement_utilization(self) -> float:
        """Unique (row, col) grid addresses / n_blocks. 1.0 ⟹ every block has
        its own cell; ``1/n_blocks`` ⟹ total pile-up (the collapse mode)."""
        row_idx, col_idx = self.placements()
        cells = (row_idx * self.grid + col_idx).tolist()
        return len(set(cells)) / self.n_blocks

    def placement_overlap_loss(self) -> torch.Tensor:
        """Differentiable anti-pile-up guard: mean pairwise overlap of the
        blocks' soft joint address distributions, scalar.

        Uses the Bhattacharyya coefficient ``Σ√(p·q)`` of the joint row⊗col
        Lorentzian distributions: identical addresses ⟹ 1; blocks spread over
        distinct cells ⟹ ≈0. The fab adds this to the training loss so the
        placement cannot silently collapse onto one grid cell.
        """
        if self.n_blocks < 2:
            return self.row_logits.new_zeros(())
        row = self._soft_address(self.row_logits)  # (K, G)
        col = self._soft_address(self.col_logits)  # (K, G)
        joint = row[:, :, None] * col[:, None, :]  # (K, G, G)
        sqrt_joint = joint.clamp_min(_OVERLAP_EPS).sqrt().reshape(self.n_blocks, -1)
        overlap = sqrt_joint @ sqrt_joint.t()  # Bhattacharyya coefficients
        off = overlap - torch.diag(overlap.diagonal())
        n_off = self.n_blocks * (self.n_blocks - 1)
        return off.sum() / n_off

    def assembled_rank(self) -> int:
        """Numerical rank of the assembled weight. Bounded by ``n_blocks·b`` by
        construction; address pile-up collapses it further — a live capacity
        diagnostic alongside the utilization gate."""
        with torch.no_grad():
            return int(torch.linalg.matrix_rank(self.assemble_weight().float()).item())
