"""NM-C8 — shared weight-dictionary virtual-depth mixer (collapse n_layers).

All virtual layers read their mixing weight from ONE shared bank of ``n_basis``
basis-matrices ``M_b ∈ R^{d×d}``; layer ``l`` MATERIALIZES its weight as a
per-layer linear combination

    W_l = Σ_b c_{l,b} · M_b

and applies it as a residual update ``x ← x + α_l · (W_l · x)``. Parameters =
``n_basis·d²`` (the shared bank) + ``n_layers·n_basis`` (per-layer coefficients)
+ ``n_layers`` (ReZero scales), versus the ``n_layers·d²`` of ``n_layers``
independent dense layers — for ``n_basis ≪ n_layers`` a large, depth-growing
cut. This is the **collapse-n_layers** lever (Lever 3): the same idea as a
weight dictionary / dictionary-learning across layers, with the basis SHARED and
only the per-layer coefficients varying.

The novel hook — and the structural difference from the ALBERT weight-tying
baseline — is the **anti-collapse diversity guard**: if the basis ``{M_b}``
degenerates to one matrix (or the coefficients all collapse to a single basis),
every layer's ``W_l`` becomes a scalar multiple of the same matrix ⟹ silent
weight tying = ALBERT (the baseline to beat). ``basis_diversity_loss()`` measures
the collinearity of the basis (off-diagonal of the normalized Gram matrix of
flattened basis matrices) and ``coeff_rank()`` measures whether the per-layer
coefficient vectors actually span the basis — the fab's ranking/improver adds
the diversity term to the training loss so the dictionary stays expressive.

Distinct from NM-C7 (recurrent depth = ONE shared ``W`` applied ``k`` times,
``W^k``): NM-C8 materializes a DIFFERENT ``W_l`` per layer from a shared basis
bank (a product ``∏_l (I + α_l W_l)`` of distinct matrices, not a power of one).
Distinct from codex's single-op ``shared_basis_proj`` (a rank-``k`` vector basis
``R^{k×d}`` factorizing ONE projection): NM-C8 is a bank of full ``d×d``
basis-MATRICES shared across N layers via per-layer coefficients.

Identity-at-init: per-layer ReZero scales ``α_l = 0`` ⟹ the whole stack is
``x`` exactly. Pointwise per token (each token mixed only by ``W_l``, never with
another token) ⟹ ``cross_token_mixing ≈ 0`` ⟹ passes the NM-11 softmax-twin
detector and is NM-10-measurable (it IS a ``[B,L,D]→[B,L,D]`` mixer). Registry
wiring DEFERRED (NM-C3/C5/C7/C10/C15/C16 convention — ship the mechanism, wire
once codex's NM-1 lands).
"""

from __future__ import annotations

import math

import torch
from torch import nn

_DIVERSITY_EPS = 1e-6


def weight_dict_param_count(dim: int, n_layers: int, n_basis: int) -> int:
    """Exact trainable parameter count.

    ``n_basis·d²`` (shared basis bank) + ``n_layers·n_basis`` (per-layer
    coefficients) + ``n_layers`` (ReZero scales). The independent-layer baseline
    is ``n_layers·d²``; the cut is real whenever ``n_basis < n_layers``.
    """
    if dim < 1 or n_layers < 1 or n_basis < 1:
        raise ValueError(
            f"need dim>=1, n_layers>=1, n_basis>=1; got {dim=}, {n_layers=}, {n_basis=}"
        )
    return n_basis * dim * dim + n_layers * n_basis + n_layers


class WeightDictionaryMix(nn.Module):
    """NM-C8 — virtual-depth mixer materializing per-layer weights from a shared
    basis bank.

    ``forward(x)`` applies ``n_layers`` residual layers; each materializes
    ``W_l = Σ_b c_{l,b} M_b`` from the shared bank and updates
    ``x ← x + α_l · (W_l x)``. ``materialize_weight(l)`` / ``materialize_weights()``
    expose the per-layer weights for inspection; ``basis_diversity_loss()`` and
    ``coeff_rank()`` are the anti-collapse diagnostics.
    """

    def __init__(
        self,
        dim: int,
        *,
        n_layers: int = 4,
        n_basis: int = 2,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.n_layers = int(n_layers)
        self.n_basis = int(n_basis)
        if self.dim < 1:
            raise ValueError(f"dim must be >= 1, got {self.dim}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if self.n_basis < 1:
            raise ValueError(f"n_basis must be >= 1, got {self.n_basis}")

        # Shared basis bank: the ONLY d²-scale storage, read by every layer.
        # std = 1/sqrt(d) so a materialized W_l (Σ_b c_{l,b} M_b) is well-scaled
        # for an N(0,1) input once the ReZero scale opens.
        basis_std = 1.0 / math.sqrt(self.dim)
        self.basis = nn.Parameter(
            torch.randn(self.n_basis, self.dim, self.dim) * basis_std
        )
        # Per-layer coefficients over the basis. Small random init so each layer
        # starts with a DISTINCT materialized weight (identity-at-init is handled
        # by the ReZero scales below; coeffs are nonzero so the basis receives
        # gradient as soon as a scale opens).
        coeff_std = 1.0 / math.sqrt(self.n_basis)
        self.coeffs = nn.Parameter(torch.randn(self.n_layers, self.n_basis) * coeff_std)
        # Per-layer ReZero scales: 0 at init ⟹ every layer is x ← x + 0 ⟹ the
        # whole stack is the identity (safe drop-in).
        self.layer_scales = nn.Parameter(torch.zeros(self.n_layers))

    @property
    def num_parameters(self) -> int:
        return weight_dict_param_count(self.dim, self.n_layers, self.n_basis)

    def materialize_weights(self) -> torch.Tensor:
        """All per-layer weights ``(n_layers, d, d)`` = ``coeffs @ basis``.

        ``W_l = Σ_b c_{l,b} M_b`` via einsum ``lk,kij->lij`` (the shared basis
        index ``k`` is contracted between coefficients and bank). Vectorized so
        the forward never rebuilds a Python-loop sum per layer.
        """
        return torch.einsum("lk,kij->lij", self.coeffs, self.basis)

    def materialize_weight(self, layer_idx: int) -> torch.Tensor:
        """The materialized weight ``W_l`` for layer ``layer_idx``: ``(d, d)``."""
        if not 0 <= layer_idx < self.n_layers:
            raise IndexError(f"layer_idx {layer_idx} out of [0, {self.n_layers})")
        return torch.einsum("k,kij->ij", self.coeffs[layer_idx], self.basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.materialize_weights()  # (n_layers, d, d)
        out = x
        for layer in range(self.n_layers):
            w_l = weights[layer]  # (d, d)
            update = torch.einsum("ij,...j->...i", w_l, out)
            out = out + self.layer_scales[layer] * update
        return out

    def basis_diversity_loss(self) -> torch.Tensor:
        """Anti-collapse guard: collinearity of the basis ``{M_b}`` (scalar).

        Computes the normalized Gram matrix of the flattened basis matrices and
        returns the mean squared off-diagonal entry. Independent basis matrices
        → ≈0; a degenerate bank (all ``M_b`` equal, the ALBERT regime) → ≈1.
        The fab adds this to the training loss so the dictionary does not
        silently collapse to a single shared matrix.
        """
        flat = self.basis.reshape(self.n_basis, -1)  # (n_basis, d*d)
        gram = flat @ flat.t()  # (n_basis, n_basis)
        diag = gram.diagonal().clamp_min(_DIVERSITY_EPS)
        # Correlation matrix: diag-normalized so the measure is scale-invariant.
        denom = diag.unsqueeze(0) * diag.unsqueeze(1)
        corr = gram / denom.clamp_min(_DIVERSITY_EPS).sqrt()
        if self.n_basis < 2:
            return corr.new_zeros(())
        off = corr - torch.eye(self.n_basis, device=corr.device, dtype=corr.dtype)
        n_off = self.n_basis * (self.n_basis - 1)
        return (off * off).sum() / n_off

    def coeff_rank(self) -> int:
        """Numerical rank of the per-layer coefficient matrix ``(n_layers, n_basis)``.

        ``n_basis`` (full rank) ⟹ the layers genuinely use distinct basis
        combinations; 1 ⟹ all layers are scalar multiples of one basis vector
        (the weight-tying degeneracy). A live anti-collapse diagnostic.
        """
        with torch.no_grad():
            rank = torch.linalg.matrix_rank(self.coeffs.float())
        return int(rank.item())
