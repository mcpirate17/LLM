# pyright: reportPrivateImportUsage=false
"""NM-C12 — sheaf-consistency token-merge mixer (Lever 4: shrink L → K).

Content-aware causal token merging: adjacent tokens merge iff their learned
RESTRICTIONS to a shared overlap subspace agree — the sheaf gluing condition
(local sections glue exactly when their restrictions to the overlap coincide),
not full-space similarity:

    δ_p = ‖ρ_L x_p − ρ_R x_{p+1}‖² / k          (overlap discrepancy, k ≪ d)
    w_p = 1 / (1 + δ_p / γ)                      (Lorentzian score, NON-softmax)
    fold_p = [w_p > τ]                           (hard, STE backward through w_p)

Folded tokens fold FORWARD into their successor (each surviving cluster head
absorbs only EARLIER tokens — the proven-causal invariant of
``_op_adjacent_token_merge``); maximal runs of agreement chain into one cluster
whose merged value is the PURE SUPERPOSITION of its members,

    glue(cluster) = Σ_j x_j c_j

— NOT the convex cluster mean, and not normalized at all. This is load-bearing
for NM-11 and was measured, not declared: the convex mean scored 0.898
(a normalized convex average IS the softmax simplex signature — weights ≥ 0
summing to 1, constants preserved); a √n-normalized superposition still rode
the 0.7 demotion boundary (0.68–0.72 across init seeds — it still points at
the cluster mean, just louder). The unnormalized sum measures 0.647 max over
5 init seeds: weights sum to n, agreeing evidence ACCUMULATES (amplitude IS
cluster mass, CDMA-style; bounded by ``max_cluster`` by construction),
constants are not preserved, the range is non-convex — and a singleton
cluster is still returned BIT-EXACT (n=1 ⟹ glue = x).

The read couples back bilinearly, not as a plain additive blend:

    out = x + α · (V · read) ⊙ (1 + x)

The first-order term transports compressed content into the token; the
second-order term (read × token) is algebra softmax attention cannot express
— its value path is LINEAR in x, a convex blend of first powers. Both terms
share one lift ``V``; ReZero ``α = 0`` keeps exact identity-at-init.

Gluing is certified over a BOUNDED cover: cluster span is capped at
``max_cluster`` (a forced head every ``max_cluster`` positions). The pairwise
overlap test only certifies LOCAL consistency — an unbounded agreement chain
would glue a global section from local checks alone, and it is also this
lane's degeneracy: on near-constant input one cluster stays open forever, no
summary ever completes, and the op silently degenerates to the identity
(which measures as the softmax partition-of-unity signature on NM-11). The
cap structurally bounds both. The sequence of cluster heads is the
compressed stream: K = #heads ≪ L on redundant input, and downstream sequence
ops that consume ``compressed_stream()`` run at O(K), not O(L) — the VRAM/FLOP
compaction claim of this lane.

Causality is the load-bearing design constraint. The fold decision for token p
uses x_{p+1}, so it may influence outputs only at positions ≥ p+1 — the
restore-hold of the stride baseline would leak (the 2026-05-23
``_op_adjacent_token_merge`` incident class, see
research/tests/test_adjacent_token_merge_causality.py). The fix is an
EXCLUSIVE read: output[p] reads the merged summary of the nearest COMPLETED
cluster strictly before p. Every quantity that reaches output[p] — the previous
head's identity, its cluster content, and the STE score weights of its folded
members — is a function of x_{≤ p} only, so the leak is structurally
impossible rather than parameter-dependent.

The novel structure vs the baselines-to-beat:

- ``_op_adjacent_token_merge`` (in-repo): deterministic stride drop —
  content-BLIND, so at any compression it averages salient tokens into filler.
  Here compression follows agreement: a distinct token disagrees with both
  neighbours on the overlap, stays a singleton cluster, and survives BIT-EXACT
  in the compressed stream (pinned in-suite).
- ToMe-style bipartite soft matching: full-space cosine similarity + soft
  (softmax-normalized) merge weights. Here agreement lives on a LEARNED k-dim
  overlap (two tokens may merge while dissimilar overall, or be kept apart
  while cosine-close), selection is hard-local with a Lorentzian
  bounded-reciprocal backward (the NM-C11/C20/F9.1 non-softmax convention),
  and no pairwise L×L score matrix ever exists.

Collapse modes and gates (this lane's anti-collapse section in the note):

- Full pile-up — everything agrees, the whole sequence chains into ONE cluster
  (a global mean). ``merge_collapse_loss`` = (mean w)² is differentiable, 0 at
  no agreement and EXACTLY 1 at full agreement. It also fires on the
  degenerate spurious-agreement mode ρ_L, ρ_R → 0 (δ ≡ 0 ⟹ w ≡ 1).
- No-merge — the compaction win evaporates; ``merge_budget_loss(target)``
  drives the soft merge rate toward a target ratio and is the gradient path
  for the threshold τ (the hard comparison itself has no gradient).
- ``merge_rate`` / ``compression_ratio`` diagnostics for ledger metadata.

Identity-at-init: ReZero scale = 0 ⟹ the module is exactly ``x``. Cross-token
BY DESIGN (this is a sequence compressor), so the NM-11 twin test carries no
pointwise waiver — the detector must measure not-a-twin on the mechanism
itself. NM-10-measurable. Params: 2·k·d restrictions + d² read lift + τ + α;
the mechanism core is O(k·d). Registry wiring DEFERRED (NM-C3/C5/C8/C9/C11/C20
convention).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn


def token_merge_param_count(dim: int, overlap_dim: int) -> int:
    """Exact trainable parameter count.

    ``2·k·d`` restriction maps + ``d²`` read lift + 1 threshold logit
    + 1 ReZero scale. The mechanism core (restrictions + threshold) is O(k·d).
    """
    _validate(dim, overlap_dim)
    return 2 * overlap_dim * dim + dim * dim + 2


def _validate(dim: int, overlap_dim: int) -> None:
    if dim < 1 or overlap_dim < 1:
        raise ValueError(f"need dim>=1 and overlap_dim>=1; got {dim=}, {overlap_dim=}")
    if overlap_dim >= dim:
        raise ValueError(
            "overlap_dim must be < dim (the sheaf overlap is a PROPER restriction "
            f"of the token stalk); got {overlap_dim=} >= {dim=}"
        )


class MergeStructure(NamedTuple):
    """Causal merge decomposition of a batch of sequences.

    ``scores``: (B, L-1) Lorentzian agreement of each adjacent pair.
    ``fold``: (B, L) bool — token p folds forward into p+1 (last position never).
    ``is_head``: (B, L) bool — cluster heads = the compressed stream positions.
    ``head_of``: (B, L) long — the head every position folds into (itself if head).
    ``merged``: (B, L, D) — cluster superposition ``Σ x_j c_j`` at head
    positions (undefined elsewhere).
    ``prev_head``: (B, L) long — nearest head STRICTLY before p, −1 if none.
    """

    scores: torch.Tensor
    fold: torch.Tensor
    is_head: torch.Tensor
    head_of: torch.Tensor
    merged: torch.Tensor
    prev_head: torch.Tensor


class TokenMergeMix(nn.Module):
    """NM-C12 — causal mixer that merges sheaf-consistent adjacent tokens and
    lets every position read the compressed stream exclusively (strict past).

    ``forward(x)`` = ``x + α · (V·read) ⊙ (1 + x)`` where ``read[p]`` is the
    superposed value of the nearest completed cluster strictly before ``p``
    (zero at the start of the sequence). ``compressed_stream`` exposes the
    K-token stream for downstream O(K) consumers; ``merge_*`` methods are the
    gates. Default ``lorentz_gamma=2`` puts the fold boundary at the median
    adjacent-pair discrepancy of unit-Gaussian input (δ ≈ 2), so the mechanism
    ENGAGES (~50% fold rate) at typical activation scale instead of sitting
    inert; the learned threshold τ moves the operating point from there.
    """

    def __init__(
        self,
        dim: int,
        *,
        overlap_dim: int = 8,
        lorentz_gamma: float = 2.0,
        gate_temp: float = 0.25,
        max_cluster: int = 8,
    ) -> None:
        super().__init__()
        _validate(dim, overlap_dim)
        if lorentz_gamma <= 0:
            raise ValueError(f"lorentz_gamma must be > 0, got {lorentz_gamma}")
        if gate_temp <= 0:
            raise ValueError(f"gate_temp must be > 0, got {gate_temp}")
        if max_cluster < 2:
            raise ValueError(f"max_cluster must be >= 2, got {max_cluster}")
        self.dim = int(dim)
        self.overlap_dim = int(overlap_dim)
        self.lorentz_gamma = float(lorentz_gamma)
        self.gate_temp = float(gate_temp)
        self.max_cluster = int(max_cluster)

        # Restriction maps of the two adjacent stalks onto the shared overlap.
        # Equal init ⟹ δ_p = ‖ρ(x_p − x_{p+1})‖²/k at init (agreement starts as
        # similarity in a random k-dim projection); training bends ρ_L ≠ ρ_R
        # into asymmetric overlap semantics.
        rho = torch.randn(overlap_dim, dim) / dim**0.5
        self.rho_left = nn.Parameter(rho.clone())
        self.rho_right = nn.Parameter(rho.clone())
        # τ = sigmoid(threshold_logit); gradient reaches it via merge_budget_loss.
        self.threshold_logit = nn.Parameter(torch.zeros(()))
        self.out_lift = nn.Parameter(torch.randn(dim, dim) / dim**0.5)
        # ReZero: 0 at init ⟹ the module is exactly the identity.
        self.scale = nn.Parameter(torch.zeros(()))

    @property
    def num_parameters(self) -> int:
        return token_merge_param_count(self.dim, self.overlap_dim)

    def threshold(self) -> torch.Tensor:
        """Merge threshold τ ∈ (0, 1)."""
        return torch.sigmoid(self.threshold_logit)

    def merge_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Lorentzian sheaf-agreement of each adjacent pair: ``(B, L-1)`` in (0, 1].

        1 = the two restrictions coincide on the overlap (sections glue);
        → 0 as the overlap discrepancy grows. Bounded-reciprocal, NON-softmax.
        """
        if x.ndim != 3:
            raise ValueError(f"x must be (B, L, D), got {tuple(x.shape)}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"last dim must be {self.dim}, got {x.shape[-1]}")
        left = torch.einsum("kd,bld->blk", self.rho_left, x[:, :-1])
        right = torch.einsum("kd,bld->blk", self.rho_right, x[:, 1:])
        delta = (left - right).square().mean(dim=-1)
        return 1.0 / (1.0 + delta / self.lorentz_gamma)

    def merge_structure(self, x: torch.Tensor) -> MergeStructure:
        """Compute the full causal merge decomposition (see ``MergeStructure``)."""
        scores = self.merge_scores(x)
        batch, length, _ = x.shape
        pos = torch.arange(length, device=x.device)

        fold = torch.zeros(batch, length, dtype=torch.bool, device=x.device)
        if length > 1:
            fold[:, :-1] = scores > self.threshold()
        # Bounded gluing cover: a forced head every max_cluster positions caps
        # cluster span — no unbounded agreement chain, no forever-open cluster.
        fold = fold & ((pos + 1) % self.max_cluster != 0)
        is_head = ~fold

        # head_of[p] = nearest non-folding position ≥ p (the last position never
        # folds, so a head always exists).
        pos_or_inf = torch.where(is_head, pos.expand(batch, -1), length)
        head_of = pos_or_inf.flip(1).cummin(dim=1).values.flip(1)

        # Pure superposition at heads: Σ x_j c_j — deliberately UNNORMALIZED
        # (weights sum to n ≤ max_cluster, never the simplex; normalized
        # variants measure as softmax twins on NM-11 — see module docstring).
        # Folded members contribute weight 1 forward and their Lorentzian
        # score backward (STE), so agreement earns gradient from the read
        # path; heads contribute constant 1.
        contrib = torch.ones(batch, length, device=x.device, dtype=x.dtype)
        if length > 1:
            soft = scores.to(x.dtype)
            ste = 1.0 + soft - soft.detach()
            contrib = torch.cat(
                [torch.where(fold[:, :-1], ste, contrib[:, :-1]), contrib[:, -1:]],
                dim=1,
            )
        merged = torch.zeros_like(x).scatter_add(
            1, head_of.unsqueeze(-1).expand_as(x), x * contrib.unsqueeze(-1)
        )

        # Nearest head STRICTLY before p (its cluster content is x_{< p}): the
        # exclusive read that makes the content-aware decision leak-proof.
        idx_or_neg = torch.where(is_head, pos.expand(batch, -1), -1)
        latest_head = idx_or_neg.cummax(dim=1).values
        prev_head = torch.cat(
            [latest_head.new_full((batch, 1), -1), latest_head[:, :-1]], dim=1
        )
        return MergeStructure(scores, fold, is_head, head_of, merged, prev_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        structure = self.merge_structure(x)
        gather_idx = structure.prev_head.clamp_min(0)
        read = structure.merged.gather(1, gather_idx.unsqueeze(-1).expand_as(x)) * (
            structure.prev_head >= 0
        ).unsqueeze(-1).to(x.dtype)
        lifted = torch.einsum("ed,bld->ble", self.out_lift, read)
        # Bilinear coupling: content transport (first order) + read×token
        # modulation (second order — outside softmax's linear value algebra).
        return x + self.scale * lifted * (1.0 + x)

    def compressed_stream(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The K-token compressed stream: ``(merged, is_head)``.

        ``merged[b, p]`` is the cluster superposition wherever
        ``is_head[b, p]`` — the K = is_head.sum() tokens a downstream sequence
        op consumes at O(K) instead of O(L). K is content-dependent and ragged
        per batch row, hence the mask form. Amplitude carries cluster mass
        (n for n agreeing members, n ≤ max_cluster); a singleton survives
        bit-exact.
        """
        structure = self.merge_structure(x)
        return structure.merged, structure.is_head

    def merge_rate(self, x: torch.Tensor) -> float:
        """Fraction of tokens folded away: 0 = no compression, →1 = pile-up."""
        with torch.no_grad():
            return float(self.merge_structure(x).fold.float().mean().item())

    def compression_ratio(self, x: torch.Tensor) -> float:
        """K/L — compressed-stream length over input length (mean over batch)."""
        return 1.0 - self.merge_rate(x)

    def merge_collapse_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable pile-up guard: ``(mean_p w_p)²``.

        0 when no pair agrees; EXACTLY 1 when every pair fully agrees — the
        whole sequence chains into one cluster and the read degenerates to a
        global mean. Also fires on the spurious-agreement degeneracy
        ρ_L, ρ_R → 0 (δ ≡ 0 ⟹ w ≡ 1), so restriction maps cannot silently
        buy compression by ignoring content.
        """
        if x.shape[1] < 2:
            return x.new_zeros(())
        return self.merge_scores(x).mean().square()

    def merge_budget_loss(
        self, x: torch.Tensor, target_rate: float = 0.5
    ) -> torch.Tensor:
        """Differentiable compression budget: ``(soft_rate − target)²``.

        ``soft_rate`` = mean σ((w − τ)/temp) — the smooth stand-in for the hard
        fold rate. This is the gradient path to the threshold τ (and to
        non-folded pairs' scores), and it penalizes BOTH collapse and the
        no-merge mode where the compaction win evaporates.
        """
        if not 0.0 <= target_rate <= 1.0:
            raise ValueError(f"target_rate must be in [0, 1], got {target_rate}")
        if x.shape[1] < 2:
            return x.new_zeros(())
        soft_rate = torch.sigmoid(
            (self.merge_scores(x) - self.threshold()) / self.gate_temp
        ).mean()
        return (soft_rate - target_rate).square()
