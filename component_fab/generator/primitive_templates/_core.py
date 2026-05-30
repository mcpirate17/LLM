"""Novel primitive templates the fab can synthesize from ProposalSpecs.

Each class is a small, self-contained ``nn.Module`` that materializes one
math-axis choice the property miner surfaced as unrealized:

- ``TropicalAttention`` — replaces softmax+sum with max-plus over
  (affinity + value), giving sparse winner-take-all attention. Semiring
  swap from the standard euclidean inner product.
- ``TropicalStateSpace`` — combines max-plus algebra with an SSM-style
  running state: ``s[t] = max(A + s[t-1], B @ x[t])``. To our knowledge
  this combination is not in the literature; it sits in the unbuilt
  ``(tropical, O(L), has_state)`` corner of property space.
- ``TopKLinear`` — projects then keeps only the ``k`` largest activations
  per position. Projection swap from dense to top-k sparsity.
- ``FourierBasisLane`` — applies a learned complex linear along the
  sequence axis in the rFFT basis. Basis swap from content to frequency.

All four preserve ``[B, L, D]`` shape and produce finite gradients at init.
"""

from __future__ import annotations


import torch
from torch import nn

from component_fab.harness.rope import RotaryEmbedding, apply_rope


def _disable_torch_compile(fn):
    try:
        return torch.compiler.disable(fn)
    except Exception:
        try:
            return torch._dynamo.disable(fn)
        except Exception:
            return fn


@_disable_torch_compile
def _cumsum_dim1_eager(x: torch.Tensor) -> torch.Tensor:
    """Keep sequence scans out of Inductor's unstable SplitScan lowering."""
    return x.cumsum(dim=1)


@_disable_torch_compile
def _cummax_dim1_eager(x: torch.Tensor) -> torch.Tensor:
    """Keep sequence scans out of Inductor's unstable SplitScan lowering
    (vectorized ``TropicalStateSpace``)."""
    return x.cummax(dim=1).values


def _reciprocal_attn_logits(
    raw: torch.Tensor,
    boost: torch.Tensor,
    tri: torch.Tensor | None,
) -> torch.Tensor:
    """Reciprocal (mutual q→k AND k→q) addressing logits in log space.

    ``raw`` is the unmasked ``q·k`` affinity ``(..., S, S)``; ``boost`` the
    broadcastable ``tanh`` reciprocity strength; ``tri`` the bool
    upper-triangular future mask (or ``None`` for non-causal).

    Future keys are set to exact ``-inf`` (no probability floor → no causal
    leak), and the **combined** logits are re-masked to ``-inf`` afterwards.
    The re-mask is load-bearing: ``log_softmax`` of a ``-inf``-masked row gives
    exactly ``-inf`` at future positions, and ``boost·(-inf)`` is ``nan`` for
    any ``boost <= 0`` (including the ``boost == 0`` init) — which would poison
    the whole row. Re-masking overwrites that ``nan`` before any downstream
    softmax/logsumexp. Shared by every reciprocal/semiring lane.
    """
    reverse = raw.transpose(-2, -1)
    if tri is not None:
        scores = raw.masked_fill(tri, float("-inf"))
        reverse = reverse.masked_fill(tri, float("-inf"))
    else:
        scores = raw
    reciprocal_log = torch.log_softmax(reverse, dim=-1)
    if tri is not None:
        # Future reciprocal_log is -inf. Pin it to a FINITE 0 before the multiply:
        # causality already comes from scores=-inf (so logits stay -inf at future),
        # and this avoids boost·(-inf)=nan in BOTH the forward (boost==0 init) and
        # the backward (∂logits/∂boost = reciprocal_log = -inf → 0·-inf=nan in the
        # gradient reduction). Masking the combined logits afterwards fixes only the
        # forward, not this gradient — so the floor must live here.
        reciprocal_log = reciprocal_log.masked_fill(tri, 0.0)
    return scores + boost * reciprocal_log


class _QKVRopeAttentionBase(nn.Module):
    def __init__(
        self,
        dim: int,
        causal: bool = True,
        *,
        use_rope: bool = False,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale = float(dim) ** -0.5
        self.dim = dim
        self.causal = causal
        self.rope = RotaryEmbedding(dim, max_seq_len=max_seq_len) if use_rope else None

    def _project_affinity_values(
        self,
        x: torch.Tensor,
        *,
        causal_mask_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if self.rope is not None:
            cos, sin = self.rope(seq_len, device=x.device, dtype=x.dtype)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        affinity = torch.einsum("bid,bjd->bij", q, k) * self.scale
        if self.causal:
            mask = torch.triu(
                torch.full(
                    (seq_len, seq_len),
                    causal_mask_value,
                    device=x.device,
                    dtype=x.dtype,
                ),
                diagonal=1,
            )
            affinity = affinity + mask
        return affinity, v


def _causal_sparsemax(logits: torch.Tensor) -> torch.Tensor:
    """Sparsemax along the last dim with a causal mask already baked in.

    Inputs are expected to have masked positions set to a large negative
    value (not ``-inf`` — sparsemax's cumsum is NaN-fragile). Returns a
    sparse probability tensor whose nonzero entries sum to 1 along the
    last axis.
    """
    sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
    k = torch.arange(1, logits.size(-1) + 1, device=logits.device, dtype=logits.dtype)
    cumsum = sorted_logits.cumsum(dim=-1)
    support = 1 + k * sorted_logits > cumsum
    k_star = support.to(logits.dtype).sum(dim=-1, keepdim=True).clamp_min(1.0)
    gather_idx = (k_star.long() - 1).clamp_min(0)
    cumsum_at_star = cumsum.gather(-1, gather_idx)
    tau = (cumsum_at_star - 1) / k_star
    return (logits - tau).clamp_min(0.0)


def _pick_n_heads(dim: int, preferred: int = 8) -> int:
    """Largest head count <= ``preferred`` that evenly divides ``dim`` and
    leaves an even head_dim (RoPE requires even dims). Falls back to 1."""
    for h in (preferred, 6, 4, 3, 2):
        if dim % h == 0 and (dim // h) % 2 == 0:
            return h
    return 1


def _heads_for_head_dim(dim: int, target_head_dim: int) -> int:
    """Pick the head count whose head_dim (a divisor of ``dim`` with even size,
    RoPE requires even) is closest to ``target_head_dim``.

    Fixing head_dim — rather than n_heads — keeps every width's per-head subspace
    near the empirical induction-nearest sweet spot (~96 dims): the nano single
    head (dim 96) scores indNear 0.46, but the SAME mechanism collapses both when
    the head is too wide (dim 576 single head → 0.115) and too narrow (dim 12,
    i.e. 8 heads at width 96 → 0.135). So we add heads as the model widens and
    hold head_dim fixed."""
    best_h, best_gap = 1, abs(dim - target_head_dim)
    for h in range(1, dim + 1):
        if dim % h != 0:
            continue
        hd = dim // h
        if hd % 2 != 0:
            continue
        gap = abs(hd - target_head_dim)
        if gap < best_gap:
            best_h, best_gap = h, gap
    return best_h


__all__ = [
    "_disable_torch_compile",
    "_cumsum_dim1_eager",
    "_cummax_dim1_eager",
    "_reciprocal_attn_logits",
    "_QKVRopeAttentionBase",
    "_causal_sparsemax",
    "_pick_n_heads",
    "_heads_for_head_dim",
]
