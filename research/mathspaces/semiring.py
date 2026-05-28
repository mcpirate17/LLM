"""Learnable-semiring attention.

Standard attention aggregates values through the (+, ×) ring:

    out_i = Σ_j w_ij · v_j        with  w = softmax(scores)

This op generalises the *value-aggregation operation* to a parametric semiring
whose ⊕ slides from arithmetic-mean (β→0) to max (β→+∞) to min (β→−∞), via a
learned per-head exponent β. Using the weighted log-sum-exp (free-energy) form:

    out_ic = (1/β) · log Σ_j w_ij · exp(β · v_jc)
           = (1/β) · log( w @ exp(β·v) )

Limits (proven by Taylor expansion of the LSE around β=0):
  β → 0    :  Σ_j w_ij v_jc          — exact softmax attention (arithmetic mean)
  β → +∞   :  max_j v_jc (on support) — winner-take-all value selection
  β → −∞   :  min_j v_jc              — value min-pooling

Distinct from α-entmax: entmax changes weight *sparsity* while still doing a
weighted *sum* of values; this changes the *aggregation semiring* itself. The
capability bet: β>0 heads copy the single most-relevant value rather than
blending (matched to induction / exact retrieval), β≈0 heads average.

Cost is O(B·H·S²·hd) — identical to vanilla attention; the exp(β·v) trick avoids
any (S × S × hd) intermediate. Research-path op (no flash/fused kernel); fine at
screening scale.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# |β| below this uses the analytic β→0 limit (Σ w v) to avoid 0/0.
_BETA_EPS = 1e-3
# β is clamped to ±this — an exp-overflow guardrail; learned β is single-digit.
_BETA_MAX = 15.0


def semiring_value_aggregate(
    logw: torch.Tensor, v: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Weighted log-sum-exp aggregation of ``v`` under log-weights ``logw``.

    Args:
        logw: log attention weights, shape ``(B, H, Sq, Sk)`` (rows log-normalised;
            masked positions = ``-inf``).
        v:    values, shape ``(B, H, Sk, hd)``.
        beta: per-head semiring exponent, shape ``(H,)`` (any real).

    Returns:
        ``(B, H, Sq, hd)`` aggregated output.
    """
    w = logw.exp()  # (B, H, Sq, Sk) — a proper attention distribution per row
    out_sum = w @ v  # β→0 analytic limit == softmax attention

    # Clamp β to an exp-safe band (guardrail against pathological training values;
    # the learned regime is single-digit). b_safe avoids a 0/0 in the dead branch
    # of the torch.where below, which would otherwise poison β's gradient.
    beta = beta.clamp(-_BETA_MAX, _BETA_MAX)
    b = beta.view(1, -1, 1, 1)  # (1, H, 1, 1)
    small = b.abs() < _BETA_EPS
    b_safe = torch.where(small, torch.ones_like(b), b)

    bv = b_safe * v  # (B, H, Sk, hd)
    # Stabilise: factor exp(max_j βv_jc) per (batch, head, channel) out of the sum.
    # The shift cancels exactly; it only bounds exp() (bv − m ≤ 0).
    m = bv.max(dim=2, keepdim=True).values  # (B, H, 1, hd)
    e = (bv - m).exp()  # (B, H, Sk, hd) ∈ (0, 1]
    num = (w @ e).clamp_min(1e-20)  # (B, H, Sq, hd)
    out_semi = (m + num.log()) / b_safe  # (m broadcasts over Sq)

    return torch.where(small, out_sum, out_semi)


def semiring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Causal learnable-semiring self-attention.

    Args:
        q, k, v: ``(B, H, S, hd)``.
        beta:    per-head exponent ``(H,)``.
        scale:   attention-logit scale (typically ``hd**-0.5``).

    Returns:
        ``(B, H, S, hd)`` attention output.
    """
    scores = (q @ k.transpose(-2, -1)) * scale  # (B, H, S, S)
    S = q.shape[-2]
    if S > 1:
        causal = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
    logw = F.log_softmax(scores, dim=-1)  # (B, H, S, S)
    return semiring_value_aggregate(logw, v, beta)
