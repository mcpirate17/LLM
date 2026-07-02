"""Parametric op-synthesis substrate (P0).

The system today *selects* token-mixers from a fixed library of ~187 hand-written
ops; genuinely-new mechanisms (tropical, semiring, reciprocal-rank,
data-dependent-decay) are all written in Python by a human/agent. This module is
the first step toward letting the system *invent* mixers instead: an op is a
composition of STAGES, each a learnable family with an **identity-at-init**
guarantee, so a sampled composition starts as plain softmax attention and the
optimizer (or a generator) mutates it from there.

P0 implements three stages — Address (q,k -> scores), Score-norm (scores ->
weights), Aggregate (weights,v -> out) — each with a default that recovers
standard softmax attention and one or more alternatives whose learnable knob is
initialized at the identity value. The contract, asserted by
``test_parametric_ops``: for EVERY ``StageSpec``, the op at init equals plain
single-head softmax attention (so any sampled mechanism is stable, finite, and
gradient-carrying before training). Grammar/dispatch registration is P1 and lives
in gemini-owned ``research/synthesis/{primitives,op_roles,compiler_ops_*}.py`` —
deliberately not touched here to avoid colliding with concurrent work.

Seed pattern this generalizes: ``compiler_ops_attention._op_reciprocal_semiring_
attention`` (a scaffold + ``reciprocal_logit_scale``/``semiring_beta`` knobs that
vanish at init). Here that pattern becomes a composable, sampled substrate.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

ADDRESS_FAMILIES = ("dot", "reciprocal", "cosine")
SCORE_NORM_FAMILIES = ("softmax", "sharpen", "tsallis_q", "renyi", "entmax_alpha")
AGGREGATE_FAMILIES = ("mean", "semiring")

_NEG = -1e9
# How far the learnable Tsallis/Rényi order q may move from 1 (softmax):
# q = 1 + tanh(knob) * _Q_SPAN ∈ (1-span, 1+span). Kept clear of the q→0
# singularity. In this normalized q-exponential ("q-softmax") convention q<1 is
# the sparse hard-cutoff regime and q>1 is the heavy-tailed / flatter regime
# (this is the Tsallis-q map, NOT the entmax-α map, whose sparsity sign flips).
_Q_SPAN = 0.8
# entmax-α order range: α = 1 + _ENTMAX_ALPHA_SPAN*tanh(knob) ∈ (1-span, 1+span).
# α=1 is softmax, α=2 is sparsemax. Span 1.0 lets the knob reach the sparsemax
# endpoint. Distinct from tsallis_q: entmax is the convex projection, so α>1
# produces *exact* zero weights (hard sparsity) the q-exponential normalizer
# never achieves.
_ENTMAX_ALPHA_SPAN = 1.0
# α-window over which the identity-blend gate ramps from pure softmax (α=1) to
# pure entmax. Saturates at α = 1 + width so the deep-sparse regime is PURE
# entmax (exact zeros), not a blend — a softmax blend would leak strictly-positive
# weight onto every position and erase the exact-sparsity property.
_ENTMAX_BLEND_WIDTH = 0.5


@dataclass(frozen=True)
class StageSpec:
    """One concrete op = one family choice per stage.

    The default ``StageSpec()`` is plain softmax attention; every other spec also
    equals softmax attention *at init* (its alternative knobs start at identity).
    """

    address: str = "dot"
    score_norm: str = "softmax"
    aggregate: str = "mean"

    def __post_init__(self) -> None:
        if self.address not in ADDRESS_FAMILIES:
            raise ValueError(f"unknown address family: {self.address!r}")
        if self.score_norm not in SCORE_NORM_FAMILIES:
            raise ValueError(f"unknown score_norm family: {self.score_norm!r}")
        if self.aggregate not in AGGREGATE_FAMILIES:
            raise ValueError(f"unknown aggregate family: {self.aggregate!r}")

    @property
    def key(self) -> str:
        return f"{self.address}|{self.score_norm}|{self.aggregate}"


def all_stage_specs() -> list[StageSpec]:
    """Every P0 mechanism (Cartesian product of the stage families)."""
    return [
        StageSpec(a, s, g)
        for a, s, g in itertools.product(
            ADDRESS_FAMILIES, SCORE_NORM_FAMILIES, AGGREGATE_FAMILIES
        )
    ]


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
    )


class ParametricMix(nn.Module):
    """A single-head mixer whose math is the composition named by ``spec``.

    Single head keeps P0 simple and the identity-at-init proof exact. Every
    alternative family carries a learnable knob initialized so the stage reduces
    to its default, hence the whole op = softmax attention at init.
    """

    def __init__(self, dim: int, spec: StageSpec | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.spec = spec or StageSpec()
        self.scale = float(dim) ** -0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        # Identity-at-init knobs (all registered; only the spec's stages use them).
        self.reciprocal_logit_scale = nn.Parameter(torch.zeros(1))  # tanh(0)=0 -> dot
        self.cosine_gate = nn.Parameter(torch.zeros(1))  # tanh(0)=0 -> dot
        self.log_tau = nn.Parameter(torch.zeros(1))  # exp(0)=1 -> softmax
        self.semiring_beta = nn.Parameter(torch.zeros(1))  # 0 -> weighted mean
        self.tsallis_q_delta = nn.Parameter(
            torch.zeros(1)
        )  # tanh(0)=0 -> q=1 (softmax)
        self.renyi_q_delta = nn.Parameter(torch.zeros(1))  # tanh(0)=0 -> q=1 (softmax)
        self.renyi_log_beta = nn.Parameter(torch.zeros(1))  # exp(0)=1 -> softmax
        self.entmax_alpha_delta = nn.Parameter(
            torch.zeros(1)
        )  # tanh(0)=0 -> α=1 (softmax)

    # ── stages ──────────────────────────────────────────────────────
    def _address(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """(q,k) -> causal-masked raw scores (B,S,S). Default: scaled dot."""
        seq = q.shape[1]
        mask = _causal_mask(seq, q.device)
        dot = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.spec.address == "dot":
            scores = dot
        elif self.spec.address == "reciprocal":
            rev = dot.transpose(-2, -1).masked_fill(mask, _NEG)
            recip = torch.softmax(rev, dim=-1).clamp(min=1e-6)
            boost = torch.tanh(self.reciprocal_logit_scale)
            scores = dot + boost * torch.log(recip)
        else:  # cosine: gated deviation from dot (gate->0 at init)
            qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
            cos = torch.matmul(qn, kn.transpose(-2, -1)) * self.scale
            scores = dot + torch.tanh(self.cosine_gate) * (cos - dot)
        return scores.masked_fill(mask, _NEG)

    @staticmethod
    def _q_exp(z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Tsallis q-exponential ``[1+(1-q)z]_+**(1/(1-q))``, smooth at ``q=1``.

        ``exp_q -> exp`` as ``q -> 1``; the ``log(1+(1-q)z)/(1-q)`` limit is taken
        via a short Taylor series near ``q=1`` so the op is not only equal to
        softmax at init but also carries a *nonzero* gradient wrt ``q`` there
        (the knob is learnable, not inert). ``[·]_+`` gives the sparse cutoff.
        """
        a = 1.0 - q
        az = a * z
        pos = (1.0 + az) > 0.0
        # Margin must survive float32 rounding: 1e-12 << float32 eps (1.19e-7), so
        # -1.0 + 1e-12 collapses to exactly -1.0 and log1p(-1.0) = -inf. The where
        # cutoff below then computes 0 * -inf = NaN in the backward. 1e-4 stays
        # representable (log1p ~ -9.2, finite) and only bites in the already-zeroed
        # hard-cutoff region, so the forward is unchanged.
        az_safe = az.clamp(min=-1.0 + 1e-4)
        small = a.abs() < 1e-3
        a_den = torch.where(small, torch.ones_like(a), a)
        exact = torch.log1p(az_safe) / a_den
        series = z * (1.0 - 0.5 * az + (az * az) / 3.0)  # log(1+az)/a as a->0
        t = torch.where(small, series, exact)
        return torch.where(pos, torch.exp(t), torch.zeros_like(t))

    def _q_softmax(
        self, scores: torch.Tensor, q: torch.Tensor, beta: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Normalized q-exponential over keys; equals softmax at ``q=1, beta=1``."""
        masked = scores <= (_NEG / 2.0)
        z = scores if beta is None else scores * beta
        z_max = z.masked_fill(masked, _NEG).amax(dim=-1, keepdim=True)
        z = (z - z_max).masked_fill(masked, -60.0)  # valid <= 0; masked -> exp ~ 0
        expq = self._q_exp(z, q).masked_fill(masked, 0.0)
        return expq / expq.sum(dim=-1, keepdim=True).clamp(min=1e-20)

    def _entmax_alpha(self, scores: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """α-entmax via convex-projection bisection; softmax at α=1, sparsemax at α=2.

        Tensor-α bisection (differentiable through α), distinct from the float-α
        ``compiler_ops_attention._entmax_bisect`` which serves fixed-config op
        dispatch and clamps α≥1.01 (so it can never equal softmax). Masked score
        positions (``<= _NEG/2``) collapse to exactly zero weight — the hard
        sparsity that makes entmax a different family from tsallis_q. ``alpha`` is
        expected pre-clamped to ``[1+ε, 2]`` by the caller; the α=1 softmax point
        is reached by the blend gate in ``_score_norm``, not by this method.
        """
        masked = scores <= (_NEG / 2.0)
        z = (
            scores - scores.masked_fill(masked, _NEG).amax(dim=-1, keepdim=True)
        ).clamp(min=-20.0)
        scaled = z * (alpha - 1.0)
        power = 1.0 / (alpha - 1.0)
        tau_lo = scaled.amin(dim=-1, keepdim=True) - 1.0
        tau_hi = scaled.amax(dim=-1, keepdim=True)
        for _ in range(20):
            tau = 0.5 * (tau_lo + tau_hi)
            probs = torch.clamp(scaled - tau, min=0.0).pow(power)
            too_large = probs.sum(dim=-1, keepdim=True) > 1.0
            tau_lo = torch.where(too_large, tau, tau_lo)
            tau_hi = torch.where(too_large, tau_hi, tau)
        probs = torch.clamp(scaled - tau_hi, min=0.0).pow(power)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return probs.masked_fill(masked, 0.0)

    def _score_norm(self, scores: torch.Tensor) -> torch.Tensor:
        """scores -> weights (B,S,S) over keys. Default: softmax.

        ``tsallis_q``: normalized Tsallis q-exponential with a learnable order
        ``q = 1 + tanh(knob)*_Q_SPAN`` (q<1 sharpens toward a sparse hard-cutoff
        read, q>1 gives heavier tails / flatter; q=1 is softmax). ``renyi``: the
        full ``(q, beta)`` surface — the
        q-exponential of a learnably sharpened score ``beta*scores``. Both equal
        softmax at init and are *not* softmax twins away from it (the normalizer
        is a q-deformed exponential, not the Gibbs exponential).
        """
        if self.spec.score_norm == "softmax":
            return torch.softmax(scores, dim=-1)
        if self.spec.score_norm == "sharpen":
            return torch.softmax(scores * torch.exp(self.log_tau), dim=-1)
        if self.spec.score_norm == "tsallis_q":
            q = 1.0 + torch.tanh(self.tsallis_q_delta) * _Q_SPAN
            return self._q_softmax(scores, q)
        if self.spec.score_norm == "entmax_alpha":
            alpha = 1.0 + _ENTMAX_ALPHA_SPAN * torch.tanh(self.entmax_alpha_delta)
            alpha_safe = torch.clamp(alpha, min=1.0 + 1e-2, max=2.0)
            # Blend gate is 0 at α=1 (init) so the output is *exactly* softmax
            # there, and saturates at 1 by α = 1 + _ENTMAX_BLEND_WIDTH so the
            # deep-sparse regime is pure entmax (exact zeros). The near-softmax
            # band blends; past it the projection's hard sparsity is unblended.
            gate = torch.clamp((alpha - 1.0) / _ENTMAX_BLEND_WIDTH, min=0.0, max=1.0)
            return (1.0 - gate) * torch.softmax(
                scores, dim=-1
            ) + gate * self._entmax_alpha(scores, alpha_safe)
        q = 1.0 + torch.tanh(self.renyi_q_delta) * _Q_SPAN  # renyi
        return self._q_softmax(scores, q, beta=torch.exp(self.renyi_log_beta))

    def _aggregate(self, weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """(weights,v) -> out (B,S,D). Default: weighted mean (weights @ v).

        ``semiring``: out_id = sum_j p_ij^d v_jd with p^d = softmax_j(log w_ij +
        beta v_jd). At beta=0, p^d = softmax(log w) = w (w is already normalized),
        so it reduces exactly to the weighted mean; beta>0 slides toward max-pool
        (tropical), beta<0 toward min — the mean<->max semiring spectrum.
        """
        if self.spec.aggregate == "mean":
            return torch.matmul(weights, v)
        logw = torch.log(weights.clamp(min=1e-9)).unsqueeze(-1)  # (B,S,S,1)
        vb = v.unsqueeze(1)  # (B,1,S,D)
        p = torch.softmax(logw + self.semiring_beta * vb, dim=2)  # over key dim
        return (p * vb).sum(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        weights = self._score_norm(self._address(q, k))
        return self.o(self._aggregate(weights, v))


def build_parametric_mix(dim: int, spec: StageSpec | None = None) -> ParametricMix:
    """Factory mirroring the lane_factory(dim) convention used by the graders."""
    return ParametricMix(dim, spec)
