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
SCORE_NORM_FAMILIES = ("softmax", "sharpen")
AGGREGATE_FAMILIES = ("mean", "semiring")

_NEG = -1e9


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

    def _score_norm(self, scores: torch.Tensor) -> torch.Tensor:
        """scores -> weights (B,S,S) over keys. Default: softmax."""
        if self.spec.score_norm == "softmax":
            return torch.softmax(scores, dim=-1)
        return torch.softmax(scores * torch.exp(self.log_tau), dim=-1)  # sharpen

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
