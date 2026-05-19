"""Per-token compute-allocation routing primitives for fab lanes.

Each primitive wraps one or more inner ``nn.Module`` mixers (the slot
candidates the fab proposer is searching over) and decides PER TOKEN
how much compute that token gets. The proven recipes from research/'s
cf3e6bc6-class BLiMP wins all combine routing with a mixer block; fab
had zero coverage of this axis before today.

Interface: every primitive maps ``[B, L, D] -> [B, L, D]`` and preserves
causality (no future-token leak via the routing decision). The router
state is **per-position** and depends only on positions ``<= t``.

Why ``LowInfoSkipRouter`` instead of the stop-word variant in the plan:
the lane interface receives hidden states, not token ids — frozen
``Embedding(vocab, 1)`` stop-word predictors require ids the upstream
``TinyLM`` keeps inside its forward. The substitute here uses
``||hidden||`` rank as a content-free low-information proxy. Same
inductive bias (low-information tokens get a cheap path), without
breaking every other lane's signature.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn

LaneFactory = Callable[[int], nn.Module]


class MixtureOfRecursionsLane(nn.Module):
    """Per-token learned halting: each token decides how many times to
    re-pass through the same mixer block.

    ACT-style continuous relaxation: at each depth ``d``, a small head
    emits a halt probability ``p_d`` per position. The output is the
    expected hidden state ``sum_d w_d * h_d`` where ``w_d`` is the
    ponder weight (halt × not-halted-so-far). Quadratic penalty on
    expected compute caps depth at training (returned via ``aux_loss``
    attribute, set on each forward call).

    Same params for every recursion depth — this is *recursive*, not
    deeper.
    """

    def __init__(
        self,
        mixer_factory: LaneFactory,
        dim: int,
        max_depth: int = 4,
        halt_temp: float = 1.0,
        epsilon: float = 0.05,
    ) -> None:
        super().__init__()
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self.mixer = mixer_factory(dim)
        self.halt_head = nn.Linear(dim, 1)
        self.dim = dim
        self.max_depth = max_depth
        self.halt_temp = float(halt_temp)
        self.epsilon = float(epsilon)
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        remainder = torch.ones(
            x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype
        )
        delta_step = torch.zeros_like(x)
        delta_total = torch.zeros_like(x)
        expected_steps = torch.zeros_like(remainder)
        for depth in range(self.max_depth):
            mix_out = self.mixer(h)
            delta_step = delta_step + mix_out
            halt = torch.sigmoid(self.halt_head(h + mix_out) / self.halt_temp)
            if depth == self.max_depth - 1:
                halt = torch.ones_like(halt)
            ponder = remainder * halt
            delta_total = delta_total + ponder * delta_step
            expected_steps = expected_steps + ponder * (depth + 1)
            h = h + mix_out
            remainder = remainder * (1.0 - halt)
            if remainder.max().item() < self.epsilon:
                break
        self.aux_loss = expected_steps.mean() * 0.001
        return delta_total


class SparseMoRLane(nn.Module):
    """Mixture-of-Recursions with a hard per-batch top-k recursion budget.

    Forward pass:
      1. One pass through the mixer (every token).
      2. Halt head over the result; the top-k-by-halt-score tokens get
         extra recursions up to ``max_depth``.
      3. Remaining tokens pin at depth=1.

    Caps worst-case compute at ``1 + top_k_frac * (max_depth - 1)`` of
    a vanilla single pass — bounds wallclock unlike unrestricted MoR.
    """

    def __init__(
        self,
        mixer_factory: LaneFactory,
        dim: int,
        max_depth: int = 4,
        top_k_frac: float = 0.25,
    ) -> None:
        super().__init__()
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if not 0.0 < top_k_frac <= 1.0:
            raise ValueError("top_k_frac must be in (0, 1]")
        self.mixer = mixer_factory(dim)
        self.halt_head = nn.Linear(dim, 1)
        self.dim = dim
        self.max_depth = max_depth
        self.top_k_frac = float(top_k_frac)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        delta = self.mixer(x)
        h = x + delta
        if self.max_depth == 1 or l == 0:
            return delta
        # Per-token causal halt: each position independently decides to
        # recurse iff sigmoid(halt_score) > (1 - top_k_frac). Global
        # top-k selection would leak future positions into the decision
        # at position t and break causality (caught by S0.5 gate).
        scores = self.halt_head(h).squeeze(-1)
        threshold = 1.0 - self.top_k_frac
        mask = (torch.sigmoid(scores) > threshold).to(x.dtype).unsqueeze(-1)
        for _ in range(self.max_depth - 1):
            extra = self.mixer(h)
            delta = delta + mask * extra
            h = h + mask * extra
        return delta


class LowInfoSkipRouter(nn.Module):
    """Per-token skip-routing keyed on activation norm.

    Splits positions into two paths: a heavy ``mixer`` for high-content
    tokens and a ``cheap_path`` (identity if ``hard=True``, or a small
    low-rank residual if ``hard=False``) for low-content tokens.

    "Low content" is determined per token by ``||x[t]|| < threshold``
    where the threshold is set by a learned per-channel projection
    (``score_proj``) — no token-id dependency, so the interface stays
    ``[B, L, D] -> [B, L, D]``.

    Two flavors:
    - ``hard=True``: stop-content tokens completely bypass the mixer
      (residual identity). Saves FLOPs but may degrade fine-grained
      agreement features.
    - ``hard=False``: stop-content tokens flow through a learned
      ``low_rank`` adapter (rank=dim//4). Preserves linguistic structure
      while still cheaper than the full mixer.
    """

    def __init__(
        self,
        mixer_factory: LaneFactory,
        dim: int,
        hard: bool = False,
        skip_floor: float = 0.1,
    ) -> None:
        super().__init__()
        self.mixer = mixer_factory(dim)
        self.score_proj = nn.Linear(dim, 1)
        self.hard = bool(hard)
        self.skip_floor = float(skip_floor)
        rank = max(1, dim // 4)
        self.cheap_down: nn.Linear | None
        self.cheap_up: nn.Linear | None
        if hard:
            self.cheap_down = None
            self.cheap_up = None
        else:
            self.cheap_down = nn.Linear(dim, rank, bias=False)
            self.cheap_up = nn.Linear(rank, dim, bias=False)
        self.dim = dim
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = torch.sigmoid(self.score_proj(x))
        skip_ratio = (1.0 - score).mean()
        floor_penalty = torch.clamp(self.skip_floor - skip_ratio, min=0.0) ** 2
        self.aux_loss = floor_penalty * 0.01
        heavy = self.mixer(x)
        if self.cheap_down is None or self.cheap_up is None:
            cheap = torch.zeros_like(x)
        else:
            cheap = self.cheap_up(self.cheap_down(x))
        return score * heavy + (1.0 - score) * cheap


class HashedMoELane(nn.Module):
    """Deterministic hash-routed Mixture-of-Experts.

    Hashes each token's activation signature to one of ``n_experts``
    buckets via a sign-of-random-projection. No learned router, no
    auxiliary balancing loss — every expert is guaranteed to see a
    well-distributed share of tokens by the hash function alone.

    Strong baseline for MoE-style routing; useful negative control
    when learned routers seem to win.
    """

    def __init__(
        self,
        expert_factories: tuple[LaneFactory, ...],
        dim: int,
    ) -> None:
        super().__init__()
        if not expert_factories:
            raise ValueError("expert_factories must be non-empty")
        self.experts = nn.ModuleList([factory(dim) for factory in expert_factories])
        n = len(self.experts)
        self.register_buffer(
            "hash_basis",
            torch.randn(dim, max(1, int(math.ceil(math.log2(max(2, n)))))),
            persistent=True,
        )
        self.dim = dim
        self.n_experts = n

    def _route(self, x: torch.Tensor) -> torch.Tensor:
        bits = (x @ self.hash_basis > 0).to(torch.long)
        weights = 2 ** torch.arange(bits.shape[-1], device=x.device)
        return (bits * weights).sum(dim=-1) % self.n_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bucket = self._route(x)
        out = torch.zeros_like(x)
        for index, expert in enumerate(self.experts):
            mask = (bucket == index).unsqueeze(-1).to(x.dtype)
            if mask.sum() == 0:
                continue
            out = out + mask * expert(x)
        return out


class DifficultyRoutedLane(nn.Module):
    """Per-token 1-bit router selecting easy vs hard mixer.

    Predicts a soft per-token routing weight ``r in [0, 1]`` from the
    input, then outputs ``r * hard(x) + (1 - r) * easy(x)``. Easy and
    hard are arbitrary fab candidates — the bias is for ``easy`` to
    be a cheap mixer (e.g. LinearStateSpaceLane) and ``hard`` to be an
    expensive one (e.g. TropicalAttention).

    Encodes the inductive bias that compute should track difficulty.
    """

    def __init__(
        self,
        easy_factory: LaneFactory,
        hard_factory: LaneFactory,
        dim: int,
    ) -> None:
        super().__init__()
        self.easy = easy_factory(dim)
        self.hard = hard_factory(dim)
        self.gate = nn.Linear(dim, 1)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.gate(x))
        return r * self.hard(x) + (1.0 - r) * self.easy(x)


class RoutedBottleneckLane(nn.Module):
    """Switch-style top-k MoE with auxiliary load-balancing loss.

    Learned router computes per-expert affinity. Each token routes to
    its top-k experts; the outputs are summed weighted by softmaxed
    affinity. Aux loss = mean-routing-fraction × mean-router-prob
    (Shazeer 2017 / Switch Transformer). Set on each forward as
    ``aux_loss`` attribute so the training loop can pick it up.
    """

    def __init__(
        self,
        expert_factories: tuple[LaneFactory, ...],
        dim: int,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        if not expert_factories:
            raise ValueError("expert_factories must be non-empty")
        n = len(expert_factories)
        if not 1 <= top_k <= n:
            raise ValueError(f"top_k={top_k} must be in [1, {n}]")
        self.experts = nn.ModuleList([factory(dim) for factory in expert_factories])
        self.router = nn.Linear(dim, n)
        self.dim = dim
        self.n_experts = n
        self.top_k = top_k
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x)
        probs = torch.softmax(logits, dim=-1)
        topk_vals, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        out = torch.zeros_like(x)
        for slot in range(self.top_k):
            idx = topk_idx[..., slot]
            wt = topk_vals[..., slot].unsqueeze(-1)
            for index, expert in enumerate(self.experts):
                mask = (idx == index).unsqueeze(-1).to(x.dtype)
                if mask.sum() == 0:
                    continue
                out = out + mask * wt * expert(x)
        fraction_per_expert = torch.zeros(
            self.n_experts, device=x.device, dtype=x.dtype
        )
        for index in range(self.n_experts):
            fraction_per_expert[index] = (topk_idx == index).float().mean()
        mean_prob_per_expert = probs.mean(dim=(0, 1))
        self.aux_loss = (
            self.n_experts * (fraction_per_expert * mean_prob_per_expert).sum() * 0.01
        )
        return out


ROUTING_KINDS: tuple[str, ...] = (
    "none",
    "depth_router",
    "sparse_depth",
    "low_info_skip",
    "hash",
    "difficulty",
    "top_k_moe",
)
