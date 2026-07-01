"""Per-token routing lanes that preserve ``[B, L, D]`` shape and causality."""

from __future__ import annotations

import math
from typing import Callable, cast

import torch
from torch import nn

from component_fab.harness.top_ar_block import RMSNorm

LaneFactory = Callable[[int], nn.Module]


class RecursionSite(nn.Module):
    """Per-token learned recursion over any shape-preserving weighted site.

    The wrapped module must preserve the ``[B, L, D]`` contract. This is the
    reusable core for making recursion a search axis over weighted sites instead
    of a one-off mixer lane.
    """

    def __init__(
        self,
        module: nn.Module,
        dim: int,
        max_depth: int = 4,
        halt_temp: float = 1.0,
        epsilon: float = 0.05,
        site_name: str = "mixer",
    ) -> None:
        super().__init__()
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self.mixer = module
        # Pre-norm the recursion input (same fix as block_templates.Recursive
        # DepthBlock): without it the mixer is fed its own geometrically
        # growing residual stream, which NaNs deep/high-gain variants.
        self.norm = RMSNorm(dim)
        self.halt_head = nn.Linear(dim, 1)
        self.dim = dim
        self.max_depth = max_depth
        self.halt_temp = float(halt_temp)
        self.epsilon = float(epsilon)
        self.site_name = str(site_name)
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(
                f"RecursionSite expects [B, L, {self.dim}], got {tuple(x.shape)}"
            )
        h = x
        remainder = torch.ones(
            x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype
        )
        delta_step = torch.zeros_like(x)
        delta_total = torch.zeros_like(x)
        expected_steps = torch.zeros_like(remainder)
        for depth in range(self.max_depth):
            mix_out = self.mixer(self.norm(h))
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


class MixtureOfRecursionsLane(RecursionSite):
    """Per-token learned halting over repeated applications of one mixer."""

    def __init__(
        self,
        mixer_factory: LaneFactory,
        dim: int,
        max_depth: int = 4,
        halt_temp: float = 1.0,
        epsilon: float = 0.05,
    ) -> None:
        super().__init__(
            mixer_factory(dim),
            dim,
            max_depth=max_depth,
            halt_temp=halt_temp,
            epsilon=epsilon,
            site_name="mixer",
        )


# Canonical apply order for stacked site recursion. ``block`` is intentionally
# absent: block-level recursion already lives in ``block_templates`` as the
# ``recursive_depth`` / ``recursive_depth_router`` templates, so duplicating it
# here would just give two paths to the same thing.
RECURSION_SITES: tuple[str, ...] = ("embedding", "mixer", "ffn", "router")


class SiteRecursionStack(nn.Module):
    """Per-token recursion over several weighted sites, not just the mixer.

    Each requested site's module is wrapped in a :class:`RecursionSite` and
    applied residually in the fixed :data:`RECURSION_SITES` order, so the axis
    can list sites in any order and the compiled lane is deterministic. The
    stack itself preserves the ``[B, L, D]`` contract and returns the total
    residual delta (like a single ``RecursionSite``), so it drops into the same
    ``x = x + lane(norm(x))`` slot a mixer lane occupies. This is the
    "recursion anywhere there are weights" generalization of the depth router.
    """

    def __init__(
        self,
        sites: dict[str, nn.Module],
        dim: int,
        *,
        depths: dict[str, int] | None = None,
        default_depth: int = 4,
    ) -> None:
        super().__init__()
        if not sites:
            raise ValueError("SiteRecursionStack requires at least one site")
        unknown = sorted(set(sites) - set(RECURSION_SITES))
        if unknown:
            raise ValueError(
                f"unknown recursion sites {unknown}; supported={list(RECURSION_SITES)}"
            )
        depths = depths or {}
        ordered = tuple(name for name in RECURSION_SITES if name in sites)
        self.site_names = ordered
        self.stages = nn.ModuleList(
            RecursionSite(
                sites[name],
                dim,
                max_depth=int(depths.get(name, default_depth)),
                site_name=name,
            )
            for name in ordered
        )
        self.dim = dim
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(
                f"SiteRecursionStack expects [B, L, {self.dim}], got {tuple(x.shape)}"
            )
        h = x
        aux = x.new_zeros(())
        for stage in self.stages:
            site = cast(RecursionSite, stage)
            h = h + site(h)
            aux = aux + site.aux_loss
        self.aux_loss = aux
        return h - x


class SparseMoRLane(nn.Module):
    """Mixture-of-Recursions with a hard per-token recursion budget."""

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
        # Pre-norm the recursion input — same NaN fix as MixtureOfRecursionsLane.
        self.norm = RMSNorm(dim)
        self.halt_head = nn.Linear(dim, 1)
        self.dim = dim
        self.max_depth = max_depth
        self.top_k_frac = float(top_k_frac)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        delta = self.mixer(self.norm(x))
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
            extra = self.mixer(self.norm(h))
            delta = delta + mask * extra
            h = h + mask * extra
        return delta


class LowInfoSkipRouter(nn.Module):
    """Skip-route low-norm positions to an identity or low-rank path."""

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
    """Deterministic hash-routed Mixture-of-Experts."""

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
    """Per-token soft router selecting easy vs hard mixer."""

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


class AuctionCapacityRouter(nn.Module):
    """Causal hard-capacity router with auction-style expert prices.

    MiniMax-M3-align M3X-R1. This is a non-softmax router: tokens bid for
    experts with raw linear scores, experts accumulate prices as they are used,
    and hard capacity masks prevent overfilled experts from receiving more
    tokens. The scan is left-to-right, so token ``t`` depends only on bids from
    tokens ``<= t`` and never leaks future route demand into earlier positions.
    """

    def __init__(
        self,
        dim: int,
        n_experts: int = 4,
        capacity_factor: float = 1.0,
        price_step: float = 1.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if n_experts < 2:
            raise ValueError("n_experts must be >= 2")
        if capacity_factor < 1.0:
            raise ValueError("capacity_factor must be >= 1.0")
        if price_step <= 0.0:
            raise ValueError("price_step must be positive")
        self.bid_proj = nn.Linear(dim, n_experts)
        self.dim = dim
        self.n_experts = n_experts
        self.capacity_factor = float(capacity_factor)
        self.price_step = float(price_step)

    def capacity(self, seq_len: int) -> int:
        """Per-batch per-expert token capacity for a sequence length."""
        return max(1, math.ceil(self.capacity_factor * seq_len / self.n_experts))

    @staticmethod
    def _surrogate_weights(adjusted: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        masked = adjusted.masked_fill(~available, -1e9)
        shifted = masked - masked.min(dim=-1, keepdim=True).values
        positive = torch.relu(shifted) * available.to(adjusted.dtype)
        empty = positive.sum(dim=-1, keepdim=True) <= 1e-12
        fallback = available.to(adjusted.dtype)
        positive = torch.where(empty, fallback, positive)
        return positive / positive.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def route_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Hard one-hot route weights ``[B, L, n_experts]`` with STE gradients."""
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(
                f"AuctionCapacityRouter expects [B, L, {self.dim}], got {tuple(x.shape)}"
            )
        batch, seq_len, _ = x.shape
        bids = self.bid_proj(x)
        prices = bids.new_zeros(batch, self.n_experts)
        loads = torch.zeros(
            batch, self.n_experts, dtype=torch.long, device=bids.device
        )
        capacity = self.capacity(seq_len)
        routes: list[torch.Tensor] = []
        for pos in range(seq_len):
            available = loads < capacity
            adjusted = bids[:, pos, :] - prices
            masked = adjusted.masked_fill(~available, -1e9)
            chosen = masked.argmax(dim=-1)
            hard = torch.zeros_like(adjusted).scatter_(1, chosen.unsqueeze(-1), 1.0)
            surrogate = self._surrogate_weights(adjusted, available)
            routes.append(hard + surrogate - surrogate.detach())
            loads = loads + hard.to(torch.long)
            prices = prices + hard * self.price_step
        return torch.stack(routes, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.route_weights(x)


class AuctionCapacityRoutedLane(nn.Module):
    """Capacity-balanced hard-routed expert lane.

    The router is the mechanism under test: it performs a causal auction over a
    small set of pointwise experts, giving every expert bounded load without a
    softmax probability simplex or post-hoc load-balancing loss.
    """

    def __init__(self, dim: int, n_experts: int = 4) -> None:
        super().__init__()
        self.router = AuctionCapacityRouter(dim, n_experts=n_experts)
        self.experts = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(n_experts)]
        )
        self.output_scale = nn.Parameter(torch.full((dim,), 20.0))
        self.dim = dim
        self.n_experts = n_experts
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def route_weights(self, x: torch.Tensor) -> torch.Tensor:
        return self.router.route_weights(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.router.route_weights(x)
        expert_out = torch.stack([expert(x) for expert in self.experts], dim=-2)
        loads = weights.detach().sum(dim=(0, 1))
        target = loads.mean().clamp_min(1.0)
        self.aux_loss = ((loads - target) / target).pow(2).mean() * 0.001
        return self.output_scale * (weights.unsqueeze(-1) * expert_out).sum(dim=-2)


class RoutedBottleneckLane(nn.Module):
    """Switch-style top-k MoE with auxiliary load-balancing loss."""

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
    "site_recursion",
    "sparse_depth",
    "low_info_skip",
    "hash",
    "difficulty",
    "auction_capacity",
    "top_k_moe",
)
