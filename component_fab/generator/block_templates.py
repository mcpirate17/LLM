"""Block-level templates for multi-lane fab compositions."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from component_fab.harness.top_ar_block import RMSNorm

LaneFactory = Callable[[int], nn.Module]

BLOCK_TEMPLATES: tuple[str, ...] = (
    "latent_compress",
    "three_lane_adaptive",
    "recursive_depth",
    "gated_parallel",
    "recursive_depth_router",  # full block recurses with halt routing per token
    "sparse_moe_block",  # block-level MoE: anchor + N experts via top-k router
    "hetero_moe_block",  # heterogeneous experts: anchor + diverse-class lanes via softmax router
    "hyperbolic_bridge",  # euclidean ↔ Poincaré-ball chart bridging
    "attn_spectral_filter",  # attention composed with spectral filtering
    "graph_attention",  # edge-conditioned attention with learned adjacency
    "top_ar_block",  # dual-mixer scaffold from fp 7fb0412ec57a1213 (top AR-curriculum scorer)
)


class LatentCompressBlock(nn.Module):
    """Run the mixer in a compressed inner dimension and project back."""

    def __init__(
        self,
        mixer_factory: LaneFactory,
        dim: int,
        compress: int = 2,
    ) -> None:
        super().__init__()
        if compress < 1:
            raise ValueError("compress must be >= 1")
        inner = max(4, dim // compress)
        self.down = nn.Linear(dim, inner, bias=False)
        self.mixer = mixer_factory(inner)
        self.up = nn.Linear(inner, dim, bias=False)
        self.dim = dim
        self.inner_dim = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.down(x)
        m = self.mixer(z)
        return self.up(m)


class ThreeLaneAdaptive(nn.Module):
    """Three parallel mixer lanes summed by a per-token softmax gate."""

    def __init__(
        self,
        anchor_factory: LaneFactory,
        attn_factory: LaneFactory,
        ssm_factory: LaneFactory,
        dim: int,
        *,
        load_balance: bool = False,
        load_balance_gamma: float = 1e-3,
    ) -> None:
        super().__init__()
        self.lane_a = anchor_factory(dim)
        self.lane_b = attn_factory(dim)
        self.lane_c = ssm_factory(dim)
        self.gate = nn.Linear(dim, 3)
        self.dim = dim
        self.load_balance = bool(load_balance)
        self.load_balance_gamma = float(load_balance_gamma)
        self.register_buffer("_moe_balance_bias", torch.zeros(3), persistent=True)

    def _update_load_balance(self, biased_logits: torch.Tensor) -> None:
        if not self.training or not self.load_balance or self.load_balance_gamma <= 0:
            return
        with torch.no_grad():
            chosen = biased_logits.argmax(dim=-1)
            counts = torch.bincount(chosen.flatten(), minlength=3).to(
                self._moe_balance_bias
            )[:3]
            target = counts.sum() / 3.0
            self._moe_balance_bias.add_(self.load_balance_gamma * (target - counts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.lane_a(x)
        b = self.lane_b(x)
        c = self.lane_c(x)
        biased_logits = self.gate(x) + self._moe_balance_bias.detach()
        self._update_load_balance(biased_logits)
        weights = torch.softmax(biased_logits, dim=-1)
        return weights[..., 0:1] * a + weights[..., 1:2] * b + weights[..., 2:3] * c


class RecursiveDepthBlock(nn.Module):
    """Repeated mixer application with sequence-level soft halting."""

    def __init__(
        self,
        mixer_factory: LaneFactory,
        dim: int,
        max_depth: int = 3,
    ) -> None:
        super().__init__()
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self.mixer = mixer_factory(dim)
        # Pre-norm the recursion input: without it the residual stream grows
        # geometrically across depth (mixer fed its own growing output), which
        # NaNs deep/high-gain variants. RMSNorm bounds each step's input so the
        # accumulated stream grows at most linearly (Universal-Transformer style).
        self.norm = RMSNorm(dim)
        self.halt_head = nn.Linear(dim, 1)
        self.dim = dim
        self.max_depth = max_depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        total = torch.zeros_like(x)
        remainder = torch.ones(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)
        for depth in range(self.max_depth):
            mix_out = self.mixer(self.norm(h))
            pooled = (h + mix_out).mean(dim=1, keepdim=True)
            halt = torch.sigmoid(self.halt_head(pooled))
            if depth == self.max_depth - 1:
                halt = torch.ones_like(halt)
            total = total + remainder * mix_out
            h = h + mix_out
            remainder = remainder * (1.0 - halt)
        return total


class GatedParallelBlock(nn.Module):
    """Two parallel mixers + per-position learned residual gate.

    ``y = sigmoid(g(x) + bias) * lane_a(x) + (1 - sigmoid(g(x) + bias)) * lane_b(x)``.

    The two-lane formulation is strictly less expressive than three-lane
    but lets us isolate "is mixing more useful than parallel-A-or-B".
    Lane choices: anchor primitive + a wavelet multiscale lane (the
    cheapest expressive non-attention mixer in the catalog).

    ``bias`` is a single scalar updated outside backprop based on observed
    per-batch gate imbalance (DeepSeek-V3 aux-loss-free load balance). Without
    it, the naive sigmoid gate drifts toward whichever lane edges ahead on the
    current gradient signal and abandons the other lane's circuits — see the
    Obsidian note ``tropical_gate_120m_pretrain_README`` §5.3 / §5.4 for
    the 120M failure-mode analysis. The bias starts at zero so existing
    checkpoints load identically and behavior matches the naked gate on step 1.
    """

    def __init__(
        self,
        anchor_factory: LaneFactory,
        wavelet_factory: LaneFactory,
        dim: int,
        *,
        load_balance: bool = False,
        load_balance_gamma: float = 1e-3,
    ) -> None:
        super().__init__()
        self.lane_a = anchor_factory(dim)
        self.lane_b = wavelet_factory(dim)
        self.gate = nn.Linear(dim, 1)
        self.dim = dim
        self.load_balance = bool(load_balance)
        self.load_balance_gamma = float(load_balance_gamma)
        self.register_buffer("_moe_balance_delta", torch.zeros(1), persistent=True)

    def _update_load_balance(self, biased_logit: torch.Tensor) -> None:
        if not self.training or not self.load_balance or self.load_balance_gamma <= 0:
            return
        with torch.no_grad():
            frac_a = (biased_logit > 0).float().mean()
            adjustment = self.load_balance_gamma * (1.0 - 2.0 * frac_a)
            self._moe_balance_delta.add_(adjustment.reshape_as(self._moe_balance_delta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_logit = self.gate(x).squeeze(-1)
        biased_logit = raw_logit + self._moe_balance_delta.detach()
        self._update_load_balance(biased_logit)
        g = torch.sigmoid(biased_logit).unsqueeze(-1)
        return g * self.lane_a(x) + (1.0 - g) * self.lane_b(x)


class RecursiveDepthRouterBlock(nn.Module):
    """Per-token ACT-style recursion over mixer plus FFN projection."""

    def __init__(
        self,
        anchor_factory: LaneFactory,
        dim: int,
        max_depth: int = 4,
        ffn_mult: int = 2,
    ) -> None:
        super().__init__()
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self.mixer = anchor_factory(dim)
        # Pre-norm the mixer and FFN inputs inside the recursion. Without it the
        # residual stream (h = h + block_delta, re-fed to the mixer each step)
        # grows geometrically with depth × mixer-gain and NaNs deep/high-gain
        # variants at random init — the bulk of the screen's "unstable" rejects.
        # Pre-norming bounds each step's contribution (Universal-Transformer /
        # MoR style), so the stream grows at most linearly.
        self.norm_mix = RMSNorm(dim)
        self.norm_ffn = RMSNorm(dim)
        self.ffn_in = nn.Linear(dim, dim * ffn_mult)
        self.ffn_out = nn.Linear(dim * ffn_mult, dim)
        self.halt = nn.Linear(dim, 1)
        self.max_depth = max_depth
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        delta_total = torch.zeros_like(x)
        remainder = torch.ones(
            x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype
        )
        for depth in range(self.max_depth):
            mix_out = self.mixer(self.norm_mix(h))
            h_after_mix = h + mix_out
            ffn_out = self.ffn_out(
                torch.nn.functional.gelu(self.ffn_in(self.norm_ffn(h_after_mix)))
            )
            block_delta = mix_out + ffn_out
            halt = torch.sigmoid(self.halt(h_after_mix + ffn_out))
            if depth == self.max_depth - 1:
                halt = torch.ones_like(halt)
            ponder = remainder * halt
            delta_total = delta_total + ponder * block_delta
            h = h + block_delta
            remainder = remainder * (1.0 - halt)
            if remainder.max().item() < 0.05:
                break
        return delta_total


class SparseMoEBlock(nn.Module):
    """Block-level top-k MoE over anchor and expert blocks."""

    def __init__(
        self,
        anchor_factory: LaneFactory,
        expert_factories: tuple[LaneFactory, ...],
        dim: int,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        if not expert_factories:
            raise ValueError("expert_factories must be non-empty")
        n = len(expert_factories)
        top_k = min(top_k, n)
        self.anchor = anchor_factory(dim)
        self.experts = nn.ModuleList([f(dim) for f in expert_factories])
        self.router = nn.Linear(dim, n)
        self.dim = dim
        self.top_k = top_k
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchor_out = self.anchor(x)
        h = x + anchor_out
        probs = torch.softmax(self.router(h), dim=-1)
        topk_vals, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        expert_out = torch.zeros_like(x)
        for slot in range(self.top_k):
            idx = topk_idx[..., slot]
            wt = topk_vals[..., slot].unsqueeze(-1)
            for index, expert in enumerate(self.experts):
                mask = (idx == index).unsqueeze(-1).to(x.dtype)
                if mask.sum() == 0:
                    continue
                expert_out = expert_out + mask * wt * expert(x)
        self.aux_loss = (probs.mean(dim=(0, 1)) * probs.mean(dim=(0, 1))).sum() * 0.01
        return anchor_out + expert_out


class HeteroMoEBlock(nn.Module):
    """Soft-mixed MoE block with heterogeneous expert factories."""

    def __init__(
        self,
        anchor_factory: LaneFactory,
        hetero_factories: tuple[LaneFactory, ...],
        dim: int,
    ) -> None:
        super().__init__()
        if not hetero_factories:
            raise ValueError("hetero_factories must be non-empty")
        self.anchor = anchor_factory(dim)
        self.experts = nn.ModuleList([f(dim) for f in hetero_factories])
        # +1 for anchor in the soft mix
        self.gate = nn.Linear(dim, len(hetero_factories) + 1)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.gate(x), dim=-1)
        out = weights[..., 0:1] * self.anchor(x)
        for index, expert in enumerate(self.experts):
            out = out + weights[..., index + 1 : index + 2] * expert(x)
        return out


class HyperbolicBridgeBlock(nn.Module):
    """Gated Euclidean and Poincare-ball mixer bridge."""

    def __init__(
        self, anchor_factory: LaneFactory, dim: int, eps: float = 1e-4
    ) -> None:
        super().__init__()
        from .primitive_templates import PoincareAttention

        self.anchor_euclidean = anchor_factory(dim)
        self.anchor_hyperbolic = PoincareAttention(dim)
        self.to_hyperbolic = nn.Linear(dim, dim)
        self.to_euclidean = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, 1)
        self.dim = dim
        self.eps = float(eps)

    def _to_ball(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return x * (torch.tanh(norm * 0.5) / norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.anchor_euclidean(x)
        h_in = self._to_ball(self.to_hyperbolic(x))
        h_mixed = self.anchor_hyperbolic(h_in)
        h = self.to_euclidean(h_mixed)
        g = torch.sigmoid(self.gate(x))
        return g * e + (1.0 - g) * h


class AttnSpectralFilterBlock(nn.Module):
    """Apply a learned spectral filter to anchor output."""

    def __init__(
        self, anchor_factory: LaneFactory, dim: int, max_seq_len: int = 128
    ) -> None:
        super().__init__()
        from .primitive_templates import FourierBasisLane

        self.anchor = anchor_factory(dim)
        self.spectral = FourierBasisLane(dim, max_seq_len=max_seq_len)
        # Per-token gate over (anchor, spectral, anchor+spectral residual).
        self.gate = nn.Linear(dim, 3)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.anchor(x)
        s = self.spectral(a)
        weights = torch.softmax(self.gate(x), dim=-1)
        return (
            weights[..., 0:1] * a + weights[..., 1:2] * s + weights[..., 2:3] * (a + s)
        )


class GraphAttentionBlock(nn.Module):
    """Edge-conditioned attention with learned low-rank adjacency."""

    def __init__(
        self,
        anchor_factory: LaneFactory,
        dim: int,
        graph_rank: int | None = None,
        causal: bool = True,
    ) -> None:
        super().__init__()
        graph_rank = graph_rank or max(2, dim // 8)
        self.anchor = anchor_factory(dim)  # primary mixer
        self.qg = nn.Linear(dim, graph_rank, bias=False)
        self.kg = nn.Linear(dim, graph_rank, bias=False)
        self.adj = nn.Parameter(torch.randn(graph_rank, graph_rank) / (graph_rank**0.5))
        self.v = nn.Linear(dim, dim, bias=False)
        self.scale = float(graph_rank) ** -0.5
        self.dim = dim
        self.causal = causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        qg = self.qg(x)  # [B, L, R]
        kg = self.kg(x)  # [B, L, R]
        v = self.v(x)  # [B, L, D]
        # Edge weight = qg · adj · kg^T
        adj_kg = kg @ self.adj.t()  # [B, L, R]
        affinity = torch.einsum("bir,bjr->bij", qg, adj_kg) * self.scale
        if self.causal:
            mask = torch.triu(
                torch.full((l, l), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            affinity = affinity + mask
        weights = torch.softmax(affinity, dim=-1)
        edge_mixed = torch.einsum("bij,bjd->bid", weights, v)
        anchor_out = self.anchor(x)
        return 0.5 * anchor_out + 0.5 * edge_mixed
