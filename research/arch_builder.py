"""
Architecture Builder

Translates an ArchSpec (morphological box choices) into a working PyTorch nn.Module.
Each dimension maps to concrete module implementations.

Design principle: modular composition. Each dimension choice produces an nn.Module
that conforms to a standard interface, then they're composed into a full model.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    RMSNorm,
    DynamicNorm,
    GroupNormWrapper,
    SigmoidNorm,
    RoPE,
    ALiBi,
    ConvPositional,
    LearnedAbsolutePositional,
    RandomFourierPositional,
)

from .morphological_box import ArchSpec
from .arch_builder_config import BuildConfig


# ── Token Representation Modules ───────────────────────────────────────


class DenseRepresentation(nn.Module):
    """Standard: just pass through."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x


class BinaryHashRepresentation(nn.Module):
    """Binary hash codes with straight-through estimator."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        # Straight-through: binarize forward, pass gradient through
        binary = (h > 0).float()
        return h + (binary - h).detach()  # STE

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj_out(x)


class SparseTopKRepresentation(nn.Module):
    """Keep only top-k activations per token."""

    def __init__(self, dim: int, k_ratio: float = 0.25):
        super().__init__()
        self.k = max(1, int(dim * k_ratio))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        topk_vals, topk_idx = x.topk(self.k, dim=-1)
        sparse = torch.zeros_like(x)
        sparse.scatter_(-1, topk_idx, topk_vals)
        # STE: pass gradient through
        return x + (sparse - x).detach()

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ComplexRepresentation(nn.Module):
    """Split dim into real and imaginary, operate in complex space."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Just reshape — the mixing layers will handle complex ops
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MultiResolutionRepresentation(nn.Module):
    """Concatenate fine (identity) and coarse (avg-pooled) representations."""

    def __init__(self, dim: int, pool_size: int = 4):
        super().__init__()
        self.fine_dim = dim // 2
        self.coarse_dim = dim - dim // 2
        self.proj_fine = nn.Linear(dim, self.fine_dim)
        self.proj_coarse = nn.Linear(dim, self.coarse_dim)
        self.pool_size = pool_size

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        fine = self.proj_fine(x)
        # Avg pool along sequence dimension
        B, S, D = x.shape
        pad = (self.pool_size - S % self.pool_size) % self.pool_size
        if pad > 0:
            x_padded = F.pad(x, (0, 0, 0, pad))
        else:
            x_padded = x
        pooled = x_padded.reshape(B, -1, self.pool_size, D).mean(dim=2)
        # Upsample back
        coarse = pooled.repeat_interleave(self.pool_size, dim=1)[:, :S, :]
        coarse = self.proj_coarse(coarse)
        return torch.cat([fine, coarse], dim=-1)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ResidualQuantizedRepresentation(nn.Module):
    """Residual VQ: quantize, then quantize the residual."""

    def __init__(self, dim: int, n_codes: int = 256, n_quantizers: int = 2):
        super().__init__()
        self.codebooks = nn.ParameterList(
            [
                nn.Parameter(torch.randn(n_codes, dim) * 0.01)
                for _ in range(n_quantizers)
            ]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        quantized = torch.zeros_like(x)
        for codebook in self.codebooks:
            # Find nearest code
            dists = torch.cdist(
                residual, codebook.unsqueeze(0).expand(x.shape[0], -1, -1)
            )
            indices = dists.argmin(dim=-1)
            codes = codebook[indices]
            quantized = quantized + codes
            residual = residual - codes.detach()
        # STE
        return x + (quantized - x).detach()

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MixtureEmbeddingRepresentation(nn.Module):
    """Represent tokens as soft mixture of learned prototypes."""

    def __init__(self, dim: int, n_prototypes: int = 64):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, dim) * 0.02)
        self.gate = nn.Linear(dim, n_prototypes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.gate(x), dim=-1)  # (B, S, n_proto)
        return torch.einsum("bsp,pd->bsd", weights, self.prototypes)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ── Token Mixing Modules ──────────────────────────────────────────────


class SoftmaxAttention(nn.Module):
    """Standard multi-head softmax attention."""

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_seq_len: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # GQA expansion
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Causal mask
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn.masked_fill_(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out)


class LinearAttention(nn.Module):
    """Linear attention with ELU kernel."""

    def __init__(self, dim: int, n_heads: int, **kwargs):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = (
            F.elu(
                self.q_proj(x)
                .reshape(B, S, self.n_heads, self.head_dim)
                .transpose(1, 2)
            )
            + 1
        )
        k = (
            F.elu(
                self.k_proj(x)
                .reshape(B, S, self.n_heads, self.head_dim)
                .transpose(1, 2)
            )
            + 1
        )
        v = self.v_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Causal linear attention via cumsum
        kv = torch.einsum("bhsd,bhse->bhsde", k, v)
        kv_cumsum = kv.cumsum(dim=2)
        k_cumsum = k.cumsum(dim=2)

        out = torch.einsum("bhsd,bhsde->bhse", q, kv_cumsum)
        denom = (
            torch.einsum("bhsd,bhsd->bhs", q, k_cumsum).unsqueeze(-1).clamp(min=1e-6)
        )
        out = out / denom

        return self.o_proj(out.transpose(1, 2).reshape(B, S, -1))


class ConvMixer(nn.Module):
    """Pure convolution stack for token mixing."""

    def __init__(self, dim: int, kernel_sizes: Tuple[int, ...] = (3, 5, 7), **kwargs):
        super().__init__()
        self.convs = nn.ModuleList()
        for ks in kernel_sizes:
            self.convs.append(nn.Conv1d(dim, dim, ks, padding=ks - 1, groups=dim))
        self.proj = nn.Linear(dim * len(kernel_sizes), dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        x_t = x.transpose(1, 2)  # (B, D, S)
        outs = []
        for conv in self.convs:
            out = conv(x_t)[:, :, :S]  # causal: trim right padding
            outs.append(out)
        combined = torch.cat(outs, dim=1).transpose(1, 2)  # (B, S, D*n)
        return self.proj(combined)


class StateSpaceMixer(nn.Module):
    """Simplified S4-style state space model."""

    def __init__(self, dim: int, state_dim: int = 16, **kwargs):
        super().__init__()
        self.state_dim = state_dim
        self.A = nn.Parameter(torch.randn(dim, state_dim) * 0.01)
        self.B = nn.Linear(dim, dim * state_dim, bias=False)
        self.C = nn.Linear(dim * state_dim, dim, bias=False)
        self.D = nn.Parameter(torch.ones(dim))
        self.dt_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from .synthesis.compiler import _parallel_associative_scan

        B, S, D = x.shape
        N = self.state_dim
        dt = F.softplus(self.dt_proj(x))  # (B, S, D)

        # Per-timestep input-dependent decay via true parallel scan
        log_a = (self.A.view(1, 1, D, N) * dt.unsqueeze(-1)).clamp(
            -10, 0
        )  # (B, S, D, N)
        b_x = self.B(x).reshape(B, S, D, N)

        # Reshape so sequence dim is last: (B, D, N, S)
        log_a_t = log_a.permute(0, 2, 3, 1).contiguous()
        b_x_t = b_x.permute(0, 2, 3, 1).contiguous()

        h_t = _parallel_associative_scan(log_a_t, b_x_t)

        # (B, D, N, S) -> (B*S, D*N) for output projection
        h = h_t.permute(0, 3, 1, 2).reshape(B * S, D * N)

        y = self.C(h).reshape(B, S, D)
        return y + x * self.D


class FourierMixer(nn.Module):
    """FFT-based global mixing."""

    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim) * 0.02)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFT along sequence dimension
        x_freq = torch.fft.rfft(x, dim=1)
        # Learnable frequency-domain filter
        x_freq = x_freq * self.weight.unsqueeze(0).unsqueeze(0)
        x_time = torch.fft.irfft(x_freq, n=x.shape[1], dim=1)
        out_dim = getattr(self, "out_dim", x_time.shape[-1])
        return (
            self.proj(x_time)
            if self.proj.out_features == out_dim
            else self.proj(x_time[:, :, :out_dim])
        )


class CompressedAttention(nn.Module):
    """Compress tokens before attention (CCGQA-inspired but simpler)."""

    def __init__(self, dim: int, n_heads: int, compression_factor: int = 4, **kwargs):
        super().__init__()
        self.compress = nn.Conv1d(
            dim,
            dim,
            kernel_size=compression_factor,
            stride=compression_factor,
            bias=False,
        )
        self.decompress = nn.ConvTranspose1d(
            dim,
            dim,
            kernel_size=compression_factor,
            stride=compression_factor,
            bias=False,
        )
        self.attn = SoftmaxAttention(
            dim, n_heads, n_heads, kwargs.get("max_seq_len", 512)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # Pad to multiple of compression factor
        cf = self.compress.kernel_size[0]
        pad = (cf - S % cf) % cf
        if pad > 0:
            x_padded = F.pad(x, (0, 0, 0, pad))
        else:
            x_padded = x

        compressed = self.compress(x_padded.transpose(1, 2)).transpose(1, 2)
        attended = self.attn(compressed)
        decompressed = self.decompress(attended.transpose(1, 2)).transpose(1, 2)
        return decompressed[:, :S, :]


class CrossAttentionPool(nn.Module):
    """Cross-attend to learned query tokens, then project back."""

    def __init__(self, dim: int, n_heads: int, n_queries: int = 32, **kwargs):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, dim) * 0.02)
        self.cross_attn = SoftmaxAttention(dim, n_heads, n_heads, n_queries)
        self.back_proj = nn.Linear(dim, dim)
        self.n_queries = n_queries

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        queries = self.queries.expand(B, -1, -1)
        # Simple cross-attention: queries attend to x
        qkv_input = torch.cat([queries, x], dim=1)  # (B, n_q + S, D)
        attended = self.cross_attn(qkv_input)
        # Take only the query outputs, project back to seq length
        q_out = attended[:, : self.n_queries]
        # Broadcast back to full sequence
        weights = F.softmax(torch.einsum("bsd,bqd->bsq", x, q_out), dim=-1)
        return self.back_proj(torch.einsum("bsq,bqd->bsd", weights, q_out))


class RandomFeatureAttention(nn.Module):
    """Random feature approximation of softmax attention."""

    def __init__(self, dim: int, n_heads: int, n_features: int = 64, **kwargs):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.n_features = n_features
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        # Random projection matrix (fixed)
        self.register_buffer(
            "omega", torch.randn(self.head_dim, n_features) / math.sqrt(n_features)
        )

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """Random feature map."""
        proj = x @ self.omega  # (B, H, S, n_features)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) / math.sqrt(
            self.n_features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        q_feat = self._phi(q)
        k_feat = self._phi(k)

        # Linear attention with random features
        kv = torch.einsum("bhsd,bhse->bhde", k_feat, v)
        out = torch.einsum("bhsd,bhde->bhse", q_feat, kv)
        denom = q_feat.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        out = out / denom

        return self.o_proj(out.transpose(1, 2).reshape(B, S, -1))


class DifferentiableSortMixer(nn.Module):
    """Sort-based mixing: sort tokens by learned key, mix, unsort."""

    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.key_proj = nn.Linear(dim, 1)
        self.mix = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        keys = self.key_proj(x).squeeze(-1)  # (B, S)
        # Soft sort via sigmoid temperature
        sorted_idx = keys.argsort(dim=-1)
        # Gather in sorted order
        sorted_x = x.gather(1, sorted_idx.unsqueeze(-1).expand(-1, -1, D))
        # Causal convolution in sorted space
        mixed = self.mix(sorted_x)
        # Unsort
        unsorted_idx = sorted_idx.argsort(dim=-1)
        return mixed.gather(1, unsorted_idx.unsqueeze(-1).expand(-1, -1, D))


class GraphAttention(nn.Module):
    """Dynamic graph attention with learned adjacency."""

    def __init__(self, dim: int, n_heads: int, **kwargs):
        super().__init__()
        self.attn = SoftmaxAttention(
            dim, n_heads, n_heads, kwargs.get("max_seq_len", 512)
        )
        self.edge_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add edge features (pairwise learned transform)
        edges = self.edge_proj(x)
        return self.attn(x + edges)


class IntegralKernelMixer(nn.Module):
    """Functional operator-style token mixing via low-rank integral kernels."""

    def __init__(self, dim: int, n_basis: int = 16, **kwargs):
        super().__init__()
        self.n_basis = n_basis
        self.kernel_in = nn.Linear(dim, n_basis, bias=False)
        self.kernel_out = nn.Linear(dim, n_basis, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Build latent basis anchors from sequence-wide weighted integration
        in_weights = F.softmax(
            self.kernel_in(x), dim=1
        )  # (B, S, K), normalized over sequence
        values = self.value_proj(x)  # (B, S, D)
        anchors = torch.einsum("bsk,bsd->bkd", in_weights, values)  # (B, K, D)

        # Per-token function coefficients project anchors back to token space
        out_weights = F.softmax(
            self.kernel_out(x), dim=-1
        )  # (B, S, K), normalized over basis
        mixed = torch.einsum("bsk,bkd->bsd", out_weights, anchors)
        return self.out_proj(mixed)


# ── Channel Mixing Modules ─────────────────────────────────────────────


class SwiGLUMLP(nn.Module):
    """Standard SwiGLU MLP."""

    def __init__(self, dim: int, mlp_ratio: float = 3.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoETopK(nn.Module):
    """Simple top-k Mixture of Experts."""

    def __init__(
        self, dim: int, n_experts: int = 4, topk: int = 2, mlp_ratio: float = 3.0
    ):
        super().__init__()
        self.experts = nn.ModuleList(
            [SwiGLUMLP(dim, mlp_ratio) for _ in range(n_experts)]
        )
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.topk = topk
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        logits = self.gate(x)
        weights, indices = logits.topk(self.topk, dim=-1)
        weights = F.softmax(weights, dim=-1)

        # Record routing telemetry
        if hasattr(self, "routing_telemetry"):
            # Compute entropy
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()

            # Utilization
            counts = torch.histc(
                indices.float(), bins=self.n_experts, min=0, max=self.n_experts - 1
            )

            rt = self.routing_telemetry
            rt["tokens_total"] = rt.get("tokens_total", 0) + B * S
            rt["tokens_processed"] = rt.get("tokens_processed", 0) + B * S
            rt["expert_counts"] = (
                rt.get("expert_counts", torch.zeros(self.n_experts, device=x.device))
                + counts
            )
            rt["entropy_sum"] = rt.get("entropy_sum", 0.0) + entropy
            rt["count"] = rt.get("count", 0) + 1
        else:
            # Initialize telemetry placeholder if not present (will be extracted by runner)
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
            counts = torch.histc(
                indices.float(), bins=self.n_experts, min=0, max=self.n_experts - 1
            )
            self.routing_telemetry = {
                "tokens_total": B * S,
                "tokens_processed": B * S,
                "expert_counts": counts,
                "entropy_sum": entropy,
                "count": 1,
            }

        # Simple loop-based routing (good enough for small models)
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (indices == i).any(dim=-1)
            if mask.any():
                expert_input = x[mask]
                expert_weight = weights[indices == i].reshape(-1, 1)
                # Accumulate weighted expert outputs
                output[mask] += expert(expert_input) * expert_weight

        return output


class KANSplineMLP(nn.Module):
    """KAN-inspired learnable activation functions via B-splines."""

    def __init__(self, dim: int, mlp_ratio: float = 3.0, n_knots: int = 8):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        # Learnable spline coefficients
        self.spline_coeffs = nn.Parameter(torch.randn(hidden, n_knots) * 0.01)
        self.register_buffer("knots", torch.linspace(-3, 3, n_knots))

    def _spline_activation(self, x: torch.Tensor) -> torch.Tensor:
        # B-spline basis evaluation (simplified: RBF-like)
        diffs = x.unsqueeze(-1) - self.knots  # (..., n_knots)
        basis = torch.exp(-(diffs**2))  # Gaussian basis
        return (basis * self.spline_coeffs).sum(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        h = self._spline_activation(h)
        return self.down(h)


class RWKVChannelMix(nn.Module):
    """RWKV-style channel mixing with time-shift."""

    def __init__(self, dim: int, mlp_ratio: float = 3.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.mix_k = nn.Parameter(torch.ones(dim) * 0.5)
        self.mix_r = nn.Parameter(torch.ones(dim) * 0.5)
        self.key = nn.Linear(dim, hidden, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Time-shift: shift by one position
        shifted = F.pad(x[:, :-1], (0, 0, 1, 0))
        xk = x * self.mix_k + shifted * (1 - self.mix_k)
        xr = x * self.mix_r + shifted * (1 - self.mix_r)
        k = torch.square(torch.relu(self.key(xk)))
        return torch.sigmoid(self.receptance(xr)) * self.value(k)


class Conv1dGLU(nn.Module):
    """1D convolution with GLU gating."""

    def __init__(self, dim: int, kernel_size: int = 5, mlp_ratio: float = 3.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.conv = nn.Conv1d(
            dim, hidden * 2, kernel_size, padding=kernel_size - 1, groups=1
        )
        self.proj = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2))[:, :, : x.shape[1]]  # causal
        h = h.transpose(1, 2)
        gate, val = h.chunk(2, dim=-1)
        return self.proj(F.silu(gate) * val)


class PolynomialExpansion(nn.Module):
    """Polynomial feature expansion."""

    def __init__(self, dim: int, degree: int = 2):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim, bias=False)
        self.proj_out = nn.Linear(dim, dim, bias=False)
        self.degree = degree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        result = h
        power = h
        for _ in range(self.degree - 1):
            power = power * h
            result = result + power
        return self.proj_out(result)


class BasisExpansionLayer(nn.Module):
    """Function-space basis expansion with learned per-token coefficients."""

    def __init__(self, dim: int, n_basis: int = 8):
        super().__init__()
        self.n_basis = n_basis
        self.coeff_proj = nn.Linear(dim, n_basis, bias=False)
        self.basis_proj = nn.Linear(dim, dim * n_basis, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        coeff = F.softmax(self.coeff_proj(x), dim=-1)  # (B, S, K)
        basis = self.basis_proj(x).reshape(B, S, self.n_basis, D)
        basis = torch.tanh(basis)
        mixed = (coeff.unsqueeze(-1) * basis).sum(dim=2)
        return self.out_proj(mixed)


class ImplicitFixedPointLayer(nn.Module):
    """Implicit fixed-point style channel transform with damped iterations."""

    def __init__(self, dim: int, hidden_mult: float = 2.0, n_steps: int = 4):
        super().__init__()
        hidden = max(dim, int(dim * hidden_mult))
        self.in_proj = nn.Linear(dim, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, dim, bias=False)
        self.residual_proj = nn.Linear(dim, dim, bias=False)
        self.n_steps = n_steps
        self.damping = nn.Parameter(torch.tensor(0.5))

    def _f(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        update = self.out_proj(F.silu(self.in_proj(h))) + self.residual_proj(x)
        return torch.tanh(update)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        alpha = torch.sigmoid(self.damping)
        for _ in range(self.n_steps):
            h = alpha * self._f(h, x) + (1.0 - alpha) * h
        return h


class ProductKeyMemory(nn.Module):
    """Product-key memory lookup."""

    def __init__(self, dim: int, n_keys: int = 64):
        super().__init__()
        self.half_dim = dim // 2
        self.keys_a = nn.Parameter(torch.randn(n_keys, self.half_dim) * 0.02)
        self.keys_b = nn.Parameter(torch.randn(n_keys, dim - self.half_dim) * 0.02)
        self.values = nn.Parameter(torch.randn(n_keys * n_keys, dim) * 0.02)
        self.n_keys = n_keys
        self.topk = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qa, qb = x[..., : self.half_dim], x[..., self.half_dim :]

        scores_a = torch.einsum("bsd,kd->bsk", qa, self.keys_a)
        scores_b = torch.einsum("bsd,kd->bsk", qb, self.keys_b)

        top_a = scores_a.topk(self.topk, dim=-1)
        top_b = scores_b.topk(self.topk, dim=-1)

        # Product indices
        idx_a = top_a.indices.unsqueeze(-1).expand(-1, -1, -1, self.topk)
        idx_b = top_b.indices.unsqueeze(-2).expand(-1, -1, self.topk, -1)
        product_idx = (idx_a * self.n_keys + idx_b).reshape(B, S, -1)

        # Weights
        w_a = F.softmax(top_a.values, dim=-1).unsqueeze(-1)
        w_b = F.softmax(top_b.values, dim=-1).unsqueeze(-2)
        weights = (w_a * w_b).reshape(B, S, -1)

        # Lookup
        vals = self.values[product_idx.reshape(-1)].reshape(B, S, -1, D)
        return (vals * weights.unsqueeze(-1)).sum(dim=2)


class IdentityMLP(nn.Module):
    """No channel mixing — identity."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ── Compute Routing Modules ───────────────────────────────────────────


class UniformRouting(nn.Module):
    """No routing — all tokens processed."""

    def forward(self, x: torch.Tensor, block: nn.Module) -> torch.Tensor:
        return block(x)


class MoDTopKRouting(nn.Module):
    """Mixture-of-Depths: only top-k tokens get full computation."""

    def __init__(self, dim: int, capacity_factor: float = 0.75):
        super().__init__()
        self.router = nn.Linear(dim, 1, bias=False)
        self.capacity_factor = capacity_factor

    def forward(self, x: torch.Tensor, block: nn.Module) -> torch.Tensor:
        B, S, D = x.shape
        k = max(1, int(S * self.capacity_factor))
        scores = self.router(x).squeeze(-1)
        topk_vals, topk_idx = scores.topk(k, dim=-1)
        weights = torch.sigmoid(topk_vals)

        # Record adaptive telemetry
        savings = 1.0 - (k / S)
        if hasattr(self, "adaptive_telemetry"):
            at = self.adaptive_telemetry
            at["savings_sum"] = at.get("savings_sum", 0.0) + savings
            at["count"] = at.get("count", 0) + 1
        else:
            self.adaptive_telemetry = {"savings_sum": savings, "count": 1}

        # Gather selected tokens
        selected = x.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, D))
        processed = block(selected) * weights.unsqueeze(-1)

        # Scatter back
        output = x.clone()
        output.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, D), processed)
        return output


class EarlyExitRouting(nn.Module):
    """Early exit: tokens can skip remaining layers."""

    def __init__(self, dim: int, threshold: float = 0.5):
        super().__init__()
        self.exit_gate = nn.Linear(dim, 1, bias=True)
        nn.init.constant_(self.exit_gate.bias, 2.0)  # bias toward continuing

    def forward(self, x: torch.Tensor, block: nn.Module) -> torch.Tensor:
        gate = torch.sigmoid(self.exit_gate(x))  # (B, S, 1)
        processed = block(x)
        return gate * processed + (1 - gate) * x


class LayerDropRouting(nn.Module):
    """Stochastic layer drop during training."""

    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor, block: nn.Module) -> torch.Tensor:
        if self.training and torch.rand(1).item() < self.drop_prob:
            return x
        return block(x)


class TokenMerging(nn.Module):
    """Merge similar tokens to reduce sequence length, then unmerge."""

    def __init__(self, dim: int, merge_ratio: float = 0.5):
        super().__init__()
        self.merge_ratio = merge_ratio

    def forward(self, x: torch.Tensor, block: nn.Module) -> torch.Tensor:
        B, S, D = x.shape
        n_keep = max(1, int(S * self.merge_ratio))

        # Simple similarity-based merging
        sim = torch.einsum(
            "bsd,btd->bst", F.normalize(x, dim=-1), F.normalize(x, dim=-1)
        )
        sim.diagonal(dim1=1, dim2=2).fill_(float("-inf"))

        # For each token, find most similar
        _, merge_target = sim.max(dim=-1)

        # Keep first n_keep tokens, merge rest into nearest
        kept = x[:, :n_keep]
        processed = block(kept)

        # Expand back (simple: repeat last token for dropped positions)
        output = torch.zeros_like(x)
        output[:, :n_keep] = processed
        output[:, n_keep:] = processed[:, -1:].expand(-1, S - n_keep, -1)
        return output


class CascadeRouting(nn.Module):
    """Progressive cascade: easy tokens exit early."""

    def __init__(self, dim: int):
        super().__init__()
        self.difficulty = nn.Linear(dim, 1, bias=True)
        nn.init.constant_(self.difficulty.bias, 0.0)

    def forward(self, x: torch.Tensor, block: nn.Module) -> torch.Tensor:
        diff = torch.sigmoid(self.difficulty(x))
        processed = block(x)
        return diff * processed + (1 - diff) * x


class SpeculativeRouting(nn.Module):
    """Speculative: run cheap path, gate with quality check.

    Implements adaptive routing with fallback to expensive path when
    quality threshold not met. Used for efficiency-accuracy tradeoffs.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.quality_gate = nn.Linear(dim, 1, bias=True)
        nn.init.constant_(self.quality_gate.bias, 1.0)

    def forward(self, x: torch.Tensor, block: nn.Module) -> torch.Tensor:
        # Run the full block
        processed = block(x)
        gate = torch.sigmoid(self.quality_gate(processed))  # Quality gating
        return gate * processed + (1 - gate) * x


class AdaptiveRecursionRouting(nn.Module):
    """MoR-style: variable recursion depth per token."""

    def __init__(self, dim: int, max_depth: int = 3):
        super().__init__()
        self.depth_router = nn.Linear(dim, max_depth, bias=False)
        self.max_depth = max_depth

    def forward(self, x: torch.Tensor, block: nn.Module) -> torch.Tensor:
        depth_weights = F.softmax(self.depth_router(x), dim=-1)  # (B, S, max_depth)

        # Record adaptive telemetry
        avg_depth = (
            (depth_weights * torch.arange(1, self.max_depth + 1, device=x.device))
            .sum(dim=-1)
            .mean()
            .item()
        )
        savings = 1.0 - (avg_depth / self.max_depth)
        if hasattr(self, "adaptive_telemetry"):
            at = self.adaptive_telemetry
            at["savings_sum"] = at.get("savings_sum", 0.0) + savings
            at["depth_sum"] = at.get("depth_sum", 0.0) + avg_depth
            at["count"] = at.get("count", 0) + 1
        else:
            self.adaptive_telemetry = {
                "savings_sum": savings,
                "depth_sum": avg_depth,
                "count": 1,
            }

        outputs = [x]
        current = x
        for d in range(self.max_depth):
            current = block(current)
            outputs.append(current)

        # Weighted sum across depths
        stacked = torch.stack(outputs[: self.max_depth], dim=-1)  # (B, S, D, max_depth)
        return (stacked * depth_weights.unsqueeze(2)).sum(dim=-1)


# ── Topology: Block Wiring ─────────────────────────────────────────────


class SequentialTopology(nn.Module):
    """Standard sequential stack."""

    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


class UNetTopology(nn.Module):
    """U-Net with skip connections."""

    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        n = len(blocks)
        self.encoder = blocks[: n // 2]
        self.decoder = blocks[n // 2 :]
        self.skip_projs = nn.ModuleList(
            [
                nn.Linear(
                    blocks[0].dim if hasattr(blocks[0], "dim") else 256,
                    blocks[0].dim if hasattr(blocks[0], "dim") else 256,
                )
                for _ in range(min(len(self.encoder), len(self.decoder)))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.encoder:
            x = x + block(x)
            skips.append(x)

        for i, block in enumerate(self.decoder):
            x = x + block(x)
            if i < len(skips):
                x = x + self.skip_projs[i](skips[-(i + 1)])
        return x


class DenseNetTopology(nn.Module):
    """Each layer receives input from all previous layers."""

    def __init__(self, blocks: nn.ModuleList, dim: int):
        super().__init__()
        self.blocks = blocks
        n = len(blocks)
        # Projection for each layer to handle concatenated inputs
        self.projs = nn.ModuleList([nn.Linear(dim * (i + 1), dim) for i in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for i, block in enumerate(self.blocks):
            concat = torch.cat(features, dim=-1)
            h = self.projs[i](concat)
            out = h + block(h)
            features.append(out)
        return features[-1]


class ParallelStreamsTopology(nn.Module):
    """Two parallel streams merged at intervals."""

    def __init__(self, blocks: nn.ModuleList, dim: int):
        super().__init__()
        n = len(blocks)
        half = n // 2
        self.stream_a = blocks[:half]
        self.stream_b = blocks[half:]
        self.merge_proj = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x, x
        for i in range(min(len(self.stream_a), len(self.stream_b))):
            a = a + self.stream_a[i](a)
            b = b + self.stream_b[i](b)
            # Merge every 2 steps
            if i % 2 == 1:
                merged = self.merge_proj(torch.cat([a, b], dim=-1))
                a, b = merged, merged
        return a


class HourglassTopology(nn.Module):
    """Progressive downsample then upsample."""

    def __init__(self, blocks: nn.ModuleList, dim: int):
        super().__init__()
        n = len(blocks)
        self.blocks = blocks
        self.down = nn.AvgPool1d(2)
        self.up = nn.Upsample(scale_factor=2)
        self.mid_idx = n // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # Encoder (full resolution)
        for block in self.blocks[: self.mid_idx]:
            x = x + block(x)

        # Downsample
        x_down = self.down(x.transpose(1, 2)).transpose(1, 2)

        # Middle blocks (half resolution)
        for block in self.blocks[self.mid_idx : self.mid_idx + 1]:
            x_down = x_down + block(x_down)

        # Upsample
        x_up = self.up(x_down.transpose(1, 2)).transpose(1, 2)[:, :S]

        # Decoder (full resolution)
        x = x + x_up
        for block in self.blocks[self.mid_idx + 1 :]:
            x = x + block(x)
        return x


class FeedbackTopology(nn.Module):
    """Sequential with feedback from last to first (2 iterations)."""

    def __init__(self, blocks: nn.ModuleList, dim: int):
        super().__init__()
        self.blocks = blocks
        self.feedback_proj = nn.Linear(dim, dim)
        self.n_iterations = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feedback = torch.zeros_like(x)
        for iteration in range(self.n_iterations):
            h = x + self.feedback_proj(feedback)
            for block in self.blocks:
                h = h + block(h)
            feedback = h
        return h


class FractalTopology(nn.Module):
    """Fractal: run blocks at multiple depths, combine."""

    def __init__(self, blocks: nn.ModuleList, dim: int):
        super().__init__()
        self.blocks = blocks
        self.combine = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.blocks) <= 1:
            return self.blocks[0](x) + x if self.blocks else x

        # Path 1: all blocks sequential
        h1 = x
        for block in self.blocks:
            h1 = h1 + block(h1)

        # Path 2: skip every other block
        h2 = x
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                h2 = h2 + block(h2)

        return self.combine(torch.cat([h1, h2], dim=-1))


class MixtureOfPathsTopology(nn.Module):
    """Two parallel paths through blocks, soft-mixed per token."""

    def __init__(self, blocks: nn.ModuleList, dim: int):
        super().__init__()
        n = len(blocks)
        half = max(1, n // 2)
        self.path_a = blocks[:half]
        self.path_b = blocks[half:]
        self.gate = nn.Linear(dim, 1, bias=True)
        nn.init.constant_(self.gate.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x
        for block in self.path_a:
            a = a + block(a)

        b = x
        for block in self.path_b:
            b = b + block(b)

        g = torch.sigmoid(self.gate(x))  # (B, S, 1)
        return g * a + (1 - g) * b


# ── The Block ──────────────────────────────────────────────────────────


class ExplorerBlock(nn.Module):
    """A single transformer block assembled from morphological choices."""

    def __init__(
        self,
        token_mixer: nn.Module,
        channel_mixer: nn.Module,
        norm1: nn.Module,
        norm2: nn.Module,
    ):
        super().__init__()
        self.token_mixer = token_mixer
        self.channel_mixer = channel_mixer
        self.norm1 = norm1
        self.norm2 = norm2
        self.dim = None  # set by builder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        h = self.norm1(x) if self.norm1 is not None else x
        h = self.token_mixer(h)
        x = x + h
        # Pre-norm MLP
        h = self.norm2(x) if self.norm2 is not None else x
        h = self.channel_mixer(h)
        return h  # residual added by topology


class RoutedBlock(nn.Module):
    """Block wrapped with compute routing."""

    def __init__(self, block: ExplorerBlock, routing: nn.Module):
        super().__init__()
        self.block = block
        self.routing = routing
        self.dim = None  # set by builder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.routing, UniformRouting):
            return self.block(x)
        return self.routing(x, self.block)


# ── The Full Model ─────────────────────────────────────────────────────


class ExplorerModel(nn.Module):
    """A complete model assembled from an ArchSpec."""

    def __init__(
        self,
        spec: ArchSpec,
        config: BuildConfig,
        tok_repr: nn.Module,
        pos_enc: Optional[nn.Module],
        topology: nn.Module,
        head_norm: nn.Module,
    ):
        super().__init__()
        self.spec = spec
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.tok_repr = tok_repr
        self.pos_enc = pos_enc
        self.topology = topology
        self.head_norm = head_norm
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = (
            self.tok_repr.encode(x)
            if hasattr(self.tok_repr, "encode")
            else self.tok_repr(x)
        )

        if self.pos_enc is not None:
            x = self.pos_enc(x)

        x = self.topology(x)

        x = self.tok_repr.decode(x) if hasattr(self.tok_repr, "decode") else x
        x = self.head_norm(x) if self.head_norm is not None else x
        return self.lm_head(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Builder ────────────────────────────────────────────────────────────


def _build_norm(choice: str, dim: int) -> Optional[nn.Module]:
    norms = {
        "rmsnorm_pre": lambda: RMSNorm(dim),
        "layernorm_pre": lambda: nn.LayerNorm(dim),
        "no_norm": lambda: None,
        "dynamic_norm": lambda: DynamicNorm(dim),
        "group_norm": lambda: GroupNormWrapper(min(8, dim), dim),
        "sigmoid_norm": lambda: SigmoidNorm(dim),
    }
    factory = norms.get(choice)
    if factory is None:
        raise ValueError(f"Unknown normalization: {choice}")
    return factory()


def _build_token_repr(choice: str, dim: int) -> nn.Module:
    reprs = {
        "dense_float": lambda: DenseRepresentation(dim),
        "binary_hash": lambda: BinaryHashRepresentation(dim),
        "sparse_topk": lambda: SparseTopKRepresentation(dim),
        "complex_valued": lambda: ComplexRepresentation(dim),
        "quaternion": lambda: ComplexRepresentation(
            dim
        ),  # same impl, different semantic
        "multi_resolution": lambda: MultiResolutionRepresentation(dim),
        "mixture_embedding": lambda: MixtureEmbeddingRepresentation(dim),
        "residual_quantized": lambda: ResidualQuantizedRepresentation(dim),
    }
    return reprs[choice]()


def _build_token_mixer(choice: str, cfg: BuildConfig) -> nn.Module:
    mixers = {
        "softmax_attention": lambda: SoftmaxAttention(
            cfg.dim, cfg.n_heads, cfg.n_kv_heads, cfg.max_seq_len
        ),
        "linear_attention": lambda: LinearAttention(cfg.dim, cfg.n_heads),
        "conv_only": lambda: ConvMixer(cfg.dim),
        "state_space": lambda: StateSpaceMixer(cfg.dim),
        "fourier_mixing": lambda: FourierMixer(cfg.dim),
        "graph_attention": lambda: GraphAttention(
            cfg.dim, cfg.n_heads, max_seq_len=cfg.max_seq_len
        ),
        "random_feature_attention": lambda: RandomFeatureAttention(
            cfg.dim, cfg.n_heads
        ),
        "differentiable_sort": lambda: DifferentiableSortMixer(cfg.dim),
        "compressed_attention": lambda: CompressedAttention(
            cfg.dim, cfg.n_heads, cfg.compression_factor, max_seq_len=cfg.max_seq_len
        ),
        "cross_attention_pool": lambda: CrossAttentionPool(cfg.dim, cfg.n_heads),
        "integral_kernel_mixing": lambda: IntegralKernelMixer(cfg.dim),
    }
    return mixers[choice]()


def _build_channel_mixer(choice: str, cfg: BuildConfig) -> nn.Module:
    mixers = {
        "swiglu_mlp": lambda: SwiGLUMLP(cfg.dim, cfg.mlp_ratio),
        "moe_topk": lambda: MoETopK(
            cfg.dim, cfg.moe_num_experts, cfg.moe_topk, cfg.mlp_ratio
        ),
        "kan_spline": lambda: KANSplineMLP(cfg.dim, cfg.mlp_ratio),
        "rwkv_channel": lambda: RWKVChannelMix(cfg.dim, cfg.mlp_ratio),
        "conv1d_glu": lambda: Conv1dGLU(cfg.dim),
        "polynomial_expansion": lambda: PolynomialExpansion(cfg.dim),
        "product_key_memory": lambda: ProductKeyMemory(cfg.dim),
        "basis_expansion_layer": lambda: BasisExpansionLayer(cfg.dim),
        "implicit_fixed_point": lambda: ImplicitFixedPointLayer(cfg.dim),
        "identity_skip": lambda: IdentityMLP(),
    }
    return mixers[choice]()


def _build_pos_enc(choice: str, cfg: BuildConfig) -> Optional[nn.Module]:
    encs = {
        "rope": lambda: RoPE(cfg.dim, cfg.max_seq_len),
        "alibi": lambda: ALiBi(cfg.dim, cfg.max_seq_len),
        "none": lambda: None,
        "learned_absolute": lambda: LearnedAbsolutePositional(cfg.dim, cfg.max_seq_len),
        "random_fourier": lambda: RandomFourierPositional(cfg.dim, cfg.max_seq_len),
        "convolutional": lambda: ConvPositional(cfg.dim),
    }
    return encs[choice]()


def _build_compute_routing(choice: str, dim: int) -> nn.Module:
    routes = {
        "uniform": lambda: UniformRouting(),
        "mod_topk": lambda: MoDTopKRouting(dim),
        "early_exit": lambda: EarlyExitRouting(dim),
        "adaptive_recursion": lambda: AdaptiveRecursionRouting(dim),
        "layerdrop": lambda: LayerDropRouting(),
        "token_merge": lambda: TokenMerging(dim),
        "cascade": lambda: CascadeRouting(dim),
        "speculative": lambda: SpeculativeRouting(dim),
    }
    return routes[choice]()


def _build_topology(choice: str, blocks: nn.ModuleList, dim: int) -> nn.Module:
    topos = {
        "sequential": lambda: SequentialTopology(blocks),
        "u_net": lambda: UNetTopology(blocks),
        "fractal": lambda: FractalTopology(blocks, dim),
        "dense_net": lambda: DenseNetTopology(blocks, dim),
        "parallel_streams": lambda: ParallelStreamsTopology(blocks, dim),
        "hourglass": lambda: HourglassTopology(blocks, dim),
        "mixture_of_paths": lambda: MixtureOfPathsTopology(blocks, dim),
        "feedback_loop": lambda: FeedbackTopology(blocks, dim),
    }
    return topos[choice]()


def build_model(spec: ArchSpec, config: Optional[BuildConfig] = None) -> ExplorerModel:
    """
    Build a complete model from an ArchSpec.

    Args:
        spec: The architecture specification
        config: Build configuration (dims, layers, etc.)

    Returns:
        An ExplorerModel ready for training
    """
    if config is None:
        config = BuildConfig()

    c = spec.choices

    # Build components
    tok_repr = _build_token_repr(c["token_representation"], config.dim)
    pos_enc = _build_pos_enc(c["positional_encoding"], config)

    # Build blocks with compute routing
    blocks = nn.ModuleList()
    for _ in range(config.n_layers):
        token_mixer = _build_token_mixer(c["token_mixing"], config)
        channel_mixer = _build_channel_mixer(c["channel_mixing"], config)
        norm1 = _build_norm(c["normalization"], config.dim)
        norm2 = _build_norm(c["normalization"], config.dim)
        block = ExplorerBlock(token_mixer, channel_mixer, norm1, norm2)
        block.dim = config.dim
        # Wrap with compute routing
        routing = _build_compute_routing(c["compute_routing"], config.dim)
        wrapped = RoutedBlock(block, routing)
        wrapped.dim = config.dim
        blocks.append(wrapped)

    # Build topology
    topology = _build_topology(c["topology"], blocks, config.dim)

    # Head norm
    head_norm = _build_norm(c["normalization"], config.dim)

    model = ExplorerModel(spec, config, tok_repr, pos_enc, topology, head_norm)
    return model
