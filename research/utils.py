"""
Utility functions and shared modules for neural architecture research.
"""

from __future__ import annotations
import torch
import torch.nn as nn

def straight_through_estimator(original: torch.Tensor, modified: torch.Tensor) -> torch.Tensor:
    """
    Bypasses non-differentiable operations entirely in the backward pass.
    Forward: returns modified
    Backward: gradient flows through original
    """
    return original + (modified - original).detach()

# ── Normalization Modules ──────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class DynamicNorm(nn.Module):
    """Per-token learned normalization scale."""
    def __init__(self, dim: int):
        super().__init__()
        self.scale_proj = nn.Linear(dim, dim)
        self.rms = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.sigmoid(self.scale_proj(x.detach()))
        return self.rms(x) * scale


class GroupNormWrapper(nn.Module):
    """GroupNorm wrapped for (B, S, D) format."""
    def __init__(self, num_groups: int, dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class SigmoidNorm(nn.Module):
    """Sigmoid-gated normalization."""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) * torch.sigmoid(self.gate(x))


# ── Positional Encoding Modules ───────────────────────────────────────

class RoPE(nn.Module):
    """Rotary Position Embeddings."""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.shape[1]
        cos = self.cos_cached[:S].unsqueeze(0)
        sin = self.sin_cached[:S].unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class ALiBi(nn.Module):
    """ALiBi-inspired position encoding (simplified: adds distance-weighted bias)."""
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.proj = nn.Linear(1, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        positions = torch.arange(S, device=x.device, dtype=x.dtype).unsqueeze(1) / S
        pe = self.proj(positions).unsqueeze(0)  # (1, S, D)
        return x + pe


class ConvPositional(nn.Module):
    """Implicit position from causal convolution."""
    def __init__(self, dim: int, kernel_size: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size - 1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)


class LearnedAbsolutePositional(nn.Module):
    """Learned absolute position embeddings."""
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.embed = nn.Embedding(max_seq_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        positions = torch.arange(S, device=x.device)
        return x + self.embed(positions).unsqueeze(0)


class RandomFourierPositional(nn.Module):
    """Random Fourier features for position encoding."""
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(1, dim // 2) * 0.1, requires_grad=False)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        positions = torch.arange(S, device=x.device, dtype=x.dtype).unsqueeze(1)
        features = positions * self.proj
        pe = torch.cat([torch.sin(features), torch.cos(features)], dim=-1)
        return x + self.linear(pe).unsqueeze(0)
