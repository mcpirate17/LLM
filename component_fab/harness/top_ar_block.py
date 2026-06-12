"""Top-AR scaffold from fp `7fb0412ec57a1213` (0.9046 ar_curriculum_auc_pair_final).

Reproduces the block structure that wins AR-curriculum at the 10-30M
param-budget regime. The block is dual-mixer with explicit per-mixer
linear_proj projections and a 3-way residual to the original input:

    input -> LN -> RMS -> mixer_a -> linear_proj
                                     -> +input (mid)
    mid -> RMS -> conv1d_seq -> swiglu -> RMS -> mixer_b -> linear_proj
                                                              -> +mid
                  -> +input (3-way residual)
                  -> final RMS

The ``mixer_a_factory`` / ``mixer_b_factory`` slots are pluggable so the
same scaffold can host either the original (``TropicalAttention`` +
``LocalWindowAttention``) or a substituted lane (e.g. our 2-lane
``GatedParallelBlock(TropicalAttention, SparsemaxAttention)`` in the
mixer_a slot).
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn

from .primitives import (
    CausalDepthwiseConv1d as CausalConv1dSeq,
    RMSNorm,
    SwiGLU,
)

__all__ = [
    "CausalConv1dSeq",
    "LocalWindowAttention",
    "RMSNorm",
    "SwiGLU",
    "TopArchBlock",
]

LaneFactory = Callable[[int], nn.Module]


class LocalWindowAttention(nn.Module):
    """Parameter-free local-window self-attention.

    Mirrors ``_op_local_window_attn`` (research/synthesis/compiler_ops_attention.py:373):
    Q=K=V=x, causal + sliding-window mask, softmax. No learned projections.
    """

    def __init__(self, dim: int, window_size: int = 16) -> None:
        super().__init__()
        self.dim = int(dim)
        self.window_size = int(window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        W = min(self.window_size, S)
        x_work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        scores = torch.bmm(x_work, x_work.transpose(-2, -1)) / math.sqrt(D)
        row_idx = torch.arange(S, device=x.device).unsqueeze(1)
        col_idx = torch.arange(S, device=x.device).unsqueeze(0)
        mask = (col_idx > row_idx) | (row_idx - col_idx >= W)
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.bmm(attn, x_work).to(dtype=x.dtype)


class TopArchBlock(nn.Module):
    """Dual-mixer block with 3-way residual; reproduces fp `7fb0412ec57a1213`.

    Forward (faithful to the graph_json — node 13 emits ``input + mid + h2``):

        h1 = proj_a(mixer_a(rms1(ln(x))))
        mid = x + h1                                       # node 5
        h2 = proj_b(mixer_b(rms3(swiglu(conv1d(rms2(mid))))))
        return rms_final(x + mid + h2)                     # nodes 13, 14
    """

    def __init__(
        self,
        dim: int,
        mixer_a_factory: LaneFactory,
        mixer_b_factory: LaneFactory,
        *,
        mlp_ratio: float = 4.0,
        conv_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.ln = nn.LayerNorm(dim)
        self.rms1 = RMSNorm(dim)
        self.mixer_a = mixer_a_factory(dim)
        self.proj_a = nn.Linear(dim, dim, bias=False)
        self.rms2 = RMSNorm(dim)
        self.conv1d = CausalConv1dSeq(dim, kernel_size=conv_kernel)
        self.swiglu = SwiGLU(dim, mlp_ratio=mlp_ratio)
        self.rms3 = RMSNorm(dim)
        self.mixer_b = mixer_b_factory(dim)
        self.proj_b = nn.Linear(dim, dim, bias=False)
        self.rms_final = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.proj_a(self.mixer_a(self.rms1(self.ln(x))))
        mid = x + h1
        h2 = self.proj_b(
            self.mixer_b(self.rms3(self.swiglu(self.conv1d(self.rms2(mid)))))
        )
        return self.rms_final(x + mid + h2)
