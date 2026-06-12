"""In-context probe — wrap a lane in a winner-shaped stack and short-train.

The bog-standard ``LaneTestBlock`` grades intrinsic op behavior. This
probe goes one step further: it embeds the lane inside a 2-block
transformer-ish stack and runs a small Adam loop on a synthetic
position-mixing task. The "ratio" of (initial_loss / final_loss)
captures whether the lane is trainable inside a real architecture.

The training task is causal running-mean reconstruction:
``target[i] = mean(x[0:i+1])``. A lane that mixes positions can learn
this; one that doesn't will saturate at a high loss.

Decoupled from research/synthesis — pure torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .primitives import RMSNorm, causal_running_mean
from .standard_block import LaneTestBlock

logger = logging.getLogger(__name__)


class WinnerLikeBlock(nn.Module):
    """Lane block + FFN block + output norm — a tiny transformer-ish stack."""

    def __init__(self, lane: nn.Module, dim: int, ffn_mult: int = 2) -> None:
        super().__init__()
        self.lane_block = LaneTestBlock(lane, dim)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )
        self.out_norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lane_block(x)
        x = x + self.ffn(self.ffn_norm(x))
        return self.out_norm(x)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    initial_loss: float
    final_loss: float
    loss_ratio_initial_over_final: float
    n_steps: int
    trained_successfully: bool


def short_training_probe(
    lane: nn.Module,
    *,
    dim: int = 32,
    seq_len: int = 32,
    n_steps: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    seed: int = 0,
    target_fn: "Callable[[torch.Tensor], torch.Tensor] | None" = None,
) -> ProbeResult:
    """Train a WinnerLikeBlock(lane) for n_steps on a position-mixing task.

    Default task is causal running-mean. Pass ``target_fn`` to grade on a
    different task (used by the sprint-5 multi-task probe suite).
    """
    if target_fn is None:
        target_fn = causal_running_mean
    torch.manual_seed(seed)
    model = WinnerLikeBlock(lane, dim).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses: list[float] = []
    try:
        for _ in range(n_steps):
            x = torch.randn(batch_size, seq_len, dim, device=device)
            with torch.no_grad():
                target = target_fn(x)
            y = model(x)
            loss = (y - target).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        trained = True
    except Exception:  # noqa: BLE001 - one broken lane must not abort grading
        logger.warning("short_training_probe failed; scoring 0.0", exc_info=True)
        trained = False
    if not losses:
        return ProbeResult(
            initial_loss=float("nan"),
            final_loss=float("nan"),
            loss_ratio_initial_over_final=0.0,
            n_steps=0,
            trained_successfully=False,
        )

    window = max(1, n_steps // 10)
    initial = sum(losses[:window]) / len(losses[:window])
    final = sum(losses[-window:]) / len(losses[-window:])
    ratio = initial / max(final, 1e-12)
    return ProbeResult(
        initial_loss=initial,
        final_loss=final,
        loss_ratio_initial_over_final=ratio,
        n_steps=len(losses),
        trained_successfully=trained,
    )
