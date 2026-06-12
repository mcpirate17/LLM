"""Synthetic probe tasks for in-context grading.

A "task" is a callable that takes a batch input ``[B, L, D]`` and returns
a target tensor of the same shape. The in-context probe trains a
``WinnerLikeBlock`` on the task and reports loss reduction.

Tasks span the spectrum from trivially-mixable (running_mean) to harder
(copy from random previous position, causal max). A lane that can learn
multiple tasks is more interesting than one that only learns the trivial
ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .primitives import causal_running_mean

TaskFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True, slots=True)
class ProbeTask:
    name: str
    target_fn: TaskFn
    difficulty: str  # "easy" / "medium" / "hard"


def _causal_max(x: torch.Tensor) -> torch.Tensor:
    return x.cummax(dim=1).values


def _shifted_copy(x: torch.Tensor) -> torch.Tensor:
    """Target at position i = input at position i-1, zero at position 0.

    Requires the lane to actually shift information across positions.
    """
    out = torch.zeros_like(x)
    out[:, 1:] = x[:, :-1]
    return out


def _periodic_average(x: torch.Tensor) -> torch.Tensor:
    """Target at position i = mean of x[i-3], x[i-2], x[i-1], x[i] (boxcar).

    A short causal convolution. Mixes positions in a narrow window.
    """
    pad = torch.nn.functional.pad(x, (0, 0, 3, 0))
    return (pad[:, :-3] + pad[:, 1:-2] + pad[:, 2:-1] + pad[:, 3:]) / 4.0


def _copy_from_uniform_past(x: torch.Tensor) -> torch.Tensor:
    """Target at position i = x[uniform-random j <= i].

    Genuinely requires the lane to copy across long distances. Random index
    per (batch, position) so a fixed-offset shift won't solve it. Tests
    associative-retrieval ability.
    """
    batch_size, seq_len, _ = x.shape
    # Pre-generate one random source index per (batch, position), j <= i.
    positions = (
        torch.arange(seq_len, device=x.device).view(1, -1).expand(batch_size, -1)
    )
    # u ~ U(0, 1) ; source = floor(u * (i + 1))
    u = torch.rand(batch_size, seq_len, device=x.device)
    source = (u * (positions.float() + 1)).long().clamp(0, seq_len - 1)
    # Gather along sequence dim
    source_expanded = source.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return torch.gather(x, 1, source_expanded)


def _causal_induction(x: torch.Tensor) -> torch.Tensor:
    """Target at position i = x[k] where k is the largest j < i with x[j,0] > 0.

    A simple induction pattern: find a "key" earlier in the sequence (first
    feature positive) and emit the value at that position. Falls back to the
    current position if no such key exists. Tests pattern-conditioned lookup.
    Vectorized: cummax over masked position indices, shifted one step right
    (strictly-previous key), then a gather.
    """
    batch_size, seq_len, dim = x.shape
    positions = torch.arange(seq_len, device=x.device)
    keyed = torch.where(
        x[..., 0] > 0,
        positions.unsqueeze(0).expand(batch_size, -1),
        torch.full_like(positions, -1).unsqueeze(0).expand(batch_size, -1),
    )
    last_key_at_or_before = keyed.cummax(dim=1).values
    last_key_before = torch.cat(
        [
            torch.full((batch_size, 1), -1, dtype=torch.long, device=x.device),
            last_key_at_or_before[:, :-1],
        ],
        dim=1,
    )
    source = torch.where(last_key_before >= 0, last_key_before, positions.unsqueeze(0))
    return torch.gather(x, 1, source.unsqueeze(-1).expand(-1, -1, dim))


def _running_parity(x: torch.Tensor) -> torch.Tensor:
    """Target at position i depends on the cumulative parity of x[j, 0] > 0 for j <= i.
    Outputs x or -x depending on if the running count of positive elements is odd/even.
    Tests hard boolean state tracking (typical SSM advantage over attention).
    """
    signs = (x[..., 0] > 0).float()
    parity = (signs.cumsum(dim=1) % 2) * 2 - 1.0  # +1 or -1
    return x * parity.unsqueeze(-1)


DEFAULT_PROBE_TASKS: tuple[ProbeTask, ...] = (
    ProbeTask(name="running_mean", target_fn=causal_running_mean, difficulty="easy"),
    ProbeTask(name="periodic_average", target_fn=_periodic_average, difficulty="easy"),
    ProbeTask(name="causal_max", target_fn=_causal_max, difficulty="medium"),
    ProbeTask(name="shifted_copy", target_fn=_shifted_copy, difficulty="hard"),
    ProbeTask(
        name="copy_from_uniform_past",
        target_fn=_copy_from_uniform_past,
        difficulty="hard",
    ),
    ProbeTask(name="causal_induction", target_fn=_causal_induction, difficulty="hard"),
    ProbeTask(name="running_parity", target_fn=_running_parity, difficulty="hard"),
)
