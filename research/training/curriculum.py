"""
Curriculum Strategy Synthesis

Generate novel data presentation strategies:
- Growing sequence length
- Difficulty-based sorting
- Spaced repetition
- Masking pattern generation
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from ._curriculum_native import load_curriculum_native

_FIXED = 0
_GROWING = 1
_OSCILLATING = 2
_OSCILLATING_SCALE = 8.0 * math.pi


@dataclass(slots=True)
class CurriculumStrategy:
    """A synthesized curriculum/data presentation strategy."""

    name: str
    description: str = ""
    seed: int = 0

    # Parameters
    warmup_steps: int = 100
    seq_len_schedule: str = "fixed"  # "fixed", "growing", "oscillating"
    initial_seq_len: int = 32
    max_seq_len: int = 512
    _cached_seq_len_schedule: str = field(init=False, repr=False, default="")
    _schedule_code: int = field(init=False, repr=False, default=_FIXED)
    _seq_span: int = field(init=False, repr=False, default=0)
    _warmup_den: int = field(init=False, repr=False, default=1)
    _cached_initial_seq_len: int = field(init=False, repr=False, default=32)
    _cached_max_seq_len: int = field(init=False, repr=False, default=512)
    _cached_warmup_steps: int = field(init=False, repr=False, default=100)
    _osc_total_steps: int = field(init=False, repr=False, default=-1)
    _osc_scale: float = field(init=False, repr=False, default=_OSCILLATING_SCALE)

    def __post_init__(self) -> None:
        self._refresh_schedule_cache()

    def _refresh_schedule_cache(self) -> None:
        schedule = self.seq_len_schedule
        if schedule == "growing":
            code = _GROWING
        elif schedule == "oscillating":
            code = _OSCILLATING
        else:
            code = _FIXED
        self._cached_seq_len_schedule = schedule
        self._schedule_code = code
        self._seq_span = int(self.max_seq_len) - int(self.initial_seq_len)
        self._warmup_den = max(int(self.warmup_steps), 1)
        self._cached_initial_seq_len = int(self.initial_seq_len)
        self._cached_max_seq_len = int(self.max_seq_len)
        self._cached_warmup_steps = int(self.warmup_steps)
        self._osc_total_steps = -1

    def get_seq_len(self, step: int, total_steps: int) -> int:
        """Get sequence length for current step."""
        if (
            self._cached_seq_len_schedule != self.seq_len_schedule
            or self._cached_initial_seq_len != self.initial_seq_len
            or self._cached_max_seq_len != self.max_seq_len
            or self._cached_warmup_steps != self.warmup_steps
        ):
            self._refresh_schedule_cache()

        code = self._schedule_code
        if code == _FIXED:
            return self.max_seq_len
        if code == _GROWING:
            progress = step / self._warmup_den
            if progress >= 1.0:
                return self.max_seq_len
            return int(self.initial_seq_len + progress * self._seq_span)

        total = int(total_steps)
        if total != self._osc_total_steps:
            self._osc_total_steps = total
            self._osc_scale = (
                _OSCILLATING_SCALE / total if total > 1 else _OSCILLATING_SCALE
            )
        frac = 0.5 + 0.5 * math.sin(step * self._osc_scale)
        return int(self.initial_seq_len + frac * self._seq_span)

    def seq_len_tensor(
        self,
        total_steps: int,
        start: int = 0,
        stop: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute a contiguous int64 schedule in native code for hot loops."""
        if (
            self._cached_seq_len_schedule != self.seq_len_schedule
            or self._cached_initial_seq_len != self.initial_seq_len
            or self._cached_max_seq_len != self.max_seq_len
            or self._cached_warmup_steps != self.warmup_steps
        ):
            self._refresh_schedule_cache()
        end = int(total_steps) if stop is None else int(stop)
        return load_curriculum_native().schedule_seq_lens(
            int(self._schedule_code),
            int(self.initial_seq_len),
            int(self.max_seq_len),
            int(self.warmup_steps),
            int(total_steps),
            int(start),
            end,
        )

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "seq_len_schedule": self.seq_len_schedule,
            "initial_seq_len": self.initial_seq_len,
            "max_seq_len": self.max_seq_len,
            "seed": self.seed,
        }


# ── Synthesis ─────────────────────────────────────────────────────────


def synthesize_curriculum(
    max_seq_len: int = 512,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> CurriculumStrategy:
    """Generate a random curriculum strategy."""
    rng = rng if rng is not None else random.Random(seed)

    seq_schedule = rng.choice(["fixed", "growing", "growing", "oscillating"])
    name = f"curriculum_{seq_schedule}"

    return CurriculumStrategy(
        name=name,
        description=f"Seq schedule: {seq_schedule}",
        seed=seed or 0,
        warmup_steps=rng.choice([50, 100, 200]),
        seq_len_schedule=seq_schedule,
        initial_seq_len=rng.choice([16, 32, 64]),
        max_seq_len=max_seq_len,
    )
