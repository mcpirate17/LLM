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
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ._native import load_training_native


_FIXED = 0
_GROWING = 1
_OSCILLATING = 2
_OSCILLATING_SCALE = 8.0 * math.pi
_SCHEDULE_CODES = {"fixed": _FIXED, "growing": _GROWING, "oscillating": _OSCILLATING}


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

    def get_seq_len(self, step: int, total_steps: int) -> int:
        """Get sequence length for current step.

        Python fallback for duck-typed callers; hot loops use the native
        ``seq_len_tensor`` schedule instead.
        """
        code = _SCHEDULE_CODES.get(self.seq_len_schedule, _FIXED)
        if code == _FIXED:
            return self.max_seq_len
        span = int(self.max_seq_len) - int(self.initial_seq_len)
        if code == _GROWING:
            progress = step / max(int(self.warmup_steps), 1)
            if progress >= 1.0:
                return self.max_seq_len
            return int(self.initial_seq_len + progress * span)

        total = int(total_steps)
        scale = _OSCILLATING_SCALE / total if total > 1 else _OSCILLATING_SCALE
        frac = 0.5 + 0.5 * math.sin(step * scale)
        return int(self.initial_seq_len + frac * span)

    def seq_len_tensor(
        self,
        total_steps: int,
        start: int = 0,
        stop: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute a contiguous int64 schedule in native code for hot loops."""
        end = int(total_steps) if stop is None else int(stop)
        return load_training_native().schedule_seq_lens(
            _SCHEDULE_CODES.get(self.seq_len_schedule, _FIXED),
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
