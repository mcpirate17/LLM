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
        """Get sequence length for current step."""
        if self.seq_len_schedule == "fixed":
            return self.max_seq_len
        elif self.seq_len_schedule == "growing":
            progress = min(1.0, step / max(self.warmup_steps, 1))
            return int(
                self.initial_seq_len
                + progress * (self.max_seq_len - self.initial_seq_len)
            )
        elif self.seq_len_schedule == "oscillating":
            cycle = math.sin(2 * math.pi * step / max(total_steps, 1) * 4)
            frac = 0.5 + 0.5 * cycle
            return int(
                self.initial_seq_len + frac * (self.max_seq_len - self.initial_seq_len)
            )
        return self.max_seq_len

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
