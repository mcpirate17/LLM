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
from typing import Dict, List, Optional


@dataclass(slots=True)
class CurriculumStrategy:
    """A synthesized curriculum/data presentation strategy."""

    name: str
    components: List[str] = field(default_factory=list)
    description: str = ""
    seed: int = 0

    # Parameters
    warmup_steps: int = 100
    seq_len_schedule: str = "fixed"  # "fixed", "growing", "oscillating"
    initial_seq_len: int = 32
    max_seq_len: int = 512
    masking_pattern: str = "causal"  # "causal", "prefix", "random_span", "checkerboard"

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
            "components": self.components,
            "seq_len_schedule": self.seq_len_schedule,
            "initial_seq_len": self.initial_seq_len,
            "max_seq_len": self.max_seq_len,
            "masking_pattern": self.masking_pattern,
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
    mask_pattern = rng.choice(
        ["causal", "causal", "prefix", "random_span", "checkerboard"]
    )
    components = [seq_schedule, mask_pattern]

    name = f"curriculum_{'_'.join(components[:3])}"

    return CurriculumStrategy(
        name=name,
        components=components,
        description=f"Seq schedule: {seq_schedule}, Mask: {mask_pattern}",
        seed=seed or 0,
        warmup_steps=rng.choice([50, 100, 200]),
        seq_len_schedule=seq_schedule,
        initial_seq_len=rng.choice([16, 32, 64]),
        max_seq_len=max_seq_len,
        masking_pattern=mask_pattern,
    )
