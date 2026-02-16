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

import torch


@dataclass
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
    difficulty_sort: bool = False
    spaced_repetition: bool = False

    def get_seq_len(self, step: int, total_steps: int) -> int:
        """Get sequence length for current step."""
        if self.seq_len_schedule == "fixed":
            return self.max_seq_len
        elif self.seq_len_schedule == "growing":
            progress = min(1.0, step / max(self.warmup_steps, 1))
            return int(self.initial_seq_len + progress * (self.max_seq_len - self.initial_seq_len))
        elif self.seq_len_schedule == "oscillating":
            cycle = math.sin(2 * math.pi * step / max(total_steps, 1) * 4)
            frac = 0.5 + 0.5 * cycle
            return int(self.initial_seq_len + frac * (self.max_seq_len - self.initial_seq_len))
        return self.max_seq_len

    def get_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get attention mask for current strategy."""
        if self.masking_pattern == "causal":
            return torch.tril(torch.ones(seq_len, seq_len, device=device))
        elif self.masking_pattern == "prefix":
            # First 25% is bidirectional, rest is causal
            prefix_len = max(1, seq_len // 4)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            mask[:, :prefix_len] = 1.0
            return mask
        elif self.masking_pattern == "random_span":
            # Random contiguous spans are masked
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            # Randomly make some spans bidirectional
            n_spans = max(1, seq_len // 16)
            for _ in range(n_spans):
                start = random.randint(0, seq_len - 4)
                end = min(start + random.randint(2, 8), seq_len)
                mask[start:end, start:end] = 1.0
            return mask
        elif self.masking_pattern == "checkerboard":
            # Alternating causal/bidirectional at block level
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            block_size = max(4, seq_len // 8)
            for i in range(0, seq_len, block_size * 2):
                end = min(i + block_size, seq_len)
                mask[i:end, i:end] = 1.0
            return mask
        return torch.tril(torch.ones(seq_len, seq_len, device=device))

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "components": self.components,
            "seq_len_schedule": self.seq_len_schedule,
            "initial_seq_len": self.initial_seq_len,
            "max_seq_len": self.max_seq_len,
            "masking_pattern": self.masking_pattern,
            "difficulty_sort": self.difficulty_sort,
            "spaced_repetition": self.spaced_repetition,
            "seed": self.seed,
        }


# ── Synthesis ─────────────────────────────────────────────────────────

def synthesize_curriculum(
    max_seq_len: int = 512,
    seed: Optional[int] = None,
) -> CurriculumStrategy:
    """Generate a random curriculum strategy."""
    rng = random.Random(seed)

    seq_schedule = rng.choice(["fixed", "growing", "growing", "oscillating"])
    mask_pattern = rng.choice(["causal", "causal", "prefix", "random_span", "checkerboard"])
    diff_sort = rng.random() < 0.3
    spaced_rep = rng.random() < 0.2

    components = [seq_schedule, mask_pattern]
    if diff_sort:
        components.append("difficulty_sort")
    if spaced_rep:
        components.append("spaced_repetition")

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
        difficulty_sort=diff_sort,
        spaced_repetition=spaced_rep,
    )
