"""
Training Program

A complete training specification combining architecture,
loss function, optimizer, and curriculum — all potentially synthesized.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .loss_synthesis import SynthesizedLoss, synthesize_loss
from .optimizer_synthesis import SynthesizedOptimizer, synthesize_optimizer
from .curriculum import CurriculumStrategy, synthesize_curriculum


@dataclass
class TrainingProgram:
    """Complete training specification."""
    name: str
    loss: SynthesizedLoss
    optimizer: SynthesizedOptimizer
    curriculum: CurriculumStrategy
    # Training hyperparameters
    n_steps: int = 500
    batch_size: int = 4
    max_grad_norm: float = 1.0
    # Initialization
    init_scheme: str = "default"  # "default", "small", "orthogonal", "spectral"
    init_scale: float = 1.0
    seed: int = 0

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "loss": self.loss.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "curriculum": self.curriculum.to_dict(),
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "max_grad_norm": self.max_grad_norm,
            "init_scheme": self.init_scheme,
            "init_scale": self.init_scale,
            "seed": self.seed,
        }

    def describe(self) -> str:
        lines = [
            f"Training Program: {self.name}",
            f"  Loss: {self.loss.name} ({len(self.loss.components)} components)",
            f"  Optimizer: {self.optimizer.name} (lr={self.optimizer.lr:.2e})",
            f"  Curriculum: {self.curriculum.seq_len_schedule} seq, {self.curriculum.masking_pattern} mask",
            f"  Init: {self.init_scheme} (scale={self.init_scale:.2f})",
            f"  Steps: {self.n_steps}, Batch: {self.batch_size}",
        ]
        return "\n".join(lines)


# ── Init Schemes ──────────────────────────────────────────────────────

INIT_SCHEMES = [
    "default",      # PyTorch defaults
    "small",        # Small initialization (scale 0.02)
    "orthogonal",   # Orthogonal initialization
    "spectral",     # Spectral normalization of init
]


def synthesize_training_program(
    n_steps: int = 500,
    max_seq_len: int = 512,
    seed: Optional[int] = None,
) -> TrainingProgram:
    """Generate a complete random training program."""
    rng = random.Random(seed)

    loss = synthesize_loss(seed=rng.randint(0, 2**32))
    optimizer = synthesize_optimizer(seed=rng.randint(0, 2**32))
    curriculum = synthesize_curriculum(max_seq_len=max_seq_len, seed=rng.randint(0, 2**32))
    init_scheme = rng.choice(INIT_SCHEMES)
    init_scale = rng.choice([0.02, 0.05, 0.1, 1.0])

    name = f"tp_{loss.name[:10]}_{optimizer.name[:10]}_{curriculum.name[:10]}"

    return TrainingProgram(
        name=name,
        loss=loss,
        optimizer=optimizer,
        curriculum=curriculum,
        n_steps=n_steps,
        batch_size=rng.choice([2, 4, 8]),
        max_grad_norm=rng.choice([0.5, 1.0, 2.0]),
        init_scheme=init_scheme,
        init_scale=init_scale,
        seed=seed or 0,
    )
