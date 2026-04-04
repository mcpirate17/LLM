from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ._optimizer_factory import build_optimizer
from ._optimizer_muon import MuonOptimizer
from .sparse_training import RigLOptimizer

OPTIMIZER_RECIPES = [
    ("adamw_standard", ["adamw"], "Standard AdamW"),
    ("adamw_high_momentum", ["adamw", "high_momentum"], "AdamW with high momentum"),
    ("adamw_low_momentum", ["adamw", "low_momentum"], "AdamW with low momentum"),
    ("muon", ["muon"], "Momentum optimizer with orthogonalized 2D updates"),
    ("rigl_sparse", ["rigl_sparse"], "RigL dynamic sparse training"),
]


@dataclass(slots=True)
class SynthesizedOptimizer:
    """Compact optimizer recipe used by training-program synthesis."""

    name: str
    components: List[str] = field(default_factory=list)
    lr: float = 3e-4
    weight_decay: float = 0.01
    description: str = ""
    seed: int = 0

    def create(self, params, **kwargs) -> torch.optim.Optimizer:
        lr = float(kwargs.get("lr", self.lr))
        weight_decay = float(kwargs.get("weight_decay", self.weight_decay))
        components = set(self.components)

        if "rigl_sparse" in components:
            return RigLOptimizer(params, lr=lr, weight_decay=weight_decay)

        if "muon" in components:
            return MuonOptimizer(params, lr=lr, weight_decay=weight_decay)

        betas = (0.9, 0.95)
        if "high_momentum" in components:
            betas = (0.95, 0.99)
        elif "low_momentum" in components:
            betas = (0.8, 0.95)
        return build_optimizer(
            params,
            optimizer_type="adamw",
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "components": self.components,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "description": self.description,
            "seed": self.seed,
        }


def synthesize_optimizer(seed: Optional[int] = None) -> SynthesizedOptimizer:
    """Generate a supported optimizer recipe."""
    rng = random.Random(seed)
    name, components, description = rng.choice(OPTIMIZER_RECIPES)
    lr = 10 ** rng.uniform(-4.5, -3.0)
    weight_decay = 10 ** rng.uniform(-3, -1)
    return SynthesizedOptimizer(
        name=name,
        components=list(components),
        lr=lr,
        weight_decay=weight_decay,
        description=description,
        seed=seed or 0,
    )
