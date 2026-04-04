"""
Loss Function Synthesis

Generate loss functions from primitives instead of using only cross-entropy.
Each synthesized loss takes (logits, targets) and returns a scalar.

Examples of what this could produce:
- Rank-weighted cross-entropy (weight by token rarity)
- Tropical cross-entropy (min-plus on log probabilities)
- Spectral loss (penalize in frequency domain)
- Contrastive variants (push apart non-target logits)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from ._loss_components import (
    LOG_PROB_COMPONENTS,
    PROB_COMPONENTS,
    compute_component_fast,
    compute_spectral_component,
)


@dataclass(slots=True)
class LossComponent:
    """A single component of a synthesized loss."""

    name: str
    weight: float = 1.0
    description: str = ""


@dataclass(slots=True)
class SynthesizedLoss:
    """A synthesized loss function."""

    name: str
    components: List[LossComponent] = field(default_factory=list)
    description: str = ""
    seed: int = 0
    _component_names: tuple[str, ...] = field(init=False, repr=False)
    _component_weights: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._component_names = tuple(component.name for component in self.components)
        self._component_weights = tuple(
            float(component.weight) for component in self.components
        )

    def compute(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the synthesized loss.
        Optimized to share intermediate results like log_softmax across components.
        """
        if logits.dim() == 3:
            _, _, vocab_size = logits.shape
            flat_logits = logits.reshape(-1, vocab_size)
            spectral_logits: Optional[torch.Tensor] = logits
        elif logits.dim() == 2:
            flat_logits = logits
            spectral_logits = None
        else:
            raise ValueError(
                f"Expected logits with 2 or 3 dims, got shape {logits.shape}"
            )

        flat_targets = targets.reshape(-1)

        # Pre-compute shared results
        log_probs = None
        probs = None

        total = flat_logits.new_zeros(())
        for comp_name, comp_weight in zip(
            self._component_names, self._component_weights
        ):
            # Handle components that need original 3D shape
            if comp_name == "spectral_loss":
                if spectral_logits is not None:
                    total = total + comp_weight * compute_spectral_component(
                        spectral_logits
                    )
                continue

            # Lazy compute log_probs/probs if needed by this component
            if comp_name in LOG_PROB_COMPONENTS:
                if log_probs is None:
                    log_probs = torch.nn.functional.log_softmax(flat_logits, dim=-1)
            if comp_name in PROB_COMPONENTS:
                if probs is None:
                    probs = torch.nn.functional.softmax(flat_logits, dim=-1)

            total = total + comp_weight * compute_component_fast(
                comp_name, flat_logits, flat_targets, log_probs, probs
            )
        return total

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "components": [
                {"name": c.name, "weight": c.weight, "description": c.description}
                for c in self.components
            ],
            "description": self.description,
            "seed": self.seed,
        }


# ── Available Components ──────────────────────────────────────────────

LOSS_COMPONENTS = [
    LossComponent("cross_entropy", 1.0, "Standard cross-entropy"),
    LossComponent("label_smoothed_ce", 1.0, "Label-smoothed cross-entropy"),
    LossComponent("rank_weighted_ce", 1.0, "Rank-weighted cross-entropy"),
    LossComponent("tropical_ce", 0.5, "Tropical (min-plus) cross-entropy"),
    LossComponent("spectral_loss", 0.1, "Spectral regularization"),
    LossComponent("contrastive_push", 0.3, "Contrastive margin loss"),
    LossComponent("entropy_reg", 0.1, "Entropy regularization"),
    LossComponent("gradient_penalty", 0.01, "Logit magnitude penalty"),
    LossComponent("kl_uniform", 0.05, "KL from uniform"),
]


def synthesize_loss(seed: Optional[int] = None) -> SynthesizedLoss:
    """Generate a random loss function from components."""
    rng = random.Random(seed)

    # Always include some form of CE
    ce_variants = [
        "cross_entropy",
        "label_smoothed_ce",
        "rank_weighted_ce",
        "tropical_ce",
    ]
    primary = rng.choice(ce_variants)

    components = [LossComponent(primary, 1.0)]

    # Add 0-3 auxiliary components
    n_aux = rng.randint(0, 3)
    aux_pool = [
        c for c in LOSS_COMPONENTS if c.name != primary and c.name not in ce_variants
    ]
    for comp in rng.sample(aux_pool, min(n_aux, len(aux_pool))):
        weight = rng.uniform(0.01, 0.5)
        components.append(LossComponent(comp.name, weight, comp.description))

    name_parts = [c.name.split("_")[0] for c in components]
    name = "loss_" + "_".join(name_parts)

    return SynthesizedLoss(
        name=name,
        components=components,
        description=f"Synthesized loss with {len(components)} components",
        seed=seed or 0,
    )
