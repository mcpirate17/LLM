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

        log_probs = None
        total = flat_logits.new_zeros(())
        for comp_name, comp_weight in zip(
            self._component_names, self._component_weights
        ):
            if comp_name == "spectral_loss":
                if spectral_logits is not None:
                    total = total + comp_weight * compute_spectral_component(
                        spectral_logits
                    )
                continue

            if comp_name in LOG_PROB_COMPONENTS and log_probs is None:
                log_probs = torch.nn.functional.log_softmax(flat_logits, dim=-1)

            total = total + comp_weight * compute_component_fast(
                comp_name, flat_logits, flat_targets, log_probs
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
#
# Catalogue of (name, description) for synthesizable loss components.
# `synthesize_loss` always assigns aux weights via `rng.uniform(...)`, so
# carrying default weights here would be misleading dead data.

_CE_VARIANTS: tuple[str, ...] = (
    "cross_entropy",
    "label_smoothed_ce",
    "rank_weighted_ce",
    "tropical_ce",
)

LOSS_COMPONENTS: dict[str, str] = {
    "cross_entropy": "Standard cross-entropy",
    "label_smoothed_ce": "Label-smoothed cross-entropy",
    "rank_weighted_ce": "Rank-weighted cross-entropy",
    "tropical_ce": "Tropical (min-plus) cross-entropy",
    "spectral_loss": "Spectral regularization",
    "contrastive_push": "Contrastive margin loss",
    "entropy_reg": "Entropy regularization",
    "gradient_penalty": "Logit magnitude penalty",
    "kl_uniform": "KL from uniform",
}


def synthesize_loss(
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> SynthesizedLoss:
    """Generate a random loss function: one CE-style primary + 0–3 aux terms."""
    rng = rng if rng is not None else random.Random(seed)
    primary = rng.choice(_CE_VARIANTS)
    components = [LossComponent(primary, 1.0)]

    aux_names = [
        n for n in LOSS_COMPONENTS if n != primary and n not in _CE_VARIANTS
    ]
    n_aux = rng.randint(0, 3)
    for name in rng.sample(aux_names, min(n_aux, len(aux_names))):
        components.append(
            LossComponent(name, rng.uniform(0.01, 0.5), LOSS_COMPONENTS[name])
        )

    pretty = "loss_" + "_".join(c.name.split("_")[0] for c in components)
    return SynthesizedLoss(
        name=pretty,
        components=components,
        description=f"Synthesized loss with {len(components)} components",
        seed=seed or 0,
    )
