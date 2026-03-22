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

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class LossComponent:
    """A single component of a synthesized loss."""

    name: str
    weight: float = 1.0
    description: str = ""


@dataclass
class SynthesizedLoss:
    """A synthesized loss function."""

    name: str
    components: List[LossComponent] = field(default_factory=list)
    description: str = ""
    seed: int = 0

    def compute(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the synthesized loss.
        Optimized to share intermediate results like log_softmax across components.
        """
        B, S, V = logits.shape
        flat_logits = logits.reshape(-1, V)
        flat_targets = targets.reshape(-1)

        # Pre-compute shared results
        log_probs = None
        probs = None

        total = torch.tensor(0.0, device=logits.device)
        for comp in self.components:
            # Handle components that need original 3D shape
            if comp.name == "spectral_loss":
                total = total + comp.weight * _compute_spectral_component(logits)
                continue

            # Lazy compute log_probs/probs if needed by this component
            if comp.name in {
                "label_smoothed_ce",
                "rank_weighted_ce",
                "tropical_ce",
                "contrastive_push",
                "kl_uniform",
            }:
                if log_probs is None:
                    log_probs = F.log_softmax(flat_logits, dim=-1)
            if comp.name in {"entropy_reg"}:
                if probs is None:
                    probs = F.softmax(flat_logits, dim=-1)

            total = total + comp.weight * _compute_component_fast(
                comp.name, flat_logits, flat_targets, log_probs, probs
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


# ── Loss Components ───────────────────────────────────────────────────


def _compute_spectral_component(logits: torch.Tensor) -> torch.Tensor:
    """Compute high-frequency energy penalty on the logit sequence."""
    if logits.dim() >= 3:
        freq = torch.fft.rfft(logits.float(), dim=1)
        high_freq_energy = freq[:, freq.shape[1] // 2 :].abs().mean()
        return high_freq_energy * 0.01
    return torch.tensor(0.0, device=logits.device)


def _loss_cross_entropy(flat_logits, flat_targets, log_probs, probs):
    return F.cross_entropy(flat_logits, flat_targets)


def _loss_label_smoothed_ce(flat_logits, flat_targets, log_probs, probs):
    if log_probs is None:
        log_probs = F.log_softmax(flat_logits, dim=-1)
    smooth = 0.1
    nll = -log_probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
    smooth_loss = -log_probs.mean(dim=-1)
    return ((1 - smooth) * nll + smooth * smooth_loss).mean()


def _loss_rank_weighted_ce(flat_logits, flat_targets, log_probs, probs):
    if log_probs is None:
        log_probs = F.log_softmax(flat_logits, dim=-1)
    nll = -log_probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
    ranks = flat_logits.argsort(dim=-1, descending=True) == flat_targets.unsqueeze(1)
    rank_pos = ranks.float().argmax(dim=-1).float()
    weights = 1.0 + torch.log1p(rank_pos)
    return (nll * weights).mean()


def _loss_tropical_ce(flat_logits, flat_targets, log_probs, probs):
    if log_probs is None:
        log_probs = F.log_softmax(flat_logits, dim=-1)
    target_log_probs = log_probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
    mask = torch.ones_like(log_probs).scatter_(1, flat_targets.unsqueeze(1), 0)
    max_other = (log_probs * mask + (1 - mask) * -1e9).max(dim=-1).values
    margin = max_other - target_log_probs
    return F.relu(margin + 1.0).mean()


def _loss_contrastive_push(flat_logits, flat_targets, log_probs, probs):
    target_logits = flat_logits.gather(1, flat_targets.unsqueeze(1))
    topk, _ = flat_logits.topk(6, dim=-1)
    return F.relu(topk[:, 1:] - target_logits + 0.5).mean()


def _loss_entropy_reg(flat_logits, flat_targets, log_probs, probs):
    if probs is None:
        probs = F.softmax(flat_logits, dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
    return entropy * 0.1


def _loss_gradient_penalty(flat_logits, flat_targets, log_probs, probs):
    return flat_logits.pow(2).mean() * 0.001


def _loss_kl_uniform(flat_logits, flat_targets, log_probs, probs):
    if log_probs is None:
        log_probs = F.log_softmax(flat_logits, dim=-1)
    V = flat_logits.shape[-1]
    return -(log_probs.mean(dim=-1) + math.log(V)).mean()


_LOSS_DISPATCH = {
    "cross_entropy": _loss_cross_entropy,
    "label_smoothed_ce": _loss_label_smoothed_ce,
    "rank_weighted_ce": _loss_rank_weighted_ce,
    "tropical_ce": _loss_tropical_ce,
    "contrastive_push": _loss_contrastive_push,
    "entropy_reg": _loss_entropy_reg,
    "gradient_penalty": _loss_gradient_penalty,
    "kl_uniform": _loss_kl_uniform,
}


def _compute_component_fast(
    name: str,
    flat_logits: torch.Tensor,
    flat_targets: torch.Tensor,
    log_probs: Optional[torch.Tensor] = None,
    probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute a single loss component using shared intermediates."""
    fn = _LOSS_DISPATCH.get(name, _loss_cross_entropy)
    return fn(flat_logits, flat_targets, log_probs, probs)


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
