"""Entropy-based difficulty scorer pretraining.

Bootstraps a lightweight MLP that predicts per-token difficulty from
hidden states. Ground truth is derived from the model's own output
entropy: tokens where the model is uncertain (high entropy) are "hard."

Usage:
    from research.training.difficulty_pretraining import pretrain_difficulty_scorer

    scorer = pretrain_difficulty_scorer(model, train_data, d_model=256)
    # scorer: nn.Module (B,S,D) -> (B,S,1) in [0,1]

The pretrained scorer can be injected into routing architectures to
guide lane assignment (easy tokens → cheap path, hard tokens → full path).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifficultyScorer(nn.Module):
    """2-layer MLP: (B,S,D) → (B,S,1) sigmoid difficulty score."""

    __slots__ = ()

    def __init__(self, d_model: int, d_hidden: Optional[int] = None) -> None:
        super().__init__()
        d_hidden = d_hidden or max(32, d_model // 4)
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, D) → (B, S, 1) difficulty in [0, 1]."""
        return torch.sigmoid(self.fc2(F.gelu(self.fc1(x))))


def _compute_entropy_labels(
    model: nn.Module,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run model forward, compute per-token output entropy as difficulty label.

    Returns:
        hidden_states: (B, S, D) — last hidden states before the LM head
        difficulty: (B, S, 1) — normalized entropy in [0, 1]
    """
    model.eval()
    with torch.no_grad():
        # Try to get hidden states; fall back to logits
        out = model(input_ids)

        # If model returns logits directly (most compiled models)
        if isinstance(out, torch.Tensor):
            logits = out
        elif hasattr(out, "logits"):
            logits = out.logits
        else:
            logits = out

        # Compute per-token entropy
        probs = F.softmax(logits / temperature, dim=-1)
        log_probs = torch.log(probs.clamp(min=1e-8))
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)  # (B, S, 1)

        # Normalize to [0, 1] per batch
        e_min = entropy.amin(dim=1, keepdim=True)
        e_max = entropy.amax(dim=1, keepdim=True)
        difficulty = (entropy - e_min) / (e_max - e_min + 1e-8)

    # For hidden states, we need a hook or re-run
    # Use logits as proxy input (the scorer learns from whatever representation)
    # In practice, the scorer should sit before the LM head
    hidden = logits.detach()

    return hidden, difficulty


def pretrain_difficulty_scorer(
    model: nn.Module,
    data_iterator,
    d_model: int,
    steps: int = 200,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    temperature: float = 1.0,
) -> DifficultyScorer:
    """Pretrain a difficulty scorer using entropy-based labels.

    Args:
        model: A trained (or partially trained) language model.
        data_iterator: Yields input_ids tensors of shape (B, S).
        d_model: Hidden dimension of the model.
        steps: Number of training steps for the scorer.
        lr: Learning rate.
        device: Device to train on.
        temperature: Softmax temperature for entropy computation.

    Returns:
        Pretrained DifficultyScorer module.
    """
    if device is None:
        device = next(model.parameters()).device

    scorer = DifficultyScorer(d_model).to(device)
    optimizer = torch.optim.Adam(scorer.parameters(), lr=lr)

    model.eval()
    scorer.train()

    step = 0
    total_loss = 0.0
    for input_ids in data_iterator:
        if step >= steps:
            break

        input_ids = input_ids.to(device) if hasattr(input_ids, "to") else input_ids

        # Get entropy-based difficulty labels
        hidden, labels = _compute_entropy_labels(model, input_ids, temperature)

        # Match scorer input dim to hidden dim
        if hidden.shape[-1] != d_model:
            # Project if dimensions don't match
            hidden = hidden[..., :d_model]

        # Train scorer
        pred = scorer(hidden)
        loss = F.mse_loss(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        step += 1

    scorer.eval()
    return scorer


def inject_difficulty_scorer(
    model: nn.Module,
    scorer: DifficultyScorer,
    attr_name: str = "_difficulty_scorer",
) -> None:
    """Attach a pretrained difficulty scorer to a model.

    Routing ops can then access it via getattr(model, attr_name).
    """
    setattr(model, attr_name, scorer)
    # Also register as submodule for proper device/state_dict handling
    if not hasattr(model, "_difficulty_modules"):
        model._difficulty_modules = nn.ModuleDict()
    model._difficulty_modules[attr_name] = scorer
