"""Noise sensitivity evaluation for robustness testing.

Measures how much a model's loss degrades when Gaussian noise
is injected into embedding outputs, providing a robustness signal
independent of training data.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import torch
import torch.nn as nn

from research.defaults import VOCAB_SIZE
from .utils import measure_loss as _measure_loss_util

logger = logging.getLogger(__name__)


def evaluate_noise_sensitivity(
    model: nn.Module,
    input_batches: List[torch.Tensor],
    device: torch.device,
    noise_levels: Sequence[float] = (0.01, 0.05, 0.1),
    vocab_size: int = 0,
) -> Dict:
    """Evaluate model robustness to Gaussian noise on embeddings.

    Hooks into the first layer after embedding and adds calibrated
    Gaussian noise at each level. Measures loss degradation.

    Args:
        model: Trained model to evaluate.
        input_batches: List of input_ids tensors.
        device: Device for evaluation.
        noise_levels: Standard deviations of noise to inject.
        vocab_size: Vocab size for loss computation (auto-detected if 0).

    Returns:
        Dict with per-level results and overall noise_sensitivity_score.
    """
    if not input_batches:
        return {"noise_sensitivity_score": 0.0, "levels": {}}

    model.eval()

    # Auto-detect vocab_size from model head
    if vocab_size <= 0:
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Linear) and m.out_features > 1000:
                vocab_size = m.out_features
                break
        if vocab_size <= 0:
            vocab_size = VOCAB_SIZE

    # Measure baseline loss (no noise)
    baseline_loss = _measure_loss(model, input_batches, device, vocab_size)
    if baseline_loss is None or baseline_loss <= 0:
        return {"noise_sensitivity_score": 0.0, "levels": {}}

    # Find the embedding layer to hook
    embed_module = None
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            embed_module = m
            break

    if embed_module is None:
        return {"noise_sensitivity_score": 0.0, "levels": {}, "error": "no_embedding"}

    results_by_level = {}
    for noise_std in noise_levels:
        # Register hook to add noise after embedding
        noise_container = {"std": noise_std}

        def _add_noise(module, input, output, nc=noise_container):
            noise = torch.randn_like(output) * nc["std"]
            return output + noise

        handle = embed_module.register_forward_hook(_add_noise)
        try:
            noisy_loss = _measure_loss(model, input_batches, device, vocab_size)
        finally:
            handle.remove()

        if noisy_loss is not None:
            degradation = (noisy_loss - baseline_loss) / max(baseline_loss, 1e-8)
            results_by_level[noise_std] = {
                "noisy_loss": round(noisy_loss, 6),
                "degradation": round(degradation, 4),
                "robust": degradation < 0.5,
            }

    # Score: fraction of noise levels where degradation < 50%
    robust_count = sum(1 for r in results_by_level.values() if r.get("robust", False))
    total = len(results_by_level) or 1
    score = robust_count / total

    return {
        "noise_sensitivity_score": round(score, 4),
        "baseline_loss": round(baseline_loss, 6),
        "levels": results_by_level,
    }


def _measure_loss(
    model: nn.Module,
    input_batches: List[torch.Tensor],
    device: torch.device,
    vocab_size: int,
) -> float | None:
    """Measure average cross-entropy loss over batches."""
    return _measure_loss_util(model, input_batches, device, vocab_size)
