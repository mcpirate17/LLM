"""Long-context scaling sweep for robustness evaluation.

Tests how well a model handles increasing sequence lengths
compared to its base performance at the default length.
"""

from __future__ import annotations

import gc
import logging
from typing import Callable, Dict, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from .training_core import run_training_loop
from .utils import language_model_loss


def run_long_context_sweep(
    make_model_fn: Callable[[], nn.Module],
    vocab_size: int,
    device: torch.device,
    base_loss: float,
    seq_lens: Sequence[int] = (512, 1024),
    n_steps: int = 200,
    batch_size: int = 2,
    lr: float = 3e-4,
) -> Dict:
    """Train at increasing sequence lengths and measure loss scaling.

    Args:
        make_model_fn: Callable that returns a fresh model instance.
        vocab_size: Vocabulary size for random data generation.
        device: Training device.
        base_loss: Reference loss at the model's default sequence length.
        seq_lens: Sequence lengths to test (sorted ascending recommended).
        n_steps: Training steps per sequence length.
        batch_size: Batch size per training run.
        lr: Learning rate.

    Returns:
        Dict with scaling_results, max_viable_len, and long_context_score.
    """
    scaling_results = {}
    max_viable_len = 0

    for seq_len in sorted(seq_lens):
        try:
            model = make_model_fn().to(device)
            model.train()
            _data_gen = torch.Generator(device=device).manual_seed(42)

            def compute_loss(_step: int) -> torch.Tensor:
                input_ids = torch.randint(
                    0,
                    vocab_size,
                    (batch_size, seq_len),
                    device=device,
                    generator=_data_gen,
                )
                try:
                    with torch.amp.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=(device.type == "cuda"),
                    ):
                        logits = model(input_ids)
                        return language_model_loss(logits, input_ids, vocab_size)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.info("OOM at seq_len=%d, stopping sweep", seq_len)
                        return torch.tensor(float("inf"), device=device)
                    raise

            result = run_training_loop(
                model.parameters(),
                compute_loss,
                n_steps=n_steps,
                optimizer_name="adamw",
                lr=lr,
                clip_grad=1.0,
            )
            final_loss = result.final_loss

            loss_ratio = (
                final_loss / max(base_loss, 1e-8) if base_loss > 0 else float("inf")
            )
            scaling_results[seq_len] = {
                "final_loss": round(final_loss, 6),
                "loss_ratio": round(loss_ratio, 4),
                "viable": loss_ratio < 2.0,
            }
            if loss_ratio < 2.0:
                max_viable_len = seq_len

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                scaling_results[seq_len] = {
                    "final_loss": None,
                    "loss_ratio": None,
                    "viable": False,
                    "error": "OOM",
                }
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                break
            raise
        except Exception as e:
            scaling_results[seq_len] = {
                "final_loss": None,
                "loss_ratio": None,
                "viable": False,
                "error": str(e)[:100],
            }

    # Graded score from tested lengths:
    # loss_ratio<=1.0 -> 1.0, loss_ratio>=2.0 -> 0.0, linear in-between.
    per_len_scores = []
    for r in scaling_results.values():
        lr_ratio = r.get("loss_ratio")
        if lr_ratio is None:
            continue
        per_len_scores.append(max(0.0, min(1.0, 2.0 - float(lr_ratio))))
    long_context_score = (
        (sum(per_len_scores) / len(per_len_scores)) if per_len_scores else 0.0
    )

    return {
        "scaling_results": scaling_results,
        "max_viable_len": max_viable_len,
        "long_context_score": round(long_context_score, 4),
    }
