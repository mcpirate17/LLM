"""
Shared utilities for model evaluation and micro-training.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def safe_json_load(raw: Any) -> Any:
    """Safely parse a JSON string, returning None on failure.

    If *raw* is already a dict/list (i.e. pre-parsed), return it directly.
    """
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        import json

        return json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def safe_parse_float(value: Any) -> Optional[float]:
    """Convert *value* to float, returning None on failure.

    Thin wrapper around ``shared_utils.safe_float`` — handles strings,
    numpy scalars, torch tensors, bytes blobs, and Python numerics.
    """
    from research.scientist.shared_utils import safe_float

    return safe_float(value, default=None)


def tokenize_string(text: str, vocab_size: int) -> List[int]:
    """Tokenize a string into a list of integers (UTF-8 bytes modulo vocab_size)."""
    return [b % vocab_size for b in text.encode("utf-8", errors="ignore")]


def tokenize_file(path: Path, vocab_size: int) -> List[int]:
    """Tokenize a text file into a list of integers (UTF-8 bytes modulo vocab_size)."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return tokenize_string(text, vocab_size)


def make_batches(
    tokens: List[int],
    batch_size: int,
    seq_len: int,
    n_batches: int,
    device: torch.device,
    seed: int = 42,
) -> List[torch.Tensor]:
    """Create a list of randomized (B, S) batches from a list of tokens."""
    if len(tokens) < seq_len + 1:
        return []
    t = torch.tensor(tokens, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    max_start = len(tokens) - seq_len - 1
    # Pre-compute all start indices at once
    all_starts = torch.randint(0, max_start, (n_batches, batch_size), generator=gen)
    # Build index matrix: (n_batches, batch_size, seq_len)
    offsets = torch.arange(seq_len).unsqueeze(0).unsqueeze(0)  # (1, 1, S)
    indices = all_starts.unsqueeze(-1) + offsets  # (N, B, S)
    # Gather all batches at once, then move to device
    all_tokens = t[indices.reshape(-1)].reshape(n_batches, batch_size, seq_len)
    return [all_tokens[i].to(device) for i in range(n_batches)]


def micro_train_loop(
    model: nn.Module,
    batches: List[torch.Tensor],
    vocab_size: int,
    n_steps: int = 200,
    lr: float = 3e-4,
    clip_grad: float = 1.0,
    warmup_steps: int = 10,
    loss_trajectory: Optional[dict] = None,
) -> float:
    """Perform a short training loop on the provided batches.

    Includes LR warmup to handle architectures with extreme initial logits.
    If training diverges (NaN loss), retries once at 1/10th LR.

    If *loss_trajectory* is not None, it is populated with
    ``{step_number: loss_value}`` for every step (1-indexed).
    """
    model.train()
    if not batches:
        return float("inf")

    def _run(model: nn.Module, run_lr: float) -> float:
        opt = torch.optim.AdamW(model.parameters(), lr=run_lr)
        final_loss = float("inf")
        for step in range(n_steps):
            # LR warmup: ramp from 0 to run_lr over warmup_steps
            if step < warmup_steps:
                warmup_factor = (step + 1) / warmup_steps
                for pg in opt.param_groups:
                    pg["lr"] = run_lr * warmup_factor

            batch = batches[step % len(batches)]
            opt.zero_grad(set_to_none=True)
            logits = model(batch)
            sl = logits[:, :-1].contiguous()
            if sl.shape[-1] > vocab_size:
                sl = sl[..., :vocab_size]

            loss = F.cross_entropy(
                sl.reshape(-1, sl.shape[-1]), batch[:, 1:].reshape(-1)
            )

            if not torch.isfinite(loss):
                logger.warning(
                    "Micro-train loss is not finite at step %d (lr=%.1e)", step, run_lr
                )
                return float("inf")

            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            final_loss = loss.item()
            if loss_trajectory is not None:
                loss_trajectory[step + 1] = final_loss
        return final_loss

    result = _run(model, lr)
    if not math.isfinite(result):
        # Retry with 1/10th LR for architectures with extreme init
        logger.info("Micro-train diverged at lr=%.1e, retrying at %.1e", lr, lr * 0.1)
        if loss_trajectory is not None:
            loss_trajectory.clear()
        # Re-init weights for clean retry
        for m in model.modules():
            if hasattr(m, "reset_parameters"):
                try:
                    m.reset_parameters()
                except Exception:
                    pass
        result = _run(model, lr * 0.1)
    return result


def measure_loss(
    model: nn.Module,
    input_batches: List[torch.Tensor],
    device: torch.device,
    vocab_size: int = 0,
) -> Optional[float]:
    """Measure average cross-entropy loss over batches without training.

    If vocab_size is 0 or not provided, uses the model's output dimension.
    """
    if not input_batches:
        return None
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in input_batches:
            try:
                batch = batch.to(device)
                logits = model(batch)
                v = vocab_size if vocab_size > 0 else logits.shape[-1]
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, v),
                    batch[:, 1:].reshape(-1),
                )
                if torch.isfinite(loss):
                    losses.append(loss.item())
            except Exception:
                continue
    return sum(losses) / len(losses) if losses else None


def cleanup_model(device: torch.device) -> None:
    """Standard cleanup after model training/eval — clear CUDA cache + gc."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    import gc

    gc.collect()


def compute_perplexity(
    model: nn.Module,
    batches: List[torch.Tensor],
    vocab_size: int,
) -> Optional[float]:
    """Compute exponential of cross-entropy loss over batches."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in batches:
            logits = model(batch)
            sl = logits[:, :-1].contiguous()
            if sl.shape[-1] > vocab_size:
                sl = sl[..., :vocab_size]
            loss = F.cross_entropy(
                sl.reshape(-1, sl.shape[-1]), batch[:, 1:].reshape(-1), reduction="sum"
            )
            if torch.isfinite(loss):
                total_loss += loss.item()
                total_tokens += batch[:, 1:].numel()

    if total_tokens == 0:
        return None
    return math.exp(min(total_loss / total_tokens, 20.0))
