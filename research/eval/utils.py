"""
Shared utilities for model evaluation and micro-training.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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
    # Gather all batches at once, move to device in one transfer, then split
    all_tokens = t[indices.reshape(-1)].reshape(n_batches, batch_size, seq_len)
    all_tokens = all_tokens.to(device)
    return [all_tokens[i] for i in range(n_batches)]


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


@torch.no_grad()
def mean_token_log_prob(
    model: nn.Module,
    token_ids: List[int],
    vocab_size: int,
    device: str,
    start_pos: int = 0,
    max_seq_len: int = 512,
) -> float:
    """Mean log-probability of tokens ``[start_pos+1:]`` under the model.

    Shared scoring primitive for HellaSwag (continuation scoring) and
    BLiMP (sentence probability comparison).

    Args:
        start_pos: First position whose *next-token prediction* is scored.
                   For whole-sequence scoring (BLiMP), use 0.
                   For continuation scoring (HellaSwag), use ``ctx_len - 1``.
    """
    if len(token_ids) < 2:
        return float("-inf")
    if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    logits = model(input_ids)

    if logits.shape[-1] > vocab_size:
        logits = logits[..., :vocab_size]

    log_probs = F.log_softmax(logits[0], dim=-1)

    end = len(token_ids) - 1
    if start_pos >= end:
        return float("-inf")

    targets = input_ids[0, start_pos + 1 : end + 1]
    pred_log_probs = log_probs[start_pos:end]
    return pred_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).mean().item()


def iter_eligible_params(
    model: nn.Module,
) -> "Iterable[tuple[str, torch.nn.Parameter]]":
    """Yield ``(name, param)`` for 2-D+ trainable params, excluding embeddings.

    Shared filter for pruning, quantization, and sparsity analysis.
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2:
            continue
        if "embed" in name.lower():
            continue
        yield name, param


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
