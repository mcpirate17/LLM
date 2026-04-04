"""
Shared utilities for model evaluation and micro-training.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def tokenize_string(text: str, vocab_size: int) -> np.ndarray:
    """Tokenize text as UTF-8 bytes modulo vocab size using native NumPy ops."""
    encoded = text.encode("utf-8", errors="ignore")
    if not encoded:
        return np.empty(0, dtype=np.int64)
    byte_view = np.frombuffer(encoded, dtype=np.uint8)
    if vocab_size < 256:
        return np.remainder(byte_view, vocab_size).astype(np.int64, copy=False)
    if vocab_size == 256:
        return byte_view.astype(np.int64, copy=False)
    return np.remainder(byte_view.astype(np.int64, copy=False), vocab_size)


def tokenize_file(path: Path, vocab_size: int) -> np.ndarray:
    """Tokenize a text file as UTF-8 bytes modulo vocab size."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return tokenize_string(text, vocab_size)


def make_batches(
    tokens: Sequence[int] | np.ndarray,
    batch_size: int,
    seq_len: int,
    n_batches: int,
    device: torch.device,
    seed: int = 42,
) -> List[torch.Tensor]:
    """Create randomized (B, S) batches from a token sequence."""
    if len(tokens) < seq_len + 1:
        return []
    t = torch.as_tensor(tokens, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    max_start = len(tokens) - seq_len - 1
    all_starts = torch.randint(0, max_start, (n_batches, batch_size), generator=gen)
    offsets = torch.arange(seq_len).view(1, 1, seq_len)
    indices = all_starts.unsqueeze(-1) + offsets
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


def compute_grad_norm(model: nn.Module) -> float:
    """Compute total L2 gradient norm across all parameters."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    try:
        norms = torch._foreach_norm(grads, 2)
        norm_vec = torch.stack([n.detach() for n in norms])
        return float(torch.linalg.vector_norm(norm_vec, ord=2).item())
    except RuntimeError:
        total = 0.0
        for grad in grads:
            total += grad.data.float().norm().item() ** 2
        return total**0.5


@torch.no_grad()
def batched_span_mean_log_probs(
    model: nn.Module,
    sequences: Sequence[Sequence[int] | np.ndarray],
    start_positions: Sequence[int],
    vocab_size: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Return mean token log-prob over per-sequence scoring spans.

    For sequence i, scores predictions over token positions
    ``[start_positions[i] + 1, len(sequence_i) - 1]``.
    Invalid or empty spans return ``-inf``.
    """
    n_seq = len(sequences)
    if n_seq == 0:
        return torch.empty(0, dtype=torch.float32)

    lengths = torch.as_tensor([len(seq) for seq in sequences], dtype=torch.long)
    starts = torch.as_tensor(start_positions, dtype=torch.long)
    valid = lengths >= 2
    if not bool(valid.any()):
        return torch.full((n_seq,), float("-inf"), dtype=torch.float32)

    valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
    valid_sequences = [sequences[int(i)] for i in valid_idx.tolist()]
    valid_lengths = lengths[valid_idx]
    valid_starts = starts[valid_idx]

    dev = torch.device(device)
    max_len = int(valid_lengths.max().item())
    padded = torch.zeros((len(valid_sequences), max_len), dtype=torch.long, device=dev)
    for row, seq in enumerate(valid_sequences):
        seq_tensor = torch.as_tensor(seq, dtype=torch.long, device=dev)
        padded[row, : seq_tensor.numel()] = seq_tensor

    logits = model(padded)
    if logits.shape[-1] > vocab_size:
        logits = logits[..., :vocab_size]
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    targets = padded[:, 1:]
    token_lps = log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)

    positions = torch.arange(max_len - 1, device=dev).unsqueeze(0)
    span_mask = (positions >= valid_starts.unsqueeze(1)) & (
        positions < (valid_lengths - 1).unsqueeze(1)
    )
    token_counts = span_mask.sum(dim=1)
    mean_lps = torch.full(
        (len(valid_sequences),), float("-inf"), dtype=torch.float32, device=dev
    )
    valid_spans = token_counts > 0
    if bool(valid_spans.any()):
        sums = (token_lps * span_mask).sum(dim=1)
        mean_lps[valid_spans] = sums[valid_spans] / token_counts[valid_spans]

    out = torch.full((n_seq,), float("-inf"), dtype=torch.float32, device=dev)
    out[valid_idx] = mean_lps
    return out.cpu()


@torch.no_grad()
def mean_token_log_prob(
    model: nn.Module,
    token_ids: Sequence[int] | np.ndarray,
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
    score = batched_span_mean_log_probs(
        model,
        [token_ids],
        [start_pos],
        vocab_size=vocab_size,
        device=device,
    )
    return float(score[0].item())


def iter_eligible_params(
    model: nn.Module,
) -> Iterable[tuple[str, torch.nn.Parameter]]:
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
