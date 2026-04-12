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

from ._runner_native import load_runner_native
from research.training._data_native import load_data_native

logger = logging.getLogger(__name__)


def make_adamw(
    params,
    *,
    lr: float,
    fused_if_available: bool = True,
    **kwargs,
):
    """Create AdamW, using fused kernels on CUDA when the local build supports it."""
    if fused_if_available and torch.cuda.is_available():
        try:
            return torch.optim.AdamW(params, lr=lr, fused=True, **kwargs)
        except TypeError:
            pass
    return torch.optim.AdamW(params, lr=lr, **kwargs)


def language_model_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy over next-token logits with vocab clipping."""
    try:
        return load_runner_native().next_token_cross_entropy(
            logits,
            targets,
            int(vocab_size),
            str(reduction),
        )
    except Exception:
        score_logits = logits[:, :-1].contiguous()
        if score_logits.shape[-1] > vocab_size:
            score_logits = score_logits[..., :vocab_size]
        return F.cross_entropy(
            score_logits.reshape(-1, score_logits.shape[-1]),
            targets[:, 1:].reshape(-1),
            reduction=reduction,
        )


def clip_grad_norm(
    parameters: Iterable[torch.Tensor],
    max_norm: float,
) -> torch.Tensor:
    """Clip dense gradients through the native runner when possible."""
    params = [param for param in parameters if param.grad is not None]
    if not params:
        return torch.zeros((), dtype=torch.float32)
    try:
        return load_runner_native().clip_grad_norm_(
            [param.grad for param in params],
            float(max_norm),
            1e-6,
        )
    except Exception:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)


def move_batches_to_device(
    batches: Sequence[torch.Tensor],
    device: str | torch.device,
) -> List[torch.Tensor]:
    """Move a batch list to a device without reallocating when already resident."""
    target = torch.device(device)
    out: List[torch.Tensor] = []
    for batch in batches:
        if batch.device == target:
            out.append(batch)
        else:
            out.append(batch.to(target, non_blocking=(target.type == "cuda")))
    return out


def tokenize_string(text: str, vocab_size: int) -> np.ndarray:
    """Tokenize text as UTF-8 bytes modulo vocab size using native NumPy ops."""
    if not text:
        return np.empty(0, dtype=np.int64)
    try:
        return load_data_native().byte_tokenize_utf8(text, int(vocab_size)).numpy()
    except Exception:
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
    try:
        return (
            load_data_native()
            .byte_tokenize_file_utf8(str(path), int(vocab_size))
            .numpy()
        )
    except Exception:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return tokenize_string(text, vocab_size)


def make_batches(
    tokens: Sequence[int] | np.ndarray,
    batch_size: int,
    seq_len: int,
    n_batches: int,
    device: str | torch.device,
    seed: int = 42,
) -> List[torch.Tensor]:
    """Create randomized (B, S) batches from a token sequence."""
    if len(tokens) < seq_len + 1:
        return []
    t = torch.as_tensor(tokens, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    max_start = len(tokens) - seq_len - 1
    all_starts = torch.randint(0, max_start, (n_batches, batch_size), generator=gen)
    try:
        native = load_data_native()
        flat_batches = native.gather_token_batch(
            t.contiguous(),
            all_starts.reshape(-1).contiguous(),
            int(seq_len),
        )
        all_tokens = flat_batches.reshape(n_batches, batch_size, seq_len)
    except Exception:
        offsets = torch.arange(seq_len).view(1, 1, seq_len)
        indices = all_starts.unsqueeze(-1) + offsets
        all_tokens = t[indices.reshape(-1)].reshape(n_batches, batch_size, seq_len)
    all_tokens = all_tokens.to(torch.device(device))
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

    from .training_core import run_training_loop

    def _run(model: nn.Module, run_lr: float) -> float:
        def compute_loss(step: int) -> torch.Tensor:
            batch = batches[step % len(batches)]
            logits = model(batch)
            return language_model_loss(logits, batch, vocab_size)

        result = run_training_loop(
            model.parameters(),
            compute_loss,
            n_steps=n_steps,
            optimizer_name="adamw",
            lr=run_lr,
            clip_grad=clip_grad,
            warmup_steps=warmup_steps,
            loss_trajectory=loss_trajectory,
        )
        if result.diverged:
            logger.warning(
                "Micro-train loss is not finite after %d steps (lr=%.1e)",
                result.steps_completed,
                run_lr,
            )
        return result.final_loss

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
                except Exception as exc:
                    logger.debug("reset_parameters() failed during retry: %s", exc)
        result = _run(model, lr * 0.1)
    return result


def micro_train_and_measure_perplexity(
    model: nn.Module,
    train_batches: List[torch.Tensor],
    val_batches: List[torch.Tensor],
    vocab_size: int,
    *,
    n_train_steps: int,
    lr: float,
) -> tuple[Optional[float], float, Optional[float]]:
    """Shared in-place micro-train + pre/post perplexity measurement flow."""
    pre_ppl = compute_perplexity(model, val_batches, vocab_size)
    train_final_loss = micro_train_loop(
        model,
        train_batches,
        vocab_size,
        n_steps=n_train_steps,
        lr=lr,
    )
    post_ppl = compute_perplexity(model, val_batches, vocab_size)
    return pre_ppl, train_final_loss, post_ppl


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
    skipped = 0
    first_error: Exception | None = None
    with torch.no_grad():
        for batch in input_batches:
            try:
                batch = batch.to(device)
                logits = model(batch)
                v = vocab_size if vocab_size > 0 else logits.shape[-1]
                loss = language_model_loss(logits, batch, v)
                if torch.isfinite(loss):
                    losses.append(loss.item())
            except Exception as exc:
                skipped += 1
                if first_error is None:
                    first_error = exc
    if skipped:
        logger.warning(
            "measure_loss skipped %d batches; first error: %s", skipped, first_error
        )
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


def _pad_sequences_python(
    sequences: list[list[int]],
    device: str | torch.device,
) -> tuple[torch.Tensor, int]:
    """Fallback: pad variable-length sequences into (batch, max_len) int64 tensor."""
    max_len = max((len(s) for s in sequences), default=0)
    if max_len == 0:
        return torch.zeros((len(sequences), 1), dtype=torch.long), 0
    padded_np = np.zeros((len(sequences), max_len), dtype=np.int64)
    for row, seq in enumerate(sequences):
        seq_np = np.asarray(seq, dtype=np.int64)
        padded_np[row, : seq_np.size] = seq_np
    padded = torch.from_numpy(padded_np)
    dev = torch.device(device)
    if dev.type == "cuda":
        padded = padded.pin_memory().to(dev, non_blocking=True)
    elif dev != torch.device("cpu"):
        padded = padded.to(dev)
    return padded, max_len


def _span_mean_python(
    token_lps: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    max_len: int,
    dev: torch.device,
) -> torch.Tensor:
    """Fallback: compute span-masked mean token log-probs."""
    positions = torch.arange(max_len - 1, device=dev).unsqueeze(0)
    span_mask = (positions >= starts.unsqueeze(1)) & (
        positions < (lengths - 1).unsqueeze(1)
    )
    token_counts = span_mask.sum(dim=1)
    n = token_lps.size(0)
    mean_lps = torch.full((n,), float("-inf"), dtype=torch.float32, device=dev)
    valid_spans = token_counts > 0
    if bool(valid_spans.any()):
        sums = (token_lps * span_mask).sum(dim=1)
        mean_lps[valid_spans] = sums[valid_spans] / token_counts[valid_spans]
    return mean_lps


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
    valid_idx_list = valid_idx.tolist()
    valid_starts_cpu = starts[valid_idx]
    valid_lengths_cpu = lengths[valid_idx]

    # Try native padding (eliminates Python loop + numpy intermediary)
    _native_ext = None
    try:
        from ._eval_native import load_eval_native

        _native_ext = load_eval_native()
        valid_seqs = [list(sequences[i]) for i in valid_idx_list]
        padded, valid_lengths_dev, max_len = _native_ext.pad_sequences_native(
            valid_seqs, str(device)
        )
        dev = padded.device
        valid_starts_dev = valid_starts_cpu.to(dev, non_blocking=(dev.type == "cuda"))
    except Exception:
        _native_ext = None
        valid_seqs_py = [list(sequences[i]) for i in valid_idx_list]
        padded, max_len = _pad_sequences_python(valid_seqs_py, device)
        dev = padded.device
        valid_lengths_dev = valid_lengths_cpu.to(dev, non_blocking=(dev.type == "cuda"))
        valid_starts_dev = valid_starts_cpu.to(dev, non_blocking=(dev.type == "cuda"))

    logits = model(padded)
    if logits.shape[-1] > vocab_size:
        logits = logits[..., :vocab_size]
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    targets = padded[:, 1:]
    token_lps = log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)

    # Try native span scoring
    if _native_ext is not None:
        try:
            mean_lps = _native_ext.span_mean_log_probs_native(
                token_lps, valid_starts_dev, valid_lengths_dev, max_len
            )
        except Exception:
            mean_lps = _span_mean_python(
                token_lps, valid_starts_dev, valid_lengths_dev, max_len, dev
            )
    else:
        mean_lps = _span_mean_python(
            token_lps, valid_starts_dev, valid_lengths_dev, max_len, dev
        )

    out = torch.full((n_seq,), float("-inf"), dtype=torch.float32, device=dev)
    out[valid_idx] = mean_lps
    return out.cpu() if dev.type != "cpu" else out


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
            loss = language_model_loss(logits, batch, vocab_size, reduction="sum")
            if torch.isfinite(loss):
                total_loss += loss.item()
                total_tokens += batch[:, 1:].numel()

    if total_tokens == 0:
        return None
    return math.exp(min(total_loss / total_tokens, 20.0))
