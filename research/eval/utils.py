"""
Shared utilities for model evaluation and micro-training.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.scientist.shared_utils import clamp
from research.training._native import load_training_native

logger = logging.getLogger(__name__)


def clip01(value: float) -> float:
    """Clamp a probability/accuracy lift to [0, 1]."""
    return clamp(float(value), 0.0, 1.0)


def chance_lift(acc: float, chance: float) -> float:
    """Above-chance fraction: ``(acc - chance) / (1 - chance)``, clamped at chance ≥ 1."""
    chance = clip01(chance)
    if chance >= 1.0:
        return 0.0
    return (float(acc) - chance) / (1.0 - chance)


def model_vocab_size(model: nn.Module) -> int | None:
    """Best-effort vocab size: prefer ``model.vocab_size`` then first ``nn.Embedding``."""
    vocab_size = getattr(model, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            return int(module.num_embeddings)
    return None


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
    return load_training_native().next_token_cross_entropy(
        logits,
        targets,
        int(vocab_size),
        str(reduction),
    )


def clip_grad_norm(
    parameters: Iterable[torch.Tensor],
    max_norm: float,
) -> torch.Tensor:
    """Clip dense gradients in place by total L2 norm."""
    params = [param for param in parameters if param.grad is not None]
    if not params:
        return torch.zeros((), dtype=torch.float32)
    return load_training_native().clip_grad_norm_(
        [param.grad for param in params],
        float(max_norm),
        1e-6,
    )


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


# tiktoken adapters are cached per encoding so repeated calls don't re-init.
_TIKTOKEN_CACHE: dict[str, Any] = {}


def _get_tiktoken_encoder(encoding_name: str = "cl100k_base"):
    """Return a tiktoken Encoding, cached per encoding name."""
    enc = _TIKTOKEN_CACHE.get(encoding_name)
    if enc is None:
        import tiktoken  # local import — only required on the BPE path

        enc = tiktoken.get_encoding(encoding_name)
        _TIKTOKEN_CACHE[encoding_name] = enc
    return enc


def tokenize_words_serial(
    enc, words: Sequence[str], *, encoding_name: str = "cl100k_base"
) -> List[int]:
    """Encode each word with a leading space; require single-token result.

    Serial ``encode`` per word — measured ~10× faster than ``encode_batch``
    for very short single-word inputs (rust thread/setup overhead dominates
    when each item is just a few bytes). Use this for any nano-style probe
    that builds tensors from short word lists.
    """
    out: List[int] = []
    for w in words:
        ids = enc.encode(" " + w, allowed_special=set())
        if len(ids) != 1:
            raise ValueError(f"word {w!r} not single-token under {encoding_name}")
        out.append(int(ids[0]))
    return out


def pack_token_rows(
    rows: Sequence[Sequence[int]],
    device: torch.device,
    *,
    pad_id: int = 0,
) -> torch.Tensor:
    """Numpy fill + single H2D transfer (vs N per-row torch.tensor copies).

    Wins on GPU (one transfer instead of N); near-parity on CPU. Used by
    ar_gate and nano_bind to pack ragged token-id rows into a padded
    [N, max_len] tensor.
    """
    max_len = max(len(r) for r in rows)
    out_np = np.full((len(rows), max_len), int(pad_id), dtype=np.int64)
    for i, row in enumerate(rows):
        if row:
            out_np[i, : len(row)] = row
    return torch.from_numpy(out_np).to(device)


def tokenize_string(
    text: str,
    vocab_size: int,
    *,
    tokenizer: str = "tiktoken",
    tiktoken_encoding: str = "cl100k_base",
) -> np.ndarray:
    """Tokenize text. Default is cl100k_base BPE to match the training
    corpus (research/corpus/wikitext103_train.npy). ``tokenizer='byte'``
    selects the legacy UTF-8 byte path."""
    if not text:
        return np.empty(0, dtype=np.int64)
    tok = (tokenizer or "byte").strip().lower()
    if tok in ("tiktoken", "bpe", "gpt2", "cl100k", "cl100k_base"):
        enc_name = tiktoken_encoding
        if tok in ("gpt2",):
            enc_name = "gpt2"
        elif tok in ("cl100k", "cl100k_base"):
            enc_name = "cl100k_base"
        ids = _get_tiktoken_encoder(enc_name).encode(text, allowed_special=set())
        arr = np.asarray(ids, dtype=np.int64)
        # Clip to model's vocab to mirror the byte path's behavior.
        if vocab_size and arr.size:
            np.minimum(arr, int(vocab_size) - 1, out=arr)
        return arr
    return load_training_native().byte_tokenize_utf8(text, int(vocab_size)).numpy()


def tokenize_file(
    path: Path,
    vocab_size: int,
    *,
    tokenizer: str = "tiktoken",
    tiktoken_encoding: str = "cl100k_base",
) -> np.ndarray:
    """Tokenize a text file. Default is cl100k_base BPE to match the training
    corpus. ``tokenizer='byte'`` selects the legacy native-C++ byte path.
    """
    tok = (tokenizer or "byte").strip().lower()
    if tok in ("tiktoken", "bpe", "gpt2", "cl100k", "cl100k_base"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return tokenize_string(
            text,
            vocab_size,
            tokenizer=tok,
            tiktoken_encoding=tiktoken_encoding,
        )
    return (
        load_training_native()
        .byte_tokenize_file_utf8(str(path), int(vocab_size))
        .numpy()
    )


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
    t = torch.as_tensor(tokens, dtype=torch.long).contiguous()
    gen = torch.Generator().manual_seed(seed)
    max_start = len(tokens) - seq_len - 1
    all_starts = torch.randint(0, max_start, (n_batches, batch_size), generator=gen)
    flat_batches = load_training_native().gather_token_batch(
        t,
        all_starts.reshape(-1).contiguous(),
        int(seq_len),
    )
    all_tokens = flat_batches.reshape(n_batches, batch_size, seq_len)
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
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    valid_batches = torch.zeros((), device=device, dtype=torch.long)
    skipped = 0
    first_error: Exception | None = None
    with torch.inference_mode():
        for batch in input_batches:
            try:
                batch = batch.to(device)
                logits = model(batch)
                v = vocab_size if vocab_size > 0 else logits.shape[-1]
                loss = language_model_loss(logits, batch, v)
                finite = torch.isfinite(loss)
                total_loss = total_loss + torch.where(
                    finite,
                    loss.detach().to(torch.float64),
                    total_loss.new_zeros(()),
                )
                valid_batches = valid_batches + finite.to(torch.long)
            except Exception as exc:
                skipped += 1
                if first_error is None:
                    first_error = exc
    if skipped:
        logger.warning(
            "measure_loss skipped %d batches; first error: %s", skipped, first_error
        )
    count = int(valid_batches.item())
    if count == 0:
        return None
    return float((total_loss / count).item())


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
    valid_starts_cpu = starts[valid_idx]

    from ._eval_native import load_eval_native

    native = load_eval_native()
    valid_seqs = [list(sequences[i]) for i in valid_idx.tolist()]
    padded, valid_lengths_dev, max_len = native.pad_sequences_native(
        valid_seqs, str(device)
    )
    dev = padded.device
    valid_starts_dev = valid_starts_cpu.to(dev, non_blocking=(dev.type == "cuda"))

    logits = model(padded)
    if logits.shape[-1] > vocab_size:
        logits = logits[..., :vocab_size]
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    targets = padded[:, 1:]
    token_lps = log_probs.gather(2, targets.unsqueeze(2)).squeeze(2)

    mean_lps = native.span_mean_log_probs_native(
        token_lps, valid_starts_dev, valid_lengths_dev, max_len
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
    total_loss: torch.Tensor | None = None
    total_tokens: torch.Tensor | None = None
    with torch.inference_mode():
        for batch in batches:
            logits = model(batch)
            loss = language_model_loss(logits, batch, vocab_size, reduction="sum")
            if total_loss is None:
                total_loss = torch.zeros((), device=loss.device, dtype=torch.float64)
                total_tokens = torch.zeros((), device=loss.device, dtype=torch.long)
            finite = torch.isfinite(loss)
            token_count = torch.tensor(
                batch[:, 1:].numel(), device=loss.device, dtype=torch.long
            )
            total_loss = total_loss + torch.where(
                finite,
                loss.detach().to(torch.float64),
                total_loss.new_zeros(()),
            )
            total_tokens = total_tokens + torch.where(
                finite,
                token_count,
                total_tokens.new_zeros(()),
            )

    if total_tokens is None:
        return None
    token_total = int(total_tokens.item())
    if token_total == 0:
        return None
    mean_loss = float((total_loss / token_total).item())
    return math.exp(min(mean_loss, 20.0))
