"""Shared helpers for controlled language probes."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import _get_tiktoken_encoder


def dedupe_lower_words(words: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for word in words:
        cleaned = word.strip().lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return tuple(out)


def encode_controlled_text(
    text: str,
    *,
    vocab_size: int,
    tokenizer: str,
    tiktoken_encoding: str,
) -> tuple[int, ...]:
    tok = (tokenizer or "tiktoken").strip().lower()
    if tok in ("byte", "bytes"):
        ids = tuple(int(b) for b in text.encode("utf-8"))
    else:
        enc_name = "gpt2" if tok == "gpt2" else tiktoken_encoding
        ids = tuple(
            int(i)
            for i in _get_tiktoken_encoder(enc_name).encode(text, allowed_special=set())
        )
    if ids and max(ids) >= int(vocab_size):
        raise ValueError(
            f"token id {max(ids)} exceeds model vocab_size={int(vocab_size)}"
        )
    return ids


def next_token_loss(
    model: nn.Module,
    batch: torch.Tensor,
    *,
    vocab_size: int,
    pad_id: int = 0,
) -> torch.Tensor:
    logits = model(batch)
    if logits.shape[-1] > vocab_size:
        logits = logits[..., :vocab_size]
    targets = batch[:, 1:].contiguous()
    pred = logits[:, :-1, :].contiguous()
    mask = targets != pad_id
    if not bool(mask.any()):
        return pred.sum() * 0.0
    return F.cross_entropy(pred[mask].float(), targets[mask])
