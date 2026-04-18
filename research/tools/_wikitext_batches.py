from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def ensure_full_wikitext(min_train_bytes: int = 5_000_000) -> tuple[Path, Path]:
    """Ensure the full WikiText-103 cache exists, not the tiny screening cache."""
    from research.eval.wikitext_eval import (
        _DEFAULT_MAX_CHARS_TRAIN,
        _DEFAULT_MAX_CHARS_VAL,
        _WIKITEXT_CACHE_DIR,
        _download_wikitext,
    )

    cache_dir = _WIKITEXT_CACHE_DIR / "wikitext-103-raw-v1"
    train_path = cache_dir / "train.txt"
    val_path = cache_dir / "validation.txt"
    if train_path.exists() and train_path.stat().st_size < min_train_bytes:
        logger.warning(
            "Cached WikiText train is only %d bytes; deleting screening cache and re-downloading full corpus",
            train_path.stat().st_size,
        )
        train_path.unlink(missing_ok=True)
        val_path.unlink(missing_ok=True)
    return _download_wikitext(
        max_chars_train=_DEFAULT_MAX_CHARS_TRAIN,
        max_chars_val=_DEFAULT_MAX_CHARS_VAL,
    )


@dataclass(slots=True)
class WikitextBatchSource:
    train_tokens: torch.Tensor
    val_tokens: torch.Tensor
    batch_size: int
    seq_len: int
    vocab_size: int
    max_val_batches: int = 64
    _stride: int = field(init=False, repr=False)
    _train_window_count: int = field(init=False, repr=False)
    _val_window_count: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.train_tokens = self.train_tokens.to(dtype=torch.long, device="cpu")
        self.val_tokens = self.val_tokens.to(dtype=torch.long, device="cpu")
        self._stride = int(self.batch_size) * int(self.seq_len)
        train_windows = int(self.train_tokens.numel()) // self._stride
        if train_windows < 2:
            raise RuntimeError(
                f"WikiText too small: {int(self.train_tokens.numel())} tokens, need at least {self._stride * 2}"
            )
        self._train_window_count = train_windows
        self._val_window_count = min(
            int(self.max_val_batches), int(self.val_tokens.numel()) // self._stride
        )

    @property
    def train_window_count(self) -> int:
        return self._train_window_count

    @property
    def val_window_count(self) -> int:
        return self._val_window_count

    def sample_train_batch(
        self,
        *,
        device: str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        index = int(
            torch.randint(
                self._train_window_count,
                (1,),
                generator=generator,
            ).item()
        )
        start = index * self._stride
        end = start + self._stride
        batch = self.train_tokens[start:end].reshape(self.batch_size, self.seq_len)
        return batch.to(device=device, non_blocking=True)

    def iter_val_batches(self, *, device: str):
        for index in range(self._val_window_count):
            start = index * self._stride
            end = start + self._stride
            batch = self.val_tokens[start:end].reshape(self.batch_size, self.seq_len)
            yield batch.to(device=device, non_blocking=True)


def load_wikitext_batch_source(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    tokenizer_encoding: str = "gpt2",
    max_val_batches: int = 64,
) -> WikitextBatchSource:
    train_path, val_path = ensure_full_wikitext()
    train_size = train_path.stat().st_size
    val_size = val_path.stat().st_size
    logger.info(
        "WikiText train: %s (%.1f MB), val: %s (%.1f KB)",
        train_path,
        train_size / 1e6,
        val_path,
        val_size / 1e3,
    )

    import tiktoken

    enc = tiktoken.get_encoding(tokenizer_encoding)
    train_text = train_path.read_text(encoding="utf-8", errors="replace")
    val_text = val_path.read_text(encoding="utf-8", errors="replace")
    train_tokens = torch.as_tensor(
        enc.encode(train_text, allowed_special=set()),
        dtype=torch.long,
    )
    val_tokens = torch.as_tensor(
        enc.encode(val_text, allowed_special=set()),
        dtype=torch.long,
    )
    if vocab_size > 0 and vocab_size < enc.n_vocab:
        train_tokens = torch.remainder(train_tokens, int(vocab_size))
        val_tokens = torch.remainder(val_tokens, int(vocab_size))
    logger.info(
        "Tokenized WikiText: train=%d tokens, val=%d tokens",
        int(train_tokens.numel()),
        int(val_tokens.numel()),
    )
    return WikitextBatchSource(
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        max_val_batches=max_val_batches,
    )
