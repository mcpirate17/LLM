"""
Tiny corpus data pipeline for training-token batch generation.

MVP goals:
- lightweight TXT/JSONL ingestion
- pluggable tokenizer adapter interface
- deterministic batch sampling using caller-provided torch.Generator
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Protocol

import numpy as np
import torch

from ._data_native import load_data_native

logger = logging.getLogger(__name__)


def _byte_tokenize_tensor(text: str, vocab_size: int) -> torch.Tensor:
    if vocab_size <= 0 or not text:
        return torch.empty(0, dtype=torch.long)
    return load_data_native().byte_tokenize_utf8(text, int(vocab_size))


def _whitespace_hash_tokenize_tensor(text: str, vocab_size: int) -> torch.Tensor:
    if vocab_size <= 0 or not text:
        return torch.empty(0, dtype=torch.long)
    return load_data_native().whitespace_hash_tokenize(text, int(vocab_size))


class TokenizerAdapter(Protocol):
    """Tokenizer adapter interface for corpus-mode training."""

    def encode(self, text: str, vocab_size: int) -> List[int]:
        """Encode text to token IDs bounded to vocab_size."""


class ByteTokenizer:
    """Deterministic byte tokenizer with modulo vocab projection."""

    __slots__ = ()

    def encode(self, text: str, vocab_size: int) -> List[int]:
        return _byte_tokenize_tensor(text, vocab_size).tolist()


class WhitespaceHashTokenizer:
    """Simple hashed-whitespace tokenizer for word-like segmentation."""

    __slots__ = ()

    def encode(self, text: str, vocab_size: int) -> List[int]:
        return _whitespace_hash_tokenize_tensor(text, vocab_size).tolist()


class TiktokenAdapter:
    """Production-grade BPE tokenizer via tiktoken (GPT-2 vocabulary).

    Uses the GPT-2 BPE encoding (50,257 subword tokens). Token IDs are
    projected into [0, vocab_size) via modulo so the model's embedding
    layer does not need to change. This preserves architecture fingerprints
    while giving proper subword segmentation.
    """

    __slots__ = ("_enc", "native_vocab_size")

    def __init__(self, encoding_name: str = "gpt2"):
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding_name)
        self.native_vocab_size = self._enc.n_vocab

    def encode(self, text: str, vocab_size: int) -> List[int]:
        ids = self._enc.encode(text, allowed_special=set())
        if vocab_size > 0 and vocab_size < self.native_vocab_size:
            return [t % vocab_size for t in ids]
        return ids


@dataclass(slots=True)
class CorpusConfig:
    path: str
    fmt: str = "auto"  # auto|txt|jsonl
    text_key: str = "text"
    tokenizer: str = "byte"  # byte|whitespace|tiktoken
    max_chars: int = 200_000
    train_fraction: float = 0.9
    val_fraction: float = 0.1
    tiktoken_encoding: str = "gpt2"  # gpt2|cl100k_base


class CorpusTokenBatcher:
    """Loads corpus text once and emits sampled token batches."""

    __slots__ = (
        "config",
        "vocab_size",
        "path",
        "_tokenizer",
        "_tokens",
        "_train_tokens",
        "_val_tokens",
        "_native_ext",
        "_newline_tokens",
    )

    def __init__(self, config: CorpusConfig, vocab_size: int):
        self.config = config
        self.vocab_size = int(vocab_size)
        self.path = Path(config.path)
        self._tokenizer = self._build_tokenizer(
            config.tokenizer, config.tiktoken_encoding
        )
        self._native_ext = None
        self._newline_tokens: Optional[torch.Tensor] = None
        self._tokens = self._load_tokens()
        self._train_tokens, self._val_tokens = self._split_tokens(self._tokens)

    @property
    def ready(self) -> bool:
        return int(self._tokens.numel()) > 1

    def _split_tokens(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.numel() == 0:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty
        train_frac = max(0.0, min(1.0, float(self.config.train_fraction or 0.0)))
        val_frac = max(0.0, min(1.0, float(self.config.val_fraction or 0.0)))
        # Normalize if sum > 1
        total = train_frac + val_frac
        if total <= 0:
            train_frac = 1.0
            val_frac = 0.0
        elif total > 1.0:
            train_frac = train_frac / total
            val_frac = val_frac / total

        n_tokens = int(tokens.numel())
        split_idx = int(n_tokens * train_frac)
        split_idx = max(1, min(n_tokens, split_idx))
        train_tokens = tokens[:split_idx]
        val_tokens = tokens[split_idx:] if val_frac > 0 else tokens.new_empty(0)
        return train_tokens, val_tokens

    def _build_tokenizer(
        self, name: str, encoding_name: str = "gpt2"
    ) -> TokenizerAdapter:
        lowered = (name or "byte").strip().lower()
        if lowered == "whitespace":
            return WhitespaceHashTokenizer()
        if lowered in ("tiktoken", "bpe", "gpt2", "cl100k", "cl100k_base"):
            return TiktokenAdapter(encoding_name=encoding_name or "gpt2")
        return ByteTokenizer()

    def _detect_format(self) -> str:
        fmt = (self.config.fmt or "auto").strip().lower()
        if fmt in {"txt", "jsonl"}:
            return fmt
        suffix = self.path.suffix.lower()
        if suffix == ".jsonl":
            return "jsonl"
        return "txt"

    def _tokens_from_text(self, text: str) -> torch.Tensor:
        if not text:
            return torch.empty(0, dtype=torch.long)
        if isinstance(self._tokenizer, ByteTokenizer):
            return _byte_tokenize_tensor(text, self.vocab_size)
        if isinstance(self._tokenizer, WhitespaceHashTokenizer):
            return _whitespace_hash_tokenize_tensor(text, self.vocab_size)

        encoded = self._tokenizer.encode(text, self.vocab_size)
        if not encoded:
            return torch.empty(0, dtype=torch.long)
        return torch.as_tensor(encoded, dtype=torch.long)

    def _separator_tokens(self) -> torch.Tensor:
        if self._newline_tokens is None:
            self._newline_tokens = self._tokens_from_text("\n")
        return self._newline_tokens

    def _load_text_tokens(self, text: str) -> torch.Tensor:
        return self._tokens_from_text(text[: self.config.max_chars])

    def _load_jsonl_tokens(self) -> torch.Tensor:
        token_chunks: List[torch.Tensor] = []
        chars = 0
        first_chunk = True
        separator = self._separator_tokens()

        with self.path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error("Failed to decode JSON line in corpus: %s", e)
                    raise e

                if isinstance(item, dict):
                    value = item.get(self.config.text_key)
                    text = value if isinstance(value, str) else ""
                elif isinstance(item, str):
                    text = item
                else:
                    text = ""

                if not text:
                    continue

                remaining = self.config.max_chars - chars
                if remaining <= 0:
                    break
                clipped = text[:remaining]
                chunk_tokens = self._tokens_from_text(clipped)
                if chunk_tokens.numel() == 0:
                    chars += len(clipped)
                    continue
                if not first_chunk and separator.numel() > 0:
                    token_chunks.append(separator)
                token_chunks.append(chunk_tokens)
                chars += len(clipped)
                first_chunk = False

        if not token_chunks:
            return torch.empty(0, dtype=torch.long)
        if len(token_chunks) == 1:
            return token_chunks[0]
        return torch.cat(token_chunks)

    def _load_tokens(self) -> torch.Tensor:
        if not self.path.exists() or not self.path.is_file():
            logger.warning("Corpus path not found: %s", self.path)
            return torch.empty(0, dtype=torch.long)

        # Pretokenized .npy: load directly, skip text encoding
        if self.path.suffix == ".npy":
            tokens_np = np.load(str(self.path), mmap_mode="r")
            tokens_np = np.asarray(tokens_np)
            if tokens_np.dtype != np.int64:
                tokens_np = tokens_np.astype(np.int64, copy=False)
            if self.vocab_size > 0:
                tokens_np = np.remainder(tokens_np, self.vocab_size).astype(
                    np.int64, copy=False
                )
            return torch.from_numpy(np.ascontiguousarray(tokens_np))

        fmt = self._detect_format()

        try:
            if fmt == "jsonl":
                return self._load_jsonl_tokens()

            with self.path.open("r", encoding="utf-8", errors="ignore") as handle:
                return self._load_text_tokens(handle.read(self.config.max_chars))
        except Exception as exc:
            logger.error("Corpus load failed from %s: %s", self.path, exc)
            raise exc

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
        generator: torch.Generator,
        device: torch.device,
        split: str = "train",
        timer: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[torch.Tensor]:
        if not self.ready or seq_len <= 0 or batch_size <= 0:
            return None

        if str(split).lower() == "val":
            tokens = self._val_tokens
        else:
            tokens = self._train_tokens

        if tokens is None or tokens.numel() == 0:
            return None

        max_start = int(tokens.numel()) - seq_len - 1
        if max_start < 0:
            return None

        sample_t0 = time.perf_counter()
        starts = torch.randint(
            0,
            max_start + 1,
            (batch_size,),
            generator=generator,
            device="cpu",
        )
        if timer is not None:
            timer("start_index_sampling_ms", (time.perf_counter() - sample_t0) * 1000.0)

        if self._native_ext is None:
            self._native_ext = load_data_native()
        gather_t0 = time.perf_counter()
        batch = self._native_ext.gather_token_batch(tokens, starts, seq_len)
        if timer is not None:
            timer("native_gather_ms", (time.perf_counter() - gather_t0) * 1000.0)
        if device.type == "cpu":
            return batch
        pin_t0 = time.perf_counter()
        batch = batch.pin_memory()
        if timer is not None:
            timer("pin_memory_ms", (time.perf_counter() - pin_t0) * 1000.0)
        h2d_t0 = time.perf_counter()
        batch = batch.to(device, non_blocking=True)
        if timer is not None:
            timer("h2d_copy_ms", (time.perf_counter() - h2d_t0) * 1000.0)
        return batch
