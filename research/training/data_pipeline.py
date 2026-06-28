"""
Corpus data pipeline for training-token batch generation.

Hot-path requirements:
- zero-copy ingest where possible (.npy mmap; native single-pass JSONL parse)
- one allocation per token buffer; no per-record tensors
- one pinned host buffer reused for the entire training loop
- deterministic batch sampling using a caller-provided torch.Generator
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np
import torch

from research.synthesis._json_compat import loads_json

from ._native import load_training_native

logger = logging.getLogger(__name__)

# Tokenized corpora are large (WikiText-103 int64 ≈ 300 MB) and immutable for
# a given (path, mtime, vocab, tokenizer, ...) tuple. Cache the tensor at
# module level so every CorpusTokenBatcher with matching key shares the same
# underlying buffer. Without this, each new ExperimentRunner re-tokenizes —
# observed leak: ~870 MB/iter, OOM-killing the box at iter ~140.
_TOKEN_CACHE_LOCK = threading.Lock()
_TOKEN_CACHE_MAX_ENTRIES = 4
_TOKEN_CACHE: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()


def _token_cache_get(key: tuple) -> Optional[torch.Tensor]:
    with _TOKEN_CACHE_LOCK:
        tensor = _TOKEN_CACHE.get(key)
        if tensor is not None:
            _TOKEN_CACHE.move_to_end(key)
        return tensor


def _token_cache_put(key: tuple, tensor: torch.Tensor) -> None:
    with _TOKEN_CACHE_LOCK:
        _TOKEN_CACHE[key] = tensor
        _TOKEN_CACHE.move_to_end(key)
        while len(_TOKEN_CACHE) > _TOKEN_CACHE_MAX_ENTRIES:
            _TOKEN_CACHE.popitem(last=False)


class TokenizerAdapter(Protocol):
    """Tokenizer that produces a 1-D int64 token tensor in one shot."""

    def encode_to_tensor(self, text: str, vocab_size: int) -> torch.Tensor:
        """Encode text to an int64 token tensor projected into [0, vocab_size)."""


class ByteTokenizer:
    """Deterministic byte tokenizer (modulo vocab projection)."""

    __slots__ = ()

    def encode_to_tensor(self, text: str, vocab_size: int) -> torch.Tensor:
        if vocab_size <= 0 or not text:
            return torch.empty(0, dtype=torch.long)
        return load_training_native().byte_tokenize_utf8(text, int(vocab_size))


class WhitespaceHashTokenizer:
    """FNV-1a hashed whitespace tokenizer."""

    __slots__ = ()

    def encode_to_tensor(self, text: str, vocab_size: int) -> torch.Tensor:
        if vocab_size <= 0 or not text:
            return torch.empty(0, dtype=torch.long)
        return load_training_native().whitespace_hash_tokenize(text, int(vocab_size))


class TiktokenAdapter:
    """GPT-2 BPE via tiktoken, projected to vocab_size with modulo."""

    __slots__ = ("_enc", "native_vocab_size")

    def __init__(self, encoding_name: str = "gpt2"):
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding_name)
        self.native_vocab_size = self._enc.n_vocab

    def encode_to_tensor(self, text: str, vocab_size: int) -> torch.Tensor:
        ids = self._enc.encode(text, allowed_special=set())
        if not ids:
            return torch.empty(0, dtype=torch.long)
        tensor = torch.as_tensor(ids, dtype=torch.long)
        if vocab_size > 0 and vocab_size < self.native_vocab_size:
            load_training_native().project_int64_modulo_inplace(tensor, int(vocab_size))
        return tensor


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
        "_pinned_batch",
        "_pinned_shape",
    )

    def __init__(self, config: CorpusConfig, vocab_size: int):
        self.config = config
        self.vocab_size = int(vocab_size)
        self.path = Path(config.path)
        self._tokenizer = self._build_tokenizer(
            config.tokenizer, config.tiktoken_encoding
        )
        self._native_ext = load_training_native()
        self._pinned_batch: Optional[torch.Tensor] = None
        self._pinned_shape: Optional[tuple[int, int]] = None
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

    def _load_text_tokens(self) -> torch.Tensor:
        max_chars = max(0, int(self.config.max_chars))
        if max_chars == 0:
            return torch.empty(0, dtype=torch.long)
        if isinstance(self._tokenizer, ByteTokenizer):
            return self._native_ext.byte_tokenize_file_prefix_utf8(
                str(self.path),
                int(self.vocab_size),
                max_chars,
            )
        with self.path.open("r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read(max_chars)
        return self._tokenizer.encode_to_tensor(text, self.vocab_size)

    def _load_jsonl_tokens(self) -> torch.Tensor:
        max_chars = max(0, int(self.config.max_chars))
        if max_chars == 0 or self.vocab_size <= 0:
            return torch.empty(0, dtype=torch.long)

        # Single-pass native path for byte tokenization: no Python json.loads,
        # no per-record tensors, one allocation.
        if isinstance(self._tokenizer, ByteTokenizer):
            return self._native_ext.jsonl_byte_tokenize_file(
                str(self.path),
                str(self.config.text_key),
                int(self.vocab_size),
                max_chars,
            )

        # Non-byte tokenizers still need string decoding. Parse with the
        # shared fast-JSON helper, accumulate strings, then tokenize in one
        # shot at the end.
        text_key = str(self.config.text_key)
        chars = 0
        chunks: list[str] = []
        with self.path.open("rb") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                try:
                    item = loads_json(raw)
                except Exception:
                    logger.error("Failed to decode JSON line in corpus: %s", self.path)
                    raise
                if isinstance(item, dict):
                    value = item.get(text_key)
                    text = value if isinstance(value, str) else ""
                elif isinstance(item, str):
                    text = item
                else:
                    text = ""
                if not text:
                    continue
                remaining = max_chars - chars
                if remaining <= 0:
                    break
                clipped = text if len(text) <= remaining else text[:remaining]
                if chunks:
                    chunks.append("\n")
                    chars += 1
                chunks.append(clipped)
                chars += len(clipped)

        if not chunks:
            return torch.empty(0, dtype=torch.long)
        return self._tokenizer.encode_to_tensor("".join(chunks), self.vocab_size)

    def _load_tokens(self) -> torch.Tensor:
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Corpus path not found: {self.path}. Callers that tolerate a "
                "missing corpus must check the path before constructing "
                "CorpusTokenBatcher."
            )

        try:
            stat = self.path.stat()
            cache_key = (
                str(self.path.resolve()),
                int(stat.st_mtime_ns),
                int(stat.st_size),
                int(self.vocab_size),
                type(self._tokenizer).__name__,
                int(self.config.max_chars),
                str(self.config.fmt or "auto"),
                str(self.config.text_key or ""),
                str(self.config.tiktoken_encoding or ""),
            )
        except OSError:
            cache_key = None

        if cache_key is not None:
            cached = _token_cache_get(cache_key)
            if cached is not None:
                return cached

        tokens = self._load_tokens_uncached()

        if (
            cache_key is not None
            and isinstance(tokens, torch.Tensor)
            and tokens.numel() > 0
        ):
            _token_cache_put(cache_key, tokens)
        return tokens

    def _load_tokens_uncached(self) -> torch.Tensor:
        # Pretokenized .npy: keep the mmap zero-copy when dtype matches and
        # values already fit the vocab range. Otherwise materialize once and
        # project in-place natively — never re-project per batch.
        if self.path.suffix == ".npy":
            tokens_np = np.load(str(self.path), mmap_mode="r")
            already_int64 = tokens_np.dtype == np.int64
            needs_modulo = self.vocab_size > 0
            if already_int64 and (
                not needs_modulo
                or (
                    int(tokens_np.min(initial=0)) >= 0
                    and int(tokens_np.max(initial=0)) < self.vocab_size
                )
            ):
                # mmap is read-only; we never write through this view, so the
                # PyTorch read-only-array warning is benign — silence it locally.
                import warnings as _w

                with _w.catch_warnings():
                    _w.filterwarnings(
                        "ignore",
                        message="The given NumPy array is not writable",
                        category=UserWarning,
                    )
                    return torch.from_numpy(tokens_np)
            if already_int64:
                tokens = torch.from_numpy(np.asarray(tokens_np))
            else:
                tokens = torch.from_numpy(np.asarray(tokens_np, dtype=np.int64))
            if needs_modulo:
                self._native_ext.project_int64_modulo_inplace(tokens, self.vocab_size)
            return tokens

        fmt = self._detect_format()

        try:
            if fmt == "jsonl":
                return self._load_jsonl_tokens()
            return self._load_text_tokens()
        except Exception:
            logger.error("Corpus load failed from %s", self.path)
            raise

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
        tokens = self._val_tokens if str(split).lower() == "val" else self._train_tokens
        if tokens is None or tokens.numel() == 0:
            return None
        max_start = int(tokens.numel()) - seq_len - 1
        if max_start < 0:
            return None

        t0 = time.perf_counter() if timer is not None else 0.0
        starts = torch.randint(
            0,
            int(tokens.numel()) - seq_len,
            (batch_size,),
            generator=generator,
            device="cpu",
        )
        if timer is not None:
            timer("start_index_sampling_ms", (time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()

        if device.type != "cuda":
            batch = self._native_ext.gather_token_batch(tokens, starts, seq_len)
            if timer is not None:
                timer("native_gather_ms", (time.perf_counter() - t0) * 1000.0)
            return batch

        out = self._cuda_pinned_buffer(batch_size, seq_len)
        self._native_ext.gather_token_batch_into(tokens, starts, seq_len, out)
        if timer is not None:
            timer("native_gather_ms", (time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
        batch = out.to(device, non_blocking=True)
        if timer is not None:
            timer("h2d_copy_ms", (time.perf_counter() - t0) * 1000.0)
        return batch

    def _cuda_pinned_buffer(self, batch_size: int, seq_len: int) -> torch.Tensor:
        shape = (batch_size, seq_len)
        if self._pinned_batch is None or self._pinned_shape != shape:
            self._pinned_batch = torch.empty(shape, dtype=torch.long, pin_memory=True)
            self._pinned_shape = shape
        return self._pinned_batch
