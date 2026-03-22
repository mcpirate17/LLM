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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TokenizerAdapter(Protocol):
    """Tokenizer adapter interface for corpus-mode training."""

    def encode(self, text: str, vocab_size: int) -> List[int]:
        """Encode text to token IDs bounded to vocab_size."""


class ByteTokenizer:
    """Deterministic byte tokenizer with modulo vocab projection."""

    def encode(self, text: str, vocab_size: int) -> List[int]:
        if vocab_size <= 0:
            return []
        return [b % vocab_size for b in text.encode("utf-8", errors="ignore")]


class WhitespaceHashTokenizer:
    """Simple hashed-whitespace tokenizer for word-like segmentation."""

    def encode(self, text: str, vocab_size: int) -> List[int]:
        if vocab_size <= 0:
            return []
        ids: List[int] = []
        for token in text.split():
            token_id = abs(hash(token)) % vocab_size
            ids.append(token_id)
        return ids


class TiktokenAdapter:
    """Production-grade BPE tokenizer via tiktoken (GPT-2 vocabulary).

    Uses the GPT-2 BPE encoding (50,257 subword tokens). Token IDs are
    projected into [0, vocab_size) via modulo so the model's embedding
    layer does not need to change. This preserves architecture fingerprints
    while giving proper subword segmentation.
    """

    def __init__(self, encoding_name: str = "gpt2"):
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding_name)
        self.native_vocab_size = self._enc.n_vocab

    def encode(self, text: str, vocab_size: int) -> List[int]:
        ids = self._enc.encode(text, allowed_special=set())
        if vocab_size > 0 and vocab_size < self.native_vocab_size:
            return [t % vocab_size for t in ids]
        return ids


@dataclass
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

    def __init__(self, config: CorpusConfig, vocab_size: int):
        self.config = config
        self.vocab_size = int(vocab_size)
        self.path = Path(config.path)
        self._tokenizer = self._build_tokenizer(
            config.tokenizer, config.tiktoken_encoding
        )
        self._tokens = self._load_tokens()
        self._train_tokens, self._val_tokens = self._split_tokens(self._tokens)

    @property
    def token_count(self) -> int:
        return len(self._tokens)

    @property
    def ready(self) -> bool:
        return len(self._tokens) > 1

    def _split_tokens(self, tokens: List[int]) -> tuple[np.ndarray, np.ndarray]:
        if not tokens:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        arr = np.array(tokens, dtype=np.int64)
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

        split_idx = int(len(arr) * train_frac)
        split_idx = max(1, min(len(arr), split_idx))
        train_tokens = arr[:split_idx]
        val_tokens = arr[split_idx:] if val_frac > 0 else np.array([], dtype=np.int64)
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

    def _load_tokens(self) -> List[int]:
        if not self.path.exists() or not self.path.is_file():
            logger.warning("Corpus path not found: %s", self.path)
            return []

        # Pretokenized .npy: load directly, skip text encoding
        if self.path.suffix == ".npy":
            import numpy as np

            tokens = np.load(str(self.path)).tolist()
            # Modulo vocab_size for safety
            return [int(t) % self.vocab_size for t in tokens]

        fmt = self._detect_format()
        text_chunks: List[str] = []
        chars = 0

        try:
            if fmt == "jsonl":
                with self.path.open("r", encoding="utf-8") as handle:
                    for raw in handle:
                        line = raw.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue

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
                        text_chunks.append(clipped)
                        chars += len(clipped)
            else:
                text = self.path.read_text(encoding="utf-8", errors="ignore")
                text_chunks.append(text[: self.config.max_chars])

            joined = "\n".join(text_chunks)
            tokens = self._tokenizer.encode(joined, self.vocab_size)
            return [int(t) for t in tokens]
        except Exception as exc:
            logger.warning("Corpus load failed from %s: %s", self.path, exc)
            return []

    def sample_batch(
        self,
        batch_size: int,
        seq_len: int,
        generator: torch.Generator,
        device: torch.device,
        split: str = "train",
    ) -> Optional[torch.Tensor]:
        if not self.ready or seq_len <= 0 or batch_size <= 0:
            return None

        if str(split).lower() == "val":
            tokens = self._val_tokens
        else:
            tokens = self._train_tokens

        if tokens is None or len(tokens) == 0:
            return None

        max_start = len(tokens) - seq_len - 1
        if max_start < 0:
            return None

        starts = torch.randint(
            0,
            max_start + 1,
            (batch_size,),
            generator=generator,
            device="cpu",
        )

        # Vectorized batch extraction via NumPy advanced indexing.
        # Build a (batch_size, seq_len) index array and gather in one shot —
        # no Python loop, no intermediate list of lists.
        starts_np = starts.numpy()
        offsets = np.arange(seq_len, dtype=np.int64)
        indices = starts_np[:, np.newaxis] + offsets  # (batch_size, seq_len)
        batch_np = tokens[indices]  # single contiguous gather

        batch = torch.from_numpy(batch_np).pin_memory()
        return batch.to(device, non_blocking=True)
