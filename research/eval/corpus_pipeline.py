"""Shared text corpus download, caching, tokenization, and batching."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .utils import make_batches, move_batches_to_device, tokenize_file


@dataclass(frozen=True, slots=True)
class TextSplitSpec:
    split: str
    filename: str
    max_chars: int


_BATCH_CACHE_MAX_ENTRIES = 16
_batch_cache: "collections.OrderedDict[tuple, tuple[List[torch.Tensor], int]]" = (
    collections.OrderedDict()
)
_TOKEN_CACHE_MAX_ENTRIES = 8
_token_cache: "collections.OrderedDict[tuple, np.ndarray]" = collections.OrderedDict()


def _trim_text_chunks(chunks: Iterable[str], max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    parts: List[str] = []
    total = 0
    saw_chunk = False
    for chunk in chunks:
        if not chunk:
            continue
        saw_chunk = True
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            parts.append(chunk[:remaining])
            break
        parts.append(chunk)
        total += len(chunk)
    if not saw_chunk:
        return ""
    return "".join(parts)


def cache_hf_text_splits(
    *,
    cache_dir: Path,
    dataset_name: str,
    split_specs: Iterable[TextSplitSpec],
    config_name: str | None = None,
    trust_remote_code: bool = True,
    streaming: bool = False,
    text_field: str = "text",
    sample_to_text: Optional[Callable[[dict], str]] = None,
    load_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_specs = list(split_specs)
    paths = {spec.split: cache_dir / spec.filename for spec in split_specs}
    if all(path.exists() for path in paths.values()):
        return paths

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace `datasets` package required for corpus evaluation. "
            "Install with: pip install datasets"
        ) from exc

    load_kwargs = dict(load_kwargs or {})
    shared_dataset = None
    if not streaming:
        shared_dataset = load_dataset(
            dataset_name,
            config_name,
            **load_kwargs,
        )

    for spec in split_specs:
        path = paths[spec.split]
        if path.exists():
            continue
        if streaming:
            dataset_split = load_dataset(
                dataset_name,
                config_name,
                split=spec.split,
                streaming=True,
                **load_kwargs,
            )
        else:
            dataset_split = shared_dataset[spec.split]

        if sample_to_text is None:
            chunks = (
                (sample.get(text_field, "") if isinstance(sample, dict) else "")
                for sample in dataset_split
            )
        else:
            chunks = (sample_to_text(sample) for sample in dataset_split)
        text = _trim_text_chunks(
            (chunk for chunk in chunks if chunk.strip()), spec.max_chars
        )
        path.write_text(text, encoding="utf-8")

    return paths


def _cache_key(
    namespace: str,
    path: Path,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    split_tag: str,
    seed: int,
) -> tuple:
    return (
        namespace,
        str(path),
        int(path.stat().st_mtime_ns) if path.exists() else 0,
        vocab_size,
        seq_len,
        batch_size,
        n_batches,
        split_tag,
        seed,
    )


def _get_cached_batch_entry(
    cache_key: tuple, device: str | torch.device
) -> Optional[tuple[List[torch.Tensor], int]]:
    cached = _batch_cache.get(cache_key)
    if cached is None:
        return None
    _batch_cache.move_to_end(cache_key)
    batches, token_count = cached
    return move_batches_to_device(batches, device), token_count


def _get_cached_batches(
    cache_key: tuple, device: str | torch.device
) -> Optional[List[torch.Tensor]]:
    cached = _get_cached_batch_entry(cache_key, device)
    if cached is None:
        return None
    return cached[0]


def _put_cached_batches(
    cache_key: tuple, batches: List[torch.Tensor], token_count: int = -1
) -> None:
    _batch_cache[cache_key] = (batches, int(token_count))
    while len(_batch_cache) > _BATCH_CACHE_MAX_ENTRIES:
        _batch_cache.popitem(last=False)


def _token_cache_key(path: Path, vocab_size: int) -> tuple:
    return (
        str(path),
        int(path.stat().st_mtime_ns) if path.exists() else 0,
        int(vocab_size),
    )


def _get_cached_tokens(path: Path, vocab_size: int) -> np.ndarray:
    cache_key = _token_cache_key(path, vocab_size)
    tokens = _token_cache.get(cache_key)
    if tokens is not None:
        _token_cache.move_to_end(cache_key)
        return tokens
    tokens = tokenize_file(path, vocab_size)
    _token_cache[cache_key] = tokens
    _token_cache.move_to_end(cache_key)
    while len(_token_cache) > _TOKEN_CACHE_MAX_ENTRIES:
        _token_cache.popitem(last=False)
    return tokens


def prepare_text_split_batches(
    *,
    namespace: str,
    train_path: Path,
    val_path: Path,
    vocab_size: int,
    seq_len: int,
    train_batch_size: int,
    eval_batch_size: int,
    n_train_batches: int,
    n_eval_batches: int,
    device: str | torch.device,
    train_seed: int = 42,
    val_seed: int = 123,
) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], int, int]:
    train_key = _cache_key(
        namespace,
        train_path,
        vocab_size,
        seq_len,
        train_batch_size,
        n_train_batches,
        "train",
        train_seed,
    )
    val_key = _cache_key(
        namespace,
        val_path,
        vocab_size,
        seq_len,
        eval_batch_size,
        n_eval_batches,
        "validation",
        val_seed,
    )
    train_cached = _get_cached_batch_entry(train_key, device)
    val_cached = _get_cached_batch_entry(val_key, device)
    train = train_cached[0] if train_cached is not None else None
    val = val_cached[0] if val_cached is not None else None
    if train_cached is not None and val_cached is not None:
        train, train_token_count = train_cached
        val, val_token_count = val_cached
        return train, val, train_token_count, val_token_count

    train_tokens = _get_cached_tokens(train_path, vocab_size)
    val_tokens = _get_cached_tokens(val_path, vocab_size)
    if len(train_tokens) < seq_len + 1 or len(val_tokens) < seq_len + 1:
        return None, None, len(train_tokens), len(val_tokens)

    if train is None:
        train_cpu = make_batches(
            train_tokens,
            train_batch_size,
            seq_len,
            n_train_batches,
            "cpu",
            seed=train_seed,
        )
        if train_cpu:
            _put_cached_batches(train_key, train_cpu, len(train_tokens))
            train = move_batches_to_device(train_cpu, device)
    if val is None:
        val_cpu = make_batches(
            val_tokens,
            eval_batch_size,
            seq_len,
            n_eval_batches,
            "cpu",
            seed=val_seed,
        )
        if val_cpu:
            _put_cached_batches(val_key, val_cpu, len(val_tokens))
            val = move_batches_to_device(val_cpu, device)

    return train, val, len(train_tokens), len(val_tokens)


def prepare_text_corpus_split_batches(
    *,
    path: Path,
    namespace: str,
    vocab_size: int,
    seq_len: int,
    train_batch_size: int,
    eval_batch_size: int,
    n_train_batches: int,
    n_eval_batches: int,
    device: str | torch.device,
    train_fraction: float = 0.9,
    train_seed: int = 42,
    val_seed: int = 123,
) -> Tuple[Optional[List[torch.Tensor]], Optional[List[torch.Tensor]], int]:
    train_key = _cache_key(
        namespace,
        path,
        vocab_size,
        seq_len,
        train_batch_size,
        n_train_batches,
        "train_fractional",
        train_seed,
    ) + (float(train_fraction),)
    val_key = _cache_key(
        namespace,
        path,
        vocab_size,
        seq_len,
        eval_batch_size,
        n_eval_batches,
        "validation_fractional",
        val_seed,
    ) + (float(train_fraction),)
    train_cached = _get_cached_batch_entry(train_key, device)
    val_cached = _get_cached_batch_entry(val_key, device)
    train = train_cached[0] if train_cached is not None else None
    val = val_cached[0] if val_cached is not None else None
    if train_cached is not None and val_cached is not None:
        train, token_count = train_cached
        val, _ = val_cached
        return train, val, token_count

    tokens = _get_cached_tokens(path, vocab_size)
    if len(tokens) < seq_len + 1:
        return None, None, len(tokens)

    train_tokens, val_tokens = split_token_array(tokens, train_fraction=train_fraction)
    if len(train_tokens) < seq_len + 1 or len(val_tokens) < seq_len + 1:
        return None, None, len(tokens)

    if train is None:
        train_cpu = make_batches(
            train_tokens,
            train_batch_size,
            seq_len,
            n_train_batches,
            "cpu",
            seed=train_seed,
        )
        if train_cpu:
            _put_cached_batches(train_key, train_cpu, len(tokens))
            train = move_batches_to_device(train_cpu, device)
    if val is None:
        val_cpu = make_batches(
            val_tokens,
            eval_batch_size,
            seq_len,
            n_eval_batches,
            "cpu",
            seed=val_seed,
        )
        if val_cpu:
            _put_cached_batches(val_key, val_cpu, len(tokens))
            val = move_batches_to_device(val_cpu, device)

    return train, val, len(tokens)


def split_token_array(
    tokens: np.ndarray,
    *,
    train_fraction: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    if len(tokens) <= 0 or train_fraction <= 0.0:
        split_idx = 0
    elif train_fraction >= 1.0:
        split_idx = int(len(tokens))
    else:
        split_idx = int(len(tokens) * train_fraction)
    return tokens[:split_idx], tokens[split_idx:]
