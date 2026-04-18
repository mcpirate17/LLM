"""Streaming FineWeb-Edu + UltraChat data iterator with background prefetch.

Uses a background thread for tokenization + batch assembly,
pin_memory for async H2D transfer, and a numpy ring buffer.
"""

from __future__ import annotations

import logging
import queue
import threading

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Batch size for encode_ordinary_batch (2.3x faster than per-text encode)
_TEXT_BATCH_SIZE = 64


def get_data_iterator(
    batch_size: int, seq_len: int, device: str, tokenizer_name: str = "gpt2"
):
    """Streaming FineWeb-Edu + UltraChat with background prefetch.

    Returns (get_batch_fn, vocab_size). Call get_batch_fn.stop() to shut down.
    """
    from datasets import load_dataset
    import tiktoken

    enc = tiktoken.get_encoding(tokenizer_name)
    vocab_size = enc.n_vocab
    logger.info(f"Tokenizer: {tokenizer_name}, vocab_size={vocab_size}")
    tokens_per_batch = batch_size * (seq_len + 1)

    # ── Streaming dataset iterators ──
    logger.info("Loading FineWeb-Edu (streaming)...")
    fw_ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    logger.info("Loading UltraChat (streaming)...")
    uc_ds = load_dataset(
        "stingning/ultrachat",
        split="train",
        streaming=True,
    )

    fw_iter = iter(fw_ds)
    uc_iter = iter(uc_ds)

    # ── Ring buffer (numpy for O(1) slicing) ──
    buf_capacity = tokens_per_batch * 16
    ring = np.empty(buf_capacity, dtype=np.int32)
    ring_len = 0
    step_counter = 0

    def _next_text():
        nonlocal fw_iter, uc_iter, step_counter
        step_counter += 1
        use_ultrachat = step_counter % 10 < 3

        if use_ultrachat:
            try:
                example = next(uc_iter)
                messages = example.get("data") or example.get("messages") or []
                if isinstance(messages, list):
                    return "\n".join(str(m) for m in messages)
                return str(messages)
            except StopIteration:
                uc_iter = iter(
                    load_dataset(
                        "stingning/ultrachat",
                        split="train",
                        streaming=True,
                    )
                )
                return _next_text()
        else:
            try:
                return next(fw_iter).get("text", "")
            except StopIteration:
                fw_iter = iter(
                    load_dataset(
                        "HuggingFaceFW/fineweb-edu",
                        name="sample-10BT",
                        split="train",
                        streaming=True,
                    )
                )
                return _next_text()

    def _fill_ring():
        nonlocal ring, ring_len, buf_capacity
        target = tokens_per_batch * 8
        while ring_len < target:
            texts = []
            while (
                len(texts) < _TEXT_BATCH_SIZE
                and ring_len + len(texts) * 200 < target * 2
            ):
                text = _next_text()
                if len(text) >= 50:
                    texts.append(text)
                if not texts:
                    break
            if not texts:
                continue
            all_token_lists = enc.encode_ordinary_batch(texts)
            for tokens in all_token_lists:
                n_tok = len(tokens)
                if n_tok == 0:
                    continue
                if ring_len + n_tok > buf_capacity:
                    buf_capacity = max(buf_capacity * 2, ring_len + n_tok)
                    new_ring = np.empty(buf_capacity, dtype=np.int32)
                    new_ring[:ring_len] = ring[:ring_len]
                    ring = new_ring
                ring[ring_len : ring_len + n_tok] = tokens
                ring_len += n_tok

    def _extract_batch_np():
        nonlocal ring_len
        if ring_len < tokens_per_batch:
            _fill_ring()
        batch = ring[:tokens_per_batch].copy()
        remaining = ring_len - tokens_per_batch
        if remaining > 0:
            ring[:remaining] = ring[tokens_per_batch : tokens_per_batch + remaining]
        ring_len = remaining
        return batch

    # Initial fill
    _fill_ring()
    logger.info(f"Buffer ready: {ring_len:,} tokens (70% FineWeb-Edu + 30% UltraChat)")

    # ── Prefetch queue ──
    prefetch_q: queue.Queue = queue.Queue(maxsize=8)
    _stop_event = threading.Event()

    def _prefetch_worker():
        use_cuda = device.startswith("cuda")
        while not _stop_event.is_set():
            try:
                batch_np = _extract_batch_np()
                t = torch.from_numpy(batch_np).long().reshape(batch_size, seq_len + 1)
                if use_cuda:
                    t = t.pin_memory()
                prefetch_q.put(t, timeout=5.0)
            except queue.Full:
                continue
            except Exception as e:
                logger.error(f"Prefetch worker error: {e}")
                break

    worker = threading.Thread(target=_prefetch_worker, daemon=True)
    worker.start()

    def _get_batch():
        t = prefetch_q.get()
        non_blocking = device.startswith("cuda")
        t = t.to(device, non_blocking=non_blocking)
        return t[:, :seq_len], t[:, 1 : seq_len + 1]

    _get_batch.stop = lambda: _stop_event.set()
    return _get_batch, vocab_size
