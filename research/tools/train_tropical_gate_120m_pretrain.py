"""Restartable 120M pretraining harness for the tropical gated fab winner.

This is intentionally a focused orchestration CLI.  The hot work stays in
PyTorch/NumPy/tiktoken/native repo helpers, while this module owns restart,
checkpoint, data-mix, schedule, logging, and eval bookkeeping.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import os
import random
import signal
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from component_fab.harness.tiny_lm import count_trainable_params
from research.defaults import PROJECT_ROOT, VOCAB_SIZE
from research.tools.scaling_blimp_study import (
    PARAM_SIZING,
    _build_tinylm,
    _saved_winner_factory,
)


DEFAULT_PROPOSAL_ID = "improve_tropical_gate_block_gated_parallel_84f0ccd08a"
DEFAULT_RUN_PREFIX = "120m_tropical_gate_pretrain"
SMOKE_SIZING = {"dim": 32, "n_blocks": 1}
LOG = logging.getLogger("train_tropical_gate_120m_pretrain")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=_json_safe, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _run_text(cmd: list[str], *, max_chars: int = 12_000) -> str:
    try:
        out = subprocess.check_output(
            cmd,
            cwd=PROJECT_ROOT,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
    except Exception as exc:  # noqa: BLE001
        return f"unavailable:{type(exc).__name__}:{exc}"
    return out[-max_chars:]


def git_snapshot() -> dict[str, Any]:
    return {
        "commit": _run_text(["git", "rev-parse", "HEAD"], max_chars=80).strip(),
        "branch": _run_text(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], max_chars=120
        ).strip(),
        "dirty_status_short": _run_text(["git", "status", "--short"], max_chars=24_000),
    }


def rng_state_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng_state(payload: dict[str, Any] | None) -> None:
    if not isinstance(payload, dict):
        return
    if payload.get("python") is not None:
        random.setstate(payload["python"])
    if payload.get("numpy") is not None:
        np.random.set_state(payload["numpy"])
    if payload.get("torch") is not None:
        torch_state = payload["torch"]
        if torch.is_tensor(torch_state):
            torch_state = torch_state.detach().cpu().to(torch.uint8)
        torch.set_rng_state(torch_state)
    if payload.get("cuda") is not None and torch.cuda.is_available():
        cuda_states = []
        for state in payload["cuda"]:
            if torch.is_tensor(state):
                state = state.detach().cpu().to(torch.uint8)
            cuda_states.append(state)
        torch.cuda.set_rng_state_all(cuda_states)


def _cpu_rng_state(state: Any) -> Any:
    if torch.is_tensor(state):
        return state.detach().cpu().to(torch.uint8)
    return state


class WarmupCosineSchedule:
    """Linear warmup followed by cosine decay, keyed by completed steps."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        learning_rate: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
        completed_steps: int = 0,
    ) -> None:
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        self.min_lr = float(min_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.completed_steps = int(completed_steps)
        self.apply(self.completed_steps)

    def lr_at(self, completed_steps: int) -> float:
        s = int(completed_steps)
        if self.total_steps <= 0:
            return self.min_lr
        if self.warmup_steps > 0 and s < self.warmup_steps:
            return self.learning_rate * float(s + 1) / float(self.warmup_steps)
        denom = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (s - self.warmup_steps) / denom))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.learning_rate - self.min_lr) * cosine

    def apply(self, completed_steps: int) -> float:
        self.completed_steps = int(completed_steps)
        lr = self.lr_at(self.completed_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "next_lr": self.lr_at(self.completed_steps),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.learning_rate = float(state["learning_rate"])
        self.min_lr = float(state["min_lr"])
        self.warmup_steps = int(state["warmup_steps"])
        self.total_steps = int(state["total_steps"])
        self.apply(int(state.get("completed_steps", 0)))


class TokenSource:
    name: str

    @property
    def n_tokens(self) -> int | None:
        raise NotImplementedError

    def next_batch(
        self, *, batch_size: int, seq_len: int, vocab_size: int, device: torch.device
    ) -> torch.Tensor:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError

    def coverage(self, seq_len: int) -> dict[str, Any]:
        n = self.n_tokens
        return {
            "name": self.name,
            "tokens_available": n,
            "tokens_emitted": int(getattr(self, "tokens_emitted", 0)),
            "epoch_equivalent": (
                round(float(getattr(self, "tokens_emitted", 0)) / float(n), 6)
                if n
                else None
            ),
        }


class MemmapWindowSource(TokenSource):
    def __init__(
        self, name: str, path: Path, *, seed: int, split: str = "train"
    ) -> None:
        self.name = name
        self.path = path
        self.split = split
        self.arr = np.load(str(path), mmap_mode="r")
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.tokens_emitted = 0

    @property
    def n_tokens(self) -> int:
        return int(self.arr.shape[0])

    def next_batch(
        self, *, batch_size: int, seq_len: int, vocab_size: int, device: torch.device
    ) -> torch.Tensor:
        max_start = self.n_tokens - int(seq_len) - 1
        if max_start <= 0:
            raise ValueError(f"{self.name} too small for seq_len={seq_len}")
        starts = torch.randint(
            0, max_start, (int(batch_size),), generator=self.generator
        )
        rows = [
            np.asarray(self.arr[int(s) : int(s) + int(seq_len) + 1]) for s in starts
        ]
        batch = torch.as_tensor(np.stack(rows), dtype=torch.long)
        if vocab_size > 0:
            batch.remainder_(int(vocab_size))
        self.tokens_emitted += int(batch_size) * int(seq_len)
        return batch.to(device, non_blocking=device.type == "cuda")

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "memmap",
            "name": self.name,
            "path": str(self.path),
            "split": self.split,
            "generator_state": self.generator.get_state(),
            "tokens_emitted": self.tokens_emitted,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("generator_state") is not None:
            self.generator.set_state(_cpu_rng_state(state["generator_state"]))
        self.tokens_emitted = int(state.get("tokens_emitted", 0))


class TextTokenSource(TokenSource):
    def __init__(self, name: str, path: Path, *, vocab_size: int, seed: int) -> None:
        self.name = name
        self.path = path
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.tokens = self._load_tokens(vocab_size)
        self.tokens_emitted = 0

    def _encode(self, text: str, vocab_size: int) -> list[int]:
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            ids = enc.encode(text, allowed_special=set())
            return [int(x) % int(vocab_size) for x in ids]
        except Exception:
            raw = text.encode("utf-8", errors="ignore")
            return [int(x) % int(vocab_size) for x in raw]

    def _load_tokens(self, vocab_size: int) -> torch.Tensor:
        text = self.path.read_text(encoding="utf-8", errors="ignore")
        ids = self._encode(text, vocab_size)
        if len(ids) < 4:
            raise ValueError(f"text source {self.path} yielded too few tokens")
        return torch.as_tensor(ids, dtype=torch.long)

    @property
    def n_tokens(self) -> int:
        return int(self.tokens.numel())

    def next_batch(
        self, *, batch_size: int, seq_len: int, vocab_size: int, device: torch.device
    ) -> torch.Tensor:
        max_start = self.n_tokens - int(seq_len) - 1
        if max_start <= 0:
            raise ValueError(f"{self.name} too small for seq_len={seq_len}")
        starts = torch.randint(
            0, max_start, (int(batch_size),), generator=self.generator
        )
        offsets = torch.arange(int(seq_len) + 1, dtype=torch.long).unsqueeze(0)
        batch = self.tokens[starts.unsqueeze(1) + offsets].remainder(int(vocab_size))
        self.tokens_emitted += int(batch_size) * int(seq_len)
        return batch.to(device, non_blocking=device.type == "cuda")

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "text",
            "name": self.name,
            "path": str(self.path),
            "generator_state": self.generator.get_state(),
            "tokens_emitted": self.tokens_emitted,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("generator_state") is not None:
            self.generator.set_state(_cpu_rng_state(state["generator_state"]))
        self.tokens_emitted = int(state.get("tokens_emitted", 0))


class JsonlStreamingSource(TokenSource):
    """Restartable local JSONL token stream over sharded text corpora."""

    def __init__(
        self,
        name: str,
        files: Iterable[Path],
        *,
        vocab_size: int,
        seed: int,
        text_key: str = "text",
        shuffle_files: bool = True,
        max_text_chars: int = 65536,
    ) -> None:
        self.name = name
        self.files = [Path(p) for p in files]
        if not self.files:
            raise ValueError(f"{name}: no JSONL files")
        self.text_key = text_key
        self.vocab_size = int(vocab_size)
        self.max_text_chars = max(50, int(max_text_chars))
        self.file_order = list(range(len(self.files)))
        rng = random.Random(int(seed))
        if shuffle_files:
            rng.shuffle(self.file_order)
        self.file_pos = 0
        self.byte_offset = 0
        self.token_buffer: list[int] = []
        self.docs_seen = 0
        self.docs_truncated = 0
        self.tokens_emitted = 0
        self._encoder = None
        self._handle = None
        self._open_current()

    @property
    def n_tokens(self) -> None:
        return None

    def _encode(self, text: str) -> list[int]:
        try:
            import tiktoken

            if self._encoder is None:
                self._encoder = tiktoken.get_encoding("cl100k_base")
            return [
                int(x) % self.vocab_size
                for x in self._encoder.encode(text, allowed_special=set())
            ]
        except Exception:
            return [
                int(x) % self.vocab_size for x in text.encode("utf-8", errors="ignore")
            ]

    def _open_current(self) -> None:
        if self._handle is not None:
            self._handle.close()
        path = self.files[self.file_order[self.file_pos]]
        self._handle = path.open("rb")
        if self.byte_offset:
            self._handle.seek(int(self.byte_offset))

    def _advance_file(self) -> None:
        self.file_pos = (self.file_pos + 1) % len(self.file_order)
        self.byte_offset = 0
        self._open_current()

    def _next_line_text(self) -> str | None:
        assert self._handle is not None
        while True:
            raw = self._handle.readline()
            if not raw:
                self._advance_file()
                continue
            self.byte_offset = int(self._handle.tell())
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if isinstance(row, dict):
                value = row.get(self.text_key)
                if isinstance(value, str):
                    text = value
                else:
                    parts = [
                        row.get(key)
                        for key in (
                            "prompt",
                            "teacher_completion",
                            "completion",
                            "response",
                            "answer",
                            "input",
                            "output",
                        )
                    ]
                    text = "\n\n".join(part for part in parts if isinstance(part, str))
                    if not text:
                        text = "\n\n".join(
                            value for value in row.values() if isinstance(value, str)
                        )
            elif isinstance(row, str):
                text = row
            else:
                text = ""
            if len(text) >= 50:
                self.docs_seen += 1
                if len(text) > self.max_text_chars:
                    self.docs_truncated += 1
                    text = text[: self.max_text_chars]
                return text

    def _fill(self, needed: int) -> None:
        while len(self.token_buffer) < needed:
            text = self._next_line_text()
            if not text:
                continue
            ids = self._encode(text)
            if ids:
                self.token_buffer.extend(ids)
                self.token_buffer.append(self.vocab_size - 1)

    def next_batch(
        self, *, batch_size: int, seq_len: int, vocab_size: int, device: torch.device
    ) -> torch.Tensor:
        needed = int(batch_size) * (int(seq_len) + 1)
        self._fill(needed)
        raw = self.token_buffer[:needed]
        del self.token_buffer[:needed]
        batch = torch.as_tensor(raw, dtype=torch.long).view(
            int(batch_size), int(seq_len) + 1
        )
        if int(vocab_size) != self.vocab_size:
            batch.remainder_(int(vocab_size))
        self.tokens_emitted += int(batch_size) * int(seq_len)
        return batch.to(device, non_blocking=device.type == "cuda")

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "jsonl_stream",
            "name": self.name,
            "files": [str(p) for p in self.files],
            "file_order": self.file_order,
            "file_pos": self.file_pos,
            "byte_offset": self.byte_offset,
            "token_buffer": self.token_buffer,
            "docs_seen": self.docs_seen,
            "docs_truncated": self.docs_truncated,
            "tokens_emitted": self.tokens_emitted,
            "text_key": self.text_key,
            "max_text_chars": self.max_text_chars,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.file_order = [int(x) for x in state.get("file_order", self.file_order)]
        self.file_pos = int(state.get("file_pos", 0))
        self.byte_offset = int(state.get("byte_offset", 0))
        self.token_buffer = [int(x) for x in state.get("token_buffer", [])]
        self.docs_seen = int(state.get("docs_seen", 0))
        self.docs_truncated = int(state.get("docs_truncated", 0))
        self.max_text_chars = max(
            50, int(state.get("max_text_chars", self.max_text_chars))
        )
        self.tokens_emitted = int(state.get("tokens_emitted", 0))
        self._open_current()

    def coverage(self, seq_len: int) -> dict[str, Any]:
        path = self.files[self.file_order[self.file_pos]]
        return {
            "name": self.name,
            "tokens_available": None,
            "tokens_emitted": int(self.tokens_emitted),
            "docs_seen": int(self.docs_seen),
            "docs_truncated": int(self.docs_truncated),
            "current_file": str(path),
            "file_pos": int(self.file_pos),
            "n_files": len(self.files),
            "byte_offset": int(self.byte_offset),
            "buffered_tokens": len(self.token_buffer),
        }

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class PackedPtTokenSource(TokenSource):
    """Restartable stream over local torch chunks containing tokenized examples."""

    def __init__(
        self,
        name: str,
        files: Iterable[Path],
        *,
        vocab_size: int,
        seed: int,
        separator_token: int | None = None,
        shuffle_files: bool = True,
    ) -> None:
        self.name = name
        self.files = [Path(p) for p in files if Path(p).is_file()]
        if not self.files:
            raise ValueError(f"{name}: no packed .pt files")
        self.vocab_size = int(vocab_size)
        self.separator_token = (
            min(int(vocab_size) - 1, 50256)
            if separator_token is None
            else int(separator_token)
        )
        self.file_order = list(range(len(self.files)))
        rng = random.Random(int(seed))
        if shuffle_files:
            rng.shuffle(self.file_order)
        self.file_pos = 0
        self.record_pos = 0
        self.token_buffer: list[int] = []
        self.records_seen = 0
        self.records_skipped = 0
        self.chunks_loaded = 0
        self.chunks_missing = 0
        self.tokens_emitted = 0
        self._records: list[Any] = []
        self._load_current_chunk()

    @property
    def n_tokens(self) -> None:
        return None

    @staticmethod
    def _torch_load_chunk(path: Path) -> Any:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")
        except (FileNotFoundError, OSError):
            raise
        except Exception:
            return torch.load(path, map_location="cpu", weights_only=False)

    def _load_current_chunk(self, *, count_load: bool = True) -> None:
        last_error: BaseException | None = None
        for _ in range(len(self.file_order)):
            path = self.files[self.file_order[self.file_pos]]
            try:
                payload = self._torch_load_chunk(path)
                if isinstance(payload, dict):
                    for key in ("records", "samples", "examples", "data"):
                        value = payload.get(key)
                        if isinstance(value, list):
                            payload = value
                            break
                if not isinstance(payload, list):
                    raise ValueError(f"expected list-like .pt chunk at {path}")
            except (
                FileNotFoundError,
                OSError,
                EOFError,
                RuntimeError,
                ValueError,
            ) as exc:
                last_error = exc
                self.chunks_missing += 1
                self.file_pos = (self.file_pos + 1) % len(self.file_order)
                self.record_pos = 0
                continue
            self._records = payload
            self.record_pos = min(int(self.record_pos), len(self._records))
            if count_load:
                self.chunks_loaded += 1
            return
        raise RuntimeError(
            f"{self.name}: no readable packed .pt chunks; last_error={last_error}"
        )

    def _advance_file(self) -> None:
        self.file_pos = (self.file_pos + 1) % len(self.file_order)
        self.record_pos = 0
        self._load_current_chunk()

    def _record_ids(self, record: Any) -> list[int]:
        value = record.get("input_ids") if isinstance(record, dict) else record
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, list):
            return []
        ids: list[int] = []
        for token in value:
            try:
                token_id = int(token)
            except Exception:
                continue
            if token_id >= 0:
                ids.append(token_id % self.vocab_size)
        return ids

    def _fill(self, needed: int) -> None:
        while len(self.token_buffer) < needed:
            if self.record_pos >= len(self._records):
                self._advance_file()
                continue
            record = self._records[self.record_pos]
            self.record_pos += 1
            ids = self._record_ids(record)
            if len(ids) < 2:
                self.records_skipped += 1
                continue
            self.token_buffer.extend(ids)
            self.token_buffer.append(self.separator_token)
            self.records_seen += 1

    def next_batch(
        self, *, batch_size: int, seq_len: int, vocab_size: int, device: torch.device
    ) -> torch.Tensor:
        needed = int(batch_size) * (int(seq_len) + 1)
        self._fill(needed)
        raw = self.token_buffer[:needed]
        del self.token_buffer[:needed]
        batch = torch.as_tensor(raw, dtype=torch.long).view(
            int(batch_size), int(seq_len) + 1
        )
        if int(vocab_size) != self.vocab_size:
            batch.remainder_(int(vocab_size))
        self.tokens_emitted += int(batch_size) * int(seq_len)
        return batch.to(device, non_blocking=device.type == "cuda")

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "packed_pt",
            "name": self.name,
            "files": [str(p) for p in self.files],
            "file_order": self.file_order,
            "file_pos": self.file_pos,
            "record_pos": self.record_pos,
            "token_buffer": self.token_buffer,
            "records_seen": self.records_seen,
            "records_skipped": self.records_skipped,
            "chunks_loaded": self.chunks_loaded,
            "chunks_missing": self.chunks_missing,
            "tokens_emitted": self.tokens_emitted,
            "separator_token": self.separator_token,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        file_order = [int(x) for x in state.get("file_order", self.file_order)]
        saved_files = [str(x) for x in state.get("files", [])]
        saved_file_pos = int(state.get("file_pos", 0))
        saved_current_path: str | None = None
        if saved_files and file_order:
            try:
                saved_current_path = saved_files[
                    file_order[saved_file_pos % len(file_order)]
                ]
            except Exception:
                saved_current_path = None
        if len(file_order) == len(self.files):
            self.file_order = file_order
        self.file_pos = saved_file_pos % len(self.file_order)
        if saved_current_path is not None:
            path_to_idx = {str(path): idx for idx, path in enumerate(self.files)}
            wanted_idx = path_to_idx.get(saved_current_path)
            if wanted_idx is not None and wanted_idx in self.file_order:
                self.file_pos = self.file_order.index(wanted_idx)
        self.record_pos = int(state.get("record_pos", 0))
        self.token_buffer = [int(x) for x in state.get("token_buffer", [])]
        self.records_seen = int(state.get("records_seen", 0))
        self.records_skipped = int(state.get("records_skipped", 0))
        self.chunks_loaded = int(state.get("chunks_loaded", 0))
        self.chunks_missing = int(state.get("chunks_missing", 0))
        self.tokens_emitted = int(state.get("tokens_emitted", 0))
        self.separator_token = (
            int(state.get("separator_token", self.separator_token)) % self.vocab_size
        )
        self._load_current_chunk(count_load=False)

    def coverage(self, seq_len: int) -> dict[str, Any]:
        path = self.files[self.file_order[self.file_pos]]
        return {
            "name": self.name,
            "tokens_available": None,
            "tokens_emitted": int(self.tokens_emitted),
            "records_seen": int(self.records_seen),
            "records_skipped": int(self.records_skipped),
            "chunks_loaded": int(self.chunks_loaded),
            "chunks_missing": int(self.chunks_missing),
            "current_file": str(path),
            "file_pos": int(self.file_pos),
            "record_pos": int(self.record_pos),
            "n_files": len(self.files),
            "buffered_tokens": len(self.token_buffer),
        }


class MixedPretrainLoader:
    def __init__(
        self,
        sources: list[tuple[TokenSource, float]],
        *,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        device: torch.device,
        seed: int,
    ) -> None:
        live = [(src, float(weight)) for src, weight in sources if float(weight) > 0]
        if not live:
            raise ValueError("no pretrain sources available")
        self.sources = [src for src, _ in live]
        weights = torch.as_tensor([weight for _, weight in live], dtype=torch.double)
        self.probs = (weights / weights.sum()).tolist()
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.device = device
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.batches = 0
        self.samples_by_source = {src.name: 0 for src in self.sources}

    def next_batch(self) -> tuple[torch.Tensor, str]:
        idx = int(
            torch.multinomial(
                torch.as_tensor(self.probs), 1, generator=self.generator
            ).item()
        )
        src = self.sources[idx]
        batch = src.next_batch(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        self.batches += 1
        self.samples_by_source[src.name] = (
            self.samples_by_source.get(src.name, 0) + self.batch_size
        )
        return batch, src.name

    def state_dict(self) -> dict[str, Any]:
        return {
            "generator_state": self.generator.get_state(),
            "batches": self.batches,
            "samples_by_source": self.samples_by_source,
            "probs": self.probs,
            "sources": {src.name: src.state_dict() for src in self.sources},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("generator_state") is not None:
            self.generator.set_state(_cpu_rng_state(state["generator_state"]))
        self.batches = int(state.get("batches", 0))
        self.samples_by_source = dict(
            state.get("samples_by_source", self.samples_by_source)
        )
        by_name = state.get("sources", {})
        for src in self.sources:
            if src.name in by_name:
                src.load_state_dict(by_name[src.name])

    def coverage(self) -> dict[str, Any]:
        return {
            "batches": int(self.batches),
            "samples_by_source": dict(self.samples_by_source),
            "source_probabilities": {
                self.sources[i].name: float(self.probs[i])
                for i in range(len(self.sources))
            },
            "sources": [src.coverage(self.seq_len) for src in self.sources],
        }

    def close(self) -> None:
        for src in self.sources:
            close = getattr(src, "close", None)
            if close is not None:
                close()


def discover_pretrain_sources(
    args: argparse.Namespace, device: torch.device
) -> tuple[MixedPretrainLoader, dict[str, Any]]:
    source_specs: list[tuple[TokenSource, float]] = []
    discovery: dict[str, Any] = {"available": [], "missing": [], "notes": []}

    wt_train = Path(args.wikitext_train)
    if wt_train.exists():
        source_specs.append(
            (
                MemmapWindowSource(
                    "wikitext103_train", wt_train, seed=int(args.seed) + 11
                ),
                float(args.wikitext_weight),
            )
        )
        discovery["available"].append(
            {"name": "wikitext103_train", "path": str(wt_train)}
        )
    else:
        discovery["missing"].append(
            {"name": "wikitext103_train", "path": str(wt_train)}
        )

    fineweb_root = Path(args.fineweb_jsonl_root)
    fineweb_files = sorted(fineweb_root.glob("*/*.jsonl"))[
        : int(args.max_fineweb_shards)
    ]
    if fineweb_files:
        source_specs.append(
            (
                JsonlStreamingSource(
                    "finefineweb_local_jsonl",
                    fineweb_files,
                    vocab_size=int(args.vocab_size),
                    seed=int(args.seed) + 23,
                    max_text_chars=int(args.jsonl_max_text_chars),
                ),
                float(args.fineweb_weight),
            )
        )
        discovery["available"].append(
            {
                "name": "finefineweb_local_jsonl",
                "root": str(fineweb_root),
                "n_shards": len(fineweb_files),
            }
        )
    else:
        discovery["missing"].append(
            {"name": "finefineweb_local_jsonl", "root": str(fineweb_root)}
        )
        discovery["notes"].append(
            "FineFineWeb local JSONL missing; external/local cache required for full intended mix."
        )

    nano_path = Path(args.nano_corpus)
    if nano_path.exists():
        source_specs.append(
            (
                TextTokenSource(
                    "nano_corpus_v4",
                    nano_path,
                    vocab_size=int(args.vocab_size),
                    seed=int(args.seed) + 37,
                ),
                float(args.nano_weight),
            )
        )
        discovery["available"].append(
            {"name": "nano_corpus_v4", "path": str(nano_path)}
        )
    else:
        discovery["missing"].append({"name": "nano_corpus_v4", "path": str(nano_path)})

    for jsonl_path in [Path(p) for p in (args.extra_jsonl or [])]:
        if jsonl_path.exists():
            source_specs.append(
                (
                    JsonlStreamingSource(
                        f"extra_jsonl:{jsonl_path.name}",
                        [jsonl_path],
                        vocab_size=int(args.vocab_size),
                        seed=int(args.seed) + 41 + len(source_specs),
                        max_text_chars=int(args.jsonl_max_text_chars),
                    ),
                    float(args.extra_jsonl_weight),
                )
            )
            discovery["available"].append(
                {"name": f"extra_jsonl:{jsonl_path.name}", "path": str(jsonl_path)}
            )
        else:
            discovery["missing"].append(
                {"name": f"extra_jsonl:{jsonl_path.name}", "path": str(jsonl_path)}
            )

    for idx, spec in enumerate(args.packed_pt_source or []):
        name, root, weight = _parse_packed_pt_source(spec)
        if root.is_dir():
            pt_files = [path for path in sorted(root.rglob("*.pt")) if path.is_file()]
        elif root.is_file():
            pt_files = [root]
        else:
            pt_files = []
        if pt_files:
            source_specs.append(
                (
                    PackedPtTokenSource(
                        name,
                        pt_files,
                        vocab_size=int(args.vocab_size),
                        seed=int(args.seed) + 701 + idx,
                    ),
                    weight,
                )
            )
            discovery["available"].append(
                {
                    "name": name,
                    "root": str(root),
                    "n_files": len(pt_files),
                    "kind": "packed_pt",
                }
            )
        else:
            discovery["missing"].append(
                {"name": name, "root": str(root), "kind": "packed_pt"}
            )

    loader = MixedPretrainLoader(
        source_specs,
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        vocab_size=int(args.vocab_size),
        device=device,
        seed=int(args.seed) + 101,
    )
    discovery["mix"] = loader.coverage()["source_probabilities"]
    return loader, discovery


def _parse_packed_pt_source(spec: str) -> tuple[str, Path, float]:
    parts = str(spec).split("=", 2)
    if len(parts) != 3 or not parts[0] or not parts[1]:
        raise ValueError(
            "--packed-pt-source must use NAME=PATH=WEIGHT, "
            "for example pleias=/mnt/data/LLM/training_pleias_synth/processed=0.08"
        )
    try:
        weight = float(parts[2])
    except ValueError as exc:
        raise ValueError(f"invalid packed source weight in {spec!r}") from exc
    return parts[0], Path(parts[1]), weight


def load_val_source(args: argparse.Namespace) -> MemmapWindowSource | None:
    path = Path(args.wikitext_val)
    if not path.exists():
        return None
    return MemmapWindowSource(
        "wikitext103_val", path, seed=int(args.seed) + 503, split="val"
    )


def lm_loss(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    logits = model(batch[:, :-1])
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)
    )


def _is_lazy_compile_failure(exc: BaseException) -> bool:
    module = type(exc).__module__
    name = type(exc).__name__
    if module.startswith("torch._dynamo") or module.startswith("torch._inductor"):
        return True
    return name in {"BackendCompilerFailed", "InductorError", "Unsupported"}


def _compact_exception_message(exc: BaseException, *, max_chars: int = 220) -> str:
    message = " ".join(str(exc).split())
    for marker in ("Set TORCHDYNAMO_VERBOSE=", "For even more developer context"):
        pos = message.find(marker)
        if pos >= 0:
            message = message[:pos].strip()
    if len(message) > max_chars:
        message = message[: max_chars - 3].rstrip() + "..."
    return message or type(exc).__name__


def configure_torch_performance() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.recompile_limit = 32
    except Exception:
        pass
    try:
        import torch._inductor.config as inductor_config

        inductor_config.triton.cudagraphs = False
        if hasattr(inductor_config.triton, "cudagraph_trees"):
            inductor_config.triton.cudagraph_trees = False
    except Exception:
        pass


def mark_cudagraph_step_begin() -> None:
    if not torch.cuda.is_available():
        return
    try:
        marker = torch.compiler.cudagraph_mark_step_begin
    except AttributeError:
        return
    marker()


def maybe_compile_training_model(
    model: nn.Module,
    *,
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
) -> nn.Module:
    if not bool(args.compile):
        return model
    if device.type != "cuda":
        logger.warning(
            "torch.compile requested but device=%s; using eager model", device
        )
        return model
    if not hasattr(torch, "compile"):
        msg = "torch.compile is unavailable in this PyTorch build"
        if bool(args.compile_required):
            raise RuntimeError(msg)
        logger.warning("%s; using eager model", msg)
        return model
    try:
        logger.info(
            "compiling training forward with torch.compile mode=%s fullgraph=%s dynamic=%s",
            args.compile_mode,
            bool(args.compile_fullgraph),
            bool(args.compile_dynamic),
        )
        compiled = torch.compile(
            model,
            mode=str(args.compile_mode),
            fullgraph=bool(args.compile_fullgraph),
            dynamic=bool(args.compile_dynamic),
        )
        logger.info("torch.compile training forward ready")
        return compiled
    except Exception as exc:  # noqa: BLE001
        if bool(args.compile_required):
            raise
        logger.exception("torch.compile failed; continuing with eager model: %s", exc)
        return model


@torch.no_grad()
def eval_validation_ppl(
    model: nn.Module,
    val_source: MemmapWindowSource | None,
    *,
    n_batches: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if val_source is None:
        return {
            "status": "failed",
            "error": "wikitext103_val.npy not found",
            "metrics": {},
        }
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(int(n_batches)):
        batch = val_source.next_batch(
            batch_size=int(batch_size),
            seq_len=int(seq_len),
            vocab_size=int(vocab_size),
            device=device,
        )
        loss = lm_loss(model, batch)
        losses.append(float(loss.item()))
    if was_training:
        model.train()
    mean_loss = float(sum(losses) / max(1, len(losses)))
    return {
        "status": "ok",
        "metrics": {
            "validation_loss": mean_loss,
            "wikitext_ppl": float(math.exp(min(mean_loss, 30.0))),
            "n_eval_batches": int(n_batches),
        },
    }


def _dataclass_to_metrics(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "to_dict"):
        return dict(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": obj}


def run_one_eval(
    name: str,
    fn,
    *,
    context: dict[str, Any],
    metrics_path: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    t0 = time.monotonic()
    base = {
        "event": "eval",
        "eval_name": name,
        "step": int(context["step"]),
        "tokens_seen": int(context["tokens_seen"]),
        "lr": float(context["lr"]),
        "checkpoint_path": context.get("checkpoint_path"),
        "wall_time_s": round(time.monotonic() - float(context["started_at"]), 3),
    }
    try:
        result = fn()
        if isinstance(result, dict) and "status" in result and "metrics" in result:
            row = {**base, **result}
        else:
            row = {
                **base,
                "status": "ok",
                "metrics": _dataclass_to_metrics(result),
                "error": None,
            }
    except Exception as exc:  # noqa: BLE001
        row = {
            **base,
            "status": "failed",
            "metrics": {},
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
    row["elapsed_s"] = round(time.monotonic() - t0, 3)
    _append_jsonl(metrics_path, row)
    if row["status"] == "ok":
        logger.info("eval %s ok metrics=%s", name, row.get("metrics", {}))
    else:
        logger.error("eval %s %s error=%s", name, row["status"], row.get("error"))
    return row


def _long_context_seq_lens(
    args: argparse.Namespace, logger: logging.Logger
) -> tuple[int, ...]:
    requested = tuple(int(x) for x in str(args.long_context_seq_lens).split(",") if x)
    max_len = int(args.seq_len)
    usable = tuple(x for x in requested if 0 < x <= max_len)
    if not usable:
        usable = (max_len,)
    dropped = tuple(x for x in requested if x > max_len)
    if dropped:
        logger.warning(
            "dropping long-context eval lengths beyond model max_seq_len=%d: requested=%s using=%s",
            max_len,
            requested,
            usable,
        )
    return usable


def run_eval_suite(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    lane_factory,
    val_source: MemmapWindowSource | None,
    context: dict[str, Any],
    metrics_path: Path,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    suite = [x.strip() for x in str(args.eval_suite).split(",") if x.strip()]
    if suite == ["none"]:
        return []
    rows: list[dict[str, Any]] = []
    device = str(args.device)
    long_context_seq_lens = _long_context_seq_lens(args, logger)

    registry = {
        "validation": lambda: eval_validation_ppl(
            model,
            val_source,
            n_batches=int(args.n_eval_batches),
            batch_size=int(args.eval_batch_size),
            seq_len=int(args.seq_len),
            vocab_size=int(args.vocab_size),
            device=torch.device(device),
        ),
        "blimp": lambda: (
            __import__("research.eval.blimp_eval", fromlist=["evaluate_blimp"])
            .evaluate_blimp(
                model,
                vocab_size=int(args.vocab_size),
                device=device,
                n_per_subtask=int(args.blimp_n_per_subtask),
                max_seq_len=int(args.seq_len),
            )
            .to_dict()
        ),
        "hellaswag": lambda: __import__(
            "research.eval.hellaswag_eval", fromlist=["evaluate_hellaswag"]
        ).evaluate_hellaswag(
            model,
            int(args.vocab_size),
            device,
            n_examples=int(args.hellaswag_n_examples),
        ),
        "induction": lambda: __import__(
            "research.eval.native_induction",
            fromlist=["induction_score_gold", "induction_result_metadata"],
        ).induction_result_metadata(
            __import__(
                "research.eval.native_induction", fromlist=["induction_score_gold"]
            ).induction_score_gold(model, device=device, seed=int(args.seed))
        ),
        "induction_intermediate": lambda: (
            __import__(
                "research.eval.induction_intermediate_probe",
                fromlist=["run_induction_intermediate"],
            )
            .run_induction_intermediate(
                model,
                n_train_steps=int(args.probe_train_steps),
                n_eval=int(args.probe_eval_examples),
                batch_size=int(args.probe_batch_size),
                device=device,
                timeout_s=float(args.probe_timeout_s),
            )
            .to_dict()
        ),
        "associative_recall": lambda: _dataclass_to_metrics(
            __import__(
                "research.eval.associative_recall",
                fromlist=["associative_recall_score"],
            ).associative_recall_score(
                model,
                n_train_steps=int(args.probe_train_steps),
                n_eval=int(args.probe_eval_examples),
                batch_size=int(args.probe_batch_size),
                device=device,
                timeout_s=float(args.probe_timeout_s),
            )
        ),
        "ar_curriculum": lambda: (
            __import__(
                "research.eval.ar_curriculum_probe",
                fromlist=["ar_curriculum_probe", "ARCurriculumConfig"],
            )
            .ar_curriculum_probe(
                model,
                cfg=__import__(
                    "research.eval.ar_curriculum_probe", fromlist=["ARCurriculumConfig"]
                ).ARCurriculumConfig(
                    steps_per_stage=int(args.ar_curriculum_steps_per_stage),
                    batch_size=int(args.probe_batch_size),
                    eval_batches=max(
                        1,
                        int(args.probe_eval_examples)
                        // max(1, int(args.probe_batch_size)),
                    ),
                    timeout_s=float(args.probe_timeout_s),
                    copy_model=True,
                ),
                device=device,
            )
            .to_dict()
        ),
        "binding_v2": lambda: (
            __import__(
                "research.eval.binding_intermediate_probe",
                fromlist=["run_binding_intermediate"],
            )
            .run_binding_intermediate(
                model,
                n_train_steps=int(args.probe_train_steps),
                n_eval=int(args.probe_eval_examples),
                train_batch_size=int(args.probe_batch_size),
                eval_batch_size=int(args.probe_batch_size),
                device=device,
                timeout_s=float(args.probe_timeout_s),
            )
            .to_dict()
        ),
        "nanobind": lambda: _dataclass_to_metrics(
            __import__(
                "component_fab.harness.nano_bind_probe", fromlist=["nano_bind_gate"]
            ).nano_bind_gate(
                lane_factory(int(args.smoke_probe_dim)),
                dim=int(args.smoke_probe_dim),
                n_train_steps=max(5, min(60, int(args.probe_train_steps))),
                batch_size=int(args.probe_batch_size),
                seed=int(args.seed),
            )
        ),
        "language_control": lambda: (
            __import__(
                "research.eval.language_control_probe",
                fromlist=["language_control_probe"],
            )
            .language_control_probe(
                model,
                n_train_steps=int(args.probe_train_steps),
                batch_size=int(args.probe_batch_size),
                timeout_s=float(args.probe_timeout_s),
                device=device,
                preserve_state=True,
            )
            .to_dict()
        ),
        "permutation_composition": lambda: (
            __import__(
                "research.eval.permutation_composition_probe",
                fromlist=["permutation_composition_score"],
            )
            .permutation_composition_score(
                model,
                n_train_steps=int(args.probe_train_steps),
                n_eval_batches=max(
                    1,
                    int(args.probe_eval_examples) // max(1, int(args.probe_batch_size)),
                ),
                batch_size=int(args.probe_batch_size),
                device=device,
            )
            .to_dict()
        ),
        "long_range_ar": lambda: _dataclass_to_metrics(
            __import__(
                "research.eval.long_range_ar", fromlist=["long_range_ar_score"]
            ).long_range_ar_score(
                model,
                seq_lens=long_context_seq_lens,
                n_train_steps=int(args.probe_train_steps),
                n_eval=int(args.probe_eval_examples),
                batch_size=int(args.probe_batch_size),
                device=device,
                timeout_s=float(args.probe_timeout_s),
            )
        ),
        "passkey": lambda: _dataclass_to_metrics(
            __import__(
                "research.eval.passkey_retrieval", fromlist=["passkey_retrieval_score"]
            ).passkey_retrieval_score(
                model,
                seq_lens=long_context_seq_lens,
                n_train_steps=int(args.probe_train_steps),
                n_eval=int(args.probe_eval_examples),
                batch_size=int(args.probe_batch_size),
                device=device,
                timeout_s=float(args.probe_timeout_s),
            )
        ),
    }

    for name in suite:
        fn = registry.get(name)
        if fn is None:
            rows.append(
                run_one_eval(
                    name,
                    lambda n=name: {
                        "status": "failed",
                        "metrics": {},
                        "error": f"unknown eval '{n}'",
                    },
                    context=context,
                    metrics_path=metrics_path,
                    logger=logger,
                )
            )
            continue
        rows.append(
            run_one_eval(
                name, fn, context=context, metrics_path=metrics_path, logger=logger
            )
        )
    return rows


def save_checkpoint(
    *,
    checkpoint_dir: Path,
    reason: str,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineSchedule,
    scaler: Any,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    tokens_seen: int,
    eval_history: list[dict[str, Any]],
    data_loader: MixedPretrainLoader,
    logger: logging.Logger,
) -> str:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if reason == "periodic":
        name = f"step_{int(step):06d}.pt"
    else:
        name = f"{reason}_step_{int(step):06d}.pt"
    path = checkpoint_dir / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": 1,
        "reason": reason,
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": rng_state_payload(),
        "args": vars(args),
        "metadata": metadata,
        "eval_history": eval_history,
        "data_mix_state": data_loader.state_dict(),
        "git": git_snapshot(),
    }
    torch.save(payload, tmp)
    tmp.replace(path)
    latest = checkpoint_dir / "latest.pt"
    latest_tmp = latest.with_suffix(latest.suffix + ".tmp")
    try:
        if latest_tmp.exists():
            latest_tmp.unlink()
        os.link(path, latest_tmp)
    except OSError:
        shutil.copy2(path, latest_tmp)
    latest_tmp.replace(latest)
    logger.info(
        "saved checkpoint reason=%s step=%d path=%s latest=%s",
        reason,
        step,
        path,
        latest,
    )
    return str(path)


def load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def setup_logging(output_dir: Path, *, append: bool = False) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_tropical_gate_120m_pretrain")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(
        output_dir / "train.log", mode="a" if append else "w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def build_model_and_factory(
    args: argparse.Namespace,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    if args.size == "smoke":
        dim, n_blocks = SMOKE_SIZING["dim"], SMOKE_SIZING["n_blocks"]
    else:
        sizing = PARAM_SIZING[str(args.size)]
        dim, n_blocks = int(sizing["dim"]), int(sizing["n_blocks"])
    label, lane_factory, axes = _saved_winner_factory(str(args.proposal_id))
    model = _build_tinylm(
        lane_factory,
        dim=dim,
        n_blocks=n_blocks,
        vocab_size=int(args.vocab_size),
        max_seq_len=int(args.seq_len),
        use_ffn=bool(args.use_ffn),
    )
    setattr(model, "vocab_size", int(args.vocab_size))
    build_meta = {
        "proposal_label": label,
        "math_axes": axes,
        "dim": dim,
        "n_blocks": n_blocks,
        "vocab_size": int(args.vocab_size),
        "use_ffn": bool(args.use_ffn),
    }
    return model, lane_factory, build_meta


def compute_training_budget(
    args: argparse.Namespace, param_count: int
) -> dict[str, int]:
    target_tokens = int(
        math.ceil(float(args.chinchilla_tokens_per_param) * int(param_count))
    )
    effective_batch_tokens = (
        int(args.batch_size) * int(args.seq_len) * int(args.grad_accum_steps)
    )
    total_steps = (
        int(args.total_steps)
        if args.total_steps is not None
        else int(math.ceil(target_tokens / effective_batch_tokens))
    )
    return {
        "actual_param_count": int(param_count),
        "chinchilla_target_token_visits": int(target_tokens),
        "effective_batch_tokens": int(effective_batch_tokens),
        "total_optimizer_steps": int(total_steps),
    }


@dataclass
class InterruptState:
    requested: bool = False
    signum: int | None = None


def install_signal_handlers(state: InterruptState) -> None:
    def _handler(signum, _frame):
        state.requested = True
        state.signum = int(signum)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def cuda_memory_summary(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"allocated_mib": 0.0, "reserved_mib": 0.0, "max_allocated_mib": 0.0}
    return {
        "allocated_mib": round(torch.cuda.memory_allocated(device) / (1024**2), 1),
        "reserved_mib": round(torch.cuda.memory_reserved(device) / (1024**2), 1),
        "max_allocated_mib": round(
            torch.cuda.max_memory_allocated(device) / (1024**2), 1
        ),
    }


def _format_source_counts(source_names: list[str]) -> str:
    counts = Counter(source_names)
    if not counts:
        return "-"
    return ",".join(f"{name}:{count}" for name, count in counts.most_common())


def _format_source_probs(coverage: dict[str, Any]) -> str:
    probs = coverage.get("source_probabilities") or {}
    if not isinstance(probs, dict):
        return "-"
    return ",".join(f"{name}={float(prob):.3g}" for name, prob in probs.items())


def _known_epoch_equivalent(coverage: dict[str, Any]) -> float | None:
    total_emitted = 0
    total_available = 0
    for source in coverage.get("sources") or []:
        if not isinstance(source, dict):
            continue
        available = source.get("tokens_available")
        emitted = source.get("tokens_emitted")
        if available is None or emitted is None:
            continue
        total_available += int(available)
        total_emitted += int(emitted)
    if total_available <= 0:
        return None
    return float(total_emitted) / float(total_available)


def train(args: argparse.Namespace) -> int:
    configure_torch_performance()
    resume_payload = None
    if args.resume:
        resume_payload = load_checkpoint(Path(args.resume), args.device)

    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir, append=bool(args.resume))
    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_dir = output_dir / "checkpoints"
    device = torch.device(args.device)
    interrupt_state = InterruptState()
    install_signal_handlers(interrupt_state)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    model, lane_factory, build_meta = build_model_and_factory(args)
    model.to(device)
    param_count = count_trainable_params(model)
    budget = compute_training_budget(args, param_count)
    args.total_steps = budget["total_optimizer_steps"]

    data_loader, data_discovery = discover_pretrain_sources(args, device)
    val_source = load_val_source(args)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = WarmupCosineSchedule(
        optimizer,
        learning_rate=float(args.learning_rate),
        min_lr=float(args.min_lr),
        warmup_steps=int(args.warmup_steps),
        total_steps=int(args.total_steps),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(args.amp and device.type == "cuda")
    )

    global_step = 0
    tokens_seen = 0
    eval_history: list[dict[str, Any]] = []
    metadata = {
        "run_started_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "model": build_meta,
        "budget": budget,
        "data": data_discovery,
        "git": git_snapshot(),
    }

    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        if resume_payload.get("scaler_state_dict") and scaler is not None:
            scaler.load_state_dict(resume_payload["scaler_state_dict"])
        scheduler.learning_rate = float(args.learning_rate)
        scheduler.min_lr = float(args.min_lr)
        scheduler.warmup_steps = int(args.warmup_steps)
        scheduler.total_steps = int(args.total_steps)
        for group in optimizer.param_groups:
            group["weight_decay"] = float(args.weight_decay)
        restore_rng_state(resume_payload.get("rng_state"))
        data_loader.load_state_dict(resume_payload.get("data_mix_state", {}))
        global_step = int(resume_payload.get("step", 0))
        tokens_seen = int(resume_payload.get("tokens_seen", 0))
        eval_history = list(resume_payload.get("eval_history", []))
        metadata["resumed_from"] = str(args.resume)
        metadata["resume_scheduler_state"] = resume_payload.get("scheduler_state_dict")
        scheduler.apply(global_step)
        logger.info(
            "resumed checkpoint=%s step=%d tokens_seen=%d next_lr=%.8g",
            args.resume,
            global_step,
            tokens_seen,
            scheduler.lr_at(global_step),
        )

    train_model = maybe_compile_training_model(
        model, args=args, device=device, logger=logger
    )

    _append_jsonl(
        metrics_path,
        {
            "event": "metadata",
            "step": global_step,
            "tokens_seen": tokens_seen,
            **metadata,
        },
    )
    logger.info(
        "start proposal=%s size=%s params=%d target_tokens=%d total_steps=%d effective_batch_tokens=%d",
        args.proposal_id,
        args.size,
        param_count,
        budget["chinchilla_target_token_visits"],
        budget["total_optimizer_steps"],
        budget["effective_batch_tokens"],
    )
    initial_coverage = data_loader.coverage()
    logger.info(
        "data mix source_probs=%s source_count=%d",
        _format_source_probs(initial_coverage),
        len(initial_coverage.get("sources") or []),
    )

    started_at = time.monotonic()
    last_log = started_at
    last_tokens = tokens_seen
    grad_spikes = 0
    best_ppl = float("inf")
    grad_accum = int(args.grad_accum_steps)
    max_steps = int(args.total_steps)
    clip_norm = float(args.grad_clip_norm)
    last_checkpoint_path: str | None = None
    nonfinite_grad_skips = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        while global_step < max_steps:
            if interrupt_state.requested:
                last_checkpoint_path = save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    reason=f"signal_{interrupt_state.signum or 'interrupt'}",
                    step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                    metadata=metadata,
                    tokens_seen=tokens_seen,
                    eval_history=eval_history,
                    data_loader=data_loader,
                    logger=logger,
                )
                return 130

            lr = scheduler.apply(global_step)
            accum_loss = 0.0
            source_names: list[str] = []
            tokens_this_step = 0
            optimizer.zero_grad(set_to_none=True)
            for micro in range(grad_accum):
                full_batch, source_name = data_loader.next_batch()
                source_names.append(source_name)
                if train_model is not model:
                    mark_cudagraph_step_begin()
                with torch.amp.autocast(
                    "cuda", enabled=bool(args.amp and device.type == "cuda")
                ):
                    try:
                        loss = lm_loss(train_model, full_batch) / float(grad_accum)
                    except Exception as exc:  # noqa: BLE001
                        if (
                            train_model is model
                            or bool(args.compile_required)
                            or not _is_lazy_compile_failure(exc)
                        ):
                            raise
                        logger.warning(
                            "torch.compile failed during lazy codegen (%s: %s); disabling compiled wrapper and retrying current microbatch eagerly",
                            type(exc).__name__,
                            _compact_exception_message(exc),
                        )
                        train_model = model
                        torch._dynamo.reset()
                        loss = lm_loss(train_model, full_batch) / float(grad_accum)
                if not torch.isfinite(loss):
                    last_checkpoint_path = save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        reason="emergency_nonfinite_loss",
                        step=global_step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        args=args,
                        metadata=metadata,
                        tokens_seen=tokens_seen,
                        eval_history=eval_history,
                        data_loader=data_loader,
                        logger=logger,
                    )
                    raise FloatingPointError(
                        f"nonfinite loss at step={global_step} micro={micro}"
                    )
                scaler.scale(loss).backward()
                accum_loss += float(loss.detach().item())
                tokens_this_step += int(args.batch_size) * int(args.seq_len)
                if interrupt_state.requested:
                    break

            if interrupt_state.requested:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            grad_norm_f = float(
                grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm
            )
            if not math.isfinite(grad_norm_f):
                nonfinite_grad_skips += 1
                logger.warning(
                    "nonfinite grad at step=%d skip=%d/%d loss=%.4f lr=%.8g src=%s; skipping optimizer update",
                    global_step,
                    nonfinite_grad_skips,
                    int(args.nonfinite_grad_patience),
                    accum_loss,
                    lr,
                    _format_source_counts(source_names),
                )
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                if nonfinite_grad_skips >= int(args.nonfinite_grad_patience):
                    last_checkpoint_path = save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        reason="emergency_nonfinite_grad",
                        step=global_step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        args=args,
                        metadata=metadata,
                        tokens_seen=tokens_seen,
                        eval_history=eval_history,
                        data_loader=data_loader,
                        logger=logger,
                    )
                    raise FloatingPointError(
                        f"repeated nonfinite grad at step={global_step}"
                    )
                continue
            nonfinite_grad_skips = 0
            if grad_norm_f > float(args.grad_spike_threshold):
                grad_spikes += 1
            else:
                grad_spikes = 0
            if grad_spikes >= int(args.grad_spike_patience):
                last_checkpoint_path = save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    reason="emergency_grad_spikes",
                    step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                    metadata=metadata,
                    tokens_seen=tokens_seen,
                    eval_history=eval_history,
                    data_loader=data_loader,
                    logger=logger,
                )
                raise FloatingPointError(f"repeated grad spikes: norm={grad_norm_f}")

            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            tokens_seen += tokens_this_step
            scheduler.apply(global_step)

            now = time.monotonic()
            if (
                global_step <= 5
                or global_step % int(args.log_every) == 0
                or now - last_log >= float(args.log_interval_s)
            ):
                dt = max(1e-9, now - last_log)
                tps = (tokens_seen - last_tokens) / dt
                remaining = max_steps - global_step
                eta_s = remaining * (now - started_at) / max(1, global_step)
                mem = cuda_memory_summary(device)
                coverage = data_loader.coverage()
                approx_ppl = math.exp(min(float(accum_loss), 30.0))
                known_epoch = _known_epoch_equivalent(coverage)
                row = {
                    "event": "train",
                    "step": global_step,
                    "tokens_seen": tokens_seen,
                    "loss": accum_loss,
                    "approx_train_ppl": approx_ppl,
                    "grad_norm": grad_norm_f,
                    "lr": scheduler.lr_at(global_step),
                    "tokens_per_sec": tps,
                    "eta_s": eta_s,
                    "sources": source_names,
                    "data_coverage": coverage,
                    "gpu_memory": mem,
                }
                _append_jsonl(metrics_path, row)
                logger.info(
                    "step=%d/%d loss=%.4f ppl~=%.1f grad=%.3f lr=%.8g tok/s=%.0f tokens=%.3fb eta=%.1fh epoch_known=%s src=%s gpu_alloc=%.0fMiB gpu_max=%.0fMiB",
                    global_step,
                    max_steps,
                    accum_loss,
                    approx_ppl,
                    grad_norm_f,
                    scheduler.lr_at(global_step),
                    tps,
                    tokens_seen / 1e9,
                    eta_s / 3600.0,
                    "-" if known_epoch is None else f"{known_epoch:.4f}",
                    _format_source_counts(source_names),
                    float(mem.get("allocated_mib", 0.0)),
                    float(mem.get("max_allocated_mib", 0.0)),
                )
                last_log = now
                last_tokens = tokens_seen

            should_ckpt = (
                global_step % int(args.checkpoint_every) == 0
                or global_step == max_steps
            )
            should_eval = (
                global_step % int(args.eval_every) == 0 or global_step == max_steps
            )
            if should_ckpt:
                last_checkpoint_path = save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    reason="periodic",
                    step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                    metadata=metadata,
                    tokens_seen=tokens_seen,
                    eval_history=eval_history,
                    data_loader=data_loader,
                    logger=logger,
                )
            if should_eval:
                if last_checkpoint_path is None:
                    last_checkpoint_path = save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        reason="pre_eval",
                        step=global_step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        args=args,
                        metadata=metadata,
                        tokens_seen=tokens_seen,
                        eval_history=eval_history,
                        data_loader=data_loader,
                        logger=logger,
                    )
                ctx = {
                    "step": global_step,
                    "tokens_seen": tokens_seen,
                    "lr": scheduler.lr_at(global_step),
                    "checkpoint_path": last_checkpoint_path,
                    "started_at": started_at,
                }
                rows = run_eval_suite(
                    args=args,
                    model=model,
                    lane_factory=lane_factory,
                    val_source=val_source,
                    context=ctx,
                    metrics_path=metrics_path,
                    logger=logger,
                )
                eval_history.extend(rows)
                for row in rows:
                    metrics = row.get("metrics") or {}
                    ppl = metrics.get("wikitext_ppl")
                    if row.get("eval_name") == "validation" and ppl is not None:
                        ppl_f = float(ppl)
                        if ppl_f < best_ppl:
                            best_ppl = ppl_f
                        elif best_ppl < float("inf") and ppl_f > best_ppl * float(
                            args.ppl_explosion_factor
                        ):
                            last_checkpoint_path = save_checkpoint(
                                checkpoint_dir=checkpoint_dir,
                                reason="emergency_ppl_explosion",
                                step=global_step,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=scaler,
                                args=args,
                                metadata=metadata,
                                tokens_seen=tokens_seen,
                                eval_history=eval_history,
                                data_loader=data_loader,
                                logger=logger,
                            )
                            raise FloatingPointError(
                                f"validation ppl explosion: {ppl_f} > {args.ppl_explosion_factor}x {best_ppl}"
                            )
                model.train()

    except KeyboardInterrupt:
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            reason="keyboard_interrupt",
            step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            args=args,
            metadata=metadata,
            tokens_seen=tokens_seen,
            eval_history=eval_history,
            data_loader=data_loader,
            logger=logger,
        )
        return 130
    finally:
        data_loader.close()

    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        reason="completed",
        step=global_step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        args=args,
        metadata=metadata,
        tokens_seen=tokens_seen,
        eval_history=eval_history,
        data_loader=data_loader,
        logger=logger,
    )
    logger.info("completed step=%d tokens_seen=%d", global_step, tokens_seen)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--proposal-id", default=DEFAULT_PROPOSAL_ID)
    p.add_argument("--size", default="120M", choices=[*PARAM_SIZING.keys(), "smoke"])
    p.add_argument("--chinchilla-tokens-per-param", type=float, default=20.0)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=20000)
    p.add_argument("--checkpoint-every", type=int, default=20000)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--amp", action="store_true", default=None)
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable torch.compile for the training forward pass only; checkpoints remain eager-compatible.",
    )
    p.add_argument(
        "--compile-mode",
        default="max-autotune-no-cudagraphs",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
    )
    p.add_argument(
        "--compile-fullgraph", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument(
        "--compile-dynamic", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument(
        "--compile-required",
        action="store_true",
        help="Exit instead of falling back to eager mode if torch.compile fails.",
    )
    p.add_argument("--use-ffn", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--grad-spike-threshold", type=float, default=10.0)
    p.add_argument("--grad-spike-patience", type=int, default=5)
    p.add_argument(
        "--nonfinite-grad-patience",
        type=int,
        default=3,
        help="Skip isolated nonfinite-gradient optimizer updates; stop after this many consecutive nonfinite gradients.",
    )
    p.add_argument("--ppl-explosion-factor", type=float, default=2.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--log-interval-s", type=float, default=30.0)
    p.add_argument(
        "--wikitext-train",
        default=str(PROJECT_ROOT / "research" / "corpus" / "wikitext103_train.npy"),
    )
    p.add_argument(
        "--wikitext-val",
        default=str(PROJECT_ROOT / "research" / "corpus" / "wikitext103_val.npy"),
    )
    p.add_argument("--fineweb-jsonl-root", default="/mnt/data/hf_finefineweb")
    p.add_argument("--max-fineweb-shards", type=int, default=2000)
    p.add_argument(
        "--jsonl-max-text-chars",
        type=int,
        default=65536,
        help="Cap each streaming JSONL document before tokenization to avoid CPU stalls on pathological long lines.",
    )
    p.add_argument(
        "--nano-corpus",
        default=str(
            PROJECT_ROOT / "research" / "data" / "nano_corpus" / "nano_corpus_v4.txt"
        ),
    )
    p.add_argument(
        "--extra-jsonl",
        action="append",
        default=[
            str(PROJECT_ROOT / "HYDRA" / "data" / "distill_all.jsonl"),
            str(PROJECT_ROOT / "HYDRA" / "data" / "distill_reasoning.jsonl"),
        ],
    )
    p.add_argument(
        "--packed-pt-source",
        action="append",
        default=None,
        help="Add a tokenized .pt corpus as NAME=PATH=WEIGHT; PATH may be a file or directory of .pt chunks.",
    )
    p.add_argument("--fineweb-weight", type=float, default=0.70)
    p.add_argument("--wikitext-weight", type=float, default=0.24)
    p.add_argument("--nano-weight", type=float, default=0.01)
    p.add_argument("--extra-jsonl-weight", type=float, default=0.025)
    p.add_argument(
        "--eval-suite",
        default=(
            "validation,blimp,hellaswag,induction,induction_intermediate,"
            "associative_recall,ar_curriculum,binding_v2,nanobind,"
            "language_control,permutation_composition,long_range_ar,passkey"
        ),
    )
    p.add_argument("--n-eval-batches", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--blimp-n-per-subtask", type=int, default=25)
    p.add_argument("--hellaswag-n-examples", type=int, default=200)
    p.add_argument("--probe-train-steps", type=int, default=300)
    p.add_argument("--probe-eval-examples", type=int, default=128)
    p.add_argument("--probe-batch-size", type=int, default=8)
    p.add_argument("--probe-timeout-s", type=float, default=180.0)
    p.add_argument("--ar-curriculum-steps-per-stage", type=int, default=200)
    p.add_argument("--long-context-seq-lens", default="256,512,1024")
    p.add_argument("--smoke-probe-dim", type=int, default=32)
    return p


def merge_resume_args(
    cli_args: argparse.Namespace, saved_args: dict[str, Any]
) -> argparse.Namespace:
    merged = dict(saved_args)
    for key, value in vars(cli_args).items():
        merged.setdefault(key, value)
    override_keys = {"resume", "device"}
    if cli_args.output_dir is not None:
        override_keys.add("output_dir")
    for runtime_key in (
        "amp",
        "compile",
        "compile_mode",
        "compile_fullgraph",
        "compile_dynamic",
        "compile_required",
        "learning_rate",
        "min_lr",
        "warmup_steps",
        "weight_decay",
        "grad_clip_norm",
        "grad_spike_threshold",
        "nonfinite_grad_patience",
        "total_steps",
        "eval_every",
        "checkpoint_every",
        "log_every",
        "long_context_seq_lens",
        "fineweb_weight",
        "wikitext_weight",
        "nano_weight",
        "extra_jsonl_weight",
        "jsonl_max_text_chars",
        "packed_pt_source",
    ):
        if (
            hasattr(cli_args, runtime_key)
            and getattr(cli_args, runtime_key) is not None
        ):
            override_keys.add(runtime_key)
    for key in override_keys:
        merged[key] = getattr(cli_args, key)
    if cli_args.resume is not None:
        merged["resume"] = cli_args.resume
    return argparse.Namespace(**merged)


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if args.resume:
        payload = load_checkpoint(Path(args.resume), args.device)
        args = merge_resume_args(args, dict(payload.get("args", {})))
        if args.output_dir is None:
            args.output_dir = Path(args.resume).resolve().parents[1]
    if args.output_dir is None:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            PROJECT_ROOT / "research" / "runtime" / f"{DEFAULT_RUN_PREFIX}_{stamp}"
        )
    args.output_dir = Path(args.output_dir)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
