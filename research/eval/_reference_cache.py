"""Shared cache and seeded-training helpers for eval reference models."""

from __future__ import annotations

import gc
import math
import sqlite3
from pathlib import Path
from typing import Callable, Iterable, TypeVar


T = TypeVar("T")


def open_sqlite_cache(
    cache_path: str | Path,
    *,
    schema_statements: Iterable[str],
    wal: bool = False,
) -> tuple[Path, sqlite3.Connection]:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    for statement in schema_statements:
        conn.execute(statement)
    conn.commit()
    return path, conn


def average_finite_reference_runs(
    n_seeds: int,
    train_once: Callable[[int], tuple[float, T]],
) -> tuple[float, list[float], T | None]:
    losses: list[float] = []
    aux_value: T | None = None
    for seed in range(max(1, int(n_seeds))):
        final_loss, aux = train_once(seed)
        if seed == 0:
            aux_value = aux
        gc.collect()
        if math.isfinite(final_loss):
            losses.append(final_loss)
    avg_loss = sum(losses) / len(losses) if losses else float("inf")
    return avg_loss, losses, aux_value
