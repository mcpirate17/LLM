"""Data-driven op-property accessors backed by ``meta_analysis.db``.

The ``op_property_catalog`` table holds 60+ static + empirical columns per
op (``op_category``, ``op_dynamical_has_state``,
``op_geometric_receptive_field``, ``eval_count``, ``s1_pass_count``,
``mean_loss`` …). The catalog rebuilds offline from observations and is
the single source of truth for properties that aren't fully captured by
``PRIMITiVE_REGISTRY`` alone (empirical s1/loss rollups; declared
properties that some downstream code needs without importing the
synthesis registry).

This module mirrors ``_slot_constraints_loader.py``: query once at first
use, cache per process, fall back when the DB is unreachable.

Common use cases:

  # All ops the catalog declares as attention (used to be hardcoded in
  # graph_features.py:_ATTENTION_OPS — future refactor target).
  attention_ops = query_ops_by_category("attention", fallback=_HARD_ATTN)

  # All stateful ops (recurrent / SSM / mLSTM / etc.) without naming them.
  stateful = query_ops_by_property("op_dynamical_has_state", lambda v: v == 1)

  # One property for one op, raw.
  rf = query_op_property("mla_attention", "op_geometric_receptive_field")

Callers must always provide a sensible ``fallback`` — the meta DB may be
missing in test environments, may not have been rebuilt after a fresh
primitive was added, or may simply not carry the requested property.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
META_DB = REPO / "research/meta_analysis.db"


_cache_lock = threading.Lock()
_row_cache: dict[str, dict[str, Any]] | None = None


def _connect() -> Optional[sqlite3.Connection]:
    if not META_DB.exists():
        logger.info("op_catalog_loader: meta_analysis.db missing — fallbacks only")
        return None
    try:
        return sqlite3.connect(f"file:{META_DB}?mode=ro&immutable=0", uri=True)
    except sqlite3.Error:
        logger.exception("op_catalog_loader: connect failed")
        return None


def _load_rows() -> dict[str, dict[str, Any]]:
    """Eager load: one row per op_name, columns as dict."""
    conn = _connect()
    if conn is None:
        return {}
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM op_property_catalog").fetchall()
    except sqlite3.Error:
        logger.exception("op_catalog_loader: SELECT op_property_catalog failed")
        return {}
    finally:
        conn.close()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row["op_name"]
        if not name:
            continue
        out[str(name)] = {key: row[key] for key in row.keys()}
    return out


def _ensure_cache() -> dict[str, dict[str, Any]]:
    global _row_cache
    with _cache_lock:
        if _row_cache is None:
            _row_cache = _load_rows()
            logger.info("op_catalog_loader: cache built (%d ops)", len(_row_cache))
        return _row_cache


def reset_cache() -> None:
    """Drop the loader's cache (used by unit tests)."""
    global _row_cache
    with _cache_lock:
        _row_cache = None


def query_op_property(op_name: str, field_name: str) -> Optional[Any]:
    """Return one column for one op, or ``None`` if missing.

    NB: the catalog has its own schema; field_name must be a real column.
    """
    rows = _ensure_cache()
    row = rows.get(op_name)
    if row is None:
        return None
    return row.get(field_name)


def query_ops_by_category(
    category: str, *, fallback: Tuple[str, ...] = ()
) -> frozenset[str]:
    """Return op_names with ``op_category == category``.

    Falls back to ``fallback`` when:
      - the meta DB is absent (test env, fresh checkout)
      - the catalog hasn't been rebuilt since a new op was added
      - the predicate matches no ops in the cache
    """
    rows = _ensure_cache()
    matches = {name for name, row in rows.items() if row.get("op_category") == category}
    if not matches:
        return frozenset(fallback)
    return frozenset(matches)


def query_ops_by_property(
    field_name: str,
    predicate: Callable[[Any], bool],
    *,
    fallback: Tuple[str, ...] = (),
) -> frozenset[str]:
    """Return op_names whose ``field_name`` value satisfies ``predicate``.

    Use cases:
      - ``op_dynamical_has_state``: pick out recurrent / state-bearing ops.
      - ``op_geometric_receptive_field``: pick long-range vs local ops.
      - ``op_composition_residual_safe``: pick ops safe to wrap in a residual.

    Same fallback semantics as :func:`query_ops_by_category`.
    """
    rows = _ensure_cache()
    matches = {name for name, row in rows.items() if predicate(row.get(field_name))}
    if not matches:
        return frozenset(fallback)
    return frozenset(matches)


def group_ops_by_property(field_name: str) -> dict[Any, frozenset[str]]:
    """Return ``{value → frozenset(op_names)}`` for one column.

    Useful for clustering ops by a categorical property (e.g. all ops
    bucketed by ``op_geometric_receptive_field`` in {'local', 'long', None}).
    Empty dict when the meta DB is absent.
    """
    rows = _ensure_cache()
    buckets: dict[Any, set[str]] = defaultdict(set)
    for name, row in rows.items():
        buckets[row.get(field_name)].add(name)
    return {key: frozenset(values) for key, values in buckets.items()}
