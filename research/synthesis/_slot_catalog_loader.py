"""Data-driven slot-property accessors backed by ``meta_analysis.db``.

The ``slot_property_catalog`` table holds 60+ declared columns per
(template, slot_index): ``slot_role``, ``slot_role_family``,
``slot_position_fraction``, the seven ``slot_accepts_*`` flags
(attention/ssm/routing/compression/memory/norm/math_space),
``slot_search_width_prior``, ``slot_pressure_prior``,
``slot_dynamical_*`` / ``slot_spectral_*`` / ``slot_composition_*`` family,
plus the JSON-encoded ``slot_classes_json`` allowlist itself.

Companion to:
- ``_slot_constraints_loader`` — empirical pass-cohort fills from
  ``slot_observations`` (joined with the lab notebook).
- ``_op_catalog_loader`` — declared + empirical *op* properties from
  ``op_property_catalog``.

This module is purely additive: no existing grammar path consumes it
yet. Future refactors of ``_MIXER_CLASSES`` / ``_FFN_CLASSES`` /
``_BOTTLENECK_CLASSES`` hardcoded tuples should query this instead.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
META_DB = REPO / "research/meta_analysis.db"


_cache_lock = threading.Lock()
_row_cache: dict[tuple[str, int], dict[str, Any]] | None = None


# Mapping from short class names (used in `slot_accepts(...)`) to the
# real column names in slot_property_catalog. Keeps callers from having
# to remember the ``slot_accepts_`` prefix every time.
_SHORT_ACCEPTS_COLUMNS: dict[str, str] = {
    "attention": "slot_accepts_attention",
    "ssm": "slot_accepts_ssm",
    "routing": "slot_accepts_routing",
    "compression": "slot_accepts_compression",
    "memory": "slot_accepts_memory",
    "norm": "slot_accepts_norm",
    "math_space": "slot_accepts_math_space",
}


def _connect() -> Optional[sqlite3.Connection]:
    if not META_DB.exists():
        logger.info("slot_catalog_loader: meta_analysis.db missing — fallbacks only")
        return None
    try:
        return sqlite3.connect(f"file:{META_DB}?mode=ro&immutable=0", uri=True)
    except sqlite3.Error:
        logger.exception("slot_catalog_loader: connect failed")
        return None


def _load_rows() -> dict[tuple[str, int], dict[str, Any]]:
    conn = _connect()
    if conn is None:
        return {}
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM slot_property_catalog").fetchall()
    except sqlite3.Error:
        logger.exception("slot_catalog_loader: SELECT slot_property_catalog failed")
        return {}
    finally:
        conn.close()
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        name = row["template_name"]
        idx = row["slot_index"]
        if not name or idx is None:
            continue
        out[(str(name), int(idx))] = {key: row[key] for key in row.keys()}
    return out


def _ensure_cache() -> dict[tuple[str, int], dict[str, Any]]:
    global _row_cache
    with _cache_lock:
        if _row_cache is None:
            _row_cache = _load_rows()
            logger.info(
                "slot_catalog_loader: cache built (%d (template, slot) rows)",
                len(_row_cache),
            )
        return _row_cache


def reset_cache() -> None:
    """Drop the loader's cache (used by unit tests)."""
    global _row_cache
    with _cache_lock:
        _row_cache = None


def query_slot_property(
    template_name: str, slot_index: int, field_name: str
) -> Optional[Any]:
    """Return one column for one (template, slot_index), or None if missing."""
    rows = _ensure_cache()
    row = rows.get((template_name, int(slot_index)))
    if row is None:
        return None
    return row.get(field_name)


def query_slot_row(template_name: str, slot_index: int) -> Optional[Mapping[str, Any]]:
    """Return the full row dict for one (template, slot_index)."""
    rows = _ensure_cache()
    row = rows.get((template_name, int(slot_index)))
    if row is None:
        return None
    return dict(row)


def slot_accepts(template_name: str, slot_index: int, class_name: str) -> bool:
    """Whether the declared slot accepts ``class_name`` (short form).

    ``class_name`` must be one of: ``attention``, ``ssm``, ``routing``,
    ``compression``, ``memory``, ``norm``, ``math_space``. Returns False
    when the slot or the catalog is missing — callers wanting "unknown
    means yes" should fall back explicitly.
    """
    column = _SHORT_ACCEPTS_COLUMNS.get(class_name)
    if column is None:
        return False
    value = query_slot_property(template_name, slot_index, column)
    return bool(value)


def slot_classes_for(
    template_name: str, slot_index: int, fallback: Tuple[str, ...] = ()
) -> Tuple[str, ...]:
    """Return the declared ``slot_classes_json`` allowlist as a tuple.

    Falls back to ``fallback`` when the slot is missing, the JSON is
    malformed, or the parsed value is empty.
    """
    raw = query_slot_property(template_name, slot_index, "slot_classes_json")
    if not raw:
        return tuple(fallback)
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return tuple(fallback)
    if not isinstance(parsed, list) or not parsed:
        return tuple(fallback)
    return tuple(str(item) for item in parsed if isinstance(item, str))


def query_slots_by_property(
    field_name: str,
    predicate: Callable[[Any], bool],
    *,
    fallback: Tuple[Tuple[str, int], ...] = (),
) -> frozenset[Tuple[str, int]]:
    """Return (template, slot_index) keys whose ``field_name`` matches ``predicate``.

    Falls back to ``fallback`` when nothing matches (or the catalog is
    absent).
    """
    rows = _ensure_cache()
    matches = {key for key, row in rows.items() if predicate(row.get(field_name))}
    if not matches:
        return frozenset(fallback)
    return frozenset(matches)


def slots_for_template(template_name: str) -> Tuple[int, ...]:
    """Return slot indices declared for ``template_name``, ascending."""
    rows = _ensure_cache()
    indices = sorted(idx for (tpl, idx) in rows.keys() if tpl == template_name)
    return tuple(indices)


def group_slots_by_property(field_name: str) -> dict[Any, frozenset[Tuple[str, int]]]:
    """Bucket (template, slot_index) keys by their ``field_name`` value."""
    rows = _ensure_cache()
    buckets: dict[Any, set[Tuple[str, int]]] = defaultdict(set)
    for key, row in rows.items():
        buckets[row.get(field_name)].add(key)
    return {key: frozenset(values) for key, values in buckets.items()}
