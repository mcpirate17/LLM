"""Tests for the slot_property_catalog loader.

Pins API shape, cache semantics, fallback behaviour, and a sanity check
against the live ``research/meta_analysis.db`` schema (skipped when the
DB is absent in a test environment).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from research.synthesis._slot_catalog_loader import (
    META_DB,
    group_slots_by_property,
    query_slot_property,
    query_slot_row,
    query_slots_by_property,
    reset_cache,
    slot_accepts,
    slot_classes_for,
    slots_for_template,
)


pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _reset_cache_between_tests():
    reset_cache()
    yield
    reset_cache()


def test_fallback_when_meta_db_missing(tmp_path: Path) -> None:
    fake_db = tmp_path / "nonexistent.db"
    with patch("research.synthesis._slot_catalog_loader.META_DB", fake_db):
        reset_cache()
        assert query_slot_property("any", 0, "any_field") is None
        assert query_slot_row("any", 0) is None
        assert slot_accepts("any", 0, "attention") is False
        assert slot_classes_for("any", 0, fallback=("fb",)) == ("fb",)
        assert query_slots_by_property(
            "slot_accepts_attention", lambda v: v == 1, fallback=(("fb_t", 0),)
        ) == frozenset({("fb_t", 0)})
        assert slots_for_template("any") == ()
        assert group_slots_by_property("any") == {}


def test_unknown_slot_returns_none() -> None:
    if not META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    assert query_slot_property("__no_such_template__", 0, "slot_role") is None
    assert query_slot_row("__no_such_template__", 0) is None


def test_unknown_class_name_returns_false() -> None:
    """``slot_accepts`` returns False for an unknown short class name."""
    if not META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    # Pick any real (template, slot) from the catalog.
    rows = query_slots_by_property("slot_index", lambda v: v == 0)
    assert rows, "expected at least one slot at index 0"
    template, idx = next(iter(rows))
    assert slot_accepts(template, idx, "__not_a_real_class__") is False


def test_real_db_has_known_slot() -> None:
    if not META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    # adaptive_conv_ffn is in every recent catalog build.
    row = query_slot_row("adaptive_conv_ffn", 0)
    assert row is not None
    assert row.get("template_name") == "adaptive_conv_ffn"
    assert row.get("slot_index") == 0


def test_slot_classes_for_parses_json() -> None:
    if not META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    classes = slot_classes_for("adaptive_conv_ffn", 0)
    assert isinstance(classes, tuple)
    assert classes, "expected non-empty class list for adaptive_conv_ffn.slot0"
    assert all(isinstance(c, str) for c in classes)


def test_slot_classes_for_falls_back_on_missing() -> None:
    if not META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    classes = slot_classes_for("__no_such_template__", 0, fallback=("fb",))
    assert classes == ("fb",)


def test_slots_for_template_ordered() -> None:
    if not META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    indices = slots_for_template("adaptive_conv_ffn")
    assert indices, "expected at least one slot for adaptive_conv_ffn"
    assert list(indices) == sorted(indices), "slot indices must be ascending"


def test_slot_accepts_attention_predicate() -> None:
    if not META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    attn_slots = query_slots_by_property("slot_accepts_attention", lambda v: v == 1)
    assert attn_slots, "expected at least one attention-accepting slot"
    # Each result is a (template, slot_index) tuple.
    for entry in list(attn_slots)[:3]:
        assert isinstance(entry, tuple) and len(entry) == 2
        assert isinstance(entry[0], str) and isinstance(entry[1], int)


def test_group_by_role_family_returns_buckets() -> None:
    if not META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    buckets = group_slots_by_property("slot_role_family")
    assert len(buckets) >= 3, f"expected multiple role families, got {sorted(buckets)}"
    for _family, slots in buckets.items():
        assert slots, "every bucket should have at least one slot"


def test_cache_is_used_across_calls() -> None:
    """Two consecutive queries hit the cache (no second DB open)."""
    if not META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    # Warm cache.
    _ = query_slot_row("adaptive_conv_ffn", 0)
    with patch(
        "research.synthesis._slot_catalog_loader._connect",
        side_effect=RuntimeError("cache should prevent this"),
    ):
        # Should not raise — cache is warm.
        v = query_slot_row("adaptive_conv_ffn", 0)
        assert v is not None
