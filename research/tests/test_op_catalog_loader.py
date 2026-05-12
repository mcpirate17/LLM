"""Tests for the op_property_catalog loader (handoff item F).

The loader makes ``research/meta_analysis.db.op_property_catalog`` reachable
from grammar code via a small, cached, fallback-aware API. These tests
pin: cache behaviour, fallback semantics, and the integration shape
against the actual repo meta DB (skipped when absent).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from research.synthesis._op_catalog_loader import (
    META_DB,
    group_ops_by_property,
    query_op_property,
    query_ops_by_category,
    query_ops_by_property,
    reset_cache,
)


pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _reset_cache_between_tests():
    reset_cache()
    yield
    reset_cache()


def test_fallback_when_meta_db_missing(tmp_path: Path) -> None:
    """All accessors must return their fallback gracefully if the DB is gone."""
    fake_db = tmp_path / "nonexistent.db"
    with patch("research.synthesis._op_catalog_loader.META_DB", fake_db):
        reset_cache()
        assert query_ops_by_category("attention", fallback=("a", "b")) == frozenset(
            {"a", "b"}
        )
        assert query_ops_by_property(
            "op_dynamical_has_state", lambda v: v == 1, fallback=("c",)
        ) == frozenset({"c"})
        assert query_op_property("anything", "any_field") is None
        assert group_ops_by_property("anything") == {}


def test_unknown_op_returns_none() -> None:
    """A real query for an op not in the catalog returns None for any field."""
    if not META_DB.exists():
        pytest.skip("meta_analysis.db not present in this checkout")
    assert query_op_property("__definitely_not_a_real_op__", "op_category") is None


def test_fallback_for_no_matches() -> None:
    """When the predicate matches zero ops, return fallback (not empty set)."""
    if not META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    fallback = ("fb_a", "fb_b")
    got = query_ops_by_category("__no_such_category__zzz__", fallback=fallback)
    assert got == frozenset(fallback)


def test_real_db_has_known_op() -> None:
    """Sanity check: an op we know exists is in the cache."""
    if not META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    # softmax_attention is in every recent catalog build.
    cat = query_op_property("softmax_attention", "op_category")
    assert cat is not None, "softmax_attention missing from op_property_catalog"


def test_group_by_category_returns_at_least_one_bucket() -> None:
    """The catalog always has multiple op_category values when present."""
    if not META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    buckets = group_ops_by_property("op_category")
    assert len(buckets) >= 3, f"expected multiple categories, got {sorted(buckets)}"
    # Every bucket has at least one op.
    for cat, ops in buckets.items():
        assert ops, f"bucket {cat!r} is empty"


def test_query_ops_by_property_predicate_lambda() -> None:
    """Predicate-based selection works on a real column."""
    if not META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    stateful = query_ops_by_property("op_dynamical_has_state", lambda v: v == 1)
    # Some ops in the catalog declare op_dynamical_has_state=1.
    assert stateful, "expected at least one stateful op declared in the catalog"


def test_cache_is_used_across_calls() -> None:
    """Two consecutive queries hit the same cached dict (no second DB open)."""
    if not META_DB.exists():
        pytest.skip("meta_analysis.db not present")
    # First call populates the cache.
    _ = query_op_property("softmax_attention", "op_category")
    # Patch _connect to ensure no further DB opens happen.
    with patch(
        "research.synthesis._op_catalog_loader._connect",
        side_effect=RuntimeError("cache should hide this call"),
    ):
        # Should not raise — cache is warm.
        v = query_op_property("softmax_attention", "op_category")
        assert v is not None
