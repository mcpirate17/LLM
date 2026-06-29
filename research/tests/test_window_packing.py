"""Unit tests for corpus-window packing (Workstream D inc. 4)."""

from __future__ import annotations

import numpy as np
import pytest

from research.training.window_packing import (
    find_doc_boundaries,
    pack_window_starts,
)


def test_find_doc_boundaries() -> None:
    tokens = np.array([1, 2, 9, 3, 4, 9, 5], dtype=np.int64)
    assert find_doc_boundaries(tokens, eot_id=9).tolist() == [2, 5]


def test_contiguous_starts_in_range_and_deterministic() -> None:
    starts = pack_window_starts(100, 8, 16, "contiguous", np.random.default_rng(0))
    assert starts.shape == (8,)
    assert (starts >= 0).all() and (starts <= 100 - 16).all()
    again = pack_window_starts(100, 8, 16, "contiguous", np.random.default_rng(0))
    assert np.array_equal(starts, again)


def test_doc_boundary_windows_never_cross_a_boundary() -> None:
    # boundaries at 20 and 45 -> docs [0,20), [21,45), [46,100)
    n = 100
    boundaries = np.array([20, 45], dtype=np.int64)
    window = 8
    rng = np.random.default_rng(1)
    starts = pack_window_starts(
        n, 200, window, "doc_boundary", rng, boundaries=boundaries
    )
    for s in starts:
        # no boundary may fall inside [s, s + window)
        assert not ((boundaries >= s) & (boundaries < s + window)).any()
        assert 0 <= s <= n - window


def test_doc_boundary_requires_boundaries() -> None:
    with pytest.raises(ValueError, match="requires document boundaries"):
        pack_window_starts(100, 4, 8, "doc_boundary", np.random.default_rng(0))


def test_doc_boundary_raises_when_no_doc_long_enough() -> None:
    # docs [0,5),[6,10),[11,15),[16,20) are all length <= 5; window 8 fits none.
    boundaries = np.array([5, 10, 15], dtype=np.int64)
    with pytest.raises(ValueError, match="no document is long enough"):
        pack_window_starts(
            20, 4, 8, "doc_boundary", np.random.default_rng(0), boundaries=boundaries
        )


def test_unimplemented_packs_fail_loud() -> None:
    for pack in ("length_bucketed", "best_fit"):
        with pytest.raises(NotImplementedError):
            pack_window_starts(100, 4, 8, pack, np.random.default_rng(0))


def test_unknown_pack_raises() -> None:
    with pytest.raises(ValueError, match="unknown pack"):
        pack_window_starts(100, 4, 8, "nope", np.random.default_rng(0))


def test_window_too_long_fails_loud() -> None:
    with pytest.raises(ValueError, match="corpus too short"):
        pack_window_starts(8, 4, 8, "contiguous", np.random.default_rng(0))
