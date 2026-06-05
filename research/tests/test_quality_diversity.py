"""Tests for the MAP-Elites quality-diversity archive (diversity generator M3)."""

from __future__ import annotations

import pytest

from research.synthesis.quality_diversity import (
    BehaviorAxis,
    MapElitesArchive,
    select_diverse,
)


def _desc(reach: float, dep: float, gate: float) -> dict[str, float]:
    return {
        "long_range_reach": reach,
        "content_dependence": dep,
        "content_match_gating": gate,
    }


# ── axis geometry ────────────────────────────────────────────────────
def test_axis_binning_is_right_open() -> None:
    axis = BehaviorAxis("x", (0.05, 0.25))
    assert axis.n_bins == 3
    assert axis.bin_of(0.0) == 0
    assert axis.bin_of(0.04) == 0
    assert axis.bin_of(0.05) == 1  # on-edge goes up
    assert axis.bin_of(0.2) == 1
    assert axis.bin_of(0.25) == 2
    assert axis.bin_of(0.9) == 2


def test_axis_bounds_have_open_ends() -> None:
    axis = BehaviorAxis("x", (0.05, 0.25))
    assert axis.bounds_of(0) == (None, 0.05)
    assert axis.bounds_of(1) == (0.05, 0.25)
    assert axis.bounds_of(2) == (0.25, None)


def test_axis_rejects_unsorted_or_empty_edges() -> None:
    with pytest.raises(ValueError):
        BehaviorAxis("x", ())
    with pytest.raises(ValueError):
        BehaviorAxis("x", (0.3, 0.1))


def test_default_archive_has_27_cells() -> None:
    archive = MapElitesArchive()
    assert len(archive.axes) == 3
    assert archive.total_cells == 27


# ── best-per-niche replacement ───────────────────────────────────────
def test_keeps_best_per_niche_not_global_topk() -> None:
    archive = MapElitesArchive()
    # Two strong candidates in the SAME niche, one weak in a DISTINCT niche.
    archive.add("a", _desc(0.4, 0.4, 0.2), fitness=0.9)
    archive.add("b", _desc(0.45, 0.45, 0.25), fitness=0.8)  # same top niche as a
    archive.add("c", _desc(0.0, 0.0, 0.0), fitness=0.1)  # non-binder niche

    elites = archive.elites
    keys = {e.key for e in elites}
    # Global top-2 would be {a, b}; diversity keeps the distinct niches {a, c}.
    assert keys == {"a", "c"}
    assert archive.filled == 2


def test_strict_improvement_only_ties_keep_incumbent() -> None:
    archive = MapElitesArchive()
    assert archive.add("a", _desc(0.4, 0.4, 0.2), 0.5) is True
    assert archive.add("b", _desc(0.41, 0.41, 0.21), 0.5) is False  # tie -> keep a
    assert archive.add("c", _desc(0.42, 0.42, 0.22), 0.6) is True  # beats -> replace
    assert archive._cells[archive.niche_for(_desc(0.4, 0.4, 0.2))].key == "c"


def test_best_and_coverage() -> None:
    archive = MapElitesArchive()
    archive.add("a", _desc(0.4, 0.4, 0.2), 0.9)
    archive.add("c", _desc(0.0, 0.0, 0.0), 0.1)
    best = archive.best()
    assert best is not None and best.key == "a"
    assert archive.coverage() == pytest.approx(2 / 27)


# ── niche bookkeeping ────────────────────────────────────────────────
def test_missing_descriptor_fails_loud() -> None:
    archive = MapElitesArchive()
    with pytest.raises(KeyError):
        archive.niche_for({"long_range_reach": 0.4})


def test_empty_niches_complement_filled() -> None:
    archive = MapElitesArchive()
    archive.add("a", _desc(0.4, 0.4, 0.2), 0.9)
    empties = archive.empty_niches()
    assert len(empties) == archive.total_cells - archive.filled
    assert archive.niche_for(_desc(0.4, 0.4, 0.2)) not in empties


def test_niche_bounds_round_trip() -> None:
    archive = MapElitesArchive()
    niche = archive.niche_for(_desc(0.4, 0.4, 0.2))
    bounds = archive.niche_bounds(niche)
    # 0.4 lands in the top bin of every default axis -> open upper end.
    for lo, hi in bounds.values():
        assert hi is None
        assert lo is not None and lo <= 0.4


# ── select_diverse ───────────────────────────────────────────────────
def _rec(rid: str, reach: float, dep: float, gate: float, fit: float) -> dict:
    return {"id": rid, "d": _desc(reach, dep, gate), "fit": fit}


def _by_id(r: dict) -> str:
    return r["id"]


def _by_desc(r: dict) -> dict[str, float]:
    return r["d"]


def _by_fit(r: dict) -> float:
    return r["fit"]


def test_select_diverse_spreads_then_backfills() -> None:
    records = [
        _rec("a", 0.4, 0.4, 0.2, 0.90),
        _rec("b", 0.45, 0.45, 0.25, 0.85),  # same niche as a (loses)
        _rec("c", 0.0, 0.0, 0.0, 0.30),  # distinct niche
        _rec("d", 0.1, 0.2, 0.05, 0.20),  # distinct niche
    ]
    # k=2: pure best-per-niche -> a (0.90) and the next-best niche elite c (0.30),
    # NOT b (0.85) which shares a's niche.
    picked = select_diverse(
        records, k=2, descriptors=_by_desc, fitness=_by_fit, key=_by_id
    )
    assert [r["id"] for r in picked] == ["a", "c"]

    # k=4 with 3 filled niches -> 3 elites + 1 backfill (highest-fitness loser, b).
    picked_all = select_diverse(
        records, k=4, descriptors=_by_desc, fitness=_by_fit, key=_by_id
    )
    assert set(r["id"] for r in picked_all) == {"a", "c", "d", "b"}


def test_select_diverse_no_backfill_caps_at_niche_count() -> None:
    records = [
        _rec("a", 0.4, 0.4, 0.2, 0.9),
        _rec("b", 0.45, 0.45, 0.25, 0.8),  # same niche
    ]
    picked = select_diverse(
        records,
        k=5,
        descriptors=lambda r: r["d"],
        fitness=lambda r: r["fit"],
        key=lambda r: r["id"],
        backfill=False,
    )
    assert [r["id"] for r in picked] == ["a"]


def test_select_diverse_empty_and_zero_k() -> None:
    assert (
        select_diverse([], k=3, descriptors=_by_desc, fitness=_by_fit, key=_by_id) == []
    )
    assert (
        select_diverse(
            [_rec("a", 0.4, 0.4, 0.2, 0.9)],
            k=0,
            descriptors=_by_desc,
            fitness=_by_fit,
            key=_by_id,
        )
        == []
    )


def test_custom_axes_respected() -> None:
    axes = [BehaviorAxis("long_range_reach", (0.5,))]  # 2 bins, one descriptor
    archive = MapElitesArchive(axes=tuple(axes))
    assert archive.total_cells == 2
    archive.add("lo", {"long_range_reach": 0.1}, 0.5)
    archive.add("hi", {"long_range_reach": 0.9}, 0.4)
    assert archive.filled == 2  # distinct niches despite lower fitness on hi
