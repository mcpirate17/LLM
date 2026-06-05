"""Tests for archive-guided generation (diversity generator M4)."""

from __future__ import annotations

import pytest

from research.synthesis._motifs_fab import fab_invention_ops
from research.synthesis.archive_guided import (
    _OP_BEHAVIOR_SIGNATURE,
    _registered_ops,
    archive_guidance,
    exploration_config_from_archive,
)
from research.synthesis.quality_diversity import MapElitesArchive, default_behavior_axes


def _full_archive() -> MapElitesArchive:
    """An archive with EVERY niche filled (no empty niches to target)."""

    archive = MapElitesArchive()
    axes = default_behavior_axes()
    # Place a candidate at the centre of each bin on each axis -> fills every cell.
    import itertools

    centres = []
    for axis in axes:
        # one representative descriptor value per bin
        vals = []
        lo_edge = axis.edges[0]
        vals.append(lo_edge - 0.01)  # bin 0
        for i in range(len(axis.edges) - 1):
            vals.append((axis.edges[i] + axis.edges[i + 1]) / 2)  # interior bins
        vals.append(axis.edges[-1] + 0.01)  # last bin
        centres.append(vals)
    for combo in itertools.product(*centres):
        desc = {axis.name: v for axis, v in zip(axes, combo)}
        archive.add(key=str(combo), descriptors=desc, fitness=1.0)
    return archive


def test_every_fab_invention_op_has_a_behavior_signature() -> None:
    # M4 must be able to target every fab invention; the two tables stay in sync.
    missing = set(fab_invention_ops()) - set(_OP_BEHAVIOR_SIGNATURE)
    assert not missing, f"fab ops without a behavior signature: {missing}"


def test_signature_ops_all_exist_in_registry() -> None:
    # Fail-safe: never emit a target op the grammar can't build.
    registered = _registered_ops()
    assert registered, "no signature op resolved against the primitive registry"
    assert "tropical_attention" in registered


def test_full_archive_emits_no_targets() -> None:
    guidance = archive_guidance(_full_archive())
    assert guidance.coverage == pytest.approx(1.0)
    assert guidance.reachable_empty == 0
    assert guidance.target_ops == frozenset()
    config, _ = exploration_config_from_archive(_full_archive())
    assert config is None  # nothing to explore -> caller keeps base grammar


def test_forced_empty_binder_niche_targets_binder_ops() -> None:
    """A forced-empty strong-binder niche shifts exploration toward binder ops."""

    archive = _full_archive()
    axes = default_behavior_axes()
    # The strong-binder corner = top bin on all three axes (HIGH/HIGH/HIGH).
    binder_niche = tuple(axis.n_bins - 1 for axis in axes)
    # Evict its incumbent so the niche reads as empty.
    archive._cells.pop(binder_niche, None)
    assert binder_niche in archive.empty_niches()

    guidance = archive_guidance(archive)
    assert guidance.reachable_empty >= 1
    # Binder ops (HIGH,HIGH,HIGH signature) must be recommended for this niche.
    assert "tropical_attention" in guidance.target_ops
    assert {"product_key_memory", "role_slot_attention"} & guidance.target_ops
    # A pure fixed-routing SSM op (LOW content_dependence) is NOT a binder fit.
    assert "state_space" not in guidance.target_ops
    # The directive maps to a usable exploration grammar.
    config, _ = exploration_config_from_archive(archive)
    assert config is not None
    assert config.exploration_targets == guidance.target_ops
    assert config.exploration_boost_factor > 1.0


def test_underfilled_niche_targeted_when_threshold_set() -> None:
    # A filled-but-low-fitness binder niche is targeted when underfilled_below is set.
    archive = _full_archive()
    axes = default_behavior_axes()
    binder_niche = tuple(axis.n_bins - 1 for axis in axes)
    # Re-occupy the binder corner with a weak elite (fitness below threshold).
    bounds = archive.niche_bounds(binder_niche)
    weak_desc = {
        name: (lo + 0.01) if lo is not None else (hi - 0.01 if hi is not None else 0.0)
        for name, (lo, hi) in bounds.items()
    }
    archive._cells.pop(binder_niche, None)
    archive.add(key="weak_binder", descriptors=weak_desc, fitness=0.05)
    empty_only = archive_guidance(archive)
    with_underfilled = archive_guidance(archive, underfilled_below=0.5)
    assert "tropical_attention" not in empty_only.target_ops  # niche is filled
    assert "tropical_attention" in with_underfilled.target_ops  # low fitness → target


def test_boost_scales_with_coverage_gap() -> None:
    # Emptier archive -> stronger exploration boost (bounded by max_boost).
    sparse = MapElitesArchive()
    sparse.add(
        key="one",
        descriptors={
            "long_range_reach": 0.3,
            "content_dependence": 0.4,
            "content_match_gating": 0.2,
        },
        fitness=1.0,
    )
    full = archive_guidance(_full_archive(), base_boost=4.0)
    lean = archive_guidance(sparse, base_boost=4.0)
    assert lean.boost_factor > full.boost_factor or full.target_ops == frozenset()
    assert lean.boost_factor <= 12.0


def test_unreachable_niche_left_empty() -> None:
    # A niche no op's signature can reach within radius is not chased.
    archive = MapElitesArchive()
    # Radius 0: only exact-signature niches are reachable, the rest are unreachable.
    guidance = archive_guidance(archive, radius=0)
    assert guidance.unreachable_empty > 0
