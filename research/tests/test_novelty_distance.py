"""NM-10: geometric-novelty coordinate = distance from the softmax/attention basin.

The discovery machine must steer toward non-softmax geometry. These tests pin the
contract of ``research.synthesis.novelty_distance``: the basin is MEASURED and
reproducible, a softmax-shaped operator is inside its own basin, a qualitatively
different operator is far, the NM-11 twin score sharpens the signal, and — the
mission assertion — a MAP-Elites archive with the novelty axis KEEPS a
far-from-softmax mechanism even when its raw capability fitness is lower.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from research.synthesis.novelty_distance import (
    NOVELTY_AXIS_NAME,
    SOFTMAX_BASIN_NAMES,
    _SoftmaxAttentionBasin,
    augment_with_novelty,
    geometric_novelty,
    novelty_aware_axes,
    novelty_behavior_axis,
    softmax_basin_signatures,
)
from research.synthesis.physics_descriptors import (
    PHYSICS_DESCRIPTOR_NAMES,
    PhysicsDescriptorProbe,
)
from research.synthesis.quality_diversity import MapElitesArchive

_DIM = 16


def _probe() -> PhysicsDescriptorProbe:
    return PhysicsDescriptorProbe(dim=_DIM, n_seeds=2)


# ── basin measurement ────────────────────────────────────────────────────────


def test_basin_signatures_are_measured_and_complete() -> None:
    sigs = softmax_basin_signatures(dim=_DIM)
    assert [s.name for s in sigs] == list(SOFTMAX_BASIN_NAMES)
    for sig in sigs:
        assert set(sig.descriptors) == set(PHYSICS_DESCRIPTOR_NAMES)
        for v in sig.descriptors.values():
            assert isinstance(v, float)
            assert v == v and abs(v) != float("inf")  # finite


def test_basin_signatures_are_cached_per_dim() -> None:
    a = softmax_basin_signatures(dim=_DIM)
    b = softmax_basin_signatures(dim=_DIM)
    assert a is b  # same cached objects (fixed reference point)


def test_basin_signatures_are_reproducible_across_processes_like() -> None:
    # Re-measuring from a fresh cache key path must match: clear the cache,
    # re-measure, compare. (The basin seed + probe seeds are fixed.)
    from research.synthesis import novelty_distance as nd

    saved = nd.softmax_basin_signatures(dim=_DIM)
    assert nd._BASIN_CACHE.get(_DIM) is saved
    nd._BASIN_CACHE.pop(_DIM, None)
    try:
        fresh = nd._BASIN_CACHE.get(_DIM)
        assert fresh is None
        redo = nd.softmax_basin_signatures(dim=_DIM)
    finally:
        nd._BASIN_CACHE[_DIM] = saved  # restore
    for a, b in zip(saved, redo):
        assert a.name == b.name
        for k in PHYSICS_DESCRIPTOR_NAMES:
            assert a.descriptors[k] == pytest.approx(b.descriptors[k], abs=1e-9)


# ── distance contract ─────────────────────────────────────────────────────────


def test_basin_signature_is_in_its_own_basin() -> None:
    """The measured softmax / mean-pool fingerprints are distance 0 from the
    nearest basin (themselves) — the basin is well-defined."""
    sigs = softmax_basin_signatures(dim=_DIM)
    for sig in sigs:
        assert geometric_novelty(sig.descriptors, basins=sigs) == pytest.approx(
            0.0, abs=1e-12
        )


def test_softmax_attention_is_closer_to_basin_than_identity() -> None:
    """A second, differently-seeded softmax attention is a region-mate of the
    basin; a pointwise identity op is geometrically farther. This is the core
    claim: 'softmax-shaped' is a region, not one point, and a genuinely different
    operator lands farther away."""
    probe = _probe()
    sigs = softmax_basin_signatures(dim=_DIM)

    torch.manual_seed(777)  # different seed -> different projections, same class
    other_softmax = _SoftmaxAttentionBasin(_DIM).eval()
    identity = nn.Identity()
    soft_desc = probe.describe_operator(other_softmax)
    id_desc = probe.describe_operator(identity)

    soft_novelty = geometric_novelty(soft_desc, basins=sigs)
    id_novelty = geometric_novelty(id_desc, basins=sigs)
    assert soft_novelty < id_novelty  # softmax beats identity on novelty distance
    assert id_novelty > 0.0
    assert soft_novelty > 0.0  # distinct random instance, not literally the basin


def test_twin_score_sharpens_but_does_not_erase() -> None:
    """NM-11 fold-in: twin_score=1 collapses novelty toward the floor, twin_score=0
    is a no-op (factor 1.0), and the 0.25 floor keeps geometry audible."""
    sigs = softmax_basin_signatures(dim=_DIM)
    far = {name: 1.0 for name in PHYSICS_DESCRIPTOR_NAMES}  # identity-like
    base = geometric_novelty(far, basins=sigs, twin_score=None)
    twin_zero = geometric_novelty(far, basins=sigs, twin_score=0.0)
    twin_one = geometric_novelty(far, basins=sigs, twin_score=1.0)
    assert twin_zero == pytest.approx(base)  # 0.25 + 0.75*(1-0) = 1.0
    assert twin_one == pytest.approx(base * 0.25)  # 0.25 + 0.75*(1-1) = 0.25
    assert twin_one < twin_zero  # confirmed softmax-twin -> less novel


def test_twin_score_rejects_out_of_range() -> None:
    sigs = softmax_basin_signatures(dim=_DIM)
    desc = {name: 0.5 for name in PHYSICS_DESCRIPTOR_NAMES}
    with pytest.raises(ValueError):
        geometric_novelty(desc, basins=sigs, twin_score=1.5)
    with pytest.raises(ValueError):
        geometric_novelty(desc, basins=sigs, twin_score=-0.1)


def test_missing_axis_fails_loud() -> None:
    sigs = softmax_basin_signatures(dim=_DIM)
    partial = {"perm_equivariance": 0.5}  # missing the other physics axes
    with pytest.raises(KeyError):
        geometric_novelty(partial, basins=sigs)


def test_empty_basins_rejected() -> None:
    desc = {name: 0.5 for name in PHYSICS_DESCRIPTOR_NAMES}
    with pytest.raises(ValueError):
        geometric_novelty(desc, basins=[])


# ── archive integration ───────────────────────────────────────────────────────


def test_augment_with_novelty_preserves_and_adds() -> None:
    sigs = softmax_basin_signatures(dim=_DIM)
    phys = {name: 0.5 for name in PHYSICS_DESCRIPTOR_NAMES}
    out = augment_with_novelty(phys, basins=sigs)
    assert NOVELTY_AXIS_NAME in out
    assert set(phys) <= set(out)
    assert phys == {k: out[k] for k in phys}  # originals unchanged
    # input not mutated
    assert NOVELTY_AXIS_NAME not in phys


def test_novelty_axis_bins_coarsely() -> None:
    axis = novelty_behavior_axis()
    assert axis.name == NOVELTY_AXIS_NAME
    assert axis.n_bins == 5
    assert axis.bin_of(0.0) == 0  # inside the basin
    assert axis.bin_of(1.0) == 1  # adjacent
    assert axis.bin_of(3.0) == 2  # far bulk
    assert axis.bin_of(5.0) == 3  # very far
    assert axis.bin_of(50.0) == 4  # blow-up quarantine


# G3 drift pin: the 30 deep-registry elites from the 8000-iter
# `deep_registry_discovery` run, with their MEASURED standardized novelty
# distances (recomputed by `research/tools/audit_novelty_axis_distribution.py`,
# committed `bbec2a4a`; source JSON `research/reports/novelty_axis_audit_*.json`
# is auto-pruned so the values are pinned here, not loaded). Each pair is
# (raw_novelty_distance, expected_bin_under_the_5-bin_edges). If the production
# edges drift away from (0.75, 1.75, 4.0, 16.0) the per-elite bins diverge from
# the independent bisect below and this test fails loud — same silent-drift class
# as the gates-5/6/8 incident.
_DEEP_REGISTRY_ELITE_DISTANCES: tuple[float, ...] = (
    175.588571906197,
    29.74163646760285,
    2.8046893867849847,
    4.029766346577487,
    2.6266030302240493,
    2.1151659717942954,
    2.7826606325871355,
    2.1825526821486583,
    2.21554600350526,
    2.271191705323358,
    2.1376905514433133,
    1.9121607534654048,
    1.8527967093709785,
    1.6893305225332376,
    1.3040239577997164,
    13.753866814069712,
    2.4300017731544235,
    2.126444897983714,
    1.258142649429461,
    1.5558333603764678,
    1.8053560793985188,
    1.5352605847657235,
    1.7318472511837293,
    1.6261354233713146,
    1.6758981989437132,
    3.298213515570017,
    1.3317225093653962,
    1.364470410112312,
    1.4917347218151247,
    1.9289865875267411,
)


def test_novelty_axis_five_bin_drift_pin() -> None:
    """The 5-bin novelty edges must stay pinned to (0.75, 1.75, 4.0, 16.0).

    The two extra edges sit in the natural data gaps 3.298→4.030 and
    13.75→29.74; the >=16 bin quarantines numerically-expansive blow-up
    descriptors so they cannot squat on the far niches stable novel mechanisms
    should own (G3 QC1 verdict, `research/notes/nm_verification_split_plan_*.md`).
    """
    import bisect

    pinned_edges = (0.75, 1.75, 4.0, 16.0)
    axis = novelty_behavior_axis()
    assert tuple(axis.edges) == pinned_edges
    assert axis.n_bins == 5

    histogram = [0, 0, 0, 0, 0]
    for dist in _DEEP_REGISTRY_ELITE_DISTANCES:
        expected = bisect.bisect_right(pinned_edges, dist)
        assert axis.bin_of(dist) == expected, (
            f"novelty distance {dist} binned to {axis.bin_of(dist)}, "
            f"expected {expected} under edges {pinned_edges} — edges drifted"
        )
        histogram[expected] += 1

    # Distribution the QC1 verdict recorded: no elite inside the basin, 11
    # adjacent, a 15-elite far bulk, and 2+2 splitting the very-far / blow-up
    # tail that the old 3-bin axis lumped into one 97×-wide bin.
    assert histogram == [0, 11, 15, 2, 2]


def test_novelty_aware_axes_extend_physics() -> None:
    from research.synthesis.physics_descriptors import physics_behavior_axes

    axes = novelty_aware_axes()
    assert [a.name for a in axes[:-1]] == [a.name for a in physics_behavior_axes()]
    assert axes[-1].name == NOVELTY_AXIS_NAME


def test_archive_keeps_far_from_softmax_mechanism() -> None:
    """THE mission assertion: a far-from-softmax mechanism must SURVIVE in the
    archive even when its capability fitness is lower than a near-softmax one.
    MAP-Elites keeps the best per niche; with the novelty axis, the far niche is
    a distinct cell the low-fitness far operator owns — so softmax-shaped
    candidates cannot crowd out geometrically novel ones."""
    sigs = softmax_basin_signatures(dim=_DIM)
    archive = MapElitesArchive(axes=novelty_aware_axes())

    near = augment_with_novelty(dict(sigs[0].descriptors), basins=sigs)  # in-basin
    far_phys = {name: 1.0 for name in PHYSICS_DESCRIPTOR_NAMES}  # identity-like
    far = augment_with_novelty(far_phys, basins=sigs)

    # near and far differ on perm_equivariance (0.x vs 1.0) AND novelty -> distinct
    # niches; give the far one a LOWER capability fitness.
    assert archive.add("near", near, fitness=0.90)
    assert archive.add("far", far, fitness=0.45)

    keys = {e.key for e in archive.elites}
    assert keys == {"near", "far"}  # both niches held; far not evicted
    far_elite = next(e for e in archive.elites if e.key == "far")
    assert far_elite.fitness == 0.45  # preserved despite the higher-fitness near
