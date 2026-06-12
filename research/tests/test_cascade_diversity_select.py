"""Diversity-preserving exploit selection in the CPU screening cascade (M3 wiring)."""

from __future__ import annotations

from research.tools.cpu_screening_cascade import MechProfile, Scored, _select


def _mech(novelty: float = 0.0, mech: float = 0.0) -> MechProfile:
    return MechProfile(
        n_mix=1,
        mixer_depth=1,
        sum_mem=0.0,
        n_global=0,
        alg_div=0,
        n_novel_mix=0,
        mech_score=mech,
        novelty=novelty,
        lit_family="x",
        lit_model="x",
        lit_match_type="novel",
    )


def _scored(
    fp: str,
    reach: float,
    dep: float,
    gate: float,
    cap: float | None,
    *,
    novelty: float = 0.0,
    measured: bool = True,
) -> Scored:
    s = Scored(
        fingerprint=fp,
        ops=["a"],
        profile=_mech(novelty=novelty),
        quality={},
        graph_dict={"nodes": {}},
    )
    s.measured_score = cap
    if measured:
        s.measured_descriptors = {
            "long_range_reach": reach,
            "content_dependence": dep,
            "content_match_gating": gate,
        }
    return s


def test_global_topk_collapses_to_same_niche() -> None:
    kept = [
        _scored("a", 0.4, 0.4, 0.2, 0.90),  # niche A
        _scored("b", 0.45, 0.45, 0.25, 0.85),  # niche A (same) — global #2
        _scored("c", 0.0, 0.0, 0.0, 0.30),  # niche B
    ]
    picked = {s.fingerprint for s in _select(kept, 2, 0, diversity_exploit=False)}
    assert picked == {"a", "b"}  # both top-cap, same behavior niche


def test_diversity_exploit_spreads_across_niches() -> None:
    kept = [
        _scored("a", 0.4, 0.4, 0.2, 0.90),  # niche A
        _scored("b", 0.45, 0.45, 0.25, 0.85),  # niche A (same)
        _scored("c", 0.0, 0.0, 0.0, 0.30),  # niche B
    ]
    picked = {s.fingerprint for s in _select(kept, 2, 0, diversity_exploit=True)}
    assert picked == {"a", "c"}  # distinct niches, not the global #2


def test_diversity_backfills_from_unmeasured_tail_last() -> None:
    kept = [
        _scored("a", 0.4, 0.4, 0.2, 0.90),  # niche A
        _scored("b", 0.45, 0.45, 0.25, 0.85),  # niche A (measured loser)
        _scored("c", 0.0, 0.0, 0.0, 0.30),  # niche B
        _scored("d", 0.0, 0.0, 0.0, None, measured=False),  # un-measured tail
    ]
    picked = [s.fingerprint for s in _select(kept, 3, 0, diversity_exploit=True)]
    # 2 niches fill first (a, c); the 3rd slot backfills the measured loser b
    # before the un-measured d.
    assert set(picked) == {"a", "c", "b"}
    assert "d" not in picked


def test_explore_reserve_still_added_by_novelty() -> None:
    kept = [
        _scored("a", 0.4, 0.4, 0.2, 0.90, novelty=0.0),
        _scored("c", 0.0, 0.0, 0.0, 0.30, novelty=9.0),  # most novel
    ]
    # exploit=1 (a by niche/cap), explore=1 reserve -> the novel c joins.
    picked = {s.fingerprint for s in _select(kept, 1, 1, diversity_exploit=True)}
    assert picked == {"a", "c"}
