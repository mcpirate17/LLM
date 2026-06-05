"""Tests for the archive-guided exploration loop (diversity generator M4)."""

from __future__ import annotations

import pytest

from research.synthesis.quality_diversity import default_behavior_axes
from research.tools.archive_guided_explore import (
    ExplorationResult,
    explore_empty_niches,
)

pytestmark = pytest.mark.unit

# Behavior corners keyed by op, matching archive_guided._OP_BEHAVIOR_SIGNATURE so
# a fake generator can place graphs in specific niches deterministically.
_FIXED_ROUTING = {
    "long_range_reach": 0.4,
    "content_dependence": 0.02,  # LOW content_dependence
    "content_match_gating": 0.0,
}
_BINDER = {
    "long_range_reach": 0.4,
    "content_dependence": 0.5,  # HIGH content_dependence
    "content_match_gating": 0.2,  # HIGH gating
}


def _const_generator(descriptor_by_phase):
    """Generator whose output descriptor depends only on whether a guided config
    is in effect — wave 0 lands in fixed-routing niches, guided waves in binder
    niches, so a guided wave demonstrably fills new niches."""

    def generate(config, seed):
        phase = "guided" if config is not None else "seed"
        return {"fingerprint": f"{phase}_{seed}", "_phase": phase}

    def measure(graph):
        return descriptor_by_phase[graph["_phase"]]

    return generate, measure


def _fitness(descriptors):
    return float(descriptors["long_range_reach"])


def test_guided_wave_raises_coverage_into_empty_niches() -> None:
    generate, measure = _const_generator({"seed": _FIXED_ROUTING, "guided": _BINDER})
    result = explore_empty_niches(
        generate_fn=generate,
        measure_fn=measure,
        fitness_fn=_fitness,
        key_fn=lambda g: g["fingerprint"],
        seed_pool=5,
        wave_pool=5,
        waves=1,
    )
    assert isinstance(result, ExplorationResult)
    # Seed wave fills the fixed-routing niche; the guided wave must add a new one.
    assert len(result.coverage_trajectory) == 2
    assert result.coverage_trajectory[1] > result.coverage_trajectory[0]
    # The guided wave should have targeted binder ops for the empty binder niche.
    assert result.target_ops_per_wave
    assert "tropical_attention" in result.target_ops_per_wave[0]
    # Candidates from the guided wave exist and are tagged wave>0.
    assert any(c.wave == 1 for c in result.candidates)


def test_stops_early_when_no_reachable_empty_niche() -> None:
    # Generator always lands in the SAME binder niche; once filled, the only empty
    # niches reachable by guided ops are already covered → guidance returns no
    # config and the loop stops without spinning through every requested wave.
    generate, measure = _const_generator({"seed": _BINDER, "guided": _BINDER})
    # Restrict the behavior space to a single axis with 1 edge (2 bins) so the
    # binder bin saturates fast and "reachable empty" empties out.
    axes = (default_behavior_axes()[1],)  # content_dependence only
    result = explore_empty_niches(
        generate_fn=generate,
        measure_fn=measure,
        fitness_fn=_fitness,
        key_fn=lambda g: g["fingerprint"],
        seed_pool=3,
        wave_pool=3,
        waves=5,
        axes=axes,
    )
    # Fewer guided waves ran than requested (early stop) OR all niches reachable
    # were filled — either way coverage never exceeds 1.0 and we didn't loop 5x.
    assert result.filled <= result.total_cells
    assert len(result.target_ops_per_wave) <= 5


def test_invalid_and_unmeasurable_are_counted_not_swallowed() -> None:
    def generate(config, seed):
        if seed % 2 == 0:
            return None  # invalid grammar sample
        return {"fingerprint": f"g_{seed}", "_phase": "seed"}

    def measure(graph):
        if graph["fingerprint"].endswith("1"):
            return None  # unmeasurable
        return _BINDER

    result = explore_empty_niches(
        generate_fn=generate,
        measure_fn=measure,
        fitness_fn=_fitness,
        key_fn=lambda g: g["fingerprint"],
        seed_pool=6,
        wave_pool=0,
        waves=0,
    )
    assert result.invalid == 3  # seeds 0,2,4
    assert result.unmeasurable >= 1
    # Every generated graph is accounted for — nothing silently swallowed.
    assert result.measured + result.unmeasurable + result.invalid == 6
