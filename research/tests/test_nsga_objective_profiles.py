"""Tests for the expanded NSGA-II objective profiles.

Default behavior (fitness + novelty) must remain unchanged. Capability
profiles must extend the Pareto comparison without crashing on populations
that lack the new attributes.
"""

from __future__ import annotations

from types import SimpleNamespace

from research.search._nsga import (
    DEFAULT_OBJECTIVES,
    OBJECTIVE_PROFILES,
    _safe_objective_value,
    fast_non_dominated_sort,
    nsga2_rank,
    resolve_objectives,
)


def _ind(**attrs):
    obj = SimpleNamespace(**attrs)
    obj.pareto_rank = 0
    obj.crowding_dist = 0.0
    return obj


def test_default_objectives_unchanged():
    assert DEFAULT_OBJECTIVES == [("fitness", "max"), ("novelty", "max")]


def test_resolve_objectives_named_profile():
    cap = resolve_objectives("capability")
    keys = [name for name, _ in cap]
    assert "ar_gate_score" in keys
    assert "binding_intermediate_auc" in keys
    assert "fitness" in keys


def test_resolve_objectives_unknown_falls_back_to_default():
    assert resolve_objectives("doesnotexist") == DEFAULT_OBJECTIVES


def test_resolve_objectives_passthrough_sequence():
    custom = [("foo", "max"), ("bar", "min")]
    assert list(resolve_objectives(custom)) == custom


def test_safe_objective_value_returns_zero_for_missing_attr():
    bare = _ind(fitness=0.7)
    assert _safe_objective_value(bare, "fitness") == 0.7
    assert _safe_objective_value(bare, "ar_gate_score") == 0.0


def test_safe_objective_value_handles_none():
    holder = _ind(fitness=None)
    assert _safe_objective_value(holder, "fitness") == 0.0


def test_capability_profile_with_partial_population_does_not_crash():
    """An older individual without ar_gate_score must still get sorted."""
    a = _ind(fitness=0.9, novelty=0.4, ar_gate_score=0.6, binding_intermediate_auc=0.5)
    b = _ind(fitness=0.7, novelty=0.5)  # missing capability metrics
    c = _ind(fitness=0.5, novelty=0.6, ar_gate_score=0.8, binding_intermediate_auc=0.7)
    fronts = fast_non_dominated_sort(
        [a, b, c], objectives=OBJECTIVE_PROFILES["capability"]
    )
    assert len(fronts) >= 1
    # Pareto rank annotated on every individual
    assert all(getattr(ind, "pareto_rank", None) for ind in (a, b, c))


def test_default_profile_pareto_unchanged_for_2d_case():
    a = _ind(fitness=0.9, novelty=0.5)
    b = _ind(fitness=0.5, novelty=0.9)
    c = _ind(fitness=0.4, novelty=0.4)  # dominated
    nsga2_rank([a, b, c])
    assert a.pareto_rank == 1
    assert b.pareto_rank == 1
    assert c.pareto_rank > 1


def test_evolution_config_propagates_objective_profile():
    """EvolutionConfig.objective_profile flows into nsga2_rank via resolve_objectives."""
    from research.search.evolution import EvolutionConfig

    cfg = EvolutionConfig()
    assert cfg.objective_profile is None  # default preserves prior behavior
    cfg_cap = EvolutionConfig(objective_profile="capability")
    assert cfg_cap.objective_profile == "capability"
    resolved = resolve_objectives(cfg_cap.objective_profile)
    keys = [name for name, _ in resolved]
    assert "ar_gate_score" in keys
    assert "binding_intermediate_auc" in keys
