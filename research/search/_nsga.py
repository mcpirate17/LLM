from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from research.frontier_kernel import crowding_distances, pareto_ranks

PARETO_FRONT_RANK = 1
DEFAULT_OBJECTIVES: List[Tuple[str, str]] = [("fitness", "max"), ("novelty", "max")]

# Named objective profiles for capability-aware multi-objective Pareto sort.
# Selected via search_objective_profile in RunConfig (default: "fitness_novelty"
# preserves prior behavior). Each entry is (attribute_name, "max"|"min").
# Missing attributes resolve to 0.0 via _safe_objective_value, so populations
# can opt into richer profiles without retrofitting every individual class.
OBJECTIVE_PROFILES: Dict[str, List[Tuple[str, str]]] = {
    "fitness_novelty": [("fitness", "max"), ("novelty", "max")],
    "capability": [
        ("fitness", "max"),
        ("ar_gate_score", "max"),
        ("binding_intermediate_auc", "max"),
        ("novelty", "max"),
    ],
    "balanced": [
        ("fitness", "max"),
        ("ar_gate_score", "max"),
        ("binding_intermediate_auc", "max"),
        ("param_efficiency", "max"),
        ("novelty", "max"),
    ],
    "efficient": [
        ("fitness", "max"),
        ("param_efficiency", "max"),
        ("compute_cost", "min"),
        ("novelty", "max"),
    ],
}


def _safe_objective_value(ind: object, attr_name: str) -> float:
    """Read an objective attribute with a default of 0.0 when missing.

    Lets callers extend objective profiles without crashing on heterogeneous
    populations (e.g. older individuals lacking ar_gate_score).
    """
    return float(getattr(ind, attr_name, 0.0) or 0.0)


def resolve_objectives(
    profile: Optional[str | Sequence[Tuple[str, str]]] = None,
) -> Sequence[Tuple[str, str]]:
    """Return an objective tuple list from a named profile or pass-through."""
    if profile is None:
        return DEFAULT_OBJECTIVES
    if isinstance(profile, str):
        return OBJECTIVE_PROFILES.get(profile, DEFAULT_OBJECTIVES)
    return profile


def fast_non_dominated_sort(
    population: List[object],
    objectives: Sequence[Tuple[str, str]] = DEFAULT_OBJECTIVES,
) -> List[List[object]]:
    if not population:
        return []

    maximize = tuple(direction == "max" for _, direction in objectives)
    attr_names = [name for name, _ in objectives]
    vals = np.asarray(
        [
            [_safe_objective_value(ind, attr_name) for attr_name in attr_names]
            for ind in population
        ],
        dtype=np.float32,
    )
    return _fronts_from_rank_array(
        population,
        pareto_ranks(vals, maximize=maximize),
    )


def _fronts_from_rank_array(
    population: List[object], ranks: np.ndarray
) -> List[List[object]]:
    fronts: List[List[object]] = []
    if ranks.size == 0:
        return fronts
    max_rank = int(np.max(ranks))
    for rank in range(1, max_rank + 1):
        indices = np.where(ranks == rank)[0]
        if indices.size == 0:
            continue
        front: List[object] = []
        for idx in indices:
            population[int(idx)].pareto_rank = rank
            front.append(population[int(idx)])
        fronts.append(front)
    return fronts


def assign_crowding_distance(
    front: List[object],
    objectives: Sequence[Tuple[str, str]] = DEFAULT_OBJECTIVES,
) -> None:
    attr_names = [attr for attr, _ in objectives]
    objective_matrix = np.asarray(
        [[_safe_objective_value(ind, attr) for attr in attr_names] for ind in front],
        dtype=np.float32,
    )
    for ind, distance in zip(front, crowding_distances(objective_matrix)):
        ind.crowding_dist = float(distance)


def nsga2_rank(
    population: List[object],
    objectives: Optional[Sequence[Tuple[str, str]]] = None,
) -> List[object]:
    if not population:
        return population

    objs = objectives if objectives is not None else DEFAULT_OBJECTIVES
    fronts = fast_non_dominated_sort(population, objs)
    for front in fronts:
        assign_crowding_distance(front, objs)
    population.sort(key=lambda x: (x.pareto_rank, -x.crowding_dist))
    return population
