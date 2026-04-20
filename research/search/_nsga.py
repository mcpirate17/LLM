from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from research.frontier_kernel import crowding_distances, pareto_ranks

PARETO_FRONT_RANK = 1
DEFAULT_OBJECTIVES: List[Tuple[str, str]] = [("fitness", "max"), ("novelty", "max")]


def fast_non_dominated_sort(
    population: List[object],
    objectives: Sequence[Tuple[str, str]] = DEFAULT_OBJECTIVES,
) -> List[List[object]]:
    if not population:
        return []

    maximize = tuple(direction == "max" for _, direction in objectives)
    attr_names = [name for name, _ in objectives]
    vals = np.asarray(
        [[getattr(ind, attr_name) for attr_name in attr_names] for ind in population],
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
        [[getattr(ind, attr) for attr in attr_names] for ind in front],
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
