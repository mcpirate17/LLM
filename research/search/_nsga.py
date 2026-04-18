from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .native_nsga import compute_crowding_distances, compute_pareto_ranks

PARETO_FRONT_RANK = 1
DEFAULT_OBJECTIVES: List[Tuple[str, str]] = [("fitness", "max"), ("novelty", "max")]


def fast_non_dominated_sort(
    population: List[object],
    objectives: Sequence[Tuple[str, str]] = DEFAULT_OBJECTIVES,
) -> List[List[object]]:
    if not population:
        return []

    signs = np.array(
        [1.0 if direction == "max" else -1.0 for _, direction in objectives],
        dtype=np.float32,
    )
    attr_names = [name for name, _ in objectives]
    vals = (
        np.array(
            [
                [getattr(ind, attr_name) for attr_name in attr_names]
                for ind in population
            ],
            dtype=np.float32,
        )
        * signs
    )

    native_ranks = compute_pareto_ranks(vals)
    if native_ranks is not None:
        return _fronts_from_rank_array(population, native_ranks)
    return _fast_non_dominated_sort_in_python(population, vals)


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


def _fast_non_dominated_sort_in_python(
    population: List[object], vals: np.ndarray
) -> List[List[object]]:
    diff = vals[:, np.newaxis, :] - vals[np.newaxis, :, :]
    ge_all = np.all(diff >= 0, axis=2)
    gt_any = np.any(diff > 0, axis=2)
    dominates = ge_all & gt_any
    domination_count = dominates.sum(axis=0)

    fronts: List[List[object]] = []
    rank = 1
    remaining = np.ones(len(population), dtype=bool)

    while True:
        front_mask = remaining & (domination_count == 0)
        front_indices = np.where(front_mask)[0]
        if len(front_indices) == 0:
            break

        front: List[object] = []
        for i in front_indices:
            population[int(i)].pareto_rank = rank
            front.append(population[int(i)])
        fronts.append(front)

        remaining[front_indices] = False
        for i in front_indices:
            dominated_by_i = np.where(dominates[int(i)] & remaining)[0]
            domination_count[dominated_by_i] -= 1

        rank += 1

    return fronts


def assign_crowding_distance(
    front: List[object],
    objectives: Sequence[Tuple[str, str]] = DEFAULT_OBJECTIVES,
) -> None:
    if len(front) <= 2:
        for ind in front:
            ind.crowding_dist = float("inf")
        return

    attr_names = [attr for attr, _ in objectives]
    objective_matrix = np.asarray(
        [[getattr(ind, attr) for attr in attr_names] for ind in front],
        dtype=np.float32,
    )
    native_distances = compute_crowding_distances(objective_matrix)
    if native_distances is not None:
        for ind, distance in zip(front, native_distances):
            ind.crowding_dist = float(distance)
        return
    _assign_crowding_distance_in_python(front, objective_matrix)


def _assign_crowding_distance_in_python(
    front: List[object], objective_matrix: np.ndarray
) -> None:
    n = len(front)
    for ind in front:
        ind.crowding_dist = 0.0

    for objective_idx in range(objective_matrix.shape[1]):
        order = np.argsort(objective_matrix[:, objective_idx], kind="mergesort")
        obj_values = objective_matrix[order, objective_idx]
        span = float(obj_values[-1]) - float(obj_values[0])
        front[int(order[0])].crowding_dist = float("inf")
        front[int(order[-1])].crowding_dist = float("inf")
        if span <= 0.0:
            continue
        inv_span = 1.0 / span
        for pos in range(1, n - 1):
            idx = int(order[pos])
            if np.isinf(front[idx].crowding_dist):
                continue
            front[idx].crowding_dist += float(
                (obj_values[pos + 1] - obj_values[pos - 1]) * inv_span
            )


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
