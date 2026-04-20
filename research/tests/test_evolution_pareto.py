"""Tests for vectorized NSGA-II fast_non_dominated_sort."""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from research.frontier_kernel import crowding_distances, pareto_ranks
from research.search.evolution import Individual
from research.search._nsga import (
    assign_crowding_distance,
    fast_non_dominated_sort,
)
from research.search.native_nsga import (
    compute_pareto_frontier_mask,
    compute_crowding_distances,
    compute_pareto_ranks,
    load_native_nsga_lib,
    reset_native_nsga_lib,
)

pytestmark = pytest.mark.unit


def _make_individual(fitness: float, novelty: float) -> Individual:
    graph = MagicMock()
    graph.fingerprint.return_value = f"fp_{fitness}_{novelty}"
    return Individual(graph=graph, fitness=fitness, novelty=novelty)


def test_two_objective_known_fronts():
    """Verify correct Pareto fronts on a known 2-objective case.

    Layout (fitness, novelty):
        A(3, 1)  B(1, 3)  — Pareto front 1 (non-dominated)
        C(2, 1)  D(1, 2)  — Pareto front 2
        E(1, 1)            — Pareto front 3
    """
    a = _make_individual(3.0, 1.0)
    b = _make_individual(1.0, 3.0)
    c = _make_individual(2.0, 1.0)
    d = _make_individual(1.0, 2.0)
    e = _make_individual(1.0, 1.0)

    fronts = fast_non_dominated_sort([a, b, c, d, e])

    assert len(fronts) == 3

    # Front 0 (rank 1): A and B are non-dominated
    front0_set = set(id(ind) for ind in fronts[0])
    assert id(a) in front0_set
    assert id(b) in front0_set
    assert len(fronts[0]) == 2

    # Front 1 (rank 2): C and D
    front1_set = set(id(ind) for ind in fronts[1])
    assert id(c) in front1_set
    assert id(d) in front1_set
    assert len(fronts[1]) == 2

    # Front 2 (rank 3): E
    assert fronts[2] == [e]

    # Verify pareto_rank assignments
    assert a.pareto_rank == 1
    assert b.pareto_rank == 1
    assert c.pareto_rank == 2
    assert d.pareto_rank == 2
    assert e.pareto_rank == 3


def test_empty_population():
    assert fast_non_dominated_sort([]) == []


def test_single_individual():
    ind = _make_individual(1.0, 1.0)
    fronts = fast_non_dominated_sort([ind])
    assert len(fronts) == 1
    assert fronts[0] == [ind]
    assert ind.pareto_rank == 1


def test_all_identical():
    """All identical individuals form a single non-dominated front."""
    inds = [_make_individual(1.0, 1.0) for _ in range(5)]
    fronts = fast_non_dominated_sort(inds)
    assert len(fronts) == 1
    assert len(fronts[0]) == 5


def test_minimization_objective():
    """Test with a min objective (lower = better)."""
    a = _make_individual(3.0, 1.0)  # high fitness, low novelty (good for min)
    b = _make_individual(
        1.0, 3.0
    )  # low fitness (good for max), high novelty (bad for min)

    objectives = [("fitness", "max"), ("novelty", "min")]
    fronts = fast_non_dominated_sort([a, b], objectives=objectives)

    # Both are non-dominated: a is better on both (higher fitness, lower novelty)
    # Actually a dominates b: fitness 3>1 (max), novelty 1<3 (min, so 1 is better)
    assert len(fronts) == 2
    assert fronts[0] == [a]
    assert fronts[1] == [b]


def test_assign_crowding_distance_sets_boundary_infinity():
    front = [
        _make_individual(1.0, 4.0),
        _make_individual(2.0, 3.0),
        _make_individual(3.0, 2.0),
        _make_individual(4.0, 1.0),
    ]

    assign_crowding_distance(front)

    inf_count = sum(math.isinf(ind.crowding_dist) for ind in front)
    assert inf_count >= 2


def test_fast_non_dominated_sort_routes_through_shared_frontier_kernel(monkeypatch):
    import research.search._nsga as nsga_mod

    population = [
        _make_individual(3.0, 1.0),
        _make_individual(1.0, 3.0),
        _make_individual(2.0, 1.0),
    ]
    stub = MagicMock(return_value=np.array([1, 1, 2], dtype=np.int32))
    monkeypatch.setattr(nsga_mod, "pareto_ranks", stub)

    fronts = fast_non_dominated_sort(population)

    assert [[ind.fitness, ind.novelty] for ind in fronts[0]] == [[3.0, 1.0], [1.0, 3.0]]
    assert [[ind.fitness, ind.novelty] for ind in fronts[1]] == [[2.0, 1.0]]
    stub.assert_called_once()
    args, kwargs = stub.call_args
    np.testing.assert_allclose(
        args[0],
        np.asarray([[3.0, 1.0], [1.0, 3.0], [2.0, 1.0]], dtype=np.float32),
    )
    assert kwargs == {"maximize": (True, True)}


def test_assign_crowding_distance_routes_through_shared_frontier_kernel(monkeypatch):
    import research.search._nsga as nsga_mod

    front = [
        _make_individual(1.0, 4.0),
        _make_individual(2.0, 2.5),
        _make_individual(3.0, 2.0),
    ]
    stub = MagicMock(return_value=np.array([np.inf, 0.75, np.inf], dtype=np.float32))
    monkeypatch.setattr(nsga_mod, "crowding_distances", stub)

    assign_crowding_distance(front)

    assert math.isinf(front[0].crowding_dist)
    assert front[1].crowding_dist == pytest.approx(0.75)
    assert math.isinf(front[2].crowding_dist)
    stub.assert_called_once()
    args, kwargs = stub.call_args
    np.testing.assert_allclose(
        args[0],
        np.asarray([[1.0, 4.0], [2.0, 2.5], [3.0, 2.0]], dtype=np.float32),
    )
    assert kwargs == {}


def test_native_crowding_distance_matches_python_reference():
    reset_native_nsga_lib()
    if load_native_nsga_lib() is None:
        pytest.skip("native frontier library unavailable")

    objective_matrix = np.asarray(
        [[1.0, 4.0], [2.0, 2.5], [3.0, 2.0], [4.0, 1.0], [2.5, 3.0]],
        dtype=np.float32,
    )

    expected = crowding_distances(objective_matrix)
    native_distances = compute_crowding_distances(objective_matrix)

    assert native_distances is not None
    for expected_distance, native in zip(expected, native_distances):
        if math.isinf(float(expected_distance)):
            assert math.isinf(float(native))
        else:
            assert math.isclose(
                float(expected_distance), float(native), rel_tol=1e-6, abs_tol=1e-6
            )


def test_native_pareto_ranks_match_fronts():
    reset_native_nsga_lib()
    if load_native_nsga_lib() is None:
        pytest.skip("native frontier library unavailable")

    objective_matrix = np.asarray(
        [[3.0, 1.0], [1.0, 3.0], [2.0, 1.0], [1.0, 2.0], [1.0, 1.0]],
        dtype=np.float32,
    )

    ranks = compute_pareto_ranks(objective_matrix)
    assert ranks is not None
    assert list(ranks) == [1, 1, 2, 2, 3]


def test_frontier_kernel_drives_search_ranking_outputs():
    objective_matrix = np.asarray(
        [[3.0, 1.0], [1.0, 3.0], [2.0, 1.0], [1.0, 2.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    assert pareto_ranks(objective_matrix).tolist() == [1, 1, 2, 2, 3]


def test_native_frontier_mask_compatibility_shim():
    reset_native_nsga_lib()
    if load_native_nsga_lib() is None:
        pytest.skip("native frontier library unavailable")

    objective_matrix = np.asarray(
        [[3.0, 1.0], [1.0, 3.0], [2.0, 1.0], [1.0, 2.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    mask = compute_pareto_frontier_mask(objective_matrix)
    assert mask is not None
    assert mask.tolist() == [True, True, False, False, False]
