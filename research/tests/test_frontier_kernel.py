import numpy as np
import pytest
from unittest.mock import MagicMock

from research.frontier_kernel import (
    crowding_distances,
    load_native_frontier_lib,
    native_pareto_frontier_mask,
    pareto_frontier_mask,
    pareto_ranks,
    reset_native_frontier_lib,
)
from research.scientist.analytics.frontier import pareto_mask

pytestmark = pytest.mark.unit


def _legacy_analytics_pareto_mask(points, *, minimize=None):
    costs = np.asarray(points, dtype=np.float64)
    n_rows, n_dims = costs.shape
    if n_rows == 0:
        return np.zeros(0, dtype=bool)

    if minimize is not None:
        directions = np.asarray(tuple(minimize), dtype=bool)
        costs = costs.copy()
        costs[:, ~directions] *= -1.0

    sort_keys = tuple(costs[:, col] for col in range(n_dims - 1, -1, -1))
    order = np.lexsort(sort_keys)

    frontier = []
    for idx in order:
        point = costs[idx]
        if frontier:
            frontier_costs = costs[frontier]
            dominated = np.all(frontier_costs <= point, axis=1) & np.any(
                frontier_costs < point, axis=1
            )
            if bool(np.any(dominated)):
                continue

            dominates_existing = np.all(point <= frontier_costs, axis=1) & np.any(
                point < frontier_costs, axis=1
            )
            if bool(np.any(dominates_existing)):
                frontier = [
                    frontier[i]
                    for i, is_dominated in enumerate(dominates_existing)
                    if not bool(is_dominated)
                ]

        frontier.append(int(idx))

    mask = np.zeros(n_rows, dtype=bool)
    mask[frontier] = True
    return mask


def _legacy_search_pareto_ranks(objective_matrix):
    values = np.asarray(objective_matrix, dtype=np.float32)
    if values.shape[0] == 0:
        return np.empty(0, dtype=np.int32)

    diff = values[:, np.newaxis, :] - values[np.newaxis, :, :]
    dominates = np.all(diff >= 0, axis=2) & np.any(diff > 0, axis=2)
    domination_count = dominates.sum(axis=0)

    ranks = np.zeros(values.shape[0], dtype=np.int32)
    remaining = np.ones(values.shape[0], dtype=bool)
    rank = 1
    while True:
        front_indices = np.where(remaining & (domination_count == 0))[0]
        if front_indices.size == 0:
            break
        ranks[front_indices] = rank
        remaining[front_indices] = False
        for idx in front_indices:
            dominated = np.where(dominates[int(idx)] & remaining)[0]
            domination_count[dominated] -= 1
        rank += 1
    return ranks


def _legacy_crowding_distances(objective_matrix):
    values = np.asarray(objective_matrix, dtype=np.float32)
    if values.shape[0] <= 2:
        return np.full(values.shape[0], np.inf, dtype=np.float32)

    distances = np.zeros(values.shape[0], dtype=np.float32)
    for objective_idx in range(values.shape[1]):
        order = np.argsort(values[:, objective_idx], kind="mergesort")
        obj_values = values[order, objective_idx]
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        span = float(obj_values[-1] - obj_values[0])
        if span <= 0.0:
            continue
        inv_span = 1.0 / span
        for pos in range(1, values.shape[0] - 1):
            idx = int(order[pos])
            if np.isinf(distances[idx]):
                continue
            distances[idx] += float(
                (obj_values[pos + 1] - obj_values[pos - 1]) * inv_span
            )
    return distances


REGRESSION_FIXTURES = (
    {
        "name": "analytics_minimize_2d",
        "points": np.array(
            [[0.5, 100.0], [1.0, 200.0], [0.8, 50.0]],
            dtype=np.float32,
        ),
        "minimize": (True, True),
        "expected_mask": [True, False, True],
        "expected_ranks": [1, 2, 1],
    },
    {
        "name": "analytics_mixed_2d",
        "points": np.array(
            [[0.40, 100.0], [0.55, 220.0], [0.60, 80.0]],
            dtype=np.float32,
        ),
        "minimize": (True, False),
        "expected_mask": [True, True, False],
        "expected_ranks": [1, 1, 2],
    },
    {
        "name": "search_maximize_2d",
        "points": np.array(
            [[3.0, 1.0], [1.0, 3.0], [2.0, 1.0], [1.0, 2.0], [1.0, 1.0]],
            dtype=np.float32,
        ),
        "maximize": (True, True),
        "expected_mask": [True, True, False, False, False],
        "expected_ranks": [1, 1, 2, 2, 3],
    },
    {
        "name": "tradeoff_3d",
        "points": np.array(
            [[0.30, 500.0, 100.0], [0.40, 300.0, 500.0], [0.50, 600.0, 200.0]],
            dtype=np.float32,
        ),
        "minimize": (True, True, True),
        "expected_mask": [True, True, False],
        "expected_ranks": [1, 1, 2],
    },
)


@pytest.mark.parametrize("fixture", REGRESSION_FIXTURES, ids=lambda case: case["name"])
def test_frontier_kernel_matches_regression_fixtures(fixture):
    kwargs = {}
    if "maximize" in fixture:
        kwargs["maximize"] = fixture["maximize"]
    if "minimize" in fixture:
        kwargs["minimize"] = fixture["minimize"]

    mask = pareto_frontier_mask(fixture["points"], **kwargs)
    ranks = pareto_ranks(fixture["points"], **kwargs)

    assert mask.tolist() == fixture["expected_mask"]
    assert ranks.tolist() == fixture["expected_ranks"]


@pytest.mark.parametrize("fixture", REGRESSION_FIXTURES, ids=lambda case: case["name"])
def test_frontier_kernel_matches_legacy_outputs(fixture):
    if "maximize" in fixture:
        legacy_ranks = _legacy_search_pareto_ranks(fixture["points"])
        ranks = pareto_ranks(fixture["points"], maximize=fixture["maximize"])
        mask = pareto_frontier_mask(fixture["points"], maximize=fixture["maximize"])
    else:
        legacy_mask = _legacy_analytics_pareto_mask(
            fixture["points"],
            minimize=fixture["minimize"],
        )
        legacy_ranks = _legacy_search_pareto_ranks(
            np.asarray(fixture["points"], dtype=np.float32)
            * np.where(np.asarray(fixture["minimize"], dtype=bool), -1.0, 1.0)
        )
        ranks = pareto_ranks(fixture["points"], minimize=fixture["minimize"])
        mask = pareto_frontier_mask(fixture["points"], minimize=fixture["minimize"])
        assert mask.tolist() == legacy_mask.tolist()

    assert ranks.tolist() == legacy_ranks.tolist()
    assert mask.tolist() == (legacy_ranks == 1).tolist()


def test_analytics_wrapper_preserves_minimization_default():
    points = np.array(
        [[0.5, 100.0], [1.0, 200.0], [0.8, 50.0]],
        dtype=np.float32,
    )
    assert pareto_mask(points).tolist() == [True, False, True]


def test_analytics_wrapper_routes_through_shared_frontier_kernel(monkeypatch):
    points = np.array(
        [[0.5, 100.0], [1.0, 200.0], [0.8, 50.0]],
        dtype=np.float32,
    )
    stub = MagicMock(return_value=np.array([True, False, True], dtype=bool))
    monkeypatch.setattr(
        "research.scientist.analytics.frontier.pareto_frontier_mask", stub
    )

    mask = pareto_mask(points)

    assert mask.tolist() == [True, False, True]
    stub.assert_called_once()
    args, kwargs = stub.call_args
    np.testing.assert_allclose(args[0], points)
    assert kwargs == {"minimize": (True, True)}


def test_crowding_distances_match_legacy_reference():
    objective_matrix = np.asarray(
        [[1.0, 4.0], [2.0, 2.5], [3.0, 2.0], [4.0, 1.0], [2.5, 3.0]],
        dtype=np.float32,
    )

    expected = _legacy_crowding_distances(objective_matrix)
    actual = crowding_distances(objective_matrix)

    assert np.array_equal(np.isinf(actual), np.isinf(expected))
    finite = ~np.isinf(expected)
    assert np.allclose(actual[finite], expected[finite], atol=1e-6, rtol=1e-6)


def test_native_frontier_mask_matches_rank_front_when_available():
    reset_native_frontier_lib()
    if load_native_frontier_lib() is None:
        pytest.skip("native frontier library unavailable")

    objective_matrix = np.asarray(
        [[3.0, 1.0], [1.0, 3.0], [2.0, 1.0], [1.0, 2.0], [1.0, 1.0]],
        dtype=np.float32,
    )

    native_mask = native_pareto_frontier_mask(objective_matrix)
    assert native_mask is not None
    assert native_mask.tolist() == [True, True, False, False, False]
