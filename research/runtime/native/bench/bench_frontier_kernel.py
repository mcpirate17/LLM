#!/usr/bin/env python3
"""Benchmark the unified frontier kernel against the legacy Python paths.

This compares the canonical shared backend to the two superseded implementations:
the analytics Pareto mask and the search NSGA ranking/crowding Python kernels.
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable

import numpy as np

from research.frontier_kernel import (
    crowding_distances,
    load_native_frontier_lib,
    pareto_frontier_mask,
    pareto_ranks,
)


def _legacy_analytics_pareto_mask(points, *, minimize=None):
    costs = np.asarray(points, dtype=np.float64)
    n_rows, n_dims = costs.shape
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
    diff = values[:, np.newaxis, :] - values[np.newaxis, :, :]
    dominates = np.all(diff >= 0.0, axis=2) & np.any(diff > 0.0, axis=2)
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


def _build_cases(seed: int):
    rng = np.random.default_rng(seed)
    return (
        (
            "medium",
            rng.random((512, 3), dtype=np.float32),
            (True, True, True),
        ),
        (
            "large",
            rng.random((2048, 4), dtype=np.float32),
            (True, True, True, True),
        ),
    )


def _print_result_row(label: str, before_us: float, after_us: float) -> None:
    speedup = before_us / after_us if after_us > 0.0 else float("inf")
    print(
        f"{label:<26} before={before_us:>10.1f} us  after={after_us:>10.1f} us  speedup={speedup:>7.2f}x"
    )


def run_benchmarks(*, iterations: int, seed: int) -> None:
    print(
        f"native_backend={'yes' if load_native_frontier_lib() is not None else 'no'}  iterations={iterations}"
    )
    for size_name, minimize_matrix, minimize in _build_cases(seed):
        maximize_matrix = -minimize_matrix
        legacy_mask = _legacy_analytics_pareto_mask(minimize_matrix, minimize=minimize)
        unified_mask = pareto_frontier_mask(minimize_matrix, minimize=minimize)
        legacy_ranks = _legacy_search_pareto_ranks(maximize_matrix)
        unified_ranks = pareto_ranks(maximize_matrix)
        legacy_crowding = _legacy_crowding_distances(maximize_matrix)
        unified_crowding = crowding_distances(maximize_matrix)

        if not np.array_equal(legacy_mask, unified_mask):
            raise AssertionError(
                f"{size_name}: unified frontier mask diverged from legacy analytics output"
            )
        if not np.array_equal(legacy_ranks, unified_ranks):
            raise AssertionError(
                f"{size_name}: unified pareto ranks diverged from legacy search output"
            )
        if not np.array_equal(np.isinf(legacy_crowding), np.isinf(unified_crowding)):
            raise AssertionError(f"{size_name}: crowding boundary infinities diverged")
        finite = ~np.isinf(legacy_crowding)
        if not np.allclose(
            legacy_crowding[finite],
            unified_crowding[finite],
            atol=1e-6,
            rtol=1e-6,
        ):
            raise AssertionError(f"{size_name}: crowding distances diverged")

        print(
            f"\n[{size_name}] rows={minimize_matrix.shape[0]} objectives={minimize_matrix.shape[1]}"
        )
        _print_result_row(
            "analytics_mask",
            _median_runtime_us(
                _legacy_analytics_pareto_mask,
                minimize_matrix,
                iterations=iterations,
                minimize=minimize,
            ),
            _median_runtime_us(
                pareto_frontier_mask,
                minimize_matrix,
                iterations=iterations,
                minimize=minimize,
            ),
        )
        _print_result_row(
            "search_ranks",
            _median_runtime_us(
                _legacy_search_pareto_ranks,
                maximize_matrix,
                iterations=iterations,
            ),
            _median_runtime_us(
                pareto_ranks,
                maximize_matrix,
                iterations=iterations,
            ),
        )
        _print_result_row(
            "crowding_distance",
            _median_runtime_us(
                _legacy_crowding_distances,
                maximize_matrix,
                iterations=iterations,
            ),
            _median_runtime_us(
                crowding_distances,
                maximize_matrix,
                iterations=iterations,
            ),
        )


def _median_runtime_us(fn, *args, iterations: int, **kwargs) -> float:
    for _ in range(min(3, iterations)):
        fn(*args, **kwargs)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn(*args, **kwargs)
        samples.append((time.perf_counter_ns() - start) / 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_benchmarks(iterations=args.iterations, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
