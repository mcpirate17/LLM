from __future__ import annotations

import ctypes
import logging
from typing import Any, Iterable

import numpy as np

from ._native_runtime import load_native_runtime_lib

logger = logging.getLogger(__name__)

_NATIVE_FRONTIER_LIB: Any = False


def load_native_frontier_lib() -> Any:
    global _NATIVE_FRONTIER_LIB
    if _NATIVE_FRONTIER_LIB is not False:
        return _NATIVE_FRONTIER_LIB

    lib = load_native_runtime_lib(
        ("aria_nsga_crowding_distance", "aria_nsga_pareto_ranks"),
        logger,
    )
    if lib is None:
        _NATIVE_FRONTIER_LIB = None
        return None

    lib.aria_nsga_pareto_ranks.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.aria_nsga_pareto_ranks.restype = ctypes.c_int32

    lib.aria_nsga_crowding_distance.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.aria_nsga_crowding_distance.restype = ctypes.c_int32

    if hasattr(lib, "aria_nsga_pareto_frontier_mask"):
        lib.aria_nsga_pareto_frontier_mask.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_uint8),
        ]
        lib.aria_nsga_pareto_frontier_mask.restype = ctypes.c_int32

    _NATIVE_FRONTIER_LIB = lib
    return lib


def reset_native_frontier_lib() -> None:
    global _NATIVE_FRONTIER_LIB
    _NATIVE_FRONTIER_LIB = False


def native_pareto_ranks(objective_matrix: np.ndarray) -> np.ndarray | None:
    lib = load_native_frontier_lib()
    if lib is None:
        return None

    values = _as_objective_matrix(objective_matrix)
    if values.shape[0] == 0:
        return np.empty(0, dtype=np.int32)

    ranks = np.empty(values.shape[0], dtype=np.int32)
    status = lib.aria_nsga_pareto_ranks(
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(values.shape[0]),
        int(values.shape[1]),
        ranks.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    if status != 0:
        return None
    return ranks


def native_crowding_distances(objective_matrix: np.ndarray) -> np.ndarray | None:
    lib = load_native_frontier_lib()
    if lib is None:
        return None

    values = _as_objective_matrix(objective_matrix)
    if values.shape[0] == 0:
        return np.empty(0, dtype=np.float32)

    distances = np.empty(values.shape[0], dtype=np.float32)
    status = lib.aria_nsga_crowding_distance(
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(values.shape[0]),
        int(values.shape[1]),
        distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if status != 0:
        return None
    return distances


def native_pareto_frontier_mask(objective_matrix: np.ndarray) -> np.ndarray | None:
    lib = load_native_frontier_lib()
    if lib is None or not hasattr(lib, "aria_nsga_pareto_frontier_mask"):
        return None

    values = _as_objective_matrix(objective_matrix)
    if values.shape[0] == 0:
        return np.empty(0, dtype=bool)

    mask_raw = np.empty(values.shape[0], dtype=np.uint8)
    status = lib.aria_nsga_pareto_frontier_mask(
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(values.shape[0]),
        int(values.shape[1]),
        mask_raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    if status != 0:
        return None
    return mask_raw.astype(bool, copy=False)


def pareto_ranks(
    objective_matrix: np.ndarray,
    *,
    maximize: Iterable[bool] | None = None,
    minimize: Iterable[bool] | None = None,
) -> np.ndarray:
    values = _normalize_objectives(
        objective_matrix,
        maximize=maximize,
        minimize=minimize,
    )
    native = native_pareto_ranks(values)
    if native is not None:
        return native
    return _pareto_ranks_python(values)


def pareto_frontier_mask(
    objective_matrix: np.ndarray,
    *,
    maximize: Iterable[bool] | None = None,
    minimize: Iterable[bool] | None = None,
) -> np.ndarray:
    values = _normalize_objectives(
        objective_matrix,
        maximize=maximize,
        minimize=minimize,
    )
    native = native_pareto_frontier_mask(values)
    if native is not None:
        return native

    native_ranks = native_pareto_ranks(values)
    if native_ranks is not None:
        return native_ranks == 1
    return _pareto_ranks_python(values) == 1


def crowding_distances(objective_matrix: np.ndarray) -> np.ndarray:
    values = _as_objective_matrix(objective_matrix)
    native = native_crowding_distances(values)
    if native is not None:
        return native
    return _crowding_distances_python(values)


def _as_objective_matrix(objective_matrix: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(np.asarray(objective_matrix, dtype=np.float32))
    if values.ndim != 2:
        raise ValueError("objective_matrix must be rank-2")
    return values


def _normalize_objectives(
    objective_matrix: np.ndarray,
    *,
    maximize: Iterable[bool] | None,
    minimize: Iterable[bool] | None,
) -> np.ndarray:
    values = _as_objective_matrix(objective_matrix)
    if maximize is not None and minimize is not None:
        raise ValueError("provide only one of maximize or minimize")
    if values.shape[0] == 0:
        return values
    if maximize is None and minimize is None:
        return values

    if maximize is not None:
        directions = np.asarray(tuple(maximize), dtype=bool)
        if directions.shape != (values.shape[1],):
            raise ValueError("maximize must match the number of objective columns")
        if np.all(directions):
            return values
        normalized = values.copy()
        normalized[:, ~directions] *= -1.0
        return normalized

    directions = np.asarray(tuple(minimize), dtype=bool)
    if directions.shape != (values.shape[1],):
        raise ValueError("minimize must match the number of objective columns")
    if not np.any(directions):
        return values
    normalized = values.copy()
    normalized[:, directions] *= -1.0
    return normalized


def _pareto_ranks_python(objective_matrix: np.ndarray) -> np.ndarray:
    n_rows = objective_matrix.shape[0]
    if n_rows == 0:
        return np.empty(0, dtype=np.int32)

    diff = objective_matrix[:, np.newaxis, :] - objective_matrix[np.newaxis, :, :]
    dominates = np.all(diff >= 0.0, axis=2) & np.any(diff > 0.0, axis=2)
    domination_count = dominates.sum(axis=0, dtype=np.int32)

    ranks = np.zeros(n_rows, dtype=np.int32)
    remaining = np.ones(n_rows, dtype=bool)
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


def _crowding_distances_python(objective_matrix: np.ndarray) -> np.ndarray:
    n_rows = objective_matrix.shape[0]
    if n_rows == 0:
        return np.empty(0, dtype=np.float32)
    if n_rows <= 2:
        return np.full(n_rows, np.inf, dtype=np.float32)

    distances = np.zeros(n_rows, dtype=np.float32)
    for objective_idx in range(objective_matrix.shape[1]):
        order = np.argsort(objective_matrix[:, objective_idx], kind="mergesort")
        obj_values = objective_matrix[order, objective_idx]
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf

        span = float(obj_values[-1] - obj_values[0])
        if span <= 0.0:
            continue

        inv_span = 1.0 / span
        for pos in range(1, n_rows - 1):
            idx = int(order[pos])
            if np.isinf(distances[idx]):
                continue
            distances[idx] += float(
                (obj_values[pos + 1] - obj_values[pos - 1]) * inv_span
            )

    return distances
