from __future__ import annotations

import ctypes
import logging
from typing import Any

import numpy as np

from ._native_runtime import load_native_runtime_lib

logger = logging.getLogger(__name__)

_NATIVE_NSGA_LIB: Any = False


def load_native_nsga_lib() -> Any:
    global _NATIVE_NSGA_LIB
    if _NATIVE_NSGA_LIB is not False:
        return _NATIVE_NSGA_LIB

    lib = load_native_runtime_lib(
        ("aria_nsga_crowding_distance", "aria_nsga_pareto_ranks"),
        logger,
    )
    if lib is None:
        _NATIVE_NSGA_LIB = None
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
    _NATIVE_NSGA_LIB = lib
    return lib


def reset_native_nsga_lib() -> None:
    global _NATIVE_NSGA_LIB
    _NATIVE_NSGA_LIB = False


def compute_crowding_distances(objective_matrix: np.ndarray) -> np.ndarray | None:
    lib = load_native_nsga_lib()
    if lib is None:
        return None

    values = np.ascontiguousarray(objective_matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("objective_matrix must be rank-2")

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


def compute_pareto_ranks(objective_matrix: np.ndarray) -> np.ndarray | None:
    lib = load_native_nsga_lib()
    if lib is None:
        return None

    values = np.ascontiguousarray(objective_matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("objective_matrix must be rank-2")

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
