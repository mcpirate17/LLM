from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_NATIVE_NSGA_LIB: Any = False


def load_native_nsga_lib() -> Any:
    global _NATIVE_NSGA_LIB
    if _NATIVE_NSGA_LIB is not False:
        return _NATIVE_NSGA_LIB

    lib = None
    for path in (
        Path(__file__).resolve().parents[1]
        / "runtime"
        / "native"
        / "build"
        / "libaria_native_runtime.so",
        Path(__file__).resolve().parents[1]
        / "runtime"
        / "native"
        / "build_current"
        / "libaria_native_runtime.so",
    ):
        if not path.exists():
            continue
        try:
            candidate = ctypes.CDLL(
                str(path), mode=os.RTLD_LOCAL | getattr(os, "RTLD_LAZY", 1)
            )
            if hasattr(candidate, "aria_nsga_crowding_distance") and hasattr(
                candidate, "aria_nsga_pareto_ranks"
            ):
                lib = candidate
                break
        except OSError as exc:
            logger.debug("Failed to load NSGA runtime at %s: %s", path, exc)
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
