from __future__ import annotations

import ctypes
import logging
import os
from typing import Any

import numpy as np

from ._native_runtime import load_native_runtime_lib

logger = logging.getLogger(__name__)

_NATIVE_SEARCH_LIB: Any = False


def load_native_search_metrics_lib() -> Any:
    global _NATIVE_SEARCH_LIB
    if _NATIVE_SEARCH_LIB is not False:
        return _NATIVE_SEARCH_LIB

    lib = load_native_runtime_lib(
        (
            "aria_behavior_mean_k_nearest",
            "aria_behavior_topk_nearest_indices",
            "aria_behavior_pairwise_median",
            "aria_behavior_neighbor_counts",
        ),
        logger,
    )
    if lib is None:
        _NATIVE_SEARCH_LIB = None
        return None

    lib.aria_behavior_mean_k_nearest.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.aria_behavior_mean_k_nearest.restype = ctypes.c_int32

    lib.aria_behavior_topk_nearest_indices.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.aria_behavior_topk_nearest_indices.restype = ctypes.c_int32

    lib.aria_behavior_pairwise_median.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.aria_behavior_pairwise_median.restype = ctypes.c_int32

    lib.aria_behavior_neighbor_counts.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.aria_behavior_neighbor_counts.restype = ctypes.c_int32

    _NATIVE_SEARCH_LIB = lib
    return lib


def reset_native_search_metrics_lib() -> None:
    global _NATIVE_SEARCH_LIB
    _NATIVE_SEARCH_LIB = False


def _as_feature_matrix(feature_matrix: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(feature_matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("feature_matrix must be rank-2")
    return arr


def _as_feature_vector(feature_vector: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(feature_vector, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("feature_vector must be rank-1")
    return arr


def archive_mean_k_nearest(
    feature_matrix: np.ndarray, target: np.ndarray, k: int
) -> float | None:
    lib = load_native_search_metrics_lib()
    if lib is None:
        return None

    fm = _as_feature_matrix(feature_matrix)
    vec = _as_feature_vector(target)
    out = ctypes.c_float()
    status = lib.aria_behavior_mean_k_nearest(
        fm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(fm.shape[0]),
        int(fm.shape[1]),
        vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(k),
        ctypes.byref(out),
    )
    if status != 0:
        return None
    return float(out.value)


def topk_nearest_indices(
    feature_matrix: np.ndarray, target: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray] | None:
    lib = load_native_search_metrics_lib()
    if lib is None:
        return None

    fm = _as_feature_matrix(feature_matrix)
    vec = _as_feature_vector(target)
    used_k = min(int(k), int(fm.shape[0]))
    out_indices = np.empty(used_k, dtype=np.int32)
    out_distances = np.empty(used_k, dtype=np.float32)
    status = lib.aria_behavior_topk_nearest_indices(
        fm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(fm.shape[0]),
        int(fm.shape[1]),
        vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        used_k,
        out_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    if status <= 0:
        return None
    return out_indices[:status], out_distances[:status]


def pairwise_median_and_neighbor_counts(
    feature_matrix: np.ndarray,
) -> tuple[float, np.ndarray] | None:
    lib = load_native_search_metrics_lib()
    if lib is None:
        return None

    fm = _as_feature_matrix(feature_matrix)
    if fm.shape[0] < 2:
        return None

    median = ctypes.c_float()
    median_status = lib.aria_behavior_pairwise_median(
        fm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(fm.shape[0]),
        int(fm.shape[1]),
        ctypes.byref(median),
    )
    if median_status != 0:
        return None

    out_counts = np.empty(int(fm.shape[0]), dtype=np.int32)
    count_status = lib.aria_behavior_neighbor_counts(
        fm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(fm.shape[0]),
        int(fm.shape[1]),
        ctypes.c_float(float(median.value)),
        out_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    if count_status != 0:
        return None

    return float(median.value), out_counts
