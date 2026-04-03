from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass

import numpy as np

from .native_analysis_bindings import load_native_graph_analysis_lib

logger = logging.getLogger(__name__)


def _load_native_graph_analysis_lib():
    return load_native_graph_analysis_lib()


@dataclass(slots=True)
class DeadParameterizedMask:
    mask: np.ndarray
    backend: str


def dead_parameterized_mask_natively(
    *, reachable_mask: np.ndarray, parameterized_flags: np.ndarray
) -> DeadParameterizedMask | None:
    lib = _load_native_graph_analysis_lib()
    if lib is None or not hasattr(lib, "aria_graph_dead_parameterized_mask"):
        return None

    reachable_mask = np.ascontiguousarray(reachable_mask, dtype=np.int32)
    parameterized_flags = np.ascontiguousarray(parameterized_flags, dtype=np.int32)
    dead_mask = np.zeros(reachable_mask.shape[0], dtype=np.int32)

    status = lib.aria_graph_dead_parameterized_mask(
        int(reachable_mask.shape[0]),
        reachable_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        parameterized_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        dead_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    if status != 0:
        logger.debug("aria_graph_dead_parameterized_mask failed with status=%d", status)
        return None

    return DeadParameterizedMask(
        mask=dead_mask.astype(bool, copy=False), backend="native"
    )


def dead_parameterized_mask_in_python(
    *, reachable_mask: np.ndarray, parameterized_flags: np.ndarray
) -> DeadParameterizedMask:
    reachable_mask = np.asarray(reachable_mask, dtype=bool)
    parameterized_flags = np.asarray(parameterized_flags, dtype=bool)
    return DeadParameterizedMask(
        mask=(~reachable_mask) & parameterized_flags,
        backend="python",
    )


def dead_parameterized_mask(
    *, reachable_mask: np.ndarray, parameterized_flags: np.ndarray
) -> DeadParameterizedMask:
    native_result = dead_parameterized_mask_natively(
        reachable_mask=reachable_mask,
        parameterized_flags=parameterized_flags,
    )
    if native_result is not None:
        return native_result
    return dead_parameterized_mask_in_python(
        reachable_mask=reachable_mask,
        parameterized_flags=parameterized_flags,
    )
