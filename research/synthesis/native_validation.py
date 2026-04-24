from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass

import numpy as np

from .native_analysis_bindings import load_native_graph_analysis_lib

logger = logging.getLogger(__name__)


def _load_native_graph_analysis_lib():
    return load_native_graph_analysis_lib()


class _AriaValidationSummary(ctypes.Structure):
    _fields_ = [
        ("risky_op_count", ctypes.c_int32),
        ("parameterized_op_count", ctypes.c_int32),
        ("unknown_op_count", ctypes.c_int32),
        ("max_projection_chain_depth", ctypes.c_int32),
    ]


@dataclass(slots=True)
class ValidationSummary:
    risky_op_count: int
    parameterized_op_count: int
    unknown_op_count: int
    max_projection_chain_depth: int
    backend: str


def summarize_validation_natively(
    *,
    known_op_flags: np.ndarray,
    risky_op_flags: np.ndarray,
    parameterized_op_flags: np.ndarray,
    norm_op_flags: np.ndarray,
    linear_op_flags: np.ndarray,
) -> ValidationSummary | None:
    lib = _load_native_graph_analysis_lib()
    if lib is None or not hasattr(lib, "aria_graph_validation_summary"):
        return None

    known_op_flags = np.ascontiguousarray(known_op_flags, dtype=np.int32)
    risky_op_flags = np.ascontiguousarray(risky_op_flags, dtype=np.int32)
    parameterized_op_flags = np.ascontiguousarray(
        parameterized_op_flags, dtype=np.int32
    )
    norm_op_flags = np.ascontiguousarray(norm_op_flags, dtype=np.int32)
    linear_op_flags = np.ascontiguousarray(linear_op_flags, dtype=np.int32)

    result = _AriaValidationSummary()
    status = lib.aria_graph_validation_summary(
        int(known_op_flags.shape[0]),
        known_op_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        risky_op_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        parameterized_op_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        norm_op_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        linear_op_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(result),
    )
    if status != 0:
        logger.debug("aria_graph_validation_summary failed with status=%d", status)
        return None

    return ValidationSummary(
        risky_op_count=int(result.risky_op_count),
        parameterized_op_count=int(result.parameterized_op_count),
        unknown_op_count=int(result.unknown_op_count),
        max_projection_chain_depth=int(result.max_projection_chain_depth),
        backend="native",
    )


def summarize_validation_in_python(
    *,
    known_op_flags: np.ndarray,
    risky_op_flags: np.ndarray,
    parameterized_op_flags: np.ndarray,
    norm_op_flags: np.ndarray,
    linear_op_flags: np.ndarray,
) -> ValidationSummary:
    known_op_flags = np.asarray(known_op_flags, dtype=bool)
    risky_op_flags = np.asarray(risky_op_flags, dtype=bool)
    parameterized_op_flags = np.asarray(parameterized_op_flags, dtype=bool)
    norm_op_flags = np.asarray(norm_op_flags, dtype=bool)
    linear_op_flags = np.asarray(linear_op_flags, dtype=bool)

    max_projection_chain_depth = 0
    projection_chain_depth = 0
    for idx in range(int(known_op_flags.shape[0])):
        if not known_op_flags[idx]:
            continue
        if norm_op_flags[idx]:
            projection_chain_depth = 0
            continue
        if linear_op_flags[idx]:
            projection_chain_depth += 1
            if projection_chain_depth > max_projection_chain_depth:
                max_projection_chain_depth = projection_chain_depth

    return ValidationSummary(
        risky_op_count=int((known_op_flags & risky_op_flags).sum()),
        parameterized_op_count=int((known_op_flags & parameterized_op_flags).sum()),
        unknown_op_count=int((~known_op_flags).sum()),
        max_projection_chain_depth=max_projection_chain_depth,
        backend="python",
    )


def summarize_validation(
    *,
    known_op_flags: np.ndarray,
    risky_op_flags: np.ndarray,
    parameterized_op_flags: np.ndarray,
    norm_op_flags: np.ndarray,
    linear_op_flags: np.ndarray,
) -> ValidationSummary:
    if int(len(known_op_flags)) <= 48:
        return summarize_validation_in_python(
            known_op_flags=known_op_flags,
            risky_op_flags=risky_op_flags,
            parameterized_op_flags=parameterized_op_flags,
            norm_op_flags=norm_op_flags,
            linear_op_flags=linear_op_flags,
        )
    native_result = summarize_validation_natively(
        known_op_flags=known_op_flags,
        risky_op_flags=risky_op_flags,
        parameterized_op_flags=parameterized_op_flags,
        norm_op_flags=norm_op_flags,
        linear_op_flags=linear_op_flags,
    )
    if native_result is not None:
        return native_result
    return summarize_validation_in_python(
        known_op_flags=known_op_flags,
        risky_op_flags=risky_op_flags,
        parameterized_op_flags=parameterized_op_flags,
        norm_op_flags=norm_op_flags,
        linear_op_flags=linear_op_flags,
    )
