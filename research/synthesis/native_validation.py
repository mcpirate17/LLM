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
    if lib is None:
        raise RuntimeError("native graph validation runtime is unavailable")
    if not hasattr(lib, "aria_graph_validation_summary"):
        raise RuntimeError("native graph validation summary symbol is unavailable")

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
        raise RuntimeError(f"aria_graph_validation_summary failed with status={status}")

    return ValidationSummary(
        risky_op_count=int(result.risky_op_count),
        parameterized_op_count=int(result.parameterized_op_count),
        unknown_op_count=int(result.unknown_op_count),
        max_projection_chain_depth=int(result.max_projection_chain_depth),
        backend="native",
    )


def summarize_validation(
    *,
    known_op_flags: np.ndarray,
    risky_op_flags: np.ndarray,
    parameterized_op_flags: np.ndarray,
    norm_op_flags: np.ndarray,
    linear_op_flags: np.ndarray,
) -> ValidationSummary:
    return summarize_validation_natively(
        known_op_flags=known_op_flags,
        risky_op_flags=risky_op_flags,
        parameterized_op_flags=parameterized_op_flags,
        norm_op_flags=norm_op_flags,
        linear_op_flags=linear_op_flags,
    )


def effective_depth_natively(
    *,
    op_codes: np.ndarray,
    input_indices: np.ndarray,
    effective_depth_weights: np.ndarray,
    discount_successor_u8: np.ndarray,
) -> float | None:
    lib = _load_native_graph_analysis_lib()
    if lib is None or not hasattr(lib, "aria_graph_effective_depth"):
        return None

    op_codes = np.ascontiguousarray(op_codes, dtype=np.int32)
    input_indices = np.ascontiguousarray(input_indices, dtype=np.int32)
    effective_depth_weights = np.ascontiguousarray(
        effective_depth_weights, dtype=np.float32
    )
    discount_successor_u8 = np.ascontiguousarray(discount_successor_u8, dtype=np.uint8)
    out_depth = ctypes.c_double(0.0)
    status = lib.aria_graph_effective_depth(
        int(op_codes.shape[0]),
        op_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        input_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        effective_depth_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        discount_successor_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        int(effective_depth_weights.shape[0]),
        ctypes.byref(out_depth),
    )
    if status != 0:
        logger.debug("aria_graph_effective_depth failed with status=%d", status)
        return None
    return float(out_depth.value)
