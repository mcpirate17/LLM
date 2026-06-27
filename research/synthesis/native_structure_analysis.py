from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .native_analysis_bindings import AriaGraphAnalysisResult


@dataclass(slots=True)
class StructuralAnalysisResult:
    has_gradient_path: bool
    reachable_count: int
    depth: int
    has_cycle: bool
    param_estimate: int
    reachable_mask: Optional[np.ndarray]
    backend: str


def analyze_ir_with_native_runtime(
    ir: Any,
    *,
    include_reachable: bool,
    load_native_graph_analysis_lib: Callable[[], Any],
) -> Optional[StructuralAnalysisResult]:
    lib = load_native_graph_analysis_lib()
    if lib is None:
        raise RuntimeError("native graph analysis runtime is unavailable")
    if not hasattr(lib, "aria_graph_analyze_ir"):
        raise RuntimeError("native graph analysis symbol is unavailable")

    op_codes = np.ascontiguousarray(ir.op_codes, dtype=np.int32)
    input_indices = np.ascontiguousarray(ir.input_indices, dtype=np.int32)
    param_estimates = getattr(ir, "param_estimates", None)
    if param_estimates is None:
        param_estimates = np.zeros(op_codes.shape[0], dtype=np.int64)
    else:
        param_estimates = np.ascontiguousarray(param_estimates, dtype=np.int64)

    reachable_mask = None
    reachable_ptr = None
    if include_reachable:
        reachable_mask = np.zeros(op_codes.shape[0], dtype=np.int32)
        reachable_ptr = reachable_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    result = AriaGraphAnalysisResult()
    status = lib.aria_graph_analyze_ir(
        int(op_codes.shape[0]),
        op_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        input_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        int(ir.output_node_idx),
        param_estimates.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.byref(result),
        reachable_ptr,
    )
    if status != 0:
        raise RuntimeError(f"aria_graph_analyze_ir failed with status={status}")

    return StructuralAnalysisResult(
        has_gradient_path=bool(result.has_gradient_path),
        reachable_count=int(result.reachable_count),
        depth=int(result.depth),
        has_cycle=bool(result.has_cycle),
        param_estimate=int(result.param_estimate),
        reachable_mask=reachable_mask.astype(bool, copy=False)
        if reachable_mask is not None
        else None,
        backend="native",
    )
