from __future__ import annotations

import ctypes
import importlib
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BOUND_NATIVE_LIB: Any = False
ARIA_CORE_MODULE: Any = False


class AriaGraphAnalysisResult(ctypes.Structure):
    _fields_ = [
        ("has_gradient_path", ctypes.c_int32),
        ("reachable_count", ctypes.c_int32),
        ("depth", ctypes.c_int32),
        ("has_cycle", ctypes.c_int32),
        ("param_estimate", ctypes.c_int64),
    ]


class AriaDimFlowSummary(ctypes.Structure):
    _fields_ = [
        ("reachable_param_count", ctypes.c_int32),
        ("reachable_param_estimate", ctypes.c_int64),
        ("reachable_nontrivial_ops", ctypes.c_int32),
        ("reachable_ops", ctypes.c_int32),
        ("kv_cacheable", ctypes.c_int32),
    ]


class AriaEdgeValidation(ctypes.Structure):
    _fields_ = [
        ("freq_mismatch_bits", ctypes.c_int32),
        ("reduce_full_dim_bits", ctypes.c_int32),
        ("binary_dim_mismatch", ctypes.c_int32),
        ("full_dim_input_bits", ctypes.c_int32),
    ]


class AriaPackedValidationResult(ctypes.Structure):
    _fields_ = [
        ("analysis", AriaGraphAnalysisResult),
        ("dim_flow", AriaDimFlowSummary),
        ("effective_depth", ctypes.c_double),
        ("edge_error_count", ctypes.c_int32),
        ("dead_parameterized_count", ctypes.c_int32),
    ]


def load_native_graph_analysis_lib() -> Any:
    global BOUND_NATIVE_LIB
    if BOUND_NATIVE_LIB is not False:
        return BOUND_NATIVE_LIB

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
            lib = ctypes.CDLL(
                str(path), mode=os.RTLD_LOCAL | getattr(os, "RTLD_LAZY", 1)
            )
            break
        except OSError as exc:
            logger.debug("Failed to load graph-analysis runtime at %s: %s", path, exc)
    if lib is None or not hasattr(lib, "aria_graph_analyze_ir"):
        BOUND_NATIVE_LIB = None
        return None

    fn = lib.aria_graph_analyze_ir
    fn.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(AriaGraphAnalysisResult),
        ctypes.POINTER(ctypes.c_int32),
    ]
    fn.restype = ctypes.c_int32

    summary_fn = getattr(lib, "aria_graph_dim_flow_summary", None)
    if summary_fn is not None:
        summary_fn.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(AriaDimFlowSummary),
        ]
        summary_fn.restype = ctypes.c_int32

    edge_fn = getattr(lib, "aria_graph_validate_edges", None)
    if edge_fn is not None:
        edge_fn.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(AriaEdgeValidation),
        ]
        edge_fn.restype = ctypes.c_int32

    validation_fn = getattr(lib, "aria_graph_validation_summary", None)
    if validation_fn is not None:
        validation_fn.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_void_p,
        ]
        validation_fn.restype = ctypes.c_int32

    dead_param_fn = getattr(lib, "aria_graph_dead_parameterized_mask", None)
    if dead_param_fn is not None:
        dead_param_fn.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        dead_param_fn.restype = ctypes.c_int32

    packed_validation_fn = getattr(lib, "aria_graph_validate_packed_ir", None)
    if packed_validation_fn is not None:
        packed_validation_fn.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(AriaPackedValidationResult),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(AriaEdgeValidation),
            ctypes.POINTER(ctypes.c_int32),
        ]
        packed_validation_fn.restype = ctypes.c_int32

    effective_depth_fn = getattr(lib, "aria_graph_effective_depth", None)
    if effective_depth_fn is not None:
        effective_depth_fn.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double),
        ]
        effective_depth_fn.restype = ctypes.c_int32

    dim_flow_flags_fn = getattr(lib, "aria_graph_build_dim_flow_flags", None)
    if dim_flow_flags_fn is not None:
        dim_flow_flags_fn.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        dim_flow_flags_fn.restype = ctypes.c_int32

    param_formula_fn = getattr(lib, "aria_eval_param_formula", None)
    if param_formula_fn is not None:
        param_formula_fn.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int64),
        ]
        param_formula_fn.restype = ctypes.c_int32

    BOUND_NATIVE_LIB = lib
    return lib


def try_import_aria_core() -> Any:
    global ARIA_CORE_MODULE
    if ARIA_CORE_MODULE is not False:
        return ARIA_CORE_MODULE
    try:
        module = importlib.import_module("aria_core")
    except Exception:
        module = None
    ARIA_CORE_MODULE = module
    return module


def reset_bindings() -> None:
    global BOUND_NATIVE_LIB, ARIA_CORE_MODULE
    BOUND_NATIVE_LIB = False
    ARIA_CORE_MODULE = False
