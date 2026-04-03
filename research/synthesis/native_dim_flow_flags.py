from __future__ import annotations

import ctypes

import numpy as np

from .native_analysis_bindings import load_native_graph_analysis_lib


def build_dim_flow_flags_natively(
    *,
    op_codes: np.ndarray,
    param_estimates: np.ndarray,
    opcode_has_params: np.ndarray,
    opcode_nontrivial: np.ndarray,
    opcode_kv_breaking: np.ndarray,
    opcode_kind: np.ndarray,
    opcode_full_dim: np.ndarray,
) -> dict[str, np.ndarray] | None:
    lib = load_native_graph_analysis_lib()
    if lib is None or not hasattr(lib, "aria_graph_build_dim_flow_flags"):
        return None

    op_codes = np.ascontiguousarray(op_codes, dtype=np.int32)
    param_estimates = np.ascontiguousarray(param_estimates, dtype=np.int64)
    opcode_has_params = np.ascontiguousarray(opcode_has_params, dtype=np.int32)
    opcode_nontrivial = np.ascontiguousarray(opcode_nontrivial, dtype=np.int32)
    opcode_kv_breaking = np.ascontiguousarray(opcode_kv_breaking, dtype=np.int32)
    opcode_kind = np.ascontiguousarray(opcode_kind, dtype=np.int32)
    opcode_full_dim = np.ascontiguousarray(opcode_full_dim, dtype=np.int32)

    has_params_flags = np.zeros(op_codes.shape[0], dtype=np.int32)
    nontrivial_flags = np.zeros(op_codes.shape[0], dtype=np.int32)
    kv_breaking_flags = np.zeros(op_codes.shape[0], dtype=np.int32)
    op_kind_flags = np.zeros(op_codes.shape[0], dtype=np.int32)
    full_dim_flags = np.zeros(op_codes.shape[0], dtype=np.int32)

    status = lib.aria_graph_build_dim_flow_flags(
        int(op_codes.shape[0]),
        op_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        param_estimates.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        opcode_has_params.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        opcode_nontrivial.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        opcode_kv_breaking.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        opcode_kind.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        opcode_full_dim.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        has_params_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        nontrivial_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        kv_breaking_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        op_kind_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        full_dim_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    if status != 0:
        return None

    return {
        "has_params_flags": has_params_flags,
        "nontrivial_flags": nontrivial_flags,
        "kv_breaking_flags": kv_breaking_flags,
        "op_kind_flags": op_kind_flags,
        "full_dim_flags": full_dim_flags,
    }
