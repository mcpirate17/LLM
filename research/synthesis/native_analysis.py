from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .native_analysis_bindings import (
    AriaDimFlowSummary,
    AriaEdgeValidation,
    load_native_graph_analysis_lib,
    reset_bindings as _reset_native_analysis_bindings,
    try_import_aria_core,
)
from .native_structure_analysis import (
    StructuralAnalysisResult,
    analyze_ir_in_python,
    analyze_ir_with_aria_core,
    analyze_ir_with_native_runtime,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DimFlowSummary:
    reachable_param_count: int
    reachable_param_estimate: int
    reachable_nontrivial_ops: int
    reachable_ops: int
    kv_cacheable: bool
    backend: str


@dataclass(slots=True)
class EdgeValidationResult:
    freq_mismatch_bits: np.ndarray
    reduce_full_dim_bits: np.ndarray
    binary_dim_mismatch: np.ndarray
    full_dim_input_bits: np.ndarray
    backend: str


def _load_native_graph_analysis_lib() -> Any:
    return load_native_graph_analysis_lib()


def _try_import_aria_core() -> Any:
    return try_import_aria_core()


def reset_native_analysis_bindings() -> None:
    _reset_native_analysis_bindings()


def analyze_ir_natively(
    ir: Any, *, include_reachable: bool = False
) -> Optional[StructuralAnalysisResult]:
    aria_core_result = analyze_ir_with_aria_core(
        ir,
        include_reachable=include_reachable,
        try_import_aria_core=_try_import_aria_core,
    )
    if aria_core_result is not None:
        return aria_core_result

    return analyze_ir_with_native_runtime(
        ir,
        include_reachable=include_reachable,
        load_native_graph_analysis_lib=_load_native_graph_analysis_lib,
    )


def analyze_ir(ir: Any, *, include_reachable: bool = False) -> StructuralAnalysisResult:
    native_result = analyze_ir_natively(ir, include_reachable=include_reachable)
    if native_result is not None:
        return native_result
    return analyze_ir_in_python(ir, include_reachable=include_reachable)


def summarize_dim_flow_natively(
    *,
    reachable_mask: np.ndarray,
    has_params_flags: np.ndarray,
    param_estimates: np.ndarray,
    nontrivial_flags: np.ndarray,
    kv_breaking_flags: np.ndarray,
) -> Optional[DimFlowSummary]:
    lib = _load_native_graph_analysis_lib()
    if lib is None or not hasattr(lib, "aria_graph_dim_flow_summary"):
        return None

    reachable_mask = np.ascontiguousarray(reachable_mask, dtype=np.int32)
    has_params_flags = np.ascontiguousarray(has_params_flags, dtype=np.int32)
    param_estimates = np.ascontiguousarray(param_estimates, dtype=np.int64)
    nontrivial_flags = np.ascontiguousarray(nontrivial_flags, dtype=np.int32)
    kv_breaking_flags = np.ascontiguousarray(kv_breaking_flags, dtype=np.int32)

    result = AriaDimFlowSummary()
    status = lib.aria_graph_dim_flow_summary(
        int(reachable_mask.shape[0]),
        reachable_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        has_params_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        param_estimates.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        nontrivial_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        kv_breaking_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(result),
    )
    if status != 0:
        logger.debug("aria_graph_dim_flow_summary failed with status=%d", status)
        return None

    return DimFlowSummary(
        reachable_param_count=int(result.reachable_param_count),
        reachable_param_estimate=int(result.reachable_param_estimate),
        reachable_nontrivial_ops=int(result.reachable_nontrivial_ops),
        reachable_ops=int(result.reachable_ops),
        kv_cacheable=bool(result.kv_cacheable),
        backend="native",
    )


def summarize_dim_flow_in_python(
    *,
    reachable_mask: np.ndarray,
    has_params_flags: np.ndarray,
    param_estimates: np.ndarray,
    nontrivial_flags: np.ndarray,
    kv_breaking_flags: np.ndarray,
) -> DimFlowSummary:
    reachable_mask = np.asarray(reachable_mask, dtype=bool)
    has_params_flags = np.asarray(has_params_flags, dtype=bool)
    param_estimates = np.asarray(param_estimates, dtype=np.int64)
    nontrivial_flags = np.asarray(nontrivial_flags, dtype=bool)
    kv_breaking_flags = np.asarray(kv_breaking_flags, dtype=bool)

    reachable_params = reachable_mask & has_params_flags
    return DimFlowSummary(
        reachable_param_count=int(reachable_params.sum()),
        reachable_param_estimate=int(param_estimates[reachable_params].sum()),
        reachable_nontrivial_ops=int((reachable_mask & nontrivial_flags).sum()),
        reachable_ops=int(reachable_mask.sum()),
        kv_cacheable=not bool((reachable_mask & kv_breaking_flags).any()),
        backend="python",
    )


def summarize_dim_flow(
    *,
    reachable_mask: np.ndarray,
    has_params_flags: np.ndarray,
    param_estimates: np.ndarray,
    nontrivial_flags: np.ndarray,
    kv_breaking_flags: np.ndarray,
) -> DimFlowSummary:
    native_result = summarize_dim_flow_natively(
        reachable_mask=reachable_mask,
        has_params_flags=has_params_flags,
        param_estimates=param_estimates,
        nontrivial_flags=nontrivial_flags,
        kv_breaking_flags=kv_breaking_flags,
    )
    if native_result is not None:
        return native_result
    return summarize_dim_flow_in_python(
        reachable_mask=reachable_mask,
        has_params_flags=has_params_flags,
        param_estimates=param_estimates,
        nontrivial_flags=nontrivial_flags,
        kv_breaking_flags=kv_breaking_flags,
    )


def validate_edges_natively(
    *,
    reachable_mask: np.ndarray,
    input_indices: np.ndarray,
    node_dims: np.ndarray,
    node_seq_flags: np.ndarray,
    op_kind_flags: np.ndarray,
    full_dim_flags: np.ndarray,
    model_dim: int,
) -> Optional[EdgeValidationResult]:
    lib = _load_native_graph_analysis_lib()
    if lib is None or not hasattr(lib, "aria_graph_validate_edges"):
        return None

    reachable_mask = np.ascontiguousarray(reachable_mask, dtype=np.int32)
    input_indices = np.ascontiguousarray(input_indices, dtype=np.int32)
    node_dims = np.ascontiguousarray(node_dims, dtype=np.int32)
    node_seq_flags = np.ascontiguousarray(node_seq_flags, dtype=np.int32)
    op_kind_flags = np.ascontiguousarray(op_kind_flags, dtype=np.int32)
    full_dim_flags = np.ascontiguousarray(full_dim_flags, dtype=np.int32)
    out = np.zeros(
        reachable_mask.shape[0],
        dtype=[
            ("freq_mismatch_bits", np.int32),
            ("reduce_full_dim_bits", np.int32),
            ("binary_dim_mismatch", np.int32),
            ("full_dim_input_bits", np.int32),
        ],
    )

    status = lib.aria_graph_validate_edges(
        int(reachable_mask.shape[0]),
        reachable_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        input_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        node_dims.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        node_seq_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        op_kind_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        full_dim_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        int(model_dim),
        out.ctypes.data_as(ctypes.POINTER(AriaEdgeValidation)),
    )
    if status != 0:
        logger.debug("aria_graph_validate_edges failed with status=%d", status)
        return None

    return EdgeValidationResult(
        freq_mismatch_bits=out["freq_mismatch_bits"].copy(),
        reduce_full_dim_bits=out["reduce_full_dim_bits"].copy(),
        binary_dim_mismatch=out["binary_dim_mismatch"].copy(),
        full_dim_input_bits=out["full_dim_input_bits"].copy(),
        backend="native",
    )


def validate_edges_in_python(
    *,
    reachable_mask: np.ndarray,
    input_indices: np.ndarray,
    node_dims: np.ndarray,
    node_seq_flags: np.ndarray,
    op_kind_flags: np.ndarray,
    full_dim_flags: np.ndarray,
    model_dim: int,
) -> EdgeValidationResult:
    n_nodes = int(len(reachable_mask))
    freq_mismatch_bits = np.zeros(n_nodes, dtype=np.int32)
    reduce_full_dim_bits = np.zeros(n_nodes, dtype=np.int32)
    binary_dim_mismatch = np.zeros(n_nodes, dtype=np.int32)
    full_dim_input_bits = np.zeros(n_nodes, dtype=np.int32)

    reachable_mask = np.asarray(reachable_mask, dtype=bool)
    for idx in range(n_nodes):
        if not reachable_mask[idx]:
            continue
        parents = input_indices[idx]
        for slot, parent in enumerate(parents):
            parent = int(parent)
            if parent == -1:
                continue
            if node_seq_flags[parent] and op_kind_flags[idx] not in (1, 2):
                freq_mismatch_bits[idx] |= 1 << slot
            if node_dims[parent] == 1 and full_dim_flags[idx]:
                reduce_full_dim_bits[idx] |= 1 << slot
            if full_dim_flags[idx] and node_dims[parent] != model_dim:
                full_dim_input_bits[idx] |= 1 << slot
        if op_kind_flags[idx] == 3 and parents[0] != -1 and parents[1] != -1:
            d0 = node_dims[int(parents[0])]
            d1 = node_dims[int(parents[1])]
            if d0 != d1 and d0 != 1 and d1 != 1:
                binary_dim_mismatch[idx] = 1

    return EdgeValidationResult(
        freq_mismatch_bits=freq_mismatch_bits,
        reduce_full_dim_bits=reduce_full_dim_bits,
        binary_dim_mismatch=binary_dim_mismatch,
        full_dim_input_bits=full_dim_input_bits,
        backend="python",
    )


def validate_edges(
    *,
    reachable_mask: np.ndarray,
    input_indices: np.ndarray,
    node_dims: np.ndarray,
    node_seq_flags: np.ndarray,
    op_kind_flags: np.ndarray,
    full_dim_flags: np.ndarray,
    model_dim: int,
) -> EdgeValidationResult:
    native_result = validate_edges_natively(
        reachable_mask=reachable_mask,
        input_indices=input_indices,
        node_dims=node_dims,
        node_seq_flags=node_seq_flags,
        op_kind_flags=op_kind_flags,
        full_dim_flags=full_dim_flags,
        model_dim=model_dim,
    )
    if native_result is not None:
        return native_result
    return validate_edges_in_python(
        reachable_mask=reachable_mask,
        input_indices=input_indices,
        node_dims=node_dims,
        node_seq_flags=node_seq_flags,
        op_kind_flags=op_kind_flags,
        full_dim_flags=full_dim_flags,
        model_dim=model_dim,
    )
