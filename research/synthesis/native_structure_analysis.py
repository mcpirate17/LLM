from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .native_analysis_bindings import AriaGraphAnalysisResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StructuralAnalysisResult:
    has_gradient_path: bool
    reachable_count: int
    depth: int
    has_cycle: bool
    param_estimate: int
    reachable_mask: Optional[np.ndarray]
    backend: str


def analyze_ir_with_aria_core(
    ir: Any,
    *,
    include_reachable: bool,
    try_import_aria_core: Callable[[], Any],
) -> Optional[StructuralAnalysisResult]:
    aria_core = try_import_aria_core()
    if aria_core is None or not hasattr(aria_core, "analyze_graph"):
        return None

    op_codes = np.ascontiguousarray(ir.op_codes, dtype=np.int32)
    input_indices = np.ascontiguousarray(ir.input_indices, dtype=np.int32)
    n_nodes = int(op_codes.shape[0])
    output_node_idx = int(ir.output_node_idx)
    input_node_candidates = np.flatnonzero(op_codes == 0)
    input_node_idx = int(input_node_candidates[0]) if input_node_candidates.size else -1

    edges: list[list[int]] = []
    for target_idx in range(n_nodes):
        for src_idx in input_indices[target_idx]:
            src = int(src_idx)
            if src != -1:
                edges.append([src, target_idx])

    try:
        result = aria_core.analyze_graph(
            n_nodes,
            edges,
            op_codes.tolist(),
            output_node_idx,
            input_node_idx,
        )
    except Exception as exc:
        logger.debug("aria_core.analyze_graph failed: %s", exc)
        return None

    if not result.get("valid", False):
        return None

    reachable_nodes = np.asarray(result.get("reachable_nodes", []), dtype=np.int32)
    reachable_mask = np.zeros(n_nodes, dtype=bool)
    if reachable_nodes.size:
        reachable_mask[reachable_nodes] = True

    param_estimate = 0
    param_estimates = getattr(ir, "param_estimates", None)
    if param_estimates is not None and reachable_nodes.size:
        param_estimate = int(np.asarray(param_estimates)[reachable_mask].sum())

    return StructuralAnalysisResult(
        has_gradient_path=bool(result.get("has_input_path", False)),
        reachable_count=int(reachable_nodes.size),
        depth=int(result.get("max_depth", 0)),
        has_cycle=False,
        param_estimate=param_estimate,
        reachable_mask=reachable_mask if include_reachable else None,
        backend="aria_core",
    )


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


def analyze_ir_in_python(
    ir: Any, *, include_reachable: bool = False
) -> StructuralAnalysisResult:
    n_nodes = int(len(ir.op_codes))
    reachable_mask = np.zeros(n_nodes, dtype=bool)
    has_gradient_path = False
    reachable_count = 0

    if 0 <= int(ir.output_node_idx) < n_nodes:
        stack = [int(ir.output_node_idx)]
        while stack:
            node_idx = stack.pop()
            if reachable_mask[node_idx]:
                continue
            reachable_mask[node_idx] = True
            reachable_count += 1
            if int(ir.op_codes[node_idx]) == 0:
                has_gradient_path = True
            for parent_idx in ir.input_indices[node_idx]:
                parent = int(parent_idx)
                if parent != -1 and not reachable_mask[parent]:
                    stack.append(parent)

    in_degree = np.zeros(n_nodes, dtype=np.int32)
    children = [[] for _ in range(n_nodes)]
    for node_idx in range(n_nodes):
        for parent_idx in ir.input_indices[node_idx]:
            parent = int(parent_idx)
            if parent != -1:
                in_degree[node_idx] += 1
                children[parent].append(node_idx)

    queue = [idx for idx, deg in enumerate(in_degree.tolist()) if deg == 0]
    topo_depth = np.zeros(n_nodes, dtype=np.int32)
    head = 0
    visited = 0
    while head < len(queue):
        node_idx = queue[head]
        head += 1
        visited += 1
        next_depth = int(topo_depth[node_idx]) + 1
        for child in children[node_idx]:
            if next_depth > topo_depth[child]:
                topo_depth[child] = next_depth
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    param_estimate = 0
    param_estimates = getattr(ir, "param_estimates", None)
    if param_estimates is not None and reachable_count:
        param_estimate = int(np.asarray(param_estimates)[reachable_mask].sum())

    return StructuralAnalysisResult(
        has_gradient_path=has_gradient_path,
        reachable_count=reachable_count,
        depth=int(topo_depth[reachable_mask].max()) if reachable_count else 0,
        has_cycle=visited < n_nodes,
        param_estimate=param_estimate,
        reachable_mask=reachable_mask if include_reachable else None,
        backend="python",
    )
