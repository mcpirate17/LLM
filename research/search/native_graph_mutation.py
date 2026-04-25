from __future__ import annotations

import ctypes
import logging
from functools import lru_cache
from typing import Any

import numpy as np

from ..synthesis.primitives import OPCODE_MAP, REVERSE_OPCODE_MAP, get_primitive
from ._native_runtime import load_native_runtime_lib

logger = logging.getLogger(__name__)

_NATIVE_GRAPH_MUTATION_LIB: Any = False


def load_native_graph_mutation_lib() -> Any:
    global _NATIVE_GRAPH_MUTATION_LIB
    if _NATIVE_GRAPH_MUTATION_LIB is not False:
        return _NATIVE_GRAPH_MUTATION_LIB

    lib = load_native_runtime_lib(("aria_graph_mutation_plan",), logger)
    if lib is None:
        _NATIVE_GRAPH_MUTATION_LIB = None
        return None

    lib.aria_graph_mutation_plan.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.aria_graph_mutation_plan.restype = ctypes.c_int32
    _NATIVE_GRAPH_MUTATION_LIB = lib
    return lib


@lru_cache(maxsize=1)
def _opcode_metadata_tables() -> tuple[np.ndarray, np.ndarray]:
    n_opcodes = max(OPCODE_MAP.values()) + 1
    category_ids = np.full(n_opcodes, -1, dtype=np.int32)
    input_arities = np.zeros(n_opcodes, dtype=np.int32)
    category_index: dict[object, int] = {}

    for opcode in range(1, n_opcodes):
        op_name = REVERSE_OPCODE_MAP.get(opcode)
        if op_name is None:
            continue
        primitive = get_primitive(op_name)
        category_id = category_index.setdefault(primitive.category, len(category_index))
        category_ids[opcode] = category_id
        input_arities[opcode] = primitive.n_inputs

    return category_ids, input_arities


def plan_local_mutation_trials(
    op_codes: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    lib = load_native_graph_mutation_lib()
    if lib is None:
        return None

    graph_opcodes = np.ascontiguousarray(op_codes, dtype=np.int32)
    if graph_opcodes.ndim != 1:
        raise ValueError("op_codes must be rank-1")

    category_ids, input_arities = _opcode_metadata_tables()
    if graph_opcodes.size == 0 or category_ids.size <= 1:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    empty_indices = np.empty(0, dtype=np.int32)
    empty_opcodes = np.empty(0, dtype=np.int32)
    out_pair_count = ctypes.c_int32()
    probe_status = lib.aria_graph_mutation_plan(
        int(graph_opcodes.shape[0]),
        graph_opcodes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        int(category_ids.shape[0]),
        category_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        input_arities.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_uint64(int(seed) & ((1 << 64) - 1)),
        0,
        empty_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        empty_opcodes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(out_pair_count),
    )
    max_pairs = int(out_pair_count.value)
    if max_pairs <= 0 and probe_status == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
    if max_pairs <= 0 or probe_status not in (0, -1):
        return None

    out_node_indices = np.empty(max_pairs, dtype=np.int32)
    out_candidate_opcodes = np.empty(max_pairs, dtype=np.int32)
    status = lib.aria_graph_mutation_plan(
        int(graph_opcodes.shape[0]),
        graph_opcodes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        int(category_ids.shape[0]),
        category_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        input_arities.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_uint64(int(seed) & ((1 << 64) - 1)),
        max_pairs,
        out_node_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_candidate_opcodes.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.byref(out_pair_count),
    )
    if status != 0:
        return None

    used_pairs = int(out_pair_count.value)
    return out_node_indices[:used_pairs], out_candidate_opcodes[:used_pairs]
