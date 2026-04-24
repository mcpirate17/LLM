from __future__ import annotations

import ctypes
from functools import lru_cache
from typing import Optional

from .native_analysis_bindings import load_native_graph_analysis_lib


@lru_cache(maxsize=512)
def evaluate_param_formula_natively(formula: str) -> Optional[int]:
    lib = load_native_graph_analysis_lib()
    if lib is None or not hasattr(lib, "aria_eval_param_formula"):
        return None

    fn = lib.aria_eval_param_formula
    out_value = ctypes.c_int64()
    status = fn(formula.encode("ascii"), ctypes.byref(out_value))
    if status != 0:
        return None
    return int(out_value.value)
