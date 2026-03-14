from __future__ import annotations

from typing import Any, Dict

import aria_core


def _coerce_graph_ir(graph_or_ir: Any) -> Any:
    if hasattr(graph_or_ir, "op_codes") and hasattr(graph_or_ir, "input_indices"):
        return graph_or_ir
    lower_to_ir = getattr(graph_or_ir, "lower_to_ir", None)
    if callable(lower_to_ir):
        return lower_to_ir()
    raise TypeError("smoke_test_graph expects a ComputationGraph or ComputationGraphIR")


def smoke_test_graph(graph_or_ir: Any, d_model: int, seq_len: int) -> Dict[str, bool]:
    graph_ir = _coerce_graph_ir(graph_or_ir)
    result = aria_core._C.smoke_test_graph(graph_ir, int(d_model), int(seq_len))
    return {
        "ok": bool(result.get("ok", False)),
        "has_params": bool(result.get("has_params", False)),
        "grad_flows": bool(result.get("grad_flows", False)),
        "no_nan": bool(result.get("no_nan", result.get("no_unsafe", False))),
    }
