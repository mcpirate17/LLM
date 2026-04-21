"""Rust scheduler execution helpers for full-graph forward/backward.

Extracted from ``dispatch.py`` to keep the main dispatch module focused on
op-level routing. Only the lower-level "call into aria-scheduler with this
JSON IR" glue lives here. Importers should continue to use ``dispatch``
for the public surface.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Dict, Optional

from .tensor_bridge import to_native_array

logger = logging.getLogger(__name__)


def _get_rust_scheduler() -> Any:
    """Resolve the Rust scheduler via the dispatch module.

    Tests monkeypatch ``dispatch._try_import_rust_scheduler`` and
    ``dispatch._compile_rust_graph_handle`` to inject fakes. Routing through
    ``dispatch`` keeps those patches effective for functions that live in
    this sibling module without creating a circular import at module load.
    """
    from . import dispatch as _dispatch

    return _dispatch._try_import_rust_scheduler()


def _get_graph_handle(graph_json: str) -> Any:
    from . import dispatch as _dispatch

    return _dispatch._compile_rust_graph_handle(graph_json)


@lru_cache(maxsize=256)
def _compile_rust_graph_handle(graph_json: str) -> Any:
    rust = _get_rust_scheduler()
    if rust is None or not hasattr(rust, "compile_graph_ir_handle"):
        return None
    try:
        return rust.compile_graph_ir_handle(graph_json)
    except Exception as exc:
        logger.debug("Rust graph-handle compilation failed: %s", exc)
        return None


def _prepare_graph_input(
    graph: Any,
    input_data: Any,
    *,
    ir_json: Optional[str] = None,
):
    """Shared setup for graph dispatch: convert input + serialize IR.

    Returns (x_np, graph_json) where x_np is a float32 numpy array
    and graph_json is the serialized native IR JSON string.
    """
    from ...synthesis.native_ir_converter import graph_to_native_ir_json

    x_np = to_native_array(input_data)
    graph_json = ir_json if ir_json is not None else graph_to_native_ir_json(graph)
    return x_np, graph_json


def _reshape_graph_output(graph: Any, x_np: Any, y_flat: Any) -> Any:
    import numpy as np

    y_np = np.array(y_flat, dtype=np.float32)
    if hasattr(graph, "output_node") and graph.output_node and x_np.ndim >= 3:
        shape = graph.output_node.output_shape
        target_shape = (x_np.shape[0], x_np.shape[1], shape.dim)
        return y_np.reshape(target_shape)
    return y_np


def _execute_rust_graph_forward_saved(
    *,
    graph_json: str,
    x_np: Any,
    graph: Any,
) -> Optional[Dict[str, Any]]:
    rust = _get_rust_scheduler()
    if rust is None or not (
        hasattr(rust, "execute_graph_forward_saved_arrays_handle")
        or hasattr(rust, "execute_graph_forward_saved_arrays")
        or hasattr(rust, "execute_graph_forward_saved")
    ):
        return None

    import numpy as np

    graph_handle = _get_graph_handle(graph_json)

    if graph_handle is not None and hasattr(
        rust, "execute_graph_forward_saved_compiled_arrays_handle"
    ):
        result = rust.execute_graph_forward_saved_compiled_arrays_handle(
            graph_handle,
            x_np,
        )
        return {
            "output": _reshape_graph_output(graph, x_np, result["output"]),
            "saved_activations": result["saved_state"],
            "ir_json": graph_json,
            "arena_bytes_used": int(result.get("arena_bytes_used", 0)),
            "arena_capacity": int(result.get("arena_capacity", 0)),
        }
    if hasattr(rust, "execute_graph_forward_saved_arrays_handle"):
        result = rust.execute_graph_forward_saved_arrays_handle(graph_json, x_np)
        return {
            "output": _reshape_graph_output(graph, x_np, result["output"]),
            "saved_activations": result["saved_state"],
            "ir_json": graph_json,
            "arena_bytes_used": int(result.get("arena_bytes_used", 0)),
            "arena_capacity": int(result.get("arena_capacity", 0)),
        }
    if hasattr(rust, "execute_graph_forward_saved_arrays"):
        result = rust.execute_graph_forward_saved_arrays(graph_json, x_np)
    else:
        result = rust.execute_graph_forward_saved(graph_json, x_np.ravel().tolist())
    saved_activations = {
        int(node_id): np.asarray(values, dtype=np.float32)
        for node_id, values in dict(result.get("saved_activations", {})).items()
    }
    return {
        "output": _reshape_graph_output(graph, x_np, result["output"]),
        "saved_activations": saved_activations,
        "ir_json": graph_json,
        "arena_bytes_used": int(result.get("arena_bytes_used", 0)),
        "arena_capacity": int(result.get("arena_capacity", 0)),
    }


def _execute_rust_graph_forward_saved_multi_input(
    *,
    graph_json: str,
    native_inputs: list[Any],
    output_shape: tuple[int, ...] | None,
) -> Optional[Dict[str, Any]]:
    rust = _get_rust_scheduler()
    if rust is None or not (
        hasattr(rust, "execute_graph_forward_saved_multi_input_arrays_handle")
        or hasattr(rust, "execute_graph_forward_saved_multi_input_arrays")
        or hasattr(rust, "execute_graph_forward_saved_multi_input")
    ):
        return None

    import numpy as np

    graph_handle = _get_graph_handle(graph_json)

    if graph_handle is not None and hasattr(
        rust, "execute_graph_forward_saved_multi_input_compiled_arrays_handle"
    ):
        result = rust.execute_graph_forward_saved_multi_input_compiled_arrays_handle(
            graph_handle,
            native_inputs,
        )
        output = np.asarray(result["output"], dtype=np.float32)
        if output_shape is not None:
            output = output.reshape(output_shape)
        return {
            "output": output,
            "saved_activations": result["saved_state"],
            "ir_json": graph_json,
            "arena_bytes_used": int(result.get("arena_bytes_used", 0)),
            "arena_capacity": int(result.get("arena_capacity", 0)),
        }
    if hasattr(rust, "execute_graph_forward_saved_multi_input_arrays_handle"):
        result = rust.execute_graph_forward_saved_multi_input_arrays_handle(
            graph_json,
            native_inputs,
        )
        output = np.asarray(result["output"], dtype=np.float32)
        if output_shape is not None:
            output = output.reshape(output_shape)
        return {
            "output": output,
            "saved_activations": result["saved_state"],
            "ir_json": graph_json,
            "arena_bytes_used": int(result.get("arena_bytes_used", 0)),
            "arena_capacity": int(result.get("arena_capacity", 0)),
        }
    if hasattr(rust, "execute_graph_forward_saved_multi_input_arrays"):
        result = rust.execute_graph_forward_saved_multi_input_arrays(
            graph_json,
            native_inputs,
        )
    else:
        result = rust.execute_graph_forward_saved_multi_input(
            graph_json,
            [value.ravel().tolist() for value in native_inputs],
        )
    output = np.asarray(result["output"], dtype=np.float32)
    if output_shape is not None:
        output = output.reshape(output_shape)
    saved_activations = {
        int(node_id): np.asarray(values, dtype=np.float32)
        for node_id, values in dict(result.get("saved_activations", {})).items()
    }
    return {
        "output": output,
        "saved_activations": saved_activations,
        "ir_json": graph_json,
        "arena_bytes_used": int(result.get("arena_bytes_used", 0)),
        "arena_capacity": int(result.get("arena_capacity", 0)),
    }


def _execute_rust_graph_backward(
    *,
    graph_json: str,
    grad_np: Any,
    saved_activations: Any,
) -> Optional[Dict[int, Any]]:
    rust = _get_rust_scheduler()
    if rust is None or not (
        hasattr(rust, "execute_graph_backward_arrays_handle")
        or hasattr(rust, "execute_graph_backward_handle")
        or hasattr(rust, "execute_graph_backward_arrays")
        or hasattr(rust, "execute_graph_backward")
    ):
        return None

    import numpy as np

    graph_handle = _get_graph_handle(graph_json)

    if not isinstance(saved_activations, Mapping):
        if graph_handle is not None and hasattr(
            rust, "execute_graph_backward_compiled_arrays_handle"
        ):
            result = rust.execute_graph_backward_compiled_arrays_handle(
                graph_handle,
                grad_np,
                saved_activations,
            )
            return {
                int(node_id): np.asarray(values, dtype=np.float32)
                for node_id, values in dict(result.get("grads", {})).items()
            }
        if hasattr(rust, "execute_graph_backward_arrays_handle"):
            result = rust.execute_graph_backward_arrays_handle(
                graph_json,
                grad_np,
                saved_activations,
            )
            return {
                int(node_id): np.asarray(values, dtype=np.float32)
                for node_id, values in dict(result.get("grads", {})).items()
            }
        if hasattr(rust, "execute_graph_backward_handle"):
            result = rust.execute_graph_backward_handle(
                graph_json,
                grad_np.ravel().tolist(),
                saved_activations,
            )
            return {
                int(node_id): np.asarray(values, dtype=np.float32)
                for node_id, values in dict(result.get("grads", {})).items()
            }
        return None

    rust_saved = {
        int(node_id): np.asarray(values, dtype=np.float32)
        for node_id, values in saved_activations.items()
    }
    if hasattr(rust, "execute_graph_backward_arrays"):
        result = rust.execute_graph_backward_arrays(
            graph_json,
            grad_np,
            rust_saved,
        )
    else:
        result = rust.execute_graph_backward(
            graph_json,
            grad_np.ravel().tolist(),
            {
                int(node_id): np.asarray(values, dtype=np.float32).ravel().tolist()
                for node_id, values in rust_saved.items()
            },
        )
    return {
        int(node_id): np.asarray(values, dtype=np.float32)
        for node_id, values in dict(result.get("grads", {})).items()
    }
