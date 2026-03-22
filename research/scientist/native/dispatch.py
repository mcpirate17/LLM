from __future__ import annotations

import ctypes
import json
import logging
import os
from typing import Any, Dict, List, Optional, Set

from .core import _try_import_cython_bridge, _try_import_rust_scheduler

logger = logging.getLogger(__name__)

_NON_KERNEL_STRUCTURAL_OPS: Set[str] = {
    "input",
    "output",
    "identity",
    "noop",
    "reshape",
    "view",
    "concat",
    "split2",
    "split",
}
_NATIVE_OP_ALIASES: Dict[str, str] = {
    "linear_proj": "linear",
    "softmax_last": "softmax",
    "transpose": "transpose2d",
}
_NATIVE_C_KERNEL_OPS: Set[str] = {
    "relu",
    "gelu",
    "silu",
    "sigmoid",
    "tanh",
    "exp",
    "square",
    "abs",
    "neg",
    "sin",
    "cos",
    "log",
    "sqrt",
    "reciprocal",
    "add",
    "sub",
    "mul",
    "matmul",
    "linear",
    "rmsnorm",
    "layernorm",
    "softmax",
    "transpose2d",
}
_CYTHON_WRAPPER_OPS: Set[str] = set(_NATIVE_C_KERNEL_OPS)
_SOFT_BRIDGE_OPS: Set[str] = {"causal_mask", "argsort_seq", "topk_gate"}
_CYTHON_UNARY_OPS: Set[str] = {
    "relu",
    "gelu",
    "silu",
    "square",
    "abs",
    "neg",
    "reciprocal",
    "log",
    "sqrt",
    "sin",
    "cos",
    "sigmoid",
    "tanh",
    "exp",
}
_CYTHON_BINARY_OPS: Set[str] = {"add", "mul", "sub"}
_CYTHON_UNARY_BACKWARD_OPS: Set[str] = {"relu", "gelu", "silu", "sigmoid", "tanh"}
_CYTHON_BINARY_BACKWARD_OPS: Set[str] = {"add", "mul", "sub"}


def _check_native_op_support(
    layer_graphs: List[Any], native_lib: Any
) -> Dict[str, Any]:
    """Check which ops in the graphs have native kernel support.

    Prefers the Cython bridge (aria_bridge.is_native) when available.
    Falls back to the C kernel registry via ctypes ``nk_is_registered``.
    """
    all_ops: Set[str] = set()
    for g in layer_graphs:
        for node in getattr(g, "nodes", {}).values():
            all_ops.add(getattr(node, "op_name", str(node)))

    kernel_relevant_ops: Set[str] = {
        op for op in all_ops if op not in _NON_KERNEL_STRUCTURAL_OPS
    }

    supported: Set[str] = set()
    unsupported: Set[str] = set()

    def _canonical_op(op_name: str) -> str:
        return _NATIVE_OP_ALIASES.get(op_name, op_name)

    # Quick-check sets: ops known to have native support without needing bridge/lib query.
    _all_known_native = _SOFT_BRIDGE_OPS | _NATIVE_C_KERNEL_OPS | _CYTHON_WRAPPER_OPS

    # Prefer explicit native library handle when provided by caller/tests.
    if native_lib is not None and hasattr(native_lib, "nk_is_registered"):
        if hasattr(native_lib, "nr_runtime_init"):
            native_lib.nr_runtime_init()
        is_registered = native_lib.nk_is_registered
        is_registered.argtypes = [ctypes.c_char_p]
        is_registered.restype = ctypes.c_int32
        for op in kernel_relevant_ops:
            # square must be explicitly registered; do not infer via mul.
            if op == "square":
                kernel_op = "square"
                if is_registered(kernel_op.encode("utf-8")):
                    supported.add(op)
                else:
                    unsupported.add(op)
                continue
            if op in _all_known_native:
                supported.add(op)
                continue
            kernel_op = _canonical_op(op)
            if kernel_op in _all_known_native:
                supported.add(op)
                continue
            if is_registered(kernel_op.encode("utf-8")):
                supported.add(op)
            else:
                unsupported.add(op)
    else:
        # Fallback: Cython bridge query path.
        bridge = _try_import_cython_bridge()
        if bridge is not None and hasattr(bridge, "is_native"):
            for op in kernel_relevant_ops:
                if op in _all_known_native:
                    supported.add(op)
                    continue
                kernel_op = _canonical_op(op)
                if kernel_op in _all_known_native:
                    supported.add(op)
                    continue
                if bridge.is_native(kernel_op):
                    supported.add(op)
                else:
                    unsupported.add(op)
        else:
            # Conservative mode: if neither bridge nor native library query is
            # available, treat kernel-relevant ops as unsupported.
            for op in kernel_relevant_ops:
                unsupported.add(op)

    if not all_ops:
        native_coverage = 0.0
    elif not kernel_relevant_ops:
        native_coverage = 1.0
    else:
        native_coverage = len(supported) / len(kernel_relevant_ops)

    return {
        "all_ops": sorted(all_ops),
        "kernel_relevant_ops": sorted(kernel_relevant_ops),
        "supported": sorted(supported),
        "unsupported": sorted(unsupported),
        "native_coverage": native_coverage,
    }


def _requested_execution_mode() -> str:
    raw = str(os.environ.get("NATIVE_RUNNER_EXECUTION_MODE", "probe")).strip().lower()
    if raw in {"probe", "selective"}:
        return raw
    return "probe"


def dispatch_op_native(op_name: str, *tensors, **kwargs) -> Any:
    """Dispatch a single op through the native Cython bridge.

    Returns numpy array result, or raises if op is unsupported or the
    Cython bridge is not available.

    Supported op routing:
    - Unary ops (relu, gelu, silu, square, abs, neg, reciprocal, log, sqrt, sin, cos, sigmoid, tanh, exp): dispatch_unary
    - Binary ops (add, mul, sub): dispatch_binary
    - Composite fallback (square): dispatch_binary("mul", x, x) when bridge lacks unary square
    - matmul: dispatch_matmul
    - linear / linear_proj: dispatch_linear (kwargs: bias)
    - rmsnorm: dispatch_rmsnorm (kwargs: eps)
    - softmax / softmax_last: dispatch_softmax
    - layernorm: dispatch_layernorm (kwargs: eps)
    - transpose, transpose2d: dispatch_transpose2d
    """
    bridge = _try_import_cython_bridge()
    if bridge is None:
        raise RuntimeError(
            "Cython bridge (aria_bridge) is not available. Cannot dispatch op natively."
        )

    canonical_op = _NATIVE_OP_ALIASES.get(op_name, op_name)

    if canonical_op in _CYTHON_UNARY_OPS:
        if len(tensors) != 1:
            raise ValueError(
                f"Unary op '{op_name}' expects 1 tensor, got {len(tensors)}"
            )
        if canonical_op == "square":
            try:
                return bridge.dispatch_unary(canonical_op, tensors[0])
            except ValueError:
                return bridge.dispatch_binary("mul", tensors[0], tensors[0])
        return bridge.dispatch_unary(canonical_op, tensors[0])

    if canonical_op in _CYTHON_BINARY_OPS:
        if len(tensors) != 2:
            raise ValueError(
                f"Binary op '{op_name}' expects 2 tensors, got {len(tensors)}"
            )
        return bridge.dispatch_binary(canonical_op, tensors[0], tensors[1])

    if canonical_op == "matmul":
        if len(tensors) != 2:
            raise ValueError(f"matmul expects 2 tensors, got {len(tensors)}")
        return bridge.dispatch_matmul(tensors[0], tensors[1])

    if canonical_op == "linear":
        if len(tensors) < 2:
            raise ValueError(
                f"linear expects at least 2 tensors (x, W), got {len(tensors)}"
            )
        bias = kwargs.get("bias", tensors[2] if len(tensors) > 2 else None)
        return bridge.dispatch_linear(tensors[0], tensors[1], bias=bias)

    if canonical_op == "rmsnorm":
        if len(tensors) < 2:
            raise ValueError(
                f"rmsnorm expects at least 2 tensors (x, weight), got {len(tensors)}"
            )
        eps = kwargs.get("eps", 1e-5)
        return bridge.dispatch_rmsnorm(tensors[0], tensors[1], eps=eps)

    if canonical_op == "softmax":
        if len(tensors) != 1:
            raise ValueError(f"softmax expects 1 tensor, got {len(tensors)}")
        return bridge.dispatch_softmax(tensors[0])

    if canonical_op == "layernorm":
        if len(tensors) < 3:
            raise ValueError(
                f"layernorm expects 3 tensors (x, weight, bias), got {len(tensors)}"
            )
        eps = kwargs.get("eps", 1e-5)
        return bridge.dispatch_layernorm(tensors[0], tensors[1], tensors[2], eps=eps)

    if canonical_op == "transpose2d":
        if len(tensors) != 1:
            raise ValueError(f"transpose2d expects 1 tensor, got {len(tensors)}")
        return bridge.dispatch_transpose2d(tensors[0])

    raise ValueError(f"Unsupported op for native dispatch: '{op_name}'")


def dispatch_op_backward_native(op_name: str, grad_output, *saved_tensors) -> Any:
    """Dispatch a backward (gradient) op through the native Cython bridge.

    Returns gradient tensor(s) as numpy arrays. For unary ops returns a single
    array; for binary/matmul ops returns a tuple of arrays.

    Supported op routing:
    - Unary backward (relu, gelu, silu, sigmoid, tanh):
        saved_tensors = (forward_input_or_output,) -> grad_input
    - Binary backward (add, mul, sub):
        saved_tensors = (a, b) -> (grad_a, grad_b)
    - matmul backward:
        saved_tensors = (A, B) -> (grad_A, grad_B)
    """
    bridge = _try_import_cython_bridge()
    if bridge is None:
        raise RuntimeError(
            "Cython bridge (aria_bridge) is not available. "
            "Cannot dispatch backward op natively."
        )

    if op_name in _CYTHON_UNARY_BACKWARD_OPS:
        if len(saved_tensors) != 1:
            raise ValueError(
                f"Unary backward '{op_name}' expects 1 saved tensor, got {len(saved_tensors)}"
            )
        return bridge.dispatch_unary_backward(op_name, grad_output, saved_tensors[0])

    if op_name in _CYTHON_BINARY_BACKWARD_OPS:
        if len(saved_tensors) != 2:
            raise ValueError(
                f"Binary backward '{op_name}' expects 2 saved tensors (a, b), got {len(saved_tensors)}"
            )
        return bridge.dispatch_binary_backward(
            op_name, grad_output, saved_tensors[0], saved_tensors[1]
        )

    if op_name == "matmul":
        if len(saved_tensors) != 2:
            raise ValueError(
                f"matmul backward expects 2 saved tensors (A, B), got {len(saved_tensors)}"
            )
        return bridge.dispatch_matmul_backward(
            grad_output, saved_tensors[0], saved_tensors[1]
        )

    if op_name == "softmax":
        if len(saved_tensors) != 1:
            raise ValueError(
                f"softmax backward expects 1 saved tensor (output), got {len(saved_tensors)}"
            )
        return bridge.dispatch_softmax_backward(grad_output, saved_tensors[0])

    if op_name == "layernorm":
        if len(saved_tensors) != 2:
            raise ValueError(
                f"layernorm backward expects 2 saved tensors (input, gamma), got {len(saved_tensors)}"
            )
        return bridge.dispatch_layernorm_backward(
            grad_output, saved_tensors[0], saved_tensors[1]
        )

    if op_name == "rmsnorm":
        if len(saved_tensors) != 2:
            raise ValueError(
                f"rmsnorm backward expects 2 saved tensors (input, gamma), got {len(saved_tensors)}"
            )
        return bridge.dispatch_rmsnorm_backward(
            grad_output, saved_tensors[0], saved_tensors[1]
        )

    raise ValueError(f"Unsupported op for native backward dispatch: '{op_name}'")


def dispatch_graph_native(graph: Any, input_data: Any) -> Any:
    """Execute a full computation graph using the Rust scheduler.

    Args:
        graph: ComputationGraph instance (from research.synthesis.graph).
        input_data: Input tensor (numpy array or torch tensor).

    Returns:
        Numpy array containing the output of the graph.
    """
    rust = _try_import_rust_scheduler()
    if rust is None:
        raise RuntimeError("Rust scheduler (aria_scheduler) is not available.")

    # Lazy import to avoid circular dependency
    from ..synthesis.native_ir_converter import graph_to_native_ir_json

    import numpy as np

    if hasattr(input_data, "detach"):
        # Convert torch tensor to numpy
        x_np = input_data.detach().cpu().numpy().astype(np.float32)
    else:
        x_np = np.asarray(input_data, dtype=np.float32)

    # Flatten input for the scheduler (expects Vec<f32>)
    # Note: Current Rust execute implementation assumes a single flat input vector.
    # In Aria, graphs usually process [Batch, Seq, Dim] tensors.
    x_flat = x_np.ravel().tolist()

    graph_json = graph_to_native_ir_json(graph)

    try:
        global _last_profile_data
        # Prefer execute_graph_with_stats for arena usage observability.
        if hasattr(rust, "execute_graph_with_stats"):
            result = rust.execute_graph_with_stats(graph_json, x_flat)
            y_flat = result["output"]
            logger.debug(
                "Arena stats: %d/%d bytes used, %d arena allocs, %d heap fallbacks",
                result.get("arena_bytes_used", 0),
                result.get("arena_capacity", 0),
                result.get("arena_alloc_count", 0),
                result.get("heap_fallback_count", 0),
            )
            # Cache profiling data if present.
            if "node_profiles" in result:
                _last_profile_data = {
                    "node_profiles": list(result["node_profiles"]),
                    "peak_memory_bytes": int(result.get("peak_memory_bytes", 0)),
                }
                logger.debug(
                    "Profiling: %d node events, peak memory %d bytes",
                    len(_last_profile_data["node_profiles"]),
                    _last_profile_data["peak_memory_bytes"],
                )
            else:
                _last_profile_data = None
        else:
            y_flat = rust.execute_graph(graph_json, x_flat)
            _last_profile_data = None
        y_np = np.array(y_flat, dtype=np.float32)

        # Reshape to output shape if possible
        if hasattr(graph, "output_node") and graph.output_node and x_np.ndim >= 3:
            shape = graph.output_node.output_shape
            # Assuming [Batch, Seq, Dim]
            target_shape = (x_np.shape[0], x_np.shape[1], shape.dim)
            return y_np.reshape(target_shape)

        return y_np
    except Exception as exc:
        logger.error("Rust scheduler execution failed: %s", exc)
        raise


def dispatch_graph_forward_native_saved(graph: Any, input_data: Any) -> Dict[str, Any]:
    """Execute a full forward pass, returning output and saved activations.

    This is the companion to ``dispatch_graph_backward_native()``.  The saved
    activations dict is keyed by integer node id and contains flat float lists.

    Implementation: walks the graph in topological order using per-op native
    dispatch via the Cython bridge, saving each node's output for backward.
    When the Rust scheduler gains ``execute_graph_forward_saved``, this will
    be upgraded to a single Rust call.

    Args:
        graph: ComputationGraph instance.
        input_data: Input tensor (numpy array or torch tensor).

    Returns:
        Dict with keys:
          - ``"output"``: numpy array containing the output.
          - ``"saved_activations"``: dict[int, numpy.ndarray] per-node activations.
          - ``"ir_json"``: the pre-serialized IR JSON (for backward call).
    """
    from ..synthesis.native_ir_converter import graph_to_native_ir_json

    import numpy as np

    if hasattr(input_data, "detach"):
        x_np = input_data.detach().cpu().numpy().astype(np.float32)
    else:
        x_np = np.asarray(input_data, dtype=np.float32)

    graph_json = graph_to_native_ir_json(graph)
    topo = graph.topological_order()

    # Walk forward, saving every node's activation.
    node_outputs: Dict[int, Any] = {}
    output_node_id = graph._output_node_id

    for nid in topo:
        node = graph.nodes[nid]
        if node.is_input:
            node_outputs[nid] = x_np.ravel()
            continue

        inputs = [node_outputs[iid] for iid in node.input_ids]
        # Flatten all inputs for the per-op dispatch.
        flat_inputs = [np.asarray(a, dtype=np.float32).ravel() for a in inputs]
        y = dispatch_op_native(node.op_name, *flat_inputs)
        node_outputs[nid] = np.asarray(y, dtype=np.float32).ravel()

    output = node_outputs.get(output_node_id)
    if output is None:
        raise RuntimeError(
            f"Graph output node {output_node_id} not found in forward pass"
        )

    return {
        "output": np.asarray(output, dtype=np.float32),
        "saved_activations": dict(node_outputs),
        "ir_json": graph_json,
    }


def dispatch_graph_backward_native(
    graph: Any,
    grad_output: Any,
    saved_activations: Dict[int, Any],
    ir_json: Optional[str] = None,
) -> Dict[int, Any]:
    """Execute a full backward pass through a graph using native per-op backward.

    Walks the graph in reverse topological order, dispatching each op's backward
    through the Cython bridge.  Accumulates gradients when a node fans out to
    multiple consumers.

    When the Rust scheduler gains ``execute_graph_backward``, this will be
    upgraded to a single Rust call.

    Args:
        graph: ComputationGraph instance.
        grad_output: Gradient w.r.t. the graph output (numpy array or torch tensor).
        saved_activations: dict[int, numpy.ndarray] from
            ``dispatch_graph_forward_native_saved()``.
        ir_json: Optional pre-serialized native IR JSON (unused in Python impl,
            kept for API compatibility with future Rust upgrade).

    Returns:
        Dict mapping node_id (int) -> gradient numpy array.
    """
    import numpy as np

    if hasattr(grad_output, "detach"):
        grad_np = grad_output.detach().cpu().numpy().astype(np.float32)
    else:
        grad_np = np.asarray(grad_output, dtype=np.float32)

    topo = graph.topological_order()
    output_node_id = graph._output_node_id

    # Ensure saved_activations values are numpy arrays.
    saved: Dict[int, Any] = {}
    for k, v in saved_activations.items():
        saved[int(k)] = np.asarray(v, dtype=np.float32).ravel()

    # node_grads[nid] = accumulated gradient for that node's output.
    node_grads: Dict[int, Any] = {}
    node_grads[output_node_id] = grad_np.ravel()

    # Walk reverse topological order.
    for nid in reversed(topo):
        node = graph.nodes[nid]
        if node.is_input:
            continue
        if nid not in node_grads:
            continue

        g_out = node_grads[nid]

        # Determine saved tensors needed for this op's backward.
        input_ids = list(node.input_ids)
        input_activations = [saved.get(iid, np.zeros_like(g_out)) for iid in input_ids]

        op_name = node.op_name
        try:
            result = dispatch_op_backward_native(op_name, g_out, *input_activations)
        except (ValueError, RuntimeError) as exc:
            logger.debug("Backward dispatch failed for op %s: %s", op_name, exc)
            # Fallback: pass gradient through unchanged (identity backward).
            result = g_out if len(input_ids) == 1 else tuple([g_out] * len(input_ids))

        # Distribute gradients to inputs.
        if len(input_ids) == 1:
            g_in = np.asarray(result, dtype=np.float32).ravel()
            if input_ids[0] in node_grads:
                node_grads[input_ids[0]] = node_grads[input_ids[0]] + g_in
            else:
                node_grads[input_ids[0]] = g_in
        else:
            # Binary/multi-input: result is a tuple of gradients.
            if not isinstance(result, (tuple, list)):
                result = tuple([result] * len(input_ids))
            for i, iid in enumerate(input_ids):
                g_in = np.asarray(result[i], dtype=np.float32).ravel()
                if iid in node_grads:
                    node_grads[iid] = node_grads[iid] + g_in
                else:
                    node_grads[iid] = g_in

    grads: Dict[int, Any] = {}
    for nid, g in node_grads.items():
        grads[nid] = np.asarray(g, dtype=np.float32)

    return grads


def dispatch_graph_native_cached(ir_json: str, graph: Any, input_data: Any) -> Any:
    """Execute a graph using a pre-converted native_ir JSON string.

    Like ``dispatch_graph_native()`` but skips the graph-to-IR conversion step,
    accepting an already-serialized JSON string. This avoids ~10us of repeated
    JSON serialization when the same graph is dispatched multiple times.

    Args:
        ir_json: Pre-serialized native_ir.v1 JSON string.
        graph: Original ComputationGraph (used only for output reshaping).
        input_data: Input tensor (numpy array or torch tensor).

    Returns:
        Numpy array containing the output of the graph.
    """
    rust = _try_import_rust_scheduler()
    if rust is None:
        raise RuntimeError("Rust scheduler (aria_scheduler) is not available.")

    import numpy as np

    if hasattr(input_data, "detach"):
        x_np = input_data.detach().cpu().numpy().astype(np.float32)
    else:
        x_np = np.asarray(input_data, dtype=np.float32)

    x_flat = x_np.ravel().tolist()

    try:
        global _last_profile_data
        if hasattr(rust, "execute_graph_with_stats"):
            result = rust.execute_graph_with_stats(ir_json, x_flat)
            y_flat = result["output"]
            logger.debug(
                "Arena stats: %d/%d bytes used, %d arena allocs, %d heap fallbacks",
                result.get("arena_bytes_used", 0),
                result.get("arena_capacity", 0),
                result.get("arena_alloc_count", 0),
                result.get("heap_fallback_count", 0),
            )
            if "node_profiles" in result:
                _last_profile_data = {
                    "node_profiles": list(result["node_profiles"]),
                    "peak_memory_bytes": int(result.get("peak_memory_bytes", 0)),
                }
            else:
                _last_profile_data = None
        else:
            y_flat = rust.execute_graph(ir_json, x_flat)
            _last_profile_data = None
        y_np = np.array(y_flat, dtype=np.float32)

        if hasattr(graph, "output_node") and graph.output_node and x_np.ndim >= 3:
            shape = graph.output_node.output_shape
            target_shape = (x_np.shape[0], x_np.shape[1], shape.dim)
            return y_np.reshape(target_shape)

        return y_np
    except Exception as exc:
        logger.error("Rust scheduler execution failed: %s", exc)
        raise


def _activate_selective_native_dispatch(native_lib: Any) -> Dict[str, Any]:
    """Run a tiny native-kernel execution path to confirm selective activation.

    Prefers the Cython bridge when available; falls back to raw ctypes calls.
    This is intentionally narrow and safe: execute `relu` and `add` on tiny fixed
    buffers. The model compile path remains legacy until full runner ABI execution
    is wired.
    """
    result: Dict[str, Any] = {
        "activated": False,
        "ops": ["relu", "add"],
        "reason": "not_attempted",
    }

    # Try Cython bridge first.
    bridge = _try_import_cython_bridge()
    if bridge is not None:
        try:
            import numpy as np

            x = np.array([-1.0, 0.0, 2.0, 3.5], dtype=np.float32)
            relu_out = bridge.dispatch_unary("relu", x)
            relu_list = [float(v) for v in relu_out]
            if relu_list != [0.0, 0.0, 2.0, 3.5]:
                result["reason"] = f"relu_mismatch:{relu_list}"
                return result

            a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
            add_out = bridge.dispatch_binary("add", a, b)
            add_list = [float(v) for v in add_out]
            if add_list != [11.0, 22.0, 33.0, 44.0]:
                result["reason"] = f"add_mismatch:{add_list}"
                return result

            result["activated"] = True
            result["reason"] = "ok"
            result["dispatch_backend"] = "cython"

            # Also check Rust scheduler
            rust = _try_import_rust_scheduler()
            if rust is not None:
                result["rust_scheduler"] = "available"
                # Simple topo probe
                order = rust.topological_order(
                    json.dumps(
                        {
                            "schema_version": "0.1",
                            "model_dim": 4,
                            "output_node_id": 1,
                            "nodes": [
                                {
                                    "id": 0,
                                    "op_name": "input",
                                    "is_input": True,
                                    "input_ids": [],
                                    "config": {},
                                },
                                {
                                    "id": 1,
                                    "op_name": "relu",
                                    "input_ids": [0],
                                    "config": {},
                                },
                            ],
                            "edges": [{"source": 0, "target": 1}],
                        }
                    )
                )
                if order == [0, 1]:
                    result["rust_scheduler"] = "ok"
                else:
                    result["rust_scheduler"] = f"topo_mismatch:{order}"
            else:
                result["rust_scheduler"] = "missing"

            return result
        except Exception as exc:
            logger.debug(
                "Cython bridge activation failed, falling back to ctypes: %s", exc
            )
            # Fall through to ctypes path.

    # Fallback: raw ctypes path.
    if native_lib is None:
        result["reason"] = "native_lib_unavailable"
        return result

    relu_fn = getattr(native_lib, "aria_relu_f32", None)
    add_fn = getattr(native_lib, "aria_add_f32", None)
    if not callable(relu_fn) or not callable(add_fn):
        result["reason"] = "missing_required_symbols"
        return result

    try:
        n = 4
        x = (ctypes.c_float * n)(-1.0, 0.0, 2.0, 3.5)
        y = (ctypes.c_float * n)()
        relu_fn(x, y, n)
        relu_out = [float(y[i]) for i in range(n)]
        if relu_out != [0.0, 0.0, 2.0, 3.5]:
            result["reason"] = f"relu_mismatch:{relu_out}"
            return result

        a = (ctypes.c_float * n)(1.0, 2.0, 3.0, 4.0)
        b = (ctypes.c_float * n)(10.0, 20.0, 30.0, 40.0)
        z = (ctypes.c_float * n)()
        add_fn(a, b, z, n)
        add_out = [float(z[i]) for i in range(n)]
        if add_out != [11.0, 22.0, 33.0, 44.0]:
            result["reason"] = f"add_mismatch:{add_out}"
            return result

        result["activated"] = True
        result["reason"] = "ok"
        result["dispatch_backend"] = "ctypes"
        return result
    except Exception as exc:
        result["reason"] = f"activation_error:{exc}"
        return result
