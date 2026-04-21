from __future__ import annotations

import ctypes
import json
import logging
from collections.abc import Mapping
import os
from typing import Any, Dict, List, Optional, Set

from .core import _try_import_cython_bridge, _try_import_rust_scheduler
from .tensor_bridge import (
    supports_host_array_bridge,
    to_device_tensor,
    to_native_array,
    to_native_flat_array,
)
from ._dispatch_constants import (
    NATIVE_STRUCTURAL_OPS,
    _CYTHON_BINARY_BACKWARD_OPS,
    _CYTHON_BINARY_OPS,
    _CYTHON_UNARY_BACKWARD_OPS,
    _CYTHON_UNARY_OPS,
    _CYTHON_WRAPPER_OPS,
    _NATIVE_C_KERNEL_OPS,
    _NATIVE_OP_ALIASES,
    _NON_KERNEL_STRUCTURAL_OPS,
    _PER_OP_BRIDGE_ONLY_OPS,
    _RUST_SCHEDULER_UNSUPPORTED_OPS,
    _SOFT_BRIDGE_OPS,
)
from ._dispatch_rust_exec import (
    _compile_rust_graph_handle,
    _execute_rust_graph_backward,
    _execute_rust_graph_forward_saved,
    _execute_rust_graph_forward_saved_multi_input,
    _prepare_graph_input,
    _reshape_graph_output,
)

logger = logging.getLogger(__name__)
_last_profile_data: Dict[str, Any] | None = None
__all__ = ["_PER_OP_BRIDGE_ONLY_OPS"]


def _native_profiling_enabled(rust: Any) -> bool:
    return bool(
        rust is not None
        and hasattr(rust, "profiler_enabled")
        and rust.profiler_enabled()
    )


def _set_last_profile_data(value: Dict[str, Any] | None) -> None:
    global _last_profile_data
    _last_profile_data = value
    try:
        from . import profiling as native_profiling

        native_profiling._last_profile_data = value
    except Exception:
        logger.debug("Failed to sync native profiling cache", exc_info=True)


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

    supported: Set[str] = {op for op in all_ops if op in NATIVE_STRUCTURAL_OPS}
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
            for op in kernel_relevant_ops:
                if op in _all_known_native:
                    supported.add(op)
                    continue
                kernel_op = _canonical_op(op)
                if kernel_op in _all_known_native:
                    supported.add(op)
                    continue
                unsupported.add(op)

    if not all_ops:
        native_coverage = 0.0
    elif not kernel_relevant_ops:
        native_coverage = 1.0
    else:
        native_coverage = len(supported & kernel_relevant_ops) / len(
            kernel_relevant_ops
        )

    return {
        "all_ops": sorted(all_ops),
        "kernel_relevant_ops": sorted(kernel_relevant_ops),
        "supported": sorted(supported),
        "unsupported": sorted(unsupported),
        "native_coverage": native_coverage,
    }


def scheduler_compatible_ops(supported_ops: Set[str]) -> Set[str]:
    """Filter native-supported ops down to the subset safe for Rust subgraph dispatch."""
    return set(supported_ops) - _RUST_SCHEDULER_UNSUPPORTED_OPS


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
    - gated_linear: dispatch_gated_linear (kwargs: bias, bias_gate)
    - rwkv_time_mixing: dispatch_rwkv_time_mixing
    """
    bridge = _try_import_cython_bridge()
    if bridge is None:
        raise RuntimeError(
            "Cython bridge (aria_bridge) is not available. Cannot dispatch op natively."
        )
    if not supports_host_array_bridge(*tensors, *kwargs.values()):
        raise RuntimeError(
            "Host array bridge does not support non-CPU tensors for native op dispatch."
        )

    canonical_op = _NATIVE_OP_ALIASES.get(op_name, op_name)
    native_tensors = tuple(
        None
        if t is None
        else (
            to_native_flat_array(t)
            if canonical_op in _CYTHON_UNARY_OPS | _CYTHON_BINARY_OPS
            else to_native_array(t)
        )
        for t in tensors
    )

    if canonical_op in _CYTHON_UNARY_OPS:
        if len(native_tensors) != 1:
            raise ValueError(
                f"Unary op '{op_name}' expects 1 tensor, got {len(tensors)}"
            )
        if canonical_op == "square":
            try:
                return bridge.dispatch_unary(canonical_op, native_tensors[0])
            except ValueError:
                return bridge.dispatch_binary(
                    "mul", native_tensors[0], native_tensors[0]
                )
        return bridge.dispatch_unary(canonical_op, native_tensors[0])

    if canonical_op in _CYTHON_BINARY_OPS:
        if len(native_tensors) != 2:
            raise ValueError(
                f"Binary op '{op_name}' expects 2 tensors, got {len(tensors)}"
            )
        return bridge.dispatch_binary(
            canonical_op, native_tensors[0], native_tensors[1]
        )

    if canonical_op == "matmul":
        if len(native_tensors) != 2:
            raise ValueError(f"matmul expects 2 tensors, got {len(tensors)}")
        return bridge.dispatch_matmul(native_tensors[0], native_tensors[1])

    if canonical_op == "linear":
        if len(native_tensors) < 2:
            raise ValueError(
                f"linear expects at least 2 tensors (x, W), got {len(tensors)}"
            )
        bias_value = kwargs.get(
            "bias", native_tensors[2] if len(native_tensors) > 2 else None
        )
        bias = None if bias_value is None else to_native_array(bias_value)
        return bridge.dispatch_linear(native_tensors[0], native_tensors[1], bias=bias)

    if canonical_op == "rmsnorm":
        if len(native_tensors) < 2:
            raise ValueError(
                f"rmsnorm expects at least 2 tensors (x, weight), got {len(tensors)}"
            )
        eps = kwargs.get("eps", 1e-5)
        return bridge.dispatch_rmsnorm(native_tensors[0], native_tensors[1], eps=eps)

    if canonical_op == "softmax":
        if len(native_tensors) != 1:
            raise ValueError(f"softmax expects 1 tensor, got {len(tensors)}")
        return bridge.dispatch_softmax(native_tensors[0])

    if canonical_op == "layernorm":
        if len(native_tensors) < 3:
            raise ValueError(
                f"layernorm expects 3 tensors (x, weight, bias), got {len(tensors)}"
            )
        eps = kwargs.get("eps", 1e-5)
        return bridge.dispatch_layernorm(
            native_tensors[0], native_tensors[1], native_tensors[2], eps=eps
        )

    if canonical_op == "transpose2d":
        if len(native_tensors) != 1:
            raise ValueError(f"transpose2d expects 1 tensor, got {len(tensors)}")
        return bridge.dispatch_transpose2d(native_tensors[0])

    if canonical_op == "gated_linear":
        if len(native_tensors) < 3:
            raise ValueError("gated_linear expects at least 3 tensors (x, W, W_gate)")
        return bridge.dispatch_gated_linear(
            native_tensors[0],
            native_tensors[1],
            native_tensors[2],
            bias=kwargs.get("bias"),
            bias_gate=kwargs.get("bias_gate"),
        )

    if canonical_op == "rwkv_time_mixing":
        if len(native_tensors) != 6:
            raise ValueError(
                "rwkv_time_mixing expects 6 tensors (x, w_decay, u_bonus, W_k, W_v, W_r)"
            )
        return bridge.dispatch_rwkv_time_mixing(
            native_tensors[0],
            native_tensors[1],
            native_tensors[2],
            native_tensors[3],
            native_tensors[4],
            native_tensors[5],
        )

    if canonical_op == "rwkv_channel":
        if len(native_tensors) != 6:
            raise ValueError(
                "rwkv_channel expects 6 tensors (x, mix_k, mix_r, W_k, W_r, W_v)"
            )
        return bridge.dispatch_rwkv_channel(
            native_tensors[0],
            native_tensors[1],
            native_tensors[2],
            native_tensors[3],
            native_tensors[4],
            native_tensors[5],
        )

    if canonical_op == "depth_weighted_proj":
        if len(native_tensors) != 3:
            raise ValueError(
                "depth_weighted_proj expects 3 tensors (x, depth_scorer, step_projs)"
            )
        return bridge.dispatch_depth_weighted_proj(
            native_tensors[0],
            native_tensors[1],
            native_tensors[2],
        )

    if canonical_op == "conv1d_seq":
        if len(native_tensors) not in {2, 3}:
            raise ValueError("conv1d_seq expects x, weight, and optional bias")
        return bridge.dispatch_conv1d_seq(
            native_tensors[0],
            native_tensors[1],
            bias=native_tensors[2] if len(native_tensors) > 2 else None,
        )

    if canonical_op == "swiglu":
        if len(native_tensors) not in {4, 5, 6, 7}:
            raise ValueError(
                "swiglu expects x, W_gate, W_up, W_down, and optional biases"
            )
        return bridge.dispatch_swiglu(
            native_tensors[0],
            native_tensors[1],
            native_tensors[2],
            native_tensors[3],
            bias_gate=native_tensors[4] if len(native_tensors) > 4 else None,
            bias_up=native_tensors[5] if len(native_tensors) > 5 else None,
            bias_down=native_tensors[6] if len(native_tensors) > 6 else None,
        )

    if canonical_op == "softmax_attention":
        if len(native_tensors) != 5:
            raise ValueError("softmax_attention expects 5 tensors (x, Wq, Wk, Wv, Wo)")
        n_heads = int(kwargs.get("n_heads", 0))
        if n_heads <= 0:
            raise ValueError("softmax_attention requires positive n_heads")
        return bridge.dispatch_softmax_attention(
            native_tensors[0],
            native_tensors[1],
            native_tensors[2],
            native_tensors[3],
            native_tensors[4],
            n_heads=n_heads,
        )

    if canonical_op == "linear_attention":
        if len(native_tensors) != 5:
            raise ValueError("linear_attention expects 5 tensors (x, Wq, Wk, Wv, Wo)")
        return bridge.dispatch_linear_attention(
            native_tensors[0],
            native_tensors[1],
            native_tensors[2],
            native_tensors[3],
            native_tensors[4],
        )

    if canonical_op == "selective_scan":
        if len(native_tensors) != 5:
            raise ValueError(
                "selective_scan expects 5 tensors (x, A_log, dt_proj, B_weight, C_weight)"
            )
        return bridge.dispatch_selective_scan_compiled(
            native_tensors[0],
            native_tensors[1],
            native_tensors[2],
            native_tensors[3],
            native_tensors[4],
        )

    if canonical_op == "state_space":
        if len(native_tensors) != 7:
            raise ValueError(
                "state_space expects 7 tensors (x, ssm_A, ssm_B_weight, ssm_C_weight, ssm_D, ssm_dt_weight, ssm_dt_bias)"
            )
        return bridge.dispatch_state_space_compiled(
            native_tensors[0],
            native_tensors[1],
            native_tensors[2],
            native_tensors[3],
            native_tensors[4],
            native_tensors[5],
            native_tensors[6],
        )

    if canonical_op == "gated_delta":
        if len(native_tensors) != 7:
            raise ValueError(
                "gated_delta expects 7 tensors (x, q_weight, k_weight, v_weight, alpha_weight, beta_weight, o_weight)"
            )
        n_heads = int(kwargs.get("n_heads", 0))
        if n_heads <= 0:
            raise ValueError("gated_delta requires positive n_heads")
        return bridge.dispatch_gated_delta_compiled(
            native_tensors[0],
            native_tensors[1],
            native_tensors[2],
            native_tensors[3],
            native_tensors[4],
            native_tensors[5],
            native_tensors[6],
            n_heads=n_heads,
        )

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
    grad_native = (
        to_native_flat_array(grad_output)
        if op_name in _CYTHON_UNARY_BACKWARD_OPS | _CYTHON_BINARY_BACKWARD_OPS
        else to_native_array(grad_output)
    )
    saved_native = tuple(
        to_native_flat_array(t)
        if op_name in _CYTHON_UNARY_BACKWARD_OPS | _CYTHON_BINARY_BACKWARD_OPS
        else to_native_array(t)
        for t in saved_tensors
    )

    if op_name in _CYTHON_UNARY_BACKWARD_OPS:
        if len(saved_native) != 1:
            raise ValueError(
                f"Unary backward '{op_name}' expects 1 saved tensor, got {len(saved_tensors)}"
            )
        return bridge.dispatch_unary_backward(op_name, grad_native, saved_native[0])

    if op_name in _CYTHON_BINARY_BACKWARD_OPS:
        if len(saved_native) != 2:
            raise ValueError(
                f"Binary backward '{op_name}' expects 2 saved tensors (a, b), got {len(saved_tensors)}"
            )
        return bridge.dispatch_binary_backward(
            op_name, grad_native, saved_native[0], saved_native[1]
        )

    if op_name == "matmul":
        if len(saved_native) != 2:
            raise ValueError(
                f"matmul backward expects 2 saved tensors (A, B), got {len(saved_tensors)}"
            )
        return bridge.dispatch_matmul_backward(
            grad_native, saved_native[0], saved_native[1]
        )

    if op_name == "linear":
        if len(saved_native) != 2:
            raise ValueError(
                f"linear backward expects 2 saved tensors (x, W), got {len(saved_tensors)}"
            )
        return bridge.dispatch_matmul_backward(
            grad_native, saved_native[0], saved_native[1]
        )

    if op_name == "softmax":
        if len(saved_native) != 1:
            raise ValueError(
                f"softmax backward expects 1 saved tensor (output), got {len(saved_tensors)}"
            )
        return bridge.dispatch_softmax_backward(grad_native, saved_native[0])

    if op_name == "layernorm":
        if len(saved_native) != 2:
            raise ValueError(
                f"layernorm backward expects 2 saved tensors (input, gamma), got {len(saved_tensors)}"
            )
        return bridge.dispatch_layernorm_backward(
            grad_native, saved_native[0], saved_native[1]
        )

    if op_name == "rmsnorm":
        if len(saved_native) != 2:
            raise ValueError(
                f"rmsnorm backward expects 2 saved tensors (input, gamma), got {len(saved_tensors)}"
            )
        return bridge.dispatch_rmsnorm_backward(
            grad_native, saved_native[0], saved_native[1]
        )

    raise ValueError(f"Unsupported op for native backward dispatch: '{op_name}'")


def dispatch_graph_backward_native_cached(
    ir_json: str,
    grad_output: Any,
    saved_activations: Any,
) -> Dict[int, Any]:
    import numpy as np

    grad_np = to_native_array(grad_output)
    rust_grads = _execute_rust_graph_backward(
        graph_json=ir_json,
        grad_np=grad_np,
        saved_activations=(
            {
                int(node_id): np.asarray(values, dtype=np.float32)
                for node_id, values in saved_activations.items()
            }
            if isinstance(saved_activations, Mapping)
            else saved_activations
        ),
    )
    if rust_grads is None:
        raise RuntimeError("Rust scheduler does not expose graph backward execution.")
    return rust_grads


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

    x_np, graph_json = _prepare_graph_input(graph, input_data)

    # Flatten input for the scheduler (expects Vec<f32>)
    try:
        profiling_enabled = _native_profiling_enabled(rust)
        if profiling_enabled and hasattr(rust, "execute_graph_with_stats_arrays"):
            result = rust.execute_graph_with_stats_arrays(graph_json, x_np)
            y_flat = result["output"]
            logger.debug(
                "Arena stats: %d/%d bytes used, %d arena allocs, %d heap fallbacks",
                result.get("arena_bytes_used", 0),
                result.get("arena_capacity", 0),
                result.get("arena_alloc_count", 0),
                result.get("heap_fallback_count", 0),
            )
            if "node_profiles" in result:
                _set_last_profile_data(
                    {
                        "node_profiles": list(result["node_profiles"]),
                        "peak_memory_bytes": int(result.get("peak_memory_bytes", 0)),
                    }
                )
            else:
                _set_last_profile_data(None)
        elif profiling_enabled and hasattr(rust, "execute_graph_with_stats"):
            x_flat = x_np.ravel().tolist()
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
                _set_last_profile_data(
                    {
                        "node_profiles": list(result["node_profiles"]),
                        "peak_memory_bytes": int(result.get("peak_memory_bytes", 0)),
                    }
                )
                logger.debug(
                    "Profiling: %d node events, peak memory %d bytes",
                    len(_last_profile_data["node_profiles"]),
                    _last_profile_data["peak_memory_bytes"],
                )
            else:
                _set_last_profile_data(None)
        elif hasattr(rust, "execute_graph_arrays"):
            y_flat = rust.execute_graph_arrays(graph_json, x_np)
            _set_last_profile_data(None)
        else:
            x_flat = x_np.ravel().tolist()
            y_flat = rust.execute_graph(graph_json, x_flat)
            _set_last_profile_data(None)
        return _reshape_graph_output(graph, x_np, y_flat)
    except Exception as exc:
        logger.error("Rust scheduler execution failed: %s", exc)
        raise


def dispatch_graph_forward_native_saved(
    graph: Any,
    input_data: Any,
    *,
    ir_json: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a full forward pass, returning output and saved activations.

    This is the companion to ``dispatch_graph_backward_native()``.  The saved
    activations dict is keyed by integer node id and contains flat float lists.

    Prefers the Rust scheduler's native forward-saved path when available.
    Falls back to the legacy per-op Python walk only when the Rust entrypoint
    is unavailable or fails.

    Args:
        graph: ComputationGraph instance.
        input_data: Input tensor (numpy array or torch tensor).

    Returns:
        Dict with keys:
          - ``"output"``: numpy array containing the output.
          - ``"saved_activations"``: dict[int, numpy.ndarray] per-node activations.
          - ``"ir_json"``: the pre-serialized IR JSON (for backward call).
    """

    x_np, graph_json = _prepare_graph_input(graph, input_data, ir_json=ir_json)
    try:
        rust_result = _execute_rust_graph_forward_saved(
            graph_json=graph_json,
            x_np=x_np,
            graph=graph,
        )
        if rust_result is not None:
            return rust_result
    except Exception as exc:
        logger.debug("Rust forward-saved dispatch failed: %s", exc)

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
        flat_inputs = [to_native_flat_array(a) for a in inputs]
        y = dispatch_op_native(node.op_name, *flat_inputs)
        node_outputs[nid] = to_native_flat_array(y)

    output = node_outputs.get(output_node_id)
    if output is None:
        raise RuntimeError(
            f"Graph output node {output_node_id} not found in forward pass"
        )

    return {
        "output": to_native_array(output),
        "saved_activations": dict(node_outputs),
        "ir_json": graph_json,
    }


def dispatch_graph_backward_native(
    graph: Any,
    grad_output: Any,
    saved_activations: Any,
    ir_json: Optional[str] = None,
) -> Dict[int, Any]:
    """Execute a full backward pass through a graph using native per-op backward.

    Prefers the Rust scheduler's native graph backward when available.
    Falls back to the legacy Python reverse walk only when the Rust entrypoint
    is unavailable or fails.

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

    grad_np = to_native_array(grad_output)

    graph_json = ir_json
    if graph_json is None:
        _, graph_json = _prepare_graph_input(graph, grad_np)

    try:
        rust_grads = _execute_rust_graph_backward(
            graph_json=graph_json,
            grad_np=grad_np,
            saved_activations=saved_activations,
        )
        if rust_grads is not None:
            return rust_grads
    except Exception as exc:
        logger.debug("Rust backward dispatch failed: %s", exc)

    topo = graph.topological_order()
    output_node_id = graph._output_node_id

    if not isinstance(saved_activations, Mapping):
        raise RuntimeError(
            "Opaque native saved activations require Rust handle-backed backward execution."
        )

    # Ensure saved_activations values are numpy arrays.
    saved: Dict[int, Any] = {}
    for k, v in saved_activations.items():
        saved[int(k)] = to_native_flat_array(v)

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
            g_in = to_native_flat_array(result)
            if input_ids[0] in node_grads:
                node_grads[input_ids[0]] = node_grads[input_ids[0]] + g_in
            else:
                node_grads[input_ids[0]] = g_in
        else:
            # Binary/multi-input: result is a tuple of gradients.
            if not isinstance(result, (tuple, list)):
                result = tuple([result] * len(input_ids))
            for i, iid in enumerate(input_ids):
                g_in = to_native_flat_array(result[i])
                if iid in node_grads:
                    node_grads[iid] = node_grads[iid] + g_in
                else:
                    node_grads[iid] = g_in

    grads: Dict[int, Any] = {}
    for nid, g in node_grads.items():
        grads[nid] = to_native_array(g)

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

    x_np = to_native_array(input_data)
    try:
        graph_handle = _compile_rust_graph_handle(ir_json)
        profiling_enabled = _native_profiling_enabled(rust)
        if (
            not profiling_enabled
            and graph_handle is not None
            and hasattr(rust, "execute_graph_compiled_arrays_handle")
        ):
            y_flat = rust.execute_graph_compiled_arrays_handle(graph_handle, x_np)
            _set_last_profile_data(None)
        elif graph_handle is not None and hasattr(
            rust, "execute_graph_with_stats_compiled_arrays"
        ):
            result = rust.execute_graph_with_stats_compiled_arrays(graph_handle, x_np)
            y_flat = result["output"]
            logger.debug(
                "Arena stats: %d/%d bytes used, %d arena allocs, %d heap fallbacks",
                result.get("arena_bytes_used", 0),
                result.get("arena_capacity", 0),
                result.get("arena_alloc_count", 0),
                result.get("heap_fallback_count", 0),
            )
            if "node_profiles" in result:
                _set_last_profile_data(
                    {
                        "node_profiles": list(result["node_profiles"]),
                        "peak_memory_bytes": int(result.get("peak_memory_bytes", 0)),
                    }
                )
            else:
                _set_last_profile_data(None)
        elif profiling_enabled and hasattr(rust, "execute_graph_with_stats_arrays"):
            result = rust.execute_graph_with_stats_arrays(ir_json, x_np)
            y_flat = result["output"]
            logger.debug(
                "Arena stats: %d/%d bytes used, %d arena allocs, %d heap fallbacks",
                result.get("arena_bytes_used", 0),
                result.get("arena_capacity", 0),
                result.get("arena_alloc_count", 0),
                result.get("heap_fallback_count", 0),
            )
            if "node_profiles" in result:
                _set_last_profile_data(
                    {
                        "node_profiles": list(result["node_profiles"]),
                        "peak_memory_bytes": int(result.get("peak_memory_bytes", 0)),
                    }
                )
            else:
                _set_last_profile_data(None)
        elif profiling_enabled and hasattr(rust, "execute_graph_with_stats"):
            x_flat = x_np.ravel().tolist()
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
                _set_last_profile_data(
                    {
                        "node_profiles": list(result["node_profiles"]),
                        "peak_memory_bytes": int(result.get("peak_memory_bytes", 0)),
                    }
                )
            else:
                _set_last_profile_data(None)
        elif hasattr(rust, "execute_graph_arrays"):
            y_flat = rust.execute_graph_arrays(ir_json, x_np)
            _set_last_profile_data(None)
        else:
            x_flat = x_np.ravel().tolist()
            y_flat = rust.execute_graph(ir_json, x_flat)
            _set_last_profile_data(None)
        return _reshape_graph_output(graph, x_np, y_flat)
    except Exception as exc:
        logger.error("Rust scheduler execution failed: %s", exc)
        raise


def dispatch_graph_native_multi_input_cached(
    ir_json: str,
    input_data: list[Any] | tuple[Any, ...],
    *,
    output_shape: tuple[int, ...] | None = None,
) -> Any:
    """Execute a graph using distinct input buffers bound to distinct input nodes."""
    if not supports_host_array_bridge(*input_data):
        raise RuntimeError(
            "Host array bridge does not support non-CPU tensors for multi-input native graph dispatch."
        )
    rust = _try_import_rust_scheduler()
    if rust is None:
        raise RuntimeError("Rust scheduler (aria_scheduler) is not available.")

    import numpy as np

    native_inputs = [to_native_array(value) for value in input_data]

    try:
        graph_handle = _compile_rust_graph_handle(ir_json)
        profiling_enabled = _native_profiling_enabled(rust)
        if (
            not profiling_enabled
            and graph_handle is not None
            and hasattr(rust, "execute_graph_multi_input_compiled_arrays_handle")
        ):
            y_flat = rust.execute_graph_multi_input_compiled_arrays_handle(
                graph_handle,
                native_inputs,
            )
            _set_last_profile_data(None)
        elif graph_handle is not None and hasattr(
            rust, "execute_graph_multi_input_compiled_arrays_with_stats"
        ):
            result = rust.execute_graph_multi_input_compiled_arrays_with_stats(
                graph_handle,
                native_inputs,
            )
            y_flat = result["output"]
            logger.debug(
                "Arena stats: %d/%d bytes used, %d arena allocs, %d heap fallbacks",
                result.get("arena_bytes_used", 0),
                result.get("arena_capacity", 0),
                result.get("arena_alloc_count", 0),
                result.get("heap_fallback_count", 0),
            )
            _set_last_profile_data(None)
        elif profiling_enabled and hasattr(
            rust, "execute_graph_multi_input_arrays_with_stats"
        ):
            result = rust.execute_graph_multi_input_arrays_with_stats(
                ir_json,
                native_inputs,
            )
            y_flat = result["output"]
            logger.debug(
                "Arena stats: %d/%d bytes used, %d arena allocs, %d heap fallbacks",
                result.get("arena_bytes_used", 0),
                result.get("arena_capacity", 0),
                result.get("arena_alloc_count", 0),
                result.get("heap_fallback_count", 0),
            )
            _set_last_profile_data(None)
        elif profiling_enabled and hasattr(
            rust, "execute_graph_multi_input_with_stats"
        ):
            flat_inputs = [value.ravel().tolist() for value in native_inputs]
            result = rust.execute_graph_multi_input_with_stats(ir_json, flat_inputs)
            y_flat = result["output"]
            logger.debug(
                "Arena stats: %d/%d bytes used, %d arena allocs, %d heap fallbacks",
                result.get("arena_bytes_used", 0),
                result.get("arena_capacity", 0),
                result.get("arena_alloc_count", 0),
                result.get("heap_fallback_count", 0),
            )
            _set_last_profile_data(None)
        elif hasattr(rust, "execute_graph_multi_input_arrays"):
            y_flat = rust.execute_graph_multi_input_arrays(ir_json, native_inputs)
            _set_last_profile_data(None)
        elif hasattr(rust, "execute_graph_multi_input"):
            flat_inputs = [value.ravel().tolist() for value in native_inputs]
            y_flat = rust.execute_graph_multi_input(ir_json, flat_inputs)
            _set_last_profile_data(None)
        else:
            raise RuntimeError(
                "Rust scheduler does not expose multi-input graph execution."
            )
        y_np = np.asarray(y_flat, dtype=np.float32)
        if output_shape is not None:
            y_np = y_np.reshape(output_shape)
        reference = input_data[0] if input_data else y_np
        return to_device_tensor(y_np, reference=reference)
    except Exception as exc:
        logger.error("Rust multi-input scheduler execution failed: %s", exc)
        raise


def dispatch_graph_forward_saved_multi_input_cached(
    ir_json: str,
    input_data: list[Any] | tuple[Any, ...],
    *,
    output_shape: tuple[int, ...] | None = None,
) -> Dict[str, Any]:
    if not supports_host_array_bridge(*input_data):
        raise RuntimeError(
            "Host array bridge does not support non-CPU tensors for saved multi-input native graph dispatch."
        )
    native_inputs = [to_native_array(value) for value in input_data]
    rust_result = _execute_rust_graph_forward_saved_multi_input(
        graph_json=ir_json,
        native_inputs=native_inputs,
        output_shape=output_shape,
    )
    if rust_result is None:
        raise RuntimeError(
            "Rust scheduler does not expose multi-input forward-saved execution."
        )
    reference = input_data[0] if input_data else rust_result["output"]
    rust_result["output"] = to_device_tensor(rust_result["output"], reference=reference)
    return rust_result


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
