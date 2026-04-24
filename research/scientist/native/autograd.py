from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional, Set

import torch

from .capability import classify_graph_native_capability
from ...synthesis.compiler_op_utils import _get_stacked_params
from .dispatch import (
    _PER_OP_BRIDGE_ONLY_OPS,
    dispatch_graph_backward_native,
    dispatch_graph_forward_native_saved,
    dispatch_graph_native,
    dispatch_graph_native_cached,
    dispatch_op_native,
)
from .single_op_bound import dispatch_single_op_bound_native
from .tensor_bridge import supports_host_array_bridge, to_device_tensor

logger = logging.getLogger(__name__)
_NATIVE_AUTOGRAD_MODULE = None


def _get_native_autograd_dispatch():
    """Resolve native autograd lazily once to avoid a circular import at module load."""
    global _NATIVE_AUTOGRAD_MODULE
    if _NATIVE_AUTOGRAD_MODULE is None:
        from .. import native_autograd

        _NATIVE_AUTOGRAD_MODULE = native_autograd
    return (
        _NATIVE_AUTOGRAD_MODULE.NATIVE_AUTOGRAD_SUPPORTED_OPS,
        _NATIVE_AUTOGRAD_MODULE.native_autograd_dispatch,
    )


def _dispatch_op_native_compat(op_name: str, *tensors: Any, **kwargs: Any) -> Any:
    legacy_facade = sys.modules.get("research.scientist.native_runner")
    facade_dispatch = getattr(legacy_facade, "dispatch_op_native", None)
    if callable(facade_dispatch) and facade_dispatch is not dispatch_op_native:
        return facade_dispatch(op_name, *tensors, **kwargs)
    return dispatch_op_native(op_name, *tensors, **kwargs)


class NativeSubgraphFunction:
    """torch.autograd.Function for full-graph native forward + backward.

    Instead of N per-op Python-to-C roundtrips in both forward and backward,
    this does 1 Rust call for the entire forward pass and 1 Rust call for
    the entire backward pass.  Activations are saved by the Rust forward
    (``execute_graph_forward_saved``) and fed back into the Rust backward
    (``execute_graph_backward``).

    This is a factory: call ``NativeSubgraphFunction.make(graph, ir_json)``
    to obtain a concrete ``torch.autograd.Function`` subclass bound to a
    specific graph.
    """

    @staticmethod
    def make(graph: Any, ir_json: Optional[str] = None):
        """Return a ``torch.autograd.Function`` subclass for *graph*.

        The returned class captures *graph* and *ir_json* in its closure so
        that ``apply(x)`` is the only user-facing call.
        """
        import torch

        _graph = graph
        _ir_json = ir_json

        class _SubgraphFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # Perform a full native forward, saving activations.
                fwd = dispatch_graph_forward_native_saved(_graph, x, ir_json=_ir_json)
                output_np = fwd["output"]
                saved_activations = fwd["saved_activations"]
                used_ir_json = fwd["ir_json"]

                import numpy as np

                output_tensor = torch.from_numpy(
                    np.asarray(output_np, dtype=np.float32)
                ).to(x.device)

                # Reshape to match input batch/seq dims when possible.
                if x.ndim >= 3 and output_tensor.ndim == 1:
                    try:
                        output_tensor = output_tensor.reshape(x.shape)
                    except RuntimeError:
                        pass

                # Save non-tensor data on ctx for backward.
                ctx._saved_activations = saved_activations
                ctx._ir_json = used_ir_json
                ctx._input_shape = x.shape
                ctx._graph = _graph

                return output_tensor

            @staticmethod
            def backward(ctx, grad_output):
                saved_activations = ctx._saved_activations
                used_ir_json = ctx._ir_json
                graph_ref = ctx._graph
                input_shape = ctx._input_shape

                # Single Rust call for the entire backward pass.
                grads = dispatch_graph_backward_native(
                    graph_ref,
                    grad_output,
                    saved_activations,
                    ir_json=used_ir_json,
                )

                # The input node gradient is what we need.  Find the input
                # node id (the node with is_input=True).
                import numpy as np

                nodes = getattr(graph_ref, "nodes", {})
                input_node_id = None
                for nid, node in nodes.items():
                    if getattr(node, "is_input", False):
                        input_node_id = nid
                        break

                if input_node_id is not None and input_node_id in grads:
                    grad_np = grads[input_node_id]
                    grad_tensor = torch.from_numpy(
                        np.asarray(grad_np, dtype=np.float32)
                    ).to(grad_output.device)
                    try:
                        grad_tensor = grad_tensor.reshape(input_shape)
                    except RuntimeError:
                        pass
                    return grad_tensor

                # Fallback: return grad_output as-is (identity gradient).
                return grad_output

        return _SubgraphFn


class SubgraphDispatcher:
    """Dispatches entire computation subgraphs through the Rust scheduler.

    When all ops in a ComputationGraph (or contiguous subgraph) are
    native-supported, this class converts the graph to native_ir.v1 and
    executes it as a single ``dispatch_graph_native()`` call — avoiding
    per-op Python-to-C roundtrips.

    If any op in the graph is unsupported, ``try_dispatch()`` returns
    ``None`` and the caller falls back to per-op dispatch.

    When the input tensor requires gradients, ``try_dispatch()`` routes
    through ``NativeSubgraphFunction`` so that both forward and backward
    are executed as single Rust calls (1 forward + 1 backward instead of
    N per-op roundtrips each way).
    """

    def __init__(self, graph: Any, supported_ops: Set[str]):
        self._graph = graph
        self._capability = classify_graph_native_capability(graph, supported_ops)
        self._supported_ops = self._capability.scheduler_supported_ops
        self._all_native = self._capability.all_native
        self._dispatch_count = 0
        self._fallback_count = 0
        self._last_refusal_reason: str | None = self._capability.refusal_reason
        # Lazily-created autograd Function subclass (cached after first grad dispatch).
        self._autograd_fn: Any = None
        # Pre-convert graph to native_ir JSON once; reuse across dispatches.
        self._ir_json: Optional[str] = None
        try:
            from ...synthesis.native_support import graph_has_bound_params

            if graph_has_bound_params(graph):
                self._all_native = False
                self._last_refusal_reason = (
                    "bound_param_graph_requires_bound_dispatcher"
                )
        except Exception:
            pass
        if self._all_native:
            try:
                from ...synthesis.native_ir_converter import graph_to_native_ir_json

                self._ir_json = graph_to_native_ir_json(graph)
            except Exception as exc:
                logger.debug("Failed to pre-convert graph to IR JSON: %s", exc)
                self._ir_json = None

    @property
    def all_native(self) -> bool:
        return self._all_native

    def try_dispatch(self, x: Any) -> Any:
        """Try to execute the full graph natively.

        Uses the pre-cached native_ir JSON to avoid re-serializing the graph
        on every call. Falls back to ``dispatch_graph_native()`` if the
        cached IR is not available.

        When *x* is a ``torch.Tensor`` that requires gradients, the call is
        routed through ``NativeSubgraphFunction`` so that both forward and
        backward are single Rust calls (instead of N per-op roundtrips).

        Args:
            x: Input tensor (torch.Tensor or numpy array).

        Returns:
            Output tensor on success, or ``None`` if subgraph dispatch
            is not possible (caller should fall back to per-op path).
        """
        if not self._all_native:
            self._last_refusal_reason = self._capability.refusal_reason
            return None
        if not supports_host_array_bridge(x):
            self._last_refusal_reason = "host_array_bridge_unsupported_device"
            return None

        try:
            # ── Autograd path: input requires grad → use NativeSubgraphFunction
            if getattr(x, "requires_grad", False):
                if self._autograd_fn is None:
                    self._autograd_fn = NativeSubgraphFunction.make(
                        self._graph, self._ir_json
                    )
                result = self._autograd_fn.apply(x)
                self._dispatch_count += 1
                self._last_refusal_reason = None
                return result

            # ── Inference path: numpy-based, no autograd ──
            if self._ir_json is not None:
                result = dispatch_graph_native_cached(self._ir_json, self._graph, x)
            else:
                result = dispatch_graph_native(self._graph, x)
            self._dispatch_count += 1
            self._last_refusal_reason = None

            # Convert back to torch if input was torch
            if hasattr(x, "detach"):
                return to_device_tensor(result, reference=x)
            return result
        except Exception as exc:
            logger.debug("Subgraph dispatch failed: %s, falling back to per-op", exc)
            exc_text = str(exc).lower()
            if (
                "not registered in native runtime" in exc_text
                or "unsupported op" in exc_text
            ):
                self._all_native = False
                self._last_refusal_reason = "runtime_op_unavailable"
            else:
                self._last_refusal_reason = "subgraph_dispatch_error"
            self._fallback_count += 1
            return None

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "all_native": self._all_native,
            "subgraph_dispatches": self._dispatch_count,
            "subgraph_fallbacks": self._fallback_count,
            "last_refusal_reason": self._last_refusal_reason,
            "scheduler_unsupported_ops": sorted(
                self._capability.scheduler_unsupported_ops
            ),
        }


class NativeForwardWrapper:
    """Wraps a compiled model to route supported ops through native C kernels.

    When enabled, intercepts tensor operations during forward pass and
    dispatches them through the Cython bridge instead of PyTorch.
    """

    def __init__(self, model: Any, supported_ops: Set[str]):
        self.model = model
        self.supported_ops = supported_ops
        self._dispatch_count = 0
        self._fallback_count = 0
        self._last_fallback_reason: str | None = None

    def _module_dispatch_args(
        self,
        op_name: str,
        module: Any,
        tensors: tuple[Any, ...],
    ) -> tuple[tuple[Any, ...], Dict[str, Any]]:
        if op_name == "gated_linear":
            if module is None or len(tensors) != 1:
                raise ValueError("gated_linear native dispatch requires module and x")
            return (
                tensors[0],
                module.linear_weight,
                module.gate_weight,
            ), {
                "bias": getattr(module, "linear_bias", None),
                "bias_gate": getattr(module, "gate_bias", None),
            }

        if op_name in {"linear_proj", "linear_proj_down", "linear_proj_up"}:
            if module is None or len(tensors) != 1:
                raise ValueError(f"{op_name} native dispatch requires module and x")
            return (
                tensors[0],
                module.weight,
            ), {}

        if op_name == "rmsnorm":
            if module is None or len(tensors) != 1:
                raise ValueError("rmsnorm native dispatch requires module and x")
            return (
                tensors[0],
                module.weight,
            ), {"eps": 1e-6}

        if op_name == "layernorm":
            if module is None or len(tensors) != 1:
                raise ValueError("layernorm native dispatch requires module and x")
            return (
                tensors[0],
                module.weight,
                module.bias,
            ), {"eps": 1e-5}

        if op_name == "rwkv_time_mixing":
            if module is None or len(tensors) != 1:
                raise ValueError(
                    "rwkv_time_mixing native dispatch requires module and x"
                )
            return (
                tensors[0],
                module.w_decay,
                module.u_bonus,
                module.W_k,
                module.W_v,
                module.W_r,
            ), {}

        if op_name == "conv1d_seq":
            if module is None or len(tensors) != 1:
                raise ValueError("conv1d_seq native dispatch requires module and x")
            conv_bias = getattr(module, "conv_bias", None)
            if conv_bias is None:
                conv_bias = module.conv_weight.new_zeros(module.conv_weight.shape[0])
            return (
                tensors[0],
                module.conv_weight,
                conv_bias,
            ), {}

        if op_name == "rwkv_channel":
            if module is None or len(tensors) != 1:
                raise ValueError("rwkv_channel native dispatch requires module and x")
            return (
                tensors[0],
                module.mix_k,
                module.mix_r,
                module.key_proj.weight,
                module.receptance_proj.weight,
                module.value_proj.weight,
            ), {}

        if op_name in {
            "depth_weighted_proj",
            "adaptive_recursion",
            "gated_lane_blend",
            "route_lanes",
            "depth_gated_transform",
            "route_recursion",
        }:
            if module is None or len(tensors) != 1:
                raise ValueError(
                    "depth_weighted_proj native dispatch requires module and x"
                )
            if op_name in {"gated_lane_blend", "route_lanes"}:
                scorer = module.lane_scorer
                stack_name = "lane_projs"
            elif op_name in {"depth_gated_transform", "route_recursion"}:
                scorer = module.depth_scorer
                stack_name = "depth_projs"
            else:
                scorer = module.depth_scorer
                stack_name = "step_projs"
            max_depth = int(scorer.shape[0])
            return (
                tensors[0],
                scorer,
                _get_stacked_params(module, stack_name, max_depth, tensors[0].dtype),
            ), {}

        if op_name == "swiglu_mlp":
            if module is None or len(tensors) != 1:
                raise ValueError("swiglu_mlp native dispatch requires module and x")
            return (
                tensors[0],
                module.gate_proj.weight,
                module.up_proj.weight,
                module.down_proj.weight,
                getattr(module.gate_proj, "bias", None),
                getattr(module.up_proj, "bias", None),
                getattr(module.down_proj, "bias", None),
            ), {}

        if op_name == "softmax_attention":
            if module is None or len(tensors) != 1:
                raise ValueError(
                    "softmax_attention native dispatch requires module and x"
                )
            return (
                tensors[0],
                module.q_proj.weight,
                module.k_proj.weight,
                module.v_proj.weight,
                module.o_proj.weight,
            ), {"n_heads": int(module.n_heads)}

        if op_name == "linear_attention":
            if module is None or len(tensors) != 1:
                raise ValueError(
                    "linear_attention native dispatch requires module and x"
                )
            return (
                tensors[0],
                module.q_proj.weight,
                module.k_proj.weight,
                module.v_proj.weight,
                module.o_proj.weight,
            ), {}

        if op_name == "selective_scan":
            if module is None or len(tensors) != 1:
                raise ValueError("selective_scan native dispatch requires module and x")
            return (
                tensors[0],
                module.A_log,
                module.dt_proj,
                module.B_proj.weight,
                module.C_proj.weight,
            ), {}

        if op_name == "state_space":
            if module is None or len(tensors) != 1:
                raise ValueError("state_space native dispatch requires module and x")
            return (
                tensors[0],
                module.ssm_A,
                module.ssm_B.weight,
                module.ssm_C.weight,
                module.ssm_D,
                module.ssm_dt.weight,
                module.ssm_dt.bias,
            ), {}

        if op_name == "gated_delta":
            if module is None or len(tensors) != 1:
                raise ValueError("gated_delta native dispatch requires module and x")
            dim = int(getattr(module, "model_dim", module.q_proj.weight.shape[0]))
            n_heads = int(getattr(module, "_gated_delta_heads", min(8, dim)))
            return (
                tensors[0],
                module.q_proj.weight,
                module.k_proj.weight,
                module.v_proj.weight,
                module.alpha_proj.weight,
                module.beta_proj.weight,
                module.o_proj.weight,
            ), {"n_heads": n_heads}

        return tensors, {}

    def dispatch(
        self, op_name: str, *tensors: Any, module: Any = None, **kwargs: Any
    ) -> Any:
        """Try native dispatch, fall back to numpy/torch if needed.

        When any input tensor requires gradients **and** the op has a native
        backward kernel, the call is routed through the corresponding
        ``torch.autograd.Function`` subclass (see ``native_autograd.py``)
        so that gradient computation flows through the C backward kernels.

        Returns the result tensor/array on success, or ``None`` to signal
        the caller should use the original PyTorch implementation.
        """
        if op_name in self.supported_ops or op_name in _PER_OP_BRIDGE_ONLY_OPS:
            try:
                dispatch_tensors, dispatch_kwargs = self._module_dispatch_args(
                    op_name, module, tensors
                )
                if not supports_host_array_bridge(
                    *dispatch_tensors, *dispatch_kwargs.values()
                ):
                    self._last_fallback_reason = "host_array_bridge_unsupported_device"
                    return None
                native_op_name = (
                    "linear"
                    if op_name in {"linear_proj", "linear_proj_down", "linear_proj_up"}
                    else op_name
                )
                if kwargs:
                    dispatch_kwargs = {**dispatch_kwargs, **kwargs}
                # Check if any torch input requires grad → use autograd path
                any_requires_grad = torch.is_grad_enabled() and any(
                    getattr(t, "requires_grad", False) for t in dispatch_tensors
                )
                if any_requires_grad:
                    if module is not None:
                        result = dispatch_single_op_bound_native(
                            op_name,
                            module,
                            tensors[0],
                            supported_ops=self.supported_ops,
                        )
                        if result is not None:
                            self._dispatch_count += 1
                            self._last_fallback_reason = None
                            return result
                    supported_ops, native_dispatch = _get_native_autograd_dispatch()
                    if native_op_name in supported_ops:
                        result = native_dispatch(native_op_name, *dispatch_tensors)
                        self._dispatch_count += 1
                        self._last_fallback_reason = None
                        return result
                    self._fallback_count += 1
                    self._last_fallback_reason = "native_backward_unavailable"
                    return None

                result = _dispatch_op_native_compat(
                    native_op_name, *dispatch_tensors, **dispatch_kwargs
                )
                self._dispatch_count += 1
                self._last_fallback_reason = None

                # Convert back to torch if input was torch
                if dispatch_tensors and hasattr(dispatch_tensors[0], "detach"):
                    return to_device_tensor(result, reference=dispatch_tensors[0])
                return result
            except Exception as exc:
                logger.debug(
                    "Native dispatch failed for %s: %s, falling back", op_name, exc
                )
                self._fallback_count += 1
                self._last_fallback_reason = "per_op_dispatch_error"
        else:
            self._last_fallback_reason = "op_not_native_supported"
        return None  # Signal to caller: use original implementation

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "native_dispatches": self._dispatch_count,
            "fallbacks": self._fallback_count,
            "last_fallback_reason": self._last_fallback_reason,
        }
