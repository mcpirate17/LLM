from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

from .dispatch import (
    dispatch_graph_backward_native,
    dispatch_graph_forward_native_saved,
    dispatch_graph_native,
    dispatch_graph_native_cached,
    dispatch_op_native,
)

logger = logging.getLogger(__name__)


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
                fwd = dispatch_graph_forward_native_saved(_graph, x)
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
        self._supported_ops = supported_ops
        self._all_native = self._check_all_native()
        self._dispatch_count = 0
        self._fallback_count = 0
        # Lazily-created autograd Function subclass (cached after first grad dispatch).
        self._autograd_fn: Any = None
        # Pre-convert graph to native_ir JSON once; reuse across dispatches.
        self._ir_json: Optional[str] = None
        if self._all_native:
            try:
                from ...synthesis.native_ir_converter import graph_to_native_ir_json

                self._ir_json = graph_to_native_ir_json(graph)
            except Exception:
                logger.debug("Failed to pre-convert graph to IR JSON")
                self._ir_json = None

    def _check_all_native(self) -> bool:
        """Return True if every non-input op in the graph is native-supported."""
        nodes = getattr(self._graph, "nodes", None)
        if not isinstance(nodes, dict) or not nodes:
            return False
        for node in nodes.values():
            if getattr(node, "is_input", False):
                continue
            op_name = getattr(node, "op_name", "")
            if op_name not in self._supported_ops:
                return False
        return True

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
                return result

            # ── Inference path: numpy-based, no autograd ──
            if self._ir_json is not None:
                result = dispatch_graph_native_cached(self._ir_json, self._graph, x)
            else:
                result = dispatch_graph_native(self._graph, x)
            self._dispatch_count += 1

            # Convert back to torch if input was torch
            if hasattr(x, "detach"):
                import torch
                import numpy as np

                return torch.from_numpy(np.asarray(result, dtype=np.float32))
            return result
        except Exception as exc:
            logger.debug("Subgraph dispatch failed: %s, falling back to per-op", exc)
            self._fallback_count += 1
            return None

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "all_native": self._all_native,
            "subgraph_dispatches": self._dispatch_count,
            "subgraph_fallbacks": self._fallback_count,
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

    def dispatch(self, op_name: str, *tensors: Any) -> Any:
        """Try native dispatch, fall back to numpy/torch if needed.

        When any input tensor requires gradients **and** the op has a native
        backward kernel, the call is routed through the corresponding
        ``torch.autograd.Function`` subclass (see ``native_autograd.py``)
        so that gradient computation flows through the C backward kernels.

        Returns the result tensor/array on success, or ``None`` to signal
        the caller should use the original PyTorch implementation.
        """
        if op_name in self.supported_ops:
            try:
                # Check if any torch input requires grad → use autograd path
                any_requires_grad = any(
                    getattr(t, "requires_grad", False) for t in tensors
                )
                if any_requires_grad:
                    from ..native_autograd import (
                        NATIVE_AUTOGRAD_SUPPORTED_OPS,
                        native_autograd_dispatch,
                    )

                    if op_name in NATIVE_AUTOGRAD_SUPPORTED_OPS:
                        result = native_autograd_dispatch(op_name, *tensors)
                        self._dispatch_count += 1
                        return result

                import numpy as np

                # Convert torch tensors to numpy for C dispatch
                np_inputs: List[Any] = []
                for t in tensors:
                    if hasattr(t, "detach"):
                        np_inputs.append(t.detach().cpu().numpy().astype(np.float32))
                    else:
                        np_inputs.append(np.asarray(t, dtype=np.float32))

                result = dispatch_op_native(op_name, *np_inputs)
                self._dispatch_count += 1

                # Convert back to torch if input was torch
                if tensors and hasattr(tensors[0], "detach"):
                    import torch

                    return torch.from_numpy(np.asarray(result, dtype=np.float32))
                return result
            except Exception as exc:
                logger.debug(
                    "Native dispatch failed for %s: %s, falling back", op_name, exc
                )
                self._fallback_count += 1
        return None  # Signal to caller: use original implementation

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "native_dispatches": self._dispatch_count,
            "fallbacks": self._fallback_count,
        }
