"""Shared fallback handler templates for component kernel shims."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn


def make_identity_handler(component_type: str):
    """Create a ComponentHandler class that returns input tensor as-is.

    This is intended for UI preview and non-native fallback paths where the
    component is semantically pass-through in the current designer runtime.
    """

    class ComponentHandler:  # noqa: D401
        def validate_config(self, config: Dict[str, Any]) -> List[str]:
            return []

        def build(self, config: Dict[str, Any]) -> nn.Module:
            return nn.Identity()

        def forward(self, inputs: Dict[str, Any], config: Dict[str, Any]):
            if "x" not in inputs:
                raise KeyError(
                    f"{component_type} identity fallback requires input port 'x'"
                )
            return {"y": inputs["x"]}

    return ComponentHandler


def make_native_temperature_handler(component_type: str, native_op_name: str):
    """Create NativeComponentHandler subclass for (x, temperature) kernels."""
    from components.base import NativeComponentHandler

    class ComponentHandler(NativeComponentHandler):  # noqa: D401
        def _get_native_args(self, inputs, config):
            x = inputs["x"].detach().contiguous().float()
            temperature = config.get("temperature", 0.1)
            return (x, temperature)

        def _fallback(self, inputs, config):
            return {"y": inputs["x"]}

    ComponentHandler.native_op_name = native_op_name
    return ComponentHandler


class _StubModule(nn.Module):
    """Bare module stub for mathspace execute functions.

    Most execute functions guard attribute access with ``hasattr`` and
    fall back to identity when weight/bias are absent.  For the few
    that unconditionally require specific attributes (hyp_linear,
    rotor_transform, hyperbolic_norm, grouped_linear), the component
    handler provides a custom stub instead.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def make_mathspace_unary_handler(
    native_op_name: str,
    execute_fn_path: str,
    *,
    native_args_fn=None,
):
    """Create a NativeComponentHandler that delegates its fallback to a
    research.mathspaces execute function (unary: ``execute_fn(module, x)``
    or variadic: ``execute_fn(module, *inputs)``).

    ``execute_fn_path`` is a dotted import path like
    ``"research.mathspaces.clifford.execute_clifford_attention"``.

    ``native_args_fn`` optionally customises the args tuple sent to
    aria_core; by default it sends ``(x_detached_contiguous_f32,)``.
    """
    from components.base import NativeComponentHandler

    _execute_fn = None

    def _resolve():
        nonlocal _execute_fn
        if _execute_fn is None:
            mod_path, fn_name = execute_fn_path.rsplit(".", 1)
            import importlib

            _execute_fn = getattr(importlib.import_module(mod_path), fn_name)
        return _execute_fn

    class ComponentHandler(NativeComponentHandler):
        def _get_native_args(self, inputs, config):
            if native_args_fn is not None:
                return native_args_fn(inputs, config)
            x = inputs["x"].detach().contiguous().float()
            return (x,)

        def _fallback(self, inputs, config):
            fn = _resolve()
            x = inputs["x"]
            return {"y": fn(_StubModule(), x)}

    ComponentHandler.native_op_name = native_op_name
    return ComponentHandler


def make_mathspace_binary_handler(
    native_op_name: str,
    execute_fn_path: str,
    *,
    native_args_fn=None,
):
    """Like ``make_mathspace_unary_handler`` but for binary ops
    (``execute_fn(module, x, y)``)."""
    from components.base import NativeComponentHandler

    _execute_fn = None

    def _resolve():
        nonlocal _execute_fn
        if _execute_fn is None:
            mod_path, fn_name = execute_fn_path.rsplit(".", 1)
            import importlib

            _execute_fn = getattr(importlib.import_module(mod_path), fn_name)
        return _execute_fn

    class ComponentHandler(NativeComponentHandler):
        def _get_native_args(self, inputs, config):
            if native_args_fn is not None:
                return native_args_fn(inputs, config)
            x = inputs.get("x", inputs.get("a")).detach().contiguous().float()
            y = inputs.get("y", inputs.get("b", x))
            if y is not x:
                y = y.detach().contiguous().float()
            return (x, y)

        def _fallback(self, inputs, config):
            fn = _resolve()
            x = inputs.get("x", inputs.get("a"))
            y = inputs.get("y", inputs.get("b", x))
            return {"y": fn(_StubModule(), x, y)}

    ComponentHandler.native_op_name = native_op_name
    return ComponentHandler


def make_embedding_lookup_handler(component_type: str):
    """Create a minimal embedding lookup fallback for preview and tests."""

    class ComponentHandler:  # noqa: D401
        def validate_config(self, config: Dict[str, Any]) -> List[str]:
            return []

        def build(self, config: Dict[str, Any]) -> None:
            return None

        def forward(self, inputs: Dict[str, Any], config: Dict[str, Any]):
            if "indices" not in inputs:
                raise KeyError(
                    f"{component_type} embedding fallback requires input port 'indices'"
                )
            indices = inputs["indices"]
            if not torch.is_tensor(indices):
                indices = torch.as_tensor(indices)
            if indices.dim() == 3:
                indices = indices[..., 0]
            indices = indices.to(dtype=torch.long)
            d_model = 256
            device = indices.device
            base = torch.arange(d_model, device=device, dtype=torch.float32).view(
                1, 1, -1
            )
            y = (indices.unsqueeze(-1).float() + base).remainder(97.0) / 97.0
            return {"y": y}

    return ComponentHandler
