"""Shared fallback handler templates for component kernel shims."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

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


def make_unary_handler(
    component_type: str,
    op: Callable[[Any], Any],
    input_port: str = "x",
):
    """Create a unary ComponentHandler with shared build/forward behavior."""

    class _UnaryModule(nn.Module):
        def forward(self, x):
            return op(x)

    class ComponentHandler:  # noqa: D401
        def validate_config(self, config: Dict[str, Any]) -> List[str]:
            return []

        def build(self, config: Dict[str, Any]) -> nn.Module:
            return _UnaryModule()

        def forward(self, inputs: Dict[str, Any], config: Dict[str, Any]):
            if input_port not in inputs:
                raise KeyError(
                    f"{component_type} unary fallback requires input port '{input_port}'"
                )
            return {"y": op(inputs[input_port])}

    return ComponentHandler


def make_binary_handler(
    component_type: str,
    op: Callable[[Any, Any], Any],
    left_port: str = "a",
    right_port: str = "b",
):
    """Create a binary ComponentHandler with shared build/forward behavior."""

    class _BinaryModule(nn.Module):
        def forward(self, a, b):
            return op(a, b)

    class ComponentHandler:  # noqa: D401
        def validate_config(self, config: Dict[str, Any]) -> List[str]:
            return []

        def build(self, config: Dict[str, Any]) -> nn.Module:
            return _BinaryModule()

        def forward(self, inputs: Dict[str, Any], config: Dict[str, Any]):
            if left_port not in inputs or right_port not in inputs:
                raise KeyError(
                    f"{component_type} binary fallback requires input ports "
                    f"'{left_port}' and '{right_port}'"
                )
            return {"y": op(inputs[left_port], inputs[right_port])}

    return ComponentHandler


def make_passthrough_handler(
    component_type: str,
    input_port: str = "x",
    output_port: str = "y",
):
    """Create a handler that forwards one port to one output."""

    class ComponentHandler:  # noqa: D401
        def validate_config(self, config: Dict[str, Any]) -> List[str]:
            return []

        def build(self, config: Dict[str, Any]) -> nn.Module:
            return nn.Identity()

        def forward(self, inputs: Dict[str, Any], config: Dict[str, Any]):
            if input_port not in inputs:
                raise KeyError(
                    f"{component_type} passthrough fallback requires input port "
                    f"'{input_port}'"
                )
            return {output_port: inputs[input_port]}

    return ComponentHandler


def make_sort_stub_handler(component_type: str, input_port: str = "x"):
    """Create a fallback for sort-like ops: pass x through and emit zero indices."""

    class ComponentHandler:  # noqa: D401
        def validate_config(self, config: Dict[str, Any]) -> List[str]:
            return []

        def build(self, config: Dict[str, Any]) -> nn.Module:
            return nn.Identity()

        def forward(self, inputs: Dict[str, Any], config: Dict[str, Any]):
            if input_port not in inputs:
                raise KeyError(
                    f"{component_type} sort-stub fallback requires input port "
                    f"'{input_port}'"
                )
            x = inputs[input_port]
            idx = torch.zeros(x.shape[0], x.shape[1], dtype=torch.long, device=x.device)
            return {"y": x, "idx": idx}

    return ComponentHandler


def safe_log(x):
    return torch.log(torch.clamp(x.abs(), min=1e-8))


def safe_sqrt(x):
    return torch.sqrt(torch.clamp(x.abs(), min=1e-8))


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
