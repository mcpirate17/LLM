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
            base = torch.arange(d_model, device=device, dtype=torch.float32).view(1, 1, -1)
            y = (indices.unsqueeze(-1).float() + base).remainder(97.0) / 97.0
            return {"y": y}

    return ComponentHandler
