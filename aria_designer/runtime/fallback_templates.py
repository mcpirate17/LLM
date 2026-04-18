"""Shared fallback handler templates for component kernel shims."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

try:
    import aria_core

    _HAS_ARIA_CORE = True
except ImportError:
    _HAS_ARIA_CORE = False


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
    from aria_designer.components.base import NativeComponentHandler

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
    from aria_designer.components.base import NativeComponentHandler

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


def make_route_topk_handler():
    """Create a native-first top-k routing handler."""

    class ComponentHandler:  # noqa: D401
        def validate_config(self, config: Dict[str, Any]) -> List[str]:
            return []

        def build(self, config: Dict[str, Any]) -> None:
            return None

        def forward(self, inputs: Dict[str, Any], config: Dict[str, Any]):
            scores = inputs["scores"]
            if scores.dim() != 3:
                raise ValueError("route_topk expects scores with shape [B, S, K]")
            k = max(1, min(int(config.get("k", 1)), scores.size(-1)))
            if _HAS_ARIA_CORE:
                try:
                    indices, weights = aria_core.route_topk_indices_f32(
                        scores.detach().contiguous().float(), k
                    )
                    return {"indices": indices, "weights": weights}
                except Exception:
                    pass
            weights, indices = torch.topk(scores, k=k, dim=-1)
            return {"indices": indices, "weights": weights}

    return ComponentHandler


def make_route_argmax_handler(
    component_name: str,
    output_name: str,
    config_key: str,
    default_limit: int,
):
    """Create an argmax-style routing handler."""

    class ComponentHandler:  # noqa: D401
        def validate_config(self, config: Dict[str, Any]) -> List[str]:
            return []

        def build(self, config: Dict[str, Any]) -> None:
            return None

        def forward(self, inputs: Dict[str, Any], config: Dict[str, Any]):
            scores = inputs["scores"]
            if scores.dim() != 3:
                raise ValueError(
                    f"{component_name} expects scores with shape [B, S, D]"
                )
            limit = max(
                1, min(int(config.get(config_key, default_limit)), scores.size(-1))
            )
            if component_name == "route_recursion" and _HAS_ARIA_CORE:
                try:
                    depth = aria_core.route_recursion_depth_f32(
                        scores[..., :limit].detach().contiguous().float()
                    )
                    return {output_name: depth}
                except Exception:
                    pass
            return {output_name: torch.argmax(scores[..., :limit], dim=-1)}

    return ComponentHandler


def make_token_merge_handler():
    """Create a native-first token merge handler."""

    class ComponentHandler:  # noqa: D401
        def validate_config(self, config: Dict[str, Any]) -> List[str]:
            return []

        def build(self, config: Dict[str, Any]) -> None:
            return None

        def forward(self, inputs: Dict[str, Any], config: Dict[str, Any]):
            x = inputs["x"]
            if x.dim() != 3:
                raise ValueError("token_merge expects x with shape [B, S, D]")
            seq_len = x.shape[1]
            n_keep = max(1, min(int(config.get("n_keep", seq_len)), seq_len))
            if _HAS_ARIA_CORE:
                try:
                    y, restore_map = aria_core.token_merge_simple_f32(
                        x.detach().contiguous().float(), n_keep
                    )
                    return {"y": y, "restore_map": restore_map}
                except Exception:
                    pass
            batch_size = x.shape[0]
            restore_row = torch.arange(
                seq_len, device=x.device, dtype=torch.long
            ).clamp(max=n_keep - 1)
            restore_map = restore_row.unsqueeze(0).expand(batch_size, -1)
            return {"y": x[:, :n_keep, :], "restore_map": restore_map}

    return ComponentHandler


def make_basis_expansion_handler():
    """Create a native-first Fourier basis expansion handler."""

    class ComponentHandler:  # noqa: D401
        def validate_config(self, config: Dict[str, Any]) -> List[str]:
            return []

        def build(self, config: Dict[str, Any]) -> None:
            return None

        def forward(self, inputs: Dict[str, Any], config: Dict[str, Any]):
            x = inputs["x"]
            n_bases = max(1, int(config.get("n_bases", 4)))
            if _HAS_ARIA_CORE:
                try:
                    freqs = torch.arange(
                        1, n_bases + 1, device=x.device, dtype=torch.float32
                    )
                    return {
                        "y": aria_core.basis_expansion_f32(
                            x.detach().contiguous().float(), freqs, n_bases
                        )
                    }
                except Exception:
                    pass
            features = []
            for freq in range(1, n_bases + 1):
                scaled = x * float(freq)
                features.extend((torch.sin(scaled), torch.cos(scaled)))
            return {"y": torch.cat(features, dim=-1)}

    return ComponentHandler
