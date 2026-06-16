"""Discover replaceable component sites inside LLM-style ``nn.Module`` trees.

This is the model-side counterpart to ``scope_existing``. ``scope_existing``
mines the property catalog; this module inspects a live architecture and finds
the submodules fab can transplant or replace with a novel mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ComponentSite:
    path: str
    class_name: str
    role: str
    replaceability: str
    param_count: int
    trainable_param_count: int
    child_count: int
    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None

    @property
    def is_replaceable(self) -> bool:
        return self.replaceability != "fixed_scaffold"


def _param_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    return int(
        sum(
            p.numel()
            for p in module.parameters(recurse=True)
            if not trainable_only or p.requires_grad
        )
    )


def _shape_of(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, torch.Tensor):
        return tuple(int(dim) for dim in value.shape)
    if isinstance(value, (tuple, list)):
        for item in value:
            shape = _shape_of(item)
            if shape is not None:
                return shape
    if isinstance(value, Mapping):
        for item in value.values():
            shape = _shape_of(item)
            if shape is not None:
                return shape
    return None


def _role_for(path: str, module: nn.Module) -> str:
    lowered = f"{path}.{type(module).__name__}".lower()
    if "embed" in lowered:
        return "embedding"
    if "lm_head" in lowered or lowered.endswith(".head") or ".head." in lowered:
        return "head"
    if isinstance(module, nn.Linear):
        return "projection"
    if "norm" in lowered:
        return "norm"
    if type(module).__name__.lower().endswith("block"):
        return "block"
    if any(token in lowered for token in ("mlp", "ffn", "feedforward", "swiglu")):
        return "ffn"
    if any(token in lowered for token in ("router", "moe", "expert", "route")):
        return "routing"
    if any(token in lowered for token in ("compress", "bottleneck", "latent")):
        return "compression"
    if any(token in lowered for token in ("memory", "ssm", "state", "scan", "delta")):
        return "state_mixer"
    if any(token in lowered for token in ("attention", "attn", "mixer", "lane")):
        return "token_mixer"
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        return "token_mixer"
    return "other"


def _replaceability_for(
    *,
    role: str,
    input_shape: tuple[int, ...] | None,
    output_shape: tuple[int, ...] | None,
) -> str:
    if role in {"embedding", "head", "norm"}:
        return "fixed_scaffold"
    if (
        input_shape is not None
        and input_shape == output_shape
        and len(input_shape) >= 3
    ):
        return "drop_in_sequence_module"
    if role in {"token_mixer", "state_mixer", "routing", "compression", "ffn"}:
        return "candidate_component"
    return "fixed_scaffold"


def _run_with_hooks(
    model: nn.Module, sample_input: Any
) -> dict[str, tuple[tuple[int, ...] | None, tuple[int, ...] | None]]:
    shapes: dict[str, tuple[tuple[int, ...] | None, tuple[int, ...] | None]] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(path: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            shapes[path] = (_shape_of(inputs), _shape_of(output))

        return hook

    for path, module in model.named_modules():
        if not path:
            continue
        hooks.append(module.register_forward_hook(make_hook(path)))
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            if isinstance(sample_input, Mapping):
                model(**sample_input)
            elif isinstance(sample_input, tuple):
                model(*sample_input)
            else:
                model(sample_input)
    finally:
        for handle in hooks:
            handle.remove()
        model.train(was_training)
    return shapes


def find_component_sites(
    model: nn.Module,
    *,
    sample_input: Any | None = None,
    include_fixed: bool = False,
) -> list[ComponentSite]:
    """Return component sites ordered by replacement priority.

    ``sample_input`` is optional. When supplied, forward hooks record real
    input/output shapes, allowing exact detection of drop-in ``[B, L, D]`` sites.
    Without it, role heuristics still identify likely mixers and routers.
    """

    shapes = _run_with_hooks(model, sample_input) if sample_input is not None else {}
    sites: list[ComponentSite] = []
    for path, module in model.named_modules():
        if not path:
            continue
        child_count = sum(1 for _ in module.children())
        input_shape, output_shape = shapes.get(path, (None, None))
        role = _role_for(path, module)
        replaceability = _replaceability_for(
            role=role,
            input_shape=input_shape,
            output_shape=output_shape,
        )
        if replaceability == "fixed_scaffold" and not include_fixed:
            continue
        sites.append(
            ComponentSite(
                path=path,
                class_name=type(module).__name__,
                role=role,
                replaceability=replaceability,
                param_count=_param_count(module),
                trainable_param_count=_param_count(module, trainable_only=True),
                child_count=child_count,
                input_shape=input_shape,
                output_shape=output_shape,
            )
        )
    priority = {
        "drop_in_sequence_module": 0,
        "candidate_component": 1,
        "fixed_scaffold": 2,
    }
    role_priority = {
        "token_mixer": 0,
        "state_mixer": 1,
        "routing": 2,
        "compression": 3,
        "ffn": 4,
    }
    return sorted(
        sites,
        key=lambda site: (
            priority.get(site.replaceability, 9),
            role_priority.get(site.role, 9),
            -site.trainable_param_count,
            site.path,
        ),
    )


def replaceable_component_paths(
    model: nn.Module,
    *,
    sample_input: Any | None = None,
) -> tuple[str, ...]:
    """Compact helper for callers that only need paths."""

    return tuple(
        site.path
        for site in find_component_sites(model, sample_input=sample_input)
        if site.is_replaceable
    )
