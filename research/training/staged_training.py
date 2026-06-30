"""Apply staged-training genotypes to PyTorch modules.

This is intentionally a small, loop-agnostic helper. A runner can call
``apply_train_stage`` before constructing an optimizer for that stage, then use
the returned trainable parameters. No optimizer state is mutated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from torch import nn

from research.synthesis.training_regime_grammar import TrainStageSpec


@dataclass(frozen=True, slots=True)
class TrainabilityReport:
    target: str
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]
    trainable_param_count: int
    total_param_count: int


def _is_embedding_name(name: str) -> bool:
    lowered = name.lower()
    return any(
        marker in lowered
        for marker in (
            "embed",
            "embedding",
            "token_emb",
            "pos_emb",
            "wte",
            "wpe",
        )
    )


def _is_lm_head_name(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith("lm_head") or ".lm_head" in lowered


def _name_matches_target(name: str, target: str) -> bool:
    lowered = name.lower()
    if target == "all":
        return True
    if target == "embeddings":
        return _is_embedding_name(lowered)
    if target == "lm_head":
        return _is_lm_head_name(lowered)
    if target == "embedding_lm_head":
        return _is_embedding_name(lowered) or _is_lm_head_name(lowered)
    if target == "non_embedding":
        return not _is_embedding_name(lowered)
    if target == "mixer":
        return any(marker in lowered for marker in ("mixer", "lane", "attn"))
    if target == "ffn":
        return any(
            marker in lowered
            for marker in ("ffn", "mlp", "feed_forward", "swiglu", "fc1", "fc2")
        )
    if target == "router":
        return any(marker in lowered for marker in ("router", "route", "gate", "halt"))
    if target == "norms":
        return "norm" in lowered or ".ln" in lowered
    if target == "carrier":
        return any(marker in lowered for marker in ("lane_partner", "carrier", "partner"))
    if target == "loss_lane":
        return any(
            marker in lowered
            for marker in ("lane_loss", "loss_lane", "loss_specialist", "monster")
        )
    raise ValueError(f"unknown train target {target!r}")


def matched_parameter_names(module: nn.Module, target: str) -> tuple[str, ...]:
    """Parameter names selected by a stage target."""

    return tuple(
        name for name, _param in module.named_parameters() if _name_matches_target(name, target)
    )


def _named_parameters(module: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return list(module.named_parameters())


def trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [param for param in module.parameters() if param.requires_grad]


def apply_train_stage(
    module: nn.Module,
    stage: TrainStageSpec,
    *,
    allow_empty: bool = False,
) -> TrainabilityReport:
    """Apply one stage's freeze/unfreeze policy to ``module``.

    ``freeze_others=True`` makes the target exclusive. ``False`` is additive:
    existing trainable parameters stay trainable and the target is unfrozen.
    """

    named = _named_parameters(module)
    matched = {name for name, _param in named if _name_matches_target(name, stage.target)}
    if not matched and not allow_empty:
        raise ValueError(f"stage target {stage.target!r} matched no parameters")

    if stage.freeze_others:
        for name, param in named:
            param.requires_grad_(name in matched)
    else:
        for name, param in named:
            if name in matched:
                param.requires_grad_(True)

    trainable_names = tuple(name for name, param in named if param.requires_grad)
    frozen_names = tuple(name for name, param in named if not param.requires_grad)
    trainable_count = sum(param.numel() for _name, param in named if param.requires_grad)
    total_count = sum(param.numel() for _name, param in named)
    if trainable_count <= 0 and not allow_empty:
        raise ValueError(f"stage target {stage.target!r} left no trainable parameters")
    return TrainabilityReport(
        target=stage.target,
        trainable_names=trainable_names,
        frozen_names=frozen_names,
        trainable_param_count=trainable_count,
        total_param_count=total_count,
    )


def apply_stage_sequence(
    module: nn.Module,
    stages: Iterable[TrainStageSpec],
    *,
    allow_empty: bool = False,
) -> tuple[TrainabilityReport, ...]:
    """Apply stages in order, returning each intermediate trainability report."""

    return tuple(
        apply_train_stage(module, stage, allow_empty=allow_empty) for stage in stages
    )
