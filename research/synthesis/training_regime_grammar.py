"""Training-regime search grammar for staged and partial training.

Architecture search already mutates the model; Workstream E makes *how the
model is trained* a logged genotype too. The first increment is deliberately
CPU-only and loop-agnostic: it defines serializable staged-training axes that a
runner can consume without changing corpus loading, optimizer internals, or GPU
jobs.

Loss monsters stay scaffolds. Regimes here are graded on downstream capability
at a matched token/step budget: AR, induction, binding, MQAR, and BLiMP deltas
over the same architecture trained normally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

TRAIN_TARGETS: tuple[str, ...] = (
    "all",
    "embeddings",
    "lm_head",
    "embedding_lm_head",
    "non_embedding",
    "mixer",
    "ffn",
    "router",
    "norms",
    "carrier",
    "loss_lane",
)
FREEZE_POLICIES: tuple[str, ...] = ("freeze_others", "additive_unfreeze")
OPTIMIZERS: tuple[str, ...] = ("adamw", "muon", "sgd")
SCHEDULERS: tuple[str, ...] = ("cosine", "constant")

AXIS_TRAIN_REGIME = "op_train_regime"
AXIS_TRAIN_STAGES = "op_train_stages"
AXIS_FREEZE_SCHEDULE = "op_freeze_schedule"
AXIS_OPTIMIZER = "op_train_optimizer"
AXIS_BASE_LR = "op_train_base_lr"
AXIS_WEIGHT_DECAY = "op_train_weight_decay"
AXIS_SCHEDULER = "op_train_scheduler"
AXIS_WARMUP_FRACTION = "op_train_warmup_fraction"
AXIS_MAX_GRAD_NORM = "op_train_max_grad_norm"


@dataclass(frozen=True, slots=True)
class TrainStageSpec:
    """One stage of a staged-training genotype."""

    target: str
    steps: int
    lr_scale: float = 1.0
    freeze_others: bool = True
    reset_optimizer: bool = True

    def __post_init__(self) -> None:
        if self.target not in TRAIN_TARGETS:
            raise ValueError(f"unknown train target {self.target!r}; valid={TRAIN_TARGETS}")
        if int(self.steps) <= 0:
            raise ValueError(f"stage steps must be positive, got {self.steps}")
        if float(self.lr_scale) <= 0.0:
            raise ValueError(f"lr_scale must be positive, got {self.lr_scale}")

    @property
    def freeze_policy(self) -> str:
        return "freeze_others" if self.freeze_others else "additive_unfreeze"

    @property
    def key(self) -> str:
        mode = "freeze" if self.freeze_others else "add"
        reset = "reset" if self.reset_optimizer else "keep"
        return f"{self.target}:{int(self.steps)}:{mode}:{float(self.lr_scale):g}:{reset}"


@dataclass(frozen=True, slots=True)
class TrainingRegimeSpec:
    """Full training-regime genotype carried alongside architecture axes."""

    name: str
    stages: tuple[TrainStageSpec, ...]
    optimizer: str = "adamw"
    base_lr: float = 3e-4
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    warmup_fraction: float = 0.05
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("training regime name must be non-empty")
        if not self.stages:
            raise ValueError("training regime needs at least one stage")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"unknown optimizer {self.optimizer!r}; valid={OPTIMIZERS}")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"unknown scheduler {self.scheduler!r}; valid={SCHEDULERS}")
        if float(self.base_lr) <= 0.0:
            raise ValueError(f"base_lr must be positive, got {self.base_lr}")
        if float(self.weight_decay) < 0.0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )
        if not 0.0 <= float(self.warmup_fraction) < 1.0:
            raise ValueError(
                f"warmup_fraction must be in [0, 1), got {self.warmup_fraction}"
            )
        if float(self.max_grad_norm) <= 0.0:
            raise ValueError(
                f"max_grad_norm must be positive, got {self.max_grad_norm}"
            )

    @property
    def total_steps(self) -> int:
        return sum(int(stage.steps) for stage in self.stages)

    @property
    def key(self) -> str:
        return f"{self.name}|{serialize_train_stages(self.stages)}"


def serialize_train_stage(stage: TrainStageSpec) -> str:
    return stage.key


def parse_train_stage(raw: str) -> TrainStageSpec:
    parts = [part.strip() for part in str(raw).split(":")]
    if len(parts) != 5:
        raise ValueError(
            "train stage must be target:steps:freeze|add:lr_scale:reset|keep, "
            f"got {raw!r}"
        )
    target, steps, mode, lr_scale, reset = parts
    if mode not in ("freeze", "add"):
        raise ValueError(f"unknown freeze mode {mode!r}; expected freeze or add")
    if reset not in ("reset", "keep"):
        raise ValueError(f"unknown optimizer reset flag {reset!r}; expected reset or keep")
    return TrainStageSpec(
        target=target,
        steps=int(steps),
        freeze_others=(mode == "freeze"),
        lr_scale=float(lr_scale),
        reset_optimizer=(reset == "reset"),
    )


def serialize_train_stages(stages: tuple[TrainStageSpec, ...]) -> str:
    if not stages:
        raise ValueError("cannot serialize an empty stage list")
    return "|".join(serialize_train_stage(stage) for stage in stages)


def parse_train_stages(raw: str) -> tuple[TrainStageSpec, ...]:
    parts = [part.strip() for part in str(raw).split("|") if part.strip()]
    if not parts:
        raise ValueError("op_train_stages must contain at least one stage")
    return tuple(parse_train_stage(part) for part in parts)


def training_regime_to_axes(spec: TrainingRegimeSpec) -> dict[str, Any]:
    """Flatten a training genotype into proposal/ledger axes."""

    freeze_schedule = ">".join(
        f"{stage.target}={stage.freeze_policy}" for stage in spec.stages
    )
    return {
        AXIS_TRAIN_REGIME: spec.name,
        AXIS_TRAIN_STAGES: serialize_train_stages(spec.stages),
        AXIS_FREEZE_SCHEDULE: freeze_schedule,
        AXIS_OPTIMIZER: spec.optimizer,
        AXIS_BASE_LR: float(spec.base_lr),
        AXIS_WEIGHT_DECAY: float(spec.weight_decay),
        AXIS_SCHEDULER: spec.scheduler,
        AXIS_WARMUP_FRACTION: float(spec.warmup_fraction),
        AXIS_MAX_GRAD_NORM: float(spec.max_grad_norm),
    }


def training_regime_from_axes(axes: dict[str, Any]) -> TrainingRegimeSpec:
    """Rebuild a training genotype from proposal/ledger axes."""

    stages = parse_train_stages(str(axes.get(AXIS_TRAIN_STAGES) or "all:1:freeze:1:reset"))
    return TrainingRegimeSpec(
        name=str(axes.get(AXIS_TRAIN_REGIME) or "custom"),
        stages=stages,
        optimizer=str(axes.get(AXIS_OPTIMIZER) or "adamw"),
        base_lr=float(axes.get(AXIS_BASE_LR) or 3e-4),
        weight_decay=float(axes.get(AXIS_WEIGHT_DECAY) or 0.01),
        scheduler=str(axes.get(AXIS_SCHEDULER) or "cosine"),
        warmup_fraction=float(axes.get(AXIS_WARMUP_FRACTION) or 0.05),
        max_grad_norm=float(axes.get(AXIS_MAX_GRAD_NORM) or 1.0),
    )


def _single_stage_all() -> TrainingRegimeSpec:
    return TrainingRegimeSpec(
        name="all_train",
        stages=(TrainStageSpec("all", 1000, freeze_others=True),),
    )


def _embed_then_all() -> TrainingRegimeSpec:
    return TrainingRegimeSpec(
        name="embed_warm_then_all",
        stages=(
            TrainStageSpec("embedding_lm_head", 200, lr_scale=1.5, freeze_others=True),
            TrainStageSpec("all", 800, lr_scale=1.0, freeze_others=True),
        ),
    )


def _carrier_then_all() -> TrainingRegimeSpec:
    return TrainingRegimeSpec(
        name="carrier_warm_then_all",
        stages=(
            TrainStageSpec("carrier", 250, lr_scale=1.25, freeze_others=True),
            TrainStageSpec("all", 750, lr_scale=1.0, freeze_others=True),
        ),
    )


def _body_then_all() -> TrainingRegimeSpec:
    return TrainingRegimeSpec(
        name="body_warm_then_all",
        stages=(
            TrainStageSpec("non_embedding", 250, lr_scale=1.25, freeze_others=True),
            TrainStageSpec("all", 750, lr_scale=1.0, freeze_others=True),
        ),
    )


def _router_then_all() -> TrainingRegimeSpec:
    return TrainingRegimeSpec(
        name="router_warm_then_all",
        stages=(
            TrainStageSpec("router", 150, lr_scale=2.0, freeze_others=True),
            TrainStageSpec("all", 850, lr_scale=1.0, freeze_others=True),
        ),
    )


_IMPLEMENTED_REGIMES = (
    _single_stage_all,
    _embed_then_all,
    _body_then_all,
    _carrier_then_all,
    _router_then_all,
)


def implemented_training_regimes() -> dict[str, TrainingRegimeSpec]:
    """Implemented regime roster keyed by name."""

    regimes = [factory() for factory in _IMPLEMENTED_REGIMES]
    return {regime.name: regime for regime in regimes}


def sample_training_regime_spec(gen: torch.Generator) -> TrainingRegimeSpec:
    """Sample an implemented training regime deterministically from ``gen``."""

    idx = int(torch.randint(len(_IMPLEMENTED_REGIMES), (1,), generator=gen))
    return _IMPLEMENTED_REGIMES[idx]()
