"""Adversarial episodic retention probes for lane-level TinyLM comparisons.

The standard long-gap task stores one pair across repetitions of one NOISE
token. These probes vary filler content, insert competing writes, increase the
number of simultaneous associations, and evaluate beyond the training length.
Key/value assignments are resampled per example, preventing global pair
memorization from masquerading as retention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import nn

from .binding_taskgen import (
    TokenBatch,
    generate_retention_batch as _generate_retention_batch_vectorized,
    pair_positions as _pair_positions,
)
from .training_probe import build_tiny_lm, seeded_generator, train_token_task

FillerMode = Literal["constant", "random"]

__all__ = [
    "DEFAULT_RETENTION_TASKS",
    "FillerMode",
    "RetentionResult",
    "RetentionTask",
    "_pair_positions",
    "generate_retention_batch",
    "run_retention_task",
]


@dataclass(frozen=True, slots=True)
class RetentionTask:
    name: str
    train_seq_len: int
    eval_seq_len: int
    n_pairs: int
    filler_mode: FillerMode
    n_keys: int = 8
    n_values: int = 8
    n_fillers: int = 16

    def __post_init__(self) -> None:
        if self.train_seq_len < 16 or self.eval_seq_len < 16:
            raise ValueError("retention sequence lengths must be at least 16")
        if self.eval_seq_len < self.train_seq_len:
            raise ValueError("eval_seq_len must be >= train_seq_len")
        if not 1 <= self.n_pairs <= min(self.n_keys, self.n_values):
            raise ValueError(
                "n_pairs must fit distinct keys and values: "
                f"pairs={self.n_pairs}, keys={self.n_keys}, values={self.n_values}"
            )
        if self.filler_mode not in ("constant", "random"):
            raise ValueError(f"unsupported filler_mode: {self.filler_mode!r}")
        if self.n_fillers <= 0:
            raise ValueError("n_fillers must be positive")
        if self.n_pairs > 1 and self.train_seq_len < 2 * self.n_pairs + 8:
            raise ValueError(
                "train_seq_len too short for non-overlapping writes: "
                f"seq={self.train_seq_len}, pairs={self.n_pairs}"
            )

    @property
    def vocab_size(self) -> int:
        return self.n_keys + self.n_values + self.n_fillers + 3

    @property
    def chance_accuracy(self) -> float:
        return 1.0 / self.n_values


@dataclass(frozen=True, slots=True)
class RetentionResult:
    task_name: str
    mixer_label: str
    eval_accuracy: float
    chance_accuracy: float
    chance_normalized_lift: float
    train_loss_initial: float
    train_loss_final: float
    train_accuracy_final: float
    n_params: int


DEFAULT_RETENTION_TASKS: tuple[RetentionTask, ...] = (
    RetentionTask("constant_one_256", 256, 256, 1, "constant"),
    RetentionTask("varied_one_256", 256, 256, 1, "random"),
    RetentionTask("intervening_four_256", 256, 256, 4, "random"),
    RetentionTask("load_eight_256", 256, 256, 8, "random"),
    RetentionTask("extrapolate_one_128_to_256", 128, 256, 1, "random"),
)


def generate_retention_batch(
    task: RetentionTask,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator,
) -> TokenBatch:
    """Generate episodic pairs followed by a query for the earliest pair."""
    return _generate_retention_batch_vectorized(task, batch_size, seq_len, generator)


def run_retention_task(
    lane_factory: Callable[[int], nn.Module],
    task: RetentionTask,
    *,
    mixer_label: str,
    dim: int = 64,
    n_blocks: int = 2,
    n_train_steps: int = 500,
    batch_size: int = 32,
    learning_rate: float = 3e-3,
    n_eval_batches: int = 8,
    seed: int = 0,
    device: str = "cpu",
) -> RetentionResult:
    """Train on one episodic retention condition and score held-out episodes.

    A lane that crashes or produces a non-finite loss is logged loudly by
    the shared training core and scored 0.0 (eval accuracy), never silently
    swallowed.
    """
    rng = seeded_generator(seed)
    model = build_tiny_lm(
        lane_factory,
        vocab_size=task.vocab_size,
        dim=dim,
        n_blocks=n_blocks,
        max_seq_len=task.eval_seq_len,
        use_position_embedding=False,
        use_rope=False,
        device=device,
    )
    trace = train_token_task(
        model,
        lambda gen: generate_retention_batch(task, batch_size, task.train_seq_len, gen),
        rng=rng,
        eval_at_steps=(n_train_steps,),
        eval_batch_fn=lambda gen: generate_retention_batch(
            task, batch_size, task.eval_seq_len, gen
        ),
        eval_seed=seed + 10007,
        n_eval_batches=n_eval_batches,
        learning_rate=learning_rate,
        device=device,
        probe=f"retention:{task.name}:{mixer_label}",
    )
    checkpoint = trace.checkpoint_at(n_train_steps)
    eval_accuracy = checkpoint.eval_accuracy if checkpoint is not None else 0.0
    chance = task.chance_accuracy
    normalized_lift = (eval_accuracy - chance) / (1.0 - chance)
    return RetentionResult(
        task_name=task.name,
        mixer_label=mixer_label,
        eval_accuracy=eval_accuracy,
        chance_accuracy=chance,
        chance_normalized_lift=normalized_lift,
        train_loss_initial=trace.initial_loss,
        train_loss_final=trace.final_loss,
        train_accuracy_final=trace.final_train_accuracy,
        n_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
