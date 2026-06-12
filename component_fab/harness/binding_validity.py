"""Validity-controlled episodic binding tasks.

The historical harder-binding suite is retained for score continuity. These
tasks isolate benchmark semantics that its generators currently conflate:

* unique-key recall has exactly one value per queried key;
* distinct-key interference adds competing writes without changing the target;
* same-key overwrite explicitly defines the latest value as the answer;
* episodic composition randomizes values per example, preventing a global
  ``(entity, attribute) -> value`` rule from solving the task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import nn

from .binding_taskgen import (
    TokenBatch,
    generate_validity_compositional_batch,
    generate_validity_flat_batch,
)
from .harder_binding_tasks import HardBindingResult
from .training_probe import build_tiny_lm, seeded_generator, train_token_task

BindingValidityKind = Literal[
    "unique_multi_query",
    "distinct_key_interference",
    "same_key_overwrite",
    "episodic_compositional",
]

_N_SPECIAL = 4
BINDING_VALIDITY_VERSION = "episodic-v2-random-query-2026-06-07"


@dataclass(frozen=True, slots=True)
class BindingValidityTask:
    name: str
    kind: BindingValidityKind
    seq_len: int = 64
    n_keys: int = 8
    n_values: int = 8
    n_pairs: int = 4
    n_queries: int = 2
    n_entities: int = 4
    n_attributes: int = 4
    scatter_writes: bool = False

    def __post_init__(self) -> None:
        if self.seq_len < 24:
            raise ValueError("seq_len must be at least 24")
        if self.n_values < 2:
            raise ValueError("n_values must be at least 2")
        if self.kind == "episodic_compositional":
            if self.n_pairs > self.n_entities * self.n_attributes:
                raise ValueError("n_pairs exceeds available entity-attribute pairs")
        elif self.n_pairs > self.n_keys:
            raise ValueError("n_pairs exceeds available distinct keys")
        if not 1 <= self.n_queries <= self.n_pairs:
            raise ValueError("n_queries must be between 1 and n_pairs")
        write_width = 3 if self.kind == "episodic_compositional" else 2
        query_width = 5 if self.kind == "episodic_compositional" else 4
        required = self.n_pairs * write_width + self.n_queries * query_width
        if required > self.seq_len:
            raise ValueError(
                f"seq_len {self.seq_len} is too short for {required} task tokens"
            )

    @property
    def value_start(self) -> int:
        if self.kind == "episodic_compositional":
            return self.n_entities + self.n_attributes
        return self.n_keys

    @property
    def vocab_size(self) -> int:
        return self.value_start + self.n_values + _N_SPECIAL

    @property
    def special(self) -> dict[str, int]:
        base = self.vocab_size - _N_SPECIAL
        return {"PAD": base, "QUERY": base + 1, "ANS": base + 2, "NOISE": base + 3}

    @property
    def chance_accuracy(self) -> float:
        return 1.0 / self.n_values


@dataclass(frozen=True, slots=True)
class LegacyBatchAudit:
    examples: int
    examples_with_duplicate_keys: int
    examples_with_conflicting_values: int

    @property
    def duplicate_key_rate(self) -> float:
        return self.examples_with_duplicate_keys / max(1, self.examples)

    @property
    def conflicting_value_rate(self) -> float:
        return self.examples_with_conflicting_values / max(1, self.examples)


DEFAULT_BINDING_VALIDITY_TASKS: tuple[BindingValidityTask, ...] = (
    BindingValidityTask("episodic_unique_multi_query", "unique_multi_query"),
    BindingValidityTask(
        "episodic_distinct_key_interference",
        "distinct_key_interference",
        n_pairs=6,
        n_queries=2,
    ),
    BindingValidityTask(
        "episodic_same_key_overwrite",
        "same_key_overwrite",
        n_pairs=2,
        n_queries=1,
    ),
    BindingValidityTask(
        "episodic_compositional",
        "episodic_compositional",
        n_pairs=6,
        n_queries=2,
    ),
)

HARD_BINDING_VALIDITY_TASKS: tuple[BindingValidityTask, ...] = (
    BindingValidityTask(
        "hard_unique_12_pairs_6_queries_128",
        "unique_multi_query",
        seq_len=128,
        n_keys=16,
        n_pairs=12,
        n_queries=6,
    ),
    BindingValidityTask(
        "hard_interference_16_pairs_8_queries_256",
        "distinct_key_interference",
        seq_len=256,
        n_keys=16,
        n_pairs=16,
        n_queries=8,
    ),
    BindingValidityTask(
        "hard_variable_layout_12_pairs_6_queries_128",
        "unique_multi_query",
        seq_len=128,
        n_keys=16,
        n_pairs=12,
        n_queries=6,
        scatter_writes=True,
    ),
    BindingValidityTask(
        "hard_compositional_16_pairs_8_queries_128",
        "episodic_compositional",
        seq_len=128,
        n_entities=8,
        n_attributes=8,
        n_pairs=16,
        n_queries=8,
        scatter_writes=True,
    ),
)


def binding_validity_load_ladder() -> tuple[BindingValidityTask, ...]:
    """Return increasing pair/load/length conditions for ceiling analysis."""
    configs = (
        (4, 2, 64),
        (8, 4, 128),
        (16, 8, 256),
        (32, 8, 512),
    )
    return tuple(
        BindingValidityTask(
            f"load_{n_pairs}_queries_{n_queries}_len_{seq_len}",
            "distinct_key_interference",
            seq_len=seq_len,
            n_keys=n_pairs,
            n_pairs=n_pairs,
            n_queries=n_queries,
            scatter_writes=True,
        )
        for n_pairs, n_queries, seq_len in configs
    )


def audit_flat_writes(
    ids: torch.Tensor,
    *,
    n_write_pairs: int,
    key_upper_bound: int,
) -> LegacyBatchAudit:
    """Measure duplicate and conflicting flat ``[key, value]`` writes."""
    duplicate = 0
    conflicting = 0
    for row in ids:
        values_by_key: dict[int, set[int]] = {}
        write_count = 0
        for pair_index in range(n_write_pairs):
            offset = 2 * pair_index
            key = int(row[offset].item())
            value = int(row[offset + 1].item())
            if not 0 <= key < key_upper_bound:
                continue
            values_by_key.setdefault(key, set()).add(value)
            write_count += 1
        if write_count > len(values_by_key):
            duplicate += 1
        if any(len(values) > 1 for values in values_by_key.values()):
            conflicting += 1
    return LegacyBatchAudit(
        examples=int(ids.shape[0]),
        examples_with_duplicate_keys=duplicate,
        examples_with_conflicting_values=conflicting,
    )


def generate_binding_validity_batch(
    task: BindingValidityTask,
    batch_size: int,
    generator: torch.Generator,
) -> TokenBatch:
    if task.kind == "episodic_compositional":
        return generate_validity_compositional_batch(task, batch_size, generator)
    return generate_validity_flat_batch(task, batch_size, generator)


def run_binding_validity_task(
    lane_factory: Callable[[int], nn.Module],
    task: BindingValidityTask,
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
) -> HardBindingResult:
    """Train and evaluate one validity-controlled episodic task."""
    rng = seeded_generator(seed)
    model = build_tiny_lm(
        lane_factory,
        vocab_size=task.vocab_size,
        dim=dim,
        n_blocks=n_blocks,
        max_seq_len=task.seq_len,
        use_position_embedding=True,
        device=device,
    )
    trace = train_token_task(
        model,
        lambda gen: generate_binding_validity_batch(task, batch_size, gen),
        rng=rng,
        eval_at_steps=(n_train_steps,),
        eval_batch_fn=lambda gen: generate_binding_validity_batch(
            task, batch_size, gen
        ),
        eval_seed=seed + 10007,
        n_eval_batches=n_eval_batches,
        learning_rate=learning_rate,
        device=device,
        probe=f"binding_validity:{task.name}:{mixer_label}",
    )
    checkpoint = trace.checkpoint_at(n_train_steps)
    return HardBindingResult(
        task_name=task.name,
        mixer_label=mixer_label,
        train_loss_initial=trace.initial_loss,
        train_loss_final=trace.final_loss,
        train_accuracy_final=trace.final_train_accuracy,
        eval_accuracy=checkpoint.eval_accuracy if checkpoint is not None else 0.0,
        chance_accuracy=task.chance_accuracy,
        converged=trace.converged,
        n_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
