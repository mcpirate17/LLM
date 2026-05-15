"""In-context validator — multi-task training probe for a lane module.

Runs ``short_training_probe`` once per task in the probe suite and packages
the aggregate as a JSON-serializable scorecard. A lane "learned signal"
when its mean loss reduction across the suite is meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Sequence


from torch import nn

from ..harness.probe_block import ProbeResult, short_training_probe
from ..harness.probe_tasks import DEFAULT_PROBE_TASKS, ProbeTask
from ..proposer.spec_generator import ProposalSpec


@dataclass(frozen=True, slots=True)
class InContextScorecard:
    proposal_id: str
    name: str
    category: str
    per_task: dict[str, dict[str, Any]]
    aggregate_loss_ratio: float  # max-over-tasks: best-task learning signal
    mean_loss_ratio: float  # mean-over-tasks: breadth of learning
    learned_signal: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


def _grade_one_task(
    lane: nn.Module,
    task: ProbeTask,
    *,
    dim: int,
    seq_len: int,
    n_steps: int,
    seed: int,
) -> dict[str, Any]:
    result: ProbeResult = short_training_probe(
        lane,
        dim=dim,
        seq_len=seq_len,
        n_steps=n_steps,
        seed=seed,
        target_fn=task.target_fn,
    )
    return {
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "loss_ratio_initial_over_final": result.loss_ratio_initial_over_final,
        "trained_successfully": result.trained_successfully,
        "difficulty": task.difficulty,
    }


def validate_in_context(
    spec: ProposalSpec,
    lane: nn.Module,
    *,
    dim: int = 32,
    seq_len: int = 32,
    n_steps: int = 100,
    learning_threshold: float = 1.25,
    tasks: Sequence[ProbeTask] = DEFAULT_PROBE_TASKS,
) -> InContextScorecard:
    """Grade ``lane`` on every task in ``tasks``; report aggregate signal."""
    per_task: dict[str, dict[str, Any]] = {}
    ratios: list[float] = []
    for index, task in enumerate(tasks):
        per_task[task.name] = _grade_one_task(
            lane,
            task,
            dim=dim,
            seq_len=seq_len,
            n_steps=n_steps,
            seed=index,
        )
        if per_task[task.name]["trained_successfully"]:
            ratios.append(per_task[task.name]["loss_ratio_initial_over_final"])
    aggregate_ratio = max(ratios) if ratios else 0.0
    mean_ratio = mean(ratios) if ratios else 0.0
    # A lane that meaningfully learns ANY of the tasks is interesting signal —
    # different primitives are good at different things (local conv learns
    # periodic_average, state-bearing op learns shifted_copy, etc.).
    learned = aggregate_ratio >= learning_threshold
    return InContextScorecard(
        proposal_id=spec.proposal_id,
        name=spec.name,
        category=spec.category,
        per_task=per_task,
        aggregate_loss_ratio=aggregate_ratio,
        mean_loss_ratio=mean_ratio,
        learned_signal=learned,
    )
