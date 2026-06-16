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

_TASKS_BY_NAME: dict[str, ProbeTask] = {task.name: task for task in DEFAULT_PROBE_TASKS}
_LONG_GAP_TASKS = (
    "shifted_copy",
    "copy_from_uniform_past",
    "causal_induction",
    "running_parity",
)
_BINDING_TASKS = (
    "copy_from_uniform_past",
    "causal_induction",
    "shifted_copy",
)
_PHYSICS_MIN_STEPS = 80
_PHYSICS_PROBE_LR = 3e-3


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
    lr: float = 1e-3,
) -> dict[str, Any]:
    result: ProbeResult = short_training_probe(
        lane,
        dim=dim,
        seq_len=seq_len,
        n_steps=n_steps,
        seed=seed,
        lr=lr,
        target_fn=task.target_fn,
    )
    return {
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "loss_ratio_initial_over_final": result.loss_ratio_initial_over_final,
        "trained_successfully": result.trained_successfully,
        "difficulty": task.difficulty,
    }


def physics_probe_tasks_for_spec(spec: ProposalSpec) -> tuple[ProbeTask, ...]:
    axes = spec.math_axes
    if axes.get("op_search_track") != "physics_atom":
        return DEFAULT_PROBE_TASKS
    target = str(axes.get("op_physics_target") or "")
    if target.startswith("long_gap"):
        names = _LONG_GAP_TASKS
    elif target in {"binding_content_addressed_state", "broad_kv_content_lookup"}:
        names = _BINDING_TASKS
    else:
        names = ("copy_from_uniform_past", "causal_induction")
    return tuple(_TASKS_BY_NAME[name] for name in names)


def physics_probe_steps_for_spec(spec: ProposalSpec, n_steps: int) -> int:
    if spec.math_axes.get("op_search_track") != "physics_atom":
        return n_steps
    return max(n_steps, _PHYSICS_MIN_STEPS)


def physics_probe_lr_for_spec(spec: ProposalSpec, lr: float = 1e-3) -> float:
    if spec.math_axes.get("op_search_track") != "physics_atom":
        return lr
    return max(lr, _PHYSICS_PROBE_LR)


def validate_in_context(
    spec: ProposalSpec,
    lane: nn.Module,
    *,
    dim: int = 32,
    seq_len: int = 32,
    n_steps: int = 100,
    lr: float = 1e-3,
    learning_threshold: float = 1.25,
    tasks: Sequence[ProbeTask] | None = None,
) -> InContextScorecard:
    """Grade ``lane`` on every task in ``tasks``; report aggregate signal."""
    tasks = tuple(tasks) if tasks is not None else physics_probe_tasks_for_spec(spec)
    n_steps = physics_probe_steps_for_spec(spec, n_steps)
    lr = physics_probe_lr_for_spec(spec, lr)
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
            lr=lr,
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
        notes=(
            (
                f"physics_probe_tasks={','.join(task.name for task in tasks)}",
                f"physics_probe_steps={n_steps}",
                f"physics_probe_lr={lr}",
            )
            if spec.math_axes.get("op_search_track") == "physics_atom"
            else ()
        ),
    )
