"""SSM-fair capability axis: state-tracking / compression probe scoring.

The ``harder_binding_tasks`` suite is all key-value associative recall — the
axis where attention beats SSMs by construction (Zoology). It cannot fairly rank
a non-QKV mechanism (routing / compression / state-space), which trades exact
recall for O(L) memory and long-range state. This module grades a lane on the
complementary axis: the ``probe_tasks`` MSE state-tracking / copy / compression
battery (``running_parity`` is literally annotated "typical SSM advantage over
attention"). The score is continuous MSE loss-reduction (higher SNR than accuracy
— AI2 "Signal and Noise"), reported per task and grouped into axes so the output
is a 2-D Pareto profile, not a single "beats frontier" bit.

``score_state_tracking`` is the per-lane primitive shared by the cohort tool
(``research/tools/grade_ssm_fair_cohort.py``) and by component_fab selection, so
the same number drives the leaderboard and the promotion signal.
"""

from __future__ import annotations

from typing import Callable

from torch import nn

from .probe_block import short_training_probe
from .probe_tasks import DEFAULT_PROBE_TASKS, ProbeTask

LaneFactory = Callable[[int], nn.Module]

# Axis grouping of the probe battery. state_tracking + copy_compression are the
# SSM/non-QKV-favoured axes; recall_induction is the attention-favoured contrast.
AXES: dict[str, tuple[str, ...]] = {
    "state_tracking": (
        "running_parity",
        "causal_max",
        "running_mean",
        "periodic_average",
    ),
    "copy_compression": ("shifted_copy", "copy_from_uniform_past"),
    "recall_induction": ("causal_induction",),
}

_TASKS_BY_NAME: dict[str, ProbeTask] = {t.name: t for t in DEFAULT_PROBE_TASKS}


def axis_of(task_name: str) -> str:
    """Return the axis a probe task belongs to ('unknown' if ungrouped)."""
    for axis, names in AXES.items():
        if task_name in names:
            return axis
    return "unknown"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def score_state_tracking(
    lane_factory: LaneFactory,
    *,
    dim: int = 32,
    seq_len: int = 32,
    n_steps: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    seeds: tuple[int, ...] = (0,),
    device: str = "cpu",
    task_names: tuple[str, ...] | None = None,
) -> dict:
    """Grade ``lane_factory`` on the state-tracking / compression probe battery.

    For every requested task and seed, trains a fresh ``WinnerLikeBlock(lane)`` via
    ``short_training_probe`` and records the MSE loss-reduction ratio
    (initial/final; higher = more learned). Returns ``{"per_task": {name:
    {ratio, final_loss}}, "per_axis": {axis: mean_ratio}, "overall": mean_ratio}``.

    A fresh lane is built per (task, seed) so no weights leak across tasks.
    """
    names = task_names or tuple(t.name for t in DEFAULT_PROBE_TASKS)
    per_task: dict[str, dict[str, float | str]] = {}
    for name in names:
        task = _TASKS_BY_NAME.get(name)
        if task is None:
            continue
        ratios: list[float] = []
        finals: list[float] = []
        for seed in seeds:
            res = short_training_probe(
                lane_factory(dim),
                dim=dim,
                seq_len=seq_len,
                n_steps=n_steps,
                batch_size=batch_size,
                lr=lr,
                device=device,
                seed=seed,
                target_fn=task.target_fn,
            )
            ratios.append(res.loss_ratio_initial_over_final)
            finals.append(res.final_loss)
        per_task[name] = {
            "ratio": _mean(ratios),
            "final_loss": _mean(finals),
            "axis": axis_of(name),
        }

    per_axis: dict[str, float] = {}
    for axis, axis_names in AXES.items():
        vals = [float(per_task[n]["ratio"]) for n in axis_names if n in per_task]
        if vals:
            per_axis[axis] = _mean(vals)

    overall = _mean([float(v["ratio"]) for v in per_task.values()])
    return {"per_task": per_task, "per_axis": per_axis, "overall": overall}
