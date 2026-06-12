"""Shared helpers for one-shot recall-probe experiment scripts.

All four test_*.py scripts share:
  1. The ``UniversalRecallLane`` implementation (pooling + latching + slotted table).
  2. A "build comparisons list → loop run_one_task_checkpoints → dump JSON" runner.

Import from here; do not duplicate in sibling scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from component_fab.harness.harder_binding_tasks import (
    HardBindingTask,
    default_hard_binding_tasks,
    run_one_task_checkpoints,
)

# ---------------------------------------------------------------------------
# Shared architecture
# ---------------------------------------------------------------------------


class UniversalRecallLane(nn.Module):
    """Synthesizes Pooling, Latching, and Slotted Tables.

    Aims to be a general-purpose non-QKV recall expert.
    """

    def __init__(
        self,
        dim: int,
        n_slots: int = 16,
        memory_dim: int = 16,
        latch_len: int = 3,
        pool_period: int = 4,
    ) -> None:
        super().__init__()
        self.q = nn.Linear(dim, memory_dim, bias=False)
        self.k = nn.Linear(dim, memory_dim, bias=False)
        self.v = nn.Linear(dim, memory_dim, bias=False)
        self.latch_mix = nn.Linear(memory_dim * latch_len, memory_dim)
        self.write_route = nn.Linear(dim, n_slots)
        self.out = nn.Linear(memory_dim, dim, bias=False)
        self.n_slots = n_slots
        self.memory_dim = memory_dim
        self.latch_len = latch_len
        self.pool_period = pool_period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        slot_keys = torch.zeros(
            batch_size, self.n_slots, self.memory_dim, device=x.device, dtype=x.dtype
        )
        slot_vals = torch.zeros(
            batch_size, self.n_slots, self.memory_dim, device=x.device, dtype=x.dtype
        )
        key_latch = [
            torch.zeros(batch_size, self.memory_dim, device=x.device, dtype=x.dtype)
            for _ in range(self.latch_len)
        ]
        pool_accum = torch.zeros(batch_size, dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            token = x[:, t]
            pool_accum = pool_accum + token
            if (t + 1) % self.pool_period == 0:
                pooled_token = pool_accum / float(self.pool_period)
                pool_accum = torch.zeros_like(pool_accum)
            else:
                pooled_token = token
            kt = torch.tanh(self.k(pooled_token))
            vt = self.v(pooled_token)
            qt = torch.tanh(self.q(token))
            latched_context = self.latch_mix(torch.cat(key_latch, dim=-1))
            w_route = torch.softmax(self.write_route(pooled_token), dim=-1)
            w_idx = w_route.argmax(dim=-1)
            mask = (
                torch.nn.functional.one_hot(w_idx, num_classes=self.n_slots)
                .unsqueeze(-1)
                .to(x.dtype)
            )
            slot_keys = slot_keys * (1.0 - mask) + mask * latched_context.unsqueeze(1)
            slot_vals = slot_vals * (1.0 - mask) + mask * vt.unsqueeze(1)
            read_weights = torch.softmax(
                torch.einsum("bd,bsd->bs", qt, slot_keys), dim=-1
            )
            read = torch.einsum("bs,bsd->bd", read_weights, slot_vals)
            outputs.append(self.out(read))
            key_latch = key_latch[1:] + [kt]
        return torch.stack(outputs, dim=1)


# ---------------------------------------------------------------------------
# Shared runner
# ---------------------------------------------------------------------------

#: Type alias for one entry in the comparisons list.
Comparison = tuple[str, Callable[[int], nn.Module], str]


def run_comparisons(
    comparisons: list[Comparison],
    *,
    steps: int,
    dim: int,
    device: str,
    out_path: str | Path,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Run ``run_one_task_checkpoints`` for each (name, factory, task_name) triple.

    Parameters
    ----------
    comparisons:
        List of ``(lane_name, factory, task_name)`` tuples.
        Multiple entries with the same ``lane_name`` are merged into one result dict.
    steps:
        Number of training steps (single checkpoint at this step).
    dim:
        Model hidden dimension.
    device:
        PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.
    out_path:
        JSON file path where results are written.  Parent directories are
        created automatically.
    seed:
        Random seed passed to both task generation and training.

    Returns
    -------
    dict mapping lane_name → {task_name → eval_accuracy}
    """
    tasks: list[HardBindingTask] = list(default_hard_binding_tasks(seed=seed))

    results: dict[str, dict[str, float]] = {}
    for name, factory, tn in comparisons:
        task = next((t for t in tasks if t.name == tn), None)
        if task is None:
            raise ValueError(f"Task {tn!r} not found in default_hard_binding_tasks")
        print(f"Running {name} on {tn}...", flush=True)
        rows = run_one_task_checkpoints(
            factory,
            task,
            eval_at_steps=(steps,),
            dim=dim,
            seed=seed,
            device=device,
            mixer_label=name,
        )
        results.setdefault(name, {})[tn] = rows[steps].eval_accuracy
        print(f"DONE: {name} {tn}: {results[name][tn]:.4f}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    return results
