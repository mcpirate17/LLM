"""Harder-than-nano_bind discrete/symbolic binding tasks for fab lanes.

These probes sit between ``nano_bind`` (single continuous key, single
slot, K=4 classes) and full language modeling. The point is a stronger
fab-level answer than nano_bind without paying the cost of wikitext +
BLiMP, while still controlling for **mixer alone** by running candidate
and baseline mixers in the same ``TinyLM`` wrapper at identical params,
steps, and optimizer.

Tasks (all discrete-token, train + held-out eval):

- ``multi_query_kv_recall``: N (k,v) pairs followed by Q query keys.
  Model emits the matching v at each query position.
- ``distractor_kv_recall``: Like above, but each true key has a
  distractor key sharing a near-duplicate embedding-relevant prefix.
  Tests exact-match vs heuristic-similarity binding.
- ``long_gap_recall``: One (k,v) at the start, query 64-256 positions
  later. Pure long-context binding capability.
- ``variable_layout_recall``: Query order is randomized (no canonical
  ordered query cycle).
- ``compositional_binding``: (entity, attribute) -> value. Train and
  held-out splits use disjoint (e,a) combinations from the same
  entities + attributes — tests whether the lane learns binding as a
  mechanism vs memorizes positions.
- ``heldout_pair_recall``: train on a 90% subset of (k,v) pairings;
  eval on the held-out 10%. Same as compositional but on flat (k,v).

All tasks emit ``(ids, query_positions, target_ids)`` so we can train
with cross-entropy at the query positions only. Batch generation is the
vectorized :mod:`.binding_taskgen`; the train/eval loop is the shared
:mod:`.training_probe` core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .binding_taskgen import (
    RESERVED_TOKENS as _RESERVED_TOKENS,
    TokenBatch,
    generate_hard_compositional_batch,
    generate_hard_kv_batch,
    generate_hard_long_gap_batch,
)
from .tiny_lm import (
    DEFAULT_BASELINE_NAMES,
    TinyLM,
    lane_factory_for_baseline,
)
from .training_probe import (
    build_tiny_lm,
    seeded_generator,
    train_token_task,
)


@dataclass(frozen=True, slots=True)
class HardBindingTask:
    name: str
    seq_len: int
    n_keys: int
    n_values: int
    vocab_size: int
    train_pairs: tuple[tuple[int, int], ...]
    eval_pairs: tuple[tuple[int, int], ...]
    n_pairs_in_seq: int
    n_queries: int
    distractors_per_key: int = 0
    long_gap_min: int = 0
    long_gap_max: int = 0
    variable_layout: bool = False
    n_entities: int = 0
    n_attributes: int = 0


@dataclass(frozen=True, slots=True)
class HardBindingResult:
    task_name: str
    mixer_label: str
    train_loss_initial: float
    train_loss_final: float
    train_accuracy_final: float
    eval_accuracy: float
    chance_accuracy: float
    converged: bool
    n_params: int


_BATCH_GENERATORS: dict[
    str,
    Callable[[HardBindingTask, int, bool, torch.Generator], TokenBatch],
] = {
    "multi_query_kv_recall": generate_hard_kv_batch,
    "distractor_kv_recall": generate_hard_kv_batch,
    "long_gap_recall": generate_hard_long_gap_batch,
    "variable_layout_recall": generate_hard_kv_batch,
    "compositional_binding": generate_hard_compositional_batch,
    "heldout_pair_recall": generate_hard_kv_batch,
}


# ---------- Task factory ----------


def _build_heldout_split(
    n_keys: int, n_values: int, holdout_frac: float, seed: int
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """All (k, v) combinations split into train / eval pools."""
    rng = torch.Generator().manual_seed(seed)
    pairs = [(k, n_keys + v) for k in range(n_keys) for v in range(n_values)]
    perm = torch.randperm(len(pairs), generator=rng).tolist()
    n_eval = max(1, int(len(pairs) * holdout_frac))
    eval_pairs = tuple(pairs[i] for i in perm[:n_eval])
    train_pairs = tuple(pairs[i] for i in perm[n_eval:])
    return train_pairs, eval_pairs


def _build_compositional_split(
    n_entities: int, n_attributes: int, n_values: int, holdout_frac: float, seed: int
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    int,
]:
    """Compositional (e, a) -> v split where train/eval (e,a) combos are disjoint.

    Values are deterministic from (e, a) so the model has to compose the
    binding mechanism rather than memorize.
    """
    rng = torch.Generator().manual_seed(seed)
    encoded_ea_pairs = [
        (e * n_attributes + a, n_entities + n_attributes + ((e + 7 * a) % n_values))
        for e in range(n_entities)
        for a in range(n_attributes)
    ]
    perm = torch.randperm(len(encoded_ea_pairs), generator=rng).tolist()
    n_eval = max(1, int(len(encoded_ea_pairs) * holdout_frac))
    eval_pairs = tuple(encoded_ea_pairs[i] for i in perm[:n_eval])
    train_pairs = tuple(encoded_ea_pairs[i] for i in perm[n_eval:])
    vocab_size = n_entities + n_attributes + n_values + _RESERVED_TOKENS
    return train_pairs, eval_pairs, vocab_size


def _kv_task(
    name: str,
    seq_len: int,
    n_keys: int,
    n_values: int,
    vocab: int,
    train_pairs: tuple[tuple[int, int], ...],
    eval_pairs: tuple[tuple[int, int], ...],
    *,
    n_pairs_in_seq: int = 4,
    n_queries: int = 4,
    distractors_per_key: int = 0,
    long_gap_min: int = 0,
    long_gap_max: int = 0,
    variable_layout: bool = False,
) -> HardBindingTask:
    return HardBindingTask(
        name=name,
        seq_len=seq_len,
        n_keys=n_keys,
        n_values=n_values,
        vocab_size=vocab,
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
        n_pairs_in_seq=n_pairs_in_seq,
        n_queries=n_queries,
        distractors_per_key=distractors_per_key,
        long_gap_min=long_gap_min,
        long_gap_max=long_gap_max,
        variable_layout=variable_layout,
    )


_KV_TASK_OVERRIDES: tuple[tuple[str, dict[str, int | bool]], ...] = (
    ("multi_query_kv_recall", {}),
    ("distractor_kv_recall", {"distractors_per_key": 1}),
    (
        "long_gap_recall",
        {
            "n_pairs_in_seq": 1,
            "n_queries": 1,
            "long_gap_min": 64,
            "long_gap_max": 240,
            "_use_long_seq": True,
        },
    ),
    ("variable_layout_recall", {"variable_layout": True}),
    ("heldout_pair_recall", {}),
)


def default_hard_binding_tasks(
    *,
    seed: int = 0,
    seq_len_short: int = 64,
    seq_len_long: int = 256,
) -> tuple[HardBindingTask, ...]:
    """Six tasks at fixed scale. Same params + steps across all six."""
    n_keys, n_values = 8, 8
    train_pairs, eval_pairs = _build_heldout_split(n_keys, n_values, 0.125, seed)
    vocab = n_keys + n_values + _RESERVED_TOKENS

    kv_built: list[HardBindingTask] = []
    for name, overrides_const in _KV_TASK_OVERRIDES:
        overrides = dict(overrides_const)  # avoid mutating module-level constant
        seq_len = (
            seq_len_long if overrides.pop("_use_long_seq", False) else seq_len_short
        )
        kv_built.append(
            _kv_task(
                name,
                seq_len,
                n_keys,
                n_values,
                vocab,
                train_pairs,
                eval_pairs,
                **overrides,
            )
        )

    n_ent, n_attr, n_val = 4, 4, 8
    train_ea, eval_ea, vocab_ea = _build_compositional_split(
        n_ent, n_attr, n_val, 0.25, seed
    )
    composite = HardBindingTask(
        name="compositional_binding",
        seq_len=seq_len_short,
        n_keys=0,
        n_values=n_val,
        vocab_size=vocab_ea,
        train_pairs=train_ea,
        eval_pairs=eval_ea,
        n_pairs_in_seq=6,
        n_queries=2,
        n_entities=n_ent,
        n_attributes=n_attr,
    )
    # Insert compositional second-to-last to keep ordering aligned with prior layout.
    return tuple(kv_built[:-1]) + (composite,) + (kv_built[-1],)


# ---------- Training + scoring (thin wrappers over training_probe) ----------


def _task_model(
    lane_factory: Callable[[int], nn.Module],
    task: HardBindingTask,
    *,
    dim: int,
    n_blocks: int,
    device: str,
) -> TinyLM:
    return build_tiny_lm(
        lane_factory,
        vocab_size=task.vocab_size,
        dim=dim,
        n_blocks=n_blocks,
        max_seq_len=task.seq_len,
        use_position_embedding=True,
        device=device,
    )


def _chance_accuracy(task: HardBindingTask) -> float:
    return 1.0 / max(1, task.n_values or task.vocab_size)


def run_one_task(
    lane_factory: Callable[[int], nn.Module],
    task: HardBindingTask,
    *,
    mixer_label: str,
    dim: int = 64,
    n_blocks: int = 2,
    n_train_steps: int = 500,
    batch_size: int = 32,
    learning_rate: float = 3e-3,
    seed: int = 0,
    device: str = "cpu",
) -> HardBindingResult:
    """Train a TinyLM(lane) on ``task`` and report held-out accuracy.

    Same hyperparameters for every candidate and baseline — the only
    moving part is ``lane_factory``.
    """
    rng = seeded_generator(seed)
    model = _task_model(lane_factory, task, dim=dim, n_blocks=n_blocks, device=device)
    generate = _BATCH_GENERATORS[task.name]
    trace = train_token_task(
        model,
        lambda gen: generate(task, batch_size, False, gen),
        rng=rng,
        eval_at_steps=(n_train_steps,),
        eval_batch_fn=lambda gen: generate(task, batch_size, True, gen),
        eval_seed=None,  # legacy semantics: eval rng continues the train rng
        learning_rate=learning_rate,
        device=device,
        probe=f"{task.name}:{mixer_label}",
    )
    checkpoint = trace.checkpoint_at(n_train_steps)
    return HardBindingResult(
        task_name=task.name,
        mixer_label=mixer_label,
        train_loss_initial=trace.initial_loss,
        train_loss_final=trace.final_loss,
        train_accuracy_final=trace.final_train_accuracy,
        eval_accuracy=checkpoint.eval_accuracy if checkpoint is not None else 0.0,
        chance_accuracy=_chance_accuracy(task),
        converged=trace.converged,
        n_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )


def run_one_task_checkpoints(
    lane_factory: Callable[[int], nn.Module],
    task: HardBindingTask,
    *,
    eval_at_steps: tuple[int, ...],
    mixer_label: str,
    dim: int = 64,
    n_blocks: int = 2,
    batch_size: int = 32,
    learning_rate: float = 3e-3,
    seed: int = 0,
    device: str = "cpu",
    n_eval_batches: int = 8,
) -> dict[int, HardBindingResult]:
    """Train ONE trajectory and read held-out accuracy at each checkpoint.

    Trains a single ``TinyLM(lane)`` to ``max(eval_at_steps)`` and evaluates
    held-out binding accuracy at every step in ``eval_at_steps`` — i.e. the
    "checkpoint at 2K, continue to 3K" pattern, so the standard (2K) and
    thorough (3K) numbers come from the same run with no retraining. Eval uses a
    fresh, deterministically-seeded generator at every checkpoint, so the eval
    set is identical across checkpoints AND across models (fairer than the
    single-shot ``run_one_task``, whose eval rng inherits the training state).

    Returns ``{step: HardBindingResult}``. On failure every requested step gets a
    ``converged=False`` row so callers can still aggregate.
    """
    steps = sorted({int(s) for s in eval_at_steps if int(s) > 0})
    if not steps:
        raise ValueError("eval_at_steps must contain at least one positive step")

    rng = seeded_generator(seed)
    model = _task_model(lane_factory, task, dim=dim, n_blocks=n_blocks, device=device)
    generate = _BATCH_GENERATORS[task.name]
    trace = train_token_task(
        model,
        lambda gen: generate(task, batch_size, False, gen),
        rng=rng,
        eval_at_steps=steps,
        eval_batch_fn=lambda gen: generate(task, batch_size, True, gen),
        eval_seed=seed + 10007,
        n_eval_batches=n_eval_batches,
        learning_rate=learning_rate,
        device=device,
        probe=f"{task.name}:{mixer_label}:checkpoints",
    )
    chance = _chance_accuracy(task)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results: dict[int, HardBindingResult] = {
        row.step: HardBindingResult(
            task_name=task.name,
            mixer_label=mixer_label,
            train_loss_initial=trace.initial_loss,
            train_loss_final=row.train_loss,
            train_accuracy_final=row.train_accuracy,
            eval_accuracy=row.eval_accuracy,
            chance_accuracy=chance,
            converged=True,
            n_params=n_params,
        )
        for row in trace.checkpoints
    }
    for step in steps:  # fill any checkpoint missed by a mid-training failure
        results.setdefault(
            step,
            HardBindingResult(
                task_name=task.name,
                mixer_label=mixer_label,
                train_loss_initial=float("nan"),
                train_loss_final=float("nan"),
                train_accuracy_final=0.0,
                eval_accuracy=0.0,
                chance_accuracy=chance,
                converged=False,
                n_params=n_params,
            ),
        )
    return results


def run_harder_binding_suite(
    candidate_factory: Callable[[int], nn.Module],
    candidate_label: str,
    *,
    tasks: tuple[HardBindingTask, ...] | None = None,
    baseline_names: tuple[str, ...] = DEFAULT_BASELINE_NAMES,
    dim: int = 64,
    n_blocks: int = 2,
    n_train_steps: int = 500,
    batch_size: int = 32,
    learning_rate: float = 3e-3,
    seed: int = 0,
) -> dict[str, list[HardBindingResult]]:
    """Run the candidate lane and each baseline against every task.

    Returns ``{task_name: [candidate_result, *baseline_results]}``.
    """
    tasks = tasks if tasks is not None else default_hard_binding_tasks(seed=seed)
    out: dict[str, list[HardBindingResult]] = {}
    for task in tasks:
        rows: list[HardBindingResult] = []
        rows.append(
            run_one_task(
                candidate_factory,
                task,
                mixer_label=candidate_label,
                dim=dim,
                n_blocks=n_blocks,
                n_train_steps=n_train_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
            )
        )
        for name in baseline_names:
            rows.append(
                run_one_task(
                    lane_factory_for_baseline(name),
                    task,
                    mixer_label=name,
                    dim=dim,
                    n_blocks=n_blocks,
                    n_train_steps=n_train_steps,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    seed=seed,
                )
            )
        out[task.name] = rows
    return out
