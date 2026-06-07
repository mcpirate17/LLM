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
- ``variable_layout_recall``: Positions of (k,v) pairs and queries are
  randomized within the sequence (no canonical ``[k,v,k,v,q,a]`` shape).
- ``compositional_binding``: (entity, attribute) -> value. Train and
  held-out splits use disjoint (e,a) combinations from the same
  entities + attributes — tests whether the lane learns binding as a
  mechanism vs memorizes positions.
- ``heldout_pair_recall``: train on a 90% subset of (k,v) pairings;
  eval on the held-out 10%. Same as compositional but on flat (k,v).

All tasks emit ``(ids, query_positions, target_ids)`` so we can train
with cross-entropy at the query positions only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .tiny_lm import (
    DEFAULT_BASELINE_NAMES,
    TinyLM,
    TinyLMConfig,
    lane_factory_for_baseline,
)


# Special token ids reserved at the END of vocab. The task generator
# carves the rest of the vocab into keys / values / entities / attrs.
_RESERVED_TOKENS = 4  # PAD, QUERY, ANS, NOISE


def _reserved_offsets(vocab_size: int) -> dict[str, int]:
    base = vocab_size - _RESERVED_TOKENS
    return {"PAD": base, "QUERY": base + 1, "ANS": base + 2, "NOISE": base + 3}


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
    compositional: bool = False
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


# ---------- Batch generators ----------


def _sample_pairs(
    pool: tuple[tuple[int, int], ...],
    n: int,
    rng: torch.Generator,
) -> list[tuple[int, int]]:
    """Sample ``n`` pairs uniformly from ``pool``."""
    indices = torch.randint(0, len(pool), (n,), generator=rng)
    return [pool[i] for i in indices.tolist()]


def _lay_pairs(
    ids: torch.Tensor, b: int, cursor: int, pairs: list[tuple[int, int]]
) -> int:
    for k, v in pairs:
        ids[b, cursor] = k
        ids[b, cursor + 1] = v
        cursor += 2
    return cursor


def _lay_distractors(
    ids: torch.Tensor,
    b: int,
    cursor: int,
    seq_pairs: list[tuple[int, int]],
    task: HardBindingTask,
    rng: torch.Generator,
) -> int:
    for k, v in seq_pairs:
        for _ in range(task.distractors_per_key):
            distractor_v = int(
                torch.randint(0, task.n_values, (1,), generator=rng).item()
            )
            if distractor_v == v - task.n_keys:
                distractor_v = (distractor_v + 1) % task.n_values
            if cursor + 1 >= task.seq_len:
                return cursor
            ids[b, cursor] = k
            ids[b, cursor + 1] = task.n_keys + distractor_v
            cursor += 2
    return cursor


def _pad_noise_for_gap(
    ids: torch.Tensor,
    b: int,
    cursor: int,
    task: HardBindingTask,
    rng: torch.Generator,
    noise_token: int,
) -> int:
    if task.long_gap_min <= 0:
        return cursor
    gap_target = int(
        torch.randint(
            task.long_gap_min, task.long_gap_max + 1, (1,), generator=rng
        ).item()
    )
    while cursor < min(gap_target, task.seq_len - 4):
        ids[b, cursor] = noise_token
        cursor += 1
    return cursor


def _lay_queries(
    ids: torch.Tensor,
    query_positions: torch.Tensor,
    target_ids: torch.Tensor,
    b: int,
    cursor: int,
    seq_pairs: list[tuple[int, int]],
    task: HardBindingTask,
    rng: torch.Generator,
    res: dict[str, int],
) -> None:
    if task.variable_layout:
        order = torch.randperm(len(seq_pairs), generator=rng).tolist()
        query_keys = [
            seq_pairs[order[i % len(seq_pairs)]][0] for i in range(task.n_queries)
        ]
    else:
        query_keys = [seq_pairs[i % len(seq_pairs)][0] for i in range(task.n_queries)]
    for qi, qk in enumerate(query_keys):
        if cursor + 3 >= task.seq_len:
            break
        ids[b, cursor] = res["QUERY"]
        ids[b, cursor + 1] = qk
        ids[b, cursor + 2] = res["ANS"]
        true_v = next(v for k, v in seq_pairs if k == qk)
        query_positions[b, qi] = cursor + 2
        target_ids[b, qi] = true_v
        cursor += 4


def _generate_basic_batch(
    task: HardBindingTask,
    batch_size: int,
    eval_split: bool,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-query KV recall (plus distractor / gap / variable-layout variants).

    Sequence layout (fixed unless ``task.variable_layout``):
    ``[k1, v1, k2, v2, ..., kN, vN, QUERY, q1, ANS, ?, QUERY, q2, ANS, ?, ...]``
    """
    pool = task.eval_pairs if eval_split else task.train_pairs
    res = _reserved_offsets(task.vocab_size)
    ids = torch.full((batch_size, task.seq_len), res["PAD"], dtype=torch.long)
    query_positions = torch.zeros((batch_size, task.n_queries), dtype=torch.long)
    target_ids = torch.zeros((batch_size, task.n_queries), dtype=torch.long)
    for b in range(batch_size):
        seq_pairs = _sample_pairs(pool, task.n_pairs_in_seq, rng)
        cursor = _lay_pairs(ids, b, 0, seq_pairs)
        if task.distractors_per_key > 0:
            cursor = _lay_distractors(ids, b, cursor, seq_pairs, task, rng)
        cursor = _pad_noise_for_gap(ids, b, cursor, task, rng, res["NOISE"])
        _lay_queries(
            ids, query_positions, target_ids, b, cursor, seq_pairs, task, rng, res
        )
    return ids, query_positions, target_ids


def _generate_long_gap_batch(
    task: HardBindingTask,
    batch_size: int,
    eval_split: bool,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One (k,v), then noise filler of length in [long_gap_min, long_gap_max],
    then QUERY k ANS ?. Pure long-context binding test.
    """
    pool = task.eval_pairs if eval_split else task.train_pairs
    res = _reserved_offsets(task.vocab_size)
    ids = torch.full((batch_size, task.seq_len), res["NOISE"], dtype=torch.long)
    query_positions = torch.zeros((batch_size, 1), dtype=torch.long)
    target_ids = torch.zeros((batch_size, 1), dtype=torch.long)
    for b in range(batch_size):
        k, v = _sample_pairs(pool, 1, rng)[0]
        ids[b, 0] = k
        ids[b, 1] = v
        gap = int(
            torch.randint(
                task.long_gap_min, task.long_gap_max + 1, (1,), generator=rng
            ).item()
        )
        q_pos = min(2 + gap, task.seq_len - 4)
        ids[b, q_pos] = res["QUERY"]
        ids[b, q_pos + 1] = k
        ids[b, q_pos + 2] = res["ANS"]
        query_positions[b, 0] = q_pos + 2
        target_ids[b, 0] = v
    return ids, query_positions, target_ids


def _generate_compositional_batch(
    task: HardBindingTask,
    batch_size: int,
    eval_split: bool,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(entity, attribute) -> value. Held-out split uses (e,a) pairs that
    never appear together in train, even though e and a both appear in train.
    """
    pool = task.eval_pairs if eval_split else task.train_pairs
    res = _reserved_offsets(task.vocab_size)
    ids = torch.full((batch_size, task.seq_len), res["PAD"], dtype=torch.long)
    query_positions = torch.zeros((batch_size, task.n_queries), dtype=torch.long)
    target_ids = torch.zeros((batch_size, task.n_queries), dtype=torch.long)
    for b in range(batch_size):
        sampled = _sample_pairs(pool, task.n_pairs_in_seq, rng)
        cursor = 0
        # Pair format here: (encoded_ea, v). encoded_ea is e*A + a + offset.
        # We expand into [e, a, v] triples in the sequence.
        triples = []
        for encoded_ea, v in sampled:
            e = encoded_ea // task.n_attributes
            a = encoded_ea % task.n_attributes
            triples.append((e, task.n_entities + a, v))
        for e, a_tok, v in triples:
            if cursor + 3 > task.seq_len:
                break
            ids[b, cursor] = e
            ids[b, cursor + 1] = a_tok
            ids[b, cursor + 2] = v
            cursor += 3
        # Query: ask about a random triple's (e, a).
        for qi in range(task.n_queries):
            if cursor + 4 > task.seq_len:
                break
            e, a_tok, v = triples[qi % len(triples)]
            ids[b, cursor] = res["QUERY"]
            ids[b, cursor + 1] = e
            ids[b, cursor + 2] = a_tok
            ids[b, cursor + 3] = res["ANS"]
            query_positions[b, qi] = cursor + 3
            target_ids[b, qi] = v
            cursor += 4
    return ids, query_positions, target_ids


_BATCH_GENERATORS: dict[
    str,
    Callable[
        [HardBindingTask, int, bool, torch.Generator],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
] = {
    "multi_query_kv_recall": _generate_basic_batch,
    "distractor_kv_recall": _generate_basic_batch,
    "long_gap_recall": _generate_long_gap_batch,
    "variable_layout_recall": _generate_basic_batch,
    "compositional_binding": _generate_compositional_batch,
    "heldout_pair_recall": _generate_basic_batch,
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
        compositional=True,
        n_entities=n_ent,
        n_attributes=n_attr,
    )
    # Insert compositional second-to-last to keep ordering aligned with prior layout.
    return tuple(kv_built[:-1]) + (composite,) + (kv_built[-1],)


# ---------- Training + scoring ----------


def _gather_logits_at(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """``logits[B, L, V]`` + ``positions[B, Q]`` -> ``[B, Q, V]``."""
    b, q = positions.shape
    v = logits.shape[-1]
    pos_expanded = positions.unsqueeze(-1).expand(b, q, v)
    return logits.gather(1, pos_expanded)


@dataclass(slots=True)
class _TrainTrace:
    initial_loss: float
    final_loss: float
    final_train_acc: float
    converged: bool


def _train_one_lm(
    model: TinyLM,
    task: HardBindingTask,
    rng: torch.Generator,
    n_train_steps: int,
    batch_size: int,
    learning_rate: float,
    device: str = "cpu",
) -> _TrainTrace:
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generate = _BATCH_GENERATORS[task.name]
    initial_loss = float("nan")
    final_loss = float("nan")
    final_train_acc = 0.0
    try:
        model.train()
        for step in range(n_train_steps):
            ids, qpos, tgt = generate(task, batch_size, False, rng)
            if device != "cpu":
                ids, qpos, tgt = ids.to(device), qpos.to(device), tgt.to(device)
            logits = model(ids)
            qlogits = _gather_logits_at(logits, qpos)
            loss = nn.functional.cross_entropy(
                qlogits.reshape(-1, qlogits.shape[-1]),
                tgt.reshape(-1),
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step == 0:
                initial_loss = float(loss.item())
            final_loss = float(loss.item())
            if step == n_train_steps - 1:
                final_train_acc = float(
                    (qlogits.argmax(dim=-1) == tgt).float().mean().item()
                )
    except Exception:  # noqa: BLE001
        return _TrainTrace(initial_loss, final_loss, final_train_acc, converged=False)
    return _TrainTrace(initial_loss, final_loss, final_train_acc, converged=True)


def _eval_one_lm(
    model: TinyLM,
    task: HardBindingTask,
    rng: torch.Generator,
    batch_size: int,
    n_eval_batches: int = 8,
    device: str = "cpu",
) -> float:
    generate = _BATCH_GENERATORS[task.name]
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_eval_batches):
            ids, qpos, tgt = generate(task, batch_size, True, rng)
            if device != "cpu":
                ids, qpos, tgt = ids.to(device), qpos.to(device), tgt.to(device)
            logits = model(ids)
            qlogits = _gather_logits_at(logits, qpos)
            preds = qlogits.argmax(dim=-1)
            correct += int((preds == tgt).sum().item())
            total += int(tgt.numel())
    return correct / max(1, total)


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
    torch.manual_seed(seed)
    cfg = TinyLMConfig(
        vocab_size=task.vocab_size,
        dim=dim,
        n_blocks=n_blocks,
        use_position_embedding=True,
        max_seq_len=task.seq_len,
    )
    model = TinyLM(lane_factory, cfg).to(device)
    rng = torch.Generator().manual_seed(seed)
    trace = _train_one_lm(
        model, task, rng, n_train_steps, batch_size, learning_rate, device=device
    )
    eval_acc = (
        _eval_one_lm(model, task, rng, batch_size, device=device)
        if trace.converged
        else 0.0
    )
    chance = 1.0 / max(1, task.n_values or task.vocab_size)
    return HardBindingResult(
        task_name=task.name,
        mixer_label=mixer_label,
        train_loss_initial=trace.initial_loss,
        train_loss_final=trace.final_loss,
        train_accuracy_final=trace.final_train_acc,
        eval_accuracy=eval_acc,
        chance_accuracy=chance,
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

    torch.manual_seed(seed)
    cfg = TinyLMConfig(
        vocab_size=task.vocab_size,
        dim=dim,
        n_blocks=n_blocks,
        use_position_embedding=True,
        max_seq_len=task.seq_len,
    )
    model = TinyLM(lane_factory, cfg).to(device)
    results = _train_eval_checkpoints(
        model,
        task,
        steps=steps,
        mixer_label=mixer_label,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
        n_eval_batches=n_eval_batches,
    )
    chance = 1.0 / max(1, task.n_values or task.vocab_size)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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


def _train_eval_checkpoints(
    model: TinyLM,
    task: HardBindingTask,
    *,
    steps: list[int],
    mixer_label: str,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
    n_eval_batches: int,
) -> dict[int, HardBindingResult]:
    """Train ``model`` to ``max(steps)``, emit a result row at each checkpoint."""
    checkpoints = set(steps)
    rng = torch.Generator().manual_seed(seed)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generate = _BATCH_GENERATORS[task.name]
    chance = 1.0 / max(1, task.n_values or task.vocab_size)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results: dict[int, HardBindingResult] = {}
    initial_loss = float("nan")
    try:
        for step in range(1, steps[-1] + 1):
            model.train()
            ids, qpos, tgt = generate(task, batch_size, False, rng)
            if device != "cpu":
                ids, qpos, tgt = ids.to(device), qpos.to(device), tgt.to(device)
            qlogits = _gather_logits_at(model(ids), qpos)
            loss = nn.functional.cross_entropy(
                qlogits.reshape(-1, qlogits.shape[-1]), tgt.reshape(-1)
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step == 1:
                initial_loss = float(loss.item())
            if step not in checkpoints:
                continue
            train_acc = float((qlogits.argmax(dim=-1) == tgt).float().mean().item())
            eval_rng = torch.Generator().manual_seed(seed + 10007)
            eval_acc = _eval_one_lm(
                model, task, eval_rng, batch_size, n_eval_batches, device=device
            )
            results[step] = HardBindingResult(
                task_name=task.name,
                mixer_label=mixer_label,
                train_loss_initial=initial_loss,
                train_loss_final=float(loss.item()),
                train_accuracy_final=train_acc,
                eval_accuracy=eval_acc,
                chance_accuracy=chance,
                converged=True,
                n_params=n_params,
            )
    except Exception:  # noqa: BLE001 - caller fills missing checkpoints
        pass
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
