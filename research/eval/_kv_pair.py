"""Shared associative key-value retrieval primitives.

Used by `ar_intermediate_probe` and `ar_validation` (and the test suite).
The two probes only differ in budget/scoring; the batch construction,
evaluation, and training step are identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import clip_grad_norm


@dataclass(frozen=True, slots=True)
class KVPairTable:
    train_keys: torch.Tensor
    train_values: torch.Tensor
    held_keys: torch.Tensor
    held_values: torch.Tensor
    vocab_lo: int
    value_lo: int
    value_hi: int
    n_value_classes: int

    @property
    def total_token_span(self) -> int:
        return int(self.value_hi - self.vocab_lo)


@dataclass(frozen=True, slots=True)
class KVProbeRuntime:
    n_eval: int
    batch_size: int
    pairs_per_example: int
    sep_token: int
    ans_token: int
    device: torch.device
    episodic_values: bool


@dataclass(frozen=True, slots=True)
class KVTrainingLoopResult:
    steps_done: int
    status: str
    error: str | None = None


def build_kv_pair_table(
    *,
    seed: int,
    vocab_lo: int,
    n_key_tokens: int,
    n_value_tokens: int,
    n_value_classes: int,
    n_train_pairs: int,
    n_held_pairs: int,
    pairs_per_example: int | None = None,
) -> KVPairTable:
    """Build deterministic disjoint train/held key/value pairs."""
    total_pairs = int(n_train_pairs) + int(n_held_pairs)
    if total_pairs <= 0:
        raise ValueError("at least one train or held pair is required")
    if int(n_key_tokens) < total_pairs * 2:
        raise ValueError("n_key_tokens must provide two unique tokens per pair")
    if int(n_value_tokens) <= 0:
        raise ValueError("n_value_tokens must be positive")
    if int(n_value_classes) <= 0 or int(n_value_classes) > int(n_value_tokens):
        raise ValueError("n_value_classes must be in [1, n_value_tokens]")
    if pairs_per_example is not None and int(pairs_per_example) < 2:
        raise ValueError("pairs_per_example must be at least 2")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    key_tokens = torch.randperm(int(n_key_tokens), generator=gen)[: total_pairs * 2]
    key_tokens = key_tokens.reshape(total_pairs, 2) + int(vocab_lo)

    value_lo = int(vocab_lo) + int(n_key_tokens)
    value_offsets = torch.arange(total_pairs, dtype=torch.long) % int(n_value_tokens)
    value_offsets = value_offsets[torch.randperm(total_pairs, generator=gen)]
    values = value_lo + value_offsets

    n_train = int(n_train_pairs)
    return KVPairTable(
        train_keys=key_tokens[:n_train].contiguous(),
        train_values=values[:n_train].contiguous(),
        held_keys=key_tokens[n_train:].contiguous(),
        held_values=values[n_train:].contiguous(),
        vocab_lo=int(vocab_lo),
        value_lo=value_lo,
        value_hi=value_lo + int(n_value_tokens),
        n_value_classes=int(n_value_classes),
    )


def kv_table_to_device(table: KVPairTable, device: torch.device) -> KVPairTable:
    return KVPairTable(
        train_keys=table.train_keys.to(device),
        train_values=table.train_values.to(device),
        held_keys=table.held_keys.to(device),
        held_values=table.held_values.to(device),
        vocab_lo=table.vocab_lo,
        value_lo=table.value_lo,
        value_hi=table.value_hi,
        n_value_classes=table.n_value_classes,
    )


def kv_value_classes(values: torch.Tensor, table: KVPairTable) -> torch.Tensor:
    return (values - int(table.value_lo)).remainder(int(table.n_value_classes))


def make_kv_pair_batch(
    table: KVPairTable,
    *,
    split: str,
    batch_size: int,
    pairs_per_example: int,
    sep_token: int,
    ans_token: int,
    device: torch.device,
    generator: torch.Generator,
    episodic_values: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate one vectorized AR pair-retrieval batch.

    Sequence shape: ``[N pairs of (k0,k1,v) || SEP || query_k0 query_k1 ANS]``,
    pairs in a per-row random order. Returns ``(ids, value_targets, value_class_targets)``.
    """
    if split not in {"train", "held"}:
        raise ValueError("split must be 'train' or 'held'")
    if int(pairs_per_example) < 2:
        raise ValueError("pairs_per_example must be at least 2")

    query_keys = table.train_keys if split == "train" else table.held_keys
    query_values = table.train_values if split == "train" else table.held_values
    if query_keys.numel() == 0:
        raise ValueError(f"{split} split has no pairs")

    batch = int(batch_size)
    n_pairs = int(pairs_per_example)
    value_span = int(table.value_hi - table.value_lo)
    if episodic_values and value_span < n_pairs:
        raise ValueError("episodic value span must cover pairs_per_example")

    q_idx = torch.randint(
        0, query_keys.shape[0], (batch,), device=device, generator=generator
    )
    d_idx = torch.randint(
        0,
        table.train_keys.shape[0],
        (batch, n_pairs - 1),
        device=device,
        generator=generator,
    )

    keys = torch.empty((batch, n_pairs, 2), dtype=torch.long, device=device)
    values = torch.empty((batch, n_pairs), dtype=torch.long, device=device)
    keys[:, 0, :] = query_keys.index_select(0, q_idx)
    flat_d = d_idx.reshape(-1)
    keys[:, 1:, :] = table.train_keys.index_select(0, flat_d).reshape(
        batch, n_pairs - 1, 2
    )
    if episodic_values:
        value_order = torch.argsort(
            torch.rand((batch, value_span), device=device, generator=generator),
            dim=1,
        )[:, :n_pairs]
        values[:, :] = value_order + int(table.value_lo)
    else:
        values[:, 0] = query_values.index_select(0, q_idx)
        values[:, 1:] = table.train_values.index_select(0, flat_d).reshape(
            batch, n_pairs - 1
        )

    order = torch.argsort(
        torch.rand((batch, n_pairs), device=device, generator=generator), dim=1
    )
    keys = keys.gather(1, order.unsqueeze(-1).expand(-1, -1, 2))
    values = values.gather(1, order)

    seq_len = 3 * n_pairs + 4
    ids = torch.empty((batch, seq_len), dtype=torch.long, device=device)
    pair_pos = torch.arange(n_pairs, device=device)
    ids[:, pair_pos * 3] = keys[:, :, 0]
    ids[:, pair_pos * 3 + 1] = keys[:, :, 1]
    ids[:, pair_pos * 3 + 2] = values
    sep_pos = 3 * n_pairs
    ids[:, sep_pos] = int(sep_token)
    ids[:, sep_pos + 1] = query_keys[q_idx, 0]
    ids[:, sep_pos + 2] = query_keys[q_idx, 1]
    ids[:, sep_pos + 3] = int(ans_token)
    query_pos = (order == 0).to(torch.long).argmax(dim=1)
    targets = values.gather(1, query_pos.unsqueeze(1)).squeeze(1)
    return ids, targets, kv_value_classes(targets, table)


@torch.no_grad()
def evaluate_kv_split(
    model: nn.Module,
    table: KVPairTable,
    *,
    split: str,
    n_eval: int,
    batch_size: int,
    pairs_per_example: int,
    sep_token: int,
    ans_token: int,
    device: torch.device,
    seed: int,
    episodic_values: bool,
) -> tuple[float, float]:
    """Average exact-pair and value-class accuracy.

    Counts accumulate as on-device tensors and sync once at the end of the
    eval — avoids per-batch ``.item()`` calls that would force a GPU stall.
    """
    model.eval()
    exact_total = torch.zeros((), dtype=torch.long, device=device)
    class_total = torch.zeros((), dtype=torch.long, device=device)
    total = 0
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    ans_pos = 3 * int(pairs_per_example) + 3
    remaining = int(n_eval)
    while remaining > 0:
        bs = min(int(batch_size), remaining)
        ids, targets, target_classes = make_kv_pair_batch(
            table,
            split=split,
            batch_size=bs,
            pairs_per_example=pairs_per_example,
            sep_token=sep_token,
            ans_token=ans_token,
            device=device,
            generator=gen,
            episodic_values=episodic_values,
        )
        logits = model(ids)
        pred = logits[:, ans_pos, table.value_lo : table.value_hi].argmax(dim=-1)
        pred = pred + int(table.value_lo)
        exact_total += (pred == targets).sum()
        class_total += (kv_value_classes(pred, table) == target_classes).sum()
        total += bs
        remaining -= bs
    denom = max(total, 1)
    return int(exact_total.item()) / denom, int(class_total.item()) / denom


def evaluate_kv_train_and_held(
    model: nn.Module,
    table: KVPairTable,
    *,
    n_eval: int,
    batch_size: int,
    pairs_per_example: int,
    sep_token: int,
    ans_token: int,
    device: torch.device,
    train_seed: int,
    held_seed: int,
    episodic_values: bool,
) -> tuple[float, float, float]:
    train_acc, _train_class = evaluate_kv_split(
        model,
        table,
        split="train",
        n_eval=n_eval,
        batch_size=batch_size,
        pairs_per_example=pairs_per_example,
        sep_token=sep_token,
        ans_token=ans_token,
        device=device,
        seed=train_seed,
        episodic_values=episodic_values,
    )
    held_pair, held_class = evaluate_kv_split(
        model,
        table,
        split="held",
        n_eval=n_eval,
        batch_size=batch_size,
        pairs_per_example=pairs_per_example,
        sep_token=sep_token,
        ans_token=ans_token,
        device=device,
        seed=held_seed,
        episodic_values=episodic_values,
    )
    return train_acc, held_pair, held_class


def evaluate_kv_probe_checkpoint(
    model: nn.Module,
    table: KVPairTable,
    *,
    runtime: KVProbeRuntime,
    base_seed: int,
    step: int | None = None,
) -> tuple[float, float, float]:
    if step is None:
        train_seed = int(base_seed) + 30_000
        held_seed = int(base_seed) + 40_000
    else:
        train_seed = int(base_seed) + 10_000 + int(step)
        held_seed = int(base_seed) + 20_000 + int(step)
    return evaluate_kv_train_and_held(
        model,
        table,
        n_eval=runtime.n_eval,
        batch_size=runtime.batch_size,
        pairs_per_example=runtime.pairs_per_example,
        sep_token=runtime.sep_token,
        ans_token=runtime.ans_token,
        device=runtime.device,
        train_seed=train_seed,
        held_seed=held_seed,
        episodic_values=runtime.episodic_values,
    )


def train_kv_one_batch(
    model: nn.Module,
    ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    opt: torch.optim.Optimizer,
    table: KVPairTable,
    ans_pos: int,
) -> torch.Tensor | None:
    """One backward step over ``cross_entropy(value_logits, targets)``.

    Returns the detached on-device loss tensor (callers ``.item()`` only at
    eval-every boundaries) or ``None`` when the loss is non-finite.
    """
    opt.zero_grad(set_to_none=True)
    logits = model(ids)
    pred = logits[:, ans_pos, table.value_lo : table.value_hi].float()
    loss = F.cross_entropy(pred, targets - int(table.value_lo))
    if not torch.isfinite(loss):
        return None
    loss.backward()
    clip_grad_norm(model.parameters(), 1.0)
    opt.step()
    return loss.detach()


def train_kv_probe_step(
    model: nn.Module,
    table: KVPairTable,
    *,
    runtime: KVProbeRuntime,
    generator: torch.Generator,
    opt: torch.optim.Optimizer,
    ans_pos: int,
) -> torch.Tensor | None:
    ids, targets, _classes = make_kv_pair_batch(
        table,
        split="train",
        batch_size=runtime.batch_size,
        pairs_per_example=runtime.pairs_per_example,
        sep_token=runtime.sep_token,
        ans_token=runtime.ans_token,
        device=runtime.device,
        generator=generator,
        episodic_values=runtime.episodic_values,
    )
    return train_kv_one_batch(
        model,
        ids,
        targets,
        opt=opt,
        table=table,
        ans_pos=ans_pos,
    )


def run_kv_probe_training_loop(
    model: nn.Module,
    table: KVPairTable,
    *,
    runtime: KVProbeRuntime,
    generator: torch.Generator,
    opt: torch.optim.Optimizer,
    ans_pos: int,
    train_steps: int,
    eval_every: int,
    deadline: float,
    base_seed: int,
    monotonic_time: Callable[[], float],
    on_eval: Callable[[int, torch.Tensor, float, float, float], None],
) -> KVTrainingLoopResult:
    steps_done = 0
    for step in range(1, int(train_steps) + 1):
        if monotonic_time() > deadline:
            return KVTrainingLoopResult(steps_done=steps_done, status="timeout")
        loss = train_kv_probe_step(
            model,
            table,
            runtime=runtime,
            generator=generator,
            opt=opt,
            ans_pos=ans_pos,
        )
        if loss is None:
            return KVTrainingLoopResult(
                steps_done=steps_done,
                status="error",
                error="non_finite_loss",
            )
        steps_done = step
        if step % int(eval_every) == 0 or step == int(train_steps):
            train_acc, held_pair, held_class = evaluate_kv_probe_checkpoint(
                model,
                table,
                runtime=runtime,
                base_seed=base_seed,
                step=step,
            )
            on_eval(step, loss, train_acc, held_pair, held_class)
    return KVTrainingLoopResult(steps_done=steps_done, status="ok")
