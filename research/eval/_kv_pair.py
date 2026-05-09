"""Shared associative key-value retrieval primitives.

Used by `ar_intermediate_probe` and `ar_validation` (and the test suite).
The two probes only differ in budget/scoring; the batch construction,
evaluation, and training step are identical.
"""

from __future__ import annotations

from dataclasses import dataclass

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
