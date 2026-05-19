"""Shared staged associative-recall curriculum primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import clip_grad_norm

StageConfig = Mapping[str, int]

VOCAB_LO = 1000

STAGE_CONFIGS_PROBE: tuple[dict[str, int], ...] = (
    {
        "n_keys": 8,
        "n_values": 4,
        "pairs_per_example": 1,
        "n_train_pairs": 3,
        "n_held_pairs": 1,
    },
    {
        "n_keys": 16,
        "n_values": 6,
        "pairs_per_example": 2,
        "n_train_pairs": 6,
        "n_held_pairs": 2,
    },
    {
        "n_keys": 32,
        "n_values": 8,
        "pairs_per_example": 2,
        "n_train_pairs": 8,
        "n_held_pairs": 4,
    },
    {
        "n_keys": 64,
        "n_values": 12,
        "pairs_per_example": 2,
        "n_train_pairs": 16,
        "n_held_pairs": 8,
    },
    {
        "n_keys": 128,
        "n_values": 16,
        "pairs_per_example": 3,
        "n_train_pairs": 32,
        "n_held_pairs": 16,
    },
    {
        "n_keys": 256,
        "n_values": 24,
        "pairs_per_example": 4,
        "n_train_pairs": 64,
        "n_held_pairs": 24,
    },
)

STAGE_CONFIGS_DEFAULT: tuple[dict[str, int], ...] = (
    dict(STAGE_CONFIGS_PROBE[2]),
    dict(STAGE_CONFIGS_PROBE[3]),
    dict(STAGE_CONFIGS_PROBE[4]),
    dict(STAGE_CONFIGS_PROBE[5]),
    {
        "n_keys": 512,
        "n_values": 32,
        "pairs_per_example": 6,
        "n_train_pairs": 96,
        "n_held_pairs": 32,
    },
    {
        "n_keys": 1024,
        "n_values": 48,
        "pairs_per_example": 9,
        "n_train_pairs": 128,
        "n_held_pairs": 48,
    },
)

STAGE_CONFIGS_FINE: tuple[dict[str, int], ...] = (
    *(dict(cfg) for cfg in STAGE_CONFIGS_PROBE),
    dict(STAGE_CONFIGS_DEFAULT[4]),
    {
        "n_keys": 768,
        "n_values": 40,
        "pairs_per_example": 7,
        "n_train_pairs": 112,
        "n_held_pairs": 40,
    },
    dict(STAGE_CONFIGS_DEFAULT[5]),
)


@dataclass(frozen=True, slots=True)
class StageSpec:
    stage_idx: int
    n_key_tokens: int
    n_value_tokens: int
    pairs_per_example: int
    n_train_pairs: int
    n_held_pairs: int
    train_keys: torch.Tensor
    train_values: torch.Tensor
    held_keys: torch.Tensor
    held_values: torch.Tensor
    value_lo: int
    value_hi: int


def build_stage_specs(
    seed: int,
    stage_configs: tuple[StageConfig, ...] = STAGE_CONFIGS_DEFAULT,
    *,
    vocab_lo: int = VOCAB_LO,
    device: torch.device | None = None,
) -> list[StageSpec]:
    """Allocate disjoint token ranges per stage and build deterministic pair tables."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    cursor = int(vocab_lo)
    stages: list[StageSpec] = []
    for i, cfg in enumerate(stage_configs):
        key_lo = cursor
        cursor += int(cfg["n_keys"])
        value_lo = cursor
        cursor += int(cfg["n_values"])

        total = int(cfg["n_train_pairs"]) + int(cfg["n_held_pairs"])
        if int(cfg["n_keys"]) < total * 2:
            raise ValueError(
                f"stage {i}: n_keys={cfg['n_keys']} too small for {total} pairs"
            )
        key_perm = torch.randperm(int(cfg["n_keys"]), generator=gen)[: total * 2]
        key_pairs = key_perm.reshape(total, 2) + key_lo

        value_offsets = torch.arange(total, dtype=torch.long) % int(cfg["n_values"])
        value_offsets = value_offsets[torch.randperm(total, generator=gen)]
        values = value_lo + value_offsets

        n_train = int(cfg["n_train_pairs"])
        stage = StageSpec(
            stage_idx=i,
            n_key_tokens=int(cfg["n_keys"]),
            n_value_tokens=int(cfg["n_values"]),
            pairs_per_example=int(cfg["pairs_per_example"]),
            n_train_pairs=int(cfg["n_train_pairs"]),
            n_held_pairs=int(cfg["n_held_pairs"]),
            train_keys=key_pairs[:n_train].contiguous(),
            train_values=values[:n_train].contiguous(),
            held_keys=key_pairs[n_train:].contiguous(),
            held_values=values[n_train:].contiguous(),
            value_lo=value_lo,
            value_hi=value_lo + int(cfg["n_values"]),
        )
        stages.append(stage_to_device(stage, device) if device is not None else stage)
    return stages


def stage_to_device(stage: StageSpec, device: torch.device) -> StageSpec:
    return StageSpec(
        stage_idx=stage.stage_idx,
        n_key_tokens=stage.n_key_tokens,
        n_value_tokens=stage.n_value_tokens,
        pairs_per_example=stage.pairs_per_example,
        n_train_pairs=stage.n_train_pairs,
        n_held_pairs=stage.n_held_pairs,
        train_keys=stage.train_keys.to(device),
        train_values=stage.train_values.to(device),
        held_keys=stage.held_keys.to(device),
        held_values=stage.held_values.to(device),
        value_lo=stage.value_lo,
        value_hi=stage.value_hi,
    )


def stage_value_classes(
    values: torch.Tensor, stage: StageSpec, n_classes: int = 4
) -> torch.Tensor:
    return (values - int(stage.value_lo)).remainder(int(n_classes))


def make_stage_batch(
    stage: StageSpec,
    *,
    split: str,
    batch_size: int,
    sep_token: int,
    ans_token: int,
    device: torch.device,
    generator: torch.Generator,
    episodic_values: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one staged associative-recall batch."""
    if split == "train":
        keys = stage.train_keys
        values = stage.train_values
    elif split == "held":
        keys = stage.held_keys
        values = stage.held_values
    else:
        raise ValueError(f"split must be 'train' or 'held' (got {split!r})")
    if keys.numel() == 0:
        raise ValueError(f"stage {stage.stage_idx} {split} split has no pairs")

    batch = int(batch_size)
    n_pairs = int(stage.pairs_per_example)
    value_span = int(stage.value_hi - stage.value_lo)
    if episodic_values and value_span < n_pairs:
        raise ValueError(
            f"stage {stage.stage_idx}: value span {value_span} < pairs_per_example {n_pairs}"
        )

    q_idx = torch.randint(
        0, keys.shape[0], (batch,), device=device, generator=generator
    )
    d_idx = torch.randint(
        0,
        stage.train_keys.shape[0],
        (batch, n_pairs - 1),
        device=device,
        generator=generator,
    )

    ex_keys = torch.empty((batch, n_pairs, 2), dtype=torch.long, device=device)
    ex_values = torch.empty((batch, n_pairs), dtype=torch.long, device=device)
    ex_keys[:, 0, :] = keys.index_select(0, q_idx)
    flat_d = d_idx.reshape(-1)
    ex_keys[:, 1:, :] = stage.train_keys.index_select(0, flat_d).reshape(
        batch, n_pairs - 1, 2
    )
    if episodic_values:
        scores = torch.rand((batch, value_span), device=device, generator=generator)
        order = torch.argsort(scores, dim=1)[:, :n_pairs]
        ex_values[:, :] = order + int(stage.value_lo)
    else:
        ex_values[:, 0] = values.index_select(0, q_idx)
        ex_values[:, 1:] = stage.train_values.index_select(0, flat_d).reshape(
            batch, n_pairs - 1
        )

    shuffle = torch.argsort(
        torch.rand((batch, n_pairs), device=device, generator=generator), dim=1
    )
    ex_keys = ex_keys.gather(1, shuffle.unsqueeze(-1).expand(-1, -1, 2))
    ex_values = ex_values.gather(1, shuffle)

    seq_len = 3 * n_pairs + 4
    ids = torch.empty((batch, seq_len), dtype=torch.long, device=device)
    pair_pos = torch.arange(n_pairs, device=device)
    ids[:, pair_pos * 3] = ex_keys[:, :, 0]
    ids[:, pair_pos * 3 + 1] = ex_keys[:, :, 1]
    ids[:, pair_pos * 3 + 2] = ex_values
    sep_pos = 3 * n_pairs
    ids[:, sep_pos] = int(sep_token)
    ids[:, sep_pos + 1] = keys[q_idx, 0]
    ids[:, sep_pos + 2] = keys[q_idx, 1]
    ids[:, sep_pos + 3] = int(ans_token)

    query_pos = (shuffle == 0).to(torch.long).argmax(dim=1)
    targets = ex_values.gather(1, query_pos.unsqueeze(1)).squeeze(1)
    return ids, targets


def train_stage_one_batch(
    model: nn.Module,
    ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    opt: torch.optim.Optimizer,
    stage: StageSpec,
    ans_pos: int,
) -> float | None:
    opt.zero_grad(set_to_none=True)
    logits = model(ids)
    pred = logits[:, ans_pos, stage.value_lo : stage.value_hi].float()
    loss = F.cross_entropy(pred, targets - int(stage.value_lo))
    if not torch.isfinite(loss):
        return None
    loss.backward()
    clip_grad_norm(model.parameters(), 1.0)
    opt.step()
    return float(loss.detach().item())


@torch.no_grad()
def evaluate_stage(
    model: nn.Module,
    stage: StageSpec,
    *,
    sep_token: int,
    ans_token: int,
    device: torch.device,
    seed: int,
    eval_batches: int,
    batch_size: int,
    episodic_values: bool = True,
) -> tuple[float, float]:
    model.eval()
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    n_pairs = int(stage.pairs_per_example)
    ans_pos = 3 * n_pairs + 3
    pair_correct = class_correct = total = 0
    for _ in range(eval_batches):
        ids, targets = make_stage_batch(
            stage,
            split="held",
            batch_size=batch_size,
            sep_token=sep_token,
            ans_token=ans_token,
            device=device,
            generator=gen,
            episodic_values=episodic_values,
        )
        logits = model(ids)
        pred = logits[:, ans_pos, stage.value_lo : stage.value_hi].argmax(dim=-1)
        pred = pred + int(stage.value_lo)
        pair_correct += int((pred == targets).sum().item())
        class_correct += int(
            (stage_value_classes(pred, stage) == stage_value_classes(targets, stage))
            .sum()
            .item()
        )
        total += int(targets.shape[0])
    return pair_correct / max(total, 1), class_correct / max(total, 1)


def chance_class_acc(stage: StageSpec, n_classes: int = 4) -> float:
    return 1.0 / float(n_classes)


def chance_pair_acc(stage: StageSpec) -> float:
    return 1.0 / float(stage.n_value_tokens)


def required_vocab_size_for_stage_configs(
    stage_configs: tuple[StageConfig, ...],
    *,
    vocab_lo: int = VOCAB_LO,
    extra_special_tokens: int = 2,
) -> int:
    span = sum(int(c["n_keys"]) + int(c["n_values"]) for c in stage_configs)
    return int(vocab_lo) + span + int(extra_special_tokens)
