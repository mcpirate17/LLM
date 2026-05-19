# pyright: reportPrivateImportUsage=false
"""Nano-induction probe — soft signal for 2-hop content-addressable retrieval.

Companion to ``nano_bind_probe`` / AR-Gate, but tests the canonical
**induction circuit**: at position L-1, retrieve what followed an earlier
occurrence of the cue token. Unlike binding (1-hop key → label), induction
requires composing two hops: match-on-content, then read-the-successor.

Per-example layout (synthetic ``[B, L, D]`` continuous setting):

    pos:  0    ..   p1   p1+1   ..   p2     ..   L-1
    val:  rand ..   K_c  V_c    rand K_c    rand SLOT(=0)

where ``K_c`` and ``V_c`` are per-class key/value vectors (resampled per
``seed``). A linear head reads ``features[:, -1, :]`` and predicts ``c``
from ``K`` classes.

The lane is expected to be wrapped in **two stacked ``LaneTestBlock``s**
(see the validator) because the induction circuit fundamentally requires
depth — one layer to do content-match, one to read-and-propagate. A
1-layer lane cannot pass; that's a known structural fact, not a flaw
of the lane.

Semantics: **soft** signal. Records ``max_accuracy`` and a boolean
``above_baseline`` (max ≥ random + margin). The validator never
hard-rejects on induction — pure WTA architectures are expected to score
low here, and we want their score recorded, not zeroed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class NanoInductionResult:
    accuracies: tuple[float, ...]
    max_accuracy: float
    final_accuracy: float
    random_baseline: float
    above_baseline: bool
    margin: float
    notes: tuple[str, ...] = field(default_factory=tuple)


def _make_class_vectors(
    n_classes: int, dim: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-class (key, value) vectors, scaled to dominate random distractors."""
    keys = torch.randn(n_classes, dim, generator=generator) * 2.0
    values = torch.randn(n_classes, dim, generator=generator) * 2.0
    return keys, values


def _sample_induction_batch(
    batch_size: int,
    seq_len: int,
    dim: int,
    n_classes: int,
    *,
    class_keys: torch.Tensor,
    class_values: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One batch of the induction layout (see module docstring)."""
    if seq_len < 8:
        raise ValueError("seq_len must be >= 8 for the induction layout")
    labels = torch.randint(0, n_classes, (batch_size,), generator=generator)
    x = torch.randn(batch_size, seq_len, dim, generator=generator)

    q1_hi = max(1, seq_len // 4 - 1)
    q3_lo = max(seq_len // 2 + 1, 3 * seq_len // 4)
    q3_hi = seq_len - 2
    if q3_lo >= q3_hi:
        q3_lo = q3_hi - 1
    p1 = torch.randint(0, q1_hi, (batch_size,), generator=generator)
    p2 = torch.randint(q3_lo, q3_hi, (batch_size,), generator=generator)

    rows = torch.arange(batch_size)
    x[rows, p1] = class_keys[labels]
    x[rows, p1 + 1] = class_values[labels]
    x[rows, p2] = class_keys[labels]
    x[:, -1] = 0.0
    return x, labels


def _step_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        return float((logits.argmax(dim=-1) == labels).float().mean().item())


def _train_induction(
    lane_block: nn.Module,
    head: nn.Linear,
    *,
    dim: int,
    seq_len: int,
    n_classes: int,
    n_train_steps: int,
    checkpoint_at_steps: tuple[int, ...],
    learning_rate: float,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[list[float], float]:
    """Run the training loop; return (accuracy_at_checkpoints, final_acc)."""
    optimizer = torch.optim.Adam(
        list(lane_block.parameters()) + list(head.parameters()),
        lr=learning_rate,
    )
    class_keys, class_values = _make_class_vectors(n_classes, dim, generator)
    accuracies: list[float] = []
    final_acc = 0.0
    lane_block.train()
    for step in range(1, n_train_steps + 1):
        x, labels = _sample_induction_batch(
            batch_size,
            seq_len,
            dim,
            n_classes,
            class_keys=class_keys,
            class_values=class_values,
            generator=generator,
        )
        logits = head(lane_block(x)[:, -1, :])
        loss = nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in checkpoint_at_steps:
            final_acc = _step_accuracy(logits, labels)
            accuracies.append(final_acc)
    lane_block.eval()
    return accuracies, final_acc


def nano_induction_gate(
    lane_block: nn.Module,
    *,
    dim: int = 32,
    seq_len: int = 24,
    n_classes: int = 8,
    n_train_steps: int = 150,
    checkpoint_at_steps: tuple[int, ...] = (50, 100, 150),
    learning_rate: float = 3e-3,
    batch_size: int = 16,
    margin: float = 0.05,
    seed: int = 0,
) -> NanoInductionResult:
    """Train ``lane_block`` + linear head briefly on the induction task.

    Soft scorecard only. ``above_baseline`` flips True when the max
    checkpoint accuracy exceeds ``1/n_classes + margin``.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    head = nn.Linear(dim, n_classes)
    baseline = 1.0 / float(n_classes)
    try:
        accuracies, final_acc = _train_induction(
            lane_block,
            head,
            dim=dim,
            seq_len=seq_len,
            n_classes=n_classes,
            n_train_steps=n_train_steps,
            checkpoint_at_steps=checkpoint_at_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            generator=generator,
        )
    except Exception as exc:  # noqa: BLE001
        return NanoInductionResult(
            accuracies=(),
            max_accuracy=0.0,
            final_accuracy=0.0,
            random_baseline=baseline,
            above_baseline=False,
            margin=margin,
            notes=(f"{type(exc).__name__}: {exc}",),
        )

    max_acc = max(accuracies) if accuracies else 0.0
    return NanoInductionResult(
        accuracies=tuple(accuracies),
        max_accuracy=max_acc,
        final_accuracy=final_acc,
        random_baseline=baseline,
        above_baseline=max_acc >= baseline + margin,
        margin=margin,
    )
