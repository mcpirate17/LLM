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


@dataclass(frozen=True, slots=True)
class NanoInductionNearestResult:
    accuracies: tuple[float, ...]
    max_accuracy: float | None
    final_accuracy: float | None
    random_baseline: float
    status: str
    train_steps: int
    protocol_version: str
    elapsed_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "accuracies": list(self.accuracies),
            "max_accuracy": self.max_accuracy,
            "final_accuracy": self.final_accuracy,
            "random_baseline": self.random_baseline,
            "status": self.status,
            "train_steps": self.train_steps,
            "protocol_version": self.protocol_version,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


NANO_INDUCTION_NEAREST_PROTOCOL_VERSION = "nano_induction_nearest_v1_steps120"


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


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _make_nearest_batch(
    batch_size: int,
    seq_len: int,
    dim: int,
    n_keys: int,
    n_values: int,
    *,
    key_vectors: torch.Tensor,
    value_vectors: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if seq_len < 12:
        raise ValueError("seq_len must be >= 12 for nearest induction")
    if n_keys < 2 or n_values < 2:
        raise ValueError("n_keys and n_values must both be >= 2")

    keys = torch.randint(
        0, n_keys, (batch_size, seq_len), generator=generator, device=device
    )
    values = torch.randint(
        0, n_values, (batch_size, seq_len), generator=generator, device=device
    )
    query_keys = torch.randint(
        0, n_keys, (batch_size,), generator=generator, device=device
    )
    nearest_values = torch.randint(
        0, n_values, (batch_size,), generator=generator, device=device
    )
    older_values = torch.randint(
        0, n_values, (batch_size,), generator=generator, device=device
    )

    rows = torch.arange(batch_size, device=device)
    older_pos = torch.randint(
        1, max(2, seq_len // 3), (batch_size,), generator=generator, device=device
    )
    nearest_pos = torch.randint(
        max(3, seq_len // 2),
        seq_len - 2,
        (batch_size,),
        generator=generator,
        device=device,
    )
    keys[rows, older_pos] = query_keys
    values[rows, older_pos] = older_values
    keys[rows, nearest_pos] = query_keys
    values[rows, nearest_pos] = nearest_values
    keys[:, -1] = query_keys

    x = key_vectors[keys] + value_vectors[values]
    x[:, -1, :] = key_vectors[query_keys]
    x = x + 0.02 * torch.randn(
        batch_size, seq_len, dim, generator=generator, device=device
    )
    return x, nearest_values


def _nearest_accuracy(
    body: nn.Module,
    head: nn.Linear,
    *,
    dim: int,
    seq_len: int,
    n_keys: int,
    n_values: int,
    eval_batch: int,
    key_vectors: torch.Tensor,
    value_vectors: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    was_training = body.training
    body.eval()
    head.eval()
    with torch.no_grad():
        x, labels = _make_nearest_batch(
            eval_batch,
            seq_len,
            dim,
            n_keys,
            n_values,
            key_vectors=key_vectors,
            value_vectors=value_vectors,
            generator=generator,
            device=device,
        )
        logits = head(body(x)[:, -1, :])
        acc = _step_accuracy(logits, labels)
    body.train(was_training)
    head.train(was_training)
    return acc


def nano_induction_nearest(
    body: nn.Module,
    *,
    dim: int,
    seq_len: int = 40,
    n_keys: int = 16,
    n_values: int = 16,
    n_train_steps: int = 120,
    checkpoint_at_steps: tuple[int, ...] = (60, 120),
    learning_rate: float = 3e-3,
    batch_size: int = 12,
    eval_batch: int = 96,
    seed: int = 0,
) -> NanoInductionNearestResult:
    """Train a continuous graph on nearest-matching-key value retrieval.

    ``body`` must map ``(B, S, dim)`` continuous embeddings to the same shape.
    Metric values are populated only for clean ``status == "ok"`` runs; failed
    runs carry status/error so downstream persistence does not invent zeros.
    """
    import time

    t0 = time.perf_counter()
    device = _module_device(body)
    try:
        generator = torch.Generator(device=device).manual_seed(seed)
    except RuntimeError:
        generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    baseline = 1.0 / float(n_values)
    head = nn.Linear(dim, n_values).to(device)
    body = body.to(device)
    key_vectors = torch.randn(n_keys, dim, generator=generator, device=device) * 2.0
    value_vectors = torch.randn(n_values, dim, generator=generator, device=device) * 2.0
    optimizer = torch.optim.Adam(
        list(body.parameters()) + list(head.parameters()), lr=learning_rate
    )
    accuracies: list[float] = []
    status = "ok"
    error = None
    try:
        body.train()
        head.train()
        for step in range(1, int(n_train_steps) + 1):
            x, labels = _make_nearest_batch(
                batch_size,
                seq_len,
                dim,
                n_keys,
                n_values,
                key_vectors=key_vectors,
                value_vectors=value_vectors,
                generator=generator,
                device=device,
            )
            logits = head(body(x)[:, -1, :])
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                status = "nonfinite_loss"
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step in checkpoint_at_steps:
                accuracies.append(
                    round(
                        _nearest_accuracy(
                            body,
                            head,
                            dim=dim,
                            seq_len=seq_len,
                            n_keys=n_keys,
                            n_values=n_values,
                            eval_batch=eval_batch,
                            key_vectors=key_vectors,
                            value_vectors=value_vectors,
                            generator=generator,
                            device=device,
                        ),
                        4,
                    )
                )
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    ok = status == "ok" and bool(accuracies)
    final_acc = accuracies[-1] if ok else None
    max_acc = max(accuracies) if ok else None
    return NanoInductionNearestResult(
        accuracies=tuple(accuracies),
        max_accuracy=max_acc,
        final_accuracy=final_acc,
        random_baseline=baseline,
        status=status,
        train_steps=int(n_train_steps),
        protocol_version=NANO_INDUCTION_NEAREST_PROTOCOL_VERSION,
        elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 3),
        error=error,
    )
