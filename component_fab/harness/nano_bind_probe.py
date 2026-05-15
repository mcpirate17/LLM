"""AR Gate / NanoBind 0.5 — S0.5 binding-test hard rejection gate.

Adapted from ``research/eval/nano_bind.py`` (and the S0.5-tier
``research/eval/ar_gate.py`` which reuses the same persistent-zero
semantic). The research probes reject architectures whose slot-ending
accuracy is **at-or-below random across every checkpoint** of a short
training sweep — frequency-mode-collapse degenerates.

Adapted to fab's ``[B, L, D]`` continuous setting: a synthetic K-class
binding task where each input has a "key prefix" at position 0 and the
lane must propagate that signal to the binding slot at position -1
where a small classification head reads it.

No-go semantic: persistent-at-baseline accuracy ⇒ reject. Passing means
"eligible to continue evaluation" — not a positive ranking signal
(matches the original probe's intent).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class NanoBindResult:
    accuracies: tuple[float, ...]
    max_accuracy: float
    random_baseline: float
    rejected_persistent_zero: bool
    passed: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


class _BindingHead(nn.Module):
    """Tiny classification head — projects the binding-slot output to K logits."""

    def __init__(self, dim: int, n_classes: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _sample_binding_batch(
    batch_size: int,
    seq_len: int,
    dim: int,
    n_classes: int,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each example: random tokens with a per-class "key prefix" embedded at position 0.

    Output target is a class index; lane must propagate the key signal
    from position 0 to the binding slot (position -1).
    """
    class_keys = torch.randn(n_classes, dim, generator=generator) * 2.0
    labels = torch.randint(0, n_classes, (batch_size,), generator=generator)
    x = torch.randn(batch_size, seq_len, dim, generator=generator)
    x[:, 0, :] = class_keys[labels]
    return x, labels


def nano_bind_gate(
    lane_block: nn.Module,
    *,
    dim: int = 32,
    seq_len: int = 16,
    n_classes: int = 4,
    n_train_steps: int = 60,
    checkpoint_at_steps: tuple[int, ...] = (20, 40, 60),
    learning_rate: float = 3e-3,
    batch_size: int = 16,
    seed: int = 0,
) -> NanoBindResult:
    """Train ``lane_block`` + a small head briefly on the binding task.

    Returns ``rejected_persistent_zero`` when accuracy at every checkpoint
    is at-or-below the random baseline by more than a tiny margin — that's
    the hard no-go signal. ``passed`` is the inverse (eligibility, not
    quality).
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    head = _BindingHead(dim, n_classes)
    optimizer = torch.optim.Adam(
        list(lane_block.parameters()) + list(head.parameters()),
        lr=learning_rate,
    )
    random_baseline = 1.0 / float(n_classes)
    # Tight margin: a primitive that can't beat random+epsilon on a tiny
    # K-class binding task fails the persistent-zero check. The original
    # nano_bind used "exactly 0.00" — we use "no checkpoint exceeded random
    # by more than epsilon" which is the equivalent for continuous-MSE setting.
    margin = 0.01
    accuracies: list[float] = []
    try:
        lane_block.train()
        for step in range(1, n_train_steps + 1):
            x, labels = _sample_binding_batch(
                batch_size,
                seq_len,
                dim,
                n_classes,
                generator=generator,
            )
            features = lane_block(x)
            logits = head(features[:, -1, :])
            loss = nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step in checkpoint_at_steps:
                with torch.no_grad():
                    accuracies.append(
                        float((logits.argmax(dim=-1) == labels).float().mean().item())
                    )
    except Exception as exc:  # noqa: BLE001
        return NanoBindResult(
            accuracies=tuple(accuracies),
            max_accuracy=0.0,
            random_baseline=random_baseline,
            rejected_persistent_zero=True,
            passed=False,
            notes=(f"{type(exc).__name__}: {exc}",),
        )

    lane_block.eval()
    max_acc = max(accuracies) if accuracies else 0.0
    persistent_zero = bool(accuracies) and all(
        acc <= random_baseline + margin for acc in accuracies
    )
    return NanoBindResult(
        accuracies=tuple(accuracies),
        max_accuracy=max_acc,
        random_baseline=random_baseline,
        rejected_persistent_zero=persistent_zero,
        passed=not persistent_zero,
    )
