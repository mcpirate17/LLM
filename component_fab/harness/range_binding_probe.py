"""Distance-resolved binding probe — component-level sparse/long-range mixing.

The other component gates only exercise mixing to ``seq_len <= 32``
(``nano_bind`` uses one fixed range; ``measure_erf`` is a single Jacobian over
32 positions). Nothing graded a *component* on whether it can carry a binding
signal across a long, sparse gap before paying for an LM-scale fingerprint —
exactly the failure mode (a lane that binds at 32 but collapses at 256) that
slips through. This probe closes that gap, mirroring the LM-level
``binding_range`` sweep at the cheap component tier.

Protocol (one trained model, distance-resolved eval — like the LM probe):

1. Train the lane + a linear head on a *mixed-distance* binding task: a
   per-class key sits at a random distance ``d ∈ [1, max_distance]`` before the
   readout (final) position, the rest of the length-``(max_distance+1)``
   sequence is random distractors, and the head classifies the key from the
   final position. The model must bind across *all* distances at once.
2. Evaluate on a large held-out set with the key placed at each probe distance
   exactly, giving a low-variance accuracy-vs-distance curve.

The **effective binding range** is the largest distance whose held-out accuracy
clears ``1/n_classes + margin``. **Soft signal** — recorded and ranked, never
sets ``eliminated_by`` (a pure-local lane legitimately decays with distance;
we want that profile recorded, not zeroed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .standard_block import LaneTestBlock
from .training_probe import train_lane_head

DEFAULT_DISTANCES: tuple[int, ...] = (8, 16, 32, 64, 128, 256)


@dataclass(frozen=True, slots=True)
class RangeBindingResult:
    distances: tuple[int, ...]
    per_distance_accuracy: dict[int, float]
    aggregate_accuracy: float  # mean over distances — overall range-mixing
    max_accuracy: float
    effective_distance: int  # largest distance above baseline+margin (0 = none)
    random_baseline: float
    above_baseline: bool
    margin: float
    notes: tuple[str, ...] = field(default_factory=tuple)


def _place_keys(
    x: torch.Tensor, labels: torch.Tensor, class_keys: torch.Tensor, dists: torch.Tensor
) -> None:
    """In-place: put each example's class key at ``seq_len-1-dist`` (causal gap)."""
    seq_len = x.shape[1]
    rows = torch.arange(x.shape[0])
    pos = (seq_len - 1) - dists
    x[rows, pos] = class_keys[labels]


def _train_mixed_distance(
    block: nn.Module,
    head: nn.Linear,
    *,
    dim: int,
    n_classes: int,
    max_distance: int,
    class_keys: torch.Tensor,
    n_train_steps: int,
    learning_rate: float,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> None:
    seq_len = max_distance + 1

    def sample() -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.randint(0, n_classes, (batch_size,), generator=generator)
        dists = torch.randint(1, max_distance + 1, (batch_size,), generator=generator)
        x = torch.randn(batch_size, seq_len, dim, generator=generator)
        _place_keys(x, labels, class_keys, dists)
        return x, labels

    block.train()
    train_lane_head(
        lambda x: head(block(x.to(device))[:, -1, :]),
        list(block.parameters()) + list(head.parameters()),
        sample,
        lambda logits, labels: nn.functional.cross_entropy(logits, labels.to(device)),
        n_train_steps=n_train_steps,
        learning_rate=learning_rate,
    )
    block.eval()


@torch.no_grad()
def _eval_at_distance(
    block: nn.Module,
    head: nn.Linear,
    *,
    dim: int,
    n_classes: int,
    distance: int,
    class_keys: torch.Tensor,
    n_eval: int,
    generator: torch.Generator,
    device: torch.device,
) -> float:
    """Held-out accuracy with the key placed at exactly ``distance``."""
    seq_len = distance + 1
    labels = torch.randint(0, n_classes, (n_eval,), generator=generator)
    x = torch.randn(n_eval, seq_len, dim, generator=generator)
    _place_keys(x, labels, class_keys, torch.full((n_eval,), distance))
    logits = head(block(x.to(device))[:, -1, :])
    return float((logits.argmax(-1).cpu() == labels).float().mean().item())


def range_binding_gate(
    lane: nn.Module,
    *,
    dim: int = 32,
    distances: tuple[int, ...] = DEFAULT_DISTANCES,
    n_classes: int = 8,
    n_train_steps: int = 150,
    learning_rate: float = 3e-3,
    batch_size: int = 32,
    n_eval: int = 256,
    margin: float = 0.08,
    seed: int = 0,
) -> RangeBindingResult:
    """Train once on mixed distances, eval distance-resolved; report the curve.

    Soft scorecard. ``effective_distance`` is the longest gap the lane still
    binds across (held-out accuracy >= ``1/n_classes + margin``).
    """
    torch.manual_seed(seed)
    device = next(lane.parameters()).device
    baseline = 1.0 / float(n_classes)
    max_distance = max(distances)
    generator = torch.Generator().manual_seed(seed)
    block = LaneTestBlock(lane, dim).to(device)
    head = nn.Linear(dim, n_classes).to(device)
    # Kept on CPU: key placement builds ``x`` on CPU, which only moves to the
    # lane's device at the forward call (avoids a CPU/GPU placement mismatch).
    class_keys = torch.randn(n_classes, dim, generator=generator) * 2.0
    per_distance: dict[int, float] = {}
    try:
        _train_mixed_distance(
            block,
            head,
            dim=dim,
            n_classes=n_classes,
            max_distance=max_distance,
            class_keys=class_keys,
            n_train_steps=n_train_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            generator=generator,
            device=device,
        )
        eval_gen = torch.Generator().manual_seed(seed + 1)
        for distance in distances:
            per_distance[distance] = _eval_at_distance(
                block,
                head,
                dim=dim,
                n_classes=n_classes,
                distance=distance,
                class_keys=class_keys,
                n_eval=n_eval,
                generator=eval_gen,
                device=device,
            )
    except Exception as exc:  # noqa: BLE001 — broken lane scores 0, not crash grading
        return RangeBindingResult(
            distances=tuple(distances),
            per_distance_accuracy=per_distance,
            aggregate_accuracy=0.0,
            max_accuracy=0.0,
            effective_distance=0,
            random_baseline=baseline,
            above_baseline=False,
            margin=margin,
            notes=(f"{type(exc).__name__}: {exc}",),
        )

    accs = list(per_distance.values())
    above = [d for d, a in per_distance.items() if a >= baseline + margin]
    aggregate = sum(accs) / len(accs) if accs else 0.0
    return RangeBindingResult(
        distances=tuple(distances),
        per_distance_accuracy=per_distance,
        aggregate_accuracy=aggregate,
        max_accuracy=max(accs) if accs else 0.0,
        effective_distance=max(above) if above else 0,
        random_baseline=baseline,
        above_baseline=bool(above),
        margin=margin,
    )
