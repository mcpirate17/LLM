# pyright: reportPrivateImportUsage=false
"""Probe lane circuit stability under interfering gradient pressure.

CPU-only research tool. For each candidate lane, runs two probes:

1. **Gradient Dispersion (GD)** — instant, structural. Forward + backward;
   per output position, fraction of input positions with non-zero gradient.
   Counts the architecture's parallel gradient pathways.

2. **NI-Stable** — ~1s training-based. Train induction → eval pre →
   train binding (interference) → re-eval induction. Retention =
   acc_post / acc_pre. Tests whether the lane HOLDS a circuit under
   competing gradient pressure.

Motivation (2026-05-17): the BLiMP-only fab screener crowned
``improve_tropical_gate_block_gated_parallel_84f0ccd08a`` (tropical +
wavelet under a per-token sigmoid gate). At 120M / 143K steps, the model
formed an induction circuit (AUC 0.87 at step 60K) but could not hold
it — induction oscillated 0.01 ↔ 0.87 across evals, binding eroded
monotonically 0.978 → 0.862. NI 0.5 by itself did not flag this
architecture at screening time. NI-Stable does:

    softmax / sparsemax / tropical (single)    retention 0.96–0.99 ✓
    tropical+wavelet (the failing winner)      retention 0.88        ⚠
    tropical+sparsemax+wavelet (worse, not
        better — more lanes = more gate
        freedom to abandon any one)            retention 0.84        ⚠
    simplified_mamba                            retention 0.54        ✗

The failure mode is **gate collapse in GatedParallelBlock**, not the WTA
mixer itself. Single-lane tropical retains perfectly; gated 2-lane
loses ~12 pts of retention; gated 3-lane is worse. Recommended
screening rule: retention ≥ 0.95 for promotion eligibility.

Run from repo root:

    CUDA_VISIBLE_DEVICES= python -m research.tools.probe_lane_circuit_stability
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import torch
from torch import nn

from component_fab.generator.block_templates import (
    GatedParallelBlock,
    ThreeLaneAdaptive,
)
from component_fab.generator.primitive_templates import (
    MultiscaleWaveletLane,
    SparsemaxAttention,
    TropicalAttention,
)
from component_fab.harness.nano_bind_probe import _sample_binding_batch
from component_fab.harness.nano_induction_probe import (
    _make_class_vectors,
    _sample_induction_batch,
)
from component_fab.harness.standard_block import LaneTestBlock
from component_fab.harness.tiny_lm import CausalConv1dLane, SoftmaxCausalAttention


_LaneFactory = Callable[[], nn.Module]


def gradient_dispersion(
    make_lane: _LaneFactory,
    *,
    dim: int = 32,
    seq_len: int = 24,
    threshold: float = 1e-6,
    n_samples: int = 4,
) -> tuple[float, float]:
    """Mean ± std fraction of input positions with non-zero gradient per output."""
    block = nn.Sequential(
        LaneTestBlock(make_lane(), dim), LaneTestBlock(make_lane(), dim)
    )
    block.eval()
    samples: list[float] = []
    for _ in range(n_samples):
        x = torch.randn(1, seq_len, dim, requires_grad=True)
        out = block(x)
        per_output: list[float] = []
        for o in range(seq_len):
            if x.grad is not None:
                x.grad.zero_()
            out[0, o, :].sum().backward(retain_graph=(o < seq_len - 1))
            assert x.grad is not None
            per_pos_norm = x.grad[0].norm(dim=-1)
            per_output.append((per_pos_norm > threshold).float().mean().item())
        samples.append(sum(per_output) / len(per_output))
    return statistics.mean(samples), statistics.stdev(samples) if len(
        samples
    ) > 1 else 0.0


def _train_step(
    block: nn.Module,
    head: nn.Linear,
    batch: tuple[torch.Tensor, torch.Tensor],
    opt: torch.optim.Optimizer,
) -> float:
    """One Adam step on ``head(block(x)[:, -1, :])`` against ``y``. Returns batch accuracy."""
    x, y = batch
    logits = head(block(x)[:, -1, :])
    loss = nn.functional.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float((logits.argmax(-1) == y).float().mean())


def _ind_batch(
    batch_size: int,
    seq_len: int,
    dim: int,
    n_classes: int,
    keys: torch.Tensor,
    vals: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _sample_induction_batch(
        batch_size,
        seq_len,
        dim,
        n_classes,
        class_keys=keys,
        class_values=vals,
        generator=generator,
    )


def _eval_induction(
    block: nn.Module,
    head: nn.Linear,
    *,
    dim: int,
    seq_len: int,
    n_classes: int,
    keys: torch.Tensor,
    vals: torch.Tensor,
    generator: torch.Generator,
    batch_size: int,
    n_batches: int,
) -> float:
    """Mean accuracy on induction over ``n_batches`` batches (no grad)."""
    block.eval()
    accs: list[float] = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = _ind_batch(
                batch_size, seq_len, dim, n_classes, keys, vals, generator
            )
            logits = head(block(x)[:, -1, :])
            accs.append(float((logits.argmax(-1) == y).float().mean()))
    return statistics.mean(accs)


def ni_stable(
    make_lane: _LaneFactory,
    *,
    dim: int = 32,
    seq_len: int = 24,
    n_classes: int = 8,
    n_form: int = 120,
    n_interfere: int = 80,
    n_eval_batches: int = 20,
    learning_rate: float = 3e-3,
    batch_size: int = 16,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Train induction, interfere with binding, re-eval induction.

    Returns ``(acc_pre, acc_post, retention)`` with
    ``retention = acc_post / max(acc_pre, 1e-6)``.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    block = nn.Sequential(
        LaneTestBlock(make_lane(), dim), LaneTestBlock(make_lane(), dim)
    )
    head_ind = nn.Linear(dim, n_classes)
    head_bin = nn.Linear(dim, n_classes)
    keys, vals = _make_class_vectors(n_classes, dim, generator)
    opt = torch.optim.Adam(
        list(block.parameters())
        + list(head_ind.parameters())
        + list(head_bin.parameters()),
        lr=learning_rate,
    )

    block.train()
    for _ in range(n_form):
        _train_step(
            block,
            head_ind,
            _ind_batch(batch_size, seq_len, dim, n_classes, keys, vals, generator),
            opt,
        )
    accs_pre = [
        _train_step(
            block,
            head_ind,
            _ind_batch(batch_size, seq_len, dim, n_classes, keys, vals, generator),
            opt,
        )
        for _ in range(n_eval_batches)
    ]
    for _ in range(n_interfere):
        bin_batch = _sample_binding_batch(
            batch_size, seq_len, dim, n_classes, generator=generator
        )
        _train_step(block, head_bin, bin_batch, opt)

    acc_pre = statistics.mean(accs_pre)
    acc_post = _eval_induction(
        block,
        head_ind,
        dim=dim,
        seq_len=seq_len,
        n_classes=n_classes,
        keys=keys,
        vals=vals,
        generator=generator,
        batch_size=batch_size,
        n_batches=n_eval_batches,
    )
    return acc_pre, acc_post, acc_post / max(acc_pre, 1e-6)


def default_lane_lineup(dim: int) -> dict[str, _LaneFactory]:
    """The lineup used in the 2026-05-17 study (so results stay comparable)."""
    return {
        "softmax": lambda: SoftmaxCausalAttention(dim),
        "sparsemax": lambda: SparsemaxAttention(dim),
        "tropical (WTA)": lambda: TropicalAttention(dim),
        "wavelet": lambda: MultiscaleWaveletLane(dim),
        "conv": lambda: CausalConv1dLane(dim),
        "tropical+wavelet (current winner)": lambda: GatedParallelBlock(
            lambda d: TropicalAttention(d),
            lambda d: MultiscaleWaveletLane(d),
            dim,
        ),
        "tropical+sparsemax+wavelet (3-lane)": lambda: ThreeLaneAdaptive(
            lambda d: TropicalAttention(d),
            lambda d: SparsemaxAttention(d),
            lambda d: MultiscaleWaveletLane(d),
            dim,
        ),
    }


def run(
    cases: dict[str, _LaneFactory], *, dim: int, seq_len: int, seeds: tuple[int, ...]
) -> None:
    print(
        f"{'lane':<42} {'GD':>10} {'NI_pre':>8} {'NI_post':>8} {'retain':>7} {'t_s':>6}"
    )
    print("-" * 90)
    for name, mk in cases.items():
        t0 = time.time()
        gd_m, gd_s = gradient_dispersion(mk, dim=dim, seq_len=seq_len)
        pres: list[float] = []
        posts: list[float] = []
        rets: list[float] = []
        for sd in seeds:
            pre, post, ret = ni_stable(mk, dim=dim, seq_len=seq_len, seed=sd)
            pres.append(pre)
            posts.append(post)
            rets.append(ret)
        dt = time.time() - t0
        print(
            f"{name:<42} {gd_m:>7.3f}±{gd_s:.2f}  "
            f"{statistics.mean(pres):>7.3f} {statistics.mean(posts):>8.3f} "
            f"{statistics.mean(rets):>7.3f} {dt:>6.1f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    run(
        default_lane_lineup(args.dim),
        dim=args.dim,
        seq_len=args.seq_len,
        seeds=tuple(args.seeds),
    )


if __name__ == "__main__":
    main()
