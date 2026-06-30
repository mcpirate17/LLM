"""Score the real compression op(s) inside a compiled fab candidate.

``compression_quality.measure_compression_quality`` scores an arbitrary
``(compress, restore)`` pair. This module supplies the missing half: a small
contract a compression lane exposes so the grading path can pull the
candidate's *own* compress/restore out of the compiled graph and persist a
real scorecard — instead of scoring a fixed ``Linear -> Linear`` (the only
thing ``run_probe_bench`` ever measured) and never writing it to the ledger.

A lane opts in by implementing ``compression_probe_pair`` (see
``SupportsCompressionProbe``). ``score_compression_in_module`` walks the
compiled module, scores every opted-in sub-lane, and aggregates conservatively
(worst latent utilization / worst reconstruction across sub-lanes) so a single
weak compressor still surfaces. Returns ``{}`` when the candidate declares no
compression op — never a fabricated score.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import torch
from torch import nn

from component_fab.metrics.compression_quality import measure_compression_quality

CompressFn = Callable[[torch.Tensor], torch.Tensor]
RestoreFn = Callable[[torch.Tensor], torch.Tensor]


@runtime_checkable
class SupportsCompressionProbe(Protocol):
    """A lane that can expose its internal compress/restore bottleneck.

    ``compression_probe_pair`` returns ``(compress_fn, restore_fn,
    latent_dim_declared)`` where ``compress_fn`` maps ``(B, S, D) -> (B, S,
    latent)`` and ``restore_fn`` maps ``(B, S, latent) -> (B, S, D)``, with the
    invariant ``restore_fn(compress_fn(x)) == forward(x)`` (the lane's readout
    path). ``latent_dim_declared`` is the lane's declared latent budget so the
    effective-rank ratio measures how much of that budget is actually used.
    """

    def compression_probe_pair(self) -> tuple[CompressFn, RestoreFn, int]: ...


def score_compression_in_module(
    module: nn.Module,
    *,
    dim: int,
    seq_len: int,
    batch_size: int = 8,
    n_trials: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Score every compression sub-lane in ``module``; ``{}`` if none.

    The aggregate is conservative: the worst (lowest) effective-rank ratio and
    the worst (highest) reconstruction MSE across all compression sub-lanes, so
    one under-utilized / poorly-reconstructing compressor flags the candidate.
    """

    probes: list[SupportsCompressionProbe] = [
        sub for sub in module.modules() if isinstance(sub, SupportsCompressionProbe)
    ]
    if not probes:
        return {}

    rank_ratios: list[float] = []
    reconstruct_mses: list[float] = []
    flops_reductions: list[float] = []
    for lane in probes:
        compress_fn, restore_fn, latent_dim = lane.compression_probe_pair()
        card = measure_compression_quality(
            compress_fn,
            restore_fn,
            input_dim=dim,
            latent_dim_declared=latent_dim,
            seq_len=seq_len,
            batch_size=batch_size,
            n_trials=n_trials,
            seed=seed,
        )
        rank_ratios.append(float(card.effective_rank_ratio))
        reconstruct_mses.append(float(card.reconstruction_mse))
        flops_reductions.append(float(card.flops_per_token_reduction))

    worst_rank_ratio = min(rank_ratios)
    worst_reconstruct = max(reconstruct_mses)
    return {
        "compression_declared": True,
        "compression_n_ops": len(probes),
        # quality headline == latent utilization (higher is better).
        "compression_quality": round(worst_rank_ratio, 4),
        "compression_effective_rank_ratio": round(worst_rank_ratio, 4),
        "compression_reconstruct_mse": round(worst_reconstruct, 4),
        "compression_ratio": round(max(flops_reductions), 4),
    }
