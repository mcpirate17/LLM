"""Multi-seed capability bench for the surprise-memory primitive family.

Compares the test-time (Titans/TTT-style) surprise-memory lanes against
references on the capability probes that matter for this research line —
binding, induction, associative recall — and an *exploratory* ultrametric
(hierarchical) recall probe that gives ``PadicSurpriseMemoryLane`` a regime
where its p-adic addressing can pay off (the standard probes use i.i.d. flat
keys with no hierarchy to exploit).

Capability metrics at this scale are high-variance (single-seed numbers
swing wildly), so every metric is reported as ``median ± pstdev`` over
``--seeds`` runs, re-initializing the lane and re-seeding each probe.

Usage:
    python -m component_fab.tools.run_surprise_memory_bench
    python -m component_fab.tools.run_surprise_memory_bench --seeds 8 --dim 32
    python -m component_fab.tools.run_surprise_memory_bench --no-hierarchical
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from component_fab.generator.memory_primitives import CausalFastWeightMemoryLane
from component_fab.harness.capability_probes import make_ar_probe, train_and_score
from component_fab.harness.nano_bind_probe import nano_bind_gate
from component_fab.harness.nano_induction_probe import nano_induction_gate
from component_fab.harness.reference_lanes import LaneFactory, REFERENCE_LANES
from component_fab.harness.standard_block import LaneTestBlock
from component_fab.tools._cli import write_report
from component_fab.validator.capability import _stacked_induction_block

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = _REPO / "component_fab" / "catalog" / "surprise_memory_bench_latest.json"

# Reference baselines + the surprise-memory family. References frame the
# result: TropicalAttention is the O(L^2) ceiling, CausalFastWeightMemory is
# the pure-Hebbian O(L) memory the surprise rule is meant to beat.
DEFAULT_CANDIDATES: dict[str, LaneFactory] = {
    "ref_tropical_attention": REFERENCE_LANES["tropical_attention"],
    "ref_linear_ssm": REFERENCE_LANES["linear_ssm"],
    "base_hebbian_fastweight": lambda d: CausalFastWeightMemoryLane(d),
    "tropical_surprise_memory": REFERENCE_LANES["tropical_surprise_memory"],
    "padic_surprise_memory": REFERENCE_LANES["padic_surprise_memory"],
}


@dataclass(frozen=True, slots=True)
class MetricSummary:
    median: float
    mean: float
    pstdev: float
    values: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "median": round(self.median, 4),
            "mean": round(self.mean, 4),
            "pstdev": round(self.pstdev, 4),
            "values": [round(v, 4) for v in self.values],
        }

    def cell(self) -> str:
        return f"{self.median:.2f}±{self.pstdev:.2f}"


def _summarize(values: list[float]) -> MetricSummary:
    return MetricSummary(
        median=statistics.median(values),
        mean=statistics.fmean(values),
        pstdev=statistics.pstdev(values) if len(values) > 1 else 0.0,
        values=tuple(values),
    )


# --------------------- exploratory hierarchical recall ---------------------


def _sample_hierarchical_batch(
    batch_size: int,
    seq_len: int,
    dim: int,
    *,
    n_coarse: int,
    n_fine: int,
    fine_scale: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-context ultrametric recall: predict the COARSE class of a cue whose
    fine code was never paired in-context.

    Keys factor as ``key(c, f) = coarse[c] + fine_scale * fine[f]`` so two
    keys sharing ``c`` are ultrametrically close. Layout per example:

        p1: key(c, f_study)   p1+1: coarse[c] (value marker)
        p2: key(c, f_query)   (f_query != f_study — novel exact key)
        -1: SLOT (=0); head reads here and predicts c.

    A flat exact-match memory must rely on the coarse component dominating
    distance; a hierarchical (coarse-pooled) memory can match on ``c``
    regardless of the fine code — the gap widens as ``fine_scale`` grows.
    """
    if seq_len < 8:
        raise ValueError("seq_len must be >= 8 for the hierarchical layout")
    coarse = torch.randn(n_coarse, dim, generator=generator) * 2.0
    fine = torch.randn(n_fine, dim, generator=generator)
    labels = torch.randint(0, n_coarse, (batch_size,), generator=generator)
    f_study = torch.randint(0, n_fine, (batch_size,), generator=generator)
    f_query = (
        f_study + torch.randint(1, n_fine, (batch_size,), generator=generator)
    ) % n_fine

    x = torch.randn(batch_size, seq_len, dim, generator=generator)
    q1_hi = max(1, seq_len // 4 - 1)
    q3_lo = max(seq_len // 2 + 1, 3 * seq_len // 4)
    q3_hi = seq_len - 2
    if q3_lo >= q3_hi:
        q3_lo = q3_hi - 1
    p1 = torch.randint(0, q1_hi, (batch_size,), generator=generator)
    p2 = torch.randint(q3_lo, q3_hi, (batch_size,), generator=generator)
    rows = torch.arange(batch_size)
    x[rows, p1] = coarse[labels] + fine_scale * fine[f_study]
    x[rows, p1 + 1] = coarse[labels]
    x[rows, p2] = coarse[labels] + fine_scale * fine[f_query]
    x[:, -1] = 0.0
    return x, labels


def hierarchical_recall_accuracy(
    lane_block: nn.Module,
    *,
    dim: int,
    seq_len: int,
    n_coarse: int = 6,
    n_fine: int = 4,
    fine_scale: float = 1.5,
    n_train_steps: int = 150,
    learning_rate: float = 3e-3,
    batch_size: int = 16,
    seed: int = 0,
) -> float:
    """Best-checkpoint accuracy on the ultrametric recall probe."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    head = nn.Linear(dim, n_coarse)
    optimizer = torch.optim.Adam(
        list(lane_block.parameters()) + list(head.parameters()), lr=learning_rate
    )
    best = 0.0
    try:
        lane_block.train()
        for step in range(1, n_train_steps + 1):
            x, labels = _sample_hierarchical_batch(
                batch_size,
                seq_len,
                dim,
                n_coarse=n_coarse,
                n_fine=n_fine,
                fine_scale=fine_scale,
                generator=generator,
            )
            logits = head(lane_block(x)[:, -1, :])
            loss = nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % (n_train_steps // 3) == 0:
                with torch.no_grad():
                    best = max(
                        best,
                        float((logits.argmax(-1) == labels).float().mean().item()),
                    )
    except Exception:  # noqa: BLE001 — a broken lane scores 0, not crash the sweep
        return 0.0
    lane_block.eval()
    return best


# ------------------------------ probe runners ------------------------------


def _probe_runners(
    dim: int, seq_len: int, include_hierarchical: bool
) -> dict[str, Callable[[LaneFactory, int], float]]:
    ar_easy = make_ar_probe(n_pairs=2, name="ar_easy", n_train_steps=40)
    ar_medium = make_ar_probe(n_pairs=5, name="ar_medium", n_train_steps=60)

    def nb(make: LaneFactory, seed: int) -> float:
        torch.manual_seed(seed)
        return nano_bind_gate(
            LaneTestBlock(make(dim), dim), dim=dim, seq_len=seq_len, seed=seed
        ).max_accuracy

    def ind(make: LaneFactory, seed: int) -> float:
        torch.manual_seed(seed)
        return nano_induction_gate(
            _stacked_induction_block(make(dim), dim),
            dim=dim,
            seq_len=max(seq_len, 24),
            n_classes=8,
            seed=seed,
        ).max_accuracy

    def ar(probe):
        def run(make: LaneFactory, seed: int) -> float:
            torch.manual_seed(seed)
            return train_and_score(
                LaneTestBlock(make(dim), dim).train(),
                probe,
                seq_len=seq_len,
                dim=dim,
                seed=seed,
            ).relative_recall

        return run

    def hier(make: LaneFactory, seed: int) -> float:
        torch.manual_seed(seed)
        return hierarchical_recall_accuracy(
            LaneTestBlock(make(dim), dim), dim=dim, seq_len=max(seq_len, 24), seed=seed
        )

    runners: dict[str, Callable[[LaneFactory, int], float]] = {
        "nano_bind": nb,
        "induction": ind,
        "ar_easy": ar(ar_easy),
        "ar_medium": ar(ar_medium),
    }
    if include_hierarchical:
        runners["hier_recall"] = hier
    return runners


def run_bench(
    *,
    seeds: int,
    dim: int,
    seq_len: int,
    candidates: dict[str, LaneFactory] = DEFAULT_CANDIDATES,
    include_hierarchical: bool = True,
) -> dict[str, dict[str, MetricSummary]]:
    runners = _probe_runners(dim, seq_len, include_hierarchical)
    seed_list = list(range(seeds))
    results: dict[str, dict[str, MetricSummary]] = {}
    for name, make in candidates.items():
        per_metric: dict[str, MetricSummary] = {}
        for metric, run in runners.items():
            per_metric[metric] = _summarize([run(make, s) for s in seed_list])
        results[name] = per_metric
    return results


def _print_table(results: dict[str, dict[str, MetricSummary]], seeds: int) -> None:
    metrics = list(next(iter(results.values())).keys())
    header = f"{'candidate':26}" + "".join(f"{m:>13}" for m in metrics)
    print(header)
    print("-" * len(header))
    for name, per_metric in results.items():
        row = f"{name:26}" + "".join(f"{per_metric[m].cell():>13}" for m in metrics)
        print(row)
    print(f"\n(median±pstdev over {seeds} seeds)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="surprise-memory capability bench")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--no-hierarchical", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    results = run_bench(
        seeds=args.seeds,
        dim=args.dim,
        seq_len=args.seq_len,
        include_hierarchical=not args.no_hierarchical,
    )
    _print_table(results, args.seeds)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seeds": args.seeds,
        "dim": args.dim,
        "seq_len": args.seq_len,
        "include_hierarchical": not args.no_hierarchical,
        "results": {
            name: {metric: summ.as_dict() for metric, summ in per_metric.items()}
            for name, per_metric in results.items()
        },
    }
    print()
    write_report(
        report,
        default_dir=DEFAULT_OUT.parent,
        prefix="surprise_memory_bench",
        output=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
