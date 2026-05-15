"""Benchmark component_fab probe wall time and best-effort memory cost.

Generates ``component_fab/catalog/probe_costs.json`` by default. The
benchmark covers the sprint-8 probe stack at the TODO-requested sizes:

- dim=16, seq_len=16, batch=8
- dim=64, seq_len=64, batch=8

Training probes intentionally instantiate a fresh module per measured call
so optimizer state and learned weights do not leak between repeats.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import resource
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import nn

from component_fab.generator.primitive_templates import TropicalAttention
from component_fab.harness.capability_probes import make_ar_probe, train_and_score
from component_fab.harness.erf_probe import measure_erf
from component_fab.harness.nano_bind_probe import nano_bind_gate
from component_fab.harness.probe_block import WinnerLikeBlock, short_training_probe
from component_fab.harness.standard_block import (
    LaneTestBlock,
    lane_forward_for_mix_speed,
)
from component_fab.metrics.compression_quality import measure_compression_quality
from component_fab.metrics.mix_speed import measure_mix_speed
from component_fab.metrics.routing_health import measure_routing_health


_REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = _REPO / "component_fab" / "catalog" / "probe_costs.json"


@dataclass(frozen=True, slots=True)
class ProbeSize:
    dim: int
    seq_len: int
    batch_size: int

    @property
    def label(self) -> str:
        return f"d{self.dim}_l{self.seq_len}_b{self.batch_size}"


DEFAULT_SIZES: tuple[ProbeSize, ...] = (
    # Keep these in the order called out in the sprint-9 TODO.
    ProbeSize(dim=16, seq_len=16, batch_size=8),
    ProbeSize(dim=64, seq_len=64, batch_size=8),
)


@dataclass(frozen=True, slots=True)
class ProbeTiming:
    probe: str
    size: dict[str, int]
    repeats: int
    warmups: int
    wall_ms_mean: float
    wall_ms_min: float
    wall_ms_max: float
    python_peak_bytes_max: int
    rss_delta_bytes_max: int
    output: dict[str, Any]


class _LearnedSoftmaxRouter(nn.Module):
    def __init__(self, dim: int, n_lanes: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, n_lanes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.gate(x), dim=-1)


def _rss_bytes() -> int:
    """Return process max RSS in bytes on Linux/macOS."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes. This repo runs primarily on Linux,
    # but the guard keeps local macOS runs roughly correct.
    if usage < 10_000_000:
        return int(usage * 1024)
    return int(usage)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _scorecard(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return _json_safe(value)
    return {"value": _json_safe(value)}


def _measure(
    factory: Callable[[], Any],
    *,
    probe: str,
    size: ProbeSize,
    repeats: int,
    warmups: int,
) -> ProbeTiming:
    for _ in range(warmups):
        gc.collect()
        torch.manual_seed(0)
        factory()

    wall_times: list[float] = []
    python_peaks: list[int] = []
    rss_deltas: list[int] = []
    last_output: dict[str, Any] = {}
    for _ in range(repeats):
        gc.collect()
        torch.manual_seed(0)
        rss_before = _rss_bytes()
        tracemalloc.start()
        t0 = time.perf_counter()
        output = factory()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_after = _rss_bytes()

        wall_times.append(wall_ms)
        python_peaks.append(int(peak))
        rss_deltas.append(max(0, int(rss_after - rss_before)))
        last_output = _scorecard(output)

    return ProbeTiming(
        probe=probe,
        size=asdict(size),
        repeats=repeats,
        warmups=warmups,
        wall_ms_mean=statistics.fmean(wall_times),
        wall_ms_min=min(wall_times),
        wall_ms_max=max(wall_times),
        python_peak_bytes_max=max(python_peaks),
        rss_delta_bytes_max=max(rss_deltas),
        output=last_output,
    )


def _mix_speed_factory(
    size: ProbeSize, metric_trials: int, train_steps: int
) -> Callable[[], Any]:
    del train_steps

    def run() -> Any:
        lane = TropicalAttention(size.dim, causal=True).eval()
        return measure_mix_speed(
            lane_forward_for_mix_speed(lane, size.dim),
            seq_len=size.seq_len,
            feature_dim=size.dim,
            batch_size=size.batch_size,
            n_trials=metric_trials,
        )

    return run


def _routing_health_factory(
    size: ProbeSize, metric_trials: int, train_steps: int
) -> Callable[[], Any]:
    del train_steps

    def run() -> Any:
        return measure_routing_health(
            _LearnedSoftmaxRouter(size.dim, n_lanes=3),
            n_lanes=3,
            seq_len=size.seq_len,
            feature_dim=size.dim,
            batch_size=size.batch_size,
            n_trials=metric_trials,
        )

    return run


def _compression_quality_factory(
    size: ProbeSize, metric_trials: int, train_steps: int
) -> Callable[[], Any]:
    del train_steps
    latent = max(1, size.dim // 4)

    def run() -> Any:
        return measure_compression_quality(
            nn.Linear(size.dim, latent),
            nn.Linear(latent, size.dim),
            input_dim=size.dim,
            latent_dim_declared=latent,
            seq_len=size.seq_len,
            batch_size=size.batch_size,
            n_trials=metric_trials,
        )

    return run


def _s05_gate_factory(
    size: ProbeSize, metric_trials: int, train_steps: int
) -> Callable[[], Any]:
    del metric_trials, train_steps

    def run() -> Any:
        return measure_s05(size)

    return run


def measure_s05(size: ProbeSize) -> Any:
    from component_fab.harness.capability_probes import causality_stability_gate

    return causality_stability_gate(
        WinnerLikeBlock(TropicalAttention(size.dim, causal=True), size.dim).eval(),
        seq_len=size.seq_len,
        dim=size.dim,
        batch_size=size.batch_size,
    )


def _erf_factory(
    size: ProbeSize, metric_trials: int, train_steps: int
) -> Callable[[], Any]:
    del metric_trials, train_steps

    def run() -> Any:
        return measure_erf(
            LaneTestBlock(TropicalAttention(size.dim, causal=True), size.dim),
            seq_len=size.seq_len,
            dim=size.dim,
            batch_size=size.batch_size,
        )

    return run


def _nano_bind_factory(
    size: ProbeSize, metric_trials: int, train_steps: int
) -> Callable[[], Any]:
    del metric_trials

    def run() -> Any:
        return nano_bind_gate(
            LaneTestBlock(TropicalAttention(size.dim, causal=True), size.dim),
            dim=size.dim,
            seq_len=size.seq_len,
            batch_size=size.batch_size,
            n_train_steps=train_steps,
            checkpoint_at_steps=(
                max(1, train_steps // 3),
                max(1, 2 * train_steps // 3),
                train_steps,
            ),
        )

    return run


def _ar_factory(
    n_pairs: int, name: str
) -> Callable[[ProbeSize, int, int], Callable[[], Any]]:
    def factory(
        size: ProbeSize, metric_trials: int, train_steps: int
    ) -> Callable[[], Any]:
        del metric_trials
        seq_len = max(size.seq_len, 2 * n_pairs + 2)

        def run() -> Any:
            probe = make_ar_probe(n_pairs=n_pairs, name=name, n_train_steps=train_steps)
            return train_and_score(
                WinnerLikeBlock(TropicalAttention(size.dim, causal=True), size.dim),
                probe,
                seq_len=seq_len,
                dim=size.dim,
            )

        return run

    return factory


def _in_context_factory(
    size: ProbeSize, metric_trials: int, train_steps: int
) -> Callable[[], Any]:
    del metric_trials

    def run() -> Any:
        return short_training_probe(
            TropicalAttention(size.dim, causal=True),
            dim=size.dim,
            seq_len=size.seq_len,
            batch_size=size.batch_size,
            n_steps=train_steps,
        )

    return run


PROBE_FACTORIES: dict[str, Callable[[ProbeSize, int, int], Callable[[], Any]]] = {
    "mix_speed": _mix_speed_factory,
    "routing_health": _routing_health_factory,
    "compression_quality": _compression_quality_factory,
    "s05_gate": _s05_gate_factory,
    "erf_density": _erf_factory,
    "nano_bind": _nano_bind_factory,
    "ar_easy": _ar_factory(2, "ar_easy"),
    "ar_medium": _ar_factory(5, "ar_medium"),
    "in_context_running_mean": _in_context_factory,
}


def run_probe_bench(
    *,
    sizes: Iterable[ProbeSize] = DEFAULT_SIZES,
    probes: Iterable[str] = PROBE_FACTORIES.keys(),
    out: Path = DEFAULT_OUT,
    repeats: int = 1,
    warmups: int = 1,
    metric_trials: int = 8,
    train_steps: int = 60,
) -> dict[str, Any]:
    probe_names = tuple(probes)
    unknown = sorted(set(probe_names) - set(PROBE_FACTORIES))
    if unknown:
        raise ValueError(f"unknown probe benchmark(s): {', '.join(unknown)}")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if train_steps <= 0:
        raise ValueError("train_steps must be positive")
    if metric_trials <= 0:
        raise ValueError("metric_trials must be positive")

    timings: list[ProbeTiming] = []
    for size in sizes:
        for probe in probe_names:
            factory = PROBE_FACTORIES[probe](size, metric_trials, train_steps)
            timings.append(
                _measure(
                    factory,
                    probe=probe,
                    size=size,
                    repeats=repeats,
                    warmups=warmups,
                )
            )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "repeats": repeats,
        "warmups": warmups,
        "metric_trials": metric_trials,
        "train_steps": train_steps,
        "sizes": [asdict(size) for size in sizes],
        "benchmarks": [_json_safe(asdict(timing)) for timing in timings],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def _parse_size(raw: str) -> ProbeSize:
    parts = raw.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("size must be DIM,SEQ_LEN,BATCH")
    try:
        dim, seq_len, batch = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("size values must be integers") from exc
    if dim <= 0 or seq_len <= 0 or batch <= 0:
        raise argparse.ArgumentTypeError("size values must be positive")
    return ProbeSize(dim=dim, seq_len=seq_len, batch_size=batch)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="component_fab probe timing benchmark")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--size",
        action="append",
        type=_parse_size,
        help="Benchmark size as DIM,SEQ_LEN,BATCH. May be repeated.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--metric-trials", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=60)
    parser.add_argument(
        "--probe",
        action="append",
        choices=sorted(PROBE_FACTORIES),
        help="Probe to benchmark. May be repeated. Defaults to all.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = run_probe_bench(
        sizes=tuple(args.size) if args.size else DEFAULT_SIZES,
        probes=tuple(args.probe) if args.probe else tuple(PROBE_FACTORIES),
        out=args.out,
        repeats=args.repeats,
        warmups=args.warmups,
        metric_trials=args.metric_trials,
        train_steps=args.train_steps,
    )
    print(f"wrote {args.out} ({len(report['benchmarks'])} probe-size measurements)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
