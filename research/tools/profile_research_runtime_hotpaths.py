from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import random
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from research.eval.choice_scoring import concat_choice_tokens
from research.eval.fingerprint_types import BehavioralFingerprint
from research.eval.metrics import batch_novelty_scores
from research.eval.routing_telemetry import collect_routing_telemetry
from research.search._behavior_archive import BehaviorArchive
from research.search._nsga import nsga2_rank
from research.synthesis.grammar import GrammarConfig, generate_layer_graph

_GRAPH_FIXTURE_CACHE: dict[tuple[int, int], list[Any]] = {}


@dataclass(slots=True)
class DummyIndividual:
    fitness: float
    novelty: float
    fingerprint: str = ""
    pareto_rank: int = 0
    crowding_dist: float = 0.0


class RoutingLeaf(nn.Module):
    def __init__(self, width: int, idx: int) -> None:
        super().__init__()
        base = torch.arange(width, dtype=torch.float32)
        self.routing_telemetry = {
            "tokens_total": width * 8 + idx,
            "keep_count": width * 5,
            "drop_count": width * 3,
            "default_path_count": idx % 7,
            "routed_token_count": width * 4,
            "sparse_span_count": idx % 5,
            "sparse_span_width_sum": float(width + idx),
            "sparse_span_width_count": 1,
            "sparse_span_coverage_tokens": width * 3,
            "confidence_sum": float(width),
            "confidence_sq_sum": float(width) * 0.5,
            "confidence_count": width,
            "route_strength_sum": float(width) * 0.25,
            "route_strength_count": width,
            "branch_weight_sum": base,
            "branch_weight_count": 1,
            "branch_dominance_sum": 0.25,
            "routed_branch_share_sum": 0.5,
            "medium_branch_share_sum": 0.25,
            "hard_branch_share_sum": 0.25,
            "lane_histogram": base.remainder(5) + 1,
            "confidence_histogram": base.remainder(3) + 1,
            "lane_count": width,
            "routing_mode": "topk",
            "gate_type": "softmax",
            "span_type": "dense",
        }


class RoutingModel(nn.Module):
    def __init__(self, n_modules: int, width: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(RoutingLeaf(width, i) for i in range(n_modules))


def _fingerprint(seed: int) -> BehavioralFingerprint:
    r = random.Random(seed)
    return BehavioralFingerprint(
        interaction_locality=r.random(),
        interaction_sparsity=r.random(),
        interaction_symmetry=r.random(),
        interaction_hierarchy=r.random(),
        isotropy=r.random(),
        rank_ratio=r.random(),
        sensitivity_uniformity=r.random(),
        cka_vs_transformer=r.random(),
        cka_vs_ssm=r.random(),
        cka_vs_conv=r.random(),
        jacobian_spectral_norm=r.random() * 5.0,
        jacobian_effective_rank=r.random() * 16.0,
        routing_selectivity=r.random(),
        routing_compute_ratio=r.random() * 2.0,
        hierarchy_fitness=r.random(),
        gromov_delta=r.random() * 0.3,
        novelty_score=r.random(),
    )


def _build_graphs(n_graphs: int, seed: int) -> list[Any]:
    grammar = GrammarConfig(max_depth=8, model_dim=64)
    graphs = []
    attempts = 0
    while len(graphs) < n_graphs and attempts < n_graphs * 8:
        attempts += 1
        try:
            graphs.append(generate_layer_graph(grammar, seed=seed + attempts * 17))
        except (ValueError, RuntimeError):
            continue
    if len(graphs) < n_graphs:
        raise RuntimeError(f"only generated {len(graphs)}/{n_graphs} graphs")
    return graphs


def _graph_fixture(n_graphs: int, seed: int) -> list[Any]:
    key = (n_graphs, seed)
    graphs = _GRAPH_FIXTURE_CACHE.get(key)
    if graphs is None:
        graphs = _build_graphs(n_graphs, seed)
        _GRAPH_FIXTURE_CACHE[key] = graphs
    for graph in graphs:
        graph._cache.clear()
    return graphs


def workload_graph_generation(size: int) -> None:
    _build_graphs(size, random.randrange(1, 10_000_000))


def workload_behavior_archive(size: int) -> None:
    archive = BehaviorArchive(max_size=size)
    individuals = []
    for i in range(size):
        ind = DummyIndividual(
            fitness=(i % 101) / 100.0,
            novelty=(i % 37) / 37.0,
            fingerprint=f"g{i}",
        )
        individuals.append(ind)
        archive.add(f"g{i}", _fingerprint(i), ind)  # type: ignore[arg-type]
    for i in range(size):
        archive.novelty_of(_fingerprint(100_000 + i), k=15)
    for _ in range(max(5, size // 16)):
        archive.suggest_exploit_target(k=5)
    archive.update_individuals(individuals)  # type: ignore[arg-type]


def workload_nsga(size: int) -> None:
    rng = random.Random(123)
    population = [
        DummyIndividual(fitness=rng.random(), novelty=rng.random()) for _ in range(size)
    ]
    nsga2_rank(population)  # type: ignore[arg-type]


def workload_novelty(size: int) -> None:
    graphs = _graph_fixture(size, 17)
    batch_novelty_scores(graphs)


def workload_fingerprint(size: int) -> None:
    graphs = _graph_fixture(size, 29)
    for graph in graphs:
        graph.fingerprint()
        graph.depth()
        graph.n_params_estimate()


def workload_routing_telemetry(size: int) -> None:
    model = RoutingModel(size, 16)
    for _ in range(16):
        collect_routing_telemetry(model, capture_heatmaps=False)


def workload_choice_concat(size: int) -> None:
    prefix = np.arange(128, dtype=np.int64)
    choices = [np.arange((i % 64) + 1, dtype=np.int64) for i in range(size)]
    for choice in choices:
        concat_choice_tokens(prefix, choice, max_seq_len=160)


WORKLOADS: dict[str, tuple[Callable[[int], None], dict[str, int]]] = {
    "behavior_archive": (
        workload_behavior_archive,
        {"small": 64, "medium": 256, "large": 1024},
    ),
    "nsga_rank": (workload_nsga, {"small": 128, "medium": 512, "large": 2048}),
    "graph_generation": (
        workload_graph_generation,
        {"small": 32, "medium": 128, "large": 512},
    ),
    "novelty_scores": (workload_novelty, {"small": 32, "medium": 128, "large": 512}),
    "graph_fingerprint": (
        workload_fingerprint,
        {"small": 32, "medium": 128, "large": 512},
    ),
    "routing_telemetry": (
        workload_routing_telemetry,
        {"small": 16, "medium": 128, "large": 512},
    ),
    "choice_concat": (
        workload_choice_concat,
        {"small": 256, "medium": 2048, "large": 16384},
    ),
}


def _time_once(fn: Callable[[int], None], size: int) -> float:
    t0 = time.perf_counter()
    fn(size)
    return (time.perf_counter() - t0) * 1000.0


def _time_repeated(
    fn: Callable[[int], None], size: int, repeats: int
) -> dict[str, float]:
    fn(size)
    samples = [_time_once(fn, size) for _ in range(repeats)]
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _profile(fn: Callable[[int], None], size: int, top_n: int) -> list[dict[str, Any]]:
    profile = cProfile.Profile()
    profile.enable()
    fn(size)
    profile.disable()
    stats = pstats.Stats(profile).sort_stats("cumtime")
    rows = []
    for func, stat in list(stats.stats.items()):
        ccalls, ncalls, total, cumulative, callers = stat
        filename, line, name = func
        if "/research/" not in filename:
            continue
        rows.append(
            {
                "file": filename,
                "line": line,
                "function": name,
                "primitive_calls": ccalls,
                "total_calls": ncalls,
                "self_ms": total * 1000.0,
                "cumulative_ms": cumulative * 1000.0,
            }
        )
    rows.sort(key=lambda row: row["cumulative_ms"], reverse=True)
    return rows[:top_n]


def _allocations(fn: Callable[[int], None], size: int) -> dict[str, float | int]:
    tracemalloc.start()
    t0 = time.perf_counter()
    fn(size)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "elapsed_ms": elapsed_ms,
        "current_bytes": current,
        "peak_bytes": peak,
    }


def _line_profile(fn: Callable[[int], None], size: int) -> str | None:
    try:
        from line_profiler import LineProfiler
    except Exception:
        return None

    profiler = LineProfiler()
    profiler.add_function(fn)
    wrapped = profiler(fn)
    wrapped(size)
    output = io.StringIO()
    profiler.print_stats(stream=output)
    return output.getvalue()


def run_audit(repeats: int, profile_size: str, top_n: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "repeats": repeats,
        "profile_size": profile_size,
        "workloads": {},
    }
    for name, (fn, sizes) in WORKLOADS.items():
        workload_result: dict[str, Any] = {"sizes": {}}
        for size_name, size in sizes.items():
            workload_result["sizes"][size_name] = {
                "n": size,
                **_time_repeated(fn, size, repeats),
            }
        selected_size = sizes[profile_size]
        workload_result["cprofile_top"] = _profile(fn, selected_size, top_n)
        workload_result["allocations"] = _allocations(fn, selected_size)
        workload_result["line_profile"] = _line_profile(fn, selected_size)
        result["workloads"][name] = workload_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile research runtime hot paths")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--profile-size", choices=["small", "medium", "large"], default="medium"
    )
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = run_audit(
        repeats=max(1, args.repeats),
        profile_size=args.profile_size,
        top_n=max(1, args.top_n),
    )
    text = json.dumps(data, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
