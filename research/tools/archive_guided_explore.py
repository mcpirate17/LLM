"""Archive-guided exploration: generate the diversity the population lacks (M4).

The runnable closed loop for the diversity generator
(``diversity_generator_charter_2026-06-03.md`` M4). The synthesis library
provides the pieces — ``quality_diversity.MapElitesArchive`` (behavior space),
``archive_guided`` (empty niches → ``GrammarConfig.exploration_targets``) — and
this tool drives them:

    seed pool (default grammar) → measure descriptors → MAP-Elites archive
      → archive-guided grammar → guided wave → measure → archive (coverage↑) → …

So instead of one default-grammar pass that collapses onto the grammar-favored
region, later waves are steered into the empty behavior niches. The output is a
jsonl of measured candidates (the guided-wave graphs that landed in previously
empty niches first) for the screener / grader to pick up.

The orchestration core (``explore_empty_niches``) takes injected generate /
measure / fitness callables, so it is unit-testable without torch; ``main``
wires the real grammar generator and the measured-descriptor instrument.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from research.synthesis.archive_guided import exploration_config_from_archive
from research.synthesis.quality_diversity import (
    BehaviorAxis,
    MapElitesArchive,
    default_behavior_axes,
)

logger = logging.getLogger(__name__)

# (config_or_None, seed) -> graph dict, or None if generation failed/invalid.
GenerateFn = Callable[[Any, int], "dict[str, Any] | None"]
# graph dict -> measured behavior descriptors, or None if unmeasurable.
MeasureFn = Callable[["dict[str, Any]"], "Mapping[str, float] | None"]
# descriptors -> scalar fitness for the archive.
FitnessFn = Callable[["Mapping[str, float]"], float]


@dataclass(frozen=True, slots=True)
class NicheCandidate:
    """One measured candidate placed in the behavior archive."""

    key: str
    wave: int
    niche: tuple[int, ...]
    fitness: float
    descriptors: Mapping[str, float]
    graph: Any


@dataclass(slots=True)
class ExplorationResult:
    """Outcome of an archive-guided exploration run."""

    coverage_trajectory: list[float] = field(default_factory=list)
    target_ops_per_wave: list[tuple[str, ...]] = field(default_factory=list)
    filled: int = 0
    total_cells: int = 0
    measured: int = 0
    unmeasurable: int = 0
    invalid: int = 0
    candidates: list[NicheCandidate] = field(default_factory=list)


def _ingest(
    *,
    archive: MapElitesArchive,
    wave: int,
    count: int,
    seed0: int,
    config: Any,
    generate_fn: GenerateFn,
    measure_fn: MeasureFn,
    fitness_fn: FitnessFn,
    key_fn: Callable[[dict[str, Any]], str],
    result: ExplorationResult,
) -> int:
    """Generate ``count`` graphs with ``config``, measure, and add to the archive.

    Returns the number of niches newly filled by this wave (coverage gain in cells).
    """

    before = archive.filled
    for offset in range(count):
        graph = generate_fn(config, seed0 + offset)
        if graph is None:
            result.invalid += 1
            continue
        descriptors = measure_fn(graph)
        if descriptors is None:
            result.unmeasurable += 1
            continue
        result.measured += 1
        try:
            niche = archive.niche_for(descriptors)
        except KeyError:
            # measure_fn returned a descriptor vector missing a behavior axis;
            # that is a measurement contract bug, not a candidate to drop silently.
            result.unmeasurable += 1
            continue
        fitness = fitness_fn(descriptors)
        archive.add(key_fn(graph), descriptors, fitness, payload=graph)
        result.candidates.append(
            NicheCandidate(
                key=key_fn(graph),
                wave=wave,
                niche=niche,
                fitness=fitness,
                descriptors=dict(descriptors),
                graph=graph,
            )
        )
    return archive.filled - before


def explore_empty_niches(
    *,
    generate_fn: GenerateFn,
    measure_fn: MeasureFn,
    fitness_fn: FitnessFn,
    key_fn: Callable[[dict[str, Any]], str],
    seed_pool: int,
    wave_pool: int,
    waves: int = 2,
    seed0: int = 0,
    axes: Sequence[BehaviorAxis] | None = None,
    guidance_kwargs: Mapping[str, Any] | None = None,
) -> ExplorationResult:
    """Run the seed → archive → guided-wave loop and return the trajectory.

    Args:
        seed_pool: graphs to generate with the default grammar (wave 0).
        wave_pool: graphs to generate per guided wave.
        waves: number of guided waves after the seed wave. Stops early when the
            archive has no reachable empty niche left to target.
        seed0: base seed; each generated graph uses a distinct seed offset.
        axes: behavior space (defaults to the coarse 27-niche archive).
        guidance_kwargs: forwarded to ``archive_guidance`` (radius, base_boost…).
    """

    archive = MapElitesArchive(axes=tuple(axes) if axes else default_behavior_axes())
    result = ExplorationResult(total_cells=archive.total_cells)

    next_seed = seed0
    _ingest(
        archive=archive,
        wave=0,
        count=seed_pool,
        seed0=next_seed,
        config=None,
        generate_fn=generate_fn,
        measure_fn=measure_fn,
        fitness_fn=fitness_fn,
        key_fn=key_fn,
        result=result,
    )
    next_seed += seed_pool
    result.coverage_trajectory.append(archive.coverage())

    gkw = dict(guidance_kwargs or {})
    for wave in range(1, waves + 1):
        config, guidance = exploration_config_from_archive(archive, **gkw)
        result.target_ops_per_wave.append(tuple(sorted(guidance.target_ops)))
        if config is None:
            logger.info("wave %d: no reachable empty niche — stopping early", wave)
            break
        gained = _ingest(
            archive=archive,
            wave=wave,
            count=wave_pool,
            seed0=next_seed,
            config=config,
            generate_fn=generate_fn,
            measure_fn=measure_fn,
            fitness_fn=fitness_fn,
            key_fn=key_fn,
            result=result,
        )
        next_seed += wave_pool
        result.coverage_trajectory.append(archive.coverage())
        logger.info(
            "wave %d: boost=%.2f targets=%d filled +%d → coverage %.3f",
            wave,
            guidance.boost_factor,
            len(guidance.target_ops),
            gained,
            archive.coverage(),
        )

    result.filled = archive.filled
    return result


# --------------------------------------------------------------------------- #
# CLI — wires the real grammar generator + measured-descriptor instrument
# --------------------------------------------------------------------------- #
def _real_generate_fn() -> GenerateFn:
    from research.synthesis.grammar import GrammarConfig, generate_layer_graph

    def generate(config: Any, seed: int) -> dict[str, Any] | None:
        try:
            graph = generate_layer_graph(config or GrammarConfig(), seed=seed)
        except Exception:  # noqa: BLE001 - invalid grammar samples are expected, counted
            return None
        return graph.to_dict()

    return generate


def _real_measure_fn(device: str | None) -> MeasureFn:
    from research.tools.measured_descriptors import MeasuredDescriptorExtractor

    mdx = MeasuredDescriptorExtractor(device=device, n_seeds=1)

    def measure(graph: dict[str, Any]) -> Mapping[str, float] | None:
        return mdx.descriptors(json.dumps(graph, separators=(",", ":")))

    return measure


def _real_fitness_fn() -> FitnessFn:
    from research.tools.measured_descriptors import capability_score_from_descriptors

    return lambda d: capability_score_from_descriptors(dict(d))


def _graph_key(graph: dict[str, Any]) -> str:
    return graph.get("fingerprint") or json.dumps(graph, sort_keys=True)


def _write_candidates(out: Path, result: ExplorationResult) -> None:
    """Write guided-wave candidates (diversity-targeted) for the screener/grader."""

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for c in result.candidates:
            if c.wave == 0:
                continue  # seed-wave graphs are the baseline, not new diversity
            f.write(
                json.dumps(
                    {
                        "fingerprint": c.key,
                        "wave": c.wave,
                        "niche": list(c.niche),
                        "fitness": round(c.fitness, 6),
                        "descriptors": {
                            k: round(v, 6) for k, v in c.descriptors.items()
                        },
                        "graph": c.graph,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-pool", type=int, default=200)
    parser.add_argument("--wave-pool", type=int, default=200)
    parser.add_argument("--waves", type=int, default=2)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--out",
        default="research/reports/archive_guided_explore.jsonl",
        help="candidate jsonl (guided-wave graphs) for the screener/grader",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    t0 = time.time()
    result = explore_empty_niches(
        generate_fn=_real_generate_fn(),
        measure_fn=_real_measure_fn(args.device),
        fitness_fn=_real_fitness_fn(),
        key_fn=_graph_key,
        seed_pool=args.seed_pool,
        wave_pool=args.wave_pool,
        waves=args.waves,
        seed0=args.seed0,
        guidance_kwargs={"radius": args.radius},
    )
    out = Path(args.out)
    _write_candidates(out, result)
    report = {
        "elapsed_s": round(time.time() - t0, 1),
        "coverage_trajectory": [round(c, 4) for c in result.coverage_trajectory],
        "filled": result.filled,
        "total_cells": result.total_cells,
        "target_ops_per_wave": [list(t) for t in result.target_ops_per_wave],
        "measured": result.measured,
        "unmeasurable": result.unmeasurable,
        "invalid": result.invalid,
        "guided_candidates": sum(1 for c in result.candidates if c.wave > 0),
        "out": out.as_posix(),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
