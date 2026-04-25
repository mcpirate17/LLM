from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

from ..eval.metrics import novelty_score as structural_novelty_score
from ..synthesis.graph import ComputationGraph
from ..synthesis.grammar import GrammarConfig
from ..eval.fingerprint import BehavioralFingerprint
from ._behavior_archive import (
    BehaviorArchive,
)
from .evolution import Individual, EvolutionConfig, evolutionary_search


@dataclass(slots=True)
class NoveltySearchConfig:
    """Configuration for novelty search."""

    archive_size: int = 200
    k_nearest: int = 15  # compare to K nearest neighbors in archive
    archive_threshold: float = 0.3  # min novelty to enter archive
    # Blend of novelty and fitness
    novelty_weight: float = 0.8
    fitness_weight: float = 0.2
    # Evolution params
    population_size: int = 50
    n_generations: int = 20
    grammar_config: Optional[GrammarConfig] = None
    fresh_injection_rate: float = 0.1
    archive_stall_window: int = 3
    archive_stall_patience: int = 2
    archive_threshold_floor: float = 0.18
    adaptation_step: float = 0.05
    max_fresh_injection_rate: float = 0.35
    max_novelty_weight: float = 0.92
    exploit_prob: float = 0.2
    local_mutation_prob: float = 0.3
    debug: bool = False


@dataclass(slots=True)
class NoveltySearchResult:
    """Result of a novelty search run."""

    best_individuals: List[Individual] = field(default_factory=list)
    archive_size: int = 0
    generations_run: int = 0
    total_evaluated: int = 0
    most_novel_fingerprint: Optional[str] = None
    novelty_scores: List[float] = field(default_factory=list)
    adaptation_events: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "n_best": len(self.best_individuals),
            "archive_size": self.archive_size,
            "generations_run": self.generations_run,
            "total_evaluated": self.total_evaluated,
            "most_novel_fingerprint": self.most_novel_fingerprint,
            "novelty_score_range": (
                min(self.novelty_scores) if self.novelty_scores else 0,
                max(self.novelty_scores) if self.novelty_scores else 0,
            ),
            "adaptation_events": list(self.adaptation_events),
        }


def novelty_search(
    fitness_fn: Callable[[ComputationGraph], float],
    fingerprint_fn: Optional[
        Callable[[ComputationGraph], BehavioralFingerprint]
    ] = None,
    config: Optional[NoveltySearchConfig] = None,
    seed: int = 42,
    callback: Optional[Callable[[int, List[Individual], BehaviorArchive], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> NoveltySearchResult:
    """Run novelty search over computation graphs.

    Combines evolutionary search with novelty-based selection.
    Maintains a behavioral archive and rewards programs that
    behave differently from everything seen before.
    """
    if config is None:
        config = NoveltySearchConfig()

    archive = BehaviorArchive(max_size=config.archive_size)
    result = NoveltySearchResult()
    adaptation_state = {
        "last_archive_size": 0,
        "recent_growth": [],
        "stall_count": 0,
        "last_best_fitness": 0.0,
    }
    novelty_fn = _NoveltyEvaluator(
        archive=archive,
        config=config,
        fingerprint_fn=fingerprint_fn,
        score_sink=result.novelty_scores,
    )

    def gen_callback(gen: int, population: List[Individual]):
        result.generations_run = gen + 1
        result.total_evaluated += len(population)
        archive.update_individuals(population)
        _update_adaptive_novelty_policy(
            generation=gen + 1,
            archive=archive,
            population=population,
            ns_config=config,
            evo_config=evo_config,
            state=adaptation_state,
            result=result,
        )
        if callback:
            callback(gen, population, archive)

    # Run evolution with novelty-based selection
    evo_config = EvolutionConfig(
        population_size=config.population_size,
        n_generations=config.n_generations,
        fitness_weight=config.fitness_weight,
        novelty_weight=config.novelty_weight,
        fresh_injection_rate=config.fresh_injection_rate,
        grammar_config=config.grammar_config,
        exploit_prob=config.exploit_prob,
        local_mutation_prob=config.local_mutation_prob,
    )

    final_population = evolutionary_search(
        fitness_fn=fitness_fn,
        novelty_fn=novelty_fn,
        config=evo_config,
        seed=seed,
        callback=gen_callback,
        stop_check=stop_check,
        archive=archive,
    )

    result.best_individuals = final_population[:10]
    result.archive_size = archive.size()
    if final_population:
        result.most_novel_fingerprint = max(
            final_population, key=lambda x: x.novelty
        ).fingerprint

    return result


class _NoveltyEvaluator:
    __slots__ = ("archive", "config", "fingerprint_fn", "score_sink")

    def __init__(
        self,
        *,
        archive: BehaviorArchive,
        config: NoveltySearchConfig,
        fingerprint_fn: Optional[Callable[[ComputationGraph], BehavioralFingerprint]],
        score_sink: List[float],
    ) -> None:
        self.archive = archive
        self.config = config
        self.fingerprint_fn = fingerprint_fn
        self.score_sink = score_sink

    def __call__(
        self, graph: ComputationGraph, population: List[ComputationGraph]
    ) -> float:
        del population
        scores = self.batch_scores([graph])
        return scores[0] if scores else 0.0

    def batch_scores(self, graphs: List[ComputationGraph]) -> List[float]:
        if self.fingerprint_fn is None:
            scores = [_structural_novelty(graph) for graph in graphs]
            self.score_sink.extend(scores)
            return scores

        behaviors: List[BehavioralFingerprint | None] = []
        scores: List[float] = [0.0] * len(graphs)
        behavioral_indices: List[int] = []
        behavioral_fps: List[BehavioralFingerprint] = []

        for idx, graph in enumerate(graphs):
            try:
                behavior = self.fingerprint_fn(graph)
            except Exception as exc:
                if self.config.debug:
                    logger.exception(
                        "DEBUG novelty: fingerprint computation failed for %s",
                        graph.fingerprint()[:16],
                    )
                else:
                    logger.warning("Novelty fingerprint failed: %s", exc)
                behaviors.append(None)
                continue

            behaviors.append(behavior)
            if behavior is None:
                scores[idx] = _structural_novelty(graph)
                if self.config.debug:
                    logger.info(
                        "DEBUG novelty: fingerprint_fn returned None for %s, falling back to structural",
                        graph.fingerprint()[:16],
                    )
                continue
            behavioral_indices.append(idx)
            behavioral_fps.append(behavior)

        archive_scores = self.archive.novelty_of_many(
            behavioral_fps, k=self.config.k_nearest
        )
        archive_additions: List[tuple[str, BehavioralFingerprint, float]] = []
        for local_idx, novelty in enumerate(archive_scores):
            idx = behavioral_indices[local_idx]
            graph = graphs[idx]
            behavior = behaviors[idx]
            if novelty is None:
                scores[idx] = _structural_novelty(graph)
                if self.config.debug:
                    logger.info(
                        "DEBUG novelty: behavior_vector returned None for %s, falling back to structural",
                        graph.fingerprint()[:16],
                    )
                continue

            novelty_value = float(novelty)
            scores[idx] = novelty_value
            if behavior is not None and novelty_value >= float(
                self.config.archive_threshold
            ):
                archive_additions.append((graph.fingerprint(), behavior, novelty_value))

        for graph_hash, behavior, novelty_value in archive_additions:
            self.archive.add(graph_hash, behavior)
            if self.config.debug:
                logger.info(
                    "DEBUG novelty: archived %s (novelty=%.3f >= threshold=%.3f, archive_size=%d)",
                    graph_hash[:16],
                    novelty_value,
                    self.config.archive_threshold,
                    self.archive.size(),
                )

        self.score_sink.extend(scores)
        return scores


def _structural_novelty(graph: ComputationGraph) -> float:
    return structural_novelty_score(graph).structural_novelty


def _update_adaptive_novelty_policy(
    *,
    generation: int,
    archive: BehaviorArchive,
    population: List[Individual],
    ns_config: NoveltySearchConfig,
    evo_config: EvolutionConfig,
    state: Dict[str, object],
    result: NoveltySearchResult,
) -> None:
    """Increase exploration pressure when archive growth stalls."""
    archive_size = archive.size()
    best_fitness = max((ind.fitness for ind in population), default=0.0)
    best_novelty = max((ind.novelty for ind in population), default=0.0)
    growth = archive_size - int(state["last_archive_size"])
    recent_growth = list(state["recent_growth"])
    recent_growth.append(growth)
    window = max(1, int(ns_config.archive_stall_window))
    if len(recent_growth) > window:
        recent_growth = recent_growth[-window:]
    state["recent_growth"] = recent_growth
    state["last_archive_size"] = archive_size

    improved = best_fitness > float(state["last_best_fitness"]) + 1e-6
    state["last_best_fitness"] = max(float(state["last_best_fitness"]), best_fitness)
    archive_stalled = len(recent_growth) >= window and sum(recent_growth) <= 0
    state["stall_count"] = (
        int(state["stall_count"]) + 1 if archive_stalled and not improved else 0
    )
    if int(state["stall_count"]) < max(1, int(ns_config.archive_stall_patience)):
        return

    step = max(0.01, float(ns_config.adaptation_step))
    evo_config.fresh_injection_rate = min(
        float(ns_config.max_fresh_injection_rate),
        evo_config.fresh_injection_rate + step,
    )
    evo_config.mutation_rate = min(0.95, evo_config.mutation_rate + step)
    evo_config.crossover_rate = max(0.05, evo_config.crossover_rate - step)
    evo_config.novelty_weight = min(
        float(ns_config.max_novelty_weight),
        evo_config.novelty_weight + step,
    )
    evo_config.fitness_weight = max(0.05, 1.0 - evo_config.novelty_weight)
    ns_config.archive_threshold = max(
        float(ns_config.archive_threshold_floor),
        ns_config.archive_threshold - step,
    )
    adaptation = {
        "generation": float(generation),
        "archive_size": float(archive_size),
        "best_fitness": float(best_fitness),
        "best_novelty": float(best_novelty),
        "fresh_injection_rate": float(evo_config.fresh_injection_rate),
        "mutation_rate": float(evo_config.mutation_rate),
        "crossover_rate": float(evo_config.crossover_rate),
        "novelty_weight": float(evo_config.novelty_weight),
        "fitness_weight": float(evo_config.fitness_weight),
        "archive_threshold": float(ns_config.archive_threshold),
    }
    result.adaptation_events.append(adaptation)
    state["stall_count"] = 0
    logger.info(
        "Novelty adaptation at gen %d: archive=%d best_fit=%.3f best_novelty=%.3f fresh=%.2f novelty_w=%.2f threshold=%.2f",
        generation,
        archive_size,
        best_fitness,
        best_novelty,
        evo_config.fresh_injection_rate,
        evo_config.novelty_weight,
        ns_config.archive_threshold,
    )
