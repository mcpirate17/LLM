"""
Novelty Search

Instead of optimizing for fitness (lower loss), reward BEHAVIORAL NOVELTY.
This prevents convergence to known solutions and encourages exploration
of genuinely new computation patterns.

Key idea: maintain an archive of behaviors seen so far. New programs are
rewarded based on how DIFFERENT they are from everything in the archive.
"""

from __future__ import annotations

import logging
import random as _random_module
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from ..synthesis.graph import ComputationGraph
from ..synthesis.grammar import GrammarConfig, generate_layer_graph
from ..eval.fingerprint import BehavioralFingerprint, compute_fingerprint
from .evolution import Individual, EvolutionConfig, evolutionary_search


@dataclass
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


class BehaviorArchive:
    """Archive of behavioral fingerprints seen during search.

    Uses reservoir sampling for eviction so that all time periods are
    represented equally, avoiding the recency bias of FIFO eviction.
    """

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.entries: List[Tuple[str, BehavioralFingerprint]] = []
        self._total_seen = 0  # total items offered (for reservoir sampling)
        self._rng = _random_module.Random(42)

    def add(self, graph_hash: str, behavior: BehavioralFingerprint):
        """Add a behavior to the archive using reservoir sampling."""
        self._total_seen += 1
        if len(self.entries) < self.max_size:
            self.entries.append((graph_hash, behavior))
        else:
            # Reservoir sampling (Algorithm R): each new item replaces a
            # random existing entry with probability max_size / total_seen.
            j = self._rng.randint(0, self._total_seen - 1)
            if j < self.max_size:
                self.entries[j] = (graph_hash, behavior)

    def novelty_of(self, behavior: BehavioralFingerprint, k: int = 15) -> float:
        """Compute novelty of a behavior relative to the archive.

        Novelty = mean distance to K nearest neighbors in archive.
        """
        if not self.entries:
            return 1.0  # Everything is novel when archive is empty

        distances = []
        for _, archived in self.entries:
            dist = _behavior_distance(behavior, archived)
            distances.append(dist)

        distances.sort()
        k = min(k, len(distances))
        return sum(distances[:k]) / k

    def size(self) -> int:
        return len(self.entries)


def _behavior_distance(a: BehavioralFingerprint, b: BehavioralFingerprint) -> float:
    """Euclidean distance between two behavioral fingerprints."""
    features_a = [
        a.interaction_locality, a.interaction_sparsity,
        a.interaction_symmetry, a.interaction_hierarchy,
        a.isotropy, a.rank_ratio,
        a.sensitivity_uniformity,
        a.cka_vs_transformer, a.cka_vs_ssm, a.cka_vs_conv,
    ]
    features_b = [
        b.interaction_locality, b.interaction_sparsity,
        b.interaction_symmetry, b.interaction_hierarchy,
        b.isotropy, b.rank_ratio,
        b.sensitivity_uniformity,
        b.cka_vs_transformer, b.cka_vs_ssm, b.cka_vs_conv,
    ]
    return sum((fa - fb) ** 2 for fa, fb in zip(features_a, features_b)) ** 0.5


@dataclass
class NoveltySearchResult:
    """Result of a novelty search run."""
    best_individuals: List[Individual] = field(default_factory=list)
    archive_size: int = 0
    generations_run: int = 0
    total_evaluated: int = 0
    most_novel_fingerprint: Optional[str] = None
    novelty_scores: List[float] = field(default_factory=list)

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
        }


def novelty_search(
    fitness_fn: Callable[[ComputationGraph], float],
    fingerprint_fn: Optional[Callable[[ComputationGraph], BehavioralFingerprint]] = None,
    config: Optional[NoveltySearchConfig] = None,
    seed: int = 42,
    callback: Optional[Callable[[int, List[Individual], BehaviorArchive], None]] = None,
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

    def novelty_fn(graph: ComputationGraph,
                   population: List[ComputationGraph]) -> float:
        """Compute novelty score for a graph."""
        if fingerprint_fn is None:
            # Structural novelty only
            from ..eval.metrics import novelty_score as struct_novelty
            m = struct_novelty(graph)
            return m.structural_novelty

        try:
            behavior = fingerprint_fn(graph)
            novelty = archive.novelty_of(behavior, k=config.k_nearest)

            # Add to archive if novel enough
            if novelty >= config.archive_threshold:
                archive.add(graph.fingerprint(), behavior)

            result.novelty_scores.append(novelty)
            return novelty
        except Exception as e:
            logger.warning("Novelty computation failed: %s", e)
            return 0.0

    def gen_callback(gen: int, population: List[Individual]):
        result.generations_run = gen + 1
        result.total_evaluated += len(population)
        if callback:
            callback(gen, population, archive)

    # Run evolution with novelty-based selection
    evo_config = EvolutionConfig(
        population_size=config.population_size,
        n_generations=config.n_generations,
        fitness_weight=config.fitness_weight,
        novelty_weight=config.novelty_weight,
        grammar_config=config.grammar_config,
    )

    final_population = evolutionary_search(
        fitness_fn=fitness_fn,
        novelty_fn=novelty_fn,
        config=evo_config,
        seed=seed,
        callback=gen_callback,
    )

    result.best_individuals = final_population[:10]
    result.archive_size = archive.size()
    if final_population:
        result.most_novel_fingerprint = max(
            final_population, key=lambda x: x.novelty
        ).fingerprint

    return result
