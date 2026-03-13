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
import math
import random as _random_module
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from ..synthesis.graph import ComputationGraph
from ..synthesis.grammar import GrammarConfig
from ..eval.fingerprint import BehavioralFingerprint
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

    Uses NumPy for vectorized distance computation and reservoir sampling
    for eviction to maintain historical diversity without recency bias.
    """

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.entries: List[Tuple[str, BehavioralFingerprint]] = []
        # Cached feature matrix for vectorized distance
        self._feature_matrix: Optional[np.ndarray] = None
        self._total_seen = 0
        self._rng = _random_module.Random(42)

    def _update_cache(self):
        """Update the internal NumPy feature matrix."""
        if not self.entries:
            self._feature_matrix = None
            return
        vectors = [_behavior_vector(fp) for _, fp in self.entries]
        self._feature_matrix = np.array(vectors, dtype=np.float32)

    def _append_to_cache(self, behavior: BehavioralFingerprint) -> None:
        """Append a single vector to the feature matrix (avoids full rebuild)."""
        vec = np.array(_behavior_vector(behavior), dtype=np.float32).reshape(1, -1)
        if self._feature_matrix is None:
            self._feature_matrix = vec
        else:
            self._feature_matrix = np.vstack([self._feature_matrix, vec])

    def _replace_in_cache(self, idx: int, behavior: BehavioralFingerprint) -> None:
        """Replace a single row in the feature matrix (avoids full rebuild)."""
        if self._feature_matrix is None:
            self._update_cache()
            return
        vec = np.array(_behavior_vector(behavior), dtype=np.float32)
        self._feature_matrix[idx] = vec

    def add(self, graph_hash: str, behavior: BehavioralFingerprint):
        """Add a behavior to the archive using reservoir sampling."""
        self._total_seen += 1
        if len(self.entries) < self.max_size:
            self.entries.append((graph_hash, behavior))
            self._append_to_cache(behavior)
        else:
            j = self._rng.randint(0, self._total_seen - 1)
            if j < self.max_size:
                self.entries[j] = (graph_hash, behavior)
                self._replace_in_cache(j, behavior)

    def novelty_of(self, behavior: BehavioralFingerprint, k: int = 15) -> float:
        """Compute novelty of a behavior relative to the archive.

        Novelty = mean distance to K nearest neighbors in archive.
        Vectorized via NumPy for high-performance orchestration.
        """
        if not self.entries or self._feature_matrix is None:
            return 1.0

        target = np.array(_behavior_vector(behavior), dtype=np.float32)
        
        # Vectorized Euclidean distance across entire archive
        # d(x, y) = sqrt(1/N * sum((x-y)^2))
        diff = self._feature_matrix - target
        dist_sq = np.mean(np.square(diff), axis=1)
        distances = np.sqrt(dist_sq)

        # Get K nearest neighbors
        k = min(k, len(distances))
        if k == 0:
            return 1.0
        
        # partition is O(N) compared to sort O(N log N)
        if len(distances) > k:
            k_nearest = np.partition(distances, k-1)[:k]
        else:
            k_nearest = distances
            
        return float(np.mean(k_nearest))

    def size(self) -> int:
        return len(self.entries)


def _behavior_distance(a: BehavioralFingerprint, b: BehavioralFingerprint) -> float:
    """RMS distance between sanitized behavior vectors (stable range: [0, 1])."""
    fa = np.array(_behavior_vector(a), dtype=np.float32)
    fb = np.array(_behavior_vector(b), dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(fa - fb))))


def _behavior_vector(fp: BehavioralFingerprint) -> List[float]:
    """Extract novelty features with robust sanitization for outliers/NaNs."""
    raw_features = [
        fp.interaction_locality, fp.interaction_sparsity,
        fp.interaction_symmetry, fp.interaction_hierarchy,
        fp.isotropy, fp.rank_ratio,
        fp.sensitivity_uniformity,
        fp.cka_vs_transformer, fp.cka_vs_ssm, fp.cka_vs_conv,
    ]
    return [_sanitize_unit_feature(v) for v in raw_features]


def _sanitize_unit_feature(value: float) -> float:
    """Clip expected unit-scale features to [0, 1], replacing invalid values."""
    try:
        v = float(value)
    except Exception:
        return 0.5
    if not math.isfinite(v):
        return 0.5
    return min(1.0, max(0.0, v))


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
            # fingerprint_fn may return None on failure — fall back to structural novelty
            if behavior is None:
                from ..eval.metrics import novelty_score as struct_novelty
                m = struct_novelty(graph)
                return m.structural_novelty

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
        stop_check=stop_check,
    )

    result.best_individuals = final_population[:10]
    result.archive_size = archive.size()
    if final_population:
        result.most_novel_fingerprint = max(
            final_population, key=lambda x: x.novelty
        ).fingerprint

    return result
