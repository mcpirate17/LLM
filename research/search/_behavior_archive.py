from __future__ import annotations

import math
import random as _random_module
from heapq import nlargest
from typing import Iterable, List, Optional, Tuple

import numpy as np

from ..eval.fingerprint import BehavioralFingerprint
from .evolution import Individual
from .native_metrics import (
    archive_mean_k_nearest,
    pairwise_median_and_neighbor_counts,
)


class BehaviorArchive:
    """Archive of behavioral fingerprints seen during search."""

    _INITIAL_CAPACITY = 256
    _FEATURE_DIM = 16
    _MAX_BATCH_DISTANCE_ELEMENTS = 1_000_000
    __slots__ = (
        "max_size",
        "_graph_hashes",
        "_hash_to_index",
        "_individuals",
        "_feature_buf",
        "_capacity",
        "_size",
        "_total_seen",
        "_rng",
        "_density_cache",
        "_exploit_order_cache",
    )

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._graph_hashes: List[str] = []
        self._hash_to_index: dict[str, int] = {}
        self._individuals: List[Optional[Individual]] = []
        self._feature_buf: np.ndarray = np.empty(
            (self._INITIAL_CAPACITY, self._FEATURE_DIM), dtype=np.float32
        )
        self._capacity: int = self._INITIAL_CAPACITY
        self._size: int = 0
        self._total_seen = 0
        self._rng = _random_module.Random(42)
        self._density_cache: Tuple[float, np.ndarray] | None = None
        self._exploit_order_cache: list[int] | None = None

    @property
    def _feature_matrix(self) -> np.ndarray | None:
        if self._size == 0:
            return None
        return self._feature_buf[: self._size]

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        new_cap = self._capacity
        while new_cap < needed:
            new_cap *= 2
        new_buf = np.empty((new_cap, self._FEATURE_DIM), dtype=np.float32)
        new_buf[: self._size] = self._feature_buf[: self._size]
        self._feature_buf = new_buf
        self._capacity = new_cap

    def _append_to_cache(self, vector: np.ndarray) -> None:
        self._ensure_capacity(self._size + 1)
        self._feature_buf[self._size] = vector
        self._size += 1
        self._density_cache = None
        self._exploit_order_cache = None

    def _replace_in_cache(self, idx: int, vector: np.ndarray) -> None:
        self._feature_buf[idx] = vector
        self._density_cache = None
        self._exploit_order_cache = None

    def add(
        self,
        graph_hash: str,
        behavior: BehavioralFingerprint,
        individual: Optional[Individual] = None,
    ) -> None:
        vector = _behavior_array(behavior)
        if vector is None:
            return

        existing_idx = self._hash_to_index.get(graph_hash)
        if existing_idx is not None:
            self._individuals[existing_idx] = individual
            self._replace_in_cache(existing_idx, vector)
            return

        self._total_seen += 1
        if self._size < self.max_size:
            self._graph_hashes.append(graph_hash)
            self._hash_to_index[graph_hash] = self._size
            self._individuals.append(individual)
            self._append_to_cache(vector)
            return

        j = self._rng.randint(0, self._total_seen - 1)
        if j < self.max_size:
            old_hash = self._graph_hashes[j]
            self._hash_to_index.pop(old_hash, None)
            self._graph_hashes[j] = graph_hash
            self._hash_to_index[graph_hash] = j
            self._individuals[j] = individual
            self._replace_in_cache(j, vector)

    def novelty_of(self, behavior: BehavioralFingerprint, k: int = 15) -> float | None:
        fm = self._feature_matrix
        if fm is None:
            return 1.0

        target = _behavior_array(behavior)
        if target is None:
            return None

        k = min(k, int(fm.shape[0]))
        if k == 0:
            return 1.0

        native_mean = archive_mean_k_nearest(fm, target, k)
        if native_mean is not None:
            return native_mean

        diff = fm - target
        distances = np.sqrt(np.mean(np.square(diff), axis=1))
        k_nearest = (
            np.partition(distances, k - 1)[:k] if len(distances) > k else distances
        )
        return float(np.mean(k_nearest))

    def novelty_of_many(
        self, behaviors: Iterable[BehavioralFingerprint], k: int = 15
    ) -> List[float | None]:
        targets = _behavior_matrix(behaviors)
        if targets is None:
            return []

        valid_mask = ~np.isnan(targets[:, 0])
        results: List[float | None] = [None] * int(targets.shape[0])
        if not np.any(valid_mask):
            return results

        fm = self._feature_matrix
        valid_indices = np.where(valid_mask)[0]
        if fm is None or fm.shape[0] == 0:
            for idx in valid_indices:
                results[int(idx)] = 1.0
            return results

        used_k = min(max(int(k), 1), int(fm.shape[0]))
        max_rows = max(
            1,
            self._MAX_BATCH_DISTANCE_ELEMENTS
            // max(1, int(fm.shape[0]) * self._FEATURE_DIM),
        )
        for start in range(0, int(valid_indices.size), max_rows):
            chunk_indices = valid_indices[start : start + max_rows]
            chunk = targets[chunk_indices]
            diffs = chunk[:, np.newaxis, :] - fm[np.newaxis, :, :]
            distances = np.sqrt(np.mean(np.square(diffs), axis=2))
            if distances.shape[1] > used_k:
                nearest = np.partition(distances, used_k - 1, axis=1)[:, :used_k]
            else:
                nearest = distances
            means = np.mean(nearest, axis=1)
            for idx, novelty in zip(chunk_indices, means, strict=True):
                results[int(idx)] = float(novelty)
        return results

    def update_individuals(self, population: List[Individual]) -> None:
        by_hash = {ind.fingerprint: ind for ind in population}
        for i, graph_hash in enumerate(self._graph_hashes):
            ind = by_hash.get(graph_hash)
            if ind is not None:
                self._individuals[i] = ind
        self._exploit_order_cache = None

    def size(self) -> int:
        return self._size

    def top_by_fitness(self, k: int = 5) -> List[Individual]:
        return nlargest(
            k,
            (ind for ind in self._individuals if ind is not None),
            key=lambda ind: ind.fitness,
        )

    def suggest_exploit_target(self, k: int = 3) -> List[Individual]:
        fm = self._feature_matrix
        if fm is None or self._size < 2:
            return self.top_by_fitness(k)

        cached_stats = self._density_cache
        if cached_stats is not None:
            median_dist, neighbor_counts = cached_stats
        else:
            native_stats = pairwise_median_and_neighbor_counts(fm)
            if native_stats is not None:
                median_dist, neighbor_counts = native_stats
            else:
                diffs = fm[:, np.newaxis, :] - fm[np.newaxis, :, :]
                all_dists = np.sqrt(np.mean(np.square(diffs), axis=2))
                mask = ~np.eye(self._size, dtype=bool)
                median_dist = float(np.median(all_dists[mask]))
                neighbor_counts = np.sum(all_dists < median_dist, axis=1) - 1
            self._density_cache = (float(median_dist), neighbor_counts)

        if median_dist <= 0:
            return self.top_by_fitness(k)

        ordered = self._exploit_order_cache
        if ordered is None:
            scores = np.full(self._size, -np.inf, dtype=np.float32)
            for i, ind in enumerate(self._individuals[: self._size]):
                if ind is None:
                    continue
                scores[i] = float(ind.fitness) - 0.1 * float(neighbor_counts[i])
            ordered = np.argsort(scores)[::-1].tolist()
            self._exploit_order_cache = ordered
        targets: List[Individual] = []
        for idx in ordered:
            ind = self._individuals[idx]
            if ind is None:
                continue
            targets.append(ind)
            if len(targets) >= k:
                break
        return targets


def _behavior_distance(a: BehavioralFingerprint, b: BehavioralFingerprint) -> float:
    va = _behavior_array(a)
    vb = _behavior_array(b)
    if va is None or vb is None:
        return 1.0
    return float(np.sqrt(np.mean(np.square(va - vb))))


def _behavior_array(fp: BehavioralFingerprint) -> np.ndarray | None:
    if _all_behavior_features_missing(fp):
        return None
    out = np.empty(BehaviorArchive._FEATURE_DIM, dtype=np.float32)
    out[0] = _sanitize_unit_feature(fp.interaction_locality)
    out[1] = _sanitize_unit_feature(fp.interaction_sparsity)
    out[2] = _sanitize_unit_feature(fp.interaction_symmetry)
    out[3] = _sanitize_unit_feature(fp.interaction_hierarchy)
    out[4] = _sanitize_unit_feature(fp.isotropy)
    out[5] = _sanitize_unit_feature(fp.rank_ratio)
    out[6] = _sanitize_unit_feature(fp.sensitivity_uniformity)
    out[7] = _sanitize_unit_feature(fp.cka_vs_transformer)
    out[8] = _sanitize_unit_feature(fp.cka_vs_ssm)
    out[9] = _sanitize_unit_feature(fp.cka_vs_conv)
    out[10] = _sanitize_scaled_feature(fp.jacobian_spectral_norm, scale=5.0)
    out[11] = _sanitize_scaled_feature(fp.jacobian_effective_rank, scale=16.0)
    out[12] = _sanitize_unit_feature(fp.routing_selectivity)
    out[13] = _sanitize_scaled_feature(fp.routing_compute_ratio, scale=2.0)
    out[14] = _sanitize_unit_feature(fp.hierarchy_fitness)
    out[15] = _sanitize_scaled_feature(fp.gromov_delta, scale=0.3)
    return out


def _behavior_matrix(
    fps: Iterable[BehavioralFingerprint],
) -> np.ndarray | None:
    rows = list(fps)
    if not rows:
        return None
    out = np.empty((len(rows), BehaviorArchive._FEATURE_DIM), dtype=np.float32)
    for idx, fp in enumerate(rows):
        vector = _behavior_array(fp)
        if vector is None:
            out[idx, :] = np.nan
        else:
            out[idx, :] = vector
    return out


def _all_behavior_features_missing(fp: BehavioralFingerprint) -> bool:
    return (
        fp.interaction_locality is None
        and fp.interaction_sparsity is None
        and fp.interaction_symmetry is None
        and fp.interaction_hierarchy is None
        and fp.isotropy is None
        and fp.rank_ratio is None
        and fp.sensitivity_uniformity is None
        and fp.cka_vs_transformer is None
        and fp.cka_vs_ssm is None
        and fp.cka_vs_conv is None
        and fp.jacobian_spectral_norm is None
        and fp.jacobian_effective_rank is None
        and fp.routing_selectivity is None
        and fp.routing_compute_ratio is None
        and fp.hierarchy_fitness is None
        and fp.gromov_delta is None
    )


def _behavior_vector(fp: BehavioralFingerprint) -> List[float] | None:
    vector = _behavior_array(fp)
    if vector is None:
        return None
    return vector.tolist()


def _sanitize_unit_feature(value: float | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float):
        v = value
    else:
        try:
            v = float(value)
        except Exception:
            return 0.0
    if not math.isfinite(v):
        return 0.0
    if v <= 0.0:
        return 0.0
    if v >= 1.0:
        return 1.0
    return v


def _sanitize_scaled_feature(value: float | None, scale: float) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float):
        v = value
    else:
        try:
            v = float(value)
        except Exception:
            return 0.0
    if not math.isfinite(v) or v < 0:
        return 0.0
    return v / (v + scale)
