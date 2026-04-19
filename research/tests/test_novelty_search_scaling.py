import pytest
import math
import unittest
import numpy as np

from research.eval.fingerprint import BehavioralFingerprint
from research.search.native_metrics import (
    archive_mean_k_nearest,
    load_native_search_metrics_lib,
    pairwise_median_and_neighbor_counts,
    reset_native_search_metrics_lib,
    topk_nearest_indices,
)
from research.search._behavior_archive import BehaviorArchive, _behavior_distance

pytestmark = pytest.mark.unit


class TestNoveltySearchScaling(unittest.TestCase):
    def test_behavior_distance_clamps_extreme_feature_values(self):
        a = BehavioralFingerprint()
        b = BehavioralFingerprint(interaction_symmetry=142003.109)

        dist = _behavior_distance(a, b)

        self.assertTrue(math.isfinite(dist))
        self.assertLessEqual(dist, 1.0)
        self.assertGreater(dist, 0.0)

    def test_behavior_distance_handles_non_finite_values(self):
        a = BehavioralFingerprint(cka_vs_transformer=float("nan"))
        b = BehavioralFingerprint(cka_vs_transformer=float("inf"))

        dist = _behavior_distance(a, b)

        self.assertTrue(math.isfinite(dist))
        self.assertGreaterEqual(dist, 0.0)
        self.assertLessEqual(dist, 1.0)

    def test_archive_novelty_remains_bounded_with_outliers(self):
        archive = BehaviorArchive(max_size=10)
        archive.add("a", BehavioralFingerprint(interaction_locality=-9999))
        archive.add("b", BehavioralFingerprint(interaction_locality=9999))

        novelty = archive.novelty_of(
            BehavioralFingerprint(interaction_locality=0.25), k=2
        )

        self.assertTrue(math.isfinite(novelty))
        self.assertGreaterEqual(novelty, 0.0)
        self.assertLessEqual(novelty, 1.0)

    def test_native_search_metrics_matches_numpy_reference(self):
        reset_native_search_metrics_lib()
        if load_native_search_metrics_lib() is None:
            self.skipTest("native search metrics runtime not built")

        feature_matrix = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        target = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)

        diff = feature_matrix - target
        distances = np.sqrt(np.mean(np.square(diff), axis=1))

        native_mean = archive_mean_k_nearest(feature_matrix, target, 2)
        self.assertIsNotNone(native_mean)
        self.assertAlmostEqual(
            native_mean, float(np.mean(np.partition(distances, 1)[:2]))
        )

        native_neighbors = topk_nearest_indices(feature_matrix, target, 4)
        self.assertIsNotNone(native_neighbors)
        order, native_distances = native_neighbors
        np.testing.assert_array_equal(order, np.argsort(distances).astype(np.int32))
        np.testing.assert_allclose(native_distances, np.sort(distances), atol=1e-6)

    def test_native_pairwise_stats_match_numpy_reference(self):
        reset_native_search_metrics_lib()
        if load_native_search_metrics_lib() is None:
            self.skipTest("native search metrics runtime not built")

        feature_matrix = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.2, 0.2, 0.2, 0.2],
                [0.6, 0.6, 0.6, 0.6],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        native_stats = pairwise_median_and_neighbor_counts(feature_matrix)
        self.assertIsNotNone(native_stats)
        native_median, native_counts = native_stats

        diffs = feature_matrix[:, np.newaxis, :] - feature_matrix[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.mean(np.square(diffs), axis=2))
        mask = ~np.eye(feature_matrix.shape[0], dtype=bool)
        expected_median = float(np.median(dist_matrix[mask]))
        expected_counts = np.sum(dist_matrix < expected_median, axis=1) - 1

        self.assertAlmostEqual(native_median, expected_median, places=6)
        np.testing.assert_array_equal(native_counts, expected_counts.astype(np.int32))


if __name__ == "__main__":
    unittest.main()
