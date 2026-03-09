import pytest
import math
import unittest

from research.eval.fingerprint import BehavioralFingerprint
from research.search.novelty_search import BehaviorArchive, _behavior_distance

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

        novelty = archive.novelty_of(BehavioralFingerprint(interaction_locality=0.25), k=2)

        self.assertTrue(math.isfinite(novelty))
        self.assertGreaterEqual(novelty, 0.0)
        self.assertLessEqual(novelty, 1.0)


if __name__ == "__main__":
    unittest.main()
