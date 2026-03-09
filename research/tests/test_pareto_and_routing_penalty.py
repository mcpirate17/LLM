"""Tests for 3D Pareto-front tracking (task 6.3) and routing penalty (task 6.4)."""

import pytest
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestCompositeScoreRoutingPenalty(unittest.TestCase):
    """Test the routing overhead penalty in compute_composite_score()."""

    def score(self, **kwargs):
        from research.scientist.notebook import LabNotebook
        return LabNotebook.compute_composite_score(**kwargs)

    def test_no_routing_no_penalty(self):
        """No routing_savings means no penalty."""
        s1 = self.score(screening_lr=0.5, screening_nov=0.8)
        s2 = self.score(screening_lr=0.5, screening_nov=0.8, routing_savings=None)
        self.assertEqual(s1, s2)

    def test_good_routing_gets_bonus(self):
        """Routing with real savings (>0.05) gets bonus, no penalty."""
        base = self.score(screening_lr=0.5, screening_nov=0.8)
        with_routing = self.score(screening_lr=0.5, screening_nov=0.8,
                                  routing_savings=0.3)
        self.assertGreater(with_routing, base)

    def test_wasteful_routing_penalized(self):
        """Routing that saves nothing AND doesn't improve loss is penalized."""
        # routing_savings=0.01 (<0.05) and screening_lr=0.98 (>0.95)
        base = self.score(screening_lr=0.98, screening_nov=0.5)
        penalized = self.score(screening_lr=0.98, screening_nov=0.5,
                               routing_savings=0.01)
        self.assertLess(penalized, base)

    def test_wasteful_routing_good_loss_no_penalty(self):
        """Routing that saves nothing but has great loss is NOT penalized."""
        # routing_savings=0.01 but screening_lr=0.3 (<0.95) — good loss
        base = self.score(screening_lr=0.3, screening_nov=0.8)
        with_routing = self.score(screening_lr=0.3, screening_nov=0.8,
                                  routing_savings=0.01)
        # Should get bonus from routing_savings (small) but no penalty
        # because loss is good
        self.assertGreaterEqual(with_routing, base)

    def test_penalty_scales_with_waste(self):
        """More wasteful routing (lower savings) gets bigger penalty."""
        p1 = self.score(screening_lr=0.98, screening_nov=0.5,
                        routing_savings=0.04)  # close to threshold
        p2 = self.score(screening_lr=0.98, screening_nov=0.5,
                        routing_savings=0.00)  # maximally wasteful
        self.assertLess(p2, p1)

    def test_penalty_uses_best_available_loss(self):
        """Penalty checks val_baseline first, then falls back."""
        # val_baseline=0.5 (good) — no penalty even with zero savings
        s = self.score(screening_lr=0.99, screening_nov=0.5,
                       val_baseline=0.5, routing_savings=0.0)
        base = self.score(screening_lr=0.99, screening_nov=0.5,
                          val_baseline=0.5)
        # Should get tiny bonus from routing_savings but no penalty
        self.assertGreaterEqual(s, base)

    def test_promote_to_tier_passes_routing(self):
        """promote_to_tier passes routing_savings and compression_ratio to scoring."""
        import inspect
        from research.scientist.notebook import LabNotebook
        src = inspect.getsource(LabNotebook.promote_to_tier)
        self.assertIn("routing_savings=d.get(\"routing_savings_ratio\")", src)
        self.assertIn("compression_ratio=d.get(\"compression_ratio\")", src)


class TestParetoFrontier3D(unittest.TestCase):
    """Test the 3D Pareto frontier logic."""

    def _make_analytics(self, rows):
        """Create mock analytics with given program_results rows."""
        from research.scientist.analytics import ExperimentAnalytics

        mock_nb = MagicMock()
        # Mock the connection to return rows as dicts
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            type('Row', (), {
                '__getitem__': lambda self, k: self.__dict__[k],
                'keys': lambda self: [k for k in self.__dict__ if not k.startswith('_')],
            })() for _ in rows
        ]
        # Actually just use sqlite3.Row-like dicts
        # Simpler: patch the method directly
        analytics = ExperimentAnalytics.__new__(ExperimentAnalytics)
        analytics.nb = mock_nb
        return analytics

    def test_empty_frontier(self):
        from research.scientist.analytics import ExperimentAnalytics
        analytics = ExperimentAnalytics.__new__(ExperimentAnalytics)
        analytics.nb = MagicMock()
        analytics.nb.conn.execute.return_value.fetchall.return_value = []

        result = analytics.efficiency_frontier_3d()
        self.assertEqual(result["frontier"], [])
        self.assertEqual(result["total_candidates"], 0)
        self.assertEqual(result["frontier_count"], 0)

    def test_single_program_is_pareto(self):
        from research.scientist.analytics import ExperimentAnalytics
        analytics = ExperimentAnalytics.__new__(ExperimentAnalytics)

        class FakeRow(dict):
            def keys(self):
                return list(super().keys())

        row = FakeRow({
            "result_id": "r1", "graph_fingerprint": "fp1",
            "final_loss": 1.0, "flops_forward": 1000,
            "param_count": 500, "novelty_score": 0.5,
            "loss_ratio": 0.8, "baseline_loss_ratio": 0.9,
            "graph_json": None, "routing_savings_ratio": None,
            "compression_ratio": None,
        })

        analytics.nb = MagicMock()
        analytics.nb.conn.execute.return_value.fetchall.return_value = [row]

        result = analytics.efficiency_frontier_3d()
        self.assertEqual(result["frontier_count"], 1)
        self.assertEqual(result["dominated_count"], 0)

    def test_dominated_point_excluded(self):
        """A point dominated on all 3 dimensions is excluded from frontier."""
        from research.scientist.analytics import ExperimentAnalytics
        analytics = ExperimentAnalytics.__new__(ExperimentAnalytics)

        def make_row(**kw):
            defaults = {
                "graph_fingerprint": "fp", "novelty_score": 0.5,
                "loss_ratio": 0.8, "baseline_loss_ratio": 0.9,
                "graph_json": None, "routing_savings_ratio": None,
                "compression_ratio": None,
            }
            defaults.update(kw)
            return defaults

        rows = [
            make_row(result_id="good", final_loss=0.5, flops_forward=500, param_count=100),
            make_row(result_id="bad", final_loss=1.0, flops_forward=1000, param_count=200),
        ]

        analytics.nb = MagicMock()
        analytics.nb.conn.execute.return_value.fetchall.return_value = rows

        result = analytics.efficiency_frontier_3d()
        self.assertEqual(result["frontier_count"], 1)
        self.assertEqual(result["dominated_count"], 1)
        self.assertEqual(result["frontier"][0]["result_id"], "good")

    def test_non_dominated_tradeoffs_both_on_frontier(self):
        """Two programs with different tradeoffs are both Pareto-optimal."""
        from research.scientist.analytics import ExperimentAnalytics
        analytics = ExperimentAnalytics.__new__(ExperimentAnalytics)

        def make_row(**kw):
            defaults = {
                "graph_fingerprint": "fp", "novelty_score": 0.5,
                "loss_ratio": 0.8, "baseline_loss_ratio": 0.9,
                "graph_json": None, "routing_savings_ratio": None,
                "compression_ratio": None,
            }
            defaults.update(kw)
            return defaults

        rows = [
            # Low loss but high compute
            make_row(result_id="accurate", final_loss=0.3, flops_forward=2000, param_count=500),
            # Higher loss but very efficient
            make_row(result_id="efficient", final_loss=0.8, flops_forward=200, param_count=50),
        ]

        analytics.nb = MagicMock()
        analytics.nb.conn.execute.return_value.fetchall.return_value = rows

        result = analytics.efficiency_frontier_3d()
        self.assertEqual(result["frontier_count"], 2)
        self.assertEqual(result["dominated_count"], 0)
        ids = {p["result_id"] for p in result["frontier"]}
        self.assertEqual(ids, {"accurate", "efficient"})

    def test_compression_creates_pareto_point(self):
        """A program with worse loss/flops but better compression survives."""
        from research.scientist.analytics import ExperimentAnalytics
        analytics = ExperimentAnalytics.__new__(ExperimentAnalytics)

        def make_row(**kw):
            defaults = {
                "graph_fingerprint": "fp", "novelty_score": 0.5,
                "loss_ratio": 0.8, "baseline_loss_ratio": 0.9,
                "graph_json": None, "routing_savings_ratio": None,
                "compression_ratio": None,
            }
            defaults.update(kw)
            return defaults

        rows = [
            # Good loss, good flops, no compression
            make_row(result_id="a", final_loss=0.5, flops_forward=1000, param_count=1000),
            # Slightly worse loss, same flops, but 4x compressed
            make_row(result_id="b", final_loss=0.6, flops_forward=1000,
                     param_count=1000, compression_ratio=0.25),
        ]

        analytics.nb = MagicMock()
        analytics.nb.conn.execute.return_value.fetchall.return_value = rows

        result = analytics.efficiency_frontier_3d()
        # b has effective_params=250 vs a's 1000, so b is not dominated
        self.assertEqual(result["frontier_count"], 2)
        ids = {p["result_id"] for p in result["frontier"]}
        self.assertEqual(ids, {"a", "b"})

    def test_three_way_dominance(self):
        """With 3 programs, one dominates another but not the third."""
        from research.scientist.analytics import ExperimentAnalytics
        analytics = ExperimentAnalytics.__new__(ExperimentAnalytics)

        def make_row(**kw):
            defaults = {
                "graph_fingerprint": "fp", "novelty_score": 0.5,
                "loss_ratio": 0.8, "baseline_loss_ratio": 0.9,
                "graph_json": None, "routing_savings_ratio": None,
                "compression_ratio": None,
            }
            defaults.update(kw)
            return defaults

        rows = [
            make_row(result_id="best",   final_loss=0.3, flops_forward=500, param_count=100),
            make_row(result_id="worst",  final_loss=0.5, flops_forward=600, param_count=200),
            make_row(result_id="tradeoff", final_loss=0.4, flops_forward=300, param_count=500),
        ]

        analytics.nb = MagicMock()
        analytics.nb.conn.execute.return_value.fetchall.return_value = rows

        result = analytics.efficiency_frontier_3d()
        # "best" dominates "worst" (better on all 3)
        # "tradeoff" has fewer flops but more params than "best" — not dominated
        self.assertEqual(result["frontier_count"], 2)
        self.assertEqual(result["dominated_count"], 1)
        ids = {p["result_id"] for p in result["frontier"]}
        self.assertIn("best", ids)
        self.assertIn("tradeoff", ids)
        self.assertNotIn("worst", ids)


class TestDashboardRoutingPenalty(unittest.TestCase):
    """Verify scoringEngine.js has the routing overhead penalty."""

    def test_penalty_function_exists(self):
        js_path = Path(__file__).resolve().parent.parent / "dashboard" / "src" / "utils" / "scoringEngine.js"
        content = js_path.read_text()
        self.assertIn("computeRoutingOverheadPenalty", content)
        self.assertIn("routingOverheadPenalty", content)


if __name__ == "__main__":
    unittest.main()
