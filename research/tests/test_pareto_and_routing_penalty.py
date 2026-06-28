"""Tests for 3D Pareto-front tracking (task 6.3) and routing penalty (task 6.4)."""

import pytest
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _pareto_row(**kw):
    defaults = {
        "graph_fingerprint": "fp",
        "novelty_score": 0.5,
        "loss_ratio": 0.8,
        "baseline_loss_ratio": 0.9,
        "graph_json": None,
        "routing_savings_ratio": None,
        "compression_ratio": None,
    }
    defaults.update(kw)
    return defaults


class TestParetoFrontier3D(unittest.TestCase):
    """Test the 3D Pareto frontier logic."""

    def _make_analytics(self, rows):
        """Create mock analytics with given program_results rows."""
        from research.scientist.analytics import ExperimentAnalytics

        mock_nb = MagicMock()
        # Mock the connection to return rows as dicts
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            type(
                "Row",
                (),
                {
                    "__getitem__": lambda self, k: self.__dict__[k],
                    "keys": lambda self: [
                        k for k in self.__dict__ if not k.startswith("_")
                    ],
                },
            )()
            for _ in rows
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

        row = FakeRow(
            {
                "result_id": "r1",
                "graph_fingerprint": "fp1",
                "final_loss": 1.0,
                "flops_forward": 1000,
                "param_count": 500,
                "novelty_score": 0.5,
                "loss_ratio": 0.8,
                "baseline_loss_ratio": 0.9,
                "graph_json": None,
                "routing_savings_ratio": None,
                "compression_ratio": None,
            }
        )

        analytics.nb = MagicMock()
        analytics.nb.conn.execute.return_value.fetchall.return_value = [row]

        result = analytics.efficiency_frontier_3d()
        self.assertEqual(result["frontier_count"], 1)
        self.assertEqual(result["dominated_count"], 0)

    def test_dominated_point_excluded(self):
        """A point dominated on all 3 dimensions is excluded from frontier."""
        from research.scientist.analytics import ExperimentAnalytics

        analytics = ExperimentAnalytics.__new__(ExperimentAnalytics)

        rows = [
            _pareto_row(
                result_id="good", final_loss=0.5, flops_forward=500, param_count=100
            ),
            _pareto_row(
                result_id="bad", final_loss=1.0, flops_forward=1000, param_count=200
            ),
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

        rows = [
            # Low loss but high compute
            _pareto_row(
                result_id="accurate",
                final_loss=0.3,
                flops_forward=2000,
                param_count=500,
            ),
            # Higher loss but very efficient
            _pareto_row(
                result_id="efficient", final_loss=0.8, flops_forward=200, param_count=50
            ),
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

        rows = [
            # Good loss, good flops, no compression
            _pareto_row(
                result_id="a", final_loss=0.5, flops_forward=1000, param_count=1000
            ),
            # Slightly worse loss, same flops, but 4x compressed
            _pareto_row(
                result_id="b",
                final_loss=0.6,
                flops_forward=1000,
                param_count=1000,
                compression_ratio=0.25,
            ),
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

        rows = [
            _pareto_row(
                result_id="best", final_loss=0.3, flops_forward=500, param_count=100
            ),
            _pareto_row(
                result_id="worst", final_loss=0.5, flops_forward=600, param_count=200
            ),
            _pareto_row(
                result_id="tradeoff", final_loss=0.4, flops_forward=300, param_count=500
            ),
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


if __name__ == "__main__":
    unittest.main()
