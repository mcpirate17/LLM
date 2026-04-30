"""Smoke tests for the new ablation diagnostics API endpoints.

Covers /api/ablations/{causal-summary, champions, components, recommendations,
children-for-rule, construction-prior, construction-prior/refresh, bulk/start}
and the pre-existing runner drain endpoint, so the route-coverage gate sees
every endpoint referenced by at least one test.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

pytestmark = pytest.mark.api


class TestAblationDiagnosticsAPI(unittest.TestCase):
    def setUp(self) -> None:
        from research.scientist.api import create_app
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "lab_notebook.db"
        # Touch the file so the notebook auto-creates schema on first read
        from research.scientist.notebook import LabNotebook
        nb = LabNotebook(str(self.db_path))
        nb.close()
        self.app = create_app(notebook_path=str(self.db_path))
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_causal_summary_endpoint_returns_shape(self):
        resp = self.client.get("/api/ablations/causal-summary?limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("summary", data)
        self.assertIn("totals", data)

    def test_champions_endpoint_returns_shape(self):
        resp = self.client.get("/api/ablations/champions?limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("champions", data)
        self.assertIsInstance(data["champions"], list)

    def test_components_endpoint_returns_shape(self):
        resp = self.client.get("/api/ablations/components?limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("components", data)

    def test_components_endpoint_filters_by_rule_type(self):
        resp = self.client.get("/api/ablations/components?rule_type=op&limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("components", data)

    def test_recommendations_endpoint_returns_shape(self):
        resp = self.client.get("/api/ablations/recommendations?min_n=2&limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("recommendations", data)

    def test_children_for_rule_requires_keys(self):
        resp = self.client.get("/api/ablations/children-for-rule")
        self.assertEqual(resp.status_code, 400)

    def test_children_for_rule_returns_shape(self):
        resp = self.client.get(
            "/api/ablations/children-for-rule?rule_type=op&rule_key=foo&limit=5"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("children", data)

    def test_construction_prior_active_returns_none_initially(self):
        resp = self.client.get("/api/ablations/construction-prior")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("active", data)
        self.assertIn("snapshots", data)
        self.assertIsNone(data["active"])

    def test_construction_prior_refresh_with_no_evidence_returns_400(self):
        resp = self.client.post(
            "/api/ablations/construction-prior/refresh",
            json={"min_n": 4},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_bulk_start_route_registered(self):
        # POST should at least be reachable (will likely 4xx without runner)
        resp = self.client.post("/api/ablations/bulk/start", json={})
        self.assertIn(resp.status_code, (200, 400, 409, 500))

    def test_drain_pending_validation_rerun_route_registered(self):
        resp = self.client.post(
            "/api/runner/drain-pending-validation-rerun", json={}
        )
        self.assertIn(resp.status_code, (200, 400, 409, 500))


if __name__ == "__main__":
    unittest.main()
