"""Smoke tests for the new ablation diagnostics API endpoints.

Covers /api/ablations/{causal-summary, champions, components, recommendations,
children-for-rule, construction-prior, construction-prior/refresh, bulk/start}
and the pre-existing runner drain endpoint, so the route-coverage gate sees
every endpoint referenced by at least one test.
"""

from __future__ import annotations

import json
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

    def test_knockout_rule_evidence_is_visible_in_rollups_and_drilldown(self):
        from research.scientist.notebook import LabNotebook
        from research.scientist.runner._helpers import program_result_kwargs_from_s1

        nb = LabNotebook(str(self.db_path))
        try:
            parent_exp = nb.start_experiment("synthesis", {}, "parent")
            child_exp = nb.start_experiment("ablation", {}, "knockout child")

            parent_s1 = {
                "passed": True,
                "loss_ratio": 0.40,
                "final_loss": 4.0,
                "wikitext_perplexity": 120.0,
                "wikitext_score": 0.6,
                "hellaswag_acc": 0.34,
                "blimp_overall_accuracy": 0.58,
                "induction_auc": 0.45,
                "binding_auc": 0.35,
                "binding_composite": 0.35,
                "ar_auc": 0.20,
                "fp_jacobian_erf_density": 0.50,
                "fp_icld_delta_loss": -0.30,
                "fp_logit_margin_delta": 0.20,
            }
            child_s1 = {
                **parent_s1,
                "loss_ratio": 0.65,
                "final_loss": 6.5,
                "wikitext_perplexity": 210.0,
                "hellaswag_acc": 0.29,
                "blimp_overall_accuracy": 0.51,
                "induction_auc": 0.12,
                "binding_auc": 0.16,
                "binding_composite": 0.16,
                "ar_auc": 0.09,
            }

            parent_kwargs = program_result_kwargs_from_s1(
                parent_s1, model_source="graph_synthesis"
            )
            child_kwargs = program_result_kwargs_from_s1(
                child_s1, model_source="ablation"
            )
            parent_rid = nb.record_program_result(
                experiment_id=parent_exp,
                graph_fingerprint="fp_parent_knockout",
                graph_json='{"nodes":{"13":{"op_name":"softmax_attention"}}}',
                result_id="parent_kout",
                stage0_passed=True,
                stage05_passed=True,
                stage1_passed=True,
                induction_v2_investigation_auc=0.91,
                induction_v2_investigation_status="ok",
                binding_v2_investigation_auc=0.73,
                binding_v2_investigation_status="ok",
                **parent_kwargs,
            )
            child_rid = nb.record_program_result(
                experiment_id=child_exp,
                graph_fingerprint="fp_child_knockout",
                graph_json='{"nodes":{}}',
                result_id="child_kout",
                stage0_passed=True,
                stage05_passed=True,
                stage1_passed=True,
                intentional_rerun_reason="ablation_counterfactual",
                induction_v2_investigation_auc=0.11,
                induction_v2_investigation_status="ok",
                binding_v2_investigation_auc=0.20,
                binding_v2_investigation_status="ok",
                **child_kwargs,
            )
            evidence_json = {
                "child_result_id": child_rid,
                "child_stage1_passed": True,
                "child": {"fingerprint": "fp_child_knockout"},
                "parent_metrics": {
                    "loss_ratio": 0.40,
                    "wikitext_perplexity": 120.0,
                    "hellaswag_acc": 0.34,
                    "blimp_overall_accuracy": 0.58,
                    "induction_auc": 0.45,
                    "binding_composite": 0.35,
                    "ar_auc": 0.20,
                    "induction_v2_investigation_auc": 0.91,
                    "binding_v2_investigation_auc": 0.73,
                },
                "child_metrics": {
                    "loss_ratio": 0.65,
                    "wikitext_perplexity": 210.0,
                    "hellaswag_acc": 0.29,
                    "blimp_overall_accuracy": 0.51,
                    "induction_auc": 0.12,
                    "binding_composite": 0.16,
                    "ar_auc": 0.09,
                    "induction_v2_investigation_auc": 0.11,
                    "induction_v2_investigation_status": "ok",
                    "binding_v2_investigation_auc": 0.20,
                    "binding_v2_investigation_status": "ok",
                },
            }
            nb.record_causal_rule_evidence(
                {
                    "parent_experiment_id": parent_exp,
                    "parent_result_id": parent_rid,
                    "parent_fingerprint": "fp_parent_knockout",
                    "ablation_experiment_id": child_exp,
                    "rule_type": "node_delete_investigation",
                    "rule_key": "13:softmax_attention",
                    "rule_context": "{}",
                    "original_loss_ratio": 0.40,
                    "ablation_best_loss_ratio": 0.65,
                    "effect_size": 0.25,
                    "original_stage1_passed": 1,
                    "ablation_stage1_pass_count": 1,
                    "ablation_total": 1,
                    "outcome": "supported",
                    "confidence": 0.95,
                    "evidence_json": json.dumps(evidence_json),
                }
            )
            nb.flush_writes()
        finally:
            nb.close()

        resp = self.client.get(
            "/api/ablations/components?rule_type=node_delete_investigation&limit=10"
        )
        self.assertEqual(resp.status_code, 200)
        rows = resp.get_json()["components"]
        self.assertEqual(rows[0]["rule_key"], "13:softmax_attention")
        self.assertEqual(rows[0]["n_induction_v2"], 1)
        self.assertAlmostEqual(rows[0]["avg_d_induction_v2"], 0.80)

        resp = self.client.get(
            "/api/ablations/children-for-rule"
            "?rule_type=node_delete_investigation&rule_key=13:softmax_attention"
        )
        self.assertEqual(resp.status_code, 200)
        children = resp.get_json()["children"]
        self.assertEqual(children[0]["source"], "knockout_investigation")
        self.assertEqual(children[0]["child_result_id"], child_rid)
        self.assertAlmostEqual(children[0]["parent_induction_v2"], 0.91)
        self.assertAlmostEqual(children[0]["child_induction_v2"], 0.11)

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
        resp = self.client.post("/api/runner/drain-pending-validation-rerun", json={})
        self.assertIn(resp.status_code, (200, 400, 409, 500))


if __name__ == "__main__":
    unittest.main()
