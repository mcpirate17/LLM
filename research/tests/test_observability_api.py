"""Tests for observability API endpoints (P0–P3)."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

try:
    from flask import Flask  # noqa: F401

    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _stage1_fixture_kwargs(loss_ratio: float, novelty_score: float) -> dict:
    from research.scientist.runner._helpers import program_result_kwargs_from_s1

    return program_result_kwargs_from_s1(
        {
            "passed": True,
            "final_loss": 4.5,
            "loss_ratio": loss_ratio,
            "wikitext_perplexity": 150.0,
            "wikitext_score": 0.55,
            "screening_wikitext_metric_version": "unit_test_wikitext_v1",
            "hellaswag_acc": 0.31,
            "hellaswag_status": "ran",
            "blimp_overall_accuracy": 0.55,
            "blimp_status": "ran",
            "induction_screening_auc": 0.21,
            "binding_screening_auc": 0.18,
            "binding_screening_composite": 0.12,
            "ar_legacy_auc": 0.06,
        },
        model_source="graph_synthesis",
        extra={
            "stage1_passed": True,
            "novelty_score": novelty_score,
            "data_mode": "random",
            "tokenizer_mode": "byte",
            "vocab_size": 256,
        },
    )


@unittest.skipUnless(HAS_TORCH and HAS_FLASK, "requires torch and flask")
class TestObservabilityAPI(unittest.TestCase):
    """Test all observability endpoints return 200 with expected keys."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.tmpdir, "test_obs.db")

        from research.scientist.api import create_app
        from research.scientist.notebook import LabNotebook

        cls.app = create_app(notebook_path=cls.db_path)
        cls.client = cls.app.test_client()

        # Seed data
        nb = LabNotebook(cls.db_path)
        exp_id = nb.start_experiment("synthesis", {"n_programs": 5}, "test hyp")

        graph = {
            "nodes": {
                "0": {"op_name": "linear_proj"},
                "1": {"op_name": "gelu"},
                "2": {"op_name": "layernorm"},
            }
        }
        result_ids = []
        for i in range(5):
            rid = nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=f"obs_fp_{i:03d}",
                graph_json=json.dumps(graph),
                stage0_passed=(i < 4),
                stage05_passed=(i < 3),
                stage1_passed=(i < 2),
                loss_ratio=0.8 - i * 0.1 if i < 4 else None,
                novelty_score=0.5,
                wikitext_perplexity=8.0 + i,
                hellaswag_acc=0.30 + i * 0.01,
                blimp_overall_accuracy=0.52 + i * 0.01,
                induction_screening_auc=0.04 + i * 0.005,
                binding_screening_auc=0.05 + i * 0.005,
                binding_screening_composite=0.06 + i * 0.005,
                ar_legacy_auc=0.03 + i * 0.005,
                induction_intermediate_auc=0.20 + i * 0.01,
                binding_intermediate_auc=0.29 + i * 0.01,
                ar_curriculum_auc_pair_final=0.40 + i * 0.02,
                ar_curriculum_s0_retention=0.60 + i * 0.01,
                ar_curriculum_max_passing_stage=4,
                language_control_s05_sentence_assoc_score=0.80,
                language_control_s05_binding_order_acc=0.70,
                language_control_s05_binding_score=0.74,
                language_control_s10_sentence_assoc_score=0.82,
                language_control_s10_binding_order_acc=0.72,
                language_control_s10_binding_score=0.76,
                language_control_investigation_sentence_assoc_score=0.84,
                language_control_investigation_binding_order_acc=0.70,
                language_control_investigation_binding_score=0.72,
                error_type="shape_mismatch" if i >= 4 else None,
            )
            result_ids.append(rid)
        nb.flush_writes()
        for rid in result_ids:
            for op_name in ("linear_proj", "gelu", "layernorm"):
                nb.conn.execute(
                    "INSERT OR IGNORE INTO program_graph_ops "
                    "(result_id, graph_fingerprint, op_name) VALUES (?, ?, ?)",
                    (rid, f"ops_{rid}", op_name),
                )

        # Seed a learning_log error entry
        nb.conn.execute(
            "INSERT INTO learning_log (timestamp, event_type, description, evidence) "
            "VALUES (?, 'error', 'test error description', '{}')",
            (time.time(),),
        )
        # Seed a grammar weights event
        nb.conn.execute(
            "INSERT INTO learning_log (timestamp, event_type, description, old_weights, new_weights) "
            "VALUES (?, 'grammar_weights_applied', 'weight update', ?, ?)",
            (
                time.time(),
                json.dumps({"linear_proj": 1.0, "gelu": 1.5}),
                json.dumps({"linear_proj": 1.2, "gelu": 1.3}),
            ),
        )
        # Seed an insight with predictions
        nb.conn.execute(
            "INSERT INTO insights (insight_id, timestamp, category, content, "
            "n_predictions, n_correct, alpha, beta_, status) "
            "VALUES (?, ?, 'pattern', 'test insight', 10, 7, 8.0, 4.0, 'active')",
            ("ins_test_001", time.time()),
        )
        # Seed a leaderboard entry using first real result_id
        first_rid = result_ids[0] if result_ids else "fallback_rid"
        nb.conn.execute(
            "INSERT INTO leaderboard (entry_id, result_id, timestamp, model_source, tier, "
            "screening_loss_ratio, composite_score, result_cohort, trust_label, "
            "comparability_label, evaluation_protocol_version, scoring_config_hash) "
            "VALUES (?, ?, ?, 'graph_synthesis', 'screening', 0.75, 0.6, "
            "'search', 'candidate_screening', 'screening_only', 'screening_v1', 'fixture_hash')",
            ("lb_test_001", first_rid, time.time()),
        )
        nb.conn.commit()

        nb.update_op_success_rates(exp_id)
        nb.complete_experiment(exp_id, {"n_programs": 5})
        nb.flush_writes()
        nb.close()
        from research.scientist.api_routes._observability_core import (
            refresh_observability_caches,
        )

        refresh_observability_caches()

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    # ── P0 endpoints ──

    def test_error_log(self):
        resp = self.client.get("/api/observability/error-log")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("errors", data)
        self.assertIn("count", data)
        self.assertGreaterEqual(data["count"], 1)

    def test_experiment_lifecycle(self):
        resp = self.client.get("/api/observability/experiment-lifecycle")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("experiments", data)
        self.assertIn("count", data)
        self.assertGreaterEqual(data["count"], 1)
        # Check orphan field present
        saw_mismatch = False
        for exp in data["experiments"]:
            self.assertIn("orphan", exp)
            self.assertIn("persisted_program_rows", exp)
            self.assertIn("persisted_stage0_passed", exp)
            self.assertIn("persisted_stage1_passed", exp)
            self.assertIn("count_discrepancy", exp)
            self.assertIn("count_mismatch", exp)
            if exp["count_mismatch"]:
                saw_mismatch = True
        self.assertTrue(saw_mismatch)

    def test_experiment_lifecycle_cleanup(self):
        resp = self.client.post("/api/observability/experiment-lifecycle/cleanup")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("cleaned", data)

    def test_throughput(self):
        resp = self.client.get("/api/observability/throughput")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        for window in ("1h", "6h", "24h"):
            self.assertIn(window, data)
            self.assertIn("total", data[window])
            self.assertIn("s0_rate", data[window])

    # ── P1 endpoints ──

    def test_op_pairs(self):
        resp = self.client.get("/api/observability/op-pairs")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("pairs", data)
        self.assertIn("total_pairs", data)
        if data["pairs"]:
            pair = data["pairs"][0]
            self.assertIn("op_a", pair)
            self.assertIn("op_b", pair)
            self.assertIn("n", pair)

    def test_loss_distribution(self):
        resp = self.client.get("/api/observability/loss-distribution")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("distributions", data)
        if data["distributions"]:
            d = data["distributions"][0]
            for key in ("op", "n", "min", "q1", "median", "q3", "max", "mean"):
                self.assertIn(key, d)

    def test_resource_utilization(self):
        resp = self.client.get("/api/observability/resource-utilization")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("cpu_percent", data)
        self.assertIn("ram_percent", data)

    def test_api_health(self):
        from research.scientist.api_routes._api_health import (
            API_HEALTH_COUNTERS,
            API_HEALTH_LOCK,
        )

        with API_HEALTH_LOCK:
            API_HEALTH_COUNTERS["/api/test:2xx"] += 1
        resp = self.client.get("/api/observability/api-health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("counters", data)
        self.assertIn("/api/test:2xx", data["counters"])

    def test_db_health_includes_entity_accounting(self):
        resp = self.client.get("/api/observability/db-health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("row_counts", data)
        self.assertIn("entity_counts", data)
        self.assertIn("row_volume", data["entity_counts"])
        self.assertIn("run_volume", data["entity_counts"])
        self.assertIn("graph_volume", data["entity_counts"])
        self.assertIn("training_curve_density", data["entity_counts"])

    def test_reporting_data_accounting_endpoint(self):
        resp = self.client.get("/api/reporting/data-accounting")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("row_volume", data)
        self.assertIn("run_volume", data)
        self.assertIn("graph_volume", data)
        self.assertIn("filtering", data)

    # ── P2 endpoints ──

    def test_grammar_evolution(self):
        resp = self.client.get("/api/observability/grammar-evolution")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("events", data)
        self.assertIn("count", data)
        self.assertGreaterEqual(data["count"], 1)
        if data["events"]:
            self.assertIn("changes", data["events"][0])

    def test_failure_patterns(self):
        resp = self.client.get("/api/observability/failure-patterns")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("patterns", data)

    def test_failure_patterns_prefer_root_cause_and_failure_op(self):
        from research.scientist.notebook import LabNotebook
        from research.scientist.api_routes._observability_core import (
            refresh_observability_caches,
        )

        nb = LabNotebook(self.db_path)
        exp_id = nb.start_experiment("synthesis", {"n_programs": 1}, "obs root cause")
        graph = {
            "nodes": {
                "0": {"op_name": "input"},
                "1": {"op_name": "rmsnorm"},
                "2": {"op_name": "swiglu_mlp"},
                "3": {"op_name": "add"},
            }
        }
        nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="obs_fp_root_cause",
            graph_json=json.dumps(graph),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=False,
            error_type="insufficient_learning",
            failure_op="swiglu_mlp",
            failure_details_json=json.dumps(
                {
                    "error_type": "insufficient_learning",
                    "failure_op": "swiglu_mlp",
                    "root_cause_code": "generalization_failure",
                }
            ),
        )
        nb.flush_writes()
        nb.close()
        refresh_observability_caches()

        resp = self.client.get("/api/observability/failure-patterns?top_ops=5")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        by_error = {p["error_type"]: p for p in data["patterns"]}
        self.assertIn("s1_generalization_failure", by_error)
        top_ops = by_error["s1_generalization_failure"]["top_ops"]
        self.assertTrue(top_ops)
        self.assertEqual(top_ops[0]["op"], "swiglu_mlp")
        self.assertNotIn("rmsnorm", [entry["op"] for entry in top_ops])

    def test_leaderboard_dynamics(self):
        resp = self.client.get("/api/observability/leaderboard-dynamics")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("daily", data)
        self.assertIn("recent_promotions", data)
        self.assertTrue(data.get("trusted_only"))
        self.assertGreaterEqual(len(data["recent_promotions"]), 1)

    def test_insight_effectiveness(self):
        resp = self.client.get("/api/observability/insight-effectiveness")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("insights", data)
        self.assertIn("count", data)
        self.assertGreaterEqual(data["count"], 1)
        if data["insights"]:
            ins = data["insights"][0]
            self.assertIn("accuracy", ins)
            self.assertIn("bayesian_mean", ins)

    # ── P3 endpoints ──

    def test_db_health(self):
        resp = self.client.get("/api/observability/db-health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("db_size_mb", data)
        self.assertIn("wal_size_mb", data)
        self.assertIn("row_counts", data)
        self.assertIn("program_results", data["row_counts"])

    # ── Existing endpoints still work ──

    def test_health(self):
        resp = self.client.get("/api/observability/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("components", data)
        self.assertIn("total", data)

    def test_alerts(self):
        resp = self.client.get("/api/observability/alerts")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("alerts", data)

    def test_health_refresh(self):
        resp = self.client.post("/api/observability/health/refresh")
        self.assertEqual(resp.status_code, 200)

    def test_failure_blocklist(self):
        resp = self.client.get("/api/observability/failure-blocklist")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("blocklist", data)

    # ── Phase 4A: Component health observability repair ──

    def test_health_grid_time_window(self):
        """Verify ?window= param filters to recent data only."""
        resp_all = self.client.get("/api/observability/health?window=all")
        self.assertEqual(resp_all.status_code, 200)
        data_all = resp_all.get_json()
        self.assertIn("window", data_all)
        self.assertEqual(data_all["window"], "all")

        # 1h window — our test data was inserted recently so should appear
        resp_1h = self.client.get("/api/observability/health?window=1h")
        self.assertEqual(resp_1h.status_code, 200)
        data_1h = resp_1h.get_json()
        self.assertEqual(data_1h["window"], "1h")
        # Should have components (our test data is recent)
        self.assertGreater(data_1h["total"], 0)

    def test_health_grid_data_source_labels(self):
        """Verify all components have a data_source field."""
        resp = self.client.get("/api/observability/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        for comp in data.get("components", []):
            self.assertIn(
                "data_source", comp, f"Component {comp.get('op')} missing data_source"
            )
            self.assertIn(
                comp["data_source"], {"search", "search+profiling", "profiling_only"}
            )

    def test_health_grid_skips_missing_graph_artifact(self):
        """A missing cold graph artifact should not blank the health grid."""
        import shutil

        from research.scientist.api import create_app
        from research.scientist.api_routes._observability_core import (
            refresh_observability_caches,
        )
        from research.scientist.notebook import LabNotebook
        from research.scientist.notebook.artifact_store import parse_artifact_pointer
        from research.scientist.notebook.artifact_store import NotebookArtifactStore
        from research.tools.externalize_notebook_artifacts import (
            run as externalize_artifacts,
        )

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "missing_artifact_obs.db")
        try:
            nb = LabNotebook(db_path)
            exp_id = nb.start_experiment(
                "synthesis", {"n_programs": 2}, "missing graph artifact obs"
            )
            graph_a = json.dumps({"nodes": {"0": {"op_name": "linear_proj"}}})
            graph_b = json.dumps({"nodes": {"0": {"op_name": "gelu"}}})
            rid_a = nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint="obs_missing_artifact_a",
                graph_json=graph_a,
                stage0_passed=True,
                **_stage1_fixture_kwargs(loss_ratio=0.8, novelty_score=0.2),
            )
            nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint="obs_missing_artifact_b",
                graph_json=graph_b,
                stage0_passed=True,
                **_stage1_fixture_kwargs(loss_ratio=0.7, novelty_score=0.3),
            )
            nb.flush_writes()
            nb.update_op_success_rates(exp_id)
            nb.close()

            externalize_artifacts(
                db_path=Path(db_path),
                min_bytes=16,
                apply=True,
                limit=None,
                vacuum=False,
                include_graph_json=True,
                graph_json_cold_only=False,
            )

            nb = LabNotebook(db_path)
            try:
                row = nb.conn.execute(
                    "SELECT graph_json FROM program_results_compat WHERE result_id = ?",
                    (rid_a,),
                ).fetchone()
                pointer = parse_artifact_pointer(row["graph_json"])
                self.assertIsNotNone(pointer)
                (NotebookArtifactStore(db_path).root / pointer["path"]).unlink()
            finally:
                nb.close()

            refresh_observability_caches()
            client = create_app(notebook_path=db_path).test_client()
            resp = client.get("/api/observability/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertGreater(data.get("total", 0), 0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_op_success_rates_weighted_averaging(self):
        """Run update_op_success_rates twice, verify weighted average."""
        from research.scientist.notebook import LabNotebook

        tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(tmpdir, "test_avg.db")
        nb = LabNotebook(db_path)

        # Experiment 1: op "linear_proj" with loss_ratio=1.0
        exp1 = nb.start_experiment("synthesis", {"n_programs": 2}, "avg test 1")
        graph = json.dumps({"nodes": {"0": {"op_name": "linear_proj"}}})
        for idx in range(4):
            nb.record_program_result(
                experiment_id=exp1,
                graph_fingerprint=f"avg_fp1_{idx}",
                graph_json=graph,
                stage0_passed=True,
                **_stage1_fixture_kwargs(loss_ratio=1.0, novelty_score=0.2),
            )
        nb.flush_writes()
        nb.update_op_success_rates(exp1)

        # Check after first experiment
        rates1 = {r["op_name"]: r for r in nb.get_op_success_rates()}
        self.assertAlmostEqual(rates1["linear_proj"]["avg_loss_ratio"], 1.0, places=2)

        # Experiment 2: same op with loss_ratio=0.0, 4 more samples
        exp2 = nb.start_experiment("synthesis", {"n_programs": 2}, "avg test 2")
        for idx in range(4):
            nb.record_program_result(
                experiment_id=exp2,
                graph_fingerprint=f"avg_fp2_{idx}",
                graph_json=graph,
                stage0_passed=True,
                **_stage1_fixture_kwargs(loss_ratio=0.0, novelty_score=0.8),
            )
        nb.flush_writes()
        nb.update_op_success_rates(exp2)

        # Weighted average: (1.0*4 + 0.0*4) / 8 = 0.5
        rates2 = {r["op_name"]: r for r in nb.get_op_success_rates()}
        self.assertAlmostEqual(rates2["linear_proj"]["avg_loss_ratio"], 0.5, places=2)
        self.assertEqual(rates2["linear_proj"]["n_used"], 8)
        nb.close()
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_profiled_only_ops_labeled(self):
        """Verify ops with Used=0 from profiling have data_source 'profiling_only'."""
        resp = self.client.get("/api/observability/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        for comp in data.get("components", []):
            if comp.get("n_used", 0) == 0 and comp.get("data_source"):
                self.assertEqual(
                    comp["data_source"],
                    "profiling_only",
                    f"Op {comp.get('op')} with n_used=0 should be profiling_only",
                )

    def test_component_health_uses_corrected_counts_for_display(self):
        """Excluded runtime-only failures should not render as 0% effective rates."""
        from research.scientist.api_routes._observability_core import (
            _build_component_entry,
        )

        row = {
            "op_name": "long_conv_hyena",
            "n_used": 60,
            "n_stage0_passed": 0,
            "n_stage05_passed": 0,
            "n_stage1_passed": 0,
        }
        stored_rates = {"long_conv_hyena": {"n": 59, "s0": 0, "s1": 0}}
        corrected_rates = {
            "long_conv_hyena": {"n": 0, "s0": 0, "s1": 0, "excluded": 59}
        }

        payload = _build_component_entry(
            row,
            stored_rates=stored_rates,
            corrected_rates=corrected_rates,
            grad_health={},
            metric_overlays={},
            max_n_used=59,
        )

        self.assertEqual(payload["n_used"], 0)
        self.assertIsNone(payload["s0_rate"])
        self.assertIsNone(payload["s05_rate"])
        self.assertIsNone(payload["s1_rate"])
        self.assertEqual(payload["raw_n_used"], 59)
        self.assertIn("excluded 59 runtime-only failures", payload["reasons"])

    def test_health_components_include_observability_metrics(self):
        """Component rows expose the same downstream metrics shown in template views."""
        resp = self.client.get("/api/observability/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertGreater(data.get("total", 0), 0)
        comp = data["components"][0]
        for key in (
            "s05_rate",
            "avg_loss_ratio",
            "avg_validation_loss_ratio",
            "avg_induction_screening_auc",
            "avg_binding_screening_auc",
            "avg_binding_screening_composite",
            "avg_ar_legacy_auc",
            "avg_induction_intermediate_auc",
            "avg_binding_intermediate_auc",
            "avg_ar_curriculum_auc_pair_final",
            "avg_ar_curriculum_s0_retention",
            "avg_ar_curriculum_max_passing_stage",
            "n_ar_curriculum",
            "avg_language_control_s05_score",
            "avg_language_control_s10_score",
            "avg_language_control_investigation_score",
            "avg_language_control_investigation_sentence_assoc_score",
            "avg_hellaswag_acc",
            "top_failure_reason",
        ):
            self.assertIn(key, comp)
        from research.scientist.api_routes._observability_core import (
            _component_language_control_metrics,
            _load_component_metric_overlays,
        )
        from research.scientist.notebook import LabNotebook

        nb = LabNotebook(self.db_path, read_only=True)
        try:
            overlays = _load_component_metric_overlays(nb, "all")
        finally:
            nb.close()
        metric_overlay = next(
            (
                item
                for item in overlays.values()
                if item.get("avg_language_control_investigation_sentence_assoc_score")
                is not None
            ),
            None,
        )
        self.assertIsNotNone(metric_overlay)
        metric_payload = _component_language_control_metrics(metric_overlay)
        self.assertAlmostEqual(
            metric_payload["avg_language_control_investigation_score"], 0.78
        )
        self.assertAlmostEqual(metric_overlay["avg_binding_screening_composite"], 0.07)
        self.assertAlmostEqual(metric_overlay["avg_ar_legacy_auc"], 0.04)
        self.assertAlmostEqual(metric_overlay["avg_induction_intermediate_auc"], 0.22)
        self.assertAlmostEqual(metric_overlay["avg_binding_intermediate_auc"], 0.31)
        self.assertAlmostEqual(metric_overlay["avg_ar_curriculum_auc_pair_final"], 0.44)
        self.assertAlmostEqual(metric_overlay["avg_ar_curriculum_s0_retention"], 0.62)
        self.assertAlmostEqual(
            metric_overlay["avg_ar_curriculum_max_passing_stage"], 4.0
        )
        self.assertEqual(metric_overlay["n_ar_curriculum"], 5)


if __name__ == "__main__":
    unittest.main()
