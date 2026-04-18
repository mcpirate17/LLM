"""
Integration Tests for the AI Scientist Research Pipeline

Tests the full stack: notebook schema, leaderboard lifecycle,
auto-escalation pipeline, API endpoints, mode selection, and
novelty scoring fixes.

Run: cd /path/to/LLM && python -m unittest research.tests.test_integration -v
"""

import pytest
from research.scientist.runtime_events import get_runtime_event_services, stop_runtime_event_services
from research.scientist.notebook.notebook_core import _ThreadSafeConnectionWrapper
import importlib
import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

pytestmark = pytest.mark.unit

# Detect available dependencies
try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False


# Import modules that don't require torch directly
# (bypass scientist/__init__.py which eagerly imports runner)
def _import_module(dotted_path):
    """Import a submodule without triggering parent __init__.py."""
    return importlib.import_module(dotted_path)


try:
    from research.scientist.notebook import LabNotebook, ExperimentEntry

    HAS_NOTEBOOK = True
except Exception as e:
    HAS_NOTEBOOK = False
    print(f"Notebook import failed: {e}")

try:
    HAS_PERSONA = True
except Exception as e:
    HAS_PERSONA = False
    print(f"Persona import failed: {e}")

try:
    import research.scientist.llm.prompts as _prompts_mod  # noqa: F401

    HAS_PROMPTS = True
except Exception as e:
    HAS_PROMPTS = False
    print(f"Prompts import failed: {e}")

try:
    import research.scientist.llm.context as _context_mod  # noqa: F401

    HAS_CONTEXT = True
except Exception as e:
    HAS_CONTEXT = False
    print(f"Context import failed: {e}")


# ── Test 1: Notebook & Leaderboard ──


@unittest.skipUnless(HAS_NOTEBOOK, "requires notebook module")
class TestNotebook(unittest.TestCase):
    """Test notebook schema, CRUD, and leaderboard lifecycle."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_notebook.db")
        self.nb = LabNotebook(self.db_path)

    def tearDown(self):
        stop_runtime_event_services(self.nb.db_path)
        self.nb.close()

    def test_schema_all_tables_exist(self):
        """Verify all 9+ tables are created."""
        tables = [
            row[0]
            for row in self.nb.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        expected = [
            "experiments",
            "entries",
            "program_results",
            "metrics_log",
            "insights",
            "training_curves",
            "op_success_rates",
            "learning_log",
            "leaderboard",
        ]
        for t in expected:
            self.assertIn(t, tables, f"Missing table: {t}")

    def test_program_results_experiment_index_exists(self):
        """program_results(experiment_id) should be indexed for large-query performance."""
        indexes = self.nb.conn.execute(
            "PRAGMA index_list('program_results')"
        ).fetchall()
        has_experiment_index = False
        for idx in indexes:
            idx_name = idx[1]
            cols = self.nb.conn.execute(f"PRAGMA index_info('{idx_name}')").fetchall()
            col_names = [c[2] for c in cols]
            if col_names == ["experiment_id"]:
                has_experiment_index = True
                break
        self.assertTrue(
            has_experiment_index, "Missing index on program_results(experiment_id)"
        )

    def test_phase4_indexes_exist(self):
        """Hot query paths should have the new Phase 4 indexes."""
        program_indexes = {
            row[1]: [
                c[2]
                for c in self.nb.conn.execute(
                    f"PRAGMA index_info('{row[1]}')"
                ).fetchall()
            ]
            for row in self.nb.conn.execute(
                "PRAGMA index_list('program_results')"
            ).fetchall()
        }
        leaderboard_indexes = {
            row[1]: [
                c[2]
                for c in self.nb.conn.execute(
                    f"PRAGMA index_info('{row[1]}')"
                ).fetchall()
            ]
            for row in self.nb.conn.execute(
                "PRAGMA index_list('leaderboard')"
            ).fetchall()
        }

        self.assertIn(["stage1_passed"], program_indexes.values())
        self.assertIn(["graph_fingerprint"], program_indexes.values())
        self.assertIn(["routing_mode"], program_indexes.values())
        self.assertIn(["model_source"], leaderboard_indexes.values())

    def test_designer_run_lineage_upsert_and_query(self):
        """Designer run lineage rows should upsert and round-trip structured payloads."""
        self.nb.save_designer_run_lineage(
            run_id="eval_test_lineage_1",
            workflow_id="wf_lineage_test",
            workflow_version=3,
            graph_fingerprint="fp_lineage_test",
            status="success",
            source="aria_designer",
            total_time_ms=123.4,
            metrics={"overall_novelty": 0.42},
            payload={"status": "success"},
        )
        row = self.nb.get_designer_run_lineage("eval_test_lineage_1")
        self.assertIsNotNone(row)
        self.assertEqual(row["workflow_id"], "wf_lineage_test")
        self.assertEqual(row["workflow_version"], 3)
        self.assertEqual(row["graph_fingerprint"], "fp_lineage_test")
        self.assertEqual(row["status"], "success")
        self.assertAlmostEqual(float(row["total_time_ms"]), 123.4, places=2)
        self.assertEqual(row["metrics"].get("overall_novelty"), 0.42)

        # Upsert same run_id updates status and metrics.
        self.nb.save_designer_run_lineage(
            run_id="eval_test_lineage_1",
            workflow_id="wf_lineage_test",
            status="failed_sandbox",
            metrics={"overall_novelty": 0.1},
        )
        row2 = self.nb.get_designer_run_lineage("eval_test_lineage_1")
        self.assertEqual(row2["status"], "failed_sandbox")
        self.assertEqual(row2["metrics"].get("overall_novelty"), 0.1)

    def test_experiment_lifecycle(self):
        """Start → complete → query an experiment."""
        exp_id = self.nb.start_experiment(
            experiment_type="synthesis",
            config={"n_programs": 10},
            hypothesis="Test hypothesis",
        )
        self.assertIsNotNone(exp_id)

        # Record a program result
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="abc123",
            graph_json='{"nodes": {}}',
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.5,
            novelty_score=0.7,
        )
        self.assertIsNotNone(result_id)

        # Complete the experiment
        self.nb.complete_experiment(
            experiment_id=exp_id,
            results={"stage1_passed": 1, "total": 10},
            aria_summary="Test summary",
            aria_mood="excited",
        )

    def test_purge_empty_experiments_deletes_attribution_reports_before_hypotheses(
        self,
    ):
        exp_id = self.nb.start_experiment(
            experiment_type="synthesis",
            config={"n_programs": 1},
            hypothesis="FK cleanup regression",
        )
        hypothesis_id = self.nb.record_hypothesis(
            campaign_id=None,
            experiment_id=exp_id,
            prediction="cleanup should not violate FKs",
            reasoning="regression test",
            test_method="unit",
            success_metric="purge completes",
        )
        self.nb.record_attribution_report(
            hypothesis_id=hypothesis_id,
            supporting_experiments=[exp_id],
            ablation_experiments=[],
            outcome="pending",
            report={"note": "fk regression"},
        )
        self.nb.conn.execute(
            "UPDATE experiments SET status = 'failed' WHERE experiment_id = ?",
            (exp_id,),
        )
        self.nb.conn.commit()

        purged = self.nb.purge_empty_experiments()

        self.assertEqual(purged, 1)
        exp = self.nb.conn.execute(
            "SELECT 1 FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        self.assertIsNone(exp)
        reports = self.nb.conn.execute(
            "SELECT COUNT(*) FROM attribution_reports WHERE hypothesis_id = ?",
            (hypothesis_id,),
        ).fetchone()[0]
        self.assertEqual(reports, 0)

    def test_record_program_result_generates_normalized_data_provenance(self):
        exp_id = self.nb.start_experiment(
            experiment_type="synthesis",
            config={"n_programs": 1},
            hypothesis="provenance normalization",
        )
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="prov123",
            graph_json='{"nodes": {}}',
            model_source="graph_synthesis",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.4,
            wikitext_perplexity=7.5,
            novelty_score=0.6,
            data_mode="corpus",
            corpus_path="/tmp/corpus.txt",
            corpus_format="txt",
            corpus_text_key="text",
            corpus_train_fraction=0.9,
            corpus_val_fraction=0.1,
            corpus_max_chars=200000,
            tokenizer_mode="tiktoken",
            tiktoken_encoding="cl100k_base",
            screening_wikitext_metric_version="screening_wikitext_v1",
            induction_probe_metric_version="induction_probe_v2",
        )
        self.nb.flush_writes()
        row = self.nb.get_program_detail(result_id)
        self.assertIsNotNone(row)
        payload = json.loads(row["data_provenance_json"])
        self.assertEqual(payload["corpus_id"], "file:/tmp/corpus.txt")
        self.assertEqual(payload["tokenizer_id"], "tiktoken:cl100k_base")
        self.assertEqual(payload["split_id"], "train=0.900;val=0.100")
        self.assertTrue(payload["provenance_complete"])
        self.assertTrue(payload["eligible_for_promotion"])

    def test_data_accounting_summary_separates_rows_runs_graphs_and_curves(self):
        exp_id = self.nb.start_experiment(
            experiment_type="synthesis",
            config={"n_programs": 3},
            hypothesis="accounting summary",
        )
        rid_pass = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_a",
            graph_json='{"nodes": {"0": {"op_name": "linear_proj"}}}',
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.4,
            trust_label="candidate_grade",
            comparability_label="candidate_comparable",
            evaluation_protocol_version="candidate_grade_v1",
            train_budget_steps=128,
            hellaswag_acc=0.3,
            induction_auc=0.02,
            binding_auc=0.01,
            wikitext_perplexity=12.0,
            data_provenance_json=json.dumps(
                {"eligible_for_screening_model_training": True}
            ),
        )
        rid_fail_same_graph = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_a",
            graph_json='{"nodes": {"0": {"op_name": "linear_proj"}}}',
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=False,
            error_type="insufficient_learning",
            trust_label="runtime_observation",
            comparability_label="partial",
            evaluation_protocol_version="runtime_observation_v1",
        )
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_b",
            graph_json='{"nodes": {"0": {"op_name": "gelu"}}}',
            stage0_passed=False,
            stage05_passed=False,
            stage1_passed=False,
            error_type="shape_mismatch",
            trust_label="exploratory",
            comparability_label="noncomparable",
            evaluation_protocol_version="forced_exploration_v1",
        )
        self.nb.flush_writes()
        self.nb.store_training_curve(
            rid_pass,
            [
                {"step": 0, "loss": 1.0, "grad_norm": 0.5, "step_time_ms": 1.0},
                {"step": 1, "loss": 0.8, "grad_norm": 0.4, "step_time_ms": 1.1},
                {"step": 2, "loss": 0.6, "grad_norm": 0.3, "step_time_ms": 1.2},
            ],
        )
        self.nb.store_training_curve(
            rid_fail_same_graph,
            [{"step": 0, "loss": 1.2, "grad_norm": 0.7, "step_time_ms": 1.0}],
        )
        self.nb.upsert_leaderboard(
            entry_id="lb_accounting",
            result_id=rid_pass,
            timestamp=1.0,
            model_source="graph_synthesis",
            tier="screening",
            result_cohort="search",
            trust_label="candidate_grade",
            comparability_label="candidate_comparable",
            evaluation_protocol_version="candidate_grade_v1",
        )
        self.nb.flush_writes()

        summary = self.nb.get_data_accounting_summary()

        self.assertEqual(summary["row_volume"]["program_result_rows"], 3)
        self.assertEqual(summary["row_volume"]["training_curve_rows"], 3)
        self.assertEqual(summary["run_volume"]["unique_runs"], 3)
        self.assertEqual(summary["graph_volume"]["unique_graphs"], 2)
        self.assertEqual(summary["graph_volume"]["unique_graph_protocols"], 3)
        self.assertEqual(summary["graph_volume"]["promotable_graphs"], 1)
        self.assertEqual(summary["graph_volume"]["downstream_full_bundle_graphs"], 1)
        self.assertEqual(summary["filtering"]["runs_filtered_pre_s0"], 1)
        self.assertEqual(summary["filtering"]["runs_reaching_s1_pass"], 1)
        self.assertEqual(
            summary["training_curve_density"]["runs_with_training_curves"], 1
        )
        self.assertEqual(
            summary["training_curve_density"]["runs_without_training_curves"], 2
        )
        self.assertEqual(
            summary["leaderboard_tiers"]["screening"]["entries"],
            1,
        )

        dashboard_summary = self.nb.get_dashboard_summary()
        self.assertIn("data_accounting", dashboard_summary)
        self.assertEqual(
            dashboard_summary["data_accounting"]["graph_volume"]["unique_graphs"],
            2,
        )

    def test_candidate_grade_without_complete_provenance_stays_partial(self):
        exp_id = self.nb.start_experiment(
            experiment_type="synthesis",
            config={"n_programs": 1},
            hypothesis="strict comparability",
        )
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="prov_partial_1",
            graph_json='{"nodes": {}}',
            model_source="graph_synthesis",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.5,
            wikitext_perplexity=8.0,
            screening_wikitext_metric_version="screening_wikitext_v1",
        )
        self.nb.flush_writes()
        row = self.nb.get_program_detail(result_id)
        self.assertEqual(row["trust_label"], "candidate_grade")
        self.assertEqual(row["comparability_label"], "partial")
        payload = json.loads(row["data_provenance_json"])
        self.assertFalse(payload["eligible_for_promotion"])
        self.assertFalse(payload["eligible_for_screening_model_training"])
        self.assertEqual(payload["comparability_reason"], "missing_corpus_id")
        self.assertIn("missing_corpus_id", payload["comparability_gaps"])

    def test_candidate_grade_with_wikitext_metric_recovers_missing_metric_version(self):
        exp_id = self.nb.start_experiment(
            experiment_type="investigation",
            config={
                "data_mode": "corpus",
                "corpus_path": "/tmp/corpus.txt",
                "corpus_format": "txt",
                "corpus_text_key": "text",
                "corpus_train_fraction": 0.9,
                "corpus_val_fraction": 0.1,
                "corpus_max_chars": 200000,
                "tokenizer_mode": "tiktoken",
                "tiktoken_encoding": "cl100k_base",
            },
            hypothesis="recover missing screening metric version",
        )
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="prov_recover_1",
            graph_json='{"nodes": {}}',
            model_source="graph_synthesis",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.4,
            wikitext_perplexity=7.5,
            provenance_complete=False,
        )
        self.nb.flush_writes()
        row = self.nb.get_program_detail(result_id)
        self.assertEqual(row["trust_label"], "candidate_grade")
        self.assertEqual(row["comparability_label"], "candidate_comparable")
        payload = json.loads(row["data_provenance_json"])
        self.assertEqual(
            payload["screening_wikitext_metric_version"], "screening_wikitext_v1"
        )
        self.assertTrue(payload["eligible_for_promotion"])
        self.assertEqual(payload["comparability_reason"], "comparable")
        self.assertEqual(payload["comparability_gaps"], [])

    def test_runtime_negative_with_complete_provenance_is_screening_trainable_only(
        self,
    ):
        exp_id = self.nb.start_experiment(
            experiment_type="synthesis",
            config={
                "data_mode": "corpus",
                "corpus_path": "/tmp/corpus.txt",
                "corpus_format": "txt",
                "corpus_text_key": "text",
                "corpus_train_fraction": 0.9,
                "corpus_val_fraction": 0.1,
                "corpus_max_chars": 1000,
                "tokenizer_mode": "tiktoken",
                "tiktoken_encoding": "cl100k_base",
            },
            hypothesis="screening negative",
        )
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="neg123",
            graph_json='{"nodes": {}}',
            model_source="graph_synthesis",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=False,
            novelty_score=0.1,
            screening_wikitext_metric_version="screening_wikitext_v1",
            induction_probe_metric_version="induction_probe_v2",
        )
        self.nb.flush_writes()
        row = self.nb.get_program_detail(result_id)
        payload = json.loads(row["data_provenance_json"])
        self.assertEqual(row["trust_label"], "runtime_observation")
        self.assertEqual(row["comparability_label"], "partial")
        self.assertFalse(payload["eligible_for_promotion"])
        self.assertTrue(payload["eligible_for_screening_model_training"])
        self.assertEqual(payload["screening_model_training_role"], "negative")

    def test_record_program_result_hydrates_provenance_from_experiment_config(self):
        exp_id = self.nb.start_experiment(
            experiment_type="synthesis",
            config={
                "data_mode": "corpus",
                "corpus_path": "/tmp/from_exp_config.txt",
                "corpus_format": "txt",
                "corpus_text_key": "text",
                "corpus_train_fraction": 0.9,
                "corpus_val_fraction": 0.1,
                "corpus_max_chars": 50000,
                "tokenizer_mode": "tiktoken",
                "tiktoken_encoding": "cl100k_base",
            },
            hypothesis="hydrate provenance from experiment config",
        )
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="prov_from_exp_cfg",
            graph_json='{"nodes": {}}',
            model_source="graph_synthesis",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.41,
            wikitext_perplexity=7.4,
            screening_wikitext_metric_version="screening_wikitext_v1",
        )
        self.nb.flush_writes()
        row = self.nb.get_program_detail(result_id)
        payload = json.loads(row["data_provenance_json"])
        self.assertEqual(payload["corpus_id"], "file:/tmp/from_exp_config.txt")
        self.assertEqual(payload["tokenizer_id"], "tiktoken:cl100k_base")
        self.assertTrue(payload["provenance_complete"])
        self.assertEqual(row["comparability_label"], "candidate_comparable")

    def test_record_program_result_hydrates_random_mode_from_experiment_config(self):
        exp_id = self.nb.start_experiment(
            experiment_type="forced_exploration",
            config={
                "data_mode": "random",
                "tokenizer_mode": "byte",
                "vocab_size": 32000,
                "mode": "forced",
                "threshold": 50,
                "device": "cuda",
                "s1_steps": 500,
                "rapid_steps": 150,
            },
            hypothesis="hydrate random-mode provenance from experiment config",
        )
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="prov_random_from_exp_cfg",
            graph_json='{"nodes": {}}',
            model_source="forced_exploration",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.39,
            induction_auc=0.04,
            induction_probe_metric_version="induction_probe_v2",
        )
        self.nb.flush_writes()
        row = self.nb.get_program_detail(result_id)
        payload = json.loads(row["data_provenance_json"])
        self.assertEqual(payload["corpus_id"], "synthetic:random_tokens")
        self.assertEqual(payload["tokenizer_id"], "byte:vocab32000")
        self.assertTrue(payload["provenance_complete"])
        self.assertEqual(payload["experiment_mode"], "forced")
        self.assertEqual(payload["exploration_threshold"], 50)
        self.assertEqual(payload["execution_device"], "cuda")
        self.assertEqual(payload["s1_steps"], 500)
        self.assertEqual(payload["rapid_steps"], 150)
        self.assertEqual(row["trust_label"], "exploratory")
        self.assertEqual(row["comparability_label"], "partial")
        self.assertEqual(row["evaluation_protocol_version"], "forced_exploration_v1")

    def test_record_program_result_hydrates_model_source_from_experiment_config(self):
        exp_id = self.nb.start_experiment(
            experiment_type="synthesis",
            config={
                "model_source": "grammar",
                "data_mode": "corpus",
                "corpus_path": "/tmp/grammar_source.txt",
                "tokenizer_mode": "tiktoken",
                "tiktoken_encoding": "cl100k_base",
            },
            hypothesis="hydrate model source from experiment config",
        )
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="prov_model_source_from_exp_cfg",
            graph_json='{"nodes": {}}',
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.37,
        )
        self.nb.flush_writes()
        row = self.nb.get_program_detail(result_id)
        payload = json.loads(row["data_provenance_json"])
        self.assertEqual(row["model_source"], "grammar")
        self.assertEqual(row["result_cohort"], "search")
        self.assertEqual(row["trust_label"], "candidate_screening")
        self.assertEqual(payload["model_source"], "grammar")
        self.assertEqual(payload["result_cohort"], "search")

    def test_record_program_result_hydrates_backfill_metadata_from_experiment_config(
        self,
    ):
        exp_id = self.nb.start_experiment(
            experiment_type="backfill",
            config={
                "data_mode": "corpus",
                "corpus_path": "/tmp/backfill_corpus.txt",
                "corpus_format": "txt",
                "corpus_text_key": "text",
                "corpus_train_fraction": 0.9,
                "corpus_val_fraction": 0.1,
                "corpus_max_chars": 50000,
                "tokenizer_mode": "tiktoken",
                "tiktoken_encoding": "cl100k_base",
                "backfill_template": "hybrid_sparse_triplet_router",
                "backfill_phase": "stack",
                "backfill_weight_mode": "random",
                "backfill_n_programs": 20,
                "device": "cuda",
            },
            hypothesis="hydrate backfill metadata from experiment config",
        )
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="prov_backfill_cfg",
            graph_json='{"nodes": {}}',
            model_source="graph_synthesis",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.42,
            wikitext_perplexity=7.6,
            screening_wikitext_metric_version="screening_wikitext_v1",
        )
        self.nb.flush_writes()
        row = self.nb.get_program_detail(result_id)
        payload = json.loads(row["data_provenance_json"])
        self.assertEqual(payload["backfill_template"], "hybrid_sparse_triplet_router")
        self.assertEqual(payload["backfill_phase"], "stack")
        self.assertEqual(payload["backfill_weight_mode"], "random")
        self.assertEqual(payload["backfill_n_programs"], 20)
        self.assertEqual(payload["execution_device"], "cuda")
        self.assertEqual(row["trust_label"], "backfill_observation")
        self.assertEqual(row["comparability_label"], "reconstructed_init_variant")

    def test_template_slot_observability_exposes_diagnosis_and_actions(self):
        """Template observability should expose richer evidence for template tuning."""
        exp_id = self.nb.start_experiment(
            experiment_type="synthesis",
            config={"n_programs": 3},
            hypothesis="template observability",
        )
        weak_graph = {
            "metadata": {
                "templates_used": ["attn_sparse_test"],
                "motifs_used": ["motif_a"],
                "template_slot_usage": [
                    {
                        "template_name": "attn_sparse_test",
                        "slot_index": 0,
                        "slot_key": "attn_sparse_test.slot0",
                        "slot_classes": ["attention"],
                        "selected_motif": "motif_a",
                    }
                ],
            }
        }
        strong_graph = {
            "metadata": {
                "templates_used": ["attn_strong_test"],
                "motifs_used": ["motif_b"],
                "template_slot_usage": [],
            }
        }
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="weak_template_fp_1",
            graph_json=json.dumps(weak_graph),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=False,
            loss_ratio=0.92,
            validation_loss_ratio=0.97,
            novelty_confidence=0.4,
            induction_auc=0.0,
            binding_auc=0.0,
            ar_auc=0.0,
            hellaswag_acc=0.25,
            screening_hellaswag_correct=2,
            screening_hellaswag_total=8,
            screening_wikitext_status="ok",
            error_type="insufficient_learning",
        )
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="strong_template_fp_1",
            graph_json=json.dumps(strong_graph),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.32,
            validation_loss_ratio=0.35,
            novelty_confidence=0.8,
            induction_auc=0.08,
            binding_auc=0.09,
            ar_auc=0.07,
            hellaswag_acc=0.31,
            screening_hellaswag_correct=3,
            screening_hellaswag_total=8,
            screening_wikitext_status="ok",
        )
        self.nb.flush_writes()

        with patch(
            "research.scientist.notebook.notebook_misc.TEMPLATES",
            {"attn_sparse_test": object(), "attn_strong_test": object()},
        ):
            summary = self.nb.get_template_slot_observability(limit=8)

        self.assertIn("all_templates", summary)
        by_name = {row["name"]: row for row in summary["all_templates"]}
        self.assertIn("attn_sparse_test", by_name)
        self.assertIn("attn_strong_test", by_name)
        weak = by_name["attn_sparse_test"]
        strong = by_name["attn_strong_test"]
        self.assertEqual(weak["evidence_level"], "insufficient")
        self.assertEqual(weak["structural_category"], "untested")
        self.assertGreaterEqual(len(weak["diagnosis"]), 1)
        self.assertGreaterEqual(len(weak["actions"]), 1)
        self.assertEqual(weak["screening_metric_coverage"]["induction"], 1)
        self.assertEqual(weak["screening_metric_coverage"]["hellaswag"], 2)
        self.assertAlmostEqual(strong["avg_induction_auc"], 0.08)
        self.assertAlmostEqual(strong["avg_binding_auc"], 0.09)
        self.assertEqual(strong["structural_category"], "untested")
        self.assertEqual(summary["summary"]["templates_tracked"], 2)
        self.assertIn("all_slots", summary)
        self.assertEqual(summary["all_slots"][0]["slot_key"], "attn_sparse_test.slot0")

    def test_template_slot_observability_filters_legacy_templates_from_all_templates(
        self,
    ):
        """Legacy template names from historical rows should not appear as active templates."""
        exp_id = self.nb.start_experiment(
            experiment_type="synthesis",
            config={"n_programs": 2},
            hypothesis="template observability legacy filtering",
        )
        active_graph = {
            "metadata": {
                "templates_used": ["attn_active_test"],
                "motifs_used": ["motif_a"],
                "template_slot_usage": [],
            }
        }
        legacy_graph = {
            "metadata": {
                "templates_used": ["0_legacy_template"],
                "motifs_used": ["motif_b"],
                "template_slot_usage": [],
            }
        }
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="active_template_fp",
            graph_json=json.dumps(active_graph),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.31,
            validation_loss_ratio=0.34,
        )
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="legacy_template_fp",
            graph_json=json.dumps(legacy_graph),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=False,
            loss_ratio=0.91,
            validation_loss_ratio=0.95,
            error_type="insufficient_learning",
        )
        self.nb.flush_writes()

        with patch(
            "research.scientist.notebook.notebook_misc.TEMPLATES",
            {"attn_active_test": object()},
        ):
            summary = self.nb.get_template_slot_observability(limit=8)

        active_names = {row["name"] for row in summary["all_templates"]}
        inactive_names = {row["name"] for row in summary["inactive_templates"]}
        self.assertIn("attn_active_test", active_names)
        self.assertNotIn("0_legacy_template", active_names)
        self.assertIn("0_legacy_template", inactive_names)

    def test_template_slot_observability_includes_zero_run_active_templates(self):
        """Active templates should appear in observability even before any runs land."""
        with patch(
            "research.scientist.notebook.notebook_misc.TEMPLATES",
            {
                "codex_ssm_retention_block": object(),
                "codex_ssm_delta_memory_block": object(),
            },
        ):
            summary = self.nb.get_template_slot_observability(limit=8)

        by_name = {row["name"]: row for row in summary["all_templates"]}
        self.assertIn("codex_ssm_retention_block", by_name)
        self.assertIn("codex_ssm_delta_memory_block", by_name)
        self.assertEqual(by_name["codex_ssm_retention_block"]["n_used"], 0)
        self.assertEqual(
            by_name["codex_ssm_retention_block"]["evidence_level"], "insufficient"
        )
        self.assertEqual(
            by_name["codex_ssm_retention_block"]["structural_category"], "untested"
        )
        self.assertIn(
            "Backfill this template before changing weights.",
            by_name["codex_ssm_retention_block"]["actions"],
        )
        self.assertEqual(summary["summary"]["templates_tracked"], 2)

    def test_template_slot_observability_requires_real_support_for_strong_label(self):
        """Strong labels require repeated fingerprint wins, not sparse one-offs."""
        exp_id = self.nb.start_experiment(
            experiment_type="synthesis",
            config={"n_programs": 16},
            hypothesis="template observability evidence-backed labels",
        )
        gpt2_graph = {
            "metadata": {
                "templates_used": ["gpt2_reference"],
                "motifs_used": ["motif_ref"],
                "template_slot_usage": [],
            }
        }
        strong_graph = {
            "metadata": {
                "templates_used": ["attn_family_winner"],
                "motifs_used": ["motif_win"],
                "template_slot_usage": [],
            }
        }
        sparse_graph = {
            "metadata": {
                "templates_used": ["attn_sparse_candidate"],
                "motifs_used": ["motif_sparse"],
                "template_slot_usage": [],
            }
        }
        for idx in range(4):
            self.nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=f"gpt2_ref_fp_{idx}",
                graph_json=json.dumps(gpt2_graph),
                stage0_passed=True,
                stage05_passed=True,
                stage1_passed=True,
                loss_ratio=0.50,
                validation_loss_ratio=0.52,
                induction_auc=0.02,
                binding_auc=0.03,
                ar_auc=0.03,
                hellaswag_acc=0.28,
            )
        for idx in range(12):
            self.nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=f"winner_fp_{idx}",
                graph_json=json.dumps(strong_graph),
                stage0_passed=True,
                stage05_passed=True,
                stage1_passed=True,
                loss_ratio=0.39 if idx < 8 else 0.44,
                validation_loss_ratio=0.43 if idx < 8 else 0.46,
                induction_auc=0.05,
                binding_auc=0.08,
                ar_auc=0.07,
                hellaswag_acc=0.33,
            )
        for idx in range(2):
            self.nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=f"sparse_fp_{idx}",
                graph_json=json.dumps(sparse_graph),
                stage0_passed=True,
                stage05_passed=True,
                stage1_passed=True,
                loss_ratio=0.36,
                validation_loss_ratio=0.40,
                induction_auc=0.06,
                binding_auc=0.09,
                ar_auc=0.08,
                hellaswag_acc=0.34,
            )
        self.nb.flush_writes()

        with patch(
            "research.scientist.notebook.notebook_misc.TEMPLATES",
            {
                "gpt2_reference": object(),
                "attn_family_winner": object(),
                "attn_sparse_candidate": object(),
            },
        ):
            summary = self.nb.get_template_slot_observability(limit=8)

        by_name = {row["name"]: row for row in summary["all_templates"]}
        self.assertEqual(by_name["gpt2_reference"]["structural_category"], "reference")
        self.assertEqual(by_name["attn_family_winner"]["structural_category"], "strong")
        self.assertEqual(
            by_name["attn_sparse_candidate"]["structural_category"], "untested"
        )
        self.assertEqual(by_name["attn_family_winner"]["unique_fingerprints"], 12)
        self.assertEqual(
            by_name["attn_family_winner"]["stage1_unique_fingerprints"], 12
        )
        self.assertGreaterEqual(
            len(by_name["attn_family_winner"]["reference_beating_metrics"]), 2
        )

    def test_template_slot_observability_flags_repeated_low_loss_families(self):
        """Repeated low-loss survivor families should be surfaced explicitly."""
        exp_id = self.nb.start_experiment(
            experiment_type="synthesis",
            config={"n_programs": 4},
            hypothesis="template observability low-loss family",
        )
        depth_graph = {
            "metadata": {
                "templates_used": ["depth_gated_block"],
                "motifs_used": ["attn_latent_compress", "conv_swiglu"],
                "template_slot_usage": [],
            }
        }
        control_graph = {
            "metadata": {
                "templates_used": ["control_template"],
                "motifs_used": ["motif_c"],
                "template_slot_usage": [],
            }
        }
        for idx, loss in enumerate((0.38, 0.41, 0.44), start=1):
            self.nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=f"depth_low_loss_{idx}",
                graph_json=json.dumps(depth_graph),
                stage0_passed=True,
                stage05_passed=True,
                stage1_passed=True,
                loss_ratio=loss,
                validation_loss_ratio=loss + 0.02,
                induction_auc=0.01,
                binding_auc=0.004,
                hellaswag_acc=0.24,
            )
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="control_template_fp",
            graph_json=json.dumps(control_graph),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.52,
            validation_loss_ratio=0.56,
            induction_auc=0.03,
            binding_auc=0.02,
            hellaswag_acc=0.31,
        )
        self.nb.flush_writes()

        with patch(
            "research.scientist.notebook.notebook_misc.TEMPLATES",
            {"depth_gated_block": object(), "control_template": object()},
        ):
            summary = self.nb.get_template_slot_observability(limit=8)

        by_name = {row["name"]: row for row in summary["all_templates"]}
        depth_row = by_name["depth_gated_block"]
        self.assertEqual(depth_row["repeated_low_loss_count"], 3)
        self.assertEqual(depth_row["very_low_loss_count"], 1)
        self.assertTrue(depth_row["repeated_low_loss_family"])
        self.assertAlmostEqual(depth_row["survivor_loss_median"], 0.41)
        self.assertEqual(
            summary["low_loss_template_families"][0]["name"], "depth_gated_block"
        )
        self.assertIn(
            "depth_gated_block",
            summary["summary"]["repeated_low_loss_templates"],
        )

    def test_experiment_trends_stabilize_tiny_runs_with_confidence_fields(self):
        """Tiny-run S1 rates should be damped and expose confidence metadata."""
        tiny_exp = self.nb.start_experiment("synthesis", {}, "tiny")
        self.nb.complete_experiment(
            experiment_id=tiny_exp,
            results={
                "total": 1,
                "stage1_passed": 1,
                "stage0_passed": 1,
                "stage05_passed": 1,
            },
        )

        stable_exp = self.nb.start_experiment("synthesis", {}, "stable")
        self.nb.complete_experiment(
            experiment_id=stable_exp,
            results={
                "total": 80,
                "stage1_passed": 8,
                "stage0_passed": 80,
                "stage05_passed": 40,
            },
        )

        trends = self.nb.get_experiment_trends(limit=10)
        self.assertGreaterEqual(len(trends), 2)

        tiny_entry = next(t for t in trends if t["experiment_id"] == tiny_exp)
        stable_entry = next(t for t in trends if t["experiment_id"] == stable_exp)

        self.assertIn("adjusted_s1_pass_rate", tiny_entry)
        self.assertIn("s1_confidence_lower", tiny_entry)
        self.assertIn("s1_confidence_upper", tiny_entry)
        self.assertIn("s1_confidence_halfwidth", tiny_entry)
        self.assertIn("trend_weight", tiny_entry)
        self.assertIn("trend_confidence", tiny_entry)
        self.assertIn("trend_mode", tiny_entry)

        self.assertLess(tiny_entry["adjusted_s1_pass_rate"], tiny_entry["s1_pass_rate"])
        self.assertLess(tiny_entry["trend_weight"], stable_entry["trend_weight"])
        self.assertEqual(tiny_entry["trend_confidence"], "low")
        self.assertIn(stable_entry["trend_confidence"], {"medium", "high"})

    def test_start_experiment_records_code_version(self):
        """Experiment config should always include code_version metadata."""
        with patch.dict(
            os.environ, {"RESEARCH_CODE_VERSION": "test-version"}, clear=False
        ):
            LabNotebook._cached_code_version = None
            exp_id = self.nb.start_experiment(
                experiment_type="synthesis",
                config={"n_programs": 3},
                hypothesis="version test",
            )

        row = self.nb.conn.execute(
            "SELECT config_json FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        self.assertIsNotNone(row)
        config = json.loads(row["config_json"])
        self.assertEqual(config.get("code_version"), "test-version")

    def test_log_learning_event_accepts_extra_keyword_metadata(self):
        """Learning log should accept extra metadata kwargs without raising."""
        self.nb.log_learning_event(
            "chat_config_adjusted",
            "Adjusted chat config",
            changes={"max_depth": 4},
            ignored={"unknown_key": 1},
        )
        rows = self.nb.get_learning_log(limit=5)
        self.assertGreaterEqual(len(rows), 1)
        latest = rows[0]
        self.assertEqual(latest["event_type"], "chat_config_adjusted")
        self.assertIsInstance(latest.get("evidence"), str)
        self.assertIn("changes", latest.get("evidence") or "")

    def test_log_learning_event_tolerates_sqlite_operational_error(self):
        """Learning log should be best-effort and not abort on SQLite write failure."""
        with patch.object(
            _ThreadSafeConnectionWrapper,
            "execute",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            self.nb.log_learning_event(
                "chat_config_adjusted",
                "Adjusted chat config",
                changes={"max_depth": 4},
            )

    def test_create_preregistration_tolerates_sqlite_operational_error(self):
        """Preregistration creation should retry on a degraded primary connection."""
        preregistration = {
            "hypothesis": {
                "statement": "Short runs will still persist preregistration rows.",
                "variables": ["runtime_events"],
                "expected_direction": "positive",
                "success_criteria": "row exists",
            },
            "analysis_plan": {
                "primary_metrics": ["launch_success"],
                "secondary_metrics": ["status_visibility"],
                "thresholds": {"launch_success": 1.0},
                "baseline_comparison": "pre-fallback behavior",
            },
            "falsification_conditions": ["insert does not persist"],
            "confounders_checklist": [{"name": "sqlite_primary_connection", "checked": True}],
            "exploratory": False,
        }

        original_execute = _ThreadSafeConnectionWrapper.execute

        def flaky_execute(conn, sql, params=()):
            if "INSERT INTO hypothesis_preregistrations" in sql:
                raise sqlite3.OperationalError("disk I/O error")
            return original_execute(conn, sql, params)

        with patch.object(
            _ThreadSafeConnectionWrapper,
            "execute",
            autospec=True,
            side_effect=flaky_execute,
        ):
            prereg_id = self.nb.create_preregistration(
                experiment_type="synthesis",
                preregistration=preregistration,
            )

        row = self.nb.conn.execute(
            """SELECT preregistration_id, status
               FROM hypothesis_preregistrations
               WHERE preregistration_id = ?""",
            (prereg_id,),
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["status"], "registered")

    def test_leaderboard_upsert_and_query(self):
        """Leaderboard CRUD operations."""
        # Create an experiment and program first
        exp_id = self.nb.start_experiment("synthesis", {}, "test")
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp001",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.4,
            novelty_score=0.8,
        )

        # Upsert to leaderboard
        entry_id = self.nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            architecture_desc="test_arch",
            screening_loss_ratio=0.4,
            screening_novelty=0.8,
            screening_passed=True,
            tier="screening",
        )
        self.assertIsNotNone(entry_id)

        # Query leaderboard
        entries = self.nb.get_leaderboard()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["tier"], "screening")
        self.assertGreater(entries[0]["composite_score"], 0)

        # Filter by tier
        screening = self.nb.get_leaderboard(tier="screening")
        self.assertEqual(len(screening), 1)
        empty = self.nb.get_leaderboard(tier="investigation")
        self.assertEqual(len(empty), 0)

    def test_leaderboard_upsert_updates_existing(self):
        """Upserting same result_id updates, doesn't duplicate."""
        exp_id = self.nb.start_experiment("synthesis", {}, "test")
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp002",
            graph_json="{}",
            model_source="graph_synthesis",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.5,
            wikitext_perplexity=7.2,
            data_mode="corpus",
            corpus_path="/tmp/upsert_existing_corpus.txt",
            corpus_format="txt",
            corpus_text_key="text",
            corpus_train_fraction=0.9,
            corpus_val_fraction=0.1,
            corpus_max_chars=12000,
            tokenizer_mode="tiktoken",
            tiktoken_encoding="cl100k_base",
            screening_wikitext_metric_version="screening_wikitext_v1",
        )

        # First upsert
        self.nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            screening_loss_ratio=0.5,
            tier="screening",
        )

        # Second upsert with investigation data
        self.nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            screening_loss_ratio=0.5,
            investigation_loss_ratio=0.4,
            investigation_robustness=0.6,
            tier="investigation",
        )

        entries = self.nb.get_leaderboard()
        self.assertEqual(len(entries), 1)  # not duplicated
        self.assertEqual(entries[0]["tier"], "investigation")
        self.assertAlmostEqual(entries[0]["investigation_robustness"], 0.6)

    def test_promote_to_tier(self):
        """Promoting a leaderboard entry updates tier + results."""
        exp_id = self.nb.start_experiment("synthesis", {}, "test")
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp003",
            graph_json="{}",
            model_source="graph_synthesis",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.3,
            wikitext_perplexity=7.0,
            data_mode="corpus",
            corpus_path="/tmp/promote_corpus.txt",
            corpus_format="txt",
            corpus_text_key="text",
            corpus_train_fraction=0.9,
            corpus_val_fraction=0.1,
            corpus_max_chars=10000,
            tokenizer_mode="tiktoken",
            tiktoken_encoding="cl100k_base",
            screening_wikitext_metric_version="screening_wikitext_v1",
        )

        entry_id = self.nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            screening_loss_ratio=0.3,
            tier="screening",
        )

        # Promote to investigation
        self.nb.promote_to_tier(
            entry_id=entry_id,
            tier="investigation",
            investigation_loss_ratio=0.25,
            investigation_robustness=0.7,
        )

        entries = self.nb.get_leaderboard()
        self.assertEqual(entries[0]["tier"], "investigation")

    def test_promote_to_tier_blocks_partial_candidate_rows(self):
        exp_id = self.nb.start_experiment("synthesis", {}, "test")
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_partial_promote",
            graph_json="{}",
            model_source="graph_synthesis",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.4,
            wikitext_perplexity=8.1,
            screening_wikitext_metric_version="screening_wikitext_v1",
        )
        entry_id = self.nb.upsert_leaderboard(
            result_id=result_id,
            model_source="graph_synthesis",
            screening_loss_ratio=0.4,
            tier="screening",
        )

        self.nb.promote_to_tier(
            entry_id=entry_id,
            tier="validation",
            validation_loss_ratio=0.35,
        )

        entry = self.nb.get_leaderboard_entry(result_id)
        self.assertEqual(entry["tier"], "screening")

    def test_upsert_leaderboard_uses_wikitext_and_investigation_flags(self):
        """Leaderboard scoring should incorporate real-token quality and failed investigation evidence."""
        exp_id = self.nb.start_experiment("synthesis", {}, "test")
        rid_a = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_wt_a",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.02,
            novelty_score=0.8,
            novelty_confidence=0.8,
        )
        rid_b = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_wt_b",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.02,
            novelty_score=0.8,
            novelty_confidence=0.8,
        )

        e_weak = self.nb.upsert_leaderboard(
            result_id=rid_a,
            model_source="graph_synthesis",
            screening_loss_ratio=0.02,
            screening_novelty=0.8,
            investigation_loss_ratio=0.2,
            investigation_robustness=0.3333,
            investigation_passed=False,
            validation_passed=False,
            wikitext_score=0.45,
            wikitext_perplexity=180.0,
            tier="screened_out",
        )
        e_strong = self.nb.upsert_leaderboard(
            result_id=rid_b,
            model_source="graph_synthesis",
            screening_loss_ratio=0.02,
            screening_novelty=0.8,
            investigation_loss_ratio=0.2,
            investigation_robustness=0.3333,
            investigation_passed=False,
            validation_passed=False,
            wikitext_score=0.67,
            wikitext_perplexity=35.0,
            tier="screened_out",
        )

        rows = {row["entry_id"]: row for row in self.nb.get_leaderboard(limit=10)}
        self.assertGreater(
            rows[e_strong]["composite_score"], rows[e_weak]["composite_score"]
        )

    def test_upsert_leaderboard_uses_shared_score_kwargs_builder(self):
        """Live leaderboard scoring should match the shared scoring-kwargs path."""
        from research.scientist.leaderboard_scoring import build_score_kwargs

        exp_id = self.nb.start_experiment("synthesis", {}, "score parity")
        rid = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_score_parity",
            graph_json="{}",
            stage1_passed=True,
            novelty_confidence=0.7,
        )
        entry_id = self.nb.upsert_leaderboard(
            result_id=rid,
            model_source="graph_synthesis",
            screening_loss_ratio=0.08,
            screening_novelty=0.9,
            routing_savings_ratio=0.4,
            wikitext_score=0.55,
            wikitext_perplexity=120.0,
            investigation_passed=False,
            validation_passed=False,
        )
        row = self.nb.conn.execute(
            "SELECT * FROM leaderboard WHERE entry_id = ?",
            (entry_id,),
        ).fetchone()
        expected = self.nb.compute_composite_score(
            **build_score_kwargs(
                self.nb.conn,
                self.nb,
                rid,
                dict(row),
                False,
            )
        )
        self.assertAlmostEqual(row["composite_score"], expected, places=6)

    def test_screening_wikitext_metadata_round_trips(self):
        """Screening WikiText provenance fields should persist without ad hoc blobs."""
        exp_id = self.nb.start_experiment("synthesis", {}, "wiki metadata")
        rid = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_wiki_meta",
            graph_json="{}",
            stage1_passed=True,
            wikitext_perplexity=48.5,
            wikitext_score=0.62,
            wikitext_pre_perplexity=210.0,
            wikitext_ppl_improvement=0.231,
            screening_wikitext_status="ok",
            screening_wikitext_metric_version="screening_wikitext_v1",
            screening_wikitext_variant="wikitext-2-raw-v1",
            screening_wikitext_elapsed_ms=2345.6,
            screening_wikitext_budget_json='{"n_train_steps":50}',
        )
        self.nb.flush_writes()
        self.nb.upsert_leaderboard(
            result_id=rid,
            model_source="graph_synthesis",
            screening_loss_ratio=0.05,
            screening_novelty=0.7,
            wikitext_perplexity=48.5,
            wikitext_score=0.62,
            wikitext_pre_perplexity=210.0,
            wikitext_ppl_improvement=0.231,
            screening_wikitext_status="ok",
            screening_wikitext_metric_version="screening_wikitext_v1",
            screening_wikitext_variant="wikitext-2-raw-v1",
            screening_wikitext_elapsed_ms=2345.6,
            screening_wikitext_budget_json='{"n_train_steps":50}',
        )

        pr = self.nb.conn.execute(
            "SELECT wikitext_pre_perplexity, wikitext_ppl_improvement, "
            "screening_wikitext_status, screening_wikitext_metric_version, "
            "screening_wikitext_variant, screening_wikitext_elapsed_ms, "
            "screening_wikitext_budget_json "
            "FROM program_results WHERE result_id = ?",
            (rid,),
        ).fetchone()
        lb = self.nb.get_leaderboard(limit=5)[0]

        self.assertEqual(pr["screening_wikitext_status"], "ok")
        self.assertEqual(
            pr["screening_wikitext_metric_version"], "screening_wikitext_v1"
        )
        self.assertEqual(pr["screening_wikitext_variant"], "wikitext-2-raw-v1")
        self.assertAlmostEqual(pr["wikitext_pre_perplexity"], 210.0)
        self.assertAlmostEqual(pr["wikitext_ppl_improvement"], 0.231)
        self.assertAlmostEqual(pr["screening_wikitext_elapsed_ms"], 2345.6)
        self.assertEqual(pr["screening_wikitext_budget_json"], '{"n_train_steps":50}')
        self.assertEqual(lb["screening_wikitext_status"], "ok")
        self.assertEqual(
            lb["screening_wikitext_metric_version"], "screening_wikitext_v1"
        )

    def test_external_benchmarks_merge_screening_payload(self):
        """Screening payload should merge into external_benchmarks_json cleanly."""
        payload = {
            "screening_wikitext": {
                "metric_version": "screening_wikitext_v1",
                "status": "ok",
                "metrics": {"wikitext_perplexity": 48.5},
            }
        }
        exp_id = self.nb.start_experiment("synthesis", {}, "external payload")
        rid = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_external_payload",
            graph_json="{}",
            stage1_passed=True,
        )
        self.nb.flush_writes()
        self.assertTrue(self.nb.set_external_benchmarks(rid, payload))

        row = self.nb.conn.execute(
            "SELECT external_benchmarks_json FROM program_results WHERE result_id = ?",
            (rid,),
        ).fetchone()
        parsed = json.loads(row["external_benchmarks_json"])
        self.assertEqual(
            parsed["screening_wikitext"]["metrics"]["wikitext_perplexity"],
            48.5,
        )

    def test_screening_probe_metadata_round_trips(self):
        """Rapid-screening and binding-probe metadata should persist explicitly."""
        exp_id = self.nb.start_experiment("synthesis", {}, "probe metadata")
        rid = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_probe_meta",
            graph_json="{}",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            rapid_screening_passed=True,
            rapid_screening_elapsed_ms=1812.0,
            rapid_screening_steps_completed=150,
            rapid_screening_max_steps=150,
            rapid_screening_degraded=False,
            rapid_screening_degraded_reasons_json='["loss_spike"]',
            rapid_screening_metrics_json='{"steps_completed":150,"has_routing":false}',
            induction_auc=0.008,
            induction_gap_accuracies_json='{"4":0.1,"8":0.2}',
            induction_probe_train_steps=1000,
            induction_probe_eval_examples=100,
            induction_probe_batch_size=16,
            induction_probe_gaps_json="[4,8,16,32,64]",
            induction_probe_elapsed_ms=5976.0,
            binding_auc=0.006,
            binding_distance_accuracies_json='{"2":0.2,"4":0.1}',
            binding_probe_eval_examples=100,
            binding_probe_distances_json="[2,4,8,16,32,64]",
            binding_probe_elapsed_ms=606.0,
            binding_composite=0.004,
            train_budget_steps=500,
            screening_hellaswag_correct=7,
            screening_hellaswag_total=10,
            screening_hellaswag_elapsed_ms=321.0,
        )
        self.nb.flush_writes()
        row = self.nb.conn.execute(
            "SELECT rapid_screening_passed, rapid_screening_elapsed_ms, "
            "rapid_screening_steps_completed, rapid_screening_max_steps, "
            "rapid_screening_degraded_reasons_json, rapid_screening_metrics_json, "
            "induction_auc, induction_gap_accuracies_json, "
            "induction_probe_train_steps, induction_probe_eval_examples, "
            "induction_probe_batch_size, induction_probe_gaps_json, "
            "induction_probe_elapsed_ms, binding_auc, "
            "binding_distance_accuracies_json, binding_probe_eval_examples, "
            "binding_probe_distances_json, binding_probe_elapsed_ms, "
            "binding_composite, train_budget_steps, "
            "screening_hellaswag_correct, screening_hellaswag_total, "
            "screening_hellaswag_elapsed_ms "
            "FROM program_results WHERE result_id = ?",
            (rid,),
        ).fetchone()

        self.assertEqual(row["rapid_screening_passed"], 1)
        self.assertAlmostEqual(row["rapid_screening_elapsed_ms"], 1812.0)
        self.assertEqual(row["rapid_screening_steps_completed"], 150)
        self.assertEqual(row["rapid_screening_max_steps"], 150)
        self.assertEqual(row["rapid_screening_degraded_reasons_json"], '["loss_spike"]')
        self.assertEqual(
            row["rapid_screening_metrics_json"],
            '{"steps_completed":150,"has_routing":false}',
        )
        self.assertAlmostEqual(row["induction_auc"], 0.008)
        self.assertEqual(row["induction_gap_accuracies_json"], '{"4":0.1,"8":0.2}')
        self.assertEqual(row["induction_probe_train_steps"], 1000)
        self.assertEqual(row["induction_probe_eval_examples"], 100)
        self.assertEqual(row["induction_probe_batch_size"], 16)
        self.assertEqual(row["induction_probe_gaps_json"], "[4,8,16,32,64]")
        self.assertAlmostEqual(row["induction_probe_elapsed_ms"], 5976.0)
        self.assertAlmostEqual(row["binding_auc"], 0.006)
        self.assertEqual(row["binding_distance_accuracies_json"], '{"2":0.2,"4":0.1}')
        self.assertEqual(row["binding_probe_eval_examples"], 100)
        self.assertEqual(row["binding_probe_distances_json"], "[2,4,8,16,32,64]")
        self.assertAlmostEqual(row["binding_probe_elapsed_ms"], 606.0)
        self.assertAlmostEqual(row["binding_composite"], 0.004)
        self.assertEqual(row["train_budget_steps"], 500)
        self.assertEqual(row["screening_hellaswag_correct"], 7)
        self.assertEqual(row["screening_hellaswag_total"], 10)
        self.assertAlmostEqual(row["screening_hellaswag_elapsed_ms"], 321.0)

    def test_insights_crud(self):
        """Record and query insights."""
        exp_id = self.nb.start_experiment("synthesis", {}, "test")
        insight_id = self.nb.record_insight(
            category="pattern",
            content="Test insight content",
            experiment_id=exp_id,
            confidence=0.8,
        )
        self.assertIsNotNone(insight_id)

        insights = self.nb.get_insights()
        self.assertEqual(len(insights), 1)
        self.assertEqual(insights[0]["category"], "pattern")

    def test_entries_crud(self):
        """Add and query notebook entries."""
        entry_id = self.nb.add_entry(
            ExperimentEntry(
                entry_type="decision",
                title="Test Decision",
                content="We decided to test things.",
            )
        )
        self.assertIsNotNone(entry_id)

        entries = self.nb.get_entries()
        self.assertGreater(len(entries), 0)

    def test_add_entry_tolerates_sqlite_operational_error(self):
        """Notebook entries should be best-effort on a degraded primary connection."""
        with patch.object(
            _ThreadSafeConnectionWrapper,
            "execute",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            entry_id = self.nb.add_entry(
                ExperimentEntry(
                    entry_type="observation",
                    title="Transient entry",
                    content="Should not abort the run.",
                    experiment_id="exp_transient_1",
                )
            )
        self.assertIsNotNone(entry_id)

    def test_has_fingerprint_tolerates_sqlite_operational_error(self):
        """Fingerprint existence checks should degrade to a safe miss on SQLite failure."""
        with patch.object(
            _ThreadSafeConnectionWrapper,
            "execute",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            self.assertFalse(self.nb.has_fingerprint("fp_transient_1"))

    def test_create_healer_task_tolerates_sqlite_operational_error(self):
        """Healer bookkeeping should not cascade another failure during recovery."""
        with patch.object(
            _ThreadSafeConnectionWrapper,
            "execute",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            task_id = self.nb.create_healer_task(
                experiment_id="exp_transient_2",
                trigger_type="runtime_error",
                scope="test scope",
                reproduction_steps=["python -m pytest research/tests/test_notebook.py"],
                acceptance_tests=["python -m pytest research/tests/test_notebook.py"],
                model_endpoint="local",
                sandbox_policy={"allowed_commands": ["python -m pytest"]},
                trigger_payload={"source": "unit-test"},
            )
        self.assertTrue(task_id.startswith("heal-"))

    def test_add_healer_event_tolerates_sqlite_operational_error(self):
        """Healer event logging should be best-effort under SQLite failures."""
        with patch.object(
            _ThreadSafeConnectionWrapper,
            "execute",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            event_id = self.nb.add_healer_event(
                "heal-transient-1",
                "opened",
                state="open",
                payload={"source": "unit-test"},
            )
        self.assertIsNotNone(event_id)

    def test_get_recent_experiments_tolerates_sqlite_operational_error(self):
        """Recent history queries should degrade to an empty list on SQLite failure."""
        with patch.object(
            _ThreadSafeConnectionWrapper,
            "execute",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            self.assertEqual(self.nb.get_recent_experiments(5), [])

    def test_get_insights_tolerates_sqlite_operational_error(self):
        """Insight queries should degrade to empty results on SQLite failure."""
        with patch.object(
            _ThreadSafeConnectionWrapper,
            "execute",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            self.assertEqual(self.nb.get_insights(limit=5), [])

    def test_get_knowledge_tolerates_sqlite_operational_error(self):
        """Knowledge queries should degrade to empty results on SQLite failure."""
        with patch.object(
            _ThreadSafeConnectionWrapper,
            "execute",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            self.assertEqual(self.nb.get_knowledge(), [])

    def test_record_insight_tolerates_sqlite_operational_error(self):
        """Insight persistence should be best-effort under SQLite failure."""
        with patch.object(
            _ThreadSafeConnectionWrapper,
            "execute",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            insight_id = self.nb.record_insight(
                category="failure_mode",
                content="Transient failure-mode insight",
                experiment_id="exp_transient_3",
                confidence=0.7,
            )
        self.assertIsNotNone(insight_id)

    def test_get_healer_task_tolerates_sqlite_operational_error(self):
        """Healer task lookup should degrade to None on SQLite failure."""
        with patch.object(
            _ThreadSafeConnectionWrapper,
            "execute",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            self.assertIsNone(self.nb.get_healer_task("heal-transient-2"))

    def test_get_recent_healer_tasks_tolerates_sqlite_operational_error(self):
        """Recent healer task queries should degrade to empty results on SQLite failure."""
        with patch.object(
            _ThreadSafeConnectionWrapper,
            "execute",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            self.assertEqual(self.nb.get_recent_healer_tasks(5), [])

    def test_get_leaderboard_tolerates_sqlite_operational_error(self):
        """Leaderboard reads should degrade to empty results on SQLite failure."""
        with patch.object(
            _ThreadSafeConnectionWrapper,
            "execute",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            self.assertEqual(self.nb.get_leaderboard(limit=5), [])

    def test_complete_experiment_tolerates_sqlite_operational_error(self):
        """Completion persistence should be best-effort under SQLite failure."""
        exp_id = self.nb.start_experiment(
            experiment_type="synthesis",
            config={"n_programs": 1},
            hypothesis="completion degradation",
        )

        with patch.object(
            _ThreadSafeConnectionWrapper,
            "execute",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            self.nb.complete_experiment(
                experiment_id=exp_id,
                results={"stage1_passed": 1, "total": 1},
                aria_summary="Completed despite notebook I/O failure",
            )

    def test_fail_experiment_tolerates_sqlite_operational_error(self):
        """Failure persistence should be best-effort under SQLite failure."""
        exp_id = self.nb.start_experiment(
            experiment_type="synthesis",
            config={"n_programs": 1},
            hypothesis="failure degradation",
        )

        with patch.object(
            _ThreadSafeConnectionWrapper,
            "execute",
            side_effect=sqlite3.OperationalError("disk I/O error"),
        ):
            self.nb.fail_experiment(
                experiment_id=exp_id,
                error="disk I/O error",
                results={"total": 1},
            )

    def test_dashboard_summary(self):
        """Dashboard summary returns expected keys."""
        summary = self.nb.get_dashboard_summary()
        expected_keys = [
            "total_experiments",
            "total_programs_evaluated",
            "stage1_survivors",
            "survival_rate",
        ]
        for k in expected_keys:
            self.assertIn(k, summary)

    def test_dashboard_headline_summary_skips_heavy_sections(self):
        """Headline summary must not compute observability or data-accounting payloads."""
        with (
            patch.object(
                self.nb,
                "get_data_accounting_summary",
                side_effect=AssertionError("data accounting should not run"),
            ),
            patch.object(
                self.nb,
                "get_template_slot_observability",
                side_effect=AssertionError("template observability should not run"),
            ),
        ):
            summary = self.nb.get_dashboard_headline_summary()

        self.assertEqual(
            summary["data_accounting"],
            {
                "row_volume": {},
                "run_volume": {},
                "graph_volume": {},
                "filtering": {},
                "training_curve_density": {},
                "leaderboard_tiers": {},
            },
        )
        self.assertEqual(summary["template_observability"], {})
        self.assertIn("total_programs_evaluated", summary)

    def test_dashboard_summary_cache_invalidates_after_write(self):
        """Dashboard summary cache should refresh after notebook writes commit."""
        before = self.nb.get_dashboard_summary()
        exp_id = self.nb.start_experiment("synthesis", {}, "cache-refresh")
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_cache_refresh",
            graph_json="{}",
            stage1_passed=True,
        )
        self.nb.flush_writes()
        after = self.nb.get_dashboard_summary()
        self.assertEqual(
            after["total_programs_evaluated"], before["total_programs_evaluated"] + 1
        )
        self.assertEqual(after["stage1_survivors"], before["stage1_survivors"] + 1)

    def test_graph_structural_counts_supports_dict_nodes(self):
        """Structural op counting should handle dict-based graph JSON without refetching repeatedly."""
        exp_id = self.nb.start_experiment("synthesis", {}, "counts")
        graph_json = json.dumps(
            {
                "nodes": {
                    "a": {"op_name": "route_topk"},
                    "b": {"op_name": "block_sparse_linear"},
                    "c": {"op_name": "moe_topk"},
                }
            }
        )
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_counts",
            graph_json=graph_json,
            stage1_passed=True,
        )
        self.nb.flush_writes()

        counts = self.nb._graph_structural_counts(result_id)
        self.assertEqual(counts["routing"], 2)
        self.assertEqual(counts["sparse"], 1)
        self.assertEqual(counts["moe"], 2)

    def test_get_top_programs_sort_options(self):
        """get_top_programs supports all sort options."""
        exp_id = self.nb.start_experiment("synthesis", {}, "test")
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_sort",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.5,
            novelty_score=0.6,
        )

        for sort_by in [
            "novelty_score",
            "loss_ratio",
            "structural_novelty",
            "behavioral_novelty",
            "validation_loss_ratio",
            "discovery_loss_ratio",
        ]:
            programs = self.nb.get_top_programs(5, sort_by=sort_by)
            self.assertIsInstance(programs, list)

    def test_training_curve_storage(self):
        """Store and retrieve training curves."""
        exp_id = self.nb.start_experiment("synthesis", {}, "test")
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_curve",
            graph_json="{}",
            stage1_passed=True,
        )
        self.nb.flush_writes()

        curve = [
            {"step": 0, "loss": 5.0, "grad_norm": 1.0},
            {"step": 1, "loss": 4.5, "grad_norm": 0.9},
            {"step": 2, "loss": 4.0, "grad_norm": 0.8},
        ]
        self.nb.store_training_curve(result_id, curve)

        retrieved = self.nb.get_training_curve(result_id)
        self.assertEqual(len(retrieved), 3)
        self.assertAlmostEqual(retrieved[0]["loss"], 5.0)

    def test_training_curve_ignored_without_persisted_survivor(self):
        """Curve writes should be skipped when the parent result does not exist."""
        self.nb.store_training_curve(
            "missing_result",
            [
                {"step": 0, "loss": 1.0, "grad_norm": 0.5},
            ],
        )
        retrieved = self.nb.get_training_curve("missing_result")
        self.assertEqual(retrieved, [])

    def test_training_curve_deleted_with_parent_program(self):
        """Purging junk programs should cascade and remove attached curves."""
        exp_id = self.nb.start_experiment("synthesis", {}, "junk")
        result_id = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_junk",
            graph_json="{}",
            stage0_passed=False,
            stage1_passed=False,
            error_type="compile_error",
        )
        self.nb.flush_writes()
        self.nb.conn.execute(
            """INSERT INTO training_curves (result_id, step, loss, grad_norm, step_time_ms)
               VALUES (?, 0, 3.0, 1.0, 1.0)""",
            (result_id,),
        )
        self.nb.conn.commit()

        deleted = self.nb.purge_junk_programs()
        self.assertEqual(deleted["deleted"], 1)
        remaining = self.nb.conn.execute(
            "SELECT COUNT(*) FROM training_curves WHERE result_id = ?",
            (result_id,),
        ).fetchone()[0]
        self.assertEqual(remaining, 0)

    def test_top_op_combinations_handles_malformed_graph_json(self):
        """Analytics top_op_combinations should skip malformed JSON and still aggregate valid pairs."""
        from research.scientist.analytics import ExperimentAnalytics

        exp_id = self.nb.start_experiment("synthesis", {}, "combo-test")
        valid_graph_1 = json.dumps(
            {
                "nodes": {
                    "a": {"op_name": "relu"},
                    "b": {"op_name": "gelu"},
                    "c": {"op_name": "input"},
                },
                "variant": 1,
            }
        )
        valid_graph_2 = json.dumps(
            {
                "nodes": {
                    "a": {"op_name": "relu"},
                    "b": {"op_name": "gelu"},
                    "c": {"op_name": "input"},
                },
                "variant": 2,
            }
        )
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_combo_1",
            graph_json=valid_graph_1,
            stage1_passed=True,
            novelty_score=0.7,
        )
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_combo_2",
            graph_json=valid_graph_2,
            stage1_passed=True,
            novelty_score=0.5,
        )
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_combo_bad",
            graph_json="{bad-json",
            stage1_passed=True,
            novelty_score=0.9,
        )

        self.nb.flush_writes()

        analytics = ExperimentAnalytics(self.nb)
        combos = analytics.top_op_combinations(n=3)

        self.assertGreaterEqual(len(combos), 1)
        first = combos[0]
        self.assertEqual(first["ops"], ["gelu", "relu"])
        self.assertEqual(first["count"], 2)

    def test_experiment_clusters_model_selection_and_consensus(self):
        """experiment_clusters should select a sensible k and expose model-selection diagnostics."""
        from research.scientist.analytics import ExperimentAnalytics

        cluster_a = [
            {"s1": 74, "novelty": 1.25, "loss": 0.58, "duration": 95.0},
            {"s1": 70, "novelty": 1.18, "loss": 0.62, "duration": 92.0},
            {"s1": 77, "novelty": 1.32, "loss": 0.55, "duration": 98.0},
        ]
        cluster_b = [
            {"s1": 9, "novelty": 0.12, "loss": 1.62, "duration": 36.0},
            {"s1": 6, "novelty": 0.08, "loss": 1.74, "duration": 32.0},
            {"s1": 11, "novelty": 0.15, "loss": 1.58, "duration": 38.0},
        ]

        for i, spec in enumerate(cluster_a + cluster_b):
            exp_id = self.nb.start_experiment(
                "synthesis", {"n_programs": 100}, f"cluster-{i}"
            )
            self.nb.complete_experiment(
                exp_id,
                {
                    "total": 100,
                    "stage0_passed": 100,
                    "stage05_passed": 100,
                    "stage1_passed": spec["s1"],
                    "best_loss_ratio": spec["loss"],
                    "best_novelty_score": spec["novelty"],
                },
            )
            self.nb.conn.execute(
                "UPDATE experiments SET duration_seconds = ? WHERE experiment_id = ?",
                (spec["duration"], exp_id),
            )
        self.nb.conn.commit()

        clusters = ExperimentAnalytics(self.nb).experiment_clusters(n_clusters=5)
        self.assertIsNotNone(clusters)
        self.assertEqual(clusters["n_experiments"], 6)
        self.assertEqual(clusters["n_clusters"], 2)
        self.assertIn("model_selection", clusters)

        model_selection = clusters["model_selection"]
        self.assertIn("candidate_ks", model_selection)
        self.assertIn("selected_k", model_selection)
        self.assertIn("silhouette", model_selection)
        self.assertIn("consensus", model_selection)
        self.assertGreaterEqual(model_selection["selected_k"], 2)
        self.assertLessEqual(model_selection["selected_k"], 5)

        avg_s1_rates = sorted(c["avg_s1_rate"] for c in clusters["clusters"])
        self.assertLess(avg_s1_rates[0], 0.2)
        self.assertGreater(avg_s1_rates[-1], 0.65)
        self.assertGreater(clusters["stability_score"], 0.6)

    def test_experiment_clusters_include_failure_signature_features(self):
        """Clustering should use failure signatures to separate experiments with similar top-level metrics."""
        from research.scientist.analytics import ExperimentAnalytics

        for i in range(3):
            exp_id = self.nb.start_experiment(
                "synthesis", {"n_programs": 10}, f"compile-heavy-{i}"
            )
            for j in range(10):
                is_success = j >= 5
                self.nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=f"fp_compile_{i}_{j}",
                    graph_json="{}",
                    stage0_passed=is_success,
                    stage05_passed=is_success,
                    stage1_passed=is_success,
                    error_type=None if is_success else "compile_error",
                    stage_at_death=None if is_success else "stage0",
                )
            self.nb.complete_experiment(
                exp_id,
                {
                    "total": 10,
                    "stage0_passed": 5,
                    "stage05_passed": 5,
                    "stage1_passed": 5,
                    "best_loss_ratio": 0.9,
                    "best_novelty_score": 0.45,
                },
            )
            self.nb.conn.execute(
                "UPDATE experiments SET duration_seconds = ? WHERE experiment_id = ?",
                (60.0 + i, exp_id),
            )

        stage1_errors = ["nan_output", "overflow", "timeout", "nan_output", "overflow"]
        for i in range(3):
            exp_id = self.nb.start_experiment(
                "synthesis", {"n_programs": 10}, f"stage1-heavy-{i}"
            )
            for j in range(10):
                is_success = j >= 5
                self.nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=f"fp_stage1_{i}_{j}",
                    graph_json="{}",
                    stage0_passed=True,
                    stage05_passed=True,
                    stage1_passed=is_success,
                    error_type=None if is_success else stage1_errors[j],
                    stage_at_death=None if is_success else "stage1",
                )
            self.nb.complete_experiment(
                exp_id,
                {
                    "total": 10,
                    "stage0_passed": 10,
                    "stage05_passed": 10,
                    "stage1_passed": 5,
                    "best_loss_ratio": 0.9,
                    "best_novelty_score": 0.45,
                },
            )
            self.nb.conn.execute(
                "UPDATE experiments SET duration_seconds = ? WHERE experiment_id = ?",
                (60.0 + i, exp_id),
            )

        self.nb.flush_writes()
        self.nb.conn.commit()

        clusters = ExperimentAnalytics(self.nb).experiment_clusters(n_clusters=4)
        self.assertIsNotNone(clusters)
        self.assertEqual(clusters["n_experiments"], 6)
        self.assertIn("compile_fail_rate", clusters["feature_keys"])
        self.assertIn("error_diversity", clusters["feature_keys"])
        self.assertIn("model_selection", clusters)
        self.assertIn(clusters["model_selection"]["selected_k"], (2, 3))

        compile_rates = sorted(c["avg_compile_fail_rate"] for c in clusters["clusters"])
        stage1_rates = sorted(c["avg_stage1_fail_rate"] for c in clusters["clusters"])
        error_diversities = sorted(
            c["avg_error_diversity"] for c in clusters["clusters"]
        )

        self.assertLess(compile_rates[0], 0.1)
        self.assertGreater(compile_rates[-1], 0.4)
        self.assertLess(stage1_rates[0], 0.1)
        self.assertGreater(stage1_rates[-1], 0.4)
        self.assertLess(error_diversities[0], 0.1)
        self.assertGreater(error_diversities[-1], 0.5)

    def test_experiment_clusters_include_trajectory_features(self):
        """Clustering should separate experiments with matched aggregates but opposite temporal trajectories."""
        from research.scientist.analytics import ExperimentAnalytics

        pending_updates = []

        def _record_experiment_with_trajectory(
            prefix: str, improving: bool, offset: float
        ):
            exp_id = self.nb.start_experiment("synthesis", {"n_programs": 10}, prefix)
            base_ts = 1000.0 + offset

            for step in range(10):
                if improving:
                    is_success = step >= 5
                    novelty = 0.2 + (0.06 * step)
                    loss_ratio = 1.3 - (0.06 * step)
                else:
                    is_success = step < 5
                    novelty = 0.8 - (0.06 * step)
                    loss_ratio = 0.7 + (0.06 * step)

                result_id = self.nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=f"fp_{prefix}_{step}",
                    graph_json="{}",
                    stage0_passed=True,
                    stage05_passed=True,
                    stage1_passed=is_success,
                    novelty_score=novelty,
                    loss_ratio=loss_ratio,
                    error_type=None if is_success else "stage1_error",
                    stage_at_death=None if is_success else "stage1",
                )
                pending_updates.append((base_ts + step, result_id))

            self.nb.complete_experiment(
                exp_id,
                {
                    "total": 10,
                    "stage0_passed": 10,
                    "stage05_passed": 10,
                    "stage1_passed": 5,
                    "best_loss_ratio": 0.7,
                    "best_novelty_score": 0.8,
                },
            )
            self.nb.conn.execute(
                "UPDATE experiments SET duration_seconds = ? WHERE experiment_id = ?",
                (75.0, exp_id),
            )

        for i in range(3):
            _record_experiment_with_trajectory(
                f"traj_up_{i}", improving=True, offset=i * 20
            )
            _record_experiment_with_trajectory(
                f"traj_down_{i}", improving=False, offset=200 + (i * 20)
            )

        self.nb.flush_writes()
        for ts, rid in pending_updates:
            self.nb.conn.execute(
                "UPDATE program_results SET timestamp = ? WHERE result_id = ?",
                (ts, rid),
            )
        self.nb.conn.commit()

        clusters = ExperimentAnalytics(self.nb).experiment_clusters(n_clusters=4)
        self.assertIsNotNone(clusters)
        self.assertEqual(clusters["n_experiments"], 6)

        self.assertIn("stage1_momentum", clusters["feature_keys"])
        self.assertIn("novelty_momentum", clusters["feature_keys"])
        self.assertIn("loss_improvement_momentum", clusters["feature_keys"])
        self.assertIn("outcome_volatility", clusters["feature_keys"])
        self.assertIn("outcome_peak_timing", clusters["feature_keys"])
        self.assertIn("recovery_lag", clusters["feature_keys"])
        self.assertIn("stage1_transition_timing", clusters["feature_keys"])
        self.assertIn("primary_change_point_timing", clusters["feature_keys"])
        self.assertIn("stage1_transition_density", clusters["feature_keys"])
        self.assertIn("change_point_confidence", clusters["feature_keys"])
        self.assertIn("windowed_change_dispersion", clusters["feature_keys"])
        self.assertIn("window_change_localization", clusters["feature_keys"])
        self.assertIn("transition_gap_entropy", clusters["feature_keys"])

        self.assertIn("model_selection", clusters)
        self.assertIn(clusters["model_selection"]["selected_k"], (2, 3))

        stage1_momentum = sorted(c["avg_stage1_momentum"] for c in clusters["clusters"])
        novelty_momentum = sorted(
            c["avg_novelty_momentum"] for c in clusters["clusters"]
        )
        loss_momentum = sorted(
            c["avg_loss_improvement_momentum"] for c in clusters["clusters"]
        )
        peak_timing = sorted(c["avg_outcome_peak_timing"] for c in clusters["clusters"])
        recovery_lag = sorted(c["avg_recovery_lag"] for c in clusters["clusters"])
        transition_timing = [
            c["avg_stage1_transition_timing"] for c in clusters["clusters"]
        ]
        change_point_timing = [
            c["avg_primary_change_point_timing"] for c in clusters["clusters"]
        ]
        transition_density = [
            c["avg_stage1_transition_density"] for c in clusters["clusters"]
        ]
        change_point_conf = [
            c["avg_change_point_confidence"] for c in clusters["clusters"]
        ]
        change_dispersion = [
            c["avg_windowed_change_dispersion"] for c in clusters["clusters"]
        ]
        change_localization = [
            c["avg_window_change_localization"] for c in clusters["clusters"]
        ]
        transition_gap_entropy = [
            c["avg_transition_gap_entropy"] for c in clusters["clusters"]
        ]

        self.assertLess(stage1_momentum[0], -0.6)
        self.assertGreater(stage1_momentum[-1], 0.6)
        self.assertLess(novelty_momentum[0], -0.3)
        self.assertGreater(novelty_momentum[-1], 0.3)
        self.assertLess(loss_momentum[0], -0.3)
        self.assertGreater(loss_momentum[-1], 0.3)
        self.assertLess(peak_timing[0], 0.35)
        self.assertGreater(peak_timing[-1], 0.65)
        self.assertLess(recovery_lag[0], 0.35)
        self.assertGreater(recovery_lag[-1], 0.8)
        self.assertTrue(all(0.0 <= t <= 1.0 for t in transition_timing))
        self.assertTrue(all(0.0 <= t <= 1.0 for t in change_point_timing))
        self.assertTrue(all(0.0 <= t <= 1.0 for t in transition_density))
        self.assertTrue(all(0.0 <= t <= 1.0 for t in change_point_conf))
        self.assertTrue(all(d >= 0.0 for d in change_dispersion))
        self.assertTrue(all(0.0 <= t <= 1.0 for t in change_localization))
        self.assertTrue(all(0.0 <= t <= 1.0 for t in transition_gap_entropy))


class TestStaleExperimentCleanup(unittest.TestCase):
    """Test cleanup_stale_experiments marks zombies as failed."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self.tmpdir, "test_cleanup.db")
        from research.scientist.notebook.__init__ import LabNotebook

        self.nb = LabNotebook(db_path)

    def tearDown(self):
        self.nb.close()

    def test_cleanup_marks_stale_as_failed(self):
        """Experiments running longer than timeout should be marked failed."""
        import time as _time

        exp_id = self.nb.start_experiment("synthesis", {}, "stale test")
        # Backdate started_at to 2 hours ago
        two_hours_ago = _time.time() - 7200
        self.nb.conn.execute(
            "UPDATE experiments SET started_at = ? WHERE experiment_id = ?",
            (two_hours_ago, exp_id),
        )
        self.nb.conn.commit()

        # Verify it's still running
        row = self.nb.conn.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        self.assertEqual(row["status"], "running")

        # Cleanup with 60-minute timeout
        cleaned = self.nb.cleanup_stale_experiments(timeout_minutes=60)
        self.assertEqual(cleaned, 1)

        # Verify it's now failed
        row = self.nb.conn.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        self.assertEqual(row["status"], "failed")
        records = [
            record.event
            for record in get_runtime_event_services(self.nb.db_path).spool.replay()
            if record.event.run_id == exp_id
            and record.event.event_type == "experiment_failed"
        ]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].payload["reason"], "stale_recovery_cleanup")

    def test_cleanup_ignores_recent_running(self):
        """Recently started experiments should not be cleaned up."""
        exp_id = self.nb.start_experiment("synthesis", {}, "recent test")
        # Don't backdate — it just started

        cleaned = self.nb.cleanup_stale_experiments(timeout_minutes=60)
        self.assertEqual(cleaned, 0)

        row = self.nb.conn.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        self.assertEqual(row["status"], "running")

    def test_cleanup_ignores_completed(self):
        """Completed experiments should not be affected by cleanup."""
        exp_id = self.nb.start_experiment("synthesis", {}, "done test")
        self.nb.complete_experiment(
            exp_id,
            {
                "total": 1,
                "stage0_passed": 1,
                "stage1_passed": 0,
            },
        )

        # Backdate to look stale
        import time as _time

        self.nb.conn.execute(
            "UPDATE experiments SET started_at = ? WHERE experiment_id = ?",
            (_time.time() - 7200, exp_id),
        )
        self.nb.conn.commit()

        cleaned = self.nb.cleanup_stale_experiments(timeout_minutes=60)
        self.assertEqual(cleaned, 0)

    def test_cleanup_marks_startup_failed_without_progress(self):
        """Running experiments with no progress should be cleaned by startup-failure threshold."""
        import time as _time

        exp_id = self.nb.start_experiment("validation", {}, "startup fail test")
        twenty_minutes_ago = _time.time() - (20 * 60)
        self.nb.conn.execute(
            "UPDATE experiments SET started_at = ? WHERE experiment_id = ?",
            (twenty_minutes_ago, exp_id),
        )
        self.nb.conn.commit()

        cleaned = self.nb.cleanup_stale_experiments(
            timeout_minutes=60,
            startup_failure_minutes=15,
        )
        self.assertEqual(cleaned, 1)

        row = self.nb.conn.execute(
            "SELECT status, results_json FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        self.assertEqual(row["status"], "failed")
        results = json.loads(row["results_json"] or "{}")
        self.assertEqual(
            results.get("failure_reason"),
            "Startup failed before any progress was recorded",
        )

    def test_cleanup_keeps_recent_progress_even_if_old(self):
        """Experiments with progress signals should not be marked startup-failed."""
        import time as _time

        exp_id = self.nb.start_experiment("validation", {}, "progress test")
        twenty_minutes_ago = _time.time() - (20 * 60)
        self.nb.conn.execute(
            "UPDATE experiments SET started_at = ? WHERE experiment_id = ?",
            (twenty_minutes_ago, exp_id),
        )
        self.nb.conn.commit()

        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="progress_fp",
            graph_json='{"nodes": {}}',
            stage0_passed=True,
            stage05_passed=False,
            stage1_passed=False,
            loss_ratio=1.5,
        )
        self.nb.flush_writes()

        cleaned = self.nb.cleanup_stale_experiments(
            timeout_minutes=60,
            startup_failure_minutes=15,
        )
        self.assertEqual(cleaned, 0)

        row = self.nb.conn.execute(
            "SELECT status FROM experiments WHERE experiment_id = ?",
            (exp_id,),
        ).fetchone()
        self.assertEqual(row["status"], "running")

    def test_cleanup_returns_zero_when_nothing_stale(self):
        """No stale experiments means cleanup returns 0."""
        cleaned = self.nb.cleanup_stale_experiments(timeout_minutes=60)
        self.assertEqual(cleaned, 0)


class TestNotebookThreadAffinity(unittest.TestCase):
    """Notebook connections should be safe to use from runner-owned threads."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self.tmpdir, "test_thread_affinity.db")
        from research.scientist.notebook.__init__ import LabNotebook

        self.nb = LabNotebook(db_path)

    def tearDown(self):
        self.nb.close()

    def test_default_connection_allows_cross_thread_reads(self):
        import threading

        errors = []

        def worker():
            try:
                row = self.nb.conn.execute("SELECT 1 AS value").fetchone()
                self.assertEqual(row["value"], 1)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])


class TestLeaderboardDedup(unittest.TestCase):
    """Test leaderboard fingerprint deduplication."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self.tmpdir, "test_dedup.db")
        from research.scientist.notebook.__init__ import LabNotebook

        self.nb = LabNotebook(db_path)

    def tearDown(self):
        self.nb.close()

    def test_leaderboard_dedup_by_fingerprint(self):
        """Same fingerprint should only appear once, keeping best score."""
        exp_id = self.nb.start_experiment("synthesis", {}, "test")

        # Create two results with same fingerprint
        r1 = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="same_fp",
            graph_json='{"nodes": {}}',
            stage1_passed=True,
            loss_ratio=0.5,
            novelty_score=0.6,
        )
        r2 = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="same_fp",
            graph_json='{"nodes": {}}',
            stage1_passed=True,
            loss_ratio=0.3,
            novelty_score=0.9,
        )
        # Third with different fingerprint
        r3 = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="different_fp",
            graph_json='{"nodes": {}}',
            stage1_passed=True,
            loss_ratio=0.4,
            novelty_score=0.7,
        )

        self.nb.flush_writes()

        self.nb.upsert_leaderboard(
            result_id=r1,
            model_source="test",
            screening_loss_ratio=0.5,
            screening_novelty=0.6,
            screening_passed=True,
            tier="screening",
        )
        self.nb.upsert_leaderboard(
            result_id=r2,
            model_source="test",
            screening_loss_ratio=0.3,
            screening_novelty=0.9,
            screening_passed=True,
            tier="screening",
        )
        self.nb.upsert_leaderboard(
            result_id=r3,
            model_source="test",
            screening_loss_ratio=0.4,
            screening_novelty=0.7,
            screening_passed=True,
            tier="screening",
        )

        entries = self.nb.get_leaderboard()
        # Should be 2 (deduped same_fp), not 3
        self.assertEqual(len(entries), 2)
        # The entry for same_fp should be the one with higher composite_score
        # (r2 has better loss_ratio=0.3 and novelty=0.9)

    def test_leaderboard_dedup_keeps_distinct_fingerprints(self):
        """Different fingerprints should all appear."""
        exp_id = self.nb.start_experiment("synthesis", {}, "test")

        for i, fp in enumerate(["fp_a", "fp_b", "fp_c"]):
            rid = self.nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=fp,
                graph_json="{}",
                stage1_passed=True,
                loss_ratio=0.4 + i * 0.1,
            )
            self.nb.flush_writes()
            self.nb.upsert_leaderboard(
                result_id=rid,
                model_source="test",
                screening_loss_ratio=0.4 + i * 0.1,
                screening_passed=True,
                tier="screening",
            )

        entries = self.nb.get_leaderboard()
        self.assertEqual(len(entries), 3)

    def test_get_investigated_fingerprints(self):
        """Fingerprints at investigation+ tier should be returned."""
        exp_id = self.nb.start_experiment("synthesis", {}, "test")

        # Create screening entry (should NOT appear)
        rid1 = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_screening",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.4,
        )
        self.nb.upsert_leaderboard(
            result_id=rid1,
            model_source="test",
            screening_loss_ratio=0.4,
            screening_passed=True,
            tier="screening",
        )

        # Create investigation entry (SHOULD appear)
        rid2 = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_investigated",
            graph_json="{}",
            model_source="graph_synthesis",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.3,
            wikitext_perplexity=7.1,
            data_mode="corpus",
            corpus_path="/tmp/investigated_fp.txt",
            corpus_format="txt",
            corpus_text_key="text",
            corpus_train_fraction=0.9,
            corpus_val_fraction=0.1,
            corpus_max_chars=12000,
            tokenizer_mode="tiktoken",
            tiktoken_encoding="cl100k_base",
            screening_wikitext_metric_version="screening_wikitext_v1",
        )
        self.nb.upsert_leaderboard(
            result_id=rid2,
            model_source="test",
            screening_loss_ratio=0.3,
            screening_passed=True,
            tier="investigation",
        )

        # Create validation entry (SHOULD appear)
        rid3 = self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_validated",
            graph_json="{}",
            model_source="graph_synthesis",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=True,
            loss_ratio=0.2,
            wikitext_perplexity=6.9,
            data_mode="corpus",
            corpus_path="/tmp/validated_fp.txt",
            corpus_format="txt",
            corpus_text_key="text",
            corpus_train_fraction=0.9,
            corpus_val_fraction=0.1,
            corpus_max_chars=12000,
            tokenizer_mode="tiktoken",
            tiktoken_encoding="cl100k_base",
            screening_wikitext_metric_version="screening_wikitext_v1",
        )
        self.nb.upsert_leaderboard(
            result_id=rid3,
            model_source="test",
            screening_loss_ratio=0.2,
            screening_passed=True,
            tier="validation",
        )

        self.nb.flush_writes()
        fps = self.nb.get_investigated_fingerprints()
        self.assertNotIn("fp_screening", fps)
        self.assertIn("fp_investigated", fps)
        self.assertIn("fp_validated", fps)
        self.assertEqual(len(fps), 2)

    def test_leaderboard_consistency_report_distinguishes_descendants(self):
        exp_screen = self.nb.start_experiment("synthesis", {}, "screen")
        exp_inv = self.nb.start_experiment("investigation", {}, "inv")

        screening_rid = self.nb.record_program_result(
            experiment_id=exp_screen,
            graph_fingerprint="fp_shared",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.2,
        )
        descendant_rid = self.nb.record_program_result(
            experiment_id=exp_inv,
            graph_fingerprint="fp_shared",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.1,
        )
        missing_screening_rid = self.nb.record_program_result(
            experiment_id=exp_screen,
            graph_fingerprint="fp_uncovered",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.3,
        )
        self.nb.flush_writes()
        self.nb.upsert_leaderboard(
            result_id=screening_rid,
            model_source="test",
            screening_loss_ratio=0.2,
            screening_passed=True,
            tier="screening",
        )

        report = self.nb.get_leaderboard_consistency_report()
        self.assertEqual(report["stage1_program_rows"], 3)
        self.assertEqual(report["direct_stage1_leaderboard_rows"], 1)
        self.assertEqual(report["descendant_stage1_rows_without_direct_entry"], 1)
        self.assertEqual(report["missing_screening_leaderboard_rows"], 1)
        self.assertIn(
            missing_screening_rid, report["samples"]["missing_screening_result_ids"]
        )
        self.assertIn(descendant_rid, report["samples"]["descendant_result_ids"])

    def test_backfill_missing_screening_leaderboard_entries_only_fills_uncovered_screening(
        self,
    ):
        exp_screen = self.nb.start_experiment("synthesis", {}, "screen")
        exp_inv = self.nb.start_experiment("investigation", {}, "inv")

        covered_rid = self.nb.record_program_result(
            experiment_id=exp_screen,
            graph_fingerprint="fp_covered",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.2,
            novelty_score=0.6,
        )
        descendant_rid = self.nb.record_program_result(
            experiment_id=exp_inv,
            graph_fingerprint="fp_covered",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.15,
            novelty_score=0.7,
        )
        missing_rid = self.nb.record_program_result(
            experiment_id=exp_screen,
            graph_fingerprint="fp_missing",
            graph_json="{}",
            stage1_passed=True,
            loss_ratio=0.25,
            novelty_score=0.8,
        )
        self.nb.flush_writes()
        self.nb.upsert_leaderboard(
            result_id=covered_rid,
            model_source="test",
            screening_loss_ratio=0.2,
            screening_novelty=0.6,
            screening_passed=True,
            tier="screening",
        )

        result = self.nb.backfill_missing_screening_leaderboard_entries()
        self.assertEqual(result["created_entries"], 1)
        self.assertEqual(result["result_ids"], [missing_rid])
        self.assertIsNotNone(self.nb.get_leaderboard_entry(missing_rid))
        self.assertIsNone(self.nb.get_leaderboard_entry(descendant_rid))


if __name__ == "__main__":
    unittest.main()
