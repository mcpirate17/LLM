"""
Integration Tests for the AI Scientist Research Pipeline

Tests the full stack: notebook schema, leaderboard lifecycle,
auto-escalation pipeline, API endpoints, mode selection, and
novelty scoring fixes.

Run: cd /path/to/LLM && python -m unittest research.tests.test_integration -v
"""

import pytest
import importlib
import json
import os
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

        exp = self.nb.get_experiment(exp_id)
        self.assertEqual(exp["status"], "completed")

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
        self.assertGreaterEqual(len(weak["diagnosis"]), 1)
        self.assertGreaterEqual(len(weak["actions"]), 1)
        self.assertEqual(weak["screening_metric_coverage"]["induction"], 1)
        self.assertEqual(weak["screening_metric_coverage"]["hellaswag"], 2)
        self.assertAlmostEqual(strong["avg_induction_auc"], 0.08)
        self.assertAlmostEqual(strong["avg_binding_auc"], 0.09)
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
            stage1_passed=True,
            loss_ratio=0.5,
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
            stage1_passed=True,
            loss_ratio=0.3,
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
        valid_graph = json.dumps(
            {
                "nodes": {
                    "a": {"op_name": "relu"},
                    "b": {"op_name": "gelu"},
                    "c": {"op_name": "input"},
                }
            }
        )
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_combo_1",
            graph_json=valid_graph,
            stage1_passed=True,
            novelty_score=0.7,
        )
        self.nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint="fp_combo_2",
            graph_json=valid_graph,
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
            stage1_passed=True,
            loss_ratio=0.3,
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
            stage1_passed=True,
            loss_ratio=0.2,
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
