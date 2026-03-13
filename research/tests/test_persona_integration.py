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
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit

# Detect available dependencies
try:
    import torch
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
    from research.scientist.persona import Aria
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


# TestAriaModeSelecion lives in test_runner_mode_selection.py


# ── Test 5: Context Builders ──


@unittest.skipUnless(HAS_CONTEXT, "requires context module")
class TestContextBuilders(unittest.TestCase):
    """Test LLM context building functions."""

    def test_mode_selection_context(self):
        """Mode selection context includes key information."""
        from research.scientist.llm.context_experiment import build_mode_selection_context

        ctx = build_mode_selection_context(
            recent_experiments=[
                {"n_stage1_passed": 2, "n_programs_generated": 50,
                 "best_novelty_score": 0.7, "experiment_type": "synthesis"},
            ],
            leaderboard=[
                {"tier": "screening", "screening_loss_ratio": 0.5,
                 "composite_score": 0.6, "result_id": "r1"},
            ],
            current_mode="synthesis",
            n_experiments_in_session=3,
        )

        self.assertIn("synthesis", ctx)
        self.assertIn("3", ctx)  # n_experiments_in_session

    def test_investigation_context(self):
        """Investigation context includes candidate data."""
        from research.scientist.llm.context_experiment import build_investigation_context

        ctx = build_investigation_context(
            candidates=[{"result_id": "r1", "loss_ratio": 0.4}],
            leaderboard=[{"tier": "screening", "composite_score": 0.5}],
        )
        self.assertIn("Investigation Phase", ctx)
        self.assertIn("r1", ctx)

    def test_validation_context(self):
        """Validation context includes investigation results."""
        from research.scientist.llm.context_experiment import build_validation_context

        ctx = build_validation_context(
            candidates=[{"result_id": "r1", "investigation_loss_ratio": 0.3}],
            investigation_results=[{"result_id": "r1", "robustness": 0.7}],
        )
        self.assertIn("Validation Phase", ctx)



# ── Test 8: Prompt Templates ──


@unittest.skipUnless(HAS_PROMPTS, "requires prompts module")
class TestPrompts(unittest.TestCase):
    """Verify all prompt templates exist and have correct placeholders."""

    def test_all_prompts_importable(self):
        from research.scientist.llm.prompts import (
            ANALYSIS_PROMPT,
            HYPOTHESIS_PROMPT,
            SUMMARY_PROMPT,
            FINGERPRINT_EXPLANATION_PROMPT,
            STRATEGY_PROMPT,
            SUGGESTION_PROMPT,
            REPORT_PROMPT,
            INVESTIGATION_HYPOTHESIS_PROMPT,
            VALIDATION_ANALYSIS_PROMPT,
            BREAKTHROUGH_ANNOUNCEMENT_PROMPT,
            MODE_SELECTION_PROMPT,
        )
        # All should have {context} placeholder
        for name, prompt in [
            ("ANALYSIS", ANALYSIS_PROMPT),
            ("HYPOTHESIS", HYPOTHESIS_PROMPT),
            ("SUMMARY", SUMMARY_PROMPT),
            ("FINGERPRINT", FINGERPRINT_EXPLANATION_PROMPT),
            ("STRATEGY", STRATEGY_PROMPT),
            ("SUGGESTION", SUGGESTION_PROMPT),
            ("REPORT", REPORT_PROMPT),
            ("INVESTIGATION", INVESTIGATION_HYPOTHESIS_PROMPT),
            ("VALIDATION_ANALYSIS", VALIDATION_ANALYSIS_PROMPT),
            ("BREAKTHROUGH", BREAKTHROUGH_ANNOUNCEMENT_PROMPT),
            ("MODE_SELECTION", MODE_SELECTION_PROMPT),
        ]:
            self.assertIn("{context}", prompt,
                          f"{name}_PROMPT missing {{context}} placeholder")

    def test_validation_prompt_has_hypothesis_placeholder(self):
        from research.scientist.llm.prompts import VALIDATION_PROMPT
        self.assertIn("{hypothesis}", VALIDATION_PROMPT)



# ── Test 9: Persona Methods ──


@unittest.skipUnless(HAS_PERSONA, "requires persona module")
class TestPersona(unittest.TestCase):
    """Verify Aria persona has all required methods."""

    def setUp(self):
        self.aria = Aria()

    def test_all_methods_exist(self):
        """Aria should have all expected public methods."""
        methods = [
            "greet", "react_to_discovery", "react_to_failure",
            "begin_analysis", "formulate_hypothesis",
            "experiment_summary", "analyze_results",
            "explain_fingerprint", "plan_strategy",
            "suggest_experiment", "validate_hypothesis",
            "explain_learning", "generate_report_narrative",
            "get_status", "add_insight",
            # Phase methods
            "formulate_investigation_hypothesis",
            "formulate_validation_hypothesis",
            "announce_breakthrough",
            # Mode selection
            "recommend_next_mode",
        ]
        for m in methods:
            self.assertTrue(hasattr(self.aria, m),
                            f"Aria missing method: {m}")
            self.assertTrue(callable(getattr(self.aria, m)),
                            f"Aria.{m} is not callable")

    def test_greet_returns_string(self):
        msg = self.aria.greet()
        self.assertIsInstance(msg, str)
        self.assertGreater(len(msg), 0)

    def test_get_status_returns_dict(self):
        status = self.aria.get_status()
        self.assertIn("name", status)
        self.assertIn("mood", status)
        self.assertIn("llm_enabled", status)

    def test_rule_based_hypothesis(self):
        hyp = self.aria.formulate_hypothesis()
        self.assertIsInstance(hyp, str)
        self.assertGreater(len(hyp), 0)

    def test_rule_based_summary(self):
        results = {"total": 50, "stage0_passed": 30,
                   "stage05_passed": 20, "stage1_passed": 2,
                   "novel_count": 1}
        summary = self.aria.experiment_summary(results)
        self.assertIsInstance(summary, str)
        self.assertIn("50", summary)

    def test_rule_based_investigation_hypothesis(self):
        hyp = self.aria.formulate_investigation_hypothesis()
        self.assertIsInstance(hyp, str)
        self.assertIn("training", hyp.lower())

    def test_rule_based_validation_hypothesis(self):
        hyp = self.aria.formulate_validation_hypothesis()
        self.assertIsInstance(hyp, str)

    def test_hypothesis_critique_returns_gate_and_checks(self):
        self.aria._get_llm = MagicMock(return_value=None)
        critique = self.aria.critique_hypothesis("Try something new")
        self.assertIn("verdict", critique)
        self.assertIn("gate", critique)
        self.assertIn(critique["gate"], {"pass", "warn", "fail"})
        self.assertIn("checks", critique)
        self.assertIn("missing_fields", critique)
        self.assertIsInstance(critique["checks"], list)
        self.assertIsInstance(critique["missing_fields"], list)
        check_keys = {c.get("key") for c in critique["checks"] if isinstance(c, dict)}
        self.assertTrue({"testability", "measurable_metric", "confound_risk", "fallback_plan"}.issubset(check_keys))

    def test_hypothesis_critique_flags_underspecified_refinement(self):
        self.aria._get_llm = MagicMock(return_value=None)
        critique = self.aria.critique_hypothesis(
            "Fingerprint refinement: locally mutate selected architecture with intent=balanced."
        )
        concerns = " ".join(critique.get("concerns") or []).lower()
        self.assertIn("source-selection rule", concerns)
        self.assertIn("mutation operators", concerns)
        self.assertIn("intent", concerns)
        self.assertIn("success criteria", concerns)

    def test_announce_breakthrough(self):
        msg = self.aria.announce_breakthrough()
        self.assertIsInstance(msg, str)
        self.assertIn("BREAKTHROUGH", msg)

    def test_assess_breakthrough_evidence_publication_grade(self):
        evidence = self.aria.assess_breakthrough_evidence(metrics={
            "seeds_passed": 6,
            "total_seeds": 6,
            "val_baseline_ratio": 0.82,
            "multi_seed_std": 0.018,
            "ood_robustness": 0.8,
            "hp_robustness": 0.85,
        })
        self.assertEqual(evidence["label"], "publication_grade")
        self.assertIn(evidence["confidence_band"], {"high", "medium", "low"})

    def test_assess_breakthrough_evidence_provisional_for_low_seed_count(self):
        evidence = self.aria.assess_breakthrough_evidence(metrics={
            "seeds_passed": 3,
            "total_seeds": 3,
            "val_baseline_ratio": 0.82,
            "multi_seed_std": 0.018,
        })
        self.assertEqual(evidence["label"], "provisional")
        self.assertIn("seed_count_below_publication_threshold", evidence["reasons"])

    def test_announce_breakthrough_provisional_language(self):
        msg = self.aria.announce_breakthrough(metrics={
            "seeds_passed": 3,
            "total_seeds": 3,
            "val_baseline_ratio": 0.93,
            "multi_seed_std": 0.05,
        })
        self.assertIn("BREAKTHROUGH SIGNAL DETECTED", msg)
        self.assertIn("PROVISIONAL", msg)

    def test_cost_tracking(self):
        self.aria.reset_cost_tracking()
        self.assertEqual(self.aria.total_tokens, 0)
        self.assertEqual(self.aria.total_cost, 0.0)

    def test_unknown_backend_cost_logs_warning_once(self):
        class _Resp:
            tokens_used = 100

        class _Backend:
            name = "mystery-backend"

        self.aria._llm = _Backend()
        with patch("research.scientist.persona_llm.logger.warning") as warn:
            self.aria._track_cost(_Resp())
            self.aria._track_cost(_Resp())
            self.assertEqual(warn.call_count, 1)
        self.assertGreater(self.aria.total_cost, 0.0)


class TestAnthropicBackendConfig(unittest.TestCase):
    """Backend config defaults should be resilient to model deprecations."""

    def test_default_model_uses_alias(self):
        with patch.dict(os.environ, {}, clear=True):
            from research.scientist.llm.anthropic import AnthropicBackend, DEFAULT_ANTHROPIC_MODEL
            backend = AnthropicBackend()
            self.assertEqual(backend.model, DEFAULT_ANTHROPIC_MODEL)

    def test_env_model_override_wins(self):
        with patch.dict(os.environ, {"ANTHROPIC_MODEL": "custom-model"}, clear=True):
            from research.scientist.llm.anthropic import AnthropicBackend
            backend = AnthropicBackend()
            self.assertEqual(backend.model, "custom-model")



class TestPersonaOptimizerAwareness(unittest.TestCase):
    """Tests for optimizer diversity awareness in persona."""

    def test_strategy_index_8_produces_valid_recommendation(self):
        """Strategy index 8 with low novelty returns diversification rec."""
        from research.scientist.persona import Aria
        aria = Aria()
        # With avg_novelty < 0.4 in analytics_data, the standard exploration
        # strategy triggers the low-novelty diversification path.
        data = {
            "total_s1_survivors": 5,
            "n_experiments_in_session": 8,
            "investigation_ready": 0,
            "validation_ready": 0,
            "analytics_data": {"avg_novelty": 0.0},
            "recent_exploration_modes": ["evolution"] * 5,
            "recent_failure_count": 1,
            "leaderboard_diversity": 3,
            "leaderboard_size": 10,
            "optimizer_counts": {"AdamW": 50},
            "optimizer_diversity": 1,
        }
        rec = aria._rule_based_mode_recommendation(data)
        self.assertEqual(rec["mode"], "synthesis")
        self.assertIn("diversify", rec["reasoning"].lower())

    def test_suggestion_template_includes_alternative_rules(self):
        """At least one suggestion config mentions alternative learning rules."""
        from research.scientist.persona import Aria
        aria = Aria()
        found = False
        # Rotate through all suggestion templates
        for i in range(20):
            aria.state.experiments_today = i
            suggestion = aria._rule_based_suggestion()
            if "optimizer_preference" in suggestion.get("config", {}):
                found = True
                self.assertIn("alternative", suggestion["reasoning"].lower())
                break
        self.assertTrue(found, "No suggestion template has optimizer_preference")



class TestContextBuilderExpanded(unittest.TestCase):
    """Tests for expanded context builder sections."""

    def test_op_registry_section_populated(self):
        """Op registry section should list all primitives by category."""
        from research.scientist.llm.context_experiment import _build_op_registry_section
        import research.scientist.llm.context_experiment as ctx_mod
        ctx_mod._OP_REGISTRY_CACHE = None  # Force rebuild
        section = _build_op_registry_section()
        self.assertIn("Available Ops", section)
        self.assertIn("excluded_ops", section)
        self.assertIn("elementwise_unary", section)
        self.assertIn("relu", section)
        self.assertIn("matmul", section)

    def test_category_weight_hint_in_context(self):
        """Grammar weights section should include category_weights hint."""
        from research.scientist.llm.context_experiment import build_rich_context
        ctx = build_rich_context(
            results={"total": 10, "stage0_passed": 5, "stage1_passed": 1},
            analytics_data={
                "grammar_weights": {"parameterized": 2.0},
                "default_weights": {"parameterized": 1.0},
            },
        )
        self.assertIn("Set category_weights in CONFIG", ctx)

    def test_excluded_ops_hint_in_negative_results(self):
        """Negative results section should suggest using excluded_ops."""
        from research.scientist.llm.context_experiment import build_rich_context
        ctx = build_rich_context(
            results={"total": 10, "stage0_passed": 5, "stage1_passed": 1},
            analytics_data={
                "negative_results": {
                    "failed_ops": [
                        {"op_name": "bad_op", "n_used": 10, "failure_stage": "stage0", "confidence": 0.9},
                    ],
                },
            },
        )
        self.assertIn("Use excluded_ops in CONFIG to ban these", ctx)

    def test_designer_telemetry_section(self):
        """Designer telemetry should render in context when present."""
        from research.scientist.llm.context_experiment import build_rich_context
        ctx = build_rich_context(
            results={"total": 10, "stage0_passed": 5, "stage1_passed": 1},
            analytics_data={
                "designer_telemetry": {
                    "bridge_gap_report": {
                        "unsupported_components": 3,
                        "total_components": 50,
                        "gaps": [{"component_id": "comp_a"}, {"component_id": "comp_b"}],
                    },
                    "builtin_blocks": ["MLP", "Attention", "FFN"],
                },
            },
        )
        self.assertIn("Designer Integration:", ctx)
        self.assertIn("Bridge gap: 3 of 50", ctx)
        self.assertIn("comp_a", ctx)
        self.assertIn("MLP", ctx)

    def test_designer_telemetry_absent_gracefully(self):
        """Missing designer telemetry should not break context building."""
        from research.scientist.llm.context_experiment import build_rich_context
        ctx = build_rich_context(
            results={"total": 10, "stage0_passed": 5, "stage1_passed": 1},
            analytics_data={},
        )
        self.assertNotIn("Designer Integration:", ctx)


class TestRuleBasedStrategies(unittest.TestCase):
    """Tests for expanded rule-based strategy configs in persona."""

    def test_strategy_keys_match_runconfig(self):
        """All strategy config keys should be valid RunConfig or grammar override keys."""
        from research.scientist.persona import Aria
        from research.scientist.runner import RunConfig
        aria = Aria()
        suggestion = aria._rule_based_suggestion()
        config = suggestion.get("config", {})

        valid_runconfig_keys = set(RunConfig.__dataclass_fields__.keys())
        # Keys handled by _apply_recommendation
        valid_override_keys = {
            "math_space_weight", "category_weights", "excluded_ops", "op_weights",
            "grammar_split_prob", "grammar_merge_prob", "grammar_risky_op_prob",
            "grammar_freq_domain_prob", "structured_sparsity_bias", "optimizer_preference",
        }
        valid_keys = valid_runconfig_keys | valid_override_keys

        for key in config:
            self.assertIn(key, valid_keys,
                          f"Strategy key '{key}' not in RunConfig or override keys")

    def test_all_strategies_have_valid_keys(self):
        """Cycle through all strategies and verify keys are valid."""
        from research.scientist.persona import Aria
        from research.scientist.runner import RunConfig
        aria = Aria()

        valid_runconfig_keys = set(RunConfig.__dataclass_fields__.keys())
        valid_override_keys = {
            "math_space_weight", "category_weights", "excluded_ops", "op_weights",
            "grammar_split_prob", "grammar_merge_prob", "grammar_risky_op_prob",
            "grammar_freq_domain_prob", "structured_sparsity_bias", "optimizer_preference",
        }
        valid_keys = valid_runconfig_keys | valid_override_keys

        for i in range(9):  # 9 strategies
            aria.state.experiments_today = i
            suggestion = aria._rule_based_suggestion()
            config = suggestion.get("config", {})
            for key in config:
                if isinstance(config[key], dict):
                    continue  # category_weights is a nested dict, not a RunConfig key
                self.assertIn(key, valid_keys,
                              f"Strategy {i} key '{key}' not valid")

    def test_functional_heavy_strategy_exists(self):
        """Strategy 9 (index 8) should be the functional-heavy config."""
        from research.scientist.persona import Aria
        aria = Aria()
        aria.state.experiments_today = 8
        suggestion = aria._rule_based_suggestion()
        config = suggestion.get("config", {})
        self.assertIn("category_weights", config)
        self.assertAlmostEqual(config["category_weights"]["functional"], 3.0)
        self.assertAlmostEqual(config["category_weights"]["elementwise_unary"], 2.5)

    def test_split_merge_uses_grammar_prefix(self):
        """Strategy 5 should use grammar_split_prob, not split_prob."""
        from research.scientist.persona import Aria
        aria = Aria()
        aria.state.experiments_today = 4  # index 4 = strategy 5
        suggestion = aria._rule_based_suggestion()
        config = suggestion.get("config", {})
        self.assertIn("grammar_split_prob", config)
        self.assertNotIn("split_prob", config)

    def test_high_risk_uses_grammar_prefix(self):
        """Strategy 6 should use grammar_risky_op_prob, not risky_op_prob."""
        from research.scientist.persona import Aria
        aria = Aria()
        aria.state.experiments_today = 5  # index 5 = strategy 6
        suggestion = aria._rule_based_suggestion()
        config = suggestion.get("config", {})
        self.assertIn("grammar_risky_op_prob", config)
        self.assertNotIn("risky_op_prob", config)



if __name__ == '__main__':
    unittest.main()
