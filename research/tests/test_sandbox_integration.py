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

pytestmark = pytest.mark.pipeline

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


class TestPackageWiring(unittest.TestCase):
    """Ensure explicitly connected package modules remain importable."""

    def test_mathspaces_exports_modules(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        init_path = os.path.join(repo_root, "mathspaces", "__init__.py")
        with open(init_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("from . import clifford, compression, hyperbolic, padic, spiking, tropical", content)
        self.assertIn("from .registry import register_all_mathspaces", content)
        self.assertIn('"hyperbolic"', content)
        self.assertIn('"tropical"', content)
        self.assertIn('"padic"', content)
        self.assertIn('"clifford"', content)
        self.assertIn('"spiking"', content)

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_mathspace_registry_includes_hyp_distance(self):
        from research.mathspaces.registry import register_all_mathspaces
        from research.synthesis.primitives import PRIMITIVE_REGISTRY

        register_all_mathspaces()
        self.assertIn("hyp_distance", PRIMITIVE_REGISTRY)
        op = PRIMITIVE_REGISTRY["hyp_distance"]
        self.assertEqual(op.category.value, "math_space")
        self.assertEqual(op.n_inputs, 2)
        self.assertTrue(hasattr(op, "execute_fn"))

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_external_op_nonfinite_sanitization_and_telemetry(self):
        from research.synthesis.compiler import _execute_op
        from research.synthesis.primitives import PrimitiveOp, OpCategory, PRIMITIVE_REGISTRY, register_external_primitive

        op_name = "test_nonfinite_mathspace_op"
        op = PrimitiveOp(
            name=op_name,
            category=OpCategory.MATH_SPACE,
            n_inputs=1,
            shape_rule="identity",
            description="test external op",
        )

        def _execute_fn(module, x):
            return x / 0.0

        object.__setattr__(op, "execute_fn", _execute_fn)
        register_external_primitive(op)
        module = torch.nn.Module()
        x = torch.ones(2, 3, 4)
        try:
            out = _execute_op(module, op_name, (x,), {})
            self.assertTrue(torch.isfinite(out).all())
            telemetry = getattr(module, "mathspace_telemetry", {})
            self.assertIn(op_name, telemetry)
            self.assertGreaterEqual(telemetry[op_name]["calls"], 1)
            self.assertGreater(telemetry[op_name]["nonfinite_elements"], 0)
            self.assertGreaterEqual(telemetry[op_name]["sanitized_calls"], 1)
        finally:
            PRIMITIVE_REGISTRY.pop(op_name, None)

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_mathspace_phase2_ops_registered(self):
        from research.mathspaces.registry import register_all_mathspaces
        from research.synthesis.primitives import PRIMITIVE_REGISTRY

        register_all_mathspaces()
        for op_name in ("hyp_tangent_nonlinear", "tropical_center", "padic_gate", "grade_mix"):
            self.assertIn(op_name, PRIMITIVE_REGISTRY)
            op = PRIMITIVE_REGISTRY[op_name]
            self.assertEqual(op.category.value, "math_space")
            self.assertEqual(op.n_inputs, 1)
            self.assertTrue(hasattr(op, "execute_fn"))

    @unittest.skipUnless(HAS_TORCH, "requires torch")
    def test_mathspace_phase2_ops_execute_shape_and_finite(self):
        from research.mathspaces.registry import register_all_mathspaces
        from research.synthesis.primitives import PRIMITIVE_REGISTRY

        register_all_mathspaces()
        x = torch.randn(2, 5, 16)
        module = torch.nn.Module()
        for op_name in ("hyp_tangent_nonlinear", "tropical_center", "padic_gate", "grade_mix"):
            op = PRIMITIVE_REGISTRY[op_name]
            out = op.execute_fn(module, x)
            self.assertEqual(tuple(out.shape), tuple(x.shape))
            self.assertTrue(torch.isfinite(out).all(), f"{op_name} produced non-finite values")

    def test_llm_package_exports_context_and_prompts(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        init_path = os.path.join(repo_root, "scientist", "llm", "__init__.py")
        with open(init_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("from . import context, prompts", content)
        self.assertIn("from .backend import", content)
        self.assertIn('"context"', content)
        self.assertIn('"prompts"', content)



# ── Test 10: Dashboard Component Consistency ──


class TestDashboardConsistency(unittest.TestCase):
    """Verify dashboard components and API endpoints are properly wired."""

    @classmethod
    def setUpClass(cls):
        import glob
        cls.repo_root = os.path.dirname(os.path.dirname(__file__))
        cls.component_dir = os.path.join(
            cls.repo_root, "dashboard", "src", "components")
        cls.component_files = glob.glob(
            os.path.join(cls.component_dir, "*.js"))
        cls.app_js = os.path.join(
            cls.repo_root, "dashboard", "src", "App.js")
        cls.api_py = os.path.join(cls.repo_root, "scientist", "api.py")

    def _read_file(self, path):
        with open(path, "r") as f:
            return f.read()

    def test_all_components_imported_in_app(self):
        """Every component should be imported in App.js."""
        app_content = self._read_file(self.app_js)

        # Components that are used inside other components, not App.js
        nested_only = {
            "GraphViewer", "FailureAnalysis", "AriaAvatar", "ReportGallery", "ReportDetail",
        }

        for filepath in self.component_files:
            basename = os.path.basename(filepath)
            # Skip test files, utility/preset files that aren't React components
            if basename.endswith(".test.js") or basename[0].islower():
                continue
            name = basename.replace(".js", "")
            if name in nested_only:
                continue
            self.assertIn(
                f"import {name}",
                app_content,
                f"Component {name} not imported in App.js",
            )

    def test_all_components_have_default_export(self):
        """Every component file should have a default export."""
        for filepath in self.component_files:
            basename = os.path.basename(filepath)
            # Skip test files and utility/preset files that aren't React components
            if basename.endswith(".test.js") or basename[0].islower():
                continue
            content = self._read_file(filepath)
            name = basename.replace(".js", "")
            has_named_default = f"export default {name}" in content
            has_default_function = f"export default function {name}" in content
            self.assertTrue(
                has_named_default or has_default_function,
                f"{name}.js missing default export for {name}",
            )

    def test_no_orphaned_api_fetch_urls(self):
        """All fetch URLs in components should match real API endpoints."""
        import re

        known_api_patterns = {
            "/api/dashboard", "/api/status", "/api/system/status", "/api/native-runner/capability",
            "/api/native-runner/canary/refresh",
            "/api/experiments", "/api/programs", "/api/trends",
            "/api/trends/context",
            "/api/insights", "/api/entries", "/api/live-feed", "/api/leaderboard",
            "/api/report", "/api/events", "/api/progress",
            "/api/config", "/api/validate",
            "/api/aria/recommendation", "/api/aria/strategy",
            "/api/strategy/briefing",
            "/api/llm/config",
            "/api/analytics/op-success", "/api/analytics/failure-patterns",
            "/api/analytics/grammar-weights", "/api/analytics/efficiency-frontier",
            "/api/analytics/learning-log", "/api/analytics/experiment-clusters",
            "/api/analytics/routing-health", "/api/analytics/math-family-coverage",
            "/api/analytics/mathspace-impact",
            "/api/analytics/routing-comparison",
            "/api/analytics/gating-diagnostics",
            "/api/analytics/gate-health",
            "/api/analytics/compression-coverage",
            "/api/analytics/learning-summary",
            "/api/analytics/insight-interactions",
            "/api/analytics/learning-trajectory",
            "/api/analytics/control-comparison",
            "/api/metrics/",
            "/api/experiments/start", "/api/experiments/stop",
            "/api/experiments/",
            "/api/campaigns", "/api/hypotheses",
            "/api/knowledge",
            "/api/decision-packet/",
            "/api/reproducibility-manifest/",
            "/api/analytics/negative-results",
            "/api/aria/chat",
            "/api/aria/chat/history",
            "/api/aria/chat/message",
            "/api/aria/chat/compact",
            "/api/aria/cycle-status",
            "/api/aria/cycle-control",
            "/api/aria/cycle-history",
            "/api/aria/agent/status/",
            "/api/aria/agent/spawn",
            "/api/aria/tools",
            "/api/aria/diagnose",
            "/api/designer/lineage",
            "/api/designer/ensure-running",
            "/api/designer/touch",
            "/api/actions",
            "/api/discoveries",
            "/api/aria/autonomy",
            "/api/aria/activity",
        }

        for filepath in self.component_files:
            content = self._read_file(filepath)
            # Find all fetch/API calls
            urls = re.findall(
                r'[`\'"](?:\$\{[^}]*\})?(/api/[a-z/_-]+)', content)
            for url in urls:
                # Normalize: remove dynamic segments
                base_url = re.sub(r'/\$\{[^}]*\}', '/', url)
                base_url = base_url.rstrip("/")

                matched = any(
                    base_url.startswith(pattern.rstrip("/"))
                    for pattern in known_api_patterns
                )
                self.assertTrue(
                    matched,
                    f"Orphaned API URL in {os.path.basename(filepath)}: {url}",
                )

    def test_aria_chat_panel_auto_analysis_uses_single_briefing_endpoint(self):
        chat_panel_path = os.path.join(self.component_dir, "AriaChatPanel.js")
        content = self._read_file(chat_panel_path)
        self.assertIn("/api/strategy/briefing", content)
        self.assertNotIn("/api/aria/strategy", content)
        self.assertNotIn("/api/aria/recommendation", content)
        self.assertIn("Auto: Off (Manual only)", content)
        self.assertNotIn("Auto: Run-only", content)
        self.assertNotIn("Auto: Always", content)
        self.assertIn("Ask for Action", content)
        self.assertIn("Self-fix: .py/.js", content)
        self.assertIn("details sent to local agent", content)
        self.assertIn("/api/aria/agent/status/${encodeURIComponent(taskId)}/summary", content)
        self.assertIn("Open full task details", content)
        self.assertIn("Guardrails (", content)

    def test_event_bus_hook_contract_exposes_subscribe_for_action_queue(self):
        hook_path = os.path.join(self.repo_root, "dashboard", "src", "hooks", "useEventBus.js")
        action_queue_path = os.path.join(self.component_dir, "ActionQueue.js")
        hook_content = self._read_file(hook_path)
        action_content = self._read_file(action_queue_path)
        self.assertIn("subscribe: ctx?.subscribe", hook_content)
        self.assertIn("const eventBus = useEventBus()", action_content)
        self.assertIn("if (typeof subscribe !== 'function') return undefined;", action_content)

    def test_dashboard_wires_auto_repair_started_event_to_chat(self):
        app_content = self._read_file(self.app_js)
        chat_panel_path = os.path.join(self.component_dir, "AriaChatPanel.js")
        chat_content = self._read_file(chat_panel_path)

        self.assertIn("aria-auto-repair-started", app_content)
        self.assertIn("emitAutoRepairStarted", app_content)
        # Auto-repair UI moved into ActionQueue; state/handlers remain in App.js
        self.assertIn("autoRepairTasks", app_content)
        self.assertIn("window.addEventListener('aria-auto-repair-started'", chat_content)
        self.assertIn("Auto-repair agent started", chat_content)

    def test_dashboard_wires_production_readiness_panel(self):
        app_content = self._read_file(self.app_js)
        # Production readiness data still consumed; UI replaced by ActionQueue
        self.assertIn("production_readiness", app_content)
        # fingerprint diagnostics fetch in useAriaData hook
        hook_content = self._read_file(os.path.join(
            self.repo_root, "dashboard", "src", "hooks", "useAriaData.js"))
        self.assertIn("/api/diagnostics/fingerprint", hook_content)
        self.assertIn("handleRunProductionTemplate", app_content)

    def test_architecture_drawer_auto_starts_designer(self):
        drawer_path = os.path.join(self.component_dir, "ArchitectureDrawer.js")
        content = self._read_file(drawer_path)
        self.assertIn("/api/designer/ensure-running", content)
        self.assertIn("/api/designer/touch", content)
        self.assertIn("/api/designer/lineage?limit=20", content)
        self.assertIn("Starting Aria Designer", content)
        self.assertNotIn("Run: cd aria_designer/ui && npm run dev", content)

    def test_architecture_drawer_embedded_bridge_handshake(self):
        drawer_path = os.path.join(self.component_dir, "ArchitectureDrawer.js")
        content = self._read_file(drawer_path)
        # Embedded iframe should signal readiness, then receive load-result.
        self.assertIn("embedded-ready", content)
        self.assertIn("load-result", content)
        # Parent should listen for graph load success/error signals.
        self.assertIn("graph-loaded", content)
        self.assertIn("graph-load-error", content)

    def test_dashboard_wires_code_healer_panel(self):
        app_content = self._read_file(self.app_js)
        # Healer state still in App.js; UI moved into ActionQueue
        self.assertIn("healerTasks", app_content)

    def test_strategy_advisor_marks_actionability_and_sanitizes_pseudo_code(self):
        strategy_path = os.path.join(self.component_dir, "StrategyAdvisor.js")
        content = self._read_file(strategy_path)
        self.assertIn("Actionable", content)
        self.assertIn("Advice only", content)
        self.assertIn("sanitizeBriefingText", content)
        self.assertIn("details sent to local agent", content)

    def test_strategy_advisor_preserves_full_suggested_config_passthrough(self):
        strategy_path = os.path.join(self.component_dir, "StrategyAdvisor.js")
        content = self._read_file(strategy_path)
        self.assertIn("const fullConfig = { ...suggestedConfig };", content)
        self.assertIn("delete fullConfig.hypothesis;", content)
        self.assertIn("delete fullConfig.result_ids;", content)
        self.assertIn("...fullConfig,", content)
        self.assertIn("sparseCoverage", content)
        self.assertIn("Sparse coverage:", content)

    def test_aria_status_sanitizes_hypothesis_summary(self):
        status_path = os.path.join(self.component_dir, "AriaStatus.js")
        content = self._read_file(status_path)
        self.assertIn("sanitizeHypothesisText", content)
        self.assertIn("summarizedHypothesis", content)
        self.assertNotIn('{aria.current_hypothesis}', content)

    def test_tab_names_match_content(self):
        """All tab names in App.js should have corresponding content blocks."""
        app_content = self._read_file(self.app_js)

        # Extract tab list from nav
        import re
        tab_match = re.search(r"\[([^\]]+)\]\.map\(tab", app_content)
        if tab_match:
            tabs_str = tab_match.group(1)
            tabs = re.findall(r"'(\w[\w-]*)'", tabs_str)

            for tab in tabs:
                # Each tab should have activeTab === 'tabname'
                self.assertIn(
                    f"activeTab === '{tab}'",
                    app_content,
                    f"Tab '{tab}' has no content block in App.js",
                )

    def test_onSelectProgram_wired(self):
        """Components with onSelectProgram should receive it as prop."""
        app_content = self._read_file(self.app_js)
        # These components should pass onSelectProgram
        for comp in ["TopPrograms", "Leaderboard", "ExperimentDetail"]:
            self.assertIn(
                f"onSelectProgram={{handleSelectProgram}}",
                app_content,
                f"{comp} should pass onSelectProgram prop",
            )

    def test_sse_event_types_handled(self):
        """All SSE event types emitted by runner should be handled in LiveFeed."""
        livefeed_path = os.path.join(self.component_dir, "LiveFeed.js")
        content = self._read_file(livefeed_path)

        expected_events = [
            "program_evaluated",
            "experiment_started",
            "experiment_completed",
            "experiment_failed",
            "investigation_started",
            "investigation_completed",
            "validation_started",
            "validation_completed",
            "breakthrough_detected",
            "mode_selected",
        ]

        for event in expected_events:
            self.assertIn(
                f"'{event}'",
                content,
                f"LiveFeed.js missing handler for SSE event: {event}",
            )

    def test_live_feed_filters_non_renderable_events_to_avoid_blank_rows(self):
        livefeed_path = os.path.join(self.component_dir, "LiveFeed.js")
        content = self._read_file(livefeed_path)
        self.assertIn("RENDERABLE_EVENT_TYPES", content)
        self.assertIn("normalizeLiveFeedEvent", content)
        self.assertIn("if (!RENDERABLE_EVENT_TYPES.has(normalizedType)) return null;", content)
        self.assertIn("annotateGenerationHistory", content)
        self.assertIn("not in current feed history", content)

    def test_frontend_api_routes_exist_in_backend(self):
        """All frontend /api paths should map to a backend Flask route."""
        import re

        api_content = self._read_file(self.api_py)
        route_re = re.compile(r"@app\.route\(\s*['\"](/api/[^'\"]+)['\"]")
        backend_routes = [self._normalize_route(r) for r in route_re.findall(api_content)]

        for filepath in self.component_files + [self.app_js]:
            content = self._read_file(filepath)
            found = re.findall(r"/api/[A-Za-z0-9_\-/${}]+(?:/[A-Za-z0-9_\-/${}]+)*", content)
            for path in found:
                normalized = self._normalize_route(path)
                matched = any(self._route_matches(b, normalized) for b in backend_routes)
                self.assertTrue(
                    matched,
                    f"Frontend route has no backend mapping: {path} in {os.path.basename(filepath)}",
                )

    def test_strategy_advisor_breakthrough_count_uses_tier(self):
        """StrategyAdvisor should derive tier counts from tier + use canonical summary keys."""
        strategy_path = os.path.join(self.component_dir, "StrategyAdvisor.js")
        content = self._read_file(strategy_path)

        self.assertIn("const tier = normalizeTier(entry);", content)
        self.assertIn("const effectiveTier = tier || 'screening';", content)
        self.assertIn("tierSummary[effectiveTier] += 1;", content)
        self.assertIn("if (effectiveTier === 'breakthrough')", content)
        self.assertIn("total_programs_evaluated", content)

    def test_research_report_uses_stage1_survivors_summary_key(self):
        """ReportDetail should read stage1_survivors (with legacy fallback)."""
        detail_path = os.path.join(self.component_dir, "ReportDetail.js")
        content = self._read_file(detail_path)
        self.assertIn("const s1Survivors = s.stage1_survivors ?? s.total_s1_passed ?? 0;", content)

    def test_research_report_wires_scoped_query_builder_controls(self):
        detail_path = os.path.join(self.component_dir, "ReportDetail.js")
        content = self._read_file(detail_path)
        self.assertIn("/api/report/query", content)
        self.assertIn("Generate Scoped Report", content)
        self.assertIn("Load Full Details", content)
        self.assertIn("theme", content)
        self.assertIn("trend", content)
        self.assertIn("fast: fast ? '1' : '0'", content)

    def test_investigation_actions_use_eligibility_gating_hooks(self):
        """App + candidate views should wire explicit eligibility gating for investigate/queue actions."""
        app_content = self._read_file(self.app_js)
        leaderboard_content = self._read_file(os.path.join(self.component_dir, "Leaderboard.js"))
        top_programs_content = self._read_file(os.path.join(self.component_dir, "TopPrograms.js"))
        program_detail_content = self._read_file(os.path.join(self.component_dir, "ProgramDetail.js"))

        # eligibilityByResultId is now derived from shared AriaData context
        self.assertIn("buildEligibilityByResultId(leaderboardEntries", app_content)
        self.assertIn("eligibilityByResultId", app_content)
        self.assertIn("filter(resultId => eligibilityByResultId[resultId]?.investigationEligible)", app_content)
        self.assertIn("eligibilityByResultId={eligibilityByResultId}", app_content)
        self.assertIn("intent: item?.intent === 'validation' ? 'validation' : 'investigation'", app_content)
        self.assertIn("const stillEligibleForIntent = intent === 'validation'", app_content)
        self.assertIn("filter(item => item.intent === 'investigation')", app_content)
        self.assertIn("filter(item => item.intent === 'validation')", app_content)

        self.assertIn("function candidateEligibility(entry)", leaderboard_content)
        self.assertIn("already_investigated_unchanged", leaderboard_content)
        self.assertIn("disabled={!isQueued && !eligibility.queueEligible}", leaderboard_content)
        self.assertIn("const queueIntent = eligibility.validationEligible", leaderboard_content)
        self.assertIn("Queue Validate", leaderboard_content)
        self.assertIn("intent: queueIntent", leaderboard_content)

        self.assertIn("eligibilityByResultId", top_programs_content)
        self.assertIn("queueEligible", top_programs_content)
        self.assertIn("Ineligible", top_programs_content)
        self.assertIn("const queueIntent = eligibility?.validationEligible", top_programs_content)
        self.assertIn("Queue Investigate", top_programs_content)

        self.assertIn("eligibilityByResultId", program_detail_content)
        self.assertIn("Already investigated", program_detail_content)

    def test_program_detail_refinement_intent_actions_are_wired(self):
        """ProgramDetail should expose intent-specific fingerprint refinement actions."""
        program_detail_content = self._read_file(os.path.join(self.component_dir, "ProgramDetail.js"))
        # Core refinement launch infrastructure
        self.assertIn("const handleLaunchRefinement = async", program_detail_content)
        self.assertIn("refine_intent: intent", program_detail_content)
        self.assertIn("Refinement Trace", program_detail_content)
        self.assertIn("Open Refinement Run", program_detail_content)
        self.assertIn("/api/experiments/${latestRefineLaunch.experimentId}", program_detail_content)
        self.assertIn("setLatestRefineLaunch", program_detail_content)
        self.assertIn("setRefineLaunchHistory", program_detail_content)
        self.assertIn("Recent Refinement Launches", program_detail_content)
        self.assertIn("Open Fingerprint", program_detail_content)
        self.assertIn("View Top Refined Result", program_detail_content)
        self.assertIn("lastRefinedCandidate", program_detail_content)
        self.assertIn("newCandidates", program_detail_content)
        self.assertIn("New Fingerprints", program_detail_content)
        # Data-driven refinement via RefinementAdvisor
        self.assertIn("RefinementAdvisor", program_detail_content)
        self.assertIn("onLaunchRefinement", program_detail_content)
        self.assertIn("Refine with Recommendation", program_detail_content)

    def test_program_detail_refinement_rationale_panel_is_wired(self):
        """ProgramDetail should render refinement rationale from graph metadata."""
        program_detail_content = self._read_file(os.path.join(self.component_dir, "ProgramDetail.js"))
        self.assertIn("function RefinementRationale({ program })", program_detail_content)
        self.assertIn("function RefinementLineage({ program, onViewInLeaderboard })", program_detail_content)
        self.assertIn("program?.graph_json_parsed?.metadata", program_detail_content)
        self.assertIn("program?.lineage_chain", program_detail_content)
        self.assertIn("refinement.intent_score", program_detail_content)
        self.assertIn("refinement.intent_score_breakdown", program_detail_content)
        self.assertIn("weighted_terms", program_detail_content)
        self.assertIn("Refinement Rationale", program_detail_content)
        self.assertIn("Refinement Lineage", program_detail_content)
        self.assertIn("Components:", program_detail_content)
        self.assertIn("learning-guided refinement", program_detail_content)

    def test_control_panel_renders_hypothesis_missing_fields(self):
        """ControlPanel should show checklist chips for missing hypothesis fields."""
        content = self._read_file(os.path.join(self.component_dir, "ControlPanel.js"))
        self.assertIn("Missing fields:", content)
        self.assertIn("critique.missing_fields", content)
        self.assertIn("source_selection_rule", content)
        self.assertIn("confounders_checklist", content)

    def test_top_programs_copy_clarifies_program_vs_fingerprint_and_shows_leading_fingerprints(self):
        content = self._read_file(os.path.join(self.component_dir, "TopPrograms.js"))
        self.assertIn("Candidate Programs (Raw Survivors)", content)
        self.assertIn("Program Fingerprint ID is the architecture identity for that row", content)
        self.assertIn("Architecture identity for each program row; the same fingerprint can appear multiple times when rerun.", content)
        self.assertIn("Fingerprint Leaderboard (Deduplicated Architecture IDs)", content)

    def test_learning_trajectory_minimum_threshold_copy_uses_backend_contract(self):
        """LearningPanel should avoid hard-coded trajectory threshold copy drift."""
        learning_panel_content = self._read_file(os.path.join(self.component_dir, "LearningPanel.js"))

        self.assertIn("const minimumExperiments = Math.max(2, Number(trajectory?.min_experiments_required) || 5);", learning_panel_content)
        self.assertIn("Need at least {minimumExperiments} experiments to compute a learning trajectory.", learning_panel_content)
        self.assertNotIn("Need at least 3 experiments to compute a learning trajectory.", learning_panel_content)

    def test_trend_charts_show_stabilized_s1_and_confidence_bands(self):
        """TrendCharts should consume stabilized data and wire adaptation refresh context."""
        trend_content = self._read_file(os.path.join(self.component_dir, "TrendCharts.js"))

        self.assertIn("valueKey=\"adjusted_s1_pass_rate\"", trend_content)
        self.assertIn("bandLowerKey=\"s1_confidence_lower\"", trend_content)
        self.assertIn("bandUpperKey=\"s1_confidence_upper\"", trend_content)
        scoring_engine = self._read_file(os.path.join(self.component_dir, "..", "utils", "scoringEngine.js"))
        self.assertIn("reliabilityMultiplier", scoring_engine)
        self.assertIn("trend_confidence", trend_content)
        self.assertIn("/api/trends", trend_content)
        self.assertIn("setInterval(fetchTrendContext, 10000)", trend_content)
        self.assertIn("Adaptation outcomes (recent)", trend_content)

    def test_research_report_mentions_deduplicated_fingerprint_rankings(self):
        """Discovery rankings should explain fingerprint dedup and repeat metadata."""
        report_content = self._read_file(os.path.join(self.component_dir, "ResearchReport.js"))
        detail_content = self._read_file(os.path.join(self.component_dir, "ReportDetail.js"))
        rankings_content = self._read_file(os.path.join(self.component_dir, "report", "DiscoveryRankings.js"))
        report_utils_content = self._read_file(os.path.join(self.component_dir, "report", "reportUtils.js"))

        self.assertIn("ReportGallery", report_content)
        self.assertIn("ReportDetail", report_content)

        self.assertIn("fingerprint-deduplicated", rankings_content)
        self.assertIn("Grouped view", rankings_content)
        self.assertIn("Expanded reruns", rankings_content)
        self.assertIn("Same architecture repeated means reruns of one fingerprint", rankings_content)
        self.assertIn("expandedPrograms", rankings_content)
        self.assertIn("top_programs_expanded", detail_content)
        self.assertIn("repeat_count", rankings_content)
        self.assertIn("repeat_experiment_span", rankings_content)
        self.assertIn("eligibilityByResultId", rankings_content)
        self.assertIn("Queue Validate", rankings_content)
        self.assertIn("Ineligible", rankings_content)
        self.assertIn("reportQueueReasonLabel", rankings_content)
        self.assertIn("reportQueueReasonLabel", report_utils_content)
        self.assertIn("Unique Architectures vs Reruns", detail_content)
        self.assertIn("architecture_rerun_telemetry", detail_content)

    def test_learning_panel_mentions_unique_vs_rerun_telemetry(self):
        """LearningPanel should show unique architecture vs rerun concentration metrics."""
        learning_panel_content = self._read_file(os.path.join(self.component_dir, "LearningPanel.js"))
        self.assertIn("Unique Architectures vs Reruns", learning_panel_content)
        self.assertIn("architecture_rerun_telemetry", learning_panel_content)
        self.assertIn("Top fingerprint concentration", learning_panel_content)

    def test_learning_panel_wires_fingerprint_diagnostics_card(self):
        """LearningPanel should render fingerprint sensitivity skip diagnostics via shared context."""
        learning_panel_content = self._read_file(os.path.join(self.component_dir, "LearningPanel.js"))
        self.assertIn("Fingerprint Diagnostics", learning_panel_content)
        self.assertIn("Sensitivity skips:", learning_panel_content)
        self.assertIn("fingerprintDiagnostics", learning_panel_content)
        # fingerprint fetch is now in useAriaData hook
        hook_content = self._read_file(os.path.join(
            self.repo_root, "dashboard", "src", "hooks", "useAriaData.js"))
        self.assertIn("/api/diagnostics/fingerprint", hook_content)
        self.assertIn("sensitivity_skips", hook_content)

    def test_learning_panel_wires_insight_synergy_matrix(self):
        learning_panel_content = self._read_file(os.path.join(self.component_dir, "LearningPanel.js"))
        self.assertIn("Insight Synergy Matrix", learning_panel_content)
        self.assertIn("Positive Pairs", learning_panel_content)
        self.assertIn("Conflicting Pairs", learning_panel_content)
        self.assertIn("/api/analytics/insight-interactions", learning_panel_content)

    @staticmethod
    def _normalize_route(path: str) -> str:
        import re

        p = path.split("?", 1)[0]
        p = re.sub(r"<[^>]+>", "*", p)
        p = re.sub(r"\$\{[^}]+\}", "*", p)
        p = re.sub(r"//+", "/", p)
        return p.rstrip("/") or "/"

    @staticmethod
    def _route_matches(backend: str, frontend: str) -> bool:
        if backend == frontend:
            return True

        b_parts = [p for p in backend.strip("/").split("/") if p]
        f_parts = [p for p in frontend.strip("/").split("/") if p]
        if len(b_parts) != len(f_parts):
            return False

        for b, f in zip(b_parts, f_parts):
            if b == "*" or f == "*":
                continue
            if b != f:
                return False
        return True

    def test_strategy_advisor_compute_strategy_includes_data_sources(self):
        """Every computeStrategy() return path must include a non-empty dataSources array."""
        strategy_path = os.path.join(self.component_dir, "StrategyAdvisor.js")
        content = self._read_file(strategy_path)
        # All 10 return paths in computeStrategy should have dataSources
        self.assertIn("dataSources: [", content)
        # Check that key rules include specific metric names
        self.assertIn("metric: 'Total Experiments'", content)
        self.assertIn("metric: 'Breakthrough Candidates'", content)
        self.assertIn("metric: 'S1 Pass Rate'", content)
        self.assertIn("metric: 'Under-tested Math Families'", content)
        self.assertIn("metric: 'Consecutive Zero-Survivor Runs'", content)
        self.assertIn("metric: 'Pipeline Status'", content)

    def test_strategy_advisor_briefing_data_source_extraction(self):
        """extractBriefingDataSources should convert evidence fields into dataSources format."""
        strategy_path = os.path.join(self.component_dir, "StrategyAdvisor.js")
        content = self._read_file(strategy_path)
        self.assertIn("function extractBriefingDataSources(evidence)", content)
        self.assertIn("metric: 'Learning Trend'", content)
        self.assertIn("metric: 'Recent Avg S1 Rate'", content)
        self.assertIn("metric: 'Sparsity Coverage'", content)
        self.assertIn("metric: 'Pipeline Distribution'", content)

    def test_strategy_advisor_data_source_badge_renders_tooltip(self):
        """DataSourceBadge must render tooltip with data source details on hover."""
        strategy_path = os.path.join(self.component_dir, "StrategyAdvisor.js")
        content = self._read_file(strategy_path)
        self.assertIn("function DataSourceBadge(", content)
        self.assertIn("Recommended Action", content)
        self.assertIn("Data Sources", content)
        self.assertIn("mergedDataSources", content)
        # Tooltip should show comparison text and navigable tab links
        self.assertIn("formatComparison(ds)", content)
        self.assertIn("onNavigateEvidence(ds.tab)", content)



class TestScaleUpFix(unittest.TestCase):
    """Test scale-up no longer passes invalid columns to record_program_result."""

    def test_scale_up_metrics_no_invalid_columns(self):
        """Verify _extract_graph_metrics doesn't produce non-schema keys."""
        from research.scientist.runner import ExperimentRunner
        from research.synthesis.grammar import generate_layer_graph

        graph = generate_layer_graph(seed=42)
        runner = ExperimentRunner.__new__(ExperimentRunner)
        runner._math_spaces_registered = False
        metrics = runner._extract_graph_metrics(graph)

        # These keys must NOT appear (they caused the scale-up crash)
        forbidden_keys = {"source_result_id", "scale_up_steps",
                          "scale_up_batch_size", "scale_up_seq_len"}
        for key in forbidden_keys:
            self.assertNotIn(key, metrics,
                             f"Forbidden key '{key}' found in graph metrics")


class TestSandboxShapeValidation(unittest.TestCase):
    """Tests for sandbox logits shape validation (#23)."""

    @unittest.skipUnless(HAS_TORCH, "torch required")
    def test_correct_shape_passes(self):
        """Model with correct (B, S, V) output passes shape check."""
        from research.eval.sandbox import safe_eval

        class GoodModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(1000, 64)
                self.linear = torch.nn.Linear(64, 1000)

            def forward(self, x):
                return self.linear(self.embed(x))

        result = safe_eval(GoodModel(), batch_size=2, seq_len=16,
                           vocab_size=1000, device="cpu",
                           run_stability_probe=False)
        self.assertNotEqual(result.error_type, "shape_mismatch",
                            f"Unexpected shape_mismatch: {result.error}")

    @unittest.skipUnless(HAS_TORCH, "torch required")
    def test_wrong_batch_dim_fails(self):
        """Model that returns wrong batch dimension is caught."""
        from research.eval.sandbox import safe_eval

        class BadBatchModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(1000, 64)
                self.linear = torch.nn.Linear(64, 1000)

            def forward(self, x):
                out = self.linear(self.embed(x))
                # Return only first sample — wrong batch dim
                return out[:1]

        result = safe_eval(BadBatchModel(), batch_size=2, seq_len=16,
                           vocab_size=1000, device="cpu",
                           run_stability_probe=False)
        self.assertEqual(result.error_type, "shape_mismatch")
        self.assertIn("(1, 16, 1000)", result.error)

    @unittest.skipUnless(HAS_TORCH, "torch required")
    def test_wrong_vocab_dim_fails(self):
        """Model that returns wrong vocab dimension is caught."""
        from research.eval.sandbox import safe_eval

        class BadVocabModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(1000, 64)
                self.linear = torch.nn.Linear(64, 500)  # Wrong vocab dim

            def forward(self, x):
                return self.linear(self.embed(x))

        result = safe_eval(BadVocabModel(), batch_size=2, seq_len=16,
                           vocab_size=1000, device="cpu",
                           run_stability_probe=False)
        self.assertEqual(result.error_type, "shape_mismatch")
        self.assertIn("vocab", result.error.lower())

    @unittest.skipUnless(HAS_TORCH, "torch required")
    def test_2d_output_fails(self):
        """Model that returns 2D output (missing seq dim) is caught."""
        from research.eval.sandbox import safe_eval

        class FlatModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(1000, 64)
                self.linear = torch.nn.Linear(64, 1000)

            def forward(self, x):
                return self.linear(self.embed(x)).reshape(-1, 1000)

        result = safe_eval(FlatModel(), batch_size=2, seq_len=16,
                           vocab_size=1000, device="cpu",
                           run_stability_probe=False)
        self.assertEqual(result.error_type, "shape_mismatch")



class TestSandboxCudaDetection(unittest.TestCase):
    """Tests for CUDA fatal error detection and health probing in sandbox."""

    def test_is_cuda_fatal_device_side_assert(self):
        from research.eval.sandbox import is_cuda_fatal

        err = RuntimeError("CUDA error: device-side assert triggered")
        self.assertTrue(is_cuda_fatal(err))

    def test_is_cuda_fatal_illegal_memory(self):
        from research.eval.sandbox import is_cuda_fatal

        err = RuntimeError("CUDA error: an illegal memory access was encountered")
        self.assertTrue(is_cuda_fatal(err))

    def test_is_cuda_fatal_context_destroyed(self):
        from research.eval.sandbox import is_cuda_fatal

        err = RuntimeError("context is destroyed")
        self.assertTrue(is_cuda_fatal(err))

    def test_is_cuda_fatal_normal_error(self):
        from research.eval.sandbox import is_cuda_fatal

        err = RuntimeError("some normal runtime error")
        self.assertFalse(is_cuda_fatal(err))

    def test_is_cuda_fatal_oom_is_not_fatal(self):
        """OOM is recoverable and should NOT be classified as fatal."""
        from research.eval.sandbox import is_cuda_fatal

        err = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        self.assertFalse(is_cuda_fatal(err))

    def test_safe_eval_categorizes_cuda_fatal(self):
        """Mock a device-side assert to verify safe_eval returns cuda_fatal."""
        import torch.nn as nn
        from unittest.mock import patch
        from research.eval.sandbox import safe_eval

        model = nn.Linear(32, 32)
        with patch.object(
            nn.Module, "to",
            side_effect=RuntimeError("CUDA error: device-side assert triggered"),
        ):
            result = safe_eval(model, device="cpu")
        self.assertFalse(result.passed)
        self.assertEqual(result.error_type, "cuda_fatal")
        self.assertIn("device-side assert", result.error)

    def test_sandbox_result_has_cuda_fatal_type(self):
        """SandboxResult should be able to carry cuda_fatal error_type."""
        from research.eval.sandbox import SandboxResult

        r = SandboxResult(error_type="cuda_fatal", error="test")
        d = r.to_dict()
        self.assertEqual(d["error_type"], "cuda_fatal")




if __name__ == '__main__':
    unittest.main()
