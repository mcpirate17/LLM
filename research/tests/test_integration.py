"""
Integration smoke tests — verifies core imports work.

All substantive tests have been moved to domain-specific files:
  - test_notebook.py           (notebook schema, CRUD, leaderboard)
  - test_novelty.py            (novelty scoring, calibration, baselines)
  - test_api_integration.py    (API endpoints, SSE, chat, recommendations)
  - test_pipeline_integration.py (runner, escalation, diagnostics, pipeline)
  - test_persona_integration.py (persona, prompts, context builders, strategy)
  - test_primitives_integration.py (primitives, math spaces, compression, sparse)
  - test_sandbox_integration.py (sandbox validation, CUDA detection, scale-up)
  - test_reference_architectures.py (CKA artifacts, reference arch validation)
  - test_runner_mode_selection.py (mode selection, aria modes)
  - test_dashboard_schema.py   (dashboard payload consistency)
  - test_package_wiring.py     (import wiring, dead code audit)
  - test_synthesis_integration.py (grammar, evolution, morphological, frontier)

Run domain tests:  pytest -m unit  |  pytest -m api  |  pytest -m pipeline
"""

import importlib
import unittest

import pytest

pytestmark = pytest.mark.unit


class TestCoreImports(unittest.TestCase):
    """Smoke test: verify core modules are importable."""

    def test_notebook_importable(self):
        mod = importlib.import_module("research.scientist.notebook")
        self.assertTrue(hasattr(mod, "LabNotebook"))

    def test_grammar_importable(self):
        mod = importlib.import_module("research.synthesis.grammar")
        self.assertTrue(hasattr(mod, "GrammarConfig"))

    def test_graph_importable(self):
        mod = importlib.import_module("research.synthesis.graph")
        self.assertTrue(hasattr(mod, "ComputationGraph"))

    def test_compiler_importable(self):
        mod = importlib.import_module("research.synthesis.compiler")
        self.assertTrue(hasattr(mod, "compile_model"))

    def test_persona_importable(self):
        mod = importlib.import_module("research.scientist.persona")
        self.assertTrue(hasattr(mod, "Aria"))

    def test_api_importable(self):
        mod = importlib.import_module("research.scientist.api")
        self.assertTrue(hasattr(mod, "create_app"))
