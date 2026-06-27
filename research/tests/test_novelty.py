"""
Integration Tests for the AI Scientist Research Pipeline

Novelty scoring tests (calibration/baseline-data-fn/CKA-artifact suites
split into sibling files on 2026-06-13).

Run: cd /path/to/LLM && python -m unittest research.tests.test_integration -v
"""

import pytest
import importlib
import unittest

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


@unittest.skipUnless(HAS_TORCH, "requires torch for graph/metrics modules")
class TestNoveltyScoring(unittest.TestCase):
    """Test that novelty scoring no longer always returns 1.0."""

    def _make_graph(self, n_ops=5, op_names=None):
        """Create a simple computation graph for testing."""
        from research.synthesis.graph import ComputationGraph

        graph = ComputationGraph(model_dim=256)

        # Add input node
        input_id = graph.add_input()

        if op_names is None:
            op_names = ["relu", "gelu", "tanh", "sigmoid", "silu"]

        prev_id = input_id
        for op_name in op_names[:n_ops]:
            prev_id = graph.add_op(op_name, [prev_id])

        graph.set_output(prev_id)
        return graph

    def test_structural_novelty_not_always_one(self):
        """Structural novelty should NOT be ~1.0 for every graph."""
        from research.eval.metrics import novelty_score

        # Simple graph with few unique ops (all the same)
        simple_graph = self._make_graph(
            op_names=["relu", "relu", "relu", "relu", "relu"]
        )

        # Diverse graph with many unique ops
        diverse_graph = self._make_graph(
            op_names=["relu", "gelu", "tanh", "sigmoid", "silu"]
        )

        simple_nov = novelty_score(simple_graph)
        diverse_nov = novelty_score(diverse_graph)

        # Neither should be 1.0 (the old bug)
        self.assertLess(
            simple_nov.structural_novelty,
            0.95,
            "Simple graph should NOT have max novelty",
        )

        # Diverse should be higher than simple
        self.assertGreater(
            diverse_nov.structural_novelty,
            simple_nov.structural_novelty,
            "Diverse graph should have higher novelty than simple",
        )

    def test_no_fingerprint_discount(self):
        """Without behavioral fingerprint, overall_novelty should be discounted."""
        from research.eval.metrics import novelty_score

        graph = self._make_graph()
        nov = novelty_score(graph, fingerprint=None)

        # Should be discounted (0.6x structural)
        expected_max = nov.structural_novelty * 0.6 + 0.01  # small tolerance
        self.assertLessEqual(
            nov.overall_novelty,
            expected_max,
            "No-fingerprint novelty should be discounted",
        )

    def test_with_fingerprint_uses_behavioral(self):
        """With fingerprint, overall should use 70% behavioral weight."""
        from research.eval.metrics import novelty_score
        from research.eval.fingerprint import BehavioralFingerprint

        graph = self._make_graph()
        fp = BehavioralFingerprint(
            novelty_score=0.9,
            cka_vs_transformer=0.3,
            cka_vs_ssm=0.2,
            cka_vs_conv=0.1,
        )

        nov = novelty_score(graph, fingerprint=fp)

        # raw_novelty should be weighted: 0.3 * structural + 0.7 * behavioral
        # overall_novelty = raw_novelty * CKA reference penalty
        expected_raw = 0.3 * nov.structural_novelty + 0.7 * 0.9
        self.assertAlmostEqual(nov.raw_novelty, expected_raw, places=2)
        # overall_novelty should be <= raw_novelty (penalty is <= 1.0)
        self.assertLessEqual(nov.overall_novelty, expected_raw + 0.01)

    def test_behavior_signature_contributes_to_fingerprint_novelty(self):
        """Fingerprint novelty is not just 1 - max(CKA)."""
        # Both moved during the fingerprint module split:
        #   BehavioralFingerprint -> fingerprint_types
        #   _blend_behavioral_novelty -> fingerprint_scoring (now public)
        from research.eval.fingerprint_runtime import (
            blend_behavioral_novelty as _blend_behavioral_novelty,
        )
        from research.eval.fingerprint_types import BehavioralFingerprint

        fp = BehavioralFingerprint(
            cka_vs_transformer=0.3,
            cka_vs_ssm=0.2,
            cka_vs_conv=0.1,
            interaction_locality=1.0,
            interaction_sparsity=0.0,
            interaction_symmetry=1.0,
            interaction_hierarchy=0.0,
            isotropy=1.0,
            rank_ratio=0.0,
            sensitivity_uniformity=1.0,
            hierarchy_fitness=1.0,
        )
        expected_cka_only = 0.7
        blended = _blend_behavioral_novelty(fp)
        self.assertGreater(blended, expected_cka_only)
        self.assertLessEqual(blended, 1.0)

    def test_duplicate_penalization(self):
        """Exact duplicate fingerprints should be penalized."""
        from research.eval.metrics import novelty_score

        graph = self._make_graph()
        fp_str = graph.fingerprint()

        nov_fresh = novelty_score(graph, known_fingerprints=[])
        nov_dup = novelty_score(graph, known_fingerprints=[fp_str])

        self.assertGreater(
            nov_fresh.overall_novelty,
            nov_dup.overall_novelty,
            "Duplicate should be penalized",
        )


if __name__ == "__main__":
    unittest.main()
