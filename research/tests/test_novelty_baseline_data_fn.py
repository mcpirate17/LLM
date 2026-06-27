"""Baseline training with custom data functions.

Split from the test_novelty.py omnibus on 2026-06-13."""

import pytest
import importlib
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
class TestBaselineDataFn(unittest.TestCase):
    """Tests for baseline training with custom data functions."""

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_baseline_with_data_fn(self):
        """Baseline trains using provided data_fn instead of random tokens."""
        from research.eval.baseline import TransformerBaseline
        import torch

        call_count = [0]

        def fake_data(batch_size, seq_len, dev):
            call_count[0] += 1
            return torch.randint(0, 1024, (batch_size, seq_len), device=dev)

        with tempfile.TemporaryDirectory() as tmpdir:
            bl = TransformerBaseline(cache_path=os.path.join(tmpdir, "bl.db"))
            loss = bl.get_baseline_loss(
                d_model=64,
                seq_len=32,
                n_steps=5,
                vocab_size=1024,
                batch_size=2,
                device="cpu",
                data_fn=fake_data,
                data_tag="test",
            )
            self.assertTrue(0 < loss < 20)
            self.assertGreater(call_count[0], 0, "data_fn should have been called")

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_baseline_compare_with_data_fn(self):
        """compare() passes data_fn through to training."""
        from research.eval.baseline import TransformerBaseline
        import torch

        def fake_data(batch_size, seq_len, dev):
            return torch.randint(0, 1024, (batch_size, seq_len), device=dev)

        with tempfile.TemporaryDirectory() as tmpdir:
            bl = TransformerBaseline(cache_path=os.path.join(tmpdir, "bl.db"))
            ratio = bl.compare(
                program_loss=5.0,
                d_model=64,
                seq_len=32,
                n_steps=5,
                vocab_size=1024,
                batch_size=2,
                device="cpu",
                data_fn=fake_data,
                data_tag="test",
            )
            self.assertIsInstance(ratio, float)
            self.assertGreater(ratio, 0)

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_data_tag_separates_cache(self):
        """Different data_tags produce separate cache entries."""
        from research.eval.baseline import TransformerBaseline

        with tempfile.TemporaryDirectory() as tmpdir:
            bl = TransformerBaseline(cache_path=os.path.join(tmpdir, "bl.db"))
            key_random = bl._config_key(64, 32, 5, 1024, data_tag="random")
            key_hydra = bl._config_key(64, 32, 5, 1024, data_tag="hydra")
            self.assertNotEqual(key_random, key_hydra)

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_hydra_batch_fallback_to_random(self):
        """_get_hydra_batch returns None when HYDRA loader fails to init."""
        from research.scientist.runner import ExperimentRunner, RunConfig
        import torch

        config = RunConfig(data_mode="hydra", hydra_project_root="/nonexistent")
        runner = ExperimentRunner.__new__(ExperimentRunner)
        runner._hydra_loader = None
        runner._hydra_iter = None
        runner._hydra_signature = ""
        # Mock the import to raise immediately rather than risk hanging
        with patch.dict("sys.modules", {"hydra.data": None}):
            result = runner._get_hydra_batch(config, 2, 32, torch.device("cpu"))
        self.assertIsNone(result, "Should return None when HYDRA unavailable")

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_make_baseline_data_fn_random_mode(self):
        """_make_baseline_data_fn returns (None, 'random', False) for random mode."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        config = RunConfig(data_mode="random")
        runner = ExperimentRunner.__new__(ExperimentRunner)
        data_fn, data_tag, cache_data_fn = runner._make_baseline_data_fn(config)
        self.assertIsNone(data_fn)
        self.assertEqual(data_tag, "random")
        self.assertFalse(cache_data_fn)

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_make_baseline_data_fn_hydra_mode(self):
        """_make_baseline_data_fn returns a callable for hydra mode."""
        from research.scientist.runner import ExperimentRunner, RunConfig

        config = RunConfig(data_mode="hydra")
        runner = ExperimentRunner.__new__(ExperimentRunner)
        data_fn, data_tag, cache_data_fn = runner._make_baseline_data_fn(config)
        self.assertIsNotNone(data_fn)
        self.assertEqual(data_tag, "hydra")
        self.assertFalse(cache_data_fn)


if __name__ == "__main__":
    unittest.main()
