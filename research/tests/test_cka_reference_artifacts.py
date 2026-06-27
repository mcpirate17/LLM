"""CKA reference artifact loader/validator/cache tests.

Split from the test_novelty.py omnibus on 2026-06-13."""

import pytest
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path

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
class TestCkaReferenceArtifacts(unittest.TestCase):
    """Tests for CKA reference artifact loader/validator/cache (#28/#43 Phase A)."""

    def _make_artifact_dir(self, tmpdir, manifest_override=None, families=None):
        """Helper: create a valid artifact directory with manifest and .pt files."""
        import torch

        art_dir = os.path.join(tmpdir, "cka_references", "v1")
        os.makedirs(art_dir, exist_ok=True)

        manifest = {
            "artifact_version": "v1",
            "schema_version": "1",
            "created_at": "2026-01-01T00:00:00Z",
            "code_version": "test",
            "reference_families": ["transformer", "ssm", "conv"],
            "probe_protocol_hash": "abc123",
            "activation_shape": [16, 32],
            "quality_flags": {"overall": "good"},
        }
        if manifest_override:
            manifest.update(manifest_override)

        with open(os.path.join(art_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f)

        # Create .pt files
        shape = manifest["activation_shape"]
        for family in families or ["transformer", "ssm", "conv"]:
            data = {
                "activations": torch.randn(shape[0], shape[1]),
                "config": {"family": family},
                "training_info": {},
            }
            torch.save(data, os.path.join(art_dir, f"{family}.pt"))

        return art_dir

    def test_load_manifest_valid(self):
        """Valid manifest loads without error."""
        from research.eval.cka_references import load_manifest

        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            m = load_manifest(Path(art_dir))
            self.assertEqual(m.artifact_version, "v1")
            self.assertEqual(m.schema_version, "1")
            self.assertEqual(set(m.reference_families), {"transformer", "ssm", "conv"})
            self.assertEqual(m.activation_shape, [16, 32])

    def test_load_manifest_missing_file(self):
        """Missing manifest.json raises ValueError."""
        from research.eval.cka_references import load_manifest

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError, msg="No manifest.json"):
                load_manifest(Path(d))

    def test_load_manifest_malformed_json(self):
        """Malformed JSON raises ValueError."""
        from research.eval.cka_references import load_manifest

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "manifest.json")
            with open(p, "w") as f:
                f.write("{bad json")
            with self.assertRaises(ValueError):
                load_manifest(Path(d))

    def test_load_manifest_missing_fields(self):
        """Manifest missing required fields raises ValueError."""
        from research.eval.cka_references import load_manifest
        import json

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "manifest.json")
            with open(p, "w") as f:
                json.dump({"artifact_version": "v1"}, f)
            with self.assertRaises(ValueError, msg="missing required fields"):
                load_manifest(Path(d))

    def test_load_manifest_unsupported_schema(self):
        """Unsupported schema version raises ValueError."""
        from research.eval.cka_references import load_manifest

        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d, {"schema_version": "99"})
            with self.assertRaises(ValueError, msg="Unsupported schema"):
                load_manifest(Path(art_dir))

    def test_load_manifest_missing_family(self):
        """Manifest with incomplete families raises ValueError."""
        from research.eval.cka_references import load_manifest

        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(
                d, {"reference_families": ["transformer"]}
            )
            with self.assertRaises(ValueError, msg="missing reference families"):
                load_manifest(Path(art_dir))

    def test_load_manifest_bad_activation_shape(self):
        """Invalid activation_shape raises ValueError."""
        from research.eval.cka_references import load_manifest

        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d, {"activation_shape": [0, 32]})
            with self.assertRaises(ValueError):
                load_manifest(Path(art_dir))

    def test_load_reference_activations_valid(self):
        """Valid .pt files load as tensors with correct shape."""
        from research.eval.cka_references import (
            load_manifest,
            load_reference_activations,
        )

        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            m = load_manifest(Path(art_dir))
            refs = load_reference_activations(Path(art_dir), m)
            self.assertEqual(set(refs.keys()), {"transformer", "ssm", "conv"})
            for t in refs.values():
                self.assertEqual(tuple(t.shape[-2:]), (16, 32))

    def test_load_reference_activations_missing_file(self):
        """Missing .pt file raises ValueError."""
        from research.eval.cka_references import (
            load_manifest,
            load_reference_activations,
        )

        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            os.remove(os.path.join(art_dir, "ssm.pt"))
            m = load_manifest(Path(art_dir))
            with self.assertRaises(ValueError, msg="Missing artifact file"):
                load_reference_activations(Path(art_dir), m)

    def test_load_reference_activations_shape_mismatch(self):
        """Tensor with wrong shape raises ValueError."""
        from research.eval.cka_references import (
            load_manifest,
            load_reference_activations,
        )
        import torch

        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            # Overwrite one file with wrong shape
            torch.save(
                {"activations": torch.randn(8, 32)},
                os.path.join(art_dir, "conv.pt"),
            )
            m = load_manifest(Path(art_dir))
            with self.assertRaises(ValueError, msg="shape mismatch"):
                load_reference_activations(Path(art_dir), m)

    def test_store_no_artifacts_returns_none(self):
        """ReferenceCkaStore with no artifacts returns None references."""
        from research.eval.cka_references import ReferenceCkaStore

        with tempfile.TemporaryDirectory() as d:
            store = ReferenceCkaStore(artifact_dir=os.path.join(d, "nonexistent"))
            self.assertIsNone(store.get_references())
            self.assertFalse(store.is_artifact_backed)
            meta = store.get_metadata()
            self.assertEqual(meta["cka_source"], "none")

    def test_store_with_valid_artifacts(self):
        """ReferenceCkaStore loads valid artifacts successfully."""
        from research.eval.cka_references import ReferenceCkaStore

        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            store = ReferenceCkaStore(artifact_dir=art_dir)
            refs = store.get_references()
            self.assertIsNotNone(refs)
            self.assertEqual(set(refs.keys()), {"transformer", "ssm", "conv"})
            self.assertTrue(store.is_artifact_backed)
            meta = store.get_metadata()
            self.assertEqual(meta["cka_source"], "artifact")
            self.assertEqual(meta["cka_artifact_version"], "v1")

    def test_store_reference_similarity_cache_reuses_matrices(self):
        """Prepared reference similarity matrices are cached across calls."""
        from research.eval.cka_references import ReferenceCkaStore

        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            store = ReferenceCkaStore(artifact_dir=art_dir)
            first = store.get_reference_similarities()
            second = store.get_reference_similarities()
            self.assertIsNotNone(first)
            self.assertIs(first, second)
            self.assertEqual(set(first.keys()), {"transformer", "ssm", "conv"})

    def test_store_reset_clears_cache(self):
        """reset() clears loaded state so next access reloads."""
        from research.eval.cka_references import ReferenceCkaStore

        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            store = ReferenceCkaStore(artifact_dir=art_dir)
            self.assertTrue(store.is_artifact_backed)
            store.reset()
            # Point to nonexistent dir after reset
            store._artifact_dir = os.path.join(d, "gone")
            self.assertFalse(store.is_artifact_backed)

    def test_store_metadata_provenance_fields(self):
        """Metadata includes all expected provenance fields."""
        from research.eval.cka_references import ReferenceCkaStore

        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            store = ReferenceCkaStore(artifact_dir=art_dir)
            meta = store.get_metadata()
            self.assertIn("cka_source", meta)
            self.assertIn("cka_artifact_version", meta)
            self.assertIn("cka_probe_protocol_hash", meta)
            self.assertIn("cka_reference_quality", meta)
            self.assertEqual(meta["cka_reference_quality"], "good")

    # ── Phase C: Runtime CKA switchover tests ──

    def test_compute_reference_cka_with_artifacts(self):
        """_compute_reference_cka uses artifact activations when provided."""
        import torch
        from research.eval.fingerprint_runtime import (
            compute_reference_cka as _compute_reference_cka,
        )

        # Create fake candidate reps and reference activations
        S, D = 16, 32
        reps = torch.randn(1, S, D)
        ref_activations = {
            "transformer": torch.randn(S, D),
            "ssm": torch.randn(S, D),
            "conv": torch.randn(S, D),
        }
        result = _compute_reference_cka(reps, ref_activations=ref_activations)
        self.assertTrue(result["_succeeded"])
        for family in ("transformer", "ssm", "conv"):
            self.assertGreaterEqual(result[family], 0.0)
            self.assertLessEqual(result[family], 1.0)

    def test_compute_reference_cka_without_artifacts_fails_closed(self):
        """_compute_reference_cka should not invent reference scores."""
        import torch
        from research.eval.fingerprint_runtime import (
            compute_reference_cka as _compute_reference_cka,
        )

        reps = torch.randn(1, 16, 32)
        result = _compute_reference_cka(reps, ref_activations=None)
        self.assertFalse(result["_succeeded"])
        for family in ("transformer", "ssm", "conv"):
            self.assertEqual(result[family], 0.0)

    def test_compute_reference_cka_seq_len_mismatch(self):
        """Artifact CKA handles different seq lengths between candidate and reference."""
        import torch
        from research.eval.fingerprint_runtime import (
            compute_reference_cka as _compute_reference_cka,
        )

        reps = torch.randn(1, 16, 32)  # seq_len=16
        ref_activations = {
            "transformer": torch.randn(24, 32),  # seq_len=24 (longer)
            "ssm": torch.randn(8, 32),  # seq_len=8 (shorter)
            "conv": torch.randn(16, 32),  # seq_len=16 (same)
        }
        result = _compute_reference_cka(reps, ref_activations=ref_activations)
        self.assertTrue(result["_succeeded"])

    def test_fingerprint_records_cka_source(self):
        """Fingerprint records cka_source provenance."""
        from research.eval.cka_references import reset_default_store
        from research.tests._cka_fingerprint_helpers import make_tiny_fingerprint

        reset_default_store()  # ensure clean state

        fp = make_tiny_fingerprint()
        self.assertIn(fp.cka_source, ("artifact", "none"))

    def test_fingerprint_reports_none_when_no_artifacts(self):
        """Fingerprint should report missing references, not heuristic stand-ins."""
        from research.tests._cka_fingerprint_helpers import (
            fingerprint_with_missing_references,
        )

        fp = fingerprint_with_missing_references()
        self.assertEqual(fp.cka_source, "none")

    def test_fingerprint_cka_provenance_fields_exist(self):
        """BehavioralFingerprint has cka_source and cka_artifact_version fields."""
        from research.tests._cka_fingerprint_helpers import (
            assert_fingerprint_provenance_fields,
        )

        assert_fingerprint_provenance_fields(self)

    # Each export runs the full reference-probe protocol (~3 min on CPU) —
    # slow lane only. Sole copy: the twin class in
    # test_reference_architectures.py was deduped 2026-06-12.
    @pytest.mark.slow
    def test_export_produces_loadable_artifacts(self):
        """Export tool produces artifacts that ReferenceCkaStore can load."""
        from research.tests._cka_fingerprint_helpers import (
            assert_export_produces_loadable_artifacts,
        )

        assert_export_produces_loadable_artifacts(self)

    @pytest.mark.slow
    def test_export_deterministic(self):
        """Same seed produces same probe_protocol_hash."""
        from research.tests._cka_fingerprint_helpers import assert_export_deterministic

        assert_export_deterministic(self)


def test_populate_cka_without_artifacts_keeps_no_reference_reason(monkeypatch):
    from research.eval.fingerprint import BehavioralFingerprint
    from research.eval.fingerprint_runtime import populate_cka

    class _MissingArtifactStore:
        def get_references(self):
            return None

        def get_reference_similarities(self):
            return None

        def get_metadata(self):
            return {
                "cka_source": "none",
                "cka_artifact_version": None,
                "cka_probe_protocol_hash": None,
                "cka_reference_quality": "none",
                "cka_similarity_path": "compute_reference_cka",
            }

    monkeypatch.setattr(
        "research.eval.fingerprint_runtime._get_default_store",
        lambda: _MissingArtifactStore(),
    )

    fp = BehavioralFingerprint()
    reps = torch.randn(8, 16)
    _, cka_all_zero = populate_cka(fp, reps, include=True)

    assert cka_all_zero is True
    assert fp.cka_source == "none"
    assert fp.novelty_valid_for_promotion is False
    assert fp.novelty_validity_reason == "no_reference_available"


if __name__ == "__main__":
    unittest.main()
