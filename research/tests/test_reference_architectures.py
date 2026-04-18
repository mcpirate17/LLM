"""Tests for reference architecture builders and registration pipeline."""

import pytest
import torch

from research.synthesis.reference_architectures import (
    REFERENCE_ARCHITECTURES,
    build_reference,
    list_references,
)
from research.synthesis.compiler import compile_model
from research.synthesis.graph import ComputationGraph

pytestmark = [pytest.mark.unit, pytest.mark.slow]


class TestReferenceArchitectureBuilders:
    """Test that each reference architecture builds valid graphs."""

    @pytest.fixture(params=list(REFERENCE_ARCHITECTURES.keys()))
    def arch_key(self, request):
        return request.param

    def test_builds_without_error(self, arch_key):
        g = build_reference(arch_key, d_model=64)
        assert isinstance(g, ComputationGraph)
        assert g.n_ops() > 0

    def test_has_valid_output(self, arch_key):
        g = build_reference(arch_key, d_model=64)
        assert g.output_node is not None
        assert g.output_node.output_shape.dim == 64
        assert g.output_node.output_shape.is_standard

    def test_has_gradient_path(self, arch_key):
        g = build_reference(arch_key, d_model=64)
        assert g.has_gradient_path()

    def test_has_metadata(self, arch_key):
        g = build_reference(arch_key, d_model=64)
        assert "architecture" in g.metadata
        assert "reference_name" in g.metadata

    def test_unique_fingerprints(self):
        fps = set()
        for key in REFERENCE_ARCHITECTURES:
            g = build_reference(key, d_model=64)
            fp = g.fingerprint()
            assert fp not in fps, f"Duplicate fingerprint for {key}"
            fps.add(fp)

    def test_different_d_models(self, arch_key):
        for d in [64, 128, 256]:
            g = build_reference(arch_key, d_model=d)
            assert g.output_node.output_shape.dim == d


class TestReferenceCompilation:
    """Test that reference architectures compile to valid torch models."""

    @pytest.fixture(params=list(REFERENCE_ARCHITECTURES.keys()))
    def arch_key(self, request):
        return request.param

    def test_compiles_to_model(self, arch_key):
        g = build_reference(arch_key, d_model=64)
        model = compile_model([g], vocab_size=1000, max_seq_len=32)
        assert isinstance(model, torch.nn.Module)

    def test_forward_pass(self, arch_key):
        g = build_reference(arch_key, d_model=64)
        model = compile_model([g], vocab_size=1000, max_seq_len=32)
        input_ids = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            output = model(input_ids)
        assert output.shape[0] == 2  # batch
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_backward_pass(self, arch_key):
        g = build_reference(arch_key, d_model=64)
        model = compile_model([g], vocab_size=1000, max_seq_len=32)
        input_ids = torch.randint(0, 1000, (2, 16))
        output = model(input_ids)
        loss = output.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        assert has_grad, f"{arch_key} has no gradients flowing"

    def test_multi_layer_model(self, arch_key):
        layers = [build_reference(arch_key, d_model=64) for _ in range(3)]
        model = compile_model(layers, vocab_size=1000, max_seq_len=32)
        input_ids = torch.randint(0, 1000, (2, 16))
        with torch.no_grad():
            output = model(input_ids)
        assert not torch.isnan(output).any()


class TestListReferences:
    def test_lists_all(self):
        refs = list_references()
        assert len(refs) == len(REFERENCE_ARCHITECTURES)
        for ref in refs:
            assert "key" in ref
            assert "name" in ref
            assert "paradigm" in ref

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            build_reference("nonexistent")


import json
import os
import tempfile
import unittest
from pathlib import Path


class TestCkaReferenceArtifacts(unittest.TestCase):
    """Tests for CKA reference artifact loader/validator/cache (#28/#43 Phase A)."""

    def _make_artifact_dir(self, tmpdir, manifest_override=None, families=None):
        """Helper: create a valid artifact directory with manifest and .pt files."""
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

    def test_reference_similarity_cache_reuses_precomputed_matrices(self):
        """ReferenceCkaStore caches prepared similarity matrices across calls."""
        from research.eval.cka_references import ReferenceCkaStore

        with tempfile.TemporaryDirectory() as d:
            art_dir = self._make_artifact_dir(d)
            store = ReferenceCkaStore(artifact_dir=art_dir)
            first = store.get_reference_similarities()
            second = store.get_reference_similarities()
            self.assertIsNotNone(first)
            self.assertIs(first, second)
            self.assertEqual(set(first.keys()), {"transformer", "ssm", "conv"})
            for sim in first.values():
                self.assertEqual(tuple(sim.shape), (16, 16))

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
        from research.eval.fingerprint_cka import compute_reference_cka as _compute_reference_cka

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
        from research.eval.fingerprint_cka import compute_reference_cka as _compute_reference_cka

        reps = torch.randn(1, 16, 32)
        result = _compute_reference_cka(reps, ref_activations=None)
        self.assertFalse(result["_succeeded"])
        for family in ("transformer", "ssm", "conv"):
            self.assertEqual(result[family], 0.0)

    def test_compute_reference_cka_seq_len_mismatch(self):
        """Artifact CKA handles different seq lengths between candidate and reference."""
        from research.eval.fingerprint_cka import compute_reference_cka as _compute_reference_cka

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

        reset_default_store()  # ensure clean state

        fp = self._make_fingerprint()
        self.assertIn(fp.cka_source, ("artifact", "none"))

    def test_fingerprint_reports_none_when_no_artifacts(self):
        """Fingerprint should report missing references, not heuristic stand-ins."""
        from unittest.mock import patch
        from research.eval import cka_references
        from research.eval.cka_references import ReferenceCkaStore, reset_default_store

        reset_default_store()
        # Force a store pointing to nonexistent dir
        fake_store = ReferenceCkaStore(artifact_dir="/nonexistent/path")
        with patch.object(cka_references, "_default_store", fake_store):
            with patch.object(
                cka_references, "_default_lock", cka_references.threading.Lock()
            ):
                # Override get_default_store to return our fake store
                with patch(
                    "research.eval.cka_references.get_default_store",
                    return_value=fake_store,
                ):
                    fp = self._make_fingerprint()
        self.assertEqual(fp.cka_source, "none")
        reset_default_store()

    def _make_fingerprint(self):
        """Helper: compute fingerprint on a tiny model."""
        import torch.nn as nn
        from research.eval.fingerprint import compute_fingerprint

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.linear = nn.Linear(32, 100)

            def forward(self, x):
                return self.linear(self.embed(x))

        model = TinyModel()
        return compute_fingerprint(
            model, seq_len=8, model_dim=32, vocab_size=100, device="cpu", n_probes=4
        )

    def test_fingerprint_cka_provenance_fields_exist(self):
        """BehavioralFingerprint has cka_source and cka_artifact_version fields."""
        from research.eval.fingerprint import BehavioralFingerprint

        fp = BehavioralFingerprint()
        self.assertEqual(fp.cka_source, "none")
        self.assertIsNone(fp.cka_artifact_version)
        d = fp.to_dict()
        self.assertIn("cka_source", d)
        self.assertIn("cka_artifact_version", d)

    def test_export_produces_loadable_artifacts(self):
        """Export tool produces artifacts that ReferenceCkaStore can load."""
        from research.tools.export_cka_references import export_artifacts
        from research.eval.cka_references import ReferenceCkaStore

        with tempfile.TemporaryDirectory() as d:
            art_dir = str(Path(d) / "refs" / "v1")
            export_artifacts(
                output_dir=art_dir,
                seed=123,
                n_steps=10,
                device="cpu",
            )
            store = ReferenceCkaStore(artifact_dir=art_dir)
            refs = store.get_references()
            self.assertIsNotNone(refs)
            self.assertEqual(set(refs.keys()), {"transformer", "ssm", "conv"})
            self.assertTrue(store.is_artifact_backed)
            meta = store.get_metadata()
            self.assertEqual(meta["cka_source"], "artifact")
            self.assertEqual(meta["cka_artifact_version"], "v1")

    def test_export_deterministic(self):
        """Same seed produces same probe_protocol_hash."""
        from research.tools.export_cka_references import export_artifacts

        with tempfile.TemporaryDirectory() as d:
            d1 = str(Path(d) / "run1")
            d2 = str(Path(d) / "run2")
            export_artifacts(output_dir=d1, seed=99, n_steps=5, device="cpu")
            export_artifacts(output_dir=d2, seed=99, n_steps=5, device="cpu")

            with open(Path(d1) / "manifest.json") as f:
                m1 = json.load(f)
            with open(Path(d2) / "manifest.json") as f:
                m2 = json.load(f)
            self.assertEqual(m1["probe_protocol_hash"], m2["probe_protocol_hash"])
            self.assertEqual(m1["activation_shape"], m2["activation_shape"])
