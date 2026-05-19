"""Shared helpers for CKA fingerprint artifact tests."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


def make_tiny_fingerprint():
    import torch.nn as nn
    from research.eval.fingerprint import compute_fingerprint

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 32)
            self.linear = nn.Linear(32, 100)

        def forward(self, x):
            return self.linear(self.embed(x))

    return compute_fingerprint(
        TinyModel(),
        seq_len=8,
        model_dim=32,
        vocab_size=100,
        device="cpu",
        n_probes=4,
    )


def fingerprint_with_missing_references():
    from research.eval import cka_references
    from research.eval.cka_references import ReferenceCkaStore, reset_default_store

    reset_default_store()
    fake_store = ReferenceCkaStore(artifact_dir="/nonexistent/path")
    with patch.object(cka_references, "_default_store", fake_store):
        with patch.object(
            cka_references, "_default_lock", cka_references.threading.Lock()
        ):
            with patch(
                "research.eval.cka_references.get_default_store",
                return_value=fake_store,
            ):
                fp = make_tiny_fingerprint()
    reset_default_store()
    return fp


def assert_fingerprint_provenance_fields(testcase) -> None:
    from research.eval.fingerprint import BehavioralFingerprint

    fp = BehavioralFingerprint()
    testcase.assertEqual(fp.cka_source, "none")
    testcase.assertIsNone(fp.cka_artifact_version)
    data = fp.to_dict()
    testcase.assertIn("cka_source", data)
    testcase.assertIn("cka_artifact_version", data)


def assert_export_produces_loadable_artifacts(testcase) -> None:
    from research.eval.cka_references import ReferenceCkaStore
    from research.tools.export_cka_references import export_artifacts

    with TemporaryDirectory() as tmpdir:
        art_dir = str(Path(tmpdir) / "refs" / "v1")
        export_artifacts(
            output_dir=art_dir,
            seed=123,
            n_steps=10,
            device="cpu",
        )
        store = ReferenceCkaStore(artifact_dir=art_dir)
        refs = store.get_references()
        testcase.assertIsNotNone(refs)
        testcase.assertEqual(set(refs.keys()), {"transformer", "ssm", "conv"})
        testcase.assertTrue(store.is_artifact_backed)
        meta = store.get_metadata()
        testcase.assertEqual(meta["cka_source"], "artifact")
        testcase.assertEqual(meta["cka_artifact_version"], "v1")


def assert_export_deterministic(testcase) -> None:
    from research.tools.export_cka_references import export_artifacts

    with TemporaryDirectory() as tmpdir:
        d1 = str(Path(tmpdir) / "run1")
        d2 = str(Path(tmpdir) / "run2")
        export_artifacts(output_dir=d1, seed=99, n_steps=5, device="cpu")
        export_artifacts(output_dir=d2, seed=99, n_steps=5, device="cpu")

        with open(Path(d1) / "manifest.json", encoding="utf-8") as handle:
            m1 = json.load(handle)
        with open(Path(d2) / "manifest.json", encoding="utf-8") as handle:
            m2 = json.load(handle)
        testcase.assertEqual(m1["probe_protocol_hash"], m2["probe_protocol_hash"])
        testcase.assertEqual(m1["activation_shape"], m2["activation_shape"])
