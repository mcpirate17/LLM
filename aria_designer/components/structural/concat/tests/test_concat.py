"""Contract tests for concat."""

import torch
import yaml
from pathlib import Path


def test_manifest_valid():
    manifest_path = Path(__file__).parent.parent / "manifest.yaml"
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    assert manifest["id"] == "concat"
    assert manifest["version"] == "1.0.0"
    assert len(manifest["outputs"]) >= 1


def test_concat_includes_both_inputs():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "concat_fallback",
        Path(__file__).parent.parent / "kernel_fallback.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    handler = mod.ComponentHandler()
    a = torch.ones(2, 3, 4)
    b = torch.full((2, 3, 5), 2.0)
    result = handler.forward({"a": a, "b": b}, {})
    y = result["y"]
    assert y.shape == (2, 3, 9)
    assert (y[:, :, :4] == 1.0).all(), "output must contain values from input a"
    assert (y[:, :, 4:] == 2.0).all(), "output must contain values from input b"
