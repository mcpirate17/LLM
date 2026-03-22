"""Contract tests for hyperbolic_norm."""

import yaml
from pathlib import Path


def test_manifest_valid():
    manifest_path = Path(__file__).parent.parent / "manifest.yaml"
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    assert manifest["id"] == "hyperbolic_norm"
    assert manifest["category"] == "math_space"
    assert manifest["version"] == "1.0.0"
    assert len(manifest["outputs"]) >= 1
    assert manifest["limits"]["deterministic"] is True
    assert "numerically_risky" in manifest["performance"]
