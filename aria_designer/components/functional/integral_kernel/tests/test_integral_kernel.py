"""Contract tests for integral_kernel."""

import yaml
from pathlib import Path


def test_manifest_valid():
    manifest_path = Path(__file__).parent.parent / "manifest.yaml"
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    assert manifest["id"] == "integral_kernel"
    assert manifest["version"] == "1.0.0"
    assert len(manifest["outputs"]) >= 1
