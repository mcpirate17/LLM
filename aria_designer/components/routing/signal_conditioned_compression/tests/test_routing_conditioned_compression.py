"""Contract tests for routing_conditioned_compression."""

import yaml
from pathlib import Path


def test_manifest_valid():
    manifest_path = Path(__file__).parent.parent / "manifest.yaml"
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    assert manifest["id"] == "signal_conditioned_compression"
    assert manifest["version"] == "1.0.0"
    assert len(manifest["outputs"]) >= 1
