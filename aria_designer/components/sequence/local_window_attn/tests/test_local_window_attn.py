"""Contract tests for local_window_attn."""

import yaml
from pathlib import Path


def test_manifest_valid():
    manifest_path = Path(__file__).parent.parent / "manifest.yaml"
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    assert manifest["id"] == "local_window_attn"
    assert manifest["version"] == "1.0.0"
    assert len(manifest["outputs"]) >= 1
