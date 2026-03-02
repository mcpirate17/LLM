"""Contract tests for binary_hash."""
import yaml
from pathlib import Path


def test_manifest_valid():
    manifest_path = Path(__file__).parent.parent / "manifest.yaml"
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    assert manifest["id"] == "binary_hash"
    assert manifest["version"] == "1.0.0"
    assert len(manifest["outputs"]) >= 1
