"""Contract tests for differentiable_sort."""
import yaml
from pathlib import Path


def test_manifest_valid():
    manifest_path = Path(__file__).parent.parent / "manifest.yaml"
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    assert manifest["id"] == "differentiable_sort"
    assert manifest["version"] == "1.0.0"
    assert len(manifest["outputs"]) >= 1
