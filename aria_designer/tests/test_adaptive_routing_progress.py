"""Progress tests for adaptive tri-lane routing bring-up."""
from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import yaml
from fastapi.testclient import TestClient

# Add api/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))


ADAPTIVE_MANIFESTS = [
    Path("components/routing/difficulty_scorer/manifest.yaml"),
    Path("components/routing/lane_router/manifest.yaml"),
    Path("components/structural/conditional_dispatch/manifest.yaml"),
    Path("components/structural/conditional_gather/manifest.yaml"),
    Path("components/functional/load_balance_loss/manifest.yaml"),
    Path("components/control_flow/training_phase_gate/manifest.yaml"),
]

ADAPTIVE_EXAMPLES = [
    Path("ui/public/examples/adaptive_trilane_v1.json"),
    Path("ui/public/examples/adaptive_trilane_v2.json"),
    Path("ui/public/examples/adaptive_trilane_v3.json"),
]


@pytest.fixture(scope="module")
def client():
    """Create test client with temporary database."""
    from app import database as db
    from app.main import app

    with tempfile.TemporaryDirectory() as tmpdir:
        db.init_db(Path(tmpdir) / "test.db")
        from app.loader import scan_and_load

        count = scan_and_load()
        assert count > 0, "No components loaded"

        with TestClient(app) as c:
            yield c


def test_adaptive_manifests_have_python_fallback_paths():
    root = Path(__file__).resolve().parents[1]
    for rel_manifest in ADAPTIVE_MANIFESTS:
        manifest_path = root / rel_manifest
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = yaml.safe_load(fh)

        impl = manifest.get("implementation") or {}
        assert impl.get("python") == "kernel_fallback.py", f"Missing fallback path in {manifest_path}"
        assert (manifest_path.parent / "kernel_fallback.py").exists(), f"Missing kernel_fallback.py for {manifest_path}"


@pytest.mark.parametrize("manifest_relpath", ADAPTIVE_MANIFESTS)
def test_adaptive_component_handlers_forward_contract(manifest_relpath: Path):
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / manifest_relpath
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)

    fallback_path = manifest_path.parent / "kernel_fallback.py"
    spec = importlib.util.spec_from_file_location(f"handler_{manifest['id']}", str(fallback_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    handler = module.ComponentHandler()
    config = {}
    for key, schema in (manifest.get("params") or {}).items():
        if schema.get("default") is not None:
            config[key] = schema["default"]

    inputs = {}
    for inp in manifest.get("inputs", []):
        name = inp["name"]
        if name == "a":
            inputs[name] = torch.randn(1, 8, 64)
        elif name == "b":
            inputs[name] = torch.randn(1, 8, 64)
        else:
            inputs[name] = torch.randn(1, 8, 64)

    outputs = handler.forward(inputs, config)
    assert isinstance(outputs, dict)
    assert "y" in outputs
    assert torch.isfinite(outputs["y"]).all()


@pytest.mark.parametrize("example_path", ADAPTIVE_EXAMPLES)
def test_adaptive_trilane_examples_compile(client, example_path: Path):
    root = Path(__file__).resolve().parents[1]
    workflow_path = root / example_path
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))

    response = client.post("/api/v1/workflows/compile", json={"workflow": workflow})
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("compiled") is True, f"Compile failed for {example_path}: {payload.get('error')}"
    assert payload.get("node_count", 0) > 0
