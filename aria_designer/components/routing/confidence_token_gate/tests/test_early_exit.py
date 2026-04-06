"""Contract tests for early_exit."""

import yaml
import torch
from pathlib import Path

from aria_designer.components.routing.confidence_token_gate.kernel_fallback import (
    ComponentHandler,
)


def test_manifest_valid():
    manifest_path = Path(__file__).parent.parent / "manifest.yaml"
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    assert manifest["id"] == "confidence_token_gate"
    assert manifest["version"] == "1.0.0"
    assert len(manifest["outputs"]) >= 1


def test_confidence_token_gate_zeroes_easy_tokens():
    handler = ComponentHandler()
    x = torch.full((2, 4, 8), 10.0)

    y = handler.forward({"x": x}, {"threshold": 0.5})["y"]

    assert torch.allclose(y, torch.zeros_like(x), atol=1e-6)
