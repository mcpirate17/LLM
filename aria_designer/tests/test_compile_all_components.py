"""Comprehensive compile+forward test for every aria_designer component.

Tests the full pipeline: manifest → kernel_fallback.py → compile_workflow() →
WorkflowModule.forward() for all ~200 components. This catches import errors,
missing symbols, and runtime failures that unit-level handler tests miss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
import yaml

from aria_designer.runtime.compiler import compile_workflow

_ARIA_ROOT = Path(__file__).resolve().parent.parent
COMPONENTS_DIR = _ARIA_ROOT / "components"

# Categories that require specialized I/O setup (file handles, datasets)
_SKIP_CATEGORIES = {"data_io"}

# Components with known pre-existing issues (not caused by our changes)
_SKIP_COMPONENTS: set[str] = set()


def _discover_components() -> List[Tuple[str, Dict[str, Any]]]:
    """Discover all components and their manifests."""
    results = []
    for manifest_path in sorted(COMPONENTS_DIR.glob("*/*/manifest.yaml")):
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        if not manifest:
            continue
        category = manifest.get("category", "")
        cid = manifest.get("id", "")
        component_type = f"{category}/{cid}"
        results.append((component_type, manifest))
    return results


def _build_workflow(
    component_type: str,
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a minimal workflow graph: source(s) → component → output.

    Reads manifest inputs/outputs to wire ports correctly. Handles:
    - Single-input (x→y): graph_input → op → graph_output
    - Binary (a,b→y): two graph_inputs → op → graph_output
    - Index inputs: creates separate integer source
    - Multiple outputs: wires first tensor output to graph_output
    """
    inputs = manifest.get("inputs", [])
    outputs = manifest.get("outputs", [])

    # Build default config from params_schema defaults
    config: Dict[str, Any] = {}
    params = manifest.get("params_schema") or manifest.get("params", {})
    for k, v in (params or {}).items():
        if isinstance(v, dict) and v.get("default") is not None:
            config[k] = v["default"]
        elif isinstance(v, dict) and v.get("type") == "integer":
            config[k] = 256

    nodes = []
    edges = []
    edge_id = 0

    # Determine primary output port
    primary_out = "y"
    if outputs:
        primary_out = outputs[0]["name"]

    # Create source nodes + edges for each input
    for inp in inputs:
        port_name = inp["name"]
        dtype = inp.get("dtype", "tensor")
        src_id = f"src_{port_name}"

        nodes.append(
            {
                "id": src_id,
                "component_type": "graph_input",
                "params": {},
            }
        )
        edges.append(
            {
                "id": f"e{edge_id}",
                "source": src_id,
                "target": "op",
                "source_port": "out",
                "target_port": port_name,
            }
        )
        edge_id += 1

    # If no inputs defined, create a default source
    if not inputs:
        nodes.append(
            {
                "id": "src_x",
                "component_type": "graph_input",
                "params": {},
            }
        )
        edges.append(
            {
                "id": f"e{edge_id}",
                "source": "src_x",
                "target": "op",
                "source_port": "out",
                "target_port": "x",
            }
        )
        edge_id += 1

    # The component under test
    nodes.append(
        {
            "id": "op",
            "component_type": component_type,
            "params": config,
        }
    )

    # Output node
    nodes.append(
        {
            "id": "sink",
            "component_type": "graph_output",
            "params": {},
        }
    )
    edges.append(
        {
            "id": f"e{edge_id}",
            "source": "op",
            "target": "sink",
            "source_port": primary_out,
            "target_port": "in",
        }
    )

    return {
        "schema_version": "workflow_graph.v1",
        "workflow_id": f"test_{component_type.replace('/', '_')}",
        "name": f"Test {component_type}",
        "nodes": nodes,
        "edges": edges,
    }


def _make_input_tensor(
    dtype: str, shape: Tuple[int, ...] = (1, 16, 256)
) -> torch.Tensor:
    """Create a dummy input tensor appropriate for the given dtype."""
    if dtype == "index":
        return torch.randint(0, min(shape[-1], 16), shape[:2])
    elif dtype == "mask":
        return (torch.rand(*shape[:2]) > 0.5).float()
    elif dtype == "scalar":
        return torch.tensor(1.0)
    elif dtype == "dataset":
        return torch.randn(*shape)
    else:
        # tensor, complex_tensor, or unknown
        return torch.randn(*shape)


# ── Test discovery ──────────────────────────────────────────────────

_ALL_COMPONENTS = _discover_components()


def _component_ids():
    """Generate pytest parametrize IDs."""
    return [ct for ct, _ in _ALL_COMPONENTS]


@pytest.mark.parametrize(
    "component_type,manifest",
    _ALL_COMPONENTS,
    ids=_component_ids(),
)
def test_component_compiles_and_runs(component_type: str, manifest: Dict[str, Any]):
    """Test that a component can be compiled into a WorkflowModule and run forward."""
    category = manifest.get("category", "")
    cid = manifest.get("id", "")

    if category in _SKIP_CATEGORIES:
        pytest.skip(f"Skipping {category} category (requires specialized I/O)")

    if component_type in _SKIP_COMPONENTS:
        pytest.skip(f"Known pre-existing issue: {component_type}")

    # Check kernel_fallback.py exists
    fallback_path = COMPONENTS_DIR / category / cid / "kernel_fallback.py"
    if not fallback_path.exists():
        pytest.skip(f"No kernel_fallback.py for {component_type}")

    # Build workflow
    workflow = _build_workflow(component_type, manifest)

    # Compile
    model = compile_workflow(workflow, str(COMPONENTS_DIR))

    # Build inputs — one tensor per source node
    inputs_spec = manifest.get("inputs", [])
    source_inputs: Dict[str, torch.Tensor] = {}

    if inputs_spec:
        for inp in inputs_spec:
            src_id = f"src_{inp['name']}"
            dtype = inp.get("dtype", "tensor")
            source_inputs[src_id] = _make_input_tensor(dtype)
    else:
        source_inputs["src_x"] = torch.randn(1, 16, 256)

    # Forward pass
    with torch.no_grad():
        result = model(source_inputs)

    # Verify we got output
    assert result, f"{component_type}: forward() returned empty result"

    # Check output tensors are finite (where applicable)
    for key, val in result.items():
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(val).all(), (
                f"{component_type}: NaN/Inf in output '{key}'"
            )
