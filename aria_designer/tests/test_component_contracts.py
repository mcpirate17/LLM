"""Automated contract verification for all Aria Designer components."""
import os
import sys
import yaml
import torch
import pytest
from pathlib import Path

# Add api/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

def get_all_components():
    components_dir = Path(__file__).parent.parent / "components"
    components = []
    for root, dirs, files in os.walk(components_dir):
        if "manifest.yaml" in files:
            components.append(Path(root))
    return components

@pytest.mark.parametrize("comp_path", get_all_components())
def test_component_contract(comp_path):
    # 1. Load manifest
    manifest_path = comp_path / "manifest.yaml"
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    
    # 2. Check basics
    assert "id" in manifest
    assert "category" in manifest
    
    # Skip data_io for now as it involves complex file side effects
    if manifest["category"] == "data_io":
        pytest.skip("Skipping data_io contract tests")
    
    # 3. Try to load fallback kernel
    fallback_path = comp_path / "kernel_fallback.py"
    if not fallback_path.exists():
        pytest.skip(f"No fallback kernel for {manifest['id']}")
        
    import importlib.util
    spec = importlib.util.spec_from_file_location("handler", str(fallback_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    handler = module.ComponentHandler()
    
    # 4. Prepare dummy inputs
    inputs = {}
    for inp in manifest.get("inputs", []):
        name = inp["name"]
        dtype = inp["dtype"]
        if dtype in ("tensor", "complex_tensor"):
            # Default shape (B, S, D) = (1, 16, 256)
            inputs[name] = torch.randn(1, 16, 256)
        elif dtype == "index":
            # Random indices
            inputs[name] = torch.randint(0, 16, (1, 16))
        elif dtype == "mask":
            inputs[name] = (torch.rand(1, 16) > 0.5).float()
        elif dtype == "scalar":
            inputs[name] = torch.tensor(1.0)
        elif dtype == "dataset":
            pytest.skip("Dataset inputs require specialized test setup")
        else:
            # Fallback for unknown types
            inputs[name] = torch.randn(1, 16, 256)
            
    # 5. Run forward pass
    config = {}
    params = manifest.get("params_schema") or manifest.get("params", {})
    for k, v in params.items():
        if v.get("default") is not None:
            config[k] = v["default"]
        elif v["type"] == "integer":
            config[k] = 256
            
    # Create dummy files for IO components to avoid FileNotFoundError
    if manifest["id"] in ("binary_file_reader", "file_loader", "csv_reader"):
        dummy_file = Path("data.bin") if manifest["id"] == "binary_file_reader" else Path("data.csv")
        if not dummy_file.exists():
            dummy_file.write_text("0,0,0\n0,0,0")
            
    try:
        outputs = handler.forward(inputs, config)
    except Exception as e:
        pytest.fail(f"Component {manifest['id']} failed forward pass: {e}")
    finally:
        # Cleanup dummy files
        for f in ["data.bin", "data.csv"]:
            if Path(f).exists():
                try: os.remove(f)
                except: pass
        
    # 6. Verify outputs
    for out in manifest.get("outputs", []):
        assert out["name"] in outputs, f"Missing output '{out['name']}'"
        val = outputs[out["name"]]
        if out["dtype"] == "tensor" and isinstance(val, torch.Tensor):
            assert torch.isfinite(val).all(), f"NaN/Inf detected in output of {manifest['id']}"
