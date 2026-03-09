
import pytest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from research.scientist.designer_utils import get_designer_components

pytestmark = pytest.mark.designer

def test_get_components():
    components = get_designer_components()
    print(f"Total components: {len(components)}")
    assert len(components) > 50
    
    # Check for Input
    inputs = [c for c in components if c["id"] == "io/input"]
    assert len(inputs) == 1
    assert inputs[0]["category"] == "io"
    
    # Check for ReLU
    relus = [c for c in components if c["id"].endswith("/relu")]
    assert len(relus) == 1
    assert relus[0]["category"] == "math"
    assert len(relus[0]["inputs"]) == 1
    
    # Check for Add
    adds = [c for c in components if c["id"].endswith("/add")]
    assert len(adds) == 1
    assert len(adds[0]["inputs"]) == 2
    
    # Check for Fused Linear GELU
    fuseds = [c for c in components if c["id"].endswith("/fused_linear_gelu")]
    assert len(fuseds) == 1
    assert "out_dim" in fuseds[0]["params_schema"]

if __name__ == "__main__":
    try:
        test_get_components()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
