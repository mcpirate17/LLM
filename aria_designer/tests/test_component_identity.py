from __future__ import annotations
import pytest
from aria_designer.api.app.component_identity import (
    canonicalize_component_id,
    canonicalize_workflow,
    discover_concepts,
)

def test_canonicalize_component_id():
    # Aliases
    assert canonicalize_component_id("difficulty") == "routing/difficulty_scorer"
    assert canonicalize_component_id("scorer") == "routing/difficulty_scorer"
    assert canonicalize_component_id("attention") == "mixing/softmax_attention"
    assert canonicalize_component_id("norm") == "linear_algebra/rmsnorm"
    assert canonicalize_component_id("linear") == "linear_algebra/linear_proj"
    
    # Leaf names
    assert canonicalize_component_id("relu") == "math/relu"
    assert canonicalize_component_id("rmsnorm_pre") == "normalization/rmsnorm_pre"
    
    # Already canonical
    assert canonicalize_component_id("io/input") == "io/input"
    assert canonicalize_component_id("math/relu") == "math/relu"
    
    # Mixed/Legacy but has correct leaf
    assert canonicalize_component_id("normalization/rmsnorm") == "linear_algebra/rmsnorm"
    
    # Unknown
    assert canonicalize_component_id("unknown/id") == "unknown/id"
    assert canonicalize_component_id("completely_new_op") == "completely_new_op"

def test_canonicalize_workflow():
    wf = {
        "nodes": [
            {"id": "n1", "component_type": "input"},
            {"id": "n2", "component_type": "difficulty_scorer"},
            {"id": "n3", "component_type": "linear_proj"},
            {"id": "n4", "component_type": "output_head"}
        ]
    }
    canonicalize_workflow(wf)
    
    assert wf["nodes"][0]["component_type"] == "io/input"
    assert wf["nodes"][1]["component_type"] == "routing/difficulty_scorer"
    assert wf["nodes"][2]["component_type"] == "linear_algebra/linear_proj"
    assert wf["nodes"][3]["component_type"] == "io/output_head"

def test_discover_concepts():
    message = "I want a transformer with ultrametric attention and a difficulty scorer"
    found = discover_concepts(message)
    
    concepts = {f["concept"] for f in found}
    assert "transformer" in concepts or "attention" in concepts
    assert "ultrametric" in concepts
    assert "difficulty" in concepts or "scorer" in concepts
    
    types = {f["component_type"] for f in found}
    assert "mixing/softmax_attention" in types
    assert "math_space/ultrametric_attention" in types
    assert "routing/difficulty_scorer" in types
