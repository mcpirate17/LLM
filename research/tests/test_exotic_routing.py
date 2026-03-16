import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from research.synthesis.compiler import compile_graph
from research.synthesis.graph import ComputationGraph

pytestmark = pytest.mark.unit

def test_mixed_recursion_gate():
    B, S, D = 2, 4, 8
    max_depth = 3
    x = torch.randn(B, S, D)
    # Scores: high entropy or similar, just random for depth selection
    scores = torch.randn(B, S, max_depth)
    
    g = ComputationGraph(D)
    in_node = g.add_input()
    score_node = g.add_input() # Dummy for scores
    out_node = g.add_op("mixed_recursion_gate", [in_node, score_node], {"max_depth": max_depth})
    g.set_output(out_node)
    
    model = compile_graph(g)
    
    # Let's test the op directly by finding it in the model
    op_module = None
    for m in model.modules():
        if hasattr(m, 'step_projs'):
            op_module = m
            break
    
    assert op_module is not None
    # OpModule.forward(*inputs, context=None)
    result = op_module(x, scores) 
    assert result.shape == (B, S, D)
    # Check if recursion happened (not identity)
    assert not torch.allclose(result, x)

def test_token_type_routing_flow():
    B, S, D = 2, 8, 16
    x = torch.randn(B, S, D)
    
    g = ComputationGraph(D)
    in_idx = g.add_input()
    # 1. Classify tokens
    class_idx = g.add_op("token_type_classifier", [in_idx], {"n_classes": 3})
    # 2. Compute entropy (routing signal)
    entropy_idx = g.add_op("entropy_score", [class_idx], {})
    # 3. Use signal for conditional compression
    out_idx = g.add_op("routing_conditioned_compression", [in_idx, entropy_idx], {})
    g.set_output(out_idx)
    
    model = compile_graph(g)
    out = model(x)
    
    assert out.shape == (B, S, D)
    assert not torch.allclose(out, x)

def test_progressive_compression():
    B, S, D = 2, 4, 16
    x = torch.randn(B, S, D)
    
    g = ComputationGraph(D)
    in_idx = g.add_input()
    out_idx = g.add_op("progressive_compression_gate", [in_idx], {})
    g.set_output(out_idx)
    
    model = compile_graph(g)
    out = model(x)
    
    assert out.shape == (B, S, D)
    assert not torch.allclose(out, x)

def test_compression_mixture_experts():
    B, S, D = 2, 4, 16
    x = torch.randn(B, S, D)
    
    g = ComputationGraph(D)
    in_idx = g.add_input()
    class_idx = g.add_op("token_type_classifier", [in_idx], {"n_classes": 2})
    out_idx = g.add_op("compression_mixture_experts", [in_idx, class_idx], {})
    g.set_output(out_idx)
    
    model = compile_graph(g)
    out = model(x)
    
    assert out.shape == (B, S, D)
    assert not torch.allclose(out, x)

def test_relu_gate_routing():
    B, S, D = 2, 4, 16
    x = torch.randn(B, S, D)
    
    g = ComputationGraph(D)
    in_idx = g.add_input()
    out_idx = g.add_op("relu_gate_routing", [in_idx], {"n_experts": 4})
    g.set_output(out_idx)
    
    model = compile_graph(g)
    out = model(x)
    
    assert out.shape == (B, S, D)
    # ReMoE is differentiable and non-identity
    assert not torch.allclose(out, x)

def test_ternary_projection():
    B, S, D = 2, 4, 16
    x = torch.randn(B, S, D)
    
    g = ComputationGraph(D)
    in_idx = g.add_input()
    out_idx = g.add_op("ternary_projection", [in_idx], {})
    g.set_output(out_idx)
    
    model = compile_graph(g)
    out = model(x)
    
    assert out.shape == (B, S, D)
    assert not torch.allclose(out, x)

if __name__ == "__main__":
    pytest.main([__file__])
