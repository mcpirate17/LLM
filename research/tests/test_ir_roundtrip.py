
import torch
import torch.nn as nn
import numpy as np
import pytest
from research.synthesis.graph import ComputationGraph
from research.synthesis.compiler import compile_model
from research.synthesis.serializer import graph_to_json, graph_from_json

def test_ir_roundtrip_equivalence():
    """Test that Graph -> IR (Compiled) produces same output as Graph -> Standard (Compiled)."""
    torch.manual_seed(42)
    dim = 64
    vocab_size = 1000
    seq_len = 32
    batch_size = 2
    
    # Create a non-trivial graph
    g = ComputationGraph(dim)
    i1 = g.add_input()
    
    # Add some ops
    n1 = g.add_op("linear_proj", [i1], config={"out_dim": dim})
    n2 = g.add_op("relu", [n1])
    n3 = g.add_op("rmsnorm", [n2])
    
    # Add a residual path
    n4 = g.add_op("add", [n3, i1])
    
    # Add a parameterized op with different dimensions
    n5 = g.add_op("linear_proj", [n4], config={"out_dim": dim})
    
    g.set_output(n5)
    
    # Compile with standard backend
    model_std = compile_model([g], vocab_size=vocab_size, max_seq_len=seq_len, use_ir=False)
    model_std.eval()
    
    # Compile with IR backend
    model_ir = compile_model([g], vocab_size=vocab_size, max_seq_len=seq_len, use_ir=True)
    model_ir.eval()
    
    # Ensure parameters are identical (copy from std to ir)
    # This is necessary because they are initialized randomly
    with torch.no_grad():
        for p_std, p_ir in zip(model_std.parameters(), model_ir.parameters()):
            p_ir.copy_(p_std)
            
    # Run forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        out_std = model_std(input_ids)
        out_ir = model_ir(input_ids)
        
    # Check equivalence
    diff = torch.abs(out_std - out_ir).max().item()
    print(f"Max difference: {diff}")
    
    assert diff < 1e-5, f"IR output differs from standard output: {diff}"

def test_ir_serialization_roundtrip():
    """Test that Graph -> JSON -> Graph -> IR produces correct results."""
    dim = 32
    g = ComputationGraph(dim)
    i1 = g.add_input()
    n1 = g.add_op("gelu", [i1])
    n2 = g.add_op("linear_proj", [n1], config={"out_dim": dim})
    g.set_output(n2)
    
    js = graph_to_json(g)
    g2 = graph_from_json(js)
    
    model = compile_model([g2], use_ir=True)
    assert model is not None
    
    input_ids = torch.randint(0, 32000, (1, 16))
    out = model(input_ids)
    assert out.shape == (1, 16, 32000)

if __name__ == "__main__":
    test_ir_roundtrip_equivalence()
    test_ir_serialization_roundtrip()
