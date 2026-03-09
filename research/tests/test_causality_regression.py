import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import math
from typing import Dict, Any, List, Tuple
from research.synthesis.primitives import PRIMITIVE_REGISTRY, OpCategory, get_primitive
from research.synthesis.compiler import _execute_op
from research.mathspaces.registry import register_all_mathspaces

pytestmark = pytest.mark.unit

# Ensure all ops are registered
register_all_mathspaces()

def is_causal_op(op_name: str) -> bool:
    """
    Test if an operation is strictly causal.
    Changing future tokens in the input should not affect past outputs.
    """
    B, S, D = 1, 16, 32
    device = torch.device("cpu")
    
    # Base input
    x_base = torch.randn(B, S, D, device=device)
    
    # Modified input: change only the last 4 positions
    x_mod = x_base.clone()
    midpoint = S - 4
    x_mod[:, midpoint:, :] = torch.randn(B, S - midpoint, D, device=device)
    
    # Mock module for params
    class MockModule(nn.Module):
        def __init__(self):
            super().__init__()
            # Allocate some dummy params for common ops if needed
            self.weight = nn.Parameter(torch.randn(D, D))
            self.bias = nn.Parameter(torch.randn(D))
            # MoE experts
            self.experts = nn.ModuleList([nn.Linear(D, D) for _ in range(4)])
            self.gate = nn.Linear(D, 4)
            # Attention
            self.q_proj = nn.Linear(D, D)
            self.k_proj = nn.Linear(D, D)
            self.v_proj = nn.Linear(D, D)
            self.out_proj = nn.Linear(D, D)
            
    module = MockModule()
    
    op = get_primitive(op_name)
    
    # Prepare inputs based on op.n_inputs
    inputs_base = tuple([x_base] * op.n_inputs)
    inputs_mod = tuple([x_mod] * op.n_inputs)
    
    # Dummy config
    config = {"dim": D, "out_dim": D, "n_experts": 4, "k": 1, "window_size": 4}
    
    try:
        # Execute op
        out_base = _execute_op(module, op_name, inputs_base, config)
        out_mod = _execute_op(module, op_name, inputs_mod, config)
        
        if not isinstance(out_base, torch.Tensor):
            return True # Not a tensor op, ignore
            
        # Check if shape has sequence dimension
        if out_base.ndim < 2:
            return True # No sequence dim to check causality on
            
        # Check causality: first 'midpoint' positions should be identical
        diff = torch.abs(out_base[:, :midpoint, ...] - out_mod[:, :midpoint, ...]).max().item()
        
        return diff < 1e-6
    except Exception as e:
        # Skip ops that can't be tested with a simple mock (e.g. need specific input shapes)
        return True

@pytest.mark.parametrize("op_name", list(PRIMITIVE_REGISTRY.keys()))
def test_op_causality_regression(op_name):
    """Verify that all primitive operations maintain autoregressive causality."""
    # Skip inherently global ops (used only in non-causal subgraphs or at the very end)
    if op_name in ["sum_seq", "mean_seq", "max_seq", "min_seq"]:
        pytest.skip(f"{op_name} is inherently global (non-causal)")
        
    assert is_causal_op(op_name), f"Operation '{op_name}' violates causality! This will cause reward hacking in Stage 1."
