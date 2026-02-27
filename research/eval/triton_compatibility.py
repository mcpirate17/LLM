"""
Triton/Flash Compatibility: Checks if a graph has native high-performance kernel coverage.
Flags architectures relying on slow fallbacks.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set
from research.synthesis.graph import ComputationGraph

@dataclass
class TritonCompatibilityResult:
    """Detailed Triton/Flash compatibility report."""
    coverage_score: float = 0.0  # 0-1 fraction of ops with Triton coverage
    native_op_count: int = 0
    fallback_op_count: int = 0
    slow_fallbacks: List[str] = field(default_factory=list)
    triton_capable_ops: List[str] = field(default_factory=list)
    has_flash_attention: bool = False
    is_fully_optimized: bool = False

# Ops with known Triton/Flash coverage in the ecosystem
TRITON_NATIVE_OPS = {
    # From synthesis/kernels.py
    "linear_proj",  # can use block_sparse or fused
    "relu",         # often fused
    "gelu",         # often fused
    "rmsnorm",      # triton_rmsnorm exists
    "layernorm",    # usually has triton version
    "local_window_attn", # triton_local_attn exists
    
    # From aria_core.gpu (migrated from LA3/HYDRA)
    "lightning_attn3",
    "lightning_attn3_no_decay",
    "rope",
    "swiglu",
    "fused_qk_norm",
    
    # Standard high-perf ops
    "softmax_attention", # FlashAttention
    "selective_scan",    # Mamba/Triton
}

# Ops known to be slow fallbacks
SLOW_FALLBACK_OPS = {
    "sort_seq",
    "fixed_point_iter",
    "integral_kernel",
    "basis_expansion",
    "moe_topk",  # when not using specialized triton
}

def check_triton_compatibility(graph: ComputationGraph) -> TritonCompatibilityResult:
    """Analyze a graph for Triton/Flash hardware acceleration coverage."""
    total_ops = 0
    native_ops = 0
    fallback_ops = 0
    slow_fallbacks = []
    triton_capable = []
    has_flash = False
    
    for node in graph.nodes.values():
        if node.is_input:
            continue
            
        total_ops += 1
        op_name = node.op_name
        
        if op_name in TRITON_NATIVE_OPS:
            native_ops += 1
            triton_capable.append(op_name)
            if op_name == "softmax_attention":
                has_flash = True
        else:
            fallback_ops += 1
            if op_name in SLOW_FALLBACK_OPS:
                slow_fallbacks.append(op_name)
                
    coverage = native_ops / max(total_ops, 1)
    
    return TritonCompatibilityResult(
        coverage_score=coverage,
        native_op_count=native_ops,
        fallback_op_count=fallback_ops,
        slow_fallbacks=sorted(list(set(slow_fallbacks))),
        triton_capable_ops=sorted(list(set(triton_capable))),
        has_flash_attention=has_flash,
        is_fully_optimized=(coverage == 1.0 and not slow_fallbacks)
    )
