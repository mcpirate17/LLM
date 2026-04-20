"""Structured capability-first template manifest.

This is the registration surface for role-slot and retrieval-v2 templates.
`templates.py` consumes these tables directly instead of repeating capability-
first template names across import lists, registry entries, and weight maps.
"""

from __future__ import annotations

from ._templates_role_slot_v2 import (
    tpl_conv_residual_retrieval_v2,
    tpl_latent_attn_retrieval_v2,
    tpl_state_space_retrieval_v2,
)
from ._templates_role_slots import (
    tpl_sparse_relation_graph_block,
    tpl_token_program_interpreter_block,
    tpl_typed_slot_memory_block,
)


ROLE_SLOT_TEMPLATE_REGISTRY = {
    "typed_slot_memory_block": tpl_typed_slot_memory_block,
    "sparse_relation_graph_block": tpl_sparse_relation_graph_block,
    "token_program_interpreter_block": tpl_token_program_interpreter_block,
    "conv_residual_retrieval_v2": tpl_conv_residual_retrieval_v2,
    "state_space_retrieval_v2": tpl_state_space_retrieval_v2,
    "latent_attn_retrieval_v2": tpl_latent_attn_retrieval_v2,
}


ROLE_SLOT_TEMPLATE_DEFAULT_WEIGHTS = {
    "typed_slot_memory_block": 4.5,  # Typed memory write/read over compact trunk
    "sparse_relation_graph_block": 4.0,  # Sparse relation proposal + algebraic retrieval
    "token_program_interpreter_block": 4.25,  # Routed token programs with explicit memory path
    # Role-slot v2: proven trunks + retrieval sidecar (2026-04-16)
    "conv_residual_retrieval_v2": 4.25,  # conv trunk + matmul/gather_topk sidecar
    "state_space_retrieval_v2": 4.25,  # SSM trunk + matmul/gather_topk sidecar
    "latent_attn_retrieval_v2": 4.0,  # latent_attn trunk + complementary retrieval lane
}
