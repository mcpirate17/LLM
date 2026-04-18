"""Role-slot templates for capability-first architectural search.

These templates intentionally avoid standard attention and state-space mixer
families as their primary mechanism. They combine sparse control, typed token
signals, algebraic retrieval, and compact local trunks.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph

from ._template_helpers import (
    MOTIF_CLASS_NORM,
    MotifWeights,
    _instantiate_motif,
    _pick_compatible_motif,
    template_add_op as _add,
    template_add_residual as _residual,
)
from ._template_role_slots import record_role_slot_binding


def _typed_entropy_gate(
    graph: ComputationGraph,
    node_id: int,
    *,
    context: str,
    classes: int = 4,
) -> int:
    typed = _add(
        graph,
        "token_type_classifier",
        [node_id],
        {"n_classes": classes},
        context=f"{context}.type",
    )
    entropy = _add(graph, "entropy_score", [typed], context=f"{context}.entropy")
    gated = _add(graph, "mul", [node_id, entropy], context=f"{context}.mix")
    return _add(graph, "sigmoid", [gated], context=f"{context}.sigmoid")


def tpl_typed_slot_memory_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Compression trunk with typed slot write/read sidecar and gated merge."""
    graph.metadata["_skip_global_decorators"] = True
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    trunk = _add(graph, "conv1d_seq", [normed], context="typed_slot_memory_block.trunk")
    record_role_slot_binding(
        graph,
        role_name="trunk_compression",
        selected_name="conv_local_trunk",
        input_node_id=normed,
    )

    controller_gate = _typed_entropy_gate(
        graph, normed, context="typed_slot_memory_block.controller"
    )
    record_role_slot_binding(
        graph,
        role_name="controller",
        selected_name="typed_entropy_controller",
        input_node_id=normed,
    )

    write_seed = _add(
        graph,
        "linear_proj",
        [normed],
        {"out_dim": graph.model_dim},
        context="typed_slot_memory_block.write_seed",
    )
    write_seed_norm = _add(
        graph,
        "rmsnorm",
        [write_seed],
        context="typed_slot_memory_block.write_seed_norm",
    )
    record_role_slot_binding(
        graph,
        role_name="binding_write",
        selected_name="typed_slot_write",
        input_node_id=normed,
    )
    slot_bank = _add(
        graph,
        "adjacent_token_merge",
        [write_seed_norm],
        context="typed_slot_memory_block.slot_bank",
    )
    slot_bank_skip = _add(
        graph,
        "add",
        [write_seed_norm, slot_bank],
        context="typed_slot_memory_block.slot_bank_skip",
    )
    slot_bank = _add(
        graph,
        "linear_proj",
        [slot_bank],
        {"out_dim": graph.model_dim},
        context="typed_slot_memory_block.slot_bank_post",
    )
    gated_bank = _add(
        graph,
        "mul",
        [slot_bank, controller_gate],
        context="typed_slot_memory_block.gated_bank",
    )

    query = _add(
        graph,
        "linear_proj",
        [trunk],
        {"out_dim": graph.model_dim},
        context="typed_slot_memory_block.query",
    )
    record_role_slot_binding(
        graph,
        role_name="binding_read",
        selected_name="typed_slot_read",
        input_node_id=trunk,
    )
    scores = _add(
        graph,
        "matmul",
        [query, gated_bank],
        context="typed_slot_memory_block.scores",
    )
    scores = _add(
        graph,
        "linear_proj",
        [scores],
        {"out_dim": graph.model_dim},
        context="typed_slot_memory_block.score_proj",
    )
    retrieved = _add(
        graph,
        "gather_topk",
        [gated_bank, scores],
        {"k": 4},
        context="typed_slot_memory_block.retrieve",
    )
    retrieved = _add(
        graph,
        "mul",
        [retrieved, controller_gate],
        context="typed_slot_memory_block.retrieve_gate",
    )
    retrieved = _add(
        graph,
        "add",
        [slot_bank_skip, retrieved],
        context="typed_slot_memory_block.retrieve_skip",
    )
    record_role_slot_binding(
        graph,
        role_name="merge_policy",
        selected_name="gated_add_merge",
        input_node_id=trunk,
    )
    merged = _add(
        graph,
        "add",
        [trunk, retrieved],
        context="typed_slot_memory_block.merge",
    )
    record_role_slot_binding(
        graph,
        role_name="local_mixing",
        selected_name="identity_local_mix",
        input_node_id=merged,
    )
    stabilized = _add(
        graph, "rmsnorm", [merged], context="typed_slot_memory_block.stabilize"
    )
    record_role_slot_binding(
        graph,
        role_name="stabilizer",
        selected_name="rmsnorm_stabilizer",
        input_node_id=merged,
    )
    return _residual(
        graph, input_id, stabilized, context="typed_slot_memory_block.output"
    )


def tpl_sparse_relation_graph_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Sparse relation proposal over a compact trunk with algebraic message passing."""
    graph.metadata["_skip_global_decorators"] = True
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    trunk = _add(
        graph, "conv1d_seq", [normed], context="sparse_relation_graph_block.trunk"
    )
    record_role_slot_binding(
        graph,
        role_name="trunk_compression",
        selected_name="conv_local_trunk",
        input_node_id=normed,
    )
    anchor = _add(
        graph,
        "linear_proj",
        [trunk],
        {"out_dim": graph.model_dim},
        context="sparse_relation_graph_block.anchor",
    )
    relation_seed = _add(
        graph,
        "route_topk",
        [anchor],
        context="sparse_relation_graph_block.route_topk",
    )
    record_role_slot_binding(
        graph,
        role_name="controller",
        selected_name="route_topk_relation_controller",
        input_node_id=anchor,
    )
    relation_basis = _add(
        graph,
        "linear_proj",
        [relation_seed],
        {"out_dim": graph.model_dim},
        context="sparse_relation_graph_block.relation_basis",
    )
    relation_scores = _add(
        graph,
        "matmul",
        [relation_basis, relation_basis],
        context="sparse_relation_graph_block.relation_scores",
    )
    relation_scores = _add(
        graph,
        "linear_proj",
        [relation_scores],
        {"out_dim": graph.model_dim},
        context="sparse_relation_graph_block.score_proj",
    )
    record_role_slot_binding(
        graph,
        role_name="global_retrieval",
        selected_name="cosine_topk_relations",
        input_node_id=relation_basis,
    )
    sparse_edges = _add(
        graph,
        "gather_topk",
        [relation_basis, relation_scores],
        {"k": 4},
        context="sparse_relation_graph_block.sparse_edges",
    )
    messages = _add(
        graph,
        "linear_proj",
        [sparse_edges],
        {"out_dim": graph.model_dim},
        context="sparse_relation_graph_block.message_proj",
    )
    merge_gate = _typed_entropy_gate(
        graph, trunk, context="sparse_relation_graph_block.merge_gate"
    )
    gated_messages = _add(
        graph,
        "mul",
        [messages, merge_gate],
        context="sparse_relation_graph_block.gated_messages",
    )
    record_role_slot_binding(
        graph,
        role_name="merge_policy",
        selected_name="competitive_relation_merge",
        input_node_id=trunk,
    )
    merged = _add(
        graph,
        "add",
        [trunk, gated_messages],
        context="sparse_relation_graph_block.merge",
    )
    record_role_slot_binding(
        graph,
        role_name="local_mixing",
        selected_name="identity_local_mix",
        input_node_id=merged,
    )
    stabilized = _add(
        graph, "rmsnorm", [merged], context="sparse_relation_graph_block.stabilize"
    )
    record_role_slot_binding(
        graph,
        role_name="stabilizer",
        selected_name="rmsnorm_stabilizer",
        input_node_id=merged,
    )
    return _residual(
        graph, input_id, stabilized, context="sparse_relation_graph_block.output"
    )


def tpl_token_program_interpreter_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Token-program controller that routes store/read paths through a compact memory."""
    graph.metadata["_skip_global_decorators"] = True
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    program_router = _add(
        graph,
        "n_way_sparse_router",
        [normed],
        {"n_ways": 4, "top_k": 2},
        context="token_program_interpreter_block.router",
    )
    program_proj = _add(
        graph,
        "linear_proj",
        [program_router],
        {"out_dim": graph.model_dim},
        context="token_program_interpreter_block.program_proj",
    )
    program_norm = _add(
        graph,
        "rmsnorm",
        [program_proj],
        context="token_program_interpreter_block.program_norm",
    )
    program_gate = _add(
        graph,
        "sigmoid",
        [program_proj],
        context="token_program_interpreter_block.program_gate",
    )
    record_role_slot_binding(
        graph,
        role_name="controller",
        selected_name="token_program_router",
        input_node_id=normed,
    )

    trunk = _add(
        graph, "conv1d_seq", [normed], context="token_program_interpreter_block.trunk"
    )
    record_role_slot_binding(
        graph,
        role_name="trunk_compression",
        selected_name="conv_local_trunk",
        input_node_id=normed,
    )
    record_role_slot_binding(
        graph,
        role_name="binding_write",
        selected_name="program_slot_write",
        input_node_id=program_proj,
    )
    memory = _add(
        graph,
        "adjacent_token_merge",
        [program_norm],
        context="token_program_interpreter_block.memory",
    )
    memory_skip = _add(
        graph,
        "add",
        [program_norm, memory],
        context="token_program_interpreter_block.memory_skip",
    )
    memory = _add(
        graph,
        "linear_proj",
        [memory],
        {"out_dim": graph.model_dim},
        context="token_program_interpreter_block.memory_proj",
    )

    query = _add(
        graph,
        "linear_proj",
        [trunk],
        {"out_dim": graph.model_dim},
        context="token_program_interpreter_block.query",
    )
    record_role_slot_binding(
        graph,
        role_name="binding_read",
        selected_name="program_slot_read",
        input_node_id=trunk,
    )
    lookup = _add(
        graph,
        "matmul",
        [query, memory],
        context="token_program_interpreter_block.lookup",
    )
    lookup = _add(
        graph,
        "linear_proj",
        [lookup],
        {"out_dim": graph.model_dim},
        context="token_program_interpreter_block.lookup_proj",
    )
    retrieved = _add(
        graph,
        "gather_topk",
        [memory, lookup],
        {"k": 4},
        context="token_program_interpreter_block.retrieve",
    )
    selected = _add(
        graph,
        "mul",
        [retrieved, program_gate],
        context="token_program_interpreter_block.selected",
    )
    selected = _add(
        graph,
        "add",
        [memory_skip, selected],
        context="token_program_interpreter_block.selected_skip",
    )
    record_role_slot_binding(
        graph,
        role_name="merge_policy",
        selected_name="program_select_merge",
        input_node_id=trunk,
    )
    merged = _add(
        graph,
        "add",
        [trunk, selected],
        context="token_program_interpreter_block.merge",
    )
    record_role_slot_binding(
        graph,
        role_name="local_mixing",
        selected_name="identity_local_mix",
        input_node_id=merged,
    )
    stabilized = _add(
        graph,
        "rmsnorm",
        [merged],
        context="token_program_interpreter_block.stabilize",
    )
    record_role_slot_binding(
        graph,
        role_name="stabilizer",
        selected_name="rmsnorm_stabilizer",
        input_node_id=merged,
    )
    return _residual(
        graph,
        input_id,
        stabilized,
        context="token_program_interpreter_block.output",
    )
