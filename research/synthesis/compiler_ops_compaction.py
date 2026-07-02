"""Compiler handlers for the NM-C compaction mixers (Tier D program).

Each handler dispatches to the self-contained mixer module instantiated by the
matching ``_init_*`` in ``compiled_op_params.py`` (the ``pq_embedding_moe_block``
convention: the full class is the compiled op, so search/synthesis and final
model generation share exact semantics). Missing submodules fail loud with
``AttributeError`` — a compiled graph op whose params were never initialised is
a wiring bug, never something to paper over with seeded fallback weights.

Registered via ``compiler_registry.load_split_op_modules`` like every other
``compiler_ops_*`` module. Lane docs live on the mixer classes themselves
(``research/synthesis/<name>.py``) and in
``research/notes/component_fab_compaction_lanes_2026-07-01.md``.
"""

from __future__ import annotations


def _op_monarch_mix(module, inputs, _config):
    return module.monarch_block(inputs[0])


def _op_butterfly_mix(module, inputs, _config):
    return module.butterfly_block(inputs[0])


def _op_recurrent_depth_refine(module, inputs, _config):
    return module.recurrent_depth_block(inputs[0])


def _op_weight_dictionary_mix(module, inputs, _config):
    return module.weight_dictionary_block(inputs[0])


def _op_hypernet_layer_mix(module, inputs, _config):
    return module.hypernet_block(inputs[0])


def _op_persistent_memory_refine(module, inputs, _config):
    return module.persistent_memory_block(inputs[0])


def _op_block_sparse_mix(module, inputs, _config):
    return module.block_sparse_block(inputs[0])


def _op_token_merge_mix(module, inputs, _config):
    return module.token_merge_block(inputs[0])


def _op_ternary_sign_mix(module, inputs, _config):
    return module.ternary_sign_block(inputs[0])


def _op_padic_lowprec_mix(module, inputs, _config):
    return module.padic_lowprec_block(inputs[0])


def _op_subspace_mixture_mix(module, inputs, _config):
    return module.subspace_mixture_block(inputs[0])


def _op_lowrank_state_memory(module, inputs, _config):
    return module.lowrank_state_block(inputs[0])


OP_IMPLS = {
    "monarch_mix": _op_monarch_mix,
    "butterfly_mix": _op_butterfly_mix,
    "recurrent_depth_refine": _op_recurrent_depth_refine,
    "weight_dictionary_mix": _op_weight_dictionary_mix,
    "hypernet_layer_mix": _op_hypernet_layer_mix,
    "persistent_memory_refine": _op_persistent_memory_refine,
    "block_sparse_mix": _op_block_sparse_mix,
    "token_merge_mix": _op_token_merge_mix,
    "ternary_sign_mix": _op_ternary_sign_mix,
    "padic_lowprec_mix": _op_padic_lowprec_mix,
    "subspace_mixture_mix": _op_subspace_mixture_mix,
    "lowrank_state_memory": _op_lowrank_state_memory,
}
