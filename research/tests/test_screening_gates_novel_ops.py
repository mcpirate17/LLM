# pyright: reportPrivateImportUsage=false
"""Structural screening gates must see the novel registry ops — drift pins.

The 2026-07-02 overnight campaign measured gates 5/6 killing S1 candidates
whose only routing/mixing op was a freshly registry-wired novel mechanism:
the gates' hand-maintained op lists had drifted from the registry, so the
screening funnel silently abandoned the novel branch (16 of 32 S1 candidates
structurally dropped, `gate6_no_mixing`=9, `gate5_no_routing`=5). These pins
make that drift class impossible to reintroduce silently:

- gate 5 (`_EFFICIENCY_OPS`) must be a superset of the grammar's
  `ROUTING_COMPRESSION_MOE_OPS` (it is now derived by union),
- gate 6 (`SEQUENCE_MIXING_OPS`) must contain every cross-token novel mixer,
- gate 8 (`CONTENT_ADDRESSED_OPS`) must contain the content-addressed ones,
- and channel-only novel ops must stay OUT of gate 6 (it exists to demand
  genuine token mixing).
"""

from __future__ import annotations

from research.scientist.runner.execution_screening import _EFFICIENCY_OPS
from research.scientist.runner.execution_screening_graphs import (
    CONTENT_ADDRESSED_OPS,
    SEQUENCE_MIXING_OPS,
)
from research.synthesis.grammar_support import ROUTING_COMPRESSION_MOE_OPS

CROSS_TOKEN_NOVEL_OPS = {
    "token_merge_mix",
    "recurrent_depth_refine",
    "persistent_memory_refine",
    "lowrank_state_memory",
    "idempotent_oblique_memory",
    "nilpotent_lie_scan",
    "integral_control_mixer",
    "port_hamiltonian_mix",
    "cdma_slot_binding",
    "scale_equivariant_wavelet",
}
CONTENT_ADDRESSED_NOVEL_OPS = {
    "cdma_slot_binding",
    "lowrank_state_memory",
    "persistent_memory_refine",
    "idempotent_oblique_memory",
}
CHANNEL_ONLY_NOVEL_OPS = {
    "monarch_mix",
    "butterfly_mix",
    "weight_dictionary_mix",
    "hypernet_layer_mix",
    "block_sparse_mix",
    "ternary_sign_mix",
    "subspace_mixture_mix",
}


def test_gate5_superset_of_grammar_routing_set() -> None:
    missing = ROUTING_COMPRESSION_MOE_OPS - _EFFICIENCY_OPS
    assert not missing, (
        f"gate 5 drifted from the grammar routing set; missing: {sorted(missing)}"
    )


def test_gate6_sees_cross_token_novel_mixers() -> None:
    missing = CROSS_TOKEN_NOVEL_OPS - SEQUENCE_MIXING_OPS
    assert not missing, (
        f"gate 6 would kill graphs whose only mixer is: {sorted(missing)}"
    )


def test_gate8_sees_content_addressed_novel_ops() -> None:
    missing = CONTENT_ADDRESSED_NOVEL_OPS - CONTENT_ADDRESSED_OPS
    assert not missing, f"gate 8 would deny binding-capability to: {sorted(missing)}"


def test_gate6_excludes_channel_only_novel_ops() -> None:
    """Gate 6 demands genuine token mixing — pointwise channel factorizations
    must not sneak in and weaken it."""
    leaked = CHANNEL_ONLY_NOVEL_OPS & SEQUENCE_MIXING_OPS
    assert not leaked, f"channel-only ops must not satisfy gate 6: {sorted(leaked)}"
