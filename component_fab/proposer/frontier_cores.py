"""Curated frontier-grade binder cores for the fab generator.

The autonomous loop anchors on ``underperforming_novel_ops``, which biases
generation toward weak-binding families (surprise-memory, p-adic, pure SSM).
To MATCH or BEAT frontier models a candidate must START from a proven strong
binder and add a novel mechanism — "frontier + delta". This module supplies
those proven cores as anchors so ``cross_anchor`` can graft novel donor
mechanisms (state / memory / sparsity) onto a core that already binds.

Cores are drawn from the 30M bAbI entity-holdout sweep (2026-05-31), where the
attention family and mixed tropical/sparsemax/wavelet lanes reached restricted
accuracy ~0.88-0.90 (tied with softmax_attention), while surprise-memory,
mamba, and linear-SSM sat at the chance floor (~0.52). Each core's axes are
verified to compile to the strong-binding module via ``code_generator`` and to
pass the capability gate (``can_bind=True``, not eliminated) — see
``tests/test_frontier_cores.py``.

``op_algebraic_space="tropical"`` is what makes ``cross_anchor.is_hosting_anchor``
accept these as hosts; the block-template cores keep a tropical inner anchor, so
the marker is faithful, not a hack.
"""

from __future__ import annotations

from ..improver.axis_variants import AnchorAxes

# (op_name, axes) for each proven binder. Axes are the minimal set that routes
# code_generator to the strong-binding module; cross_anchor layers donor
# mechanisms on top without disturbing the host's global mixing.
_FRONTIER_CORES: tuple[tuple[str, dict[str, object]], ...] = (
    (
        "frontier_tropical_attention",
        {
            "op_algebraic_space": "tropical",
            "op_geometric_receptive_field": "global",
            "op_dynamical_has_state": 0,
        },
    ),
    (
        "frontier_gated_parallel_tropical_sparsemax",
        {
            "op_algebraic_space": "tropical",
            "op_block_template": "gated_parallel",
            "op_block_slot_b": "sparsemax_attention",
            "op_geometric_receptive_field": "global",
        },
    ),
    (
        "frontier_three_lane_tsw",
        {
            "op_algebraic_space": "tropical",
            "op_block_template": "three_lane_adaptive",
            "op_block_slot_b": "sparsemax_attention",
            "op_block_slot_c": "multiscale_wavelet",
            "op_geometric_receptive_field": "global",
        },
    ),
)


def frontier_core_anchors() -> list[AnchorAxes]:
    """Proven strong-binder cores as host anchors for cross-anchor hybrids.

    ``pass_rate=1.0`` reflects that these are validated winners (used only as a
    sort/priority hint downstream; they carry no ledger eval history yet).
    """
    return [
        AnchorAxes(op_name=name, axes=dict(axes), eval_count=0, pass_rate=1.0)
        for name, axes in _FRONTIER_CORES
    ]


def frontier_core_names() -> tuple[str, ...]:
    return tuple(name for name, _ in _FRONTIER_CORES)
