"""Regression tests for the aggressive 2026-04-17 audit cleanup wave.

Pins the second-pass fixes:
  P1.2 — vocab-normalized screening loss thresholds
  P0.5 — requires_residual_context enforcement
  P0.9 — RAG reference's gather_topk now feeds a real cross-attention path
  P1.6 — fragile/retired templates pruned from the registry
  P1.7 — motif sampling enforces a minimum lift floor
"""

from __future__ import annotations

import math
import random

import pytest

pytestmark = pytest.mark.unit


# ── P1.2: vocab-normalized loss limits ────────────────────────────────


def test_screening_loss_limit_scales_with_vocab():
    """A vocab=50000 model must not be killed for a loss that's still well
    under its entropy ceiling (the legacy flat-500 limit was tuned for
    vocab=256 only)."""
    from research.eval.screening_rapid import RapidScreeningCheck

    g = RapidScreeningCheck()
    base = g.LOSS_AT_STEP_25_LIMIT
    expected_50k = base * math.log(50000) / math.log(g.LOSS_LIMIT_VOCAB_BASELINE)
    assert expected_50k > base * 1.5, (
        "vocab-50k limit must be at least 1.5× the base limit; otherwise the "
        "scaling is too weak to prevent vocab-bias kills"
    )


# ── P0.5: requires_residual_context is now enforced ──────────────────


def test_requires_residual_context_set_is_populated():
    from research.synthesis._context_registry import REQUIRES_RESIDUAL_CONTEXT_OPS

    assert len(REQUIRES_RESIDUAL_CONTEXT_OPS) >= 25, (
        "Audit identified 30 ops with this flag — most must be enforced"
    )
    # Spot-check a few: softmax_attention and selective_scan are the canonical
    # examples cited in the audit findings.
    assert "softmax_attention" in REQUIRES_RESIDUAL_CONTEXT_OPS
    assert "selective_scan" in REQUIRES_RESIDUAL_CONTEXT_OPS


def test_grammar_imports_residual_context_set():
    """Pin the integration: grammar must reference the enforcement set."""
    import pathlib

    src = pathlib.Path("research/synthesis/grammar.py").read_text()
    assert "REQUIRES_RESIDUAL_CONTEXT_OPS" in src, (
        "Audit fix P0.5: grammar._validate_graph must consult the enforcement set"
    )
    assert "no downstream add is reachable" in src, (
        "Audit fix P0.5: enforcement should report a clear, specific error"
    )


# ── P0.9: RAG reference actually retrieves ───────────────────────────


def test_rag_reference_has_no_dangling_ops():
    """The pre-2026-04-17 RAG baseline added `gather_topk` whose output was
    never consumed by any downstream op (the audit's "RAG = self-attention
    plus dead op" finding). Whether the new baseline keeps gather_topk and
    consumes it OR drops gather_topk entirely, the structural property is
    the same: every non-output op must have at least one downstream consumer.
    """
    from research.synthesis.reference_architectures import (
        build_retrieval_augmented_layer,
    )

    g = build_retrieval_augmented_layer(d_model=128, top_k=4)
    output_ids = {nid for nid, n in g.nodes.items() if getattr(n, "is_output", False)}
    has_consumer = {nid: False for nid in g.nodes}
    for n in g.nodes.values():
        for pid in n.input_ids:
            if pid in has_consumer:
                has_consumer[pid] = True
    dangling = [
        g.nodes[nid].op_name
        for nid, ok in has_consumer.items()
        if not ok and nid not in output_ids and not g.nodes[nid].is_input
    ]
    assert not dangling, (
        f"RAG reference still has dangling ops (no downstream consumer): {dangling}"
    )


def test_rag_reference_has_real_attention_path():
    """The RAG baseline must contain at least one attention-class op so the
    retrieval intent is not silently a no-op even when the discrete-top-k
    path is unavailable."""
    from research.synthesis.reference_architectures import (
        build_retrieval_augmented_layer,
    )

    g = build_retrieval_augmented_layer(d_model=128, top_k=4)
    op_names = {n.op_name for n in g.nodes.values()}
    attention_family = {
        "softmax_attention",
        "linear_attention",
        "diff_attention",
        "graph_attention",
        "local_window_attn",
        "gated_linear_attention",
        "latent_attention_compressor",
    }
    assert op_names & attention_family, (
        f"RAG baseline contains no attention-family op; found {sorted(op_names)}"
    )


# ── P1.6: retired templates fully pruned ─────────────────────────────


def test_retired_templates_no_longer_in_registry():
    """The 6 templates the audit retired (4 zero-weight + 2 fragile) must be
    absent from both the TEMPLATES dict and the weights dict — neither name
    nor function should be reachable through the public registry."""
    from research.synthesis.templates import (
        DEFAULT_TEMPLATE_WEIGHTS,
        TEMPLATES,
    )

    retired = {
        "attn_reciprocal_gated",
        "attn_softmax_router_sidecar",
        "multiscale_difficulty_router_blocksparse_attn_ssm",
        "multiscale_difficulty_router_easy_attn_ssm",
        "depth_gated_block_matmul",
        "attn_linear_no_matmul_ffn_v2",
    }
    assert not (set(TEMPLATES) & retired), (
        f"Retired templates still in TEMPLATES: {set(TEMPLATES) & retired}"
    )
    assert not (set(DEFAULT_TEMPLATE_WEIGHTS) & retired), (
        f"Retired templates still have weight entries: "
        f"{set(DEFAULT_TEMPLATE_WEIGHTS) & retired}"
    )


def test_template_registry_and_weights_align():
    """No orphan weights, no templates without weight."""
    from research.synthesis.templates import (
        DEFAULT_TEMPLATE_WEIGHTS,
        TEMPLATES,
    )

    orphan_weights = set(DEFAULT_TEMPLATE_WEIGHTS) - set(TEMPLATES)
    missing_weights = set(TEMPLATES) - set(DEFAULT_TEMPLATE_WEIGHTS)
    assert not orphan_weights, f"Weights without templates: {orphan_weights}"
    assert not missing_weights, f"Templates without weights: {missing_weights}"


# ── P1.7: motif lift floor ───────────────────────────────────────────


def test_motif_lift_floor_excludes_low_lift_motifs():
    from research.synthesis._motif_selection import (
        ALL_MOTIFS,
        MIN_MOTIF_LIFT,
        pick_motif,
    )

    # Identify low-lift motifs that the floor should suppress.
    low_lift = [m for m in ALL_MOTIFS if m.lift < MIN_MOTIF_LIFT]
    assert low_lift, (
        "Lift floor only matters if some motifs have lift < floor; the audit "
        "identified at least 6 such motifs (attn_softmax, etc.)"
    )

    # Walk every motif class and verify we never sample a low-lift motif.
    rng = random.Random(0)
    classes = sorted({m.motif_class for m in ALL_MOTIFS})
    sampled_names: set[str] = set()
    for cls in classes:
        for _ in range(30):
            picked = pick_motif(rng, cls)
            if picked is not None:
                sampled_names.add(picked.name)
    leaked = {m.name for m in low_lift} & sampled_names
    assert not leaked, f"Low-lift motifs leaked into sampling: {leaked}"


def test_motif_weights_override_can_resurrect_low_lift_motif():
    """Override weights must still pull a low-lift motif back into the pool —
    learned policies may want to revisit a previously-demoted motif."""
    from research.synthesis._motif_selection import (
        ALL_MOTIFS,
        MIN_MOTIF_LIFT,
        pick_motif,
    )

    low_lift = next(m for m in ALL_MOTIFS if m.lift < MIN_MOTIF_LIFT)
    rng = random.Random(0)
    # Heavy override on the low-lift motif; it should at least become reachable.
    sampled = set()
    for _ in range(50):
        picked = pick_motif(rng, low_lift.motif_class, weights={low_lift.name: 100.0})
        if picked is not None:
            sampled.add(picked.name)
    assert low_lift.name in sampled, (
        "Override weight=100 should make a low-lift motif sampleable; "
        f"{low_lift.name} was excluded entirely"
    )
