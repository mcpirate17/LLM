"""New-substrate templates added 2026-05-10 / 2026-05-11.

Each new primitive (tropical_softmax, pq_embedding, mla_attention, tree_mix)
ships with a bare-primitive template, plus a fused variant for the three
binary primitives that pair them with the empirical winner-motif slot
constraints (research/synthesis/_templates_attention_tail.py:
_LATENT_ATTN_SPARSE_FFN_FFN_CLASSES_FALLBACK etc.).

Split out of ``_templates_exotic.py`` (2026-05-11) when that file crossed
the 1250-line guardrail.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import ComputationGraph

from ._template_helpers import (
    MOTIF_CLASS_CONV,
    MOTIF_CLASS_FFN,
    MOTIF_CLASS_GATE,
    MOTIF_CLASS_MOE,
    MOTIF_CLASS_NORM,
    MotifWeights,
    _FFN_CLASSES,
    _fix_dim,
    _instantiate_motif,
    _pick_compatible_motif,
    _pick_compatible_motif_from_classes,
)
from ._templates_core import tpl_residual_block


# Tightened FFN slot classes for the fused-substrate templates below.
# These match the empirical pass cohorts of latent_attn_sparse_ffn /
# latent_attn_moe (research/reports/slot_tightening_proposal.json,
# 2026-05-04 Phase 3.2). Hardcoded fallbacks because the empirical-Bayes
# slot constraints loader keys off the template name, not the slot
# composition — these new templates haven't accumulated evidence yet, so
# the loader would fall back to the legacy classes anyway.
_SPARSE_FFN_CLASSES = (MOTIF_CLASS_CONV, MOTIF_CLASS_FFN)
_MOE_FFN_CLASSES = (MOTIF_CLASS_MOE, MOTIF_CLASS_GATE)


def tpl_tropical_softmax_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → score_proj → tropical_softmax → mul(input) → proj → [FFN] → residual.

    Uses the gradient-friendly softmin (tropical_softmax) as a per-token,
    per-feature gating distribution: low scores get high mass. Drop-in for
    sigmoid/softmax gating in tropical-flavored contexts.

    Per external_research_2026-05-10.md §3.5 — tropical_softmax replaces
    hard max with LogSumExp-temperature, preserving gradient flow.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        scores = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        gate = graph.add_op("tropical_softmax", [scores])
        gated = graph.add_op("mul", [normed, gate])
        projected = graph.add_op("linear_proj", [gated], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, projected, ffn, rng) if ffn else projected
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_pq_embedding_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → linear_proj → pq_embedding → linear_proj → [FFN] → residual.

    Product-quantized embedding block (research §2.3). The linear_proj on
    the input gives the codebook something flexible to quantize; the
    post-pq linear_proj projects back out of the quantization basin.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        proj_in = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        pq = graph.add_op("pq_embedding", [proj_in])
        proj_out = graph.add_op("linear_proj", [pq], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, proj_out, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, proj_out, ffn, rng) if ffn else proj_out
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_mla_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → query_proj + kv_proj → mla_attention → out_proj → [FFN] → residual.

    Multi-head latent attention (research §1.1). The query path stays
    full-dim; the kv path is the shared-latent compression target. This
    template is the canonical MLA-as-block synthesis pattern.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        q = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        kv = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        attn = graph.add_op("mla_attention", [q, kv])
        out = graph.add_op("linear_proj", [attn], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, out, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, out, ffn, rng) if ffn else out
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_tree_mix_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → 4 leaf projections → depth-2 binary tree of tree_mix nodes → proj → [FFN] → residual.

    Balanced binary-tree feature mixer (research §2.1, "leafed layers").
    Builds a depth-2 tree: 4 sibling linear projections of the normalized
    input act as leaves, then 3 tree_mix nodes blend them pairwise up to
    the root. Each tree_mix has its own learned sigmoid gate, so the
    grammar gets per-level asymmetric mixing along a structural axis it
    previously couldn't express.

        leaves: a, b, c, d  (4 × linear_proj of the same norm)
        level 1: ab = tree_mix(a, b)        cd = tree_mix(c, d)
        level 2: root = tree_mix(ab, cd)
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        a = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        b = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        c = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        d = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        ab = graph.add_op("tree_mix", [a, b])
        cd = graph.add_op("tree_mix", [c, d])
        root = graph.add_op("tree_mix", [ab, cd])
        projected = graph.add_op("linear_proj", [root], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, projected, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, projected, ffn, rng) if ffn else projected
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_mla_sparse_ffn_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → mla_attention → out_proj → CONV/FFN motif → residual.

    Fuses MLA (research §1.1) with the latent_attn_sparse_ffn winner motif
    (CONV + FFN tail). Bare ``tpl_mla_block`` uses the general ``_FFN_CLASSES``;
    this variant restricts the FFN slot to the empirical pass cohort of
    latent_attn_sparse_ffn (conv_swiglu, etc.). The substrate-fusion
    hypothesis: incumbent winners compose attention with the tightened
    FFN cohort; bare MLA + general FFN doesn't reach the same basin.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        q = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        kv = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        attn = graph.add_op("mla_attention", [q, kv])
        out = graph.add_op("linear_proj", [attn], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, out, rng, list(_SPARSE_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, out, ffn, rng) if ffn else out
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_pq_embedding_moe_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """Factorized Semantic Substrate (FSS) — High-depth FSB implementation.

    Derivation: uses PQ-quantization as a semantic low-pass filter to drive
    a multi-lane routing architecture. Denoised signals are processed by
    parallel lanes (Easy/Medium/Hard) and merged via codebook-aligned gates.

    Topology:
      Stem: norm → proj → act → PQ → proj → norm
      Lanes:
        1. Easy: Identity skip of quantized stem
        2. Medium: Token-conditioned lightweight mixer (Conv/Sparse)
        3. Hard: Semantic MoE (Top-K expert routing)
      Merge: Multi-lane residual merge → Token Merge Proj → Final Norm → Add

    Total Ops: ~18-22 (Passes all structural and depth gates).
    """
    template_name = "pq_embedding_moe_block"
    template_instance = int(graph.metadata.get("_active_template_instance", 0) or 0)
    D = graph.model_dim
    from ._template_helpers import (
        record_template_slot_binding,
        template_add_op as _add,
        template_add_residual as _residual,
    )
    from ._templates_routing import _single_input_op_config

    # ── Phase 1: Semantic Stem (The Bottleneck) ───────────────────
    norm1 = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    stem = _instantiate_motif(graph, input_id, norm1, rng) if norm1 else input_id

    try:
        # Project to factorized axes
        h = _add(graph, "linear_proj", [stem], {"out_dim": D}, context="fss.proj_in")
        # Non-linear feature prep
        h = _add(graph, "silu", [h], context="fss.act")
        # Information Bottleneck (Low-pass filter)
        h = _add(graph, "pq_embedding", [h], context="fss.pq")
        # Restore model space
        h = _add(graph, "linear_proj", [h], {"out_dim": D}, context="fss.proj_out")
        # Stabilize for routing
        quantized_stem = _add(graph, "rmsnorm", [h], context="fss.stem_norm")
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    # ── Phase 2: Routing Signals ──────────────────────────────────
    # Drive routing based on the denoised "palette" representation
    signal = _add(
        graph,
        "token_class_proj",
        [quantized_stem],
        {"n_classes": 4},
        context="fss.signal",
    )
    difficulty = _add(graph, "token_entropy", [signal], context="fss.difficulty")

    # ── Phase 3: Multi-Lane Execution ─────────────────────────────

    # Lane 1: Easy (Denoised identity)
    easy_lane = quantized_stem

    # Lane 2: Medium (Denoised Mixer)
    # Picks a lightweight mixer from the "medium" cohort
    medium_op = rng.choice(["conv1d_seq", "nm_sparse_linear", "block_sparse_linear"])
    medium_cfg = _single_input_op_config(medium_op, D, rng)
    if medium_op == "nm_sparse_linear":
        medium_cfg.update({"n": 2, "m": 4})
    
    medium_lane = _add(
        graph, medium_op, [quantized_stem], medium_cfg, context="fss.medium"
    )
    medium_lane = _add(
        graph, "linear_proj", [medium_lane], {"out_dim": D}, context="fss.medium_proj"
    )

    # Lane 3: Hard (Semantic MoE)
    # Uses a sparse router driven by the quantized signal
    gated_tokens = _add(
        graph,
        "hybrid_token_gate",
        [quantized_stem],
        {"threshold": 0.5},
        context="fss.token_gate",
    )
    router = _add(
        graph,
        "hybrid_sparse_router",
        [gated_tokens],
        {"span_width": 2, "lane_count": 2, "confidence_threshold": 0.5},
        context="fss.router",
    )
    
    moe_motif = _pick_compatible_motif(graph, router, rng, MOTIF_CLASS_MOE, weights)
    if moe_motif:
        hard_lane = _instantiate_motif(graph, router, moe_motif, rng)
    else:
        # MoE Top-K fallback
        hard_lane = _add(
            graph,
            "moe_topk",
            [router],
            {"num_experts": 4, "top_k": 2},
            context="fss.moe",
        )
    hard_lane = _fix_dim(graph, hard_lane)

    # ── Phase 4: Merging & Final Synthesis ────────────────────────
    
    # Merge easy + medium via difficulty gate
    merged = _add(graph, "mul", [medium_lane, difficulty], context="fss.gate_medium")
    merged = _residual(graph, easy_lane, merged, context="fss.merge_easy_medium")
    
    # Merge hard lane
    merged = _residual(graph, merged, hard_lane, context="fss.merge_hard")
    
    # Token-level consolidation
    merged_tokens = _add(
        graph, "linear_proj", [merged], {"out_dim": D}, context="fss.token_merge"
    )
    
    # Final refinement
    post = _add(graph, "rmsnorm", [merged_tokens], context="fss.post_norm")

    # Record bindings for search visibility
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=1,
        slot_key="fss.stem",
        slot_classes=["bottleneck"],
        selected_name="pq_embedding",
        selected_class="primitive",
        input_node_id=input_id,
    )
    record_template_slot_binding(
        graph,
        template_name=template_name,
        template_instance=template_instance,
        slot_index=2,
        slot_key="fss.hard_lane",
        slot_classes=["moe"],
        selected_name=moe_motif.name if moe_motif else "moe_topk",
        selected_class="motif",
        input_node_id=router,
    )

    # Global residual bypass
    try:
        return _add(graph, "add", [input_id, post], context="fss.output")
    except ValueError:
        return post


def tpl_mlstm_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → mlstm_cell → linear_proj → [FFN] → residual.

    Bare-cell template for the xLSTM matrix-memory recurrence (research §1.5).
    The cell maintains a (D, D) outer-product state addressed by per-token
    queries; the post-cell linear_proj reshapes the retrieved vector before
    the standard FFN tail and residual.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        cell = graph.add_op("mlstm_cell", [normed])
        out = graph.add_op("linear_proj", [cell], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, out, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, out, ffn, rng) if ffn else out
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_mlstm_sparse_ffn_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → mlstm_cell → linear_proj → CONV/FFN motif → residual.

    Fuses the mLSTM matrix-memory cell with the latent_attn_sparse_ffn
    winner motif's tightened FFN slot (CONV + FFN). Pairs the novel
    state form with the empirically-best post-mixer FFN cohort, matching
    the substrate-fusion hypothesis from handoff_2026-05-11.
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        cell = graph.add_op("mlstm_cell", [normed])
        out = graph.add_op("linear_proj", [cell], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, out, rng, list(_SPARSE_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, out, ffn, rng) if ffn else out
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed


def tpl_tree_mix_attention_block(
    graph: ComputationGraph,
    input_id: int,
    rng: random.Random,
    weights: MotifWeights = None,
) -> int:
    """norm → tree_mix(depth-1) → softmax_attention → FFN → residual.

    Fuses the leafed-layer mixer (research §2.1) with softmax attention.
    The tree_mix node blends two sibling projections of the normed input
    into a single hidden state; that hidden state then feeds standard
    softmax attention. This pairs the novel mixer topology with the most
    reliable feature-routing op (softmax_attention), giving the grammar a
    way to compose tree_mix with attention without the all-tree depth-2
    structure that ``tpl_tree_mix_block`` uses (and which has no S1
    evidence so far).
    """
    D = graph.model_dim
    norm = _pick_compatible_motif(graph, input_id, rng, MOTIF_CLASS_NORM, weights)
    normed = _instantiate_motif(graph, input_id, norm, rng) if norm else input_id

    try:
        leaf_a = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        leaf_b = graph.add_op("linear_proj", [normed], config={"out_dim": D})
        mixed = graph.add_op("tree_mix", [leaf_a, leaf_b])
        attn = graph.add_op("softmax_attention", [mixed])
        # Context rule (research/synthesis/_context_registry.py:351):
        # softmax_attention forbids linear_proj as a direct successor
        # ("raw attention output needs norm first"); insert rmsnorm.
        attn_norm = graph.add_op("rmsnorm", [attn])
        out = graph.add_op("linear_proj", [attn_norm], config={"out_dim": D})
    except (ValueError, KeyError):
        return tpl_residual_block(graph, input_id, rng, weights)

    ffn = _pick_compatible_motif_from_classes(
        graph, out, rng, list(_FFN_CLASSES), weights
    )
    processed = _instantiate_motif(graph, out, ffn, rng) if ffn else out
    processed = _fix_dim(graph, processed)
    try:
        return graph.add_op("add", [input_id, processed])
    except ValueError:
        return processed
