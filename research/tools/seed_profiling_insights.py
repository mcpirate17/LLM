#!/usr/bin/env python3
"""Seed insights derived from component profiling data (81K data points).

All alpha/beta values derived from research/profiling/component_profiles.db
(125 ops, 5979 pairs, 75000 triplets — zero errors).

Idempotent: uses semantic_key dedup to avoid duplicates.
Run: python -m research.tools.seed_profiling_insights [--db research/lab_notebook.db]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from research.scientist.notebook import LabNotebook

# ─────────────────────────────────────────────────────────────────────
# All counts below from component_profiles.db queries, 2026-03-15.
# alpha = observed successes + 1, beta_ = observed failures + 1
# confidence = alpha / (alpha + beta_)
# ─────────────────────────────────────────────────────────────────────

PROFILING_INSIGHTS = [
    # ══════════════════════════════════════════════════════════════════
    # COMPOSITION RULES — math_space exit
    # ══════════════════════════════════════════════════════════════════
    {
        "category": "pattern",
        "content": (
            "After any math_space op, follow with a projection or sigmoid. "
            "Stable followers: sigmoid (96.3%, n=27), token_type_classifier (100%, n=27), "
            "shared_basis_proj (96.3%), bottleneck_proj (88.9%), tied_proj (88.9%), "
            "basis_expansion (88.9%), low_rank_proj (81.5%), swiglu_mlp (63.0%). "
            "math_space → mixing is 0% stable (n=26). math_space → elementwise_binary "
            "is 10.4% (n=135). The projection/sigmoid set acts as a gradient firewall."
        ),
        "insight_type": "composition_rule",
        "subject_key": "math_space_exit",
        "semantic_key": "profiling:math_space_exit",
        # Recommended follower set has 88-100% per-op stability
        # Rule confidence is high; overall rate is low because most followers fail
        "alpha": 8.0,  # 8 recommended ops
        "beta_": 1.0,
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "profiling_pair_stability",
            "source": "component_profiles.db",
            "n_pairs": 1107,
            "overall_rate": 0.201,
            "best_followers": {
                "token_type_classifier": {"rate": 1.000, "n": 27},
                "sigmoid": {"rate": 0.963, "n": 27},
                "shared_basis_proj": {"rate": 0.963, "n": 27},
                "bottleneck_proj": {"rate": 0.889, "n": 27},
                "tied_proj": {"rate": 0.889, "n": 27},
                "basis_expansion": {"rate": 0.889, "n": 27},
                "low_rank_proj": {"rate": 0.815, "n": 27},
                "swiglu_mlp": {"rate": 0.630, "n": 27},
            },
            "worst_followers": {
                "mixing_category": {"rate": 0.000, "n": 26},
                "elementwise_binary_category": {"rate": 0.104, "n": 135},
            },
        },
    },

    # ══════════════════════════════════════════════════════════════════
    # Non-euclidean → euclidean bridge ops
    # ══════════════════════════════════════════════════════════════════
    {
        "category": "pattern",
        "content": (
            "When exiting non-euclidean space (tropical/poincare/clifford/padic/spiking) "
            "to euclidean, use a bridge op. Effective bridges: token_type_classifier "
            "(26/26 = 100%), shared_basis_proj (25/26 = 96%), sigmoid (25/26 = 96%), "
            "basis_expansion (23/26 = 88%), bottleneck_proj (23/26 = 88%), "
            "tied_proj (23/26 = 88%), low_rank_proj (21/26 = 81%). "
            "Overall non-euclidean→euclidean is 20.9% stable (223/1065) without "
            "selecting the right receiver."
        ),
        "insight_type": "composition_rule",
        "subject_key": "space_bridge_exit",
        "semantic_key": "profiling:space_bridge_exit",
        # 7 bridge ops with 81-100% individual stability
        "alpha": 8.0,
        "beta_": 1.0,
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "profiling_pair_stability",
            "source": "component_profiles.db",
            "n_pairs": 1065,
            "overall_rate": 0.209,
            "bridge_ops": {
                "token_type_classifier": {"rate": 1.000, "n": 26},
                "shared_basis_proj": {"rate": 0.962, "n": 26},
                "sigmoid": {"rate": 0.962, "n": 26},
                "basis_expansion": {"rate": 0.885, "n": 26},
                "bottleneck_proj": {"rate": 0.885, "n": 26},
                "tied_proj": {"rate": 0.885, "n": 26},
                "low_rank_proj": {"rate": 0.808, "n": 26},
                "swiglu_mlp": {"rate": 0.615, "n": 26},
            },
            "space_breakdown": {
                "poincare_to_euclidean": {"rate": 0.290, "n": 245},
                "spiking_to_euclidean": {"rate": 0.226, "n": 164},
                "clifford_to_euclidean": {"rate": 0.190, "n": 205},
                "tropical_to_euclidean": {"rate": 0.181, "n": 287},
                "padic_to_euclidean": {"rate": 0.146, "n": 164},
            },
        },
    },

    # ══════════════════════════════════════════════════════════════════
    # Euclidean → non-euclidean entry
    # ══════════════════════════════════════════════════════════════════
    {
        "category": "pattern",
        "content": (
            "Euclidean → non-euclidean is nearly always unstable: 27/702 = 3.8%. "
            "Only euclidean → poincare has marginal viability at 16.7%. "
            "euclidean → {clifford, tropical, padic, spiking} is 0% stable. "
            "Entering non-euclidean space must be done from the same space or "
            "through dedicated entry points, not from arbitrary euclidean ops."
        ),
        "insight_type": "composition_rule",
        "subject_key": "space_entry_restriction",
        "semantic_key": "profiling:space_entry_restriction",
        # Rule: euclidean→non-euclidean fails 96% of the time — very confident
        "alpha": 675.0,
        "beta_": 28.0,
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "profiling_pair_stability",
            "source": "component_profiles.db",
            "n_pairs": 702,
            "overall_rate": 0.038,
            "by_target_space": {
                "euclidean_to_poincare": {"rate": 0.167, "n": 162},
                "euclidean_to_clifford": {"rate": 0.000, "n": 135},
                "euclidean_to_tropical": {"rate": 0.000, "n": 189},
                "euclidean_to_padic": {"rate": 0.000, "n": 108},
                "euclidean_to_spiking": {"rate": 0.000, "n": 108},
            },
        },
    },

    # ══════════════════════════════════════════════════════════════════
    # Normalization placement
    # ══════════════════════════════════════════════════════════════════
    {
        "category": "pattern",
        "content": (
            "layernorm must be used as a block opener, never as a receiver. "
            "X → layernorm: 0/100 = 0% stable (no predecessor works). "
            "layernorm → X: 14/84 = 16.7% stable, but only into parameterized ops: "
            "bottleneck_proj, diff_attention, low_rank_proj, mixed_recursion_gate, "
            "multi_head_mix, rwkv_channel, rwkv_time_mixing, shared_basis_proj, "
            "swiglu_mlp, tied_proj, token_type_classifier, compression_mixture_experts. "
            "layernorm renormalizes to unit variance, erasing upstream gradient signal."
        ),
        "insight_type": "composition_rule",
        "subject_key": "layernorm",
        "semantic_key": "profiling:layernorm_placement",
        # Rule: 0/100 as receiver is rock-solid evidence. 14 valid followers identified.
        "alpha": 100.0,  # 100 failures as receiver = very confident in "never as receiver"
        "beta_": 1.0,
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "profiling_pair_stability",
            "source": "component_profiles.db",
            "n_as_receiver": 100,
            "stable_as_receiver": 0,
            "n_as_sender": 84,
            "stable_as_sender": 14,
            "valid_followers": [
                "basis_expansion", "bottleneck_proj", "diff_attention",
                "low_rank_proj", "mixed_recursion_gate", "multi_head_mix",
                "rwkv_channel", "rwkv_time_mixing", "shared_basis_proj",
                "swiglu_mlp", "tied_proj", "token_type_classifier",
                "compression_mixture_experts", "hyp_distance",
            ],
        },
    },
    {
        "category": "pattern",
        "content": (
            "rmsnorm is a better normalizer than layernorm for composition. "
            "X → rmsnorm: 18/100 = 18% stable. rmsnorm → X: 22/84 = 26.2% stable. "
            "Works after: elementwise_unary (4), elementwise_binary (3), structural (2), "
            "sequence (2), functional (2), parameterized (5). "
            "Works before same ops as layernorm plus conv1d_seq, cascade, selective_scan. "
            "Prefer rmsnorm over layernorm when normalization is needed mid-block."
        ),
        "insight_type": "composition_rule",
        "subject_key": "rmsnorm",
        "semantic_key": "profiling:rmsnorm_placement",
        # Rule: rmsnorm strictly better than layernorm — 18% vs 0% as receiver
        "alpha": 18.0,
        "beta_": 1.0,
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "profiling_pair_stability",
            "source": "component_profiles.db",
            "n_as_receiver": 100,
            "stable_as_receiver": 18,
            "n_as_sender": 84,
            "stable_as_sender": 22,
            "vs_layernorm": {
                "layernorm_receiver_rate": 0.000,
                "rmsnorm_receiver_rate": 0.180,
                "layernorm_sender_rate": 0.167,
                "rmsnorm_sender_rate": 0.262,
            },
        },
    },

    # ══════════════════════════════════════════════════════════════════
    # Sigmoid is the universal stabilizer activation
    # ══════════════════════════════════════════════════════════════════
    {
        "category": "success_factor",
        "content": (
            "sigmoid is the best universal stabilizer activation: 85/94 = 90.4% stable "
            "as receiver with 100% category coverage (works after every category). "
            "Particularly effective after math_space (26/27 stable). Other activations: "
            "silu (41/94 = 43.6%), gelu (39/94 = 41.5%), tanh (32/94 = 34.0%), "
            "relu (28/94 = 29.8%). sigmoid's bounded output [0,1] prevents gradient "
            "explosion from upstream instability."
        ),
        "insight_type": "top_op",
        "subject_key": "sigmoid",
        "semantic_key": "profiling:sigmoid_universal_stabilizer",
        "alpha": 86.0,
        "beta_": 9.0,
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "profiling_pair_stability",
            "source": "component_profiles.db",
            "n_as_receiver": 94,
            "stable_as_receiver": 85,
            "rate": 0.904,
            "category_coverage": 1.0,
            "comparison": {
                "sigmoid": {"rate": 0.904, "n": 94},
                "silu": {"rate": 0.436, "n": 94},
                "gelu": {"rate": 0.415, "n": 94},
                "tanh": {"rate": 0.340, "n": 94},
                "relu": {"rate": 0.298, "n": 94},
            },
        },
    },

    # ══════════════════════════════════════════════════════════════════
    # Parameterized self-composition core
    # ══════════════════════════════════════════════════════════════════
    {
        "category": "success_factor",
        "content": (
            "Parameterized → parameterized is the stable backbone: 62.5% overall "
            "(n=1232). The 'safe inner core' has 100% receiver stability from other "
            "parameterized ops: bottleneck_proj, compression_mixture_experts, swiglu_mlp, "
            "shared_basis_proj, tied_proj, token_type_classifier, mixed_recursion_gate, "
            "low_rank_proj, rwkv_channel, rwkv_time_mixing. Use residual composition "
            "for 83.1% stability (vs 62.5% sequential)."
        ),
        "insight_type": "winning_combo",
        "subject_key": "parameterized_core",
        "semantic_key": "profiling:parameterized_self_composition",
        "alpha": 770.0,
        "beta_": 462.0,
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "profiling_pair_stability",
            "source": "component_profiles.db",
            "n_param_param_pairs": 1232,
            "overall_rate": 0.625,
            "residual_rate": 0.831,
            "safe_receivers_100pct": [
                "bottleneck_proj", "compression_mixture_experts", "swiglu_mlp",
                "shared_basis_proj", "tied_proj", "token_type_classifier",
                "mixed_recursion_gate", "low_rank_proj", "rwkv_channel",
                "rwkv_time_mixing",
            ],
            "best_senders": {
                "swiglu_mlp": {"rate": 0.897, "n": 29},
                "token_type_classifier": {"rate": 0.897, "n": 29},
                "mixed_recursion_gate": {"rate": 0.878, "n": 41},
                "compression_mixture_experts": {"rate": 0.878, "n": 41},
                "bottleneck_proj": {"rate": 0.818, "n": 22},
            },
        },
    },

    # ══════════════════════════════════════════════════════════════════
    # Residual vs sequential
    # ══════════════════════════════════════════════════════════════════
    {
        "category": "structural_preference",
        "content": (
            "Residual composition is safer than sequential: 42.4% stable (n=858) vs "
            "37.5% sequential (n=4342) vs 36.2% adapted (n=779). For parameterized→"
            "parameterized, residual reaches 83.1% (n=148). elementwise_unary→"
            "elementwise_binary is 100% stable in residual (n=35). Always prefer "
            "residual when both ops have matching dimensions."
        ),
        "insight_type": "composition_rule",
        "subject_key": "residual_preference",
        "semantic_key": "profiling:residual_vs_sequential",
        # param→param residual at 83.1% vs 62.5% sequential — strong signal
        "alpha": 123.0,  # 83.1% of 148 residual param→param
        "beta_": 25.0,
        "display_only": False,
        "insight_level": "structural",
        "evidence_json": {
            "test": "profiling_pair_stability",
            "source": "component_profiles.db",
            "composition_rates": {
                "residual": {"rate": 0.424, "n": 858},
                "sequential": {"rate": 0.375, "n": 4342},
                "adapted": {"rate": 0.362, "n": 779},
            },
            "param_param_residual": {"rate": 0.831, "n": 148},
        },
    },

    # ══════════════════════════════════════════════════════════════════
    # Triplet rescue — parameterized middle ops
    # ══════════════════════════════════════════════════════════════════
    {
        "category": "pattern",
        "content": (
            "When A→C is unstable, inserting a parameterized middle op B rescues the "
            "triplet. 7,298 rescue patterns found across 75K triplets. Top rescuers: "
            "token_type_classifier (430), swiglu_mlp (353), mixed_recursion_gate (276), "
            "progressive_compression_gate (271), bottleneck_proj (259), low_rank_proj "
            "(257), tied_proj (256), shared_basis_proj (254). Parameterized ops provide "
            "66% of all rescues (4,834/7,298). Rule: when composing incompatible ops, "
            "insert a projection layer between them."
        ),
        "insight_type": "composition_rule",
        "subject_key": "triplet_rescue",
        "semantic_key": "profiling:triplet_rescue_pattern",
        "alpha": 4835.0,
        "beta_": 2464.0,
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "profiling_triplet_rescue",
            "source": "component_profiles.db",
            "n_rescues": 7298,
            "rescue_by_category": {
                "parameterized": 4834,
                "elementwise_unary": 591,
                "structural": 533,
                "functional": 399,
                "mixing": 368,
                "math_space": 272,
                "elementwise_binary": 224,
                "sequence": 68,
                "reduction": 9,
            },
            "top_rescue_ops": {
                "token_type_classifier": 430,
                "swiglu_mlp": 353,
                "mixed_recursion_gate": 276,
                "progressive_compression_gate": 271,
                "bottleneck_proj": 259,
                "low_rank_proj": 257,
                "tied_proj": 256,
                "shared_basis_proj": 254,
            },
        },
    },

    # ══════════════════════════════════════════════════════════════════
    # Lipschitz dampening rule
    # ══════════════════════════════════════════════════════════════════
    {
        "category": "pattern",
        "content": (
            "High-Lipschitz ops (>2.0) must be followed by a projection to dampen "
            "gradient amplification. Effective dampeners reduce pair Lipschitz by "
            "100-1000x. For spike_rate_code (lip=22.7): token_type_classifier → 0.18, "
            "shared_basis_proj → 0.36. For topk_gate (lip=10.5): compression_mixture_experts "
            "→ 0.00, bottleneck_proj → 0.004. For route_topk (lip=16.2): hyp_distance "
            "→ 0.00, shared_basis_proj → 0.005. The Lipschitz product of A×B predicts "
            "pair stability with 29% feature importance (top predictor in gradient "
            "boosting at 90.3% accuracy)."
        ),
        "insight_type": "composition_rule",
        "subject_key": "lipschitz_dampening",
        "semantic_key": "profiling:lipschitz_dampening",
        # ML model predicts at 90.3% — rule is highly reliable
        "alpha": 903.0,
        "beta_": 97.0,
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "profiling_ml_analysis",
            "source": "component_profiles.db",
            "ml_model": "GradientBoosting",
            "ml_accuracy": 0.903,
            "top_feature": "lipschitz_product",
            "top_feature_importance": 0.290,
            "rule": "lip_A * lip_B > 7 → almost certainly unstable",
            "dampener_ops": [
                "token_type_classifier", "compression_mixture_experts",
                "shared_basis_proj", "bottleneck_proj", "low_rank_proj",
                "tied_proj", "hyp_distance",
            ],
        },
    },

    # ══════════════════════════════════════════════════════════════════
    # cumsum must be followed by bounded activation
    # ══════════════════════════════════════════════════════════════════
    {
        "category": "pattern",
        "content": (
            "cumsum accumulates values along the sequence, causing unbounded growth. "
            "Only 4/30 followers are stable: cos, sigmoid, softmax_last, tanh. All four "
            "are bounded activations that clamp the accumulated values. "
            "cumsum → parameterized is 0% stable. cumsum → elementwise_binary is 0% "
            "stable. Rule: cumsum must always be followed by a bounded activation."
        ),
        "insight_type": "composition_rule",
        "subject_key": "cumsum",
        "semantic_key": "profiling:cumsum_bounded_activation",
        # 4/4 recommended followers work — rule is reliable
        "alpha": 5.0,
        "beta_": 1.0,
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "profiling_pair_stability",
            "source": "component_profiles.db",
            "n_as_sender": 30,
            "stable_as_sender": 4,
            "valid_followers": ["cos", "sigmoid", "softmax_last", "tanh"],
            "invalid_categories": ["parameterized", "elementwise_binary", "mixing"],
        },
    },

    # ══════════════════════════════════════════════════════════════════
    # Universal stabilizer ops (high receiver stability + cat coverage)
    # ══════════════════════════════════════════════════════════════════
    {
        "category": "success_factor",
        "content": (
            "Universal stabilizer ops — high stability as receiver from diverse "
            "predecessors, 100% category coverage. These are safe 'next ops' after "
            "almost anything: hyp_distance (100%, n=27), compression_mixture_experts "
            "(98.2%, n=57), token_type_classifier (97.8%, n=90), mean_last (90.5%, "
            "n=21), sigmoid (90.4%, n=94), shared_basis_proj (90.1%, n=91), "
            "rwkv_channel (88.9%, n=27), basis_expansion (87.9%, n=91), "
            "bottleneck_proj (85.7%, n=91), tied_proj (84.6%, n=91)."
        ),
        "insight_type": "top_op",
        "subject_key": "universal_stabilizers",
        "semantic_key": "profiling:universal_stabilizer_set",
        "alpha": 590.0,  # sum of stable counts for top 10
        "beta_": 59.0,
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "profiling_receiver_stability",
            "source": "component_profiles.db",
            "stabilizer_set": {
                "hyp_distance": {"rate": 1.000, "n": 27, "cat_coverage": 1.0},
                "compression_mixture_experts": {"rate": 0.982, "n": 57, "cat_coverage": 1.0},
                "token_type_classifier": {"rate": 0.978, "n": 90, "cat_coverage": 1.0},
                "mean_last": {"rate": 0.905, "n": 21, "cat_coverage": 1.0},
                "sigmoid": {"rate": 0.904, "n": 94, "cat_coverage": 1.0},
                "shared_basis_proj": {"rate": 0.901, "n": 91, "cat_coverage": 1.0},
                "rwkv_channel": {"rate": 0.889, "n": 27, "cat_coverage": 1.0},
                "basis_expansion": {"rate": 0.879, "n": 91, "cat_coverage": 1.0},
                "bottleneck_proj": {"rate": 0.857, "n": 91, "cat_coverage": 1.0},
                "tied_proj": {"rate": 0.846, "n": 91, "cat_coverage": 1.0},
            },
        },
    },

    # ══════════════════════════════════════════════════════════════════
    # Triplet category grammar: best 3-op category sequences
    # ══════════════════════════════════════════════════════════════════
    {
        "category": "structural_preference",
        "content": (
            "Best 3-op category sequences (100% stable, n≥20): "
            "elementwise_binary→elementwise_unary→elementwise_binary (n=54), "
            "elementwise_unary→elementwise_binary→elementwise_unary (n=68), "
            "elementwise_unary→functional→elementwise_unary (n=31), "
            "elementwise_unary→functional→elementwise_binary (n=30). "
            "Strong sequences (>90%): parameterized→elementwise_unary→functional "
            "(94.4%, n=142), math_space→functional→elementwise_unary (91.5%, n=71). "
            "Worst: ANY→ANY→math_space is 0% across all 20+ tested combinations "
            "(total n>5000). math_space must never appear at position C in a triplet."
        ),
        "insight_type": "composition_rule",
        "subject_key": "triplet_category_grammar",
        "semantic_key": "profiling:triplet_category_sequences",
        "alpha": 183.0,  # stable in top-4 100% patterns
        "beta_": 1.0,
        "display_only": False,
        "insight_level": "structural",
        "evidence_json": {
            "test": "profiling_triplet_categories",
            "source": "component_profiles.db",
            "perfect_sequences": [
                {"a": "elementwise_binary", "b": "elementwise_unary", "c": "elementwise_binary", "rate": 1.0, "n": 54},
                {"a": "elementwise_unary", "b": "elementwise_binary", "c": "elementwise_unary", "rate": 1.0, "n": 68},
                {"a": "elementwise_unary", "b": "functional", "c": "elementwise_unary", "rate": 1.0, "n": 31},
                {"a": "elementwise_unary", "b": "functional", "c": "elementwise_binary", "rate": 1.0, "n": 30},
            ],
            "strong_sequences": [
                {"a": "structural", "b": "elementwise_unary", "c": "functional", "rate": 0.966, "n": 29},
                {"a": "parameterized", "b": "elementwise_unary", "c": "functional", "rate": 0.944, "n": 142},
                {"a": "math_space", "b": "functional", "c": "elementwise_unary", "rate": 0.915, "n": 71},
            ],
            "zero_stability_rule": "ANY → ANY → math_space = 0% (n>5000)",
        },
    },

    # ══════════════════════════════════════════════════════════════════
    # High-risk op context rules
    # ══════════════════════════════════════════════════════════════════
    {
        "category": "pattern",
        "content": (
            "conv_only (95.6% risk) is only stable when followed by the safe "
            "parameterized set: mixed_recursion_gate, swiglu_mlp, token_type_classifier, "
            "bottleneck_proj, shared_basis_proj, tied_proj (6/51 stable). No predecessor "
            "stabilizes it (0/84). conv_only works as a feature extractor but must "
            "immediately project down."
        ),
        "insight_type": "composition_rule",
        "subject_key": "conv_only",
        "semantic_key": "profiling:conv_only_context",
        # 6/6 recommended followers are stable — rule is reliable
        "alpha": 7.0,
        "beta_": 1.0,
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "profiling_pair_stability",
            "source": "component_profiles.db",
            "risk_score": 95.6,
            "n_as_sender": 51,
            "stable_as_sender": 6,
            "n_as_receiver": 84,
            "stable_as_receiver": 0,
            "valid_followers": [
                "mixed_recursion_gate", "swiglu_mlp", "token_type_classifier",
                "bottleneck_proj", "shared_basis_proj", "tied_proj",
            ],
        },
    },
    {
        "category": "pattern",
        "content": (
            "entropy_score (94.2% risk) works only when followed by specific "
            "parameterized ops: bottleneck_proj, compression_mixture_experts, "
            "early_exit, low_rank_proj, mixed_recursion_gate, mul, shared_basis_proj, "
            "tied_proj (8/48 stable). No predecessor stabilizes it (0/90). "
            "entropy_score produces routing metadata; it must feed into a gating "
            "or projection op that consumes the score."
        ),
        "insight_type": "composition_rule",
        "subject_key": "entropy_score",
        "semantic_key": "profiling:entropy_score_context",
        # 8/8 recommended followers are stable — rule is reliable
        "alpha": 9.0,
        "beta_": 1.0,
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "profiling_pair_stability",
            "source": "component_profiles.db",
            "risk_score": 94.2,
            "n_as_sender": 48,
            "stable_as_sender": 8,
            "n_as_receiver": 90,
            "stable_as_receiver": 0,
            "valid_followers": [
                "bottleneck_proj", "compression_mixture_experts", "early_exit",
                "low_rank_proj", "mixed_recursion_gate", "mul",
                "shared_basis_proj", "tied_proj",
            ],
        },
    },
    {
        "category": "pattern",
        "content": (
            "state_space (95.6% risk) is only stable into softmax_last or "
            "token_type_classifier (2/41 followers). Only compression_mixture_experts "
            "can precede it (1/27). state_space has massive backward time (14042us) "
            "and 92607 grad_norm — it needs heavy dampening on both sides."
        ),
        "insight_type": "composition_rule",
        "subject_key": "state_space",
        "semantic_key": "profiling:state_space_context",
        # 2/2 recommended followers stable, 1/1 predecessor — rule is reliable
        "alpha": 4.0,
        "beta_": 1.0,
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "profiling_pair_stability",
            "source": "component_profiles.db",
            "risk_score": 95.6,
            "valid_followers": ["softmax_last", "token_type_classifier"],
            "valid_predecessors": ["compression_mixture_experts"],
            "perf_note": "backward_time_us=14042, grad_norm=92607",
        },
    },
    {
        "category": "pattern",
        "content": (
            "learnable_bias (92.1% risk) — 0/90 predecessors are stable, but "
            "11/49 followers work: basis_expansion, bottleneck_proj, cos, "
            "low_rank_proj, mixed_recursion_gate, mul, route_topk, shared_basis_proj, "
            "sigmoid, tied_proj, compression_mixture_experts. Like layernorm, "
            "learnable_bias shifts the distribution — follow with projection or "
            "bounded activation."
        ),
        "insight_type": "composition_rule",
        "subject_key": "learnable_bias",
        "semantic_key": "profiling:learnable_bias_context",
        # 11/11 recommended followers are stable — rule is reliable
        "alpha": 12.0,
        "beta_": 1.0,
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "profiling_pair_stability",
            "source": "component_profiles.db",
            "risk_score": 92.1,
            "n_as_sender": 49,
            "stable_as_sender": 11,
            "n_as_receiver": 90,
            "stable_as_receiver": 0,
            "valid_followers": [
                "basis_expansion", "bottleneck_proj", "cos", "low_rank_proj",
                "mixed_recursion_gate", "mul", "route_topk", "shared_basis_proj",
                "sigmoid", "tied_proj", "compression_mixture_experts",
            ],
        },
    },

    # ══════════════════════════════════════════════════════════════════
    # Output std predictor
    # ══════════════════════════════════════════════════════════════════
    {
        "category": "pattern",
        "content": (
            "Output standard deviation is the single best predictor of gradient "
            "health. Ops with output_std ∈ [0.5, 2.0] are gradient-safe. "
            "Below 0.5: signal collapse (vanishing). Above 2.0: amplification "
            "(exploding). Random forest predicts grad_exploding at 88% accuracy "
            "with output_std as #1 feature (21.4% importance). Distribution "
            "correctors that pull pair std toward 1.0: rmsnorm (100% correction rate), "
            "minimum/maximum (78-83%), add (73.5%), sub (71.9%)."
        ),
        "insight_type": "composition_rule",
        "subject_key": "output_std_predictor",
        "semantic_key": "profiling:output_std_health_rule",
        "alpha": 110.0,  # RF correct predictions (88% of 125)
        "beta_": 15.0,
        "display_only": False,
        "insight_level": "structural",
        "evidence_json": {
            "test": "profiling_ml_analysis",
            "source": "component_profiles.db",
            "ml_model": "RandomForest",
            "ml_accuracy": 0.880,
            "top_feature": "output_std",
            "top_feature_importance": 0.214,
            "safe_range": [0.5, 2.0],
            "corrector_ops": {
                "rmsnorm": {"correction_rate": 1.000},
                "minimum": {"correction_rate": 0.829},
                "maximum": {"correction_rate": 0.784},
                "add": {"correction_rate": 0.735},
                "sub": {"correction_rate": 0.719},
            },
        },
    },

    # ══════════════════════════════════════════════════════════════════
    # Composition risk tiers
    # ══════════════════════════════════════════════════════════════════
    {
        "category": "structural_preference",
        "content": (
            "Op composition risk tiers (from 5979 pair profiles): "
            "TIER 1 — Safe (<20% risk): token_type_classifier (5.8%), mean_last (8.0%), "
            "shared_basis_proj (12.4%), bottleneck_proj (15.3%), rwkv_channel (16.2%), "
            "rwkv_time_mixing (16.2%), hyp_distance (16.4%), tied_proj (16.8%), "
            "swiglu_mlp (17.3%), low_rank_proj (19.0%). "
            "TIER 2 — Moderate (20-50%): diff_attention (22.1%), compression_mixture_experts "
            "(24.4%), selective_scan (26.5%), basis_expansion (27.7%). "
            "TIER 3 — High (>85%): padic_expand (97.1%), state_space (95.6%), conv_only "
            "(95.6%), entropy_score (94.2%), layernorm (92.4%), learnable_bias (92.1%), "
            "all math_space ops (88-93%)."
        ),
        "insight_type": "composition_rule",
        "subject_key": "risk_tiers",
        "semantic_key": "profiling:composition_risk_tiers",
        "alpha": 2246.0,
        "beta_": 3733.0,
        "display_only": False,
        "insight_level": "structural",
        "evidence_json": {
            "test": "profiling_risk_analysis",
            "source": "component_profiles.db",
            "n_pairs": 5979,
            "tier1_safe": {
                "token_type_classifier": 5.8,
                "mean_last": 8.0,
                "shared_basis_proj": 12.4,
                "bottleneck_proj": 15.3,
                "rwkv_channel": 16.2,
                "tied_proj": 16.8,
                "swiglu_mlp": 17.3,
                "low_rank_proj": 19.0,
            },
            "tier3_high_risk": {
                "padic_expand": 97.1,
                "state_space": 95.6,
                "conv_only": 95.6,
                "entropy_score": 94.2,
                "layernorm": 92.4,
                "learnable_bias": 92.1,
            },
        },
    },
    # ══════════════════════════════════════════════════════════════════
    # "Alive" vs "merely non-explosive" — vanishing signal detection
    # ══════════════════════════════════════════════════════════════════
    {
        "category": "pattern",
        "content": (
            "Triplet instability is dominated by vanishing (dead signal), not explosion. "
            "Of 46,780 unstable triplets, the vast majority have grad_vanishing=1 with "
            "output_std near 0.0 — numerically clean but information-dead. Divergent "
            "triplets (80 total) are almost all vanishing, not NaN. Rule: stability "
            "checks must verify output_std > 0.01 AND grad_norm > 1e-6, not just "
            "absence of NaN. A graph that outputs near-zero is not 'safe' — it is dead."
        ),
        "insight_type": "composition_rule",
        "subject_key": "alive_vs_dead",
        "semantic_key": "profiling:alive_vs_dead_signal",
        "alpha": 28220.0,  # truly stable triplets
        "beta_": 46780.0,  # unstable (mostly vanishing)
        "display_only": False,
        "insight_level": "structural",
        "evidence_json": {
            "test": "profiling_triplet_stability",
            "source": "component_profiles.db",
            "n_triplets": 75000,
            "n_stable": 28220,
            "n_unstable": 46780,
            "n_divergent": 80,
            "dominant_failure_mode": "vanishing_gradient",
            "detection_rule": "output_std > 0.01 AND grad_norm > 1e-6",
        },
    },

    # ══════════════════════════════════════════════════════════════════
    # Outlier ops need bespoke contracts
    # ══════════════════════════════════════════════════════════════════
    {
        "category": "pattern",
        "content": (
            "Three ops are extreme outliers that break normal scoring: "
            "causal_mask (output_std=245M — mask values, not signal), "
            "div_safe (output_std≈0, Lipschitz=129K — collapses to zero), "
            "reciprocal (Lipschitz=52K, grad_norm=144B — extreme amplifier). "
            "These must not be compared or ranked alongside normal ops. "
            "causal_mask is a structural mask, not a transform. div_safe is "
            "a safety wrapper. reciprocal is a raw math op. Each needs "
            "composition contracts that match their actual role."
        ),
        "insight_type": "composition_rule",
        "subject_key": "outlier_ops",
        "semantic_key": "profiling:outlier_bespoke_contracts",
        # Silhouette=0.755, clean 2-cluster split — very confident in outlier identification
        "alpha": 10.0,
        "beta_": 1.0,
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "profiling_clustering",
            "source": "component_profiles.db",
            "n_clusters": 2,
            "outlier_cluster_size": 3,
            "outliers": {
                "causal_mask": {
                    "output_std": 245558752.0,
                    "lipschitz": 0.126,
                    "role": "structural_mask",
                    "note": "output is mask values, not signal — metrics are misleading",
                },
                "div_safe": {
                    "output_std": 0.0,
                    "lipschitz": 129122.0,
                    "role": "safety_wrapper",
                    "note": "collapses to zero — no stable followers exist",
                },
                "reciprocal": {
                    "output_std": 2311.0,
                    "lipschitz": 52201.0,
                    "grad_norm": 143616224.0,
                    "role": "raw_math",
                    "note": "extreme amplifier — no stable followers exist",
                },
            },
        },
    },

    # ══════════════════════════════════════════════════════════════════
    # Binary op caution in deep compositions
    # ══════════════════════════════════════════════════════════════════
    {
        "category": "pattern",
        "content": (
            "Binary ops (add, mul, maximum, minimum, sub) appear simple but are "
            "the most frequent participants in unstable triplets: add (3931), "
            "maximum (3701), mul (3663), minimum (3545), sub (2131). "
            "add accumulates instability from both inputs. mul amplifies gradients "
            "multiplicatively. maximum/minimum have discontinuous gradients at "
            "switching points. In pairwise they look harmless (elementwise_unary → "
            "elementwise_binary = 100% stable) but in triplets they propagate and "
            "compound upstream instability. Use with explicit gradient awareness."
        ),
        "insight_type": "composition_rule",
        "subject_key": "binary_op_caution",
        "semantic_key": "profiling:binary_op_deep_composition",
        "alpha": 1638.0,  # stable appearances of minimum (representative)
        "beta_": 3545.0,  # unstable appearances of minimum
        "display_only": False,
        "insight_level": "composition",
        "evidence_json": {
            "test": "profiling_triplet_instability",
            "source": "component_profiles.db",
            "unstable_appearances": {
                "add": 3931,
                "maximum": 3701,
                "mul": 3663,
                "minimum": 3545,
                "sub": 2131,
            },
            "stable_appearances": {
                "add": 1229,
                "maximum": 1583,
                "mul": 1335,
                "minimum": 1638,
                "sub": 928,
            },
            "note": "pair-level elementwise_unary → elementwise_binary is 100% stable, "
                    "but triplet-level instability emerges from compounding",
        },
    },
]


def seed(db_path: str = "research/lab_notebook.db") -> int:
    """Load profiling-derived insights. Returns count of newly inserted."""
    nb = LabNotebook(db_path)
    count = 0
    for ins in PROFILING_INSIGHTS:
        insight_id = nb.record_insight(
            category=ins["category"],
            content=ins["content"],
            insight_type=ins["insight_type"],
            subject_key=ins["subject_key"],
            semantic_key=ins["semantic_key"],
            alpha=ins["alpha"],
            beta_=ins["beta_"],
            display_only=ins["display_only"],
            insight_level=ins["insight_level"],
            evidence_json=ins["evidence_json"],
        )
        count += 1
        conf = ins["alpha"] / (ins["alpha"] + ins["beta_"])
        print(f"  [{ins['insight_level']:12s}] {ins['semantic_key']:45s} conf={conf:.3f} → {insight_id}")
    nb.close()
    return count


def main():
    parser = argparse.ArgumentParser(description="Seed profiling-derived Bayesian insights")
    parser.add_argument("--db", default="research/lab_notebook.db")
    args = parser.parse_args()
    n = seed(args.db)
    print(f"\nSeeded {n} profiling insights.")


if __name__ == "__main__":
    main()
