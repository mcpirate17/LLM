"""
Motif-Based Compositional Grammar

Generates valid computation graphs by composing validated motifs
into structural templates. This replaces the old random-walk grammar
that produced 93% non-viable "op soup" architectures.

Architecture:
  1. Pick 1-3 templates (weighted by success priors)
  2. For each template slot, pick a motif from the compatible class
  3. Compose templates into a single ComputationGraph
  4. Validate output shape and add residual connection

The grammar has constraints that make programs VALID (shapes compose,
gradient flows, bounded depth/params) but not necessarily USEFUL.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

from research.defaults import MODEL_DIM
from .graph import ComputationGraph, OpNode
from .grammar_support import (
    EfficiencyPrior,
    ROUTING_COMPRESSION_MOE_OPS,
    check_graph_space_consistency,
    check_shape_compat,
)
from .grammar_defaults import default_category_weights
from .primitives import (
    PRIMITIVE_REGISTRY,
    REQUIRES_RESIDUAL_BYPASS,
    get_wiring_rule,
    validate_wiring,
)
from .templates import (
    apply_template,
)
from ._dynamic_template_branch import maybe_apply_dynamic_template
from .template_rules import validate_template_graph
from .validator import validate_graph
from .native_template_selection import (
    TemplateWeightOverrides,
    make_template_weight_overrides,
)
from ._routing_capable_manifest import (
    ROUTING_CAPABLE_TEMPLATE_NAMES as _ROUTING_CAPABLE_TEMPLATE_NAMES,
)
from .generation_runtime import (
    GenerationRuntimeContext as _GenerationRuntimeContext,
    normalize_generation_config as _normalize_generation_config,
    runtime_context_for_config as _runtime_context_for_config,
)

# Alias for backward compatibility — some test files import Node from grammar
Node = OpNode
_check_graph_space_consistency = check_graph_space_consistency


@dataclass
class GrammarConfig:
    """Configuration for the graph generator."""

    model_dim: int = MODEL_DIM
    max_depth: int = 18  # depth budget (triple-lane templates need ~12-14)
    max_width: int = 4  # max parallel paths (2 or 3 way splits)
    max_ops: int = 24  # max total operations (triple-lane split3 needs ~21)
    residual_prob: float = 0.7  # probability of residual connection
    split_prob: float = 0.3  # probability of branching into parallel paths
    min_splits: int = 0  # minimum number of split-merge blocks to force
    three_way_split_prob: float = 0.0  # probability of 3-way split (vs 2-way)
    branch_depth: int = 1  # depth of subgraph processing on each branch
    max_recursion_depth: int = 4  # iteration cap for recursive ops
    stability_check: bool = True  # validate architectures before compilation
    merge_prob: float = 0.4  # probability of merging paths
    risky_op_prob: float = 0.5  # probability of using numerically risky ops
    freq_domain_prob: float = 0.15  # probability of FFT detour
    # Category weights (higher = more likely to be chosen)
    category_weights: Dict[str, float] = field(default_factory=default_category_weights)
    # Per-op weight multipliers
    op_weights: Dict[str, float] = field(default_factory=dict)

    # Structured Sparsity Constraints (Z7)
    structured_sparsity_bias: float = 0.3
    enforce_block_size: Optional[int] = None
    min_block_density: float = 0.05
    max_block_density: float = 0.5

    # Hyperbolic Promotion (Phase 3)
    hyperbolic_promotion_threshold: float = 0.6
    hyperbolic_boost_factor: float = 3.0
    _hierarchy_fitness: Optional[float] = None

    # ── DB-driven template weights ──────────────────────────────────
    # When True, loads template weights from template_stats DB table
    # (blended with static defaults). Requires backfill_stats.py to have run.
    use_db_weights: bool = True

    # ── Motif-based grammar config (Phase 6) ────────────────────────
    # Template selection weights (template_name → weight).
    # If empty, uses DEFAULT_TEMPLATE_WEIGHTS.
    template_weights: Dict[str, float] = field(default_factory=dict)
    # Motif selection weights (motif_name → weight).
    # If empty, uses motif's lift score as weight.
    motif_weights: Dict[str, float] = field(default_factory=dict)
    # Slot-specific motif priors derived from observability. Keys are
    # ``template.slotN`` identifiers and values are per-motif multipliers.
    slot_motif_weight_multipliers: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    # Slot-specific motif denylist derived from repeated toxic/failing slots.
    slot_motif_denylist: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    # Number of templates to compose per graph (1-3)
    composition_depth: int = 3  # Minimum template blocks per graph

    # ── Under-observed component exploration ──────────────────────────
    # Op names to boost during graph generation (under-observed ops).
    # Each op's containing motif gets weight multiplied by boost_factor.
    exploration_targets: FrozenSet[str] = field(default_factory=frozenset)
    exploration_boost_factor: float = 4.0  # Weight multiplier for target motifs

    # ── Wildcard slot exploration ────────────────────────────────────
    # Fraction of slots that proactively accept any motif class (exploration).
    # Also used as fallback when a slot's prescribed classes yield zero candidates.
    wildcard_slot_prob: float = 0.15

    # ── Template exploration budget ────────────────────────────────────
    # Fraction of template picks that ignore weights and select uniformly
    # from ALL templates (including zero-weighted). Ensures every template
    # gets coverage regardless of routing_mandatory or weight settings.
    template_exploration_budget: float = 0.10

    # ── Phase B (2026-05-04) — dynamically learned slots ─────────────
    # When True, _pick_compatible_motif consults the slot_constraints loader
    # and narrows the allowed motif_class tuple to those that empirically
    # pass for (template, slot_index) in the cohort. Hardcoded class tuples
    # remain as fallback when the meta DB is unavailable or no qualifying
    # data exists. Off by default; enable per-config to A/B against the
    # static-allow-list baseline.
    use_derived_slot_classes: bool = False
    # Propagate advisory AR/binding overlay opt-in to generated graph metadata.
    # The overlay itself remains outside core template/routing audit fields.
    ar_binding_overlay_enabled: bool = False
    # Optional advisory routing-decision prior. Disabled by default; when
    # enabled, generation loads a compact offline artifact and biases only
    # sample_routing_choice calls with matching (template, decision, value)
    # evidence. Missing or invalid artifacts fail closed to neutral choices.
    use_routing_decision_priors: bool = False
    routing_decision_prior_path: str = (
        "research/artifacts/routing_decision_priors/latest.json"
    )
    routing_decision_prior_strength: float = 1.0
    # Descriptor-backed dynamic templates. Enabled by default (2026-05-11):
    # the registry only consumes pre-validated mined chains, lowers them with
    # the same context-rule guards the static templates obey, and falls back
    # to the static pool on any per-attempt failure. Disable by setting
    # ``use_dynamic_template_candidates=False`` for ablation runs.
    use_dynamic_template_candidates: bool = True
    dynamic_template_candidate_path: str = (
        "research/notes/dynamic_component_candidates.json"
    )
    dynamic_template_candidate_prob: float = 0.10
    dynamic_template_candidate_strength: float = 1.0
    dynamic_template_max_candidates: int = 32
    dynamic_template_min_lowered_ops: int = 8
    # Provenance string for the use_derived_slot_classes assignment, set by
    # the A/B resolver in _slot_constraints_loader.resolve_slot_class_strategy.
    # Persisted to graph.metadata so post-hoc analysis can compare cohorts.
    # Values include "explicit_config", "strategy_derived", "strategy_static",
    # "strategy_ab_50_50". Empty string means the field was never set.
    slot_strategy_reason: str = ""

    # ── Phase C (2026-05-04) — trial template A/B harness ────────────
    # When non-empty, the exploration draw forces a uniform pick from this
    # list rather than the global weighted_pool. Picks are tagged with
    # graph.metadata['_template_trial'] = True so downstream aggregators
    # can split sa_pass stats by trial vs production.
    trial_template_names: Tuple[str, ...] = ()

    # Force every graph to use this specific template (bypass pick_template).
    # Set to a template name from TEMPLATES dict, e.g. "transformer_block".
    forced_template: Optional[str] = None

    # ── Routing-First Config (Phase 2) ────────────────────────────────
    routing_mandatory: bool = True  # Force routing structure in every graph
    routing_min_lanes: int = 2  # Minimum routing lanes (2 or 3)
    difficulty_scorer_type: str = "entropy"  # "entropy" or "learned"

    # ── Capability-First Config ───────────────────────────────────────
    # When True, screening requires at least one content-addressed op
    # (attention family or bare matmul/outer_product/gather_topk/
    # cosine_similarity). Used by ``capability_first`` preset to stop the
    # search burning investigation compute on retrieval-dead trunks.
    binding_capable_required: bool = False

    def update_bias(self, delta: float):
        """Adjust structured sparsity bias."""
        self.structured_sparsity_bias = max(
            0.0, min(1.0, self.structured_sparsity_bias + delta)
        )

    @classmethod
    def efficient(cls, model_dim: int = 256) -> "GrammarConfig":
        """Config tuned for efficiency-first architecture search (>5x GPT-2)."""
        return cls(
            model_dim=model_dim,
            max_depth=14,
            max_ops=24,
            residual_prob=0.7,
            split_prob=0.3,
            merge_prob=0.4,
            risky_op_prob=0.5,
            structured_sparsity_bias=0.8,
            category_weights={
                "elementwise_unary": 1.0,
                "elementwise_binary": 1.0,
                "reduction": 0.5,
                "linear_algebra": 1.0,
                "structural": 1.0,
                "parameterized": 2.0,
                "mixing": 2.0,
                "sequence": 1.0,
                "frequency": 0.3,
                "math_space": 1.0,
                "functional": 4.0,
            },
            template_weights={
                "sparse_moe_block": 5.0,
                "routed_bottleneck": 4.0,
                "token_merge_block": 4.0,
                "conditional_compute": 3.5,
                "difficulty_routed_block": 5.0,
                "three_lane_adaptive": 5.0,
                "cascaded_early_exit": 4.5,
                "recursive_depth_router": 4.5,
                "sparse_ffn": 3.0,
                "moe": 3.0,
                "bottleneck": 2.5,
                "transformer_block": 0.5,
                "residual_block": 1.0,
                "sequential": 0.5,
                "parallel_split": 0.5,
                "hybrid_parallel": 1.0,
                "gated_residual": 1.5,
                "dense_cascade": 0.3,
            },
            op_weights={
                "nm_sparse_linear": 4.0,
                "moe_topk": 4.0,
                "token_entropy": 3.5,
                "adjacent_token_merge": 3.5,
                "ternary_projection": 3.5,
                "block_sparse_linear": 3.5,
                "semi_structured_2_4_linear": 3.0,
                "moe_2expert": 3.0,
                "gated_linear": 2.5,
                "swiglu_mlp": 2.0,
                # Quarantine: Python-loop ops without native kernels.
                # Avoids wasting screening GPU time on slow fallback paths.
                "tropical_router": 0.01,
                "tropical_moe": 0.01,
            },
        )

    @classmethod
    def exotic(cls, model_dim: int = 256) -> "GrammarConfig":
        """Config tuned for exotic architecture exploration."""
        return cls(
            model_dim=model_dim,
            max_depth=18,
            max_ops=24,
            split_prob=0.6,
            residual_prob=0.4,
            merge_prob=0.5,
            risky_op_prob=0.7,
            category_weights={
                "elementwise_unary": 1.5,
                "elementwise_binary": 1.0,
                "reduction": 0.8,
                "linear_algebra": 1.0,
                "structural": 1.0,
                "parameterized": 3.0,
                "mixing": 3.0,
                "sequence": 1.2,
                "frequency": 1.0,
                "math_space": 1.5,
                "functional": 3.5,
            },
            template_weights={
                "moe": 4.0,
                "hybrid_parallel": 3.0,
                "parallel_split": 3.0,
                "gated_residual": 2.5,
                "sparse_ffn": 2.5,
                "difficulty_routed_block": 4.0,
                "three_lane_adaptive": 4.0,
                "cascaded_early_exit": 3.5,
                "recursive_depth_router": 3.5,
                "transformer_block": 2.0,
                "residual_block": 1.5,
                "bottleneck": 2.0,
                "dense_cascade": 1.5,
                "sequential": 1.0,
            },
            op_weights={
                "feature_sparsity": 3.0,
                "gated_lane_blend": 3.0,
                "depth_gated_transform": 2.5,
                "depth_token_mask": 3.0,
                "confidence_token_gate": 2.0,
                "depth_weighted_proj": 3.0,
                "adjacent_token_merge": 2.0,
                "learned_token_gate": 2.0,
                "cheap_verify_blend": 2.0,
                "moe_topk": 3.0,
                "difficulty_blend_3way": 3.0,
                "score_depth_blend": 2.5,
                "relu_gated_moe": 2.5,
                "latent_attention_compressor": 3.0,
                "signal_conditioned_compression": 2.5,
                "adaptive_rank_gate": 2.0,
                "dual_compression_blend": 2.5,
                "token_class_proj": 2.5,
                "token_entropy": 2.5,
            },
        )

    @classmethod
    def routing_first(cls, model_dim: int = 256) -> "GrammarConfig":
        """Config that mandates routing structure in every generated graph.

        Template selection draws ONLY from routing templates. Every graph
        will have a difficulty scorer and differential compute paths.
        """
        from .templates import ROUTING_TEMPLATES

        # Zero-out non-routing templates, boost routing ones
        tpl_weights = {
            name: (5.0 if name in ROUTING_TEMPLATES else 0.0)
            for name in (
                "residual_block",
                "sequential",
                "transformer_block",
                "parallel_split",
                "bottleneck",
                "moe",
                "hybrid_parallel",
                "gated_residual",
                "dense_cascade",
                "sparse_ffn",
                "sparse_moe_block",
                "routed_bottleneck",
                "token_merge_block",
                "conditional_compute",
                "difficulty_routed_block",
                "three_lane_adaptive",
                "cascaded_early_exit",
                "recursive_depth_router",
            )
        }
        return cls(
            model_dim=model_dim,
            max_depth=18,
            max_ops=24,
            residual_prob=0.8,
            split_prob=0.3,
            risky_op_prob=0.5,
            routing_mandatory=True,
            routing_min_lanes=2,
            difficulty_scorer_type="entropy",
            category_weights={
                "elementwise_unary": 1.0,
                "elementwise_binary": 1.5,
                "reduction": 0.5,
                "linear_algebra": 1.0,
                "structural": 1.0,
                "parameterized": 3.0,
                "mixing": 2.0,
                "sequence": 1.0,
                "frequency": 0.3,
                "math_space": 1.0,
                "functional": 4.0,
            },
            template_weights=tpl_weights,
            op_weights={
                "token_entropy": 5.0,
                "difficulty_blend_3way": 4.0,
                "confidence_token_gate": 3.5,
                "learned_token_gate": 3.5,
                "depth_weighted_proj": 3.5,
                "moe_topk": 3.0,
                "moe_2expert": 3.0,
                "adjacent_token_merge": 3.0,
                "relu_gated_moe": 2.5,
                "swiglu_mlp": 2.0,
                "gated_linear": 2.0,
                # Quarantine: Python-loop ops without native kernels.
                "tropical_router": 0.01,
                "tropical_moe": 0.01,
            },
        )

    @classmethod
    def capability_first(cls, model_dim: int = 256) -> "GrammarConfig":
        """Preset that pressures the search toward trunk+sidecar graphs.

        Samples only from role-slot templates that wire an explicit
        exact-retrieval sidecar (matmul / gather_topk) merged into a
        compression trunk via a typed-entropy or sparse-router controller.
        Boosts retrieval-family ops (matmul, outer_product, gather_topk,
        cosine_similarity, token_type_classifier, token_entropy) so the
        motif slots inside the sidecar actually pick content-addressed
        primitives.

        Use this preset when seeding runs aimed at the full metric tuple
        (ppl + binding_screening_auc + induction + ar + hellaswag). The companion
        ``routing_first`` preset still exists for pure MoE/difficulty
        routing experiments.
        """
        from .templates import CAPABILITY_FIRST_TEMPLATES, DEFAULT_TEMPLATE_WEIGHTS

        # Promote capability-first templates; keep a small positive weight on
        # the core routing templates so composition can still interleave a
        # difficulty router, but let the role-slot templates dominate.
        tpl_weights: Dict[str, float] = {}
        for name in DEFAULT_TEMPLATE_WEIGHTS:
            if name in CAPABILITY_FIRST_TEMPLATES:
                tpl_weights[name] = 6.0
            else:
                tpl_weights[name] = 0.0
        # Keep a couple of known-good low-ppl trunks at minority weight so the
        # grammar can compose them before the retrieval sidecar. Without this
        # the trunk choice is fully owned by the role-slot template internals.
        for trunk_name, weight in (
            ("conv_residual_block", 1.5),
            ("state_space_block", 1.5),
            ("mamba_reference", 1.0),
            ("adaptive_conv_ffn", 1.5),
            ("adaptive_ssm_chain", 1.5),
        ):
            if trunk_name in tpl_weights:
                tpl_weights[trunk_name] = weight

        return cls(
            model_dim=model_dim,
            max_depth=18,
            max_ops=24,
            residual_prob=0.8,
            split_prob=0.3,
            risky_op_prob=0.5,
            routing_mandatory=True,
            routing_min_lanes=2,
            difficulty_scorer_type="entropy",
            binding_capable_required=True,
            category_weights={
                "elementwise_unary": 1.0,
                "elementwise_binary": 1.5,
                "reduction": 0.5,
                "linear_algebra": 2.5,  # matmul, outer_product, cosine_similarity
                "structural": 2.0,  # gather_topk lives here
                "parameterized": 2.5,
                "mixing": 2.0,
                "sequence": 1.0,
                "frequency": 0.3,
                "math_space": 2.0,
                "functional": 3.5,
            },
            template_weights=tpl_weights,
            op_weights={
                # Retrieval sidecar: pair these with routing signals.
                "matmul": 4.0,
                "outer_product": 3.5,
                "gather_topk": 4.0,
                "cosine_similarity": 3.5,
                "token_type_classifier": 4.0,
                "token_class_proj": 4.0,
                "token_entropy": 4.0,
                "entropy_score": 4.0,
                # Trunk: ppl-winner family.
                "conv1d_seq": 3.5,
                "selective_scan": 3.0,
                "swiglu_mlp": 2.5,
                "adjacent_token_merge": 3.0,
                "nm_sparse_linear": 3.0,
                "moe_2expert": 2.5,
                # Quarantine: Python-loop ops without native kernels.
                "tropical_router": 0.01,
                "tropical_moe": 0.01,
            },
        )

    @classmethod
    def exploration(
        cls,
        target_ops: FrozenSet[str],
        model_dim: int = 256,
        boost_factor: float = 4.0,
    ) -> "GrammarConfig":
        """Config that heavily boosts under-observed ops for evidence collection.

        Args:
            target_ops: Op names to boost (e.g. from DB query for <20 observations)
            boost_factor: Weight multiplier for motifs/templates containing targets
        """
        # Exploration targets may not be routing/compression/MoE ops,
        # so relax routing_mandatory to avoid rejecting valid exploration
        # graphs. The target ops are the priority here.
        return cls(
            model_dim=model_dim,
            max_depth=18,
            max_ops=24,
            residual_prob=0.6,
            split_prob=0.4,
            risky_op_prob=0.7,
            routing_mandatory=False,
            exploration_targets=target_ops,
            exploration_boost_factor=boost_factor,
        )


# ── DB-backed template weight loader ────────────────────────────────
_ROUTING_COMPRESSION_MOE_OPS = ROUTING_COMPRESSION_MOE_OPS


_ROUTING_CAPABLE_TEMPLATES_CACHE: Optional[FrozenSet[str]] = None


_ROUTING_PROBE_SEEDS: int = 8
_ROUTING_PROBE_MIN_HITS: int = 7  # ≥ 7/8 seeds must emit routing


def _probe_routing_capable_templates() -> FrozenSet[str]:
    """Probe templates that reliably emit a ROUTING_COMPRESSION_MOE op.

    This is intentionally kept out of the runtime path. It dry-runs every
    registered template across several seeds and is used by tests/audits to
    refresh ``_ROUTING_CAPABLE_TEMPLATE_NAMES`` when template behavior changes.
    A template qualifies only if it emits a routing op in at
    least ``_ROUTING_PROBE_MIN_HITS`` of ``_ROUTING_PROBE_SEEDS`` probes —
    templates that emit routing only stochastically via motif lottery would
    defeat the rescue since the actual composition picks its own seed.
    """
    # guardrail: allow-complexity - offline template audit helper, not generation.
    # Import here to avoid circular imports at module load.
    from .templates import TEMPLATES

    routing_capable: List[str] = []
    for tpl_name in TEMPLATES:
        hits = 0
        for seed in range(_ROUTING_PROBE_SEEDS):
            try:
                probe_graph = ComputationGraph(MODEL_DIM)
                probe_inp = probe_graph.add_input()
                apply_template(
                    probe_graph,
                    probe_inp,
                    random.Random(seed),
                    template_name=tpl_name,
                )
                ops = {
                    node.op_name
                    for node in probe_graph.nodes.values()
                    if not node.is_input
                }
                if ops & _ROUTING_COMPRESSION_MOE_OPS:
                    hits += 1
            except Exception:
                # One seed's probe failure doesn't disqualify the template;
                # continue and let the hit ratio decide.
                continue
        if hits >= _ROUTING_PROBE_MIN_HITS:
            routing_capable.append(tpl_name)

    return frozenset(routing_capable)


def _get_routing_capable_templates() -> FrozenSet[str]:
    """Return names of templates that reliably emit a ROUTING_COMPRESSION_MOE op.

    Runtime graph generation uses the maintained manifest rather than probing
    all templates on first use. The probe is still available for tests/audits,
    but not on the hot path.
    """
    global _ROUTING_CAPABLE_TEMPLATES_CACHE
    if _ROUTING_CAPABLE_TEMPLATES_CACHE is not None:
        return _ROUTING_CAPABLE_TEMPLATES_CACHE

    # Import here to avoid circular imports at module load.
    from .templates import TEMPLATES

    unknown = _ROUTING_CAPABLE_TEMPLATE_NAMES - set(TEMPLATES)
    if unknown:
        raise RuntimeError(
            "Routing-capable template manifest contains unknown templates: "
            f"{sorted(unknown)}"
        )

    _ROUTING_CAPABLE_TEMPLATES_CACHE = _ROUTING_CAPABLE_TEMPLATE_NAMES
    return _ROUTING_CAPABLE_TEMPLATES_CACHE


def _config_with_efficiency_prior(
    config: GrammarConfig, prior: Optional[EfficiencyPrior]
) -> GrammarConfig:
    """Apply learned efficiency bias to op weights without mutating the caller's config."""
    if prior is None:
        return config

    biased_weights = dict(config.op_weights)
    for op_name in PRIMITIVE_REGISTRY:
        if op_name == "input":
            continue
        bias = prior.get_bias(op_name)
        if bias != 1.0:
            biased_weights[op_name] = biased_weights.get(op_name, 1.0) * bias
    return replace(config, op_weights=biased_weights)


def generate_layer_graph(
    config: Optional[GrammarConfig] = None,
    seed: Optional[int] = None,
    *,
    validate: bool = True,
    _runtime_context: _GenerationRuntimeContext | None = None,
) -> ComputationGraph:
    """Generate a computation graph for a single layer.

    Uses motif-based compositional generation:
    1. Pick 1-2 structural templates
    2. Fill each template's slots with validated motifs
    3. Add residual connection
    4. Validate output shape
    """
    if config is None:
        config = GrammarConfig()

    config = _normalize_generation_config(config)
    runtime = _runtime_context or _runtime_context_for_config(config)

    rng = random.Random(seed)
    graph = ComputationGraph(config.model_dim)
    input_id = graph.add_input()

    # Determine composition depth (how many template blocks to stack)
    if config.composition_depth > 0:
        n_templates = config.composition_depth
    else:
        n_templates = rng.choices([1, 2, 3], weights=[3, 5, 2], k=1)[0]

    graph.metadata["context_rules_version"] = "low_s1_v1"
    if config.ar_binding_overlay_enabled:
        graph.metadata["ar_binding_overlay_enabled"] = True
    if config.wildcard_slot_prob > 0:
        graph.metadata["_wildcard_slot_prob"] = config.wildcard_slot_prob
    if runtime.slot_motif_weight_multipliers:
        graph.metadata["_slot_motif_weight_multipliers"] = (
            runtime.slot_motif_weight_multipliers
        )
    if runtime.slot_motif_denylist:
        graph.metadata["_slot_motif_denylist"] = runtime.slot_motif_denylist
    if runtime.slot_adaptations:
        graph.metadata["_slot_adaptations"] = runtime.slot_adaptations
    if runtime.effective_op_weights:
        graph.metadata["_op_weights"] = runtime.effective_op_weights
    if runtime.routing_decision_priors and runtime.routing_decision_priors.get(
        "loaded"
    ):
        graph._routing_decision_prior_state = {
            "prior": runtime.routing_decision_priors,
            "strength": max(0.0, float(config.routing_decision_prior_strength)),
        }
        graph.metadata["routing_decision_prior"] = {
            "enabled": True,
            "version": runtime.routing_decision_priors.get("version"),
            "path": str(config.routing_decision_prior_path),
            "strength": max(0.0, float(config.routing_decision_prior_strength)),
        }
    if runtime.dynamic_template_candidates:
        graph.metadata["dynamic_template_candidates"] = {
            "enabled": True,
            "path": str(config.dynamic_template_candidate_path),
            "count": len(runtime.dynamic_template_candidates),
            "min_lowered_ops": int(config.dynamic_template_min_lowered_ops),
            "prob": max(
                0.0,
                min(1.0, float(config.dynamic_template_candidate_prob)),
            ),
            "strength": max(0.0, float(config.dynamic_template_candidate_strength)),
        }

    current = input_id
    for t_idx in range(n_templates):
        _iter_weights = (
            runtime.first_tpl_weights
            if (t_idx == 0 and runtime.use_efficiency_first)
            else runtime.tpl_weights
        )
        _iter_allowed_names = None

        # routing_mandatory bias: on the FIRST slot (cheap, fits budget), pick
        # from templates that reliably emit a routing/compression/MoE op. The
        # remaining slots are unrestricted, so downstream composition can still
        # add attention / FFN / etc. Without this, ~(1 - routing_frac)^n of
        # compositions drop into the hard-reject path post-build. Putting the
        # bias on the LAST slot instead creates a budget collision because the
        # routing templates are typically larger. Audit fix 2026-04-17.
        #
        # Zero out (not drop) non-routing templates: pick_template falls back
        # to DEFAULT_TEMPLATE_WEIGHTS for any key not present in the passed
        # dict, so a drop-based filter gets silently re-expanded to the full
        # registry. An explicit zero weight does stick.
        if config.routing_mandatory and t_idx == 0 and not config.forced_template:
            _iter_allowed_names = _get_routing_capable_templates()

        # Depth-aware template biasing: early blocks favor FFN/conv,
        # late blocks favor attention/SSM (per GPT-2 layer importance research).
        # Note: "attn" matches new attention templates (attn_*).
        if _iter_weights and n_templates > 1:
            _iter_weights = _depth_adjusted_template_weights(
                _iter_weights, t_idx, n_templates
            )

        max_attempts = (
            4
            if config.routing_mandatory and t_idx == 0 and not config.forced_template
            else 1
        )
        template_applied = False
        for _attempt in range(max_attempts):
            # Snapshot graph state for lightweight rollback instead of full copy.
            # Only need to track node IDs added and metadata changes.
            prev_next_id = graph._next_id
            prev_output_id = graph._output_node_id
            prev_metadata = copy.deepcopy(graph.metadata)
            graph._cache.clear()

            # Phase B.2 — propagate the dynamic-slot flag through metadata so the
            # picker (_pick_compatible_motif) can read it without altering the
            # apply_template signature.
            graph.metadata["_use_derived_slot_classes"] = bool(
                config.use_derived_slot_classes
            )
            if getattr(config, "slot_strategy_reason", ""):
                graph.metadata["slot_strategy_reason"] = str(
                    config.slot_strategy_reason
                )

            dynamic_trial, dynamic_used = maybe_apply_dynamic_template(
                graph=graph,
                current=current,
                rng=rng,
                runtime=runtime,
                config=config,
                t_idx=t_idx,
                prev_next_id=prev_next_id,
                prev_output_id=prev_output_id,
                prev_metadata=prev_metadata,
            )
            if dynamic_used:
                trial_current = dynamic_trial
            else:
                trial_current = apply_template(
                    graph,
                    current,
                    rng,
                    template_name=config.forced_template,
                    template_weights=_iter_weights,
                    motif_weights=runtime.motif_weights,
                    op_weights=runtime.effective_op_weights,
                    exploration_budget=config.template_exploration_budget,
                    allowed_template_names=_iter_allowed_names,
                    trial_template_names=config.trial_template_names,
                )

            # depth() returns 0 without a set output — point it at the trial tail
            # so the budget check sees the actual longest input-to-trail path.
            # Skip for very tight budgets where a real depth signal starves the
            # grammar: the validator's +2 headroom is the sole depth backstop
            # in that regime.
            if config.max_depth > 10:
                graph._output_node_id = trial_current
                graph._cache.pop("depth", None)

            if _graph_exceeds_final_budget(graph, config):
                # Roll back only the suffix allocated by this template. Node IDs
                # are monotonic, so there is no reason to rebuild a full key set.
                for nid in range(prev_next_id, graph._next_id):
                    del graph.nodes[nid]
                graph._next_id = prev_next_id
                graph._output_node_id = prev_output_id
                graph.metadata = prev_metadata
                graph._cache.clear()
                continue
            current = trial_current
            template_applied = True
            break
        if not template_applied:
            break

    # Record depth placement for leaderboard analysis
    tpls_used = graph.metadata.get("templates_used", [])
    if tpls_used and n_templates > 1:
        graph.metadata["layer_depths"] = {
            f"block_{i}": tpls_used[i] if i < len(tpls_used) else None
            for i in range(n_templates)
        }

    skip_global_decorators = bool(graph.metadata.get("_skip_global_decorators", False))

    # Optional spectral filter injection (driven by freq_domain_prob)
    # Wrapped in residual: FFT numerical drift is unrecoverable without skip.
    # Skip when the template already owns its merge/post path and sits near the
    # screening op budget.
    if (
        not skip_global_decorators
        and rng.random() < config.freq_domain_prob
        and "spectral_filter" in PRIMITIVE_REGISTRY
        and not _graph_exceeds_final_budget(graph, config)
    ):
        pre_spectral = current
        try:
            norm_id = graph.add_op("rmsnorm", [current])
            sf_id = graph.add_op("spectral_filter", [norm_id])
            current = graph.add_op("add", [pre_spectral, sf_id])
        except ValueError:
            # Shape mismatch in residual wrap — revert to pre-spectral state.
            # Bare spectral_filter without residual bypass causes unrecoverable
            # numerical drift, so we must not keep it.
            graph.prune_unreachable_nodes()
            current = pre_spectral
            graph.metadata["_grammar_spectral_fallback"] = True

    # Ensure output shape is (B, S, D)
    result_shape = graph.nodes[current].output_shape
    if result_shape.dim != config.model_dim:
        current = graph.add_op(
            "linear_proj", [current], config={"out_dim": config.model_dim}
        )
        graph.metadata["_grammar_output_dim_coerced"] = True

    # Optional outer residual connection (if not already added by template)
    # Check if the last op is already an add with input_id
    last_node = graph.nodes[current]
    has_outer_residual = last_node.op_name == "add" and input_id in last_node.input_ids
    if (
        not skip_global_decorators
        and not has_outer_residual
        and graph.n_ops() < config.max_ops
        and rng.random() < config.residual_prob
    ):
        try:
            current = graph.add_op("add", [input_id, current])
        except ValueError:
            # Shape mismatch on optional outer residual — keep the non-residual
            # path explicitly rather than swallowing the failure.
            current = current

    # Final LayerNorm before output head — every serious LM (GPT-2, LLaMA,
    # Mamba) has this. Without it logit scale depends on last block's variance.
    # Worth ~5-15% perplexity improvement.
    last_op = graph.nodes[current].op_name
    if last_op not in ("rmsnorm", "layernorm"):
        current = graph.add_op("rmsnorm", [current])

    graph.set_output(current)

    # Prune dead branches
    graph.prune_unreachable_nodes()

    if validate:
        _validate_graph(graph, config)

    return graph


def random_graph(
    config: Optional[GrammarConfig] = None,
    seed: Optional[int] = None,
) -> ComputationGraph:
    """Backward-compatible alias for older callers."""

    return generate_layer_graph(config=config, seed=seed)


def _validate_graph(
    graph: ComputationGraph,
    config: GrammarConfig,
    *,
    dim_flow_inputs: object | None = None,
    packed_validation: object | None = None,
) -> None:
    """Validate a generated graph and raise ValueError if invalid."""
    # Allow +2 depth headroom for multi-step motifs (e.g., 3-4 step math-space
    # motifs that include norm+op+proj) which can push templates slightly over.
    max_params = 12 * 4 * config.model_dim * config.model_dim
    validation_kwargs = {
        "max_ops": config.max_ops,
        "max_depth": config.max_depth + 2,
        "min_splits": config.min_splits,
        "max_params": max_params,
    }
    if dim_flow_inputs is not None:
        validation_kwargs["dim_flow_inputs"] = dim_flow_inputs
    if packed_validation is not None:
        validation_kwargs["packed_validation"] = packed_validation
    result = validate_graph(graph, **validation_kwargs)
    if not result.valid:
        raise ValueError(
            result.errors[0] if result.errors else "Graph validation failed"
        )

    # Algebraic space consistency check — reject graphs that mix
    # incompatible mathematical spaces (e.g., tropical after poincaré).
    space_err = check_graph_space_consistency(graph)
    if space_err is not None:
        raise ValueError(space_err)

    # Routing-mandatory check: every graph must have routing, compression, or MoE.
    # Skipped when forced_template is set or exploration budget selected the template.
    _skip_routing_check = config.forced_template or graph.metadata.get(
        "_template_exploration_used"
    )
    if config.routing_mandatory and not _skip_routing_check:
        op_names = {n.op_name for n in graph.nodes.values() if not n.is_input}
        if not op_names & _ROUTING_COMPRESSION_MOE_OPS:
            raise ValueError(
                "routing_mandatory=True but graph has no routing/compression/MoE ops"
            )

    # Depth constraint check: reject ops placed before their required layer depth.
    # The requirement lives in wiring rules; mutating the graph here leaves stale
    # cached IR/metrics behind and turns validation into silent graph rewriting.
    for nid, node in graph.nodes.items():
        if node.is_input:
            continue
        depth_rule = get_wiring_rule(node.op_name) or {}
        min_layer_depth = int(depth_rule.get("min_layer_depth", 0))
        if min_layer_depth > 0:
            if node.depth < min_layer_depth:
                raise ValueError(
                    f"{node.op_name} (id={nid}) placed at depth {node.depth} "
                    f"before min_layer_depth={min_layer_depth}"
                )

    # Residual bypass check: ops in REQUIRES_RESIDUAL_BYPASS must have a
    # downstream add that also takes the op's input (residual connection).
    successors: Dict[int, List[int]] = graph.children_map()
    add_inputs_by_source: Dict[int, set[int]] = {}
    for other_nid, other_node in graph.nodes.items():
        if other_node.op_name == "add":
            add_inputs = set(other_node.input_ids)
            for source_id in add_inputs:
                add_inputs_by_source.setdefault(source_id, set()).update(add_inputs)

    for nid, node in graph.nodes.items():
        if node.is_input or node.op_name not in REQUIRES_RESIDUAL_BYPASS:
            continue
        node_inputs = set(node.input_ids)
        if not (add_inputs_by_source.get(nid, set()) & node_inputs):
            raise ValueError(
                f"{node.op_name} (id={nid}) requires residual bypass but none found"
            )

    # requires_residual_context: weaker than residual_bypass — the op's
    # output must reach SOME downstream `add` node (not necessarily one that
    # also takes the op's input), so the unbounded output rejoins a residual
    # stream rather than feeding raw into another transform. Built from
    # CONTEXT_RULES at module load. Audit fix 2026-04-17.
    from ._context_registry import REQUIRES_RESIDUAL_CONTEXT_OPS

    if REQUIRES_RESIDUAL_CONTEXT_OPS:
        # Per-node: does any descendant of `nid` participate as input to an
        # `add` op? Cheap BFS using the `successors` map already built above.
        add_consumers: set[int] = {
            other_nid
            for other_nid, other_node in graph.nodes.items()
            if other_node.op_name == "add"
        }
        for nid, node in graph.nodes.items():
            if (
                node.is_input
                or node.op_name not in REQUIRES_RESIDUAL_CONTEXT_OPS
                or node.op_name in REQUIRES_RESIDUAL_BYPASS
            ):
                continue
            # BFS forward from nid; any reachable add is sufficient context.
            seen: set[int] = {nid}
            queue: List[int] = [nid]
            reaches_add = False
            while queue:
                cur = queue.pop()
                if cur in add_consumers:
                    reaches_add = True
                    break
                for child in successors.get(cur, ()):
                    if child not in seen:
                        seen.add(child)
                        queue.append(child)
            if not reaches_add:
                raise ValueError(
                    f"{node.op_name} (id={nid}) requires residual context but "
                    f"no downstream add is reachable from its output"
                )

    # Op wiring constraint check: validate signal producer/consumer chains
    wiring_errors = validate_wiring(graph)
    if wiring_errors:
        raise ValueError(f"Wiring constraint violated: {wiring_errors[0]}")

    # Activation constraint check: reject activation placements that
    # empirically always diverge. Checks both directions:
    #   "before" — which ops may consume this activation's output
    #   "after"  — which ops must precede this activation (predecessor check)
    from .motifs import ACTIVATION_RULES, MATH_SPACE_RULES
    from .op_roles import get_role

    for nid in sorted(graph.nodes):
        node = graph.nodes[nid]
        if node.is_input:
            continue
        rules = ACTIVATION_RULES.get(node.op_name)
        if rules is None:
            continue

        # "before" check: reject invalid successors (e.g. sigmoid→add)
        before = rules.get("before")
        if before is not None:
            for other_nid in successors.get(nid, ()):
                other_node = graph.nodes[other_nid]
                if other_node.is_input:
                    continue
                if other_node.op_name not in before:
                    raise ValueError(
                        f"Activation constraint: {node.op_name} (id={nid}) "
                        f"→ {other_node.op_name} (id={other_nid}) is not allowed; "
                        f"valid successors: {before}"
                    )

        # "after" check: reject invalid predecessors (e.g. exp after unbounded op)
        after = rules.get("after")
        if after is not None:
            for parent_id in node.input_ids:
                parent = graph.nodes.get(parent_id)
                if parent is None or parent.is_input:
                    continue
                parent_role = get_role(parent.op_name)
                if parent.op_name not in after and parent_role not in after:
                    raise ValueError(
                        f"Activation constraint: {parent.op_name} (id={parent_id}) "
                        f"→ {node.op_name} (id={nid}) is not allowed; "
                        f"valid predecessors: {after}"
                    )

    # Math-space composition check: reject math-space ops whose required
    # predecessors (must_precede) are not satisfied. Templates auto-insert
    # rmsnorm, but free-form generation can skip it.
    for nid in sorted(graph.nodes):
        node = graph.nodes[nid]
        if node.is_input:
            continue
        ms_rules = MATH_SPACE_RULES.get(node.op_name)
        if ms_rules is None:
            continue

        # must_precede: at least one parent must be from the required set
        must_precede = ms_rules.get("must_precede")
        if must_precede is not None:
            has_valid_parent = False
            for parent_id in node.input_ids:
                parent = graph.nodes.get(parent_id)
                if parent is not None and parent.op_name in must_precede:
                    has_valid_parent = True
                    break
            if not has_valid_parent:
                raise ValueError(
                    f"Math-space constraint: {node.op_name} (id={nid}) "
                    f"requires predecessor from {must_precede}"
                )

        # must_follow: this op must come after specific ops
        must_follow = ms_rules.get("must_follow")
        if must_follow is not None:
            has_valid_parent = False
            for parent_id in node.input_ids:
                parent = graph.nodes.get(parent_id)
                if parent is not None and parent.op_name in must_follow:
                    has_valid_parent = True
                    break
            if not has_valid_parent:
                raise ValueError(
                    f"Math-space constraint: {node.op_name} (id={nid}) "
                    f"must follow one of {must_follow}"
                )

        # must_follow_with: a successor from this set must consume this op
        must_follow_with = ms_rules.get("must_follow_with")
        if must_follow_with is not None:
            if not any(
                not graph.nodes[other_nid].is_input
                and graph.nodes[other_nid].op_name in must_follow_with
                for other_nid in successors.get(nid, ())
            ):
                raise ValueError(
                    f"Math-space constraint: {node.op_name} (id={nid}) "
                    f"must be followed by one of {must_follow_with}"
                )

    # Template-level structural invariants are part of legality, not metadata-only
    # guidance. Preserve the warning payload for observability, but reject the
    # graph so template-invalid programs never count as valid survivors.
    tpl_errors = validate_template_graph(graph)
    if tpl_errors:
        for err in tpl_errors:
            logger.debug("template_rule: %s", err)
        graph.metadata["template_rule_warnings"] = tpl_errors
        raise ValueError(f"Template rule violations: {tpl_errors}")


_POST_BODY_OP_RESERVE = 3  # dim-coerce + outer residual + final rmsnorm
_POST_BODY_DEPTH_RESERVE = 3  # same three hops, worst-case sequential


@lru_cache(maxsize=256)
def _template_depth_tags(template_name: str) -> tuple[bool, bool, bool]:
    name = template_name.lower()
    is_attn = "attn" in name or "attention" in name or "transformer" in name
    is_early = "conv" in name or "ffn" in name or "bottleneck" in name
    is_mamba = "mamba" in name
    return is_early, is_attn, is_mamba


_DEPTH_WEIGHT_CACHE_MAX = 256
_DEPTH_WEIGHT_CACHE: dict[
    tuple[int, int, int], tuple[Mapping[str, float], TemplateWeightOverrides]
] = {}


def _depth_adjusted_template_weights(
    weights: Mapping[str, float],
    t_idx: int,
    n_templates: int,
) -> TemplateWeightOverrides:
    """Return cached depth-adjusted template weights for native selection."""
    # guardrail: allow-complexity - cached metadata transform over template weights.
    cache_key = (id(weights), int(t_idx), int(n_templates))
    cached = _DEPTH_WEIGHT_CACHE.get(cache_key)
    if cached is not None and cached[0] is weights:
        return cached[1]

    depth_ratio = t_idx / (n_templates - 1)
    items = []
    for tpl_name, base_w in weights.items():
        is_early_template, is_attn, is_mamba = _template_depth_tags(tpl_name)
        weight = float(base_w)
        if depth_ratio < 0.33:
            if is_early_template:
                weight *= 1.5
            elif is_attn:
                weight *= 0.85
        elif depth_ratio > 0.66:
            if is_attn or is_mamba:
                weight *= 1.5
            elif is_early_template:
                weight *= 0.6
        items.append((str(tpl_name), weight))

    prepared = make_template_weight_overrides(tuple(items))
    if len(_DEPTH_WEIGHT_CACHE) >= _DEPTH_WEIGHT_CACHE_MAX:
        _DEPTH_WEIGHT_CACHE.pop(next(iter(_DEPTH_WEIGHT_CACHE)))
    _DEPTH_WEIGHT_CACHE[cache_key] = (weights, prepared)
    return prepared


def _reachable_output_depth(graph: ComputationGraph) -> int:
    """Return maintained output-ancestor depth without lowering to IR."""
    output_id = graph._output_node_id
    if output_id is None or output_id not in graph.nodes:
        return 0
    return int(graph.nodes[output_id].depth)


def _graph_exceeds_final_budget(
    graph: ComputationGraph,
    config: GrammarConfig,
) -> bool:
    """Mirror the final screening depth/op budget during generation.

    Reserves a small op/depth margin for the trailing decorators that
    generate_layer_graph always appends after the template loop
    (output-dim coercion, outer residual, final rmsnorm). Without this
    reservation the grammar routinely emits graphs 1-3 ops over
    ``config.max_ops``, which the downstream validator rejects.

    For very tight budgets (small max_ops/max_depth) the reserve is
    waived: starving templates entirely produces zero valid graphs,
    and the validator already tolerates the +2 depth headroom that
    covers the post-body decorators.
    """
    op_reserve = _POST_BODY_OP_RESERVE if config.max_ops > 16 else 0
    depth_reserve = _POST_BODY_DEPTH_RESERVE if config.max_depth > 10 else 0
    op_limit = max(1, config.max_ops - op_reserve)
    depth_limit = config.max_depth + max(0, int(config.min_splits)) * 3 - depth_reserve
    return graph.n_ops() > op_limit or _reachable_output_depth(graph) > depth_limit


# ── Shape compatibility check (used by external code) ───────────────


_check_shape_compat = check_shape_compat


from .grammar_batch import AdaptiveGenerator, BatchGenerateResult, batch_generate

__all__ = [
    "AdaptiveGenerator",
    "BatchGenerateResult",
    "GrammarConfig",
    "Node",
    "batch_generate",
    "generate_layer_graph",
    "random_graph",
]
