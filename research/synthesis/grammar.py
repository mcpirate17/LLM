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

import logging
import random
from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, List, Optional

logger = logging.getLogger(__name__)

from research.defaults import MODEL_DIM
from .graph import ComputationGraph, OpNode
from .grammar_support import (
    DBTemplateWeightCache,
    EFFICIENCY_TEMPLATES,
    EfficiencyPrior,
    OP_TO_TEMPLATE,
    ROUTING_COMPRESSION_MOE_OPS,
    SlotAdaptationCache,
    check_graph_space_consistency,
    check_shape_compat,
    compute_motif_weights_from_op_weights,
)
from .primitives import (
    PRIMITIVE_REGISTRY,
    REQUIRES_RESIDUAL_BYPASS,
    get_wiring_rule,
    validate_wiring,
)
from .templates import (
    apply_template,
)
from .context_rules import validate_context_rules
from .graph_validator import validate_dim_flow
from .template_rules import validate_template_graph
from .validator import validate_graph


@dataclass(slots=True)
class BatchGenerateResult:
    """Result of batch_generate with generation statistics."""

    graphs: List["ComputationGraph"]
    n_attempted: int  # total generate_layer_graph calls made
    n_rejected_grammar: int  # ValueError/RuntimeError during generation
    n_rejected_dedup: int  # duplicate fingerprints


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
    category_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "elementwise_unary": 2.0,
            "elementwise_binary": 1.5,
            "reduction": 0.8,
            "linear_algebra": 1.0,
            "structural": 1.0,
            "parameterized": 2.0,
            "mixing": 1.5,
            "sequence": 1.2,
            "frequency": 1.0,
            "math_space": 1.5,
            "functional": 3.0,  # Routing/gating/branching ops
        }
    )
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
        (ppl + binding_auc + induction + ar + hellaswag). The companion
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


_db_weight_cache = DBTemplateWeightCache(ttl=60.0)
_slot_adaptation_cache = SlotAdaptationCache(ttl=120.0)
_ROUTING_COMPRESSION_MOE_OPS = ROUTING_COMPRESSION_MOE_OPS


_ROUTING_CAPABLE_TEMPLATES_CACHE: Optional[FrozenSet[str]] = None


_ROUTING_PROBE_SEEDS: int = 8
_ROUTING_PROBE_MIN_HITS: int = 7  # ≥ 7/8 seeds must emit routing


def _get_routing_capable_templates() -> FrozenSet[str]:
    """Return names of templates that reliably emit a ROUTING_COMPRESSION_MOE op.

    Built once on first call by dry-running each registered template across
    several seeds. A template qualifies only if it emits a routing op in at
    least ``_ROUTING_PROBE_MIN_HITS`` of ``_ROUTING_PROBE_SEEDS`` probes —
    templates that emit routing only stochastically via motif lottery (e.g.
    residual_block, diff_attention_block) would defeat the rescue since the
    actual composition picks its own seed. Used by the routing_mandatory
    rescue in generate_layer_graph to restrict the final composition slot
    to a guaranteed-routing pool.
    """
    global _ROUTING_CAPABLE_TEMPLATES_CACHE
    if _ROUTING_CAPABLE_TEMPLATES_CACHE is not None:
        return _ROUTING_CAPABLE_TEMPLATES_CACHE

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

    _ROUTING_CAPABLE_TEMPLATES_CACHE = frozenset(routing_capable)
    logger.info(
        "Routing-capable templates: %d of %d registered "
        "(>=%d/%d probe hits; cached for routing_mandatory rescue).",
        len(_ROUTING_CAPABLE_TEMPLATES_CACHE),
        len(TEMPLATES),
        _ROUTING_PROBE_MIN_HITS,
        _ROUTING_PROBE_SEEDS,
    )
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

    # A single positive template weight is not a soft preference; it's a
    # targeted generation request. Treat it as forced_template so routing
    # rescue and fallback selection cannot silently swap in unrelated graphs.
    if not config.forced_template and config.template_weights:
        positive_templates = [
            name for name, weight in config.template_weights.items() if weight > 0.0
        ]
        if len(positive_templates) == 1:
            config = replace(config, forced_template=positive_templates[0])

    # Forced template: default to 1 block (stacking the same template 3x
    # usually exceeds depth/ops limits and fails validation).
    if config.forced_template and config.composition_depth > 1:
        config = replace(config, composition_depth=1)

    rng = random.Random(seed)
    graph = ComputationGraph(config.model_dim)
    input_id = graph.add_input()

    # Determine composition depth (how many template blocks to stack)
    if config.composition_depth > 0:
        n_templates = config.composition_depth
    else:
        n_templates = rng.choices([1, 2, 3], weights=[3, 5, 2], k=1)[0]

    # Template and motif weights flow directly into template/motif pickers.
    # Non-empty dicts carry research-signal priors from execution_screening.
    # If config has explicit weights, use them. If use_db_weights, try DB.
    if config.template_weights:
        tpl_weights = dict(config.template_weights)
    elif config.use_db_weights:
        tpl_weights = _db_weight_cache.get()  # TTL-cached, returns None if unavailable
    else:
        tpl_weights = None
    motif_weights = dict(config.motif_weights) if config.motif_weights else {}

    # Bridge op_weights → motif_weights: geometric mean of constituent op weights.
    # Always applied (not just boosts) so penalties propagate through motifs.
    if config.op_weights or config.exploration_targets:
        from .motifs import ALL_MOTIFS

    if config.op_weights:
        cached_factors = compute_motif_weights_from_op_weights(config.op_weights)
        for motif_name, (factor, default_lift) in cached_factors.items():
            current = motif_weights.get(motif_name, default_lift)
            motif_weights[motif_name] = current * factor

    # Boost motifs containing under-observed ops (exploration targets)
    if config.exploration_targets:
        for motif in ALL_MOTIFS:
            motif_ops = {step.op_name for step in motif.steps}
            if motif_ops & config.exploration_targets:
                current = motif_weights.get(motif.name, motif.lift)
                motif_weights[motif.name] = current * config.exploration_boost_factor

        if tpl_weights is None:
            from .templates import DEFAULT_TEMPLATE_WEIGHTS

            tpl_weights = dict(DEFAULT_TEMPLATE_WEIGHTS)
        for op_name in config.exploration_targets:
            tpl_name = OP_TO_TEMPLATE.get(op_name)
            if tpl_name and tpl_name in tpl_weights:
                tpl_weights[tpl_name] *= config.exploration_boost_factor

    motif_weights = motif_weights or None
    graph.metadata["context_rules_version"] = "low_s1_v1"
    if config.wildcard_slot_prob > 0:
        graph.metadata["_wildcard_slot_prob"] = config.wildcard_slot_prob
    # Load learned slot class expansions from wildcard success data
    if config.use_db_weights:
        slot_adaptations = _slot_adaptation_cache.get()
        if slot_adaptations:
            graph.metadata["_slot_adaptations"] = slot_adaptations

    # High sparsity bias → force first template from efficiency pool
    if config.structured_sparsity_bias > 0.5 and tpl_weights:
        _first_tpl_weights = {
            k: (v if k in EFFICIENCY_TEMPLATES else 0.0) for k, v in tpl_weights.items()
        }
        # Only use if at least one efficiency template has positive weight
        if any(v > 0 for v in _first_tpl_weights.values()):
            _use_efficiency_first = True
        else:
            _first_tpl_weights = None
            _use_efficiency_first = False
    else:
        _first_tpl_weights = None
        _use_efficiency_first = False

    current = input_id
    for t_idx in range(n_templates):
        _iter_weights = (
            _first_tpl_weights
            if (t_idx == 0 and _use_efficiency_first)
            else tpl_weights
        )

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
            from .templates import TEMPLATES as _ALL_TEMPLATES

            _routing_tpls = _get_routing_capable_templates()
            _base_weights = dict(_iter_weights) if _iter_weights else {}
            _iter_weights = {
                name: (_base_weights.get(name, 1.0) if name in _routing_tpls else 0.0)
                for name in _ALL_TEMPLATES
            }

        # Depth-aware template biasing: early blocks favor FFN/conv,
        # late blocks favor attention/SSM (per GPT-2 layer importance research).
        # Note: "attn" matches new attention templates (attn_*).
        if _iter_weights and n_templates > 1:
            depth_ratio = t_idx / (n_templates - 1)
            depth_weights = {}
            for tpl_name, base_w in _iter_weights.items():
                tl = tpl_name.lower()
                _is_attn = "attn" in tl or "attention" in tl or "transformer" in tl
                if depth_ratio < 0.33:
                    if "conv" in tl or "ffn" in tl or "bottleneck" in tl:
                        depth_weights[tpl_name] = base_w * 1.5
                    elif _is_attn:
                        depth_weights[tpl_name] = base_w * 0.85
                    else:
                        depth_weights[tpl_name] = base_w
                elif depth_ratio > 0.66:
                    if _is_attn or "mamba" in tl:
                        depth_weights[tpl_name] = base_w * 1.5
                    elif "bottleneck" in tl or "compress" in tl:
                        depth_weights[tpl_name] = base_w * 0.6
                    else:
                        depth_weights[tpl_name] = base_w
                else:
                    depth_weights[tpl_name] = base_w
            _iter_weights = depth_weights

        # Snapshot graph state for lightweight rollback instead of full copy.
        # Only need to track node IDs added and metadata changes.
        prev_next_id = graph._next_id
        prev_output_id = graph._output_node_id
        prev_metadata = dict(graph.metadata)
        graph._cache.clear()

        trial_current = apply_template(
            graph,
            current,
            rng,
            template_name=config.forced_template,
            template_weights=_iter_weights,
            motif_weights=motif_weights,
            op_weights=config.op_weights or None,
            exploration_budget=config.template_exploration_budget,
        )

        if _graph_exceeds_final_budget(graph, config):
            # Roll back only the suffix allocated by this template. Node IDs are
            # monotonic, so there is no reason to rebuild a full key set here.
            for nid in range(prev_next_id, graph._next_id):
                del graph.nodes[nid]
            graph._next_id = prev_next_id
            graph._output_node_id = prev_output_id
            graph.metadata = prev_metadata
            graph._cache.clear()
            break
        current = trial_current

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

    # Post-generation validation
    _validate_graph(graph, config)

    return graph


def _validate_graph(graph: ComputationGraph, config: GrammarConfig) -> None:
    """Validate a generated graph and raise ValueError if invalid."""
    # Allow +2 depth headroom for multi-step motifs (e.g., 3-4 step math-space
    # motifs that include norm+op+proj) which can push templates slightly over.
    result = validate_graph(
        graph,
        max_ops=config.max_ops,
        max_depth=config.max_depth + 2,
        min_splits=config.min_splits,
    )
    if not result.valid:
        raise ValueError(
            result.errors[0] if result.errors else "Graph validation failed"
        )

    # Dimension-flow validation also enforces the parameter budget so we
    # don't pay for an extra whole-graph reachable-path scan here.
    max_params = 12 * 4 * config.model_dim * config.model_dim
    dim_result = validate_dim_flow(graph, max_params=max_params)
    if not dim_result.valid:
        raise ValueError(dim_result.errors[0])

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
    topo_depth: Dict[int, int] = {}
    for nid, node in sorted(graph.nodes.items()):
        if node.is_input:
            topo_depth[nid] = 0
        else:
            parent_depths = [topo_depth.get(pid, 0) for pid in node.input_ids]
            topo_depth[nid] = max(parent_depths, default=0) + 1

    for nid, node in graph.nodes.items():
        if node.is_input:
            continue
        depth_rule = get_wiring_rule(node.op_name) or {}
        min_layer_depth = int(depth_rule.get("min_layer_depth", 0))
        if min_layer_depth > 0:
            depth = topo_depth.get(nid, 0)
            if depth < min_layer_depth:
                raise ValueError(
                    f"{node.op_name} (id={nid}) placed at depth {depth} "
                    f"before min_layer_depth={min_layer_depth}"
                )

    # Residual bypass check: ops in REQUIRES_RESIDUAL_BYPASS must have a
    # downstream add that also takes the op's input (residual connection).
    successors: Dict[int, List[int]] = {nid: [] for nid in graph.nodes}
    add_inputs_by_source: Dict[int, set[int]] = {}
    for other_nid, other_node in graph.nodes.items():
        for parent_id in other_node.input_ids:
            if parent_id in successors:
                successors[parent_id].append(other_nid)
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

    ctx_err = validate_context_rules(graph)
    if ctx_err is not None:
        raise ValueError(ctx_err)

    # Template-level structural invariants are part of legality, not metadata-only
    # guidance. Preserve the warning payload for observability, but reject the
    # graph so template-invalid programs never count as valid survivors.
    tpl_errors = validate_template_graph(graph)
    if tpl_errors:
        for err in tpl_errors:
            logger.debug("template_rule: %s", err)
        graph.metadata["template_rule_warnings"] = tpl_errors
        raise ValueError(f"Template rule violations: {tpl_errors}")


def _graph_exceeds_final_budget(
    graph: ComputationGraph,
    config: GrammarConfig,
) -> bool:
    """Mirror the final screening depth/op budget during generation."""
    depth_limit = config.max_depth + max(0, int(config.min_splits)) * 3
    return graph.n_ops() > config.max_ops or graph.depth() > depth_limit


def batch_generate(
    n: int,
    config: Optional[GrammarConfig] = None,
    base_seed: int = 42,
    _use_adaptive_synthesis: bool = False,
    prior: Optional[EfficiencyPrior] = None,
) -> BatchGenerateResult:
    """Generate N unique computation graphs.

    Returns BatchGenerateResult with generation statistics (n_attempted,
    n_rejected_grammar, n_rejected_dedup) to expose the true rejection rate.
    """
    if config is None:
        config = GrammarConfig()
    config = _config_with_efficiency_prior(config, prior)

    graphs: List[ComputationGraph] = []
    fingerprints: set = set()

    attempts = 0
    n_rejected_grammar = 0
    n_rejected_dedup = 0
    max_attempts = n * 10

    while len(graphs) < n and attempts < max_attempts:
        attempts += 1
        seed = base_seed + attempts * 137
        try:
            g = generate_layer_graph(config, seed=seed)
            fp = g.fingerprint()
            if fp not in fingerprints:
                fingerprints.add(fp)
                graphs.append(g)
            else:
                n_rejected_dedup += 1
        except (ValueError, RuntimeError):
            n_rejected_grammar += 1
            continue

    rejection_rate = (n_rejected_grammar + n_rejected_dedup) / max(attempts, 1)

    # Exhaustion recovery: if 0 graphs produced, relax constraints and retry.
    # Reduces composition_depth, raises max_ops/max_depth, increases exploration
    # budget. This prevents complete stalls when the grammar is saturated.
    if len(graphs) == 0 and n_rejected_grammar > 0:
        relaxed = replace(
            config,
            composition_depth=max(1, config.composition_depth - 1),
            max_ops=config.max_ops + 6,
            max_depth=config.max_depth + 6,
            template_exploration_budget=max(config.template_exploration_budget, 0.25),
            forced_template=None,  # clear forced template on recovery
        )
        logger.info(
            "batch_generate: exhaustion recovery — relaxing constraints "
            "(depth %d→%d, ops %d→%d, composition %d→%d, exploration %.0f%%)",
            config.max_depth,
            relaxed.max_depth,
            config.max_ops,
            relaxed.max_ops,
            config.composition_depth,
            relaxed.composition_depth,
            relaxed.template_exploration_budget * 100,
        )
        retry_attempts = 0
        retry_max = n * 5
        while len(graphs) < n and retry_attempts < retry_max:
            retry_attempts += 1
            attempts += 1
            seed = base_seed + attempts * 137 + 99999
            try:
                g = generate_layer_graph(relaxed, seed=seed)
                fp = g.fingerprint()
                if fp not in fingerprints:
                    fingerprints.add(fp)
                    graphs.append(g)
                else:
                    n_rejected_dedup += 1
            except (ValueError, RuntimeError):
                n_rejected_grammar += 1

    rejection_rate = (n_rejected_grammar + n_rejected_dedup) / max(attempts, 1)
    logger.info(
        "batch_generate: %d graphs from %d attempts "
        "(%d grammar failures, %d duplicates, %.0f%% rejection rate)",
        len(graphs),
        attempts,
        n_rejected_grammar,
        n_rejected_dedup,
        rejection_rate * 100,
    )

    return BatchGenerateResult(
        graphs=graphs,
        n_attempted=attempts,
        n_rejected_grammar=n_rejected_grammar,
        n_rejected_dedup=n_rejected_dedup,
    )


# ── Legacy compatibility ────────────────────────────────────────────
# AdaptiveGenerator is still referenced by some test files and the
# use_adaptive_synthesis path. Keep it functional.


class AdaptiveGenerator:
    """Adaptive generator — delegates to motif-based generation."""

    __slots__ = ("config", "prior", "model_dim", "max_params", "max_flops")

    def __init__(self, config: GrammarConfig, prior: Optional[EfficiencyPrior] = None):
        self.config = config
        self.prior = prior
        self.model_dim = config.model_dim
        self.max_params = (
            4 * self.model_dim * self.model_dim * 12
        )  # VRAM is the real constraint
        self.max_flops = 4 * (12 * self.model_dim * self.model_dim * 128)

    def generate(self, seed: Optional[int] = None) -> ComputationGraph:
        return generate_layer_graph(
            _config_with_efficiency_prior(self.config, self.prior),
            seed=seed,
        )


# ── Shape compatibility check (used by external code) ───────────────


_check_shape_compat = check_shape_compat
