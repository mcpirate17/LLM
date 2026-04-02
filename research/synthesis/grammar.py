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
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional

logger = logging.getLogger(__name__)

from research.defaults import MODEL_DIM
from .graph import ComputationGraph, OpNode, ShapeInfo
from .primitives import (
    PRIMITIVE_REGISTRY,
    PrimitiveOp,
    REQUIRES_RESIDUAL_BYPASS,
    algebraic_types_compatible,
    default_algebraic_type_for_space,
    validate_wiring,
)
from .templates import (
    apply_template,
)
from .context_rules import validate_context_rules
from .graph_validator import validate_dim_flow, check_param_budget
from .template_rules import validate_template_graph
from .validator import validate_graph


@dataclass(slots=True)
class BatchGenerateResult:
    """Result of batch_generate with generation statistics."""

    graphs: List["ComputationGraph"]
    n_attempted: int  # total generate_layer_graph calls made
    n_rejected_grammar: int  # ValueError/RuntimeError during generation
    n_rejected_dedup: int  # duplicate fingerprints


# ── Motif weight cache (avoids recomputing geometric means every generation) ──

import threading

_motif_weight_lock = threading.Lock()
_motif_weight_cache_key: Optional[tuple] = None
_motif_weight_cache_val: Dict[str, float] = {}


def _compute_motif_weights_from_op_weights(
    op_weights: Dict[str, float],
) -> Dict[str, tuple]:
    """Geometric mean of op weights per motif, cached by op_weights content.

    Returns {motif_name: (factor, default_lift)} so callers can apply
    ``motif_weights.get(name, lift) * factor``.
    """
    global _motif_weight_cache_key, _motif_weight_cache_val
    cache_key = tuple(sorted(op_weights.items()))
    with _motif_weight_lock:
        if cache_key == _motif_weight_cache_key:
            return _motif_weight_cache_val

    import math as _math
    from .motifs import ALL_MOTIFS

    result: Dict[str, tuple] = {}
    for motif in ALL_MOTIFS:
        motif_ops = [step.op_name for step in motif.steps]
        factors = [op_weights.get(op, 1.0) for op in motif_ops]
        factor = _math.exp(sum(_math.log(max(f, 0.01)) for f in factors) / len(factors))
        factor = max(0.1, min(8.0, factor))
        result[motif.name] = (factor, motif.lift)

    with _motif_weight_lock:
        _motif_weight_cache_key = cache_key
        _motif_weight_cache_val = result
    return result


# ── Algebraic Space Compatibility ────────────────────────────────────


def _compatible_space(current_space: str, op_space: str) -> bool:
    """Check if an op's algebraic space is compatible with the current context.

    Euclidean ops compose with everything. Non-euclidean ops require a
    matching space (e.g., tropical ops can only follow other tropical ops
    or euclidean ops).
    """
    return algebraic_types_compatible(
        default_algebraic_type_for_space(current_space),
        default_algebraic_type_for_space(op_space),
    )


def _check_graph_space_consistency(graph: ComputationGraph) -> Optional[str]:
    """Validate that a computation graph has no algebraic space conflicts.

    Walks the graph in topological order and checks that each op's algebraic
    space is compatible with the spaces of its input ops.  Returns None if
    consistent, or an error message describing the first conflict found.
    """
    for node_id, node in sorted(graph.nodes.items()):
        if node.op_name == "input":
            continue
        op = PRIMITIVE_REGISTRY.get(node.op_name)
        if op is None:
            continue
        op_type = op.algebraic_type

        for in_id in node.input_ids:
            in_node = graph.nodes.get(in_id)
            if in_node is None or in_node.op_name == "input":
                continue
            in_op = PRIMITIVE_REGISTRY.get(in_node.op_name)
            if in_op is None:
                continue
            in_type = in_op.algebraic_type
            if not algebraic_types_compatible(in_type, op_type):
                return (
                    f"Space conflict: {in_node.op_name} ({in_type.space}/{in_type.output_guarantee}) → "
                    f"{node.op_name} ({op_type.space}/{op_type.input_constraint})"
                )
    return None


# Alias for backward compatibility — some test files import Node from grammar
Node = OpNode


@dataclass
class GrammarConfig:
    """Configuration for the graph generator."""

    model_dim: int = MODEL_DIM
    max_depth: int = 16  # depth budget (triple-lane templates need ~12-14)
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

    # ── Routing-First Config (Phase 2) ────────────────────────────────
    routing_mandatory: bool = True  # Force routing structure in every graph
    routing_min_lanes: int = 2  # Minimum routing lanes (2 or 3)
    difficulty_scorer_type: str = "entropy"  # "entropy" or "learned"

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
            max_depth=16,
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
            max_depth=16,
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
            max_depth=16,
            max_ops=24,
            residual_prob=0.6,
            split_prob=0.4,
            risky_op_prob=0.7,
            routing_mandatory=False,
            exploration_targets=target_ops,
            exploration_boost_factor=boost_factor,
        )


# ── DB-backed template weight loader ────────────────────────────────


class _DBTemplateWeightCache:
    """TTL-bounded cache for DB template weights. No global mutable state."""

    __slots__ = ("_weights", "_expires", "_ttl")

    def __init__(self, ttl: float = 60.0):
        self._weights: Optional[Dict[str, float]] = None
        self._expires: float = 0.0
        self._ttl = ttl

    def get(
        self, db_path: str = "research/lab_notebook.db"
    ) -> Optional[Dict[str, float]]:
        """Load template weights from template_stats, cached for TTL seconds."""
        import math as _math
        import sqlite3
        import time as _time

        now = _time.time()
        if self._weights is not None and now < self._expires:
            return self._weights

        try:
            from pathlib import Path

            path = Path(db_path)
            if not path.is_absolute():
                cwd = Path.cwd()
                if (
                    cwd.name == "research"
                    and path.parts
                    and path.parts[0] == "research"
                ):
                    path = cwd.parent / db_path
                else:
                    path = path.resolve()
            if not path.exists():
                return None

            conn = sqlite3.connect(str(path), timeout=5.0)
            conn.execute("PRAGMA busy_timeout=5000")
            rows = conn.execute(
                """SELECT template_name, eval_count, s1_pass_count, mean_loss
                   FROM template_stats WHERE eval_count >= 5"""
            ).fetchall()
            conn.close()

            if not rows:
                return None

            from .templates import DEFAULT_TEMPLATE_WEIGHTS

            k = 3.0
            db_weights: Dict[str, float] = {}
            for tpl_name, eval_count, s1_count, mean_loss in rows:
                if mean_loss is None or not _math.isfinite(mean_loss):
                    continue
                s1_rate = s1_count / max(eval_count, 1)
                perf_weight = _math.exp(-k * mean_loss) * (1.0 + s1_rate)
                static_weight = DEFAULT_TEMPLATE_WEIGHTS.get(tpl_name, 1.0)
                db_weights[tpl_name] = 0.5 * static_weight + 0.5 * perf_weight

            for tpl_name, w in DEFAULT_TEMPLATE_WEIGHTS.items():
                if tpl_name not in db_weights:
                    db_weights[tpl_name] = w

            # Curiosity bonus: under-explored templates get up to 2x boost.
            # Decays naturally as they accumulate evaluations.
            eval_counts = {r[0]: r[1] for r in rows}
            if eval_counts:
                sorted_counts = sorted(eval_counts.values())
                median_evals = sorted_counts[len(sorted_counts) // 2]
                if median_evals > 0:
                    for tpl_name in db_weights:
                        n = eval_counts.get(tpl_name, 0)
                        if n < median_evals:
                            curiosity = 1.0 + (1.0 - n / median_evals)
                            db_weights[tpl_name] *= curiosity

            self._weights = db_weights
            self._expires = now + self._ttl
            logger.info(
                "Loaded DB template weights for %d templates (%.0f%% from DB)",
                len(db_weights),
                len(rows) / max(len(db_weights), 1) * 100,
            )
            return db_weights

        except Exception as e:
            logger.debug("Failed to load DB template weights: %s", e)
            return None


_db_weight_cache = _DBTemplateWeightCache(ttl=60.0)


class _SlotAdaptationCache:
    """TTL-bounded cache for slot class adaptations learned from wildcard fills."""

    __slots__ = ("_adaptations", "_expires", "_ttl")

    _MIN_EVALS = 5  # Minimum wildcard evals before promoting a class
    _MAX_EXTRA_CLASSES = 2  # Cap additional classes per slot

    def __init__(self, ttl: float = 120.0):
        self._adaptations: Optional[Dict[str, list]] = None
        self._expires: float = 0.0
        self._ttl = ttl

    def get(self, db_path: str = "research/lab_notebook.db") -> Dict[str, list]:
        """Return slot_key → [additional motif classes] learned from wildcard success."""
        import json as _json
        import sqlite3
        import time as _time

        now = _time.time()
        if self._adaptations is not None and now < self._expires:
            return self._adaptations

        adaptations: Dict[str, list] = {}
        try:
            from pathlib import Path

            path = Path(db_path)
            if not path.is_absolute():
                cwd = Path.cwd()
                if (
                    cwd.name == "research"
                    and path.parts
                    and path.parts[0] == "research"
                ):
                    path = cwd.parent / db_path
                else:
                    path = path.resolve()
            if not path.exists():
                return adaptations

            conn = sqlite3.connect(str(path), timeout=5.0)
            conn.execute("PRAGMA busy_timeout=5000")
            rows = conn.execute(
                """SELECT slot_key, slot_classes, s1_pass_count, eval_count,
                          wildcard_class_outcomes
                   FROM slot_stats
                   WHERE wildcard_count >= ?""",
                (self._MIN_EVALS,),
            ).fetchall()
            conn.close()

            for slot_key, slot_classes_json, s1_total, eval_total, wc_json in rows:
                if not wc_json:
                    continue
                try:
                    prescribed = set(_json.loads(slot_classes_json or "[]"))
                    wc_outcomes = _json.loads(wc_json)
                except (ValueError, TypeError):
                    continue

                baseline_s1_rate = s1_total / max(eval_total, 1)
                extra: list = []
                for cls, vals in wc_outcomes.items():
                    if cls in prescribed:
                        continue  # Already in the slot's class set
                    n = vals.get("n", 0)
                    s1 = vals.get("s1", 0)
                    if n < self._MIN_EVALS:
                        continue
                    cls_s1_rate = s1 / n
                    if cls_s1_rate > baseline_s1_rate:
                        extra.append(cls)
                    if len(extra) >= self._MAX_EXTRA_CLASSES:
                        break

                if extra:
                    adaptations[slot_key] = extra

            self._adaptations = adaptations
            self._expires = now + self._ttl
            if adaptations:
                logger.info(
                    "Loaded slot adaptations: %d slots with expanded classes",
                    len(adaptations),
                )
        except Exception as e:
            logger.debug("Failed to load slot adaptations: %s", e)

        return adaptations


_slot_adaptation_cache = _SlotAdaptationCache(ttl=120.0)


class EfficiencyPrior:
    """Uses historical Pareto frontier data to bias synthesis."""

    __slots__ = ("op_biases",)

    def __init__(self, frontier_data: List[Dict]):
        self.op_biases: Dict[str, float] = {}
        for p in frontier_data or []:
            graph_json = p.get("graph_json", "")
            if not graph_json:
                continue
            for motif in [
                "selective_scan",
                "tropical",
                "clifford",
                "low_rank",
                "sparse",
            ]:
                if motif in graph_json:
                    mult = 1.12 if motif == "tropical" else 1.05
                    self.op_biases[motif] = self.op_biases.get(motif, 1.0) * mult

    def get_bias(self, op_name: str) -> float:
        bias = 1.0
        for motif, multiplier in self.op_biases.items():
            if motif in op_name:
                bias *= multiplier
        return min(2.5, bias)


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
        cached_factors = _compute_motif_weights_from_op_weights(config.op_weights)
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

        # Also boost templates that serve target ops (binary-op templates)
        _OP_TO_TEMPLATE = {
            "div_safe": "safe_division",
            "maximum": "gated_maximum",
            "minimum": "gated_minimum",
            "sub": "residual_difference",
            "split3": "three_way_split",
            "outer_product": "gated_product",
            "geometric_product": "geometric_product_block",
            "tropical_matmul": "tropical_matmul_block",
            "hyp_distance": "hyp_distance_scoring",
            "cumprod_safe": "decay_sequence",
            # Phase 3: dedicated paths for underperforming ops
            "lif_neuron": "spiking_moe_block",
            "sparse_threshold": "spiking_moe_block",
            "stdp_attention": "spiking_residual_block",  # needs spiking predecessor chain
            "spike_rate_code": "spiking_moe_block",
            "hyp_linear": "hyperbolic_bridge_block",
            "hyp_tangent_nonlinear": "hyperbolic_bridge_block",
            "sparse_bottleneck_moe": "n_way_moe_block",
            "conv_only": "conv_residual_block",
            "fixed_point_iter": "iterative_refinement",
            "gated_delta": "recurrent_delta_block",
            "bottleneck_proj": "bottleneck",
            "low_rank_proj": "bottleneck",
            "confidence_token_gate": "cascaded_early_exit",
            "depth_weighted_proj": "recursive_depth_router",
            "tropical_center": "tropical_center_block",
            "tropical_attention": "tropical_center_block",
            "tropical_add": "tropical_residual",
            # 0% S1 fix: dedicated template paths
            "cumsum": "cumulative_sequence",
            "sqrt": "sqrt_gated_ffn",
            "norm_last": "reduce_attend",
            "mean_last": "reduce_attend",
            "max_last": "reduce_attend",
            "sum_last": "reduce_attend",
            # 0% S1 fix round 2
            "diff_attention": "diff_attention_block",
            "causal_mask": "causal_mix_block",
            "fused_linear_gelu": "fused_gelu_ffn",
            "exp": "exp_gated_residual",
            "integral_kernel": "integral_kernel_block",
            "sliding_window_mask": "windowed_attention",
            "local_window_attn": "local_attention_block",
            "state_space": "state_space_block",
            "rwkv_time_mixing": "rwkv_block",
            "reciprocal": "reciprocal_gated",
            "sign_ste": "sign_ste_gated",
            "log": "log_gated",
            "ultrametric_attention": "ultrametric_attention_block",
            "graph_attention": "graph_attention_block",
            # 2-input routing ops — need dedicated template for structural wiring
            "dual_compression_blend": "signal_routed_compression",
            "signal_conditioned_compression": "signal_routed_compression",
            "score_depth_blend": "mixed_recursion",
            "difficulty_blend_3way": "three_lane_adaptive",
            "relu_gated_moe": "moe",
        }
        if tpl_weights is None:
            from .templates import DEFAULT_TEMPLATE_WEIGHTS

            tpl_weights = dict(DEFAULT_TEMPLATE_WEIGHTS)
        for op_name in config.exploration_targets:
            tpl_name = _OP_TO_TEMPLATE.get(op_name)
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
    _EFFICIENCY_TEMPLATES = {
        "sparse_moe_block",
        "routed_bottleneck",
        "token_merge_block",
        "conditional_compute",
        "sparse_ffn",
        "moe",
    }
    if config.structured_sparsity_bias > 0.5 and tpl_weights:
        _first_tpl_weights = {
            k: (v if k in _EFFICIENCY_TEMPLATES else 0.0)
            for k, v in tpl_weights.items()
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

        # Depth-aware template biasing: early blocks favor FFN/conv,
        # late blocks favor attention/SSM (per GPT-2 layer importance research).
        if _iter_weights and n_templates > 1:
            depth_ratio = t_idx / (n_templates - 1)
            depth_weights = {}
            for tpl_name, base_w in _iter_weights.items():
                tl = tpl_name.lower()
                if depth_ratio < 0.33:
                    if "conv" in tl or "ffn" in tl or "bottleneck" in tl:
                        depth_weights[tpl_name] = base_w * 1.5
                    elif "attention" in tl or "transformer" in tl:
                        depth_weights[tpl_name] = base_w * 0.6
                    else:
                        depth_weights[tpl_name] = base_w
                elif depth_ratio > 0.66:
                    if "attention" in tl or "transformer" in tl or "mamba" in tl:
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
        prev_node_ids = set(graph.nodes.keys())
        prev_next_id = graph._next_id
        prev_output_id = graph._output_node_id
        prev_metadata = dict(graph.metadata)
        graph._cache.clear()

        trial_current = apply_template(
            graph,
            current,
            rng,
            template_weights=_iter_weights,
            motif_weights=motif_weights,
            op_weights=config.op_weights or None,
        )

        if _graph_exceeds_final_budget(graph, config):
            # Rollback: remove added nodes, restore metadata
            added_ids = set(graph.nodes.keys()) - prev_node_ids
            for nid in added_ids:
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

    # Optional spectral filter injection (driven by freq_domain_prob)
    # Wrapped in residual: FFT numerical drift is unrecoverable without skip.
    # rmsnorm before spectral_filter satisfies MATH_SPACE_RULES.
    if (
        rng.random() < config.freq_domain_prob
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

    # Ensure output shape is (B, S, D)
    result_shape = graph.nodes[current].output_shape
    if result_shape.dim != config.model_dim:
        current = graph.add_op(
            "linear_proj", [current], config={"out_dim": config.model_dim}
        )

    # Optional outer residual connection (if not already added by template)
    # Check if the last op is already an add with input_id
    last_node = graph.nodes[current]
    has_outer_residual = last_node.op_name == "add" and input_id in last_node.input_ids
    if (
        not has_outer_residual
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


_ROUTING_COMPRESSION_MOE_OPS: frozenset = frozenset(
    {
        # Routing — per-token path selection
        "token_entropy",
        "token_class_proj",
        "feature_sparsity",
        "gated_lane_blend",
        "depth_gated_transform",
        "difficulty_blend_3way",
        "score_depth_blend",
        "confidence_token_gate",
        "learned_token_gate",
        "cheap_verify_blend",
        "depth_weighted_proj",
        "depth_token_mask",
        "adjacent_token_merge",
        "relu_gated_moe",
        # True routing — heterogeneous expert dispatch
        "hetero_moe",
        "arch_router",
        "compute_budget_router",
        # MoE — expert selection
        "moe_topk",
        "moe_2expert",
        "sparse_bottleneck_moe",
        "tropical_moe",
        # Conditional gating — per-token activation decisions
        "topk_gate",
        "tropical_gate",
        "tropical_router",
        "sparse_threshold",
        "lif_neuron",
        "padic_gate",
        # Dynamic compression — per-token bandwidth decisions
        "signal_conditioned_compression",
        "adaptive_rank_gate",
        "dual_compression_blend",
        "latent_attention_compressor",
    }
)


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

    # Dimension-flow validation — catch skip-only paths and dim mismatches
    # that slipped through template ValueError fallbacks.
    dim_result = validate_dim_flow(graph)
    if not dim_result.valid:
        raise ValueError(dim_result.errors[0])

    # Parameter budget — reject graphs that would OOM before eval.
    # Budget: 12 transformer-equivalent layers * 4*D*D params each.
    max_params = 12 * 4 * config.model_dim * config.model_dim
    budget_result = check_param_budget(graph, max_params)
    if not budget_result.valid:
        raise ValueError(budget_result.errors[0])

    # Algebraic space consistency check — reject graphs that mix
    # incompatible mathematical spaces (e.g., tropical after poincaré).
    space_err = _check_graph_space_consistency(graph)
    if space_err is not None:
        raise ValueError(space_err)

    # Routing-mandatory check: every graph must have routing, compression, or MoE
    if config.routing_mandatory:
        op_names = {n.op_name for n in graph.nodes.values() if not n.is_input}
        if not op_names & _ROUTING_COMPRESSION_MOE_OPS:
            raise ValueError(
                "routing_mandatory=True but graph has no routing/compression/MoE ops"
            )

    # Depth constraint check: reject ops placed before their min_layer_depth.
    # Approximate layer depth by topological order from input.
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
        op = PRIMITIVE_REGISTRY.get(node.op_name)
        if op is not None and op.min_layer_depth > 0:
            depth = topo_depth.get(nid, 0)
            if depth < op.min_layer_depth:
                # Auto-correct: replace too-shallow op with identity pass-through
                logger.debug(
                    "depth_autocorrect: %s at depth %d < min %d → identity",
                    node.op_name,
                    depth,
                    op.min_layer_depth,
                )
                object.__setattr__(node, "op_name", "identity")

    # Residual bypass check: ops in REQUIRES_RESIDUAL_BYPASS must have a
    # downstream add that also takes the op's input (residual connection).
    for nid, node in graph.nodes.items():
        if node.is_input or node.op_name not in REQUIRES_RESIDUAL_BYPASS:
            continue
        # Check if any downstream consumer is an 'add' that also takes
        # one of this node's inputs (forming a skip connection).
        node_inputs = set(node.input_ids)
        has_bypass = False
        for other_nid, other_node in graph.nodes.items():
            if other_node.op_name != "add":
                continue
            other_inputs = set(other_node.input_ids)
            # The add takes both (a) something downstream of nid and
            # (b) one of the original inputs to nid → residual bypass
            if other_inputs & node_inputs:
                has_bypass = True
                break
        if not has_bypass:
            raise ValueError(
                f"{node.op_name} (id={nid}) requires residual bypass but none found"
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
            for other_nid, other_node in graph.nodes.items():
                if other_node.is_input:
                    continue
                if nid not in other_node.input_ids:
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
            has_valid_successor = False
            for other_nid, other_node in graph.nodes.items():
                if other_node.is_input:
                    continue
                if (
                    nid in other_node.input_ids
                    and other_node.op_name in must_follow_with
                ):
                    has_valid_successor = True
                    break
            if not has_valid_successor:
                raise ValueError(
                    f"Math-space constraint: {node.op_name} (id={nid}) "
                    f"must be followed by one of {must_follow_with}"
                )

    ctx_err = validate_context_rules(graph)
    if ctx_err is not None:
        raise ValueError(ctx_err)

    # Template-level structural invariants (metadata only, don't reject)
    tpl_errors = validate_template_graph(graph)
    if tpl_errors:
        for err in tpl_errors:
            logger.debug("template_rule: %s", err)
        graph.metadata["template_rule_warnings"] = tpl_errors


def _graph_exceeds_final_budget(
    graph: ComputationGraph,
    config: GrammarConfig,
) -> bool:
    """Mirror the final screening depth/op budget during generation."""
    depth_limit = config.max_depth + max(0, int(config.min_splits)) * 3
    return graph.n_ops() >= config.max_ops or graph.depth() >= depth_limit


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
        return generate_layer_graph(self.config, seed=seed)


# ── Shape compatibility check (used by external code) ───────────────


def _check_shape_compat(
    op: PrimitiveOp,
    input_shapes: List[ShapeInfo],
    model_dim: int,
    current_space: str = "euclidean",
) -> bool:
    """Quick check if an op is compatible with given input shapes and space."""
    # Algebraic space filter: reject ops from incompatible spaces
    if not algebraic_types_compatible(
        default_algebraic_type_for_space(current_space),
        op.algebraic_type,
    ):
        return False

    if not input_shapes:
        return False
    if op.n_inputs != len(input_shapes):
        return False

    s0 = input_shapes[0]

    if op.name == "split2":
        if s0.dim % 2 != 0 or s0.dim // 2 < 4:
            return False
    if op.name == "split3":
        if s0.dim % 3 != 0 or s0.dim // 3 < 4:
            return False

    if op.shape_rule == "rfft" and not s0.is_standard:
        return False
    if op.shape_rule == "irfft" and not s0.is_freq_domain:
        return False

    if op.name in (
        "local_window_attn",
        "sliding_window_mask",
        "token_pool_restore",
        "selective_scan",
        "conv1d_seq",
        "basis_expansion",
        "integral_kernel",
        "fixed_point_iter",
    ):
        if not s0.is_standard:
            return False

    _MIN_DIM_OPS = {
        "softmax_attention": 16,
        "linear_attention": 16,
        "graph_attention": 16,
        "multi_head_mix": 4,
        "selective_scan": 8,
        "state_space": 8,
        "rwkv_time_mixing": 8,
        "rwkv_channel": 8,
        "conv1d_seq": 4,
        "moe_topk": 8,
        "moe_2expert": 8,
        "swiglu_mlp": 4,
        "topk_gate": 4,
        "block_sparse_linear": 16,
        "nm_sparse_linear": 8,
        "low_rank_proj": 8,
        "bottleneck_proj": 8,
        "grouped_linear": 8,
        "shared_basis_proj": 8,
        # Parameterized ops that fail with degenerate dims from reduce_last
        "gated_linear": 8,
        "ternary_projection": 8,
        "linear_proj": 4,
        "linear_proj_down": 4,
        "linear_proj_up": 4,
        "fused_linear_gelu": 4,
        "difficulty_blend_3way": 8,
        "relu_gated_moe": 8,
    }
    min_dim = _MIN_DIM_OPS.get(op.name)
    if min_dim and s0.dim < min_dim:
        return False

    if len(input_shapes) == 2:
        s1 = input_shapes[1]
        if op.shape_rule == "binary_broadcast":
            if s0.seq != s1.seq:
                return False
            if s0.dim != s1.dim and s0.dim != 1 and s1.dim != 1:
                return False
        elif op.shape_rule in ("matmul", "concat"):
            if s0.seq != s1.seq:
                return False

    return True
