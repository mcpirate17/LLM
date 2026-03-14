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
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .graph import ComputationGraph, OpNode, ShapeInfo
from .primitives import (
    PRIMITIVE_REGISTRY, PrimitiveOp, algebraic_types_compatible,
    default_algebraic_type_for_space,
)
from .templates import (
    apply_template,
)
from .validator import validate_graph


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
    model_dim: int = 256
    min_depth: int = 3
    max_depth: int = 10
    max_width: int = 4          # max parallel paths (2 or 3 way splits)
    max_ops: int = 16           # max total operations
    max_params_ratio: float = 8.0  # max params relative to D^2
    residual_prob: float = 0.7  # probability of residual connection
    split_prob: float = 0.3     # probability of branching into parallel paths
    min_splits: int = 0         # minimum number of split-merge blocks to force
    three_way_split_prob: float = 0.0  # probability of 3-way split (vs 2-way)
    branch_depth: int = 1       # depth of subgraph processing on each branch
    max_recursion_depth: int = 4  # iteration cap for recursive ops
    stability_check: bool = True  # validate architectures before compilation
    merge_prob: float = 0.4     # probability of merging paths
    risky_op_prob: float = 0.5  # probability of using numerically risky ops
    freq_domain_prob: float = 0.15  # probability of FFT detour
    # Category weights (higher = more likely to be chosen)
    category_weights: Dict[str, float] = field(default_factory=lambda: {
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
        "functional": 1.5,
    })
    # Excluded op names
    excluded_ops: Set[str] = field(default_factory=lambda: {
        "softmax_seq", "mean_seq", "sum_seq",
        "sort_seq", "argsort_seq",
        "rfft_seq", "irfft_seq",
    })
    # Per-op weight multipliers
    op_weights: Dict[str, float] = field(default_factory=dict)

    # Structured Sparsity Constraints (Z7)
    structured_sparsity_bias: float = 0.0
    enforce_block_size: Optional[int] = None
    min_block_density: float = 0.05
    max_block_density: float = 0.5

    # Hyperbolic Promotion (Phase 3)
    hyperbolic_promotion_threshold: float = 0.6
    hyperbolic_boost_factor: float = 3.0
    _hierarchy_fitness: Optional[float] = None

    # ── Motif-based grammar config (Phase 6) ────────────────────────
    # Template selection weights (template_name → weight).
    # If empty, uses DEFAULT_TEMPLATE_WEIGHTS.
    template_weights: Dict[str, float] = field(default_factory=dict)
    # Motif selection weights (motif_name → weight).
    # If empty, uses motif's lift score as weight.
    motif_weights: Dict[str, float] = field(default_factory=dict)
    # Number of templates to compose per graph (1-3)
    composition_depth: int = 0  # 0 = auto (random 1-2)

    # ── Routing-First Config (Phase 2) ────────────────────────────────
    routing_mandatory: bool = False   # Force routing structure in every graph
    routing_min_lanes: int = 2        # Minimum routing lanes (2 or 3)
    difficulty_scorer_type: str = "entropy"  # "entropy" or "learned"

    def update_bias(self, delta: float):
        """Adjust structured sparsity bias."""
        self.structured_sparsity_bias = max(0.0, min(1.0,
                                            self.structured_sparsity_bias + delta))

    @classmethod
    def efficient(cls, model_dim: int = 256) -> "GrammarConfig":
        """Config tuned for efficiency-first architecture search (>5x GPT-2)."""
        return cls(
            model_dim=model_dim,
            min_depth=3,
            max_depth=8,
            max_ops=12,
            max_params_ratio=8.0,
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
                "entropy_router": 3.5,
                "token_merging": 3.5,
                "ternary_projection": 3.5,
                "block_sparse_linear": 3.5,
                "semi_structured_2_4_linear": 3.0,
                "moe_2expert": 3.0,
                "gated_linear": 2.5,
                "swiglu_mlp": 2.0,
            },
        )

    @classmethod
    def exotic(cls, model_dim: int = 256) -> "GrammarConfig":
        """Config tuned for exotic architecture exploration."""
        return cls(
            model_dim=model_dim,
            min_depth=3,
            max_depth=10,
            max_ops=16,
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
                "route_topk": 3.0, "route_lanes": 3.0,
                "route_recursion": 2.5, "mod_topk": 3.0,
                "early_exit": 2.0, "adaptive_recursion": 3.0,
                "token_merging": 2.0, "cascade": 2.0,
                "speculative": 2.0, "moe_topk": 3.0,
                "adaptive_lane_mixer": 3.0,
                "mixed_recursion_gate": 2.5,
                "relu_gate_routing": 2.5,
                "latent_attention_compressor": 3.0,
                "routing_conditioned_compression": 2.5,
                "progressive_compression_gate": 2.0,
                "compression_mixture_experts": 2.5,
                "token_type_classifier": 2.5,
                "entropy_router": 2.5,
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
                "residual_block", "sequential", "transformer_block",
                "parallel_split", "bottleneck", "moe", "hybrid_parallel",
                "gated_residual", "dense_cascade", "sparse_ffn",
                "sparse_moe_block", "routed_bottleneck", "token_merge_block",
                "conditional_compute", "difficulty_routed_block",
                "three_lane_adaptive", "cascaded_early_exit",
                "recursive_depth_router",
            )
        }
        return cls(
            model_dim=model_dim,
            min_depth=3,
            max_depth=10,
            max_ops=16,
            max_params_ratio=10.0,
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
                "entropy_router": 5.0,
                "adaptive_lane_mixer": 4.0,
                "early_exit": 3.5,
                "cascade": 3.5,
                "adaptive_recursion": 3.5,
                "moe_topk": 3.0,
                "moe_2expert": 3.0,
                "token_merging": 3.0,
                "relu_gate_routing": 2.5,
                "swiglu_mlp": 2.0,
                "gated_linear": 2.0,
            },
        )


class EfficiencyPrior:
    """Uses historical Pareto frontier data to bias synthesis."""
    __slots__ = ("op_biases",)

    def __init__(self, frontier_data: List[Dict]):
        self.op_biases: Dict[str, float] = {}
        for p in (frontier_data or []):
            graph_json = p.get("graph_json", "")
            if not graph_json:
                continue
            for motif in ["selective_scan", "tropical", "clifford",
                          "low_rank", "sparse"]:
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
    tpl_weights = dict(config.template_weights) if config.template_weights else None
    motif_weights = config.motif_weights or None

    # High sparsity bias → force first template from efficiency pool
    _EFFICIENCY_TEMPLATES = {
        "sparse_moe_block", "routed_bottleneck", "token_merge_block",
        "conditional_compute", "sparse_ffn", "moe",
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
        _iter_weights = _first_tpl_weights if (t_idx == 0 and _use_efficiency_first) else tpl_weights
        trial_graph = copy.deepcopy(graph)
        trial_current = apply_template(
            trial_graph, current, rng,
            template_weights=_iter_weights,
            motif_weights=motif_weights,
        )

        if _graph_exceeds_final_budget(trial_graph, config):
            break
        graph = trial_graph
        current = trial_current

    # Ensure output shape is (B, S, D)
    result_shape = graph.nodes[current].output_shape
    if result_shape.dim != config.model_dim:
        current = graph.add_op("linear_proj", [current],
                               config={"out_dim": config.model_dim})

    if result_shape.is_freq_domain and "irfft_seq" in PRIMITIVE_REGISTRY:
        current = graph.add_op("irfft_seq", [current])

    # Optional outer residual connection (if not already added by template)
    # Check if the last op is already an add with input_id
    last_node = graph.nodes[current]
    has_outer_residual = (
        last_node.op_name == "add"
        and input_id in last_node.input_ids
    )
    if (
        not has_outer_residual
        and graph.n_ops() < config.max_ops
        and rng.random() < config.residual_prob
    ):
        try:
            current = graph.add_op("add", [input_id, current])
        except ValueError:
            pass

    graph.set_output(current)

    # Prune dead branches
    graph.prune_unreachable_nodes()

    # Post-generation validation
    _validate_graph(graph, config)

    return graph


_ROUTING_OPS: frozenset = frozenset({
    "entropy_router", "token_type_classifier", "route_topk", "route_lanes",
    "route_recursion", "adaptive_lane_mixer", "mixed_recursion_gate",
    "early_exit", "cascade", "speculative", "adaptive_recursion",
    "mod_topk", "token_merging", "token_merge", "relu_gate_routing",
    "moe_topk", "moe_2expert", "routing_conditioned_compression",
})


def _validate_graph(graph: ComputationGraph, config: GrammarConfig) -> None:
    """Validate a generated graph and raise ValueError if invalid."""
    result = validate_graph(
        graph,
        max_ops=config.max_ops,
        max_depth=config.max_depth,
        max_params_ratio=config.max_params_ratio * 3,
        min_splits=config.min_splits,
    )
    if not result.valid:
        raise ValueError(result.errors[0] if result.errors else
                         "Graph validation failed")

    # Algebraic space consistency check — reject graphs that mix
    # incompatible mathematical spaces (e.g., tropical after poincaré).
    space_err = _check_graph_space_consistency(graph)
    if space_err is not None:
        raise ValueError(space_err)

    # Routing-mandatory check: reject graphs without routing ops
    if config.routing_mandatory:
        op_names = {n.op_name for n in graph.nodes.values() if not n.is_input}
        if not op_names & _ROUTING_OPS:
            raise ValueError(
                "routing_mandatory=True but graph has no routing ops"
            )


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
    use_adaptive_synthesis: bool = False,
    prior: Optional[EfficiencyPrior] = None,
) -> List[ComputationGraph]:
    """Generate N unique computation graphs."""
    if config is None:
        config = GrammarConfig()

    graphs = []
    fingerprints: set = set()

    attempts = 0
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
        except (ValueError, RuntimeError):
            continue

    return graphs


# ── Legacy compatibility ────────────────────────────────────────────
# AdaptiveGenerator is still referenced by some test files and the
# use_adaptive_synthesis path. Keep it functional.

class AdaptiveGenerator:
    """Adaptive generator — delegates to motif-based generation."""
    __slots__ = ("config", "prior", "model_dim", "max_params", "max_flops")

    def __init__(self, config: GrammarConfig,
                 prior: Optional[EfficiencyPrior] = None):
        self.config = config
        self.prior = prior
        self.model_dim = config.model_dim
        self.max_params = int(config.max_params_ratio
                              * self.model_dim * self.model_dim)
        self.max_flops = 4 * (12 * self.model_dim * self.model_dim * 128)

    def generate(self, seed: Optional[int] = None) -> ComputationGraph:
        return generate_layer_graph(self.config, seed=seed)


# ── Shape compatibility check (used by external code) ───────────────

def _check_shape_compat(
    op: PrimitiveOp, input_shapes: List[ShapeInfo], model_dim: int,
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

    if op.name in ("local_window_attn", "sliding_window_mask",
                    "token_pool_restore", "selective_scan", "conv1d_seq",
                    "basis_expansion", "integral_kernel", "fixed_point_iter"):
        if not s0.is_standard:
            return False

    _MIN_DIM_OPS = {
        "softmax_attention": 16, "linear_attention": 16,
        "graph_attention": 16, "multi_head_mix": 4,
        "selective_scan": 8, "state_space": 8,
        "rwkv_time_mixing": 8, "rwkv_channel": 8,
        "conv1d_seq": 4, "moe_topk": 8, "moe_2expert": 8,
        "swiglu_mlp": 4, "topk_gate": 4,
        "block_sparse_linear": 16, "nm_sparse_linear": 8,
        "low_rank_proj": 8, "bottleneck_proj": 8,
        "grouped_linear": 8, "shared_basis_proj": 8,
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
