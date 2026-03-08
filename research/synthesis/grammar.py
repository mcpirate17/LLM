"""
Grammar-Based Random Graph Generator

Generates valid computation graphs by recursively building from primitives,
tracking shapes at every step. This is the core creative engine — it produces
programs that have never existed before.

The grammar has constraints that make programs VALID (shapes compose, gradient
flows, bounded depth/params) but not necessarily USEFUL. That's the evaluator's job.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .primitives import (
    PrimitiveOp, OpCategory, PRIMITIVE_REGISTRY,
    list_primitives, get_primitive,
    estimate_op_params,
)
from .graph import ComputationGraph, ShapeInfo, OpNode
from .validator import validate_graph

# Alias for backward compatibility — some test files import Node from grammar
Node = OpNode



# Action selection constraints
ACTION_WEIGHT_UNARY = 2.0
ACTION_WEIGHT_PARAM = 2.0
ACTION_WEIGHT_BINARY = 1.5
ACTION_SPLIT_MULTIPLIER = 3.0
ACTION_FREQ_MULTIPLIER = 3.0
ACTION_WEIGHT_TEMPLATE = 0.8
ACTION_STOP_BASE_WEIGHT = 5.0

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
    branch_depth: int = 1       # depth of subgraph processing on each branch (1=shallow, 2+=deep)
    max_recursion_depth: int = 4  # iteration cap for recursive ops (adaptive_recursion, fixed_point_iter, etc.)
    stability_check: bool = True  # validate architectures before compilation
    merge_prob: float = 0.4     # probability of merging paths
    risky_op_prob: float = 0.6  # probability of using numerically risky ops
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
        "frequency": 1.0,  # Increased from 0.5 to encourage basic ops
        "math_space": 1.5,
        "functional": 1.5,
    })
    # Excluded op names (if any)
    # Non-causal ops are excluded by default — they break the strict causality
    # gate in autoregressive evaluation (softmax_seq operates on full sequence,
    # sort/argsort reorder sequence, rfft/irfft use global FFT).
    excluded_ops: Set[str] = field(default_factory=lambda: {
        "softmax_seq", "mean_seq", "sum_seq",
        "sort_seq", "argsort_seq",
        "rfft_seq", "irfft_seq",
    })
    # Per-op weight multipliers (op_name -> weight, default 1.0 if absent).
    # Values < 1.0 soft-penalize weak ops; values > 1.0 boost strong ops.
    # Empty by default: the runtime learning system derives weights from actual data.
    op_weights: Dict[str, float] = field(default_factory=dict)
    
    # Structured Sparsity Constraints (Z7)
    structured_sparsity_bias: float = 0.0 # 0.0 to 1.0, nudge toward sparse ops
    enforce_block_size: Optional[int] = None # if set, force this block size
    min_block_density: float = 0.05
    max_block_density: float = 0.5

    # Hyperbolic Promotion (Phase 3)
    hyperbolic_promotion_threshold: float = 0.6
    hyperbolic_boost_factor: float = 3.0
    _hierarchy_fitness: Optional[float] = None  # set by analytics

    def update_bias(self, delta: float):
        """Adjust structured sparsity bias."""
        self.structured_sparsity_bias = max(0.0, min(1.0, self.structured_sparsity_bias + delta))

    @classmethod
    def exotic(cls, model_dim: int = 256) -> "GrammarConfig":
        """Return a config tuned for exotic architecture exploration.

        Boosts branching, routing, MoE, and compression primitives to ensure
        ~25% of search budget explores complex multi-lane architectures.
        """
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
            op_weights={
                # Routing ops
                "route_topk": 3.0,
                "route_lanes": 3.0,
                "route_recursion": 2.5,
                "mod_topk": 3.0,
                "early_exit": 2.0,
                "adaptive_recursion": 3.0,
                "token_merging": 2.0,
                "cascade": 2.0,
                "speculative": 2.0,
                # MoE / gating
                "moe_topk": 3.0,
                "adaptive_lane_mixer": 3.0,
                "mixed_recursion_gate": 2.5,
                "relu_gate_routing": 2.5,
                # Compression
                "latent_attention_compressor": 3.0,
                "routing_conditioned_compression": 2.5,
                "progressive_compression_gate": 2.0,
                "compression_mixture_experts": 2.5,
                # Classification / routing support
                "token_type_classifier": 2.5,
                "entropy_router": 2.5,
            },
        )


class EfficiencyPrior:
    """
    Uses historical Pareto frontier data to bias synthesis toward efficient patterns.
    (Project Hephaestus Phase 4)
    """
    def __init__(self, frontier_data: List[Dict]):
        self.op_biases: Dict[str, float] = {}
        for p in (frontier_data or []):
            graph_json = p.get("graph_json", "")
            if not graph_json: continue
            # Look for recurring motifs in successful efficient models
            for motif in ["selective_scan", "tropical", "clifford", "low_rank", "sparse"]:
                if motif in graph_json:
                    mult = 1.12 if motif == "tropical" else 1.05
                    self.op_biases[motif] = self.op_biases.get(motif, 1.0) * mult

    def get_bias(self, op_name: str) -> float:
        bias = 1.0
        for motif, multiplier in self.op_biases.items():
            if motif in op_name:
                bias *= multiplier
        return min(2.5, bias)


class AdaptiveGenerator:
    """
    High-performance recursive generator with look-ahead budget pruning.
    
    Instead of generating a full graph and then rejecting it, this builder
    estimates FLOPs and Params at every step and prunes branches that
    mathematically cannot fit the budget.
    """
    def __init__(self, config: GrammarConfig, prior: Optional[EfficiencyPrior] = None):
        self.config = config
        self.prior = prior
        self.model_dim = config.model_dim
        self.max_params = int(config.max_params_ratio * self.model_dim * self.model_dim)
        # 4x Transformer complexity budget for search
        self.max_flops = 4 * (12 * self.model_dim * self.model_dim * 128)

        # Import Cython fast-path params estimator
        try:
            from .adaptive_sampler import c_estimate_op_params
            self._c_params_estimator = c_estimate_op_params
        except ImportError:
            self._c_params_estimator = None

    def generate(self, seed: Optional[int] = None) -> ComputationGraph:
        rng = random.Random(seed)
        graph = ComputationGraph(self.model_dim)
        input_id = graph.add_input()
        
        try:
            self._build_recursive(
                graph, rng, [input_id],
                depth=0, params_acc=0, flops_acc=0
            )
        except RecursionError:
            pass # Hard prune on depth

        # Ensure we have a valid output
        if not graph.nodes:
            # Emergency fallback: identity
            input_id = graph.add_input()
            graph.set_output(input_id)
            return graph
            
        res_id = list(graph.nodes.keys())[-1]
        try:
            graph.set_output(res_id)
        except ValueError:
            # Fix dimension mismatch at output
            res_id = graph.add_op("linear_proj", [res_id], 
                                  config={"out_dim": self.model_dim})
            graph.set_output(res_id)
            
        return graph

    def _build_recursive(self, graph, rng, nodes, depth, params_acc, flops_acc):
        if depth >= self.config.max_depth or params_acc > self.max_params or flops_acc > self.max_flops:
            return

        # Adaptive stop probability
        stop_p = max((depth / self.config.max_depth)**2, 
                     params_acc / self.max_params,
                     flops_acc / self.max_flops)
        if rng.random() < stop_p:
            return

        # Choose action
        action = _choose_action(self.config, rng, depth, len(nodes), len(graph.nodes))
        if action == "stop": return

        # Pick node and op
        node_id = rng.choice(nodes)
        d_in = graph.nodes[node_id].output_shape.dim
        
        # Budget-safe selection
        categories = [OpCategory.PARAMETERIZED, OpCategory.MIXING, OpCategory.LINEAR_ALGEBRA, 
                      OpCategory.SEQUENCE, OpCategory.ELEMENTWISE_UNARY]
        
        candidates = []
        weights = []
        
        for cat in categories:
            cat_w = self.config.category_weights.get(cat.value, 1.0)
            for op in list_primitives(cat):
                if op.n_inputs != 1 or op.name in self.config.excluded_ops:
                    continue
                
                # Fast Look-Ahead Estimate
                op_p = self._estimate_params(op, d_in)
                if params_acc + op_p > self.max_params: continue
                
                op_f = self._estimate_flops(op, d_in)
                if flops_acc + op_f > self.max_flops: continue
                
                # Check shape compatibility
                if _check_shape_compat(op, [graph.nodes[node_id].output_shape], self.model_dim):
                    candidates.append((op, op_p, op_f))
                    
                    # Apply Biases
                    op_w = self.config.op_weights.get(op.name, 1.0)
                    if self.prior:
                        op_w *= self.prior.get_bias(op.name)
                    weights.append(cat_w * op_w)

        if not candidates: return
        
        choice, op_p, op_f = rng.choices(candidates, weights=weights, k=1)[0]
        
        try:
            new_id = graph.add_op(choice.name, [node_id])
            self._build_recursive(graph, rng, nodes + [new_id], 
                                  depth + 1, params_acc + op_p, flops_acc + op_f)
        except ValueError:
            return

    def _estimate_params(self, op: PrimitiveOp, d_in: int) -> int:
        if self._c_params_estimator:
            # Fast Cython path
            try:
                return int(self._c_params_estimator(op.name.encode("utf-8"), d_in, d_in))
            except Exception:
                pass
        return estimate_op_params(op, d_in)

    def _estimate_flops(self, op: PrimitiveOp, d_in: int) -> int:
        # Simplified estimate for look-ahead (B=1, S=128)
        # For most linear-like ops, flops approx 2*S*D_in*D_out
        if op.category == OpCategory.PARAMETERIZED or op.category == OpCategory.LINEAR_ALGEBRA:
            return 2 * 128 * d_in * d_in
        return 128 * d_in # elementwise


def generate_layer_graph(
    config: Optional[GrammarConfig] = None,
    seed: Optional[int] = None,
) -> ComputationGraph:
    """Generate a random computation graph for a single layer.

    The graph takes (B, S, D) and produces (B, S, D).
    Uses recursive random construction with shape tracking.
    """
    if config is None:
        config = GrammarConfig()

    rng = random.Random(seed)
    graph = ComputationGraph(config.model_dim)

    # Start with input
    input_id = graph.add_input()

    # Build the computation
    config._split_counter = [0]  # mutable counter for forced splits
    result_id = _build_subgraph(
        graph, config, rng,
        available_nodes=[input_id],
        current_depth=0,
        n_ops_so_far=0,
        params_so_far=0,
    )

    # Ensure output shape is (B, S, D)
    result_shape = graph.nodes[result_id].output_shape
    if result_shape.dim != config.model_dim:
        # Add a linear projection to fix dimension
        result_id = graph.add_op("linear_proj", [result_id],
                                  config={"out_dim": config.model_dim})

    if result_shape.is_freq_domain and "irfft_seq" in PRIMITIVE_REGISTRY:
        # Return from frequency domain
        result_id = graph.add_op("irfft_seq", [result_id])

    # Optional residual connection
    if rng.random() < config.residual_prob:
        result_id = graph.add_op("add", [input_id, result_id])

    graph.set_output(result_id)

    # Prune dead branches: the recursive builder keeps old nodes in
    # available_nodes for skip connections, but some never connect to
    # the output path. Strip them before validation.
    graph.prune_unreachable_nodes()

    # Post-generation validation
    _validate_graph(graph, config)

    return graph


def _validate_graph(graph: ComputationGraph, config: GrammarConfig) -> None:
    """Validate a generated graph and raise ValueError if invalid."""
    result = validate_graph(
        graph,
        max_ops=config.max_ops,
        max_depth=config.max_depth,
        max_params_ratio=config.max_params_ratio,
        min_splits=config.min_splits,
    )
    if not result.valid:
        raise ValueError(result.errors[0] if result.errors else "Graph validation failed")


def _build_subgraph(
    graph: ComputationGraph,
    config: GrammarConfig,
    rng: random.Random,
    available_nodes: List[int],
    current_depth: int,
    n_ops_so_far: int,
    params_so_far: int,
) -> int:
    """Recursively build a subgraph. Returns the output node ID."""

    # Safety guard: hard cap at depth 15 regardless of config to prevent
    # Python stack overflow from unbounded grammar parameter growth.
    _HARD_DEPTH_LIMIT = 15

    # Base case: stop if we've hit limits
    if (current_depth >= config.max_depth
            or current_depth >= _HARD_DEPTH_LIMIT
            or n_ops_so_far >= config.max_ops):
        return available_nodes[-1]  # return most recent node

    # Force split_merge if we haven't met min_splits quota yet
    # (only if we have enough budget and depth remaining)
    sc = getattr(config, '_split_counter', [0])
    force_split = (
        config.min_splits > 0
        and sc[0] < config.min_splits
        and current_depth < config.max_depth - 3
        and n_ops_so_far < config.max_ops - 4
    )

    # Decide what to do
    if force_split:
        action = "split_merge"
    else:
        action = _choose_action(config, rng, current_depth, len(available_nodes),
                                n_ops_so_far)

    if action == "unary_op":
        return _apply_unary(graph, config, rng, available_nodes,
                           current_depth, n_ops_so_far, params_so_far)

    elif action == "binary_op":
        return _apply_binary(graph, config, rng, available_nodes,
                            current_depth, n_ops_so_far, params_so_far)

    elif action == "split_merge":
        return _split_and_merge(graph, config, rng, available_nodes,
                               current_depth, n_ops_so_far, params_so_far)

    elif action == "freq_detour":
        return _freq_domain_detour(graph, config, rng, available_nodes,
                                   current_depth, n_ops_so_far, params_so_far)

    elif action == "template":
        from .templates import apply_random_template
        node_id = available_nodes[-1]
        try:
            result_id = apply_random_template(graph, node_id, rng,
                                                     excluded_ops=config.excluded_ops)
            return _build_subgraph(
                graph, config, rng,
                available_nodes=available_nodes + [result_id],
                current_depth=current_depth + 2,
                n_ops_so_far=n_ops_so_far + 3,
                params_so_far=params_so_far,
            )
        except Exception:
            return node_id

    elif action == "parameterized":
        return _apply_parameterized(graph, config, rng, available_nodes,
                                    current_depth, n_ops_so_far, params_so_far)

    else:
        return available_nodes[-1]


def _choose_action(
    config: GrammarConfig, rng: random.Random,
    depth: int, n_available: int, n_ops: int,
) -> str:
    """Choose what action to take next."""
    actions = []
    weights = []

    # Always available
    actions.append("unary_op")
    weights.append(ACTION_WEIGHT_UNARY)

    actions.append("parameterized")
    weights.append(ACTION_WEIGHT_PARAM)

    # Binary ops need 2+ available nodes
    if n_available >= 2:
        actions.append("binary_op")
        weights.append(ACTION_WEIGHT_BINARY)

    # Split/merge for parallelism
    if depth < config.max_depth - 3 and n_ops < config.max_ops - 4:
        actions.append("split_merge")
        weights.append(config.split_prob * ACTION_SPLIT_MULTIPLIER)

    # Frequency domain detour (disabled when rfft_seq/irfft_seq are excluded —
    # they break causality in autoregressive models)
    if (depth < config.max_depth - 2 and n_ops < config.max_ops - 2
            and "rfft_seq" not in config.excluded_ops):
        actions.append("freq_detour")
        weights.append(config.freq_domain_prob * ACTION_FREQ_MULTIPLIER)

    # Template action (opinionated seeds from mined survivors)
    if depth < config.max_depth - 3 and n_ops < config.max_ops - 4:
        actions.append("template")
        weights.append(ACTION_WEIGHT_TEMPLATE)

    # Stop (more likely as we go deeper, but never before min_depth)
    if depth >= config.min_depth:
        actions.append("stop")
        stop_weight = ((depth - config.min_depth) / max(1, config.max_depth - config.min_depth)) ** 2 * ACTION_STOP_BASE_WEIGHT
        weights.append(stop_weight)

    return rng.choices(actions, weights=weights, k=1)[0]


def _pick_op(
    config: GrammarConfig, rng: random.Random,
    categories: List[OpCategory],
    input_shapes: List[ShapeInfo],
    model_dim: int,
) -> Optional[str]:
    """Pick a valid operation from the given categories."""
    candidates = []
    weights = []

    for cat in categories:
        cat_weight = config.category_weights.get(cat.value, 1.0)
        for op in list_primitives(cat):
            if op.name in config.excluded_ops:
                continue
            if not op.standalone:
                continue
            if op.n_inputs != len(input_shapes):
                continue
            if op.numerically_risky and rng.random() > config.risky_op_prob:
                continue
            # Check if shapes are compatible
            if _check_shape_compat(op, input_shapes, model_dim):
                candidates.append(op.name)
                op_w = config.op_weights.get(op.name, 1.0)
                
                # Apply Z7: Structured Sparsity Bias
                if config.structured_sparsity_bias > 0:
                    if op.name in {"block_sparse_linear", "nm_sparse_linear", "semi_structured_2_4_linear"}:
                        op_w *= (1.0 + config.structured_sparsity_bias * 2.0)
                    elif op.name in {"linear_proj", "linear_proj_down", "linear_proj_up"}:
                        op_w *= (1.0 - config.structured_sparsity_bias * 0.5)

                # Hyperbolic Promotion: boost hyperbolic ops when hierarchy detected
                if (config._hierarchy_fitness is not None
                        and config._hierarchy_fitness > config.hyperbolic_promotion_threshold):
                    _HYP_OPS = {"poincare_add", "exp_map", "log_map",
                                "hyp_linear", "hyp_distance", "hyp_tangent_nonlinear",
                                "hyperbolic_norm"}
                    if op.name in _HYP_OPS:
                        op_w *= config.hyperbolic_boost_factor
                
                weights.append(cat_weight * op_w)

    if not candidates:
        return None

    return rng.choices(candidates, weights=weights, k=1)[0]


def _check_shape_compat(op: PrimitiveOp, input_shapes: List[ShapeInfo], model_dim: int) -> bool:
    """Quick check if an op is compatible with given input shapes."""
    if not input_shapes:
        return False

    # Arity check: op must accept exactly the number of inputs provided
    if op.n_inputs != len(input_shapes):
        return False

    s0 = input_shapes[0]

    # Split ops need divisible dimensions and minimum output dim
    if op.name == "split2":
        if s0.dim % 2 != 0 or s0.dim // 2 < 4:
            return False
    if op.name == "split3":
        if s0.dim % 3 != 0 or s0.dim // 3 < 4:
            return False

    # FFT ops need standard seq dimension
    if op.shape_rule == "rfft" and not s0.is_standard:
        return False
    if op.shape_rule == "irfft" and not s0.is_freq_domain:
        return False

    # Sequence-dependent ops need standard (non-freq) tensors
    if op.name in ("local_window_attn", "sliding_window_mask",
                    "token_pool_restore", "selective_scan", "conv1d_seq",
                    "basis_expansion", "integral_kernel", "fixed_point_iter"):
        if not s0.is_standard:
            return False

    # Minimum dimension requirements for complex ops
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

    # Binary ops need matching seq dims
    if len(input_shapes) == 2:
        s1 = input_shapes[1]
        if op.shape_rule == "binary_broadcast":
            if s0.seq != s1.seq:
                return False
            if s0.dim != s1.dim and s0.dim != 1 and s1.dim != 1:
                return False
        elif op.shape_rule == "matmul":
            if s0.seq != s1.seq:
                return False
        elif op.shape_rule == "concat":
            if s0.seq != s1.seq:
                return False

    return True


def _apply_unary(graph, config, rng, available_nodes,
                 depth, n_ops, params_so_far) -> int:
    """Apply a unary operation."""
    node_id = rng.choice(available_nodes)
    shape = graph.nodes[node_id].output_shape

    categories = [
        OpCategory.ELEMENTWISE_UNARY,
        OpCategory.SEQUENCE,
        OpCategory.MATH_SPACE,
    ]

    op_name = _pick_op(config, rng, categories, [shape], config.model_dim)
    if op_name is None:
        return node_id

    try:
        op_config = {}
        if op_name in ("local_window_attn", "sliding_window_mask"):
            op_config["window_size"] = min(rng.choice([8, 16, 32, 64]), 32)
        elif op_name == "multi_head_mix":
            # Pick n_heads that divides D
            D = shape.dim
            candidates = [h for h in [2, 4, 8] if D % h == 0]
            op_config["n_heads"] = rng.choice(candidates) if candidates else 1
        new_id = graph.add_op(op_name, [node_id], config=op_config)
    except ValueError:
        return node_id

    # Continue building
    return _build_subgraph(
        graph, config, rng,
        available_nodes=available_nodes + [new_id],
        current_depth=depth + 1,
        n_ops_so_far=n_ops + 1,
        params_so_far=params_so_far,
    )


def _apply_binary(graph, config, rng, available_nodes,
                  depth, n_ops, params_so_far) -> int:
    """Apply a binary operation to two available nodes."""
    if len(available_nodes) < 2:
        return available_nodes[-1]

    # Pick two compatible nodes
    rng.shuffle(available_nodes)
    for i in range(len(available_nodes)):
        for j in range(i + 1, len(available_nodes)):
            s_i = graph.nodes[available_nodes[i]].output_shape
            s_j = graph.nodes[available_nodes[j]].output_shape

            categories = [OpCategory.ELEMENTWISE_BINARY,
                          OpCategory.FUNCTIONAL]
            if s_i.dim == s_j.dim:
                categories.append(OpCategory.LINEAR_ALGEBRA)
                categories.append(OpCategory.MATH_SPACE)

            op_name = _pick_op(config, rng, categories, [s_i, s_j], config.model_dim)
            if op_name is not None:
                try:
                    new_id = graph.add_op(op_name,
                                          [available_nodes[i], available_nodes[j]])
                    return _build_subgraph(
                        graph, config, rng,
                        available_nodes=[new_id] + [n for n in available_nodes
                                                    if n not in (available_nodes[i], available_nodes[j])],
                        current_depth=depth + 1,
                        n_ops_so_far=n_ops + 1,
                        params_so_far=params_so_far,
                    )
                except ValueError:
                    continue

    return available_nodes[-1]


def _apply_parameterized(graph, config, rng, available_nodes,
                         depth, n_ops, params_so_far) -> int:
    """Apply a parameterized (learnable) operation."""
    node_id = rng.choice(available_nodes)
    shape = graph.nodes[node_id].output_shape

    # Use actual input dim for param estimation, not global model_dim
    D_actual = shape.dim
    D_global = config.model_dim
    max_params = int(config.max_params_ratio * D_global * D_global)

    # Pick a parameterized op that doesn't exceed param budget
    candidates = []
    cand_cat_weights = []
    for cat in (OpCategory.PARAMETERIZED, OpCategory.MATH_SPACE, OpCategory.FUNCTIONAL):
        cat_w = config.category_weights.get(cat.value, 1.0)
        for op in list_primitives(cat):
            if op.name in config.excluded_ops:
                continue
            if not op.standalone:
                continue
            # Only pick 1-input ops here; 2-input ops go through _apply_binary
            if op.n_inputs != 1:
                continue
            if not _check_shape_compat(op, [shape], D_global):
                continue
            # Estimate params using actual input dim
            op_params = estimate_op_params(op, D_actual)
            if params_so_far + op_params <= max_params:
                candidates.append((op.name, op_params))
                cand_cat_weights.append(cat_w)

    if not candidates:
        return node_id

    # Weighted selection using per-op weights * category weights
    cand_weights = [config.op_weights.get(name, 1.0) * cand_cat_weights[i] for i, (name, _) in enumerate(candidates)]
    op_name, op_params = rng.choices(candidates, weights=cand_weights, k=1)[0]

    try:
        op_config = {}
        if op_name in ("linear_proj", "linear_proj_down", "linear_proj_up"):
            if op_name == "linear_proj_down":
                op_config["out_dim"] = shape.dim // 2
            elif op_name == "linear_proj_up":
                op_config["out_dim"] = shape.dim * 2
            else:
                op_config["out_dim"] = shape.dim
        elif op_name == "fixed_point_iter":
            op_config["n_iters"] = min(rng.choice([2, 3, 4]), config.max_recursion_depth)
            op_config["damping"] = rng.choice([0.4, 0.5, 0.6])
        elif op_name in ("adaptive_recursion", "route_recursion", "mixed_recursion_gate"):
            op_config["max_depth"] = config.max_recursion_depth
        elif op_name == "integral_kernel":
            op_config["kernel_scale"] = rng.choice([0.15, 0.25, 0.35])
        elif op_name == "block_sparse_linear":
            op_config["block_size"] = config.enforce_block_size or rng.choice([8, 16, 32])
            op_config["block_density"] = rng.uniform(config.min_block_density, config.max_block_density)
        elif op_name == "nm_sparse_linear":
            op_config["n"] = 2
            op_config["m"] = 4
        elif op_name in ("swiglu_mlp", "rwkv_channel", "moe_topk", "rwkv_time_mixing"):
            op_config["mlp_ratio"] = rng.choice([2.0, 3.0, 4.0])
        # New parameterized ops don't need special config beyond defaults
        new_id = graph.add_op(op_name, [node_id], config=op_config)
    except ValueError:
        return node_id

    return _build_subgraph(
        graph, config, rng,
        available_nodes=available_nodes + [new_id],
        current_depth=depth + 1,
        n_ops_so_far=n_ops + 1,
        params_so_far=params_so_far + op_params,
    )


def _split_and_merge(graph, config, rng, available_nodes,
                     depth, n_ops, params_so_far) -> int:
    """Split into parallel paths, process, merge back."""
    node_id = available_nodes[-1]
    shape = graph.nodes[node_id].output_shape

    # Increment split counter
    sc = getattr(config, '_split_counter', [0])
    sc[0] += 1

    # Need enough dim for split + useful ops on each part
    if shape.dim < 16:
        return node_id

    # Decide 2-way or 3-way split
    use_three_way = (
        rng.random() < config.three_way_split_prob
        and shape.dim >= 48  # need enough dim for 3 parts
        and n_ops < config.max_ops - 6
        and "split3" in PRIMITIVE_REGISTRY
        and "split3" not in config.excluded_ops
    )

    split_op = "split3" if use_three_way else "split2"
    try:
        split_id = graph.add_op(split_op, [node_id])
    except ValueError:
        return node_id

    n_paths = 3 if use_three_way else 2
    branch_depth = max(1, config.branch_depth)

    # Budget management: reserve ops for merge infrastructure + continuation
    # merge=1 (2-way) or 2 (3-way), linear_proj=1, continuation=2
    merge_overhead = (2 if use_three_way else 1) + 1 + 2
    remaining = config.max_ops - n_ops - 1  # subtract 1 for split op
    # Cap branch budget: each branch gets at most half the remaining budget
    # (after overhead), minimum 1
    branch_budget = max(1, (remaining - merge_overhead) // (n_paths + 1))

    # Process each path with a subgraph of configurable depth
    paths = []
    ops_used = 1  # for the split op
    for p in range(n_paths):
        if branch_depth >= 2 or p == 0:
            # Subgraph branch: use budget-capped _build_subgraph
            # Inflate n_ops_so_far so the branch only has branch_budget room
            capped_n_ops = max(n_ops + ops_used, config.max_ops - branch_budget)
            # Push depth closer to limit — leave room for merge overhead (3 depth)
            capped_depth = max(depth + 1, config.max_depth - 4)
            path_out = _build_subgraph(
                graph, config, rng,
                available_nodes=[split_id],
                current_depth=capped_depth,
                n_ops_so_far=capped_n_ops,
                params_so_far=params_so_far,
            )
        else:
            # Shallow branch: single op from diverse set
            branch_ops = [
                "tanh", "sigmoid", "relu", "gelu", "silu", "sin", "square",
                "learnable_scale", "learnable_bias",
                "sparse_threshold", "spike_rate_code", "padic_gate",
                "tropical_gate",
            ]
            valid_ops = [op for op in branch_ops
                         if op not in config.excluded_ops and op in PRIMITIVE_REGISTRY]
            if valid_ops:
                b_op = rng.choice(valid_ops)
                try:
                    path_out = graph.add_op(b_op, [split_id])
                except ValueError:
                    path_out = split_id
            else:
                path_out = split_id
        paths.append(path_out)
        ops_used += 1

    # Merge all paths
    try:
        if len(paths) == 2:
            merged = graph.add_op("concat", [paths[0], paths[1]])
        else:
            # 3-way: concat first two, then concat with third
            merged_ab = graph.add_op("concat", [paths[0], paths[1]])
            merged = graph.add_op("concat", [merged_ab, paths[2]])
            ops_used += 1
    except ValueError:
        return paths[0]

    # Project back to original dim if needed
    merged_shape = graph.nodes[merged].output_shape
    if merged_shape.dim != config.model_dim:
        try:
            merged = graph.add_op("linear_proj", [merged],
                                   config={"out_dim": config.model_dim})
            ops_used += 1
        except ValueError:
            return paths[0]

    return _build_subgraph(
        graph, config, rng,
        available_nodes=[merged],
        current_depth=depth + 3,
        n_ops_so_far=n_ops + ops_used + 2,
        params_so_far=params_so_far + config.model_dim * config.model_dim,
    )


def _freq_domain_detour(graph, config, rng, available_nodes,
                        depth, n_ops, params_so_far) -> int:
    """Take a detour through frequency domain."""
    node_id = available_nodes[-1]

    # Guard: rfft_seq/irfft_seq may have been removed from primitives
    if "rfft_seq" not in PRIMITIVE_REGISTRY or "irfft_seq" not in PRIMITIVE_REGISTRY:
        return node_id

    shape = graph.nodes[node_id].output_shape

    if not shape.is_standard:
        return node_id

    try:
        # Go to frequency domain
        freq_id = graph.add_op("rfft_seq", [node_id])

        # Apply some ops in frequency domain
        freq_ops = ["learnable_scale", "mul", "square", "sigmoid"]
        valid_freq_ops = [op for op in freq_ops
                         if op not in config.excluded_ops and op in PRIMITIVE_REGISTRY]
        if valid_freq_ops:
            op_name = rng.choice(valid_freq_ops)
            op = get_primitive(op_name)
            if op.n_inputs == 1:
                freq_id = graph.add_op(op_name, [freq_id])

        # Come back to time domain
        time_id = graph.add_op("irfft_seq", [freq_id])

        return _build_subgraph(
            graph, config, rng,
            available_nodes=available_nodes + [time_id],
            current_depth=depth + 3,
            n_ops_so_far=n_ops + 3,
            params_so_far=params_so_far,
        )
    except ValueError:
        return node_id


def batch_generate(
    n: int,
    config: Optional[GrammarConfig] = None,
    base_seed: int = 42,
    use_adaptive_synthesis: bool = False,
    prior: Optional[EfficiencyPrior] = None,
) -> List[ComputationGraph]:
    """Generate N random computation graphs."""
    if config is None:
        config = GrammarConfig()

    graphs = []
    fingerprints = set()

    attempts = 0
    max_attempts = n * 10
    
    # Initialize adaptive generator if requested
    generator = None
    if use_adaptive_synthesis:
        generator = AdaptiveGenerator(config, prior=prior)

    while len(graphs) < n and attempts < max_attempts:
        attempts += 1
        seed = base_seed + attempts * 137
        try:
            if generator:
                g = generator.generate(seed=seed)
            else:
                g = generate_layer_graph(config, seed=seed)
                
            fp = g.fingerprint()
            if fp not in fingerprints:
                fingerprints.add(fp)
                graphs.append(g)
        except (ValueError, RuntimeError):
            continue

    return graphs
