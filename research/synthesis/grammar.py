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
    safe_eval_formula,
)
from .graph import ComputationGraph, ShapeInfo, OpNode


@dataclass
class GrammarConfig:
    """Configuration for the graph generator."""
    model_dim: int = 256
    min_depth: int = 3
    max_depth: int = 10
    max_width: int = 4          # max parallel paths
    max_ops: int = 16           # max total operations
    max_params_ratio: float = 8.0  # max params relative to D^2
    residual_prob: float = 0.7  # probability of residual connection
    split_prob: float = 0.3     # probability of branching into parallel paths
    stability_check: bool = True  # validate architectures before compilation
    merge_prob: float = 0.4     # probability of merging paths
    risky_op_prob: float = 0.2  # probability of using numerically risky ops
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
        "functional": 1.0,
    })
    # Excluded op names (if any)
    excluded_ops: Set[str] = field(default_factory=set)
    # Per-op weight multipliers (op_name -> weight, default 1.0 if absent).
    # Values < 1.0 soft-penalize weak ops; values > 1.0 boost strong ops.
    op_weights: Dict[str, float] = field(default_factory=dict)
    
    # Structured Sparsity Constraints (Z7)
    structured_sparsity_bias: float = 0.0 # 0.0 to 1.0, nudge toward sparse ops
    enforce_block_size: Optional[int] = None # if set, force this block size
    min_block_density: float = 0.05
    max_block_density: float = 0.5

    def update_bias(self, delta: float):
        """Adjust structured sparsity bias."""
        self.structured_sparsity_bias = max(0.0, min(1.0, self.structured_sparsity_bias + delta))


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

    if result_shape.is_freq_domain:
        # Return from frequency domain
        result_id = graph.add_op("irfft_seq", [result_id])

    # Optional residual connection
    if rng.random() < config.residual_prob:
        result_id = graph.add_op("add", [input_id, result_id])

    graph.set_output(result_id)
    return graph


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

    # Decide what to do
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
            result_id = apply_random_template(graph, node_id, rng)
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
    weights.append(2.0)

    actions.append("parameterized")
    weights.append(2.0)

    # Binary ops need 2+ available nodes
    if n_available >= 2:
        actions.append("binary_op")
        weights.append(1.5)

    # Split/merge for parallelism
    if depth < config.max_depth - 3 and n_ops < config.max_ops - 4:
        actions.append("split_merge")
        weights.append(config.split_prob * 3)

    # Frequency domain detour
    if depth < config.max_depth - 2 and n_ops < config.max_ops - 2:
        actions.append("freq_detour")
        weights.append(config.freq_domain_prob * 3)

    # Template action (opinionated seeds from mined survivors)
    if depth < config.max_depth - 3 and n_ops < config.max_ops - 4:
        actions.append("template")
        weights.append(0.8)

    # Stop (more likely as we go deeper)
    actions.append("stop")
    stop_weight = (depth / config.max_depth) ** 2 * 5
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
                
                weights.append(cat_weight * op_w)

    if not candidates:
        return None

    return rng.choices(candidates, weights=weights, k=1)[0]


def _check_shape_compat(op: PrimitiveOp, input_shapes: List[ShapeInfo], model_dim: int) -> bool:
    """Quick check if an op is compatible with given input shapes."""
    if not input_shapes:
        return False

    s0 = input_shapes[0]

    # Split ops need divisible dimensions
    if op.name == "split2" and s0.dim % 2 != 0:
        return False
    if op.name == "split3" and s0.dim % 3 != 0:
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

    # multi_head_mix needs D divisible by at least 2
    if op.name == "multi_head_mix" and s0.dim < 2:
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

            categories = [OpCategory.ELEMENTWISE_BINARY]
            if s_i.dim == s_j.dim:
                categories.append(OpCategory.LINEAR_ALGEBRA)

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

    D = config.model_dim
    max_params = int(config.max_params_ratio * D * D)

    # Pick a parameterized op that doesn't exceed param budget
    candidates = []
    for op in list_primitives(OpCategory.PARAMETERIZED):
        if op.name in config.excluded_ops:
            continue
        formula = op.param_formula.replace("D", str(D))
        try:
            op_params = safe_eval_formula(formula)
        except Exception:
            op_params = D * D
        if params_so_far + op_params <= max_params:
            if _check_shape_compat(op, [shape], D):
                candidates.append((op.name, op_params))

    # Also include math space ops
    for op in list_primitives(OpCategory.MATH_SPACE):
        if op.name in config.excluded_ops:
            continue
        if op.n_inputs == 1 and _check_shape_compat(op, [shape], D):
            formula = op.param_formula.replace("D", str(D))
            try:
                op_params = safe_eval_formula(formula)
            except Exception:
                op_params = 0
            if params_so_far + op_params <= max_params:
                candidates.append((op.name, op_params))

    # Include functional operator-learning primitives
    for op in list_primitives(OpCategory.FUNCTIONAL):
        if op.name in config.excluded_ops:
            continue
        if op.n_inputs == 1 and _check_shape_compat(op, [shape], D):
            formula = op.param_formula.replace("D", str(D))
            try:
                op_params = safe_eval_formula(formula)
            except Exception:
                op_params = D * D
            if params_so_far + op_params <= max_params:
                candidates.append((op.name, op_params))

    if not candidates:
        return node_id

    # Weighted selection using per-op weights for soft penalties
    cand_weights = [config.op_weights.get(name, 1.0) for name, _ in candidates]
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
            op_config["n_iters"] = rng.choice([2, 3, 4])
            op_config["damping"] = rng.choice([0.4, 0.5, 0.6])
        elif op_name == "integral_kernel":
            op_config["kernel_scale"] = rng.choice([0.15, 0.25, 0.35])
        elif op_name == "block_sparse_linear":
            op_config["block_size"] = config.enforce_block_size or rng.choice([8, 16, 32])
            op_config["block_density"] = rng.uniform(config.min_block_density, config.max_block_density)
        elif op_name == "nm_sparse_linear":
            op_config["n"] = 2
            op_config["m"] = 4
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

    if shape.dim < 4:
        return node_id

    # Split into 2 paths
    try:
        split_id = graph.add_op("split2", [node_id])
    except ValueError:
        return node_id

    half_dim = shape.dim // 2

    # Process each path independently (shallow subgraphs)
    path_a = _build_subgraph(
        graph, config, rng,
        available_nodes=[split_id],
        current_depth=depth + 2,
        n_ops_so_far=n_ops + 1,
        params_so_far=params_so_far,
    )

    # Second path: apply a different op to the same split
    path_b_ops = [
        "tanh", "sigmoid", "relu", "gelu", "silu", "sin", "square",
        "learnable_scale", "learnable_bias",
    ]
    valid_ops = [op for op in path_b_ops
                 if op not in config.excluded_ops and op in PRIMITIVE_REGISTRY]
    if valid_ops:
        b_op = rng.choice(valid_ops)
        try:
            path_b = graph.add_op(b_op, [split_id])
        except ValueError:
            path_b = split_id
    else:
        path_b = split_id

    # Merge paths
    try:
        merged = graph.add_op("concat", [path_a, path_b])
    except ValueError:
        return path_a

    # Project back to original dim if needed
    merged_shape = graph.nodes[merged].output_shape
    if merged_shape.dim != config.model_dim:
        try:
            merged = graph.add_op("linear_proj", [merged],
                                   config={"out_dim": config.model_dim})
        except ValueError:
            return path_a

    return _build_subgraph(
        graph, config, rng,
        available_nodes=[merged],
        current_depth=depth + 3,
        n_ops_so_far=n_ops + 4,
        params_so_far=params_so_far + config.model_dim * config.model_dim,
    )


def _freq_domain_detour(graph, config, rng, available_nodes,
                        depth, n_ops, params_so_far) -> int:
    """Take a detour through frequency domain."""
    node_id = available_nodes[-1]
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
) -> List[ComputationGraph]:
    """Generate N random computation graphs."""
    if config is None:
        config = GrammarConfig()

    graphs = []
    fingerprints = set()

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
