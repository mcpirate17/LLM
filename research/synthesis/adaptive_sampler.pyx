# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False

"""
Adaptive Sampler (Project Hephaestus Phase 4)

High-performance Cython implementation of the Aria Grammar.
Integrates real-time budget (FLOPs/Params) look-ahead and efficiency priors.
"""

from libc.math cimport log2, exp, pow, sqrt, floor
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map as cpp_map
import random
import numpy as np
cimport numpy as cnp

# Import primitives and metadata from Python (cached)
from .primitives import PRIMITIVE_REGISTRY, get_primitive, OpCategory

cdef float _INF = 1e30

# ── High-Performance Parameter Estimator ─────────────────────────────

cdef long c_estimate_op_params(string op_name, int d_in, int d_out):
    """Native parameter estimation for common op patterns."""
    # This mirrors safe_eval_formula for common cases to avoid Python calls
    if op_name == b"linear_proj" or op_name == b"linear_proj_down" or op_name == b"linear_proj_up":
        return d_in * d_out
    elif op_name == b"low_rank_proj":
        # bottleneck rank = d_in // 4
        return d_in * (d_in // 4) + (d_in // 4) * d_out
    elif op_name == b"block_sparse_linear":
        # approx based on standard 0.25 density
        return int(d_in * d_out * 0.25)
    elif op_name == b"nm_sparse_linear":
        return int(d_in * d_out * 0.5)
    elif op_name == b"softmax_attention" or op_name == b"linear_attention":
        # QKV projections: 3 * D * D
        return 3 * d_in * d_in
    elif op_name == b"add" or op_name == b"mul" or op_name == b"relu":
        return 0
    # Fallback to a conservative estimate
    return d_in * d_out

# ── Efficiency Prior ───────────────────────────────────────────────

class EfficiencyPrior:
    """Uses the Pareto frontier to bias sampling toward efficient structures."""
    def __init__(self, frontier_data: list):
        self.frontier = frontier_data
        # Extract winning op frequencies from the frontier
        self.op_biases = {}
        for p in self.frontier:
            graph_json = p.get("graph_json", "")
            if not graph_json: continue
            # Basic tokenization for speed
            for op in ["selective_scan", "tropical", "clifford", "low_rank", "sparse"]:
                if op in graph_json:
                    self.op_biases[op] = self.op_biases.get(op, 1.0) * 1.1

    def get_bias(self, str op_name) -> float:
        bias = 1.0
        for k, v in self.op_biases.items():
            if k in op_name:
                bias *= v
        return min(3.0, bias)

# ── Adaptive Generator ─────────────────────────────────────────────

class AdaptiveGenerator:
    """
    Cython-backed graph generator with real-time budget pruning.
    
    Logic:
    - Prunes branches that exceed Param/FLOP budget *during* recursion.
    - Uses EfficiencyPrior to bias opcode selection.
    """
    def __init__(self, config, prior=None):
        self.config = config
        self.prior = prior
        self.model_dim = config.model_dim
        self.max_params = 4 * self.model_dim * self.model_dim * 12  # VRAM is the real constraint
        # Max FLOPs relative to a standard Transformer layer (approx 12*D^2*S)
        # We cap at 4x Transformer complexity for the search.
        self.max_flops = 4 * (12 * self.model_dim * self.model_dim * 128)

    def generate(self, seed=None):
        """Python entry point for synthesis."""
        from .graph import ComputationGraph
        rng = random.Random(seed)
        graph = ComputationGraph(self.model_dim)
        
        input_id = graph.add_input()
        
        # Start recursive build
        try:
            self._recursive_build(
                graph, rng,
                available_nodes=[input_id],
                depth=0,
                params_acc=0,
                flops_acc=0
            )
        except Exception as e:
            # If we pruned everything, just return a minimal graph
            pass
            
        # Final connection logic (standard Aria fallback)
        res_id = list(graph.nodes.keys())[-1]
        graph.set_output(res_id)
        return graph

    def _recursive_build(self, graph, rng, available_nodes, depth, params_acc, flops_acc):
        """Recursive build with look-ahead pruning."""
        if depth >= self.config.max_depth or params_acc > self.max_params or flops_acc > self.max_flops:
            return

        # 1. Choose next action
        action = self._choose_action_adaptive(rng, depth, params_acc, flops_acc)
        if action == "stop":
            return

        # 2. Pick node and op
        node_id = rng.choice(available_nodes)
        d_in = graph.nodes[node_id].output_shape.dim
        
        # 3. Filter candidates by budget
        candidates = self._get_budget_safe_ops(d_in, params_acc, flops_acc)
        if not candidates:
            return

        # 4. Weighted selection with EfficiencyPrior
        op_name = self._weighted_pick(rng, candidates)
        
        # 5. Apply and recurse
        try:
            new_id = graph.add_op(op_name, [node_id])
            # Update accumulators (fast path)
            op_p = c_estimate_op_params(op_name.encode('utf-8'), d_in, d_in)
            # Rough FLOP estimate: 2*S*D*D for linear-like
            op_f = 2 * 128 * d_in * d_in 
            
            self._recursive_build(
                graph, rng, 
                available_nodes + [new_id], 
                depth + 1,
                params_acc + op_p,
                flops_acc + op_f
            )
        except Exception:
            return

    def _choose_action_adaptive(self, rng, depth, p_acc, f_acc):
        # Nudge toward stopping as we approach budgets
        p_ratio = p_acc / self.max_params
        f_ratio = f_acc / self.max_flops
        d_ratio = depth / self.config.max_depth
        
        stop_prob = max(p_ratio, f_ratio, d_ratio**2)
        if rng.random() < stop_prob:
            return "stop"
        return "add_op"

    def _get_budget_safe_ops(self, int d_in, long p_acc, long f_acc):
        """Return ops that fit in remaining budget."""
        safe = []
        # Query primitives (this part stays Python-accessible)
        for name, op in PRIMITIVE_REGISTRY.items():
            if op.n_inputs != 1: continue
            
            # Fast Param Estimate
            op_p = c_estimate_op_params(name.encode('utf-8'), d_in, d_in)
            if p_acc + op_p > self.max_params:
                continue
                
            # Fast FLOP Estimate (conservative)
            op_f = 2 * 128 * d_in * d_in
            if f_acc + op_f > self.max_flops:
                continue
                
            safe.append(name)
        return safe

    def _weighted_pick(self, rng, candidates):
        weights = []
        for name in candidates:
            w = self.config.op_weights.get(name, 1.0)
            if self.prior:
                w *= self.prior.get_bias(name)
            weights.append(w)
        return rng.choices(candidates, weights=weights, k=1)[0]
