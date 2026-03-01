"""
Primitive Operations for Program Synthesis

~50 primitive tensor operations that serve as the "instruction set"
for generating novel computation graphs. These are BELOW the level
of any named technique — torch.matmul, torch.exp, torch.sin, etc.

Each primitive declares:
- Name and category
- Shape transformation rule (given input shapes, what's the output shape?)
- Whether it introduces learnable parameters
- Whether it preserves gradient flow
"""

from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union


class OpCategory(Enum):
    ELEMENTWISE_UNARY = "elementwise_unary"
    ELEMENTWISE_BINARY = "elementwise_binary"
    REDUCTION = "reduction"
    LINEAR_ALGEBRA = "linear_algebra"
    STRUCTURAL = "structural"
    PARAMETERIZED = "parameterized"
    MIXING = "mixing"
    SEQUENCE = "sequence"
    FREQUENCY = "frequency"
    MATH_SPACE = "math_space"
    FUNCTIONAL = "functional"


# Shape is represented as a tuple of symbolic dimensions
# We use strings for symbolic dims: "B" (batch), "S" (seq), "D" (model dim)
# and ints for concrete dims
ShapeSym = Tuple[str, ...]


@dataclass(frozen=True)
class PrimitiveOp:
    """A single primitive operation in our instruction set."""
    name: str
    category: OpCategory
    n_inputs: int  # number of tensor inputs (1 for unary, 2 for binary)
    # Shape rule: given input shape(s), returns output shape
    # None means "same as input" (most elementwise ops)
    shape_rule: str  # symbolic rule name, resolved at graph build time
    # Does this op introduce learnable parameters?
    has_params: bool = False
    # Approximate param count as function of D (model dim)
    # e.g., "D*D" for linear projection, "D" for scale/bias
    param_formula: str = "0"
    # Does this op always preserve gradient flow?
    preserves_gradient: bool = True
    # Can this op produce NaN/Inf with normal inputs?
    numerically_risky: bool = False
    # Description for debugging/display
    description: str = ""
    config_keys: Tuple[str, ...] = ()  # Required config keys
    # Can this op be placed standalone by the grammar?
    # False for routing signal helpers that produce non-standard outputs
    # (tuples, indices, reduced dims) consumed by specific routing ops.
    standalone: bool = True

    def __hash__(self):
        return hash(self.name)


# ── Protected Ops ────────────────────────────────────────────────────
# Ops that must never be hard-excluded by the auto-exclusion system.
# These ops have known root-cause fixes and should be given fair chances.
PROTECTED_OPS: frozenset = frozenset({
    "lif_neuron", "stdp_attention", "spike_rate_code", "sparse_threshold",
    "swiglu_mlp", "rwkv_channel", "reciprocal", "sliding_window_mask",
    "token_merge", "rmsnorm", "div_safe", "ultrametric_attention",
    "rotor_transform", "padic_residual", "padic_expand", "tropical_center",
    "rwkv_time_mixing", "mod_topk", "adaptive_recursion", "speculative",
    "entropy_router", "latent_attention_compressor", "token_type_classifier",
    "route_topk", "route_lanes", "route_recursion", "token_merging",
})


# ── Safe arithmetic evaluation ────────────────────────────────────────

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_eval_formula(formula: str) -> int:
    """Safely evaluate a simple arithmetic formula (no builtins/calls).

    Supports: integers, +, -, *, /, //, **, unary minus, parentheses.
    Raises ValueError on anything else (function calls, names, etc.).
    """
    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid formula: {formula!r}") from e

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.operand))
        else:
            raise ValueError(f"Unsupported expression in formula: {ast.dump(node)}")

    return int(_eval(tree))


# ── The Primitive Registry ────────────────────────────────────────────

PRIMITIVE_REGISTRY: Dict[str, PrimitiveOp] = {}
OPCODE_MAP: Dict[str, int] = {"input": 0}
REVERSE_OPCODE_MAP: Dict[int, str] = {0: "input"}


def _register(op: PrimitiveOp) -> PrimitiveOp:
    PRIMITIVE_REGISTRY[op.name] = op
    if op.name not in OPCODE_MAP:
        opcode = len(OPCODE_MAP)
        OPCODE_MAP[op.name] = opcode
        REVERSE_OPCODE_MAP[opcode] = op.name
    return op


# ── Elementwise Unary ─────────────────────────────────────────────────

_register(PrimitiveOp("neg", OpCategory.ELEMENTWISE_UNARY, 1, "identity",
                       description="Negate: -x"))
_register(PrimitiveOp("abs", OpCategory.ELEMENTWISE_UNARY, 1, "identity",
                       description="Absolute value",
                       preserves_gradient=False))  # gradient undefined at 0
_register(PrimitiveOp("exp", OpCategory.ELEMENTWISE_UNARY, 1, "identity",
                       description="Exponential",
                       numerically_risky=True))
_register(PrimitiveOp("log", OpCategory.ELEMENTWISE_UNARY, 1, "identity",
                       description="Natural logarithm (clamped input)",
                       numerically_risky=True))
_register(PrimitiveOp("sin", OpCategory.ELEMENTWISE_UNARY, 1, "identity",
                       description="Sine"))
_register(PrimitiveOp("cos", OpCategory.ELEMENTWISE_UNARY, 1, "identity",
                       description="Cosine"))
_register(PrimitiveOp("tanh", OpCategory.ELEMENTWISE_UNARY, 1, "identity",
                       description="Hyperbolic tangent"))
_register(PrimitiveOp("sigmoid", OpCategory.ELEMENTWISE_UNARY, 1, "identity",
                       description="Sigmoid"))
_register(PrimitiveOp("relu", OpCategory.ELEMENTWISE_UNARY, 1, "identity",
                       description="ReLU",
                       preserves_gradient=False))
_register(PrimitiveOp("gelu", OpCategory.ELEMENTWISE_UNARY, 1, "identity",
                       description="GELU activation"))
_register(PrimitiveOp("silu", OpCategory.ELEMENTWISE_UNARY, 1, "identity",
                       description="SiLU (Swish) activation"))
_register(PrimitiveOp("sqrt", OpCategory.ELEMENTWISE_UNARY, 1, "identity",
                       description="Square root (clamped input)",
                       numerically_risky=True))
_register(PrimitiveOp("square", OpCategory.ELEMENTWISE_UNARY, 1, "identity",
                       description="Square: x^2"))
_register(PrimitiveOp("sign_ste", OpCategory.ELEMENTWISE_UNARY, 1, "identity",
                       description="Sign with straight-through estimator",
                       preserves_gradient=False))
_register(PrimitiveOp("reciprocal", OpCategory.ELEMENTWISE_UNARY, 1, "identity",
                       description="1/x (clamped)",
                       numerically_risky=True))

# ── Elementwise Binary ────────────────────────────────────────────────

_register(PrimitiveOp("add", OpCategory.ELEMENTWISE_BINARY, 2, "binary_broadcast",
                       description="Element-wise addition"))
_register(PrimitiveOp("mul", OpCategory.ELEMENTWISE_BINARY, 2, "binary_broadcast",
                       description="Element-wise multiplication (gating)"))
_register(PrimitiveOp("sub", OpCategory.ELEMENTWISE_BINARY, 2, "binary_broadcast",
                       description="Element-wise subtraction"))
_register(PrimitiveOp("div_safe", OpCategory.ELEMENTWISE_BINARY, 2, "binary_broadcast",
                       description="Element-wise division (clamped denominator)",
                       numerically_risky=True))
_register(PrimitiveOp("maximum", OpCategory.ELEMENTWISE_BINARY, 2, "binary_broadcast",
                       description="Element-wise maximum"))
_register(PrimitiveOp("minimum", OpCategory.ELEMENTWISE_BINARY, 2, "binary_broadcast",
                       description="Element-wise minimum"))

# ── Reductions ────────────────────────────────────────────────────────

_register(PrimitiveOp("sum_last", OpCategory.REDUCTION, 1, "reduce_last",
                       description="Sum along last (feature) dimension"))
_register(PrimitiveOp("mean_last", OpCategory.REDUCTION, 1, "reduce_last",
                       description="Mean along last dimension"))
_register(PrimitiveOp("max_last", OpCategory.REDUCTION, 1, "reduce_last",
                       description="Max along last dimension"))
_register(PrimitiveOp("norm_last", OpCategory.REDUCTION, 1, "reduce_last",
                       description="L2 norm along last dimension"))
_register(PrimitiveOp("cumsum", OpCategory.REDUCTION, 1, "cumulative",
                       description="Cumulative sum along sequence dim"))
_register(PrimitiveOp("cumprod_safe", OpCategory.REDUCTION, 1, "cumulative",
                       description="Cumulative product (clamped) along seq dim",
                       numerically_risky=True))

# ── Linear Algebra ────────────────────────────────────────────────────

_register(PrimitiveOp("matmul", OpCategory.LINEAR_ALGEBRA, 2, "matmul",
                       description="Batched matrix multiply"))
_register(PrimitiveOp("outer_product", OpCategory.LINEAR_ALGEBRA, 2, "outer",
                       description="Elementwise (Hadamard) product of two inputs"))
_register(PrimitiveOp("transpose_sd", OpCategory.LINEAR_ALGEBRA, 1, "transpose_seq_dim",
                       description="Transpose sequence and feature dims"))

# ── Identity (pass-through, used by workflow_converter for uniform routing) ──

_register(PrimitiveOp("identity", OpCategory.STRUCTURAL, 1, "identity",
                       description="Pass-through (no-op)",
                       standalone=False))

# ── Structural ────────────────────────────────────────────────────────

_register(PrimitiveOp("split2", OpCategory.STRUCTURAL, 1, "split",
                       description="Split last dim into 2 equal parts",
                       config_keys=("n_splits",)))
_register(PrimitiveOp("split3", OpCategory.STRUCTURAL, 1, "split",
                       description="Split last dim into 3 equal parts",
                       config_keys=("n_splits",)))
_register(PrimitiveOp("concat", OpCategory.STRUCTURAL, 2, "concat",
                       description="Concatenate along last dimension"))
_register(PrimitiveOp("multi_head_mix", OpCategory.STRUCTURAL, 1, "identity",
                       description="Multi-head reshape + per-head L2 normalize",
                       config_keys=("n_heads",)))

# ── Parameterized (learnable) ─────────────────────────────────────────

_register(PrimitiveOp("linear_proj", OpCategory.PARAMETERIZED, 1, "linear",
                       has_params=True, param_formula="D*D",
                       description="Learned linear projection (D -> D)",
                       config_keys=("out_dim",)))
_register(PrimitiveOp("linear_proj_down", OpCategory.PARAMETERIZED, 1, "linear",
                       has_params=True, param_formula="D*D//2",
                       description="Learned linear projection (D -> D//2)"))
_register(PrimitiveOp("linear_proj_up", OpCategory.PARAMETERIZED, 1, "linear",
                       has_params=True, param_formula="D//2*D",
                       description="Learned linear projection (D//2 -> D)"))
_register(PrimitiveOp("fused_linear_gelu", OpCategory.PARAMETERIZED, 1, "linear",
                       has_params=True, param_formula="D*D",
                       description="Fused Linear + Bias + GELU (Triton-accelerated)",
                       config_keys=("out_dim",)))
_register(PrimitiveOp("learnable_scale", OpCategory.PARAMETERIZED, 1, "scale",
                       has_params=True, param_formula="D",
                       description="Learnable per-dimension scale"))
_register(PrimitiveOp("learnable_bias", OpCategory.PARAMETERIZED, 1, "bias",
                       has_params=True, param_formula="D",
                       description="Learnable per-dimension bias"))
_register(PrimitiveOp("selective_scan", OpCategory.PARAMETERIZED, 1, "identity",
                       has_params=True, param_formula="D*4",
                       description="SSM-style input-dependent state scan",
                       numerically_risky=True))
_register(PrimitiveOp("conv1d_seq", OpCategory.PARAMETERIZED, 1, "identity",
                       has_params=True, param_formula="D*3",
                       description="Depthwise 1D convolution (kernel=3) along seq dim"))
_register(PrimitiveOp("topk_gate", OpCategory.PARAMETERIZED, 1, "identity",
                       has_params=True, param_formula="D*2",
                       description="Sparse gating: project to 2 gate scores, weight feature halves"))
_register(PrimitiveOp("nm_sparse_linear", OpCategory.PARAMETERIZED, 1, "linear",
                       has_params=True, param_formula="D*D//2",
                       description="N:M structured sparse linear projection (2:4 default)",
                       config_keys=("n", "m", "out_dim")))
_register(PrimitiveOp("block_sparse_linear", OpCategory.PARAMETERIZED, 1, "linear",
                       has_params=True, param_formula="D*D//4",
                       description="Block-sparse linear projection with configurable block density",
                       config_keys=("block_size", "block_density", "out_dim")))
_register(PrimitiveOp("rmsnorm", OpCategory.PARAMETERIZED, 1, "identity",
                       has_params=True, param_formula="D",
                       description="Root Mean Square Layer Normalization (Triton-accelerated)"))
_register(PrimitiveOp("semi_structured_2_4_linear", OpCategory.PARAMETERIZED, 1, "linear",
                       has_params=True, param_formula="D*D//2",
                       description="Semi-structured 2:4 sparse linear projection with compatibility gating",
                       config_keys=("out_dim",)))

# ── Sequence Operations ───────────────────────────────────────────────

_register(PrimitiveOp("softmax_last", OpCategory.SEQUENCE, 1, "softmax",
                       description="Softmax along last dimension"))
_register(PrimitiveOp("causal_mask", OpCategory.SEQUENCE, 1, "causal_mask",
                       description="Apply causal (lower-triangular) mask"))
_register(PrimitiveOp("local_window_attn", OpCategory.SEQUENCE, 1, "identity",
                       description="Local windowed causal self-attention (Q=K=V)",
                       config_keys=("window_size",)))
_register(PrimitiveOp("sliding_window_mask", OpCategory.SEQUENCE, 1, "causal_mask",
                       description="Exponential distance decay mask for windowed composition",
                       config_keys=("window_size",)))
# ── Mixing Operations ─────────────────────────────────────────────────

_register(PrimitiveOp("softmax_attention", OpCategory.MIXING, 1, "identity",
                       has_params=True, param_formula="D*D*3",
                       description="Standard Softmax Self-Attention"))
_register(PrimitiveOp("linear_attention", OpCategory.MIXING, 1, "identity",
                       has_params=True, param_formula="D*D*3",
                       description="Linear-complexity Attention (kernel-based)"))
_register(PrimitiveOp("graph_attention", OpCategory.MIXING, 1, "identity",
                       has_params=True, param_formula="D*D",
                       description="Graph-based sequence mixing with learned adjacency"))
# Removed fourier_mixing as it inherently breaks causality in an autoregressive context.
# _register(PrimitiveOp("fourier_mixing", OpCategory.MIXING, 1, "identity",
#                        description="Unparameterized global mixing via FFT"))
_register(PrimitiveOp("state_space", OpCategory.MIXING, 1, "identity",
                       has_params=True, param_formula="D*4",
                       description="State-space sequence mixer (Mamba-style)"))
_register(PrimitiveOp("conv_only", OpCategory.MIXING, 1, "identity",
                       has_params=True, param_formula="D*3",
                       description="Depthwise convolutional sequence mixer"))

# ── Channel Mixing ────────────────────────────────────────────────────

_register(PrimitiveOp("swiglu_mlp", OpCategory.PARAMETERIZED, 1, "identity",
                       has_params=True, param_formula="D*D*3.5",
                       description="SwiGLU MLP channel mixer"))
_register(PrimitiveOp("rwkv_channel", OpCategory.PARAMETERIZED, 1, "identity",
                       has_params=True, param_formula="D*D*2",
                       description="RWKV-style time-mixing channel update"))
_register(PrimitiveOp("moe_topk", OpCategory.PARAMETERIZED, 1, "identity",
                       has_params=True, param_formula="D*D*8",
                       description="Sparse Mixture-of-Experts channel mixer",
                       config_keys=("num_experts", "top_k")))
_register(PrimitiveOp("moe_2expert", OpCategory.PARAMETERIZED, 1, "identity",
                       has_params=True, param_formula="D*D*2",
                       description="Lightweight 2-expert MoE with learned gating",
                       config_keys=()))

# ── Functional (operator-learning / neural-field) ────────────────────

# ── Reference Architecture Ops ───────────────────────────────────────

_register(PrimitiveOp("layernorm", OpCategory.PARAMETERIZED, 1, "identity",
                       has_params=True, param_formula="D*2",
                       description="Layer Normalization with learned affine"))
_register(PrimitiveOp("embedding_lookup", OpCategory.PARAMETERIZED, 1, "identity",
                       has_params=True, param_formula="32000*D",
                       description="Token embedding lookup table",
                       config_keys=("vocab_size",)))
_register(PrimitiveOp("rope_rotate", OpCategory.FUNCTIONAL, 1, "identity",
                       description="Rotary Position Embedding (RoPE)"))
_register(PrimitiveOp("gated_linear", OpCategory.PARAMETERIZED, 1, "linear",
                       has_params=True, param_formula="D*D*2",
                       description="Fused gated linear: (x@W) * sigmoid(x@W_gate)",
                       config_keys=("out_dim",)))
_register(PrimitiveOp("cosine_similarity", OpCategory.LINEAR_ALGEBRA, 2, "reduce_last",
                       description="Cosine similarity between two tensors along last dim"))
_register(PrimitiveOp("gather_topk", OpCategory.STRUCTURAL, 2, "identity",
                       description="Gather top-k vectors by score",
                       config_keys=("k",)))
_register(PrimitiveOp("rwkv_time_mixing", OpCategory.PARAMETERIZED, 1, "identity",
                       has_params=True, param_formula="D*D*3",
                       description="RWKV WKV linear attention with learned decay"))

# ── Routing Primitives (Phase 1/2) ───────────────────────────────────

_register(PrimitiveOp("route_topk", OpCategory.FUNCTIONAL, 1, "identity",
                       description="Top-k token selection: (B,S) -> (B,K) indices + weights",
                       config_keys=("k",), standalone=False))
_register(PrimitiveOp("route_lanes", OpCategory.FUNCTIONAL, 1, "identity",
                       description="Multi-lane dispatch: (B,S,L) -> (B,S) lane indices",
                       config_keys=("n_lanes",), standalone=False))
_register(PrimitiveOp("route_recursion", OpCategory.FUNCTIONAL, 1, "identity",
                       description="Adaptive recursion depth: (B,S,Dp) -> (B,S) depth",
                       config_keys=("max_depth",), standalone=False))
_register(PrimitiveOp("token_merge", OpCategory.FUNCTIONAL, 1, "identity",
                       description="Similarity-based token merging: (B,S,D) -> (B,K,D)",
                       config_keys=("n_keep",)))

# ── Routing (control-style ops operating on tensors) ──────────────────

_register(PrimitiveOp("mod_topk", OpCategory.FUNCTIONAL, 1, "identity",
                       description="Mixture-of-Depths top-k token routing (masking)",
                       config_keys=("capacity_factor",)))
_register(PrimitiveOp("early_exit", OpCategory.FUNCTIONAL, 1, "identity",
                       description="Early-exit routing (token gating)",
                       config_keys=("threshold",)))
_register(PrimitiveOp("adaptive_recursion", OpCategory.FUNCTIONAL, 1, "identity",
                       description="Adaptive recursion routing (depth gating)",
                       config_keys=("max_depth",)))
_register(PrimitiveOp("token_merging", OpCategory.FUNCTIONAL, 1, "identity",
                       description="Token merging routing (merge + restore)",
                       config_keys=("n_keep",)))
_register(PrimitiveOp("cascade", OpCategory.FUNCTIONAL, 1, "identity",
                       description="Cascade routing (difficulty-scaled gate)",
                       config_keys=("threshold",)))
_register(PrimitiveOp("speculative", OpCategory.FUNCTIONAL, 1, "identity",
                       description="Speculative routing (quality gate)",
                       config_keys=("threshold",)))

# ── Exotic Routing & Compression (Phase 4) ──────────────────────────

_register(PrimitiveOp("adaptive_lane_mixer", OpCategory.PARAMETERIZED, 2, "identity",
                       has_params=True,
                       description="Difficulty-adaptive lane routing: routes tokens to experts based on learned difficulty"))
_register(PrimitiveOp("mixed_recursion_gate", OpCategory.PARAMETERIZED, 2, "identity",
                       has_params=True,
                       description="Tokens re-enter block with different transforms per recursion step, depth conditional"))
_register(PrimitiveOp("routing_conditioned_compression", OpCategory.PARAMETERIZED, 2, "identity",
                       has_params=True,
                       description="Compression level chosen per-token by routing scores"))
_register(PrimitiveOp("token_type_classifier", OpCategory.PARAMETERIZED, 1, "identity",
                       has_params=True, param_formula="D*D",
                       description="Learned classifier to produce routing scores from token embeddings, projected back to model dim",
                       config_keys=("n_classes",), standalone=False))
_register(PrimitiveOp("entropy_router", OpCategory.FUNCTIONAL, 1, "reduce_last",
                       description="Produces routing signal based on entropy of input scores (B,S,K) -> (B,S,1)",
                       standalone=False))
_register(PrimitiveOp("progressive_compression_gate", OpCategory.PARAMETERIZED, 1, "identity",
                       has_params=True,
                       description="Learned per-layer compression schedule: heavier early, lighter late"))
_register(PrimitiveOp("compression_mixture_experts", OpCategory.PARAMETERIZED, 2, "identity",
                       has_params=True,
                       description="Routing assigns tokens to method-specific compression experts (e.g. low-rank, sparse, bottleneck)"))

# ── 2026 Frontier Exotic Ops ────────────────────────────────────────

_register(PrimitiveOp("relu_gate_routing", OpCategory.PARAMETERIZED, 1, "identity",
                       has_params=True,
                       description="Differentiable ReLU-based gating: learns optimal expert count per token"))
_register(PrimitiveOp("ternary_projection", OpCategory.PARAMETERIZED, 1, "linear",
                       has_params=True,
                       description="1.58-bit ternary simulated projection (-1, 0, 1 weights)"))
_register(PrimitiveOp("latent_attention_compressor", OpCategory.PARAMETERIZED, 1, "identity",
                       has_params=True,
                       description="Multi-Head Latent Attention (MLA) style KV cache compression"))

# ── Functional (operator-learning / neural-field) ────────────────────

_register(PrimitiveOp("basis_expansion", OpCategory.FUNCTIONAL, 1, "identity",
                       has_params=True, param_formula="D*4",
                       description="Basis-expansion layer: project through sinusoidal bases"))
_register(PrimitiveOp("integral_kernel", OpCategory.FUNCTIONAL, 1, "identity",
                       has_params=True, param_formula="D*D",
                       description="Integral kernel mixing: learned kernel over sequence positions",
                       config_keys=("kernel_scale",)))
_register(PrimitiveOp("fixed_point_iter", OpCategory.FUNCTIONAL, 1, "identity",
                       has_params=True, param_formula="D*D+D",
                       description="Implicit fixed-point iteration: x = f(x) with learned f",
                       numerically_risky=True,
                       config_keys=("n_iters", "damping")))


# ── Manifest Loading (Single Source of Truth) ─────────────────────────

def load_primitives_from_designer(components_root: Path) -> int:
    """Scan Designer components/ and register primitives from manifests."""
    import yaml
    count = 0
    # Map Designer categories to Research categories
    CAT_MAP = {
        "math": OpCategory.ELEMENTWISE_UNARY,
        "math_space": OpCategory.MATH_SPACE,
        "mixing": OpCategory.MIXING,
        "routing": OpCategory.FUNCTIONAL,
        "structural": OpCategory.STRUCTURAL,
        "reduction": OpCategory.REDUCTION,
        "linear_algebra": OpCategory.LINEAR_ALGEBRA,
        "sequence": OpCategory.SEQUENCE,
        "frequency": OpCategory.FREQUENCY,
        "parameterized": OpCategory.PARAMETERIZED,
        "functional": OpCategory.FUNCTIONAL,
        "blocks": OpCategory.STRUCTURAL,
        "channel_mixing": OpCategory.MIXING,
    }

    for manifest_path in components_root.glob("*/*/manifest.yaml"):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = yaml.safe_load(f)

            op_id = manifest.get("id")
            if not op_id or op_id in PRIMITIVE_REGISTRY:
                continue

            perf = manifest.get("performance", {})
            desc = manifest.get("description", "")
            cat_name = manifest_path.parent.parent.name

            # Skip non-graph categories: these are morphological box concepts
            # (normalization variants, positional encodings, representations,
            # data I/O, control flow) that have no _OP_DISPATCH handler and
            # would raise ValueError("Unknown op") at execution time.
            if cat_name not in CAT_MAP:
                continue

            # Skip manifest ops that have no compiler dispatch handler.
            # Many aria-designer components (blocks, mixing variants, etc.)
            # have manifests but no _execute_op implementation — registering
            # them lets the grammar sample them, only to crash at runtime.
            from .compiler import _OP_DISPATCH
            if op_id not in _OP_DISPATCH:
                continue

            # Create PrimitiveOp from manifest
            op = PrimitiveOp(
                name=op_id,
                category=CAT_MAP.get(cat_name, OpCategory.ELEMENTWISE_UNARY),
                n_inputs=len(manifest.get("inputs", [])),
                shape_rule=manifest.get("shape_rule", "identity"), # Default to identity if missing
                has_params=perf.get("has_params", False),
                param_formula=perf.get("param_formula", "0"),
                preserves_gradient=perf.get("preserves_gradient", True),
                numerically_risky=perf.get("numerically_risky", False),
                description=desc,
                config_keys=tuple(manifest.get("params", {}).keys()),
                standalone=manifest.get("standalone", True)
            )
            _register(op)
            count += 1
        except Exception:
            continue
    return count


# Load Designer primitives if available
try:
    from pathlib import Path
    _DESIGNER_COMPONENTS = Path(__file__).resolve().parents[2] / "aria_designer" / "components"
    if _DESIGNER_COMPONENTS.exists():
        load_primitives_from_designer(_DESIGNER_COMPONENTS)
except Exception:
    pass


# ── Helper Functions ──────────────────────────────────────────────────

def get_primitive(name: str) -> PrimitiveOp:
    """Get a primitive by name."""
    if name not in PRIMITIVE_REGISTRY:
        raise KeyError(f"Unknown primitive: {name}. Available: {list(PRIMITIVE_REGISTRY.keys())}")
    return PRIMITIVE_REGISTRY[name]


def list_primitives(category: Optional[OpCategory] = None) -> List[PrimitiveOp]:
    """List all primitives, optionally filtered by category."""
    ops = list(PRIMITIVE_REGISTRY.values())
    if category is not None:
        ops = [op for op in ops if op.category == category]
    return ops


def register_external_primitive(op: PrimitiveOp) -> None:
    """Register a primitive from external sources (e.g., math spaces)."""
    if op.name not in PRIMITIVE_REGISTRY:
        PRIMITIVE_REGISTRY[op.name] = op
        if op.name not in OPCODE_MAP:
            opcode = len(OPCODE_MAP)
            OPCODE_MAP[op.name] = opcode
            REVERSE_OPCODE_MAP[opcode] = op.name
