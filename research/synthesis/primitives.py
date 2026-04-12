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
import operator
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Dict, List, Optional, Tuple

from .native_param_formula import evaluate_param_formula_natively

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class AlgebraicType:
    """Input/output algebraic contract for a primitive."""

    space: str
    input_constraint: str
    output_guarantee: str


_EUCLIDEAN_TYPE = AlgebraicType("euclidean", "real", "real")


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
    # Safe for byte-level / sub-word tokenization?
    # False for ops that reorder, drop, or merge tokens in ways that
    # destroy byte-stream integrity (token_merge, mod_topk, etc.)
    byte_safe: bool = True
    # Minimum layer depth before this op may be placed (0 = any layer).
    # Prevents destructive ops from acting on raw / early embeddings.
    min_layer_depth: int = 0
    # Backward-compatible coarse algebraic space tag.
    algebraic_space: str = "euclidean"
    algebraic_type: AlgebraicType = field(default_factory=lambda: _EUCLIDEAN_TYPE)
    # Token binding range: how far back this op can see in the sequence.
    # "full" = full sequence (attention, SSM), "medium" = windowed (local_window),
    # "local" = k<=3 neighbors (conv, token_merge), "none" = non-mixer op.
    binding_range_class: str = "none"

    def __hash__(self):
        return hash(self.name)


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


def estimate_op_params(
    op: PrimitiveOp,
    d_in: int,
    d_out: Optional[int] = None,
) -> int:
    """Estimate learnable parameter count for a primitive op.

    Uses primitive formula evaluation with conservative fallback when formulas
    are malformed or unsafe.
    """
    if not op.has_params or not op.param_formula or op.param_formula == "0":
        return 0
    d_out = d_in if d_out is None else d_out
    formula = op.param_formula.replace("D_OUT", str(d_out)).replace("D", str(d_in))
    native_value = evaluate_param_formula_natively(formula)
    if native_value is not None:
        return native_value
    try:
        return safe_eval_formula(formula)
    except Exception:
        return d_in * d_out


# ── The Primitive Registry ────────────────────────────────────────────

PRIMITIVE_REGISTRY: Dict[str, PrimitiveOp] = {}
OPCODE_MAP: Dict[str, int] = {"input": 0}
REVERSE_OPCODE_MAP: Dict[int, str] = {0: "input"}


# Backward-compatible aliases for renamed ops (old name → new name).
# get_primitive() resolves these transparently so serialized graphs and
# database entries using old names continue to work.
OP_NAME_ALIASES: Dict[str, str] = {
    # Phase 1 renames (other session)
    "route_topk": "feature_sparsity",
    "route_lanes": "gated_lane_blend",
    "route_recursion": "depth_gated_transform",
    "routing_conditioned_compression": "signal_conditioned_compression",
    "relu_gate_routing": "relu_gated_moe",
    # Phase 2 renames (remaining "routing" ops that don't actually route)
    "adaptive_lane_mixer": "difficulty_blend_3way",
    "adaptive_recursion": "depth_weighted_proj",
    "cascade": "learned_token_gate",
    "compression_mixture_experts": "dual_compression_blend",
    "difficulty_scorer": "token_difficulty_proj",
    "early_exit": "confidence_token_gate",
    "entropy_score": "token_entropy",
    "mixed_recursion_gate": "score_depth_blend",
    "mod_topk": "depth_token_mask",
    "progressive_compression_gate": "adaptive_rank_gate",
    "speculative": "cheap_verify_blend",
    "token_merge": "adjacent_token_merge",
    "token_type_classifier": "token_class_proj",
    "n_way_sparse_router": "sparse_bottleneck_moe",
}


def _register(op: PrimitiveOp) -> PrimitiveOp:
    PRIMITIVE_REGISTRY[op.name] = op
    if op.name not in OPCODE_MAP:
        opcode = len(OPCODE_MAP)
        OPCODE_MAP[op.name] = opcode
        REVERSE_OPCODE_MAP[opcode] = op.name
    # Also register under any old alias names pointing to this op,
    # so serialized graphs using old names still resolve.
    for old_name, new_name in OP_NAME_ALIASES.items():
        if new_name == op.name:
            PRIMITIVE_REGISTRY[old_name] = op
            if old_name not in OPCODE_MAP:
                OPCODE_MAP[old_name] = OPCODE_MAP[op.name]
    return op


# ── Elementwise Unary ─────────────────────────────────────────────────

_register(
    PrimitiveOp(
        "neg", OpCategory.ELEMENTWISE_UNARY, 1, "identity", description="Negate: -x"
    )
)
_register(
    PrimitiveOp(
        "abs",
        OpCategory.ELEMENTWISE_UNARY,
        1,
        "identity",
        description="Absolute value",
        preserves_gradient=False,
    )
)  # gradient undefined at 0
_register(
    PrimitiveOp(
        "exp",
        OpCategory.ELEMENTWISE_UNARY,
        1,
        "identity",
        description="Exponential",
        numerically_risky=True,
    )
)
_register(
    PrimitiveOp(
        "log",
        OpCategory.ELEMENTWISE_UNARY,
        1,
        "identity",
        description="Natural logarithm (clamped input)",
        numerically_risky=True,
    )
)
_register(
    PrimitiveOp("sin", OpCategory.ELEMENTWISE_UNARY, 1, "identity", description="Sine")
)
_register(
    PrimitiveOp(
        "cos", OpCategory.ELEMENTWISE_UNARY, 1, "identity", description="Cosine"
    )
)
_register(
    PrimitiveOp(
        "tanh",
        OpCategory.ELEMENTWISE_UNARY,
        1,
        "identity",
        description="Hyperbolic tangent",
    )
)
_register(
    PrimitiveOp(
        "sigmoid", OpCategory.ELEMENTWISE_UNARY, 1, "identity", description="Sigmoid"
    )
)
_register(
    PrimitiveOp(
        "relu",
        OpCategory.ELEMENTWISE_UNARY,
        1,
        "identity",
        description="ReLU",
        preserves_gradient=False,
    )
)
_register(
    PrimitiveOp(
        "gelu",
        OpCategory.ELEMENTWISE_UNARY,
        1,
        "identity",
        description="GELU activation",
    )
)
_register(
    PrimitiveOp(
        "silu",
        OpCategory.ELEMENTWISE_UNARY,
        1,
        "identity",
        description="SiLU (Swish) activation",
    )
)
_register(
    PrimitiveOp(
        "sqrt",
        OpCategory.ELEMENTWISE_UNARY,
        1,
        "identity",
        description="Square root (clamped input)",
        numerically_risky=True,
    )
)
_register(
    PrimitiveOp(
        "square", OpCategory.ELEMENTWISE_UNARY, 1, "identity", description="Square: x^2"
    )
)
_register(
    PrimitiveOp(
        "sign_ste",
        OpCategory.ELEMENTWISE_UNARY,
        1,
        "identity",
        description="Sign with straight-through estimator",
        preserves_gradient=False,
    )
)
_register(
    PrimitiveOp(
        "reciprocal",
        OpCategory.ELEMENTWISE_UNARY,
        1,
        "identity",
        description="1/x (clamped)",
        numerically_risky=True,
    )
)

# ── Elementwise Binary ────────────────────────────────────────────────

_register(
    PrimitiveOp(
        "add",
        OpCategory.ELEMENTWISE_BINARY,
        2,
        "binary_broadcast",
        description="Element-wise addition",
    )
)
_register(
    PrimitiveOp(
        "mul",
        OpCategory.ELEMENTWISE_BINARY,
        2,
        "binary_broadcast",
        description="Element-wise multiplication (gating)",
    )
)
_register(
    PrimitiveOp(
        "sub",
        OpCategory.ELEMENTWISE_BINARY,
        2,
        "binary_broadcast",
        description="Element-wise subtraction",
    )
)
_register(
    PrimitiveOp(
        "div_safe",
        OpCategory.ELEMENTWISE_BINARY,
        2,
        "binary_broadcast",
        description="Element-wise division (clamped denominator)",
        numerically_risky=True,
    )
)
_register(
    PrimitiveOp(
        "maximum",
        OpCategory.ELEMENTWISE_BINARY,
        2,
        "binary_broadcast",
        description="Element-wise maximum",
    )
)
_register(
    PrimitiveOp(
        "minimum",
        OpCategory.ELEMENTWISE_BINARY,
        2,
        "binary_broadcast",
        description="Element-wise minimum",
    )
)

# ── Reductions ────────────────────────────────────────────────────────

_register(
    PrimitiveOp(
        "sum_last",
        OpCategory.REDUCTION,
        1,
        "reduce_last",
        description="Sum along last (feature) dimension",
    )
)
_register(
    PrimitiveOp(
        "mean_last",
        OpCategory.REDUCTION,
        1,
        "reduce_last",
        description="Mean along last dimension",
    )
)
_register(
    PrimitiveOp(
        "max_last",
        OpCategory.REDUCTION,
        1,
        "reduce_last",
        description="Max along last dimension",
    )
)
_register(
    PrimitiveOp(
        "norm_last",
        OpCategory.REDUCTION,
        1,
        "reduce_last",
        description="L2 norm along last dimension",
    )
)
_register(
    PrimitiveOp(
        "cumsum",
        OpCategory.REDUCTION,
        1,
        "cumulative",
        description="Cumulative sum along sequence dim",
    )
)
_register(
    PrimitiveOp(
        "cumprod_safe",
        OpCategory.REDUCTION,
        1,
        "cumulative",
        description="Cumulative product (clamped) along seq dim",
        numerically_risky=True,
    )
)

# ── Linear Algebra ────────────────────────────────────────────────────

_register(
    PrimitiveOp(
        "matmul",
        OpCategory.LINEAR_ALGEBRA,
        2,
        "matmul",
        description="Batched matrix multiply",
    )
)
_register(
    PrimitiveOp(
        "outer_product",
        OpCategory.LINEAR_ALGEBRA,
        2,
        "outer",
        description="Elementwise (Hadamard) product of two inputs",
    )
)
_register(
    PrimitiveOp(
        "transpose_sd",
        OpCategory.LINEAR_ALGEBRA,
        1,
        "transpose_seq_dim",
        description="Transpose sequence and feature dims",
    )
)

# ── Identity (pass-through, used by workflow_converter for uniform routing) ──

_register(
    PrimitiveOp(
        "identity",
        OpCategory.STRUCTURAL,
        1,
        "identity",
        description="Pass-through (no-op)",
        standalone=False,
    )
)

# ── Structural ────────────────────────────────────────────────────────

_register(
    PrimitiveOp(
        "split2",
        OpCategory.STRUCTURAL,
        1,
        "split",
        description="Split last dim into 2 equal parts",
        config_keys=("n_splits",),
    )
)
_register(
    PrimitiveOp(
        "split3",
        OpCategory.STRUCTURAL,
        1,
        "split",
        description="Split last dim into 3 equal parts",
        config_keys=("n_splits",),
    )
)
_register(
    PrimitiveOp(
        "concat",
        OpCategory.STRUCTURAL,
        2,
        "concat",
        description="Concatenate along last dimension",
    )
)
_register(
    PrimitiveOp(
        "multi_head_mix",
        OpCategory.STRUCTURAL,
        1,
        "identity",
        description="Multi-head reshape + per-head L2 normalize",
        config_keys=("n_heads",),
    )
)

# ── Parameterized (learnable) ─────────────────────────────────────────

_register(
    PrimitiveOp(
        "linear_proj",
        OpCategory.PARAMETERIZED,
        1,
        "linear",
        has_params=True,
        param_formula="D*D",
        description="Learned linear projection (D -> D)",
        config_keys=("out_dim",),
    )
)
_register(
    PrimitiveOp(
        "linear_proj_down",
        OpCategory.PARAMETERIZED,
        1,
        "linear",
        has_params=True,
        param_formula="D*D//2",
        description="Learned linear projection (D -> D//2)",
    )
)
_register(
    PrimitiveOp(
        "linear_proj_up",
        OpCategory.PARAMETERIZED,
        1,
        "linear",
        has_params=True,
        param_formula="D//2*D",
        description="Learned linear projection (D//2 -> D)",
    )
)
_register(
    PrimitiveOp(
        "fused_linear_gelu",
        OpCategory.PARAMETERIZED,
        1,
        "linear",
        has_params=True,
        param_formula="D*D",
        description="Fused Linear + Bias + GELU (Triton-accelerated)",
        config_keys=("out_dim",),
    )
)
_register(
    PrimitiveOp(
        "learnable_scale",
        OpCategory.PARAMETERIZED,
        1,
        "scale",
        has_params=True,
        param_formula="D",
        description="Learnable per-dimension scale",
    )
)
_register(
    PrimitiveOp(
        "learnable_bias",
        OpCategory.PARAMETERIZED,
        1,
        "bias",
        has_params=True,
        param_formula="D",
        description="Learnable per-dimension bias",
    )
)
_register(
    PrimitiveOp(
        "calibrated_branch_merge",
        OpCategory.PARAMETERIZED,
        2,
        "identity",
        has_params=True,
        param_formula="D*2+D*2+4",
        standalone=False,
        description="Calibrated two-branch merge with bounded secondary-share protection",
    )
)
_register(
    PrimitiveOp(
        "selective_scan",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*4",
        description="SSM-style input-dependent state scan",
        numerically_risky=True,
        binding_range_class="full",
    )
)
_register(
    PrimitiveOp(
        "conv1d_seq",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*3",
        description="Depthwise 1D convolution (kernel=3) along seq dim",
        binding_range_class="local",
    )
)
_register(
    PrimitiveOp(
        "topk_gate",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*2",
        description="Sparse gating: project to 2 gate scores, weight feature halves",
    )
)
_register(
    PrimitiveOp(
        "nm_sparse_linear",
        OpCategory.PARAMETERIZED,
        1,
        "linear",
        has_params=True,
        param_formula="D*D//2",
        description="N:M structured sparse linear projection (2:4 default)",
        config_keys=("n", "m", "out_dim"),
    )
)
_register(
    PrimitiveOp(
        "block_sparse_linear",
        OpCategory.PARAMETERIZED,
        1,
        "linear",
        has_params=True,
        param_formula="D*D//4",
        description="Block-sparse linear projection with configurable block density",
        config_keys=("block_size", "block_density", "out_dim"),
    )
)
_register(
    PrimitiveOp(
        "rmsnorm",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D",
        description="Root Mean Square Layer Normalization (Triton-accelerated)",
    )
)
_register(
    PrimitiveOp(
        "semi_structured_2_4_linear",
        OpCategory.PARAMETERIZED,
        1,
        "linear",
        has_params=True,
        param_formula="D*D//2",
        description="Semi-structured 2:4 sparse linear projection with compatibility gating",
        config_keys=("out_dim",),
    )
)

# ── Sequence Operations ───────────────────────────────────────────────

_register(
    PrimitiveOp(
        "softmax_last",
        OpCategory.SEQUENCE,
        1,
        "softmax",
        description="Softmax along last dimension",
    )
)
_register(
    PrimitiveOp(
        "causal_mask",
        OpCategory.SEQUENCE,
        1,
        "causal_mask",
        description="Apply causal (lower-triangular) mask",
    )
)
_register(
    PrimitiveOp(
        "local_window_attn",
        OpCategory.SEQUENCE,
        1,
        "identity",
        description="Local windowed causal self-attention (Q=K=V)",
        config_keys=("window_size",),
        binding_range_class="medium",
    )
)
_register(
    PrimitiveOp(
        "sliding_window_mask",
        OpCategory.SEQUENCE,
        1,
        "causal_mask",
        description="Exponential distance decay mask for windowed composition",
        config_keys=("window_size",),
        binding_range_class="medium",
    )
)
# ── Mixing Operations ─────────────────────────────────────────────────

_register(
    PrimitiveOp(
        "softmax_attention",
        OpCategory.MIXING,
        1,
        "identity",
        has_params=True,
        param_formula="D*D*3",
        description="Standard Softmax Self-Attention",
        binding_range_class="full",
    )
)
_register(
    PrimitiveOp(
        "linear_attention",
        OpCategory.MIXING,
        1,
        "identity",
        has_params=True,
        param_formula="D*D*3",
        description="Linear-complexity Attention (kernel-based)",
        binding_range_class="full",
    )
)
_register(
    PrimitiveOp(
        "graph_attention",
        OpCategory.MIXING,
        1,
        "identity",
        has_params=True,
        param_formula="D*D",
        description="Graph-based sequence mixing with learned adjacency",
        binding_range_class="full",
    )
)
_register(
    PrimitiveOp(
        "diff_attention",
        OpCategory.MIXING,
        1,
        "identity",
        has_params=True,
        param_formula="D*D*4",
        description="Differential attention: two softmax maps subtracted to cancel noise (ICLR 2025)",
        binding_range_class="full",
    )
)
# Removed fourier_mixing as it inherently breaks causality in an autoregressive context.
# _register(PrimitiveOp("fourier_mixing", OpCategory.MIXING, 1, "identity",
#                        description="Unparameterized global mixing via FFT"))
_register(
    PrimitiveOp(
        "state_space",
        OpCategory.MIXING,
        1,
        "identity",
        has_params=True,
        param_formula="D*4",
        description="State-space sequence mixer (Mamba-style)",
        binding_range_class="full",
    )
)
_register(
    PrimitiveOp(
        "conv_only",
        OpCategory.MIXING,
        1,
        "identity",
        has_params=True,
        param_formula="D*3",
        description="Depthwise convolutional sequence mixer",
        binding_range_class="local",
    )
)
_register(
    PrimitiveOp(
        "gated_delta",
        OpCategory.MIXING,
        1,
        "identity",
        has_params=True,
        param_formula="D*D*4",
        description="Gated delta rule: linear recurrence with decay + update gates for targeted state writes (ICLR 2025)",
        binding_range_class="full",
    )
)

# ── Channel Mixing ────────────────────────────────────────────────────

_register(
    PrimitiveOp(
        "swiglu_mlp",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D*3.5",
        description="SwiGLU MLP channel mixer",
    )
)
_register(
    PrimitiveOp(
        "rwkv_channel",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D*2",
        description="RWKV-style time-mixing channel update",
    )
)
_register(
    PrimitiveOp(
        "moe_topk",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D*8",
        description="Sparse Mixture-of-Experts channel mixer",
        config_keys=("num_experts", "top_k"),
    )
)
_register(
    PrimitiveOp(
        "moe_2expert",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D*2",
        description="Lightweight 2-expert MoE with learned gating",
        config_keys=(),
    )
)

# ── Functional (operator-learning / neural-field) ────────────────────

# ── Reference Architecture Ops ───────────────────────────────────────

_register(
    PrimitiveOp(
        "layernorm",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*2",
        description="Layer Normalization with learned affine",
    )
)
_register(
    PrimitiveOp(
        "embedding_lookup",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="64*D+D*D",
        description="Learnable codebook projection (soft VQ)",
        config_keys=("vocab_size",),
    )
)
_register(
    PrimitiveOp(
        "rope_rotate",
        OpCategory.FUNCTIONAL,
        1,
        "identity",
        description="Rotary Position Embedding (RoPE)",
    )
)
_register(
    PrimitiveOp(
        "gated_linear",
        OpCategory.PARAMETERIZED,
        1,
        "linear",
        has_params=True,
        param_formula="D*D*2",
        description="Fused gated linear: (x@W) * sigmoid(x@W_gate)",
        config_keys=("out_dim",),
    )
)
_register(
    PrimitiveOp(
        "cosine_similarity",
        OpCategory.LINEAR_ALGEBRA,
        2,
        "reduce_last",
        description="Cosine similarity between two tensors along last dim",
    )
)
_register(
    PrimitiveOp(
        "gather_topk",
        OpCategory.STRUCTURAL,
        2,
        "identity",
        description="Gather top-k vectors by score",
        config_keys=("k",),
    )
)
_register(
    PrimitiveOp(
        "rwkv_time_mixing",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D*3",
        description="RWKV WKV linear attention with learned decay",
        binding_range_class="full",
    )
)

# ── Routing Primitives (Phase 1/2) ───────────────────────────────────

_register(
    PrimitiveOp(
        "feature_sparsity",
        OpCategory.FUNCTIONAL,
        1,
        "identity",
        description="Top-k feature selection with STE mask (sparsifies feature dim, not tokens)",
        config_keys=("k",),
        standalone=False,
    )
)
_register(
    PrimitiveOp(
        "gated_lane_blend",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D*3+D*3",
        description="Learned difficulty-based lane blend: score tokens, soft-weight N internal linear projections",
        config_keys=("n_lanes",),
    )
)
_register(
    PrimitiveOp(
        "depth_gated_transform",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D*3+D*3",
        description="Learned difficulty-based depth gate: score tokens, apply variable-depth linear transforms",
        config_keys=("max_depth",),
    )
)
_register(
    PrimitiveOp(
        "adjacent_token_merge",  # was: token_merge
        OpCategory.FUNCTIONAL,
        1,
        "identity",
        description="Similarity-based token merging with seq_len restore: (B,S,D) -> (B,S,D)",
        config_keys=("n_keep",),
        byte_safe=True,  # kernel restores seq_len via nearest-neighbor mapping
        binding_range_class="local",
    )
)

# ── Routing (control-style ops operating on tensors) ──────────────────

_register(
    PrimitiveOp(
        "depth_token_mask",  # was: mod_topk
        OpCategory.FUNCTIONAL,
        1,
        "identity",
        has_params=True,
        param_formula="D",
        description="Mixture-of-Depths top-k token routing (masking)",
        config_keys=("capacity_factor",),
        byte_safe=False,
    )
)
_register(
    PrimitiveOp(
        "confidence_token_gate",  # was: early_exit
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D",
        description="Learned early-exit: confidence gate attenuates uncertain tokens",
        config_keys=("threshold",),
    )
)
_register(
    PrimitiveOp(
        "depth_weighted_proj",  # was: adaptive_recursion
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D*3+D*3",
        description="Learned per-token recursion depth with per-step transforms",
        config_keys=("max_depth",),
    )
)
_register(
    PrimitiveOp(
        "learned_token_gate",  # was: cascade
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D",
        description="Learned cascade gate: progressive difficulty-scaled token gating",
        config_keys=("threshold",),
    )
)
_register(
    PrimitiveOp(
        "cheap_verify_blend",  # was: speculative
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D+D",
        description="Speculative execution: cheap path + learned verification gate",
        config_keys=("threshold",),
    )
)
_register(
    PrimitiveOp(
        "hybrid_token_gate",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D",
        description="Cheap single-token gate for default-path vs informative-token routing",
        config_keys=("threshold",),
    )
)
_register(
    PrimitiveOp(
        "sparse_span_builder",
        OpCategory.FUNCTIONAL,
        1,
        "identity",
        description="Sparse fused pair/triplet span builder over informative tokens",
        config_keys=("span_width", "fallback_behavior"),
    )
)
_register(
    PrimitiveOp(
        "hybrid_sparse_router",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D*4",
        description="Two-stage hybrid router: token gate then sparse fused-span lane routing with default fallback",
        config_keys=("span_width", "lane_count", "confidence_threshold"),
    )
)
_register(
    PrimitiveOp(
        "lane_conditioned_block",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D",
        description="Lane-conditioned downstream block for routed spans",
        config_keys=("lane_id",),
    )
)
_register(
    PrimitiveOp(
        "default_path",
        OpCategory.FUNCTIONAL,
        1,
        "identity",
        description="Cheap/default routed bypass path",
    )
)

# ── Exotic Routing & Compression (Phase 4) ──────────────────────────

_register(
    PrimitiveOp(
        "difficulty_blend_3way",  # was: adaptive_lane_mixer
        OpCategory.PARAMETERIZED,
        2,
        "identity",
        has_params=True,
        param_formula="D*D*6",
        description="Difficulty-adaptive lane routing: routes tokens to experts based on learned difficulty",
    )
)
_register(
    PrimitiveOp(
        "score_depth_blend",  # was: mixed_recursion_gate
        OpCategory.PARAMETERIZED,
        2,
        "identity",
        has_params=True,
        param_formula="D*D*3",
        description="Tokens re-enter block with different transforms per recursion step, depth conditional",
    )
)
_register(
    PrimitiveOp(
        "signal_conditioned_compression",
        OpCategory.PARAMETERIZED,
        2,
        "identity",
        has_params=True,
        param_formula="D*D*2",
        description="Compression level chosen per-token by external signal (interpolates full-rank vs low-rank)",
    )
)
_register(
    PrimitiveOp(
        "token_class_proj",  # was: token_type_classifier
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D",
        description="Learned classifier to produce routing scores from token embeddings, projected back to model dim",
        config_keys=("n_classes",),
        standalone=False,
    )
)
_register(
    PrimitiveOp(
        "token_entropy",  # was: entropy_score
        OpCategory.FUNCTIONAL,
        1,
        "reduce_last",
        description="Shannon entropy of input scores as difficulty signal (B,S,K) -> (B,S,1)",
        standalone=False,
    )
)
_register(
    PrimitiveOp(
        "adaptive_rank_gate",  # was: progressive_compression_gate
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D*2",
        description="Per-token compression gate: learned projection decides full vs low-rank per token",
    )
)
_register(
    PrimitiveOp(
        "dual_compression_blend",  # was: compression_mixture_experts
        OpCategory.PARAMETERIZED,
        2,
        "identity",
        has_params=True,
        param_formula="D*D",
        description="Routing assigns tokens to method-specific compression experts (e.g. low-rank, sparse, bottleneck)",
    )
)

# ── 2026 Frontier Exotic Ops ────────────────────────────────────────

_register(
    PrimitiveOp(
        "relu_gated_moe",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D*4",
        description="ReLU-gated MoE (ReMoE): sparse expert activation with learned gate and internal expert projections",
    )
)
_register(
    PrimitiveOp(
        "ternary_projection",
        OpCategory.PARAMETERIZED,
        1,
        "linear",
        has_params=True,
        param_formula="D*D",
        description="1.58-bit ternary simulated projection (-1, 0, 1 weights)",
    )
)
_register(
    PrimitiveOp(
        "latent_attention_compressor",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D",
        description="Multi-Head Latent Attention (MLA) style KV cache compression",
    )
)

# ── Weight-efficient projections ─────────────────────────────────────

_register(
    PrimitiveOp(
        "low_rank_proj",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D//2",
        description="Low-rank factored projection (U @ V)",
    )
)
_register(
    PrimitiveOp(
        "grouped_linear",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D//4",
        description="Group-wise linear projection (4 groups)",
    )
)
_register(
    PrimitiveOp(
        "bottleneck_proj",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D//2",
        description="Bottleneck projection (down then up)",
    )
)
_register(
    PrimitiveOp(
        "shared_basis_proj",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*16",
        description="Shared-basis projection (8 basis vectors + mixing)",
    )
)
_register(
    PrimitiveOp(
        "tied_proj",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D//4",
        description="Tied projection (shared down+up weights)",
    )
)
_register(
    PrimitiveOp(
        "kronecker_linear",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="2*D",
        description="Kronecker-factored linear: W=A⊗B, 128x param compression",
    )
)
_register(
    PrimitiveOp(
        "sparse_bottleneck_moe",  # was: n_way_sparse_router
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="2*D*D+D*4",
        description="N-way sparse router: bottleneck experts with top-k activation",
        config_keys=("n_ways", "top_k"),
    )
)
_register(
    PrimitiveOp(
        "chebyshev_spectral_mix",
        OpCategory.MIXING,
        1,
        "identity",
        has_params=True,
        param_formula="6*D",
        description="Chebyshev polynomial spectral mixing (K*D params)",
        config_keys=("chebyshev_order",),
    )
)
# Note: Math space ops (padic_*, tropical_*, hyp_*, clifford_*, stdp_*)
# are dynamically registered by research.mathspaces.registry.register_all_mathspaces()
# Do NOT register them statically here — they need execute_fn from mathspaces.

# ── Frequency domain ─────────────────────────────────────────────────

_register(
    PrimitiveOp(
        "spectral_filter",
        OpCategory.FREQUENCY,
        1,
        "identity",
        has_params=True,
        param_formula="D",
        description="Learnable spectral filter: per-position FFT masking over feature dim",
    )
)

# ── Functional (operator-learning / neural-field) ────────────────────

_register(
    PrimitiveOp(
        "basis_expansion",
        OpCategory.FUNCTIONAL,
        1,
        "identity",
        has_params=True,
        param_formula="D*4",
        description="Basis-expansion layer: project through sinusoidal bases",
    )
)
_register(
    PrimitiveOp(
        "integral_kernel",
        OpCategory.FUNCTIONAL,
        1,
        "identity",
        has_params=True,
        param_formula="D*D",
        description="Integral kernel mixing: learned kernel over sequence positions",
        config_keys=("kernel_scale",),
    )
)
_register(
    PrimitiveOp(
        "fixed_point_iter",
        OpCategory.FUNCTIONAL,
        1,
        "identity",
        has_params=True,
        param_formula="D*D+D",
        description="Implicit fixed-point iteration: x = f(x) with learned f",
        numerically_risky=True,
        config_keys=("n_iters", "damping"),
    )
)


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
                shape_rule=manifest.get(
                    "shape_rule", "identity"
                ),  # Default to identity if missing
                has_params=perf.get("has_params", False),
                param_formula=perf.get("param_formula", "0"),
                preserves_gradient=perf.get("preserves_gradient", True),
                numerically_risky=perf.get("numerically_risky", False),
                description=desc,
                config_keys=tuple(manifest.get("params", {}).keys()),
                standalone=manifest.get("standalone", True),
            )
            _register(op)
            count += 1
        except Exception:
            continue
    return count


# Load Designer primitives if available
try:
    from pathlib import Path

    _DESIGNER_COMPONENTS = (
        Path(__file__).resolve().parents[2] / "aria_designer" / "components"
    )
    if _DESIGNER_COMPONENTS.exists():
        load_primitives_from_designer(_DESIGNER_COMPONENTS)
except Exception as _e:
    logger.warning("Failed to load designer primitives: %s", _e)


# ── Algebraic Space Tags ──────────────────────────────────────────────
# Valid spaces: "euclidean", "poincare", "tropical", "clifford", "padic",
# "spiking", "any".  Default is "euclidean" (composes with everything).
# Non-euclidean ops require matching space context in the grammar.

_ALGEBRAIC_SPACE_TAGS: Dict[str, str] = {
    # Poincaré / hyperbolic
    "poincare_add": "poincare",
    "exp_map": "poincare",
    "log_map": "poincare",
    "hyp_distance": "poincare",
    "hyp_linear": "poincare",
    "hyp_tangent_nonlinear": "poincare",
    "hyperbolic_norm": "poincare",
    # Tropical
    "tropical_add": "tropical",
    "tropical_matmul": "tropical",
    "tropical_attention": "tropical",
    "tropical_gate": "tropical",
    "tropical_center": "tropical",
    # Clifford
    "clifford_attention": "clifford",
    "geometric_product": "clifford",
    "rotor_transform": "clifford",
    "grade_select": "clifford",
    "grade_mix": "clifford",
    # p-adic
    "padic_expand": "padic",
    "padic_gate": "padic",
    "padic_residual": "padic",
    "ultrametric_attention": "padic",
    # Tropical (additional)
    "tropical_router": "tropical",
    "tropical_moe": "tropical",
    # Spiking
    "lif_neuron": "spiking",
    "spike_rate_code": "spiking",
    "stdp_attention": "spiking",
    "sparse_threshold": "spiking",
}

_ALGEBRAIC_TYPE_TAGS: Dict[str, AlgebraicType] = {
    "poincare_add": AlgebraicType("poincare", "unit_ball", "unit_ball"),
    "exp_map": AlgebraicType("poincare", "real", "unit_ball"),
    "log_map": AlgebraicType("euclidean", "unit_ball", "real"),
    "hyp_distance": AlgebraicType("poincare", "unit_ball", "real"),
    "hyp_linear": AlgebraicType("poincare", "unit_ball", "unit_ball"),
    "hyp_tangent_nonlinear": AlgebraicType("poincare", "unit_ball", "unit_ball"),
    "hyperbolic_norm": AlgebraicType("poincare", "real", "real"),
    "tropical_add": AlgebraicType("tropical", "real", "real"),
    "tropical_matmul": AlgebraicType("tropical", "real", "real"),
    "tropical_attention": AlgebraicType("tropical", "real", "real"),
    "tropical_gate": AlgebraicType("tropical", "real", "real"),
    "tropical_center": AlgebraicType("tropical", "real", "real"),
    "tropical_router": AlgebraicType("tropical", "real", "real"),
    "tropical_moe": AlgebraicType("tropical", "real", "real"),
    "padic_expand": AlgebraicType("padic", "real", "real"),
    "padic_gate": AlgebraicType("padic", "real", "real"),
    "padic_residual": AlgebraicType("padic", "real", "real"),
    "ultrametric_attention": AlgebraicType("padic", "real", "real"),
    "clifford_attention": AlgebraicType("clifford", "real", "real"),
    "geometric_product": AlgebraicType("clifford", "multivector", "multivector"),
    "rotor_transform": AlgebraicType("clifford", "real", "multivector"),
    "grade_select": AlgebraicType("clifford", "any", "real"),
    "grade_mix": AlgebraicType("clifford", "real", "real"),
    "lif_neuron": AlgebraicType("spiking", "real", "real"),
    "spike_rate_code": AlgebraicType("spiking", "real", "real"),
    "stdp_attention": AlgebraicType("spiking", "real", "real"),
    "sparse_threshold": AlgebraicType("spiking", "real", "real"),
}

_SPACE_DEFAULT_TYPES: Dict[str, AlgebraicType] = {
    "euclidean": AlgebraicType("euclidean", "real", "real"),
    "poincare": AlgebraicType("poincare", "unit_ball", "unit_ball"),
    "tropical": AlgebraicType("tropical", "tropical_tensor", "tropical_tensor"),
    "clifford": AlgebraicType("clifford", "multivector", "multivector"),
    "padic": AlgebraicType("padic", "padic_tensor", "padic_tensor"),
    "spiking": AlgebraicType("spiking", "spikes", "spikes"),
    "any": AlgebraicType("any", "any", "any"),
}

VALID_ALGEBRAIC_SPACES: frozenset = frozenset(
    {
        "euclidean",
        "poincare",
        "tropical",
        "clifford",
        "padic",
        "spiking",
        "any",
    }
)


def _apply_algebraic_space_tags() -> None:
    """Apply algebraic space tags to registered primitives.

    Uses object.__setattr__ because PrimitiveOp is frozen.
    """
    for op_name, space in _ALGEBRAIC_SPACE_TAGS.items():
        op = PRIMITIVE_REGISTRY.get(op_name)
        if op is not None:
            object.__setattr__(op, "algebraic_space", space)


_apply_algebraic_space_tags()


def _apply_algebraic_type_tags() -> None:
    for op_name, algebraic_type in _ALGEBRAIC_TYPE_TAGS.items():
        op = PRIMITIVE_REGISTRY.get(op_name)
        if op is not None:
            object.__setattr__(op, "algebraic_type", algebraic_type)
            object.__setattr__(op, "algebraic_space", algebraic_type.space)


_apply_algebraic_type_tags()


# ── True Token-Routing Ops ──────────────────────────────────────────
# These dispatch tokens to genuinely different compute types via
# gather-scatter, unlike gated mixture ops which blend identical
# linear projections with different weights.

_register(
    PrimitiveOp(
        "hetero_moe",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D*7",
        description="Heterogeneous MoE: routes tokens to attention, conv1d, or SSM experts",
    )
)
_register(
    PrimitiveOp(
        "arch_router",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D*8",
        description="Architecture router: tokens choose transformer, mamba, or MLP processing style",
    )
)
_register(
    PrimitiveOp(
        "compute_budget_router",
        OpCategory.PARAMETERIZED,
        1,
        "identity",
        has_params=True,
        param_formula="D*D*5",
        description="Adaptive compute budget: easy tokens get cheap linear, medium conv, hard attention",
    )
)


# Ops that MUST have a residual bypass around them to preserve
# information flow.  Enforced at template/validation level.
REQUIRES_RESIDUAL_BYPASS: frozenset = frozenset(
    {
        "adjacent_token_merge",
        "depth_token_mask",
        "learned_token_gate",
        "hybrid_token_gate",
    }
)


# ── Op Wiring Constraints ──────────────────────────────────────────────
# Rules for ops that require specific predecessors or signal shapes.
# The grammar validator checks these after graph generation.
#
# Format: op_name → {
#   "input_signals": {input_idx: {"from_ops": [...], "shape_hint": str}},
#   "requires_residual": bool,  — must have residual bypass around it
# }

OP_WIRING_RULES: Dict[str, dict] = {
    # Signal producers: non-standard output shape, can't be terminal
    "token_entropy": {  # was: entropy_score
        "output_shape": "(B,S,1)",
        "input_signals": {
            0: {
                "from_ops": ["token_class_proj", "linear_proj"],
                "shape_hint": "(B,S,K) class logits or projected scores",
            },
        },
        "valid_consumers": [
            "signal_conditioned_compression",  # as input[1]
            "mul",  # template gating pattern: mul(x, entropy)
        ],
        "note": "Produces (B,S,1) difficulty signal — input must be class logits, not raw activations",
    },
    "token_class_proj": {  # was: token_type_classifier
        "output_shape": "(B,S,D)",
        "valid_consumers": [
            "token_entropy",  # classifier → entropy → routing
            "dual_compression_blend",  # as input[1] with n_classes matching
            "score_depth_blend",  # as input[1] depth scores
            "signal_conditioned_compression",  # as input[1] routing signal
        ],
        "note": "Produces class scores — feeds token_entropy, MoE routing, or depth gates",
    },
    # 2-input consumers: input[1] must come from specific signal producers
    "signal_conditioned_compression": {
        "input_signals": {
            1: {
                "from_ops": ["token_entropy", "token_class_proj", "mul"],
                "shape_hint": "(B,S,1) or (B,S,K) routing signal",
            },
        },
    },
    "dual_compression_blend": {
        "input_signals": {
            1: {
                "from_ops": ["token_class_proj", "token_entropy", "linear_proj"],
                "shape_hint": "(B,S,2+) expert routing weights",
            },
        },
    },
    "score_depth_blend": {  # was: mixed_recursion_gate
        "input_signals": {
            1: {
                "from_ops": [
                    "token_class_proj",
                    "linear_proj",
                    "depth_gated_transform",
                ],
                "shape_hint": "(B,S,max_depth) depth scores",
            },
        },
    },
    "sparse_span_builder": {
        "input_signals": {
            0: {
                "from_ops": [
                    "hybrid_token_gate",
                    "confidence_token_gate",
                    "learned_token_gate",
                    "mul",
                    "linear_proj",
                ],
                "shape_hint": "(B,S,D) informative-token activations",
            },
        },
        "valid_consumers": ["hybrid_sparse_router", "lane_conditioned_block", "add"],
    },
    "hybrid_sparse_router": {
        "input_signals": {
            0: {
                "from_ops": [
                    "sparse_span_builder",
                    "hybrid_token_gate",
                    "linear_proj",
                    "identity",
                ],
                "shape_hint": "(B,S,D) span features or informative-token activations",
            },
        },
        "valid_consumers": ["lane_conditioned_block", "add", "default_path"],
        "requires_residual": True,
    },
    "difficulty_blend_3way": {  # was: adaptive_lane_mixer
        "input_signals": {
            1: {
                "from_ops": None,  # any (B,S,D) input — self-contained gate
                "shape_hint": "(B,S,D) same as input[0]",
            },
        },
    },
    # Ops that MUST have residual bypass (information-destructive)
    "adjacent_token_merge": {"requires_residual": True},
    "depth_token_mask": {"requires_residual": True},
    "confidence_token_gate": {"requires_residual": True},
    "learned_token_gate": {"requires_residual": True},
    "hybrid_token_gate": {"requires_residual": True},
    "gated_lane_blend": {"min_layer_depth": 2},
    "depth_gated_transform": {"min_layer_depth": 2},
}


def get_wiring_rule(op_name: str) -> Optional[dict]:
    """Get wiring constraints for an op, or None if unconstrained."""
    return OP_WIRING_RULES.get(op_name)


def validate_wiring(graph, errors: Optional[List[str]] = None) -> List[str]:
    """Validate that all op wiring constraints are satisfied in a graph.

    Returns list of error strings (empty = valid).
    """
    if errors is None:
        errors = []
    for nid, node in graph.nodes.items():
        if node.is_input:
            continue
        rule = OP_WIRING_RULES.get(node.op_name)
        if rule is None:
            continue

        # Check input signal constraints
        input_signals = rule.get("input_signals", {})
        for idx, constraint in input_signals.items():
            if idx >= len(node.input_ids):
                errors.append(
                    f"{node.op_name} requires input[{idx}] but only has "
                    f"{len(node.input_ids)} inputs"
                )
                continue
            from_ops = constraint.get("from_ops")
            if from_ops is None:
                continue  # any op allowed
            source_id = node.input_ids[idx]
            source_node = graph.nodes.get(source_id)
            if (
                source_node
                and not source_node.is_input
                and source_node.op_name not in from_ops
            ):
                errors.append(
                    f"{node.op_name} input[{idx}] requires signal from "
                    f"{from_ops} but got '{source_node.op_name}'. "
                    f"Expected: {constraint.get('shape_hint', 'compatible signal')}"
                )

        # Check output consumers for signal producers
        valid_consumers = rule.get("valid_consumers")
        if valid_consumers is not None:
            consumers = [
                n for n in graph.nodes.values() if not n.is_input and nid in n.input_ids
            ]
            for consumer in consumers:
                if (
                    consumer.op_name not in valid_consumers
                    and consumer.op_name != "add"
                ):
                    errors.append(
                        f"{node.op_name} output consumed by '{consumer.op_name}' "
                        f"but only valid consumers are {valid_consumers}"
                    )

    return errors


def algebraic_types_compatible(
    producer: AlgebraicType,
    consumer: AlgebraicType,
) -> bool:
    """Check output→input algebraic compatibility."""
    if consumer.input_constraint == "any" or producer.output_guarantee == "any":
        return True
    if consumer.input_constraint == producer.output_guarantee:
        return True
    if consumer.input_constraint == "real" and producer.output_guarantee == "real":
        return True
    return False


def default_algebraic_type_for_space(space: str) -> AlgebraicType:
    return _SPACE_DEFAULT_TYPES.get(space, _SPACE_DEFAULT_TYPES["euclidean"])


# ── Helper Functions ──────────────────────────────────────────────────


def get_primitive(name: str) -> PrimitiveOp:
    """Get a primitive by name, resolving aliases for renamed ops."""
    resolved = OP_NAME_ALIASES.get(name, name)
    if resolved not in PRIMITIVE_REGISTRY:
        # Lazily register mathspace ops on first miss
        from ..mathspaces.registry import register_all_mathspaces

        register_all_mathspaces()
    if resolved not in PRIMITIVE_REGISTRY:
        raise KeyError(
            f"Unknown primitive: {name}. Available: {list(PRIMITIVE_REGISTRY.keys())}"
        )
    return PRIMITIVE_REGISTRY[resolved]


_BINDING_PRIORITY = {"full": 3, "medium": 2, "local": 1, "none": 0}
_BINDING_PRIORITY_REV = {v: k for k, v in _BINDING_PRIORITY.items()}


def graph_binding_range_class(graph: "ComputationGraph") -> str:
    """Return the maximum binding range class across all ops in a graph.

    Priority: full > medium > local > none.
    """
    best = 0
    for node in graph.nodes.values():
        if node.is_input:
            continue
        op = PRIMITIVE_REGISTRY.get(node.op_name)
        if op is not None:
            best = max(best, _BINDING_PRIORITY.get(op.binding_range_class, 0))
            if best == 3:  # full — can't go higher
                return "full"
    return _BINDING_PRIORITY_REV.get(best, "none")


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
        tagged = _ALGEBRAIC_TYPE_TAGS.get(op.name)
        if tagged is not None:
            object.__setattr__(PRIMITIVE_REGISTRY[op.name], "algebraic_type", tagged)
            object.__setattr__(
                PRIMITIVE_REGISTRY[op.name], "algebraic_space", tagged.space
            )


# ── Eager mathspace registration ─────────────────────────────────────
# All direct PRIMITIVE_REGISTRY[name] reads (grammar.py, templates.py,
# compiler.py fallback, profiler.py) bypass get_primitive()'s lazy fallback.
# Register mathspace ops at import time so they are always present.
#
# Guard against circular import: mathspaces.registry imports from this module.
# When mathspaces is already mid-import, skip — mathspaces/__init__.py will
# call register_all_mathspaces() itself once registry.py finishes loading.
import sys as _sys

if "research.mathspaces.registry" not in _sys.modules:
    from ..mathspaces.registry import register_all_mathspaces

    register_all_mathspaces()
