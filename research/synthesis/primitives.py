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

    def __hash__(self):
        return hash(self.name)


# ── Shape Rules ──────────────────────────────────────────────────────
# These are symbolic. The actual shape computation happens in graph.py
# when we know concrete dimensions.

SHAPE_RULES = {
    "identity": "Input shape passes through unchanged",
    "binary_broadcast": "Broadcast binary op (shapes must be compatible)",
    "reduce_last": "Reduce last dimension: (B,S,D) -> (B,S,1)",
    "reduce_seq": "Reduce sequence dimension: (B,S,D) -> (B,1,D)",
    "matmul": "Matrix multiply: (B,S,D) x (B,D,K) -> (B,S,K) or similar",
    "outer": "Elementwise (Hadamard) product: (B,S,D) x (B,S,D) -> (B,S,D)",
    "transpose_seq_dim": "Swap seq and dim: (B,S,D) -> (B,D,S)",
    "split": "Split last dim into N parts: (B,S,D) -> N x (B,S,D//N)",
    "concat": "Concat along last dim: N x (B,S,D_i) -> (B,S,sum(D_i))",
    "linear": "Linear projection: (B,S,D_in) -> (B,S,D_out)",
    "roll": "Circular shift along sequence dim: (B,S,D) -> (B,S,D)",
    "gather": "Gather along dim using indices from argsort",
    "scatter": "Scatter values back to original positions",
    "rfft": "Real FFT along sequence dim: (B,S,D) -> (B,S//2+1,D) complex",
    "irfft": "Inverse real FFT: (B,S//2+1,D) -> (B,S,D)",
    "sort": "Sort along sequence dim, returns sorted + indices",
    "unsort": "Inverse permutation to undo a sort",
    "cumulative": "Cumulative op along sequence dim: shape unchanged",
    "softmax": "Softmax along specified dim: shape unchanged",
    "causal_mask": "Apply causal mask: shape unchanged",
    "scale": "Learnable per-dim scale: shape unchanged",
    "bias": "Learnable per-dim bias: shape unchanged",
}


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
_register(PrimitiveOp("sum_seq", OpCategory.REDUCTION, 1, "reduce_seq",
                       description="Sum along sequence dimension"))
_register(PrimitiveOp("mean_seq", OpCategory.REDUCTION, 1, "reduce_seq",
                       description="Mean along sequence dimension"))
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

# ── Structural ────────────────────────────────────────────────────────

_register(PrimitiveOp("split2", OpCategory.STRUCTURAL, 1, "split",
                       description="Split last dim into 2 equal parts",
                       config_keys=("n_splits",)))
_register(PrimitiveOp("split3", OpCategory.STRUCTURAL, 1, "split",
                       description="Split last dim into 3 equal parts",
                       config_keys=("n_splits",)))
_register(PrimitiveOp("concat", OpCategory.STRUCTURAL, 2, "concat",
                       description="Concatenate along last dimension"))
_register(PrimitiveOp("roll_seq", OpCategory.STRUCTURAL, 1, "roll",
                       description="Circular shift by 1 along sequence dim"))
_register(PrimitiveOp("roll_neg", OpCategory.STRUCTURAL, 1, "roll",
                       description="Circular shift by -1 along sequence dim"))
_register(PrimitiveOp("gather_sorted", OpCategory.STRUCTURAL, 2, "gather",
                       description="Gather elements using sort indices"))
_register(PrimitiveOp("scatter_unsort", OpCategory.STRUCTURAL, 2, "scatter",
                       description="Scatter elements back using unsort indices"))
_register(PrimitiveOp("multi_head_mix", OpCategory.STRUCTURAL, 1, "identity",
                       description="Multi-head reshape + per-head L2 normalize",
                       config_keys=("n_heads",)))

# ── Parameterized (learnable) ─────────────────────────────────────────

_register(PrimitiveOp("linear_proj", OpCategory.PARAMETERIZED, 1, "linear",
                       has_params=True, param_formula="D*D",
                       description="Learned linear projection (D -> D)"))
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
_register(PrimitiveOp("softmax_seq", OpCategory.SEQUENCE, 1, "softmax",
                       description="Softmax along sequence dimension"))
_register(PrimitiveOp("causal_mask", OpCategory.SEQUENCE, 1, "causal_mask",
                       description="Apply causal (lower-triangular) mask"))
_register(PrimitiveOp("sort_seq", OpCategory.SEQUENCE, 1, "sort",
                       description="Sort along sequence dim by learned key"))
_register(PrimitiveOp("argsort_seq", OpCategory.SEQUENCE, 1, "sort",
                       description="Argsort along sequence dim"))
_register(PrimitiveOp("local_window_attn", OpCategory.SEQUENCE, 1, "identity",
                       description="Local windowed causal self-attention (Q=K=V)",
                       config_keys=("window_size",)))
_register(PrimitiveOp("sliding_window_mask", OpCategory.SEQUENCE, 1, "causal_mask",
                       description="Exponential distance decay mask for windowed composition",
                       config_keys=("window_size",)))
_register(PrimitiveOp("token_pool_restore", OpCategory.SEQUENCE, 1, "identity",
                       description="Pool adjacent token pairs then restore via repeat"))

# ── Frequency Domain ──────────────────────────────────────────────────

_register(PrimitiveOp("rfft_seq", OpCategory.FREQUENCY, 1, "rfft",
                       description="Real FFT along sequence dimension"))
_register(PrimitiveOp("irfft_seq", OpCategory.FREQUENCY, 1, "irfft",
                       description="Inverse real FFT along sequence dim"))

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
_register(PrimitiveOp("fourier_mixing", OpCategory.MIXING, 1, "identity",
                       description="Unparameterized global mixing via FFT"))
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


def list_by_n_inputs(n: int) -> List[PrimitiveOp]:
    """List primitives by number of inputs."""
    return [op for op in PRIMITIVE_REGISTRY.values() if op.n_inputs == n]


def register_external_primitive(op: PrimitiveOp) -> None:
    """Register a primitive from external sources (e.g., math spaces)."""
    if op.name not in PRIMITIVE_REGISTRY:
        PRIMITIVE_REGISTRY[op.name] = op
        if op.name not in OPCODE_MAP:
            opcode = len(OPCODE_MAP)
            OPCODE_MAP[op.name] = opcode
            REVERSE_OPCODE_MAP[opcode] = op.name
