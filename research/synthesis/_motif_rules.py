"""Activation rules, math-space composition rules, and activation selection.

Context-aware placement rules derived from empirical analysis of 203 training runs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .op_roles import OpRole

# ── Activation substitution pool ────────────────────────────────────
# When a motif step is marked substitutable=True and has role ACTIVATE,
# these are the valid replacements.
ACTIVATION_POOL: Tuple[str, ...] = (
    "gelu",
    "silu",
    "relu",
    "tanh",
    "sigmoid",
    "sin",
    "cos",
    "abs",
    "neg",
    "square",
    "softmax_last",
    "reciprocal",
)

# ── Context-aware activation placement rules ───────────────────────
# Each activation maps to:
#   "after": set of predecessor OpRoles or specific op names where valid.
#             None means any predecessor is fine.
#   "before": set of successor op names where valid.
#              None means any successor is fine.
# Derived from empirical analysis of 203 training runs.

ACTIVATION_RULES: Dict[str, Dict] = {
    # Universal: safe everywhere
    "gelu": {"after": None, "before": None},
    "silu": {"after": None, "before": None},
    "relu": {"after": None, "before": None},
    # Risky: exp amplifies, must only follow bounded output (norm/sigmoid/tanh)
    "exp": {
        "after": {OpRole.NORMALIZE, "tanh", "sigmoid", "rmsnorm", "layernorm"},
        "before": None,
    },
    # Bounded [-1,1]: safe after up-projections, but kills signal after
    # down-projections (0% S1 with linear_proj_down across 60 entries).
    # Restrict: must feed into a projection or merge, not stand alone.
    "tanh": {"after": None, "before": {"linear_proj", "linear_proj_up", "mul", "add"}},
    # Gating: sigmoid → [0,1], useful before multiplicative/decay ops and residual add
    "sigmoid": {
        "after": None,
        "before": {
            "mul",
            "outer_product",
            "matmul",
            "cosine_similarity",
            "cumprod_safe",
            "add",
            "reciprocal",
            "log",
            "moe_topk",
            "moe_2expert",
        },
    },
    # Periodic: learnable frequency features, best after projections/norms.
    # Must feed into projection/merge — periodic output in compressed space
    # is noise without a learned transform to interpret it (0% S1 after down-proj).
    "sin": {
        "after": {OpRole.PROJECT, OpRole.NORMALIZE},
        "before": {"linear_proj", "linear_proj_up", "mul", "add"},
    },
    "cos": {
        "after": {OpRole.PROJECT, OpRole.NORMALIZE},
        "before": {"linear_proj", "linear_proj_up", "mul", "add"},
    },
    # Magnitude: loses sign info, must feed into gating/scaling
    "abs": {
        "after": {OpRole.PROJECT, OpRole.MIX},
        "before": {"mul", "learnable_scale", "topk_gate", "add"},
    },
    # Inversion: -x, works broadly but not after sign-flipping ops
    "neg": {
        "after": {OpRole.PROJECT, OpRole.MIX, OpRole.ACTIVATE},
        "before": None,
    },
    # Quadratic: magnitude-expanding, needs bounded input
    "square": {
        "after": {"ternary_projection", "tanh", "sigmoid", OpRole.NORMALIZE},
        "before": None,
    },
    # Attention-norm: only before matmul-like ops that use weights
    "softmax_last": {
        "after": {OpRole.PROJECT, OpRole.MIX},
        "before": {
            "matmul",
            "outer_product",
            "mul",
            "cosine_similarity",
            "div_safe",
            "add",
        },
    },
    # Reciprocal: bounded [0.5, 1.0] via 1/(1+sigmoid(x)), safe after norm/bounded
    "reciprocal": {
        "after": {OpRole.NORMALIZE, "sigmoid", "tanh", "rmsnorm", "layernorm"},
        "before": None,
    },
}


def _get_valid_activations(
    prev_op: Optional[str] = None,
    next_op: Optional[str] = None,
) -> List[str]:
    """Filter ACTIVATION_POOL by context rules.

    Args:
        prev_op: Name of the preceding op (checked against "after" rules).
        next_op: Name of the following op (checked against "before" rules).

    Returns:
        List of activation names valid in this context. Falls back to
        universally-safe activations if nothing matches.
    """
    from .op_roles import get_role

    prev_role = get_role(prev_op) if prev_op else None
    candidates = []
    for act in ACTIVATION_POOL:
        rules = ACTIVATION_RULES.get(act)
        if rules is None:
            candidates.append(act)
            continue
        # Check "after" constraint
        after = rules.get("after")
        if after is not None and prev_op is not None:
            if prev_op not in after and (prev_role is None or prev_role not in after):
                continue
        # Check "before" constraint
        before = rules.get("before")
        if before is not None and next_op is not None:
            if next_op not in before:
                continue
        candidates.append(act)
    # Fallback: always allow the universally safe ones
    if not candidates:
        candidates = ["gelu", "silu", "relu"]
    return candidates


# ── Math-space composition rules ─────────────────────────────────
# Tells the grammar which ops MUST be preceded by a normalizer for
# numerical stability.  Checked in templates._instantiate_motif().
#   "must_precede": predecessor must be one of these ops/roles
#   "must_follow_with": a successor from this set must appear after the op

MATH_SPACE_RULES: Dict[str, Dict] = {
    # Tropical ops: input must be bounded (they use min/max/softmax internally).
    # tropical_gate: min-based routing produces sparse gradients. Must be
    # preceded by norm, tropical_attention, or spiking ops (which normalize
    # internally via firing rate encoding). The proven chains:
    #   layernorm → tropical_attention → tropical_gate → tropical_center (8/11 S1, lr=0.079)
    #   spike_rate_code → tropical_gate → linear_proj (2 S1, lr=0.007)
    "tropical_gate": {
        "must_precede": {
            "rmsnorm",
            "layernorm",
            "tropical_attention",
            "lif_neuron",
            "spike_rate_code",
            "sparse_threshold",
        },
        "must_follow_with": {
            "linear_proj",
            "linear_proj_down",
            "gated_linear",
            "tropical_center",
        },
    },
    # tropical_attention: sequential min ops compound gradient sparsity.
    # Must be followed by a projection, tropical_center, or tropical_gate.
    # The proven architecture (8/11 S1 passes, best lr=0.079) chains:
    #   tropical_attention → tropical_gate → tropical_center → linear_proj
    # (Updated 2026-03-22: added tropical_gate as valid successor per DB evidence.)
    "tropical_attention": {
        "must_precede": {"rmsnorm", "layernorm"},
        "must_follow_with": {
            "linear_proj",
            "linear_proj_down",
            "gated_linear",
            "tropical_center",
            "tropical_gate",
        },
    },
    # tropical_center: structural centerer inside tropical motifs. Must follow
    # a tropical mixer (tropical_attention or tropical_gate) and must be
    # followed by a projection to return gradient density.
    # (Volta audit 2026-03-21: 0% S1 in freeform context, passes in
    # valid tropical_core context with loss_ratio=0.52.)
    "tropical_center": {
        "must_follow": {"tropical_attention", "tropical_gate"},
        "must_follow_with": {"linear_proj", "linear_proj_down", "tropical_gate"},
    },
    # tropical_matmul: binary min-based matmul. Needs gradient re-densification
    # via projection after the matmul. The norm requirement is satisfied by the
    # template structure (norm → proj → tropical_matmul), not by must_precede
    # which only checks direct parents (binary ops take projections as inputs).
    "tropical_matmul": {
        "must_follow_with": {"linear_proj", "linear_proj_down"},
    },
    # Clifford ops: input must be bounded (geometric product can amplify)
    "clifford_attention": {"must_precede": {"rmsnorm", "layernorm"}},
    "grade_mix": {
        "must_follow": {
            "clifford_attention",
            "rotor_transform",
            "grade_select",
            "geometric_product",
        },
        "must_follow_with": {
            "linear_proj",
            "linear_proj_down",
            "add",
            "rmsnorm",
            "layernorm",
        },
    },
    # P-adic ops: expansion doubles dim, must project back
    "padic_expand": {
        "must_precede": {"rmsnorm", "layernorm"},
        "must_follow_with": {
            "linear_proj",
            "linear_proj_down",
            "padic_residual",
            "ultrametric_attention",
        },
    },
    # State space: bound input to prevent scan explosion
    "state_space": {"must_precede": {"rmsnorm", "layernorm"}},
    # conv_only: local-only mixing is insufficient as a sole mixer. Must be
    # preceded by normalization and followed by a projection that can learn
    # non-local patterns. (Diagnosis 2026-03-20: 0% S1 rate across 40 attempts,
    # all unstable_dynamics. The conv itself is fine — it just can't carry a
    # language model alone.)
    "conv_only": {
        "must_precede": {"rmsnorm", "layernorm"},
        "must_follow_with": {
            "linear_proj",
            "linear_proj_down",
            "gated_linear",
            "fused_linear_gelu",
        },
    },
    # Spectral filter: always inside residual (handled by grammar.py fix)
    "spectral_filter": {"must_precede": {"rmsnorm", "layernorm"}},
    # Spiking ops: AlgebraicType("spiking", "real", "real") is compatible
    # with euclidean, so algebraic_types_compatible() does NOT prevent
    # placement in non-spiking contexts. These rules enforce spiking-only
    # predecessor chains.
    # (Volta audit 2026-03-21: sparse_threshold and stdp_attention both have
    # 8% compile rate when placed in non-spiking contexts. 92% failure from
    # forward_error and nan_forward.)
    "lif_neuron": {
        "must_follow_with": {
            "spike_rate_code",
            "sparse_threshold",
            "stdp_attention",
            "tropical_gate",
        },
    },
    "sparse_threshold": {
        "must_follow": {"lif_neuron", "spike_rate_code"},
    },
    "stdp_attention": {
        "must_follow": {"sparse_threshold", "spike_rate_code", "lif_neuron"},
    },
    # Hyperbolic ops: require exp_map → op → log_map bridge for Poincare
    # ball operations. Algebraic type provides partial protection but does
    # not enforce the bridge chain.
    # (Volta audit 2026-03-21: hyp_linear valid context loss_ratio=0.165,
    # default context loss_ratio=0.43. 18/74 init_poisoned, 43/74 s1_fail.)
    "hyp_linear": {
        "must_follow": {"exp_map"},
    },
    "hyp_tangent_nonlinear": {
        "must_follow": {"hyp_linear"},
        "must_follow_with": {"log_map", "linear_proj"},
    },
    # Numerically risky ops: must be preceded by norm to bound activations
    "cosine_similarity": {"must_precede": {"rmsnorm", "layernorm"}},
    "cumprod_safe": {"must_precede": {"rmsnorm", "layernorm"}},
    "div_safe": {"must_precede": {"rmsnorm", "layernorm"}},
    "exp": {"must_precede": {"rmsnorm", "layernorm"}},
    "hyperbolic_norm": {"must_precede": {"rmsnorm", "layernorm"}},
    "log": {"must_precede": {"rmsnorm", "layernorm"}},
    "reciprocal": {"must_precede": {"rmsnorm", "layernorm"}},
    "sqrt": {"must_precede": {"rmsnorm", "layernorm"}},
    # Ops with domain constraints
    "spike_rate_code": {"must_precede": {"rmsnorm", "layernorm", "lif_neuron"}},
    "padic_gate": {"must_precede": {"rmsnorm", "layernorm"}},
    "chebyshev_spectral_mix": {"must_precede": {"rmsnorm", "layernorm"}},
    "kronecker_linear": {"must_precede": {"rmsnorm", "layernorm"}},
    "sparse_bottleneck_moe": {"must_precede": {"rmsnorm", "layernorm"}},
    "integral_kernel": {"must_precede": {"rmsnorm", "layernorm"}},
    "basis_expansion": {"must_precede": {"rmsnorm", "layernorm"}},
    "rotor_transform": {"must_precede": {"rmsnorm", "layernorm"}},
    # Tropical routing ops: tropical algebraic type amplifies via min/max,
    # must be preceded by norm and followed by projection to re-densify gradients.
    "tropical_moe": {
        "must_precede": {"rmsnorm", "layernorm"},
        "must_follow_with": {"linear_proj", "linear_proj_down", "gated_linear"},
    },
    "tropical_router": {
        "must_precede": {"rmsnorm", "layernorm"},
        "must_follow_with": {"linear_proj", "linear_proj_down", "gated_linear"},
    },
    # Sequence-length-altering ops: SSM/recurrent ops assume fixed sequence
    # length and will crash or produce garbage after these ops.
    "adjacent_token_merge": {
        "must_precede": {"rmsnorm", "layernorm"},
        "must_follow_with": {
            "linear_proj",
            "linear_proj_down",
            "linear_proj_up",
            "gated_linear",
            "fused_linear_gelu",
            "gelu",
            "silu",
            "relu",
            # Data-mined: top-3 token_merge graphs (loss_ratio 0.006-0.043)
            # use conv1d_seq/swiglu_mlp after merge.  Post-merge rmsnorm
            # satisfies conv1d_seq.must_precede when conv follows merge.
            "conv1d_seq",
            "swiglu_mlp",
            "rmsnorm",
            "layernorm",
        },
    },
    "depth_token_mask": {
        "must_precede": {"rmsnorm", "layernorm"},
        "must_follow_with": {
            "linear_proj",
            "linear_proj_down",
            "linear_proj_up",
            "gated_linear",
            "fused_linear_gelu",
        },
    },
    # Ops that were in the original audit's must_precede list
    "exp_map": {"must_precede": {"rmsnorm", "layernorm", "linear_proj"}},
    "log_map": {"must_follow": {"exp_map", "poincare_add", "hyp_linear"}},
    "fixed_point_iter": {"must_precede": {"rmsnorm", "layernorm"}},
    "hyp_distance": {"must_precede": {"exp_map"}},
    "rwkv_time_mixing": {"must_precede": {"rmsnorm", "layernorm"}},
    "selective_scan": {"must_precede": {"rmsnorm", "layernorm"}},
    "n_way_sparse_router": {"must_precede": {"rmsnorm", "layernorm"}},
    # Mixing ops: 3.3% pass when pred=input, 1% when pred=add, 57% when pred=norm.
    # Every sequence-mixing op must be preceded by normalization.
    # Data: 3000 experiments, mixing ops after add → 1% S0 pass rate.
    "softmax_attention": {
        "must_precede": {
            "rmsnorm",
            "layernorm",
            "rope_rotate",
            "causal_mask",
            "sliding_window_mask",
        }
    },
    "linear_attention": {
        "must_precede": {"rmsnorm", "layernorm", "sliding_window_mask"}
    },
    "diff_attention": {"must_precede": {"rmsnorm", "layernorm"}},
    "graph_attention": {"must_precede": {"rmsnorm", "layernorm"}},
    "local_window_attn": {
        "must_precede": {"rmsnorm", "layernorm", "sliding_window_mask"}
    },
    "multi_head_mix": {"must_precede": {"rmsnorm", "layernorm"}},
    "gated_delta": {"must_precede": {"rmsnorm", "layernorm"}},
    "conv1d_seq": {"must_precede": {"rmsnorm", "layernorm"}},
    "latent_attention_compressor": {"must_precede": {"rmsnorm", "layernorm"}},
}
