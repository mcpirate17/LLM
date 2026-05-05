"""Descriptive properties for template and slot meta-analysis datasets.

The functions here are separate from the synthesis runtime. They infer
interpretable, static columns from template and slot metadata so analytics can
ask questions such as "do routing-heavy late slots fail differently than early
memory slots?" without changing generation behavior or the live notebook DB.
"""

from __future__ import annotations

import math
import re
from typing import Any


PROPERTY_VERSION = "template_slot_descriptive_v1"
OP_PROPERTY_VERSION = "op_math_descriptive_v1"

_INSTANCE_RE = re.compile(r"\[\d+\]")

_ATTENTION_TOKENS = ("attn", "attention", "retrieval", "relation")
_SSM_TOKENS = ("ssm", "scan", "state", "mamba", "rwkv", "retention", "delta")
_CONV_TOKENS = ("conv", "window", "local")
_ROUTING_TOKENS = ("router", "routing", "route", "gated", "gate", "lane")
_COMPRESSION_TOKENS = ("compress", "bottleneck", "merge", "sparse", "topk")
_MEMORY_TOKENS = ("memory", "retrieval", "retention", "state", "cache")
_MATH_SPACE_TOKENS = (
    "tropical",
    "padic",
    "clifford",
    "wavelet",
    "fourier",
    "spectral",
    "hyperbolic",
    "hyp_",
)
_FREQUENCY_TOKENS = ("fourier", "spectral", "wavelet", "frequency")
_NORM_TOKENS = ("norm", "rms", "layernorm", "stabilize")
_MOE_TOKENS = ("moe", "expert")
_RESIDUAL_TOKENS = ("residual", "skip", "transformer", "block")
_ROLE_PREFIX = "role:"


def canonical_slot_key(slot_key: str) -> str:
    return _INSTANCE_RE.sub("", str(slot_key or ""))


def _tokens(text: str) -> set[str]:
    return {part for part in re.split(r"[^a-z0-9]+", text.lower()) if part}


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in needles)


def _score(*values: bool) -> float:
    return round(sum(1.0 for value in values if value) / max(len(values), 1), 4)


def _template_family(name: str) -> str:
    lower = name.lower()
    if _has_any(lower, _ROUTING_TOKENS):
        return "routing"
    if _has_any(lower, _MEMORY_TOKENS):
        return "memory"
    if _has_any(lower, _ATTENTION_TOKENS):
        return "attention"
    if _has_any(lower, _SSM_TOKENS):
        return "state_space"
    if _has_any(lower, _MOE_TOKENS):
        return "moe"
    if _has_any(lower, _MATH_SPACE_TOKENS):
        return "math_space"
    if "mlp" in lower or "ffn" in lower:
        return "feedforward"
    return "generic"


def _template_topology(name: str, slot_count: int) -> str:
    lower = name.lower()
    if "n_way" in lower or "three_way" in lower or "multi" in lower:
        return "parallel_dag"
    if _has_any(lower, ("router", "routing", "lane", "split", "triplet")):
        return "branched_dag"
    if slot_count >= 4:
        return "slot_dag"
    if "stack" in lower or "recursive" in lower:
        return "chain_stack"
    return "chain"


def _receptive_field(name: str) -> str:
    lower = name.lower()
    if _has_any(lower, ("local", "window", "conv")):
        if _has_any(lower, ("global", "attn", "attention", "retrieval", "memory")):
            return "hybrid_local_global"
        return "local"
    if _has_any(lower, ("attn", "attention", "retrieval", "memory", "global")):
        return "global"
    if _has_any(lower, ("ssm", "scan", "state", "retention")):
        return "recurrent_global"
    return "unspecified"


def _preferred_basis(name: str) -> str:
    lower = name.lower()
    if "fourier" in lower or "spectral" in lower or "frequency" in lower:
        return "fourier"
    if "wavelet" in lower:
        return "wavelet"
    if "hadamard" in lower:
        return "hadamard"
    if _has_any(lower, ("conv", "window", "local")):
        return "locality"
    if _has_any(lower, ("attn", "attention", "retrieval", "relation")):
        return "content"
    return "identity"


def _linearity_class(text: str) -> str:
    lower = text.lower()
    if _has_any(lower, ("mask", "topk", "router", "gate", "routing")):
        return "piecewise_discrete"
    if _has_any(
        lower,
        ("attn", "attention", "bilinear", "qk", "relation", "retrieval", "memory"),
    ):
        return "bilinear"
    if _has_any(lower, ("gelu", "swiglu", "sigmoid", "exp", "softmax", "ffn", "mlp")):
        return "transcendental"
    if _has_any(lower, ("tropical", "padic", "clifford", "moe", "conditional")):
        return "nonlinear"
    if _has_any(
        lower,
        (
            "linear",
            "proj",
            "conv",
            "scan",
            "state",
            "retention",
            "norm",
            "residual",
            "merge",
            "bottleneck",
        ),
    ):
        return "linear"
    return "unknown"


def _equivariance_class(text: str) -> str:
    lower = text.lower()
    if _has_any(lower, ("conv", "local", "window", "scan", "state", "retention")):
        return "translation"
    if _has_any(lower, ("attn", "attention", "set", "relation", "graph")):
        return "permutation"
    if _has_any(lower, ("norm", "rms", "scale")):
        return "scale"
    if _has_any(lower, ("rope", "rotary")):
        return "rotation"
    return "none"


def _activation_density(text: str) -> str:
    lower = text.lower()
    if _has_any(lower, ("topk", "sparse")):
        return "k_sparse"
    if _has_any(lower, ("gate", "router", "moe", "routing")):
        return "gated"
    return "dense"


def _sparsity_pattern(text: str) -> str:
    lower = text.lower()
    if "topk" in lower:
        return "top_k"
    if _has_any(lower, ("local", "window", "conv", "block")):
        return "structured"
    if _has_any(lower, ("sparse", "moe", "router", "gate")):
        return "learned_structured"
    return "dense"


def _smoothness_class(text: str) -> str:
    lower = text.lower()
    if _has_any(lower, ("topk", "mask", "argmax", "route", "router")):
        return "piecewise_discrete"
    if _has_any(lower, ("relu", "gelu", "swiglu", "gate")):
        return "piecewise_smooth"
    if _has_any(lower, ("softmax", "sigmoid", "exp", "norm", "attn", "attention")):
        return "c_infinity"
    return "c1"


def _math_axis_properties(
    prefix: str, text: str, *, slot_count: int = 0
) -> dict[str, Any]:
    lower = text.lower()
    has_attention = _has_any(lower, _ATTENTION_TOKENS)
    has_ssm = _has_any(lower, _SSM_TOKENS)
    has_conv = _has_any(lower, _CONV_TOKENS)
    has_routing = _has_any(lower, _ROUTING_TOKENS)
    has_compression = _has_any(lower, _COMPRESSION_TOKENS)
    has_memory = _has_any(lower, _MEMORY_TOKENS)
    has_math_space = _has_any(lower, _MATH_SPACE_TOKENS)
    has_frequency = _has_any(lower, _FREQUENCY_TOKENS)
    has_norm = _has_any(lower, _NORM_TOKENS)
    has_moe = _has_any(lower, _MOE_TOKENS)
    has_residual = _has_any(lower, _RESIDUAL_TOKENS)
    is_sparse = _has_any(lower, ("sparse", "topk", "mask", "local", "window"))
    is_gated = _has_any(lower, ("gate", "gated", "router", "route", "moe"))
    is_discrete = _has_any(lower, ("topk", "argmax", "mask", "router"))
    preferred_basis = _preferred_basis(text)
    receptive_field = _receptive_field(text)

    low_pass = (
        (0.45 if has_conv else 0.0)
        + (0.35 if has_ssm else 0.0)
        + (0.25 if has_compression else 0.0)
        + (0.20 if has_frequency else 0.0)
        - (0.10 if has_routing else 0.0)
    )
    lipschitz = (
        1.0
        + (0.55 if has_attention else 0.0)
        + (0.45 if has_routing else 0.0)
        + (0.35 if has_math_space else 0.0)
        + (0.25 if has_moe else 0.0)
        - (0.20 if has_norm else 0.0)
        - (0.10 if has_compression else 0.0)
    )
    cond = (
        1.0
        + (0.35 * max(slot_count - 1, 0))
        + (0.70 if has_routing else 0.0)
        + (0.55 if has_math_space else 0.0)
        + (0.35 if has_attention else 0.0)
        - (0.30 if has_norm else 0.0)
    )
    grad_vanish = (
        0.10
        + (0.22 if has_compression else 0.0)
        + (0.18 if has_ssm else 0.0)
        + (0.15 if "sigmoid" in lower else 0.0)
        - (0.08 if has_residual else 0.0)
    )
    grad_explode = (
        0.10
        + (0.22 if has_attention else 0.0)
        + (0.22 if has_routing else 0.0)
        + (0.20 if has_math_space else 0.0)
        - (0.10 if has_norm else 0.0)
    )
    fp16_stable = (
        0.72
        - (0.18 if has_math_space else 0.0)
        - (0.15 if has_routing else 0.0)
        - (0.10 if has_attention else 0.0)
        + (0.12 if has_norm else 0.0)
        + (0.06 if has_conv else 0.0)
    )
    effective_rank = (
        0.50
        + (0.20 if has_attention else 0.0)
        + (0.16 if has_memory else 0.0)
        + (0.12 if has_math_space else 0.0)
        - (0.18 if has_compression else 0.0)
        - (0.12 if is_sparse else 0.0)
    )

    return {
        f"{prefix}_algebraic_linearity_class": _linearity_class(text),
        f"{prefix}_algebraic_equivariance": _equivariance_class(text),
        f"{prefix}_algebraic_idempotent": int(
            _has_any(lower, ("mask", "norm", "topk"))
        ),
        f"{prefix}_algebraic_involutive": int(
            _has_any(lower, ("rope", "rotary", "fft"))
        ),
        f"{prefix}_algebraic_commutes_with_norm": int(
            _has_any(lower, ("linear", "proj", "conv", "attention", "scan"))
        ),
        f"{prefix}_spectral_preferred_basis": preferred_basis,
        f"{prefix}_spectral_low_pass_strength": round(min(1.0, max(low_pass, 0.0)), 4),
        f"{prefix}_spectral_diagonalizable_prior": round(
            min(
                1.0,
                0.35 + (0.30 if has_frequency else 0.0) + (0.20 if has_conv else 0.0),
            ),
            4,
        ),
        f"{prefix}_spectral_radius_init_prior": round(
            min(3.0, max(0.1, lipschitz + (0.25 if has_ssm else 0.0))),
            4,
        ),
        f"{prefix}_geometric_lipschitz_prior": round(min(4.0, max(0.1, lipschitz)), 4),
        f"{prefix}_geometric_jacobian_rank_prior": round(
            min(1.0, max(0.05, effective_rank)),
            4,
        ),
        f"{prefix}_geometric_jacobian_cond_prior": round(min(6.0, max(0.1, cond)), 4),
        f"{prefix}_geometric_receptive_field": receptive_field,
        f"{prefix}_geometric_curvature_prior": round(
            min(
                1.0,
                0.08
                + (0.22 if has_math_space else 0.0)
                + (0.18 if has_attention else 0.0)
                + (0.14 if is_gated else 0.0),
            ),
            4,
        ),
        f"{prefix}_dynamical_causal": int(
            has_ssm or "causal" in lower or "mask" in lower
        ),
        f"{prefix}_dynamical_memory_length_class": (
            "O(L^2)"
            if has_attention
            and not _has_any(lower, ("linear_attention", "local", "window"))
            else "O(L)"
            if has_ssm or has_memory or has_conv
            else "O(1)"
        ),
        f"{prefix}_dynamical_contraction_factor_prior": round(
            min(
                1.0,
                max(
                    0.0,
                    0.55
                    + (0.18 if has_norm else 0.0)
                    + (0.15 if has_compression else 0.0)
                    - (0.14 if has_routing else 0.0),
                ),
            ),
            4,
        ),
        f"{prefix}_dynamical_exponential_decay_rate_prior": round(
            min(
                1.0,
                (0.45 if has_ssm else 0.0)
                + (0.25 if has_memory else 0.0)
                + (0.20 if has_conv else 0.0),
            ),
            4,
        ),
        f"{prefix}_dynamical_has_state": int(has_ssm or has_memory),
        f"{prefix}_numerical_hessian_conditioning_prior": round(
            min(6.0, max(0.1, cond + (0.4 if is_discrete else 0.0))), 4
        ),
        f"{prefix}_numerical_grad_vanish_propensity": round(
            min(1.0, max(0.0, grad_vanish)), 4
        ),
        f"{prefix}_numerical_grad_explode_propensity": round(
            min(1.0, max(0.0, grad_explode)), 4
        ),
        f"{prefix}_numerical_fp16_stable_prior": round(
            min(1.0, max(0.0, fp16_stable)), 4
        ),
        f"{prefix}_numerical_init_sensitivity": round(
            min(
                1.0,
                0.16
                + (0.18 if has_routing else 0.0)
                + (0.16 if has_attention else 0.0)
                + (0.14 if has_math_space else 0.0)
                + (0.04 * slot_count),
            ),
            4,
        ),
        f"{prefix}_activation_density_class": _activation_density(text),
        f"{prefix}_activation_sparsity_pattern": _sparsity_pattern(text),
        f"{prefix}_activation_effective_rank_prior": round(
            min(1.0, max(0.05, effective_rank)),
            4,
        ),
        f"{prefix}_expressivity_universal_approx_class": (
            "continuous"
            if has_attention or has_moe or has_math_space
            else "polynomial"
            if _linearity_class(text) not in {"linear", "unknown"}
            else "none"
        ),
        f"{prefix}_expressivity_depth_required_for_xor_prior": int(
            1 if is_gated or has_moe else 2 if has_attention or has_math_space else 3
        ),
        f"{prefix}_expressivity_params_for_identity_prior": round(
            1.0
            + (0.35 if has_residual else 0.0)
            + (0.25 if has_norm else 0.0)
            - (0.25 if has_compression else 0.0),
            4,
        ),
        f"{prefix}_composition_parallel_safe": int(not has_ssm and not has_memory),
        f"{prefix}_composition_residual_safe": int(
            has_norm or has_residual or not is_discrete
        ),
        f"{prefix}_composition_norm_required": (
            "pre_post"
            if has_routing or has_math_space
            else "pre"
            if has_attention or has_ssm or has_memory
            else "none"
        ),
        f"{prefix}_composition_max_stack_depth_before_collapse": int(
            2 if has_compression else 3 if has_routing or has_math_space else 6
        ),
        f"{prefix}_differentiability_smoothness_class": _smoothness_class(text),
        f"{prefix}_differentiability_needs_surrogate": (
            "straight_through"
            if is_discrete
            else "gumbel"
            if has_routing or has_moe
            else "none"
        ),
    }


def template_descriptive_properties(
    template_name: str,
    *,
    slot_count: int = 0,
) -> dict[str, Any]:
    name = str(template_name or "").strip()
    lower = name.lower()
    slot_count = max(int(slot_count or 0), 0)

    has_attention = _has_any(lower, _ATTENTION_TOKENS)
    has_ssm = _has_any(lower, _SSM_TOKENS)
    has_conv = _has_any(lower, _CONV_TOKENS)
    has_routing = _has_any(lower, _ROUTING_TOKENS)
    has_compression = _has_any(lower, _COMPRESSION_TOKENS)
    has_memory = _has_any(lower, _MEMORY_TOKENS)
    has_math_space = _has_any(lower, _MATH_SPACE_TOKENS)
    has_frequency = _has_any(lower, _FREQUENCY_TOKENS)
    has_norm = _has_any(lower, _NORM_TOKENS)
    has_moe = _has_any(lower, _MOE_TOKENS)
    has_residual = _has_any(lower, _RESIDUAL_TOKENS)
    has_parallel_paths = _has_any(lower, ("multi", "lane", "split", "triplet", "n_way"))
    has_state = has_ssm or has_memory

    branch_factor = 1
    if has_parallel_paths:
        branch_factor += 1
    if "three_way" in lower or "triplet" in lower:
        branch_factor = max(branch_factor, 3)
    if "n_way" in lower:
        branch_factor = max(branch_factor, 4)
    if "dual" in lower:
        branch_factor = max(branch_factor, 2)
    if "multi" in lower or "lane" in lower:
        branch_factor = max(branch_factor, min(4, max(2, slot_count // 2)))

    complexity = (
        1.0
        + (0.45 * slot_count)
        + (0.70 * branch_factor)
        + (0.85 if has_routing else 0.0)
        + (0.65 if has_attention else 0.0)
        + (0.55 if has_ssm else 0.0)
        + (0.50 if has_math_space else 0.0)
        + (0.45 if has_memory else 0.0)
        + (0.35 if has_compression else 0.0)
    )
    risk = (
        0.10
        + (0.13 * slot_count)
        + (0.18 if has_routing else 0.0)
        + (0.16 if has_math_space else 0.0)
        + (0.12 if branch_factor >= 3 else 0.0)
        - (0.08 if has_norm else 0.0)
        - (0.05 if has_residual else 0.0)
    )
    trainability = (
        0.55
        + (0.08 if has_norm else 0.0)
        + (0.07 if has_residual else 0.0)
        + (0.05 if has_attention else 0.0)
        - (0.04 * max(slot_count - 2, 0))
        - (0.06 if has_math_space else 0.0)
        - (0.06 if has_routing and branch_factor >= 3 else 0.0)
    )

    return {
        "template_family": _template_family(name),
        "template_topology": _template_topology(name, slot_count),
        "template_receptive_field": _receptive_field(name),
        "template_preferred_basis": _preferred_basis(name),
        "template_has_attention": int(has_attention),
        "template_has_ssm": int(has_ssm),
        "template_has_conv": int(has_conv),
        "template_has_routing": int(has_routing),
        "template_has_compression": int(has_compression),
        "template_has_memory": int(has_memory),
        "template_has_math_space": int(has_math_space),
        "template_has_frequency_domain": int(has_frequency),
        "template_has_norm": int(has_norm),
        "template_has_moe": int(has_moe),
        "template_has_residual": int(has_residual),
        "template_has_parallel_paths": int(has_parallel_paths),
        "template_has_state": int(has_state),
        "template_est_branch_factor": int(branch_factor),
        "template_est_parallel_paths": int(max(branch_factor, 1)),
        "template_slot_density": round(slot_count / max(branch_factor, 1), 4),
        "template_structural_complexity": round(complexity, 4),
        "template_routing_intensity": round(
            min(1.0, (0.35 if has_routing else 0.0) + (0.10 * branch_factor)),
            4,
        ),
        "template_memory_intensity": _score(has_memory, has_ssm, "retrieval" in lower),
        "template_compression_intensity": _score(
            has_compression, "bottleneck" in lower, "merge" in lower, "sparse" in lower
        ),
        "template_local_global_mix": _score(has_conv, has_attention or has_memory),
        "template_math_space_intensity": _score(
            has_math_space, has_frequency, "tropical" in lower or "padic" in lower
        ),
        "template_stabilization_need": round(min(1.0, max(risk, 0.0)), 4),
        "template_trainability_prior": round(min(1.0, max(trainability, 0.0)), 4),
        "template_novelty_prior": round(
            min(
                1.0,
                0.12
                + (0.16 if has_math_space else 0.0)
                + (0.12 if has_routing else 0.0)
                + (0.10 if has_memory else 0.0)
                + (0.08 if has_parallel_paths else 0.0),
            ),
            4,
        ),
        "template_expected_context_span": {
            "local": 1,
            "unspecified": 2,
            "hybrid_local_global": 3,
            "global": 4,
            "recurrent_global": 4,
        }.get(_receptive_field(name), 2),
        **_math_axis_properties("template", name, slot_count=slot_count),
    }


def _slot_role_from_classes(slot_classes: list[str]) -> str:
    for item in slot_classes:
        if item.startswith(_ROLE_PREFIX):
            return item[len(_ROLE_PREFIX) :].strip() or "role"
    return ""


def _slot_role_family(slot_key: str, slot_classes: list[str]) -> str:
    role = _slot_role_from_classes(slot_classes)
    text = " ".join([slot_key, role, *slot_classes]).lower()
    if _has_any(text, ("trunk", "default_path", "stem")):
        return "trunk"
    if _has_any(text, ("router", "route", "controller", "gate", "difficulty")):
        return "routing"
    if _has_any(text, ("retrieval", "read", "memory", "relation")):
        return "memory_retrieval"
    if _has_any(text, ("write", "delta", "state")):
        return "memory_write"
    if _has_any(text, ("merge", "blend", "post_merge", "stabilize")):
        return "merge_stabilize"
    if _has_any(text, ("compress", "bottleneck", "sparse", "topk")):
        return "compression"
    if _has_any(text, ("norm",)):
        return "normalization"
    return "motif"


def slot_descriptive_properties(
    slot_key: str,
    *,
    template_name: str = "",
    slot_index: int = 0,
    slot_count: int = 0,
    slot_classes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    canonical = canonical_slot_key(slot_key)
    template_name = str(template_name or canonical.split(".", 1)[0]).strip()
    slot_index = max(int(slot_index or 0), 0)
    slot_count = max(int(slot_count or 0), 0)
    classes = [str(item) for item in (slot_classes or []) if item is not None]
    class_text = " ".join(classes).lower()
    math_text = " ".join([canonical, template_name, *classes])
    role = _slot_role_from_classes(classes)
    role_family = _slot_role_family(canonical, classes)
    position = slot_index / max(slot_count - 1, 1) if slot_count > 1 else 0.0
    allowed_class_count = len(set(classes))
    is_wildcard = any(item in {"*", "any", "wildcard"} for item in _tokens(class_text))

    accepts_attention = _has_any(class_text, _ATTENTION_TOKENS)
    accepts_ssm = _has_any(class_text, _SSM_TOKENS)
    accepts_routing = _has_any(class_text, _ROUTING_TOKENS)
    accepts_compression = _has_any(class_text, _COMPRESSION_TOKENS)
    accepts_memory = _has_any(class_text, _MEMORY_TOKENS)
    accepts_norm = _has_any(class_text, _NORM_TOKENS)
    accepts_math_space = _has_any(class_text, _MATH_SPACE_TOKENS)

    search_width = allowed_class_count + (3 if is_wildcard else 0)
    pressure = (
        0.15
        + (0.08 * search_width)
        + (0.18 if role_family in {"routing", "memory_retrieval"} else 0.0)
        + (0.12 if accepts_math_space else 0.0)
        + (0.08 if position > 0.66 and role_family != "merge_stabilize" else 0.0)
    )

    return {
        "slot_key_canonical": canonical,
        "slot_role": role,
        "slot_role_family": role_family,
        "slot_position_fraction": round(position, 4),
        "slot_is_early": int(position <= 0.33),
        "slot_is_middle": int(0.33 < position <= 0.66),
        "slot_is_late": int(position > 0.66),
        "slot_allowed_class_count": int(allowed_class_count),
        "slot_is_wildcard": int(is_wildcard),
        "slot_accepts_attention": int(accepts_attention),
        "slot_accepts_ssm": int(accepts_ssm),
        "slot_accepts_routing": int(accepts_routing),
        "slot_accepts_compression": int(accepts_compression),
        "slot_accepts_memory": int(accepts_memory),
        "slot_accepts_norm": int(accepts_norm),
        "slot_accepts_math_space": int(accepts_math_space),
        "slot_search_width_prior": int(search_width),
        "slot_search_entropy_prior": round(math.log1p(max(search_width, 0)), 4),
        "slot_pressure_prior": round(min(1.0, pressure), 4),
        "slot_expected_contract": (
            "stabilizer"
            if role_family in {"merge_stabilize", "normalization"}
            else "routing_control"
            if role_family == "routing"
            else "memory_io"
            if role_family in {"memory_retrieval", "memory_write"}
            else "feature_transform"
        ),
        "slot_template_family": template_descriptive_properties(
            template_name, slot_count=slot_count
        )["template_family"],
        **_math_axis_properties("slot", math_text, slot_count=slot_count),
    }


TEMPLATE_PROPERTY_COLUMNS = tuple(
    template_descriptive_properties("sample_router", slot_count=2).keys()
)
SLOT_PROPERTY_COLUMNS = tuple(
    slot_descriptive_properties(
        "sample.slot0",
        template_name="sample",
        slot_index=0,
        slot_count=1,
        slot_classes=["role:router", "attention"],
    ).keys()
)


def _op_text_from_metadata(op_name: str, metadata: dict[str, Any]) -> str:
    return " ".join(
        str(part)
        for part in (
            op_name,
            metadata.get("category", ""),
            metadata.get("shape_rule", ""),
            metadata.get("description", ""),
            metadata.get("algebraic_space", ""),
            metadata.get("binding_range_class", ""),
        )
        if part
    )


def op_descriptive_properties(
    op_name: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return wide static math/property columns for one primitive op."""

    metadata = dict(metadata or {})
    name = str(op_name or "").strip()
    text = _op_text_from_metadata(name, metadata)
    lower = text.lower()
    category = str(metadata.get("category") or "unknown")
    binding_range = str(metadata.get("binding_range_class") or "none")
    has_params = bool(metadata.get("has_params"))
    preserves_gradient = bool(metadata.get("preserves_gradient", True))
    numerically_risky = bool(metadata.get("numerically_risky"))
    param_formula = str(metadata.get("param_formula") or "0")
    algebraic_space = str(metadata.get("algebraic_space") or "unknown")
    byte_safe = bool(metadata.get("byte_safe", True))
    standalone = bool(metadata.get("standalone", True))
    min_layer_depth = int(metadata.get("min_layer_depth") or 0)

    math_props = _math_axis_properties("op", text, slot_count=0)
    is_functional = category == "functional" or _has_any(
        lower, ("lambda", "compose", "map", "fold", "reduce", "function")
    )
    symbolic_affinity = _score(
        is_functional,
        category in {"math_space", "functional"},
        algebraic_space not in {"euclidean", "unknown", ""},
        _has_any(lower, ("discrete", "program", "interpreter", "symbolic", "tree")),
    )
    alternative_math_affinity = _score(
        category in {"math_space", "functional", "frequency"},
        algebraic_space not in {"euclidean", "unknown", ""},
        _has_any(lower, ("tropical", "padic", "clifford", "fourier", "wavelet")),
        is_functional,
    )
    lambda_affinity = _score(
        is_functional,
        _has_any(lower, ("lambda", "compose", "map", "fold", "reduce")),
        category in {"functional", "structural"},
        not has_params,
    )

    return {
        "op_category": category,
        "op_shape_rule": str(metadata.get("shape_rule") or ""),
        "op_n_inputs": int(metadata.get("n_inputs") or 0),
        "op_has_params": int(has_params),
        "op_param_formula": param_formula,
        "op_preserves_gradient_declared": int(preserves_gradient),
        "op_numerically_risky_declared": int(numerically_risky),
        "op_standalone": int(standalone),
        "op_byte_safe": int(byte_safe),
        "op_min_layer_depth": min_layer_depth,
        "op_algebraic_space": algebraic_space,
        "op_binding_range_class": binding_range,
        "op_description": str(metadata.get("description") or ""),
        "op_is_parameterized": int(has_params),
        "op_is_stateless": int(binding_range == "none"),
        "op_symbolic_affinity": symbolic_affinity,
        "op_lambda_calculus_affinity": lambda_affinity,
        "op_alternative_math_affinity": alternative_math_affinity,
        "op_empirical_probe_needed": int(
            numerically_risky
            or not preserves_gradient
            or category in {"math_space", "functional", "mixing", "sequence"}
            or math_props["op_differentiability_needs_surrogate"] != "none"
        ),
        **math_props,
    }


OP_PROPERTY_COLUMNS = tuple(
    op_descriptive_properties(
        "sample_lambda_map",
        metadata={
            "category": "functional",
            "n_inputs": 1,
            "shape_rule": "identity",
            "description": "Lambda calculus style map/compose candidate",
            "has_params": False,
            "preserves_gradient": True,
            "numerically_risky": False,
            "binding_range_class": "none",
        },
    ).keys()
)


def alternative_math_candidate_properties(name: str) -> dict[str, Any]:
    """Describe unimplemented math spaces to compare against observed gaps."""

    lower = name.lower()
    if "lambda" in lower:
        return {
            "candidate_name": name,
            "candidate_family": "symbolic_functional",
            "candidate_core_abstraction": "lambda_terms",
            "candidate_linearity_class": "higher_order_discrete",
            "candidate_preferred_basis": "program_structure",
            "candidate_memory_length_class": "O(L)",
            "candidate_smoothness_class": "piecewise_discrete",
            "candidate_needs_surrogate": "straight_through_or_relaxation",
            "candidate_expected_strength": "compositional_binding",
            "candidate_expected_risk": "gradient_surrogate_and_search_space",
            "candidate_best_entry_slot": "controller_or_program_interpreter",
            "candidate_probe_priority": 0.82,
        }
    if "combinator" in lower:
        return {
            "candidate_name": name,
            "candidate_family": "symbolic_functional",
            "candidate_core_abstraction": "combinatory_logic",
            "candidate_linearity_class": "higher_order_discrete",
            "candidate_preferred_basis": "program_structure",
            "candidate_memory_length_class": "O(L)",
            "candidate_smoothness_class": "piecewise_discrete",
            "candidate_needs_surrogate": "straight_through_or_relaxation",
            "candidate_expected_strength": "variable_free_composition",
            "candidate_expected_risk": "routing_collapse",
            "candidate_best_entry_slot": "routing_or_merge_slot",
            "candidate_probe_priority": 0.74,
        }
    if "category" in lower:
        return {
            "candidate_name": name,
            "candidate_family": "algebraic_structure",
            "candidate_core_abstraction": "morphisms_functors",
            "candidate_linearity_class": "compositional",
            "candidate_preferred_basis": "graph_structure",
            "candidate_memory_length_class": "O(L)",
            "candidate_smoothness_class": "c1",
            "candidate_needs_surrogate": "none_or_relaxed",
            "candidate_expected_strength": "typed_composition",
            "candidate_expected_risk": "too_abstract_for_low_level_ops",
            "candidate_best_entry_slot": "template_composer",
            "candidate_probe_priority": 0.55,
        }
    if "boolean" in lower:
        return {
            "candidate_name": name,
            "candidate_family": "discrete_logic",
            "candidate_core_abstraction": "boolean_algebra",
            "candidate_linearity_class": "piecewise_discrete",
            "candidate_preferred_basis": "logic",
            "candidate_memory_length_class": "O(1)",
            "candidate_smoothness_class": "piecewise_discrete",
            "candidate_needs_surrogate": "straight_through",
            "candidate_expected_strength": "control_and_gating",
            "candidate_expected_risk": "non_smooth_gradients",
            "candidate_best_entry_slot": "router",
            "candidate_probe_priority": 0.62,
        }
    return {
        "candidate_name": name,
        "candidate_family": "unknown",
        "candidate_core_abstraction": "unknown",
        "candidate_linearity_class": "unknown",
        "candidate_preferred_basis": "unknown",
        "candidate_memory_length_class": "unknown",
        "candidate_smoothness_class": "unknown",
        "candidate_needs_surrogate": "unknown",
        "candidate_expected_strength": "unknown",
        "candidate_expected_risk": "unknown",
        "candidate_best_entry_slot": "unknown",
        "candidate_probe_priority": 0.0,
    }


ALTERNATIVE_MATH_CANDIDATE_COLUMNS = tuple(
    alternative_math_candidate_properties("lambda_calculus").keys()
)
