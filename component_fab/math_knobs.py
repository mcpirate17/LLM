"""Canonical math-knob definitions and axis parsing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MathKnob:
    knob_id: str
    family: str
    axes: dict[str, Any]
    cost_class: str
    rationale: str


DEFAULT_MATH_KNOBS: tuple[MathKnob, ...] = (
    MathKnob(
        knob_id="calculus_finite_difference",
        family="calculus",
        axes={
            "op_calculus_operator": "causal_finite_difference_integral",
        },
        cost_class="low",
        rationale="causal derivative and integral features",
    ),
    MathKnob(
        knob_id="linear_algebra_low_rank",
        family="linear_algebra",
        axes={
            "op_linear_algebra_structure": "low_rank_factorized",
        },
        cost_class="low",
        rationale="low-rank factorized adapter",
    ),
    MathKnob(
        knob_id="sparse_matrix_banded",
        family="sparse_matrix",
        axes={
            "op_sparse_matrix_pattern": "causal_banded",
        },
        cost_class="low",
        rationale="causal banded sparse matrix adapter",
    ),
    MathKnob(
        knob_id="kernel_random_features",
        family="kernel_methods",
        axes={
            "op_kernel_feature_map": "positive_random_features",
        },
        cost_class="low",
        rationale="positive random-feature causal kernel mixer",
    ),
    MathKnob(
        knob_id="multiscale_wavelet",
        family="multiscale",
        axes={
            "op_multiscale_transform": "causal_haar",
        },
        cost_class="low",
        rationale="causal Haar-style multiscale averaging and detail mixing",
    ),
    MathKnob(
        knob_id="graph_laplacian_diffusion",
        family="graph_diffusion",
        axes={
            "op_graph_topology": "causal_path_laplacian",
        },
        cost_class="low",
        rationale="causal path-graph Laplacian diffusion",
    ),
    MathKnob(
        knob_id="lambda_functional_blend",
        family="lambda_functional",
        axes={
            "op_lambda_transform": "learned_functional_blend",
            "op_lambda_gate": "content",
            "op_lambda_basis": "identity",
        },
        cost_class="low",
        rationale="identity-biased lambda-gated functional blend",
    ),
    MathKnob(
        knob_id="tropical_knob",
        family="tropical",
        axes={
            "op_tropical_adapter": "maxplus_read",
        },
        cost_class="medium",
        rationale="stackable max-plus tropical read adapter",
    ),
    MathKnob(
        knob_id="padic_knob",
        family="padic",
        axes={
            "op_padic_adapter": "ultrametric_projection",
        },
        cost_class="medium",
        rationale="stackable ultrametric p-adic projection adapter",
    ),
    MathKnob(
        knob_id="clifford_knob",
        family="clifford",
        axes={
            "op_clifford_adapter": "geometric_product",
        },
        cost_class="medium",
        rationale="stackable Clifford geometric-product adapter",
    ),
    MathKnob(
        knob_id="hyperbolic_knob",
        family="hyperbolic",
        axes={
            "op_hyperbolic_adapter": "poincare_projection",
        },
        cost_class="medium",
        rationale="stackable Poincare-ball hyperbolic adapter",
    ),
)

AUTO_DEEPENING_MATH_KNOBS: tuple[MathKnob, ...] = (
    MathKnob(
        knob_id="calculus_causal_gradient",
        family="calculus",
        axes={
            "op_calculus_operator": "causal_gradient",
            "op_math_deepening_source": "calculus_finite_difference",
        },
        cost_class="low",
        rationale="auto-deepened calculus sibling: causal gradient features",
    ),
    MathKnob(
        knob_id="calculus_laplacian",
        family="calculus",
        axes={
            "op_calculus_operator": "causal_laplacian",
            "op_math_deepening_source": "calculus_finite_difference",
        },
        cost_class="low",
        rationale="auto-deepened calculus sibling: causal second-difference/Laplacian features",
    ),
    MathKnob(
        knob_id="calculus_lie_derivative",
        family="calculus",
        axes={
            "op_calculus_operator": "lie_derivative_along_flow",
            "op_math_deepening_source": "calculus_finite_difference",
        },
        cost_class="low",
        rationale="auto-deepened calculus sibling: derivative along learned sequence flow",
    ),
    MathKnob(
        knob_id="linear_algebra_block_low_rank",
        family="linear_algebra",
        axes={
            "op_linear_algebra_structure": "block_low_rank_factorized",
            "op_math_deepening_source": "linear_algebra_low_rank",
        },
        cost_class="low",
        rationale="auto-deepened linear-algebra sibling: block low-rank adapter",
    ),
    MathKnob(
        knob_id="sparse_matrix_dilated_banded",
        family="sparse_matrix",
        axes={
            "op_sparse_matrix_pattern": "causal_dilated_banded",
            "op_math_deepening_source": "sparse_matrix_banded",
        },
        cost_class="low",
        rationale="auto-deepened sparse-matrix sibling: dilated causal band adapter",
    ),
    MathKnob(
        knob_id="kernel_nystrom_features",
        family="kernel_methods",
        axes={
            "op_kernel_feature_map": "nystrom_landmarks",
            "op_math_deepening_source": "kernel_random_features",
        },
        cost_class="low",
        rationale="auto-deepened kernel sibling: Nystrom-style landmark features",
    ),
    MathKnob(
        knob_id="kernel_fourier_features",
        family="kernel_methods",
        axes={
            "op_kernel_feature_map": "orthogonal_fourier_features",
            "op_math_deepening_source": "kernel_random_features",
        },
        cost_class="low",
        rationale="auto-deepened kernel sibling: orthogonal Fourier random features",
    ),
    MathKnob(
        knob_id="multiscale_dyadic_diff",
        family="multiscale",
        axes={
            "op_multiscale_transform": "dyadic_diff",
            "op_math_deepening_source": "multiscale_wavelet",
        },
        cost_class="low",
        rationale="auto-deepened multiscale sibling: dyadic causal differences",
    ),
    MathKnob(
        knob_id="multiscale_laplacian_pyramid",
        family="multiscale",
        axes={
            "op_multiscale_transform": "laplacian_pyramid",
            "op_math_deepening_source": "multiscale_wavelet",
        },
        cost_class="low",
        rationale="auto-deepened multiscale sibling: causal Laplacian pyramid",
    ),
    MathKnob(
        knob_id="graph_diffusion_multihop",
        family="graph_diffusion",
        axes={
            "op_graph_topology": "causal_multihop_laplacian",
            "op_math_deepening_source": "graph_laplacian_diffusion",
        },
        cost_class="low",
        rationale="auto-deepened graph sibling: multi-hop causal path diffusion",
    ),
    MathKnob(
        knob_id="graph_diffusion_learned_path",
        family="graph_diffusion",
        axes={
            "op_graph_topology": "learned_path_laplacian",
            "op_math_deepening_source": "graph_laplacian_diffusion",
        },
        cost_class="low",
        rationale="auto-deepened graph sibling: learned path-Laplacian weighting",
    ),
    MathKnob(
        knob_id="clifford_rotor_sandwich",
        family="clifford",
        axes={
            "op_clifford_adapter": "rotor_sandwich",
            "op_math_deepening_source": "clifford_knob",
        },
        cost_class="low",
        rationale="auto-deepened Clifford sibling: pointwise rotor-sandwich geometric product",
    ),
    MathKnob(
        knob_id="lambda_functional_token_basis",
        family="lambda_functional",
        axes={
            "op_lambda_transform": "learned_functional_blend",
            "op_lambda_gate": "content",
            "op_lambda_basis": "token",
            "op_math_deepening_source": "lambda_functional_blend",
        },
        cost_class="low",
        rationale="auto-deepened lambda-functional sibling: token-basis functional blend",
    ),
)

KNOB_ID_BY_FAMILY: dict[str, str] = {
    **{knob.family: knob.knob_id for knob in DEFAULT_MATH_KNOBS},
    # Phase-2 knobs are generated by axis_variants today, but dispatch and
    # validation still need the same family fallback when op_math_knobs is absent.
    "information_geometry": "info_geom_fisher",
    "spectral_graph": "spectral_chebyshev",
    "tensor_decomp": "tensor_tucker",
}


def math_knobs_from_axes(math_axes: Mapping[str, Any]) -> tuple[str, ...]:
    """Normalize explicit or family-implied math knobs from a spec axis map."""
    raw = math_axes.get("op_math_knobs")
    if raw is None:
        family = str(math_axes.get("op_math_family") or "")
        knob = KNOB_ID_BY_FAMILY.get(family)
        return (knob,) if knob is not None else ()
    if isinstance(raw, str):
        return tuple(part.strip() for part in raw.split("+") if part.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(str(part) for part in raw if str(part))
    return ()
