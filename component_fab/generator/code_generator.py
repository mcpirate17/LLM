"""Dispatcher — ProposalSpec -> runnable nn.Module.

Heavy primitive builders still live here, but ordered dispatch mechanics,
block-slot lookup, and invention-mechanism dispatch live under
``component_fab.generator.dispatch`` so registry behavior is auditable and
unknown block slots fail loud.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from component_fab.generator.dispatch import (
    DispatchRule,
    build_block_slot_factory,
    dispatch_first,
    known_partner_kinds,
    slot_name_for_partner_kind,
)
from component_fab.generator.dispatch.invention import (
    NativeParityEvidenceError as NativeParityEvidenceError,
    dispatch_invention_mechanism,
)
from component_fab.math_knobs import math_knobs_from_axes

from ..harness.primitives import SwiGLU
from ..proposer.nas_bridge import SOURCE_NAS, load_cached_graph_json
from ..proposer.spec_generator import ProposalSpec
from .block_templates import (
    BLOCK_TEMPLATES,
    AttnSpectralFilterBlock,
    GatedParallelBlock,
    GraphAttentionBlock,
    HeteroMoEBlock,
    HyperbolicBridgeBlock,
    LatentCompressBlock,
    LossMonsterPairedBlock,
    RecursiveDepthBlock,
    RecursiveDepthRouterBlock,
    SparseMoEBlock,
    ThreeLaneAdaptive,
)
from .primitive_templates import (
    CalculusAugmentedLane,
    ChebyshevAdapterLane,
    ChebyshevSpectralLane,
    CliffordAttention,
    FiniteDifferenceCalculusLane,
    FisherAdapterLane,
    FisherAttention,
    FourierBasisLane,
    GraphDiffusionAdapterLane,
    GraphDiffusionLane,
    LambdaFunctionalAdapterLane,
    LambdaFunctionalLane,
    LinearStateSpaceLane,
    LowRankAdapterLane,
    LowRankFactorizedLane,
    MultiscaleWaveletAdapterLane,
    MultiscaleWaveletLane,
    PadicProjection,
    PoincareAttention,
    QuaternionAttention,
    RandomFeatureKernelAdapterLane,
    RandomFeatureKernelLane,
    SparseBandedAdapterLane,
    SparseBandedMatrixLane,
    SpikingActivationGate,
    TopKLinear,
    TropicalAttention,
    TropicalStateSpace,
    TropicalTopKStateSpace,
    TuckerAdapterLane,
    TuckerDecompLane,
)
from .routing_primitives import (
    RECURSION_SITES,
    ROUTING_KINDS,
    DifficultyRoutedLane,
    HashedMoELane,
    LowInfoSkipRouter,
    MixtureOfRecursionsLane,
    RoutedBottleneckLane,
    SiteRecursionStack,
    SparseMoRLane,
)


LaneFactory = Callable[[int], nn.Module]


class UndispatchableSpecError(ValueError):
    """A spec's math axes matched no generator template."""


def _physics_atom_kinds(raw: Any) -> tuple[str, ...]:
    if raw in (None, "", "identity"):
        return ()
    if isinstance(raw, str):
        return tuple(part for part in raw.split("+") if part)
    if isinstance(raw, (tuple, list)):
        return tuple(str(part) for part in raw if str(part))
    return (str(raw),)


def _axis(math_axes: dict[str, Any], key: str) -> str:
    value = math_axes.get(key)
    return "" if value is None else str(value)


def _has_state(math_axes: dict[str, Any]) -> bool:
    return bool(int(math_axes.get("op_dynamical_has_state") or 0))


def _dispatch_tropical(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float
) -> nn.Module | None:
    if _axis(math_axes, "op_algebraic_space") != "tropical":
        return None
    sparsity = _axis(math_axes, "op_activation_sparsity_pattern")
    if _has_state(math_axes) and sparsity == "top_k":
        k = max(1, int(round(dim * top_k_frac)))
        return TropicalTopKStateSpace(dim, k=k)
    if _has_state(math_axes):
        return TropicalStateSpace(dim)
    return TropicalAttention(dim)


def _dispatch_clifford(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float = 0.25
) -> nn.Module | None:
    if _axis(math_axes, "op_algebraic_space") != "clifford":
        return None
    if dim % 4 != 0:
        return nn.Linear(dim, dim)
    return CliffordAttention(dim)


def _dispatch_spiking(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float = 0.25
) -> nn.Module | None:
    if _axis(math_axes, "op_algebraic_space") != "spiking":
        return None
    return SpikingActivationGate(dim)


def _dispatch_padic(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float = 0.25
) -> nn.Module | None:
    if _axis(math_axes, "op_algebraic_space") != "padic":
        return None
    if dim % 8 != 0:
        return nn.Linear(dim, dim)
    return PadicProjection(dim, p=2, n_levels=3)


def _dispatch_quaternion(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float = 0.25
) -> nn.Module | None:
    if _axis(math_axes, "op_algebraic_space") != "quaternion":
        return None
    if dim % 4 != 0:
        return nn.Linear(dim, dim)
    return QuaternionAttention(dim)


def _dispatch_hyperbolic(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float = 0.25
) -> nn.Module | None:
    if _axis(math_axes, "op_algebraic_space") not in (
        "hyperbolic",
        "hyperbolic_poincare",
    ):
        return None
    return PoincareAttention(dim)


def _dispatch_state_kernel(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float = 0.25
) -> nn.Module | None:
    if not _has_state(math_axes):
        return None
    algebra = _axis(math_axes, "op_algebraic_space")
    if algebra in ("tropical", "clifford", "spiking", "padic"):
        return None
    return LinearStateSpaceLane(dim)


def _dispatch_axis_modifier(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float
) -> nn.Module | None:
    sparsity = _axis(math_axes, "op_activation_sparsity_pattern")
    if sparsity == "top_k":
        k = max(1, int(round(dim * top_k_frac)))
        return TopKLinear(dim, dim, k=k)
    basis = _axis(math_axes, "op_spectral_preferred_basis")
    if basis in ("fourier", "frequency"):
        return FourierBasisLane(dim)
    return None


def _dispatch_math_knob(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float
) -> nn.Module | None:
    family = _axis(math_axes, "op_math_family")
    if family == "calculus":
        operator = _axis(math_axes, "op_calculus_operator")
        if operator in ("causal_finite_difference_integral", "finite_difference"):
            return FiniteDifferenceCalculusLane(dim)
    if family == "linear_algebra":
        structure = _axis(math_axes, "op_linear_algebra_structure")
        if structure in ("low_rank_factorized", "low_rank"):
            rank = max(1, int(round(dim * top_k_frac)))
            return LowRankFactorizedLane(dim, rank=rank)
    if family == "sparse_matrix":
        pattern = _axis(math_axes, "op_sparse_matrix_pattern")
        if pattern in ("causal_banded", "banded"):
            bandwidth = max(1, min(dim, int(round(dim * top_k_frac))))
            return SparseBandedMatrixLane(dim, bandwidth=bandwidth)
    if family == "kernel_methods":
        kernel = _axis(math_axes, "op_kernel_feature_map")
        if kernel in ("positive_random_features", "random_features"):
            n_features = max(4, int(round(dim * 0.5)))
            return RandomFeatureKernelLane(dim, n_features=n_features)
    if family == "multiscale":
        transform = _axis(math_axes, "op_multiscale_transform")
        if transform in ("causal_haar", "wavelet"):
            return MultiscaleWaveletLane(dim)
    if family == "graph_diffusion":
        topology = _axis(math_axes, "op_graph_topology")
        if topology in ("causal_path_laplacian", "causal_path"):
            return GraphDiffusionLane(dim)
    if family == "lambda_functional":
        transform = _axis(math_axes, "op_lambda_transform")
        if transform in ("learned_functional_blend", ""):
            return LambdaFunctionalLane(
                dim,
                gate=_axis(math_axes, "op_lambda_gate") or "content",
                basis=_axis(math_axes, "op_lambda_basis") or "identity",
            )
    if family == "information_geometry":
        operator = _axis(math_axes, "op_info_geom_operator")
        if operator in ("fisher_attention", "fisher", ""):
            return FisherAttention(dim)
    if family == "spectral_graph":
        operator = _axis(math_axes, "op_spectral_graph_operator")
        if operator in ("chebyshev_polynomial", "chebyshev", ""):
            return ChebyshevSpectralLane(dim, n_terms=5)
    if family == "tensor_decomp":
        decomp = _axis(math_axes, "op_tensor_decomp_kind")
        if decomp in ("tucker", ""):
            return TuckerDecompLane(dim)
    return None


def _dispatch_synthesis_hint(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float
) -> nn.Module | None:
    kind = _axis(math_axes, "synthesis_kind")
    if not kind:
        return None
    if kind == "basis_swap":
        basis = _axis(math_axes, "op_spectral_preferred_basis")
        if basis in ("chebyshev", "polynomial"):
            return ChebyshevSpectralLane(dim, n_terms=5)
        if basis in ("fourier", "frequency"):
            return FourierBasisLane(dim)
        return ChebyshevSpectralLane(dim, n_terms=5)
    if kind == "projection_swap":
        return HashedMoELane(_expert_factory_pool(top_k_frac), dim)
    if kind == "state_kernel_swap":
        return LinearStateSpaceLane(dim)
    return None


def _dispatch_rule(
    name: str,
    handler: Callable[..., nn.Module | None],
    *,
    dim: int,
    top_k_frac: float,
) -> DispatchRule:
    return DispatchRule(
        name,
        lambda axes: handler(axes, dim=dim, top_k_frac=top_k_frac),
    )


def _base_dispatchers(*, dim: int, top_k_frac: float) -> tuple[DispatchRule, ...]:
    return (
        _dispatch_rule("tropical", _dispatch_tropical, dim=dim, top_k_frac=top_k_frac),
        _dispatch_rule("clifford", _dispatch_clifford, dim=dim, top_k_frac=top_k_frac),
        _dispatch_rule("spiking", _dispatch_spiking, dim=dim, top_k_frac=top_k_frac),
        _dispatch_rule("padic", _dispatch_padic, dim=dim, top_k_frac=top_k_frac),
        _dispatch_rule(
            "quaternion", _dispatch_quaternion, dim=dim, top_k_frac=top_k_frac
        ),
        _dispatch_rule(
            "hyperbolic", _dispatch_hyperbolic, dim=dim, top_k_frac=top_k_frac
        ),
        _dispatch_rule(
            "state_kernel", _dispatch_state_kernel, dim=dim, top_k_frac=top_k_frac
        ),
        _dispatch_rule(
            "axis_modifier", _dispatch_axis_modifier, dim=dim, top_k_frac=top_k_frac
        ),
        _dispatch_rule(
            "synthesis_hint", _dispatch_synthesis_hint, dim=dim, top_k_frac=top_k_frac
        ),
    )


def _base_module(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float
) -> nn.Module:
    result = dispatch_first(
        _base_dispatchers(dim=dim, top_k_frac=top_k_frac), math_axes
    )
    if result is not None:
        return result
    present = sorted(k for k, v in math_axes.items() if v is not None)
    raise UndispatchableSpecError(
        "no generator template matched these math_axes; refusing to fall back "
        f"to nn.Linear. Non-null axes: {present}"
    )


def _apply_math_knobs(
    module: nn.Module,
    math_axes: dict[str, Any],
    *,
    dim: int,
    top_k_frac: float,
) -> nn.Module:
    rank = max(1, int(round(dim * top_k_frac)))
    bandwidth = max(1, min(dim, int(round(dim * top_k_frac))))
    n_features = max(4, int(round(dim * 0.5)))
    for knob in math_knobs_from_axes(math_axes):
        if knob == "calculus_finite_difference":
            module = CalculusAugmentedLane(module, dim)
        elif knob == "linear_algebra_low_rank":
            module = LowRankAdapterLane(module, dim, rank=rank)
        elif knob == "sparse_matrix_banded":
            module = SparseBandedAdapterLane(module, dim, bandwidth=bandwidth)
        elif knob == "kernel_random_features":
            module = RandomFeatureKernelAdapterLane(module, dim, n_features=n_features)
        elif knob == "multiscale_wavelet":
            module = MultiscaleWaveletAdapterLane(module, dim)
        elif knob == "graph_laplacian_diffusion":
            module = GraphDiffusionAdapterLane(module, dim)
        elif knob == "lambda_functional_blend":
            module = LambdaFunctionalAdapterLane(
                module,
                dim,
                gate=_axis(math_axes, "op_lambda_gate") or "content",
                basis=_axis(math_axes, "op_lambda_basis") or "identity",
            )
        elif knob == "info_geom_fisher":
            module = FisherAdapterLane(module, dim)
        elif knob == "spectral_chebyshev":
            module = ChebyshevAdapterLane(module, dim)
        elif knob == "tensor_tucker":
            tucker_rank = max(2, int(round(dim * top_k_frac)))
            module = TuckerAdapterLane(module, dim, rank=tucker_rank)
    return module


def _base_lane_factory(math_axes: dict[str, Any], *, top_k_frac: float) -> LaneFactory:
    def factory(dim: int) -> nn.Module:
        axes = dict(math_axes)
        axes["op_routing_kind"] = "none"
        return generate_module(axes, dim=dim, top_k_frac=top_k_frac)

    return factory


def _recursion_sites(raw: Any) -> tuple[str, ...]:
    if raw in (None, "", "none"):
        return ("mixer",)
    if isinstance(raw, str):
        normalized = raw.replace(",", "+")
        return tuple(part.strip() for part in normalized.split("+") if part.strip())
    if isinstance(raw, (tuple, list, set)):
        return tuple(str(part).strip() for part in raw if str(part).strip())
    return (str(raw).strip(),)


# Cap on the summed per-site recursion depth. The paired probe already makes
# recursion pay for itself capability-wise (added halt/site params must beat the
# anchor), but a 4-site x deep spec could still balloon the lane; this fails
# loud at the pathological end instead of silently clamping.
MAX_RECURSION_TOTAL_DEPTH = 32


def _site_recursion_module(
    site: str,
    base: nn.Module,
    *,
    dim: int,
    top_k_frac: float,
    top_k: int,
) -> nn.Module:
    """Build the weighted submodule recursion wraps for a given site."""
    if site == "mixer":
        return base
    if site == "ffn":
        return SwiGLU(dim)
    if site == "router":
        return RoutedBottleneckLane(_expert_factory_pool(top_k_frac), dim, top_k=top_k)
    if site == "embedding":
        rank = max(1, int(round(dim * top_k_frac)))
        return LowRankFactorizedLane(dim, rank=rank)
    raise ValueError(f"unsupported recursion site {site!r}")


def _build_site_recursion(
    base: nn.Module,
    math_axes: dict[str, Any],
    *,
    dim: int,
    top_k_frac: float,
    default_depth: int,
    top_k: int,
) -> nn.Module:
    """Recurse over any weighted site (embedding/mixer/ffn/router), not just the
    mixer lane — the "recursion anywhere there are weights" search axis."""
    listed = set(_recursion_sites(math_axes.get("op_recursion_sites")))
    unsupported = sorted(listed - set(RECURSION_SITES))
    if unsupported:
        raise NotImplementedError(
            f"site_recursion supports sites={list(RECURSION_SITES)}; "
            f"unsupported={unsupported}"
        )
    # The token mixer is the lane's reason to exist — always present, recursed
    # (depth>1) only when explicitly requested; depth 1 == plain application.
    sites = tuple(s for s in RECURSION_SITES if s in listed or s == "mixer")
    depths = {
        site: (
            int(math_axes.get(f"op_max_depth_{site}") or default_depth)
            if site in listed
            else 1
        )
        for site in sites
    }
    total = sum(depths.values())
    if total > MAX_RECURSION_TOTAL_DEPTH:
        raise ValueError(
            f"site_recursion total depth {total} exceeds budget "
            f"{MAX_RECURSION_TOTAL_DEPTH} (sites={depths}); recursion must pay "
            "for itself, not bloat the lane"
        )
    modules = {
        site: _site_recursion_module(
            site, base, dim=dim, top_k_frac=top_k_frac, top_k=top_k
        )
        for site in sites
    }
    return SiteRecursionStack(modules, dim, depths=depths, default_depth=default_depth)


def _expert_factory_pool(
    top_k_frac: float,
) -> tuple[LaneFactory, LaneFactory, LaneFactory]:
    def expert_attn(dim: int) -> nn.Module:
        return TropicalAttention(dim)

    def expert_ssm(dim: int) -> nn.Module:
        return LinearStateSpaceLane(dim)

    def expert_topk(dim: int) -> nn.Module:
        k = max(1, int(round(dim * top_k_frac)))
        return TopKLinear(dim, dim, k=k)

    return (expert_attn, expert_ssm, expert_topk)


def _apply_routing_wrap(
    base: nn.Module,
    math_axes: dict[str, Any],
    *,
    dim: int,
    top_k_frac: float,
) -> nn.Module:
    kind = str(math_axes.get("op_routing_kind") or "none")
    if kind == "none" or kind not in ROUTING_KINDS:
        return base
    max_depth = int(math_axes.get("op_max_depth") or 4)
    top_k = int(math_axes.get("op_top_k") or 2)
    skip_hard = bool(int(math_axes.get("op_skip_hard") or 0))
    base_factory = _base_lane_factory(math_axes, top_k_frac=top_k_frac)
    if kind == "depth_router":
        return MixtureOfRecursionsLane(base_factory, dim, max_depth=max_depth)
    if kind == "site_recursion":
        return _build_site_recursion(
            base,
            math_axes,
            dim=dim,
            top_k_frac=top_k_frac,
            default_depth=max_depth,
            top_k=top_k,
        )
    if kind == "sparse_depth":
        return SparseMoRLane(
            base_factory, dim, max_depth=max_depth, top_k_frac=top_k_frac
        )
    if kind == "low_info_skip":
        return LowInfoSkipRouter(base_factory, dim, hard=skip_hard)
    if kind == "difficulty":

        def easy(d: int) -> nn.Module:
            return LinearStateSpaceLane(d)

        return DifficultyRoutedLane(easy, base_factory, dim)
    if kind == "hash":
        return HashedMoELane(_expert_factory_pool(top_k_frac), dim)
    if kind == "top_k_moe":
        return RoutedBottleneckLane(_expert_factory_pool(top_k_frac), dim, top_k=top_k)
    return base


def _dispatch_nas_graph(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float = 0.25
) -> nn.Module | None:
    if str(math_axes.get("op_source") or "") != SOURCE_NAS:
        return None
    fingerprint = str(math_axes.get("op_nas_fingerprint") or "")
    graph_json = load_cached_graph_json(fingerprint)
    if graph_json is None:
        raise RuntimeError(
            f"nas_graph spec {fingerprint!r}: cached graph JSON missing "
            f"(component_fab/catalog/nas_graphs/)"
        )
    from research.synthesis.compiler import compile_graph
    from research.synthesis.serializer import graph_from_json

    graph = graph_from_json(graph_json, model_dim=dim)
    return compile_graph(graph, use_ir=True)


def _dispatch_physics_atom_program(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float = 0.25
) -> nn.Module | None:
    """Build a name-free parametric atom/mixer program from physics axes."""
    del top_k_frac
    if str(math_axes.get("op_search_track") or "") != "physics_atom":
        return None
    from research.synthesis.open_discovery import ProgramSpec, build_program
    from research.synthesis.parametric_atoms import AtomSpec
    from research.synthesis.parametric_ops import StageSpec

    atom = AtomSpec(
        kinds=_physics_atom_kinds(math_axes.get("op_physics_atom_kinds")),
        norm_axis=str(math_axes.get("op_physics_norm_axis") or "channel"),
        basis_axis=str(math_axes.get("op_physics_basis_axis") or "channel"),
    )
    stage = StageSpec(
        address=str(math_axes.get("op_physics_address_family") or "dot"),
        score_norm=str(math_axes.get("op_physics_score_norm_family") or "softmax"),
        aggregate=str(math_axes.get("op_physics_aggregate_family") or "mean"),
    )
    spec = ProgramSpec(
        atom=atom,
        stage=stage,
        knob_scale=float(math_axes.get("op_physics_knob_scale") or 1.0),
    )
    seed = int(math_axes.get("op_physics_seed") or 0)
    return build_program(spec, dim=dim, seed=seed)


def _dispatch_block_template(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float
) -> nn.Module | None:
    template = str(math_axes.get("op_block_template") or "")
    if not template or template not in BLOCK_TEMPLATES:
        return None
    inner_axes = dict(math_axes)
    inner_axes.pop("op_block_template", None)
    inner_block = str(math_axes.get("op_block_inner_template") or "")
    if inner_block and inner_block in BLOCK_TEMPLATES:
        inner_axes["op_block_template"] = inner_block
        inner_axes.pop("op_block_inner_template", None)

    def anchor_factory(d: int) -> nn.Module:
        return generate_module(inner_axes, dim=d, top_k_frac=top_k_frac)

    builder = _BLOCK_TEMPLATE_BUILDERS.get(template)
    if builder is None:
        return None
    return builder(math_axes, anchor_factory, dim, top_k_frac)


def _build_latent_compress(math_axes, anchor_factory, dim, top_k_frac):
    del top_k_frac
    compress = int(math_axes.get("op_block_compress") or 2)
    return LatentCompressBlock(anchor_factory, dim, compress=compress)


def _build_three_lane_adaptive(math_axes, anchor_factory, dim, top_k_frac):
    del top_k_frac
    slot_b = _block_slot_factory(_block_slot_name(math_axes, "b", "tropical_attention"))
    slot_c = _block_slot_factory(_block_slot_name(math_axes, "c", "linear_state_space"))
    return ThreeLaneAdaptive(anchor_factory, slot_b, slot_c, dim)


def _build_recursive_depth(math_axes, anchor_factory, dim, top_k_frac):
    del top_k_frac
    max_depth = int(math_axes.get("op_max_depth") or 3)
    return RecursiveDepthBlock(anchor_factory, dim, max_depth=max_depth)


def _build_gated_parallel(math_axes, anchor_factory, dim, top_k_frac):
    del top_k_frac
    slot_b = _block_slot_factory(_block_slot_name(math_axes, "b", "multiscale_wavelet"))
    return GatedParallelBlock(anchor_factory, slot_b, dim)


def _loss_partner_factory(math_axes, anchor_factory) -> LaneFactory:
    explicit_slot = str(math_axes.get("op_block_slot_partner") or "").strip()
    if explicit_slot:
        return _block_slot_factory(explicit_slot)
    partner_kind = str(math_axes.get("op_partner_kind") or "anchor").strip()
    if partner_kind in ("", "anchor"):
        return anchor_factory
    slot_name = slot_name_for_partner_kind(partner_kind)
    if slot_name is None:
        known = ", ".join(known_partner_kinds())
        raise ValueError(
            f"unknown loss-monster partner kind {partner_kind!r}; known: {known}"
        )
    return _block_slot_factory(slot_name)


def _loss_slot_name(math_axes: dict[str, Any]) -> str:
    return str(
        math_axes.get("op_block_slot_loss")
        or math_axes.get("op_loss_monster_slot")
        or "routed_bottleneck"
    )


def _build_loss_monster_paired(math_axes, anchor_factory, dim, top_k_frac):
    del top_k_frac
    partner_factory = _loss_partner_factory(math_axes, anchor_factory)
    loss_factory = _block_slot_factory(_loss_slot_name(math_axes))
    partner_floor = float(math_axes.get("op_partner_floor") or 0.5)
    return LossMonsterPairedBlock(
        partner_factory,
        loss_factory,
        dim,
        partner_floor=partner_floor,
    )


def _build_recursive_depth_router(math_axes, anchor_factory, dim, top_k_frac):
    del top_k_frac
    max_depth = int(math_axes.get("op_max_depth") or 4)
    return RecursiveDepthRouterBlock(anchor_factory, dim, max_depth=max_depth)


def _build_sparse_moe_block(math_axes, anchor_factory, dim, top_k_frac):
    top_k = int(math_axes.get("op_top_k") or 2)
    return SparseMoEBlock(
        anchor_factory,
        _expert_factory_pool(top_k_frac),
        dim,
        top_k=top_k,
    )


def _build_hetero_moe_block(math_axes, anchor_factory, dim, top_k_frac):
    def hetero_topk(d: int) -> nn.Module:
        k = max(1, int(round(d * top_k_frac)))
        return TopKLinear(d, d, k=k)

    return HeteroMoEBlock(
        anchor_factory,
        (TropicalAttention, LinearStateSpaceLane, hetero_topk, MultiscaleWaveletLane),
        dim,
    )


def _build_top_ar_block(math_axes, anchor_factory, dim, top_k_frac):
    del top_k_frac
    from component_fab.harness.top_ar_block import TopArchBlock

    slot_b = _block_slot_factory(_block_slot_name(math_axes, "b", "local_window_attn"))
    return TopArchBlock(dim, anchor_factory, slot_b)


def _block_slot_name(math_axes: dict[str, Any], slot: str, default: str) -> str:
    return str(math_axes.get(f"op_block_slot_{slot}") or "") or default


def _block_slot_factory(name: str) -> LaneFactory:
    return build_block_slot_factory(name)


_BLOCK_TEMPLATE_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "latent_compress": _build_latent_compress,
    "three_lane_adaptive": _build_three_lane_adaptive,
    "recursive_depth": _build_recursive_depth,
    "gated_parallel": _build_gated_parallel,
    "loss_monster_paired": _build_loss_monster_paired,
    "recursive_depth_router": _build_recursive_depth_router,
    "sparse_moe_block": _build_sparse_moe_block,
    "hetero_moe_block": _build_hetero_moe_block,
    "hyperbolic_bridge": lambda _axes, anchor, dim, _tkf: HyperbolicBridgeBlock(
        anchor, dim
    ),
    "attn_spectral_filter": lambda _axes, anchor, dim, _tkf: AttnSpectralFilterBlock(
        anchor, dim
    ),
    "graph_attention": lambda _axes, anchor, dim, _tkf: GraphAttentionBlock(
        anchor, dim
    ),
    "top_ar_block": _build_top_ar_block,
}


def _default_dispatchers(*, dim: int, top_k_frac: float) -> tuple[DispatchRule, ...]:
    return (
        _dispatch_rule(
            "math_knob", _dispatch_math_knob, dim=dim, top_k_frac=top_k_frac
        ),
        DispatchRule(
            "base_module",
            lambda axes: _base_module(axes, dim=dim, top_k_frac=top_k_frac),
        ),
    )


def generate_module(
    math_axes: dict[str, Any],
    *,
    dim: int = 32,
    top_k_frac: float = 0.25,
) -> nn.Module:
    """Generate a primitive instance from a math-axis tuple."""
    nas = _dispatch_nas_graph(math_axes, dim=dim, top_k_frac=top_k_frac)
    if nas is not None:
        return nas
    block = _dispatch_block_template(math_axes, dim=dim, top_k_frac=top_k_frac)
    if block is not None:
        return block
    physics_atom = _dispatch_physics_atom_program(
        math_axes, dim=dim, top_k_frac=top_k_frac
    )
    if physics_atom is not None:
        return _apply_routing_wrap(
            physics_atom, math_axes, dim=dim, top_k_frac=top_k_frac
        )
    invention = dispatch_invention_mechanism(math_axes, dim=dim, top_k_frac=top_k_frac)
    if invention is not None:
        return _apply_routing_wrap(invention, math_axes, dim=dim, top_k_frac=top_k_frac)
    if math_axes.get("op_math_knobs") is not None:
        base = _base_module(math_axes, dim=dim, top_k_frac=top_k_frac)
        wrapped = _apply_math_knobs(base, math_axes, dim=dim, top_k_frac=top_k_frac)
        return _apply_routing_wrap(wrapped, math_axes, dim=dim, top_k_frac=top_k_frac)
    result = dispatch_first(
        _default_dispatchers(dim=dim, top_k_frac=top_k_frac), math_axes
    )
    if result is not None:
        return _apply_routing_wrap(result, math_axes, dim=dim, top_k_frac=top_k_frac)
    raise RuntimeError("unreachable module dispatch state")


def generate_module_from_spec(
    spec: ProposalSpec, *, dim: int = 32, top_k_frac: float = 0.25
) -> nn.Module:
    return generate_module(spec.math_axes, dim=dim, top_k_frac=top_k_frac)
