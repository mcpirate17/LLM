"""Dispatcher — ProposalSpec → runnable nn.Module.

Maps the spec's math axes + synthesis_kind onto the right
``primitive_templates`` class. Returns an instantiated module ready to
plug into the standard test harness.

Dispatch order (first match wins): explicit ``op_math_family`` knobs come
first because they name concrete operator mechanisms; algebraic_space comes
before sparsity otherwise because algebra determines the underlying math.
E.g. ``tropical + state + top_k`` should materialize as
``TropicalTopKStateSpace``, not bypass to ``TopKLinear``.
"""

from __future__ import annotations

from typing import Any

from torch import nn

from ..proposer.nas_bridge import SOURCE_NAS, load_cached_graph_json
from ..proposer.spec_generator import ProposalSpec
from .memory_primitives import (
    CausalFastWeightMemoryLane,
    CausalSlotRouterMemoryLane,
    DataDependentDecayMemoryLane,
    HierarchicalResidualCompressorLane,
    PadicSurpriseMemoryLane,
    SemiringSurpriseMemoryLane,
    TropicalSurpriseMemoryLane,
)
from .native_surprise_memory import (
    NativeAtlasPolySurpriseMemoryLane,
    NativeAdaptiveSemiringBiLaneSurpriseMemoryLane,
    NativeAdaptiveSemiringRopeTitansMACSurpriseMemoryLane,
    NativeBalancedSemiringBiLaneSurpriseMemoryLane,
    NativeBalancedSemiringRopeTitansMACSurpriseMemoryLane,
    NativeBalancedSemiringTitansMACSurpriseMemoryLane,
    NativeBalancedSemiringTriLaneSurpriseMemoryLane,
    NativeContextGatedSurpriseMemoryLane,
    NativeReadBeforeWriteSurpriseMemoryLane,
    NativeSemiringRopeSurpriseMemoryLane,
    NativeSemiringRopeTitansMACSurpriseMemoryLane,
    NativeSemiringSurpriseMemoryLane,
    NativeSemiringTitansMACSurpriseMemoryLane,
    NativeTitansMACSurpriseMemoryLane,
)
from .primitive_templates import (
    CalculusAugmentedLane,
    ChebyshevAdapterLane,
    ChebyshevSpectralLane,
    CliffordAttention,
    FisherAdapterLane,
    FisherAttention,
    FiniteDifferenceCalculusLane,
    FourierBasisLane,
    GraphDiffusionAdapterLane,
    GraphDiffusionLane,
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
    SparsemaxAttention,
    SpikingActivationGate,
    SymplecticResidualMixerLane,
    TopKLinear,
    TropicalAttention,
    TropicalStateSpace,
    TropicalTopKStateSpace,
    TuckerAdapterLane,
    TuckerDecompLane,
)
from .block_templates import (
    BLOCK_TEMPLATES,
    AttnSpectralFilterBlock,
    GatedParallelBlock,
    GraphAttentionBlock,
    HeteroMoEBlock,
    HyperbolicBridgeBlock,
    LatentCompressBlock,
    RecursiveDepthBlock,
    RecursiveDepthRouterBlock,
    SparseMoEBlock,
    ThreeLaneAdaptive,
)
from .routing_primitives import (
    ROUTING_KINDS,
    DifficultyRoutedLane,
    HashedMoELane,
    LowInfoSkipRouter,
    MixtureOfRecursionsLane,
    RoutedBottleneckLane,
    SparseMoRLane,
)


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
    """Generic state-bearing primitive for non-tropical / non-clifford / non-padic
    proposals declaring ``op_dynamical_has_state=1``. Algebra-specific
    state primitives (TropicalStateSpace, etc.) already fired earlier in
    the dispatch chain, so reaching here means no domain module matched.
    """
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
    # Phase 2 (2026-05-15): information_geometry, spectral_graph, tensor_decomp.
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


def _dispatch_invention_mechanism(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float = 0.25
) -> nn.Module | None:
    mechanism = _axis(math_axes, "op_invention_mechanism")
    if mechanism == "causal_fast_weight_memory":
        return CausalFastWeightMemoryLane(dim)
    if mechanism == "data_dependent_decay_memory":
        return DataDependentDecayMemoryLane(dim)
    if mechanism == "causal_slot_router_memory":
        return CausalSlotRouterMemoryLane(dim)
    if mechanism == "hierarchical_residual_compressor":
        return HierarchicalResidualCompressorLane(dim)
    if mechanism == "symplectic_residual_mixer":
        if dim % 2 != 0:
            return nn.Linear(dim, dim)
        return SymplecticResidualMixerLane(dim)
    if mechanism == "tropical_surprise_memory":
        return TropicalSurpriseMemoryLane(dim)
    if mechanism == "semiring_surprise_memory":
        return SemiringSurpriseMemoryLane(dim)
    if mechanism == "semiring_surprise_memory_rope":
        return SemiringSurpriseMemoryLane(dim, use_rope=True)
    if mechanism == "padic_surprise_memory":
        return PadicSurpriseMemoryLane(dim)
    if mechanism == "native_read_before_write_surprise_memory":
        return NativeReadBeforeWriteSurpriseMemoryLane(dim)
    if mechanism == "native_context_gated_surprise_memory":
        return NativeContextGatedSurpriseMemoryLane(dim)
    if mechanism == "native_atlas_poly_surprise_memory":
        return NativeAtlasPolySurpriseMemoryLane(dim)
    if mechanism == "native_titans_mac_surprise_memory":
        return NativeTitansMACSurpriseMemoryLane(dim)
    if mechanism == "native_semiring_surprise_memory":
        return NativeSemiringSurpriseMemoryLane(dim)
    if mechanism == "native_semiring_surprise_memory_rope":
        return NativeSemiringRopeSurpriseMemoryLane(dim)
    if mechanism == "native_semiring_titans_mac_surprise_memory":
        return NativeSemiringTitansMACSurpriseMemoryLane(dim)
    if mechanism == "native_semiring_rope_titans_mac_surprise_memory":
        return NativeSemiringRopeTitansMACSurpriseMemoryLane(dim)
    if mechanism == "native_balanced_semiring_titans_mac_surprise_memory":
        return NativeBalancedSemiringTitansMACSurpriseMemoryLane(dim)
    if mechanism == "native_balanced_semiring_rope_titans_mac_surprise_memory":
        return NativeBalancedSemiringRopeTitansMACSurpriseMemoryLane(dim)
    if mechanism == "native_balanced_semiring_bilane_surprise_memory":
        return NativeBalancedSemiringBiLaneSurpriseMemoryLane(dim)
    if mechanism == "native_balanced_semiring_trilane_surprise_memory":
        return NativeBalancedSemiringTriLaneSurpriseMemoryLane(dim)
    if mechanism == "native_adaptive_semiring_rope_titans_mac_surprise_memory":
        return NativeAdaptiveSemiringRopeTitansMACSurpriseMemoryLane(dim)
    if mechanism == "native_adaptive_semiring_bilane_surprise_memory":
        return NativeAdaptiveSemiringBiLaneSurpriseMemoryLane(dim)
    return None


def _math_knobs(math_axes: dict[str, Any]) -> tuple[str, ...]:
    raw = math_axes.get("op_math_knobs")
    if raw is None:
        family = _axis(math_axes, "op_math_family")
        if family == "calculus":
            return ("calculus_finite_difference",)
        if family == "linear_algebra":
            return ("linear_algebra_low_rank",)
        if family == "sparse_matrix":
            return ("sparse_matrix_banded",)
        if family == "kernel_methods":
            return ("kernel_random_features",)
        if family == "information_geometry":
            return ("info_geom_fisher",)
        if family == "spectral_graph":
            return ("spectral_chebyshev",)
        if family == "tensor_decomp":
            return ("tensor_tucker",)
        if family == "multiscale":
            return ("multiscale_wavelet",)
        if family == "graph_diffusion":
            return ("graph_laplacian_diffusion",)
        return ()
    if isinstance(raw, str):
        return tuple(part.strip() for part in raw.split("+") if part.strip())
    if isinstance(raw, (list, tuple)):
        return tuple(str(part) for part in raw if str(part))
    return ()


def _dispatch_synthesis_hint(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float
) -> nn.Module | None:
    """Phase 3: read ``synthesis_kind`` and pick a primitive when the
    upstream algebra/state/sparsity dispatchers don't fully determine
    the module. Without this, ``basis_swap`` falls through to
    ``nn.Linear`` because no axis carries a sparsity/state/algebra
    signal. This dispatcher gives the kind label real teeth.
    """
    kind = _axis(math_axes, "synthesis_kind")
    # synthesis_kind isn't in math_axes by default — it's a ProposalSpec
    # field. But specs may carry it forward via math_axes when set
    # explicitly. Fall back to inferring from declared axes.
    if not kind:
        return None
    if kind == "basis_swap":
        basis = _axis(math_axes, "op_spectral_preferred_basis")
        if basis in ("chebyshev", "polynomial"):
            return ChebyshevSpectralLane(dim, n_terms=5)
        if basis in ("fourier", "frequency"):
            return FourierBasisLane(dim)
        # Default basis_swap → Chebyshev (FNO-style polynomial mixing).
        return ChebyshevSpectralLane(dim, n_terms=5)
    if kind == "projection_swap":
        # Sparsity-pattern primitives. Hash routing as default.
        return HashedMoELane(_expert_factory_pool(top_k_frac), dim)
    if kind == "state_kernel_swap":
        # State-bearing primitives. Generic linear-SSM fallback.
        return LinearStateSpaceLane(dim)
    return None


# Flat dispatch table for the algebra / sparsity / synthesis chain. Each entry
# is a function with the uniform signature
#   (math_axes, *, dim: int, top_k_frac: float) -> nn.Module | None
# that returns the matching primitive (or ``None`` to fall through). Order is
# load-bearing: tropical fires before sparsity because algebra determines the
# underlying math, and ``TropicalTopKStateSpace`` (tropical + state + top_k)
# must materialize as the state primitive, not be bypassed to ``TopKLinear``.
_BASE_DISPATCHERS: tuple = (
    _dispatch_tropical,
    _dispatch_clifford,
    _dispatch_spiking,
    _dispatch_padic,
    _dispatch_quaternion,
    _dispatch_hyperbolic,
    _dispatch_state_kernel,
    _dispatch_axis_modifier,
    _dispatch_synthesis_hint,
)


def _base_module(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float
) -> nn.Module:
    for dispatcher in _BASE_DISPATCHERS:
        result = dispatcher(math_axes, dim=dim, top_k_frac=top_k_frac)
        if result is not None:
            return result
    return nn.Linear(dim, dim)


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
    for knob in _math_knobs(math_axes):
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
        elif knob == "info_geom_fisher":
            module = FisherAdapterLane(module, dim)
        elif knob == "spectral_chebyshev":
            module = ChebyshevAdapterLane(module, dim)
        elif knob == "tensor_tucker":
            tucker_rank = max(2, int(round(dim * top_k_frac)))
            module = TuckerAdapterLane(module, dim, rank=tucker_rank)
    return module


def _base_lane_factory(math_axes: dict[str, Any], *, top_k_frac: float) -> "callable":
    """Return a ``Callable[[int], nn.Module]`` that re-dispatches the base
    primitive for the given axes at the requested dim.

    Routing primitives (MoR / sparseMoR / skip / Difficulty) need this
    so each routing slot creates a fresh inner mixer rather than reusing
    one instance across many slots.
    """

    def factory(dim: int) -> nn.Module:
        axes = dict(math_axes)
        axes["op_routing_kind"] = "none"  # break recursion
        return generate_module(axes, dim=dim, top_k_frac=top_k_frac)

    return factory


def _expert_factory_pool(top_k_frac: float) -> tuple:
    """Three diverse expert factories for HashedMoE / RoutedBottleneck.

    Mixing a max-plus attention, a softmax-style state-space lane, and
    a top-k linear forces the MoE to use experts with different
    inductive biases. Sized for dim divisible by 4 (Cl(2,0) constraint
    not used here, but kept for forward compat).
    """

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
    """Wrap ``base`` in a routing primitive if the spec requests one.

    Spec axis ``op_routing_kind`` picks the wrapper; if unset or "none"
    the base module passes through unchanged. ``op_max_depth``,
    ``op_n_experts``, ``op_top_k``, ``op_skip_hard`` modulate per-kind
    knobs.
    """
    kind = str(math_axes.get("op_routing_kind") or "none")
    if kind == "none" or kind not in ROUTING_KINDS:
        return base
    max_depth = int(math_axes.get("op_max_depth") or 4)
    top_k = int(math_axes.get("op_top_k") or 2)
    skip_hard = bool(int(math_axes.get("op_skip_hard") or 0))
    base_factory = _base_lane_factory(math_axes, top_k_frac=top_k_frac)
    if kind == "depth_router":
        return MixtureOfRecursionsLane(base_factory, dim, max_depth=max_depth)
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
    """Compile a NAS-synthesized graph topology into a token-mixing lane.

    When ``op_source == "nas_graph"`` the spec carries a graph fingerprint whose
    JSON is cached by ``proposer.nas_bridge``. We reload it, re-dimension to the
    requested grading ``dim``, and compile it to an (B,L,D)->(B,L,D) module. The
    bridge already compile-tested the graph at this dim, so this is the same
    deterministic operation; a missing cache is a hard error (fail loud).
    """
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


def _dispatch_block_template(
    math_axes: dict[str, Any], *, dim: int, top_k_frac: float
) -> nn.Module | None:
    """Build a block-template module when ``op_block_template`` is set.

    Block templates compose multiple lanes around the anchor's primitive.
    The anchor's primitive becomes the inner mixer (via a factory closed
    over the anchor's axes minus ``op_block_template`` to break the
    dispatch recursion). The auxiliary lanes (attn, ssm, wavelet) are
    fixed-class baselines chosen for inductive-bias diversity.
    """
    template = str(math_axes.get("op_block_template") or "")
    if not template or template not in BLOCK_TEMPLATES:
        return None
    inner_axes = dict(math_axes)
    inner_axes.pop("op_block_template", None)
    # Day-5+: allow one level of block nesting. If op_block_inner_template
    # is set, the inner anchor IS another block (e.g. gated_parallel
    # whose anchor slot is a latent_compress block). Breaks recursion
    # after one nesting level — set op_block_inner_template only on the
    # outer spec, never on a nested spec generated by this dispatcher.
    inner_block = str(math_axes.get("op_block_inner_template") or "")
    if inner_block and inner_block in BLOCK_TEMPLATES:
        inner_axes["op_block_template"] = inner_block
        inner_axes.pop("op_block_inner_template", None)

    def anchor_factory(d: int) -> nn.Module:
        return generate_module(inner_axes, dim=d, top_k_frac=top_k_frac)

    slot_b_name = str(math_axes.get("op_block_slot_b") or "")
    slot_c_name = str(math_axes.get("op_block_slot_c") or "")

    if template == "latent_compress":
        compress = int(math_axes.get("op_block_compress") or 2)
        return LatentCompressBlock(anchor_factory, dim, compress=compress)
    if template == "three_lane_adaptive":
        slot_b = _block_slot_factory(slot_b_name or "tropical_attention")
        slot_c = _block_slot_factory(slot_c_name or "linear_state_space")
        return ThreeLaneAdaptive(anchor_factory, slot_b, slot_c, dim)
    if template == "recursive_depth":
        max_depth = int(math_axes.get("op_max_depth") or 3)
        return RecursiveDepthBlock(anchor_factory, dim, max_depth=max_depth)
    if template == "gated_parallel":
        slot_b = _block_slot_factory(slot_b_name or "multiscale_wavelet")
        return GatedParallelBlock(anchor_factory, slot_b, dim)
    if template == "recursive_depth_router":
        max_depth = int(math_axes.get("op_max_depth") or 4)
        return RecursiveDepthRouterBlock(anchor_factory, dim, max_depth=max_depth)
    if template == "sparse_moe_block":
        top_k = int(math_axes.get("op_top_k") or 2)
        return SparseMoEBlock(
            anchor_factory,
            _expert_factory_pool(top_k_frac),
            dim,
            top_k=top_k,
        )
    if template == "hetero_moe_block":
        # 4 heterogeneous experts: attn, ssm, top-k, wavelet
        def hetero_attn(d: int) -> nn.Module:
            return TropicalAttention(d)

        def hetero_ssm(d: int) -> nn.Module:
            return LinearStateSpaceLane(d)

        def hetero_topk(d: int) -> nn.Module:
            k = max(1, int(round(d * top_k_frac)))
            return TopKLinear(d, d, k=k)

        def hetero_wavelet(d: int) -> nn.Module:
            return MultiscaleWaveletLane(d)

        return HeteroMoEBlock(
            anchor_factory,
            (hetero_attn, hetero_ssm, hetero_topk, hetero_wavelet),
            dim,
        )
    if template == "hyperbolic_bridge":
        return HyperbolicBridgeBlock(anchor_factory, dim)
    if template == "attn_spectral_filter":
        return AttnSpectralFilterBlock(anchor_factory, dim)
    if template == "graph_attention":
        return GraphAttentionBlock(anchor_factory, dim)
    if template == "top_ar_block":
        # 2026-05-19: dual-mixer scaffold from fp 7fb0412ec57a1213 (top
        # AR-curriculum scorer at 0.9046 / passes all 5 stages).
        # MIXER_A = anchor (from inner_axes / op_block_inner_template),
        # MIXER_B = slot_b (default local_window_attn matches the source fp).
        from component_fab.harness.top_ar_block import TopArchBlock

        slot_b = _block_slot_factory(slot_b_name or "local_window_attn")
        return TopArchBlock(dim, anchor_factory, slot_b)
    return None


def _block_slot_factory(name: str) -> "callable":
    """Map a slot name to a fresh-instance factory. Used by block
    templates to fill non-anchor lane slots. New slot kinds register
    here without touching the templates themselves.
    """

    def _two_lane_ts(d: int) -> nn.Module:
        return GatedParallelBlock(
            lambda dd: TropicalAttention(dd),
            lambda dd: SparsemaxAttention(dd),
            d,
        )

    def _three_lane_tsw(d: int) -> nn.Module:
        return ThreeLaneAdaptive(
            lambda dd: TropicalAttention(dd),
            lambda dd: SparsemaxAttention(dd),
            lambda dd: MultiscaleWaveletLane(dd),
            d,
        )

    def _local_window(d: int) -> nn.Module:
        from component_fab.harness.top_ar_block import LocalWindowAttention

        return LocalWindowAttention(d, window_size=16)

    table = {
        "tropical_attention": TropicalAttention,
        "sparsemax_attention": SparsemaxAttention,
        "clifford_attention": lambda d: (
            CliffordAttention(d) if d % 4 == 0 else nn.Linear(d, d)
        ),
        "linear_state_space": LinearStateSpaceLane,
        "multiscale_wavelet": MultiscaleWaveletLane,
        "fourier_basis": FourierBasisLane,
        "graph_diffusion": GraphDiffusionLane,
        "fisher_attention": FisherAttention,
        "chebyshev_spectral": ChebyshevSpectralLane,
        "tucker_decomp": TuckerDecompLane,
        "quaternion": lambda d: (
            QuaternionAttention(d) if d % 4 == 0 else nn.Linear(d, d)
        ),
        "poincare": PoincareAttention,
        "random_features": RandomFeatureKernelLane,
        "low_rank": LowRankFactorizedLane,
        # Composite fab winners — usable as a slot fill inside ANY block template
        # so e.g. top_ar_block's MIXER_A slot can be filled with the 2-lane.
        "tropical_sparsemax_two_lane": _two_lane_ts,
        "tropical_sparsemax_wavelet_three_lane": _three_lane_tsw,
        # Top-AR scaffold's MIXER_B default (parameter-free local-window attention).
        "local_window_attn": _local_window,
    }
    ctor = table.get(name, MultiscaleWaveletLane)

    def factory(dim: int) -> nn.Module:
        return ctor(dim)

    return factory


# Default routing chain (math_knobs absent). math_knob fires first because
# math_family names a concrete operator mechanism; the base algebra/sparsity
# chain is the fallback. Replaces a tuple of 2 lambdas built per call.
_DEFAULT_DISPATCHERS: tuple = (
    _dispatch_math_knob,
    _base_module,
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
    invention = _dispatch_invention_mechanism(math_axes, dim=dim, top_k_frac=top_k_frac)
    if invention is not None:
        return _apply_routing_wrap(invention, math_axes, dim=dim, top_k_frac=top_k_frac)
    if math_axes.get("op_math_knobs") is not None:
        base = _base_module(math_axes, dim=dim, top_k_frac=top_k_frac)
        wrapped = _apply_math_knobs(base, math_axes, dim=dim, top_k_frac=top_k_frac)
        return _apply_routing_wrap(wrapped, math_axes, dim=dim, top_k_frac=top_k_frac)
    for dispatcher in _DEFAULT_DISPATCHERS:
        result = dispatcher(math_axes, dim=dim, top_k_frac=top_k_frac)
        if result is not None:
            return _apply_routing_wrap(
                result, math_axes, dim=dim, top_k_frac=top_k_frac
            )
    raise RuntimeError("unreachable module dispatch state")


def generate_module_from_spec(
    spec: ProposalSpec, *, dim: int = 32, top_k_frac: float = 0.25
) -> nn.Module:
    return generate_module(spec.math_axes, dim=dim, top_k_frac=top_k_frac)
