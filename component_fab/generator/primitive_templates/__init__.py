"""Novel primitive templates the fab can synthesize from ProposalSpecs.

Split from a single module into a package (god-file guardrail). Public API
is unchanged: import the same names from ``primitive_templates``.
"""

from ._core import *  # noqa: F401,F403
from ._lanes_a import *  # noqa: F401,F403
from ._lanes_b import *  # noqa: F401,F403

from ._core import (
    _disable_torch_compile,
    _cumsum_dim1_eager,
    _cummax_dim1_eager,
    _reciprocal_attn_logits,
    _QKVRopeAttentionBase,
    _causal_sparsemax,
    _pick_n_heads,
    _heads_for_head_dim,
)

__all__ = [
    "_disable_torch_compile",
    "_cumsum_dim1_eager",
    "_cummax_dim1_eager",
    "_reciprocal_attn_logits",
    "_QKVRopeAttentionBase",
    "_causal_sparsemax",
    "_pick_n_heads",
    "_heads_for_head_dim",
    "TropicalAttention",
    "SparsemaxAttention",
    "ReciprocalRankAttention",
    "PhaseLockAttention",
    "ReciprocalPrimaryRefine",
    "SparseReciprocalAttention",
    "SemiringReciprocalAttention",
    "HeteroSemiringReciprocalAttention",
    "AnisotropicSemiringReciprocalAttention",
    "FixedRankReciprocalAttention",
    "TemperedTropicalAttention",
    "TropicalStateSpace",
    "TopKLinear",
    "FourierBasisLane",
    "FiniteDifferenceCalculusLane",
    "LowRankFactorizedLane",
    "SparseBandedMatrixLane",
    "CalculusAugmentedLane",
    "LowRankAdapterLane",
    "SparseBandedAdapterLane",
    "RandomFeatureKernelLane",
    "MultiscaleWaveletLane",
    "GraphDiffusionLane",
    "RandomFeatureKernelAdapterLane",
    "MultiscaleWaveletAdapterLane",
    "GraphDiffusionAdapterLane",
    "CliffordAttention",
    "_SurrogateSpike",
    "SpikingActivationGate",
    "PadicProjection",
    "TropicalTopKStateSpace",
    "LinearStateSpaceLane",
    "FisherAttention",
    "ChebyshevSpectralLane",
    "TuckerDecompLane",
    "FisherAdapterLane",
    "ChebyshevAdapterLane",
    "TuckerAdapterLane",
    "QuaternionAttention",
    "PoincareAttention",
    "SymplecticResidualMixerLane",
]
