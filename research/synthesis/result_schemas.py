"""
Shared Result Schemas: Unified dataclasses for evaluation and synthesis results.
Consolidates BridgeResult, SandboxResult, and Fingerprint schemas.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class SandboxResult:
    """Result of sandbox evaluation (Stage 0 and 0.5)."""

    passed: bool = False
    stage: str = ""  # "compile", "forward", "backward", "stability"
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Timing (ms)
    compile_time_ms: float = 0.0
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0

    # Metrics
    param_count: int = 0
    peak_memory_mb: float = 0.0
    output_shape: Optional[str] = None

    # Gradient health
    grad_norm: float = 0.0
    has_nan_grad: bool = False
    has_zero_grad: bool = False
    has_nan_output: bool = False
    has_inf_output: bool = False

    # Numerical stability
    stability_score: float = 0.0  # 0-1, higher is more stable
    extreme_input_passed: bool = False
    random_input_passed: bool = False
    causality_passed: bool = True
    output_range: Optional[str] = None

    # Sparsity & routing telemetry
    activation_sparsity: float = 0.0
    dead_neuron_count: int = 0
    sparsity_report: Optional[Dict[str, Any]] = None
    routing_report: Optional[Dict[str, Any]] = None

    # Failure attribution
    failure_op: Optional[str] = (
        None  # Op name that caused the failure (parsed from traceback)
    )

    # Debug/Advanced
    kernel_timing: Optional[Dict[str, Any]] = None
    native_abi_probe: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FingerprintResult:
    """Result of behavioral fingerprinting (Stage 1)."""

    # CKA similarities
    cka_vs_transformer: float = 0.0
    cka_vs_ssm: float = 0.0
    cka_vs_conv: float = 0.0

    # Structural properties
    interaction_locality: float = 0.0
    interaction_sparsity: float = 0.0
    intrinsic_dim: float = 0.0
    isotropy: float = 0.0

    # Novelty scores
    structural_novelty: float = 0.0
    behavioral_novelty: float = 0.0
    overall_novelty: float = 0.0
    most_similar_to: str = ""


@dataclass
class BridgeResult:
    """Complete evaluation result from the research pipeline."""

    status: str  # "success", "error", "failed_sandbox"
    error: Optional[str] = None
    error_stage: Optional[str] = None

    # Graph info
    graph_fingerprint: Optional[str] = None
    n_ops: int = 0
    depth: int = 0
    n_params_estimate: int = 0
    has_gradient_path: bool = False

    # Metrics from Sandbox
    sandbox: SandboxResult = field(default_factory=SandboxResult)

    # Metrics from Fingerprint
    fingerprint: FingerprintResult = field(default_factory=FingerprintResult)

    # Efficiency
    compression_ratio: float = 1.0
    pruning_tolerance: float = 0.0
    sparse_op_coverage: float = 0.0
    triton_compatibility_score: float = 0.0
    efficiency_score: float = 0.0

    # Timing
    total_time_ms: float = 0.0

    @property
    def param_count(self) -> int:
        """Backward-compatible alias used by older bridge consumers/tests."""
        return int(self.n_params_estimate or 0)

    def to_dict(self) -> Dict[str, Any]:
        """Flatten nested results for backward compatibility with designer API."""
        d = {
            "status": self.status,
            "error": self.error,
            "error_stage": self.error_stage,
            "graph_fingerprint": self.graph_fingerprint,
            "n_ops": self.n_ops,
            "depth": self.depth,
            "n_params_estimate": self.n_params_estimate,
            "has_gradient_path": self.has_gradient_path,
            "compression_ratio": self.compression_ratio,
            "pruning_tolerance": self.pruning_tolerance,
            "sparse_op_coverage": self.sparse_op_coverage,
            "triton_compatibility_score": self.triton_compatibility_score,
            "efficiency_score": self.efficiency_score,
            "total_time_ms": self.total_time_ms,
        }

        # Flatten sandbox
        s_dict = self.sandbox.to_dict()
        d.update(
            {
                "sandbox_passed": s_dict["passed"],
                "compile_time_ms": s_dict["compile_time_ms"],
                "forward_time_ms": s_dict["forward_time_ms"],
                "backward_time_ms": s_dict["backward_time_ms"],
                "param_count": s_dict["param_count"],
                "peak_memory_mb": s_dict["peak_memory_mb"],
                "grad_norm": s_dict["grad_norm"],
                "stability_score": s_dict["stability_score"],
            }
        )
        if s_dict.get("routing_report"):
            d["routing_report"] = s_dict["routing_report"]
            for key, value in s_dict["routing_report"].items():
                d[f"routing_{key}"] = value

        # Flatten fingerprint
        f_dict = asdict(self.fingerprint)
        d.update(f_dict)

        # Ensure native types
        for k, v in d.items():
            if hasattr(v, "item"):
                d[k] = v.item()
        return d
