"""Shared types and constants for behavioral fingerprinting."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Optional

NOVELTY_REFERENCE_SCHEME_VERSION = "nv1"
CKA_NOVELTY_WEIGHT = 0.75
BEHAVIOR_SIGNATURE_WEIGHT = 0.25


@dataclass(slots=True)
class BehavioralFingerprint:
    """Characterizes how a model behaves, not what it computes."""

    interaction_locality: Optional[float] = 0.0
    interaction_sparsity: Optional[float] = 0.0
    interaction_symmetry: Optional[float] = 0.0
    interaction_hierarchy: Optional[float] = 0.0

    intrinsic_dim: Optional[float] = 0.0
    isotropy: Optional[float] = 0.0
    rank_ratio: Optional[float] = 0.0

    jacobian_spectral_norm: Optional[float] = 0.0
    jacobian_effective_rank: Optional[float] = 0.0
    sensitivity_uniformity: Optional[float] = 0.0

    routing_selectivity: float = 0.0
    routing_compute_ratio: float = 0.0
    routing_lane_correlation: float = 0.0
    routing_telemetry_present: Optional[bool] = None

    cka_vs_transformer: Optional[float] = 0.0
    cka_vs_ssm: Optional[float] = 0.0
    cka_vs_conv: Optional[float] = 0.0

    hierarchy_fitness: float = 0.0
    gromov_delta: float = 0.0

    novelty_score: float = 0.0
    behavior_signature_score: float = 0.0

    cka_source: str = "none"
    cka_artifact_version: Optional[str] = None
    cka_probe_protocol_hash: Optional[str] = None
    cka_reference_quality: Optional[str] = None
    similarity_path: Optional[str] = None
    novelty_reference_version: Optional[str] = None
    novelty_valid_for_promotion: bool = False
    novelty_validity_reason: str = "missing_reference"

    fingerprint_completed_post_investigation: bool = False
    fingerprint_completion_timestamp: Optional[str] = None

    analyses_succeeded: int = 0
    quality: str = "none"

    def to_dict(self) -> Dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def summary(self) -> str:
        lines = [
            f"Novelty Score: {self.novelty_score:.3f}",
            f"Interaction: locality={self.interaction_locality:.2f}, "
            f"sparsity={self.interaction_sparsity:.2f}, "
            f"hierarchy={self.interaction_hierarchy:.2f}",
            f"Geometry: intrinsic_dim={self.intrinsic_dim:.1f}, "
            f"isotropy={self.isotropy:.3f}, rank_ratio={self.rank_ratio:.3f}",
            f"Sensitivity: jacobian_rank={self.jacobian_effective_rank:.1f}, "
            f"uniformity={self.sensitivity_uniformity:.3f}",
            f"CKA similarity: transformer={self.cka_vs_transformer:.3f}, "
            f"ssm={self.cka_vs_ssm:.3f}, conv={self.cka_vs_conv:.3f}",
        ]
        return "\n".join(lines)
