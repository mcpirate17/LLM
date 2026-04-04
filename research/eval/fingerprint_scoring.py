"""Scoring helpers for behavioral fingerprints."""

from __future__ import annotations

import math
from typing import Dict, Optional

from .fingerprint_types import (
    BEHAVIOR_SIGNATURE_WEIGHT,
    CKA_NOVELTY_WEIGHT,
    BehavioralFingerprint,
    NOVELTY_REFERENCE_SCHEME_VERSION,
)

_FEATURE_BASELINES: Dict[str, tuple[float, float]] = {
    "interaction_locality": (0.35, 0.20),
    "interaction_sparsity": (0.25, 0.20),
    "interaction_symmetry": (0.40, 0.25),
    "interaction_hierarchy": (0.15, 0.15),
    "isotropy": (0.15, 0.12),
    "rank_ratio": (0.40, 0.20),
    "sensitivity_uniformity": (0.35, 0.20),
    "hierarchy_fitness": (0.08, 0.10),
    "routing_selectivity": (0.30, 0.20),
    "routing_compute_ratio": (0.50, 0.25),
    "routing_lane_correlation": (0.20, 0.15),
}

_FEATURE_NAMES_BASE = [
    "interaction_locality",
    "interaction_sparsity",
    "interaction_symmetry",
    "interaction_hierarchy",
    "isotropy",
    "rank_ratio",
    "sensitivity_uniformity",
    "hierarchy_fitness",
]

_FEATURE_NAMES_ROUTING = [
    "routing_selectivity",
    "routing_compute_ratio",
    "routing_lane_correlation",
]


def build_novelty_reference_version(
    cka_source: Optional[str],
    cka_artifact_version: Optional[str],
    cka_probe_protocol_hash: Optional[str],
) -> str:
    source = str(cka_source or "none")
    artifact = str(cka_artifact_version or "none")
    probe = str(cka_probe_protocol_hash or "none")
    return f"{NOVELTY_REFERENCE_SCHEME_VERSION}:{source}:{artifact}:{probe}"


def sanitize_unit_feature(value: float) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(val):
        return 0.5
    return min(1.0, max(0.0, val))


def behavior_signature_score(fp: BehavioralFingerprint) -> float:
    feature_names = list(_FEATURE_NAMES_BASE)
    candidates = [
        fp.interaction_locality,
        fp.interaction_sparsity,
        fp.interaction_symmetry,
        fp.interaction_hierarchy,
        fp.isotropy,
        fp.rank_ratio,
        fp.sensitivity_uniformity,
        fp.hierarchy_fitness,
    ]
    if fp.routing_telemetry_present:
        feature_names.extend(_FEATURE_NAMES_ROUTING)
        candidates.extend(
            [
                fp.routing_selectivity,
                fp.routing_compute_ratio,
                fp.routing_lane_correlation,
            ]
        )
    pairs = [
        (name, value)
        for name, value in zip(feature_names, candidates)
        if value is not None
    ]
    if not pairs:
        return 0.0

    total = 0.0
    for name, raw_value in pairs:
        value = sanitize_unit_feature(raw_value)
        mean, std = _FEATURE_BASELINES.get(name, (0.5, 0.25))
        distinctiveness = abs(value - mean) / max(std, 0.05)
        total += min(1.0, max(0.0, distinctiveness))
    return float(total / len(pairs))


def cka_distance_novelty(fp: BehavioralFingerprint) -> float:
    cka_t = fp.cka_vs_transformer if fp.cka_vs_transformer is not None else 0.0
    cka_s = fp.cka_vs_ssm if fp.cka_vs_ssm is not None else 0.0
    cka_c = fp.cka_vs_conv if fp.cka_vs_conv is not None else 0.0
    return 1.0 - max(cka_t, cka_s, cka_c, 0.01)


def blend_behavioral_novelty(fp: BehavioralFingerprint) -> float:
    if fp.cka_source in ("deferred", "degenerate") or fp.cka_vs_transformer is None:
        return behavior_signature_score(fp)
    return CKA_NOVELTY_WEIGHT * cka_distance_novelty(
        fp
    ) + BEHAVIOR_SIGNATURE_WEIGHT * behavior_signature_score(fp)
