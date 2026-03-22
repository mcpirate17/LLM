"""
Behavioral Fingerprinting

Answers: "Is this genuinely novel, or just attention with extra steps?"

Computes behavioral fingerprints that characterize HOW a model processes
information, independent of specific weights:
- Token interaction patterns (attention-like? local? hierarchical?)
- Representation geometry (intrinsic dimensionality, isotropy)
- Input sensitivity (Jacobian spectrum)
- CKA similarity vs known architectures
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, fields
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from research.env import aria_core
from research.defaults import VOCAB_SIZE

logger = logging.getLogger(__name__)
NOVELTY_REFERENCE_SCHEME_VERSION = "nv1"
CKA_NOVELTY_WEIGHT = 0.75
BEHAVIOR_SIGNATURE_WEIGHT = 0.25

_SENSITIVITY_SKIP_COUNTS: Dict[str, int] = {}
_SENSITIVITY_SKIP_LAST_LOG_TS: float = 0.0
_SENSITIVITY_SKIP_LOG_INTERVAL_S: float = 60.0


def _record_sensitivity_skip(reason: str) -> None:
    """Track skipped sensitivity probes and emit a rate-limited debug summary."""
    global _SENSITIVITY_SKIP_LAST_LOG_TS
    key = str(reason or "unknown")
    _SENSITIVITY_SKIP_COUNTS[key] = _SENSITIVITY_SKIP_COUNTS.get(key, 0) + 1

    now = time.monotonic()
    if (now - _SENSITIVITY_SKIP_LAST_LOG_TS) < _SENSITIVITY_SKIP_LOG_INTERVAL_S:
        return

    _SENSITIVITY_SKIP_LAST_LOG_TS = now
    total = sum(_SENSITIVITY_SKIP_COUNTS.values())
    breakdown = ", ".join(
        f"{name}={count}" for name, count in sorted(_SENSITIVITY_SKIP_COUNTS.items())
    )
    logger.debug("Sensitivity probes skipped (%d total): %s", total, breakdown)


def get_sensitivity_skip_stats(reset: bool = False) -> Dict[str, object]:
    """Return aggregated sensitivity-skip counters for diagnostics."""
    global _SENSITIVITY_SKIP_LAST_LOG_TS
    by_reason = dict(_SENSITIVITY_SKIP_COUNTS)
    payload = {
        "total": int(sum(by_reason.values())),
        "by_reason": by_reason,
        "log_interval_seconds": _SENSITIVITY_SKIP_LOG_INTERVAL_S,
        "last_log_monotonic": _SENSITIVITY_SKIP_LAST_LOG_TS,
    }
    if reset:
        _SENSITIVITY_SKIP_COUNTS.clear()
        _SENSITIVITY_SKIP_LAST_LOG_TS = 0.0
    return payload


@dataclass(slots=True)
class BehavioralFingerprint:
    """Characterizes how a model behaves, not what it computes."""

    # Token interaction pattern (None when behavioral probes deferred)
    interaction_locality: Optional[float] = 0.0  # 0=global, 1=purely local
    interaction_sparsity: Optional[float] = 0.0  # 0=dense, 1=sparse attention
    interaction_symmetry: Optional[float] = 0.0  # 0=asymmetric, 1=symmetric
    interaction_hierarchy: Optional[float] = (
        0.0  # how hierarchical the interaction pattern is
    )

    # Representation geometry (None when behavioral probes deferred)
    intrinsic_dim: Optional[float] = 0.0  # estimated intrinsic dimensionality
    isotropy: Optional[float] = (
        0.0  # how uniformly directions are used (0=collapsed, 1=isotropic)
    )
    rank_ratio: Optional[float] = 0.0  # effective rank / full rank

    # Input sensitivity (None when behavioral probes deferred)
    jacobian_spectral_norm: Optional[float] = 0.0
    jacobian_effective_rank: Optional[float] = 0.0
    sensitivity_uniformity: Optional[float] = (
        0.0  # how uniformly sensitive to each input token
    )

    # Routing-specific dimensions (Task 2H)
    routing_selectivity: float = 0.0  # std of difficulty scores
    routing_compute_ratio: float = 0.0  # slow/fast FLOP ratio
    routing_lane_correlation: float = 0.0  # position/content correlation
    routing_telemetry_present: Optional[bool] = (
        None  # None=unknown, True/False=explicit
    )

    # Similarity to known architectures (CKA) — None when deferred
    cka_vs_transformer: Optional[float] = 0.0
    cka_vs_ssm: Optional[float] = 0.0
    cka_vs_conv: Optional[float] = 0.0

    # Hierarchy detection (Gromov delta-hyperbolicity)
    hierarchy_fitness: float = 0.0  # 0=flat/Euclidean, 1=very tree-like
    gromov_delta: float = 0.0  # raw Gromov 4-point delta

    # Overall novelty estimate
    novelty_score: float = 0.0
    behavior_signature_score: float = 0.0

    # CKA provenance
    cka_source: str = "none"  # "artifact", "heuristic_fallback", "none"
    cka_artifact_version: Optional[str] = None
    cka_probe_protocol_hash: Optional[str] = None
    cka_reference_quality: Optional[str] = None
    similarity_path: Optional[str] = None
    novelty_reference_version: Optional[str] = None
    novelty_valid_for_promotion: bool = False
    novelty_validity_reason: str = "missing_reference"

    # Post-investigation fingerprint completion tracking
    fingerprint_completed_post_investigation: bool = False
    fingerprint_completion_timestamp: Optional[str] = None

    # Quality tracking: how many of the 4 sub-analyses succeeded
    analyses_succeeded: int = 0  # 0–4
    quality: str = "none"  # "full" (4/4), "partial" (1–3), "none" (0)

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


def compute_fingerprint(
    model: nn.Module,
    seq_len: int = 64,
    model_dim: int = 256,
    vocab_size: int = VOCAB_SIZE,
    device: str = "cuda",
    n_probes: int = 32,
    *,
    include_cka: bool = True,
    include_behavioral_probes: bool = True,
) -> BehavioralFingerprint:
    """Compute behavioral fingerprint for a model.

    Args:
        include_cka: When False, CKA fields are set to None and
            novelty_valid_for_promotion=False with cka_source="deferred".
        include_behavioral_probes: When False, Jacobian/perturbation/SVD
            probes are skipped; affected fields are set to None.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    fp = BehavioralFingerprint()

    n_succeeded = 0

    with torch.no_grad():
        # Generate probe inputs
        probe_ids = torch.randint(0, vocab_size, (n_probes, seq_len), device=dev)

        # Get intermediate representations
        reps = _get_representations(model, probe_ids, dev)

        if reps is not None and len(reps) > 0:
            if include_behavioral_probes:
                # Token interaction pattern
                interaction = _analyze_interactions(
                    model, probe_ids, dev, seq_len, vocab_size
                )
                fp.interaction_locality = interaction["locality"]
                fp.interaction_sparsity = interaction["sparsity"]
                fp.interaction_symmetry = interaction["symmetry"]
                fp.interaction_hierarchy = interaction["hierarchy"]
                if interaction.get("_succeeded"):
                    n_succeeded += 1

                # Representation geometry
                geometry = _analyze_geometry(reps)
                fp.intrinsic_dim = geometry["intrinsic_dim"]
                fp.isotropy = geometry["isotropy"]
                fp.rank_ratio = geometry["rank_ratio"]
                if geometry.get("_succeeded"):
                    n_succeeded += 1
            else:
                # Behavioral probes deferred — set fields to None
                fp.interaction_locality = None
                fp.interaction_sparsity = None
                fp.interaction_symmetry = None
                fp.interaction_hierarchy = None
                fp.intrinsic_dim = None
                fp.isotropy = None
                fp.rank_ratio = None

            # Hierarchy detection (Gromov delta-hyperbolicity)
            # Structural — always computed when reps available
            try:
                from .hierarchy_probe import hierarchy_fitness as _hf

                hf_result = _hf(reps, max_tokens=100)
                fp.hierarchy_fitness = hf_result["hierarchy_fitness"]
                fp.gromov_delta = hf_result["gromov_delta"]
            except Exception:
                pass

        if include_behavioral_probes:
            # Input sensitivity (Jacobian analysis)
            sensitivity = _analyze_sensitivity(model, dev, seq_len, vocab_size)
            fp.jacobian_spectral_norm = sensitivity["spectral_norm"]
            fp.jacobian_effective_rank = sensitivity["effective_rank"]
            fp.sensitivity_uniformity = sensitivity["uniformity"]
            if sensitivity.get("_succeeded"):
                n_succeeded += 1
        else:
            fp.jacobian_spectral_norm = None
            fp.jacobian_effective_rank = None
            fp.sensitivity_uniformity = None

        # Routing-aware analysis (Task 2H)
        try:
            routing_data = _analyze_routing(model, probe_ids, dev)
            fp.routing_selectivity = routing_data["selectivity"]
            fp.routing_compute_ratio = routing_data["compute_ratio"]
            fp.routing_lane_correlation = routing_data["lane_correlation"]
            # _analyze_routing returns all-zero defaults when model has no routing ops
            fp.routing_telemetry_present = routing_data.get("_has_routing", False)
        except Exception as e_route:
            logger.debug("Routing analysis skipped: %s", e_route)
            fp.routing_telemetry_present = False

        cka_all_zero = False

        if include_cka:
            # CKA similarity to reference architectures
            # Try artifact-backed CKA first, fall back to heuristic
            from .cka_references import get_default_store

            store = get_default_store()
            ref_activations = store.get_references()
            cka_meta = store.get_metadata()

            cka = _compute_reference_cka(reps, ref_activations=ref_activations)
            fp.cka_vs_transformer = cka.get("transformer", 0.0)
            fp.cka_vs_ssm = cka.get("ssm", 0.0)
            fp.cka_vs_conv = cka.get("conv", 0.0)
            fp.cka_source = cka_meta.get("cka_source", "none")
            fp.cka_artifact_version = cka_meta.get("cka_artifact_version")
            fp.cka_probe_protocol_hash = cka_meta.get("cka_probe_protocol_hash")
            fp.similarity_path = cka_meta.get(
                "cka_similarity_path", "_compute_reference_cka"
            )
            # Sanity gate: if all CKA scores are near-zero, the computation
            # likely failed (e.g. device mismatch, degenerate reps). Mark as
            # low-quality rather than trusting the manifest blindly.
            all_near_zero = (
                fp.cka_vs_transformer < 0.01
                and fp.cka_vs_ssm < 0.01
                and fp.cka_vs_conv < 0.01
            )
            if all_near_zero and cka.get("_succeeded"):
                logger.warning(
                    "CKA sanity gate: all three scores < 0.01 — marking "
                    "cka_reference_quality as false"
                )
                fp.cka_reference_quality = False
            else:
                fp.cka_reference_quality = cka_meta.get("cka_reference_quality")
            fp.novelty_reference_version = build_novelty_reference_version(
                fp.cka_source,
                fp.cka_artifact_version,
                fp.cka_probe_protocol_hash,
            )
            if fp.cka_source == "artifact":
                fp.novelty_valid_for_promotion = True
                fp.novelty_validity_reason = "artifact_reference"
            elif fp.cka_source == "heuristic_fallback":
                fp.novelty_valid_for_promotion = False
                fp.novelty_validity_reason = "heuristic_fallback_reference"
            else:
                fp.novelty_valid_for_promotion = False
                fp.novelty_validity_reason = "no_reference_available"

            # Degenerate CKA: all three scores near zero means the CKA computation
            # produced no usable signal. Invalidate regardless of cka_source or
            # _succeeded — arithmetic would yield novelty_score ≈ 1.0 from
            # _cka_distance_novelty = 1.0 - max(0,0,0,0.01) = 0.99.
            cka_scores = [fp.cka_vs_transformer, fp.cka_vs_ssm, fp.cka_vs_conv]
            cka_all_zero = all(abs(s) < 1e-6 for s in cka_scores)
            if cka_all_zero:
                fp.novelty_valid_for_promotion = False
                fp.novelty_validity_reason = "cka_degenerate_zeros"
                logger.warning(
                    "cka_degenerate_zeros: cka_scores=%s cka_source=%s quality=%s",
                    cka_scores,
                    fp.cka_source,
                    fp.quality,
                )

            # Kernel fallback invalidates promotion — C++ probes may have produced
            # degraded results that inflate or deflate novelty.
            from research.synthesis.compiler_op_utils import (
                kernel_fallback_occurred as _kf,
            )

            if _kf():
                fp.novelty_valid_for_promotion = False
                fp.novelty_validity_reason = (
                    fp.novelty_validity_reason + "|kernel_fallback"
                )
                logger.warning(
                    "novelty_invalidated_kernel_fallback: cka_source=%s quality=%s",
                    fp.cka_source,
                    fp.quality,
                )

            if cka.get("_succeeded"):
                n_succeeded += 1
        else:
            # CKA deferred — set fields to None, mark invalid for promotion
            fp.cka_vs_transformer = None
            fp.cka_vs_ssm = None
            fp.cka_vs_conv = None
            fp.cka_source = "deferred"
            fp.novelty_valid_for_promotion = False
            fp.novelty_validity_reason = "cka_deferred_post_investigation"

        fp.behavior_signature_score = _behavior_signature_score(fp)
        fp.novelty_score = _blend_behavioral_novelty(fp)

        # When CKA is degenerate, the blend formula produces ~0.99 from the
        # CKA distance term. Replace with behavior_signature_score alone so
        # novelty reflects only what was actually measured.
        if cka_all_zero:
            fp.novelty_score = fp.behavior_signature_score

    # Record analysis quality
    fp.analyses_succeeded = n_succeeded
    if n_succeeded == 4:
        fp.quality = "full"
    elif n_succeeded > 0:
        fp.quality = "partial"
    else:
        fp.quality = "none"

    model.train()
    return fp


def compute_structural_novelty_only(
    graph: "ComputationGraph",
) -> float:
    """Compute structural novelty score without behavioral probes or CKA.

    Reuses the existing structural analysis from metrics._novelty_score_from_ir.
    Returns a 0–1 score based on op diversity, category spread, and evenness.
    """
    from .metrics import _novelty_score_from_ir

    ir = graph.lower_to_ir()
    metrics = _novelty_score_from_ir(graph, ir, fingerprint=None)
    return float(metrics.structural_novelty)


def compute_lightning_fingerprint(
    model: nn.Module,
    seq_len: int = 64,
    model_dim: int = 256,
    device: str = "cpu",
    n_probes: int = 8,
    *,
    graph: Optional["ComputationGraph"] = None,
    structural_floor: float = 0.10,
) -> BehavioralFingerprint:
    """Lightning-fast structural-only fingerprint for pre-experiment gating.

    CKA is no longer computed at screening time — cold models produce
    degenerate CKA scores. Instead, the lightning gate uses structural
    novelty (op diversity, category spread, evenness) to reject obvious
    duplicates before investigation.

    If a ComputationGraph is provided, structural_novelty is computed
    directly from the graph IR (no forward pass needed for the gate
    decision). The forward pass still runs for hierarchy detection.
    """
    dev = torch.device(device)
    model = model.to(dev).eval()
    fp = BehavioralFingerprint()

    # Structural novelty from graph IR (if available)
    if graph is not None:
        fp.novelty_score = compute_structural_novelty_only(graph)
    else:
        fp.novelty_score = 0.0

    with torch.no_grad():
        torch.manual_seed(42)
        probe_ids = torch.randint(0, 32000, (n_probes, seq_len), device=dev)
        reps = _get_representations(model, probe_ids, dev)

        if reps is not None:
            # Hierarchy detection — structural, does not require convergence
            try:
                from .hierarchy_probe import hierarchy_fitness as _hf

                hf_result = _hf(reps, max_tokens=100)
                fp.hierarchy_fitness = hf_result["hierarchy_fitness"]
                fp.gromov_delta = hf_result["gromov_delta"]
            except Exception:
                pass

    # CKA deferred — not valid at screening time on cold models
    fp.cka_vs_transformer = None
    fp.cka_vs_ssm = None
    fp.cka_vs_conv = None
    fp.cka_source = "deferred"
    fp.novelty_valid_for_promotion = False
    fp.novelty_validity_reason = "cka_deferred_post_investigation"
    fp.behavior_signature_score = _behavior_signature_score(fp)
    fp.quality = "partial"
    fp.analyses_succeeded = 0

    return fp


def compute_gated_fingerprint(
    model: nn.Module,
    *,
    seq_len: int = 64,
    model_dim: int = 256,
    vocab_size: int = VOCAB_SIZE,
    device: str = "cpu",
    full_gate_enabled: bool = True,
    _lightning_novelty_threshold: float = 0.15,
    force_lightning_only: bool = False,
    graph: Optional["ComputationGraph"] = None,
    structural_floor: float = 0.10,
) -> Tuple[BehavioralFingerprint, bool]:
    """Run structural gating before the deferred fingerprint.

    With CKA and behavioral probes moved to post-investigation, the
    screening fingerprint only computes structural fields. The lightning
    gate rejects graphs with structural_novelty < structural_floor.
    """
    if not full_gate_enabled:
        return (
            compute_fingerprint(
                model,
                seq_len=seq_len,
                model_dim=model_dim,
                vocab_size=vocab_size,
                device=device,
                include_cka=False,
                include_behavioral_probes=False,
            ),
            True,
        )

    lightning_fp = compute_lightning_fingerprint(
        model,
        seq_len=seq_len,
        model_dim=model_dim,
        device=device,
        graph=graph,
        structural_floor=structural_floor,
    )

    # Reject below structural floor or when forced to lightning-only
    if force_lightning_only or float(lightning_fp.novelty_score or 0.0) < float(
        structural_floor
    ):
        return lightning_fp, False

    return (
        compute_fingerprint(
            model,
            seq_len=seq_len,
            model_dim=model_dim,
            vocab_size=vocab_size,
            device=device,
            include_cka=False,
            include_behavioral_probes=False,
        ),
        True,
    )


def complete_fingerprint_post_investigation(
    fp: BehavioralFingerprint,
    model: nn.Module,
    seq_len: int = 64,
    model_dim: int = 256,
    vocab_size: int = VOCAB_SIZE,
    device: str = "cuda",
    n_probes: int = 32,
) -> BehavioralFingerprint:
    """Complete a deferred fingerprint after investigation training.

    Takes a fingerprint built with include_cka=False, include_behavioral_probes=False
    and fills in the missing behavioral probes and CKA measurements using the
    post-investigation model that has converged representations.

    Idempotent: if fp.fingerprint_completed_post_investigation is already True,
    returns fp unchanged.
    """
    if fp.fingerprint_completed_post_investigation:
        return fp

    from datetime import datetime

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()

    with torch.no_grad():
        probe_ids = torch.randint(0, vocab_size, (n_probes, seq_len), device=dev)
        reps = _get_representations(model, probe_ids, dev)

    # Step 1 — Run behavioral probes
    if reps is not None and len(reps) > 0:
        # Token interaction pattern
        try:
            interaction = _analyze_interactions(
                model, probe_ids, dev, seq_len, vocab_size
            )
            fp.interaction_locality = interaction["locality"]
            fp.interaction_sparsity = interaction["sparsity"]
            fp.interaction_symmetry = interaction["symmetry"]
            fp.interaction_hierarchy = interaction["hierarchy"]
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("post_inv_interaction_probe_failed: %s", e)

        # Representation geometry
        try:
            geometry = _analyze_geometry(reps)
            fp.intrinsic_dim = geometry["intrinsic_dim"]
            fp.isotropy = geometry["isotropy"]
            fp.rank_ratio = geometry["rank_ratio"]
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("post_inv_geometry_probe_failed: %s", e)

    # Input sensitivity (Jacobian analysis)
    try:
        sensitivity = _analyze_sensitivity(model, dev, seq_len, vocab_size)
        fp.jacobian_spectral_norm = sensitivity["spectral_norm"]
        fp.jacobian_effective_rank = sensitivity["effective_rank"]
        fp.sensitivity_uniformity = sensitivity["uniformity"]
    except (ImportError, RuntimeError, AttributeError) as e:
        logger.warning("post_inv_sensitivity_probe_failed: %s", e)

    # Step 2 — Run CKA
    with torch.no_grad():
        from .cka_references import get_default_store

        store = get_default_store()
        ref_activations = store.get_references()
        cka_meta = store.get_metadata()

        cka = _compute_reference_cka(reps, ref_activations=ref_activations)
        cka_t = cka.get("transformer", 0.0)
        cka_s = cka.get("ssm", 0.0)
        cka_c = cka.get("conv", 0.0)

        # Degenerate CKA check
        if all(abs(s) < 1e-6 for s in [cka_t, cka_s, cka_c]):
            fp.cka_vs_transformer = None
            fp.cka_vs_ssm = None
            fp.cka_vs_conv = None
            fp.cka_source = "degenerate"
            fp.novelty_valid_for_promotion = False
            fp.novelty_validity_reason = "cka_degenerate_zeros"
            logger.warning(
                "cka_degenerate_zeros_post_investigation: cka_scores=[%.6f,%.6f,%.6f]",
                cka_t,
                cka_s,
                cka_c,
            )
        else:
            fp.cka_vs_transformer = cka_t
            fp.cka_vs_ssm = cka_s
            fp.cka_vs_conv = cka_c
            fp.cka_source = cka_meta.get("cka_source", "artifact")
            fp.cka_artifact_version = cka_meta.get("cka_artifact_version")
            fp.cka_probe_protocol_hash = cka_meta.get("cka_probe_protocol_hash")
            fp.cka_reference_quality = cka_meta.get("cka_reference_quality")
            fp.similarity_path = cka_meta.get(
                "cka_similarity_path", "_compute_reference_cka"
            )
            fp.novelty_reference_version = build_novelty_reference_version(
                fp.cka_source,
                fp.cka_artifact_version,
                fp.cka_probe_protocol_hash,
            )
            if fp.cka_source == "artifact":
                fp.novelty_valid_for_promotion = True
                fp.novelty_validity_reason = "artifact_reference_post_investigation"
            elif fp.cka_source == "heuristic_fallback":
                fp.novelty_valid_for_promotion = False
                fp.novelty_validity_reason = "heuristic_fallback_reference"
            else:
                fp.novelty_valid_for_promotion = False
                fp.novelty_validity_reason = "no_reference_available"

    # Step 3 — Recompute novelty blend with real values
    fp.behavior_signature_score = _behavior_signature_score(fp)
    if fp.cka_vs_transformer is not None:
        cka_distance = 1.0 - max(fp.cka_vs_transformer, fp.cka_vs_ssm, fp.cka_vs_conv)
        fp.novelty_score = (
            CKA_NOVELTY_WEIGHT * cka_distance
            + BEHAVIOR_SIGNATURE_WEIGHT * fp.behavior_signature_score
        )
    else:
        # CKA degenerate — use behavior_signature_score alone
        fp.novelty_score = fp.behavior_signature_score

    # Step 4 — Mark completion
    fp.fingerprint_completed_post_investigation = True
    fp.fingerprint_completion_timestamp = datetime.utcnow().isoformat()

    # Update quality tracking
    n_succeeded = 0
    if fp.interaction_locality is not None and fp.interaction_locality != 0.0:
        n_succeeded += 1
    if fp.intrinsic_dim is not None and fp.intrinsic_dim != 0.0:
        n_succeeded += 1
    if fp.jacobian_spectral_norm is not None and fp.jacobian_spectral_norm != 0.0:
        n_succeeded += 1
    if fp.cka_vs_transformer is not None:
        n_succeeded += 1
    fp.analyses_succeeded = n_succeeded
    if n_succeeded == 4:
        fp.quality = "full"
    elif n_succeeded > 0:
        fp.quality = "partial"
    else:
        fp.quality = "none"

    model.train()
    return fp


def build_novelty_reference_version(
    cka_source: Optional[str],
    cka_artifact_version: Optional[str],
    cka_probe_protocol_hash: Optional[str],
) -> str:
    """Stable version id used to compare novelty across time."""
    source = str(cka_source or "none")
    art = str(cka_artifact_version or "none")
    probe = str(cka_probe_protocol_hash or "none")
    return f"{NOVELTY_REFERENCE_SCHEME_VERSION}:{source}:{art}:{probe}"


def _sanitize_unit_feature(value: float) -> float:
    try:
        val = float(value)
    except Exception:
        return 0.5
    if not math.isfinite(val):
        return 0.5
    return min(1.0, max(0.0, val))


def _behavior_signature_score(fp: BehavioralFingerprint) -> float:
    """Bounded non-CKA distinctiveness signal from behavioral probes.

    Routing dimensions are excluded when routing telemetry is absent to avoid
    inflating novelty for non-routing architectures (default 0.0 maps to
    maximum distinctiveness under the abs(v-0.5)*2 formula).

    Fields set to None (deferred probes) are excluded from the average.
    """
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
    # Only include routing dims when telemetry was actually collected
    if fp.routing_telemetry_present:
        candidates.extend(
            [
                fp.routing_selectivity,
                fp.routing_compute_ratio,
                fp.routing_lane_correlation,
            ]
        )
    # Filter out None (deferred behavioral probes)
    features = [v for v in candidates if v is not None]
    if not features:
        return 0.0
    sanitized = [_sanitize_unit_feature(v) for v in features]
    return float(sum(abs(v - 0.5) * 2.0 for v in sanitized) / len(sanitized))


def _cka_distance_novelty(fp: BehavioralFingerprint) -> float:
    cka_t = fp.cka_vs_transformer if fp.cka_vs_transformer is not None else 0.0
    cka_s = fp.cka_vs_ssm if fp.cka_vs_ssm is not None else 0.0
    cka_c = fp.cka_vs_conv if fp.cka_vs_conv is not None else 0.0
    max_cka = max(cka_t, cka_s, cka_c, 0.01)
    return 1.0 - max_cka


def _blend_behavioral_novelty(fp: BehavioralFingerprint) -> float:
    # When CKA is deferred or degenerate, use behavior_signature_score alone
    # to avoid inflating novelty from the 1.0 - max(0,0,0) = 0.99 artifact.
    if fp.cka_source in ("deferred", "degenerate") or fp.cka_vs_transformer is None:
        return _behavior_signature_score(fp)
    return CKA_NOVELTY_WEIGHT * _cka_distance_novelty(
        fp
    ) + BEHAVIOR_SIGNATURE_WEIGHT * _behavior_signature_score(fp)


def _get_representations(
    model: nn.Module, input_ids: torch.Tensor, dev: torch.device
) -> Optional[torch.Tensor]:
    """Get output representations from a model."""
    try:
        logits = model(input_ids)
        return logits
    except Exception as e:
        logger.warning("Failed to get representations: %s", e)
        return None


def _analyze_interactions(
    model: nn.Module,
    input_ids: torch.Tensor,
    dev: torch.device,
    seq_len: int,
    vocab_size: int = VOCAB_SIZE,
) -> Dict[str, float]:
    """Analyze token-to-token interaction patterns."""
    result = {
        "locality": 0.5,
        "sparsity": 0.5,
        "symmetry": 0.5,
        "hierarchy": 0.5,
        "_succeeded": False,
    }

    try:
        input_ids.shape[0]
        # Compute per-token influence by masking
        # Use a single sample for efficiency
        ids = input_ids[:1]
        n_positions = min(8, seq_len)
        positions = torch.linspace(0, seq_len - 1, n_positions, device=dev).long()
        influence_matrix = _interaction_influence_matrix(
            model, ids, positions, vocab_size=vocab_size
        )
        result.update(_interaction_metrics(influence_matrix, positions))
        result["_succeeded"] = True

    except Exception as e:
        logger.warning("Interaction analysis failed: %s", e)

    return result


def _interaction_influence_matrix(
    model: nn.Module,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    """Return the perturbation influence matrix for selected token positions."""
    ids = input_ids[:1]
    base_out = model(ids)
    n_positions = int(positions.numel())
    perturbed_batch = ids.expand(n_positions, -1).clone()
    row_idx = torch.arange(n_positions, device=positions.device)
    perturbed_batch[row_idx, positions] = (
        perturbed_batch[row_idx, positions] + 1
    ) % vocab_size
    return (model(perturbed_batch) - base_out).abs().mean(dim=-1)


def _interaction_metrics(
    influence_matrix: torch.Tensor,
    positions: torch.Tensor,
) -> Dict[str, float]:
    """Compute interaction metrics via native C++ kernel."""
    inf_det = influence_matrix.detach()
    pos_det = positions.detach()
    if inf_det.device.type != "cpu":
        inf_det = inf_det.cpu()
    if pos_det.device.type != "cpu":
        pos_det = pos_det.cpu()
    native = aria_core.interaction_metrics_f32(
        inf_det.contiguous(),
        pos_det.contiguous(),
    )
    return {
        "locality": float(native[0].item()),
        "sparsity": float(native[1].item()),
        "symmetry": float(native[2].item()),
        "hierarchy": float(native[3].item()),
    }


def _analyze_routing(
    model: nn.Module,
    input_ids: torch.Tensor,
    dev: torch.device,
) -> Dict[str, float]:
    """Analyze routing-specific behavior (Task 2H)."""
    result = {
        "selectivity": 0.0,
        "compute_ratio": 0.0,
        "lane_correlation": 0.0,
        "_has_routing": False,
    }

    # Identify if model has routing ops via its graph (if accessible)
    has_routing = False
    if hasattr(model, "graph") and model.graph is not None:
        from ..synthesis.grammar import _ROUTING_OPS

        for node in model.graph.nodes.values():
            if not node.is_input and node.op_name in _ROUTING_OPS:
                has_routing = True
                break

    if not has_routing:
        return result

    result["_has_routing"] = True
    try:
        # Extract routing telemetry from the model
        # Most routing models in Aria expose 'routing_stats' or similar after a forward pass
        with torch.no_grad():
            model(input_ids)

        # 1. Routing Selectivity (Std of difficulty/gate scores)
        # Higher selectivity = model is making sharp decisions about token paths
        if hasattr(model, "last_routing_scores"):
            scores = (
                model.last_routing_scores
            )  # Expected shape (B, S, n_lanes) or similar
            if isinstance(scores, torch.Tensor) and scores.numel() > 0:
                result["selectivity"] = float(scores.std().item())

        # 2. Routing Compute Ratio (slow/fast FLOP ratio)
        # Measures how much of the compute is dynamic vs static
        if hasattr(model, "get_routing_compute_stats"):
            stats = model.get_routing_compute_stats()
            # Expecting {'slow_flops': ..., 'fast_flops': ...} or similar
            slow = stats.get("slow_flops", 0)
            fast = stats.get("fast_flops", 1)  # avoid div by zero
            result["compute_ratio"] = float(slow / max(fast, 1e-6))
        elif hasattr(model, "routing_compute_ratio"):
            result["compute_ratio"] = float(model.routing_compute_ratio)

        # 3. Routing Lane Correlation (Position vs Content correlation)
        # Do tokens at same positions always take same lanes? (Structural)
        # Or does it depend on content? (Content-aware)
        if hasattr(model, "last_routing_decisions"):
            # Shape (B, S) - lane indices
            decisions = model.last_routing_decisions
            if isinstance(decisions, torch.Tensor) and decisions.dim() >= 2:
                # Correlation of lane choice with position S
                B, S = decisions.shape[:2]
                positions = torch.arange(S, device=dev).float().expand(B, S)

                def pearson_corr(x, y):
                    mx, my = x.mean(), y.mean()
                    vx, vy = x - mx, y - my
                    return (vx * vy).sum() / (torch.norm(vx) * torch.norm(vy) + 1e-8)

                result["lane_correlation"] = float(
                    pearson_corr(decisions.float(), positions).item()
                )

    except Exception as e:
        logger.debug("Routing analysis failed: %s", e)

    return result


def _analyze_geometry(reps: torch.Tensor) -> Dict[str, float]:
    """Analyze the geometry of representation space."""
    result = {
        "intrinsic_dim": 0.0,
        "isotropy": 0.0,
        "rank_ratio": 0.0,
        "_succeeded": False,
    }

    try:
        # Flatten to (N, D)
        flat = reps.reshape(-1, reps.shape[-1]).float()
        N, D = flat.shape

        if N < 2 or D < 2:
            return result

        # Center
        flat = flat - flat.mean(dim=0, keepdim=True)

        # SVD for spectral analysis (use subset for efficiency)
        n_samples = min(N, 500)
        idx = torch.randperm(N)[:n_samples]
        subset = flat[idx]

        try:
            U, S, V = torch.linalg.svd(subset, full_matrices=False)
        except Exception as e:
            logger.debug("SVD failed in geometry analysis: %s", e)
            return result

        S = S.clamp(min=1e-10)

        # Intrinsic dimensionality (participation ratio)
        S_norm = S / S.sum()
        result["intrinsic_dim"] = (1.0 / (S_norm**2).sum()).item()

        # Isotropy: how uniform are the singular values?
        # Perfect isotropy: all singular values equal
        result["isotropy"] = (S.min() / S.max()).item()

        # Effective rank
        S_log = S_norm * torch.log(S_norm)
        entropy = -S_log.sum().item()
        result["rank_ratio"] = math.exp(entropy) / len(S)

        result["_succeeded"] = True

    except Exception as e:
        logger.warning("Geometry analysis failed: %s", e)

    return result


def _forward_model_from_embed(model: nn.Module, embed_in: torch.Tensor) -> torch.Tensor:
    """Run the model body starting from precomputed embeddings."""
    x_local = embed_in
    if hasattr(model, "pos_enc") and model.pos_enc is not None:
        x_local = model.pos_enc(x_local)
    if hasattr(model, "layers"):
        for layer in model.layers:
            x_local = layer(x_local)
        return x_local
    if hasattr(model, "topology"):
        return model.topology(x_local)
    return x_local


def _analyze_sensitivity(
    model: nn.Module,
    dev: torch.device,
    seq_len: int,
    vocab_size: int,
) -> Dict[str, float]:
    """Analyze input sensitivity via approximate Jacobian."""
    result = {
        "spectral_norm": 0.0,
        "effective_rank": 0.0,
        "uniformity": 0.0,
        "_succeeded": False,
    }

    try:
        model.eval()
        with torch.enable_grad():
            # Small batch for Jacobian estimation
            ids = torch.randint(0, vocab_size, (1, seq_len), device=dev)
            ids.requires_grad_(False)

            # Get embedding and make it require grad
            embed = model.embed(ids).detach().requires_grad_(True)

            def forward_from_embed(embed_in):
                return _forward_model_from_embed(model, embed_in)

            x = forward_from_embed(embed)

            if not x.requires_grad:
                _record_sensitivity_skip("output_no_grad")
                return result

            n_positions = max(1, min(4, seq_len))
            step = max(1, seq_len // n_positions)
            positions = torch.arange(0, seq_len, step, device=dev, dtype=torch.int64)[
                :n_positions
            ]
            sens_matrix = _collect_position_sensitivities(
                forward_from_embed, embed, positions
            )
            if sens_matrix is None:
                _record_sensitivity_skip("no_sensitivity_grads")
            if sens_matrix is not None:
                result.update(_sensitivity_metrics(sens_matrix))

            result["_succeeded"] = True

    except Exception as e:
        logger.warning("Sensitivity analysis failed: %s", e)

    return result


def _collect_position_sensitivities(
    x_or_forward: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    embed: torch.Tensor,
    positions: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Collect sensitivity rows from either a forward fn or a precomputed tensor."""
    n_pos = positions.numel()
    if n_pos == 0 or not embed.requires_grad:
        return None

    if callable(x_or_forward):
        forward_from_embed = x_or_forward
    else:
        x = x_or_forward
        try:
            grad_outputs = torch.zeros(n_pos, *x.shape, device=x.device, dtype=x.dtype)
            grad_outputs[
                torch.arange(n_pos, device=positions.device), :, positions, :
            ] = 1.0
            batched_grads = torch.autograd.grad(
                x,
                embed,
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,
                is_grads_batched=True,
            )[0]
            return batched_grads.norm(dim=-1).squeeze(1)
        except RuntimeError:

            def forward_from_embed(_):
                return x

    try:
        from torch.func import grad, vmap

        def probe_loss(embed_in: torch.Tensor, pos_idx: torch.Tensor) -> torch.Tensor:
            out = forward_from_embed(embed_in)
            return torch.index_select(out, 1, pos_idx.reshape(1)).sum()

        batched_grads = vmap(
            lambda pos_idx: grad(probe_loss, argnums=0)(embed, pos_idx)
        )(positions)
        return batched_grads.norm(dim=-1).squeeze(1)

    except (ImportError, RuntimeError):
        pass

    # Batched fallback: stack all position variants on the batch dim in a single
    # forward pass instead of N sequential passes.  vmap is unavailable here
    # (failed above), and the model may not be purely functional, so we replicate
    # the embedding along batch and run one forward call.
    n_pos = len(positions)
    embed_expanded = embed.expand(n_pos, *embed.shape[1:]).contiguous()
    embed_expanded.requires_grad_(True)
    out = forward_from_embed(embed_expanded)  # (n_pos, seq, hidden)
    # Select each position's output for the corresponding batch element
    selected = out[
        torch.arange(n_pos, device=positions.device), positions, :
    ]  # (n_pos, hidden)
    grad_out = torch.autograd.grad(
        selected.sum(),
        embed_expanded,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )[0]
    if grad_out is None:
        return None
    return grad_out.norm(dim=-1).squeeze(1)


def _sensitivity_metrics(sens_matrix: torch.Tensor) -> Dict[str, float]:
    """Compute sensitivity metrics via native C++ kernel."""
    sens_det = sens_matrix.detach()
    if sens_det.device.type != "cpu":
        sens_det = sens_det.cpu()
    native = aria_core.sensitivity_metrics_f32(sens_det.contiguous())
    return {
        "spectral_norm": float(native[0].item()),
        "uniformity": float(native[1].item()),
        "effective_rank": float(native[2].item()),
    }


def _compute_reference_cka(
    reps: Optional[torch.Tensor],
    ref_activations: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    """Compute CKA similarity to reference architecture behaviors.

    If ref_activations is provided (from artifact store), computes CKA
    against real pre-trained reference representations. Otherwise falls
    back to heuristic synthetic patterns.

    Args:
        reps: Candidate model representations.
        ref_activations: Optional dict mapping family name -> reference
            activation tensor from artifact store.
    """
    result = {"transformer": 0.0, "ssm": 0.0, "conv": 0.0, "_succeeded": False}

    if reps is None:
        return result

    try:
        # Compute self-similarity matrix of representations
        flat = reps[0].float() if reps.dim() > 2 else reps.float()
        S, D = flat.shape[-2], flat.shape[-1]
        if S < 4:
            return result

        norm = F.normalize(flat, dim=-1)
        sim = torch.mm(norm.reshape(-1, D), norm.reshape(-1, D).t())
        sim = sim[:S, :S]  # (S, S)

        if ref_activations is not None:
            # Artifact-backed CKA: compute against real reference activations
            for family in ("transformer", "ssm", "conv"):
                ref_tensor = ref_activations.get(family)
                if ref_tensor is None:
                    continue
                # Build reference self-similarity matrix, truncating/padding
                # to match candidate sequence length
                ref_flat = ref_tensor.to(device=flat.device).float()
                rS = ref_flat.shape[-2]
                use_S = min(S, rS)
                ref_norm = F.normalize(ref_flat[..., :use_S, :], dim=-1)
                rD = ref_norm.shape[-1]
                ref_sim = torch.mm(
                    ref_norm.reshape(-1, rD), ref_norm.reshape(-1, rD).t()
                )
                ref_sim = ref_sim[:use_S, :use_S]
                result[family] = _linear_cka(sim[:use_S, :use_S], ref_sim)
        else:
            # Heuristic fallback: synthetic reference patterns
            # CAVEAT: These are synthetic approximations, not empirical.
            positions = torch.arange(S, device=sim.device).float()
            dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()

            # Transformer: soft attention-like (slow decay from diagonal)
            ref_transformer = torch.exp(-dist / (S * 0.3))
            # SSM: recurrent (lower triangular with exponential decay)
            ref_ssm = torch.exp(-dist / (S * 0.15)) * (dist >= 0).float()
            ref_ssm = ref_ssm.tril()
            # Conv: local (sharp banded)
            ref_conv = (dist <= 5).float()

            result["transformer"] = _linear_cka(sim, ref_transformer)
            result["ssm"] = _linear_cka(sim, ref_ssm)
            result["conv"] = _linear_cka(sim, ref_conv)

        result["_succeeded"] = True

    except Exception as e:
        logger.warning("CKA computation failed: %s", e)

    return result


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA similarity."""
    try:
        # Optimization: use native aria_core kernel if on CPU
        if X.device.type == "cpu" and Y.device.type == "cpu":
            return aria_core.linear_cka_f32(X.contiguous(), Y.contiguous())

        X = X - X.mean()
        Y = Y - Y.mean()
        hsic_xy = (X * Y).sum()
        hsic_xx = (X * X).sum()
        hsic_yy = (Y * Y).sum()
        denom = (hsic_xx * hsic_yy).clamp(min=1e-30).sqrt()
        return (hsic_xy / denom).clamp(0, 1).item()
    except Exception as e:
        logger.debug("CKA computation error: %s", e)
        return 0.0
