"""Behavioral fingerprint orchestration and compatibility exports."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from research.defaults import VOCAB_SIZE

from .cka_references import get_default_store as _get_default_store
from .fingerprint_cka import (
    cka_from_tensor as _cka_from_tensor_impl,
    compute_reference_cka as _compute_reference_cka_impl,
)
from .fingerprint_probes import (
    analyze_geometry as _analyze_geometry_impl,
    analyze_hierarchy as _analyze_hierarchy,
    analyze_interactions as _analyze_interactions_impl,
    analyze_routing as _analyze_routing_impl,
    get_representations as _get_representations_impl,
    interaction_influence_matrix as _interaction_influence_matrix_impl,
)
from .fingerprint_scoring import (
    behavior_signature_score as _behavior_signature_score_impl,
    blend_behavioral_novelty as _blend_behavioral_novelty_impl,
    build_novelty_reference_version,
    cka_distance_novelty as _cka_distance_novelty_impl,
    sanitize_unit_feature as _sanitize_unit_feature_impl,
)
from .fingerprint_sensitivity import (
    analyze_sensitivity as _analyze_sensitivity_impl,
    collect_position_sensitivities as _collect_position_sensitivities_impl,
    forward_model_from_embed as _forward_model_from_embed_impl,
)
from .fingerprint_types import BehavioralFingerprint

logger = logging.getLogger(__name__)


def _populate_behavioral_probes(
    fp: BehavioralFingerprint,
    model: nn.Module,
    probe_ids: torch.Tensor,
    reps: Optional[torch.Tensor],
    device: torch.device,
    seq_len: int,
    vocab_size: int,
    include: bool,
) -> int:
    n_succeeded = 0

    if reps is not None and len(reps) > 0:
        if include:
            interaction = _analyze_interactions_impl(
                model,
                probe_ids,
                device,
                seq_len,
                vocab_size,
            )
            fp.interaction_locality = interaction["locality"]
            fp.interaction_sparsity = interaction["sparsity"]
            fp.interaction_symmetry = interaction["symmetry"]
            fp.interaction_hierarchy = interaction["hierarchy"]
            if interaction.get("_succeeded"):
                n_succeeded += 1

            geometry = _analyze_geometry_impl(reps)
            fp.intrinsic_dim = geometry["intrinsic_dim"]
            fp.isotropy = geometry["isotropy"]
            fp.rank_ratio = geometry["rank_ratio"]
            if geometry.get("_succeeded"):
                n_succeeded += 1
        else:
            fp.interaction_locality = None
            fp.interaction_sparsity = None
            fp.interaction_symmetry = None
            fp.interaction_hierarchy = None
            fp.intrinsic_dim = None
            fp.isotropy = None
            fp.rank_ratio = None

        try:
            hierarchy = _analyze_hierarchy(reps)
            fp.hierarchy_fitness = hierarchy["hierarchy_fitness"]
            fp.gromov_delta = hierarchy["gromov_delta"]
        except Exception as exc:
            logger.debug("Hierarchy probe skipped: %s", exc)

    if include:
        sensitivity = _analyze_sensitivity_impl(model, device, seq_len, vocab_size)
        fp.jacobian_spectral_norm = sensitivity["spectral_norm"]
        fp.jacobian_effective_rank = sensitivity["effective_rank"]
        fp.sensitivity_uniformity = sensitivity["uniformity"]
        if sensitivity.get("_succeeded"):
            n_succeeded += 1
    else:
        fp.jacobian_spectral_norm = None
        fp.jacobian_effective_rank = None
        fp.sensitivity_uniformity = None

    try:
        routing = _analyze_routing_impl(model, probe_ids, device)
        fp.routing_selectivity = routing["selectivity"]
        fp.routing_compute_ratio = routing["compute_ratio"]
        fp.routing_lane_correlation = routing["lane_correlation"]
        fp.routing_telemetry_present = routing.get("_has_routing", False)
    except Exception as exc:
        logger.debug("Routing analysis skipped: %s", exc)
        fp.routing_telemetry_present = False

    return n_succeeded


def _populate_cka(
    fp: BehavioralFingerprint,
    reps: Optional[torch.Tensor],
    include: bool,
) -> Tuple[int, bool]:
    if not include:
        fp.cka_vs_transformer = None
        fp.cka_vs_ssm = None
        fp.cka_vs_conv = None
        fp.cka_source = "deferred"
        fp.novelty_valid_for_promotion = False
        fp.novelty_validity_reason = "cka_deferred_post_investigation"
        return 0, False

    store = _get_default_store()
    cka = _compute_reference_cka_impl(
        reps,
        ref_activations=store.get_references(),
        ref_similarities=store.get_reference_similarities(),
    )
    cka_meta = store.get_metadata()
    fp.cka_vs_transformer = cka.get("transformer", 0.0)
    fp.cka_vs_ssm = cka.get("ssm", 0.0)
    fp.cka_vs_conv = cka.get("conv", 0.0)
    fp.cka_source = cka_meta.get("cka_source", "none")
    fp.cka_artifact_version = cka_meta.get("cka_artifact_version")
    fp.cka_probe_protocol_hash = cka_meta.get("cka_probe_protocol_hash")
    fp.similarity_path = cka_meta.get("cka_similarity_path", "_compute_reference_cka")

    all_near_zero = (
        fp.cka_vs_transformer < 0.01 and fp.cka_vs_ssm < 0.01 and fp.cka_vs_conv < 0.01
    )
    if all_near_zero and cka.get("_succeeded"):
        logger.warning(
            "CKA sanity gate: all three scores < 0.01 — marking cka_reference_quality as false"
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

    cka_scores = [fp.cka_vs_transformer, fp.cka_vs_ssm, fp.cka_vs_conv]
    cka_all_zero = all(abs(score) < 1e-6 for score in cka_scores)
    if cka_all_zero:
        fp.novelty_valid_for_promotion = False
        fp.novelty_validity_reason = "cka_degenerate_zeros"
        logger.warning(
            "cka_degenerate_zeros: cka_scores=%s cka_source=%s quality=%s",
            cka_scores,
            fp.cka_source,
            fp.quality,
        )

    from research.synthesis.compiler_op_utils import (
        kernel_fallback_occurred as _kernel_fallback_occurred,
    )

    if _kernel_fallback_occurred():
        fp.novelty_valid_for_promotion = False
        fp.novelty_validity_reason = fp.novelty_validity_reason + "|kernel_fallback"
        logger.warning(
            "novelty_invalidated_kernel_fallback: cka_source=%s quality=%s",
            fp.cka_source,
            fp.quality,
        )

    return (1 if cka.get("_succeeded") else 0), cka_all_zero


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
    del model_dim
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()
    fp = BehavioralFingerprint()

    with torch.no_grad():
        probe_ids = torch.randint(0, vocab_size, (n_probes, seq_len), device=dev)
        reps = _get_representations_impl(model, probe_ids)
        probe_succeeded = _populate_behavioral_probes(
            fp,
            model,
            probe_ids,
            reps,
            dev,
            seq_len,
            vocab_size,
            include=include_behavioral_probes,
        )
        cka_succeeded, cka_all_zero = _populate_cka(fp, reps, include=include_cka)
        fp.behavior_signature_score = _behavior_signature_score_impl(fp)
        fp.novelty_score = _blend_behavioral_novelty_impl(fp)
        if cka_all_zero:
            fp.novelty_score = fp.behavior_signature_score

    _set_quality(fp, probe_succeeded + cka_succeeded)
    model.train()
    return fp


def compute_structural_novelty_only(graph: "ComputationGraph") -> float:
    from .metrics import batch_novelty_scores

    metrics = batch_novelty_scores([graph], None)[0]
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
    del model_dim, structural_floor
    dev = torch.device(device)
    model = model.to(dev).eval()
    fp = BehavioralFingerprint()
    fp.novelty_score = (
        compute_structural_novelty_only(graph) if graph is not None else 0.0
    )

    with torch.no_grad():
        torch.manual_seed(42)
        probe_ids = torch.randint(0, 32000, (n_probes, seq_len), device=dev)
        reps = _get_representations_impl(model, probe_ids)
        if reps is not None:
            try:
                hierarchy = _analyze_hierarchy(reps)
                fp.hierarchy_fitness = hierarchy["hierarchy_fitness"]
                fp.gromov_delta = hierarchy["gromov_delta"]
            except Exception as exc:
                logger.debug("Hierarchy detection skipped: %s", exc)

    fp.cka_vs_transformer = None
    fp.cka_vs_ssm = None
    fp.cka_vs_conv = None
    fp.cka_source = "deferred"
    fp.novelty_valid_for_promotion = False
    fp.novelty_validity_reason = "cka_deferred_post_investigation"
    fp.behavior_signature_score = _behavior_signature_score_impl(fp)
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
    del _lightning_novelty_threshold
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
    del model_dim
    if fp.fingerprint_completed_post_investigation:
        return fp

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()

    with torch.no_grad():
        probe_ids = torch.randint(0, vocab_size, (n_probes, seq_len), device=dev)
        reps = _get_representations_impl(model, probe_ids)

    probe_succeeded = _populate_behavioral_probes(
        fp,
        model,
        probe_ids,
        reps,
        dev,
        seq_len,
        vocab_size,
        include=True,
    )
    cka_succeeded, cka_all_zero = _populate_cka(fp, reps, include=True)
    fp.behavior_signature_score = _behavior_signature_score_impl(fp)
    fp.novelty_score = _blend_behavioral_novelty_impl(fp)
    if cka_all_zero:
        fp.novelty_score = fp.behavior_signature_score

    fp.fingerprint_completed_post_investigation = True
    fp.fingerprint_completion_timestamp = datetime.utcnow().isoformat()
    _set_quality(fp, probe_succeeded + cka_succeeded)
    model.train()
    return fp


def _set_quality(fp: BehavioralFingerprint, analyses_succeeded: int) -> None:
    fp.analyses_succeeded = analyses_succeeded
    if analyses_succeeded == 4:
        fp.quality = "full"
    elif analyses_succeeded > 0:
        fp.quality = "partial"
    else:
        fp.quality = "none"


def _sanitize_unit_feature(value: float) -> float:
    return _sanitize_unit_feature_impl(value)


def _behavior_signature_score(fp: BehavioralFingerprint) -> float:
    return _behavior_signature_score_impl(fp)


def _cka_distance_novelty(fp: BehavioralFingerprint) -> float:
    return _cka_distance_novelty_impl(fp)


def _blend_behavioral_novelty(fp: BehavioralFingerprint) -> float:
    return _blend_behavioral_novelty_impl(fp)


def _get_representations(
    model: nn.Module, input_ids: torch.Tensor, dev: torch.device
) -> Optional[torch.Tensor]:
    del dev
    return _get_representations_impl(model, input_ids)


def _analyze_interactions(
    model: nn.Module,
    input_ids: torch.Tensor,
    dev: torch.device,
    seq_len: int,
    vocab_size: int = VOCAB_SIZE,
) -> Dict[str, float]:
    return _analyze_interactions_impl(model, input_ids, dev, seq_len, vocab_size)


def _interaction_influence_matrix(
    model: nn.Module,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    return _interaction_influence_matrix_impl(
        model, input_ids, positions, vocab_size=vocab_size
    )


def _interaction_metrics(
    influence_matrix: torch.Tensor, positions: torch.Tensor
) -> Dict[str, float]:
    from .fingerprint_native import interaction_metrics

    return interaction_metrics(influence_matrix, positions)


def _analyze_routing(
    model: nn.Module, input_ids: torch.Tensor, dev: torch.device
) -> Dict[str, float]:
    return _analyze_routing_impl(model, input_ids, dev)


def _analyze_geometry(reps: torch.Tensor) -> Dict[str, float]:
    return _analyze_geometry_impl(reps)


def _forward_model_from_embed(model: nn.Module, embed_in: torch.Tensor) -> torch.Tensor:
    return _forward_model_from_embed_impl(model, embed_in)


def _analyze_sensitivity(
    model: nn.Module,
    dev: torch.device,
    seq_len: int,
    vocab_size: int,
) -> Dict[str, float]:
    return _analyze_sensitivity_impl(model, dev, seq_len, vocab_size)


def _collect_position_sensitivities(
    x_or_forward: torch.Tensor,
    embed: torch.Tensor,
    positions: torch.Tensor,
) -> Optional[torch.Tensor]:
    return _collect_position_sensitivities_impl(x_or_forward, embed, positions)


def _sensitivity_metrics(sens_matrix: torch.Tensor) -> Dict[str, float]:
    from .fingerprint_native import sensitivity_metrics

    return sensitivity_metrics(sens_matrix)


def _compute_reference_cka(
    reps: Optional[torch.Tensor],
    ref_activations: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    return _compute_reference_cka_impl(reps, ref_activations)


def _cka_from_tensor(
    reps: torch.Tensor,
    ref_activations: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    return _cka_from_tensor_impl(reps, ref_activations)


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    from .fingerprint_native import linear_cka

    return linear_cka(X, Y)
