"""Behavioral fingerprint runtime: probes, CKA, scoring, orchestration.

Consolidated 2026-06-13 from fingerprint_probes.py + fingerprint_cka.py +
fingerprint_scoring.py + fingerprint_runtime.py — the three absorbed modules
had no callers outside this runtime (and tests).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from research.synthesis.graph import ComputationGraph

from research.defaults import VOCAB_SIZE

from ..synthesis.grammar import _ROUTING_COMPRESSION_MOE_OPS
from .cka_references import get_default_store as _get_default_store
from .fingerprint_native import (
    geometry_metrics,
    interaction_metrics,
    linear_cka,
    mean_abs_linear_delta,
    sequence_self_similarity,
)
from .fingerprint_sensitivity import analyze_sensitivity
from .fingerprint_types import (
    BEHAVIOR_SIGNATURE_WEIGHT,
    CKA_NOVELTY_WEIGHT,
    BehavioralFingerprint,
    NOVELTY_REFERENCE_SCHEME_VERSION,
)
from .hierarchy_probe import hierarchy_fitness

logger = logging.getLogger(__name__)


# ── Scoring (from fingerprint_scoring.py) ───────────────────────────────

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


# ── CKA reference comparison (from fingerprint_cka.py) ─────────────────


def compute_reference_cka(
    reps: Optional[torch.Tensor],
    ref_activations: Optional[Dict[str, torch.Tensor]] = None,
    ref_similarities: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    result = {"transformer": 0.0, "ssm": 0.0, "conv": 0.0, "_succeeded": False}
    if reps is None:
        return result
    try:
        result = cka_from_tensor(
            reps,
            ref_activations=ref_activations,
            ref_similarities=ref_similarities,
        )
        intermediate = getattr(reps, "_cka_intermediate_reps", None)
        if intermediate is not None:
            all_zero = all(
                abs(result[key]) < 1e-6 for key in ("transformer", "ssm", "conv")
            )
            if all_zero:
                logger.info("CKA degenerate on logits, retrying with intermediate reps")
                fallback = cka_from_tensor(
                    intermediate,
                    ref_activations=ref_activations,
                    ref_similarities=ref_similarities,
                )
                if fallback["_succeeded"] and any(
                    abs(fallback[key]) > 1e-6 for key in ("transformer", "ssm", "conv")
                ):
                    fallback["_cka_layer"] = "intermediate"
                    return fallback
    except Exception as exc:
        logger.warning("CKA computation failed: %s", exc)
    return result


def cka_from_tensor(
    reps: torch.Tensor,
    ref_activations: Optional[Dict[str, torch.Tensor]] = None,
    ref_similarities: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    result = {"transformer": 0.0, "ssm": 0.0, "conv": 0.0, "_succeeded": False}
    similarity = _self_similarity(reps)
    if similarity is None:
        return result

    sim, seq_len = similarity
    if ref_similarities is None and ref_activations is None:
        return result

    matched_reference = False
    for family in ("transformer", "ssm", "conv"):
        ref_sim = None
        if ref_similarities is not None:
            ref_sim = ref_similarities.get(family)
        if ref_sim is None and ref_activations is not None:
            ref_tensor = ref_activations.get(family)
            if ref_tensor is None:
                continue
            ref_flat = ref_tensor.float()
            ref_seq = ref_flat.shape[-2]
            use_seq = min(seq_len, ref_seq)
            ref_sim = sequence_self_similarity(ref_flat[..., :use_seq, :])
        if ref_sim is None:
            continue
        ref_seq = ref_sim.shape[-1]
        use_seq = min(seq_len, ref_seq)
        if ref_sim.device != sim.device:
            ref_sim = ref_sim.to(sim.device)
        matched_reference = True
        result[family] = linear_cka(
            sim[:use_seq, :use_seq], ref_sim[:use_seq, :use_seq]
        )

    result["_succeeded"] = matched_reference
    return result


def _self_similarity(reps: torch.Tensor) -> Optional[tuple[torch.Tensor, int]]:
    seq_len = reps.shape[-2]
    if seq_len < 4:
        return None
    if reps.dim() > 2:
        reps = reps[: min(reps.shape[0], 8)]
    return sequence_self_similarity(reps)[:seq_len, :seq_len], seq_len


# ── Probes (from fingerprint_probes.py) ────────────────────────────────


@dataclass(slots=True)
class ProbeRepresentations:
    logits: torch.Tensor
    reps: Optional[torch.Tensor]


def _replacement_ids(
    ids: torch.Tensor, positions: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    return (ids[0, positions] + 1) % vocab_size


def _perturbed_token_batch(
    ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    n_positions = int(positions.numel())
    perturbed = ids.expand(n_positions, -1).clone()
    row_idx = torch.arange(n_positions, device=positions.device)
    perturbed[row_idx, positions] = _replacement_ids(ids, positions, vocab_size)
    return perturbed


def _perturbed_embed_batch(
    model: nn.Module,
    ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    n_positions = int(positions.numel())
    base_embed = model.embed(ids)
    perturbed = base_embed.expand(n_positions, -1, -1).clone()
    row_idx = torch.arange(n_positions, device=positions.device)
    perturbed[row_idx, positions] = model.embed(
        _replacement_ids(ids, positions, vocab_size)
    )
    return base_embed, perturbed


def _resolve_representation_hook_module(model: nn.Module) -> nn.Module | None:
    cached = model.__dict__.get("_fingerprint_capture_module", None)
    if cached is False:
        return None
    if cached is not None:
        return cached

    output_projection_ids = {
        id(module)
        for name in ("lm_head", "head", "output", "classifier")
        for module in (getattr(model, name, None),)
        if isinstance(module, nn.Module)
    }

    vocab_size = None
    embed = getattr(model, "embed", None)
    if isinstance(embed, nn.Embedding):
        vocab_size = int(embed.num_embeddings)
        embed_candidate: nn.Module | None = embed
    else:
        embed_candidate = None

    last_candidate: nn.Module | None = None
    for mod in model.modules():
        if id(mod) in output_projection_ids:
            continue
        if isinstance(mod, nn.LayerNorm):
            last_candidate = mod
            continue
        if isinstance(mod, nn.Linear):
            if vocab_size is not None and int(mod.out_features) == vocab_size:
                continue
            last_candidate = mod

    resolved = last_candidate if last_candidate is not None else embed_candidate
    model.__dict__["_fingerprint_capture_module"] = resolved or False
    return resolved


def capture_probe_representations(
    model: nn.Module,
    input_ids: torch.Tensor,
) -> Optional[ProbeRepresentations]:
    """Run the model and capture the best available hidden-state-like tensor."""
    try:
        direct_impl = getattr(model, "_fingerprint_representations", None)
        if callable(direct_impl):
            logits, reps = direct_impl(input_ids)
            if isinstance(logits, torch.Tensor):
                return ProbeRepresentations(
                    logits=logits,
                    reps=reps.detach() if isinstance(reps, torch.Tensor) else None,
                )
            return None

        captured: dict[str, torch.Tensor] = {}
        hooks = []
        mod = _resolve_representation_hook_module(model)

        if mod is not None:

            def _hook(_module, _inp, out):
                if isinstance(out, torch.Tensor) and out.dim() >= 2:
                    captured["last"] = out.detach()

            hooks.append(mod.register_forward_hook(_hook))

        logits = model(input_ids)

        for hook in hooks:
            hook.remove()

        if not isinstance(logits, torch.Tensor):
            return None
        reps = captured.get("last")
        return ProbeRepresentations(logits=logits, reps=reps)
    except Exception as exc:
        logger.warning("Failed to get representations: %s", exc)
        return None


def interaction_influence_matrix(
    model: nn.Module,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    ids = input_ids[:1]
    pre_logits_from_embed = getattr(model, "_fingerprint_pre_logits_from_embed", None)
    if (
        callable(pre_logits_from_embed)
        and hasattr(model, "embed")
        and hasattr(model, "lm_head")
    ):
        base_embed, perturbed_embed = _perturbed_embed_batch(
            model, ids, positions, vocab_size=vocab_size
        )
        base_pre = pre_logits_from_embed(base_embed)
        delta = pre_logits_from_embed(perturbed_embed) - base_pre
        native_metric = mean_abs_linear_delta(delta, model.lm_head.weight)
        if native_metric is not None:
            return native_metric
        return F.linear(delta, model.lm_head.weight).abs().mean(dim=-1)

    logits_from_embed = getattr(model, "_fingerprint_logits_from_embed", None)
    if callable(logits_from_embed) and hasattr(model, "embed"):
        base_embed, perturbed_embed = _perturbed_embed_batch(
            model, ids, positions, vocab_size=vocab_size
        )
        base_out = logits_from_embed(base_embed)
        return (logits_from_embed(perturbed_embed) - base_out).abs().mean(dim=-1)

    base_out = model(ids)
    perturbed_batch = _perturbed_token_batch(ids, positions, vocab_size=vocab_size)
    return (model(perturbed_batch) - base_out).abs().mean(dim=-1)


def analyze_interactions(
    model: nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    seq_len: int,
    vocab_size: int = VOCAB_SIZE,
) -> Dict[str, float]:
    result = {
        "locality": 0.5,
        "sparsity": 0.5,
        "symmetry": 0.5,
        "hierarchy": 0.5,
        "_succeeded": False,
    }
    try:
        n_positions = min(8, seq_len)
        positions = torch.linspace(0, seq_len - 1, n_positions, device=device).long()
        influence = interaction_influence_matrix(
            model,
            input_ids[:1],
            positions,
            vocab_size=vocab_size,
        )
        result.update(interaction_metrics(influence, positions))
        result["_succeeded"] = True
    except Exception as exc:
        logger.warning("Interaction analysis failed: %s", exc)
    return result


def analyze_routing(
    model: nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    result = {
        "selectivity": 0.0,
        "compute_ratio": 0.0,
        "lane_correlation": 0.0,
        "_has_routing": False,
    }
    has_routing = False
    if hasattr(model, "graph") and model.graph is not None:
        for node in model.graph.nodes.values():
            if not node.is_input and node.op_name in _ROUTING_COMPRESSION_MOE_OPS:
                has_routing = True
                break
    if not has_routing:
        return result

    result["_has_routing"] = True
    try:
        with torch.no_grad():
            model(input_ids)

        if hasattr(model, "last_routing_scores"):
            scores = model.last_routing_scores
            if isinstance(scores, torch.Tensor) and scores.numel() > 0:
                result["selectivity"] = float(scores.std().item())

        if hasattr(model, "get_routing_compute_stats"):
            stats = model.get_routing_compute_stats()
            slow = stats.get("slow_flops", 0)
            fast = stats.get("fast_flops", 1)
            result["compute_ratio"] = float(slow / max(fast, 1e-6))
        elif hasattr(model, "routing_compute_ratio"):
            result["compute_ratio"] = float(model.routing_compute_ratio)

        if hasattr(model, "last_routing_decisions"):
            decisions = model.last_routing_decisions
            if isinstance(decisions, torch.Tensor) and decisions.dim() >= 2:
                batch_size, seq_len = decisions.shape[:2]
                positions = (
                    torch.arange(seq_len, device=device)
                    .float()
                    .expand(batch_size, seq_len)
                )
                result["lane_correlation"] = float(
                    _pearson_corr(decisions.float(), positions).item()
                )
    except Exception as exc:
        logger.debug("Routing analysis failed: %s", exc)
    return result


def analyze_geometry(reps: torch.Tensor) -> Dict[str, float]:
    result = {
        "intrinsic_dim": 0.0,
        "isotropy": 0.0,
        "rank_ratio": 0.0,
        "_succeeded": False,
    }
    try:
        flat = reps.reshape(-1, reps.shape[-1]).float()
        num_rows, width = flat.shape
        if num_rows < 2 or width < 2:
            return result

        native = geometry_metrics(reps)
        if native is not None:
            result.update(native)
            result["_succeeded"] = True
            return result

        flat = flat - flat.mean(dim=0, keepdim=True)
        subset = flat[
            torch.randperm(num_rows, device=flat.device)[: min(num_rows, 500)]
        ]
        try:
            if subset.shape[0] >= subset.shape[1]:
                gram = subset.transpose(0, 1) @ subset
            else:
                gram = subset @ subset.transpose(0, 1)
            singular_values = torch.linalg.eigvalsh(gram).clamp_min(1e-20).sqrt()
        except Exception as exc:
            logger.debug("SVD failed in geometry analysis: %s", exc)
            return result

        singular_values = singular_values.clamp(min=1e-10)
        normalized = singular_values / singular_values.sum()
        result["intrinsic_dim"] = float((1.0 / (normalized**2).sum()).item())
        result["isotropy"] = float(
            (singular_values.min() / singular_values.max()).item()
        )
        entropy = float((-(normalized * torch.log(normalized))).sum().item())
        result["rank_ratio"] = math.exp(entropy) / len(singular_values)
        result["_succeeded"] = True
    except Exception as exc:
        logger.warning("Geometry analysis failed: %s", exc)
    return result


def analyze_hierarchy(reps: torch.Tensor) -> Dict[str, float]:
    return hierarchy_fitness(reps, max_tokens=100)


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    return (x_centered * y_centered).sum() / (
        torch.norm(x_centered) * torch.norm(y_centered) + 1e-8
    )


# ── Orchestration ───────────────────────────────────────────────────────


def populate_behavioral_probes(
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
            interaction = analyze_interactions(
                model,
                probe_ids,
                device,
                seq_len,
                vocab_size,
            )
            if interaction.get("_succeeded"):
                fp.interaction_locality = interaction["locality"]
                fp.interaction_sparsity = interaction["sparsity"]
                fp.interaction_symmetry = interaction["symmetry"]
                fp.interaction_hierarchy = interaction["hierarchy"]
                n_succeeded += 1
            else:
                fp.interaction_locality = None
                fp.interaction_sparsity = None
                fp.interaction_symmetry = None
                fp.interaction_hierarchy = None

            geometry = analyze_geometry(reps)
            if geometry.get("_succeeded"):
                fp.intrinsic_dim = geometry["intrinsic_dim"]
                fp.isotropy = geometry["isotropy"]
                fp.rank_ratio = geometry["rank_ratio"]
                n_succeeded += 1
            else:
                fp.intrinsic_dim = None
                fp.isotropy = None
                fp.rank_ratio = None
        else:
            fp.interaction_locality = None
            fp.interaction_sparsity = None
            fp.interaction_symmetry = None
            fp.interaction_hierarchy = None
            fp.intrinsic_dim = None
            fp.isotropy = None
            fp.rank_ratio = None

        if include:
            try:
                hierarchy = analyze_hierarchy(reps)
                fp.hierarchy_fitness = hierarchy["hierarchy_fitness"]
                fp.gromov_delta = hierarchy["gromov_delta"]
            except Exception as exc:
                logger.debug("Hierarchy probe skipped: %s", exc)
        else:
            fp.hierarchy_fitness = None
            fp.gromov_delta = None

    sensitivity = analyze_sensitivity(model, device, seq_len, vocab_size)
    if sensitivity.get("_succeeded"):
        fp.jacobian_spectral_norm = sensitivity["spectral_norm"]
        fp.jacobian_effective_rank = sensitivity["effective_rank"]
        fp.sensitivity_uniformity = sensitivity["uniformity"]
        if include:
            n_succeeded += 1
    else:
        fp.jacobian_spectral_norm = None
        fp.jacobian_effective_rank = None
        fp.sensitivity_uniformity = None

    try:
        routing = analyze_routing(model, probe_ids, device)
        fp.routing_selectivity = routing["selectivity"]
        fp.routing_compute_ratio = routing["compute_ratio"]
        fp.routing_lane_correlation = routing["lane_correlation"]
        fp.routing_telemetry_present = routing.get("_has_routing", False)
    except Exception as exc:
        logger.debug("Routing analysis skipped: %s", exc)
        fp.routing_telemetry_present = False

    return n_succeeded


def populate_cka(
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
    cka = compute_reference_cka(
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
    fp.similarity_path = cka_meta.get("cka_similarity_path", "compute_reference_cka")

    all_near_zero = (
        fp.cka_vs_transformer < 0.01 and fp.cka_vs_ssm < 0.01 and fp.cka_vs_conv < 0.01
    )
    if all_near_zero and cka.get("_succeeded"):
        logger.warning(
            "CKA sanity gate: all three scores < 0.01 - marking cka_reference_quality as false"
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
    else:
        fp.novelty_valid_for_promotion = False
        fp.novelty_validity_reason = "no_reference_available"

    cka_scores = [fp.cka_vs_transformer, fp.cka_vs_ssm, fp.cka_vs_conv]
    cka_all_zero = all(abs(score) < 1e-6 for score in cka_scores)
    if cka_all_zero and fp.cka_source == "artifact":
        fp.novelty_valid_for_promotion = False
        fp.novelty_validity_reason = "cka_degenerate_zeros"
        logger.warning(
            "cka_degenerate_zeros: cka_scores=%s cka_source=%s quality=%s",
            cka_scores,
            fp.cka_source,
            fp.quality,
        )
    elif cka_all_zero:
        logger.info(
            "cka_zero_scores_without_artifact: cka_scores=%s cka_source=%s quality=%s",
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


def set_quality(fp: BehavioralFingerprint, analyses_succeeded: int) -> None:
    fp.analyses_succeeded = analyses_succeeded
    if analyses_succeeded == 4:
        fp.quality = "full"
    elif analyses_succeeded > 0:
        fp.quality = "partial"
    else:
        fp.quality = "none"


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
        captured = capture_probe_representations(model, probe_ids)
        reps = captured.reps if captured is not None else None
        probe_succeeded = populate_behavioral_probes(
            fp,
            model,
            probe_ids,
            reps,
            dev,
            seq_len,
            vocab_size,
            include=include_behavioral_probes,
        )
        cka_succeeded, cka_all_zero = populate_cka(fp, reps, include=include_cka)
        fp.behavior_signature_score = behavior_signature_score(fp)
        fp.novelty_score = blend_behavioral_novelty(fp)
        if cka_all_zero:
            fp.novelty_score = fp.behavior_signature_score

    set_quality(fp, probe_succeeded + cka_succeeded)
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
        captured = capture_probe_representations(model, probe_ids)
        reps = captured.reps if captured is not None else None
        if reps is not None:
            try:
                hierarchy = analyze_hierarchy(reps)
                fp.hierarchy_fitness = hierarchy["hierarchy_fitness"]
                fp.gromov_delta = hierarchy["gromov_delta"]
            except Exception as exc:
                logger.debug("Hierarchy detection skipped: %s", exc)

    # Always compute sensitivity (Jacobian spectral norm) for ML-ops training
    # data; this needs grad so it runs outside the no_grad block above.
    sensitivity = analyze_sensitivity(model, dev, seq_len, vocab_size=32000)
    if sensitivity.get("_succeeded"):
        fp.jacobian_spectral_norm = sensitivity["spectral_norm"]
        fp.jacobian_effective_rank = sensitivity["effective_rank"]
        fp.sensitivity_uniformity = sensitivity["uniformity"]
    else:
        fp.jacobian_spectral_norm = None
        fp.jacobian_effective_rank = None
        fp.sensitivity_uniformity = None

    fp.cka_vs_transformer = None
    fp.cka_vs_ssm = None
    fp.cka_vs_conv = None
    fp.cka_source = "deferred"
    fp.novelty_valid_for_promotion = False
    fp.novelty_validity_reason = "cka_deferred_post_investigation"
    fp.behavior_signature_score = behavior_signature_score(fp)
    fp.quality = "partial"
    fp.analyses_succeeded = 0
    return fp


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
        captured = capture_probe_representations(model, probe_ids)
        reps = captured.reps if captured is not None else None

    probe_succeeded = populate_behavioral_probes(
        fp,
        model,
        probe_ids,
        reps,
        dev,
        seq_len,
        vocab_size,
        include=True,
    )
    cka_succeeded, cka_all_zero = populate_cka(fp, reps, include=True)
    fp.behavior_signature_score = behavior_signature_score(fp)
    fp.novelty_score = blend_behavioral_novelty(fp)
    if cka_all_zero:
        fp.novelty_score = fp.behavior_signature_score

    fp.fingerprint_completed_post_investigation = True
    fp.fingerprint_completion_timestamp = datetime.utcnow().isoformat()
    set_quality(fp, probe_succeeded + cka_succeeded)
    model.train()
    return fp
