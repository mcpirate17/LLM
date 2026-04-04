"""CKA helpers and reference-comparison utilities."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch

from .fingerprint_native import linear_cka, sequence_self_similarity

logger = logging.getLogger(__name__)


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
    if ref_similarities is not None or ref_activations is not None:
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
            result[family] = linear_cka(
                sim[:use_seq, :use_seq], ref_sim[:use_seq, :use_seq]
            )
    else:
        positions = torch.arange(seq_len, device=sim.device).float()
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        result["transformer"] = linear_cka(sim, torch.exp(-dist / (seq_len * 0.3)))
        result["ssm"] = linear_cka(
            sim, (torch.exp(-dist / (seq_len * 0.15)) * (dist >= 0).float()).tril()
        )
        result["conv"] = linear_cka(sim, (dist <= 5).float())

    result["_succeeded"] = True
    return result


def _self_similarity(reps: torch.Tensor) -> Optional[tuple[torch.Tensor, int]]:
    seq_len = reps.shape[-2]
    if seq_len < 4:
        return None
    if reps.dim() > 2:
        reps = reps[: min(reps.shape[0], 8)]
    return sequence_self_similarity(reps)[:seq_len, :seq_len], seq_len
