from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .associative_recall import associative_recall_score
from .binding_range import binding_range_profile
from .binding_curriculum import (
    CURRICULUM_BINDING_PROTOCOL_VERSION,
    CURRICULUM_BINDING_DISTANCES,
    CURRICULUM_BINDING_EVAL_FULL,
    CURRICULUM_BINDING_EVAL_SCREENING,
    CURRICULUM_BINDING_STEPS_FULL,
    CURRICULUM_BINDING_STEPS_SCREENING,
    curriculum_binding_range_profile,
)
from .native_induction import induction_result_metadata, induction_score_gold


@dataclass(slots=True)
class FullBindingProbeResult:
    ar_legacy_auc: float
    ar_legacy_final_acc: float
    ar_legacy_timed_out: bool
    ar_legacy_above_chance: bool
    ar_elapsed_ms: float
    induction_screening_auc: float
    induction_metadata: Dict[str, Any]
    induction_elapsed_ms: float
    binding_screening_auc: float
    binding_distance_accuracies: Dict[int, float]
    binding_elapsed_ms: float
    binding_curriculum_auc: float
    binding_distance_accuracies_curriculum: Dict[int, float]
    binding_curriculum_elapsed_ms: float
    binding_curriculum_train_steps: int

    def to_result_dict(self) -> Dict[str, Any]:
        out = {
            "ar_legacy_auc": self.ar_legacy_auc,
            "ar_legacy_final_acc": self.ar_legacy_final_acc,
            "ar_legacy_timed_out": self.ar_legacy_timed_out,
            "ar_legacy_above_chance": self.ar_legacy_above_chance,
            "binding_screening_auc": self.binding_screening_auc,
            "binding_distance_accuracies": self.binding_distance_accuracies,
            "binding_probe_distances": list(CURRICULUM_BINDING_DISTANCES),
            "binding_screening_eval_examples": CURRICULUM_BINDING_EVAL_FULL,
            "binding_screening_elapsed_ms": self.binding_elapsed_ms,
            "binding_curriculum_auc": self.binding_curriculum_auc,
            "binding_distance_accuracies_curriculum": self.binding_distance_accuracies_curriculum,
            "binding_curriculum_steps": self.binding_curriculum_train_steps,
            "binding_curriculum_elapsed_ms": self.binding_curriculum_elapsed_ms,
            "binding_curriculum_protocol_version": CURRICULUM_BINDING_PROTOCOL_VERSION,
        }
        out.update(self.induction_metadata)
        return out


def compute_binding_screening_composite(
    ar_legacy_auc: float | None,
    induction_screening_auc: float,
    binding_screening_auc: float,
) -> float:
    if ar_legacy_auc is None:
        return round(0.3 * induction_screening_auc + 0.3 * binding_screening_auc, 4)
    return round(
        0.4 * ar_legacy_auc
        + 0.3 * induction_screening_auc
        + 0.3 * binding_screening_auc,
        4,
    )


def compute_local_only(
    ar_legacy_auc: float, induction_screening_auc: float, binding_screening_auc: float
) -> int:
    from research.scientist.thresholds import (
        BINDING_AR_SOFT_GATE,
        BINDING_BINDING_AUC_SOFT_GATE,
        BINDING_INDUCTION_SOFT_GATE,
    )

    return int(
        ar_legacy_auc < BINDING_AR_SOFT_GATE
        and induction_screening_auc < BINDING_INDUCTION_SOFT_GATE
        and binding_screening_auc < BINDING_BINDING_AUC_SOFT_GATE
    )


def run_screening_binding_probes(
    model, *, device: str, seed: int | None = None
) -> Dict[str, Any]:
    ind = induction_score_gold(model, device=device, seed=seed)
    zero = binding_range_profile(model, device=device, seed=seed)
    br = curriculum_binding_range_profile(
        model,
        distances=CURRICULUM_BINDING_DISTANCES,
        n_train_steps=CURRICULUM_BINDING_STEPS_SCREENING,
        n_eval=CURRICULUM_BINDING_EVAL_SCREENING,
        device=device,
        seed=seed,
    )
    out = induction_result_metadata(ind)
    out["binding_screening_auc"] = zero.auc
    out["binding_distance_accuracies"] = zero.distance_accuracies
    out["binding_screening_elapsed_ms"] = zero.elapsed_ms
    out["binding_curriculum_auc"] = br.auc
    out["binding_distance_accuracies_curriculum"] = br.distance_accuracies
    out["binding_curriculum_steps"] = br.train_steps
    out["binding_curriculum_elapsed_ms"] = br.elapsed_ms
    out["binding_curriculum_protocol_version"] = br.protocol_version
    out["binding_screening_eval_examples"] = CURRICULUM_BINDING_EVAL_SCREENING
    out["binding_probe_distances"] = list(CURRICULUM_BINDING_DISTANCES)
    out["ar_legacy_auc"] = None
    out["binding_screening_composite"] = compute_binding_screening_composite(
        None, ind.auc, zero.auc
    )
    return out


def run_full_binding_probes(model, *, device: str) -> FullBindingProbeResult:
    ar = associative_recall_score(
        model,
        n_pairs=20,
        n_eval=200,
        n_train_steps=500,
        batch_size=16,
        device=device,
    )
    ind = induction_score_gold(model, device=device)
    zero = binding_range_profile(model, device=device)
    br = curriculum_binding_range_profile(
        model,
        distances=CURRICULUM_BINDING_DISTANCES,
        n_train_steps=CURRICULUM_BINDING_STEPS_FULL,
        n_eval=CURRICULUM_BINDING_EVAL_FULL,
        device=device,
    )
    return FullBindingProbeResult(
        ar_legacy_auc=ar.auc,
        ar_legacy_final_acc=ar.final_acc,
        ar_legacy_timed_out=ar.timed_out,
        ar_legacy_above_chance=ar.above_chance,
        ar_elapsed_ms=ar.elapsed_ms,
        induction_screening_auc=ind.auc,
        induction_metadata=induction_result_metadata(ind),
        induction_elapsed_ms=ind.elapsed_ms,
        binding_screening_auc=zero.auc,
        binding_distance_accuracies=zero.distance_accuracies,
        binding_elapsed_ms=zero.elapsed_ms,
        binding_curriculum_auc=br.auc,
        binding_distance_accuracies_curriculum=br.distance_accuracies,
        binding_curriculum_elapsed_ms=br.elapsed_ms,
        binding_curriculum_train_steps=br.train_steps,
    )
