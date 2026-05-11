"""Post-S1 screening-probe runner.

Produces the 6 metrics required by the S1-completeness guardrail
(``research/scientist/notebook/program_writes.py::_enforce_s1_metric_completeness``)
on a model that has just finished S1 micro-training:

- ``wikitext_perplexity``
- ``hellaswag_acc``
- ``induction_screening_auc``
- ``binding_screening_auc``
- ``binding_screening_composite``
- ``ar_legacy_auc``

Plus auxiliary fields that the leaderboard composite consumes
(``ar_gate_score``, ``binding_curriculum_*``, etc.).

The synthesis runner has its own probe loop in
``runner/execution_training_post.py::_run_post_s1_screening_probes`` with
discovery hooks, NO-GO flagging and SSE-friendly logging. That version is
not invoked here to avoid coupling to ``RunConfig`` / mixin context. A
follow-up should refactor the runner method to call this helper.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


def _probe_wikitext(
    model: nn.Module, vocab_size: int, device: str, seq_len: int
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        from research.eval.wikitext_eval import screening_wikitext_eval

        wt = screening_wikitext_eval(model, vocab_size, device, seq_len=seq_len)
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        logger.warning("WikiText probe failed: %s", exc)
        return out
    if wt.get("wikitext_perplexity") is not None:
        out["wikitext_perplexity"] = wt["wikitext_perplexity"]
        out["wikitext_score"] = wt.get("wikitext_score")
    for key in (
        "screening_wikitext_status",
        "screening_wikitext_metric_version",
        "screening_wikitext_elapsed_ms",
        "wikitext_pre_perplexity",
        "wikitext_ppl_improvement",
    ):
        value = wt.get(key)
        if value is not None:
            out[key] = value
    return out


def _probe_hellaswag(model: nn.Module, vocab_size: int, device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        from research.eval.hellaswag_eval import screening_hellaswag_eval

        hs = screening_hellaswag_eval(model, vocab_size, device)
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        logger.warning("HellaSwag probe failed: %s", exc)
        return out
    if hs.get("hellaswag_acc") is not None:
        out["hellaswag_acc"] = hs["hellaswag_acc"]
    for key in (
        "hellaswag_status",
        "hellaswag_metric_version",
        "hellaswag_tokenizer_mode",
        "hellaswag_tiktoken_encoding",
        "hellaswag_correct",
    ):
        value = hs.get(key)
        if value is not None:
            out[key] = value
    if hs.get("hellaswag_total") is not None:
        out["hellaswag_n_examples"] = hs["hellaswag_total"]
    if hs.get("elapsed_ms") is not None:
        out["screening_hellaswag_elapsed_ms"] = hs["elapsed_ms"]
    return out


def _probe_induction(
    model: nn.Module, device: str, seed: Optional[int]
) -> tuple[Dict[str, Any], Optional[float]]:
    try:
        from research.eval.native_induction import (
            induction_result_metadata,
            induction_score_gold,
        )

        ind = induction_score_gold(model, device=device, seed=seed)
    except (RuntimeError, ValueError, TypeError, ImportError) as exc:
        logger.warning("Induction probe failed: %s", exc)
        return {}, None
    auc = float(ind.auc) if getattr(ind, "auc", None) is not None else None
    return induction_result_metadata(ind), auc


def _probe_binding_zero(
    model: nn.Module, device: str, seed: Optional[int]
) -> tuple[Dict[str, Any], Optional[float]]:
    out: Dict[str, Any] = {}
    try:
        from research.eval.binding_curriculum import (
            CURRICULUM_BINDING_DISTANCES,
            CURRICULUM_BINDING_EVAL_SCREENING,
        )
        from research.eval.binding_range import binding_range_profile

        zero = binding_range_profile(
            model,
            distances=CURRICULUM_BINDING_DISTANCES,
            n_eval=CURRICULUM_BINDING_EVAL_SCREENING,
            device=device,
            seed=seed,
        )
    except (RuntimeError, ValueError, TypeError, ImportError) as exc:
        logger.warning("Binding zero-shot probe failed: %s", exc)
        return out, None
    out["binding_screening_auc"] = zero.auc
    out["binding_distance_accuracies"] = zero.distance_accuracies
    out["binding_screening_eval_examples"] = CURRICULUM_BINDING_EVAL_SCREENING
    out["binding_probe_distances"] = list(CURRICULUM_BINDING_DISTANCES)
    out["binding_screening_elapsed_ms"] = zero.elapsed_ms
    return out, (float(zero.auc) if zero.auc is not None else None)


def _probe_binding_curriculum(
    model: nn.Module, device: str, seed: Optional[int]
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        from research.eval.binding_curriculum import (
            CURRICULUM_BINDING_DISTANCES,
            CURRICULUM_BINDING_EVAL_BATCH_SIZE,
            CURRICULUM_BINDING_EVAL_SCREENING,
            CURRICULUM_BINDING_PROTOCOL_VERSION,
            CURRICULUM_BINDING_STEPS_SCREENING,
            CURRICULUM_BINDING_TRAIN_BATCH_SIZE,
            curriculum_binding_range_profile,
        )

        br = curriculum_binding_range_profile(
            model,
            distances=CURRICULUM_BINDING_DISTANCES,
            n_train_steps=CURRICULUM_BINDING_STEPS_SCREENING,
            n_eval=CURRICULUM_BINDING_EVAL_SCREENING,
            train_batch_size=CURRICULUM_BINDING_TRAIN_BATCH_SIZE,
            eval_batch_size=CURRICULUM_BINDING_EVAL_BATCH_SIZE,
            device=device,
            seed=seed,
        )
    except (RuntimeError, ValueError, TypeError, ImportError) as exc:
        logger.warning("Binding curriculum probe failed: %s", exc)
        return out
    out["binding_curriculum_auc"] = br.auc
    out["binding_distance_accuracies_curriculum"] = br.distance_accuracies
    out["binding_curriculum_steps"] = br.train_steps
    out["binding_curriculum_elapsed_ms"] = br.elapsed_ms
    out["binding_curriculum_protocol_version"] = CURRICULUM_BINDING_PROTOCOL_VERSION
    return out


def _probe_ar_legacy(model: nn.Module, device: str) -> Dict[str, Any]:
    try:
        from research.eval.associative_recall import associative_recall_score

        ar = associative_recall_score(
            model,
            n_pairs=20,
            n_eval=200,
            n_train_steps=500,
            batch_size=16,
            device=device,
        )
    except (RuntimeError, ValueError, TypeError, ImportError) as exc:
        logger.warning("AR legacy probe failed: %s", exc)
        return {}
    return {
        "ar_legacy_auc": ar.auc,
        "ar_legacy_final_acc": ar.final_acc,
        "ar_legacy_timed_out": int(ar.timed_out),
        "ar_legacy_above_chance": int(ar.above_chance),
    }


def _probe_ar_gate(
    model: nn.Module, device: str
) -> tuple[Dict[str, Any], Optional[float]]:
    try:
        from research.eval.ar_gate import ARGateConfig, ar_gate

        nai = ar_gate(model=model, device=device, cfg=ARGateConfig(from_s1=True))
    except (RuntimeError, ValueError, TypeError, ImportError) as exc:
        logger.warning("AR-gate probe failed: %s", exc)
        return {}, None
    score = 0.6 * nai.in_dist_pair_acc + 0.4 * nai.held_class_acc
    out = {
        "ar_gate_metric_version": nai.metric_version,
        "ar_gate_in_dist_pair_acc": nai.in_dist_pair_acc,
        "ar_gate_in_dist_class_acc": nai.in_dist_class_acc,
        "ar_gate_held_pair_acc": nai.held_pair_acc,
        "ar_gate_held_class_acc": nai.held_class_acc,
        "ar_gate_score": round(score, 4),
        "ar_gate_status": nai.status,
        "ar_gate_elapsed_ms": nai.elapsed_ms,
        "ar_gate_train_steps_done": nai.finetune_steps_done,
    }
    return out, score


def _compute_binding_composite(
    induction_auc: Optional[float],
    binding_auc: Optional[float],
    ar_gate_score: Optional[float],
) -> Optional[float]:
    """Match runner/execution_training_post.py composite weights."""
    if induction_auc is None or binding_auc is None:
        return None
    if ar_gate_score is not None:
        return round(0.4 * ar_gate_score + 0.3 * induction_auc + 0.3 * binding_auc, 4)
    return round(0.3 * induction_auc + 0.3 * binding_auc, 4)


def run_post_s1_probes(
    model: nn.Module,
    *,
    vocab_size: int,
    max_seq_len: int = 128,
    device: str = "cpu",
    seed: Optional[int] = None,
    run_ar_gate: bool = True,
    run_binding_curriculum: bool = True,
) -> Dict[str, Any]:
    """Run the post-S1 probe set on a trained model.

    Each probe is wrapped in a broad try/except so a single probe failure
    doesn't sink the others; the caller decides whether to demote
    ``stage1_passed`` if required metrics are missing. Returns a flat dict
    of metric column → value, safe to splat into ``record_program_result``
    or merge with ``program_result_kwargs_from_s1``.
    """
    dev_str = str(device)
    probe_seq_len = min(128, max_seq_len)
    metrics: Dict[str, Any] = {}

    metrics.update(_probe_wikitext(model, vocab_size, dev_str, probe_seq_len))
    metrics.update(_probe_hellaswag(model, vocab_size, dev_str))

    induction_fields, induction_auc = _probe_induction(model, dev_str, seed)
    metrics.update(induction_fields)

    binding_fields, binding_auc = _probe_binding_zero(model, dev_str, seed)
    metrics.update(binding_fields)
    if run_binding_curriculum and binding_auc is not None:
        metrics.update(_probe_binding_curriculum(model, dev_str, seed))

    metrics.update(_probe_ar_legacy(model, dev_str))

    ar_gate_score: Optional[float] = None
    if run_ar_gate:
        ar_gate_fields, ar_gate_score = _probe_ar_gate(model, dev_str)
        metrics.update(ar_gate_fields)

    composite = _compute_binding_composite(induction_auc, binding_auc, ar_gate_score)
    if composite is not None:
        metrics["binding_screening_composite"] = composite

    return metrics


_REQUIRED_S1_METRICS = (
    "wikitext_perplexity",
    "hellaswag_acc",
    "induction_screening_auc",
    "binding_screening_auc",
    "binding_screening_composite",
    "ar_legacy_auc",
)


def missing_required_metrics(metrics: Dict[str, Any]) -> list[str]:
    """Return the list of S1-guardrail-required metric columns absent from ``metrics``.

    Mirrors ``_S1_REQUIRED_POST_METRIC_COLUMNS_FOR_GUARDRAIL`` in
    ``notebook/program_writes.py`` so callers can downgrade
    ``stage1_passed`` *before* hitting the write-time guardrail.
    """
    return [c for c in _REQUIRED_S1_METRICS if metrics.get(c) is None]
