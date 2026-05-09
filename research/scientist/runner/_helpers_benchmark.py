"""Runner helpers — split from _helpers. Re-exported via _helpers."""

from __future__ import annotations

import json
import logging
import queue
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from ..capability_ranker_metrics import (
    AR_INTERMEDIATE_SCORE_FIELDS as _AR_INTERMEDIATE_SCORE_FIELDS,
    BINDING_MULTISLOT_SCORE_FIELDS as _BINDING_MULTISLOT_SCORE_FIELDS,
    capability_ranker_fields,
    has_capability_ranker_evidence,
)
from ..thresholds import TIER_RANK
from ._helpers_metrics import (
    _trajectory_probe_capability_tier,
    screening_wikitext_fields,
    trajectory_probe_fields,
    v9_trajectory_fields,
)

logger = logging.getLogger(__name__)


def _build_benchmark_model(
    *,
    config,
    dev,
    model_source: str,
    arch_spec_json_str: str | None,
    graph_json_str: str | None,
    cached_json_load,
) -> Any:
    """Build a model for benchmark evaluation (shared across benchmarks)."""
    if model_source == "morphological_box" and arch_spec_json_str:
        from ...morphological_box import ArchSpec
        from ...arch_builder import BuildConfig, build_model

        spec = ArchSpec(**cached_json_load(arch_spec_json_str))
        build_cfg = BuildConfig(
            dim=config.model_dim,
            n_layers=config.n_layers,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
        )
        return build_model(spec, build_cfg).to(dev)
    elif graph_json_str:
        from ..native_runner import compile_model_native_first as compile_model
        from ...synthesis.serializer import graph_from_json

        return compile_model(
            [graph_from_json(graph_json_str)] * config.n_layers,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
        ).to(dev)
    return None


def _evaluate_investigation_benchmarks(
    *,
    config,
    dev,
    model_source: str,
    arch_spec_json_str: str | None,
    graph_json_str: str | None,
    cached_json_load,
    stop_event=None,
) -> Dict[str, Any]:
    """Run lightweight benchmark evals for investigation survivors.

    Compiles the model once and runs both WikiText and TinyStories evals
    on the same instance to avoid redundant compilation.

    ``stop_event`` is a ``threading.Event``; when set, the function aborts
    between major phases so that ``runner.stop()`` actually terminates
    background benchmark work in a bounded time instead of letting it
    grind through every queued candidate.
    """
    result: Dict[str, Any] = {
        "inv_wikitext_ppl": None,
        "inv_wikitext_score": None,
        "inv_tinystories_ppl": None,
        "inv_tinystories_score": None,
    }

    if stop_event is not None and stop_event.is_set():
        return result

    try:
        model = _build_benchmark_model(
            config=config,
            dev=dev,
            model_source=model_source,
            arch_spec_json_str=arch_spec_json_str,
            graph_json_str=graph_json_str,
            cached_json_load=cached_json_load,
        )
    except (ImportError, RuntimeError, ValueError, TypeError) as exc:
        logger.debug("Benchmark model build failed: %s", exc)
        return result

    if model is None:
        return result

    if stop_event is not None and stop_event.is_set():
        return result

    eval_seq_len = min(128, config.max_seq_len)
    result.update(
        _run_investigation_v2_probes(
            model,
            dev,
            graph_json_str=graph_json_str,
            run_induction_intermediate=bool(
                getattr(config, "investigation_run_capability_rankers", False)
            ),
            run_binding_intermediate=bool(
                getattr(config, "investigation_run_capability_rankers", False)
            ),
            run_ar_gate=True,
            run_ar_intermediate=bool(getattr(config, "run_ar_intermediate", False)),
            run_binding_multislot=bool(getattr(config, "run_binding_multislot", False)),
        )
    )

    if stop_event is not None and stop_event.is_set():
        return result

    try:
        from ...eval.wikitext_eval import evaluate_wikitext_trajectory

        wt_result = evaluate_wikitext_trajectory(
            model,
            config.vocab_size,
            dev,
            checkpoints=(100, 500, 1000),
            seq_len=eval_seq_len,
        )
        ckpts = wt_result.get("checkpoints") or {}
        ckpt_100 = ckpts.get(100) or ckpts.get("100") or {}
        ckpt_500 = ckpts.get(500) or ckpts.get("500") or {}
        ckpt_1000 = ckpts.get(1000) or ckpts.get("1000") or {}
        ppl_100 = ckpt_100.get("ppl")
        ppl_500 = ckpt_500.get("ppl")
        ppl_1000 = ckpt_1000.get("ppl")
        improvement_ratio = wt_result.get("improvement_ratio")
        result["wikitext_ppl_200"] = ppl_100  # legacy column, now stores @100
        result["wikitext_ppl_500"] = ppl_500
        result["wikitext_improvement_ratio"] = improvement_ratio
        result["wikitext_eval_steps"] = 1000 if ppl_1000 else 500
        result["eval_budget_steps"] = 1000 if ppl_1000 else 500
        # Use ppl@1000 as the screening perplexity (matches v7 anchor)
        result["wikitext_perplexity"] = ppl_1000 or ppl_500 or ppl_100
        result["evaluation_stage"] = "PROBED"
        result["capability_tier"] = _trajectory_probe_capability_tier(
            ppl_1000 or ppl_500,
            improvement_ratio,
            float(
                getattr(config, "improvement_ratio_escalation_threshold", 2.0) or 2.0
            ),
        )
        result["inv_wikitext_ppl"] = (
            wt_result.get("peak_ppl") or ppl_1000 or ppl_500 or ppl_100
        )
        result["inv_wikitext_score"] = (
            ckpt_1000.get("score")
            if ckpt_1000.get("score") is not None
            else ckpt_500.get("score")
            if ckpt_500.get("score") is not None
            else ckpt_100.get("score")
        )
        result["wikitext_trajectory_payload"] = wt_result
        if result["inv_wikitext_ppl"] is not None:
            logger.info(
                "Investigation WikiText-103 probe ppl100=%s ppl500=%s ppl1000=%s ratio=%s tier=%s",
                f"{ppl_100:.1f}" if isinstance(ppl_100, (int, float)) else "n/a",
                f"{ppl_500:.1f}" if isinstance(ppl_500, (int, float)) else "n/a",
                f"{ppl_1000:.1f}" if isinstance(ppl_1000, (int, float)) else "n/a",
                f"{improvement_ratio:.2f}"
                if isinstance(improvement_ratio, (int, float))
                else "n/a",
                result["capability_tier"],
            )
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.warning("Investigation WikiText eval skipped: %s", exc)

    if stop_event is not None and stop_event.is_set():
        del model
        return result

    try:
        from ...eval.tinystories_eval import evaluate_tinystories

        ts_result = evaluate_tinystories(
            model,
            config.vocab_size,
            dev,
            n_train_steps=200,
            seq_len=eval_seq_len,
        )
        result["inv_tinystories_ppl"] = ts_result.get("tinystories_perplexity")
        result["inv_tinystories_score"] = ts_result.get("tinystories_score")
        if result["inv_tinystories_ppl"] is not None:
            logger.info(
                "Investigation TinyStories ppl=%.1f score=%.3f",
                result["inv_tinystories_ppl"],
                result["inv_tinystories_score"] or 0,
            )
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.warning("Investigation TinyStories eval skipped: %s", exc)

    if stop_event is not None and stop_event.is_set():
        del model
        return result

    try:
        from ...eval.hellaswag_eval import evaluate_hellaswag

        hs_result = evaluate_hellaswag(
            model,
            config.vocab_size,
            dev,
            n_examples=100,
        )
        result["hellaswag_acc"] = hs_result.get("hellaswag_acc")
        result["hellaswag_status"] = hs_result.get("hellaswag_status")
        result["hellaswag_metric_version"] = hs_result.get("hellaswag_metric_version")
        result["hellaswag_tokenizer_mode"] = hs_result.get("hellaswag_tokenizer_mode")
        result["hellaswag_tiktoken_encoding"] = hs_result.get(
            "hellaswag_tiktoken_encoding"
        )
        if result["hellaswag_acc"] is not None:
            logger.info(
                "Investigation HellaSwag acc=%.1f%% (%d/%d, %.0fms)",
                result["hellaswag_acc"] * 100,
                hs_result.get("hellaswag_correct", 0),
                hs_result.get("hellaswag_total", 0),
                hs_result.get("elapsed_ms", 0),
            )
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.warning("Investigation HellaSwag eval skipped: %s", exc)

    if stop_event is not None and stop_event.is_set():
        del model
        return result

    # BLiMP linguistic minimal pairs (investigation: 50 per subtask)
    try:
        from ...eval.blimp_eval import evaluate_blimp

        blimp = evaluate_blimp(model, config.vocab_size, dev, n_per_subtask=50)
        result["blimp_overall_accuracy"] = blimp.overall_accuracy
        result["blimp_subtask_accuracies_json"] = json.dumps(blimp.subtask_accuracies)
        result["blimp_n_subtasks"] = blimp.n_subtasks
        result["blimp_status"] = blimp.status
        if blimp.overall_accuracy > 0:
            logger.info(
                "Investigation BLiMP acc=%.1f%% (%d subtasks, %d examples, %.0fms)",
                blimp.overall_accuracy * 100,
                blimp.n_subtasks,
                blimp.n_examples,
                blimp.elapsed_ms,
            )
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.warning("Investigation BLiMP eval skipped: %s", exc)

    if stop_event is not None and stop_event.is_set():
        del model
        return result

    # Binding probes: AR + induction + binding range (full suite at investigation)
    try:
        from ...eval.binding_pipeline import (
            compute_binding_screening_composite,
            compute_local_only,
            run_full_binding_probes,
        )

        probe = run_full_binding_probes(model, device=dev)
        result.update(probe.to_result_dict())
        bc = compute_binding_screening_composite(
            probe.ar_legacy_auc,
            probe.induction_screening_auc,
            probe.binding_screening_auc,
        )
        result["binding_screening_composite"] = bc
        result["local_only"] = compute_local_only(
            probe.ar_legacy_auc,
            probe.induction_screening_auc,
            probe.binding_screening_auc,
        )

        logger.info(
            "Investigation binding probes: ar=%.3f ind=%.3f bind=%.3f bc=%.3f local_only=%s "
            "(%.0f+%.0f+%.0fms)",
            probe.ar_legacy_auc,
            probe.induction_screening_auc,
            probe.binding_screening_auc,
            bc,
            bool(result["local_only"]),
            probe.ar_elapsed_ms,
            probe.induction_elapsed_ms,
            probe.binding_elapsed_ms,
        )

        # Discovery: high AR without standard attention is a priority find
        _attn_ops = {
            "softmax_attention",
            "linear_attention",
            "diff_attention",
            "graph_attention",
            "local_window_attention",
        }
        _graph_str = graph_json_str or ""
        _has_attn = any(op in _graph_str for op in _attn_ops)
        if probe.ar_legacy_auc > 0.15 and not _has_attn:
            logger.warning(
                "DISCOVERY: High AR score without full attention — "
                "ar_legacy_auc=%.3f, model_source=%s, graph=%s",
                probe.ar_legacy_auc,
                model_source,
                _graph_str[:200],
            )
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.warning("Investigation binding probes skipped: %s", exc)

    if stop_event is not None and stop_event.is_set():
        del model
        return result

    # Gemini trajectory metrics on the trained investigation model.
    # Phase tag investigation_full so ML training distinguishes lifecycle.
    try:
        from ...eval.trajectory_metrics import compute_trajectory_metrics

        _traj = compute_trajectory_metrics(
            model,
            metric_phase="investigation_full",
            device=str(dev),
            spec_norm_vocab_size=int(getattr(config, "vocab_size", 32000)),
        )
        result.update(_traj.to_column_dict())
        logger.info(
            "Investigation trajectory: erf_d=%.2f erf_var=%.0f icld=%+.4f margin=%+.4f sn=%.1f",
            _traj.jacobian_erf.density or 0.0,
            _traj.jacobian_erf.variance or 0.0,
            _traj.icld.velocity or 0.0,
            _traj.logit_margin.velocity or 0.0,
            _traj.spec_norm or 0.0,
        )
    except (ImportError, RuntimeError, ValueError, TypeError) as exc:
        logger.warning("Investigation trajectory metrics skipped: %s", exc)

    del model
    return result


def _ar_gate_result_fields(ar_gate_result: Any) -> Dict[str, Any]:
    nano_ok = str(ar_gate_result.status or "") == "ok"
    score = None
    if nano_ok:
        score = round(
            0.6 * float(ar_gate_result.in_dist_pair_acc or 0.0)
            + 0.4 * float(ar_gate_result.held_class_acc or 0.0),
            4,
        )
    return {
        "ar_gate_metric_version": ar_gate_result.metric_version,
        "ar_gate_in_dist_pair_acc": (
            ar_gate_result.in_dist_pair_acc if nano_ok else None
        ),
        "ar_gate_in_dist_class_acc": (
            ar_gate_result.in_dist_class_acc if nano_ok else None
        ),
        "ar_gate_held_pair_acc": (ar_gate_result.held_pair_acc if nano_ok else None),
        "ar_gate_held_class_acc": (ar_gate_result.held_class_acc if nano_ok else None),
        "ar_gate_score": score,
        "ar_gate_status": ar_gate_result.status,
        "ar_gate_elapsed_ms": ar_gate_result.elapsed_ms,
        "ar_gate_train_steps_done": ar_gate_result.finetune_steps_done,
    }


def _run_investigation_ar_gate_probe(
    model: Any,
    dev: Any,
    *,
    graph_json_str: str | None = None,
) -> Dict[str, Any]:
    """Run the investigation-tier AR Gate probe.

    Live investigation reruns usually only have a graph spec available in the
    background benchmark worker, so use the locked standalone/wikitext-warmup
    config from the offline AR Gate backfill path when graph JSON is present.
    If a caller only has a live model, fall back to the production from-S1 mode.
    """
    from ...eval.ar_gate import ARGateConfig, ar_gate

    if graph_json_str:
        cfg = ARGateConfig(
            seed=0,
            wikitext_warmup_steps=2500,
            finetune_steps=400,
            n_pairs_per_noun=1,
            reps=10,
            n_distractors=480,
            n_adjectives=20,
            n_objects=25,
            timeout_s=600.0,
            from_s1=False,
        )
        nano = ar_gate(graph_json=graph_json_str, device=str(dev), cfg=cfg)
    else:
        cfg = ARGateConfig(
            seed=0,
            finetune_steps=400,
            n_pairs_per_noun=1,
            reps=10,
            n_distractors=480,
            n_adjectives=20,
            n_objects=25,
            timeout_s=600.0,
            from_s1=True,
        )
        nano = ar_gate(model=model, device=str(dev), cfg=cfg)

    fields = _ar_gate_result_fields(nano)
    logger.info(
        "Investigation AR Gate probe: score=%s in_pair=%.4f held_class=%.4f status=%s",
        (
            f"{fields['ar_gate_score']:.4f}"
            if fields["ar_gate_score"] is not None
            else "None"
        ),
        nano.in_dist_pair_acc,
        nano.held_class_acc,
        nano.status,
    )
    return fields


def _ar_validation_result_fields(result: Any) -> Dict[str, Any]:
    ok = str(result.status or "") == "ok"
    return {
        "ar_validation_metric_version": result.metric_version,
        "ar_validation_final_acc": result.final_acc if ok else None,
        "ar_validation_held_pair_acc": (result.held_pair_acc if ok else None),
        "ar_validation_held_class_acc": result.held_class_acc if ok else None,
        "ar_validation_learning_curve_json": json.dumps(
            result.learning_curve or [],
            sort_keys=True,
        ),
        "ar_validation_steps_to_floor": result.steps_to_floor,
        "ar_validation_rank_score": result.score if ok else None,
        "ar_validation_status": result.status,
        "ar_validation_elapsed_ms": result.elapsed_ms,
    }


def _run_ar_validation_probe(model: Any, dev: Any) -> Dict[str, Any]:
    from ...eval.ar_validation import run_ar_validation

    result = run_ar_validation(model, device=str(dev))
    fields = _ar_validation_result_fields(result)
    logger.info(
        "Champion AR Validation probe: score=%s held_pair=%.4f held_class=%.4f status=%s",
        (
            f"{fields['ar_validation_rank_score']:.4f}"
            if fields["ar_validation_rank_score"] is not None
            else "None"
        ),
        result.held_pair_acc,
        result.held_class_acc,
        result.status,
    )
    return fields


def _probe_dict_result_fields(
    result: Any,
    *,
    metric_version_key: str,
    status_key: str,
    elapsed_ms_key: str,
    error_key: str,
    metric_keys: tuple[str, ...],
) -> Dict[str, Any]:
    fields = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    ok = str(fields.get(status_key) or "") == "ok"
    if not ok:
        for key in metric_keys:
            fields[key] = None
    return {
        metric_version_key: fields.get(metric_version_key),
        **{key: fields.get(key) for key in metric_keys},
        status_key: fields.get(status_key),
        elapsed_ms_key: fields.get(elapsed_ms_key),
        error_key: fields.get(error_key),
    }


def _ar_intermediate_result_fields(result: Any) -> Dict[str, Any]:
    fields = _probe_dict_result_fields(
        result,
        metric_version_key="ar_intermediate_metric_version",
        status_key="ar_intermediate_status",
        elapsed_ms_key="ar_intermediate_elapsed_ms",
        error_key="ar_intermediate_error",
        metric_keys=(
            *_AR_INTERMEDIATE_SCORE_FIELDS,
            "ar_intermediate_learning_curve_json",
        ),
    )
    return fields


def _binding_multislot_result_fields(result: Any) -> Dict[str, Any]:
    fields = _probe_dict_result_fields(
        result,
        metric_version_key="binding_multislot_metric_version",
        status_key="binding_multislot_status",
        elapsed_ms_key="binding_multislot_elapsed_ms",
        error_key="binding_multislot_error",
        metric_keys=(
            *_BINDING_MULTISLOT_SCORE_FIELDS,
            "binding_multislot_learning_curve_json",
        ),
    )
    return fields


def _run_ar_intermediate_probe(model: Any, dev: Any) -> Dict[str, Any]:
    from ...eval.ar_intermediate_probe import ar_intermediate_probe

    result = ar_intermediate_probe(model, device=str(dev))
    fields = _ar_intermediate_result_fields(result)
    logger.info(
        "AR intermediate probe: score=%s held_pair=%s auc_lift=%s status=%s",
        fields.get("ar_intermediate_diagnostic_score"),
        fields.get("ar_intermediate_held_pair_acc"),
        fields.get("ar_intermediate_auc_lift"),
        fields.get("ar_intermediate_status"),
    )
    return fields


def _run_binding_multislot_probe(model: Any, dev: Any) -> Dict[str, Any]:
    from ...eval.binding_multislot_probe import binding_multislot_probe

    result = binding_multislot_probe(model, device=str(dev))
    fields = _binding_multislot_result_fields(result)
    logger.info(
        "Binding multislot probe: score=%s held_slot=%s two_plus=%s status=%s",
        fields.get("binding_multislot_diagnostic_score"),
        fields.get("binding_multislot_held_entity_slot_acc"),
        fields.get("binding_multislot_two_plus_slots_acc"),
        fields.get("binding_multislot_status"),
    )
    return fields


def _run_investigation_v2_probes(
    model: Any,
    dev: Any,
    *,
    graph_json_str: str | None = None,
    run_induction_intermediate: bool = True,
    run_binding_intermediate: bool = True,
    run_ar_gate: bool = False,
    run_induction_validation: bool = False,
    induction_validation_extended_budget: bool = False,
    run_ar_validation_probe: bool = False,
    run_ar_intermediate: bool = False,
    run_binding_multislot: bool = False,
) -> Dict[str, Any]:
    """Run investigation-tier capability probes on the benchmark model."""
    result: Dict[str, Any] = {}

    if run_induction_intermediate:
        try:
            from ...eval.induction_intermediate_probe import (
                run_induction_intermediate as _run_induction_intermediate,
            )

            induction_intermediate = _run_induction_intermediate(model, device=dev)
            induction_intermediate_ok = str(induction_intermediate.status or "") == "ok"
            result.update(
                {
                    "induction_intermediate_auc": (
                        induction_intermediate.auc
                        if induction_intermediate_ok
                        else None
                    ),
                    "induction_intermediate_max_gap_acc": (
                        induction_intermediate.max_gap_acc
                        if induction_intermediate_ok
                        else None
                    ),
                    "induction_intermediate_gap_accuracies_json": json.dumps(
                        induction_intermediate.gap_accuracies or {},
                        sort_keys=True,
                    ),
                    "induction_intermediate_steps_trained": induction_intermediate.steps_trained,
                    "induction_intermediate_status": induction_intermediate.status,
                    "induction_intermediate_elapsed_ms": induction_intermediate.elapsed_ms,
                    "induction_intermediate_protocol_version": (
                        induction_intermediate.protocol_version
                    ),
                }
            )
            logger.info(
                "Capability induction-intermediate probe: auc=%.4f max_gap=%.4f status=%s",
                induction_intermediate.auc,
                induction_intermediate.max_gap_acc,
                induction_intermediate.status,
            )
        except (ImportError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Capability induction-intermediate probe skipped: %s", exc)

    if run_induction_validation:
        try:
            from ...eval.induction_validation_probe import (
                run_induction_validation_champion,
            )

            induction_validation = run_induction_validation_champion(
                model,
                device=dev,
                extended_budget=induction_validation_extended_budget,
            )
            induction_validation_ok = str(induction_validation.status or "") == "ok"
            result.update(
                {
                    "induction_validation_auc": (
                        induction_validation.auc if induction_validation_ok else None
                    ),
                    "induction_validation_max_gap_acc": (
                        induction_validation.max_gap_acc
                        if induction_validation_ok
                        else None
                    ),
                    "induction_validation_gap_accuracy_cv": (
                        induction_validation.gap_accuracy_cv
                        if induction_validation_ok
                        else None
                    ),
                    "induction_validation_gap_accuracies_json": json.dumps(
                        induction_validation.gap_accuracies or {},
                        sort_keys=True,
                    ),
                    "induction_validation_steps_trained": induction_validation.steps_trained,
                    "induction_validation_status": induction_validation.status,
                    "induction_validation_elapsed_ms": induction_validation.elapsed_ms,
                    "induction_validation_protocol_version": induction_validation.protocol_version,
                }
            )
            logger.info(
                "Champion induction-validation probe: auc=%.4f max_gap=%.4f cv=%.4f status=%s",
                induction_validation.auc,
                induction_validation.max_gap_acc,
                induction_validation.gap_accuracy_cv,
                induction_validation.status,
            )
        except (ImportError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Champion induction-validation probe skipped: %s", exc)

    if run_binding_intermediate:
        try:
            from ...eval.binding_intermediate_probe import (
                run_binding_intermediate as _run_binding_intermediate,
            )

            binding_intermediate = _run_binding_intermediate(model, device=dev)
            binding_intermediate_ok = str(binding_intermediate.status or "") == "ok"
            result.update(
                {
                    "binding_intermediate_auc": (
                        binding_intermediate.auc if binding_intermediate_ok else None
                    ),
                    "binding_intermediate_max_distance_acc": (
                        binding_intermediate.max_distance_acc
                        if binding_intermediate_ok
                        else None
                    ),
                    "binding_intermediate_distance_accuracies_json": json.dumps(
                        binding_intermediate.distance_accuracies or {},
                        sort_keys=True,
                    ),
                    "binding_intermediate_train_steps": binding_intermediate.train_steps,
                    "binding_intermediate_status": binding_intermediate.status,
                    "binding_intermediate_elapsed_ms": binding_intermediate.elapsed_ms,
                    "binding_intermediate_protocol_version": (
                        binding_intermediate.protocol_version
                    ),
                }
            )
            logger.info(
                "Capability binding-intermediate probe: auc=%.4f max_distance=%.4f status=%s",
                binding_intermediate.auc,
                binding_intermediate.max_distance_acc,
                binding_intermediate.status,
            )
        except (ImportError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Capability binding-intermediate probe skipped: %s", exc)

    if run_ar_gate:
        try:
            result.update(
                _run_investigation_ar_gate_probe(
                    model,
                    dev,
                    graph_json_str=graph_json_str,
                )
            )
        except (ImportError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Investigation AR Gate probe skipped: %s", exc)

    if run_ar_validation_probe:
        try:
            result.update(_run_ar_validation_probe(model, dev))
        except (ImportError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Champion AR Validation probe skipped: %s", exc)

    if run_ar_intermediate:
        try:
            result.update(_run_ar_intermediate_probe(model, dev))
        except (ImportError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("AR intermediate probe skipped: %s", exc)

    if run_binding_multislot:
        try:
            result.update(_run_binding_multislot_probe(model, dev))
        except (ImportError, RuntimeError, ValueError, TypeError) as exc:
            logger.warning("Binding multislot probe skipped: %s", exc)

    return result


def _evaluate_capability_rankers(
    *,
    config,
    dev,
    model_source: str,
    arch_spec_json_str: str | None,
    graph_json_str: str | None,
    cached_json_load,
    stop_event=None,
) -> Dict[str, Any]:
    """Run selective capability rankers without validation semantics."""
    if stop_event is not None and stop_event.is_set():
        return {}
    try:
        model = _build_benchmark_model(
            config=config,
            dev=dev,
            model_source=model_source,
            arch_spec_json_str=arch_spec_json_str,
            graph_json_str=graph_json_str,
            cached_json_load=cached_json_load,
        )
    except (ImportError, RuntimeError, ValueError, TypeError) as exc:
        logger.debug("Capability-ranking model build failed: %s", exc)
        return {"capability_ranking_status": "model_build_failed", "error": str(exc)}
    if model is None:
        return {"capability_ranking_status": "no_model"}
    try:
        if stop_event is not None and stop_event.is_set():
            return {}
        result = _run_investigation_v2_probes(
            model,
            dev,
            graph_json_str=graph_json_str,
            run_induction_intermediate=bool(
                getattr(config, "capability_ranking_run_intermediate", True)
            ),
            run_binding_intermediate=bool(
                getattr(config, "capability_ranking_run_intermediate", True)
            ),
            run_ar_gate=False,
            run_induction_validation=bool(
                getattr(config, "capability_ranking_run_induction_validation", False)
            ),
            induction_validation_extended_budget=bool(
                getattr(config, "champion_induction_validation_extended_budget", False)
            ),
            run_ar_validation_probe=bool(
                getattr(config, "capability_ranking_run_ar_validation", False)
            ),
            run_ar_intermediate=bool(
                getattr(config, "capability_ranking_run_ar_intermediate", True)
            ),
            run_binding_multislot=bool(
                getattr(config, "capability_ranking_run_binding_multislot", True)
            ),
        )
        result["capability_ranking_status"] = (
            "ok" if has_capability_ranker_evidence(result) else "no_metrics"
        )
        return result
    finally:
        del model


def _record_capability_ranking_result(
    *,
    nb,
    exp_id: str,
    source_result_id: str,
    source: Dict[str, Any],
    model_source: str,
    benchmark_result: Dict[str, Any],
) -> None:
    """Persist capability-ranking metrics without claiming validation evidence."""
    graph_fingerprint = str(source.get("graph_fingerprint") or "").strip()
    leaderboard_result_id = source_result_id
    existing_entry: Dict[str, Any] | None = None
    if graph_fingerprint:
        try:
            existing_fp_entry = nb.get_leaderboard_entry_by_fingerprint(
                graph_fingerprint
            )
            if existing_fp_entry and existing_fp_entry.get("result_id"):
                leaderboard_result_id = str(existing_fp_entry["result_id"])
                existing_entry = dict(existing_fp_entry)
        except Exception as exc:
            logger.debug(
                "Capability-ranking fp anchor lookup failed for %s: %s",
                graph_fingerprint[:16],
                exc,
            )
    if existing_entry is None:
        try:
            row = nb.get_leaderboard_entry(leaderboard_result_id)
            existing_entry = dict(row) if row else None
        except Exception as exc:
            logger.debug(
                "Capability-ranking leaderboard source lookup failed for %s: %s",
                leaderboard_result_id,
                exc,
            )

    def _source_or_existing(key: str) -> Any:
        value = source.get(key)
        if value not in (None, ""):
            return value
        if existing_entry:
            return existing_entry.get(key)
        return None

    ranker_fields = {
        key: value
        for key, value in capability_ranker_fields(benchmark_result).items()
        if value is not None
    }
    if not ranker_fields:
        return

    common = {
        "model_source": model_source,
        "architecture_desc": graph_fingerprint[:40],
        "graph_fingerprint": graph_fingerprint
        or _source_or_existing("graph_fingerprint"),
        "tier": _safe_tier(nb, leaderboard_result_id, "capability_ranking"),
        "screening_loss_ratio": (
            source.get("loss_ratio")
            if source.get("loss_ratio") is not None
            else _source_or_existing("screening_loss_ratio")
        ),
        "screening_novelty": (
            source.get("novelty_score")
            if source.get("novelty_score") is not None
            else _source_or_existing("screening_novelty")
        ),
        "screening_passed": True,
        "investigation_loss_ratio": _source_or_existing("investigation_loss_ratio"),
        "investigation_robustness": _source_or_existing("investigation_robustness"),
        "investigation_passed": _source_or_existing("investigation_passed"),
        "novelty_confidence": _source_or_existing("novelty_confidence"),
        "fp_jacobian_spectral_norm": _source_or_existing("fp_jacobian_spectral_norm"),
        "routing_savings_ratio": _source_or_existing("routing_savings_ratio"),
        "activation_sparsity_score": _source_or_existing("activation_sparsity_score"),
        "depth_savings_ratio": _source_or_existing("depth_savings_ratio"),
        "compression_ratio": _source_or_existing("compression_ratio"),
        "result_cohort": _source_or_existing("result_cohort"),
        "trust_label": _source_or_existing("trust_label"),
        "comparability_label": _source_or_existing("comparability_label"),
        "evaluation_protocol_version": (
            _source_or_existing("evaluation_protocol_version")
            or "capability_ranking_v1"
        ),
    }
    nb.upsert_leaderboard(
        result_id=leaderboard_result_id,
        **common,
        **ranker_fields,
    )

    set_parts = []
    set_params: List[Any] = []
    source_updates = {
        **ranker_fields,
        "evaluation_protocol_version": (
            source.get("evaluation_protocol_version") or "capability_ranking_v1"
        ),
    }
    for col, value in source_updates.items():
        if value is None:
            continue
        set_parts.append(f"{col} = ?")
        set_params.append(value)
    if set_parts:
        set_params.append(source_result_id)
        nb.conn.execute(
            f"UPDATE program_results SET {', '.join(set_parts)} WHERE result_id = ?",
            set_params,
        )
        nb._maybe_commit()


# Single-threaded pool for background benchmark evals — avoids blocking the
# investigation loop while still serialising GPU work.
_benchmark_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bench")


def _submit_benchmark_eval(
    *,
    nb,
    exp_id: str,
    source_result_id: str,
    source: Dict[str, Any],
    model_source: str,
    graph_json_str: str | None,
    arch_spec_json_str: str | None,
    n_passed: int,
    n_programs_tested: int,
    best_lr: Any,
    best_tp_json: str | None,
    robustness: float,
    investigation_passed: bool,
    config,
    dev,
    cached_json_load,
    fingerprint_incomplete: bool = False,
    best_training_curve: list[dict] | None = None,
    stop_event=None,
) -> Future:
    """Submit benchmark evals + result recording to a background thread.

    The investigation loop can continue to the next candidate immediately
    instead of blocking on 400 training steps per benchmark.

    Creates a fresh LabNotebook connection in the background thread because
    SQLite connections cannot be shared across threads (check_same_thread).

    ``stop_event`` is consulted at the top of the worker (so queued jobs
    drop immediately on stop) and threaded into the benchmark evaluator
    so mid-run aborts are honored too.
    """
    db_path = str(nb.db_path)

    def _run() -> None:
        if stop_event is not None and stop_event.is_set():
            return
        benchmark_result = _evaluate_investigation_benchmarks(
            config=config,
            dev=dev,
            model_source=model_source,
            arch_spec_json_str=arch_spec_json_str,
            graph_json_str=graph_json_str,
            cached_json_load=cached_json_load,
            stop_event=stop_event,
        )
        if stop_event is not None and stop_event.is_set():
            return
        # Create a thread-local notebook for DB writes
        from ..notebook import LabNotebook

        thread_nb = LabNotebook(db_path)
        try:
            _record_investigation_result(
                nb=thread_nb,
                exp_id=exp_id,
                source_result_id=source_result_id,
                source=source,
                model_source=model_source,
                graph_json_str=graph_json_str,
                arch_spec_json_str=arch_spec_json_str,
                n_passed=n_passed,
                n_programs_tested=n_programs_tested,
                best_lr=best_lr,
                best_tp_json=best_tp_json,
                robustness=robustness,
                investigation_passed=investigation_passed,
                benchmark_result=benchmark_result,
                fingerprint_incomplete=fingerprint_incomplete,
                best_training_curve=best_training_curve,
            )
            thread_nb.flush_writes()
        finally:
            thread_nb.close()

    return _benchmark_pool.submit(_run)


def _submit_v2_probe_eval(
    *,
    nb,
    exp_id: str,
    source_result_id: str,
    source: Dict[str, Any],
    model_source: str,
    graph_json_str: str | None,
    arch_spec_json_str: str | None,
    n_passed: int,
    n_programs_tested: int,
    best_lr: Any,
    best_tp_json: str | None,
    robustness: float,
    investigation_passed: bool,
    config,
    dev,
    cached_json_load,
    fingerprint_incomplete: bool = False,
    stop_event=None,
) -> Future:
    """Submit v2-only investigation probes when no training program passes.

    The v2 probes train their own probe heads/tasks, so they can still produce
    useful induction/binding evidence for a compiled graph even when the
    investigation training recipe did not pass.

    ``stop_event`` lets ``runner.stop()`` cancel queued jobs that haven't
    started executing yet so the background pool drains promptly.
    """
    db_path = str(nb.db_path)

    def _run() -> None:
        if stop_event is not None and stop_event.is_set():
            return
        benchmark_result: Dict[str, Any] = {}
        try:
            model = _build_benchmark_model(
                config=config,
                dev=dev,
                model_source=model_source,
                arch_spec_json_str=arch_spec_json_str,
                graph_json_str=graph_json_str,
                cached_json_load=cached_json_load,
            )
            if model is not None:
                try:
                    benchmark_result.update(
                        _run_investigation_v2_probes(
                            model,
                            dev,
                            graph_json_str=graph_json_str,
                            run_ar_gate=True,
                            run_ar_intermediate=bool(
                                getattr(config, "run_ar_intermediate", False)
                            ),
                            run_binding_multislot=bool(
                                getattr(config, "run_binding_multislot", False)
                            ),
                        )
                    )
                finally:
                    del model
        except (ImportError, RuntimeError, ValueError, TypeError) as exc:
            logger.debug("Investigation v2-only probe eval skipped: %s", exc)
        if stop_event is not None and stop_event.is_set():
            return

        from ..notebook import LabNotebook

        thread_nb = LabNotebook(db_path)
        try:
            _record_investigation_result(
                nb=thread_nb,
                exp_id=exp_id,
                source_result_id=source_result_id,
                source=source,
                model_source=model_source,
                graph_json_str=graph_json_str,
                arch_spec_json_str=arch_spec_json_str,
                n_passed=n_passed,
                n_programs_tested=n_programs_tested,
                best_lr=best_lr,
                best_tp_json=best_tp_json,
                robustness=robustness,
                investigation_passed=investigation_passed,
                benchmark_result=benchmark_result,
                fingerprint_incomplete=fingerprint_incomplete,
            )
            thread_nb.flush_writes()
        finally:
            thread_nb.close()

    return _benchmark_pool.submit(_run)


def _safe_tier(nb, result_id: str, proposed: str) -> str:
    """Return the higher of existing tier and proposed tier to prevent downgrades."""
    try:
        row = nb.conn.execute(
            "SELECT tier FROM leaderboard WHERE result_id = ?", (result_id,)
        ).fetchone()
        if row:
            existing = str(row["tier"] or "screening")
            if TIER_RANK.get(existing, 0) > TIER_RANK.get(proposed, 0):
                return existing
    except (OSError, RuntimeError) as e:
        logger.debug("_safe_tier lookup failed: %s", e)
    return proposed


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _capability_override_investigation_passed(
    *,
    benchmark_result: Dict[str, Any],
    best_lr: Any,
    screening_lr: Any,
    robustness: float,
    n_passed: int,
    n_programs_tested: int,
    fingerprint_incomplete: bool,
) -> tuple[bool, list[str]]:
    """Allow strong capability evidence to rescue a near-miss loss result."""
    if fingerprint_incomplete or n_programs_tested <= 0:
        return False, []
    if robustness < 0.5 or n_passed <= 0:
        return False, []

    lr = _to_float_or_none(best_lr)
    if lr is None or lr >= 0.65:
        return False, []

    screening = _to_float_or_none(screening_lr)
    multiplier = lr / max(screening, 1e-6) if screening is not None else None
    if multiplier is not None and multiplier > 1.25:
        return False, []

    reasons: list[str] = []
    ar_gate = _to_float_or_none(benchmark_result.get("ar_gate_score"))
    ind_v2 = _to_float_or_none(benchmark_result.get("induction_intermediate_auc"))
    bind_v2 = _to_float_or_none(benchmark_result.get("binding_intermediate_auc"))

    if ar_gate is not None and ar_gate >= 0.8:
        reasons.append(f"ar_gate={ar_gate:.3f}")
    if bind_v2 is not None and bind_v2 >= 0.08:
        reasons.append(f"binding_intermediate={bind_v2:.3f}")
    if ind_v2 is not None and ind_v2 >= 0.30:
        reasons.append(f"induction_intermediate={ind_v2:.3f}")

    if len(reasons) < 2:
        return False, []
    reasons.append(f"loss_ratio={lr:.3f}")
    reasons.append(f"robustness={robustness:.3f}")
    if multiplier is not None:
        reasons.append(f"loss_multiplier={multiplier:.3f}")
    return True, reasons


def _investigation_tier_for_result(
    *,
    investigation_passed: bool,
    fingerprint_incomplete: bool,
    n_passed: int,
    n_programs_tested: int,
) -> str:
    """Choose the canonical current-status tier for an investigation result.

    ``investigation_passed`` remains the strict gate for validation-readiness.
    A run that was fully reproducible across the complete investigation program
    set is still treated as having reached the investigation tier even when it
    failed the stricter loss-ratio gate.
    """
    if investigation_passed:
        return "investigation"
    if fingerprint_incomplete:
        return "investigation_fingerprint_incomplete"
    if n_programs_tested >= 3 and n_passed == n_programs_tested:
        return "investigation"
    if n_programs_tested > 0 and (n_passed / max(n_programs_tested, 1)) >= 0.5:
        return "investigation"
    return "investigation_failed"


def _record_investigation_result(
    *,
    nb,
    exp_id: str,
    source_result_id: str,
    source: Dict[str, Any],
    model_source: str,
    graph_json_str: str | None,
    arch_spec_json_str: str | None,
    n_passed: int,
    n_programs_tested: int,
    best_lr: Any,
    best_tp_json: str | None,
    robustness: float,
    investigation_passed: bool,
    benchmark_result: Dict[str, Any],
    fingerprint_incomplete: bool = False,
    best_training_curve: list[dict] | None = None,
) -> None:
    """Persist leaderboard and program-results updates for investigation.

    Protects existing investigation data: if the entry already has better
    investigation results (lower loss ratio, higher robustness), those are
    preserved rather than overwritten by a weaker re-investigation.
    """
    graph_fingerprint = str(source.get("graph_fingerprint") or "").strip()
    leaderboard_result_id = source_result_id
    existing_fp_entry = None
    if graph_fingerprint:
        try:
            existing_fp_entry = nb.get_leaderboard_entry_by_fingerprint(
                graph_fingerprint
            )
            if existing_fp_entry and existing_fp_entry.get("result_id"):
                leaderboard_result_id = str(existing_fp_entry["result_id"])
        except Exception as exc:
            logger.debug(
                "Fingerprint leaderboard anchor lookup failed for %s: %s",
                graph_fingerprint[:16],
                exc,
            )
    source_result_cohort = str(source.get("result_cohort") or "").strip().lower()
    source_trust_label = str(source.get("trust_label") or "").strip().lower()
    source_comparability = str(source.get("comparability_label") or "").strip().lower()
    parent_result_cohort = (
        str((existing_fp_entry or {}).get("result_cohort") or "").strip().lower()
    )
    parent_trust_label = (
        str((existing_fp_entry or {}).get("trust_label") or "").strip().lower()
    )
    parent_comparability = (
        str((existing_fp_entry or {}).get("comparability_label") or "").strip().lower()
    )

    result_cohort = parent_result_cohort or source_result_cohort or "search"
    trust_label = parent_trust_label or source_trust_label or "candidate_grade"
    comparability_label = (
        parent_comparability or source_comparability or "candidate_comparable"
    )
    if result_cohort == "backfill" or trust_label == "backfill_observation":
        result_cohort = "search"
        trust_label = "candidate_grade"
    if comparability_label == "reconstructed_init_variant":
        comparability_label = "candidate_comparable"
    evaluation_protocol_version = (
        (existing_fp_entry or {}).get("evaluation_protocol_version")
        or source.get("evaluation_protocol_version")
        or "candidate_grade_v1"
    )

    # Check if existing investigation results are better — never overwrite with worse
    existing_inv = nb.conn.execute(
        "SELECT investigation_loss_ratio, investigation_robustness, investigation_passed, "
        "investigation_best_training FROM leaderboard WHERE result_id = ?",
        (leaderboard_result_id,),
    ).fetchone()
    if existing_inv and existing_inv["investigation_passed"]:
        existing_lr = existing_inv["investigation_loss_ratio"]
        # Never overwrite a passed investigation with a failed one or worse results
        if best_lr is None or (existing_lr is not None and existing_lr <= best_lr):
            best_lr = existing_lr
            robustness = max(
                robustness, float(existing_inv["investigation_robustness"] or 0)
            )
            best_tp_json = existing_inv["investigation_best_training"] or best_tp_json
            investigation_passed = True

    if not investigation_passed:
        capability_override, capability_reasons = (
            _capability_override_investigation_passed(
                benchmark_result=benchmark_result,
                best_lr=best_lr,
                screening_lr=source.get("loss_ratio"),
                robustness=float(robustness or 0.0),
                n_passed=n_passed,
                n_programs_tested=n_programs_tested,
                fingerprint_incomplete=fingerprint_incomplete,
            )
        )
        if capability_override:
            investigation_passed = True
            logger.info(
                "Investigation capability override passed for %s: %s",
                source_result_id[:12],
                ", ".join(capability_reasons),
            )

    # HellaSwag hard gate: DISABLED — doesn't differentiate at nano scale.

    # Binding probe: informational logging only. No hard gate — probes are
    # too noisy at nano scale (Mamba fluctuates 0.01-0.13 across runs).
    # The soft penalty in compute_composite handles score reduction.
    _bp_ind = benchmark_result.get("induction_screening_auc")
    if _bp_ind is not None and _bp_ind < 0.03:
        logger.info(
            "Binding probe: %s ind=%.3f (local-only signal, soft penalty applied in scoring)",
            source_result_id[:8],
            _bp_ind,
        )

    trajectory_fields = trajectory_probe_fields(benchmark_result)
    proposed_tier = _investigation_tier_for_result(
        investigation_passed=investigation_passed,
        fingerprint_incomplete=fingerprint_incomplete,
        n_passed=n_passed,
        n_programs_tested=n_programs_tested,
    )
    nb.upsert_leaderboard(
        result_id=leaderboard_result_id,
        model_source=model_source,
        architecture_desc=graph_fingerprint[:40],
        screening_loss_ratio=source.get("loss_ratio"),
        screening_novelty=source.get("novelty_score"),
        screening_passed=True,
        investigation_loss_ratio=best_lr,
        investigation_robustness=robustness,
        investigation_best_training=best_tp_json,
        investigation_passed=investigation_passed,
        tier=_safe_tier(nb, leaderboard_result_id, proposed_tier),
        result_cohort=result_cohort,
        trust_label=trust_label,
        comparability_label=comparability_label,
        evaluation_protocol_version=evaluation_protocol_version,
        novelty_confidence=source.get("novelty_confidence"),
        fp_jacobian_spectral_norm=source.get("fp_jacobian_spectral_norm"),
        wikitext_perplexity=benchmark_result.get("inv_wikitext_ppl"),
        wikitext_score=benchmark_result.get("inv_wikitext_score"),
        tinystories_perplexity=benchmark_result.get("inv_tinystories_ppl"),
        tinystories_score=benchmark_result.get("inv_tinystories_score"),
        routing_savings_ratio=source.get("routing_savings_ratio"),
        activation_sparsity_score=source.get("activation_sparsity_score"),
        depth_savings_ratio=source.get("depth_savings_ratio"),
        compression_ratio=source.get("compression_ratio"),
        loss_improvement_rate=source.get("loss_improvement_rate"),
        hellaswag_acc=benchmark_result.get("hellaswag_acc"),
        hellaswag_metric_version=benchmark_result.get("hellaswag_metric_version"),
        hellaswag_tokenizer_mode=benchmark_result.get("hellaswag_tokenizer_mode"),
        hellaswag_tiktoken_encoding=benchmark_result.get("hellaswag_tiktoken_encoding"),
        ar_legacy_auc=benchmark_result.get("ar_legacy_auc"),
        induction_screening_auc=benchmark_result.get("induction_screening_auc"),
        binding_screening_auc=benchmark_result.get("binding_screening_auc"),
        binding_screening_composite=benchmark_result.get("binding_screening_composite"),
        local_only=benchmark_result.get("local_only"),
        **trajectory_fields,
    )

    v2_fields = {
        key: benchmark_result.get(key)
        for key in (
            "induction_intermediate_auc",
            "induction_intermediate_max_gap_acc",
            "induction_intermediate_gap_accuracies_json",
            "induction_intermediate_steps_trained",
            "induction_intermediate_status",
            "induction_intermediate_elapsed_ms",
            "induction_intermediate_protocol_version",
            "binding_intermediate_auc",
            "binding_intermediate_max_distance_acc",
            "binding_intermediate_distance_accuracies_json",
            "binding_intermediate_train_steps",
            "binding_intermediate_status",
            "binding_intermediate_elapsed_ms",
            "binding_intermediate_protocol_version",
        )
    }
    v3_fields = {
        key: benchmark_result.get(key)
        for key in (
            "induction_validation_auc",
            "induction_validation_max_gap_acc",
            "induction_validation_gap_accuracy_cv",
            "induction_validation_gap_accuracies_json",
            "induction_validation_steps_trained",
            "induction_validation_status",
            "induction_validation_elapsed_ms",
            "induction_validation_protocol_version",
        )
    }
    ar_gate_fields = {
        key: benchmark_result.get(key)
        for key in (
            "ar_gate_metric_version",
            "ar_gate_in_dist_pair_acc",
            "ar_gate_in_dist_class_acc",
            "ar_gate_held_pair_acc",
            "ar_gate_held_class_acc",
            "ar_gate_score",
            "ar_gate_status",
            "ar_gate_elapsed_ms",
            "ar_gate_train_steps_done",
        )
    }
    ar_validation_fields = {
        key: benchmark_result.get(key)
        for key in (
            "ar_validation_metric_version",
            "ar_validation_final_acc",
            "ar_validation_held_pair_acc",
            "ar_validation_held_class_acc",
            "ar_validation_learning_curve_json",
            "ar_validation_steps_to_floor",
            "ar_validation_rank_score",
            "ar_validation_status",
            "ar_validation_elapsed_ms",
        )
    }
    ar_intermediate_fields = {
        key: benchmark_result.get(key)
        for key in (
            "ar_intermediate_metric_version",
            *_AR_INTERMEDIATE_SCORE_FIELDS,
            "ar_intermediate_learning_curve_json",
            "ar_intermediate_status",
            "ar_intermediate_elapsed_ms",
            "ar_intermediate_error",
        )
    }
    binding_multislot_fields = {
        key: benchmark_result.get(key)
        for key in (
            "binding_multislot_metric_version",
            *_BINDING_MULTISLOT_SCORE_FIELDS,
            "binding_multislot_learning_curve_json",
            "binding_multislot_status",
            "binding_multislot_elapsed_ms",
            "binding_multislot_error",
        )
    }
    if any(
        value is not None
        for value in (
            *v2_fields.values(),
            *v3_fields.values(),
            *ar_gate_fields.values(),
            *ar_validation_fields.values(),
            *ar_intermediate_fields.values(),
            *binding_multislot_fields.values(),
        )
    ):
        nb.upsert_leaderboard(
            result_id=leaderboard_result_id,
            model_source=model_source,
            architecture_desc=graph_fingerprint[:40],
            tier=_safe_tier(nb, leaderboard_result_id, proposed_tier),
            result_cohort=result_cohort,
            trust_label=trust_label,
            comparability_label=comparability_label,
            evaluation_protocol_version=evaluation_protocol_version,
            **v2_fields,
            **v3_fields,
            **ar_gate_fields,
            **ar_validation_fields,
            **ar_intermediate_fields,
            **binding_multislot_fields,
        )
    # Investigation S1 metric completeness gate.  The investigation pipeline
    # runs blimp + v1 probes (induction/binding/ar) AND the v2 capability
    # probes inside _evaluate_investigation_benchmarks.  When training trips
    # an inflight check (e.g., inflight_no_progress) the model may survive
    # but downstream probes can silently skip via their try/except blocks.
    # If ANY of the 7 universal-guard required metrics is missing, claiming
    # stage1_passed=True would write a partial-data row that the universal
    # guard rejects entirely — losing the investigation observation.
    # Persist what we measured but downgrade stage1_passed to False so the
    # observation lands without violating the no-missing-data rule.
    _inv_wikitext_ppl = benchmark_result.get(
        "wikitext_perplexity"
    ) or benchmark_result.get("inv_wikitext_ppl")
    _required_s1_metrics = {
        "wikitext_perplexity": _inv_wikitext_ppl,
        "hellaswag_acc": benchmark_result.get("hellaswag_acc"),
        "blimp_overall_accuracy": benchmark_result.get("blimp_overall_accuracy"),
        "induction_screening_auc": benchmark_result.get("induction_screening_auc"),
        "binding_screening_auc": benchmark_result.get("binding_screening_auc"),
        "binding_screening_composite": benchmark_result.get(
            "binding_screening_composite"
        ),
        "ar_legacy_auc": benchmark_result.get("ar_legacy_auc"),
    }
    _missing_s1_metrics = [k for k, v in _required_s1_metrics.items() if v is None]
    _training_survived = bool(n_passed > 0)
    _stage1_passed = _training_survived and not _missing_s1_metrics
    if _training_survived and _missing_s1_metrics:
        logger.warning(
            "Investigation %s: training survived (n_passed=%d) but post-S1 probes "
            "incomplete — missing %s.  Recording stage1_passed=False to honor the "
            "no-missing-data rule; the row preserves whatever metrics did land.",
            source_result_id[:10],
            n_passed,
            _missing_s1_metrics,
        )
    result_id = nb.record_program_result(
        experiment_id=exp_id,
        graph_fingerprint=source.get("graph_fingerprint", source_result_id),
        graph_json=graph_json_str or "{}",
        intentional_rerun_reason="exact_graph_replay",
        stage0_passed=True,
        stage05_passed=True,
        stage1_passed=_stage1_passed,
        loss_ratio=best_lr,
        novelty_score=source.get("novelty_score"),
        novelty_confidence=source.get("novelty_confidence"),
        novelty_raw_score=source.get("novelty_raw_score"),
        novelty_z_score=source.get("novelty_z_score"),
        novelty_reference_version=source.get("novelty_reference_version"),
        novelty_valid_for_promotion=source.get("novelty_valid_for_promotion"),
        novelty_validity_reason=source.get("novelty_validity_reason"),
        novelty_requires_justification=source.get("novelty_requires_justification"),
        training_program_json=best_tp_json,
        model_source=model_source,
        arch_spec_json=arch_spec_json_str,
        result_cohort=result_cohort,
        trust_label=trust_label,
        comparability_label=comparability_label,
        evaluation_protocol_version=evaluation_protocol_version,
        wikitext_perplexity=_inv_wikitext_ppl,
        wikitext_score=benchmark_result.get("inv_wikitext_score"),
        tinystories_perplexity=benchmark_result.get("inv_tinystories_ppl"),
        tinystories_score=benchmark_result.get("inv_tinystories_score"),
        wikitext_ppl_200=benchmark_result.get("wikitext_ppl_200"),
        wikitext_ppl_500=benchmark_result.get("wikitext_ppl_500"),
        wikitext_improvement_ratio=benchmark_result.get("wikitext_improvement_ratio"),
        wikitext_eval_steps=benchmark_result.get("wikitext_eval_steps"),
        hellaswag_acc=benchmark_result.get("hellaswag_acc"),
        hellaswag_status=benchmark_result.get("hellaswag_status"),
        hellaswag_n_examples=benchmark_result.get("hellaswag_total"),
        hellaswag_metric_version=benchmark_result.get("hellaswag_metric_version"),
        hellaswag_tokenizer_mode=benchmark_result.get("hellaswag_tokenizer_mode"),
        hellaswag_tiktoken_encoding=benchmark_result.get("hellaswag_tiktoken_encoding"),
        # The 5 fields that prior to 2026-05-02 were silently dropped on the
        # investigation rerun row, causing every investigation re-record to
        # fail the universal S1 guard and discard the entire observation.
        blimp_overall_accuracy=benchmark_result.get("blimp_overall_accuracy"),
        blimp_subtask_accuracies_json=benchmark_result.get(
            "blimp_subtask_accuracies_json"
        ),
        blimp_n_subtasks=benchmark_result.get("blimp_n_subtasks"),
        blimp_status=benchmark_result.get("blimp_status"),
        induction_screening_auc=benchmark_result.get("induction_screening_auc"),
        binding_screening_auc=benchmark_result.get("binding_screening_auc"),
        binding_curriculum_auc=benchmark_result.get("binding_curriculum_auc"),
        binding_screening_composite=benchmark_result.get("binding_screening_composite"),
        ar_legacy_auc=benchmark_result.get("ar_legacy_auc"),
        ar_legacy_final_acc=benchmark_result.get("ar_legacy_final_acc"),
        ar_legacy_timed_out=benchmark_result.get("ar_legacy_timed_out"),
        ar_legacy_above_chance=benchmark_result.get("ar_legacy_above_chance"),
        local_only=benchmark_result.get("local_only"),
        **v2_fields,
        **v3_fields,
        **ar_gate_fields,
        **ar_validation_fields,
        **ar_intermediate_fields,
        **binding_multislot_fields,
        **v9_trajectory_fields(benchmark_result),
    )
    if result_id and best_training_curve:
        try:
            nb.store_training_curve(result_id, best_training_curve)
        except Exception as exc:
            logger.warning(
                "Investigation training-curve persist failed for %s: %s",
                str(result_id)[:12],
                exc,
            )
    source_updates = {
        "wikitext_perplexity": benchmark_result.get("inv_wikitext_ppl"),
        "wikitext_score": benchmark_result.get("inv_wikitext_score"),
        "wikitext_ppl_200": benchmark_result.get("wikitext_ppl_200"),
        "wikitext_ppl_500": benchmark_result.get("wikitext_ppl_500"),
        "wikitext_improvement_ratio": benchmark_result.get(
            "wikitext_improvement_ratio"
        ),
        "wikitext_eval_steps": benchmark_result.get("wikitext_eval_steps"),
        "hellaswag_acc": benchmark_result.get("hellaswag_acc"),
        "hellaswag_status": benchmark_result.get("hellaswag_status"),
        "hellaswag_n_examples": benchmark_result.get("hellaswag_total"),
        "hellaswag_metric_version": benchmark_result.get("hellaswag_metric_version"),
        "hellaswag_tokenizer_mode": benchmark_result.get("hellaswag_tokenizer_mode"),
        "hellaswag_tiktoken_encoding": benchmark_result.get(
            "hellaswag_tiktoken_encoding"
        ),
        "ar_legacy_auc": benchmark_result.get("ar_legacy_auc"),
        "ar_legacy_final_acc": benchmark_result.get("ar_legacy_final_acc"),
        "ar_legacy_timed_out": benchmark_result.get("ar_legacy_timed_out"),
        "ar_legacy_above_chance": benchmark_result.get("ar_legacy_above_chance"),
        "induction_screening_auc": benchmark_result.get("induction_screening_auc"),
        "binding_screening_auc": benchmark_result.get("binding_screening_auc"),
        "binding_screening_composite": benchmark_result.get(
            "binding_screening_composite"
        ),
        "local_only": benchmark_result.get("local_only"),
        "result_cohort": result_cohort,
        "trust_label": trust_label,
        "comparability_label": comparability_label,
        "evaluation_protocol_version": evaluation_protocol_version,
        **v2_fields,
        **v3_fields,
        **ar_gate_fields,
        **ar_validation_fields,
        **ar_intermediate_fields,
        **binding_multislot_fields,
        # v9 trajectory metrics — overwrite earlier-phase init/screening
        # values with investigation_full measurements. Phase tag flips so
        # ML training distinguishes the two.
        **v9_trajectory_fields(benchmark_result),
    }
    set_parts = []
    set_params: List[Any] = []
    for col, value in source_updates.items():
        if value is None:
            continue
        set_parts.append(f"{col} = ?")
        set_params.append(value)
    if set_parts:
        set_params.append(source_result_id)
        nb.conn.execute(
            f"UPDATE program_results SET {', '.join(set_parts)} WHERE result_id = ?",
            set_params,
        )
        nb.upsert_induction_metric_v2(
            graph_fingerprint=str(
                benchmark_result.get("graph_fingerprint")
                or source.get("graph_fingerprint")
                or ""
            ),
            result_id=str(source_result_id),
            row=benchmark_result,
            source_cohort="runtime",
        )
        nb._maybe_commit()
    try:
        from ...eval.wikitext_eval import trajectory_wikitext_payload

        payload = trajectory_wikitext_payload(
            benchmark_result.get("wikitext_trajectory_payload") or {}
        )
        if payload:
            nb.set_external_benchmarks(result_id, payload)
            if source_result_id != result_id:
                nb.set_external_benchmarks(source_result_id, payload)
    except (ImportError, OSError, ValueError) as e:
        logger.debug("Trajectory wikitext payload persist failed: %s", e)


def _upsert_screening_entry(nb, row: Dict[str, Any]) -> Optional[str]:
    """Create or update a screening-tier leaderboard entry from a program_results row.

    Single source of truth for screening leaderboard creation.
    Returns entry_id on success, None on failure.
    """
    result_id = row.get("result_id")
    if not result_id:
        return None
    wiki_fields = screening_wikitext_fields(row)
    return nb.upsert_leaderboard(
        result_id=result_id,
        model_source=row.get("model_source") or "graph_synthesis",
        architecture_desc=row.get("graph_fingerprint", "")[:40],
        screening_loss_ratio=row.get("loss_ratio"),
        screening_novelty=row.get("novelty_score"),
        screening_passed=True,
        validation_loss_ratio=row.get("validation_loss_ratio"),
        validation_baseline_ratio=row.get("baseline_loss_ratio"),
        tier="screening",
        novelty_confidence=row.get("novelty_confidence"),
        fp_jacobian_spectral_norm=row.get("fp_jacobian_spectral_norm"),
        routing_savings_ratio=row.get("routing_savings_ratio"),
        activation_sparsity_score=row.get("activation_sparsity_score"),
        depth_savings_ratio=row.get("depth_savings_ratio"),
        compression_ratio=row.get("compression_ratio"),
        **wiki_fields,
    )


# ── SSE Log Bridge ──────────────────────────────────────────────────────
# Bridges Python logging → SSE event queue so dashboard live feed shows
# ── Baseline comparison helper ──
# Replaces the 20-line recipe/compare block that was duplicated 6× across
# execution_validation.py and continuous_validation.py.

logger = logging.getLogger(__name__)


def run_baseline_comparison(
    *,
    get_baseline,
    resolve_recipe,
    make_data_fn,
    candidate_loss: float,
    train_result: dict,
    config,
    dev_str: str,
    split: str = "train",
    normalized: bool = False,
    program_params: int | None = None,
) -> float | dict | None:
    """Run a baseline comparison (raw or parameter-normalized).

    Args:
        get_baseline: callable returning the TransformerBaseline instance.
        resolve_recipe: callable(train_result, default_lr) → recipe dict.
        make_data_fn: callable(config, split) → (data_fn, data_tag, cache).
        candidate_loss: the loss value to compare against baseline.
        train_result: best seed dict with optimizer/lr/steps info.
        config: RunConfig instance.
        dev_str: device string ("cuda", "cpu").
        split: data split ("train" or "val").
        normalized: if True, call compare_normalized instead of compare.
        program_params: required when normalized=True.

    Returns:
        float (loss ratio) for raw comparison, dict for normalized, or None on failure.
    """
    baseline = get_baseline()
    steps = int(train_result.get("n_train_steps") or config.validation_steps)
    recipe = resolve_recipe(train_result, default_lr=config.stage1_lr)
    data_fn, data_tag, cache = make_data_fn(config, split)

    kwargs = dict(
        d_model=config.model_dim,
        seq_len=min(128, config.validation_seq_len),
        n_steps=max(1, steps),
        vocab_size=config.vocab_size,
        batch_size=config.validation_batch_size,
        lr=recipe["lr"],
        device=dev_str,
        n_layers=config.n_layers,
        optimizer_name=recipe["optimizer_name"],
        weight_decay=recipe["weight_decay"],
        momentum=recipe["momentum"],
        betas=recipe["betas"],
        data_fn=data_fn,
        data_tag=data_tag,
        cache_data_fn=cache,
    )

    if normalized:
        return baseline.compare_normalized(
            candidate_loss, program_params=int(program_params), **kwargs
        )
    return baseline.compare(candidate_loss, **kwargs)


# ── Shared post-eval helpers ──
# Deduplicate ~155 lines shared between _run_validation_thread
# and _run_inline_validation.


def build_validation_entry(
    *,
    source_result_id: str,
    source: dict | None = None,
    metrics,  # ValidationMetrics
    ev_res,  # ExternalEvalResult
    nov_conf: float,
    config,  # RunConfig
):
    """Construct a ValidationEntry from metrics + eval result."""
    from ._types import ValidationEntry

    return ValidationEntry(
        result_id=source_result_id,
        source_experiment_id=(source or {}).get("experiment_id"),
        graph_fingerprint=(source or {}).get("graph_fingerprint"),
        novelty_score=(source or {}).get("novelty_score"),
        val_loss_ratio=metrics.val_loss_ratio,
        val_baseline_ratio=metrics.val_baseline_ratio,
        val_normalized_ratio=metrics.val_normalized_ratio,
        param_efficiency=metrics.val_param_efficiency,
        multi_seed_std=metrics.multi_seed_std,
        robustness_score=metrics.robustness_score,
        is_unstable=metrics.is_unstable,
        seeds_passed=len(metrics.passed_seeds),
        total_seeds=int(getattr(config, "validation_n_seeds", 5) or 5),
        is_breakthrough=ev_res.is_breakthrough,
        flop_gated=ev_res.flop_gated,
        quant_int8_retention=ev_res.quant_int8_retention,
        quant_quality_per_byte=ev_res.quant_quality_per_byte,
        long_context_score=ev_res.long_context_score,
        noise_sensitivity_score=ev_res.noise_score,
        init_sensitivity_std=metrics.init_sensitivity_std,
        novelty_confidence=nov_conf,
        ood_robustness=ev_res.ood_result,
        sensitivity=ev_res.sensitivity_result,
        activation_sparsity_score=ev_res.activation_sparsity_score,
        dead_neuron_ratio=ev_res.dead_neuron_ratio,
        routing_collapse_score=ev_res.routing_collapse_score,
        wikitext_perplexity=ev_res.wikitext_perplexity,
        wikitext_score=ev_res.wikitext_score,
        tinystories_perplexity=ev_res.tinystories_perplexity,
        tinystories_score=ev_res.tinystories_score,
        cross_task_score=ev_res.cross_task_score,
        efficiency_wall_score=ev_res.efficiency_wall_score,
        max_viable_seq_len=ev_res.max_viable_seq_len,
        scaling_regime=ev_res.scaling_regime,
    )


def finalize_validation_results_summary(results: dict) -> None:
    """Populate validation-specific counters before persistence and summaries."""
    entries = [
        entry
        for entry in (results.get("validation_results") or [])
        if isinstance(entry, dict)
    ]
    if not entries:
        return

    validation_passed = sum(
        1 for entry in entries if int(entry.get("seeds_passed") or 0) > 0
    )
    breakthrough_count = sum(
        1 for entry in entries if bool(entry.get("is_breakthrough"))
    )

    novel_count = 0
    for entry in entries:
        novelty = entry.get("novelty_score")
        if novelty is None:
            continue
        try:
            if float(novelty) > 0.5:
                novel_count += 1
        except (TypeError, ValueError):
            continue
    if (
        novel_count == 0
        and len(entries) == 1
        and results.get("best_novelty_score") is not None
    ):
        try:
            novel_count = int(float(results["best_novelty_score"]) > 0.5)
        except (TypeError, ValueError):
            novel_count = 0

    results["validated_count"] = len(entries)
    results["validation_passed_count"] = validation_passed
    results["breakthrough_count"] = breakthrough_count
    results["novel_count"] = novel_count


def promote_validation_candidate(
    *,
    nb,
    source_result_id: str,
    source: dict,
    tier: str,
    metrics,  # ValidationMetrics
    ev_res,  # ExternalEvalResult
    novelty_cap: float | None = None,
) -> None:
    """Promote candidate to tier on leaderboard + store benchmark payload.

    Handles novelty capping (B3) and external benchmark storage.
    """
    from ..shared_utils import coerce_dict_payload

    source_row = dict(nb.get_program_detail(source_result_id) or {})
    for key, value in dict(source or {}).items():
        if value is not None:
            source_row[key] = value

    # B3: cap novelty if CKA was missing
    if novelty_cap is not None:
        _raw_novelty = source_row.get("novelty_score")
        _raw_confidence = source_row.get("novelty_confidence")
        if _raw_novelty is not None:
            _raw_novelty = float(_raw_novelty) * novelty_cap
        if _raw_confidence is not None:
            _raw_confidence = float(_raw_confidence) * novelty_cap
        logger.info(
            "validation_novelty_capped: result_id=%s cap=%.2f novelty=%.4f confidence=%.4f",
            source_result_id[:12],
            novelty_cap,
            _raw_novelty or 0.0,
            _raw_confidence or 0.0,
        )
        if _raw_novelty is not None:
            source_row["novelty_score"] = _raw_novelty
        if _raw_confidence is not None:
            source_row["novelty_confidence"] = _raw_confidence
        try:
            fp_payload = source_row.get("fingerprint_json")
            if isinstance(fp_payload, str):
                try:
                    fp_payload = json.loads(fp_payload)
                except (TypeError, ValueError, json.JSONDecodeError):
                    fp_payload = None
            if isinstance(fp_payload, dict):
                fp_payload = dict(fp_payload)
                fp_payload["novelty_score"] = _raw_novelty
                nb.sync_behavioral_fingerprint_result(
                    result_id=source_result_id,
                    fp_payload=fp_payload,
                    novelty_confidence=_raw_confidence,
                    sync_leaderboard=False,
                )
            else:
                cap_updates = []
                if _raw_novelty is not None:
                    cap_updates.append(("novelty_score", _raw_novelty))
                if _raw_confidence is not None:
                    cap_updates.append(("novelty_confidence", _raw_confidence))
                if cap_updates:
                    _set = ", ".join(f"{c} = ?" for c, _ in cap_updates)
                    _vals = [v for _, v in cap_updates] + [source_result_id]
                    nb._submit_write(
                        f"UPDATE program_results SET {_set} WHERE result_id = ?",
                        _vals,
                    )
            nb.flush_writes()
        except (OSError, RuntimeError) as e:
            logger.debug(
                "B3 novelty cap DB update failed for %s: %s",
                source_result_id[:12],
                e,
            )

    nb.merge_program_result_patch(
        result_id=source_result_id,
        clear_failure_if_stage1=True,
        validation_loss_ratio=metrics.val_loss_ratio,
        validation_baseline_ratio=metrics.val_baseline_ratio,
        baseline_loss_ratio=metrics.val_baseline_ratio,
        validation_multi_seed_std=metrics.multi_seed_std,
        validation_robustness_score=metrics.robustness_score,
        validation_is_unstable=int(metrics.is_unstable),
        validation_passed=len(metrics.passed_seeds) > 0,
        normalized_baseline_ratio=metrics.val_normalized_ratio,
        param_efficiency=metrics.val_param_efficiency,
        quant_int8_retention=ev_res.quant_int8_retention,
        quant_quality_per_byte=ev_res.quant_quality_per_byte,
        robustness_long_ctx_score=ev_res.long_context_score,
        robustness_long_ctx_scaling_score=ev_res.long_ctx_scaling_score,
        robustness_long_ctx_assoc_score=ev_res.long_ctx_assoc_score,
        robustness_long_ctx_passkey_score=ev_res.long_ctx_passkey_score,
        robustness_long_ctx_multi_hop_score=ev_res.long_ctx_multi_hop_score,
        robustness_long_ctx_retrieval_aggregate=ev_res.long_ctx_retrieval_aggregate,
        robustness_long_ctx_combined_score=ev_res.long_ctx_combined_score,
        induction_intermediate_auc=ev_res.induction_intermediate_auc,
        induction_intermediate_max_gap_acc=ev_res.induction_intermediate_max_gap_acc,
        induction_intermediate_protocol_version=ev_res.induction_intermediate_protocol_version,
        binding_intermediate_auc=ev_res.binding_intermediate_auc,
        binding_intermediate_max_distance_acc=ev_res.binding_intermediate_max_distance_acc,
        binding_intermediate_protocol_version=ev_res.binding_intermediate_protocol_version,
        permutation_composition_score=ev_res.permutation_composition_score,
        permutation_composition_train_chain_acc=ev_res.permutation_composition_train_chain_acc,
        permutation_composition_extrapolation_acc=ev_res.permutation_composition_extrapolation_acc,
        permutation_composition_n_items=ev_res.permutation_composition_n_items,
        permutation_composition_train_chain_len=ev_res.permutation_composition_train_chain_len,
        permutation_composition_eval_chain_len=ev_res.permutation_composition_eval_chain_len,
        permutation_composition_train_steps=ev_res.permutation_composition_train_steps,
        permutation_composition_chance=ev_res.permutation_composition_chance,
        permutation_composition_elapsed_ms=ev_res.permutation_composition_elapsed_ms,
        permutation_composition_status=ev_res.permutation_composition_status,
        permutation_composition_metric_version=ev_res.permutation_composition_metric_version,
        robustness_noise_score=ev_res.noise_score,
        init_sensitivity_std=metrics.init_sensitivity_std,
        fp_jacobian_spectral_norm=source_row.get("fp_jacobian_spectral_norm"),
        scaling_param_efficiency=ev_res.scaling_param_efficiency,
        scaling_d512_param_efficiency=ev_res.scaling_d512_param_efficiency,
        scaling_flop_efficiency=ev_res.scaling_flop_efficiency,
        scaling_gate_passed=ev_res.scaling_gate_passed_val,
        scaling_best_family=ev_res.scaling_best_family,
        scaling_confidence=ev_res.scaling_confidence,
        activation_sparsity_score=ev_res.activation_sparsity_score,
        dead_neuron_ratio=ev_res.dead_neuron_ratio,
        routing_collapse_score=ev_res.routing_collapse_score,
        wikitext_perplexity=ev_res.wikitext_perplexity,
        wikitext_score=ev_res.wikitext_score,
        tinystories_perplexity=ev_res.tinystories_perplexity,
        tinystories_score=ev_res.tinystories_score,
        cross_task_score=ev_res.cross_task_score,
        efficiency_wall_score=ev_res.efficiency_wall_score,
        max_viable_seq_len=ev_res.max_viable_seq_len,
        scaling_regime=ev_res.scaling_regime,
    )

    entry = nb.get_leaderboard_entry(source_result_id)
    if not entry:
        graph_fingerprint = str(source_row.get("graph_fingerprint") or "").strip()
        if graph_fingerprint:
            entry = nb.get_leaderboard_entry_by_fingerprint(graph_fingerprint)
    if not entry:
        entry_id = _upsert_screening_entry(nb, source_row)
        if entry_id:
            entry = nb.get_leaderboard_entry(source_result_id)
            if entry is None:
                entry = {
                    "entry_id": entry_id,
                    "result_id": source_result_id,
                }
    if not entry:
        return

    promote_kwargs = dict(
        entry_id=entry["entry_id"],
        tier=tier,
        validation_loss_ratio=metrics.val_loss_ratio,
        validation_baseline_ratio=metrics.val_baseline_ratio,
        validation_multi_seed_std=metrics.multi_seed_std,
        validation_robustness_score=metrics.robustness_score,
        validation_is_unstable=int(metrics.is_unstable),
        validation_passed=len(metrics.passed_seeds) > 0,
        normalized_baseline_ratio=metrics.val_normalized_ratio,
        param_efficiency=metrics.val_param_efficiency,
        quant_int8_retention=ev_res.quant_int8_retention,
        quant_quality_per_byte=ev_res.quant_quality_per_byte,
        robustness_long_ctx_score=ev_res.long_context_score,
        robustness_long_ctx_scaling_score=ev_res.long_ctx_scaling_score,
        robustness_long_ctx_assoc_score=ev_res.long_ctx_assoc_score,
        robustness_long_ctx_passkey_score=ev_res.long_ctx_passkey_score,
        robustness_long_ctx_multi_hop_score=ev_res.long_ctx_multi_hop_score,
        robustness_long_ctx_retrieval_aggregate=ev_res.long_ctx_retrieval_aggregate,
        robustness_long_ctx_combined_score=ev_res.long_ctx_combined_score,
        induction_intermediate_auc=ev_res.induction_intermediate_auc,
        induction_intermediate_max_gap_acc=ev_res.induction_intermediate_max_gap_acc,
        induction_intermediate_protocol_version=ev_res.induction_intermediate_protocol_version,
        binding_intermediate_auc=ev_res.binding_intermediate_auc,
        binding_intermediate_max_distance_acc=ev_res.binding_intermediate_max_distance_acc,
        binding_intermediate_protocol_version=ev_res.binding_intermediate_protocol_version,
        robustness_noise_score=ev_res.noise_score,
        init_sensitivity_std=metrics.init_sensitivity_std,
        fp_jacobian_spectral_norm=source_row.get("fp_jacobian_spectral_norm"),
        scaling_param_efficiency=ev_res.scaling_param_efficiency,
        scaling_d512_param_efficiency=ev_res.scaling_d512_param_efficiency,
        scaling_flop_efficiency=ev_res.scaling_flop_efficiency,
        scaling_gate_passed=ev_res.scaling_gate_passed_val,
        scaling_best_family=ev_res.scaling_best_family,
        scaling_confidence=ev_res.scaling_confidence,
        activation_sparsity_score=ev_res.activation_sparsity_score,
        dead_neuron_ratio=ev_res.dead_neuron_ratio,
        routing_collapse_score=ev_res.routing_collapse_score,
        wikitext_perplexity=ev_res.wikitext_perplexity,
        wikitext_score=ev_res.wikitext_score,
        tinystories_perplexity=ev_res.tinystories_perplexity,
        tinystories_score=ev_res.tinystories_score,
        cross_task_score=ev_res.cross_task_score,
        efficiency_wall_score=ev_res.efficiency_wall_score,
        max_viable_seq_len=ev_res.max_viable_seq_len,
        scaling_regime=ev_res.scaling_regime,
    )
    if novelty_cap is not None:
        _raw = source_row.get("novelty_score")
        if _raw is not None:
            promote_kwargs["screening_novelty"] = float(_raw) * novelty_cap

    nb.promote_to_tier(**promote_kwargs)

    # Store external benchmark payload
    external = {}
    sp = coerce_dict_payload(ev_res.scaling_result)
    if sp is not None:
        external.update(sp)
        external["scaling_comparison"] = sp
    if ev_res.long_context_details is not None:
        external["long_context"] = ev_res.long_context_details
    if external:
        nb.set_external_benchmarks(source_result_id, external)
        canonical_result_id = str(entry.get("result_id") or "").strip()
        if canonical_result_id and canonical_result_id != source_result_id:
            nb.set_external_benchmarks(canonical_result_id, external)


def handle_breakthrough(
    *,
    is_breakthrough: bool,
    trajectory_composite: float | None,
    aria,
    nb,
    exp_id: str,
    source_result_id: str,
    source: dict,
    validation_entry,  # ValidationEntry
    val_loss_ratio: float | None,
    val_baseline_ratio: float | None,
    multi_seed_std: float,
    emit_event,
) -> bool:
    """Check trajectory-aware breakthrough and emit announcement.

    Returns final is_breakthrough value.
    """
    from ..breakthrough_gates import passes_breakthrough_from_row
    from ..llm.context_experiment import build_validation_context
    from ..notebook import ExperimentEntry

    # Trajectory-aware fallback promotion: only fires when the row's full
    # gate set passes (composite floor + baseline improvement + capability
    # signal). The prior ``trajectory_composite > 300.0`` hardcode promoted
    # the d904 false positive (composite 499 with all capability metrics
    # near-random); the helper now blocks that family.
    if not is_breakthrough and trajectory_composite is not None:
        entry_row = nb.get_leaderboard_entry(source_result_id) or {}
        passed, reason = passes_breakthrough_from_row(
            dict(entry_row), composite_score=trajectory_composite
        )
        if passed:
            is_breakthrough = True
            logger.info(
                "Trajectory-aware breakthrough: %s composite=%.1f",
                source_result_id[:8],
                trajectory_composite,
            )
        else:
            logger.info(
                "Trajectory-aware breakthrough blocked: %s composite=%.1f reason=%s",
                source_result_id[:8],
                trajectory_composite,
                reason,
            )

    if is_breakthrough:
        entry_dict = (
            validation_entry.to_dict()
            if hasattr(validation_entry, "to_dict")
            else validation_entry
        )
        ctx = build_validation_context([source], [entry_dict])
        announcement = aria.announce_breakthrough(ctx)
        nb.add_entry(
            ExperimentEntry(
                entry_type="insight",
                title="BREAKTHROUGH DETECTED",
                content=announcement,
                experiment_id=exp_id,
                tags=["breakthrough"],
            )
        )
        emit_event(
            "breakthrough_detected",
            {
                "experiment_id": exp_id,
                "result_id": source_result_id,
                "val_loss_ratio": val_loss_ratio,
                "val_baseline_ratio": val_baseline_ratio,
                "multi_seed_std": multi_seed_std,
                "announcement": announcement,
            },
        )

    return is_breakthrough


# ── SSE log handler ──
# log messages without modifying every call site.

_SSE_LOG_DEDUP_WINDOW: float = 5.0  # seconds to suppress identical messages
_SSE_LOG_RATE_LIMIT: int = 10  # max events per second per logger name
_SSE_LOG_RATE_WINDOW: float = 1.0  # sliding window for rate limit


class SSELogHandler(logging.Handler):
    """Logging handler that forwards records to the runner's SSE event queue.

    Guardrails:
    - Only captures ``research.*`` loggers at INFO+
    - Deduplicates identical messages within a time window
    - Rate-limits per logger name to prevent queue saturation
    - Never persists to DB (avoids bloating the notebook)
    """

    __slots__ = (
        "_queue",
        "_dedup",
        "_rate_counts",
        "_rate_window_start",
    )

    def __init__(self, event_queue: queue.Queue):
        super().__init__(level=logging.INFO)
        self._queue = event_queue
        # {message_text: last_emit_ts}
        self._dedup: Dict[str, float] = {}
        # {logger_name: count_in_current_window}
        self._rate_counts: Dict[str, int] = {}
        self._rate_window_start: float = time.monotonic()

    def filter(self, record: logging.LogRecord) -> bool:
        # Only research.* loggers, skip werkzeug/urllib3/etc.
        return record.name.startswith("research.")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record) if self.formatter else record.getMessage()
            now = time.monotonic()

            # ── Dedup: skip identical messages within window ──
            last_seen = self._dedup.get(msg)
            if last_seen is not None and (now - last_seen) < _SSE_LOG_DEDUP_WINDOW:
                return
            self._dedup[msg] = now

            # Prune stale dedup entries periodically (every ~50 messages)
            if len(self._dedup) > 200:
                cutoff = now - _SSE_LOG_DEDUP_WINDOW
                self._dedup = {k: v for k, v in self._dedup.items() if v > cutoff}

            # ── Rate limit per logger name ──
            if (now - self._rate_window_start) >= _SSE_LOG_RATE_WINDOW:
                self._rate_counts.clear()
                self._rate_window_start = now
            count = self._rate_counts.get(record.name, 0)
            if count >= _SSE_LOG_RATE_LIMIT:
                return
            self._rate_counts[record.name] = count + 1

            # ── Push to SSE queue ──
            # Truncate short logger prefix for dashboard display
            short_name = record.name
            if short_name.startswith("research."):
                short_name = short_name[len("research.") :]

            payload = {
                "type": "log_message",
                "data": {
                    "level": record.levelname,
                    "logger": short_name,
                    "message": msg[:500],
                    "timestamp": time.time(),
                },
                "timestamp": time.time(),
            }
            self._queue.put_nowait(payload)
        except queue.Full:
            pass  # drop log events silently when queue is saturated
        except Exception:
            pass  # top-level error boundary: never break the logging pipeline
