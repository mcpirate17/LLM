"""Execution training mixin — split from execution_training."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import torch
import torch.nn as nn

from research.defaults import RUNS_DB

from ..json_utils import json_safe
from ._helpers import (
    _corpus_type_from_config,
    get_reference_losses,
    normalized_loss_ratio,
    resolve_stage1_gate_metrics,
    stage1_learning_gate,
)
from ._types import RunConfig
from .execution_training_native_boundary import (
    _TrainingLoopState,
)
from ...eval.fingerprint import compute_gated_fingerprint
from ...training.profiling import TrainingRunProfiler

import logging

logger = logging.getLogger(__name__)


def _resolve_stage1_learning_gate():
    """Honor monkeypatches against the legacy execution_training module surface."""
    gate_fn = getattr(
        __import__(
            __package__ + ".execution_training", fromlist=["stage1_learning_gate"]
        ),
        "stage1_learning_gate",
        None,
    )
    if callable(gate_fn):
        return gate_fn
    return stage1_learning_gate


class _ExecutionTrainingPostMixin:
    """Post-training metric collection + post-S1 probes."""

    __slots__ = ()

    def _collect_post_training_metrics(
        self,
        model: nn.Module,
        result: Dict[str, Any],
        config: RunConfig,
        dev: torch.device,
        loop_state: _TrainingLoopState,
        tracer,
        trace_totals_ms: Dict[str, float],
        starvation_detector,
        kernel_profiles: List[Dict[str, Any]],
        run_profiler: TrainingRunProfiler,
        graph_json: str,
        graph_data,
        use_synthesized_training: bool,
    ) -> None:
        """Collect all post-training metrics and write them into *result*.

        Covers validation/discovery loss, perf traces, learning gate,
        fingerprint, architecture telemetry, entropy gate trajectory,
        and routing metrics.  Called after the training loop finishes.
        """
        ls = loop_state

        # Optional validation loss on heldout corpus split
        validation_loss = None
        validation_loss_ratio = None
        generalization_gap = None
        if not bool(getattr(config, "profile_disable_post_eval", False)):
            try:
                with run_profiler.trace("validation_eval_ms"):
                    validation_loss = self._micro_train_optional_validation_loss(
                        model=model,
                        config=config,
                        dev=dev,
                        seq_len=ls.seq_len,
                        seed=ls.seed,
                    )
            except RuntimeError as e:
                logger.debug("Validation loss eval failed: %s", e)
                result["validation_loss_error"] = str(e)

        # Optional discovery loss on random tokens (fast triage signal)
        discovery_loss = None
        discovery_loss_ratio = None
        if not bool(getattr(config, "profile_disable_post_eval", False)):
            try:
                with run_profiler.trace("discovery_eval_ms"):
                    discovery_loss = self._micro_train_optional_discovery_loss(
                        model=model,
                        config=config,
                        dev=dev,
                        seq_len=ls.seq_len,
                        seed=ls.seed,
                    )
            except RuntimeError as e:
                logger.debug("Discovery loss eval failed: %s", e)
                result["discovery_loss_error"] = str(e)

        if validation_loss is not None and ls.initial_loss:
            validation_loss_ratio = validation_loss / max(ls.initial_loss, 1e-6)
        if validation_loss is not None and ls.final_loss is not None:
            generalization_gap = validation_loss - ls.final_loss
        if discovery_loss is not None and ls.initial_loss:
            discovery_loss_ratio = discovery_loss / max(ls.initial_loss, 1e-6)

        # Collect perf results
        if tracer is not None:
            result["perf_traces"] = tracer.get_report()
        else:
            result["perf_traces"] = {
                "summary_ms": {k: round(v, 4) for k, v in trace_totals_ms.items()},
                "traces": [],
            }
        result["gpu_starvation"] = starvation_detector.get_summary()
        if kernel_profiles:
            result["kernel_timing"] = {
                "sample_count": len(kernel_profiles),
                "samples": kernel_profiles,
                "top_ops": kernel_profiles[0].get("top_ops", []),
            }

        if ls.initial_loss is not None and ls.final_loss is not None:
            # Store both loss ratio formulas with unambiguous names:
            #   loss_ratio_raw  = final_loss / initial_loss  (relative improvement)
            #   loss_ratio_norm = final_loss / ln(vocab_size) (absolute position)
            # The auto-escalation threshold (0.18) is calibrated against RAW.
            # loss_ratio keeps RAW for backward compatibility.
            _raw = ls.final_loss / max(ls.initial_loss, 1e-6)
            _norm = normalized_loss_ratio(ls.final_loss, config.vocab_size)
            result["loss_ratio"] = _raw
            result["loss_ratio_raw"] = _raw
            result["loss_ratio_norm"] = _norm
            result["final_loss"] = ls.final_loss
            result["initial_loss"] = ls.initial_loss
            result["min_loss"] = ls.min_loss
            if validation_loss is not None:
                result["validation_loss"] = validation_loss
            if validation_loss_ratio is not None:
                result["validation_loss_ratio"] = validation_loss_ratio
            if generalization_gap is not None:
                result["generalization_gap"] = generalization_gap
            if discovery_loss is not None:
                result["discovery_loss"] = discovery_loss
            if discovery_loss_ratio is not None:
                result["discovery_loss_ratio"] = discovery_loss_ratio
            training_summary = ls.native_summary()
            result["throughput"] = training_summary["throughput"]

            # Corpus-aware learning gate (replaces fixed threshold)
            corpus_type = _corpus_type_from_config(config)
            tokenizer = str(config.tokenizer_mode or "tiktoken")
            try:
                ref_losses = get_reference_losses(
                    str(getattr(self, "notebook_path", RUNS_DB))
                )
            except (OSError, ValueError, KeyError) as e:
                logger.debug("Reference loss lookup failed: %s", e)
                ref_losses = {}
            gate_loss, raw_ratio, gate_loss_source = resolve_stage1_gate_metrics(
                initial_loss=ls.initial_loss,
                final_loss=ls.final_loss,
                validation_loss=validation_loss,
            )
            gate_passed, gate_reason = _resolve_stage1_learning_gate()(
                final_loss=gate_loss,
                loss_ratio=raw_ratio,
                initial_loss=ls.initial_loss,
                n_steps=ls.step_count,
                corpus_type=corpus_type,
                tokenizer=tokenizer,
                reference_losses=ref_losses,
            )
            result["passed"] = gate_passed
            result["gate_reason"] = gate_reason
            result["gate_loss_source"] = gate_loss_source

            # Validation loss gate: if val loss didn't improve, fail.
            if (
                result["passed"]
                and validation_loss_ratio is not None
                and validation_loss_ratio > 0.6
            ):
                result["passed"] = False
                result["error_type"] = "insufficient_learning"
                result["error"] = (
                    f"Validation loss ratio {validation_loss_ratio:.4f} > 0.60 — "
                    f"model memorized training but failed to generalize"
                )
            # Inflight checks already flagged this run — override pass
            if result.get("error_type", "").startswith("inflight_"):
                result["passed"] = False
            if not result["passed"] and result.get("error_type") is None:
                result["error_type"] = "failed_convergence"
                result["error"] = gate_reason
            if ls.initial_loss > 0:
                result["loss_improvement_rate"] = (
                    ls.initial_loss - ls.final_loss
                ) / ls.initial_loss

            # Timing stats
            result["avg_step_time_ms"] = training_summary["avg_step_time_ms"]
            result["total_train_time_ms"] = ls.total_time_ms

            # Gradient norm stats
            if training_summary["max_grad_norm"] is not None:
                result["max_grad_norm"] = training_summary["max_grad_norm"]
                result["mean_grad_norm"] = training_summary["mean_grad_norm"]
                result["grad_norm_std"] = training_summary["grad_norm_std"]

            result["n_train_steps"] = training_summary["n_train_steps"]
            result["final_lr"] = config.stage1_lr  # constant for now
            if ls.collect_curve:
                result["training_curve"] = ls.training_curve
            artifacts = run_profiler.artifacts()
            if artifacts is not None:
                result["profile_artifacts"] = {
                    "output_dir": artifacts.output_dir,
                    "summary_json": artifacts.summary_json,
                    "trace_json": artifacts.trace_json,
                }
                run_profiler.event("avg_step_time_ms", result["avg_step_time_ms"])
                run_profiler.event("throughput_tok_s", result["throughput"])
                run_profiler.event("n_train_steps", result["n_train_steps"])

            # Extract architecture-specific telemetry (MoE, MoD, MoR, etc.)
            arch_telemetry = self._extract_architecture_telemetry(model)
            result.update(arch_telemetry)

            # Entropy gate trajectory (sampled during training)
            if ls.entropy_gate_trajectory:
                result["entropy_gate_trajectory_json"] = json.dumps(
                    json_safe(ls.entropy_gate_trajectory)
                )
                if any(v < 0.05 for v in ls.entropy_gate_trajectory):
                    result["routing_collapse_score"] = 1.0

            # Routing training metrics: load-balance aux loss + derived stats
            if ls.routing_aux_loss_count > 0:
                result["routing_aux_loss_mean"] = (
                    ls.routing_aux_loss_sum / ls.routing_aux_loss_count
                )
            rt_total = result.get("routing_tokens_total", 0)
            rt_processed = result.get("routing_tokens_processed", 0)
            if rt_total > 0:
                result["routing_fast_fraction"] = max(
                    0.0,
                    1.0 - (rt_processed / rt_total),
                )
                eu_json = result.get("routing_expert_utilization_json")
                if eu_json:
                    try:
                        counts = json.loads(eu_json)
                        if counts:
                            total_c = sum(counts)
                            if total_c > 0:
                                fracs = [c / total_c for c in counts]
                                uniform = 1.0 / len(fracs)
                                # Balance = 1 - normalized MSE (1=uniform, 0=collapsed)
                                mse = sum((f - uniform) ** 2 for f in fracs) / len(
                                    fracs
                                )
                                result["routing_balance_score"] = max(
                                    0.0,
                                    1.0 - mse * len(fracs),
                                )
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Behavioral fingerprint for S1 survivors (structural-only at
            # screening; CKA + behavioral probes deferred to post-investigation)
            if (
                result.get("passed")
                and model is not None
                and not bool(getattr(config, "skip_post_s1_triage", False))
                and not bool(getattr(config, "profile_disable_post_eval", False))
            ):
                try:
                    _lr = result.get("loss_ratio", 1.0)
                    _perf_gate = float(
                        getattr(config, "fingerprint_perf_gate", 0.85) or 0.85
                    )
                    _force_lightning = _lr > _perf_gate

                    if _force_lightning:
                        logger.debug(
                            "    Investigation gating: skipping full fingerprint for poor performer (LR=%.4f > %.2f)",
                            _lr,
                            _perf_gate,
                        )

                    # Parse graph for structural novelty computation
                    _graph_obj = None
                    if graph_json:
                        try:
                            from ...synthesis.serializer import graph_from_json

                            _graph_obj = graph_from_json(graph_json)
                        except (ValueError, KeyError, json.JSONDecodeError) as e:
                            logger.debug(
                                "Graph deserialization failed for fingerprint: %s",
                                e,
                            )

                    _fp, full_ran = compute_gated_fingerprint(
                        model,
                        seq_len=min(64, config.max_seq_len),
                        model_dim=config.model_dim,
                        vocab_size=config.vocab_size,
                        device=str(dev),
                        full_gate_enabled=True,
                        force_lightning_only=_force_lightning,
                        graph=_graph_obj,
                        structural_floor=float(
                            getattr(config, "lightning_structural_floor", 0.10) or 0.10
                        ),
                    )
                    result["_behavioral_fingerprint"] = _fp.to_dict()
                    result["fingerprint_full_ran"] = full_ran
                except (RuntimeError, ValueError, TypeError) as e_fp:
                    logger.debug("Fingerprint failed in S1 worker: %s", e_fp)

    def _run_post_s1_screening_probes(
        self,
        model: nn.Module,
        result: Dict[str, Any],
        config: RunConfig,
        dev: torch.device,
        graph_json: str,
        graph_data,
    ) -> None:
        """Run post-S1 screening probes on passing candidates.

        WikiText eval, HellaSwag eval, binding probes, and post-S1 triage.
        Only runs on candidates that passed the learning gate.
        Mutates *result* in-place.
        """
        # Failed candidates do not benefit from post-S1 screening probes.
        # Keeping these on the failure path wastes seconds per reject.
        should_run_post_s1_screening_probes = bool(result.get("passed")) and (
            not bool(getattr(config, "profile_disable_post_eval", False))
        )

        # Fast WikiText perplexity at screening time
        if should_run_post_s1_screening_probes and not getattr(
            config, "skip_screening_wikitext", False
        ):
            try:
                from ...eval.wikitext_eval import screening_wikitext_eval

                wt = screening_wikitext_eval(
                    model,
                    config.vocab_size,
                    str(dev),
                    seq_len=min(128, config.max_seq_len),
                )
                result["screening_wikitext_status"] = wt.get(
                    "screening_wikitext_status"
                )
                result["screening_wikitext_metric_version"] = wt.get(
                    "screening_wikitext_metric_version"
                )
                if wt.get("wikitext_perplexity") is not None:
                    result["wikitext_perplexity"] = wt["wikitext_perplexity"]
                    result["wikitext_score"] = wt.get("wikitext_score")
                    result["wikitext_pre_perplexity"] = wt.get(
                        "wikitext_pre_perplexity"
                    )
                    result["wikitext_ppl_improvement"] = wt.get(
                        "wikitext_ppl_improvement"
                    )
                # Slope trajectory fields (for slope reprieve)
                for _slope_key in (
                    "screening_loss_10",
                    "screening_loss_25",
                    "screening_loss_50",
                    "screening_slope",
                    "screening_slope_consistent",
                ):
                    if wt.get(_slope_key) is not None:
                        result[_slope_key] = wt[_slope_key]
                logger.info(
                    "    Screening WikiText ppl=%.1f score=%.3f (%.0fms)",
                    wt["wikitext_perplexity"],
                    wt.get("wikitext_score") or 0,
                    wt.get("elapsed_ms") or 0,
                )
            except (RuntimeError, ValueError, OSError, ImportError) as e_wt:
                logger.debug("Screening WikiText eval skipped: %s", e_wt)

        # Fast HellaSwag commonsense reasoning probe at screening time
        if should_run_post_s1_screening_probes and not getattr(
            config, "skip_screening_hellaswag", False
        ):
            try:
                from ...eval.hellaswag_eval import screening_hellaswag_eval

                hs = screening_hellaswag_eval(
                    model,
                    config.vocab_size,
                    str(dev),
                )
                result["hellaswag_acc"] = hs.get("hellaswag_acc")
                result["hellaswag_status"] = hs.get("hellaswag_status")
                result["hellaswag_n_examples"] = hs.get("hellaswag_total")
                result["hellaswag_metric_version"] = hs.get("hellaswag_metric_version")
                result["hellaswag_tokenizer_mode"] = hs.get("hellaswag_tokenizer_mode")
                result["hellaswag_tiktoken_encoding"] = hs.get(
                    "hellaswag_tiktoken_encoding"
                )
                result["screening_hellaswag_correct"] = hs.get("hellaswag_correct")
                result["screening_hellaswag_total"] = hs.get("hellaswag_total")
                result["screening_hellaswag_elapsed_ms"] = hs.get("elapsed_ms")
                if hs.get("hellaswag_acc") is not None:
                    logger.info(
                        "    Screening HellaSwag acc=%.1f%% (%d/%d, %.0fms)",
                        hs["hellaswag_acc"] * 100,
                        hs.get("hellaswag_correct", 0),
                        hs.get("hellaswag_total", 0),
                        hs.get("elapsed_ms", 0),
                    )
            except (RuntimeError, ValueError, OSError, ImportError) as e_hs:
                logger.debug("Screening HellaSwag eval skipped: %s", e_hs)

        # BLiMP linguistic minimal pairs (forward-only, ~2s)
        if should_run_post_s1_screening_probes and not getattr(
            config, "skip_screening_blimp", False
        ):
            try:
                from ...eval.blimp_eval import evaluate_blimp

                blimp = evaluate_blimp(
                    model, int(config.vocab_size), str(dev), n_per_subtask=50
                )
                result["blimp_overall_accuracy"] = blimp.overall_accuracy
                result["blimp_subtask_accuracies_json"] = blimp.subtask_accuracies
                result["blimp_n_subtasks"] = blimp.n_subtasks
                result["blimp_status"] = blimp.status
                if blimp.overall_accuracy is not None:
                    logger.info(
                        "    Screening BLiMP acc=%.1f%% (%d subtasks, %s)",
                        blimp.overall_accuracy * 100,
                        blimp.n_subtasks,
                        blimp.status,
                    )
            except (RuntimeError, ValueError, OSError, ImportError) as e_bl:
                logger.debug("Screening BLiMP eval skipped: %s", e_bl)

        # Screening probes: induction and binding are independently skippable.
        want_induction_probe = should_run_post_s1_screening_probes and not (
            getattr(config, "skip_binding_probes", False)
            or getattr(config, "skip_induction_probe", False)
        )
        want_binding_probe = should_run_post_s1_screening_probes and not (
            getattr(config, "skip_binding_probes", False)
            or getattr(config, "skip_binding_probe", False)
        )
        if want_induction_probe or want_binding_probe:
            try:
                from ...eval.binding_curriculum import (
                    CURRICULUM_BINDING_PROTOCOL_VERSION,
                    CURRICULUM_BINDING_DISTANCES,
                    CURRICULUM_BINDING_EVAL_BATCH_SIZE,
                    CURRICULUM_BINDING_EVAL_SCREENING,
                    CURRICULUM_BINDING_STEPS_SCREENING,
                    CURRICULUM_BINDING_TRAIN_BATCH_SIZE,
                    curriculum_binding_range_profile,
                )
                from ...eval.binding_range import binding_range_profile
                from ...eval.native_induction import (
                    induction_result_metadata,
                    induction_score_gold,
                )

                ind = None
                if want_induction_probe:
                    ind = induction_score_gold(
                        model,
                        device=str(dev),
                        seed=getattr(config, "screening_probe_seed", None),
                    )
                    result.update(induction_result_metadata(ind))

                br = None
                if want_binding_probe:
                    zero = binding_range_profile(
                        model,
                        distances=CURRICULUM_BINDING_DISTANCES,
                        n_eval=CURRICULUM_BINDING_EVAL_SCREENING,
                        device=str(dev),
                        seed=getattr(config, "screening_probe_seed", None),
                    )
                    br = curriculum_binding_range_profile(
                        model,
                        distances=CURRICULUM_BINDING_DISTANCES,
                        n_train_steps=CURRICULUM_BINDING_STEPS_SCREENING,
                        n_eval=CURRICULUM_BINDING_EVAL_SCREENING,
                        train_batch_size=max(
                            1,
                            int(
                                getattr(
                                    config,
                                    "binding_probe_train_batch_size",
                                    CURRICULUM_BINDING_TRAIN_BATCH_SIZE,
                                )
                                or CURRICULUM_BINDING_TRAIN_BATCH_SIZE
                            ),
                        ),
                        eval_batch_size=max(
                            1,
                            int(
                                getattr(
                                    config,
                                    "binding_probe_eval_batch_size",
                                    CURRICULUM_BINDING_EVAL_BATCH_SIZE,
                                )
                                or CURRICULUM_BINDING_EVAL_BATCH_SIZE
                            ),
                        ),
                        device=str(dev),
                        seed=getattr(config, "screening_probe_seed", None),
                        offload_source_model=bool(
                            getattr(config, "binding_probe_offload_source_model", False)
                        ),
                    )
                    result["binding_screening_auc"] = zero.auc
                    result["binding_distance_accuracies"] = zero.distance_accuracies
                    result["binding_screening_eval_examples"] = (
                        CURRICULUM_BINDING_EVAL_SCREENING
                    )
                    result["binding_probe_distances"] = list(
                        CURRICULUM_BINDING_DISTANCES
                    )
                    result["binding_screening_elapsed_ms"] = zero.elapsed_ms
                    result["binding_curriculum_auc"] = br.auc
                    result["binding_distance_accuracies_curriculum"] = (
                        br.distance_accuracies
                    )
                    result["binding_curriculum_steps"] = br.train_steps
                    result["binding_curriculum_elapsed_ms"] = br.elapsed_ms
                    result["binding_curriculum_protocol_version"] = (
                        CURRICULUM_BINDING_PROTOCOL_VERSION
                    )

                # AR probe (~60s, deepcopy + 500 train steps)
                ar = None
                if not getattr(config, "skip_ar_probe", False):
                    try:
                        from ...eval.associative_recall import associative_recall_score

                        ar = associative_recall_score(
                            model,
                            n_pairs=20,
                            n_eval=200,
                            n_train_steps=500,
                            batch_size=16,
                            device=str(dev),
                        )
                        result["ar_legacy_auc"] = ar.auc
                        result["ar_legacy_final_acc"] = ar.final_acc
                        result["ar_legacy_timed_out"] = int(ar.timed_out)
                        result["ar_legacy_above_chance"] = int(ar.above_chance)
                    except (RuntimeError, ValueError, TypeError, ImportError) as e_ar:
                        logger.debug("AR probe skipped: %s", e_ar)

                if result.get("ar_legacy_auc") is None:
                    result["ar_legacy_auc"] = None

                # AR gate-INV (investigation-tier associative-recall probe).
                # Runs on a deepcopy of the live S1 model — uses the wikitext
                # priors the backbone already learned. ~20s on cuda.
                # Replaces the dead ar_legacy_auc slot in binding_screening_composite.
                nai = None
                if not getattr(config, "skip_ar_gate", False):
                    try:
                        from ...eval.ar_gate import (
                            ARGateConfig,
                            ar_gate,
                        )

                        nai = ar_gate(
                            model=model,
                            device=str(dev),
                            cfg=ARGateConfig(from_s1=True),
                        )
                        result["ar_gate_metric_version"] = nai.metric_version
                        result["ar_gate_in_dist_pair_acc"] = nai.in_dist_pair_acc
                        result["ar_gate_in_dist_class_acc"] = nai.in_dist_class_acc
                        result["ar_gate_held_pair_acc"] = nai.held_pair_acc
                        result["ar_gate_held_class_acc"] = nai.held_class_acc
                        result["ar_gate_score"] = round(
                            0.6 * nai.in_dist_pair_acc + 0.4 * nai.held_class_acc,
                            4,
                        )
                        result["ar_gate_status"] = nai.status
                        result["ar_gate_elapsed_ms"] = nai.elapsed_ms
                        result["ar_gate_train_steps_done"] = nai.finetune_steps_done
                        # Hard no-go gate (mirrors nano_bind's persistent-zero rule):
                        # both pair-match and held-class < 0.10 ⇒ frequency-collapse
                        # degenerate. The gate only flags status='ok' runs to avoid
                        # punishing transient failures (timeout / non-finite loss).
                        is_no_go = (
                            nai.status == "ok"
                            and nai.in_dist_pair_acc < 0.10
                            and nai.held_class_acc < 0.10
                        )
                        result["ar_gate_no_go"] = int(bool(is_no_go))
                        if is_no_go:
                            # Diagnostic flag only — do NOT set failure_op or
                            # demote tier. Demotion would hide the row from the
                            # dashboard listing (per
                            # ``_entry_has_promotion_path``) and lose existing
                            # metric data from view. The composite_score already
                            # penalizes the row via cap_ar=0 from nai=0.
                            logger.info(
                                "    AR gate-INV NO-GO flagged: pair=%.2f "
                                "held_class=%.2f (frequency-collapse; row stays "
                                "in tier — composite_score handles ranking)",
                                nai.in_dist_pair_acc,
                                nai.held_class_acc,
                            )
                    except (RuntimeError, ValueError, TypeError, ImportError) as e_nai:
                        logger.debug("AR gate-INV probe skipped: %s", e_nai)

                # Binding composite: 0.4 * ar_gate_score + 0.3*induction + 0.3*binding.
                # ar_legacy_auc is read-only legacy; it does not contribute weight here.
                nai_score = result.get("ar_gate_score")
                ind_val = ind.auc if ind is not None else None
                bind_val = result.get("binding_screening_auc")
                if ind_val is not None and bind_val is not None:
                    if nai_score is not None:
                        result["binding_screening_composite"] = round(
                            0.4 * nai_score + 0.3 * ind_val + 0.3 * bind_val, 4
                        )
                    else:
                        result["binding_screening_composite"] = round(
                            0.3 * ind_val + 0.3 * bind_val, 4
                        )

                logger.info(
                    "    Screening probes: induction=%s binding=%s ar=%s bc=%s",
                    (
                        f"{ind.auc:.3f} ({ind.elapsed_ms:.0f}ms)"
                        if ind is not None
                        else "skip"
                    ),
                    (
                        f"{br.auc:.3f} ({br.elapsed_ms:.0f}ms)"
                        if br is not None
                        else "skip"
                    ),
                    (
                        f"{ar.auc:.3f} ({ar.elapsed_ms:.0f}ms)"
                        if ar is not None
                        else "skip"
                    ),
                    (
                        f"{result.get('binding_screening_composite'):.3f}"
                        if result.get("binding_screening_composite") is not None
                        else "skip"
                    ),
                )

                # HIGH PRIORITY DISCOVERY: induction_screening_auc > 0.20 without
                # standard causal attention. This would be a novel mechanism
                # for exact token retrieval across gaps.
                if ind is not None and ind.auc > 0.20 and graph_data:
                    graph_nodes = []
                    if isinstance(graph_data, dict):
                        raw_nodes = graph_data.get("nodes", [])
                        if isinstance(raw_nodes, dict):
                            raw_nodes = list(raw_nodes.values())
                        graph_nodes = [
                            node
                            for node in raw_nodes
                            if isinstance(node, dict)
                            and not node.get("is_input", False)
                        ]
                    _has_attention = any(
                        n.get("op_name", n.get("op"))
                        in (
                            "softmax_attention",
                            "diff_attention",
                            "graph_attention",
                            "linear_attention",
                        )
                        for n in graph_nodes
                    )
                    if not _has_attention:
                        logger.warning(
                            "*** HIGH PRIORITY DISCOVERY: %s induction_screening_auc=%.3f "
                            "WITHOUT standard attention ops! Investigate immediately. "
                            "Graph ops: %s",
                            result.get("graph_fingerprint", "?")[:10],
                            ind.auc,
                            [n.get("op_name", n.get("op")) for n in graph_nodes],
                        )
            except (RuntimeError, ValueError, TypeError, ImportError) as e_bp:
                logger.debug("Binding probes skipped: %s", e_bp)

        # Post-S1 triage: cheap evals for composite score dimensions
        if (
            result.get("passed")
            and model is not None
            and not bool(getattr(config, "skip_post_s1_fingerprint", False))
            and not bool(getattr(config, "profile_disable_post_eval", False))
        ):
            try:
                from .execution_triage import run_triage

                _graph_for_triage = None
                if graph_json:
                    try:
                        from ...synthesis.serializer import graph_from_json

                        _graph_for_triage = graph_from_json(graph_json)
                    except (ValueError, KeyError, json.JSONDecodeError) as e:
                        logger.debug("Graph deserialization failed for triage: %s", e)
                triage = run_triage(
                    model,
                    _graph_for_triage,
                    result,
                    config.model_dim,
                )
                if triage:
                    result.update(triage)
                    _n_rt = triage.get("n_routing_ops", 0)
                    _n_sp = triage.get("n_sparse_ops", 0)
                    _n_mo = triage.get("n_moe_ops", 0)
                    _qpp = triage.get("param_efficiency", 0)
                    logger.info(
                        "    Triage: %d fields (qpp=%.2f, route=%d sparse=%d moe=%d)",
                        len(triage),
                        _qpp,
                        _n_rt,
                        _n_sp,
                        _n_mo,
                    )
            except (RuntimeError, ValueError, TypeError) as e_tri:
                logger.debug("Triage eval skipped: %s", e_tri)

            # Gemini trajectory metrics — Jacobian ERF, ICLD velocity, logit
            # margin slope, spec_norm, AND ID Collapse Rate. The latter
            # uses snapshots captured in execution_training_program at
            # ~20% and ~100% of training; if the trainer didn't capture
            # them (short run, exception, etc.) compute_trajectory_metrics
            # gracefully reports id_collapse=None. Phase tag matches the
            # screening lifecycle stage so ML training can condition on it.
            try:
                from ...eval.trajectory_metrics import compute_trajectory_metrics

                _id_early = getattr(self, "_id_collapse_early_snap", None)
                _id_late = getattr(self, "_id_collapse_late_snap", None)
                _traj = compute_trajectory_metrics(
                    model,
                    metric_phase="screening_750",
                    device=str(getattr(model, "_aria_device", "cuda")),
                    spec_norm_vocab_size=int(getattr(config, "vocab_size", 32000)),
                    id_collapse_early=_id_early,
                    id_collapse_late=_id_late,
                )
                result.update(_traj.to_column_dict())
                _id_rate = (
                    _traj.id_collapse.collapse_rate
                    if _traj.id_collapse is not None
                    else None
                )
                logger.info(
                    "    Trajectory: erf_d=%.2f erf_var=%.0f icld=%+.4f margin=%+.4f "
                    "sn=%.1f id_rate=%s",
                    _traj.jacobian_erf.density or 0.0,
                    _traj.jacobian_erf.variance or 0.0,
                    _traj.icld.velocity or 0.0,
                    _traj.logit_margin.velocity or 0.0,
                    _traj.spec_norm or 0.0,
                    f"{_id_rate:+.4f}" if _id_rate is not None else "n/a",
                )
            except (RuntimeError, ValueError, TypeError) as e_traj:
                logger.debug("Trajectory metrics skipped: %s", e_traj)
            finally:
                # Clear so the next program in this experiment starts
                # without inheriting stale snapshots.
                self._id_collapse_early_snap = None
                self._id_collapse_late_snap = None
                self._id_collapse_probe_ids = None

    # ── Scale-Up Mode ──
