"""Execution screening — per-candidate gates, pipeline, finalize. Split from execution_screening."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import torch

from ..json_utils import fast_dumps
from ..notebook import LabNotebook
from ..native_runner import compile_model_native_first as _compile_model_native
from ...synthesis.compiler import compile_model as _compile_model_legacy
from ...synthesis.validator import validate_graph
from ...eval.flops import estimate_flops
from ...eval.perf_budget import evaluate_perf_budget_gate
from .execution_screening_graphs import (
    analyze_graph_for_screening,
    structural_gate_failure,
    toxic_failure_ratio,
)
from ._helpers_gate import clear_gpu_memory
from ._helpers_metrics import (
    _native_proactive_gating,
    graph_observed_routing_ops,
    graph_routing_ops,
    routing_fast_lane_fields,
)
from ._types import RunConfig, LiveProgress

import logging

logger = logging.getLogger(__name__)


# These three names live in the sibling execution_screening module. We can't
# import them at top level because that module imports _this_ module at the
# bottom of its own body (mixin composition), which creates a circular load.
# Fetch them lazily from sys.modules — once execution_screening has finished
# loading (which it always has by the time any runner method is called via
# an instance), the lookup is a single dict access.
def _es_mod():
    from . import execution_screening as _es  # cached after first call

    return _es


def _make_experiment_results() -> Dict[str, Any]:
    """Create a fresh experiment results dict with all counters zeroed."""
    return {
        "total": 0,
        "stage0_passed": 0,
        "stage05_passed": 0,
        "rapid_screening_killed": 0,
        "rapid_screening_kill_reasons": {},
        "stage09_passed": 0,
        "stage1_passed": 0,
        "novel_count": 0,
        "best_loss_ratio": None,
        "best_novelty_score": None,
        "survivors": [],
        "skipped_proactive_gating": 0,
        "proactive_gating_failures": [],
        "funnel_counts": {
            "raw_generated": 0,
            "post_batch_dedup": 0,
            "judgment_filtered": 0,
            "post_judgment": 0,
            "screening_considered": 0,
            "dropped_runtime_dedup": 0,
            "dropped_toxic": 0,
            "dropped_proactive_gating": 0,
            "dropped_invalid_graph": 0,
            "dropped_runtime_error": 0,
            "stage0_attempted": 0,
            "stage0_passed": 0,
            "dropped_stage0": 0,
            "stage05_passed": 0,
            "dropped_stage05": 0,
            "stage09_passed": 0,
            "dropped_stage09": 0,
            "stage1_passed": 0,
            "dropped_stage1": 0,
        },
    }


class _ExecutionScreeningPipelineMixin:
    """Per-candidate gate checks, compile/eval, pipeline, finalize, execute_experiment."""

    __slots__ = ()

    def _screen_candidate_quality_gates(
        self,
        graph,
        fp: str,
        config: RunConfig,
        results: Dict,
        graph_analysis,
        _judgment_scores: Dict,
        i: int,
    ) -> Optional[str]:
        """Collect metrics, run proactive gating and graph validation.

        On pass: populates self._last_gate_program_metrics and
        self._last_gate_graph_analysis for the caller.

        Returns a skip-reason string if the candidate should be skipped,
        None if it passed.
        """
        # Collect all metrics for this program
        program_metrics: Dict[str, Any] = {}
        program_metrics.update(self._extract_graph_metrics(graph))
        j_score = _judgment_scores.get(id(graph))
        if j_score is not None:
            program_metrics["judgment_score"] = j_score

        # Estimate FLOPs
        try:
            flop_est = estimate_flops(
                graph,
                seq_len=min(128, config.max_seq_len),
                d_model=config.model_dim,
            )
            program_metrics["flops_forward"] = flop_est.flops_forward
            program_metrics["flops_per_param"] = flop_est.flops_per_param
            program_metrics["flops_per_token"] = flop_est.flops_per_token
        except Exception as e:
            logger.debug("FLOP estimate failed for %s: %s", graph.fingerprint()[:10], e)

        # Native Proactive Gating (Project Hephaestus)
        try:
            native_gating = _native_proactive_gating(graph)
            if not native_gating.get("passed", True):
                results.setdefault("skipped_proactive_gating", 0)
                results["skipped_proactive_gating"] += 1
                results["funnel_counts"]["dropped_proactive_gating"] += 1
                program_metrics["proactive_gating_reason"] = native_gating.get("reason")
                program_metrics["max_depth"] = native_gating.get("max_depth")
                program_metrics["n_toxic_motifs"] = native_gating.get("n_toxic_motifs")
                self._emit_event(
                    "program_evaluated",
                    {
                        "index": i,
                        "fingerprint": fp[:10],
                        "result": "skipped_proactive",
                        "reason": native_gating.get("reason"),
                        "max_depth": native_gating.get("max_depth"),
                    },
                )
                return "proactive_gating"
        except Exception as e:
            logger.debug("Native proactive gating failed for %s: %s", fp[:10], e)

        # Validate
        validation = validate_graph(
            graph,
            max_ops=max(1, int(config.max_ops)),
            max_depth=max(1, int(config.max_depth)),
            min_splits=config.min_splits,
        )
        if not validation.valid:
            results.setdefault("s0_validation_failures", 0)
            results["s0_validation_failures"] += 1
            results["funnel_counts"]["dropped_invalid_graph"] += 1
            self._emit_event(
                "program_evaluated",
                {
                    "index": i,
                    "fingerprint": fp[:10],
                    "result": "invalid",
                    "error": validation.errors[0] if validation.errors else "",
                },
            )
            return "invalid_graph"

        # Stash results for the caller
        self._last_gate_program_metrics = program_metrics
        self._last_gate_graph_analysis = graph_analysis
        return None

    def _screen_candidate_gates(
        self,
        graph,
        fp: str,
        failure_blocklist,
        config: RunConfig,
        results: Dict,
        nb: LabNotebook,
        _judgment_scores: Dict,
        _get_primitive,
        i: int,
    ) -> Optional[str]:
        """Run pre-compilation gates on a single candidate.

        Checks: runtime dedup, structural gate, toxic ratio, FLOPs estimate,
        proactive gating, graph validation.

        On pass: populates self._last_gate_program_metrics and
        self._last_gate_graph_analysis for the caller to consume.

        Returns a skip-reason string if the candidate should be skipped,
        None if it passed all gates.
        """
        # Real-time dedup: skip if evaluated by another process since experiment start
        if not getattr(config, "disable_runtime_dedup", False) and nb.has_fingerprint(
            fp
        ):
            results.setdefault("skipped_dedup_runtime", 0)
            results["skipped_dedup_runtime"] += 1
            results["funnel_counts"]["dropped_runtime_dedup"] += 1
            self._emit_event(
                "program_evaluated",
                {
                    "index": i,
                    "fingerprint": fp[:10],
                    "result": "skipped_dedup",
                },
            )
            return "dedup"

        # ── Structural gates + toxic pre-screen (single-pass analysis) ──
        graph_analysis = analyze_graph_for_screening(graph, _get_primitive)
        # gate8_retrieval_dead is opt-in via capability_first mode. The flag
        # lives on RunConfig as ``_capability_first_mode`` (the UI toggle);
        # the corresponding ``binding_capable_required`` field on
        # GrammarConfig is not visible at this call site. Use the RunConfig
        # flag as the source of truth so gate8 actually fires when the user
        # enables capability-first.
        gate_fail = structural_gate_failure(
            graph,
            routing_mandatory=bool(config.routing_mandatory),
            efficiency_ops=_es_mod()._EFFICIENCY_OPS,
            analysis=graph_analysis,
            binding_capable_required=bool(
                getattr(config, "_capability_first_mode", False)
            ),
        )
        if gate_fail is not None:
            results["funnel_counts"].setdefault("dropped_structural_gate", 0)
            results["funnel_counts"]["dropped_structural_gate"] += 1
            results["funnel_counts"].setdefault(f"dropped_{gate_fail}", 0)
            results["funnel_counts"][f"dropped_{gate_fail}"] += 1
            return gate_fail

        if failure_blocklist:
            _toxic_ratio = toxic_failure_ratio(failure_blocklist, graph_analysis)
            if _toxic_ratio >= 0.5:
                results.setdefault("skipped_toxic", 0)
                results["skipped_toxic"] += 1
                results["funnel_counts"]["dropped_toxic"] += 1
                self._emit_event(
                    "program_evaluated",
                    {
                        "index": i,
                        "fingerprint": fp[:10],
                        "result": "skipped_toxic",
                        "toxic_ratio": f"{_toxic_ratio:.2f}",
                    },
                )
                return "toxic"

        # Collect metrics, run proactive gating, validate
        skip = self._screen_candidate_quality_gates(
            graph=graph,
            fp=fp,
            config=config,
            results=results,
            graph_analysis=graph_analysis,
            _judgment_scores=_judgment_scores,
            i=i,
        )
        if skip is not None:
            return skip

        return None

    def _handle_s0_s05_failure(
        self,
        s0_passed: bool,
        s05_passed: bool,
        sandbox_result,
        results: Dict,
        config: RunConfig,
        nb: LabNotebook,
        exp_id: str,
        graph,
        fp: str,
        i: int,
    ) -> None:
        """Record S0/S0.5 failure: update funnel counts, emit event, persist."""
        if not s0_passed:
            results["funnel_counts"]["dropped_stage0"] += 1
        else:
            results["funnel_counts"]["dropped_stage05"] += 1
        error_type = sandbox_result.error_type or "unknown"
        results.setdefault("failure_error_types", {})
        results["failure_error_types"][error_type] = (
            results["failure_error_types"].get(error_type, 0) + 1
        )
        self._emit_event(
            "program_evaluated",
            {
                "index": i,
                "fingerprint": fp[:10],
                "result": "fail_s0" if not s0_passed else "fail_s05",
                "error": (sandbox_result.error or "")[:120] if not s0_passed else None,
                "error_type": error_type,
                "stability": f"{sandbox_result.stability_score:.2f}"
                if s0_passed and not s05_passed
                else None,
                "params": sandbox_result.param_count
                if sandbox_result.param_count
                else None,
                "memory_mb": f"{sandbox_result.peak_memory_mb:.1f}"
                if sandbox_result.peak_memory_mb
                else None,
                "has_nan": sandbox_result.has_nan_output
                or sandbox_result.has_nan_grad
                or None,
                "has_inf": sandbox_result.has_inf_output or None,
            },
        )
        if config.persist_screening_failures:
            _es_mod()._record_screening_failure(
                nb=nb,
                exp_id=exp_id,
                graph=graph,
                stage0_passed=bool(s0_passed),
                stage05_passed=bool(s05_passed),
                error_type=error_type,
                error_message=(sandbox_result.error or "")[:240] or None,
                stage_at_death="stage0" if not s0_passed else "stage05",
                stability_score=sandbox_result.stability_score,
            )

    def _screen_candidate_compile_eval(
        self,
        graph,
        graph_analysis,
        config: RunConfig,
        dev_str: str,
        results: Dict,
        nb: LabNotebook,
        exp_id: str,
        fp: str,
        program_metrics: Dict[str, Any],
        _s0_op_counts: Dict[str, Dict[str, int]],
        i: int,
    ) -> Optional[Dict[str, Any]]:
        """Compile a candidate graph and run S0/S0.5 sandbox eval.

        Returns a dict with keys (model, sandbox_result, layer_graphs,
        use_progressive, phase1_vocab) on success, or None if the candidate
        failed S0 or S0.5.
        """
        # Progressive screening: Phase 1 uses cheap qualifying vocab (32K)
        # for S0/S0.5/rapid.  Only Phase 1 survivors get recompiled at
        # the real vocab for S1 training.
        _use_progressive = (
            config.progressive_screening
            and config.vocab_size > config.qualifying_vocab_size
        )
        _phase1_vocab = (
            config.qualifying_vocab_size if _use_progressive else config.vocab_size
        )

        results["funnel_counts"]["stage0_attempted"] += 1
        # Z13: Defensive pause + GC to stabilize Torch Dynamo context if needed
        if i > 0 and i % 10 == 0:
            clear_gpu_memory()

            # More aggressive reset every 50 to clear Torch Dynamo cache
            if i % 50 == 0:
                try:
                    torch.compiler.reset()
                except (AttributeError, RuntimeError) as e:
                    logger.debug("torch.compiler.reset() unavailable: %s", e)

            time.sleep(0.1)

        layer_graphs = [graph] * config.n_layers
        _compile_t0 = time.perf_counter()
        # Capability-first templates produce novel graph topologies that
        # can trigger segfaults in the native C/Rust/Cython dispatch path
        # (the native kernels were written before these templates existed).
        # Fall back to pure PyTorch compilation which is slower but never
        # segfaults. Once the native kernels are hardened for the role-slot
        # op combinations, this guard can be removed.
        _compiler = (
            _compile_model_legacy
            if getattr(config, "_capability_first_mode", False)
            else _compile_model_native
        )
        model = _compiler(
            layer_graphs,
            vocab_size=_phase1_vocab,
            max_seq_len=config.max_seq_len,
        )
        _compile_ms = (time.perf_counter() - _compile_t0) * 1000.0
        program_metrics["compile_time_ms"] = (
            float(program_metrics.get("compile_time_ms", 0.0) or 0.0) + _compile_ms
        )
        results.setdefault("_compile_times_ms", []).append(_compile_ms)
        _eval_timeout = 60 if getattr(config, "_exotic_mode", False) else 30
        sandbox_result = self._safe_eval_for_stage(
            model,
            stage_tag="candidate_screening",
            batch_size=2,
            seq_len=min(128, config.max_seq_len),
            vocab_size=_phase1_vocab,
            device=dev_str,
            timeout_seconds=_eval_timeout,
        )
        program_metrics.update(self._extract_sandbox_metrics(sandbox_result))
        program_metrics["param_count"] = sandbox_result.param_count

        s0_passed = sandbox_result.passed
        s05_passed = (
            sandbox_result.stability_score >= config.stage05_stability_threshold
            and sandbox_result.causality_passed
        )

        if s0_passed:
            results["stage0_passed"] += 1
            results["funnel_counts"]["stage0_passed"] += 1
            with self._lock:
                self._progress.stage0_passed += 1
        if s05_passed:
            results["stage05_passed"] += 1
            results["funnel_counts"]["stage05_passed"] += 1
            with self._lock:
                self._progress.stage05_passed += 1

        # Track ALL compiled programs in _s0_op_counts so
        # merge_op_failure_counts sees both passes and failures.
        for op_name in graph_analysis.counted_ops:
            c = _s0_op_counts.setdefault(op_name, {"n_used": 0, "n_s0": 0, "n_s05": 0})
            c["n_used"] += 1
            if s0_passed:
                c["n_s0"] += 1
            if s05_passed:
                c["n_s05"] += 1

        if not s0_passed or not s05_passed:
            self._handle_s0_s05_failure(
                s0_passed=s0_passed,
                s05_passed=s05_passed,
                sandbox_result=sandbox_result,
                results=results,
                config=config,
                nb=nb,
                exp_id=exp_id,
                graph=graph,
                fp=fp,
                i=i,
            )
            return None

        return {
            "model": model,
            "sandbox_result": sandbox_result,
            "layer_graphs": layer_graphs,
            "use_progressive": _use_progressive,
            "phase1_vocab": _phase1_vocab,
        }

    def _screen_candidate_pipeline(
        self,
        i: int,
        graph,
        model,
        config: RunConfig,
        nb: LabNotebook,
        exp_id: str,
        fp: str,
        dev_str: str,
        results: Dict,
        program_metrics: Dict[str, Any],
        sandbox_result,
        layer_graphs: List,
        _use_progressive: bool,
        _phase1_vocab: int,
        stage1_config: RunConfig,
        stage09_enabled: bool,
        orchestrator,
        candidate_batch_size: int,
    ) -> bool:
        """Run S0.75 through S1 queue submission for a single candidate.

        Returns True if the candidate should be skipped (continue in the outer loop),
        False if it was successfully queued for S1.
        """
        # S0.75: Initial-loss pre-screen (5 gradient steps)
        # Architectures with init_loss > 50 have deep unscaled
        # projection chains and waste rapid-screen + S1 budget.
        try:
            _s075_dev = torch.device(dev_str)
            model.train()
            _s075_opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
            _s075_ids = torch.randint(0, _phase1_vocab, (4, 64), device=_s075_dev)
            with torch.amp.autocast(
                device_type=_s075_dev.type,
                dtype=torch.bfloat16,
                enabled=(_s075_dev.type == "cuda"),
            ):
                _s075_logits = model(_s075_ids)
                _s075_loss = torch.nn.functional.cross_entropy(
                    _s075_logits[:, :-1].reshape(-1, _s075_logits.size(-1)),
                    _s075_ids[:, 1:].reshape(-1),
                )
            _s075_init_loss = _s075_loss.item()
            program_metrics["s075_initial_loss"] = _s075_init_loss

            if (
                not math.isnan(_s075_init_loss)
                and not math.isinf(_s075_init_loss)
                and _s075_init_loss > _es_mod().INITIAL_LOSS_THRESHOLD
            ):
                results["funnel_counts"]["dropped_s075_high_init"] += 1
                self._emit_event(
                    "program_evaluated",
                    {
                        "index": i,
                        "fingerprint": fp[:10],
                        "result": "fail_s075",
                        "initial_loss": round(_s075_init_loss, 1),
                        "threshold": _es_mod().INITIAL_LOSS_THRESHOLD,
                    },
                )
                if config.persist_screening_failures:
                    _es_mod()._record_screening_failure(
                        nb=nb,
                        exp_id=exp_id,
                        graph=graph,
                        stage0_passed=True,
                        stage05_passed=True,
                        error_type="high_initial_loss",
                        error_message=(
                            f"initial_loss={_s075_init_loss:.4f} > "
                            f"{_es_mod().INITIAL_LOSS_THRESHOLD:.4f}"
                        ),
                        stage_at_death="stage075",
                        stability_score=sandbox_result.stability_score,
                    )
                del _s075_opt
                return True

            # Clean up probe state before rapid screening
            _s075_opt.zero_grad(set_to_none=True)
            del _s075_opt
        except Exception as s075_err:
            logger.warning(
                "S0.75 probe failed for graph %d, skipping check: %s",
                i,
                s075_err,
            )

        # Rapid Screening: 150-step gradient health check
        # Catches exploding grads, NaN, stalled loss, routing collapse
        # BEFORE committing to full Stage 1 training budget.
        from ...eval.screening_rapid import RapidScreeningCheck

        rapid = RapidScreeningCheck()
        results["funnel_counts"]["rapid_screen_attempted"] += 1
        rapid_result = rapid.run(
            model,
            vocab_size=_phase1_vocab,
            seq_len=min(128, config.max_seq_len),
            batch_size=2,
            device=dev_str,
        )
        program_metrics["rapid_screening_passed"] = rapid_result.passed
        program_metrics["rapid_screening_elapsed_ms"] = rapid_result.elapsed_ms
        program_metrics["rapid_screening_steps_completed"] = rapid_result.metrics.get(
            "steps_completed"
        )
        program_metrics["rapid_screening_max_steps"] = rapid.max_steps
        program_metrics["rapid_screening_gpu_minutes_saved"] = (
            rapid_result.gpu_minutes_saved
        )
        program_metrics["rapid_screening_metrics"] = rapid_result.metrics
        # Extract screening loss checkpoints for failure diagnostics
        _rm = rapid_result.metrics
        for _step, _col in (
            (10, "screening_loss_10"),
            (25, "screening_loss_25"),
            (50, "screening_loss_50"),
        ):
            _key = f"loss_at_{_step}"
            if _key not in _rm and len(_rm.get("losses", [])) >= _step:
                _rm[_key] = _rm["losses"][_step - 1]
            if _rm.get(_key) is not None:
                program_metrics[_col] = _rm[_key]
        # Extract grad norms even for failures
        if _rm.get("max_grad_norm") is not None:
            program_metrics.setdefault("max_grad_norm", _rm["max_grad_norm"])
            program_metrics.setdefault("mean_grad_norm", _rm["mean_grad_norm"])
        if rapid_result.degraded:
            program_metrics["rapid_screening_degraded"] = True
            program_metrics["rapid_screening_degraded_reasons"] = (
                rapid_result.degraded_reasons
            )
        if not rapid_result.passed:
            results["rapid_screening_killed"] += 1
            results["funnel_counts"]["dropped_rapid_screening"] += 1
            kr = rapid_result.kill_reason or "unknown"
            results["rapid_screening_kill_reasons"][kr] = (
                results["rapid_screening_kill_reasons"].get(kr, 0) + 1
            )
            program_metrics["rapid_screening_kill_reason"] = rapid_result.kill_reason
            program_metrics["rapid_screening_kill_step"] = rapid_result.kill_step
            program_metrics["rapid_screening_kill_metric"] = rapid_result.kill_metric
            self._emit_event(
                "program_evaluated",
                {
                    "index": i,
                    "fingerprint": fp[:10],
                    "result": "fail_rapid_screening",
                    "kill_reason": rapid_result.kill_reason,
                    "kill_step": rapid_result.kill_step,
                    "gpu_minutes_saved": rapid_result.gpu_minutes_saved,
                },
            )
            if config.persist_screening_failures:
                _es_mod()._record_screening_failure(
                    nb=nb,
                    exp_id=exp_id,
                    graph=graph,
                    stage0_passed=True,
                    stage05_passed=True,
                    error_type="rapid_screening_error",
                    error_message=(
                        rapid_result.kill_reason or "rapid_screening_failed"
                    )[:240],
                    stage_at_death="rapid_screening",
                    stability_score=sandbox_result.stability_score,
                    extra_metrics=program_metrics,
                )
            return True

        # Phase 2: recompile at real vocab for S1 training
        if _use_progressive:
            del model
            clear_gpu_memory()
            _compile_t0 = time.perf_counter()
            _prog_compiler = (
                _compile_model_legacy
                if getattr(config, "_capability_first_mode", False)
                else _compile_model_native
            )
            model = _prog_compiler(
                layer_graphs,
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
            )
            _compile_ms = (time.perf_counter() - _compile_t0) * 1000.0
            program_metrics["compile_time_ms"] = (
                float(program_metrics.get("compile_time_ms", 0.0) or 0.0) + _compile_ms
            )
            results.setdefault("_compile_times_ms", []).append(_compile_ms)
            program_metrics["progressive_phase2_compiled"] = True

        routing_ops = graph_routing_ops(graph)
        observed_routing_ops = graph_observed_routing_ops(graph)
        if routing_ops:
            program_metrics["routing_fast_lane_applied"] = 1
            try:
                from ...eval.wikitext_eval import screening_wikitext_eval

                _compile_t0 = time.perf_counter()
                _fl_compiler = (
                    _compile_model_legacy
                    if getattr(config, "_capability_first_mode", False)
                    else _compile_model_native
                )
                fast_lane_model = _fl_compiler(
                    layer_graphs,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                )
                _compile_ms = (time.perf_counter() - _compile_t0) * 1000.0
                program_metrics["compile_time_ms"] = (
                    float(program_metrics.get("compile_time_ms", 0.0) or 0.0)
                    + _compile_ms
                )
                results.setdefault("_compile_times_ms", []).append(_compile_ms)
                fast_lane = screening_wikitext_eval(
                    fast_lane_model,
                    config.vocab_size,
                    "cpu",
                    seq_len=min(96, config.max_seq_len),
                    n_train_steps=24,
                    n_train_batches=8,
                    n_eval_batches=3,
                    batch_size=3,
                )
                fast_lane["routing_fast_lane_applied"] = 1
                fast_lane["routing_fast_lane_status"] = fast_lane.get(
                    "screening_wikitext_status"
                )
                fast_lane["routing_fast_lane_metric_version"] = "routing_fast_lane_v1"
                fast_lane["routing_fast_lane_perplexity"] = fast_lane.get(
                    "wikitext_perplexity"
                )
                fast_lane["routing_fast_lane_score"] = fast_lane.get("wikitext_score")
                fast_lane["routing_fast_lane_pre_perplexity"] = fast_lane.get(
                    "wikitext_pre_perplexity"
                )
                fast_lane["routing_fast_lane_ppl_improvement"] = fast_lane.get(
                    "wikitext_ppl_improvement"
                )
                fast_lane["routing_fast_lane_elapsed_ms"] = fast_lane.get("elapsed_ms")
                fast_lane["routing_fast_lane_budget"] = dict(
                    fast_lane.get("screening_wikitext_budget") or {}
                )
                fast_lane["routing_fast_lane_slope"] = fast_lane.get("screening_slope")
                fast_lane["routing_fast_lane_slope_consistent"] = fast_lane.get(
                    "screening_slope_consistent"
                )
                fast_lane["routing_fast_lane_routing_ops"] = routing_ops
                fast_lane["routing_observed_ops"] = observed_routing_ops
                program_metrics.update(routing_fast_lane_fields(fast_lane))
                logger.info(
                    "    Routing fast lane trigger_ops=%s observed_ops=%s score=%.3f status=%s (%.0fms)%s",
                    ",".join(routing_ops),
                    ",".join(observed_routing_ops) if observed_routing_ops else "-",
                    float(fast_lane.get("routing_fast_lane_score") or 0.0),
                    fast_lane.get("routing_fast_lane_status") or "unknown",
                    float(fast_lane.get("routing_fast_lane_elapsed_ms") or 0.0),
                    (
                        f" error={fast_lane.get('error')}"
                        if fast_lane.get("routing_fast_lane_status") != "ok"
                        and fast_lane.get("error")
                        else ""
                    ),
                )
                # HellaSwag fast lane probe (25 examples)
                try:
                    from ...eval.hellaswag_eval import screening_hellaswag_eval

                    hs_fl = screening_hellaswag_eval(
                        fast_lane_model,
                        config.vocab_size,
                        "cpu",
                        n_examples=25,
                    )
                    program_metrics["routing_fast_lane_hellaswag_acc"] = hs_fl.get(
                        "hellaswag_acc"
                    )
                except Exception as e_hs_fl:
                    logger.debug("Fast lane HellaSwag skipped: %s", e_hs_fl)

                del fast_lane_model
            except Exception as e_fast_lane:
                logger.warning("Routing fast lane failed: %s", e_fast_lane)
                program_metrics.setdefault("routing_fast_lane_applied", 1)
                program_metrics["routing_fast_lane_status"] = "eval_failed"
                program_metrics["routing_fast_lane_metric_version"] = (
                    "routing_fast_lane_v1"
                )
                program_metrics["routing_fast_lane_routing_ops_json"] = fast_dumps(
                    routing_ops
                )

        # Stage 1: Asynchronous Execution (Z6)
        self._update_progress(current_stage="queuing_s1")

        screening_seed = self._stable_seed(exp_id, i, "screening")
        orchestrator.submit(
            index=i,
            graph=graph,
            config=stage1_config,
            seed=screening_seed,
            payload={
                "metrics": program_metrics,
                "graph": graph,
                "batch_id": i // candidate_batch_size,
                "queue_kind": "candidate_screening",
                "screening_stage": "stage09" if stage09_enabled else "stage1",
                "screening_seed": screening_seed,
            },
            model=model,  # Reuse compiled model (at real vocab)
        )
        results["funnel_counts"]["stage1_queued"] += 1
        return False

    def _finalize_experiment_results(
        self,
        exp_id: str,
        orchestrator,
        nb: LabNotebook,
        results: Dict,
        config: RunConfig,
        t_start: float,
        _s0_op_counts: Dict[str, Dict[str, int]],
    ) -> None:
        """Drain orchestrator queue and log final experiment metrics."""
        # Wait for remaining asynchronous Stage 1 evaluations
        self._update_progress(status="finalizing_evaluations")

        while (
            orchestrator.job_queue.unfinished_tasks > 0
            or not orchestrator.result_queue.empty()
        ):
            if self._stop_event.is_set():
                break
            self._process_orchestrator_results(
                orchestrator, nb, exp_id, results, config
            )
            time.sleep(0.5)

        queue_telemetry = orchestrator.get_telemetry()
        orchestrator.shutdown()
        results["queue_telemetry"] = queue_telemetry
        results["elapsed_seconds"] = time.time() - t_start
        results["perf_report"] = self._build_experiment_perf_report(
            results, queue_telemetry=queue_telemetry
        )
        results["perf_budget_gate"] = evaluate_perf_budget_gate(results["perf_report"])
        results.pop("_perf_traces", None)
        results.pop("_gpu_starvation", None)
        results.pop("_kernel_timing", None)
        if _s0_op_counts:
            results["_s0_op_counts"] = _s0_op_counts

        elapsed = results.get("elapsed_seconds", time.time() - t_start)
        self._update_progress(
            elapsed_seconds=elapsed,
            status="analyzing",
            aria_message=self.aria.begin_analysis(),
        )

        best = results.get("best_loss_ratio")
        best_str = f", best loss={best:.4f}" if best else ""
        dedup_str = ""
        if results.get("skipped_dedup", 0) > 0:
            dedup_str = f", dedup={results['skipped_dedup']} ({results.get('dedup_rate', 0) * 100:.0f}%)"
        rapid_killed = results.get("rapid_screening_killed", 0)
        rapid_str = f", rapid_killed={rapid_killed}" if rapid_killed else ""
        stage09 = results.get("stage09_passed", 0)
        stage09_str = f", S0.9={stage09}" if stage09 else ""
        logger.info(
            "Experiment %s complete: %d programs → S0=%d → S0.5=%d → S1=%d "
            "(%.1fs)%s%s%s%s%s",
            exp_id[:8],
            results["total"],
            results["stage0_passed"],
            results["stage05_passed"],
            results["stage1_passed"],
            elapsed,
            best_str,
            dedup_str,
            rapid_str,
            stage09_str,
            f", native_gating={results.get('skipped_proactive_gating', 0)}"
            if results.get("skipped_proactive_gating")
            else "",
        )

        self._live_training_context = None

    def _execute_experiment(
        self,
        exp_id: str,
        config: RunConfig,
        nb: LabNotebook,
        use_learned_grammar: bool = True,
    ) -> Dict:
        """Core experiment logic shared by single and continuous modes."""
        self._live_training_context = {"exp_id": exp_id, "phase": "synthesis"}
        stage09_enabled = bool(
            getattr(config, "enable_stage09_cheap_train_gate", False)
        )
        stage1_config = (
            _es_mod()._make_stage1_screening_config(config)
            if stage09_enabled
            else config
        )
        with self._lock:
            # Z17: Explicitly reset progress object at start of execution
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=config.n_programs,
                aria_message=f"{self.aria.NAME}: Initializing experiment {exp_id[:8]}...",
            )

        results = _make_experiment_results()

        # Phase 1: Grammar preparation
        grammar, failure_blocklist, analytics = self._prepare_grammar_config(
            exp_id, config, nb, results, use_learned_grammar=use_learned_grammar
        )

        # Phase 2: Candidate generation & filtering
        (
            graphs,
            _judgment_scores,
            dev,
            dev_str,
            orchestrator,
            candidate_batch_size,
            t_start,
        ) = self._generate_and_filter_candidates(
            exp_id,
            config,
            nb,
            grammar,
            analytics,
            results,
            use_learned_grammar=use_learned_grammar,
        )

        # Early return for morphological box (already handled inside _generate_and_filter_candidates)
        if not graphs and config.model_source == "morphological_box":
            return results

        # Track ops from S0 failures for op_success_rates (not stored in DB)
        _s0_op_counts: Dict[str, Dict[str, int]] = {}  # op -> {n_used, n_s0, n_s05}

        # Pre-import outside graph loop to avoid per-node import overhead
        try:
            from ...synthesis.primitives import get_primitive as _get_primitive
        except ImportError:
            _get_primitive = None

        # Phase 3: Per-candidate screening loop
        for i, graph in enumerate(graphs):
            if self._stop_event.is_set():
                break

            results["funnel_counts"]["screening_considered"] += 1
            fp = graph.fingerprint()
            self._update_progress(
                current_program=i + 1,
                current_fingerprint=fp[:10],
                elapsed_seconds=time.time() - t_start,
            )

            # Gate checks: dedup, structural, toxic, proactive gating, validation
            gate_skip = self._screen_candidate_gates(
                graph=graph,
                fp=fp,
                failure_blocklist=failure_blocklist,
                config=config,
                results=results,
                nb=nb,
                _judgment_scores=_judgment_scores,
                _get_primitive=_get_primitive,
                i=i,
            )
            if gate_skip is not None:
                # gate_skip is (reason, program_metrics_or_None)
                continue

            # Collect metrics and estimate FLOPs (populated by gates on pass-through)
            program_metrics = self._last_gate_program_metrics
            graph_analysis = self._last_gate_graph_analysis

            # Compile, S0/S0.5 eval, op counts
            try:
                compile_result = self._screen_candidate_compile_eval(
                    graph=graph,
                    graph_analysis=graph_analysis,
                    config=config,
                    dev_str=dev_str,
                    results=results,
                    nb=nb,
                    exp_id=exp_id,
                    fp=fp,
                    program_metrics=program_metrics,
                    _s0_op_counts=_s0_op_counts,
                    i=i,
                )
                if compile_result is None:
                    continue

                # S0.75 through S1 queue submission
                skip = self._screen_candidate_pipeline(
                    i=i,
                    graph=graph,
                    model=compile_result["model"],
                    config=config,
                    nb=nb,
                    exp_id=exp_id,
                    fp=fp,
                    dev_str=dev_str,
                    results=results,
                    program_metrics=program_metrics,
                    sandbox_result=compile_result["sandbox_result"],
                    layer_graphs=compile_result["layer_graphs"],
                    _use_progressive=compile_result["use_progressive"],
                    _phase1_vocab=compile_result["phase1_vocab"],
                    stage1_config=stage1_config,
                    stage09_enabled=stage09_enabled,
                    orchestrator=orchestrator,
                    candidate_batch_size=candidate_batch_size,
                )
                if skip:
                    continue

            except Exception as e:
                logger.error("Error evaluating graph %d: %s", i, e)
                results["funnel_counts"]["dropped_runtime_error"] += 1
                # Reset CUDA context if this was a fatal CUDA error
                if torch.cuda.is_available():
                    from ...eval.sandbox import is_cuda_fatal

                    if is_cuda_fatal(e):
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                            _probe = torch.zeros(1, device="cuda")
                            del _probe
                            torch.cuda.synchronize()
                            logger.info(
                                "CUDA context recovered after fatal error on graph %d",
                                i,
                            )
                        except (RuntimeError, OSError) as e_cuda:
                            logger.warning(
                                "CUDA context unrecoverable after fatal error on graph %d (reason: %s)",
                                i,
                                e_cuda,
                            )
                continue

            # Periodically process available results to keep the dashboard updated
            self._process_orchestrator_results(
                orchestrator, nb, exp_id, results, config
            )

        # Phase 4: Finalization
        self._finalize_experiment_results(
            exp_id, orchestrator, nb, results, config, t_start, _s0_op_counts
        )
        return results
