"""Dashboard mixin: orchestrator-result ingestion and persistence.

Handles the per-program ingestion pipeline: route stage-0.9 → stage-1
promotion, merge training metrics + perf traces, run baseline comparisons,
compute novelty, persist to the notebook, upsert leaderboard, emit events,
and resolve pending selection-insight trials."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import torch

from ..native_runner import compile_model_native_first as compile_model
from ..json_utils import json_safe
from ..notebook import LabNotebook
from ...eval.fingerprint import BehavioralFingerprint
from ...eval.diagnostic_tasks import run_diagnostic_suite
from ...eval.metrics import novelty_score
from ...synthesis.serializer import graph_to_json

logger = logging.getLogger(__name__)


_PROGRAM_RESULT_FLUSH_BATCH = 10


class _DashboardOrchestratorMixin:
    """Orchestrator-result ingestion + persistence."""

    def _process_orchestrator_results(
        self,
        orchestrator,
        nb,
        exp_id,
        results,
        config,
        wait_timeout: float = 0.0,
    ) -> int:
        """Collect and record all available results from the orchestrator."""
        job_results = orchestrator.get_results(timeout=wait_timeout)
        if not job_results:
            return 0
        for start in range(0, len(job_results), _PROGRAM_RESULT_FLUSH_BATCH):
            with nb.batch():
                for jr in job_results[start : start + _PROGRAM_RESULT_FLUSH_BATCH]:
                    self._record_orchestrator_result(jr, nb, exp_id, results, config)
        return len(job_results)

    def _merge_train_result_metrics(self, program_metrics, train_result, config):
        from ._helpers import screening_probe_fields, screening_wikitext_fields

        program_metrics["initial_loss"] = train_result.get("initial_loss")
        program_metrics["min_loss"] = train_result.get("min_loss")
        program_metrics["loss_improvement_rate"] = train_result.get(
            "loss_improvement_rate"
        )
        program_metrics["avg_step_time_ms"] = train_result.get("avg_step_time_ms")
        program_metrics["total_train_time_ms"] = train_result.get("total_train_time_ms")
        program_metrics["max_grad_norm"] = train_result.get("max_grad_norm")
        program_metrics["mean_grad_norm"] = train_result.get("mean_grad_norm")
        program_metrics["grad_norm_std"] = train_result.get("grad_norm_std")
        program_metrics["n_train_steps"] = train_result.get("n_train_steps")
        program_metrics["final_lr"] = train_result.get("final_lr")
        program_metrics["validation_loss"] = train_result.get("validation_loss")
        program_metrics["validation_loss_ratio"] = train_result.get(
            "validation_loss_ratio"
        )
        program_metrics["generalization_gap"] = train_result.get("generalization_gap")
        program_metrics["discovery_loss"] = train_result.get("discovery_loss")
        program_metrics["discovery_loss_ratio"] = train_result.get(
            "discovery_loss_ratio"
        )
        program_metrics["train_budget_steps"] = config.stage1_steps
        program_metrics.update(screening_wikitext_fields(train_result))
        program_metrics.update(screening_probe_fields(train_result))
        program_metrics.update(screening_probe_fields(program_metrics))
        program_metrics.update(
            {k: train_result.get(k) for k in train_result if k.startswith("pruning_")}
        )
        if train_result.get("error_type"):
            program_metrics["error_type"] = train_result["error_type"]
        if train_result.get("error"):
            program_metrics["error_message"] = train_result["error"]

    def _run_full_stage1_after_stage09(self, graph, config, seed):
        dev_str = config.device
        if dev_str == "cuda" and not torch.cuda.is_available():
            dev_str = "cpu"
        compile_t0 = time.perf_counter()
        rich_model = compile_model(
            [graph] * config.n_layers,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
        )
        compile_ms = (time.perf_counter() - compile_t0) * 1000.0
        return compile_ms, self._micro_train(
            model=rich_model,
            config=config,
            dev=torch.device(dev_str),
            seed=int(seed),
        )

    def _record_route_s09_to_s1(
        self,
        jr,
        graph,
        config,
        results: Dict,
        program_metrics: Dict[str, Any],
        s1_result: Dict,
        screening_seed: int,
        i: int,
    ) -> tuple:
        """Handle S0.9 routing: promote passing candidates to full S1.

        Returns (promoted_to_stage1: bool, s1_result: dict).
        """
        screening_stage = str(jr.payload.get("screening_stage") or "stage1")
        funnel = results.setdefault("funnel_counts", {})
        promoted_to_stage1 = screening_stage != "stage09"

        if screening_stage == "stage09":
            funnel["stage09_completed"] = int(funnel.get("stage09_completed", 0)) + 1
            stage09_passed = bool(s1_result.get("passed", False))
            program_metrics["stage09_passed"] = int(stage09_passed)
            program_metrics["stage09_loss_ratio"] = s1_result.get("loss_ratio")
            program_metrics["stage09_final_loss"] = s1_result.get("final_loss")
            program_metrics["stage09_total_train_time_ms"] = s1_result.get(
                "total_train_time_ms"
            )
            program_metrics["stage09_avg_step_time_ms"] = s1_result.get(
                "avg_step_time_ms"
            )
            if stage09_passed:
                results["stage09_passed"] = int(results.get("stage09_passed", 0)) + 1
                funnel["stage09_survived"] = int(funnel.get("stage09_survived", 0)) + 1
                try:
                    compile_ms, s1_result = self._run_full_stage1_after_stage09(
                        graph=graph,
                        config=config,
                        seed=screening_seed or (1000 + i),
                    )
                    program_metrics["stage09_promoted_to_s1"] = 1
                    program_metrics["compile_time_ms"] = (
                        float(program_metrics.get("compile_time_ms", 0.0) or 0.0)
                        + compile_ms
                    )
                    results.setdefault("_compile_times_ms", []).append(compile_ms)
                    promoted_to_stage1 = True
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.debug("Stage09->S1 promotion failed: %s", e)
                    s1_result = {
                        "passed": False,
                        "error_type": "stage1_promotion_failed",
                        "error": str(e),
                    }
            else:
                s1_result = {
                    "passed": False,
                    "error_type": s1_result.get("error_type") or "failed_stage09_gate",
                    "error": s1_result.get("error") or "failed_stage09_gate",
                }

        return promoted_to_stage1, s1_result

    def _record_baseline_comparisons(
        self,
        final_loss: Optional[float],
        s1_result: Dict,
        config,
        program_metrics: Dict[str, Any],
    ) -> None:
        """Run discovery, validation, and standard baseline comparisons.

        Updates program_metrics in-place with baseline ratio fields.
        """
        if final_loss is None:
            return

        try:
            baseline = self._get_baseline()
            baseline_steps = int(s1_result.get("n_train_steps") or config.stage1_steps)
            baseline_recipe = self._resolve_baseline_recipe(
                s1_result, default_lr=config.stage1_lr
            )

            # 1. Discovery Baseline (Random Tokens)
            discovery_loss = s1_result.get("discovery_loss")
            if discovery_loss is not None:
                try:
                    discovery_steps = min(5, baseline_steps // 10)
                    discovery_ratio = baseline.compare(
                        discovery_loss,
                        d_model=config.model_dim,
                        seq_len=min(128, config.max_seq_len),
                        n_steps=max(1, discovery_steps),
                        vocab_size=config.vocab_size,
                        batch_size=config.stage1_batch_size,
                        lr=baseline_recipe["lr"],
                        device=str(config.device),
                        n_layers=2,
                        data_mode="random",
                        data_tag="discovery_baseline",
                    )
                    program_metrics["discovery_baseline_ratio"] = discovery_ratio
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.debug("Discovery baseline failed: %s", e)

            # 2. Validation Baseline (Corpus)
            val_loss = s1_result.get("validation_loss")
            if val_loss is not None:
                try:
                    v_data_fn, v_data_tag, v_cache = self._make_baseline_data_fn(
                        config, split="val"
                    )
                    v_baseline_ratio = baseline.compare(
                        val_loss,
                        d_model=config.model_dim,
                        seq_len=min(128, config.max_seq_len),
                        n_steps=max(1, baseline_steps),
                        vocab_size=config.vocab_size,
                        batch_size=config.stage1_batch_size,
                        lr=baseline_recipe["lr"],
                        device=str(config.device),
                        n_layers=2,
                        data_fn=v_data_fn,
                        data_mode="corpus",
                        data_tag=v_data_tag,
                        cache_data_fn=v_cache,
                    )
                    program_metrics["validation_baseline_loss_ratio"] = v_baseline_ratio
                    program_metrics["validation_baseline_ratio"] = v_baseline_ratio
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.debug("Validation baseline comparison failed: %s", e)

            # 3. Standard Baseline (for backward compatibility / fallback)
            baseline_ratio = baseline.compare(
                final_loss,
                d_model=config.model_dim,
                seq_len=min(128, config.max_seq_len),
                n_steps=max(1, baseline_steps),
                vocab_size=config.vocab_size,
                batch_size=config.stage1_batch_size,
                lr=baseline_recipe["lr"],
                device=str(config.device),
                n_layers=2,
                data_mode="corpus" if val_loss is not None else "random",
                data_tag="standard_baseline",
            )
            program_metrics["baseline_loss_ratio"] = baseline_ratio
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug("Standard baseline comparison failed: %s", e)

    def _record_compute_novelty(
        self,
        s1_result: Dict,
        graph,
        config,
        nb,
        program_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute novelty score and extract behavioral fingerprint.

        Returns novelty_kwargs dict for record_program_result.
        """
        n_score = None
        nov = None

        try:
            fp = None
            fp_dict = s1_result.get("_behavioral_fingerprint")
            if fp_dict is not None:
                # Option B: reconstruct behavioral fingerprint from S1 worker
                fp = BehavioralFingerprint()
                for k, v in fp_dict.items():
                    if hasattr(fp, k):
                        setattr(fp, k, v)

                # Persist all behavioral fingerprint fields to DB
                program_metrics["fingerprint_json"] = json.dumps(
                    json_safe(fp.to_dict())
                )
                program_metrics["fp_interaction_locality"] = fp.interaction_locality
                program_metrics["fp_interaction_sparsity"] = fp.interaction_sparsity
                program_metrics["fp_interaction_symmetry"] = fp.interaction_symmetry
                program_metrics["fp_interaction_hierarchy"] = fp.interaction_hierarchy
                program_metrics["fp_intrinsic_dim"] = fp.intrinsic_dim
                program_metrics["fp_isotropy"] = fp.isotropy
                program_metrics["fp_rank_ratio"] = fp.rank_ratio
                program_metrics["fp_jacobian_spectral_norm"] = fp.jacobian_spectral_norm
                program_metrics["fp_jacobian_effective_rank"] = (
                    fp.jacobian_effective_rank
                )
                program_metrics["fp_sensitivity_uniformity"] = fp.sensitivity_uniformity
                program_metrics["fp_cka_vs_transformer"] = fp.cka_vs_transformer
                program_metrics["fp_cka_vs_ssm"] = fp.cka_vs_ssm
                program_metrics["fp_cka_vs_conv"] = fp.cka_vs_conv
                program_metrics["fp_hierarchy_fitness"] = fp.hierarchy_fitness
                program_metrics["fp_gromov_delta"] = fp.gromov_delta

                calibration_row = self._ensure_novelty_calibration(nb, config, fp)
                calibration = None
                if calibration_row:
                    calibration = {
                        "noise_floor_mean": calibration_row.get("noise_floor_mean"),
                        "noise_floor_std": calibration_row.get("noise_floor_std"),
                    }
                nov = novelty_score(graph, fingerprint=fp, calibration=calibration)
            else:
                # Option A fallback: structural-only novelty
                nov = novelty_score(graph)

            n_score = nov.overall_novelty
            novelty_valid, novelty_valid_reason, novelty_requires_justification = (
                self._resolve_novelty_promotion_validity(
                    config,
                    nov.novelty_valid_for_promotion,
                    nov.novelty_validity_reason,
                )
            )
            program_metrics["novelty_raw_score"] = nov.raw_novelty
            program_metrics["novelty_z_score"] = nov.novelty_z_score
            program_metrics["novelty_reference_version"] = (
                nov.novelty_reference_version
                or (fp.novelty_reference_version if fp is not None else None)
            )
            program_metrics["novelty_valid_for_promotion"] = int(novelty_valid)
            program_metrics["novelty_validity_reason"] = novelty_valid_reason
            program_metrics["novelty_requires_justification"] = int(
                novelty_requires_justification
            )
        except (ImportError, RuntimeError, ValueError, TypeError) as e:
            logger.debug(
                "Novelty scoring failed for %s: %s", graph.fingerprint()[:10], e
            )

        novelty_kwargs = {}
        if nov is not None:
            novelty_kwargs = dict(
                novelty_score=n_score,
                structural_novelty=nov.structural_novelty,
                behavioral_novelty=nov.behavioral_novelty,
                most_similar_to=nov.most_similar_to,
                novelty_confidence=nov.novelty_confidence,
            )
        return novelty_kwargs

    def _record_persist_result(
        self,
        nb,
        exp_id: str,
        graph,
        s1_result: Dict,
        s1_passed: bool,
        program_metrics: Dict[str, Any],
        novelty_kwargs: Dict[str, Any],
        results: Dict,
        loss_ratio,
        final_loss,
        throughput,
        training_curve,
    ) -> Optional[str]:
        """Persist result to notebook and store training curve.

        Returns result_id or None.
        """
        funnel = results.setdefault("funnel_counts", {})

        self._attach_ncd_metrics(graph, training_curve, program_metrics)
        rid = self._persist_program_row(
            nb=nb,
            exp_id=exp_id,
            graph=graph,
            s1_passed=s1_passed,
            program_metrics=program_metrics,
            novelty_kwargs=novelty_kwargs,
            final_loss=final_loss,
            loss_ratio=loss_ratio,
            throughput=throughput,
        )
        if rid:
            funnel["persisted_rows"] = int(funnel.get("persisted_rows", 0)) + 1
        else:
            funnel["dropped_persistence_quality_gate"] = (
                int(funnel.get("dropped_persistence_quality_gate", 0)) + 1
            )

        self._persist_training_curve_if_missing(nb, rid, training_curve)
        self._persist_screening_benchmark_payload(nb, rid, s1_result)
        return rid

    def _attach_ncd_metrics(
        self,
        graph,
        training_curve,
        program_metrics: Dict[str, Any],
    ) -> None:
        if not training_curve:
            return
        try:
            from ...eval.ncd import compute_graph_ncd

            graph_json_str = graph_to_json(graph)
            ncd_result = compute_graph_ncd(
                graph_json_str,
                training_curve,
                n_params=program_metrics.get("param_count"),
            )
            program_metrics["ncd_score"] = ncd_result["ncd_score"]
            program_metrics["ncd_description_length"] = ncd_result["description_length"]
            program_metrics["ncd_description_length_per_param"] = ncd_result[
                "description_length_per_param"
            ]
        except (ImportError, KeyError, TypeError, ValueError) as e:
            logger.debug("NCD computation failed: %s", e)

    def _persist_program_row(
        self,
        nb,
        exp_id: str,
        graph,
        s1_passed: bool,
        program_metrics: Dict[str, Any],
        novelty_kwargs: Dict[str, Any],
        final_loss,
        loss_ratio,
        throughput,
    ) -> Optional[str]:
        source_result_id = str(program_metrics.get("source_result_id") or "").strip()
        if (
            source_result_id
            and program_metrics.get("model_source") == "exact_graph_replay"
        ):
            nb.merge_program_result_patch(
                result_id=source_result_id,
                graph_fingerprint=graph.fingerprint(),
                graph_json=graph_to_json(graph),
                clear_failure_if_stage1=True,
                relabel_backfill_if_orphan=True,
                stage0_passed=True,
                stage05_passed=True,
                stage1_passed=s1_passed,
                final_loss=final_loss,
                loss_ratio=loss_ratio,
                throughput_tok_s=throughput,
                **novelty_kwargs,
                **program_metrics,
            )
            return source_result_id
        return nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=graph.fingerprint(),
            graph_json=graph_to_json(graph),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=s1_passed,
            final_loss=final_loss,
            loss_ratio=loss_ratio,
            throughput_tok_s=throughput,
            **novelty_kwargs,
            **program_metrics,
        )

    def _persist_training_curve_if_missing(self, nb, rid, training_curve) -> None:
        if not training_curve or not rid:
            return
        try:
            existing_curve = nb.conn.execute(
                "SELECT 1 FROM training_curves WHERE result_id = ? LIMIT 1",
                (rid,),
            ).fetchone()
            if existing_curve is None:
                nb.store_training_curve(rid, training_curve)
        except (OSError, RuntimeError) as e:
            logger.debug("store_training_curve failed for %s: %s", rid, e)

    def _persist_screening_benchmark_payload(self, nb, rid, s1_result: Dict) -> None:
        if not rid:
            return
        try:
            from ...eval.wikitext_eval import screening_wikitext_payload

            payload = screening_wikitext_payload(s1_result)
            if payload:
                nb.set_external_benchmarks(rid, payload)
        except (ImportError, OSError, ValueError) as e:
            logger.debug(
                "Screening benchmark payload persist failed for %s: %s", rid, e
            )

    def _record_leaderboard_and_best(
        self,
        nb,
        rid: Optional[str],
        graph,
        s1_passed: bool,
        program_metrics: Dict[str, Any],
        novelty_kwargs: Dict[str, Any],
        results: Dict,
        loss_ratio,
    ) -> None:
        """Upsert screening leaderboard entry and update best metrics."""
        if s1_passed and rid:
            nb.flush_writes()
            try:
                from ._helpers import _upsert_screening_entry

                _upsert_screening_entry(
                    nb,
                    {
                        "result_id": rid,
                        "model_source": program_metrics.get(
                            "model_source", "graph_synthesis"
                        ),
                        "graph_fingerprint": graph.fingerprint(),
                        "loss_ratio": loss_ratio,
                        "novelty_score": novelty_kwargs.get("novelty_score"),
                        "novelty_confidence": novelty_kwargs.get("novelty_confidence"),
                        "fp_jacobian_spectral_norm": program_metrics.get(
                            "fp_jacobian_spectral_norm"
                        ),
                        "routing_savings_ratio": program_metrics.get(
                            "routing_savings_ratio"
                        ),
                        "activation_sparsity_score": program_metrics.get(
                            "activation_sparsity_score"
                        ),
                        "depth_savings_ratio": program_metrics.get(
                            "depth_savings_ratio"
                        ),
                        "compression_ratio": program_metrics.get("compression_ratio"),
                        "wikitext_perplexity": program_metrics.get(
                            "wikitext_perplexity"
                        ),
                        "wikitext_score": program_metrics.get("wikitext_score"),
                    },
                )
            except (ImportError, OSError, RuntimeError) as e:
                logger.debug("Screening leaderboard upsert failed for %s: %s", rid, e)

        # Update best metrics in experiment summary
        if loss_ratio is not None:
            if (
                results["best_loss_ratio"] is None
                or loss_ratio < results["best_loss_ratio"]
            ):
                results["best_loss_ratio"] = loss_ratio

        try:
            nov = novelty_kwargs.get("novelty_score") or program_metrics.get(
                "novelty_score"
            )
            if nov is not None:
                if (
                    results["best_novelty_score"] is None
                    or nov > results["best_novelty_score"]
                ):
                    results["best_novelty_score"] = nov
        except (KeyError, TypeError) as e:
            logger.debug("Best novelty score update failed: %s", e)

    def _record_merge_metrics(
        self,
        jr,
        graph,
        s1_result: Dict,
        program_metrics: Dict[str, Any],
        results: Dict,
        config,
    ) -> None:
        """Merge training result metrics, efficiency, and perf traces."""
        self._merge_train_result_metrics(program_metrics, s1_result, config)
        self._merge_s1_telemetry(program_metrics, s1_result)

        from .synthesis import _graph_is_moe

        try:
            from ..leaderboard_scoring import compute_efficiency_multiple

            eff = compute_efficiency_multiple(
                loss_ratio=s1_result.get("loss_ratio"),
                param_count=program_metrics.get("param_count"),
                forward_time_ms=s1_result.get("forward_time_ms"),
                peak_memory_mb=s1_result.get("peak_memory_mb"),
                throughput_tok_s=s1_result.get("throughput"),
                is_moe=_graph_is_moe(graph) if graph else False,
            )
            if eff:
                program_metrics["efficiency_multiple"] = eff["geomean"]
        except (ImportError, TypeError, ValueError) as e:
            logger.debug("Efficiency multiple computation failed: %s", e)

        self._merge_perf_traces(jr, s1_result, program_metrics, results)

    def _merge_perf_traces(
        self,
        jr,
        s1_result: Dict,
        program_metrics: Dict[str, Any],
        results: Dict,
    ) -> None:
        """Merge perf/starvation/kernel traces into program_metrics and results."""
        perf_report = s1_result.get("perf_report", s1_result.get("perf_traces"))
        if perf_report:
            program_metrics["perf_report_json"] = json.dumps(json_safe(perf_report))
            results.setdefault("_perf_traces", []).append(perf_report)

        starvation_report = s1_result.get(
            "starvation_report", s1_result.get("gpu_starvation")
        )
        if starvation_report:
            program_metrics["starvation_report_json"] = json.dumps(
                json_safe(starvation_report)
            )
            results.setdefault("_gpu_starvation", []).append(starvation_report)

        kernel_timings = s1_result.get(
            "kernel_timings_ms", s1_result.get("kernel_timing")
        )
        if kernel_timings:
            program_metrics["kernel_timings_json"] = json.dumps(
                json_safe(kernel_timings)
            )
            results.setdefault("_kernel_timing", []).append(kernel_timings)

        if getattr(jr, "telemetry", None):
            program_metrics["queue_telemetry_json"] = json.dumps(
                json_safe(jr.telemetry)
            )

    def _run_diagnostic_suite_for_survivor(
        self, graph, config, program_metrics: Dict[str, Any]
    ) -> None:
        """Informational-only diagnostic suite run for S1 survivors."""
        try:
            diag_dev = str(config.device) if torch.cuda.is_available() else "cpu"
            diag_model = compile_model(
                [graph], vocab_size=config.vocab_size, max_seq_len=64
            )
            diag_result = run_diagnostic_suite(diag_model, device=diag_dev, n_steps=50)
            program_metrics["diagnostic_score"] = diag_result.diagnostic_score
            program_metrics["diagnostic_tasks_json"] = json.dumps(
                json_safe(diag_result.to_dict())
            )
        except (ImportError, RuntimeError, ValueError) as e:
            logger.debug(
                "Diagnostic suite failed for %s: %s", graph.fingerprint()[:10], e
            )

    def _emit_program_evaluated_event(
        self,
        i: int,
        graph,
        s1_passed: bool,
        loss_ratio,
        rid: Optional[str],
        throughput,
        program_metrics: Dict[str, Any],
    ) -> None:
        self._emit_event(
            "program_evaluated",
            {
                "index": i,
                "fingerprint": graph.fingerprint()[:10],
                "result": "pass" if s1_passed else "fail",
                "loss_ratio": f"{loss_ratio:.4f}" if loss_ratio is not None else None,
                "result_id": rid,
                "throughput": f"{throughput:.0f}" if throughput else None,
                "params": program_metrics.get("param_count"),
                "memory_mb": f"{program_metrics.get('peak_memory_mb', 0):.1f}"
                if program_metrics.get("peak_memory_mb")
                else None,
                "novelty": f"{program_metrics.get('novelty_score', 0):.3f}"
                if program_metrics.get("novelty_score") is not None
                else None,
            },
        )

    def _handle_s1_survivor(
        self,
        i: int,
        graph,
        config,
        final_loss,
        loss_ratio,
        s1_result: Dict,
        program_metrics: Dict[str, Any],
        results: Dict,
    ) -> None:
        """Survivor-specific bookkeeping: funnel, logs, baselines, diagnostics."""
        results["stage1_passed"] += 1
        funnel = results.setdefault("funnel_counts", {})
        funnel["stage1_survived"] = int(funnel.get("stage1_survived", 0)) + 1
        with self._lock:
            self._progress.stage1_passed += 1
        logger.info(
            "  ★ S1 SURVIVOR [%d] %s — loss_ratio=%.4f, params=%s",
            i + 1,
            graph.fingerprint()[:10],
            loss_ratio or 0,
            f"{program_metrics.get('param_count', 0):,}",
        )
        self._record_baseline_comparisons(
            final_loss, s1_result, config, program_metrics
        )
        self._run_diagnostic_suite_for_survivor(graph, config, program_metrics)

    def _record_orchestrator_result(self, jr, nb, exp_id, results, config):
        """Record a single result from the orchestrator into the notebook."""
        s1_result = jr.s1_result
        program_metrics = jr.payload["metrics"]
        graph = jr.payload["graph"]
        i = jr.index
        screening_seed = int(jr.payload.get("screening_seed") or 0)

        # Step 1: S0.9 routing
        promoted_to_stage1, s1_result = self._record_route_s09_to_s1(
            jr, graph, config, results, program_metrics, s1_result, screening_seed, i
        )

        funnel = results.setdefault("funnel_counts", {})
        if promoted_to_stage1:
            funnel["stage1_completed"] = int(funnel.get("stage1_completed", 0)) + 1

        s1_passed = s1_result.get("passed", False)
        loss_ratio = s1_result.get("loss_ratio")
        final_loss = s1_result.get("final_loss")
        throughput = s1_result.get("throughput")
        training_curve = s1_result.get("training_curve")

        # Step 2: Merge train metrics, efficiency, and perf traces
        self._record_merge_metrics(
            jr, graph, s1_result, program_metrics, results, config
        )

        # Step 3: S1 survivor bookkeeping + baselines + diagnostics
        if s1_passed:
            self._handle_s1_survivor(
                i,
                graph,
                config,
                final_loss,
                loss_ratio,
                s1_result,
                program_metrics,
                results,
            )

        # Step 4: Novelty scoring
        novelty_kwargs: Dict[str, Any] = {}
        if s1_passed:
            novelty_kwargs = self._record_compute_novelty(
                s1_result, graph, config, nb, program_metrics
            )

        # Step 5: Persist + leaderboard
        rid = self._record_persist_result(
            nb=nb,
            exp_id=exp_id,
            graph=graph,
            s1_result=s1_result,
            s1_passed=s1_passed,
            program_metrics=program_metrics,
            novelty_kwargs=novelty_kwargs,
            results=results,
            loss_ratio=loss_ratio,
            final_loss=final_loss,
            throughput=throughput,
            training_curve=training_curve,
        )
        self._record_leaderboard_and_best(
            nb=nb,
            rid=rid,
            graph=graph,
            s1_passed=s1_passed,
            program_metrics=program_metrics,
            novelty_kwargs=novelty_kwargs,
            results=results,
            loss_ratio=loss_ratio,
        )

        # Step 6: Emit event
        self._emit_program_evaluated_event(
            i, graph, s1_passed, loss_ratio, rid, throughput, program_metrics
        )

    # ── Pending selection-insight trial resolution ───────────────────────

    def _selection_insight_reward_investigate(
        self, entry: Dict[str, Any]
    ) -> Optional[float]:
        inv_pass = entry.get("investigation_passed")
        inv_loss = entry.get("investigation_loss_ratio")
        inv_rob = entry.get("investigation_robustness")
        if inv_pass is None and inv_loss is None:
            return None
        passed = 1.0 if bool(inv_pass) else 0.0
        loss_term = max(0.0, 1.0 - self._to_float(inv_loss, default=1.0))
        rob_term = max(0.0, min(1.0, self._to_float(inv_rob, default=0.0)))
        return max(0.0, min(1.0, 0.5 * passed + 0.3 * loss_term + 0.2 * rob_term))

    def _selection_insight_reward_validate(
        self, entry: Dict[str, Any]
    ) -> Optional[float]:
        val_pass = entry.get("validation_passed")
        val_loss = entry.get("validation_loss_ratio")
        val_base = entry.get("validation_baseline_ratio")
        val_std = self._to_float(entry.get("validation_multi_seed_std"), default=0.2)
        if val_pass is None and val_loss is None and val_base is None:
            return None
        passed = 1.0 if bool(val_pass) else 0.0
        if val_base is not None:
            loss_term = max(0.0, 1.0 - self._to_float(val_base, default=1.0))
        else:
            loss_term = max(0.0, 1.0 - self._to_float(val_loss, default=1.0))
        std_term = max(0.0, min(1.0, 1.0 - val_std))
        return max(0.0, min(1.0, 0.5 * passed + 0.3 * loss_term + 0.2 * std_term))

    def _selection_insight_trial_rewards(
        self, entries: List[Dict[str, Any]], context: str
    ) -> Optional[List[float]]:
        """Return per-entry realized rewards or None if any entry is unresolved."""
        if context == "auto_investigate_screening":
            reward_fn = self._selection_insight_reward_investigate
        elif context == "auto_validate_investigation":
            reward_fn = self._selection_insight_reward_validate
        else:
            return None
        rewards: List[float] = []
        for entry in entries:
            r = reward_fn(entry)
            if r is None:
                return None
            rewards.append(r)
        return rewards

    @staticmethod
    def _selection_trial_outcome(reward: float) -> str:
        if reward >= 0.55:
            return "supported"
        if reward <= 0.45:
            return "not_supported"
        return "inconclusive"

    @staticmethod
    def _selection_trial_leaderboard_rows(nb: LabNotebook) -> Dict[str, Dict[str, Any]]:
        rows = nb.conn.execute(
            """SELECT result_id, investigation_passed, investigation_loss_ratio,
                      investigation_robustness, validation_passed,
                      validation_loss_ratio, validation_baseline_ratio,
                      validation_multi_seed_std
               FROM leaderboard
               WHERE result_id IS NOT NULL"""
        ).fetchall()
        return {
            str(row["result_id"]): dict(row)
            for row in rows
            if row["result_id"] is not None
        }

    def _resolve_pending_selection_family_trials(self, nb: LabNotebook) -> None:
        """Resolve pending family trials against realized downstream outcomes."""
        try:
            trials = nb.get_pending_selection_family_trials(limit=200)
        except (OSError, RuntimeError) as e:
            logger.debug("Pending selection family trials fetch failed: %s", e)
            return
        if not trials:
            return

        by_result = self._selection_trial_leaderboard_rows(nb)
        for trial in trials:
            context = str(trial.get("context") or "")
            chosen_ids = trial.get("chosen_result_ids_json") or []
            if not isinstance(chosen_ids, list) or not chosen_ids:
                continue
            entries = [by_result.get(str(rid)) for rid in chosen_ids]
            if any(entry is None for entry in entries):
                continue
            rewards = self._selection_insight_trial_rewards(entries, context)
            if not rewards:
                continue
            reward = float(sum(rewards) / len(rewards))
            nb.resolve_selection_family_trial(
                trial_id=str(trial.get("trial_id")),
                reward=reward,
                outcome=self._selection_trial_outcome(reward),
                metadata={
                    "context": context,
                    "family": trial.get("family"),
                    "n_candidates": len(chosen_ids),
                    "resolved_from": "leaderboard",
                },
            )

    def _resolve_pending_selection_insight_trials(self, nb: LabNotebook) -> None:
        """Resolve pending insight-bundle trials once outcomes are available."""
        try:
            trials = nb.get_pending_selection_insight_trials(limit=200)
        except (OSError, RuntimeError) as e:
            logger.debug("Pending selection insight trials fetch failed: %s", e)
            return
        if not trials:
            return

        by_result = self._selection_trial_leaderboard_rows(nb)
        for trial in trials:
            context = str(trial.get("context") or "")
            chosen_ids = trial.get("chosen_result_ids_json") or []
            if not isinstance(chosen_ids, list) or not chosen_ids:
                continue
            entries = [by_result.get(str(rid)) for rid in chosen_ids]
            if any(entry is None for entry in entries):
                continue

            rewards = self._selection_insight_trial_rewards(entries, context)
            if not rewards:
                continue

            reward = float(sum(rewards) / len(rewards))
            outcome = self._selection_trial_outcome(reward)
            nb.resolve_selection_insight_trial(
                trial_id=str(trial.get("trial_id")),
                reward=reward,
                outcome=outcome,
                metadata={
                    "context": context,
                    "n_candidates": len(chosen_ids),
                    "resolved_from": "leaderboard",
                },
            )
            # Bayesian update: insights that predict well gain confidence
            try:
                trial_insight_ids = trial.get("insight_ids_json") or []
                if isinstance(trial_insight_ids, str):
                    trial_insight_ids = json.loads(trial_insight_ids)
                for insight_id in trial_insight_ids:
                    nb.update_insight_bayesian(
                        str(insight_id),
                        success=(outcome == "supported"),
                    )
            except (json.JSONDecodeError, OSError, TypeError) as e:
                logger.debug("Bayesian insight update failed: %s", e)
