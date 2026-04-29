"""Execution validation mixin — split from execution_validation."""

from __future__ import annotations

import json
import sqlite3


from ..json_utils import json_safe
from ..native_runner import compile_model_native_first as compile_model
from ._helpers import (
    clear_gpu_memory,
    screening_probe_fields,
    screening_wikitext_fields,
)
from ._types import RunConfig
from .execution_validation import _fail_loud
from ...eval.diagnostic_tasks import run_diagnostic_suite
from ...eval.fingerprint import compute_fingerprint
from ...eval.metrics import novelty_score
from ...synthesis.serializer import graph_from_json
from ...training.checkpointing import CheckpointManager

import logging

logger = logging.getLogger(__name__)


class _ExecutionValidationScaleMixin:
    """Scale-up: fetch/compile, train, metrics, baselines, evals, novelty, record."""

    __slots__ = ()

    def _scale_up_candidate(
        self,
        exp_id: str,
        source_result_id: str,
        prog_idx: int,
        total: int,
        config: RunConfig,
        scale_config: RunConfig,
        dev,
        dev_str: str,
        nb,
        results: dict,
    ) -> None:
        """Process a single scale-up candidate: fetch, compile, train, record."""
        result = self._scale_up_fetch_and_compile(
            exp_id=exp_id,
            source_result_id=source_result_id,
            prog_idx=prog_idx,
            total=total,
            config=config,
            nb=nb,
        )
        if result is None:
            return
        graph, model = result

        results["stage0_passed"] += 1
        results["stage05_passed"] += 1

        s1_result = self._scale_up_train(
            exp_id=exp_id,
            source_result_id=source_result_id,
            prog_idx=prog_idx,
            config=config,
            scale_config=scale_config,
            dev=dev,
            model=model,
        )

        program_metrics = self._extract_graph_metrics(graph)
        program_metrics["model_source"] = "graph_synthesis"

        s1_passed = s1_result.get("passed", False)
        loss_ratio = s1_result.get("loss_ratio")
        final_loss = s1_result.get("final_loss")
        throughput = s1_result.get("throughput")
        training_curve = s1_result.get("training_curve")

        self._scale_up_collect_training_metrics(program_metrics, s1_result, config)

        if s1_passed:
            results["stage1_passed"] += 1
            if final_loss is not None:
                self._scale_up_baseline_comparison(
                    program_metrics=program_metrics,
                    s1_result=s1_result,
                    final_loss=final_loss,
                    config=config,
                    dev_str=dev_str,
                    source_result_id=source_result_id,
                )

        program_metrics["stage_at_death"] = "survived" if s1_passed else "stage1"

        self._scale_up_evals(
            s1_passed=s1_passed,
            model=model,
            dev_str=dev_str,
            config=config,
            program_metrics=program_metrics,
            source_result_id=source_result_id,
        )

        n_score, nov = self._scale_up_novelty(
            s1_passed=s1_passed,
            model=model,
            graph=graph,
            config=config,
            dev_str=dev_str,
            nb=nb,
            program_metrics=program_metrics,
            source_result_id=source_result_id,
        )

        self._scale_up_record_result(
            exp_id=exp_id,
            source_result_id=source_result_id,
            prog_idx=prog_idx,
            total=total,
            config=config,
            nb=nb,
            results=results,
            graph=graph,
            model=model,
            s1_passed=s1_passed,
            loss_ratio=loss_ratio,
            final_loss=final_loss,
            throughput=throughput,
            training_curve=training_curve,
            n_score=n_score,
            nov=nov,
            program_metrics=program_metrics,
        )

    def _scale_up_fetch_and_compile(
        self,
        exp_id: str,
        source_result_id: str,
        prog_idx: int,
        total: int,
        config: RunConfig,
        nb,
    ) -> tuple | None:
        """Fetch source, deserialize graph, compile model.

        Returns (graph, model) or None if skipped.
        """
        source_program = nb.get_program_detail(source_result_id)
        if source_program is None:
            self._emit_event(
                "scale_up_progress",
                {
                    "experiment_id": exp_id,
                    "current_program": prog_idx + 1,
                    "total_programs": total,
                    "source_result_id": source_result_id,
                    "status": "skipped",
                    "error": "Source program not found",
                },
            )
            return None

        graph_json_str = source_program.get("graph_json")
        if not graph_json_str:
            raise RuntimeError(
                f"Scale-up source {source_result_id[:8]} has no graph_json"
            )

        try:
            graph = graph_from_json(graph_json_str)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self._emit_event(
                "scale_up_progress",
                {
                    "experiment_id": exp_id,
                    "current_program": prog_idx + 1,
                    "total_programs": total,
                    "source_result_id": source_result_id,
                    "status": "error",
                    "error": f"Graph deserialization failed: {e}",
                },
            )
            _fail_loud(
                "scale_up",
                f"graph deserialization failed for {source_result_id[:8]}",
                e,
            )

        try:
            layer_graphs = [graph] * config.n_layers
            model = compile_model(
                layer_graphs,
                vocab_size=config.vocab_size,
                max_seq_len=config.scale_up_seq_len,
            )
        except (RuntimeError, ValueError, TypeError) as e:
            self._emit_event(
                "scale_up_progress",
                {
                    "experiment_id": exp_id,
                    "current_program": prog_idx + 1,
                    "total_programs": total,
                    "source_result_id": source_result_id,
                    "status": "error",
                    "error": f"Compilation failed: {e}",
                },
            )
            _fail_loud(
                "scale_up",
                f"compilation failed for {source_result_id[:8]}",
                e,
            )

        return graph, model

    def _scale_up_train(
        self,
        exp_id: str,
        source_result_id: str,
        prog_idx: int,
        config: RunConfig,
        scale_config: RunConfig,
        dev,
        model,
    ) -> dict:
        """Run micro-training for a scale-up candidate with checkpoint support."""
        ckpt = CheckpointManager(checkpoint_dir=str(config.checkpoint_dir))
        resume_state = ckpt.load_phase(exp_id, "validation", prog_idx, 0)
        base_ctx = {"exp_id": exp_id, "phase": "scale_up"}
        self._live_training_context = {
            **base_ctx,
            "source_result_id": source_result_id,
            "candidate_index": prog_idx + 1,
            "total_candidates": 1,
            "training_program_index": 1,
            "total_training_programs": 1,
            "training_program_label": "scale-up",
            "run_kind": "validation",
            "checkpoint_manager": ckpt,
            "checkpoint_phase": "validation",
            "checkpoint_candidate_idx": prog_idx,
            "checkpoint_seed_idx": 0,
            "checkpoint_interval_steps": int(
                getattr(config, "phase_checkpoint_step_interval", 0) or 0
            ),
            "checkpoint_resume_state": (
                resume_state
                if resume_state and int(resume_state.get("step", 0) or 0) > 0
                else None
            ),
        }
        try:
            s1_result = self._micro_train(
                model,
                scale_config,
                dev,
                seed=self._stable_seed(exp_id, source_result_id, "scale_up"),
            )
        finally:
            self._live_training_context = base_ctx
        return s1_result

    def _scale_up_collect_training_metrics(
        self, program_metrics: dict, s1_result: dict, config: RunConfig
    ) -> None:
        """Copy training metrics from s1_result into program_metrics."""
        for key in [
            "initial_loss",
            "min_loss",
            "loss_improvement_rate",
            "avg_step_time_ms",
            "total_train_time_ms",
            "max_grad_norm",
            "mean_grad_norm",
            "grad_norm_std",
            "n_train_steps",
            "final_lr",
            "validation_loss",
            "validation_loss_ratio",
            "generalization_gap",
            "discovery_loss",
            "discovery_loss_ratio",
        ]:
            program_metrics[key] = s1_result.get(key)
        program_metrics["train_budget_steps"] = config.scale_up_steps
        program_metrics.update(screening_wikitext_fields(s1_result))
        program_metrics.update(screening_probe_fields(s1_result))
        program_metrics.update(screening_probe_fields(program_metrics))
        self._merge_s1_telemetry(program_metrics, s1_result)

    def _scale_up_baseline_comparison(
        self,
        program_metrics: dict,
        s1_result: dict,
        final_loss: float,
        config: RunConfig,
        dev_str: str,
        source_result_id: str,
    ) -> None:
        """Run baseline + val-split baseline comparisons for a scale-up candidate."""
        try:
            baseline = self._get_baseline()
            baseline_steps = int(
                s1_result.get("n_train_steps") or config.scale_up_steps
            )
            baseline_recipe = self._resolve_baseline_recipe(
                s1_result, default_lr=config.stage1_lr
            )
            bl_data_fn, bl_data_tag, bl_cache = self._make_baseline_data_fn(config)
            baseline_ratio = baseline.compare(
                final_loss,
                d_model=config.model_dim,
                seq_len=min(128, config.scale_up_seq_len),
                n_steps=max(1, baseline_steps),
                vocab_size=config.vocab_size,
                batch_size=config.scale_up_batch_size,
                lr=baseline_recipe["lr"],
                device=dev_str,
                n_layers=config.n_layers,
                optimizer_name=baseline_recipe["optimizer_name"],
                weight_decay=baseline_recipe["weight_decay"],
                momentum=baseline_recipe["momentum"],
                betas=baseline_recipe["betas"],
                data_fn=bl_data_fn,
                data_tag=bl_data_tag,
                cache_data_fn=bl_cache,
            )
            program_metrics["baseline_loss_ratio"] = baseline_ratio

            val_loss = s1_result.get("validation_loss")
            if val_loss is not None:
                v_data_fn, v_data_tag, v_cache = self._make_baseline_data_fn(
                    config, split="val"
                )
                v_baseline_ratio = baseline.compare(
                    val_loss,
                    d_model=config.model_dim,
                    seq_len=min(128, config.scale_up_seq_len),
                    n_steps=max(1, baseline_steps),
                    vocab_size=config.vocab_size,
                    batch_size=config.scale_up_batch_size,
                    lr=baseline_recipe["lr"],
                    device=dev_str,
                    n_layers=config.n_layers,
                    optimizer_name=baseline_recipe["optimizer_name"],
                    weight_decay=baseline_recipe["weight_decay"],
                    momentum=baseline_recipe["momentum"],
                    betas=baseline_recipe["betas"],
                    data_fn=v_data_fn,
                    data_tag=v_data_tag,
                    cache_data_fn=v_cache,
                )
                program_metrics["validation_baseline_loss_ratio"] = v_baseline_ratio
        except (RuntimeError, ValueError, TypeError) as exc:
            _fail_loud(
                "scale_up",
                f"baseline comparison failed for {source_result_id[:8]}",
                exc,
            )

    def _scale_up_evals(
        self,
        s1_passed: bool,
        model,
        dev_str: str,
        config: RunConfig,
        program_metrics: dict,
        source_result_id: str,
    ) -> None:
        """Run diagnostics + benchmark evals for scale-up survivors."""
        if s1_passed and model is not None:
            try:
                diag = run_diagnostic_suite(model, device=dev_str)
                program_metrics["diagnostic_tasks_json"] = json.dumps(
                    json_safe(diag.to_dict())
                )
                program_metrics["diagnostic_score"] = diag.diagnostic_score
            except (RuntimeError, ValueError) as exc:
                _fail_loud(
                    "scale_up",
                    f"diagnostic suite failed for {source_result_id[:8]}",
                    exc,
                )

        if s1_passed and model is not None:
            eval_seq_len = min(128, config.scale_up_seq_len)
            try:
                from ...eval.wikitext_eval import evaluate_wikitext_perplexity

                wt_result = evaluate_wikitext_perplexity(
                    model,
                    config.vocab_size,
                    dev_str,
                    n_train_steps=200,
                    seq_len=eval_seq_len,
                )
                program_metrics["wikitext_perplexity"] = wt_result.get(
                    "wikitext_perplexity"
                )
                program_metrics["wikitext_score"] = wt_result.get("wikitext_score")
                if program_metrics.get("wikitext_perplexity") is not None:
                    logger.info(
                        "Scale-up WikiText ppl=%.1f score=%.3f",
                        program_metrics["wikitext_perplexity"],
                        program_metrics.get("wikitext_score") or 0,
                    )
            except (ImportError, RuntimeError, ValueError) as e:
                logger.debug("Scale-up WikiText eval skipped: %s", e)
            try:
                from ...eval.tinystories_eval import evaluate_tinystories

                ts_result = evaluate_tinystories(
                    model,
                    config.vocab_size,
                    dev_str,
                    n_train_steps=200,
                    seq_len=eval_seq_len,
                )
                program_metrics["tinystories_perplexity"] = ts_result.get(
                    "tinystories_perplexity"
                )
                program_metrics["tinystories_score"] = ts_result.get(
                    "tinystories_score"
                )
                if program_metrics.get("tinystories_perplexity") is not None:
                    logger.info(
                        "Scale-up TinyStories ppl=%.1f score=%.3f",
                        program_metrics["tinystories_perplexity"],
                        program_metrics.get("tinystories_score") or 0,
                    )
            except (ImportError, RuntimeError, ValueError) as e:
                logger.debug("Scale-up TinyStories eval skipped: %s", e)

    def _scale_up_novelty(
        self,
        s1_passed: bool,
        model,
        graph,
        config: RunConfig,
        dev_str: str,
        nb,
        program_metrics: dict,
        source_result_id: str,
    ) -> tuple:
        """Compute fingerprint + novelty score; return (n_score, nov)."""
        fp = None
        calibration_row = None
        if s1_passed and model is not None:
            try:
                fp = compute_fingerprint(
                    model,
                    seq_len=min(64, config.scale_up_seq_len),
                    model_dim=config.model_dim,
                    vocab_size=config.vocab_size,
                    device=dev_str,
                )
                program_metrics["cka_source"] = fp.cka_source
                program_metrics["cka_artifact_version"] = fp.cka_artifact_version
                program_metrics["cka_probe_protocol_hash"] = fp.cka_probe_protocol_hash
                program_metrics["cka_reference_quality"] = fp.cka_reference_quality
                calibration_row = self._ensure_novelty_calibration(nb, config, fp)
            except (RuntimeError, ValueError, TypeError) as exc:
                _fail_loud(
                    "scale_up",
                    f"fingerprint computation failed for {source_result_id[:8]}",
                    exc,
                )

        calibration = None
        if calibration_row:
            calibration = {
                "noise_floor_mean": calibration_row.get("noise_floor_mean"),
                "noise_floor_std": calibration_row.get("noise_floor_std"),
            }
        nov = novelty_score(graph, fingerprint=fp, calibration=calibration)
        n_score = nov.overall_novelty

        program_metrics["novelty_raw_score"] = nov.raw_novelty
        program_metrics["novelty_z_score"] = nov.novelty_z_score
        program_metrics["novelty_reference_version"] = (
            nov.novelty_reference_version
            or (fp.novelty_reference_version if fp is not None else None)
        )

        return n_score, nov

    def _scale_up_record_result(
        self,
        exp_id: str,
        source_result_id: str,
        prog_idx: int,
        total: int,
        config: RunConfig,
        nb,
        results: dict,
        graph,
        model,
        s1_passed: bool,
        loss_ratio: float | None,
        final_loss: float | None,
        throughput: float | None,
        training_curve: list | None,
        n_score: float,
        nov,
        program_metrics: dict,
    ) -> None:
        """Resolve novelty validity, update results, persist to notebook."""
        novelty_valid, novelty_valid_reason, novelty_requires_justification = (
            self._resolve_novelty_promotion_validity(
                config,
                nov.novelty_valid_for_promotion,
                nov.novelty_validity_reason,
            )
        )
        program_metrics["novelty_valid_for_promotion"] = int(novelty_valid)
        program_metrics["novelty_validity_reason"] = novelty_valid_reason
        program_metrics["novelty_requires_justification"] = int(
            novelty_requires_justification
        )

        if s1_passed and n_score > 0.5:
            results["novel_count"] += 1
            results["survivors"].append(
                {
                    "fingerprint": graph.fingerprint(),
                    "novelty": n_score,
                    "loss_ratio": loss_ratio,
                    "novelty_valid_for_promotion": novelty_valid,
                }
            )

        if loss_ratio and (
            results["best_loss_ratio"] is None
            or loss_ratio < results["best_loss_ratio"]
        ):
            results["best_loss_ratio"] = loss_ratio
        if n_score and (
            results["best_novelty_score"] is None
            or n_score > results["best_novelty_score"]
        ):
            results["best_novelty_score"] = n_score

        result_id = source_result_id

        if training_curve and result_id:
            try:
                existing_curve = nb.conn.execute(
                    "SELECT 1 FROM training_curves WHERE result_id = ? LIMIT 1",
                    (result_id,),
                ).fetchone()
                if existing_curve is None:
                    nb.store_training_curve(result_id, training_curve)
            except (sqlite3.OperationalError, RuntimeError) as exc:
                _fail_loud(
                    "scale_up",
                    f"training curve persistence failed for {result_id[:8]}",
                    exc,
                )

        self._emit_event(
            "scale_up_progress",
            {
                "experiment_id": exp_id,
                "current_program": prog_idx + 1,
                "total_programs": total,
                "source_result_id": source_result_id,
                "status": "completed",
                "passed": s1_passed,
                "loss_ratio": round(loss_ratio, 4) if loss_ratio else None,
                "final_loss": round(final_loss, 4) if final_loss else None,
            },
        )

        del model
        clear_gpu_memory()
