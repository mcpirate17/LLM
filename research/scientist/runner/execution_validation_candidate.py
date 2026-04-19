"""Execution validation mixin — split from execution_validation."""

from __future__ import annotations

import json
import sqlite3
import time
import traceback
from typing import List

import torch

from ..json_utils import json_safe
from ..native_runner import compile_model_native_first as compile_model
from ..runtime_events import publish_lifecycle_event
from ..shared_utils import resolve_device
from ._helpers import (
    build_validation_entry,
    clear_gpu_memory,
    compute_seed_metrics,
    handle_breakthrough,
    promote_validation_candidate,
    run_baseline_comparison,
    run_trajectory_probe,
    screening_probe_fields,
    screening_wikitext_fields,
)
from ._types import RunConfig
from .execution_validation import _fail_loud
from ...eval.diagnostic_tasks import run_diagnostic_suite
from ...eval.fingerprint import compute_fingerprint
from ...eval.metrics import novelty_score
from ...synthesis.serializer import graph_from_json, graph_to_json
from ...training.checkpointing import CheckpointManager

import logging

logger = logging.getLogger(__name__)


class _ExecutionValidationCandidateMixin:
    """Per-candidate validation: CKA, record, promote, baselines, checkpoint."""

    __slots__ = ()

    def _run_single_validation_candidate(
        self,
        exp_id: str,
        source_result_id: str,
        prog_idx: int,
        result_ids: List[str],
        config: RunConfig,
        val_config,
        dev,
        dev_str: str,
        nb,
        source_map: dict,
        results: dict,
        vstatus,
        ckpt,
        t_start: float,
    ) -> None:
        """Process one validation candidate: progress, CKA, seed sweep, record."""
        self._update_progress(
            current_program=prog_idx + 1,
            status="validating",
            aria_message=(
                f"Validating {prog_idx + 1}/{len(result_ids)}: "
                f"{source_result_id[:8]}... "
                f"({config.validation_n_seeds} seeds, "
                f"{config.validation_steps} steps)"
            ),
            elapsed_seconds=time.time() - t_start,
        )

        self._emit_event(
            "validation_progress",
            {
                "experiment_id": exp_id,
                "current": prog_idx + 1,
                "total": len(result_ids),
                "source_result_id": source_result_id,
                "status": "starting",
            },
        )

        source = source_map.get(source_result_id)
        if source is None:
            return

        graph_json_str = source.get("graph_json")
        arch_spec_json_str = source.get("arch_spec_json")
        model_source = source.get("model_source") or "graph_synthesis"

        best_tp_json = self._get_validation_best_training_json(nb, source_result_id)

        _novelty_cap = self._validation_cka_check(
            source=source,
            source_result_id=source_result_id,
            model_source=model_source,
            arch_spec_json_str=arch_spec_json_str,
            graph_json_str=graph_json_str,
            config=config,
            dev=dev,
        )

        seed_results = self._run_validation_seed_sweep(
            exp_id=exp_id,
            source_result_id=source_result_id,
            model_source=model_source,
            arch_spec_json_str=arch_spec_json_str,
            graph_json_str=graph_json_str,
            config=config,
            val_config=val_config,
            dev=dev,
            best_tp_json=best_tp_json,
            progress_payload={
                "experiment_id": exp_id,
                "current": prog_idx + 1,
                "total": len(result_ids),
                "source_result_id": source_result_id,
            },
        )

        if not seed_results:
            raise RuntimeError(
                f"Validation aborted for {source_result_id[:8]}: "
                f"model failed to reconstruct for all "
                f"{config.validation_n_seeds} seeds"
            )

        self._record_validation_candidate(
            seed_results=seed_results,
            source=source,
            source_result_id=source_result_id,
            model_source=model_source,
            arch_spec_json_str=arch_spec_json_str,
            graph_json_str=graph_json_str,
            config=config,
            dev=dev,
            dev_str=dev_str,
            nb=nb,
            exp_id=exp_id,
            results=results,
            novelty_cap=_novelty_cap,
            vstatus=vstatus,
            ckpt=ckpt,
            prog_idx=prog_idx,
        )

    def _validation_cka_check(
        self,
        source: dict,
        source_result_id: str,
        model_source: str,
        arch_spec_json_str: str | None,
        graph_json_str: str | None,
        config: RunConfig,
        dev,
    ) -> float | None:
        """B3: Validate artifact-backed CKA; return novelty cap or None."""
        _fp_data = source.get("_behavioral_fingerprint") or {}
        _cka_src = _fp_data.get("cka_source", "unknown")
        if _cka_src == "artifact":
            return None

        logger.info(
            "validation_cka_check: result_id=%s cka_source=%s "
            "— attempting fingerprint completion",
            source_result_id[:12],
            _cka_src,
        )
        try:
            from ...eval.fingerprint import BehavioralFingerprint
            from ...eval.fingerprint_runtime import (
                complete_fingerprint_post_investigation,
            )

            _fp_fields = {
                k: v
                for k, v in _fp_data.items()
                if k
                in {f.name for f in BehavioralFingerprint.__dataclass_fields__.values()}
            }
            if not _fp_fields:
                return 0.5

            _fp = BehavioralFingerprint(**_fp_fields)
            _tmp_model = self._build_model_from_source(
                model_source,
                arch_spec_json_str,
                graph_json_str,
                config,
                seq_len_override=min(64, config.validation_seq_len),
            )
            if _tmp_model is None:
                logger.warning(
                    "validation_cka_model_build_failed: result_id=%s "
                    "— capping novelty at 50%%",
                    source_result_id[:12],
                )
                return 0.5

            _fp = complete_fingerprint_post_investigation(
                _fp,
                _tmp_model,
                seq_len=min(64, config.validation_seq_len),
                model_dim=config.model_dim,
                vocab_size=config.vocab_size,
                device=str(dev),
            )
            del _tmp_model
            clear_gpu_memory()

            if _fp.cka_source == "artifact":
                source["_behavioral_fingerprint"] = _fp.to_dict()
                logger.info(
                    "validation_cka_completed: result_id=%s cka_source=artifact",
                    source_result_id[:12],
                )
                return None

            logger.warning(
                "validation_cka_still_missing: result_id=%s "
                "cka_source=%s — capping novelty at 50%%",
                source_result_id[:12],
                _fp.cka_source,
            )
            return 0.5
        except (RuntimeError, ValueError, TypeError, ImportError) as e:
            logger.warning(
                "validation_cka_attempt_failed: result_id=%s error=%s "
                "— capping novelty at 50%%",
                source_result_id[:12],
                str(e),
            )
            return 0.5

    def _record_validation_candidate(
        self,
        seed_results: list,
        source: dict,
        source_result_id: str,
        model_source: str,
        arch_spec_json_str: str | None,
        graph_json_str: str | None,
        config: RunConfig,
        dev,
        dev_str: str,
        nb,
        exp_id: str,
        results: dict,
        novelty_cap: float | None,
        vstatus,
        ckpt,
        prog_idx: int,
    ) -> None:
        """Compute metrics, record result, promote, and checkpoint."""
        _sm = compute_seed_metrics(seed_results)
        passed_seeds = _sm["passed_seeds"]
        loss_ratios = _sm["loss_ratios"]
        val_loss_ratio = _sm["val_loss_ratio"]
        multi_seed_std = _sm["multi_seed_std"]
        robustness_score = _sm["robustness_score"]
        is_unstable = _sm["is_unstable"]
        init_sensitivity_std = _sm["init_sensitivity_std"]
        best_seed = _sm["best_seed"]

        _rid_short = source_result_id[:8]

        def _compare(loss, **kw):
            return run_baseline_comparison(
                get_baseline=self._get_baseline,
                resolve_recipe=self._resolve_baseline_recipe,
                make_data_fn=self._make_baseline_data_fn,
                candidate_loss=loss,
                train_result=best_seed,
                config=config,
                dev_str=dev_str,
                **kw,
            )

        (
            val_baseline_ratio,
            val_normalized_ratio,
            val_param_efficiency,
            val_split_ratio,
        ) = self._validation_baseline_comparisons(
            source=source,
            source_result_id=source_result_id,
            best_seed=best_seed,
            loss_ratios=loss_ratios,
            config=config,
            _compare=_compare,
            vstatus=vstatus,
            rid_short=_rid_short,
        )
        if len(passed_seeds) > 0:
            results["stage1_passed"] += 1
        results["stage0_passed"] += 1
        results["stage05_passed"] += 1

        source_params = int(
            (source.get("param_count") or source.get("graph_n_params_estimate") or 0)
            if source
            else 0
        )

        vstatus("external evals", _rid_short)
        ev_res = self._run_external_evals(
            config=config,
            dev=dev,
            dev_str=dev_str,
            best_seed=best_seed,
            model_source=model_source,
            arch_spec_json_str=arch_spec_json_str,
            graph_json_str=graph_json_str,
            source=source,
            source_result_id=source_result_id,
            exp_id=exp_id,
            val_loss_ratio=val_loss_ratio,
            val_baseline_ratio=val_baseline_ratio,
            val_normalized_ratio=val_normalized_ratio,
            multi_seed_std=multi_seed_std,
            passed_seeds=passed_seeds,
            source_params=source_params,
        )

        nov_conf = source.get("novelty_confidence", 0) if source else 0

        from ._types import ValidationMetrics

        _metrics = ValidationMetrics(
            val_loss_ratio=val_loss_ratio,
            multi_seed_std=multi_seed_std,
            robustness_score=robustness_score,
            is_unstable=is_unstable,
            init_sensitivity_std=init_sensitivity_std,
            val_baseline_ratio=val_baseline_ratio,
            val_normalized_ratio=val_normalized_ratio,
            val_param_efficiency=val_param_efficiency,
            validation_baseline_loss_ratio=val_split_ratio,
            passed_seeds=passed_seeds,
            best_seed=best_seed,
            source_params=int(source_params),
        )

        self._validation_promote_and_record(
            source=source,
            source_result_id=source_result_id,
            model_source=model_source,
            arch_spec_json_str=arch_spec_json_str,
            graph_json_str=graph_json_str,
            config=config,
            dev=dev,
            dev_str=dev_str,
            nb=nb,
            exp_id=exp_id,
            results=results,
            novelty_cap=novelty_cap,
            vstatus=vstatus,
            ckpt=ckpt,
            prog_idx=prog_idx,
            _metrics=_metrics,
            ev_res=ev_res,
            nov_conf=nov_conf,
            val_loss_ratio=val_loss_ratio,
            val_baseline_ratio=val_baseline_ratio,
            multi_seed_std=multi_seed_std,
            passed_seeds=passed_seeds,
            rid_short=_rid_short,
        )

    def _validation_promote_and_record(
        self,
        source: dict,
        source_result_id: str,
        model_source: str,
        arch_spec_json_str: str | None,
        graph_json_str: str | None,
        config: RunConfig,
        dev,
        dev_str: str,
        nb,
        exp_id: str,
        results: dict,
        novelty_cap: float | None,
        vstatus,
        ckpt,
        prog_idx: int,
        _metrics,
        ev_res,
        nov_conf: float,
        val_loss_ratio: float | None,
        val_baseline_ratio: float | None,
        multi_seed_std: float,
        passed_seeds: list,
        rid_short: str,
    ) -> None:
        """Build validation entry, promote, run trajectory probe, record + checkpoint."""
        validation_entry = build_validation_entry(
            source_result_id=source_result_id,
            metrics=_metrics,
            ev_res=ev_res,
            nov_conf=nov_conf,
            config=config,
        )
        tier = "breakthrough" if ev_res.is_breakthrough else "validation"
        results["validation_results"].append(validation_entry.to_dict())

        if val_loss_ratio and (
            results["best_loss_ratio"] is None
            or val_loss_ratio < results["best_loss_ratio"]
        ):
            results["best_loss_ratio"] = val_loss_ratio
        source_novelty = source.get("novelty_score")
        if source_novelty is not None and (
            results["best_novelty_score"] is None
            or source_novelty > results["best_novelty_score"]
        ):
            results["best_novelty_score"] = source_novelty

        vstatus("leaderboard promotion", rid_short)
        promote_validation_candidate(
            nb=nb,
            source_result_id=source_result_id,
            source=source,
            tier=tier,
            metrics=_metrics,
            ev_res=ev_res,
            novelty_cap=novelty_cap,
        )

        vstatus("trajectory probe (4000 steps)", rid_short)
        trajectory_composite = run_trajectory_probe(
            graph_json_str=graph_json_str,
            config=config,
            dev=dev,
            dev_str=dev_str,
            nb=nb,
            source_result_id=source_result_id,
            tier=tier,
            passed_seeds=passed_seeds,
        )

        handle_breakthrough(
            is_breakthrough=ev_res.is_breakthrough,
            trajectory_composite=trajectory_composite,
            aria=self.aria,
            nb=nb,
            exp_id=exp_id,
            source_result_id=source_result_id,
            source=source,
            validation_entry=validation_entry,
            val_loss_ratio=val_loss_ratio,
            val_baseline_ratio=val_baseline_ratio,
            multi_seed_std=multi_seed_std,
            emit_event=self._emit_event,
        )

        self._validation_record_and_checkpoint(
            source=source,
            source_result_id=source_result_id,
            model_source=model_source,
            arch_spec_json_str=arch_spec_json_str,
            graph_json_str=graph_json_str,
            nb=nb,
            exp_id=exp_id,
            passed_seeds=passed_seeds,
            val_loss_ratio=val_loss_ratio,
            val_baseline_ratio=val_baseline_ratio,
            novelty_cap=novelty_cap,
            ckpt=ckpt,
            prog_idx=prog_idx,
        )

    def _validation_baseline_comparisons(
        self,
        source: dict,
        source_result_id: str,
        best_seed: dict | None,
        loss_ratios: list,
        config: RunConfig,
        _compare,
        vstatus,
        rid_short: str,
    ) -> tuple:
        """Run baseline + normalized baseline comparisons.

        Returns ``(val_baseline_ratio, val_normalized_ratio,
        val_param_efficiency, val_split_ratio)``. The fourth element is the
        per-validation-split ratio computed from ``best_seed.validation_loss``
        (``None`` when the seed has no recorded validation loss); callers
        thread it into their own ``program_metrics`` dict for persistence.
        """
        vstatus("baseline comparison", rid_short)
        val_baseline_ratio = None
        val_split_ratio = None
        if best_seed is not None:
            try:
                val_baseline_ratio = _compare(best_seed["final_loss"])
                v_loss = best_seed.get("validation_loss")
                if v_loss is not None:
                    val_split_ratio = _compare(v_loss, split="val")
            except (RuntimeError, ValueError, TypeError) as exc:
                _fail_loud(
                    "validation",
                    f"baseline comparison failed for {source_result_id[:8]}",
                    exc,
                )

        vstatus("normalized baseline comparison", rid_short)
        val_normalized_ratio = None
        val_param_efficiency = None
        source_params = int(
            (source.get("param_count") or source.get("graph_n_params_estimate") or 0)
            if source
            else 0
        )
        if loss_ratios and best_seed is not None and source_params > 0:
            try:
                norm_result = _compare(
                    best_seed["final_loss"],
                    normalized=True,
                    program_params=source_params,
                )
                val_normalized_ratio = norm_result.get("normalized_ratio")
                val_param_efficiency = norm_result.get("param_efficiency")
            except (RuntimeError, ValueError, TypeError) as exc:
                _fail_loud(
                    "validation",
                    f"normalized baseline comparison failed for {source_result_id[:8]}",
                    exc,
                )

        return val_baseline_ratio, val_normalized_ratio, val_param_efficiency, val_split_ratio

    def _validation_record_and_checkpoint(
        self,
        source: dict,
        source_result_id: str,
        model_source: str,
        arch_spec_json_str: str | None,
        graph_json_str: str | None,
        nb,
        exp_id: str,
        passed_seeds: list,
        val_loss_ratio: float | None,
        val_baseline_ratio: float | None,
        novelty_cap: float | None,
        ckpt,
        prog_idx: int,
    ) -> None:
        """Record program result and save phase checkpoint."""
        _raw_novelty = source.get("novelty_score")
        _raw_confidence = source.get("novelty_confidence")
        if novelty_cap is not None:
            if _raw_novelty is not None:
                _raw_novelty = float(_raw_novelty) * novelty_cap
            if _raw_confidence is not None:
                _raw_confidence = float(_raw_confidence) * novelty_cap

        nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=source.get("graph_fingerprint", source_result_id),
            graph_json=graph_json_str or "{}",
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=len(passed_seeds) > 0,
            loss_ratio=val_loss_ratio,
            baseline_loss_ratio=val_baseline_ratio,
            novelty_score=_raw_novelty,
            novelty_confidence=_raw_confidence,
            novelty_raw_score=source.get("novelty_raw_score"),
            novelty_z_score=source.get("novelty_z_score"),
            novelty_reference_version=source.get("novelty_reference_version"),
            novelty_valid_for_promotion=source.get("novelty_valid_for_promotion"),
            novelty_validity_reason=source.get("novelty_validity_reason"),
            novelty_requires_justification=source.get("novelty_requires_justification"),
            model_source=model_source,
            arch_spec_json=arch_spec_json_str,
        )

        try:
            ckpt.save_phase(
                experiment_id=exp_id,
                phase="validation",
                candidate_idx=prog_idx + 1,
                seed_idx=0,
                model_state_dict={},
                optimizer_state_dict={},
                step=0,
                metrics={"completed_candidate": prog_idx},
            )
            ckpt.save_phase(
                experiment_id=exp_id,
                phase="validation",
                candidate_idx=-1,
                seed_idx=0,
                model_state_dict={},
                optimizer_state_dict={},
                step=0,
                metrics={"candidate_idx": prog_idx + 1},
            )
        except (OSError, RuntimeError) as e:
            _fail_loud(
                "validation",
                f"checkpoint save failed for candidate {prog_idx + 1}",
                e,
            )

    # ── Extracted helpers for _run_scale_up_thread ──

