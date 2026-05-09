"""Champion confirmation milestone evaluation helpers."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

from ..json_utils import json_safe
from ..native_runner import compile_model_native_first as compile_model
from ._helpers import clear_gpu_memory
from ._types import RunConfig

logger = logging.getLogger(__name__)


class ChampionConfirmationEvaluator:
    """Run champion-only hard probes without bloating scale-up orchestration."""

    __slots__ = ("owner",)

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.owner, name)

    def _scale_up_milestone_artifact_path(
        self,
        config: RunConfig,
        exp_id: str,
        source_result_id: str,
        prog_idx: int,
        step: int,
    ) -> Path:
        return (
            Path(str(config.checkpoint_dir))
            / "_investigation_artifacts"
            / exp_id
            / f"{source_result_id}_tp{prog_idx}_scale_up_step{int(step)}.pt"
        )

    def _scale_up_missing_milestone_snapshot(
        self,
        *,
        exp_id: str,
        source_result_id: str,
        prog_idx: int,
        step: int,
        path: Path,
    ) -> dict:
        self._scale_up_emit_champion_probe_event(
            {
                "experiment_id": exp_id,
                "source_result_id": source_result_id,
                "candidate_index": prog_idx + 1,
                "step": int(step),
                "path": str(path),
            },
            probe="milestone_checkpoint",
            status="missing_checkpoint",
        )
        logger.warning(
            "Champion milestone checkpoint missing: result=%s step=%s path=%s",
            source_result_id[:8],
            step,
            path,
        )
        return {"step": step, "status": "missing_checkpoint", "path": str(path)}

    def _scale_up_finalize_champion_milestones(
        self,
        *,
        program_metrics: dict,
        payloads: list[dict],
        milestones: list[int],
        exp_id: str,
        source_result_id: str,
        prog_idx: int,
    ) -> None:
        self._scale_up_apply_champion_id_collapse(payloads)
        if not payloads:
            return
        for item in payloads:
            item.pop("_id_hidden_snapshot", None)
        program_metrics["external_benchmarks_json"] = json.dumps(
            json_safe({"champion_confirmation_milestones": payloads}),
            sort_keys=True,
        )
        completed = [
            item
            for item in payloads
            if item.get("status") in (None, "ok", "partial")
            and int(item.get("step") or 0) > 0
        ]
        if completed:
            final_snapshot = max(completed, key=lambda item: int(item["step"]))
            self._scale_up_apply_champion_snapshot(program_metrics, final_snapshot)
        self._scale_up_emit_champion_probe_event(
            {
                "experiment_id": exp_id,
                "source_result_id": source_result_id,
                "candidate_index": prog_idx + 1,
                "completed_milestones": len(completed),
                "total_milestones": len(milestones),
            },
            probe="milestones",
            status="completed",
        )

    def _scale_up_champion_milestone_evals(
        self,
        exp_id: str,
        source_result_id: str,
        prog_idx: int,
        graph,
        config: RunConfig,
        dev,
        dev_str: str,
        s1_passed: bool,
        program_metrics: dict,
    ) -> None:
        """Run champion-only hard probes from saved final and floor checkpoints."""
        is_confirmation = str(getattr(config, "mode", "") or "") == "confirmation"
        if not (is_confirmation and s1_passed):
            return
        probe_device = torch.device(dev)
        if probe_device.type == "cpu" and not bool(
            getattr(config, "allow_champion_cpu_probes", False)
        ):
            snapshot = {
                "experiment_id": exp_id,
                "source_result_id": source_result_id,
                "candidate_index": prog_idx + 1,
                "status": "missing_accelerator",
                "device": str(probe_device),
            }
            program_metrics["external_benchmarks_json"] = json.dumps(
                json_safe({"champion_confirmation_milestones": [snapshot]}),
                sort_keys=True,
            )
            self._scale_up_emit_champion_probe_event(
                snapshot,
                probe="milestones",
                status="missing_accelerator",
                error="champion confirmation probes require an accelerator",
            )
            logger.warning(
                "Champion confirmation probes skipped on CPU: result=%s",
                source_result_id[:8],
            )
            return
        milestones = {
            step
            for step in (10_000, 20_000, 40_000)
            if step <= int(config.scale_up_steps)
        }
        floor_step = program_metrics.get("champion_steps_to_floor")
        interval = int(
            getattr(config, "champion_floor_checkpoint_interval_steps", 0) or 0
        )
        if floor_step is not None and interval > 0:
            rounded_floor = int(round(float(floor_step) / float(interval)) * interval)
            rounded_floor = max(
                interval, min(int(config.scale_up_steps), rounded_floor)
            )
            milestones.add(rounded_floor)
        milestones = sorted(milestones)
        payloads: list[dict] = []
        self._scale_up_emit_champion_probe_event(
            {
                "experiment_id": exp_id,
                "source_result_id": source_result_id,
                "candidate_index": prog_idx + 1,
                "total_milestones": len(milestones),
            },
            probe="milestones",
            status="started",
        )
        for step in milestones:
            if (
                getattr(self, "_stop_event", None) is not None
                and self._stop_event.is_set()
            ):
                break
            path = self._scale_up_milestone_artifact_path(
                config, exp_id, source_result_id, prog_idx, step
            )
            if not path.exists():
                payloads.append(
                    self._scale_up_missing_milestone_snapshot(
                        exp_id=exp_id,
                        source_result_id=source_result_id,
                        prog_idx=prog_idx,
                        step=step,
                        path=path,
                    )
                )
                continue
            snapshot = self._scale_up_eval_checkpoint_snapshot(
                graph=graph,
                checkpoint_path=path,
                step=step,
                config=config,
                dev=dev,
                dev_str=dev_str,
                exp_id=exp_id,
                source_result_id=source_result_id,
                prog_idx=prog_idx,
            )
            payloads.append(snapshot)

        self._scale_up_finalize_champion_milestones(
            program_metrics=program_metrics,
            payloads=payloads,
            milestones=milestones,
            exp_id=exp_id,
            source_result_id=source_result_id,
            prog_idx=prog_idx,
        )

    def _scale_up_emit_champion_probe_event(
        self,
        snapshot: dict,
        *,
        probe: str,
        status: str,
        elapsed_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        payload = {
            "experiment_id": snapshot.get("experiment_id"),
            "source_result_id": snapshot.get("source_result_id"),
            "candidate_index": snapshot.get("candidate_index"),
            "step": snapshot.get("step"),
            "probe": probe,
            "status": status,
        }
        if elapsed_ms is not None:
            payload["elapsed_ms"] = round(float(elapsed_ms), 1)
        if error:
            payload["error"] = str(error)
        if snapshot.get("path"):
            payload["path"] = snapshot.get("path")
        logger.info(
            "Champion probe %s step=%s status=%s elapsed_ms=%s",
            probe,
            payload.get("step"),
            status,
            payload.get("elapsed_ms"),
        )
        if callable(getattr(self.owner, "_emit_event", None)):
            self._emit_event("champion_probe_progress", payload)

    def _scale_up_apply_champion_id_collapse(self, payloads: list[dict]) -> None:
        snapshots = [
            item
            for item in payloads
            if item.get("_id_hidden_snapshot") is not None
            and int(item.get("step") or 0) > 0
        ]
        if len(snapshots) < 2:
            return
        early = min(snapshots, key=lambda item: int(item["step"]))
        late = max(snapshots, key=lambda item: int(item["step"]))
        try:
            from ...eval.intrinsic_dim_collapse import compute_id_collapse_rate

            collapse = compute_id_collapse_rate(
                early["_id_hidden_snapshot"],
                late["_id_hidden_snapshot"],
            )
            fields = collapse.to_dict()
            late.update(fields)
            late["id_collapse_window"] = {
                "early_step": int(early["step"]),
                "late_step": int(late["step"]),
            }
        except (RuntimeError, ValueError, TypeError, ImportError) as exc:
            late["fp_id_collapse_status"] = f"failed: {exc}"

    def _scale_up_eval_checkpoint_snapshot(
        self,
        graph,
        checkpoint_path: Path,
        step: int,
        config: RunConfig,
        dev,
        dev_str: str,
        exp_id: str | None = None,
        source_result_id: str | None = None,
        prog_idx: int | None = None,
    ) -> dict:
        snapshot: dict = {
            "experiment_id": exp_id,
            "source_result_id": source_result_id,
            "candidate_index": (int(prog_idx) + 1) if prog_idx is not None else None,
            "step": int(step),
            "status": "ok",
            "path": str(checkpoint_path),
            "device": str(dev),
        }
        model = None
        try:
            self._scale_up_emit_champion_probe_event(
                snapshot, probe="checkpoint_snapshot", status="started"
            )
            try:
                state = torch.load(
                    str(checkpoint_path), map_location="cpu", weights_only=False
                )
            except TypeError:
                state = torch.load(str(checkpoint_path), map_location="cpu")
            payload = state.get("payload") if isinstance(state, dict) else {}
            progress = (payload or {}).get("progress") or {}
            base_loss = progress.get("final_loss") or progress.get("min_loss")
            model = compile_model(
                [graph] * int(config.n_layers),
                vocab_size=int(config.vocab_size),
                max_seq_len=int(config.scale_up_seq_len),
            )
            model.load_state_dict(state["model_state_dict"], strict=True)
            model.to(dev)
            model.eval()

            def _fresh_model():
                fresh = compile_model(
                    [graph] * int(config.n_layers),
                    vocab_size=int(config.vocab_size),
                    max_seq_len=int(config.scale_up_seq_len),
                )
                fresh.load_state_dict(state["model_state_dict"], strict=True)
                return fresh

            self._scale_up_run_hard_probe_snapshot(
                model, _fresh_model, base_loss, config, dev, dev_str, snapshot
            )
            self._scale_up_emit_champion_probe_event(
                snapshot,
                probe="checkpoint_snapshot",
                status=snapshot.get("status") or "ok",
            )
        except (RuntimeError, ValueError, KeyError, OSError, TypeError) as exc:
            snapshot["status"] = "failed"
            snapshot["error"] = str(exc)
            self._scale_up_emit_champion_probe_event(
                snapshot, probe="checkpoint_snapshot", status="failed", error=str(exc)
            )
            logger.warning(
                "Champion hard probes failed: result step=%s path=%s error=%s",
                step,
                checkpoint_path,
                exc,
            )
        finally:
            del model
            clear_gpu_memory()
        return snapshot

    def _scale_up_eval_input_batches(
        self, config: RunConfig, dev, *, n_batches: int = 4
    ) -> list:
        generator = torch.Generator(device=dev).manual_seed(42)
        seq_len = min(128, int(config.scale_up_seq_len))
        batch_size = max(1, min(4, int(config.scale_up_batch_size)))
        return [
            torch.randint(
                0,
                int(config.vocab_size),
                (batch_size, seq_len),
                device=dev,
                generator=generator,
            )
            for _ in range(max(1, int(n_batches)))
        ]

    def _scale_up_run_hard_probe_snapshot(
        self,
        model,
        model_factory,
        base_loss,
        config: RunConfig,
        dev,
        dev_str: str,
        snapshot: dict,
    ) -> None:
        """Populate one milestone snapshot with hard probes; never runs AR gate."""
        self._scale_up_run_probe_guarded(
            snapshot,
            "hellaswag",
            lambda: self._scale_up_run_hellaswag_snapshot(
                model, config, dev_str, snapshot
            ),
        )
        self._scale_up_run_probe_guarded(
            snapshot,
            "blimp",
            lambda: self._scale_up_run_blimp_snapshot(model, config, dev_str, snapshot),
        )
        self._scale_up_run_probe_guarded(
            snapshot,
            "binding_probe",
            lambda: self._scale_up_run_binding_probe_snapshot(
                model, config, dev_str, snapshot
            ),
        )
        self._scale_up_run_probe_guarded(
            snapshot,
            "investigation_v2",
            lambda: self._scale_up_run_investigation_v2_snapshot(
                model, dev_str, snapshot, config
            ),
        )
        self._scale_up_run_probe_guarded(
            snapshot,
            "trajectory",
            lambda: self._scale_up_run_trajectory_snapshot(
                model, config, dev, snapshot
            ),
        )
        self._scale_up_run_probe_guarded(
            snapshot,
            "activation_sparsity",
            lambda: self._scale_up_run_sparsity_snapshot(model, config, dev, snapshot),
        )
        self._scale_up_run_probe_guarded(
            snapshot,
            "quantization",
            lambda: self._scale_up_run_quant_snapshot(
                model_factory, config, dev, snapshot
            ),
        )
        self._scale_up_run_probe_guarded(
            snapshot,
            "noise_sensitivity",
            lambda: self._scale_up_run_noise_snapshot(model, config, dev, snapshot),
        )
        self._scale_up_run_probe_guarded(
            snapshot,
            "long_ctx",
            lambda: self._scale_up_run_long_ctx_snapshot(
                model, model_factory, base_loss, config, dev, dev_str, snapshot
            ),
        )
        self._scale_up_run_probe_guarded(
            snapshot,
            "wikitext",
            lambda: self._scale_up_run_wikitext_snapshot(
                model, config, dev_str, snapshot
            ),
        )

    def _scale_up_run_probe_guarded(self, snapshot: dict, name: str, fn) -> None:
        t0 = time.perf_counter()
        self._scale_up_emit_champion_probe_event(snapshot, probe=name, status="started")
        try:
            fn()
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            snapshot[f"{name}_status"] = f"failed: {exc}"
            self._scale_up_emit_champion_probe_event(
                snapshot,
                probe=name,
                status="failed",
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                error=str(exc),
            )
            return
        self._scale_up_emit_champion_probe_event(
            snapshot,
            probe=name,
            status=snapshot.get(f"{name}_status") or "ok",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def _scale_up_run_hellaswag_snapshot(
        self, model, config: RunConfig, dev_str: str, snapshot: dict
    ) -> None:
        from ...eval.hellaswag_eval import evaluate_hellaswag

        hs = evaluate_hellaswag(model, int(config.vocab_size), dev_str)
        snapshot.update(
            {
                "hellaswag_acc": hs.get("hellaswag_acc"),
                "hellaswag_status": hs.get("hellaswag_status"),
                "hellaswag_n_examples": hs.get("hellaswag_total"),
                "hellaswag_metric_version": hs.get("hellaswag_metric_version"),
                "hellaswag_tokenizer_mode": hs.get("hellaswag_tokenizer_mode"),
                "hellaswag_tiktoken_encoding": hs.get("hellaswag_tiktoken_encoding"),
            }
        )

    def _scale_up_run_blimp_snapshot(
        self, model, config: RunConfig, dev_str: str, snapshot: dict
    ) -> None:
        from ...eval.blimp_eval import evaluate_blimp

        blimp = evaluate_blimp(model, int(config.vocab_size), dev_str, n_per_subtask=50)
        snapshot.update(
            {
                "blimp_overall_accuracy": blimp.overall_accuracy,
                "blimp_subtask_accuracies": blimp.subtask_accuracies,
                "blimp_n_subtasks": blimp.n_subtasks,
                "blimp_status": blimp.status,
                "blimp_elapsed_ms": blimp.elapsed_ms,
            }
        )

    def _scale_up_run_investigation_v2_snapshot(
        self, model, dev_str: str, snapshot: dict, config: RunConfig | None = None
    ) -> None:
        try:
            from ...eval.induction_validation_probe import (
                run_induction_validation_champion,
            )

            induction_validation = run_induction_validation_champion(
                model,
                device=dev_str,
                extended_budget=bool(
                    getattr(
                        config, "champion_induction_validation_extended_budget", False
                    )
                ),
            )
            induction_fields = induction_validation.to_dict()
            induction_status = str(
                getattr(induction_validation, "status", None)
                or induction_fields.get("induction_validation_status")
                or ""
            )
            if induction_status != "ok":
                induction_fields["induction_validation_auc"] = None
                induction_fields["induction_validation_max_gap_acc"] = None
                induction_fields["induction_validation_gap_accuracy_cv"] = None
            if "induction_validation_gap_accuracies" in induction_fields:
                induction_fields["induction_validation_gap_accuracies_json"] = (
                    json.dumps(
                        json_safe(
                            induction_fields.pop("induction_validation_gap_accuracies")
                        ),
                        sort_keys=True,
                    )
                )
            snapshot.update(induction_fields)
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            snapshot["induction_validation_status"] = f"failed: {exc}"

        try:
            from ...eval.binding_intermediate_probe import (
                run_binding_intermediate,
            )

            binding_intermediate = run_binding_intermediate(model, device=dev_str)
            binding_fields = binding_intermediate.to_dict()
            if "binding_intermediate_distance_accuracies" in binding_fields:
                binding_fields["binding_intermediate_distance_accuracies_json"] = (
                    json.dumps(
                        json_safe(
                            binding_fields.pop(
                                "binding_intermediate_distance_accuracies"
                            )
                        ),
                        sort_keys=True,
                    )
                )
            snapshot.update(binding_fields)
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            snapshot["binding_intermediate_status"] = f"failed: {exc}"

        try:
            from ...eval.ar_validation import run_ar_validation

            ar_validation = run_ar_validation(model, device=dev_str)
            snapshot.update(ar_validation.to_dict())
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            snapshot["ar_validation_status"] = f"failed: {exc}"

    def _scale_up_run_trajectory_snapshot(
        self, model, config: RunConfig, dev, snapshot: dict
    ) -> None:
        try:
            from ...eval.trajectory_metrics import (
                capture_hidden_state_snapshot,
                compute_trajectory_metrics,
            )

            seq_len = min(64, int(config.scale_up_seq_len))
            traj = compute_trajectory_metrics(
                model,
                metric_phase="champion_confirmation_milestone",
                device=dev,
                seq_len=seq_len,
                icld_seq_len=seq_len,
                icld_batch_size=max(1, min(32, int(config.scale_up_batch_size) * 4)),
                logit_margin_batch_size=max(
                    1, min(32, int(config.scale_up_batch_size) * 4)
                ),
                spec_norm_vocab_size=int(config.vocab_size),
            )
            snapshot.update(traj.to_column_dict())
            generator = torch.Generator(device=dev).manual_seed(20260506)
            probe_ids = torch.randint(
                0,
                int(config.vocab_size),
                (max(1, min(4, int(config.scale_up_batch_size))), seq_len),
                device=dev,
                generator=generator,
            )
            hidden = capture_hidden_state_snapshot(
                model,
                probe_ids,
                step=int(snapshot.get("step") or 0),
                device=dev,
            )
            snapshot["_id_hidden_snapshot"] = hidden
            snapshot["fp_id_snapshot_status"] = hidden.status
            snapshot["fp_id_snapshot_pr"] = hidden.participation_ratio
            snapshot["fp_id_snapshot_norm"] = hidden.intrinsic_dim_normalized
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            snapshot["trajectory_status"] = f"failed: {exc}"

    def _scale_up_run_sparsity_snapshot(
        self, model, config: RunConfig, dev, snapshot: dict
    ) -> None:
        try:
            from ...eval.sparsity import evaluate_activation_sparsity

            sr = evaluate_activation_sparsity(
                model,
                self._scale_up_eval_input_batches(config, dev),
                dev,
            )
            snapshot["activation_sparsity_score"] = sr.get("activation_sparsity_score")
            snapshot["dead_neuron_ratio"] = sr.get("dead_neuron_ratio")
            snapshot["activation_sparsity_details"] = sr
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            snapshot["activation_sparsity_status"] = f"failed: {exc}"

    def _scale_up_run_quant_snapshot(
        self, model_factory, config: RunConfig, dev, snapshot: dict
    ) -> None:
        quant_model = None
        try:
            from ...eval.quantization import evaluate_sparse_quant_quality

            quant_model = model_factory().to(dev)
            qr = evaluate_sparse_quant_quality(
                quant_model,
                self._scale_up_eval_input_batches(config, dev),
                dev,
            )
            if qr:
                snapshot["quant_int8_retention"] = qr.get("full_retention")
                snapshot["quant_quality_per_byte"] = qr.get("quality_per_byte")
                snapshot["quantization_details"] = qr
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            snapshot["quantization_status"] = f"failed: {exc}"
        finally:
            del quant_model
            clear_gpu_memory()

    def _scale_up_run_noise_snapshot(
        self, model, config: RunConfig, dev, snapshot: dict
    ) -> None:
        try:
            from ...eval.noise_sensitivity import evaluate_noise_sensitivity

            nr = evaluate_noise_sensitivity(
                model,
                self._scale_up_eval_input_batches(config, dev),
                dev,
                vocab_size=int(config.vocab_size),
            )
            snapshot["robustness_noise_score"] = nr.get("noise_sensitivity_score")
            snapshot["noise_sensitivity_details"] = nr
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            snapshot["noise_sensitivity_status"] = f"failed: {exc}"

    def _scale_up_run_binding_probe_snapshot(
        self, model, config: RunConfig, dev_str: str, snapshot: dict
    ) -> None:
        try:
            from ...eval.associative_recall import associative_recall_score
            from ...eval.binding_curriculum import (
                CURRICULUM_BINDING_DISTANCES,
                CURRICULUM_BINDING_EVAL_SCREENING,
                CURRICULUM_BINDING_PROTOCOL_VERSION,
                CURRICULUM_BINDING_STEPS_SCREENING,
                curriculum_binding_range_profile,
            )
            from ...eval.binding_range import binding_range_profile
            from ...eval.native_induction import (
                induction_result_metadata,
                induction_score_gold,
            )

            ind = induction_score_gold(
                model,
                device=dev_str,
                seed=getattr(config, "screening_probe_seed", None),
            )
            snapshot.update(induction_result_metadata(ind))
            zero = binding_range_profile(
                model,
                distances=CURRICULUM_BINDING_DISTANCES,
                n_eval=CURRICULUM_BINDING_EVAL_SCREENING,
                device=dev_str,
                seed=getattr(config, "screening_probe_seed", None),
            )
            curriculum = curriculum_binding_range_profile(
                model,
                distances=CURRICULUM_BINDING_DISTANCES,
                n_train_steps=CURRICULUM_BINDING_STEPS_SCREENING,
                n_eval=CURRICULUM_BINDING_EVAL_SCREENING,
                device=dev_str,
                seed=getattr(config, "screening_probe_seed", None),
            )
            ar = associative_recall_score(
                model,
                n_pairs=20,
                n_eval=200,
                n_train_steps=500,
                batch_size=16,
                device=dev_str,
            )
            snapshot.update(
                {
                    "binding_screening_auc": zero.auc,
                    "binding_distance_accuracies": zero.distance_accuracies,
                    "binding_screening_eval_examples": CURRICULUM_BINDING_EVAL_SCREENING,
                    "binding_probe_distances": list(CURRICULUM_BINDING_DISTANCES),
                    "binding_screening_elapsed_ms": zero.elapsed_ms,
                    "binding_curriculum_auc": curriculum.auc,
                    "binding_distance_accuracies_curriculum": (
                        curriculum.distance_accuracies
                    ),
                    "binding_curriculum_steps": curriculum.train_steps,
                    "binding_curriculum_elapsed_ms": curriculum.elapsed_ms,
                    "binding_curriculum_protocol_version": (
                        CURRICULUM_BINDING_PROTOCOL_VERSION
                    ),
                    "ar_legacy_auc": ar.auc,
                    "ar_legacy_final_acc": ar.final_acc,
                    "ar_legacy_timed_out": int(ar.timed_out),
                    "ar_legacy_above_chance": int(ar.above_chance),
                }
            )
            snapshot["binding_screening_composite"] = round(
                0.4 * (ar.auc or 0.0)
                + 0.3 * (ind.auc or 0.0)
                + 0.3 * (zero.auc or 0.0),
                4,
            )
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            snapshot["binding_probe_status"] = f"failed: {exc}"

    def _scale_up_run_long_ctx_snapshot(
        self,
        model,
        model_factory,
        base_loss,
        config: RunConfig,
        dev,
        dev_str: str,
        snapshot: dict,
    ) -> None:
        try:
            from ...eval.long_context import run_long_context_sweep
            from ...eval.long_range_ar import long_range_ar_score
            from ...eval.multi_hop_retrieval import multi_hop_retrieval_score
            from ...eval.passkey_retrieval import passkey_retrieval_score

            batch_size = max(1, min(16, int(config.validation_batch_size)))
            scaling = run_long_context_sweep(
                model_factory,
                int(config.vocab_size),
                dev,
                base_loss=max(float(base_loss or 1.0), 1e-6),
                seq_lens=(512, 1024),
                n_steps=min(60, max(20, int(config.validation_steps) // 100)),
                batch_size=max(1, min(2, int(config.validation_batch_size))),
                lr=float(config.stage1_lr),
            )
            assoc = long_range_ar_score(
                model,
                seq_lens=(128, 256, 512, 1024),
                n_train_steps=300,
                batch_size=batch_size,
                device=dev_str,
            )
            passkey = passkey_retrieval_score(
                model,
                seq_lens=(256, 512, 1024, 2048),
                n_train_steps=300,
                batch_size=batch_size,
                device=dev_str,
            )
            multi_hop = multi_hop_retrieval_score(
                model,
                seq_lens=(256, 512, 1024),
                hop_depths=(2, 3),
                n_train_steps=300,
                batch_size=batch_size,
                device=dev_str,
            )
            retrieval_scores = [
                s
                for s in (assoc.score, passkey.score, multi_hop.score)
                if s is not None
            ]
            aggregate = (
                round(sum(retrieval_scores) / len(retrieval_scores), 4)
                if retrieval_scores
                else None
            )
            snapshot.update(
                {
                    "robustness_long_ctx_score": scaling.get("long_context_score"),
                    "robustness_long_ctx_scaling_score": scaling.get(
                        "long_context_score"
                    ),
                    "long_context_details": scaling,
                    "max_viable_seq_len": scaling.get("max_viable_len"),
                    "robustness_long_ctx_assoc_score": assoc.score,
                    "robustness_long_ctx_passkey_score": passkey.score,
                    "robustness_long_ctx_multi_hop_score": multi_hop.score,
                    "robustness_long_ctx_retrieval_aggregate": aggregate,
                    "robustness_long_ctx_combined_score": (
                        round(
                            0.4 * scaling.get("long_context_score", 0.0)
                            + 0.6 * aggregate,
                            4,
                        )
                        if aggregate is not None
                        else scaling.get("long_context_score")
                    ),
                }
            )
        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            snapshot["long_ctx_status"] = f"failed: {exc}"

    def _scale_up_run_wikitext_snapshot(
        self, model, config: RunConfig, dev_str: str, snapshot: dict
    ) -> None:
        corpus_path = str(getattr(config, "corpus_path", "") or "").lower()
        if (
            str(getattr(config, "data_mode", "") or "") != "corpus"
            or "wikitext" not in corpus_path
        ):
            snapshot["wikitext_status"] = "skipped_non_wikitext_corpus"
            return
        try:
            from ...eval.wikitext_eval import evaluate_wikitext_perplexity

            wt = evaluate_wikitext_perplexity(
                model,
                int(config.vocab_size),
                dev_str,
                n_train_steps=200,
                seq_len=min(128, int(config.scale_up_seq_len)),
            )
            snapshot["wikitext_perplexity"] = wt.get("wikitext_perplexity")
            snapshot["wikitext_pre_perplexity"] = wt.get("wikitext_pre_perplexity")
            snapshot["wikitext_score"] = wt.get("wikitext_score")
            snapshot["wikitext_ppl_improvement"] = wt.get("wikitext_ppl_improvement")
            snapshot["wikitext_eval_steps"] = wt.get("n_train_steps")
            snapshot["screening_wikitext_variant"] = wt.get("variant")
            snapshot["screening_wikitext_elapsed_ms"] = wt.get("elapsed_ms")
            snapshot["screening_wikitext_metric_version"] = "bpe_eval_v1"
            snapshot["corpus_path"] = getattr(config, "corpus_path", None)
            snapshot["wikitext_status"] = wt.get("error") or "ok"
            snapshot["screening_wikitext_status"] = snapshot["wikitext_status"]
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            snapshot["wikitext_status"] = f"failed: {exc}"

    def _scale_up_apply_champion_snapshot(
        self, program_metrics: dict, snapshot: dict
    ) -> None:
        for key in (
            "wikitext_perplexity",
            "wikitext_pre_perplexity",
            "wikitext_score",
            "wikitext_ppl_improvement",
            "wikitext_eval_steps",
            "screening_wikitext_status",
            "screening_wikitext_metric_version",
            "screening_wikitext_variant",
            "screening_wikitext_elapsed_ms",
            "corpus_path",
            "hellaswag_acc",
            "hellaswag_status",
            "hellaswag_n_examples",
            "hellaswag_metric_version",
            "hellaswag_tokenizer_mode",
            "hellaswag_tiktoken_encoding",
            "blimp_overall_accuracy",
            "blimp_subtask_accuracies",
            "blimp_n_subtasks",
            "blimp_status",
            "blimp_elapsed_ms",
            "ar_legacy_auc",
            "ar_legacy_final_acc",
            "ar_legacy_timed_out",
            "ar_legacy_above_chance",
            "induction_screening_auc",
            "induction_validation_auc",
            "induction_validation_max_gap_acc",
            "induction_validation_gap_accuracy_cv",
            "induction_validation_gap_accuracies_json",
            "induction_validation_steps_trained",
            "induction_validation_status",
            "induction_validation_elapsed_ms",
            "induction_validation_protocol_version",
            "induction_intermediate_auc",
            "induction_intermediate_max_gap_acc",
            "induction_intermediate_gap_accuracies_json",
            "induction_intermediate_steps_trained",
            "induction_intermediate_status",
            "induction_intermediate_elapsed_ms",
            "induction_intermediate_protocol_version",
            "binding_screening_auc",
            "binding_intermediate_auc",
            "binding_intermediate_max_distance_acc",
            "binding_intermediate_distance_accuracies_json",
            "binding_intermediate_train_steps",
            "binding_intermediate_status",
            "binding_intermediate_elapsed_ms",
            "binding_intermediate_protocol_version",
            "ar_validation_metric_version",
            "ar_validation_final_acc",
            "ar_validation_held_pair_acc",
            "ar_validation_held_class_acc",
            "ar_validation_learning_curve_json",
            "ar_validation_steps_to_floor",
            "ar_validation_rank_score",
            "ar_validation_status",
            "ar_validation_elapsed_ms",
            "binding_curriculum_auc",
            "binding_screening_composite",
            "activation_sparsity_score",
            "dead_neuron_ratio",
            "quant_int8_retention",
            "quant_quality_per_byte",
            "robustness_noise_score",
            "robustness_long_ctx_score",
            "robustness_long_ctx_scaling_score",
            "long_context_details",
            "max_viable_seq_len",
            "robustness_long_ctx_assoc_score",
            "robustness_long_ctx_passkey_score",
            "robustness_long_ctx_multi_hop_score",
            "robustness_long_ctx_retrieval_aggregate",
            "robustness_long_ctx_combined_score",
            "fp_metric_phase",
            "fp_jacobian_spectral_norm",
            "fp_jacobian_effective_rank",
            "fp_sensitivity_uniformity",
            "fp_spec_norm_status",
            "fp_jacobian_erf_density",
            "fp_jacobian_erf_variance",
            "fp_jacobian_erf_decay_slope",
            "fp_jacobian_erf_last_norm",
            "fp_jacobian_erf_first_norm",
            "fp_jacobian_erf_status",
            "fp_jacobian_erf_elapsed_ms",
            "fp_icld_velocity",
            "fp_icld_early_loss",
            "fp_icld_late_loss",
            "fp_icld_delta_loss",
            "fp_icld_seq_len",
            "fp_icld_status",
            "fp_icld_elapsed_ms",
            "fp_logit_margin_velocity",
            "fp_logit_margin_initial",
            "fp_logit_margin_final",
            "fp_logit_margin_delta",
            "fp_logit_margin_n_steps",
            "fp_logit_margin_status",
            "fp_logit_margin_elapsed_ms",
            "fp_id_pr_early",
            "fp_id_pr_late",
            "fp_id_norm_early",
            "fp_id_norm_late",
            "fp_id_step_early",
            "fp_id_step_late",
            "fp_id_collapse_rate",
            "fp_id_collapse_rate_normalized",
            "fp_id_collapse_status",
            "fp_id_collapse_elapsed_ms",
        ):
            if snapshot.get(key) is not None:
                program_metrics[key] = snapshot[key]


def run_champion_milestone_evals(owner: Any, **kwargs: Any) -> None:
    ChampionConfirmationEvaluator(owner)._scale_up_champion_milestone_evals(**kwargs)
