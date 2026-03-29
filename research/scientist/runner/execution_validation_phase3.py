"""Validation phase helpers extracted from execution._run_validation_thread."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ...synthesis.serializer import graph_from_json
from ...training.training_program import synthesize_training_program
from ..native_runner import compile_model_native_first as compile_model
from ..shared_utils import resolve_device
from ._helpers import _build_source_map, clear_gpu_memory
from ._types import RunConfig

logger = logging.getLogger(__name__)


class _ExecutionValidationPhase3Mixin:
    """Split helpers for validation phase orchestration."""

    def _prepare_validation_state(
        self,
        config: RunConfig,
        result_ids: List[str],
        nb: Any,
    ) -> Tuple[Dict[str, Any], torch.device, str, RunConfig, Dict[str, Dict[str, Any]]]:
        results = {
            "total": len(result_ids),
            "stage0_passed": 0,
            "stage05_passed": 0,
            "stage1_passed": 0,
            "novel_count": 0,
            "best_loss_ratio": None,
            "best_novelty_score": None,
            "survivors": [],
            "validation_results": [],
        }

        dev = resolve_device(config.device)
        dev_str = str(dev)

        val_config = config.copy()
        val_config.stage1_steps = config.validation_steps
        val_config.stage1_batch_size = config.validation_batch_size
        val_config.max_seq_len = config.validation_seq_len
        # Scale early stopping for longer validation runs.
        step_ratio = config.validation_steps / max(config.stage1_steps, 1)
        val_config.early_stop_patience = int(config.early_stop_patience * step_ratio)
        val_config.early_stop_min_steps = int(config.early_stop_min_steps * step_ratio)

        source_map = _build_source_map(nb, result_ids)
        return results, dev, dev_str, val_config, source_map

    def _get_validation_best_training_json(
        self, nb: Any, source_result_id: str
    ) -> Optional[str]:
        entry = nb.get_leaderboard_entry(source_result_id)
        return entry.get("investigation_best_training") if entry else None

    def _reconstruct_validation_model(
        self,
        model_source: str,
        arch_spec_json_str: Optional[str],
        graph_json_str: Optional[str],
        config: RunConfig,
    ) -> Optional[nn.Module]:
        try:
            if model_source == "morphological_box" and arch_spec_json_str:
                from ...morphological_box import ArchSpec
                from ...arch_builder import BuildConfig, build_model

                spec_data = self._cached_json_load(arch_spec_json_str)
                spec = ArchSpec(**spec_data)
                build_cfg = BuildConfig(
                    dim=config.model_dim,
                    n_layers=config.n_layers,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.validation_seq_len,
                )
                return build_model(spec, build_cfg)
            if graph_json_str:
                graph = graph_from_json(graph_json_str)
                layer_graphs = [graph] * config.n_layers
                return compile_model(
                    layer_graphs,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.validation_seq_len,
                )
            return None
        except Exception as e:
            logger.debug("Model reconstruction failed: %s", e)
            return None

    def _run_validation_seed_sweep(
        self,
        exp_id: str,
        source_result_id: str,
        model_source: str,
        arch_spec_json_str: Optional[str],
        graph_json_str: Optional[str],
        config: RunConfig,
        val_config: RunConfig,
        dev: torch.device,
        best_tp_json: Optional[str],
        progress_payload: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        seed_results: List[Dict[str, Any]] = []

        for seed in range(config.validation_n_seeds):
            if self._stop_event.is_set():
                break
            torch.manual_seed(seed * 42 + 7)

            model = self._reconstruct_validation_model(
                model_source=model_source,
                arch_spec_json_str=arch_spec_json_str,
                graph_json_str=graph_json_str,
                config=config,
            )
            if model is None:
                continue

            init_scheme = "default"
            if seed == config.validation_n_seeds - 1:
                init_scheme = "xavier_uniform"
                for p in model.parameters():
                    if p.dim() >= 2:
                        nn.init.xavier_uniform_(p)

            self._emit_event(
                "validation_progress",
                {
                    **progress_payload,
                    "seed": seed + 1,
                    "total_seeds": config.validation_n_seeds,
                    "status": f"seed {seed + 1}/{config.validation_n_seeds}",
                },
            )

            if best_tp_json:
                try:
                    tp_data = self._cached_json_load(best_tp_json)
                    tp = synthesize_training_program(
                        n_steps=config.validation_steps,
                        max_seq_len=config.validation_seq_len,
                        seed=tp_data.get("seed", seed),
                    )
                    s1_result = self._train_with_program(
                        model,
                        tp,
                        val_config,
                        dev,
                        seed=self._stable_seed(
                            exp_id,
                            source_result_id,
                            seed,
                            "validation_inline_tp",
                        ),
                    )
                except Exception:
                    s1_result = self._micro_train(
                        model,
                        val_config,
                        dev,
                        seed=self._stable_seed(
                            exp_id,
                            source_result_id,
                            seed,
                            "validation_inline_micro",
                        ),
                    )
            else:
                s1_result = self._micro_train(
                    model,
                    val_config,
                    dev,
                    seed=self._stable_seed(
                        exp_id,
                        source_result_id,
                        seed,
                        "validation_inline_micro",
                    ),
                )

            seed_results.append(
                {
                    "seed": seed,
                    "init_scheme": init_scheme,
                    "passed": s1_result.get("passed", False),
                    "loss_ratio": s1_result.get("loss_ratio"),
                    "final_loss": s1_result.get("final_loss"),
                    "n_train_steps": s1_result.get("n_train_steps"),
                    "final_lr": s1_result.get("final_lr"),
                    "training_program_json": s1_result.get("training_program_json"),
                    "optimizer_class": s1_result.get("optimizer_class"),
                    "optimizer_lr": s1_result.get("optimizer_lr"),
                    "optimizer_weight_decay": s1_result.get("optimizer_weight_decay"),
                    "optimizer_momentum": s1_result.get("optimizer_momentum"),
                    "optimizer_beta1": s1_result.get("optimizer_beta1"),
                    "optimizer_beta2": s1_result.get("optimizer_beta2"),
                }
            )

            del model
            clear_gpu_memory()

        return seed_results
