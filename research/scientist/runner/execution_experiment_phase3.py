"""Phase helpers extracted from _execute_experiment for maintainability."""

from __future__ import annotations

import gc
import logging
import math
import os
import time
from typing import Any, Dict, List, Set, Tuple

import torch

from ...orchestrator.executor import WorkerPoolOrchestrator
from ...synthesis.grammar import batch_generate
from ..notebook import ExperimentEntry, LabNotebook
from ..shared_utils import resolve_device
from ._types import RunConfig

logger = logging.getLogger(__name__)


class _ExecutionExperimentPhase3Mixin:
    """Split helpers for experiment execution phase orchestration."""

    def _run_morphological_screening(
        self,
        exp_id: str,
        config: RunConfig,
        nb: LabNotebook,
        results: Dict[str, Any],
        t_start: float,
    ) -> None:
        candidates = self._generate_candidates(config, config.n_programs, "morphological_box")
        results["total"] = len(candidates)

        dev = resolve_device(config.device)
        dev_str = str(dev)

        for i, cand in enumerate(candidates):
            if self._stop_event.is_set():
                break

            with self._lock:
                self._progress.current_program = i + 1
                self._progress.current_fingerprint = (cand.fingerprint or "")[:10]
                self._progress.elapsed_seconds = time.time() - t_start

            model = cand.model
            if model is None:
                continue

            try:
                sandbox_result = self._safe_eval_for_stage(
                    model,
                    stage_tag="morph_candidate_screening",
                    batch_size=2,
                    seq_len=min(128, config.max_seq_len),
                    vocab_size=config.vocab_size,
                    device=dev_str,
                )
            except Exception as e:
                logger.error("Error evaluating morph candidate %d: %s", i, e)
                continue

            s0_passed = bool(sandbox_result.passed)
            s05_passed = (
                sandbox_result.stability_score >= config.stage05_stability_threshold
                and sandbox_result.causality_passed
            )
            if s0_passed:
                results["stage0_passed"] += 1
                with self._lock:
                    self._progress.stage0_passed += 1
            if s05_passed:
                results["stage05_passed"] += 1
                with self._lock:
                    self._progress.stage05_passed += 1

            if not s0_passed or not s05_passed:
                continue

            s1_result = self._micro_train(
                model,
                config,
                dev,
                seed=self._stable_seed(exp_id, i, "morphology"),
            )
            s1_passed = bool(s1_result.get("passed", False))
            if s1_passed:
                results["stage1_passed"] += 1
                with self._lock:
                    self._progress.stage1_passed += 1

            program_metrics: Dict[str, Any] = {}
            try:
                program_metrics.update(self._extract_sandbox_metrics(sandbox_result))
            except Exception:
                pass
            try:
                program_metrics["param_count"] = sandbox_result.param_count
            except Exception:
                pass

            for k in (
                "initial_loss",
                "final_loss",
                "min_loss",
                "loss_ratio",
                "throughput",
                "avg_step_time_ms",
                "total_train_time_ms",
                "validation_loss",
                "validation_loss_ratio",
                "generalization_gap",
                "discovery_loss",
                "discovery_loss_ratio",
            ):
                if k in s1_result:
                    program_metrics[k] = s1_result.get(k)
            from ._helpers import screening_wikitext_fields
            program_metrics.update(screening_wikitext_fields(s1_result))
            self._merge_s1_telemetry(program_metrics, s1_result)

            result_id = nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=cand.fingerprint,
                graph_json="{}",
                stage0_passed=s0_passed,
                stage05_passed=s05_passed,
                stage1_passed=s1_passed,
                loss_ratio=s1_result.get("loss_ratio"),
                final_loss=s1_result.get("final_loss"),
                model_source="morphological_box",
                arch_spec_json=cand.arch_spec_json,
                **program_metrics,
            )
            try:
                from ...eval.wikitext_eval import screening_wikitext_payload
                payload = screening_wikitext_payload(s1_result)
                if payload:
                    nb.set_external_benchmarks(result_id, payload)
            except Exception:
                pass

    def _prepare_screening_orchestrator(
        self,
        config: RunConfig,
        results: Dict[str, Any],
    ) -> Tuple[torch.device, str, WorkerPoolOrchestrator, int]:
        dev = resolve_device(config.device)
        dev_str = str(dev)

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(num_gpus)]
            num_workers = num_gpus * 2
        else:
            devices = ["cpu"]
            num_workers = 1

        remote_workers = [
            w.strip()
            for w in os.environ.get("ARIA_REMOTE_WORKERS", "").split(",")
            if w.strip()
        ]

        orchestrator = WorkerPoolOrchestrator(
            train_fn=lambda m, c, s, d: self._micro_train_async(m, c, s, d),
            num_workers=num_workers,
            max_queue_size=config.n_programs,
            devices=devices,
            remote_workers=remote_workers,
        )
        candidate_batch_size = max(1, min(32, int(math.sqrt(max(1, config.n_programs)))))
        results["candidate_batch_size"] = candidate_batch_size
        return dev, dev_str, orchestrator, candidate_batch_size

    def _dedup_graph_candidates(
        self,
        nb: LabNotebook,
        graphs: List[Any],
        grammar: Any,
        config: RunConfig,
        exp_id: str,
        results: Dict[str, Any],
    ) -> Tuple[List[Any], Set[str]]:
        try:
            existing_fps = {
                r[0]
                for r in nb.conn.execute(
                    "SELECT DISTINCT graph_fingerprint FROM program_results"
                ).fetchall()
                if r[0]
            }
        except Exception:
            existing_fps = set()

        original_count = len(graphs)
        dedup_max_rounds = 3
        dedup_target = max(1, int(original_count * 0.5))
        for dedup_round in range(dedup_max_rounds):
            novel = []
            seen_this_batch: Set[str] = set()
            for g in graphs:
                fp = g.fingerprint()
                if fp not in existing_fps and fp not in seen_this_batch:
                    novel.append(g)
                    seen_this_batch.add(fp)
            graphs = novel
            if len(graphs) >= dedup_target or config.model_source == "fingerprint_refine":
                break
            shortfall = original_count - len(graphs)
            if shortfall <= 0:
                break
            extra = batch_generate(min(shortfall * 2, original_count), grammar)
            graphs.extend(extra)
            logger.info(
                "Experiment %s dedup round %d: %d novel / %d generated, added %d extra candidates",
                exp_id[:8],
                dedup_round + 1,
                len(novel),
                original_count,
                len(extra),
            )

        for g in graphs:
            existing_fps.add(g.fingerprint())

        dedup_rate = 1.0 - (len(graphs) / max(original_count, 1))
        results["skipped_dedup"] = original_count - len(graphs)
        results["dedup_rate"] = round(dedup_rate, 3)
        results["dedup_novel_count"] = len(graphs)
        results["dedup_known_fingerprints"] = len(existing_fps)
        results["total"] = len(graphs)

        if dedup_rate > 0.1:
            logger.info(
                "Experiment %s dedup: %d/%d candidates were duplicates (%.0f%% dedup rate), %d novel candidates remain, %d known fingerprints in DB",
                exp_id[:8],
                original_count - len(graphs),
                original_count,
                dedup_rate * 100,
                len(graphs),
                len(existing_fps),
            )
        if dedup_rate > 0.8:
            logger.warning(
                "Experiment %s: grammar diversity exhaustion — %.0f%% dedup rate. "
                "Consider increasing grammar depth/ops or switching to refinement mode.",
                exp_id[:8],
                dedup_rate * 100,
            )

        return graphs, existing_fps

    def _log_generated_graph_observation(
        self,
        nb: LabNotebook,
        exp_id: str,
        graphs: List[Any],
        grammar: Any,
        config: RunConfig,
    ) -> None:
        with self._lock:
            self._progress.total_programs = len(graphs)
            self._progress.status = "evaluating"

        logger.info(
            "Experiment %s: generated %d graphs (depth=%d, ops=%d, dim=%d, device=%s)",
            exp_id[:8],
            len(graphs),
            grammar.max_depth,
            grammar.max_ops,
            config.model_dim,
            config.device,
        )

        nb.add_entry(
            ExperimentEntry(
                entry_type="observation",
                title=f"Generated {len(graphs)} computation graphs",
                content=(
                    f"Grammar: depth={grammar.max_depth}, ops={grammar.max_ops}, "
                    f"dim={config.model_dim}, math_space_weight={config.math_space_weight}"
                ),
                experiment_id=exp_id,
            )
        )
