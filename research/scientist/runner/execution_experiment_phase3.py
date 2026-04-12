"""Phase helpers extracted from _execute_experiment for maintainability."""

from __future__ import annotations

import logging
import math
import os
import time
import json
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
        candidates = self._generate_candidates(
            config, config.n_programs, "morphological_box", nb=nb
        )
        results["total"] = len(candidates)

        dev = resolve_device(config.device)
        dev_str = str(dev)

        for i, cand in enumerate(candidates):
            if self._stop_event.is_set():
                break

            self._update_progress(
                current_program=i + 1,
                current_fingerprint=(cand.fingerprint or "")[:10],
                elapsed_seconds=time.time() - t_start,
            )

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
            training_curve = s1_result.get("training_curve")
            if s1_passed:
                results["stage1_passed"] += 1
                with self._lock:
                    self._progress.stage1_passed += 1

            program_metrics: Dict[str, Any] = {}
            try:
                program_metrics.update(self._extract_sandbox_metrics(sandbox_result))
            except Exception as exc:
                logger.debug("Suppressed error: %s", exc)
            try:
                program_metrics["param_count"] = sandbox_result.param_count
            except Exception as exc:
                logger.debug("Suppressed error: %s", exc)

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
            from ._helpers import screening_probe_fields, screening_wikitext_fields

            program_metrics["train_budget_steps"] = config.stage1_steps
            program_metrics.update(screening_wikitext_fields(s1_result))
            program_metrics.update(screening_probe_fields(s1_result))
            program_metrics.update(screening_probe_fields(program_metrics))
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
            except Exception as exc:
                logger.debug("Suppressed error: %s", exc)
            if training_curve and result_id:
                try:
                    nb.store_training_curve(result_id, training_curve)
                except Exception as exc:
                    logger.debug(
                        "store_training_curve failed for %s: %s", result_id, exc
                    )

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
            # Default to one Stage 1 worker per GPU. The screening pipeline runs
            # post-train CUDA probes and may attach native dispatchers to models;
            # oversubscribing a single device with multiple Python threads has
            # caused unrecoverable native/CUDA crashes in practice.
            workers_per_gpu = max(
                1, int(os.environ.get("ARIA_WORKERS_PER_GPU", "1") or "1")
            )
            num_workers = num_gpus * workers_per_gpu
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
        candidate_batch_size = max(
            1, min(32, int(math.sqrt(max(1, config.n_programs))))
        )
        results["candidate_batch_size"] = candidate_batch_size
        return dev, dev_str, orchestrator, candidate_batch_size

    def _run_gbm_prescreener(
        self,
        *,
        nb: LabNotebook,
        graphs: List[Any],
        config: RunConfig,
        exp_id: str,
        results: Dict[str, Any],
    ) -> List[Any]:
        """Rank graphs by the runtime predictor and drop only clear losers."""

        if not config.gbm_prescreener_enabled or not graphs:
            return graphs
        from ..ml_influence_policy import component_is_allowed

        if not component_is_allowed("screening_ensemble", config):
            logger.info(
                "Ensemble pre-screener requested but blocked by ML trust policy"
            )
            return graphs

        try:
            from ..intelligence.predictor import load_runtime_ensemble
            from ...synthesis.graph_features import (
                extract_graph_features,
                enrich_with_op_stats,
                load_op_stats,
            )

            db_path = (
                str(nb.db_path)
                if hasattr(nb, "db_path")
                else "research/lab_notebook.db"
            )
            profiling_db = "research/profiling/component_profiles.db"
            ensemble = load_runtime_ensemble(profiling_db=profiling_db)
            if ensemble is None or not ensemble.is_fitted():
                logger.debug(
                    "Ensemble pre-screener disabled: no persisted predictor artifacts loaded"
                )
                return graphs

            op_stats_cache = load_op_stats(db_path)
            scored: List[tuple[float, float, float, float, Any, Dict[str, Any]]] = []
            for graph in graphs:
                graph_dict = graph.to_dict()
                features = extract_graph_features(graph_dict)
                if features:
                    nodes = graph_dict.get("nodes") or {}
                    ops = [
                        node.get("op_name", "")
                        for node in nodes.values()
                        if node.get("op_name", "") != "input"
                    ]
                    enrich_with_op_stats(features, ops, preloaded=op_stats_cache)
                planning = ensemble.predict_planning_score(
                    graph_json=graph_dict,
                    graph_features=features if features else None,
                )
                scored.append(
                    (
                        float(planning.get("planning_score", 0.0)),
                        float(planning.get("p_pass", 0.0)),
                        float(planning.get("p_induction_learner", 0.0)),
                        float(planning.get("predicted_induction_auc", 0.0)),
                        graph,
                        graph_dict,
                    )
                )

            scored.sort(key=lambda row: -row[0])
            kept: List[Any] = []
            skipped = 0
            for planning_score, p_pass, p_ind, pred_auc, graph, graph_dict in scored:
                if p_pass < config.gbm_gate_threshold:
                    skipped += 1
                    try:
                        nb.record_program_result(
                            experiment_id=exp_id,
                            graph=graph,
                            graph_json=json.dumps(graph_dict, separators=(",", ":")),
                            status="predictor_skip",
                            metrics={
                                "predicted_p_s1": p_pass,
                                "predicted_induction_auc": pred_auc,
                                "predicted_p_induction_learner": p_ind,
                                "predictor_planning_score": planning_score,
                            },
                        )
                    except (TypeError, ValueError) as exc:
                        logger.debug("Failed recording predictor_skip result: %s", exc)
                    continue
                kept.append(graph)

            results["funnel_counts"]["gbm_prescreener_skipped"] = skipped
            results["funnel_counts"]["post_gbm_prescreener"] = len(kept)
            diagnostics = (
                ensemble.diagnostics() if hasattr(ensemble, "diagnostics") else {}
            )
            planning_scores = [row[0] for row in scored]
            pass_scores = [row[1] for row in scored]
            induction_scores = [row[2] for row in scored]
            logger.info(
                "Ensemble ranker: %d graphs scored plan=[%.3f-%.3f] "
                "pass=[%.3f-%.3f] induction=[%.3f-%.3f], "
                "%d below P(pass_s1) floor (%.2f), %d kept, components=%d",
                len(scored),
                min(planning_scores) if planning_scores else 0.0,
                max(planning_scores) if planning_scores else 0.0,
                min(pass_scores) if pass_scores else 0.0,
                max(pass_scores) if pass_scores else 0.0,
                min(induction_scores) if induction_scores else 0.0,
                max(induction_scores) if induction_scores else 0.0,
                skipped,
                config.gbm_gate_threshold,
                len(kept),
                diagnostics.get("n_components", 1),
            )
            return kept
        except Exception as exc:
            logger.debug("Ensemble pre-screener unavailable: %s", exc)
            return graphs

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
                    "SELECT graph_fingerprint FROM program_results"
                ).fetchall()
                if r[0]
            }
        except Exception as exc:
            logger.debug("Failed to load existing fingerprints for dedup: %s", exc)
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
            if (
                len(graphs) >= dedup_target
                or config.model_source == "fingerprint_refine"
            ):
                break
            shortfall = original_count - len(graphs)
            if shortfall <= 0:
                break
            extra = batch_generate(min(shortfall * 2, original_count), grammar).graphs
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
        self._update_progress(total_programs=len(graphs), status="evaluating")

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
