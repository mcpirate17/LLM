"""Phase helpers extracted from _execute_experiment for maintainability."""

from __future__ import annotations

import logging
import math
import os
import time
import json
from typing import Any, Dict, List, Set, Tuple

import torch

from research.defaults import RUNS_DB

from ...orchestrator.executor import WorkerPoolOrchestrator
from ...synthesis.grammar import batch_generate
from ..notebook import ExperimentEntry, LabNotebook
from ..shared_utils import resolve_device
from ._types import RunConfig
from .screening_measured_rescue import (
    measured_rescue_config,
    rescue_skipped_candidates,
)
from .measured_rescue_observability import initialize_measured_rescue_records

logger = logging.getLogger(__name__)

_SQLITE_IN_CLAUSE_CHUNK = 900
_RANK_COMPOSITE_UNUSABLE = 1e6
_RANK_COMPOSITE_USABLE_CUTOFF = 1e5


def _resolve_p_pass_floor(
    ensemble: Any, config: RunConfig, report: Dict[str, Any]
) -> Tuple[float, str]:
    """Pick the p_pass floor and record its provenance for telemetry."""
    temporal_f1_threshold = (
        (
            (
                (report.get("ensemble_calibrated") or {}).get(
                    "temporal_holdout_evaluation"
                )
                or {}
            ).get("operating_points")
            or {}
        )
        .get("f1", {})
        .get("threshold")
    )
    explicit_floor = float(
        getattr(config, "screening_ensemble_p_pass_floor", 0.0) or 0.0
    )
    deprecated_floor = float(getattr(config, "gbm_gate_threshold", 0.0) or 0.0)
    if explicit_floor > 0.0:
        return explicit_floor, "config.screening_ensemble_p_pass_floor"
    if deprecated_floor > 0.0:
        return deprecated_floor, "config.gbm_gate_threshold"
    if temporal_f1_threshold is not None:
        try:
            temporal_f1_threshold = float(temporal_f1_threshold)
        except (TypeError, ValueError):
            temporal_f1_threshold = None
    if temporal_f1_threshold is not None and temporal_f1_threshold > 0.0:
        return (
            temporal_f1_threshold,
            "ensemble.temporal_holdout_evaluation.operating_points.f1.threshold",
        )
    return float(getattr(ensemble, "gate_threshold", 0.5)), "ensemble.gate_threshold"


def _predict_rank_composite_safe(gbm: Any, features: Dict[str, float]) -> float:
    """Wrap predict_rank_composite. Returns sentinel when unusable/missing."""
    if (
        gbm is None
        or not hasattr(gbm, "is_fitted")
        or not gbm.is_fitted()
        or not features
        or not hasattr(gbm, "predict_rank_composite")
    ):
        return _RANK_COMPOSITE_UNUSABLE
    try:
        value = float(gbm.predict_rank_composite(features))
    except (TypeError, ValueError) as exc:
        logger.debug("predict_rank_composite failed: %s", exc)
        return _RANK_COMPOSITE_UNUSABLE
    return value if math.isfinite(value) else _RANK_COMPOSITE_UNUSABLE


def _score_graphs_for_prescreener(
    ensemble: Any,
    gbm: Any,
    graphs: List[Any],
    op_stats_cache: Dict[str, Any],
) -> List[tuple[float, float, float, float, float, Any, Dict[str, Any]]]:
    """Score every graph with planning + production composite rank head."""
    from ...synthesis.graph_features import (
        extract_graph_features_bundle,
        enrich_with_op_stats,
    )

    scored: List[tuple[float, float, float, float, float, Any, Dict[str, Any]]] = []
    for graph in graphs:
        graph_dict = graph.to_dict()
        features, ops = extract_graph_features_bundle(graph_dict)
        if features:
            for op in ops:
                if op:
                    features[f"op_{op}"] = features.get(f"op_{op}", 0.0) + 1.0
            enrich_with_op_stats(features, ops, preloaded=op_stats_cache)
        planning = ensemble.predict_planning_score(
            graph_json=graph_dict,
            graph_features=features if features else None,
        )
        rank_composite = _predict_rank_composite_safe(gbm, features or {})
        scored.append(
            (
                float(planning.get("planning_score", 0.0)),
                float(planning.get("p_pass", 0.0)),
                float(planning.get("p_induction_learner", 0.0)),
                float(planning.get("predicted_induction_screening_auc", 0.0)),
                rank_composite,
                graph,
                graph_dict,
            )
        )
    return scored


def _partition_prescreener_candidates(
    nb: LabNotebook,
    scored: List[tuple[float, float, float, float, float, Any, Dict[str, Any]]],
    *,
    exp_id: str,
    p_pass_floor: float,
    floor_source: str,
    rescue_cfg: Any = None,
) -> Tuple[List[tuple[Any, float]], int, List[Dict[str, Any]]]:
    """Split scored candidates into kept_with_rank and persist skips.

    When ``rescue_cfg`` is provided (env-gated via ``ARIA_MEASURED_RESCUE``, default OFF), the
    candidates the GBM would drop are first offered to the label-free MEASURED filter; those
    flagged structurally induction-capable (``long_range_reach >= tau``) are re-admitted to
    ``kept_with_rank`` at the explore tail instead of being recorded as ``predictor_skip``.
    Additive: with ``rescue_cfg=None`` the skip set, metrics, and counts are unchanged.
    """
    kept_with_rank: List[tuple[Any, float]] = []
    would_skip: List[Tuple[Any, Dict[str, Any], Dict[str, Any]]] = []
    for (
        planning_score,
        p_pass,
        p_ind,
        pred_auc,
        rank_composite,
        graph,
        graph_dict,
    ) in scored:
        if p_pass < p_pass_floor:
            skip_metrics: Dict[str, Any] = {
                "predicted_p_s1": p_pass,
                "predicted_induction_screening_auc": pred_auc,
                "predicted_p_induction_learner": p_ind,
                "predictor_planning_score": planning_score,
                "screening_ensemble_p_pass_floor": p_pass_floor,
                "screening_ensemble_p_pass_floor_source": floor_source,
            }
            if rank_composite < _RANK_COMPOSITE_USABLE_CUTOFF:
                skip_metrics["predicted_rank_composite"] = rank_composite
            would_skip.append((graph, graph_dict, skip_metrics))
            continue
        kept_with_rank.append((graph, rank_composite))

    rescued_graphs: List[Any] = []
    rescue_records: List[Dict[str, Any]] = []
    if rescue_cfg is not None and would_skip:
        try:
            rescued_graphs, rescue_records = rescue_skipped_candidates(
                would_skip, rescue_cfg
            )
        except Exception as exc:  # a rescue failure must never break the gate
            logger.debug("measured rescue raised, skipping: %s", exc)
            rescued_graphs, rescue_records = [], []
    rescued_ids = {id(g) for g in rescued_graphs}

    for graph, graph_dict, skip_metrics in would_skip:
        if id(graph) in rescued_ids:
            continue  # re-admitted by measured rescue — do not record as a skip
        try:
            nb.record_program_result(
                experiment_id=exp_id,
                graph=graph,
                graph_json=json.dumps(graph_dict, separators=(",", ":")),
                status="predictor_skip",
                metrics=skip_metrics,
            )
        except (TypeError, ValueError) as exc:
            logger.debug("Failed recording predictor_skip result: %s", exc)

    # Rescued candidates ride the explore tail (sentinel rank): re-admitted for measurement,
    # not promoted as predicted-good.
    for graph in rescued_graphs:
        kept_with_rank.append((graph, _RANK_COMPOSITE_USABLE_CUTOFF))

    skipped = len(would_skip) - len(rescued_graphs)
    return kept_with_rank, skipped, rescue_records


def _reorder_kept_by_rank_composite(
    kept_with_rank: List[tuple[Any, float]],
) -> Tuple[List[Any], bool, List[float]]:
    """Re-rank survivors by the production composite head when usable.

    Higher composite = better predicted quality. Sentinel sinks to the bottom;
    if every survivor is unusable the sort is a no-op and the original
    planning_score order is preserved.
    """
    usable_ranks = [r for _, r in kept_with_rank if r < _RANK_COMPOSITE_USABLE_CUTOFF]
    used = bool(usable_ranks)
    if used:
        kept_with_rank.sort(
            key=lambda pair: (
                -pair[1] if pair[1] < _RANK_COMPOSITE_USABLE_CUTOFF else float("inf")
            )
        )
    return [graph for graph, _ in kept_with_rank], used, usable_ranks


def _maybe_rerank_kept_by_ar_binding_overlay(
    kept: List[Any],
    config: RunConfig,
    results: Dict[str, Any],
) -> List[Any]:
    """Apply the sibling AR/binding reranker after composite rank ordering."""
    if not getattr(config, "ar_binding_overlay_enabled", False) or not kept:
        return kept
    try:
        from ..intelligence.ar_binding_reranker import rerank_graphs_by_ar_binding

        kept, stats = rerank_graphs_by_ar_binding(kept)
    except Exception as exc:
        logger.debug("AR/binding overlay reranker unavailable: %s", exc)
        results["screening_ar_binding_overlay_used"] = False
        return kept

    results["screening_ar_binding_overlay_used"] = bool(stats.get("used"))
    results["screening_ar_binding_overlay_scored"] = int(stats.get("scored", 0) or 0)
    results["screening_ar_binding_overlay_holdout_required"] = int(
        stats.get("holdout_required", 0) or 0
    )
    results["screening_ar_binding_overlay_score_min"] = stats.get("score_min")
    results["screening_ar_binding_overlay_score_max"] = stats.get("score_max")
    return kept


def _log_prescreener_summary(
    ensemble: Any,
    scored: List[tuple],
    *,
    usable_ranks: List[float],
    skipped: int,
    p_pass_floor: float,
    n_kept: int,
    rank_composite_used: bool,
) -> None:
    """Emit the one-line ensemble-ranker diagnostic for a prescreened batch."""
    diagnostics = ensemble.diagnostics() if hasattr(ensemble, "diagnostics") else {}
    planning_scores = [row[0] for row in scored]
    pass_scores = [row[1] for row in scored]
    induction_scores = [row[2] for row in scored]
    rank_range = (
        "[%.2f-%.2f]" % (min(usable_ranks), max(usable_ranks))
        if usable_ranks
        else "unusable"
    )
    logger.info(
        "Ensemble ranker: %d graphs scored plan=[%.3f-%.3f] "
        "pass=[%.3f-%.3f] induction=[%.3f-%.3f] composite=%s, "
        "%d below P(pass_s1) floor (%.2f), %d kept (composite_reorder=%s), components=%d",
        len(scored),
        min(planning_scores) if planning_scores else 0.0,
        max(planning_scores) if planning_scores else 0.0,
        min(pass_scores) if pass_scores else 0.0,
        max(pass_scores) if pass_scores else 0.0,
        min(induction_scores) if induction_scores else 0.0,
        max(induction_scores) if induction_scores else 0.0,
        rank_range,
        skipped,
        p_pass_floor,
        n_kept,
        rank_composite_used,
        diagnostics.get("n_components", 1),
    )


class _ExecutionExperimentPhase3Mixin:
    """Split helpers for experiment execution phase orchestration."""

    def _lookup_existing_fingerprints(
        self,
        nb: LabNotebook,
        fingerprints: Set[str],
    ) -> Set[str]:
        if not fingerprints:
            return set()

        found: Set[str] = set()
        ordered = [fp for fp in fingerprints if fp]
        for start in range(0, len(ordered), _SQLITE_IN_CLAUSE_CHUNK):
            chunk = ordered[start : start + _SQLITE_IN_CLAUSE_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            rows = nb.conn.execute(
                "SELECT graph_fingerprint FROM program_results_compat "  # nosec B608  # nosemgrep: python-sql-string-formatting
                f"WHERE graph_fingerprint IN ({placeholders})",
                chunk,
            ).fetchall()
            found.update(str(row[0]) for row in rows if row[0])
        return found

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

    def _emit_measured_rescue(
        self,
        results: Dict[str, Any],
        rescue_records: List[Dict[str, Any]],
        rescue_cfg: Any,
        exp_id: str,
    ) -> None:
        """Record + announce candidates re-admitted by the label-free measured filter."""
        if not rescue_records:
            return
        results["funnel_counts"]["measured_rescued"] = len(rescue_records)
        initialize_measured_rescue_records(
            results,
            rescue_records,
            experiment_id=exp_id,
            tau=getattr(rescue_cfg, "tau", None),
            max_rescue=getattr(rescue_cfg, "max_rescue", None),
            probe_budget=getattr(rescue_cfg, "probe_budget", None),
        )
        self._emit_event(
            "measured_rescue_applied",
            {
                "experiment_id": exp_id,
                "n_rescued": len(rescue_records),
                "tau": getattr(rescue_cfg, "tau", None),
                "rescued": results.get("measured_rescue_records", []),
            },
        )

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
        # Capability-first templates are structurally novel — the GBM was
        # trained before they existed and systematically rejects them.
        # Bypass the prescreener when capability_first is active so the
        # new templates actually reach screening.
        if getattr(config, "_capability_first_mode", False):
            return graphs
        from ..ml_influence_policy import component_is_allowed
        from ..ml_influence_policy import load_predictor_metrics_report

        if not component_is_allowed("screening_ensemble", config):
            logger.info(
                "Ensemble pre-screener requested but blocked by ML trust policy"
            )
            return graphs

        try:
            from ..intelligence.predictor import load_runtime_ensemble
            from ...synthesis.graph_features import load_op_stats

            db_path = str(nb.db_path) if hasattr(nb, "db_path") else RUNS_DB
            profiling_db = "research/profiling/component_profiles.db"
            ensemble = load_runtime_ensemble(profiling_db=profiling_db)
            if ensemble is None or not ensemble.is_fitted():
                logger.debug(
                    "Ensemble pre-screener disabled: no persisted predictor artifacts loaded"
                )
                return graphs

            report = load_predictor_metrics_report()
            p_pass_floor, floor_source = _resolve_p_pass_floor(ensemble, config, report)

            op_stats_cache = load_op_stats(db_path)
            scored = _score_graphs_for_prescreener(
                ensemble, getattr(ensemble, "gbm", None), graphs, op_stats_cache
            )
            scored.sort(key=lambda row: -row[0])
            rescue_cfg = measured_rescue_config(
                device=str(getattr(config, "device", "cpu"))
            )
            kept_with_rank, skipped, rescue_records = _partition_prescreener_candidates(
                nb,
                scored,
                exp_id=exp_id,
                p_pass_floor=p_pass_floor,
                floor_source=floor_source,
                rescue_cfg=rescue_cfg,
            )
            kept, rank_composite_used, usable_ranks = _reorder_kept_by_rank_composite(
                kept_with_rank
            )
            kept = _maybe_rerank_kept_by_ar_binding_overlay(kept, config, results)

            results["funnel_counts"]["gbm_prescreener_skipped"] = skipped
            results["funnel_counts"]["post_gbm_prescreener"] = len(kept)
            self._emit_measured_rescue(results, rescue_records, rescue_cfg, exp_id)
            results["screening_ensemble_p_pass_floor"] = p_pass_floor
            results["screening_ensemble_p_pass_floor_source"] = floor_source
            results["screening_rank_composite_used"] = rank_composite_used
            _log_prescreener_summary(
                ensemble,
                scored,
                usable_ranks=usable_ranks,
                skipped=skipped,
                p_pass_floor=p_pass_floor,
                n_kept=len(kept),
                rank_composite_used=rank_composite_used,
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
        original_count = len(graphs)
        graph_fps = [(g, g.fingerprint()) for g in graphs]
        existing_fps = self._lookup_existing_fingerprints(
            nb, {fp for _, fp in graph_fps}
        )
        known_before = len(existing_fps)
        dedup_max_rounds = 3
        dedup_target = max(1, int(original_count * 0.5))
        for dedup_round in range(dedup_max_rounds):
            novel = []
            seen_this_batch: Set[str] = set()
            for g, fp in graph_fps:
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
            extra_fps = [(g, g.fingerprint()) for g in extra]
            existing_fps.update(
                self._lookup_existing_fingerprints(nb, {fp for _, fp in extra_fps})
            )
            graph_fps = [(g, g.fingerprint()) for g in graphs]
            graph_fps.extend(extra_fps)
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
        results["dedup_known_fingerprints"] = known_before
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
