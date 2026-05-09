"""Execution mixin: capability-ranking tier."""

from __future__ import annotations

import json
import time
import traceback
from typing import Any, Dict, List, Optional

from ..shared_utils import resolve_device
from ._helpers import (
    _build_source_map,
    _evaluate_capability_rankers,
    _record_capability_ranking_result,
    clear_gpu_memory,
)
from ._lifecycle import _LifecycleMixin
from ._types import RunConfig

import logging

logger = logging.getLogger(__name__)


class _ExecutionCapabilityRankingMixin:
    """Selective capability ranker execution between investigation and validation."""

    __slots__ = ()
    _publish_terminal_event = _LifecycleMixin._publish_terminal_event
    _complete_experiment_compat = _LifecycleMixin._complete_experiment_compat
    _fail_experiment_compat = _LifecycleMixin._fail_experiment_compat

    def _run_capability_ranking_thread(
        self,
        exp_id: str,
        result_ids: List[str],
        config: RunConfig,
        hypothesis: Optional[str],
    ) -> None:
        self._live_training_context = {"exp_id": exp_id, "phase": "capability_ranking"}
        nb = self._make_notebook()
        t_start = time.time()
        results: Dict[str, Any] = {
            "total": len(result_ids),
            "ranked": 0,
            "failed": 0,
            "result_ids": list(result_ids),
            "ranker_results": [],
        }
        try:
            dev = resolve_device(config.device)
            source_map = _build_source_map(nb, result_ids)
            for idx, result_id in enumerate(result_ids):
                if self._stop_event.is_set():
                    break
                source = source_map.get(result_id) or {}
                if not source:
                    results["failed"] += 1
                    results["ranker_results"].append(
                        {"result_id": result_id, "status": "missing_source"}
                    )
                    continue
                self._update_progress(
                    status="evaluating",
                    current_program=idx + 1,
                    total_programs=len(result_ids),
                    current_fingerprint=str(source.get("graph_fingerprint") or "")[:10],
                    aria_message=(
                        f"{self.aria.NAME}: Ranking capability evidence "
                        f"{idx + 1}/{len(result_ids)}..."
                    ),
                )
                graph_json = source.get("graph_json")
                graph_json_str = graph_json if isinstance(graph_json, str) else None
                arch_spec_json = source.get("arch_spec_json")
                arch_spec_json_str = (
                    arch_spec_json if isinstance(arch_spec_json, str) else None
                )
                model_source = str(source.get("model_source") or "graph_synthesis")
                ranker_result = _evaluate_capability_rankers(
                    config=config,
                    dev=dev,
                    model_source=model_source,
                    arch_spec_json_str=arch_spec_json_str,
                    graph_json_str=graph_json_str,
                    cached_json_load=json.loads,
                    stop_event=self._stop_event,
                )
                if self._stop_event.is_set():
                    break
                _record_capability_ranking_result(
                    nb=nb,
                    exp_id=exp_id,
                    source_result_id=result_id,
                    source=source,
                    model_source=model_source,
                    benchmark_result=ranker_result,
                )
                status = str(ranker_result.get("capability_ranking_status") or "")
                if status == "ok":
                    results["ranked"] += 1
                else:
                    results["failed"] += 1
                results["ranker_results"].append(
                    {
                        "result_id": result_id,
                        "status": status or "unknown",
                        "induction_intermediate_auc": ranker_result.get(
                            "induction_intermediate_auc"
                        ),
                        "binding_intermediate_auc": ranker_result.get(
                            "binding_intermediate_auc"
                        ),
                        "ar_intermediate_diagnostic_score": ranker_result.get(
                            "ar_intermediate_diagnostic_score"
                        ),
                        "binding_multislot_diagnostic_score": ranker_result.get(
                            "binding_multislot_diagnostic_score"
                        ),
                        "induction_validation_auc": ranker_result.get(
                            "induction_validation_auc"
                        ),
                        "ar_validation_rank_score": ranker_result.get(
                            "ar_validation_rank_score"
                        ),
                    }
                )
                clear_gpu_memory()

            results["elapsed_seconds"] = time.time() - t_start
            self._complete_experiment_compat(
                nb=nb,
                experiment_id=exp_id,
                results=results,
                aria_summary=(
                    "Capability ranking complete: "
                    f"{results['ranked']}/{results['total']} ranked."
                ),
            )
            self._publish_terminal_event(
                producer="runner.execution_capability_ranking",
                event_type="experiment_completed",
                exp_id=exp_id,
                payload={
                    "completed_at": time.time(),
                    "results": results,
                    "mode": "capability_ranking",
                },
            )
            self._update_progress(
                status="completed",
                aria_message=(
                    f"{self.aria.NAME}: Capability ranking complete "
                    f"({results['ranked']}/{results['total']})."
                ),
            )
            self._emit_event(
                "capability_ranking_completed",
                {"experiment_id": exp_id, **results},
            )
        except BaseException as exc:
            error = traceback.format_exc()
            logger.error("Capability ranking failed (%s): %s\n%s", exp_id, exc, error)
            self._fail_experiment_compat(
                nb=nb,
                experiment_id=exp_id,
                error=str(exc),
                results=results,
            )
            self._update_progress(
                status="failed",
                error=str(exc),
                aria_message=self.aria.react_to_failure(str(exc)),
            )
            self._emit_event(
                "capability_ranking_failed",
                {"experiment_id": exp_id, "error": str(exc)},
            )
        finally:
            self._live_training_context = None
            nb.close()
            clear_gpu_memory()
