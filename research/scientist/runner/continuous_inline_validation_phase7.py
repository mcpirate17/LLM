"""Inline validation helpers extracted from continuous._run_inline_validation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..llm.context import build_validation_context
from ..notebook import LabNotebook
from ..shared_utils import resolve_device
from ._types import LiveProgress, RunConfig


class _ContinuousInlineValidationPhase7Mixin:
    """Split helpers for continuous inline validation orchestration."""

    def _inline_validation_candidate_ids(self, config: RunConfig, leaderboard: List[Dict[str, Any]]) -> List[str]:
        candidates = [
            e
            for e in leaderboard
            if e.get("tier") == "investigation"
            and e.get("investigation_robustness") is not None
            and e["investigation_robustness"] >= config.investigation_robustness_threshold
        ]
        if not candidates:
            return []
        return [c["result_id"] for c in candidates[: config.auto_validate_top_n] if c.get("result_id")]

    def _inline_validation_bootstrap(
        self,
        config: RunConfig,
        nb: LabNotebook,
        leaderboard: List[Dict[str, Any]],
        result_ids: List[str],
        limit_str: str,
    ) -> Tuple[str, str]:
        val_details = [d or {} for d in (nb.get_program_details(result_ids) or [])]
        val_context = build_validation_context(
            val_details, [e for e in leaderboard if e.get("result_id") in result_ids]
        )
        hypothesis = self.aria.formulate_validation_hypothesis(context=val_context)
        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="validation",
            config=self._validation_config_with_result_ids(config, result_ids, "continuous_auto"),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source="llm_context",
                llm_used=True,
                fallback_used=False,
                used_context=True,
            ),
            created_by="inline_validation",
        )
        self._live_training_context = {"exp_id": exp_id, "phase": "validation"}

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="validating",
                total_programs=len(result_ids),
                estimated_cost=self.aria.total_cost,
                total_tokens=self.aria.total_tokens,
                aria_message=f"[{limit_str}|validation] Validating {len(result_ids)} candidates",
            )

        self._emit_event("validation_started", {"experiment_id": exp_id, "n_candidates": len(result_ids)})
        entry_by_result = {
            e.get("result_id"): e.get("entry_id")
            for e in leaderboard
            if e.get("result_id") and e.get("entry_id")
        }
        for rid in result_ids:
            entry_id = entry_by_result.get(rid)
            if not entry_id:
                continue
            try:
                nb.promote_to_tier(entry_id, "validation")
            except Exception:
                pass
        return exp_id, hypothesis

    def _inline_validation_prepare_runtime(
        self,
        config: RunConfig,
        nb: LabNotebook,
        result_ids: List[str],
    ):
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
        val_config = RunConfig.from_dict(config.to_dict())
        val_config.stage1_steps = config.validation_steps
        val_config.stage1_batch_size = config.validation_batch_size
        val_config.max_seq_len = config.validation_seq_len
        program_details = [d or {} for d in (nb.get_program_details(result_ids) or [])]
        source_map = {d.get("result_id"): d for d in program_details if d.get("result_id")}
        return results, dev, dev_str, val_config, source_map
