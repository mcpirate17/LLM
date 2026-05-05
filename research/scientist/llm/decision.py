"""LLM-driven next-experiment planner with local-first fallback."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .backend import create_backend_from_config
from .prompts import (
    NEXT_EXPERIMENT_PLAN_PROMPT,
    NEXT_EXPERIMENT_PLAN_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class NextExperimentPlannerConfig:
    enabled: bool = True
    local_backend: str = ""
    local_model: str = ""
    local_host: str = ""
    remote_backend: str = ""
    remote_model: str = ""
    temperature: float = 0.2
    max_tokens: int = 700
    budget_dollars: float = 0.0
    max_n_programs: int = 200
    max_time_minutes: int = 120
    min_novelty_weight: float = 0.25
    min_family_bonus_weight: float = 0.10


class NextExperimentDecisionPlanner:
    """Generates structured next-experiment plans from recent outcomes."""

    def __init__(self, config: NextExperimentPlannerConfig):
        self.config = config
        self._local_backend = None
        self._remote_backend = None

    @classmethod
    def from_run_config(cls, run_config: Any) -> "NextExperimentDecisionPlanner":
        cfg = NextExperimentPlannerConfig(
            enabled=bool(getattr(run_config, "enable_llm_decision_planner", True)),
            local_backend=str(
                getattr(run_config, "llm_decision_local_backend", "") or ""
            ),
            local_model=str(getattr(run_config, "llm_decision_local_model", "") or ""),
            local_host=str(getattr(run_config, "llm_decision_local_host", "") or ""),
            remote_backend=str(
                getattr(run_config, "llm_decision_remote_backend", "") or ""
            ),
            remote_model=str(
                getattr(run_config, "llm_decision_remote_model", "") or ""
            ),
            temperature=float(
                getattr(run_config, "llm_decision_temperature", 0.2) or 0.2
            ),
            max_tokens=int(getattr(run_config, "llm_decision_max_tokens", 700) or 700),
            budget_dollars=float(
                getattr(run_config, "llm_decision_budget_dollars", 0.0) or 0.0
            ),
            max_n_programs=int(
                getattr(run_config, "llm_decision_max_n_programs", 200) or 200
            ),
            max_time_minutes=int(
                getattr(run_config, "llm_decision_max_time_minutes", 120) or 120
            ),
            min_novelty_weight=float(
                getattr(run_config, "llm_decision_min_novelty_weight", 0.25) or 0.25
            ),
            min_family_bonus_weight=float(
                getattr(run_config, "llm_decision_min_family_bonus_weight", 0.10)
                or 0.10
            ),
        )
        return cls(cfg)

    def propose_plan(
        self,
        summary: Dict[str, Any],
        *,
        current_cost_dollars: float = 0.0,
        fallback_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a validated plan. Falls back to heuristic plan on any failure."""
        fallback = self._fallback_plan(summary, fallback_plan=fallback_plan)
        if not self.config.enabled:
            fallback["planner"] = {"source": "disabled", "backend": None}
            return self._apply_meta_strategy_bias(fallback, summary)
        if (
            self.config.budget_dollars > 0
            and current_cost_dollars >= self.config.budget_dollars
        ):
            fallback["planner"] = {"source": "budget_exhausted", "backend": None}
            return self._apply_meta_strategy_bias(fallback, summary)

        prompt = NEXT_EXPERIMENT_PLAN_PROMPT.format(
            summary_json=json.dumps(summary, indent=2, sort_keys=True),
            max_n_programs=self.config.max_n_programs,
            max_time_minutes=self.config.max_time_minutes,
        )

        for source_name, backend in (
            ("local", self._get_local_backend()),
            ("remote", self._get_remote_backend()),
        ):
            if backend is None:
                continue
            try:
                resp = backend.generate(
                    prompt,
                    system=NEXT_EXPERIMENT_PLAN_SYSTEM_PROMPT,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                candidate = self._parse_plan_payload(resp.text)
                plan = self._validate_and_harden(candidate, fallback=fallback)
                if plan is None:
                    continue
                plan = self._apply_meta_strategy_bias(plan, summary)
                plan["planner"] = {
                    "source": source_name,
                    "backend": getattr(backend, "name", None),
                    "model": getattr(resp, "model", None),
                    "tokens_used": int(getattr(resp, "tokens_used", 0) or 0),
                }
                return plan
            except Exception as exc:
                logger.debug("LLM planner (%s) failed: %s", source_name, exc)

        fallback["planner"] = {"source": "heuristic_fallback", "backend": None}
        return self._apply_meta_strategy_bias(fallback, summary)

    def _get_local_backend(self):
        if self._local_backend is not None:
            return self._local_backend
        if not self.config.local_backend:
            self._local_backend = None
            return None
        self._local_backend = create_backend_from_config(
            self.config.local_backend,
            model=self.config.local_model,
            host=self.config.local_host,
        )
        if self._local_backend and not self._local_backend.is_available():
            self._local_backend = None
        return self._local_backend

    def _get_remote_backend(self):
        if self._remote_backend is not None:
            return self._remote_backend
        if not self.config.remote_backend:
            self._remote_backend = None
            return None
        self._remote_backend = create_backend_from_config(
            self.config.remote_backend,
            model=self.config.remote_model,
        )
        if self._remote_backend and not self._remote_backend.is_available():
            self._remote_backend = None
        return self._remote_backend

    @staticmethod
    def _parse_plan_payload(text: str) -> Dict[str, Any]:
        if not text:
            return {}
        clean = text.strip()
        fence_match = re.search(r"```json\s*(\{.+?\})\s*```", clean, flags=re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass
        obj_match = re.search(r"(\{.+\})", clean, flags=re.DOTALL)
        if obj_match:
            try:
                return json.loads(obj_match.group(1))
            except json.JSONDecodeError:
                return {}
        return {}

    def _validate_and_harden(
        self, candidate: Dict[str, Any], *, fallback: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(candidate, dict):
            return None
        mode = str(candidate.get("mode") or "").strip().lower()
        if mode not in {
            "synthesis",
            "evolution",
            "novelty",
            "investigation",
            "validation",
            "refinement",
        }:
            return None
        rationale = str(candidate.get("rationale") or "").strip()
        if not rationale:
            return None
        config = candidate.get("config")
        if not isinstance(config, dict):
            config = {}

        # Hard guardrails: preserve diversity and avoid single-metric collapse.
        if mode in {"novelty", "evolution", "refinement"}:
            novelty_weight = float(
                config.get(
                    "novelty_weight", fallback["config"].get("novelty_weight", 0.5)
                )
            )
            config["novelty_weight"] = max(
                novelty_weight, self.config.min_novelty_weight
            )
            fam_bonus = float(
                config.get(
                    "selection_family_bonus_weight",
                    fallback["config"].get("selection_family_bonus_weight", 0.2),
                )
            )
            config["selection_family_bonus_weight"] = max(
                fam_bonus, self.config.min_family_bonus_weight
            )

        # Reproducibility defaults.
        config.setdefault("selection_policy", "ucb")
        config.setdefault("selection_epsilon", 0.0)

        # Cost/time bounds.
        if "n_programs" in config:
            try:
                config["n_programs"] = max(
                    4, min(int(config["n_programs"]), self.config.max_n_programs)
                )
            except (TypeError, ValueError):
                config.pop("n_programs", None)
        if "max_time_minutes" in config:
            try:
                config["max_time_minutes"] = max(
                    1,
                    min(int(config["max_time_minutes"]), self.config.max_time_minutes),
                )
            except (TypeError, ValueError):
                config.pop("max_time_minutes", None)

        return {
            "mode": mode,
            "reasoning": rationale,
            "confidence": float(candidate.get("confidence", 0.5) or 0.5),
            "config": config,
            "guardrails": candidate.get("guardrails")
            if isinstance(candidate.get("guardrails"), dict)
            else {},
        }

    @staticmethod
    def _apply_meta_strategy_bias(
        plan: Dict[str, Any],
        summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        meta_strategy = (
            summary.get("meta_profile_strategy")
            if isinstance(summary.get("meta_profile_strategy"), dict)
            else {}
        )
        if not meta_strategy.get("active"):
            return plan
        out = dict(plan)
        config = dict(out.get("config") if isinstance(out.get("config"), dict) else {})
        config_bias = (
            meta_strategy.get("config_bias")
            if isinstance(meta_strategy.get("config_bias"), dict)
            else {}
        )
        for key, value in config_bias.items():
            if key == "op_weights" and isinstance(value, dict):
                merged = dict(config.get("op_weights") or {})
                for op_name, weight in value.items():
                    if op_name not in merged:
                        merged[str(op_name)] = weight
                config["op_weights"] = merged
            elif key == "category_weights" and isinstance(value, dict):
                merged = dict(config.get("category_weights") or {})
                for category, weight in value.items():
                    if category not in merged:
                        merged[str(category)] = weight
                config["category_weights"] = merged
            else:
                config.setdefault(key, value)
        out["config"] = config

        guardrails = dict(
            out.get("guardrails") if isinstance(out.get("guardrails"), dict) else {}
        )
        guardrails.setdefault(
            "meta_profile_strategy",
            meta_strategy.get("guardrails", {}),
        )
        out["guardrails"] = guardrails
        out["meta_profile_strategy_used"] = True
        excerpt = dict(
            out.get("summary_excerpt")
            if isinstance(out.get("summary_excerpt"), dict)
            else {}
        )
        excerpt["meta_profile_strategy_bias"] = meta_strategy.get("strategy_bias")
        excerpt["top_profile_refresh_ops"] = list(
            meta_strategy.get("top_profile_refresh_ops") or []
        )[:4]
        out["summary_excerpt"] = excerpt
        if meta_strategy.get("rationale"):
            reasoning = str(out.get("reasoning") or "")
            if "Meta-profile strategy:" not in reasoning:
                out["reasoning"] = (
                    f"{reasoning} Meta-profile strategy: {meta_strategy['rationale']}"
                ).strip()
        return out

    @staticmethod
    def _fallback_plan(
        summary: Dict[str, Any], fallback_plan: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        fallback = fallback_plan if isinstance(fallback_plan, dict) else {}
        meta_strategy = (
            summary.get("meta_profile_strategy")
            if isinstance(summary.get("meta_profile_strategy"), dict)
            else {}
        )
        mode = str(fallback.get("mode") or "synthesis").strip().lower()
        if meta_strategy.get("active"):
            meta_mode = str(meta_strategy.get("recommended_next_mode") or "").lower()
            if meta_mode in {
                "synthesis",
                "evolution",
                "novelty",
                "investigation",
                "validation",
                "refinement",
            }:
                mode = meta_mode
        if mode not in {
            "synthesis",
            "evolution",
            "novelty",
            "investigation",
            "validation",
            "refinement",
        }:
            mode = "synthesis"
        reasoning = str(
            fallback.get("reasoning")
            or "Rule-based fallback plan from recent outcomes."
        )
        if meta_strategy.get("active"):
            meta_reason = str(meta_strategy.get("rationale") or "").strip()
            if meta_reason:
                reasoning = f"{reasoning} Meta-profile strategy: {meta_reason}"
        confidence = float(fallback.get("confidence", 0.45) or 0.45)
        if meta_strategy.get("active"):
            confidence = max(confidence, 0.62)
        config = (
            fallback.get("config") if isinstance(fallback.get("config"), dict) else {}
        )
        config = dict(config)
        if meta_strategy.get("active") and isinstance(
            meta_strategy.get("config_bias"), dict
        ):
            config.update(meta_strategy["config_bias"])
            # Preserve caller/fallback explicit diversity knobs if present.
            fallback_config = (
                fallback.get("config")
                if isinstance(fallback.get("config"), dict)
                else {}
            )
            for key in (
                "novelty_weight",
                "selection_family_bonus_weight",
                "selection_policy",
                "selection_epsilon",
            ):
                if key in fallback_config:
                    config[key] = fallback_config[key]
        return {
            "mode": mode,
            "reasoning": reasoning,
            "confidence": confidence,
            "config": config,
            "guardrails": {"meta_profile_strategy": meta_strategy.get("guardrails", {})}
            if meta_strategy.get("active")
            else {},
            "summary_excerpt": {
                "recent_experiment_id": summary.get("recent_experiment_id"),
                "stage1_survivors": summary.get("stage1_survivors", 0),
                "best_loss_ratio": summary.get(
                    "best_validation_loss_ratio", summary.get("best_loss_ratio")
                ),
                "best_novelty": summary.get("best_novelty"),
                "meta_profile_strategy_bias": meta_strategy.get("strategy_bias")
                if meta_strategy.get("active")
                else None,
            },
        }
