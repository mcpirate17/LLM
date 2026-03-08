"""
Experiment Runner

The autonomous experiment execution engine. Aria uses this to:
1. Generate batches of synthesized programs
2. Evaluate them through the funnel
3. Record results in the lab notebook
4. Analyze patterns and formulate new hypotheses
5. Adjust strategy based on outcomes

Supports background execution controlled from the dashboard.
"""

from __future__ import annotations

import gc
import hashlib
import json
import copy
import math
import os
import queue
import random
import re
import shlex
import threading
import time
import traceback
import uuid
import functools
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...synthesis.grammar import GrammarConfig, generate_layer_graph, batch_generate
from ..native_runner import (
    compile_model_native_first as compile_model,
    record_native_abi_parity_result,
    reset_native_runner_telemetry,
)
from ...synthesis.validator import validate_graph
from ...synthesis.serializer import graph_to_json, graph_from_json, graph_summary
from ...synthesis.primitives import get_primitive, list_primitives, PROTECTED_OPS
from ...eval.sandbox import safe_eval
from ...eval.metrics import novelty_score
from ...eval.flops import estimate_flops
from ...eval.baseline import TransformerBaseline
from ...eval.fingerprint import compute_fingerprint, BehavioralFingerprint
from ...eval.diagnostic_tasks import run_diagnostic_suite
from ...eval.perf_budget import evaluate_perf_budget_gate
from ...eval.pruning import apply_one_shot_pruning, estimate_lm_ce_loss
from ...training.training_program import synthesize_training_program, synthesize_training_program_batch
from ...training.data_pipeline import CorpusConfig, CorpusTokenBatcher
from ...training.checkpointing import CheckpointManager
from ...orchestrator.executor import WorkerPoolOrchestrator
from ..persona import Aria, get_aria
from ..notebook import LabNotebook, ExperimentEntry
from ..evidence import (
    build_evidence_pack,
    validate_selection_decision_log,
)
from ..preregistration import (
    HypothesisPreregistration,
    PreregistrationError,
    validate_preregistration,
)
from ...healer import CodeHealer
from ...healer.core import HealerTaskSpec
from ..llm.context import (build_rich_context, build_investigation_context,
                          build_validation_context, build_mode_selection_context,
                          build_hypothesis_context, build_go_no_go_context,
                          build_knowledge_extraction_context,
                          build_campaign_formulation_context,
                          build_manual_start_fallback_context)
from ..llm.decision import NextExperimentDecisionPlanner

import logging
logger = logging.getLogger(__name__)

from ._types import RunConfig, LiveProgress, _LIVE_LOSS_CURVE_MAX_POINTS, _TRAINING_STEP_SSE_EVERY


class _ScreeningMixin:
    """Config validation, CUDA health, prescreening."""

    @staticmethod
    @functools.lru_cache(maxsize=1024)
    def _cached_json_load(json_str: str) -> Any:
        """Cached JSON decoding for high-frequency access patterns."""
        if not json_str:
            return None
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _investigation_loss_multiplier(
        screening_loss_ratio: Optional[float],
        best_loss_ratio: Optional[float],
    ) -> Optional[float]:
        """Best-investigation vs screening loss-ratio multiplier.

        Returns None when either value is unavailable or screening ratio is
        near zero (to avoid unstable division).
        """
        if screening_loss_ratio is None or best_loss_ratio is None:
            return None
        if screening_loss_ratio <= 1e-8:
            return None
        return best_loss_ratio / screening_loss_ratio

    @staticmethod
    def _is_cuda_assert_error(message: Any) -> bool:
        text = str(message or "").strip().lower()
        if not text:
            return False
        return (
            ("cuda" in text and "device-side assert" in text)
            or "cudaerrorassert" in text
            or "device side assert" in text
        )

    def _cuda_health_probe(self) -> Tuple[bool, Optional[str]]:
        """Run a minimal CUDA op/sync probe to catch poisoned contexts early."""
        try:
            dev = torch.device("cuda")
            probe = torch.zeros((4,), device=dev, dtype=torch.float32)
            probe = probe + 1.0
            _ = float(probe.sum().item())
            torch.cuda.synchronize(dev)
            return True, None
        except Exception as exc:
            return False, str(exc)

    def _recent_cuda_assert_signals(self, window: int = 5) -> Dict[str, Any]:
        """Inspect recent cycle/database history for repeated CUDA assert failures."""
        window = max(1, int(window))
        signal_count = 0
        experiment_ids: List[str] = []

        with self._lock:
            recent_cycles = list(self._aria_cycle_history[-window:])

        for item in recent_cycles:
            if self._is_cuda_assert_error(item.get("error")):
                signal_count += 1
                cycle_exp = str(item.get("experiment_id") or "").strip()
                if cycle_exp:
                    experiment_ids.append(cycle_exp)

        nb = None
        try:
            nb = self._make_notebook()
            recent_experiments = nb.get_recent_experiments(window)
            for exp in recent_experiments:
                msg = exp.get("aria_summary")
                if self._is_cuda_assert_error(msg):
                    signal_count += 1
                    exp_id = str(exp.get("experiment_id") or "").strip()
                    if exp_id:
                        experiment_ids.append(exp_id)
        except Exception:
            pass
        finally:
            if nb is not None:
                try:
                    nb.close()
                except Exception:
                    pass

        seen = set()
        unique_ids: List[str] = []
        for exp_id in experiment_ids:
            if exp_id in seen:
                continue
            seen.add(exp_id)
            unique_ids.append(exp_id)

        return {
            "count": signal_count,
            "window": window,
            "experiment_ids": unique_ids,
        }

    def _build_model_from_source(
        self,
        model_source: str,
        arch_spec_json_str: Optional[str],
        graph_json_str: Optional[str],
        config: RunConfig,
        seq_len_override: Optional[int] = None,
    ) -> Optional[nn.Module]:
        """Reconstruct a model from either morphological spec or graph JSON."""
        seq_len = seq_len_override if seq_len_override is not None else config.max_seq_len
        if model_source == "morphological_box" and arch_spec_json_str:
            from ...morphological_box import ArchSpec
            from ...arch_builder import build_model, BuildConfig
            spec_data = json.loads(arch_spec_json_str)
            spec = ArchSpec(**spec_data)
            build_cfg = BuildConfig(
                dim=config.model_dim,
                n_layers=config.n_layers,
                vocab_size=config.vocab_size,
                max_seq_len=seq_len,
            )
            return build_model(spec, build_cfg)
        if graph_json_str:
            graph = graph_from_json(graph_json_str)
            layer_graphs = [graph] * config.n_layers
            return compile_model(
                layer_graphs,
                vocab_size=config.vocab_size,
                max_seq_len=seq_len,
            )
        return None

    @staticmethod
    def _env_stage_set(name: str) -> Set[str]:
        raw = str(os.environ.get(name, "") or "")
        out: Set[str] = set()
        for part in raw.split(","):
            token = part.strip().lower()
            if token:
                out.add(token)
        return out

    def _safe_eval_for_stage(
        self,
        model: nn.Module,
        *,
        stage_tag: str,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        device: str,
        run_stability_probe: bool = True,
        timeout_seconds: int = 30,
    ):
        """Stage-aware safe_eval routing for ABI primary/probe cohorts.

        Env controls:
        - `NATIVE_RUNNER_ABI_PRIMARY_STAGES`: comma list (e.g. `fitness,candidate_gen`)
        - `NATIVE_RUNNER_ABI_PROBE_STAGES`: comma list (default all via `*` fallback)
        """
        stage_key = str(stage_tag or "").strip().lower()
        primary_stages = self._env_stage_set("NATIVE_RUNNER_ABI_PRIMARY_STAGES")
        probe_stages = self._env_stage_set("NATIVE_RUNNER_ABI_PROBE_STAGES")

        use_primary = ("*" in primary_stages) or (stage_key in primary_stages)
        if probe_stages:
            use_probe = ("*" in probe_stages) or (stage_key in probe_stages)
        else:
            # Preserve existing default behavior when no stage list configured.
            use_probe = True

        sandbox_result = safe_eval(
            model,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device,
            run_stability_probe=run_stability_probe,
            timeout_seconds=timeout_seconds,
            abi_infer_probe=use_probe,
            abi_infer_primary=use_primary,
            abi_infer_primary_no_grad=True,
        )
        abi_probe = getattr(sandbox_result, "native_abi_probe", None)
        if isinstance(abi_probe, dict):
            if bool(abi_probe.get("parity_attempted")):
                record_native_abi_parity_result(abi_probe.get("parity_pass"))
            with self._lock:
                native_runner = dict(getattr(self._progress, "native_runner", {}) or {})
                native_runner["abi_last_probe"] = dict(abi_probe)
                native_runner["abi_last_stage"] = stage_key
                self._progress.native_runner = native_runner
        return sandbox_result

    def prescreen_run_config(
        self,
        config: RunConfig,
        mode: str = "single",
        auto_harden: bool = True,
    ) -> Tuple[RunConfig, Dict[str, Any]]:
        """Pre-screen launch config and optionally apply safe hardening.

        Returns a potentially hardened config plus a prescreen report payload
        suitable for API responses.
        """
        mode_norm = str(mode or "single").strip().lower()
        screened = RunConfig.from_dict(config.to_dict())
        issues: List[Dict[str, Any]] = []
        adjustments: List[Dict[str, Any]] = []
        risk_score = 0

        def _record_issue(
            key: str,
            severity: str,
            reason: str,
            old_value: Any,
            suggested_value: Any,
            risk_points: int,
            adjusted: bool,
        ) -> None:
            nonlocal risk_score
            risk_score += max(0, int(risk_points))
            issues.append({
                "key": key,
                "severity": severity,
                "reason": reason,
                "value": old_value,
                "suggested_value": suggested_value,
                "risk_points": int(risk_points),
                "adjusted": adjusted,
            })
            if adjusted:
                adjustments.append({
                    "key": key,
                    "from": old_value,
                    "to": suggested_value,
                    "reason": reason,
                })

        def _harden_min_int(field_name: str, minimum: int,
                            severity: str, reason: str, points: int) -> None:
            old = int(getattr(screened, field_name))
            if old >= minimum:
                return
            if auto_harden:
                setattr(screened, field_name, minimum)
            _record_issue(
                key=field_name,
                severity=severity,
                reason=reason,
                old_value=old,
                suggested_value=minimum,
                risk_points=points,
                adjusted=auto_harden,
            )

        _harden_min_int(
            "n_programs", 1, "high",
            "n_programs must be >= 1 to run any evaluation.",
            30,
        )
        _harden_min_int(
            "stage1_steps", 1, "high",
            "stage1_steps must be >= 1 to avoid zero-step training failures.",
            30,
        )
        _harden_min_int(
            "n_layers", 1, "medium",
            "n_layers must be >= 1 for valid model construction.",
            20,
        )
        _harden_min_int(
            "model_dim", 16, "medium",
            "Very small model_dim is brittle and can trigger invalid shapes.",
            15,
        )
        _harden_min_int(
            "max_seq_len", 16, "medium",
            "Very small max_seq_len can destabilize evaluation and diagnostics.",
            15,
        )

        data_mode = str(screened.data_mode or "random").strip().lower()
        if data_mode == "corpus" and not (screened.corpus_path or "").strip():
            if auto_harden:
                old_mode = screened.data_mode
                screened.data_mode = "random"
                _record_issue(
                    key="data_mode",
                    severity="high",
                    reason="corpus data_mode requires corpus_path; falling back to random data.",
                    old_value=old_mode,
                    suggested_value="random",
                    risk_points=30,
                    adjusted=True,
                )
            else:
                _record_issue(
                    key="data_mode",
                    severity="high",
                    reason="corpus data_mode requires corpus_path.",
                    old_value=screened.data_mode,
                    suggested_value="random",
                    risk_points=30,
                    adjusted=False,
                )

        if str(screened.device).strip().lower() == "cuda" and not torch.cuda.is_available():
            if auto_harden:
                screened.device = "cpu"
            _record_issue(
                key="device",
                severity="high",
                reason="CUDA was requested but is not available on this host.",
                old_value="cuda",
                suggested_value="cpu",
                risk_points=35,
                adjusted=auto_harden,
            )

        if str(screened.device).strip().lower() == "cuda" and torch.cuda.is_available():
            probe_ok, probe_error = self._cuda_health_probe()
            if not probe_ok:
                if auto_harden:
                    screened.device = "cpu"
                _record_issue(
                    key="device",
                    severity="high",
                    reason=(
                        "CUDA preflight probe failed before launch; likely unstable or poisoned CUDA context. "
                        "Falling back to CPU for this run."
                    ),
                    old_value="cuda",
                    suggested_value="cpu",
                    risk_points=45,
                    adjusted=auto_harden,
                )
            else:
                recent_cuda = self._recent_cuda_assert_signals(window=5)
                recent_count = int(recent_cuda.get("count") or 0)
                if recent_count >= 3:
                    experiment_ids = [str(x)[:8] for x in (recent_cuda.get("experiment_ids") or []) if str(x)]
                    exp_label = ", ".join(experiment_ids[:5]) if experiment_ids else "recent runs"
                    if auto_harden:
                        screened.device = "cpu"
                    _record_issue(
                        key="device",
                        severity="high",
                        reason=(
                            f"Detected {recent_count} recent CUDA device-side assert failures "
                            f"(e.g., {exp_label}); forcing CPU to avoid repeated 0/0 launch failures."
                        ),
                        old_value="cuda",
                        suggested_value="cpu",
                        risk_points=45,
                        adjusted=auto_harden,
                    )

        if mode_norm in {"continuous", "evolve", "novelty"}:
            _harden_min_int(
                "max_depth", 2, "medium",
                "max_depth must be >= 2 to produce meaningful architectures.",
                10,
            )
            _harden_min_int(
                "max_ops", 4, "medium",
                "max_ops must be >= 4 for search-space viability.",
                10,
            )

        if mode_norm in {"evolve", "novelty"}:
            if screened.max_depth > 12:
                old = screened.max_depth
                if auto_harden:
                    screened.max_depth = 12
                _record_issue(
                    key="max_depth",
                    severity="medium",
                    reason="Capping depth at 12 reduces recursion-overflow risk in search loops.",
                    old_value=old,
                    suggested_value=12,
                    risk_points=12,
                    adjusted=auto_harden,
                )
            if screened.max_ops > 20:
                old = screened.max_ops
                if auto_harden:
                    screened.max_ops = 20
                _record_issue(
                    key="max_ops",
                    severity="medium",
                    reason="Capping max_ops at 20 reduces recursive expansion risk.",
                    old_value=old,
                    suggested_value=20,
                    risk_points=12,
                    adjusted=auto_harden,
                )

            # Complexity-aware search caps: allow exotic multi-lane architectures
            # (MoE, MoD, routing) that need ~10-15 ops while still preventing
            # runaway recursion (hard limits remain at 12/20 in evolution).
            if screened.max_depth > 8:
                old = screened.max_depth
                if auto_harden:
                    screened.max_depth = 8
                _record_issue(
                    key="max_depth",
                    severity="medium",
                    reason="Capping max_depth at 8 for evolve/novelty (allows exotic architectures).",
                    old_value=old,
                    suggested_value=8,
                    risk_points=8,
                    adjusted=auto_harden,
                )
            if screened.max_ops > 12:
                old = screened.max_ops
                if auto_harden:
                    screened.max_ops = 12
                _record_issue(
                    key="max_ops",
                    severity="medium",
                    reason="Capping max_ops at 12 for evolve/novelty (allows exotic architectures).",
                    old_value=old,
                    suggested_value=12,
                    risk_points=8,
                    adjusted=auto_harden,
                )
            _harden_min_int(
                "n_generations", 1, "high",
                "n_generations must be >= 1 for evolution/novelty search.",
                20,
            )

        if mode_norm == "continuous":
            _harden_min_int(
                "max_experiments", 1, "medium",
                "max_experiments must be >= 1 in continuous mode.",
                10,
            )

        # ── Compression / sparse op safeguards ──
        dim = int(getattr(screened, "model_dim", 64))
        if dim % 4 != 0:
            # grouped_linear (g=4), N:M sparse (m=4), semi_structured_2_4
            # all need D divisible by 4 for clean operation
            new_dim = ((dim + 3) // 4) * 4
            if auto_harden:
                screened.model_dim = new_dim
            _record_issue(
                key="model_dim",
                severity="medium",
                reason=(
                    "model_dim not divisible by 4; compression ops "
                    "(grouped_linear, N:M sparse) may produce uneven splits. "
                    f"Rounding up {dim} -> {new_dim}."
                ),
                old_value=dim,
                suggested_value=new_dim,
                risk_points=8,
                adjusted=auto_harden,
            )

        prune_target = float(getattr(screened, "one_shot_pruning_sparsity", 0.5))
        if prune_target < 0.0 or prune_target > 0.95:
            suggested = max(0.0, min(0.95, prune_target))
            if auto_harden:
                screened.one_shot_pruning_sparsity = suggested
            _record_issue(
                key="one_shot_pruning_sparsity",
                severity="medium",
                reason="one-shot pruning sparsity should be in [0.0, 0.95].",
                old_value=prune_target,
                suggested_value=suggested,
                risk_points=8,
                adjusted=auto_harden,
            )

        eval_batches = int(getattr(screened, "one_shot_pruning_eval_batches", 4))
        if eval_batches < 1 or eval_batches > 32:
            suggested = max(1, min(32, eval_batches))
            if auto_harden:
                screened.one_shot_pruning_eval_batches = suggested
            _record_issue(
                key="one_shot_pruning_eval_batches",
                severity="low",
                reason="one-shot pruning eval batch count should be in [1, 32].",
                old_value=eval_batches,
                suggested_value=suggested,
                risk_points=4,
                adjusted=auto_harden,
            )

        risk_score = min(risk_score, 100)
        if risk_score >= 60:
            risk_level = "high"
        elif risk_score >= 25:
            risk_level = "medium"
        else:
            risk_level = "low"

        report = {
            "checked": True,
            "mode": mode_norm,
            "auto_hardened": bool(auto_harden),
            "issues": issues,
            "adjustments": adjustments,
            "issue_count": len(issues),
            "adjustment_count": len(adjustments),
            "risk_score": risk_score,
            "risk_level": risk_level,
        }
        return screened, report

    def _check_continuous_limits(self, config: RunConfig, t_start: float,
                                  n_experiments: int) -> Optional[str]:
        """Check if any continuous mode limit has been reached.

        Returns a stop reason string, or None to continue.
        """
        if n_experiments >= config.max_experiments:
            return f"Reached experiment limit ({config.max_experiments})"
        effective_max_time_minutes = self._effective_max_time_minutes(config)
        if effective_max_time_minutes > 0:
            elapsed_min = (time.time() - t_start) / 60
            if elapsed_min >= effective_max_time_minutes:
                return f"Time limit reached ({effective_max_time_minutes} min)"
        if config.max_cost_dollars > 0:
            cost = self.aria.total_cost
            if cost >= config.max_cost_dollars:
                return f"Cost limit reached (${cost:.2f} / ${config.max_cost_dollars:.2f})"

        return None

    def _detect_plateau(self, n_experiments: int) -> Optional[str]:
        """Detect when continuous mode has plateaued.

        Triggers when the last N cycles produced zero new S1 survivors
        and the mode recommendations are repeating.  Returns a stop reason
        string or None to continue.
        """
        if n_experiments < self._PLATEAU_MIN_CYCLES:
            return None

        with self._lock:
            recent = list(self._aria_cycle_history[-self._PLATEAU_WINDOW:])

        if len(recent) < self._PLATEAU_WINDOW:
            return None

        # Check if any recent cycle produced new S1 survivors
        total_delta_s1 = sum(
            int(c.get("delta_stage1_survivors") or 0) for c in recent
        )
        if total_delta_s1 > 0:
            return None  # Still making progress

        # Check mode diversity — if all recent cycles used the same mode, we're stuck
        recent_modes = [c.get("mode", "synthesis") for c in recent]
        unique_modes = set(recent_modes)

        # Check for repeated errors
        error_count = sum(1 for c in recent if c.get("error"))
        if error_count >= self._PLATEAU_WINDOW - 1:
            return (
                f"Plateau: {error_count}/{self._PLATEAU_WINDOW} recent cycles "
                f"failed with errors. Pausing to avoid wasting resources."
            )

        if total_delta_s1 == 0:
            mode_str = ", ".join(recent_modes)
            return (
                f"Plateau: 0 new S1 survivors in last {self._PLATEAU_WINDOW} "
                f"cycles (modes: {mode_str}). "
                f"Pausing — try adjusting grammar weights, config, or hypothesis."
            )

