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
import queue
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..synthesis.grammar import GrammarConfig, generate_layer_graph, batch_generate
from ..synthesis.compiler import compile_model
from ..synthesis.validator import validate_graph
from ..synthesis.serializer import graph_to_json, graph_from_json, graph_summary
from ..eval.sandbox import safe_eval
from ..eval.metrics import novelty_score
from ..eval.flops import estimate_flops
from ..eval.baseline import TransformerBaseline
from ..eval.fingerprint import compute_fingerprint
from ..eval.diagnostic_tasks import run_diagnostic_suite
from ..eval.pruning import apply_one_shot_pruning, estimate_lm_ce_loss
from ..training.loss_synthesis import synthesize_loss
from ..training.optimizer_synthesis import synthesize_optimizer
from ..training.training_program import synthesize_training_program
from ..training.data_pipeline import CorpusConfig, CorpusTokenBatcher
from ..training.checkpointing import CheckpointManager
from .persona import Aria, get_aria
from .notebook import LabNotebook, ExperimentEntry
from .llm.context import (build_experiment_context,
                          build_rich_context, build_investigation_context,
                          build_validation_context, build_mode_selection_context,
                          build_hypothesis_context, build_go_no_go_context,
                          build_knowledge_extraction_context,
                          build_campaign_report_context,
                          build_campaign_formulation_context,
                          build_manual_start_fallback_context)

import logging
logger = logging.getLogger(__name__)


@dataclass
class ModelCandidate:
    """Unified representation of a candidate model from any source."""
    source: str  # "graph_synthesis" or "morphological_box"
    model: nn.Module
    description: str
    # Source-specific data
    graph: Optional[Any] = None
    graph_json: Optional[str] = None
    arch_spec: Optional[Any] = None  # ArchSpec
    arch_spec_json: Optional[str] = None
    fingerprint: str = ""


@dataclass
class RunConfig:
    """Configuration for an experiment run."""
    n_programs: int = 100
    model_dim: int = 256
    n_layers: int = 4
    vocab_size: int = 32000
    max_seq_len: int = 256
    device: str = "cuda"
    # Stage 1 training
    stage1_steps: int = 500
    stage1_lr: float = 3e-4
    stage1_batch_size: int = 4
    # Training data source
    data_mode: str = "random"  # "random" | "corpus" | "hydra"
    corpus_path: str = ""      # TXT or JSONL path for corpus mode
    corpus_format: str = "auto"  # "auto" | "txt" | "jsonl"
    corpus_text_key: str = "text"  # JSONL key when format is jsonl
    tokenizer_mode: str = "byte"  # "byte" | "whitespace"
    corpus_max_chars: int = 200000
    # HYDRA data loader settings (data_mode="hydra")
    hydra_data_dir: str = "/home/tim/Projects/LLM/HYDRA/data"
    hydra_dataset: str = "local_jsonl"  # any HYDRA dataset name
    hydra_project_root: str = "/home/tim/Projects/LLM/HYDRA"
    # Synthesis grammar
    max_depth: int = 10
    max_ops: int = 16
    math_space_weight: float = 2.0
    residual_prob: float = 0.7
    # Continuous mode
    continuous: bool = False
    max_experiments: int = 100
    rest_between_experiments: int = 5  # seconds
    control_experiment_interval: int = 5  # run every Nth synthesis as control (0 disables)
    max_time_minutes: int = 0        # 0 = no limit
    max_cost_dollars: float = 0.0    # 0 = no limit (estimated LLM API cost)
    # Evolution search
    population_size: int = 50
    n_generations: int = 20
    tournament_size: int = 5
    mutation_rate: float = 0.7
    crossover_rate: float = 0.3
    elitism: int = 5
    novelty_weight: float = 0.5
    fitness_weight: float = 0.5
    # Novelty search
    archive_size: int = 200
    k_nearest: int = 15
    archive_threshold: float = 0.3
    # Scale-up mode
    scale_up: bool = False
    scale_up_result_ids: str = ""   # comma-separated result IDs
    scale_up_steps: int = 5000      # 10x default 500
    scale_up_batch_size: int = 8    # 2x default 4
    scale_up_seq_len: int = 512     # 2x default 256
    # One-shot pruning baseline
    one_shot_pruning_baseline: bool = False
    one_shot_pruning_method: str = "wanda"  # wanda | sparsegpt
    one_shot_pruning_sparsity: float = 0.5
    one_shot_pruning_eval_batches: int = 4
    one_shot_pruning_batch_size: int = 2
    # Automation
    auto_scale_up: bool = True         # auto-trigger scale-up when criteria met
    auto_scale_up_min_survivors: int = 3  # min S1 survivors to trigger
    auto_scale_up_min_novelty: float = 0.5  # min avg novelty of survivors
    auto_scale_up_top_n: int = 5       # how many to scale up
    auto_report: bool = True           # auto-generate report at session end
    auto_report_every_n: int = 5       # also generate report every N experiments (continuous)
    # Model source
    model_source: str = "graph_synthesis"  # "graph_synthesis", "morphological_box", "mixed"
    morph_ratio: float = 0.5           # fraction of morphological candidates in mixed mode
    # Training program variation
    use_synthesized_training: bool = False  # use random training programs
    n_training_programs: int = 3       # how many to try per candidate (investigation)
    # Investigation phase
    investigation_steps: int = 2500
    investigation_batch_size: int = 4
    investigation_max_loss_ratio_multiplier: float = 8.0
    # Validation phase
    validation_steps: int = 10000
    validation_batch_size: int = 8
    validation_seq_len: int = 512
    validation_n_seeds: int = 5
    # Auto-escalation pipeline
    auto_investigate: bool = True
    auto_investigate_min_survivors: int = 1
    auto_investigate_top_n: int = 5
    auto_validate: bool = True
    auto_validate_min_robustness: float = 0.5
    auto_validate_max_baseline_ratio: float = 0.90
    auto_validate_min_novelty_confidence: float = 0.50
    auto_validate_top_n: int = 3
    # Checkpoint/resume
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1  # save continuous checkpoint every N experiments
    resume_experiment_id: str = ""  # experiment ID to resume (empty = fresh start)
    keep_checkpoints: bool = False  # keep checkpoints after successful completion
    # Campaign system
    enable_campaigns: bool = True
    knowledge_extraction_interval: int = 3  # every N experiments
    auto_go_no_go: bool = True  # auto-record go/no-go decisions at escalation

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict) -> RunConfig:
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class LiveProgress:
    """Real-time progress of a running experiment."""
    experiment_id: str = ""
    status: str = "idle"  # idle, generating, evaluating, training, analyzing, completed, failed, stopped
    current_program: int = 0
    total_programs: int = 0
    stage0_passed: int = 0
    stage05_passed: int = 0
    stage1_passed: int = 0
    novel_count: int = 0
    current_stage: str = ""  # "validating", "stage0", "stage0.5", "stage1", "novelty"
    current_fingerprint: str = ""
    best_loss_ratio: Optional[float] = None
    best_novelty: Optional[float] = None
    elapsed_seconds: float = 0.0
    aria_message: str = ""
    error: Optional[str] = None
    # Limits tracking (continuous mode)
    estimated_cost: float = 0.0
    total_tokens: int = 0
    # Evolution/novelty progress
    current_generation: int = 0
    total_generations: int = 0
    best_fitness: Optional[float] = None
    avg_fitness: Optional[float] = None
    archive_size: int = 0
    # Preflight hypothesis critique
    hypothesis_critique: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


class ExperimentRunner:
    """Autonomous experiment execution engine with background support."""

    _ROUTING_BENCHMARK_MODES = [
        "uniform",
        "mod_topk",
        "early_exit",
        "token_merging",
        "moe_topk",
    ]
    _ROUTING_EFFICIENCY_FACTOR = {
        "uniform": 1.0,
        "mod_topk": 0.7,
        "early_exit": 0.75,
        "token_merging": 0.65,
        "moe_topk": 0.8,
    }

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

    def __init__(self, notebook_path: str = "research/lab_notebook.db"):
        self.notebook_path = notebook_path
        self.aria = get_aria()
        self._math_spaces_registered = False
        self._baseline: Optional[TransformerBaseline] = None

        # Background execution state
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._progress = LiveProgress()
        self._event_queue: queue.Queue = queue.Queue(maxsize=500)
        self._lock = threading.Lock()
        self._last_recommendation: Optional[Dict] = None
        self._active_campaign_id: Optional[str] = None
        self._current_hypothesis_id: Optional[str] = None
        self._corpus_batcher: Optional[CorpusTokenBatcher] = None
        self._corpus_signature: Optional[Tuple[str, str, str, str, int, int]] = None
        self._corpus_warned_unavailable: bool = False
        self._hydra_loader = None
        self._hydra_iter = None
        self._hydra_signature: Optional[str] = None
        self._last_cycle_summary: Optional[Dict[str, Any]] = None
        self._aria_cycle_history: List[Dict[str, Any]] = []
        self._aria_cycle_paused: bool = False
        self._aria_cycle_status: Dict[str, Any] = {
            "phase": "idle",
            "phase_label": "Idle",
            "continuous_active": False,
            "cycle_index": 0,
            "selected_mode": None,
            "last_completed_mode": None,
            "last_note": "Awaiting run.",
            "last_transition_ts": time.time(),
        }

        self._recover_stale_experiments_on_startup()

    def _ensure_math_spaces(self):
        if not self._math_spaces_registered:
            try:
                from ..mathspaces.registry import register_all_mathspaces
                register_all_mathspaces()
                self._math_spaces_registered = True
            except Exception as e:
                logger.debug("Math spaces registration failed: %s", e)

    def _get_baseline(self) -> TransformerBaseline:
        if self._baseline is None:
            self._baseline = TransformerBaseline()
        return self._baseline

    def _make_notebook(self) -> LabNotebook:
        """Create a new notebook connection (thread-safe)."""
        return LabNotebook(self.notebook_path)

    def _recover_stale_experiments_on_startup(self) -> None:
        """Clean up stale running experiments from previous crashed processes."""
        try:
            nb = self._make_notebook()
            cleaned = nb.cleanup_stale_experiments(timeout_minutes=60)
            if cleaned > 0:
                logger.info("Recovered %d stale experiments from previous crash", cleaned)
            nb.close()
        except Exception as e:
            logger.debug("Startup stale-experiment recovery failed: %s", e)

    @staticmethod
    def _stable_seed(*parts: Any) -> int:
        """Create a reproducible 31-bit seed from contextual parts."""
        key = "|".join(str(p) for p in parts)
        return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF

    @staticmethod
    def _build_hypothesis_metadata(
        source: str,
        llm_used: bool = False,
        fallback_used: bool = False,
        used_context: bool = False,
        review_status: str = "not_reviewed",
        confidence: Optional[float] = None,
        critique: Any = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "source": source,
            "llm_used": llm_used,
            "fallback_used": fallback_used,
            "used_context": used_context,
            "review_status": review_status,
            "confidence": confidence,
            "critique": critique,
        }
        if extra:
            metadata.update(extra)
        return metadata

    def _get_corpus_batcher(self, config: RunConfig) -> Optional[CorpusTokenBatcher]:
        """Lazily create or reuse corpus batcher for corpus-mode training."""
        signature = (
            str(config.corpus_path or ""),
            str(config.corpus_format or "auto"),
            str(config.corpus_text_key or "text"),
            str(config.tokenizer_mode or "byte"),
            int(config.corpus_max_chars),
            int(config.vocab_size),
        )

        if self._corpus_batcher is not None and self._corpus_signature == signature:
            return self._corpus_batcher

        path = str(config.corpus_path or "").strip()
        if not path:
            self._corpus_batcher = None
            self._corpus_signature = signature
            return None

        batcher = CorpusTokenBatcher(
            CorpusConfig(
                path=path,
                fmt=str(config.corpus_format or "auto"),
                text_key=str(config.corpus_text_key or "text"),
                tokenizer=str(config.tokenizer_mode or "byte"),
                max_chars=int(config.corpus_max_chars),
            ),
            vocab_size=int(config.vocab_size),
        )
        self._corpus_batcher = batcher
        self._corpus_signature = signature
        if not batcher.ready and not self._corpus_warned_unavailable:
            logger.warning(
                "Corpus mode requested but corpus unavailable/too small (path=%s); falling back to random tokens.",
                path,
            )
            self._corpus_warned_unavailable = True
        return batcher

    def _get_hydra_batch(
        self, config: RunConfig, batch_size: int, seq_len: int, dev: torch.device,
    ) -> Optional[torch.Tensor]:
        """Get a batch from HYDRA's universal data loader.

        Lazily initializes the loader. Returns None on failure (caller
        falls back to random tokens).
        """
        sig = f"{config.hydra_data_dir}|{config.hydra_dataset}|{batch_size}|{seq_len}"
        if self._hydra_loader is None or self._hydra_signature != sig:
            try:
                import sys
                hydra_root = config.hydra_project_root
                if hydra_root not in sys.path:
                    sys.path.insert(0, hydra_root)
                from hydra.data import create_universal_loader

                self._hydra_loader = create_universal_loader(
                    dataset=config.hydra_dataset,
                    data_dir=config.hydra_data_dir,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    vocab_size=int(config.vocab_size),
                    device="cpu",  # we move to dev below
                    num_workers=0,  # keep it simple for subprocess safety
                    seed=42,
                )
                self._hydra_iter = iter(self._hydra_loader)
                self._hydra_signature = sig
                logger.info("HYDRA data loader initialized: dataset=%s, dir=%s",
                            config.hydra_dataset, config.hydra_data_dir)
            except Exception as e:
                logger.warning("Failed to initialize HYDRA data loader: %s", e)
                self._hydra_loader = None
                self._hydra_iter = None
                return None

        # Get next batch from iterator
        try:
            batch = next(self._hydra_iter)
        except StopIteration:
            # Reset iterator
            self._hydra_iter = iter(self._hydra_loader)
            try:
                batch = next(self._hydra_iter)
            except StopIteration:
                return None

        input_ids = batch.get("input_ids")
        if input_ids is None:
            return None

        # Project token IDs into model's vocab range if needed
        vocab = int(config.vocab_size)
        if input_ids.max().item() >= vocab:
            input_ids = input_ids % vocab

        return input_ids.to(dev)

    def _sample_training_input_ids(
        self,
        config: RunConfig,
        dev: torch.device,
        batch_size: int,
        seq_len: int,
        seed: int,
    ) -> torch.Tensor:
        """Sample input IDs from configured data source with deterministic seed."""
        mode = str(config.data_mode or "random").strip().lower()
        generator = torch.Generator(device=dev)
        generator.manual_seed(int(seed))

        if mode == "hydra":
            batch = self._get_hydra_batch(config, batch_size, seq_len, dev)
            if batch is not None:
                return batch
            # Fall through to random on failure

        if mode == "corpus":
            batcher = self._get_corpus_batcher(config)
            if batcher is not None:
                batch = batcher.sample_batch(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    generator=generator,
                    device=dev,
                )
                if batch is not None:
                    return batch

        return torch.randint(
            0,
            int(config.vocab_size),
            (batch_size, seq_len),
            device=dev,
            generator=generator,
        )

    def _make_baseline_data_fn(self, config: RunConfig):
        """Build a data_fn for baseline training when using real data.

        Returns (data_fn, data_tag) tuple. data_fn is None for random mode
        (baseline uses its own random tokens). data_tag is a cache key suffix.
        """
        mode = str(config.data_mode or "random").strip().lower()
        if mode == "hydra":
            def data_fn(batch_size, seq_len, dev):
                batch = self._get_hydra_batch(config, batch_size, seq_len, dev)
                if batch is not None:
                    return batch
                return torch.randint(0, config.vocab_size, (batch_size, seq_len), device=dev)
            return data_fn, "hydra"
        return None, "random"

    @property
    def progress(self) -> LiveProgress:
        with self._lock:
            return LiveProgress(**self._progress.to_dict())

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_recommendation(self) -> Optional[Dict]:
        """Last auto-generated recommendation after experiment completion."""
        with self._lock:
            rec = self._last_recommendation
            # Clear after reading so dashboard only shows it once
            self._last_recommendation = None
            return rec

    def _emit_event(self, event_type: str, data: Dict):
        """Push an event for SSE consumers."""
        try:
            self._event_queue.put_nowait({
                "type": event_type,
                "data": data,
                "timestamp": time.time(),
            })
        except queue.Full:
            pass  # drop oldest if full

    @staticmethod
    def _aria_phase_label(phase: str) -> str:
        labels = {
            "idle": "Idle",
            "planning": "Planning",
            "running": "Running",
            "analyzing": "Analyzing",
            "paused": "Paused",
            "stopping": "Stopping",
            "completed": "Completed",
            "failed": "Failed",
        }
        return labels.get(phase, phase.replace("_", " ").title())

    def _set_aria_cycle_phase(
        self,
        phase: str,
        *,
        cycle_index: Optional[int] = None,
        selected_mode: Optional[str] = None,
        note: Optional[str] = None,
        continuous_active: Optional[bool] = None,
        emit_event: bool = True,
    ) -> None:
        """Track Aria's continuous cycle phase for observability APIs/UI."""
        with self._lock:
            payload: Dict[str, Any] = {
                "phase": str(phase or "idle"),
                "phase_label": self._aria_phase_label(str(phase or "idle")),
                "last_transition_ts": time.time(),
            }
            if cycle_index is not None:
                payload["cycle_index"] = int(cycle_index)
            if selected_mode is not None:
                payload["selected_mode"] = str(selected_mode)
            if note is not None:
                payload["last_note"] = str(note)
            if continuous_active is not None:
                payload["continuous_active"] = bool(continuous_active)
            if phase == "running" and selected_mode is not None:
                payload["last_completed_mode"] = None
            if phase in {"analyzing", "completed", "failed"} and selected_mode is not None:
                payload["last_completed_mode"] = str(selected_mode)

            self._aria_cycle_status.update(payload)
            snapshot = dict(self._aria_cycle_status)

        if emit_event:
            self._emit_event("aria_cycle_phase", snapshot)

    def get_aria_cycle_status(self) -> Dict[str, Any]:
        """Return latest Aria cycle status for dashboard/API polling."""
        with self._lock:
            cycle = dict(self._aria_cycle_status)
            progress = self._progress.to_dict()
            last_cycle = dict(self._last_cycle_summary) if self._last_cycle_summary else None
            cycle_history = [dict(item) for item in self._aria_cycle_history[-10:]]
            cycle_paused = bool(self._aria_cycle_paused)
        cycle["is_running"] = self.is_running
        cycle["progress_status"] = progress.get("status")
        cycle["aria_message"] = progress.get("aria_message")
        cycle["experiment_id"] = progress.get("experiment_id")
        cycle["last_cycle_summary"] = last_cycle
        cycle["cycle_history"] = cycle_history
        cycle["cycle_paused"] = cycle_paused
        return cycle

    def pause_aria_cycle(self) -> Dict[str, Any]:
        """Pause continuous cycle progression between experiment iterations."""
        with self._lock:
            self._aria_cycle_paused = True
            running = self.is_running
        note = (
            "Pause requested; pausing before the next cycle."
            if running
            else "Cycle is paused. Start continuous mode to resume execution."
        )
        self._set_aria_cycle_phase(
            "paused",
            continuous_active=running,
            note=note,
        )
        self._emit_event("aria_cycle_paused", {"note": note})
        return self.get_aria_cycle_status()

    def resume_aria_cycle(self) -> Dict[str, Any]:
        """Resume continuous cycle progression."""
        with self._lock:
            self._aria_cycle_paused = False
            running = self.is_running
            cycle_index = int(self._aria_cycle_status.get("cycle_index") or 0)
        self._set_aria_cycle_phase(
            "planning" if running else "idle",
            continuous_active=running,
            cycle_index=cycle_index,
            note="Cycle resumed." if running else "Cycle resumed and awaiting start.",
        )
        self._emit_event("aria_cycle_resumed", {"running": running})
        return self.get_aria_cycle_status()

    def _wait_for_cycle_resume(self, cycle_index: int) -> None:
        """Block between cycles while paused, unless stop is requested."""
        with self._lock:
            paused = bool(self._aria_cycle_paused)
        if not paused:
            return
        self._set_aria_cycle_phase(
            "paused",
            continuous_active=True,
            cycle_index=cycle_index,
            note="Cycle paused; waiting for resume.",
        )
        while not self._stop_event.is_set():
            with self._lock:
                paused = bool(self._aria_cycle_paused)
            if not paused:
                break
            time.sleep(0.5)

    def _build_aria_cycle_summary(
        self,
        *,
        cycle_index: int,
        selected_mode: str,
        mode_reasoning: str,
        mode_confidence: Optional[float],
        before_progress: Dict[str, Any],
        after_progress: Dict[str, Any],
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a compact cycle summary payload for SSE/UI/chat consumers."""
        before_total = int(before_progress.get("total_programs") or 0)
        after_total = int(after_progress.get("total_programs") or 0)
        before_s1 = int(before_progress.get("stage1_passed") or 0)
        after_s1 = int(after_progress.get("stage1_passed") or 0)

        summary = {
            "cycle_index": int(cycle_index),
            "mode": str(selected_mode or "synthesis"),
            "reasoning": str(mode_reasoning or ""),
            "confidence": float(mode_confidence or 0.0),
            "status": "failed" if error else "completed",
            "programs_total": after_total,
            "stage1_survivors": after_s1,
            "delta_programs": max(0, after_total - before_total),
            "delta_stage1_survivors": max(0, after_s1 - before_s1),
            "aria_message": str(after_progress.get("aria_message") or ""),
            "timestamp": time.time(),
        }
        if error:
            summary["error"] = str(error)
        return summary

    def run_aria_cycle(
        self,
        config: RunConfig,
        nb: LabNotebook,
        n_experiments: int,
        t_start: float,
    ) -> Dict[str, Any]:
        """Run one continuous research cycle (plan -> run -> analyze -> summarize)."""
        self._set_aria_cycle_phase(
            "planning",
            continuous_active=True,
            cycle_index=n_experiments,
            selected_mode=None,
            note=f"Planning cycle {n_experiments}.",
        )

        mode_rec = self._select_next_mode(config, nb, n_experiments)
        selected_mode = mode_rec.get("mode", "synthesis")
        mode_reasoning = mode_rec.get("reasoning", "")
        mode_confidence = mode_rec.get("confidence", 0)
        effective_max_time_minutes = self._effective_max_time_minutes(config)

        self._emit_event("mode_selected", {
            "mode": selected_mode,
            "reasoning": mode_reasoning,
            "confidence": mode_confidence,
            "experiment_number": n_experiments,
        })

        limit_info = []
        if config.max_experiments > 0:
            limit_info.append(f"exp {n_experiments}/{config.max_experiments}")
        if effective_max_time_minutes > 0:
            elapsed_min = (time.time() - t_start) / 60
            limit_info.append(f"{elapsed_min:.0f}/{effective_max_time_minutes}min")
        if config.max_cost_dollars > 0:
            limit_info.append(f"${self.aria.total_cost:.2f}/${config.max_cost_dollars:.2f}")
        limit_str = " | ".join(limit_info) if limit_info else f"exp {n_experiments}"

        pending_inv = getattr(self, "_pending_investigation", None)
        pending_val = getattr(self, "_pending_validation", None)

        if pending_inv and selected_mode != "investigation":
            selected_mode = "investigation"
            mode_reasoning = pending_inv.get("hypothesis", "Auto-investigation")
            mode_confidence = 0.9
            self._pending_investigation = None
            self._emit_event("mode_selected", {
                "mode": "investigation",
                "reasoning": "Auto-escalation: S1 survivors qualify for investigation",
                "confidence": 0.9,
                "experiment_number": n_experiments,
            })
        elif pending_val and selected_mode != "validation":
            selected_mode = "validation"
            mode_reasoning = pending_val.get("hypothesis", "Auto-validation")
            mode_confidence = 0.9
            self._pending_validation = None
            self._emit_event("mode_selected", {
                "mode": "validation",
                "reasoning": "Auto-escalation: investigation survivors qualify for validation",
                "confidence": 0.9,
                "experiment_number": n_experiments,
            })

        before_progress = self.progress.to_dict()
        cycle_error: Optional[str] = None

        try:
            self._set_aria_cycle_phase(
                "running",
                continuous_active=True,
                cycle_index=n_experiments,
                selected_mode=selected_mode,
                note=f"Running {selected_mode} cycle {n_experiments}.",
            )
            if selected_mode in ("investigation", "validation"):
                self._run_continuous_phase(
                    selected_mode, config, nb, n_experiments,
                    limit_str, mode_reasoning)
            elif selected_mode == "evolution":
                self._run_continuous_evolution(
                    config, nb, n_experiments, limit_str, mode_reasoning)
            elif selected_mode == "novelty":
                self._run_continuous_novelty(
                    config, nb, n_experiments, limit_str, mode_reasoning)
            else:
                self._run_continuous_synthesis(
                    config, nb, n_experiments, limit_str, mode_reasoning)
        except Exception as e:
            cycle_error = str(e)
            logger.warning(f"Continuous mode {selected_mode} failed: {e}")
            failed_exp_id = self._fail_active_cycle_experiment(
                nb,
                cycle_error,
                expected_mode=selected_mode,
            )
            self._emit_event("experiment_failed", {
                "experiment_number": n_experiments,
                "mode": selected_mode,
                "experiment_id": failed_exp_id,
                "error": cycle_error,
            })
        finally:
            if not self._stop_event.is_set():
                self._set_aria_cycle_phase(
                    "analyzing",
                    continuous_active=True,
                    cycle_index=n_experiments,
                    selected_mode=selected_mode,
                    note="Analyzing outcomes and preparing next recommendation.",
                )

        after_progress = self.progress.to_dict()
        summary = self._build_aria_cycle_summary(
            cycle_index=n_experiments,
            selected_mode=selected_mode,
            mode_reasoning=mode_reasoning,
            mode_confidence=mode_confidence,
            before_progress=before_progress,
            after_progress=after_progress,
            error=cycle_error,
        )
        try:
            nb.add_entry(ExperimentEntry(
                entry_type="live_feed",
                experiment_id=after_progress.get("experiment_id") or None,
                title=f"Aria cycle {n_experiments}: {selected_mode}",
                content=(
                    f"Cycle {n_experiments} completed in {selected_mode} mode "
                    f"with ΔS1={summary.get('delta_stage1_survivors', 0)}"
                ),
                metadata={
                    "live_feed_type": "aria_cycle",
                    "payload": summary,
                },
            ))
        except Exception as e:
            logger.debug("Failed to persist aria_cycle live-feed entry: %s", e)
        with self._lock:
            self._last_cycle_summary = summary
            self._aria_cycle_history.append(summary)
            if len(self._aria_cycle_history) > 50:
                self._aria_cycle_history = self._aria_cycle_history[-50:]
        self._emit_event("aria_cycle_completed", summary)
        return summary

    def get_events(self, timeout: float = 30.0):
        """Generator yielding events for SSE streaming."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                event = self._event_queue.get(timeout=1.0)
                yield event
            except queue.Empty:
                # Send keepalive
                yield {"type": "keepalive", "data": {}, "timestamp": time.time()}

    # ── Start / Stop ──

    def start_experiment(self, config: RunConfig,
                         hypothesis: Optional[str] = None) -> str:
        """Start an experiment in a background thread. Returns experiment ID."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()
        self._set_aria_cycle_phase(
            "idle",
            continuous_active=False,
            cycle_index=0,
            selected_mode=None,
            note="Single-run experiment started.",
            emit_event=False,
        )

        # Pre-generate experiment ID
        nb = self._make_notebook()
        hypothesis_metadata = {
            "source": "user_input" if hypothesis is not None else "unknown",
            "llm_used": False,
            "fallback_used": False,
            "used_context": False,
            "review_status": "not_reviewed",
            "confidence": None,
            "critique": None,
        }
        if hypothesis is None:
            context = self._build_start_experiment_hypothesis_context(nb, config)
            llm_available = self.aria._get_llm() is not None
            if llm_available and not (context or "").strip():
                context = build_manual_start_fallback_context(config.to_dict())
            result = None
            if context:
                result = self.aria.formulate_hypothesis(
                    context=context,
                    return_metadata=True,
                )
                hypothesis_metadata["used_context"] = True
            else:
                result = self.aria.formulate_hypothesis(return_metadata=True)

            if isinstance(result, tuple):
                hypothesis, meta = result
                hypothesis_metadata.update(meta or {})
            else:
                hypothesis = result
                hypothesis_metadata["source"] = (
                    "rule_based_fallback" if context else "rule_based"
                )

            if context:
                hypothesis_metadata["context_char_count"] = len(context)

        # Preflight hypothesis critique
        critique = None
        if hypothesis:
            try:
                critique_context = self._build_start_experiment_hypothesis_context(
                    nb, config,
                ) if hypothesis_metadata.get("source") == "user_input" else ""
                critique = self.aria.critique_hypothesis(
                    hypothesis, context=critique_context,
                )
                hypothesis_metadata["preflight_critique"] = critique
                hypothesis_metadata["critique"] = critique
                hypothesis_metadata["critique_confidence"] = critique.get("confidence")
                hypothesis_metadata["review_status"] = f"preflight_{critique.get('gate', 'warn')}"
            except Exception as e:
                logger.warning(f"Hypothesis critique failed: {e}")

        exp_id = nb.start_experiment(
            experiment_type="synthesis",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=hypothesis_metadata,
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=config.n_programs,
                aria_message=self.aria.greet(),
                hypothesis_critique=critique,
            )

        self._emit_event("experiment_started", {
            "experiment_id": exp_id,
            "hypothesis": hypothesis,
            "config": config.to_dict(),
            "aria_greeting": self.aria.greet(),
            "hypothesis_critique": critique,
        })

        self._thread = threading.Thread(
            target=self._run_experiment_thread,
            args=(exp_id, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def _build_start_experiment_hypothesis_context(
        self, nb: LabNotebook, config: RunConfig,
    ) -> str:
        """Build context for hypothesis generation in manual start_experiment.

        Ensures manual starts use the same context-aware hypothesis pathway as
        continuous mode whenever history/analytics are available.
        """
        try:
            recent = nb.get_recent_experiments(10)
            leaderboard = nb.get_leaderboard(limit=20)
            analytics_data = self._gather_analytics_data(nb)
            context = build_mode_selection_context(
                recent_experiments=recent,
                leaderboard=leaderboard,
                analytics_data=analytics_data,
                current_mode="synthesis",
                n_experiments_in_session=len(recent),
                cost_spent=self.aria.total_cost,
                budget=config.max_cost_dollars,
            )
            if config.max_cost_dollars > 0:
                context += (f"\n\nBudget: ${self.aria.total_cost:.2f} spent "
                            f"of ${config.max_cost_dollars:.2f}")
            return context
        except Exception as e:
            logger.debug("Failed to build manual hypothesis context: %s", e)
            return build_manual_start_fallback_context(config.to_dict())

    def start_continuous(self, config: RunConfig) -> str:
        """Start continuous experiment mode in background."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()
        with self._lock:
            self._aria_cycle_paused = False

        config.continuous = True
        self._set_aria_cycle_phase(
            "planning",
            continuous_active=True,
            cycle_index=0,
            selected_mode=None,
            note="Continuous session initialized.",
        )

        with self._lock:
            self._progress = LiveProgress(
                status="generating",
                aria_message=f"{self.aria.NAME} entering continuous research mode...",
            )

        self._thread = threading.Thread(
            target=self._run_continuous_thread,
            args=(config,),
            daemon=True,
        )
        self._thread.start()
        return "continuous"

    def start_resume(self, experiment_id: str, config: Optional[RunConfig] = None) -> str:
        """Resume an interrupted experiment from its last checkpoint.

        Looks up the experiment in the notebook, reconstructs config if needed,
        and dispatches to the appropriate thread based on experiment type.
        """
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()
        exp_data = nb.get_resumable_experiment(experiment_id)
        if exp_data is None:
            nb.close()
            raise ValueError(
                f"Experiment {experiment_id} not found or not resumable "
                "(must be 'running' or 'failed')")

        exp_type = exp_data["experiment_type"]
        hypothesis = exp_data.get("hypothesis", "")

        # Reconstruct config from stored config_json
        if config is None:
            try:
                config_dict = json.loads(exp_data["config_json"])
                config = RunConfig.from_dict(config_dict)
            except Exception:
                nb.close()
                raise ValueError(
                    f"Cannot reconstruct config for experiment {experiment_id}")

        config.resume_experiment_id = experiment_id

        # Mark experiment as running again if it was failed
        if exp_data["status"] == "failed":
            nb.conn.execute(
                "UPDATE experiments SET status = 'running' WHERE experiment_id = ?",
                (experiment_id,),
            )
            nb.conn.commit()
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=experiment_id,
                status="resuming",
                aria_message=f"Resuming {exp_type} experiment {experiment_id}...",
            )

        self._emit_event("experiment_resuming", {
            "experiment_id": experiment_id,
            "experiment_type": exp_type,
        })

        if exp_type == "continuous" or config.continuous:
            self._thread = threading.Thread(
                target=self._run_continuous_thread,
                args=(config,),
                daemon=True,
            )
        else:
            logger.warning("Resume for experiment type '%s' not yet supported, "
                           "falling back to continuous", exp_type)
            config.continuous = True
            self._thread = threading.Thread(
                target=self._run_continuous_thread,
                args=(config,),
                daemon=True,
            )

        self._thread.start()
        return experiment_id

    def stop(self):
        """Stop the current experiment gracefully."""
        self._stop_event.set()
        self.aria.state.mood = "contemplative"
        with self._lock:
            self._aria_cycle_paused = False
        self._set_aria_cycle_phase(
            "stopping",
            continuous_active=self.is_running,
            note="Stop requested; wrapping up current work.",
        )
        with self._lock:
            self._progress.status = "stopped"
            self._progress.aria_message = "Stopping... wrapping up current evaluation."
        self._emit_event("experiment_stopping", {})

    # ── Routing Benchmark Harness (Track C) ──

    @staticmethod
    def _routing_stability_from_curve(training_curve: List[Dict[str, Any]]) -> Optional[float]:
        """Compute a simple stability score from per-step loss trajectory."""
        if not training_curve:
            return None
        losses = [float(row.get("loss")) for row in training_curve if row.get("loss") is not None]
        if len(losses) < 2:
            return None
        tail = losses[max(0, len(losses) // 2):]
        if len(tail) < 2:
            return None
        mean_loss = sum(tail) / len(tail)
        if mean_loss <= 1e-8:
            return 1.0
        variance = sum((v - mean_loss) ** 2 for v in tail) / len(tail)
        std = variance ** 0.5
        cv = std / mean_loss
        return 1.0 / (1.0 + cv)

    def run_routing_benchmark(
        self,
        config: RunConfig,
        seed_set: Optional[List[int]] = None,
        modes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run fixed-budget routing benchmark across compute-routing strategies.

        Compares routing modes on identical architecture skeleton, seed set, and
        step budget. Returns compact frontier points and raw per-run metrics.
        """
        from ..morphological_box import roll
        from ..arch_builder import build_model, BuildConfig

        requested_modes = modes or list(self._ROUTING_BENCHMARK_MODES)
        supported_modes = [m for m in requested_modes if m in self._ROUTING_BENCHMARK_MODES]
        seeds = seed_set or [101, 202, 303]
        if not supported_modes:
            return {
                "available": False,
                "reason": "No supported routing modes requested",
                "modes_requested": requested_modes,
                "seed_set": seeds,
                "points": [],
                "raw_runs": [],
            }

        dev_str = config.device
        if dev_str == "cuda" and not torch.cuda.is_available():
            dev_str = "cpu"
        dev = torch.device(dev_str)

        fixed_base = {
            "token_representation": "dense_float",
            "weight_storage": "dense_matrix",
            "token_mixing": "softmax_attention",
            "channel_mixing": "swiglu_mlp",
            "topology": "sequential",
            "normalization": "rmsnorm_pre",
            "positional_encoding": "rope",
        }

        bench_config = RunConfig.from_dict(config.to_dict())
        if bench_config.stage1_steps <= 0:
            bench_config.stage1_steps = 1

        raw_runs: List[Dict[str, Any]] = []
        for routing_mode in supported_modes:
            fixed = dict(fixed_base)
            fixed["compute_routing"] = routing_mode

            for seed in seeds:
                if self._stop_event.is_set():
                    break

                run_data: Dict[str, Any] = {
                    "routing_mode": routing_mode,
                    "seed": int(seed),
                    "status": "ok",
                }
                try:
                    spec = roll(seed=int(seed), fixed=fixed)
                    model = build_model(
                        spec,
                        BuildConfig(
                            dim=int(bench_config.model_dim),
                            n_layers=int(bench_config.n_layers),
                            vocab_size=int(bench_config.vocab_size),
                            max_seq_len=int(bench_config.max_seq_len),
                        ),
                    )
                    train_result = self._micro_train(
                        model=model,
                        config=bench_config,
                        dev=dev,
                        seed=int(seed),
                    )

                    seq_len = min(128, int(bench_config.max_seq_len))
                    n_steps = int(train_result.get("n_train_steps") or bench_config.stage1_steps)
                    batch_size = int(bench_config.stage1_batch_size)
                    tokens_total = batch_size * seq_len * n_steps
                    eff_factor = float(self._ROUTING_EFFICIENCY_FACTOR.get(routing_mode, 1.0))

                    run_data.update({
                        "validation_loss": train_result.get("final_loss"),
                        "tokens_per_sec": train_result.get("throughput"),
                        "routing_stability": self._routing_stability_from_curve(
                            train_result.get("training_curve") or []
                        ),
                        "tokens_total": tokens_total,
                        "effective_token_compute": tokens_total * eff_factor,
                        "loss_ratio": train_result.get("loss_ratio"),
                    })

                    del model
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as exc:
                    run_data["status"] = "error"
                    run_data["error"] = str(exc)

                raw_runs.append(run_data)

        points: List[Dict[str, Any]] = []
        for routing_mode in supported_modes:
            mode_runs = [
                row for row in raw_runs
                if row.get("routing_mode") == routing_mode and row.get("status") == "ok"
            ]
            if not mode_runs:
                continue

            def _mean(key: str) -> Optional[float]:
                vals = [float(r[key]) for r in mode_runs if r.get(key) is not None]
                return (sum(vals) / len(vals)) if vals else None

            points.append({
                "routing_mode": routing_mode,
                "n_runs": len(mode_runs),
                "validation_loss": _mean("validation_loss"),
                "tokens_per_sec": _mean("tokens_per_sec"),
                "effective_token_compute": _mean("effective_token_compute"),
                "routing_stability": _mean("routing_stability"),
            })

        return {
            "available": len(points) > 0,
            "seed_set": seeds,
            "modes_requested": requested_modes,
            "modes_evaluated": [p["routing_mode"] for p in points],
            "points": points,
            "raw_runs": raw_runs,
            "benchmark_config": {
                "stage1_steps": int(bench_config.stage1_steps),
                "stage1_batch_size": int(bench_config.stage1_batch_size),
                "max_seq_len": int(bench_config.max_seq_len),
                "data_mode": str(bench_config.data_mode),
            },
        }

    # ── Background Threads ──

    def _run_experiment_thread(self, exp_id: str, config: RunConfig,
                                hypothesis: str):
        """Execute a single experiment in background."""
        nb = self._make_notebook()
        try:
            results = self._execute_experiment(exp_id, config, nb)
            self._persist_applied_grammar_weights(nb, exp_id, results)

            # Build rich context for LLM-enhanced methods
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)

            summary = self.aria.experiment_summary(results, context=context)
            insights = self._analyze_results(results, exp_id, nb, context=context)

            # Store LLM analysis if available
            llm_analysis = self.aria.analyze_results(results, context=context)

            # Validate hypothesis
            try:
                validation = self.aria.validate_hypothesis(hypothesis, results, context)
                if validation:
                    nb.add_entry(ExperimentEntry(
                        entry_type="analysis",
                        title="Hypothesis Validation",
                        content=validation.get("explanation", ""),
                        experiment_id=exp_id,
                        metadata={"validated": validation.get("validated", False)},
                    ))
            except Exception as e:
                logger.warning("Hypothesis validation logging failed: %s", e)

            nb.complete_experiment(
                experiment_id=exp_id,
                results=results,
                aria_summary=summary,
                aria_mood=self.aria.state.mood,
                insights=insights,
                llm_analysis=llm_analysis,
            )

            # Update op success rates after experiment
            nb.update_op_success_rates(exp_id)

            # Auto-recommend next experiment
            self._auto_recommend(results, config, hypothesis, nb)

            # Auto-escalation pipeline (investigation/validation)
            results["experiment_id"] = exp_id
            self._auto_escalate(results, config, nb, phase="screening")

            # Auto-scale-up if criteria met (legacy, kept for backward compat)
            self._maybe_auto_scale_up(results, config, nb)

            # Auto-report for single experiments
            self._maybe_auto_report(config, nb, reason="experiment_complete")

            with self._lock:
                self._progress.status = "completed"
                self._progress.aria_message = summary.split("\n")[-1] if summary else "Experiment complete."

            self._emit_event("experiment_completed", {
                "experiment_id": exp_id,
                "results": results,
                "summary": summary,
            })

        except Exception as e:
            error = traceback.format_exc()
            nb.fail_experiment(exp_id, str(e))
            with self._lock:
                self._progress.status = "failed"
                self._progress.error = str(e)
                self._progress.aria_message = self.aria.react_to_failure(str(e))
            self._emit_event("experiment_failed", {
                "experiment_id": exp_id,
                "error": str(e),
            })
        finally:
            nb.close()
            # Launch queued auto-scale-up after notebook is closed
            self._run_pending_scale_up()

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

    def _effective_max_time_minutes(self, config: RunConfig) -> int:
        """Resolve effective continuous-session time limit.

        Local LLM backends (e.g., Ollama) are treated as unconstrained by wall-clock
        timeout by default so autonomous research can continue without artificial cutoffs.
        """
        configured_limit = int(getattr(config, "max_time_minutes", 0) or 0)
        if configured_limit <= 0:
            return 0
        if self._uses_local_llm_backend():
            return 0
        return configured_limit

    def _uses_local_llm_backend(self) -> bool:
        """Whether Aria is currently configured to use a local LLM backend."""
        try:
            llm_config = self.aria.get_llm_config()
        except Exception as e:
            logger.debug("Failed to inspect LLM backend for limit policy: %s", e)
            return False
        backend = str((llm_config or {}).get("backend") or "").strip().lower()
        return backend in {"ollama", "local", "lmstudio", "llama.cpp", "llamacpp", "vllm"}

    def _fail_active_cycle_experiment(
        self,
        nb: LabNotebook,
        error: str,
        expected_mode: Optional[str] = None,
    ) -> Optional[str]:
        """Fail the currently tracked cycle experiment to avoid stale `running` rows."""
        active_exp_id = self.progress.experiment_id
        if not active_exp_id:
            return None

        try:
            row = nb.conn.execute(
                "SELECT experiment_type, status FROM experiments WHERE experiment_id = ?",
                (active_exp_id,),
            ).fetchone()
            if not row or row["status"] != "running":
                return None
            if expected_mode and row["experiment_type"] != expected_mode:
                logger.debug(
                    "Skip failing experiment %s: type mismatch (%s != %s)",
                    active_exp_id,
                    row["experiment_type"],
                    expected_mode,
                )
                return None

            nb.fail_experiment(active_exp_id, error)
            with self._lock:
                if self._progress.experiment_id == active_exp_id:
                    self._progress.status = "failed"
                    self._progress.error = error
                    self._progress.aria_message = self.aria.react_to_failure(error)
            return active_exp_id
        except Exception as finalize_error:
            logger.warning(
                "Failed to mark active experiment %s as failed: %s",
                active_exp_id,
                finalize_error,
            )
            return None

    def _is_control_experiment(self, config: RunConfig, n_experiments: int) -> bool:
        """Whether this continuous synthesis run should be a control experiment."""
        interval = int(getattr(config, "control_experiment_interval", 0) or 0)
        return interval > 0 and n_experiments > 0 and (n_experiments % interval == 0)

    def _ensure_campaign(self, config: RunConfig, nb: LabNotebook) -> Optional[str]:
        """Ensure an active campaign exists. Create one if needed."""
        if not config.enable_campaigns:
            return None

        # Check for existing active campaign
        active = nb.get_active_campaigns()
        if active:
            self._active_campaign_id = active[0]["campaign_id"]
            return self._active_campaign_id

        # Create new campaign via Aria
        recent = nb.get_recent_experiments(10)
        knowledge = nb.get_knowledge()
        all_campaigns = nb.conn.execute(
            "SELECT * FROM campaigns ORDER BY timestamp DESC LIMIT 5"
        ).fetchall()
        previous = [dict(r) for r in all_campaigns]

        context = build_campaign_formulation_context(
            recent_experiments=recent,
            knowledge=knowledge,
            previous_campaigns=previous,
        )
        camp_data = self.aria.formulate_campaign(context=context)
        post_hoc_note = (
            "\n\n[POST-HOC] Success criteria were formulated after reviewing "
            "recent experiment outcomes; treat claims as exploratory until "
            "prospective criteria are pre-registered."
        )
        campaign_id = nb.create_campaign(
            title=camp_data["title"],
            objective=camp_data["objective"],
            success_criteria=f"{camp_data['success_criteria']}{post_hoc_note}",
        )
        self._active_campaign_id = campaign_id
        self._emit_event("campaign_created", {
            "campaign_id": campaign_id,
            "title": camp_data["title"],
            "objective": camp_data["objective"],
        })
        logger.info(f"Campaign created: {camp_data['title']} ({campaign_id})")
        return campaign_id

    def _maybe_evaluate_campaign(self, config: RunConfig, nb: LabNotebook) -> None:
        """Evaluate campaign success criteria after an experiment.

        Auto-completes the campaign if criteria are met or the campaign is
        stale (10+ experiments with no criteria passing).  When a campaign
        completes, a successor campaign is formulated based on pipeline state.
        """
        if not config.enable_campaigns or not self._active_campaign_id:
            return

        try:
            evaluation = nb.evaluate_campaign_criteria(self._active_campaign_id)

            if not evaluation["all_met"] and not evaluation["stale"]:
                return  # still in progress

            campaign = nb.get_campaign(self._active_campaign_id)
            if not campaign or campaign.get("status") != "active":
                return

            # --- Complete the campaign ---
            if evaluation["all_met"]:
                reason = "criteria_met"
                findings = (
                    f"All {evaluation['n_criteria']} success criteria met. "
                    f"{evaluation['n_passing']} criteria passing."
                )
            else:
                reason = "stale"
                findings = (
                    f"Campaign stale after {len(nb.get_campaign_experiments(self._active_campaign_id))} "
                    f"experiments: {evaluation['n_at_risk']} criteria at risk, "
                    f"{evaluation['n_passing']} passing."
                )

            nb.update_campaign(
                self._active_campaign_id,
                status="completed",
                completed_at=time.time(),
                completion_reason=reason,
                findings_summary=findings,
            )

            self._emit_event("campaign_completed", {
                "campaign_id": self._active_campaign_id,
                "title": campaign.get("title", ""),
                "reason": reason,
                "findings": findings,
            })
            logger.info(
                f"Campaign completed ({reason}): "
                f"{campaign.get('title', '')} ({self._active_campaign_id})"
            )

            # --- Formulate successor campaign ---
            completed_id = self._active_campaign_id
            self._active_campaign_id = None

            # Determine next focus from pipeline state
            leaderboard_rows = nb.conn.execute(
                "SELECT tier, COUNT(*) as cnt FROM leaderboard GROUP BY tier"
            ).fetchall()
            tiers = {r["tier"]: r["cnt"] for r in leaderboard_rows}

            recent = nb.get_recent_experiments(10)
            knowledge = nb.get_knowledge()
            all_campaigns = nb.conn.execute(
                "SELECT * FROM campaigns ORDER BY timestamp DESC LIMIT 5"
            ).fetchall()
            previous = [dict(r) for r in all_campaigns]

            # Build context that includes pipeline state for Aria
            from .llm.context import build_campaign_formulation_context
            context = build_campaign_formulation_context(
                recent_experiments=recent,
                knowledge=knowledge,
                previous_campaigns=previous,
            )
            pipeline_hint = (
                f"\n\nPipeline state: "
                f"{tiers.get('screening', 0)} screening, "
                f"{tiers.get('investigation', 0)} investigation, "
                f"{tiers.get('validation', 0)} validation, "
                f"{tiers.get('breakthrough', 0)} breakthrough. "
            )
            if reason == "criteria_met":
                pipeline_hint += (
                    "Previous campaign succeeded — evolve to a more ambitious "
                    "objective (deeper investigation, validation, or scale-up)."
                )
            else:
                pipeline_hint += (
                    "Previous campaign stalled — pivot to a different approach "
                    "(novelty search, different architecture families, or "
                    "relaxed criteria)."
                )

            camp_data = self.aria.formulate_campaign(
                context=context + pipeline_hint
            )

            # Rule-based fallback: evolve based on pipeline state
            if camp_data["title"] == "Architecture Discovery Campaign":
                camp_data = self._pipeline_driven_campaign(tiers, reason)

            successor_id = nb.create_campaign(
                title=camp_data["title"],
                objective=camp_data["objective"],
                success_criteria=camp_data["success_criteria"],
                parent_id=completed_id,
            )

            # Link successor to completed campaign
            nb.update_campaign(
                completed_id,
                successor_campaign_id=successor_id,
            )

            self._active_campaign_id = successor_id
            self._emit_event("campaign_created", {
                "campaign_id": successor_id,
                "title": camp_data["title"],
                "objective": camp_data["objective"],
                "predecessor": completed_id,
            })
            logger.info(
                f"Successor campaign: {camp_data['title']} ({successor_id}) "
                f"→ replacing {completed_id}"
            )

        except Exception as e:
            logger.debug(f"Campaign evaluation failed: {e}")

    @staticmethod
    def _pipeline_driven_campaign(tiers: dict, reason: str) -> dict:
        """Deterministic campaign formulation based on pipeline state."""
        screening = tiers.get("screening", 0)
        investigation = tiers.get("investigation", 0)
        validation = tiers.get("validation", 0)
        breakthrough = tiers.get("breakthrough", 0)

        if breakthrough > 0:
            return {
                "title": "Scale-Up & Generalization",
                "objective": (
                    f"Validate {breakthrough} breakthrough architecture(s) at "
                    f"larger scale (512+ dim, longer sequences) and on diverse "
                    f"data distributions to confirm generalization."
                ),
                "success_criteria": (
                    "Breakthrough architecture maintains loss_ratio < 0.5 at "
                    "model_dim=512; OOD generalization >= 0.67; "
                    "Reproducible across 5+ random seeds with std <= 0.03"
                ),
            }
        elif validation > 0:
            return {
                "title": "Validation & Robustness",
                "objective": (
                    f"Complete multi-seed validation for {validation} candidate(s) "
                    f"and identify which architectures are robust enough for "
                    f"breakthrough consideration."
                ),
                "success_criteria": (
                    "At least 1 candidate passes validation with multi-seed "
                    "std <= 0.03 and baseline_ratio < 0.90; "
                    "Go/no-go decision recorded for each candidate"
                ),
            }
        elif investigation > 0 or screening > 0:
            total_candidates = investigation + screening
            return {
                "title": "Deep Investigation",
                "objective": (
                    f"Investigate {total_candidates} screening/investigation "
                    f"candidate(s) with extended training to identify which "
                    f"architectures warrant full validation."
                ),
                "success_criteria": (
                    "At least 1 candidate passes investigation with "
                    "loss_ratio < 0.6 and robustness > 0.7; "
                    "Clear go/no-go decision for each investigated candidate"
                ),
            }
        elif reason == "stale":
            return {
                "title": "Novelty Exploration",
                "objective": (
                    "Escape the current search region using evolution and "
                    "novelty search to discover fundamentally different "
                    "architecture patterns."
                ),
                "success_criteria": (
                    "Find 3+ architectures with loss_ratio < 0.5 and "
                    "novelty_score > 0.5; Stage-1 survival rate > 5%"
                ),
            }
        else:
            return {
                "title": "Architecture Discovery",
                "objective": (
                    "Discover novel computation patterns by exploring diverse "
                    "op combinations, math spaces, and weight storage techniques."
                ),
                "success_criteria": (
                    "Find 3+ architectures with loss_ratio < 0.5; "
                    "Stage-1 survival rate > 3%; "
                    "At least 1 go/no-go decision recorded"
                ),
            }

    def _maybe_extract_knowledge(self, config: RunConfig, nb: LabNotebook,
                                  n_experiments: int) -> None:
        """Extract knowledge every N experiments."""
        if not config.enable_campaigns:
            return
        if n_experiments <= 0 or n_experiments % config.knowledge_extraction_interval != 0:
            return

        try:
            recent = nb.get_recent_experiments(config.knowledge_extraction_interval)
            resolved = []
            if self._active_campaign_id:
                all_hyps = nb.get_campaign_hypotheses(self._active_campaign_id)
                resolved = [h for h in all_hyps
                           if h.get("status") in ("confirmed", "refuted")]

            context = build_knowledge_extraction_context(recent, resolved)
            entries = self.aria.extract_knowledge(recent, resolved, context=context)

            for entry in entries:
                # Check if knowledge already exists
                existing = nb.search_knowledge(entry.get("title", ""))
                if existing:
                    nb.validate_knowledge(existing[0]["entry_id"])
                else:
                    evidence = [e.get("experiment_id", "") for e in recent[:3]]
                    nb.add_knowledge(
                        category=entry.get("category", "principle"),
                        title=entry.get("title", ""),
                        content=entry.get("content", ""),
                        evidence=evidence,
                        confidence=entry.get("confidence", 0.5),
                    )

            if entries:
                self._emit_event("knowledge_extracted", {
                    "n_entries": len(entries),
                    "categories": list(set(e.get("category", "") for e in entries)),
                })
                logger.info(f"Knowledge extracted: {len(entries)} entries")
        except Exception as e:
            logger.debug(f"Knowledge extraction failed: {e}")

    def _end_of_session_automation(self, config: RunConfig, reason: str):
        """Run end-of-session report and scale-up. Used by both limit-reached and user-stop paths."""
        nb = self._make_notebook()
        try:
            self._maybe_auto_report(config, nb, reason=reason)
            cumulative_results = {"stage1_passed": 0, "survivors": []}
            top = nb.get_top_programs(
                config.auto_scale_up_top_n, sort_by="loss_ratio")
            for p in top:
                if p.get("stage1_passed"):
                    cumulative_results["stage1_passed"] += 1
                    cumulative_results["survivors"].append({
                        "novelty": p.get("novelty_score", 0),
                    })
            self._maybe_auto_scale_up(cumulative_results, config, nb)
        except Exception as e:
            logger.debug(f"End-of-session automation failed: {e}")
        finally:
            nb.close()

    def _run_continuous_thread(self, config: RunConfig):
        """Execute continuous experiments in background."""
        n_experiments = 0
        t_start = time.time()
        self.aria.reset_cost_tracking()
        self._set_aria_cycle_phase(
            "planning",
            continuous_active=True,
            cycle_index=0,
            selected_mode=None,
            note="Preparing continuous research loop.",
        )

        # Initialize checkpoint manager
        ckpt = CheckpointManager(config.checkpoint_dir)
        resume_id = config.resume_experiment_id

        # Resume from checkpoint if requested
        if resume_id:
            ckpt_state = ckpt.load_continuous(resume_id)
            if ckpt_state:
                n_experiments = ckpt_state.get("n_experiments", 0)
                elapsed_prior = ckpt_state.get("elapsed_seconds", 0.0)
                t_start = time.time() - elapsed_prior
                logger.info("Resuming continuous session from checkpoint: "
                            "n_experiments=%d, elapsed=%.0fs",
                            n_experiments, elapsed_prior)
                self._emit_event("checkpoint_resumed", {
                    "experiment_id": resume_id,
                    "n_experiments": n_experiments,
                    "elapsed_seconds": elapsed_prior,
                })

        # Clean up stale experiments from previous interrupted runs
        try:
            cleanup_nb = self._make_notebook()
            n_cleaned = cleanup_nb.cleanup_stale_experiments()
            if n_cleaned:
                logger.info(f"Cleaned up {n_cleaned} stale running experiments")
            cleanup_nb.close()
        except Exception as e:
            logger.debug(f"Stale experiment cleanup failed: {e}")

        # Initialize campaign
        try:
            init_nb = self._make_notebook()
            self._ensure_campaign(config, init_nb)
            init_nb.close()
        except Exception as e:
            logger.debug(f"Campaign init failed: {e}")

        while not self._stop_event.is_set():
            self._wait_for_cycle_resume(n_experiments)
            if self._stop_event.is_set():
                break

            # Check limits before starting next experiment
            stop_reason = self._check_continuous_limits(
                config, t_start, n_experiments)
            if stop_reason:
                self._end_of_session_automation(
                    config, reason=f"continuous_session_end ({stop_reason})")
                self._set_aria_cycle_phase(
                    "completed",
                    continuous_active=False,
                    cycle_index=n_experiments,
                    note=f"Session ended: {stop_reason}",
                )

                with self._lock:
                    self._progress.status = "completed"
                    self._progress.aria_message = f"Session ended: {stop_reason}"
                self._emit_event("continuous_limit_reached", {
                    "reason": stop_reason,
                    "experiments_completed": n_experiments,
                    "elapsed_minutes": (time.time() - t_start) / 60,
                    "estimated_cost": self.aria.total_cost,
                })
                # Launch queued auto-scale-up
                self._run_pending_scale_up()
                return

            n_experiments += 1
            nb = self._make_notebook()
            try:
                self.run_aria_cycle(config, nb, n_experiments, t_start)
            finally:
                nb.close()

            # Update cost in progress
            with self._lock:
                self._progress.estimated_cost = self.aria.total_cost
                self._progress.total_tokens = self.aria.total_tokens

            # Save checkpoint after every checkpoint_interval experiments
            if (config.checkpoint_interval > 0
                    and n_experiments % config.checkpoint_interval == 0):
                try:
                    ckpt_exp_id = resume_id or "continuous"
                    ckpt.save_continuous(
                        experiment_id=ckpt_exp_id,
                        config_dict=config.to_dict(),
                        n_experiments=n_experiments,
                        elapsed_seconds=time.time() - t_start,
                        extra_state={
                            "estimated_cost": self.aria.total_cost,
                            "total_tokens": self.aria.total_tokens,
                        },
                    )
                except Exception as e:
                    logger.debug("Checkpoint save failed: %s", e)

            if config.rest_between_experiments > 0 and not self._stop_event.is_set():
                time.sleep(config.rest_between_experiments)

        # Session ending (user stopped) — auto-report and auto-scale-up
        if n_experiments > 0:
            self._end_of_session_automation(
                config,
                reason=f"continuous_session_stopped (after {n_experiments} experiments)")

        with self._lock:
            elapsed_min = (time.time() - t_start) / 60
            cost_str = f" | Est. cost: ${self.aria.total_cost:.2f}" if self.aria.total_cost > 0 else ""
            self._progress.status = "completed" if not self._stop_event.is_set() else "stopped"
            self._progress.estimated_cost = self.aria.total_cost
            self._progress.total_tokens = self.aria.total_tokens
            self._progress.aria_message = (
                f"Stopped after {n_experiments} experiments ({elapsed_min:.0f}min{cost_str})."
            )
        self._set_aria_cycle_phase(
            "completed" if not self._stop_event.is_set() else "idle",
            continuous_active=False,
            cycle_index=n_experiments,
            note=(
                f"Continuous run finished after {n_experiments} experiments."
                if not self._stop_event.is_set()
                else "Continuous run stopped by user."
            ),
        )

        # Clean up checkpoints on successful completion (unless keep_checkpoints)
        if not self._stop_event.is_set() and not config.keep_checkpoints:
            try:
                ckpt_exp_id = resume_id or "continuous"
                ckpt.cleanup(ckpt_exp_id)
            except Exception as e:
                logger.debug("Checkpoint cleanup failed: %s", e)

        # Launch queued auto-scale-up
        self._run_pending_scale_up()

    def _select_next_mode(self, config: RunConfig, nb: LabNotebook,
                          n_experiments: int) -> Dict:
        """Have Aria decide the next experiment mode."""
        try:
            recent = nb.get_recent_experiments(10)
            leaderboard = nb.get_leaderboard(limit=50)
            analytics_data = self._gather_analytics_data(nb)

            context = build_mode_selection_context(
                recent_experiments=recent,
                leaderboard=leaderboard,
                analytics_data=analytics_data,
                current_mode="synthesis",
                n_experiments_in_session=n_experiments,
                cost_spent=self.aria.total_cost,
                budget=config.max_cost_dollars,
            )

            # Build fallback data for rule-based recommendation
            total_s1 = sum(e.get("n_stage1_passed", 0) for e in recent)
            novelty_scores = [
                e.get("best_novelty_score", 0) for e in recent
                if e.get("best_novelty_score") is not None
            ]
            avg_novelty = (sum(novelty_scores) / len(novelty_scores)
                           if novelty_scores else 0)

            investigation_ready = len([
                e for e in leaderboard
                if e.get("tier") == "screening"
                and e.get("screening_loss_ratio") is not None
                and e["screening_loss_ratio"] < 0.5
            ])
            validation_ready = len([
                e for e in leaderboard
                if e.get("tier") == "investigation"
                and e.get("investigation_robustness") is not None
                and e["investigation_robustness"] >= 0.5
            ])

            # Gather richer analytics for data-driven rule-based recommendation
            recent_modes = [e.get("experiment_type", "synthesis") for e in recent]
            recent_failures = [e for e in recent if e.get("status") == "failed"]
            unique_fingerprints = set()
            for e in leaderboard:
                fp = e.get("graph_fingerprint") or ""
                if fp:
                    unique_fingerprints.add(fp[:8])

            # Optimizer diversity: count distinct optimizers used
            optimizer_counts = {}
            try:
                rows = nb.db.execute(
                    "SELECT optimizer_name, COUNT(*) as cnt "
                    "FROM program_results WHERE optimizer_name IS NOT NULL "
                    "GROUP BY optimizer_name"
                ).fetchall()
                for row in rows:
                    optimizer_counts[row[0]] = row[1]
            except Exception:
                pass  # Table/column may not exist yet

            fallback_data = {
                "total_s1_survivors": total_s1,
                "avg_novelty": avg_novelty,
                "n_experiments_in_session": n_experiments,
                "investigation_ready": investigation_ready,
                "validation_ready": validation_ready,
                "analytics_data": analytics_data,
                "recent_modes": recent_modes,
                "recent_failure_count": len(recent_failures),
                "leaderboard_diversity": len(unique_fingerprints),
                "leaderboard_size": len(leaderboard),
                "optimizer_counts": optimizer_counts,
                "optimizer_diversity": len(optimizer_counts),
            }

            rec = self.aria.recommend_next_mode(
                context=context, fallback_data=fallback_data)

            nb.add_entry(ExperimentEntry(
                entry_type="decision",
                title=f"Mode Selection: {rec.get('mode', 'synthesis')}",
                content=rec.get("reasoning", ""),
                metadata={
                    "mode": rec.get("mode"),
                    "confidence": rec.get("confidence"),
                    "experiment_number": n_experiments,
                },
            ))

            return rec
        except Exception as e:
            logger.debug(f"Mode selection failed, defaulting to synthesis: {e}")
            return {"mode": "synthesis", "reasoning": "Fallback", "confidence": 0.3,
                    "config": {}}

    def _run_continuous_synthesis(self, config: RunConfig, nb: LabNotebook,
                                  n_experiments: int, limit_str: str,
                                  mode_reasoning: str):
        """Run a single synthesis experiment within continuous mode."""
        is_control = self._is_control_experiment(config, n_experiments)

        # Build context so Aria's hypothesis is informed by recent results
        recent = nb.get_recent_experiments(5)
        leaderboard = nb.get_leaderboard(limit=20)
        context = build_mode_selection_context(
            recent_experiments=recent,
            leaderboard=leaderboard,
            current_mode="synthesis",
            n_experiments_in_session=n_experiments,
        )
        if config.max_cost_dollars > 0:
            context += (f"\n\nBudget: ${self.aria.total_cost:.2f} spent "
                        f"of ${config.max_cost_dollars:.2f}")

        # Structured hypothesis (campaign-aware)
        structured_hyp = None
        hypothesis_id = None
        if config.enable_campaigns:
            try:
                knowledge = nb.get_knowledge()
                recent_hyps = []
                if self._active_campaign_id:
                    recent_hyps = nb.get_campaign_hypotheses(
                        self._active_campaign_id)[-5:]
                hyp_context = build_hypothesis_context(
                    campaign=nb.get_campaign(self._active_campaign_id) if self._active_campaign_id else None,
                    recent_hypotheses=recent_hyps,
                    knowledge=knowledge,
                    leaderboard=leaderboard,
                    recent_experiments=recent,
                )
                structured_hyp = self.aria.formulate_structured_hypothesis(
                    context=hyp_context)
                hypothesis = structured_hyp["prediction"]

                # Record structured hypothesis
                # Find parent: last unresolved hypothesis in chain
                parent_id = None
                unresolved = nb.get_unresolved_hypotheses(self._active_campaign_id)
                # Also check if previous hypothesis suggested a follow-up
                if hasattr(self, '_next_follow_up_parent') and self._next_follow_up_parent:
                    parent_id = self._next_follow_up_parent
                    self._next_follow_up_parent = None

                hypothesis_id = nb.record_hypothesis(
                    campaign_id=self._active_campaign_id,
                    prediction=structured_hyp["prediction"],
                    reasoning=structured_hyp["reasoning"],
                    test_method=structured_hyp["test_method"],
                    success_metric=structured_hyp["success_metric"],
                    parent_id=parent_id,
                    confidence=structured_hyp["confidence"],
                    metadata={
                        "source": "structured_hypothesis",
                        "llm_used": True,
                        "fallback_used": False,
                        "used_context": True,
                        "review_status": "not_reviewed",
                        "confidence": structured_hyp.get("confidence"),
                        "critique": structured_hyp.get("critique"),
                    },
                )
                self._current_hypothesis_id = hypothesis_id

                self._emit_event("hypothesis_recorded", {
                    "hypothesis_id": hypothesis_id,
                    "prediction": structured_hyp["prediction"],
                    "confidence": structured_hyp["confidence"],
                    "campaign_id": self._active_campaign_id,
                })
            except Exception as e:
                logger.debug(f"Structured hypothesis failed, using basic: {e}")
                structured_hyp = None

        if structured_hyp is None:
            result = self.aria.formulate_hypothesis(
                context=context,
                return_metadata=True,
            )
            if isinstance(result, tuple):
                hypothesis, basic_hyp_meta = result
            else:
                hypothesis = result
                basic_hyp_meta = {
                    "source": "rule_based_fallback",
                    "llm_used": False,
                    "fallback_used": True,
                    "used_context": True,
                    "review_status": "not_reviewed",
                    "confidence": None,
                    "critique": None,
                }
            hypothesis_metadata = {
                **basic_hyp_meta,
                "context_char_count": len(context),
            }
        else:
            hypothesis_metadata = {
                "source": "structured_hypothesis",
                "llm_used": True,
                "fallback_used": False,
                "used_context": True,
                "review_status": "not_reviewed",
                "confidence": structured_hyp.get("confidence"),
                "critique": structured_hyp.get("critique"),
                "hypothesis_id": hypothesis_id,
            }

        exp_config = config.to_dict()
        if is_control:
            exp_config["control_experiment"] = True
            exp_config["use_learned_grammar_weights"] = False

        exp_id = nb.start_experiment(
            experiment_type="synthesis",
            config=exp_config,
            hypothesis=hypothesis,
            hypothesis_metadata=hypothesis_metadata,
        )

        if is_control:
            nb.log_learning_event(
                "grammar_control_experiment",
                f"Experiment {exp_id} is a control run using default grammar weights",
                evidence=f"interval={config.control_experiment_interval}, experiment_number={n_experiments}",
            )

        # Link experiment to campaign
        if config.enable_campaigns and self._active_campaign_id:
            try:
                nb.conn.execute(
                    "UPDATE experiments SET campaign_id = ? WHERE experiment_id = ?",
                    (self._active_campaign_id, exp_id),
                )
                nb.conn.commit()
            except Exception as e:
                logger.warning("Campaign linking failed for %s: %s", exp_id, e)

        # Link hypothesis to experiment
        if hypothesis_id:
            try:
                nb.conn.execute(
                    "UPDATE hypotheses SET experiment_id = ?, status = 'testing' "
                    "WHERE hypothesis_id = ?",
                    (exp_id, hypothesis_id),
                )
                nb.conn.commit()
            except Exception as e:
                logger.warning("Hypothesis linking failed for %s: %s", exp_id, e)

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=config.n_programs,
                estimated_cost=self.aria.total_cost,
                total_tokens=self.aria.total_tokens,
                aria_message=f"[{limit_str}|synthesis] {hypothesis}",
            )

        self._emit_event("experiment_started", {
            "experiment_id": exp_id,
            "experiment_number": n_experiments,
            "hypothesis": hypothesis,
            "mode": "synthesis",
            "is_control_experiment": is_control,
        })

        # Diversify grammar config based on experiment number
        synth_config = self._diversify_grammar_config(config, n_experiments)

        results = self._execute_experiment(
            exp_id,
            synth_config,
            nb,
            use_learned_grammar=not is_control,
        )
        self._persist_applied_grammar_weights(nb, exp_id, results)

        context = self._build_rich_context_for_experiment(
            results, config, hypothesis, nb)
        summary = self.aria.experiment_summary(results, context=context)
        insights = self._analyze_results(results, exp_id, nb, context=context)
        llm_analysis = self.aria.analyze_results(results, context=context)

        # Structured hypothesis validation
        if structured_hyp and hypothesis_id:
            try:
                validation = self.aria.validate_structured_hypothesis(
                    structured_hyp, results, context=context)
                nb.resolve_hypothesis(
                    hypothesis_id=hypothesis_id,
                    status=validation["status"],
                    evidence=validation["evidence"],
                    summary=validation["explanation"],
                    confidence_after=validation["confidence_after"],
                )
                nb.add_entry(ExperimentEntry(
                    entry_type="analysis",
                    title=f"Hypothesis {validation['status'].upper()}",
                    content=validation["explanation"],
                    experiment_id=exp_id,
                    metadata={
                        "hypothesis_id": hypothesis_id,
                        "status": validation["status"],
                        "confidence_after": validation["confidence_after"],
                    },
                ))
                self._emit_event("hypothesis_resolved", {
                    "hypothesis_id": hypothesis_id,
                    "status": validation["status"],
                    "evidence": validation["evidence"][:200],
                    "confidence_after": validation["confidence_after"],
                })
                # If follow-up suggested, queue it for next experiment
                if validation.get("follow_up"):
                    self._next_follow_up_parent = hypothesis_id
            except Exception as e:
                logger.debug(f"Structured validation failed: {e}")
        else:
            # Fallback to old-style validation
            try:
                validation = self.aria.validate_hypothesis(hypothesis, results, context)
                if validation:
                    nb.add_entry(ExperimentEntry(
                        entry_type="analysis",
                        title="Hypothesis Validation",
                        content=validation.get("explanation", ""),
                        experiment_id=exp_id,
                        metadata={"validated": validation.get("validated", False)},
                    ))
            except Exception as e:
                logger.warning("Hypothesis validation logging failed: %s", e)

        nb.complete_experiment(
            experiment_id=exp_id, results=results,
            aria_summary=summary, aria_mood=self.aria.state.mood,
            insights=insights, llm_analysis=llm_analysis,
        )
        nb.update_op_success_rates(exp_id)
        self._auto_recommend(results, config, hypothesis, nb)

        if (config.auto_report
                and config.auto_report_every_n > 0
                and n_experiments % config.auto_report_every_n == 0):
            self._maybe_auto_report(
                config, nb,
                reason=f"periodic (every {config.auto_report_every_n}, "
                       f"after exp #{n_experiments})")

        # Knowledge extraction
        self._maybe_extract_knowledge(config, nb, n_experiments)

        # Auto-escalation: promote S1 survivors to leaderboard and
        # queue investigation/validation if criteria met
        results["experiment_id"] = exp_id
        self._auto_escalate(results, config, nb, phase="screening")
        self._maybe_evaluate_campaign(config, nb)

        self._emit_event("experiment_completed", {
            "experiment_id": exp_id, "results": results, "mode": "synthesis",
        })

    def _persist_applied_grammar_weights(
        self,
        nb: LabNotebook,
        exp_id: str,
        results: Dict[str, Any],
    ) -> None:
        """Persist applied grammar weights into experiment config_json."""
        applied = results.get("applied_grammar_weights")
        if not applied:
            return
        try:
            row = nb.conn.execute(
                "SELECT config_json FROM experiments WHERE experiment_id = ?",
                (exp_id,),
            ).fetchone()
            if row is None:
                return
            cfg_raw = row["config_json"]
            stored_config = json.loads(cfg_raw) if cfg_raw else {}
            stored_config["applied_grammar_weights"] = applied
            stored_config["grammar_weights"] = applied
            nb.conn.execute(
                "UPDATE experiments SET config_json = ? WHERE experiment_id = ?",
                (json.dumps(stored_config), exp_id),
            )
            nb.conn.commit()
        except Exception as e:
            logger.debug("Failed persisting grammar weights to config: %s", e)

    @staticmethod
    def _compute_generated_op_distribution(graphs: List[Any]) -> Dict[str, float]:
        """Compute normalized op-name distribution across generated graphs."""
        counts: Dict[str, int] = {}
        total = 0
        for graph in graphs:
            nodes = getattr(graph, "nodes", {}) or {}
            for node in nodes.values():
                op_name = getattr(node, "op_name", None)
                if not op_name or op_name == "input":
                    continue
                counts[op_name] = counts.get(op_name, 0) + 1
                total += 1

        if total <= 0:
            return {}

        return {
            op: round(count / total, 6)
            for op, count in sorted(counts.items())
        }

    @staticmethod
    def _distribution_l1_distance(
        current: Dict[str, float],
        previous: Dict[str, float],
    ) -> float:
        """Compute L1 distance between two sparse distributions."""
        keys = set(current.keys()) | set(previous.keys())
        if not keys:
            return 0.0
        return float(sum(abs(current.get(k, 0.0) - previous.get(k, 0.0)) for k in keys))

    def _compare_with_previous_synthesis_distribution(
        self,
        nb: LabNotebook,
        exp_id: str,
        current_distribution: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        """Compare generated-op distribution against previous synthesis experiment."""
        if not current_distribution:
            return None

        try:
            row = nb.conn.execute(
                """
                SELECT experiment_id, results_json
                FROM experiments
                WHERE experiment_type = 'synthesis'
                  AND experiment_id != ?
                  AND results_json IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (exp_id,),
            ).fetchone()
            if row is None:
                return None

            prev_results_raw = row["results_json"]
            prev_results = json.loads(prev_results_raw) if prev_results_raw else {}
            previous_distribution = prev_results.get("generated_op_distribution")
            if not isinstance(previous_distribution, dict) or not previous_distribution:
                return None

            l1 = self._distribution_l1_distance(current_distribution, previous_distribution)
            delta_pairs = []
            for op in set(current_distribution.keys()) | set(previous_distribution.keys()):
                delta = current_distribution.get(op, 0.0) - previous_distribution.get(op, 0.0)
                if abs(delta) > 1e-12:
                    delta_pairs.append((op, delta))
            delta_pairs.sort(key=lambda item: abs(item[1]), reverse=True)
            top_changes = [
                {"op": op, "delta": round(delta, 6)}
                for op, delta in delta_pairs[:5]
            ]

            return {
                "previous_experiment_id": row["experiment_id"],
                "l1_distance": round(l1, 6),
                "top_op_deltas": top_changes,
            }
        except Exception as e:
            logger.debug("Failed comparing generated-op distribution for %s: %s", exp_id, e)
            return None

    def _run_continuous_evolution(self, config: RunConfig, nb: LabNotebook,
                                  n_experiments: int, limit_str: str,
                                  mode_reasoning: str):
        """Run evolution search within continuous mode (inline, not threaded)."""
        from ..search.evolution import evolutionary_search, EvolutionConfig

        hypothesis = f"Evolution search: {mode_reasoning}"
        exp_id = nb.start_experiment(
            experiment_type="evolution",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source="runner_template",
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
        )

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="evolving",
                estimated_cost=self.aria.total_cost,
                total_tokens=self.aria.total_tokens,
                aria_message=f"[{limit_str}|evolution] {hypothesis[:80]}",
            )

        self._emit_event("experiment_started", {
            "experiment_id": exp_id,
            "experiment_number": n_experiments,
            "hypothesis": hypothesis,
            "mode": "evolution",
        })

        # Cap depth/ops for evolution to prevent recursion overflow
        evo_max_depth = min(config.max_depth, 12)
        evo_max_ops = min(config.max_ops, 20)

        grammar = GrammarConfig(
            max_depth=evo_max_depth,
            max_ops=evo_max_ops,
            model_dim=config.model_dim,
            residual_prob=config.residual_prob,
        )
        grammar.category_weights["math_space"] = config.math_space_weight
        evo_config = EvolutionConfig(
            population_size=config.n_programs,
            n_generations=config.n_generations,
            grammar_config=grammar,
        )
        fitness_fn = self._make_fitness_fn(config)

        def novelty_fn(graph, all_graphs):
            nov = novelty_score(graph)
            my_fp = graph.fingerprint()
            dup_count = sum(1 for g in all_graphs
                            if g.fingerprint() == my_fp) - 1
            penalty = max(0, 1 - dup_count * 0.3)
            return nov.structural_novelty * penalty

        population = evolutionary_search(
            fitness_fn=fitness_fn,
            novelty_fn=novelty_fn,
            config=evo_config,
        )

        results = {
            "total": len(population),
            "stage0_passed": sum(1 for ind in population if ind.fitness > 0),
            "stage05_passed": sum(1 for ind in population if ind.fitness > 0),
            "stage1_passed": sum(1 for ind in population if ind.fitness > 0.2),
            "novel_count": sum(1 for ind in population if ind.novelty > 0.5),
            "best_loss_ratio": 1.0 - max((ind.fitness for ind in population), default=0),
            "best_novelty_score": max((ind.novelty for ind in population), default=0),
            "survivors": [],
        }

        for ind in population[:20]:
            graph_metrics = self._extract_graph_metrics(ind.graph)
            nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=ind.fingerprint,
                graph_json=graph_to_json(ind.graph),
                stage1_passed=ind.fitness > 0.2,
                stage0_passed=ind.fitness > 0,
                stage05_passed=ind.fitness > 0,
                loss_ratio=1.0 - ind.fitness if ind.fitness > 0 else None,
                novelty_score=ind.novelty,
                novelty_confidence=0.2,
                stage_at_death="survived" if ind.fitness > 0.2 else "stage1",
                model_source="graph_synthesis",
                **graph_metrics,
            )
            if ind.fitness > 0.2:
                results["survivors"].append({
                    "fingerprint": ind.fingerprint,
                    "novelty": ind.novelty,
                    "loss_ratio": 1.0 - ind.fitness,
                })

        nb.update_op_success_rates(exp_id)
        context = self._build_rich_context_for_experiment(
            results, config, hypothesis, nb)
        summary = self.aria.experiment_summary(results, context=context)
        llm_analysis = self.aria.analyze_results(results, context=context)
        nb.complete_experiment(
            experiment_id=exp_id, results=results,
            aria_summary=summary, aria_mood=self.aria.state.mood,
            insights=self._analyze_results(results, exp_id, nb, context=context),
            llm_analysis=llm_analysis,
        )

        results["experiment_id"] = exp_id
        self._auto_escalate(results, config, nb, phase="screening")
        self._maybe_evaluate_campaign(config, nb)

        self._emit_event("experiment_completed", {
            "experiment_id": exp_id, "results": results, "mode": "evolution",
        })

    def _run_continuous_novelty(self, config: RunConfig, nb: LabNotebook,
                                 n_experiments: int, limit_str: str,
                                 mode_reasoning: str):
        """Run novelty search within continuous mode (inline, not threaded)."""
        from ..search.novelty_search import novelty_search, NoveltySearchConfig

        hypothesis = f"Novelty search: {mode_reasoning}"
        exp_id = nb.start_experiment(
            experiment_type="novelty",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source="runner_template",
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
        )

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="novelty_search",
                estimated_cost=self.aria.total_cost,
                total_tokens=self.aria.total_tokens,
                aria_message=f"[{limit_str}|novelty] {hypothesis[:80]}",
            )

        self._emit_event("experiment_started", {
            "experiment_id": exp_id,
            "experiment_number": n_experiments,
            "hypothesis": hypothesis,
            "mode": "novelty",
        })

        # Cap depth/ops for novelty search to prevent recursion overflow
        ns_max_depth = min(config.max_depth, 12)
        ns_max_ops = min(config.max_ops, 20)

        grammar = GrammarConfig(
            max_depth=ns_max_depth,
            max_ops=ns_max_ops,
            model_dim=config.model_dim,
            residual_prob=config.residual_prob,
        )
        grammar.category_weights["math_space"] = config.math_space_weight
        ns_config = NoveltySearchConfig(
            population_size=config.n_programs,
            n_generations=config.n_generations,
            grammar_config=grammar,
        )
        fitness_fn = self._make_fitness_fn(config)
        dev_str = config.device if torch.cuda.is_available() else "cpu"

        def fingerprint_fn(graph):
            """Compute behavioral fingerprint, falling back to None on failure."""
            try:
                layer_graphs = [graph] * config.n_layers
                model = compile_model(
                    layer_graphs,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                )
                fp = compute_fingerprint(
                    model,
                    seq_len=min(64, config.max_seq_len),
                    model_dim=config.model_dim,
                    vocab_size=config.vocab_size,
                    device=dev_str,
                )
                del model
                return fp
            except Exception as e:
                logger.debug("Fingerprint computation failed: %s", e)
                return None

        ns_result = novelty_search(
            fitness_fn=fitness_fn,
            fingerprint_fn=fingerprint_fn,
            config=ns_config,
        )

        results = {
            "total": ns_result.total_evaluated,
            "stage0_passed": sum(1 for ind in ns_result.best_individuals if ind.fitness > 0),
            "stage05_passed": sum(1 for ind in ns_result.best_individuals if ind.fitness > 0),
            "stage1_passed": sum(1 for ind in ns_result.best_individuals if ind.fitness > 0.2),
            "novel_count": sum(1 for ind in ns_result.best_individuals if ind.novelty > 0.5),
            "best_loss_ratio": None,
            "best_novelty_score": None,
            "survivors": [],
            "archive_size": ns_result.archive_size,
        }

        for ind in ns_result.best_individuals[:20]:
            graph_metrics = self._extract_graph_metrics(ind.graph)
            nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=ind.fingerprint,
                graph_json=graph_to_json(ind.graph),
                stage1_passed=ind.fitness > 0.2,
                stage0_passed=ind.fitness > 0,
                stage05_passed=ind.fitness > 0,
                loss_ratio=1.0 - ind.fitness if ind.fitness > 0 else None,
                novelty_score=ind.novelty,
                novelty_confidence=0.2,
                stage_at_death="survived" if ind.fitness > 0.2 else "stage1",
                model_source="graph_synthesis",
                **graph_metrics,
            )
            lr = 1.0 - ind.fitness if ind.fitness > 0 else None
            if lr is not None and (results["best_loss_ratio"] is None
                                    or lr < results["best_loss_ratio"]):
                results["best_loss_ratio"] = lr
            if ind.novelty and (results["best_novelty_score"] is None
                                 or ind.novelty > results["best_novelty_score"]):
                results["best_novelty_score"] = ind.novelty
            if ind.fitness > 0.2:
                results["survivors"].append({
                    "fingerprint": ind.fingerprint,
                    "novelty": ind.novelty,
                    "loss_ratio": 1.0 - ind.fitness,
                })

        nb.update_op_success_rates(exp_id)
        context = self._build_rich_context_for_experiment(
            results, config, hypothesis, nb)
        summary = self.aria.experiment_summary(results, context=context)
        llm_analysis = self.aria.analyze_results(results, context=context)
        nb.complete_experiment(
            experiment_id=exp_id, results=results,
            aria_summary=summary, aria_mood=self.aria.state.mood,
            insights=self._analyze_results(results, exp_id, nb, context=context),
            llm_analysis=llm_analysis,
        )

        results["experiment_id"] = exp_id
        self._auto_escalate(results, config, nb, phase="screening")
        self._maybe_evaluate_campaign(config, nb)

        self._emit_event("experiment_completed", {
            "experiment_id": exp_id, "results": results, "mode": "novelty",
        })

    def _run_continuous_phase(self, phase: str, config: RunConfig,
                               nb: LabNotebook, n_experiments: int,
                               limit_str: str, mode_reasoning: str):
        """Run investigation or validation phase inline within continuous mode."""
        leaderboard = nb.get_leaderboard(limit=50)

        if phase == "investigation":
            self._run_inline_investigation(
                config, nb, leaderboard, n_experiments, limit_str, mode_reasoning)
        elif phase == "validation":
            self._run_inline_validation(
                config, nb, leaderboard, n_experiments, limit_str, mode_reasoning)

    def _run_inline_investigation(self, config: RunConfig, nb: LabNotebook,
                                   leaderboard: list, n_experiments: int,
                                   limit_str: str, mode_reasoning: str):
        """Execute investigation phase inline (not threaded) for continuous mode."""
        # Find screening survivors with good loss ratios, skipping already-investigated archs
        investigated_fps = nb.get_investigated_fingerprints()
        candidates = [
            e for e in leaderboard
            if e.get("tier") == "screening"
            and e.get("screening_loss_ratio") is not None
            and e["screening_loss_ratio"] < 0.5
        ]
        if investigated_fps:
            before = len(candidates)
            candidates = [
                c for c in candidates
                if c.get("graph_fingerprint", c.get("architecture_desc", ""))
                not in investigated_fps
            ]
            skipped = before - len(candidates)
            if skipped:
                logger.info("Skipped %d already-investigated candidates", skipped)
        if not candidates:
            logger.info("No investigation candidates, falling back to synthesis")
            self._run_continuous_synthesis(
                config, nb, n_experiments, limit_str, mode_reasoning)
            return

        result_ids = [c["result_id"] for c in candidates[:config.auto_investigate_top_n]
                      if c.get("result_id")]
        if not result_ids:
            self._run_continuous_synthesis(
                config, nb, n_experiments, limit_str, mode_reasoning)
            return

        # Build context for hypothesis formulation
        inv_context = build_investigation_context(
            [nb.get_program_detail(rid) or {} for rid in result_ids],
            leaderboard,
        )
        hypothesis = self.aria.formulate_investigation_hypothesis(
            context=inv_context)
        exp_id = nb.start_experiment(
            experiment_type="investigation",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source="llm_context",
                llm_used=True,
                fallback_used=False,
                used_context=True,
            ),
        )

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="investigating",
                total_programs=len(result_ids),
                estimated_cost=self.aria.total_cost,
                total_tokens=self.aria.total_tokens,
                aria_message=(f"[{limit_str}|investigation] "
                              f"Studying {len(result_ids)} candidates"),
            )

        self._emit_event("investigation_started", {
            "experiment_id": exp_id,
            "n_candidates": len(result_ids),
        })

        try:
            # ── Inline investigation logic (from _run_investigation_thread) ──
            results = {
                "total": len(result_ids), "stage0_passed": 0, "stage05_passed": 0,
                "stage1_passed": 0, "novel_count": 0,
                "best_loss_ratio": None, "best_novelty_score": None,
                "survivors": [], "investigation_results": [],
            }

            dev_str = config.device if torch.cuda.is_available() else "cpu"
            dev = torch.device(dev_str)

            inv_config = RunConfig.from_dict(config.to_dict())
            inv_config.stage1_steps = config.investigation_steps
            inv_config.stage1_batch_size = config.investigation_batch_size

            for prog_idx, source_result_id in enumerate(result_ids):
                if self._stop_event.is_set():
                    break

                # Cost check mid-investigation
                if config.max_cost_dollars > 0 and self.aria.total_cost >= config.max_cost_dollars:
                    logger.info("Cost limit reached during investigation")
                    break

                with self._lock:
                    self._progress.current_program = prog_idx + 1
                    self._progress.status = "investigating"
                    self._progress.aria_message = (
                        f"Investigating {prog_idx + 1}/{len(result_ids)}: "
                        f"{source_result_id[:8]}... "
                        f"({config.n_training_programs} training programs)"
                    )

                self._emit_event("investigation_progress", {
                    "experiment_id": exp_id,
                    "current": prog_idx + 1,
                    "total": len(result_ids),
                    "source_result_id": source_result_id,
                    "status": "starting",
                })

                # Fetch source program
                source = nb.get_program_detail(source_result_id)
                if source is None:
                    continue

                graph_json_str = source.get("graph_json")
                arch_spec_json_str = source.get("arch_spec_json")
                model_source = source.get("model_source", "graph_synthesis")

                # Generate training programs
                training_programs = []
                for tp_i in range(config.n_training_programs):
                    tp = synthesize_training_program(
                        n_steps=config.investigation_steps,
                        max_seq_len=config.max_seq_len,
                        seed=tp_i + prog_idx * 1000,
                    )
                    training_programs.append(tp)

                # Test each (model x training_program) pair
                tp_results = []
                for tp_i, tp in enumerate(training_programs):
                    if self._stop_event.is_set():
                        break

                    # Reconstruct model fresh for each training program
                    try:
                        if model_source == "morphological_box" and arch_spec_json_str:
                            from ..morphological_box import ArchSpec
                            from ..arch_builder import build_model, BuildConfig
                            spec_data = json.loads(arch_spec_json_str)
                            spec = ArchSpec(**spec_data)
                            build_cfg = BuildConfig(
                                dim=config.model_dim,
                                n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.max_seq_len,
                            )
                            model = build_model(spec, build_cfg)
                        elif graph_json_str:
                            graph = graph_from_json(graph_json_str)
                            layer_graphs = [graph] * config.n_layers
                            model = compile_model(
                                layer_graphs,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.max_seq_len,
                            )
                        else:
                            continue
                    except Exception as e:
                        logger.debug(f"Model reconstruction failed: {e}")
                        continue

                    self._emit_event("investigation_progress", {
                        "experiment_id": exp_id,
                        "current": prog_idx + 1,
                        "total": len(result_ids),
                        "source_result_id": source_result_id,
                        "training_program": tp_i + 1,
                        "total_programs": len(training_programs),
                        "status": f"training with {tp.name}",
                    })

                    tp_result = self._train_with_program(
                        model,
                        tp,
                        inv_config,
                        dev,
                        seed=self._stable_seed(exp_id, source_result_id, tp_i, "investigation"),
                    )
                    tp_results.append({
                        "training_program": tp.name,
                        "passed": tp_result.get("passed", False),
                        "loss_ratio": tp_result.get("loss_ratio"),
                        "final_loss": tp_result.get("final_loss"),
                    })

                    del model
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

                # Compute robustness
                n_passed = sum(1 for r in tp_results if r.get("passed"))
                robustness = n_passed / max(len(tp_results), 1)
                best_tp = min(
                    (r for r in tp_results if r.get("loss_ratio") is not None),
                    key=lambda r: r["loss_ratio"],
                    default=None,
                )
                best_lr = best_tp["loss_ratio"] if best_tp else None
                screening_lr = source.get("loss_ratio")
                lr_multiplier = self._investigation_loss_multiplier(screening_lr, best_lr)
                brittle_risk = (
                    lr_multiplier is not None
                    and lr_multiplier > float(config.investigation_max_loss_ratio_multiplier)
                )

                if n_passed > 0:
                    results["stage1_passed"] += 1
                results["stage0_passed"] += 1
                results["stage05_passed"] += 1

                investigation_entry = {
                    "result_id": source_result_id,
                    "robustness": robustness,
                    "best_loss_ratio": best_lr,
                    "screening_loss_ratio": screening_lr,
                    "baseline_loss_ratio": source.get("baseline_loss_ratio"),
                    "novelty_confidence": source.get("novelty_confidence"),
                    "loss_ratio_multiplier": lr_multiplier,
                    "brittle_risk": brittle_risk,
                    "n_programs_passed": n_passed,
                    "n_programs_tested": len(tp_results),
                    "best_training_program": best_tp.get("training_program") if best_tp else None,
                }
                results["investigation_results"].append(investigation_entry)

                if best_lr and (results["best_loss_ratio"] is None
                                or best_lr < results["best_loss_ratio"]):
                    results["best_loss_ratio"] = best_lr

                # Update leaderboard
                best_tp_json = None
                if best_tp and best_tp.get("training_program"):
                    for tp in training_programs:
                        if tp.name == best_tp["training_program"]:
                            best_tp_json = json.dumps(tp.to_dict())
                            break

                investigation_passed = (
                    robustness >= 0.5
                    and (best_lr or 1.0) < 0.5
                    and not brittle_risk
                )

                nb.upsert_leaderboard(
                    result_id=source_result_id,
                    model_source=model_source,
                    architecture_desc=source.get("graph_fingerprint", "")[:40],
                    screening_loss_ratio=source.get("loss_ratio"),
                    screening_novelty=source.get("novelty_score"),
                    screening_passed=True,
                    investigation_loss_ratio=best_lr,
                    investigation_robustness=robustness,
                    investigation_best_training=best_tp_json,
                    investigation_passed=investigation_passed,
                    tier="investigation" if investigation_passed else "screening",
                    novelty_confidence=source.get("novelty_confidence"),
                )

                # Record result
                nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=source.get("graph_fingerprint", source_result_id),
                    graph_json=graph_json_str or "{}",
                    stage0_passed=True,
                    stage05_passed=True,
                    stage1_passed=n_passed > 0,
                    loss_ratio=best_lr,
                    novelty_score=source.get("novelty_score"),
                    novelty_confidence=source.get("novelty_confidence"),
                    training_program_json=best_tp_json,
                    model_source=model_source,
                    arch_spec_json=arch_spec_json_str,
                )

            # Complete experiment with LLM analysis
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)

            nb.complete_experiment(
                experiment_id=exp_id, results=results,
                aria_summary=summary,
                aria_mood=self.aria.state.mood,
                insights=self._analyze_results(results, exp_id, nb, context=context),
                llm_analysis=llm_analysis,
            )

            # Auto-escalate to validation if strong candidates found
            self._auto_escalate(results, config, nb, phase="investigation")

            # Knowledge extraction after investigation
            self._maybe_extract_knowledge(config, nb, n_experiments)

            self._emit_event("investigation_completed", {
                "experiment_id": exp_id, "results": results,
                "summary": summary,
            })

        except Exception as e:
            logger.warning(f"Inline investigation failed: {e}")
            nb.fail_experiment(exp_id, str(e))
            self._emit_event("investigation_completed", {
                "experiment_id": exp_id, "error": str(e),
            })

    def _run_inline_validation(self, config: RunConfig, nb: LabNotebook,
                                leaderboard: list, n_experiments: int,
                                limit_str: str, mode_reasoning: str):
        """Execute validation phase inline (not threaded) for continuous mode."""
        # Find investigation survivors with robustness
        candidates = [
            e for e in leaderboard
            if e.get("tier") == "investigation"
            and e.get("investigation_robustness") is not None
            and e["investigation_robustness"] >= 0.5
        ]
        if not candidates:
            logger.info("No validation candidates, falling back to synthesis")
            self._run_continuous_synthesis(
                config, nb, n_experiments, limit_str, mode_reasoning)
            return

        result_ids = [c["result_id"] for c in candidates[:config.auto_validate_top_n]
                      if c.get("result_id")]
        if not result_ids:
            self._run_continuous_synthesis(
                config, nb, n_experiments, limit_str, mode_reasoning)
            return

        # Build context for hypothesis formulation
        val_context = build_validation_context(
            [nb.get_program_detail(rid) or {} for rid in result_ids],
            [e for e in leaderboard if e.get("result_id") in result_ids],
        )
        hypothesis = self.aria.formulate_validation_hypothesis(
            context=val_context)
        exp_id = nb.start_experiment(
            experiment_type="validation",
            config=self._validation_config_with_result_ids(
                config,
                result_ids,
                "continuous_auto",
            ),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source="llm_context",
                llm_used=True,
                fallback_used=False,
                used_context=True,
            ),
        )

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="validating",
                total_programs=len(result_ids),
                estimated_cost=self.aria.total_cost,
                total_tokens=self.aria.total_tokens,
                aria_message=(f"[{limit_str}|validation] "
                              f"Validating {len(result_ids)} candidates"),
            )

        self._emit_event("validation_started", {
            "experiment_id": exp_id,
            "n_candidates": len(result_ids),
        })

        try:
            # ── Inline validation logic (from _run_validation_thread) ──
            results = {
                "total": len(result_ids), "stage0_passed": 0, "stage05_passed": 0,
                "stage1_passed": 0, "novel_count": 0,
                "best_loss_ratio": None, "best_novelty_score": None,
                "survivors": [], "validation_results": [],
            }

            dev_str = config.device if torch.cuda.is_available() else "cpu"
            dev = torch.device(dev_str)

            val_config = RunConfig.from_dict(config.to_dict())
            val_config.stage1_steps = config.validation_steps
            val_config.stage1_batch_size = config.validation_batch_size
            val_config.max_seq_len = config.validation_seq_len

            for prog_idx, source_result_id in enumerate(result_ids):
                if self._stop_event.is_set():
                    break

                # Cost check mid-validation
                if config.max_cost_dollars > 0 and self.aria.total_cost >= config.max_cost_dollars:
                    logger.info("Cost limit reached during validation")
                    break

                with self._lock:
                    self._progress.current_program = prog_idx + 1
                    self._progress.status = "validating"
                    self._progress.aria_message = (
                        f"Validating {prog_idx + 1}/{len(result_ids)}: "
                        f"{source_result_id[:8]}... "
                        f"({config.validation_n_seeds} seeds, "
                        f"{config.validation_steps} steps)"
                    )

                self._emit_event("validation_progress", {
                    "experiment_id": exp_id,
                    "current": prog_idx + 1,
                    "total": len(result_ids),
                    "source_result_id": source_result_id,
                    "status": "starting",
                })

                # Fetch source and leaderboard entry
                source = nb.get_program_detail(source_result_id)
                if source is None:
                    continue

                graph_json_str = source.get("graph_json")
                arch_spec_json_str = source.get("arch_spec_json")
                model_source = source.get("model_source", "graph_synthesis")

                # Get best training program from investigation
                best_tp_json = None
                for entry in leaderboard:
                    if entry.get("result_id") == source_result_id:
                        best_tp_json = entry.get("investigation_best_training")
                        break

                # Multi-seed evaluation
                seed_results = []
                for seed in range(config.validation_n_seeds):
                    if self._stop_event.is_set():
                        break

                    torch.manual_seed(seed * 42 + 7)

                    # Reconstruct model fresh
                    try:
                        if model_source == "morphological_box" and arch_spec_json_str:
                            from ..morphological_box import ArchSpec
                            from ..arch_builder import build_model, BuildConfig
                            spec_data = json.loads(arch_spec_json_str)
                            spec = ArchSpec(**spec_data)
                            build_cfg = BuildConfig(
                                dim=config.model_dim,
                                n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len,
                            )
                            model = build_model(spec, build_cfg)
                        elif graph_json_str:
                            graph = graph_from_json(graph_json_str)
                            layer_graphs = [graph] * config.n_layers
                            model = compile_model(
                                layer_graphs,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len,
                            )
                        else:
                            continue
                    except Exception as e:
                        logger.debug(f"Model reconstruction failed: {e}")
                        continue

                    self._emit_event("validation_progress", {
                        "experiment_id": exp_id,
                        "current": prog_idx + 1,
                        "total": len(result_ids),
                        "source_result_id": source_result_id,
                        "seed": seed + 1,
                        "total_seeds": config.validation_n_seeds,
                        "status": f"seed {seed + 1}/{config.validation_n_seeds}",
                    })

                    # Train (use best training program if available)
                    if best_tp_json:
                        try:
                            tp_data = json.loads(best_tp_json)
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
                                seed=self._stable_seed(exp_id, source_result_id, seed, "validation_tp"),
                            )
                        except Exception:
                            s1_result = self._micro_train(
                                model,
                                val_config,
                                dev,
                                seed=self._stable_seed(exp_id, source_result_id, seed, "validation_micro"),
                            )
                    else:
                        s1_result = self._micro_train(
                            model,
                            val_config,
                            dev,
                            seed=self._stable_seed(exp_id, source_result_id, seed, "validation_micro"),
                        )

                    seed_results.append({
                        "seed": seed,
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
                    })

                    del model
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

                # Compute validation metrics
                passed_seeds = [r for r in seed_results if r.get("passed")]
                loss_ratios = [r["loss_ratio"] for r in seed_results
                               if r.get("loss_ratio") is not None]

                val_loss_ratio = (sum(loss_ratios) / len(loss_ratios)
                                  if loss_ratios else None)
                multi_seed_std = 0.0
                if len(loss_ratios) > 1:
                    mean_lr = sum(loss_ratios) / len(loss_ratios)
                    multi_seed_std = (
                        sum((lr - mean_lr) ** 2 for lr in loss_ratios)
                        / len(loss_ratios)
                    ) ** 0.5

                # Baseline comparison at validation scale
                val_baseline_ratio = None
                if loss_ratios:
                    best_seed = min(
                        (r for r in seed_results if r.get("final_loss") is not None),
                        key=lambda r: r["final_loss"],
                        default=None,
                    )
                    if best_seed is not None:
                        try:
                            baseline = self._get_baseline()
                            baseline_steps = int(best_seed.get("n_train_steps") or config.validation_steps)
                            baseline_recipe = self._resolve_baseline_recipe(
                                best_seed, default_lr=config.stage1_lr)
                            bl_data_fn, bl_data_tag = self._make_baseline_data_fn(config)
                            val_baseline_ratio = baseline.compare(
                                best_seed["final_loss"],
                                d_model=config.model_dim,
                                seq_len=min(128, config.validation_seq_len),
                                n_steps=max(1, baseline_steps),
                                vocab_size=config.vocab_size,
                                batch_size=config.validation_batch_size,
                                lr=baseline_recipe["lr"],
                                device=dev_str,
                                n_layers=config.n_layers,
                                optimizer_name=baseline_recipe["optimizer_name"],
                                weight_decay=baseline_recipe["weight_decay"],
                                momentum=baseline_recipe["momentum"],
                                betas=baseline_recipe["betas"],
                                data_fn=bl_data_fn,
                                data_tag=bl_data_tag,
                            )
                        except Exception:
                            pass

                if len(passed_seeds) > 0:
                    results["stage1_passed"] += 1
                results["stage0_passed"] += 1
                results["stage05_passed"] += 1

                # OOD robustness check (#54): test with reference recipes
                ood_result = None
                if len(passed_seeds) > 0:
                    _gjs_ood = graph_json_str
                    _asjs_ood = arch_spec_json_str
                    _ms_ood = model_source
                    _cfg_ood = config

                    def _make_model_ood():
                        if _ms_ood == "morphological_box" and _asjs_ood:
                            from ..morphological_box import ArchSpec
                            from ..arch_builder import build_model, BuildConfig
                            spec = ArchSpec(**json.loads(_asjs_ood))
                            bc = BuildConfig(
                                dim=_cfg_ood.model_dim,
                                n_layers=_cfg_ood.n_layers,
                                vocab_size=_cfg_ood.vocab_size,
                                max_seq_len=_cfg_ood.validation_seq_len)
                            return build_model(spec, bc)
                        else:
                            g = graph_from_json(_gjs_ood)
                            return compile_model(
                                [g] * _cfg_ood.n_layers,
                                vocab_size=_cfg_ood.vocab_size,
                                max_seq_len=_cfg_ood.validation_seq_len)

                    try:
                        ood_result = self._ood_robustness_check(
                            _make_model_ood, config, dev,
                            n_steps=min(300, config.validation_steps // 3),
                            seed=self._stable_seed(
                                exp_id, source_result_id, 0, "ood"),
                        )
                        self._emit_event("ood_robustness", {
                            "experiment_id": exp_id,
                            "result_id": source_result_id,
                            "ood_robustness": ood_result.get("ood_robustness"),
                            "recipes_passed": ood_result.get("recipes_passed"),
                        })
                    except Exception as e:
                        logger.debug("OOD robustness check failed: %s", e)

                # Hyperparameter sensitivity check (#57)
                sensitivity_result = None
                if len(passed_seeds) > 0 and val_loss_ratio is not None:
                    try:
                        sensitivity_result = self._sensitivity_check(
                            _make_model_ood, config, dev,
                            base_loss_ratio=val_loss_ratio,
                            n_steps=min(300, config.validation_steps // 3),
                            seed=self._stable_seed(
                                exp_id, source_result_id, 0, "sensitivity"),
                        )
                        self._emit_event("sensitivity_check", {
                            "experiment_id": exp_id,
                            "result_id": source_result_id,
                            "hp_robustness": sensitivity_result.get("hp_robustness"),
                            "avg_deviation": sensitivity_result.get("avg_deviation"),
                        })
                    except Exception as e:
                        logger.debug("Sensitivity check failed: %s", e)

                # Determine if breakthrough — aligned with Aria publication thresholds
                ood_ok = (ood_result is not None
                          and ood_result.get("ood_robustness", 0) >= 0.67)
                hp_ok = (sensitivity_result is not None
                         and sensitivity_result.get("hp_robustness", 0) >= 0.75)
                nov_conf = source.get("novelty_confidence", 0) if source else 0
                is_breakthrough = (
                    val_baseline_ratio is not None
                    and val_baseline_ratio < 0.90
                    and multi_seed_std <= 0.03
                    and len(passed_seeds) >= 5
                    and len(passed_seeds) == config.validation_n_seeds
                    and (ood_result is None or ood_ok)
                    and (sensitivity_result is None or hp_ok)
                    and nov_conf >= 0.5
                )

                tier = "breakthrough" if is_breakthrough else "validation"

                validation_entry = {
                    "result_id": source_result_id,
                    "val_loss_ratio": val_loss_ratio,
                    "val_baseline_ratio": val_baseline_ratio,
                    "multi_seed_std": multi_seed_std,
                    "seeds_passed": len(passed_seeds),
                    "total_seeds": config.validation_n_seeds,
                    "is_breakthrough": is_breakthrough,
                    "novelty_confidence": nov_conf,
                    "ood_robustness": ood_result,
                    "sensitivity": sensitivity_result,
                }
                results["validation_results"].append(validation_entry)

                if val_loss_ratio and (results["best_loss_ratio"] is None
                                       or val_loss_ratio < results["best_loss_ratio"]):
                    results["best_loss_ratio"] = val_loss_ratio

                # Update leaderboard - find the actual entry for this result
                for entry in nb.get_leaderboard(limit=200):
                    if entry.get("result_id") == source_result_id:
                        nb.promote_to_tier(
                            entry_id=entry["entry_id"],
                            tier=tier,
                            validation_loss_ratio=val_loss_ratio,
                            validation_baseline_ratio=val_baseline_ratio,
                            validation_multi_seed_std=multi_seed_std,
                            validation_passed=len(passed_seeds) > 0,
                        )
                        break

                # Record validation result
                nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=source.get("graph_fingerprint",
                                                 source_result_id),
                    graph_json=graph_json_str or "{}",
                    stage0_passed=True,
                    stage05_passed=True,
                    stage1_passed=len(passed_seeds) > 0,
                    loss_ratio=val_loss_ratio,
                    baseline_loss_ratio=val_baseline_ratio,
                    novelty_score=source.get("novelty_score"),
                    novelty_confidence=source.get("novelty_confidence"),
                    model_source=model_source,
                    arch_spec_json=arch_spec_json_str,
                )

                # Breakthrough detection
                if is_breakthrough:
                    ctx = build_validation_context(
                        [source], [validation_entry])
                    announcement = self.aria.announce_breakthrough(ctx)
                    nb.add_entry(ExperimentEntry(
                        entry_type="insight",
                        title="BREAKTHROUGH DETECTED",
                        content=announcement,
                        experiment_id=exp_id,
                        tags=["breakthrough"],
                    ))
                    self._emit_event("breakthrough_detected", {
                        "experiment_id": exp_id,
                        "result_id": source_result_id,
                        "val_loss_ratio": val_loss_ratio,
                        "val_baseline_ratio": val_baseline_ratio,
                        "multi_seed_std": multi_seed_std,
                        "announcement": announcement,
                    })

            # Complete experiment with LLM analysis
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)

            nb.complete_experiment(
                experiment_id=exp_id, results=results,
                aria_summary=summary,
                aria_mood=self.aria.state.mood,
                insights=self._analyze_results(results, exp_id, nb, context=context),
                llm_analysis=llm_analysis,
            )

            # Knowledge extraction after validation
            self._maybe_extract_knowledge(config, nb, n_experiments)

            self._emit_event("validation_completed", {
                "experiment_id": exp_id, "results": results,
                "summary": summary,
            })

        except Exception as e:
            logger.warning(f"Inline validation failed: {e}")
            nb.fail_experiment(exp_id, str(e))
            self._emit_event("validation_completed", {
                "experiment_id": exp_id, "error": str(e),
            })

    # ── Core Execution ──

    @staticmethod
    def _diversify_grammar_config(config: RunConfig, n_experiments: int) -> RunConfig:
        """Mutate grammar parameters based on experiment number for diversity.

        Returns a shallow copy of config with adjusted grammar settings.
        Uses modular arithmetic to cycle through configurations deterministically.
        """
        import copy
        cfg = copy.copy(config)
        cycle = n_experiments % 6

        if cycle == 0:
            # Boost frequency and reduction, suppress dominant categories
            cfg.math_space_weight = 1.0
            cfg.residual_prob = 0.5
        elif cycle == 1:
            # Deeper, narrower
            cfg.max_depth = 12
            cfg.max_ops = 20
            cfg.residual_prob = 0.7
        elif cycle == 2:
            # Wider, shallower
            cfg.max_depth = 6
            cfg.max_ops = 12
            cfg.residual_prob = 0.6
        elif cycle == 3:
            # High risk, frequency focus
            cfg.math_space_weight = 3.0
            cfg.residual_prob = 0.4
        elif cycle == 4:
            # Minimal depth, low residual
            cfg.max_depth = 8
            cfg.max_ops = 10
            cfg.math_space_weight = 1.5
            cfg.residual_prob = 0.3
        else:
            # Default with boosted math space
            cfg.math_space_weight = 2.5
            cfg.max_depth = 10
            cfg.residual_prob = 0.7

        return cfg

    def _extract_graph_metrics(self, graph) -> Dict:
        """Extract structural metrics from a computation graph."""
        metrics = {}
        metrics["graph_n_ops"] = graph.n_ops()
        metrics["graph_depth"] = graph.depth()
        metrics["graph_n_params_estimate"] = graph.n_params_estimate()
        metrics["graph_has_gradient_path"] = graph.has_gradient_path()

        # Edge count
        n_edges = sum(len(n.input_ids) for n in graph.nodes.values())
        metrics["graph_n_edges"] = n_edges

        # Unique ops and category histogram
        ops_used = set()
        cat_counts: Dict[str, int] = {}
        uses_math = False
        uses_freq = False
        for node in graph.nodes.values():
            if node.is_input:
                continue
            ops_used.add(node.op_name)
            try:
                from ..synthesis.primitives import get_primitive
                op = get_primitive(node.op_name)
                cat = op.category.value
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                if cat == "math_space":
                    uses_math = True
                if cat == "frequency":
                    uses_freq = True
            except (KeyError, Exception):
                pass

        metrics["graph_n_unique_ops"] = len(ops_used)
        metrics["graph_category_histogram"] = json.dumps(cat_counts)
        metrics["graph_uses_math_spaces"] = uses_math
        metrics["graph_uses_frequency_domain"] = uses_freq

        return metrics

    def _extract_sandbox_metrics(self, sandbox_result) -> Dict:
        """Extract ALL fields from a SandboxResult."""
        metrics = {}
        metrics["compile_time_ms"] = sandbox_result.compile_time_ms
        metrics["forward_time_ms"] = sandbox_result.forward_time_ms
        metrics["backward_time_ms"] = sandbox_result.backward_time_ms
        metrics["peak_memory_mb"] = sandbox_result.peak_memory_mb
        metrics["grad_norm"] = sandbox_result.grad_norm
        metrics["stability_score"] = sandbox_result.stability_score
        metrics["extreme_input_passed"] = sandbox_result.extreme_input_passed
        metrics["random_input_passed"] = sandbox_result.random_input_passed
        metrics["has_nan_output"] = sandbox_result.has_nan_output
        metrics["has_inf_output"] = sandbox_result.has_inf_output
        metrics["has_nan_grad"] = sandbox_result.has_nan_grad
        metrics["has_zero_grad"] = sandbox_result.has_zero_grad
        metrics["error_type"] = sandbox_result.error_type
        metrics["error_message"] = sandbox_result.error

        # Parse output_range "[min, max]" string
        if sandbox_result.output_range:
            try:
                parts = sandbox_result.output_range.strip("[]").split(",")
                metrics["output_range_min"] = float(parts[0].strip())
                metrics["output_range_max"] = float(parts[1].strip())
            except (ValueError, IndexError):
                pass

        return metrics

    def _extract_sparse_metrics(self, model: Optional[nn.Module]) -> Dict:
        """Extract sparse execution telemetry from compiled layer ops."""
        if model is None or not hasattr(model, "layers"):
            return {}

        telemetry_rows: List[Dict[str, Any]] = []
        total_calls = 0
        total_fallback_calls = 0
        kernel_fallback_calls = 0
        density_sum = 0.0
        density_last_values: List[float] = []
        nm_compliant = 0
        nm_total = 0
        sparse_active_params_estimate = 0.0

        for layer in getattr(model, "layers", []):
            ops = getattr(layer, "ops", None)
            if ops is None:
                continue
            for compiled_op in ops.values():
                sparse_telemetry = getattr(compiled_op, "sparse_telemetry", None)
                if not sparse_telemetry:
                    continue
                has_weight = hasattr(compiled_op, "weight")
                weight_params = float(compiled_op.weight.numel()) if has_weight else 0.0
                for op_name, stats in sparse_telemetry.items():
                    calls = int(stats.get("calls", 0) or 0)
                    fallback_calls = int(stats.get("fallback_calls", 0) or 0)
                    density_sum_local = float(stats.get("density_sum", 0.0) or 0.0)
                    last_density = float(stats.get("last_density", 1.0) or 1.0)
                    fallback_reason = stats.get("last_fallback_reason")

                    total_calls += calls
                    total_fallback_calls += fallback_calls
                    density_sum += density_sum_local
                    density_last_values.append(last_density)
                    if fallback_reason == "kernel_unavailable":
                        kernel_fallback_calls += fallback_calls

                    if op_name in ("nm_sparse_linear", "semi_structured_2_4_linear"):
                        nm_total += 1
                        if last_density <= 0.51:
                            nm_compliant += 1

                    if weight_params > 0.0:
                        density_for_params = (density_sum_local / calls) if calls > 0 else last_density
                        sparse_active_params_estimate += weight_params * density_for_params

                    telemetry_rows.append({
                        "op_name": op_name,
                        "calls": calls,
                        "fallback_calls": fallback_calls,
                        "last_density": last_density,
                        "last_fallback_reason": fallback_reason,
                    })

        if total_calls == 0:
            return {}

        density_mean = density_sum / max(total_calls, 1)
        density_last = sum(density_last_values) / max(len(density_last_values), 1)
        nm_compliance = (nm_compliant / nm_total) if nm_total > 0 else None

        metrics: Dict[str, Any] = {
            "sparse_density_mean": density_mean,
            "sparse_density_last": density_last,
            "sparse_fallback_calls": total_fallback_calls,
            "sparse_kernel_fallback_calls": kernel_fallback_calls,
            "sparse_active_params_estimate": int(max(0.0, sparse_active_params_estimate)),
            "sparse_telemetry_json": json.dumps(telemetry_rows),
        }
        if nm_compliance is not None:
            metrics["sparse_nm_compliance"] = nm_compliance
        return metrics

    def _classify_stage_at_death(self, s0_passed: bool, s05_passed: bool,
                                  s1_passed: bool) -> str:
        """Classify which stage a program died at."""
        if not s0_passed:
            return "stage0"
        if not s05_passed:
            return "stage0.5"
        if not s1_passed:
            return "stage1"
        return "survived"

    def _execute_experiment(self, exp_id: str, config: RunConfig,
                            nb: LabNotebook,
                            use_learned_grammar: bool = True) -> Dict:
        """Core experiment logic shared by single and continuous modes."""
        results = {
            "total": 0, "stage0_passed": 0, "stage05_passed": 0,
            "stage1_passed": 0, "novel_count": 0,
            "best_loss_ratio": None, "best_novelty_score": None,
            "survivors": [],
        }

        grammar_weights = None
        excluded_ops: set = set()
        if use_learned_grammar:
            try:
                from .analytics import ExperimentAnalytics
                analytics = ExperimentAnalytics(nb)
                grammar_weights = analytics.compute_grammar_weights()
            except Exception as e:
                logger.warning("Failed computing learned grammar weights for %s: %s", exp_id, e)

            # Populate excluded_ops from negative results
            try:
                if analytics is not None:
                    neg = analytics.negative_results_synthesis()
                    for op_info in neg.get("failed_ops", []):
                        if (op_info.get("s1_rate", 1) == 0
                                and op_info.get("n_used", 0) >= 5
                                and op_info.get("confidence", 0) >= 0.7):
                            excluded_ops.add(op_info["op_name"])
                    if excluded_ops:
                        nb.log_learning_event(
                            "excluded_ops_applied",
                            f"Excluded {len(excluded_ops)} ops with 0% S1 rate: "
                            f"{', '.join(sorted(excluded_ops))}",
                            excluded_ops=sorted(excluded_ops),
                        )
            except Exception as e:
                logger.warning("Failed computing excluded ops for %s: %s", exp_id, e)

        grammar = GrammarConfig(
            model_dim=config.model_dim,
            max_depth=config.max_depth,
            max_ops=config.max_ops,
            residual_prob=config.residual_prob,
            excluded_ops=excluded_ops,
        )

        if grammar_weights:
            old_weights = dict(grammar.category_weights)
            grammar.category_weights.update(grammar_weights)
            nb.log_learning_event(
                "grammar_weights_applied",
                f"Applied learned grammar weights for experiment {exp_id}",
                old_weights=old_weights,
                new_weights=dict(grammar.category_weights),
            )
            # Persist for observability
            results["applied_grammar_weights"] = dict(grammar.category_weights)
            # Emit SSE so LiveFeed can show learning events
            n_changed = sum(1 for k in grammar_weights
                            if old_weights.get(k) != grammar_weights[k])
            self._emit_event("learning_event", {
                "event_type": "grammar_weights_applied",
                "experiment_id": exp_id,
                "n_changed": n_changed,
                "description": f"Applied learned grammar weights ({n_changed} categories changed)",
            })
        else:
            grammar.category_weights["math_space"] = config.math_space_weight

        t_start = time.time()

        # Generate graphs
        graphs = batch_generate(config.n_programs, grammar)
        results["total"] = len(graphs)
        op_distribution = self._compute_generated_op_distribution(graphs)
        if op_distribution:
            results["generated_op_distribution"] = op_distribution
            shift = self._compare_with_previous_synthesis_distribution(
                nb,
                exp_id,
                op_distribution,
            )
            if shift:
                results["generation_distribution_shift"] = shift
                nb.log_learning_event(
                    "architecture_distribution_shift",
                    f"Generated-op distribution shift recorded for synthesis experiment {exp_id}",
                    evidence=json.dumps(shift, sort_keys=True),
                )
            else:
                nb.log_learning_event(
                    "architecture_distribution_snapshot",
                    f"Captured generated-op distribution for synthesis experiment {exp_id}",
                    evidence=json.dumps({"op_distribution": op_distribution}, sort_keys=True),
                )

        with self._lock:
            self._progress.total_programs = len(graphs)
            self._progress.status = "evaluating"

        nb.add_entry(ExperimentEntry(
            entry_type="observation",
            title=f"Generated {len(graphs)} computation graphs",
            content=f"Grammar: depth={grammar.max_depth}, ops={grammar.max_ops}, "
                    f"dim={config.model_dim}, math_space_weight={config.math_space_weight}",
            experiment_id=exp_id,
        ))

        dev_str = config.device if torch.cuda.is_available() else "cpu"
        dev = torch.device(dev_str)

        for i, graph in enumerate(graphs):
            if self._stop_event.is_set():
                break

            with self._lock:
                self._progress.current_program = i + 1
                self._progress.current_fingerprint = graph.fingerprint()[:10]
                self._progress.elapsed_seconds = time.time() - t_start

            # Collect all metrics for this program
            program_metrics: Dict[str, Any] = {}

            # Extract graph structural metrics
            program_metrics.update(self._extract_graph_metrics(graph))

            # Estimate FLOPs
            try:
                flop_est = estimate_flops(graph, seq_len=min(128, config.max_seq_len),
                                          d_model=config.model_dim)
                program_metrics["flops_forward"] = flop_est.flops_forward
                program_metrics["flops_per_param"] = flop_est.flops_per_param
                program_metrics["flops_per_token"] = flop_est.flops_per_token
            except Exception as e:
                logger.debug("FLOP estimate failed for %s: %s", graph.fingerprint()[:10], e)

            # Validate
            with self._lock:
                self._progress.current_stage = "validating"

            validation = validate_graph(graph)
            if not validation.valid:
                program_metrics["stage_at_death"] = "stage0"
                program_metrics["error_type"] = "validation_error"
                program_metrics["error_message"] = "; ".join(validation.errors)
                nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=graph.fingerprint(),
                    graph_json=graph_to_json(graph),
                    stage0_error="; ".join(validation.errors),
                    **program_metrics,
                )
                self._emit_event("program_evaluated", {
                    "index": i, "fingerprint": graph.fingerprint()[:10],
                    "result": "invalid", "error": validation.errors[0] if validation.errors else "",
                })
                continue

            # Compile
            try:
                layer_graphs = [graph] * config.n_layers
                model = compile_model(
                    layer_graphs,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                )
            except Exception as e:
                program_metrics["stage_at_death"] = "stage0"
                program_metrics["error_type"] = "compile_error"
                program_metrics["error_message"] = str(e)
                nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=graph.fingerprint(),
                    graph_json=graph_to_json(graph),
                    stage0_error=str(e),
                    **program_metrics,
                )
                self._emit_event("program_evaluated", {
                    "index": i, "fingerprint": graph.fingerprint()[:10],
                    "result": "compile_error",
                })
                continue

            # Stage 0 + 0.5
            with self._lock:
                self._progress.current_stage = "stage0"

            sandbox_result = safe_eval(
                model, batch_size=2,
                seq_len=min(128, config.max_seq_len),
                vocab_size=config.vocab_size,
                device=dev_str,
            )

            # Extract all sandbox metrics
            program_metrics.update(self._extract_sandbox_metrics(sandbox_result))

            s0_passed = sandbox_result.passed
            s05_passed = sandbox_result.stability_score >= 0.5

            program_metrics["param_count"] = sandbox_result.param_count

            if s0_passed:
                results["stage0_passed"] += 1
                with self._lock:
                    self._progress.stage0_passed += 1
            if s05_passed:
                results["stage05_passed"] += 1
                with self._lock:
                    self._progress.stage05_passed += 1

            # Fingerprint (after S0 pass)
            if s0_passed:
                try:
                    fp = compute_fingerprint(
                        model,
                        seq_len=min(64, config.max_seq_len),
                        model_dim=config.model_dim,
                        vocab_size=config.vocab_size,
                        device=dev_str,
                    )
                    program_metrics["fingerprint_json"] = json.dumps(fp.to_dict())
                    program_metrics["fp_interaction_locality"] = fp.interaction_locality
                    program_metrics["fp_interaction_sparsity"] = fp.interaction_sparsity
                    program_metrics["fp_interaction_symmetry"] = fp.interaction_symmetry
                    program_metrics["fp_interaction_hierarchy"] = fp.interaction_hierarchy
                    program_metrics["fp_intrinsic_dim"] = fp.intrinsic_dim
                    program_metrics["fp_isotropy"] = fp.isotropy
                    program_metrics["fp_rank_ratio"] = fp.rank_ratio
                    program_metrics["fp_jacobian_spectral_norm"] = fp.jacobian_spectral_norm
                    program_metrics["fp_jacobian_effective_rank"] = fp.jacobian_effective_rank
                    program_metrics["fp_sensitivity_uniformity"] = fp.sensitivity_uniformity
                    program_metrics["fp_cka_vs_transformer"] = fp.cka_vs_transformer
                    program_metrics["fp_cka_vs_ssm"] = fp.cka_vs_ssm
                    program_metrics["fp_cka_vs_conv"] = fp.cka_vs_conv
                    program_metrics["cka_source"] = fp.cka_source
                    program_metrics["cka_artifact_version"] = fp.cka_artifact_version
                except Exception:
                    pass

            # Stage 1
            s1_passed = False
            loss_ratio = None
            final_loss = None
            throughput = None
            training_curve = None

            if s0_passed and s05_passed and not self._stop_event.is_set():
                with self._lock:
                    self._progress.current_stage = "stage1"

                s1_result = self._micro_train(
                    model,
                    config,
                    dev,
                    seed=self._stable_seed(exp_id, i, "screening"),
                )
                s1_passed = s1_result.get("passed", False)
                loss_ratio = s1_result.get("loss_ratio")
                final_loss = s1_result.get("final_loss")
                throughput = s1_result.get("throughput")
                training_curve = s1_result.get("training_curve")

                # Training metrics
                program_metrics["initial_loss"] = s1_result.get("initial_loss")
                program_metrics["min_loss"] = s1_result.get("min_loss")
                program_metrics["loss_improvement_rate"] = s1_result.get("loss_improvement_rate")
                program_metrics["avg_step_time_ms"] = s1_result.get("avg_step_time_ms")
                program_metrics["total_train_time_ms"] = s1_result.get("total_train_time_ms")
                program_metrics["max_grad_norm"] = s1_result.get("max_grad_norm")
                program_metrics["mean_grad_norm"] = s1_result.get("mean_grad_norm")
                program_metrics["grad_norm_std"] = s1_result.get("grad_norm_std")
                program_metrics["n_train_steps"] = s1_result.get("n_train_steps")
                program_metrics["final_lr"] = s1_result.get("final_lr")
                program_metrics["pruning_method"] = s1_result.get("pruning_method")
                program_metrics["pruning_target_sparsity"] = s1_result.get("pruning_target_sparsity")
                program_metrics["pruning_actual_sparsity"] = s1_result.get("pruning_actual_sparsity")
                program_metrics["pruning_n_params_total"] = s1_result.get("pruning_n_params_total")
                program_metrics["pruning_n_params_pruned"] = s1_result.get("pruning_n_params_pruned")
                program_metrics["pruning_dense_eval_loss"] = s1_result.get("pruning_dense_eval_loss")
                program_metrics["pruning_pruned_eval_loss"] = s1_result.get("pruning_pruned_eval_loss")
                program_metrics["pruning_quality_retention"] = s1_result.get("pruning_quality_retention")
                program_metrics["pruning_active_params_estimate"] = s1_result.get("pruning_active_params_estimate")
                program_metrics["pruning_error"] = s1_result.get("pruning_error")

                if s1_passed:
                    results["stage1_passed"] += 1
                    with self._lock:
                        self._progress.stage1_passed += 1

                    # Compare to baseline
                    if final_loss is not None:
                        try:
                            baseline = self._get_baseline()
                            baseline_steps = int(s1_result.get("n_train_steps") or config.stage1_steps)
                            baseline_recipe = self._resolve_baseline_recipe(
                                s1_result, default_lr=config.stage1_lr)
                            bl_data_fn, bl_data_tag = self._make_baseline_data_fn(config)
                            baseline_ratio = baseline.compare(
                                final_loss,
                                d_model=config.model_dim,
                                seq_len=min(128, config.max_seq_len),
                                n_steps=max(1, baseline_steps),
                                vocab_size=config.vocab_size,
                                batch_size=config.stage1_batch_size,
                                lr=baseline_recipe["lr"],
                                device=dev_str,
                                n_layers=config.n_layers,
                                optimizer_name=baseline_recipe["optimizer_name"],
                                weight_decay=baseline_recipe["weight_decay"],
                                momentum=baseline_recipe["momentum"],
                                betas=baseline_recipe["betas"],
                                data_fn=bl_data_fn,
                                data_tag=bl_data_tag,
                            )
                            program_metrics["baseline_loss_ratio"] = baseline_ratio
                        except Exception:
                            pass

            # Determine stage at death
            program_metrics.update(self._extract_sparse_metrics(model))
            program_metrics["stage_at_death"] = self._classify_stage_at_death(
                s0_passed, s05_passed, s1_passed)

            # Novelty — compute behavioral fingerprint for S1 survivors
            with self._lock:
                self._progress.current_stage = "novelty"

            fp = None
            if s1_passed and model is not None:
                try:
                    fp = compute_fingerprint(
                        model,
                        seq_len=min(64, config.max_seq_len),
                        model_dim=config.model_dim,
                        vocab_size=config.vocab_size,
                        device=dev_str,
                    )
                except Exception:
                    pass

            # Diagnostic tasks — test specific architectural capabilities
            if s1_passed and model is not None:
                try:
                    diag = run_diagnostic_suite(model, device=dev_str)
                    program_metrics["diagnostic_tasks_json"] = json.dumps(diag.to_dict())
                    program_metrics["diagnostic_score"] = diag.diagnostic_score
                except Exception:
                    pass

            nov = novelty_score(graph, fingerprint=fp)
            n_score = nov.overall_novelty

            if s1_passed and n_score > 0.5:
                results["novel_count"] += 1
                with self._lock:
                    self._progress.novel_count += 1
                results["survivors"].append({
                    "fingerprint": graph.fingerprint(),
                    "novelty": n_score,
                    "loss_ratio": loss_ratio,
                })

            if loss_ratio and (results["best_loss_ratio"] is None
                               or loss_ratio < results["best_loss_ratio"]):
                results["best_loss_ratio"] = loss_ratio
                with self._lock:
                    self._progress.best_loss_ratio = loss_ratio

            if n_score and (results["best_novelty_score"] is None
                            or n_score > results["best_novelty_score"]):
                results["best_novelty_score"] = n_score
                with self._lock:
                    self._progress.best_novelty = n_score

            # Record program result with ALL metrics
            result_id = nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=graph.fingerprint(),
                graph_json=graph_to_json(graph),
                stage0_passed=s0_passed, stage05_passed=s05_passed,
                stage1_passed=s1_passed,
                stage0_error=sandbox_result.error,
                loss_ratio=loss_ratio, final_loss=final_loss,
                throughput=throughput, novelty_score=n_score,
                structural_novelty=nov.structural_novelty,
                behavioral_novelty=nov.behavioral_novelty,
                most_similar_to=nov.most_similar_to,
                novelty_confidence=nov.novelty_confidence,
                model_source="graph_synthesis",
                **program_metrics,
            )

            # Store training curve if available
            if training_curve and result_id:
                try:
                    nb.store_training_curve(result_id, training_curve)
                except Exception:
                    pass

            nb.log_metric("stage0_pass_rate",
                          results["stage0_passed"] / (i + 1),
                          experiment_id=exp_id)

            stage_label = "S1 PASS" if s1_passed else ("S0" if s0_passed else "FAIL")
            self._emit_event("program_evaluated", {
                "index": i,
                "fingerprint": graph.fingerprint()[:10],
                "result": stage_label,
                "novelty": round(n_score, 3),
                "loss_ratio": round(loss_ratio, 4) if loss_ratio else None,
                "params": sandbox_result.param_count,
            })

            # Cleanup
            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        with self._lock:
            self._progress.elapsed_seconds = time.time() - t_start
            self._progress.status = "analyzing"
            self._progress.aria_message = self.aria.begin_analysis()

        return results

    def _resolve_baseline_recipe(
        self,
        train_result: Dict[str, Any],
        default_lr: float,
        default_weight_decay: float = 0.01,
    ) -> Dict[str, Any]:
        """Resolve baseline training recipe from observed candidate metadata."""
        optimizer_name = "adamw"

        optimizer_class = str(train_result.get("optimizer_class") or "").lower()
        if "sgd" in optimizer_class:
            optimizer_name = "sgd"

        lr = float(
            train_result.get("final_lr")
            or train_result.get("optimizer_lr")
            or default_lr
        )
        weight_decay = float(
            train_result.get("optimizer_weight_decay", default_weight_decay)
        )
        momentum = float(train_result.get("optimizer_momentum", 0.0))

        beta1 = train_result.get("optimizer_beta1")
        beta2 = train_result.get("optimizer_beta2")
        betas: Optional[Tuple[float, float]] = None
        if beta1 is not None and beta2 is not None:
            betas = (float(beta1), float(beta2))

        tp_json = train_result.get("training_program_json")
        if tp_json and not optimizer_class:
            try:
                tp = json.loads(tp_json)
                opt = tp.get("optimizer") or {}
                opt_name = str(opt.get("name") or "").lower()
                comps = [str(c).lower() for c in (opt.get("components") or [])]
                if "sgd" in opt_name or "sgd" in comps:
                    optimizer_name = "sgd"
                if "lr" in opt:
                    lr = float(opt["lr"])
                if "weight_decay" in opt:
                    weight_decay = float(opt["weight_decay"])
            except Exception as e:
                logger.debug("Failed to parse training_program_json for baseline recipe: %s", e)

        return {
            "optimizer_name": optimizer_name,
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "betas": betas,
        }

    def _micro_train(self, model: nn.Module, config: RunConfig,
                     dev: torch.device, seed: int = 42) -> Dict:
        """Run Stage 1 micro-training with comprehensive metric capture.

        Uses deterministic seeding per step so all candidates see the same
        training data in the same order, enabling fair comparison (#56).
        """
        result: Dict[str, Any] = {"passed": False}

        try:
            model = model.to(dev)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=config.stage1_lr, weight_decay=0.01)

            result["optimizer_class"] = optimizer.__class__.__name__.lower()
            if optimizer.param_groups:
                pg0 = optimizer.param_groups[0]
                result["optimizer_lr"] = float(pg0.get("lr", config.stage1_lr))
                result["optimizer_weight_decay"] = float(pg0.get("weight_decay", 0.01))
                result["optimizer_momentum"] = float(pg0.get("momentum", 0.0))
                betas = pg0.get("betas")
                if isinstance(betas, tuple) and len(betas) == 2:
                    result["optimizer_beta1"] = float(betas[0])
                    result["optimizer_beta2"] = float(betas[1])

            initial_loss = None
            final_loss = None
            min_loss = float("inf")
            total_tokens = 0
            t_start = time.perf_counter()

            step_times: List[float] = []
            grad_norms: List[float] = []
            training_curve: List[Dict] = []

            seq_len = min(128, config.max_seq_len)

            for step in range(config.stage1_steps):
                if self._stop_event.is_set():
                    break

                input_ids = self._sample_training_input_ids(
                    config=config,
                    dev=dev,
                    batch_size=config.stage1_batch_size,
                    seq_len=seq_len,
                    seed=seed + step,
                )

                t_step = time.perf_counter()

                with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                        enabled=(dev.type == "cuda")):
                    logits = model(input_ids)
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.shape[-1]),
                        input_ids[:, 1:].reshape(-1),
                    )

                if torch.isnan(loss) or torch.isinf(loss):
                    result["error"] = f"NaN/Inf loss at step {step}"
                    result["n_train_steps"] = step
                    return result

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                optimizer.step()

                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)

                t_step_end = time.perf_counter()
                step_time_ms = (t_step_end - t_step) * 1000

                loss_val = loss.item()
                if step == 0:
                    initial_loss = loss_val
                final_loss = loss_val
                min_loss = min(min_loss, loss_val)
                total_tokens += input_ids.numel()

                step_times.append(step_time_ms)
                grad_norms.append(grad_norm)

                # Record per-step data
                training_curve.append({
                    "step": step,
                    "loss": loss_val,
                    "grad_norm": grad_norm,
                    "step_time_ms": step_time_ms,
                })

            t_end = time.perf_counter()
            total_time_ms = (t_end - t_start) * 1000

            if initial_loss and final_loss:
                result["loss_ratio"] = final_loss / max(initial_loss, 1e-6)
                result["final_loss"] = final_loss
                result["initial_loss"] = initial_loss
                result["min_loss"] = min_loss
                result["throughput"] = total_tokens / (total_time_ms / 1000)
                result["passed"] = result["loss_ratio"] < 0.8

                # Compute improvement rate
                if initial_loss > 0:
                    result["loss_improvement_rate"] = (initial_loss - final_loss) / initial_loss

                # Timing stats
                result["avg_step_time_ms"] = sum(step_times) / len(step_times) if step_times else 0
                result["total_train_time_ms"] = total_time_ms

                # Gradient norm stats
                if grad_norms:
                    result["max_grad_norm"] = max(grad_norms)
                    result["mean_grad_norm"] = sum(grad_norms) / len(grad_norms)
                    mean_gn = result["mean_grad_norm"]
                    result["grad_norm_std"] = (
                        sum((g - mean_gn) ** 2 for g in grad_norms) / len(grad_norms)
                    ) ** 0.5

                result["n_train_steps"] = len(step_times)
                result["final_lr"] = config.stage1_lr  # constant for now
                result["training_curve"] = training_curve

        except Exception as e:
            result["error"] = str(e)

        if result.get("final_loss") is not None and bool(getattr(config, "one_shot_pruning_baseline", False)):
            try:
                seq_len = min(128, int(config.max_seq_len))
                eval_batches = max(1, int(getattr(config, "one_shot_pruning_eval_batches", 4)))
                eval_batch_size = max(1, int(getattr(config, "one_shot_pruning_batch_size", 2)))

                eval_inputs = [
                    self._sample_training_input_ids(
                        config=config,
                        dev=dev,
                        batch_size=eval_batch_size,
                        seq_len=seq_len,
                        seed=seed + 100_000 + i,
                    )
                    for i in range(eval_batches)
                ]

                dense_eval_loss = estimate_lm_ce_loss(model, eval_inputs, dev)

                pruned_model = copy.deepcopy(model).to(dev)
                prune_info = apply_one_shot_pruning(
                    pruned_model,
                    target_sparsity=float(getattr(config, "one_shot_pruning_sparsity", 0.5)),
                    method=str(getattr(config, "one_shot_pruning_method", "wanda")),
                )
                pruned_eval_loss = estimate_lm_ce_loss(pruned_model, eval_inputs, dev)

                quality_retention = None
                if dense_eval_loss is not None and pruned_eval_loss is not None and pruned_eval_loss > 0:
                    quality_retention = max(0.0, min(1.5, dense_eval_loss / pruned_eval_loss))

                result["pruning_method"] = prune_info.method
                result["pruning_target_sparsity"] = prune_info.target_sparsity
                result["pruning_actual_sparsity"] = prune_info.actual_sparsity
                result["pruning_n_params_total"] = prune_info.n_params_total
                result["pruning_n_params_pruned"] = prune_info.n_params_pruned
                result["pruning_dense_eval_loss"] = dense_eval_loss
                result["pruning_pruned_eval_loss"] = pruned_eval_loss
                result["pruning_quality_retention"] = quality_retention
                if prune_info.n_params_total > 0:
                    result["pruning_active_params_estimate"] = (
                        prune_info.n_params_total - prune_info.n_params_pruned
                    )

                del pruned_model
            except Exception as e:
                result["pruning_error"] = str(e)

        return result

    def _analyze_results(self, results: Dict, exp_id: str,
                         nb: LabNotebook, context: str = "") -> List[str]:
        """Analyze experiment results and generate insights."""
        # Try data-driven analytics first
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            structured = analytics.compute_insights()

            # Deduplicate: normalize numbers out of content so
            # "appears in 144 survivors" matches "appears in 145 survivors"
            def _dedup_key(text: str) -> str:
                s = re.sub(r'\d+\.\d+%?', '#', text)   # decimals / pcts
                s = re.sub(r'\b\d{2,}\b', '#', s)       # multi-digit ints
                return s

            # Build map: dedup_key -> list of existing insight rows
            existing_by_key: dict = {}
            for row in nb.get_insights(limit=500):
                key = _dedup_key(row["content"])
                existing_by_key.setdefault(key, []).append(row)

            recorded = []
            for ins in structured:
                content = ins if isinstance(ins, str) else ins.get("content", "")
                key = _dedup_key(content)
                category = ins.get("category", "pattern") if isinstance(ins, dict) else "pattern"
                confidence = ins.get("confidence", 0.7) if isinstance(ins, dict) else 0.7

                old_entries = existing_by_key.get(key, [])
                if old_entries:
                    # Supersede all old versions of this insight
                    for old in old_entries:
                        nb.supersede_insight(old["insight_id"])
                    existing_by_key[key] = []

                nb.record_insight(category, content, exp_id, confidence=confidence)
                self.aria.add_insight(content)
                recorded.append(content)
            return recorded
        except Exception:
            pass

        # Fall back to rule-based
        return self._rule_based_insights(results, exp_id, nb)

    def _rule_based_insights(self, results: Dict, exp_id: str,
                              nb: LabNotebook) -> List[str]:
        """Rule-based insight generation (always runs)."""
        insights = []
        aria = self.aria

        s0_rate = results["stage0_passed"] / max(results["total"], 1)
        s1_rate = results["stage1_passed"] / max(results["total"], 1)

        if s0_rate < 0.2:
            insight = "Low Stage 0 pass rate — grammar produces too many invalid programs. Consider tightening shape constraints."
            insights.append(insight)
            nb.record_insight("failure_mode", insight, exp_id, confidence=0.7)
            aria.add_insight(insight)

        if s0_rate > 0.5 and s1_rate < 0.01:
            insight = "Programs compile but don't learn. The operations may not compose into learnable functions. Need more parameterized ops."
            insights.append(insight)
            nb.record_insight("failure_mode", insight, exp_id, confidence=0.6)
            aria.add_insight(insight)

        if results["novel_count"] > 0:
            insight = f"Found {results['novel_count']} genuinely novel survivors! Behaviorally distinct from known architectures."
            insights.append(insight)
            nb.record_insight("success_factor", insight, exp_id, confidence=0.8)
            aria.add_insight(insight)

        if s1_rate > 0.05:
            insight = f"Strong Stage 1 pass rate ({s1_rate:.0%}). Current grammar configuration is productive."
            insights.append(insight)
            nb.record_insight("pattern", insight, exp_id, confidence=0.7)
            aria.add_insight(insight)

        return insights

    # ── Rich Context Helpers ──

    def _gather_analytics_data(self, nb: LabNotebook) -> Dict:
        """Gather all analytics data for rich context."""
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return {
                "op_success_rates": analytics.op_success_rates(),
                "structural_correlations": analytics.structural_correlations(),
                "failure_patterns": analytics.failure_patterns(),
                "top_op_combinations": analytics.top_op_combinations(10),
                "efficiency_frontier": analytics.efficiency_frontier(),
                "grammar_weights": analytics.compute_grammar_weights(),
                "default_weights": analytics.get_current_grammar_weights(),
                "learning_log": nb.get_learning_log(limit=10),
                "insights": nb.get_insights(limit=20),
                "negative_results": analytics.negative_results_synthesis(),
            }
        except Exception:
            return {}

    def _get_past_hypotheses(self, nb: LabNotebook, limit: int = 5) -> List[Dict]:
        """Get past hypotheses with their outcomes."""
        experiments = nb.get_recent_experiments(limit * 2)
        past = []
        for exp in experiments:
            hyp = exp.get("hypothesis")
            if not hyp:
                continue
            s1_count = exp.get("n_stage1_passed", 0)
            best_novelty = exp.get("best_novelty_score", 0)
            past.append({
                "hypothesis": hyp,
                "confirmed": s1_count > 0,
                "s1_count": s1_count,
                "best_novelty": best_novelty or 0,
                "experiment_id": exp.get("experiment_id"),
            })
            if len(past) >= limit:
                break
        return past

    def _auto_recommend(self, results: Dict, config: RunConfig,
                        hypothesis: str, nb: LabNotebook):
        """Auto-generate a recommendation after experiment completion."""
        try:
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)
            suggestion = self.aria.suggest_experiment(context)
            if suggestion:
                with self._lock:
                    self._last_recommendation = suggestion
                self._emit_event("aria_recommendation", {
                    "reasoning": suggestion.get("reasoning", ""),
                    "confidence": suggestion.get("confidence", 0),
                    "config": suggestion.get("config", {}),
                })
                # Store as notebook entry
                nb.add_entry(ExperimentEntry(
                    entry_type="decision",
                    title="Aria's Next Experiment Recommendation",
                    content=suggestion.get("reasoning", ""),
                    metadata={
                        "confidence": suggestion.get("confidence", 0),
                        "suggested_config": suggestion.get("config", {}),
                    },
                ))
        except Exception as e:
            logger.debug(f"Auto-recommendation failed: {e}")

    def _build_rich_context_for_experiment(
        self, results: Dict, config: RunConfig,
        hypothesis: str, nb: LabNotebook,
    ) -> str:
        """Build rich context string for an experiment."""
        analytics_data = self._gather_analytics_data(nb)
        history = nb.get_recent_experiments(10)
        past_hypotheses = self._get_past_hypotheses(nb)
        return build_rich_context(
            results=results,
            config=config.to_dict(),
            hypothesis=hypothesis,
            analytics_data=analytics_data,
            history=history,
            past_hypotheses=past_hypotheses,
        )

    # ── Automation: Auto-Scale-Up & Auto-Report ──

    def _maybe_auto_scale_up(self, results: Dict, config: RunConfig,
                              nb: LabNotebook):
        """Check if we should auto-trigger scale-up after an experiment.

        Criteria:
        1. auto_scale_up is enabled in config
        2. Enough S1 survivors (>= auto_scale_up_min_survivors)
        3. Survivors have sufficient novelty (>= auto_scale_up_min_novelty avg)
        4. Not already a scale_up experiment (avoid recursion)
        5. No experiment currently running
        """
        if not config.auto_scale_up:
            return
        if config.scale_up:
            return  # don't chain scale-ups

        survivors = results.get("survivors", [])
        s1_count = results.get("stage1_passed", 0)

        if s1_count < config.auto_scale_up_min_survivors:
            return

        # Check novelty
        if survivors:
            avg_novelty = sum(s.get("novelty", 0) for s in survivors) / len(survivors)
            if avg_novelty < config.auto_scale_up_min_novelty:
                return

        # Select top programs by loss ratio
        top_programs = nb.get_top_programs(
            config.auto_scale_up_top_n, sort_by="loss_ratio")
        result_ids = [
            p["result_id"] for p in top_programs
            if p.get("stage1_passed")
        ][:config.auto_scale_up_top_n]

        if not result_ids:
            return

        logger.info(
            f"Auto-scale-up triggered: {len(result_ids)} programs qualify "
            f"(s1={s1_count}, survivors={len(survivors)})"
        )

        # Store the intent — can't start immediately since thread is still
        # running. Schedule via a flag the main thread can pick up.
        self._pending_scale_up = {
            "result_ids": result_ids,
            "config": config,
            "hypothesis": (
                f"Auto-scale-up: validating top {len(result_ids)} performers "
                f"at {config.scale_up_steps} steps to confirm they work at scale."
            ),
        }

        self._emit_event("auto_scale_up_queued", {
            "result_ids": result_ids,
            "n_programs": len(result_ids),
            "reason": f"{s1_count} S1 survivors with avg novelty >= {config.auto_scale_up_min_novelty}",
        })

        nb.add_entry(ExperimentEntry(
            entry_type="decision",
            title="Auto-Scale-Up Triggered",
            content=(
                f"Automatically queuing scale-up validation for {len(result_ids)} "
                f"top performers. Criteria met: {s1_count} S1 survivors."
            ),
            metadata={"result_ids": result_ids},
        ))

    def _maybe_auto_report(self, config: RunConfig, nb: LabNotebook,
                            reason: str = "session_end"):
        """Auto-generate and store a research report."""
        if not config.auto_report:
            return

        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)

            report_data = {
                "summary": nb.get_dashboard_summary(),
                "top_programs": nb.get_top_programs(20, sort_by="loss_ratio"),
                "recent_experiments": nb.get_recent_experiments(100),
                "op_success_rates": analytics.op_success_rates(),
                "structural_correlations": analytics.structural_correlations(),
                "failure_patterns": analytics.failure_patterns(),
                "top_op_combinations": analytics.top_op_combinations(10),
                "efficiency_frontier": analytics.efficiency_frontier(),
                "grammar_weights": analytics.compute_grammar_weights() or {},
                "default_weights": analytics.get_current_grammar_weights(),
            }

            narrative = self.aria.generate_report_narrative(report_data)

            nb.add_entry(ExperimentEntry(
                entry_type="report",
                title=f"Research Report ({reason})",
                content=narrative,
                metadata={
                    "trigger": reason,
                    "total_experiments": report_data["summary"].get("total_experiments", 0),
                    "stage1_survivors": report_data["summary"].get("stage1_survivors", 0),
                },
            ))

            # Save as markdown file for human/LLM consumption
            nb.save_report_markdown(narrative, reason, report_data["summary"])

            self._emit_event("auto_report_generated", {
                "reason": reason,
                "narrative_length": len(narrative),
                "summary": report_data["summary"],
            })

            logger.info(f"Auto-report generated ({reason}): {len(narrative)} chars")
        except Exception as e:
            logger.warning(f"Auto-report generation failed: {e}")

    def _run_pending_scale_up(self):
        """Launch pending auto-scale-up, auto-investigation, or auto-validation."""
        # Check investigation first (higher priority)
        self._run_pending_investigation()
        if self.is_running:
            return

        # Then validation
        self._run_pending_validation()
        if self.is_running:
            return

        # Then scale-up
        pending = getattr(self, "_pending_scale_up", None)
        if pending is None:
            return
        self._pending_scale_up = None

        if self.is_running:
            return  # something else started, skip

        try:
            self.start_scale_up(
                result_ids=pending["result_ids"],
                config=pending["config"],
                hypothesis=pending["hypothesis"],
            )
        except Exception as e:
            logger.warning(f"Failed to launch auto-scale-up: {e}")

    # ── Model Source Abstraction ──

    def _generate_candidates(self, config: RunConfig, n: int,
                             source: str = "graph_synthesis") -> List[ModelCandidate]:
        """Generate candidate models from the specified source.

        source: "graph_synthesis", "morphological_box", or "mixed"
        Returns candidates that pass Stage 0 smoke test.
        """
        candidates: List[ModelCandidate] = []
        dev_str = config.device if torch.cuda.is_available() else "cpu"

        if source == "mixed":
            n_morph = int(n * config.morph_ratio)
            n_graph = n - n_morph
            candidates.extend(
                self._generate_candidates(config, n_graph, "graph_synthesis"))
            candidates.extend(
                self._generate_candidates(config, n_morph, "morphological_box"))
            return candidates

        if source == "morphological_box":
            try:
                from ..morphological_box import roll, describe_spec
                from ..arch_builder import build_model, BuildConfig

                build_cfg = BuildConfig(
                    dim=config.model_dim,
                    n_layers=config.n_layers,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                )

                for i in range(n):
                    if self._stop_event.is_set():
                        break
                    try:
                        spec = roll(seed=i + int(time.time() * 1000) % 100000,
                                    generation=0)
                        model = build_model(spec, build_cfg)
                        desc = describe_spec(spec)

                        # Quick smoke test
                        sandbox_result = safe_eval(
                            model, batch_size=2,
                            seq_len=min(128, config.max_seq_len),
                            vocab_size=config.vocab_size,
                            device=dev_str,
                        )
                        if sandbox_result.passed:
                            import json as _json
                            candidates.append(ModelCandidate(
                                source="morphological_box",
                                model=model,
                                description=desc,
                                arch_spec=spec,
                                arch_spec_json=_json.dumps(spec.to_dict()),
                                fingerprint=spec.id,
                            ))
                        else:
                            del model
                    except Exception as e:
                        logger.debug(f"Morphological candidate {i} failed: {e}")
                        continue
            except ImportError:
                logger.warning("morphological_box or arch_builder not available")
            return candidates

        # Default: graph_synthesis
        grammar = GrammarConfig(
            model_dim=config.model_dim,
            max_depth=config.max_depth,
            max_ops=config.max_ops,
            residual_prob=config.residual_prob,
        )
        grammar.category_weights["math_space"] = config.math_space_weight

        graphs = batch_generate(n, grammar)
        for graph in graphs:
            if self._stop_event.is_set():
                break
            validation = validate_graph(graph)
            if not validation.valid:
                continue
            try:
                layer_graphs = [graph] * config.n_layers
                model = compile_model(
                    layer_graphs,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                )
                sandbox_result = safe_eval(
                    model, batch_size=2,
                    seq_len=min(128, config.max_seq_len),
                    vocab_size=config.vocab_size,
                    device=dev_str,
                )
                if sandbox_result.passed:
                    candidates.append(ModelCandidate(
                        source="graph_synthesis",
                        model=model,
                        description=graph_summary(graph),
                        graph=graph,
                        graph_json=graph_to_json(graph),
                        fingerprint=graph.fingerprint(),
                    ))
                else:
                    del model
            except Exception:
                continue

        return candidates

    # ── Training with synthesized programs ──

    def _train_with_program(self, model: nn.Module, program,
                            config: RunConfig,
                            dev: torch.device,
                            seed: int = 42) -> Dict:
        """Train a model using a synthesized TrainingProgram.

        Returns same metrics dict as _micro_train() plus training_program_json.
        """
        result: Dict[str, Any] = {"passed": False}

        try:
            model = model.to(dev)
            model.train()

            # Apply init scheme
            if program.init_scheme == "small":
                for p in model.parameters():
                    if p.dim() >= 2:
                        nn.init.normal_(p, std=program.init_scale)
            elif program.init_scheme == "orthogonal":
                for m in model.modules():
                    if isinstance(m, (nn.Linear, nn.Conv1d)):
                        nn.init.orthogonal_(m.weight, gain=program.init_scale)
            elif program.init_scheme == "spectral":
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight)

            # Create optimizer from program
            try:
                optimizer = program.optimizer.create(model.parameters())
            except Exception:
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=3e-4, weight_decay=0.01)

            result["optimizer_class"] = optimizer.__class__.__name__.lower()
            if optimizer.param_groups:
                pg0 = optimizer.param_groups[0]
                result["optimizer_lr"] = float(pg0.get("lr", 3e-4))
                result["optimizer_weight_decay"] = float(pg0.get("weight_decay", 0.01))
                result["optimizer_momentum"] = float(pg0.get("momentum", 0.0))
                betas = pg0.get("betas")
                if isinstance(betas, tuple) and len(betas) == 2:
                    result["optimizer_beta1"] = float(betas[0])
                    result["optimizer_beta2"] = float(betas[1])

            n_steps = program.n_steps
            batch_size = program.batch_size
            max_grad_norm_val = program.max_grad_norm

            initial_loss = None
            final_loss = None
            min_loss = float("inf")
            total_tokens = 0
            t_start = time.perf_counter()

            step_times: List[float] = []
            grad_norms: List[float] = []
            training_curve: List[Dict] = []

            seq_len = min(128, config.max_seq_len)
            # Apply curriculum seq_len schedule
            try:
                base_seq = program.curriculum.get_seq_len(0, n_steps)
                if base_seq and base_seq > 0:
                    seq_len = min(base_seq, config.max_seq_len)
            except Exception:
                pass

            for step in range(n_steps):
                if self._stop_event.is_set():
                    break

                # Update seq_len from curriculum
                try:
                    curr_seq = program.curriculum.get_seq_len(step, n_steps)
                    if curr_seq and curr_seq > 0:
                        seq_len = min(curr_seq, config.max_seq_len)
                except Exception:
                    pass

                input_ids = self._sample_training_input_ids(
                    config=config,
                    dev=dev,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    seed=seed + step,
                )

                t_step = time.perf_counter()

                with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                        enabled=(dev.type == "cuda")):
                    logits = model(input_ids)
                    # Use synthesized loss if possible
                    try:
                        loss = program.loss.compute(
                            logits[:, :-1].reshape(-1, logits.shape[-1]),
                            input_ids[:, 1:].reshape(-1),
                        )
                    except Exception:
                        loss = F.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]),
                            input_ids[:, 1:].reshape(-1),
                        )

                if torch.isnan(loss) or torch.isinf(loss):
                    result["error"] = f"NaN/Inf loss at step {step}"
                    result["n_train_steps"] = step
                    return result

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm_val).item()
                optimizer.step()

                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)

                t_step_end = time.perf_counter()
                step_time_ms = (t_step_end - t_step) * 1000

                loss_val = loss.item()
                if step == 0:
                    initial_loss = loss_val
                final_loss = loss_val
                min_loss = min(min_loss, loss_val)
                total_tokens += input_ids.numel()

                step_times.append(step_time_ms)
                grad_norms.append(grad_norm)

                training_curve.append({
                    "step": step,
                    "loss": loss_val,
                    "grad_norm": grad_norm,
                    "step_time_ms": step_time_ms,
                })

            t_end = time.perf_counter()
            total_time_ms = (t_end - t_start) * 1000

            if initial_loss and final_loss:
                result["loss_ratio"] = final_loss / max(initial_loss, 1e-6)
                result["final_loss"] = final_loss
                result["initial_loss"] = initial_loss
                result["min_loss"] = min_loss
                result["throughput"] = total_tokens / (total_time_ms / 1000)
                result["passed"] = result["loss_ratio"] < 0.8

                if initial_loss > 0:
                    result["loss_improvement_rate"] = (initial_loss - final_loss) / initial_loss

                result["avg_step_time_ms"] = sum(step_times) / len(step_times) if step_times else 0
                result["total_train_time_ms"] = total_time_ms

                if grad_norms:
                    result["max_grad_norm"] = max(grad_norms)
                    result["mean_grad_norm"] = sum(grad_norms) / len(grad_norms)
                    mean_gn = result["mean_grad_norm"]
                    result["grad_norm_std"] = (
                        sum((g - mean_gn) ** 2 for g in grad_norms) / len(grad_norms)
                    ) ** 0.5

                result["n_train_steps"] = len(step_times)
                result["final_lr"] = getattr(optimizer, 'defaults', {}).get('lr', 3e-4)
                result["training_curve"] = training_curve
                result["training_program_json"] = json.dumps(program.to_dict())

        except Exception as e:
            result["error"] = str(e)

        return result

    # ── OOD Robustness Testing (#54) ──

    # Hand-designed reference training recipes for out-of-distribution testing.
    # Each recipe exercises a different optimizer/LR/schedule to test whether
    # a candidate's learnability is robust or just an artifact of one recipe.
    _REFERENCE_RECIPES = [
        {
            "name": "sgd_high_lr",
            "optimizer": "sgd",
            "lr": 1e-2,
            "momentum": 0.9,
            "weight_decay": 0.0,
        },
        {
            "name": "adamw_low_lr",
            "optimizer": "adamw",
            "lr": 1e-4,
            "weight_decay": 0.1,
        },
        {
            "name": "adamw_high_lr",
            "optimizer": "adamw",
            "lr": 1e-3,
            "weight_decay": 0.01,
        },
    ]

    def _ood_robustness_check(
        self,
        model_factory: Callable[[], nn.Module],
        config: RunConfig,
        dev: torch.device,
        n_steps: int = 300,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Test a candidate against hand-designed reference training recipes.

        Returns a dict with per-recipe results and an overall robustness score
        (fraction of recipes that achieved loss_ratio < 0.9).
        """
        recipe_results = []

        for recipe in self._REFERENCE_RECIPES:
            if self._stop_event.is_set():
                break

            try:
                model = model_factory().to(dev)
                model.train()

                if recipe["optimizer"] == "sgd":
                    optimizer = torch.optim.SGD(
                        model.parameters(),
                        lr=recipe["lr"],
                        momentum=recipe.get("momentum", 0.0),
                        weight_decay=recipe.get("weight_decay", 0.0),
                    )
                else:  # adamw
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=recipe["lr"],
                        weight_decay=recipe.get("weight_decay", 0.01),
                    )

                seq_len = min(128, config.max_seq_len)
                initial_loss = None
                final_loss = None

                for step in range(n_steps):
                    if self._stop_event.is_set():
                        break

                    input_ids = self._sample_training_input_ids(
                        config=config,
                        dev=dev,
                        batch_size=config.stage1_batch_size,
                        seq_len=seq_len,
                        seed=seed + step,
                    )

                    with torch.amp.autocast(
                        device_type=dev.type, dtype=torch.bfloat16,
                        enabled=(dev.type == "cuda"),
                    ):
                        logits = model(input_ids)
                        loss = F.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]),
                            input_ids[:, 1:].reshape(-1),
                        )

                    if torch.isnan(loss) or torch.isinf(loss):
                        break

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    loss_val = loss.item()
                    if step == 0:
                        initial_loss = loss_val
                    final_loss = loss_val

                loss_ratio = (final_loss / max(initial_loss, 1e-6)
                              if initial_loss and final_loss else None)
                recipe_results.append({
                    "recipe": recipe["name"],
                    "loss_ratio": round(loss_ratio, 4) if loss_ratio else None,
                    "passed": loss_ratio is not None and loss_ratio < 0.9,
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                })

                del model
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                recipe_results.append({
                    "recipe": recipe["name"],
                    "loss_ratio": None,
                    "passed": False,
                    "error": str(e),
                })

        n_passed = sum(1 for r in recipe_results if r.get("passed"))
        return {
            "recipes_tested": len(recipe_results),
            "recipes_passed": n_passed,
            "ood_robustness": n_passed / max(len(recipe_results), 1),
            "recipe_results": recipe_results,
        }

    # ── Hyperparameter Sensitivity (#57) ──

    # Perturbations to test: each is (label, param_overrides) where overrides
    # are multipliers applied to the base config values.
    _SENSITIVITY_PERTURBATIONS = [
        ("lr_half", {"lr_mult": 0.5}),
        ("lr_double", {"lr_mult": 2.0}),
        ("steps_half", {"steps_mult": 0.5}),
        ("steps_double", {"steps_mult": 2.0}),
    ]

    def _sensitivity_check(
        self,
        model_factory: Callable[[], nn.Module],
        config: RunConfig,
        dev: torch.device,
        base_loss_ratio: float,
        n_steps: int = 300,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Test whether a candidate's performance is sensitive to hyperparameter changes.

        Trains the model with ±2x learning rate and ±2x training steps.
        Returns per-perturbation loss ratios and an overall sensitivity score.
        A robust candidate should learn under all perturbations (loss_ratio < 1.0).
        """
        perturbation_results = []
        base_lr = config.stage1_lr

        for label, overrides in self._SENSITIVITY_PERTURBATIONS:
            if self._stop_event.is_set():
                break

            lr = base_lr * overrides.get("lr_mult", 1.0)
            steps = int(n_steps * overrides.get("steps_mult", 1.0))

            try:
                model = model_factory().to(dev)
                model.train()
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=lr, weight_decay=0.01)

                seq_len = min(128, config.max_seq_len)
                initial_loss = None
                final_loss = None

                for step in range(steps):
                    if self._stop_event.is_set():
                        break

                    input_ids = self._sample_training_input_ids(
                        config=config,
                        dev=dev,
                        batch_size=config.stage1_batch_size,
                        seq_len=seq_len,
                        seed=seed + step,
                    )

                    with torch.amp.autocast(
                        device_type=dev.type, dtype=torch.bfloat16,
                        enabled=(dev.type == "cuda"),
                    ):
                        logits = model(input_ids)
                        loss = F.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.shape[-1]),
                            input_ids[:, 1:].reshape(-1),
                        )

                    if torch.isnan(loss) or torch.isinf(loss):
                        break

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    loss_val = loss.item()
                    if step == 0:
                        initial_loss = loss_val
                    final_loss = loss_val

                loss_ratio = (final_loss / max(initial_loss, 1e-6)
                              if initial_loss and final_loss else None)

                # How much did loss_ratio change vs the base run?
                deviation = (abs(loss_ratio - base_loss_ratio) / max(base_loss_ratio, 1e-6)
                             if loss_ratio is not None else None)

                perturbation_results.append({
                    "perturbation": label,
                    "lr": lr,
                    "steps": steps,
                    "loss_ratio": round(loss_ratio, 4) if loss_ratio else None,
                    "deviation_from_base": round(deviation, 4) if deviation is not None else None,
                    "still_learns": loss_ratio is not None and loss_ratio < 1.0,
                })

                del model
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                perturbation_results.append({
                    "perturbation": label,
                    "loss_ratio": None,
                    "still_learns": False,
                    "error": str(e),
                })

        n_learns = sum(1 for r in perturbation_results if r.get("still_learns"))
        deviations = [r["deviation_from_base"] for r in perturbation_results
                      if r.get("deviation_from_base") is not None]
        avg_deviation = sum(deviations) / len(deviations) if deviations else None

        return {
            "perturbations_tested": len(perturbation_results),
            "perturbations_learn": n_learns,
            "hp_robustness": n_learns / max(len(perturbation_results), 1),
            "avg_deviation": round(avg_deviation, 4) if avg_deviation is not None else None,
            "perturbation_results": perturbation_results,
        }

    # ── Investigation Phase ──

    def start_investigation(self, result_ids: List[str], config: RunConfig,
                            hypothesis: Optional[str] = None) -> str:
        """Start investigation phase for selected candidates."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()

        # Tier guard: reject result IDs already at investigation tier or beyond
        tiers = nb.get_tiers_for_result_ids(result_ids)
        already_done = {
            rid: tier for rid, tier in tiers.items()
            if tier in ("investigation", "validation", "breakthrough")
        }
        if already_done:
            nb.close()
            labels = ", ".join(f"{rid} ({tier})" for rid, tier in already_done.items())
            raise ValueError(
                f"Cannot investigate: {len(already_done)} candidate(s) already "
                f"at or beyond investigation tier: {labels}"
            )

        source = "user_input" if hypothesis is not None else "runner_template"
        if hypothesis is None:
            hypothesis = (
                f"Investigation: deep study of {len(result_ids)} screening survivors "
                f"with multiple training programs to test robustness."
            )

        exp_id = nb.start_experiment(
            experiment_type="investigation",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source=source,
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=len(result_ids),
                aria_message=f"{self.aria.NAME}: Starting investigation of {len(result_ids)} candidate(s)...",
            )

        self._emit_event("investigation_started", {
            "experiment_id": exp_id,
            "hypothesis": hypothesis,
            "result_ids": result_ids,
            "n_training_programs": config.n_training_programs,
        })

        self._thread = threading.Thread(
            target=self._run_investigation_thread,
            args=(exp_id, result_ids, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def _run_investigation_thread(self, exp_id: str, result_ids: List[str],
                                   config: RunConfig, hypothesis: str):
        """Execute investigation phase in background."""
        nb = self._make_notebook()
        t_start = time.time()
        ckpt = CheckpointManager(config.checkpoint_dir)

        # Load phase checkpoint to find where we left off
        resume_from_candidate = 0
        ckpt_state = ckpt.load_phase(exp_id, "investigation", -1, 0)
        if ckpt_state:
            resume_from_candidate = ckpt_state.get("candidate_idx", 0)
            logger.info("Resuming investigation from candidate %d", resume_from_candidate)

        try:
            results = {
                "total": len(result_ids), "stage0_passed": 0, "stage05_passed": 0,
                "stage1_passed": 0, "novel_count": 0,
                "best_loss_ratio": None, "best_novelty_score": None,
                "survivors": [], "investigation_results": [],
            }

            dev_str = config.device if torch.cuda.is_available() else "cpu"
            dev = torch.device(dev_str)

            inv_config = RunConfig.from_dict(config.to_dict())
            inv_config.stage1_steps = config.investigation_steps
            inv_config.stage1_batch_size = config.investigation_batch_size

            for prog_idx, source_result_id in enumerate(result_ids):
                if prog_idx < resume_from_candidate:
                    continue
                if self._stop_event.is_set():
                    break

                with self._lock:
                    self._progress.current_program = prog_idx + 1
                    self._progress.status = "investigating"
                    self._progress.aria_message = (
                        f"Investigating {prog_idx + 1}/{len(result_ids)}: "
                        f"{source_result_id[:8]}... "
                        f"({config.n_training_programs} training programs)"
                    )
                    self._progress.elapsed_seconds = time.time() - t_start

                self._emit_event("investigation_progress", {
                    "experiment_id": exp_id,
                    "current": prog_idx + 1,
                    "total": len(result_ids),
                    "source_result_id": source_result_id,
                    "status": "starting",
                })

                # Fetch source program
                source = nb.get_program_detail(source_result_id)
                if source is None:
                    continue

                # Reconstruct model
                graph_json_str = source.get("graph_json")
                arch_spec_json_str = source.get("arch_spec_json")
                model_source = source.get("model_source", "graph_synthesis")

                # Generate training programs
                training_programs = []
                for tp_i in range(config.n_training_programs):
                    tp = synthesize_training_program(
                        n_steps=config.investigation_steps,
                        max_seq_len=config.max_seq_len,
                        seed=tp_i + prog_idx * 1000,
                    )
                    training_programs.append(tp)

                # Test each (model x training_program) pair
                tp_results = []
                for tp_i, tp in enumerate(training_programs):
                    if self._stop_event.is_set():
                        break

                    # Reconstruct model fresh for each training program
                    try:
                        if model_source == "morphological_box" and arch_spec_json_str:
                            from ..morphological_box import ArchSpec
                            from ..arch_builder import build_model, BuildConfig
                            spec_data = json.loads(arch_spec_json_str)
                            spec = ArchSpec(**spec_data)
                            build_cfg = BuildConfig(
                                dim=config.model_dim,
                                n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.max_seq_len,
                            )
                            model = build_model(spec, build_cfg)
                        elif graph_json_str:
                            graph = graph_from_json(graph_json_str)
                            layer_graphs = [graph] * config.n_layers
                            model = compile_model(
                                layer_graphs,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.max_seq_len,
                            )
                        else:
                            continue
                    except Exception as e:
                        logger.debug(f"Model reconstruction failed: {e}")
                        continue

                    self._emit_event("investigation_progress", {
                        "experiment_id": exp_id,
                        "current": prog_idx + 1,
                        "total": len(result_ids),
                        "source_result_id": source_result_id,
                        "training_program": tp_i + 1,
                        "total_programs": len(training_programs),
                        "status": f"training with {tp.name}",
                    })

                    # Train with this program
                    tp_result = self._train_with_program(
                        model,
                        tp,
                        inv_config,
                        dev,
                        seed=self._stable_seed(exp_id, source_result_id, tp_i, "investigation_inline"),
                    )
                    tp_results.append({
                        "training_program": tp.name,
                        "passed": tp_result.get("passed", False),
                        "loss_ratio": tp_result.get("loss_ratio"),
                        "final_loss": tp_result.get("final_loss"),
                    })

                    del model
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

                # Compute robustness
                n_passed = sum(1 for r in tp_results if r.get("passed"))
                robustness = n_passed / max(len(tp_results), 1)
                best_tp = min(
                    (r for r in tp_results if r.get("loss_ratio") is not None),
                    key=lambda r: r["loss_ratio"],
                    default=None,
                )
                best_lr = best_tp["loss_ratio"] if best_tp else None
                screening_lr = source.get("loss_ratio")
                lr_multiplier = self._investigation_loss_multiplier(screening_lr, best_lr)
                brittle_risk = (
                    lr_multiplier is not None
                    and lr_multiplier > float(config.investigation_max_loss_ratio_multiplier)
                )

                if n_passed > 0:
                    results["stage1_passed"] += 1
                results["stage0_passed"] += 1
                results["stage05_passed"] += 1

                investigation_entry = {
                    "result_id": source_result_id,
                    "robustness": robustness,
                    "best_loss_ratio": best_lr,
                    "screening_loss_ratio": screening_lr,
                    "baseline_loss_ratio": source.get("baseline_loss_ratio"),
                    "novelty_confidence": source.get("novelty_confidence"),
                    "loss_ratio_multiplier": lr_multiplier,
                    "brittle_risk": brittle_risk,
                    "n_programs_passed": n_passed,
                    "n_programs_tested": len(tp_results),
                    "best_training_program": best_tp.get("training_program") if best_tp else None,
                }
                results["investigation_results"].append(investigation_entry)

                if best_lr and (results["best_loss_ratio"] is None
                                or best_lr < results["best_loss_ratio"]):
                    results["best_loss_ratio"] = best_lr

                # Update leaderboard
                best_tp_json = None
                if best_tp and best_tp.get("training_program"):
                    for tp in training_programs:
                        if tp.name == best_tp["training_program"]:
                            best_tp_json = json.dumps(tp.to_dict())
                            break

                investigation_passed = (
                    robustness >= 0.5
                    and (best_lr or 1.0) < 0.5
                    and not brittle_risk
                )

                nb.upsert_leaderboard(
                    result_id=source_result_id,
                    model_source=model_source,
                    architecture_desc=source.get("graph_fingerprint", "")[:40],
                    screening_loss_ratio=source.get("loss_ratio"),
                    screening_novelty=source.get("novelty_score"),
                    screening_passed=True,
                    investigation_loss_ratio=best_lr,
                    investigation_robustness=robustness,
                    investigation_best_training=best_tp_json,
                    investigation_passed=investigation_passed,
                    tier="investigation" if investigation_passed else "screening",
                    novelty_confidence=source.get("novelty_confidence"),
                )

                # Record result
                nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=source.get("graph_fingerprint", source_result_id),
                    graph_json=graph_json_str or "{}",
                    stage0_passed=True,
                    stage05_passed=True,
                    stage1_passed=n_passed > 0,
                    loss_ratio=best_lr,
                    novelty_score=source.get("novelty_score"),
                    novelty_confidence=source.get("novelty_confidence"),
                    training_program_json=best_tp_json,
                    model_source=model_source,
                    arch_spec_json=arch_spec_json_str,
                )

                # Save checkpoint after each candidate completes
                try:
                    ckpt.save_phase(
                        experiment_id=exp_id,
                        phase="investigation",
                        candidate_idx=prog_idx + 1,
                        seed_idx=0,
                        model_state_dict={},
                        optimizer_state_dict={},
                        step=0,
                        metrics={"completed_candidate": prog_idx},
                    )
                    # Also save a progress marker at index -1 for resume
                    ckpt.save_phase(
                        experiment_id=exp_id,
                        phase="investigation",
                        candidate_idx=-1,
                        seed_idx=0,
                        model_state_dict={},
                        optimizer_state_dict={},
                        step=0,
                        metrics={"candidate_idx": prog_idx + 1},
                    )
                except Exception as e:
                    logger.debug("Investigation checkpoint save failed: %s", e)

            # Complete experiment
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)

            nb.complete_experiment(
                experiment_id=exp_id,
                results=results,
                aria_summary=summary,
                aria_mood=self.aria.state.mood,
                insights=self._analyze_results(results, exp_id, nb, context=context),
                llm_analysis=llm_analysis,
            )

            # Auto-escalate to validation
            self._auto_escalate(results, config, nb, phase="investigation")

            # Clean up investigation checkpoints on success
            if not config.keep_checkpoints:
                try:
                    ckpt.cleanup(exp_id)
                except Exception:
                    pass

            with self._lock:
                self._progress.status = "completed"
                self._progress.elapsed_seconds = time.time() - t_start
                self._progress.aria_message = summary.split("\n")[-1] if summary else "Investigation complete."

            self._emit_event("investigation_completed", {
                "experiment_id": exp_id,
                "results": results,
                "summary": summary,
            })

        except Exception as e:
            error = traceback.format_exc()
            nb.fail_experiment(exp_id, str(e))
            with self._lock:
                self._progress.status = "failed"
                self._progress.error = str(e)
                self._progress.aria_message = self.aria.react_to_failure(str(e))
            self._emit_event("experiment_failed", {
                "experiment_id": exp_id,
                "error": str(e),
            })
        finally:
            nb.close()
            self._run_pending_scale_up()

    # ── Validation Phase ──

    def start_validation(self, result_ids: List[str], config: RunConfig,
                         hypothesis: Optional[str] = None,
                         trigger: str = "manual") -> str:
        """Start validation phase for investigation survivors."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()

        # Tier guard: reject candidates already at validation tier or beyond
        tiers = nb.get_tiers_for_result_ids(result_ids)
        already_validated = {
            rid: tier for rid, tier in tiers.items()
            if tier in ("validation", "breakthrough")
        }
        if already_validated:
            nb.close()
            labels = ", ".join(f"{rid} ({tier})" for rid, tier in already_validated.items())
            raise ValueError(
                f"Cannot validate: {len(already_validated)} candidate(s) already "
                f"at or beyond validation tier: {labels}"
            )
        # Warn if known-screening candidates haven't been investigated
        # (result_ids without leaderboard entries are allowed — they may
        # come from auto-escalation paths that create entries mid-flight)
        not_investigated = {
            rid for rid in result_ids
            if tiers.get(rid) == "screening"
        }
        if not_investigated:
            nb.close()
            raise ValueError(
                f"Cannot validate: {len(not_investigated)} candidate(s) are still "
                f"at screening tier (not investigated): {', '.join(not_investigated)}"
            )

        source = "user_input" if hypothesis is not None else "runner_template"
        if hypothesis is None:
            hypothesis = (
                f"Validation: publication-grade testing of {len(result_ids)} "
                f"investigation survivors with multi-seed evaluation."
            )

        exp_id = nb.start_experiment(
            experiment_type="validation",
            config=self._validation_config_with_result_ids(config, result_ids, trigger),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source=source,
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=len(result_ids),
                aria_message=f"{self.aria.NAME}: Starting validation of {len(result_ids)} candidate(s)...",
            )

        self._emit_event("validation_started", {
            "experiment_id": exp_id,
            "hypothesis": hypothesis,
            "result_ids": result_ids,
        })

        self._thread = threading.Thread(
            target=self._run_validation_thread,
            args=(exp_id, result_ids, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def _run_validation_thread(self, exp_id: str, result_ids: List[str],
                                config: RunConfig, hypothesis: str):
        """Execute validation phase in background."""
        nb = self._make_notebook()
        t_start = time.time()
        ckpt = CheckpointManager(config.checkpoint_dir)

        # Load phase checkpoint to find where we left off
        resume_from_candidate = 0
        ckpt_state = ckpt.load_phase(exp_id, "validation", -1, 0)
        if ckpt_state:
            resume_from_candidate = ckpt_state.get("candidate_idx", 0)
            logger.info("Resuming validation from candidate %d", resume_from_candidate)

        try:
            results = {
                "total": len(result_ids), "stage0_passed": 0, "stage05_passed": 0,
                "stage1_passed": 0, "novel_count": 0,
                "best_loss_ratio": None, "best_novelty_score": None,
                "survivors": [], "validation_results": [],
            }

            dev_str = config.device if torch.cuda.is_available() else "cpu"
            dev = torch.device(dev_str)

            val_config = RunConfig.from_dict(config.to_dict())
            val_config.stage1_steps = config.validation_steps
            val_config.stage1_batch_size = config.validation_batch_size
            val_config.max_seq_len = config.validation_seq_len

            for prog_idx, source_result_id in enumerate(result_ids):
                if prog_idx < resume_from_candidate:
                    continue
                if self._stop_event.is_set():
                    break

                with self._lock:
                    self._progress.current_program = prog_idx + 1
                    self._progress.status = "validating"
                    self._progress.aria_message = (
                        f"Validating {prog_idx + 1}/{len(result_ids)}: "
                        f"{source_result_id[:8]}... "
                        f"({config.validation_n_seeds} seeds, "
                        f"{config.validation_steps} steps)"
                    )
                    self._progress.elapsed_seconds = time.time() - t_start

                self._emit_event("validation_progress", {
                    "experiment_id": exp_id,
                    "current": prog_idx + 1,
                    "total": len(result_ids),
                    "source_result_id": source_result_id,
                    "status": "starting",
                })

                # Fetch source and leaderboard entry
                source = nb.get_program_detail(source_result_id)
                if source is None:
                    continue

                graph_json_str = source.get("graph_json")
                arch_spec_json_str = source.get("arch_spec_json")
                model_source = source.get("model_source", "graph_synthesis")

                # Get best training program from investigation
                leaderboard_entries = nb.get_leaderboard()
                best_tp_json = None
                for entry in leaderboard_entries:
                    if entry.get("result_id") == source_result_id:
                        best_tp_json = entry.get("investigation_best_training")
                        break

                # Multi-seed evaluation
                seed_results = []
                for seed in range(config.validation_n_seeds):
                    if self._stop_event.is_set():
                        break

                    torch.manual_seed(seed * 42 + 7)

                    # Reconstruct model fresh
                    try:
                        if model_source == "morphological_box" and arch_spec_json_str:
                            from ..morphological_box import ArchSpec
                            from ..arch_builder import build_model, BuildConfig
                            spec_data = json.loads(arch_spec_json_str)
                            spec = ArchSpec(**spec_data)
                            build_cfg = BuildConfig(
                                dim=config.model_dim,
                                n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len,
                            )
                            model = build_model(spec, build_cfg)
                        elif graph_json_str:
                            graph = graph_from_json(graph_json_str)
                            layer_graphs = [graph] * config.n_layers
                            model = compile_model(
                                layer_graphs,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len,
                            )
                        else:
                            continue
                    except Exception as e:
                        logger.debug(f"Model reconstruction failed: {e}")
                        continue

                    self._emit_event("validation_progress", {
                        "experiment_id": exp_id,
                        "current": prog_idx + 1,
                        "total": len(result_ids),
                        "source_result_id": source_result_id,
                        "seed": seed + 1,
                        "total_seeds": config.validation_n_seeds,
                        "status": f"seed {seed + 1}/{config.validation_n_seeds}",
                    })

                    # Train (use best training program if available)
                    if best_tp_json:
                        try:
                            tp_data = json.loads(best_tp_json)
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
                                seed=self._stable_seed(exp_id, source_result_id, seed, "validation_inline_tp"),
                            )
                        except Exception:
                            s1_result = self._micro_train(
                                model,
                                val_config,
                                dev,
                                seed=self._stable_seed(exp_id, source_result_id, seed, "validation_inline_micro"),
                            )
                    else:
                        s1_result = self._micro_train(
                            model,
                            val_config,
                            dev,
                            seed=self._stable_seed(exp_id, source_result_id, seed, "validation_inline_micro"),
                        )

                    seed_results.append({
                        "seed": seed,
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
                    })

                    del model
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

                # Compute validation metrics
                passed_seeds = [r for r in seed_results if r.get("passed")]
                loss_ratios = [r["loss_ratio"] for r in seed_results
                               if r.get("loss_ratio") is not None]

                val_loss_ratio = (sum(loss_ratios) / len(loss_ratios)
                                  if loss_ratios else None)
                multi_seed_std = 0.0
                if len(loss_ratios) > 1:
                    mean_lr = sum(loss_ratios) / len(loss_ratios)
                    multi_seed_std = (
                        sum((lr - mean_lr) ** 2 for lr in loss_ratios)
                        / len(loss_ratios)
                    ) ** 0.5

                # Baseline comparison at validation scale
                val_baseline_ratio = None
                if loss_ratios:
                    best_seed = min(
                        (r for r in seed_results if r.get("final_loss") is not None),
                        key=lambda r: r["final_loss"],
                        default=None,
                    )
                    if best_seed is not None:
                        try:
                            baseline = self._get_baseline()
                            baseline_steps = int(best_seed.get("n_train_steps") or config.validation_steps)
                            baseline_recipe = self._resolve_baseline_recipe(
                                best_seed, default_lr=config.stage1_lr)
                            bl_data_fn, bl_data_tag = self._make_baseline_data_fn(config)
                            val_baseline_ratio = baseline.compare(
                                best_seed["final_loss"],
                                d_model=config.model_dim,
                                seq_len=min(128, config.validation_seq_len),
                                n_steps=max(1, baseline_steps),
                                vocab_size=config.vocab_size,
                                batch_size=config.validation_batch_size,
                                lr=baseline_recipe["lr"],
                                device=dev_str,
                                n_layers=config.n_layers,
                                optimizer_name=baseline_recipe["optimizer_name"],
                                weight_decay=baseline_recipe["weight_decay"],
                                momentum=baseline_recipe["momentum"],
                                betas=baseline_recipe["betas"],
                                data_fn=bl_data_fn,
                                data_tag=bl_data_tag,
                            )
                        except Exception:
                            pass

                if len(passed_seeds) > 0:
                    results["stage1_passed"] += 1
                results["stage0_passed"] += 1
                results["stage05_passed"] += 1

                # OOD robustness check (#54): test with reference recipes
                ood_result = None
                if len(passed_seeds) > 0:
                    _gjs_t = graph_json_str
                    _asjs_t = arch_spec_json_str
                    _ms_t = model_source
                    _cfg_t = config

                    def _make_model_t():
                        if _ms_t == "morphological_box" and _asjs_t:
                            from ..morphological_box import ArchSpec
                            from ..arch_builder import build_model, BuildConfig
                            spec = ArchSpec(**json.loads(_asjs_t))
                            bc = BuildConfig(
                                dim=_cfg_t.model_dim,
                                n_layers=_cfg_t.n_layers,
                                vocab_size=_cfg_t.vocab_size,
                                max_seq_len=_cfg_t.validation_seq_len)
                            return build_model(spec, bc)
                        else:
                            g = graph_from_json(_gjs_t)
                            return compile_model(
                                [g] * _cfg_t.n_layers,
                                vocab_size=_cfg_t.vocab_size,
                                max_seq_len=_cfg_t.validation_seq_len)

                    try:
                        ood_result = self._ood_robustness_check(
                            _make_model_t, config, dev,
                            n_steps=min(300, config.validation_steps // 3),
                            seed=self._stable_seed(
                                exp_id, source_result_id, 0, "ood"),
                        )
                    except Exception as e:
                        logger.debug("OOD robustness check failed: %s", e)

                # Hyperparameter sensitivity check (#57)
                sensitivity_result = None
                if len(passed_seeds) > 0 and val_loss_ratio is not None:
                    try:
                        sensitivity_result = self._sensitivity_check(
                            _make_model_t, config, dev,
                            base_loss_ratio=val_loss_ratio,
                            n_steps=min(300, config.validation_steps // 3),
                            seed=self._stable_seed(
                                exp_id, source_result_id, 0, "sensitivity"),
                        )
                    except Exception as e:
                        logger.debug("Sensitivity check failed: %s", e)

                # Determine if breakthrough — aligned with Aria publication thresholds
                ood_ok = (ood_result is not None
                          and ood_result.get("ood_robustness", 0) >= 0.67)
                hp_ok = (sensitivity_result is not None
                         and sensitivity_result.get("hp_robustness", 0) >= 0.75)
                nov_conf = source.get("novelty_confidence", 0) if source else 0
                is_breakthrough = (
                    val_baseline_ratio is not None
                    and val_baseline_ratio < 0.90
                    and multi_seed_std <= 0.03
                    and len(passed_seeds) >= 5
                    and len(passed_seeds) == config.validation_n_seeds
                    and (ood_result is None or ood_ok)
                    and (sensitivity_result is None or hp_ok)
                    and nov_conf >= 0.5
                )

                tier = "breakthrough" if is_breakthrough else "validation"

                validation_entry = {
                    "result_id": source_result_id,
                    "val_loss_ratio": val_loss_ratio,
                    "val_baseline_ratio": val_baseline_ratio,
                    "multi_seed_std": multi_seed_std,
                    "seeds_passed": len(passed_seeds),
                    "total_seeds": config.validation_n_seeds,
                    "is_breakthrough": is_breakthrough,
                    "novelty_confidence": nov_conf,
                    "ood_robustness": ood_result,
                    "sensitivity": sensitivity_result,
                }
                results["validation_results"].append(validation_entry)

                if val_loss_ratio and (results["best_loss_ratio"] is None
                                       or val_loss_ratio < results["best_loss_ratio"]):
                    results["best_loss_ratio"] = val_loss_ratio

                # Update leaderboard — find the actual entry for this result
                for entry in nb.get_leaderboard(limit=200):
                    if entry.get("result_id") == source_result_id:
                        nb.promote_to_tier(
                            entry_id=entry["entry_id"],
                            tier=tier,
                            validation_loss_ratio=val_loss_ratio,
                            validation_baseline_ratio=val_baseline_ratio,
                            validation_multi_seed_std=multi_seed_std,
                            validation_passed=len(passed_seeds) > 0,
                        )
                        break

                # Breakthrough detection
                if is_breakthrough:
                    ctx = build_validation_context(
                        [source], [validation_entry])
                    announcement = self.aria.announce_breakthrough(ctx)
                    nb.add_entry(ExperimentEntry(
                        entry_type="insight",
                        title="BREAKTHROUGH DETECTED",
                        content=announcement,
                        experiment_id=exp_id,
                        tags=["breakthrough"],
                    ))
                    self._emit_event("breakthrough_detected", {
                        "experiment_id": exp_id,
                        "result_id": source_result_id,
                        "val_loss_ratio": val_loss_ratio,
                        "val_baseline_ratio": val_baseline_ratio,
                        "multi_seed_std": multi_seed_std,
                        "announcement": announcement,
                    })

                # Record validation result
                nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=source.get("graph_fingerprint",
                                                 source_result_id),
                    graph_json=graph_json_str or "{}",
                    stage0_passed=True,
                    stage05_passed=True,
                    stage1_passed=len(passed_seeds) > 0,
                    loss_ratio=val_loss_ratio,
                    baseline_loss_ratio=val_baseline_ratio,
                    novelty_score=source.get("novelty_score"),
                    novelty_confidence=source.get("novelty_confidence"),
                    model_source=model_source,
                    arch_spec_json=arch_spec_json_str,
                )

                # Save checkpoint after each candidate completes
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
                    # Also save a progress marker at index -1 for resume
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
                except Exception as e:
                    logger.debug("Validation checkpoint save failed: %s", e)

            # Complete experiment
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)

            nb.complete_experiment(
                experiment_id=exp_id,
                results=results,
                aria_summary=summary,
                aria_mood=self.aria.state.mood,
                insights=self._analyze_results(results, exp_id, nb, context=context),
                llm_analysis=llm_analysis,
            )

            # Clean up validation checkpoints on success
            if not config.keep_checkpoints:
                try:
                    ckpt.cleanup(exp_id)
                except Exception:
                    pass

            with self._lock:
                self._progress.status = "completed"
                self._progress.elapsed_seconds = time.time() - t_start
                self._progress.aria_message = summary.split("\n")[-1] if summary else "Validation complete."

            self._emit_event("validation_completed", {
                "experiment_id": exp_id,
                "results": results,
                "summary": summary,
            })

        except Exception as e:
            error = traceback.format_exc()
            nb.fail_experiment(exp_id, str(e))
            with self._lock:
                self._progress.status = "failed"
                self._progress.error = str(e)
                self._progress.aria_message = self.aria.react_to_failure(str(e))
            self._emit_event("experiment_failed", {
                "experiment_id": exp_id,
                "error": str(e),
            })
        finally:
            nb.close()

    # ── Auto-Escalation Pipeline ──

    def _auto_escalate(self, results: Dict, config: RunConfig,
                       nb: LabNotebook, phase: str = "screening"):
        """Auto-escalate candidates through the research pipeline.

        Called after screening or investigation completes.
        """
        if phase == "screening" or phase == "experiment":
            # After screening: queue investigation if enough survivors
            if not config.auto_investigate:
                return
            s1_count = results.get("stage1_passed", 0)
            if s1_count < config.auto_investigate_min_survivors:
                return

            # Select top performers from the CURRENT experiment only
            # (not global top-N, which would re-promote the same candidates)
            exp_id = results.get("experiment_id")
            if exp_id:
                rows = nb.conn.execute(
                    """SELECT * FROM program_results
                       WHERE experiment_id = ? AND stage1_passed = 1
                       ORDER BY loss_ratio ASC NULLS LAST
                       LIMIT ?""",
                    (exp_id, config.auto_investigate_top_n),
                ).fetchall()
                top = [dict(r) for r in rows]
            else:
                # Fallback for callers that don't set experiment_id in results
                top = nb.get_top_programs(
                    config.auto_investigate_top_n, sort_by="loss_ratio")
            # Filter out architectures already investigated
            investigated_fps = nb.get_investigated_fingerprints()
            if investigated_fps:
                before = len(top)
                top = [p for p in top
                       if p.get("graph_fingerprint") not in investigated_fps]
                skipped = before - len(top)
                if skipped:
                    logger.info("Auto-escalate: skipped %d already-investigated archs", skipped)

            candidate_ids = [
                p["result_id"] for p in top
                if p.get("stage1_passed") and p.get("loss_ratio", 1.0) < 0.5
            ][:config.auto_investigate_top_n]

            if len(candidate_ids) < config.auto_investigate_min_survivors:
                return

            # Go/no-go decision for each candidate
            if config.auto_go_no_go and config.enable_campaigns:
                approved_ids = []
                for p in top:
                    if p["result_id"] not in candidate_ids:
                        continue
                    try:
                        # Skip if decision already exists for this result_id
                        existing_decisions = nb.get_decisions(
                            campaign_id=self._active_campaign_id)
                        already_decided = any(
                            p["result_id"] in (d.get("evidence_ids") or [])
                            for d in existing_decisions
                        )
                        if already_decided:
                            approved_ids.append(p["result_id"])
                            continue

                        go_context = build_go_no_go_context(
                            candidate=p,
                            campaign_criteria=(
                                nb.get_campaign(self._active_campaign_id or "")
                                or {}
                            ).get("success_criteria", ""),
                        )
                        decision = self.aria.generate_go_no_go(
                            subject=f"Promote {p['result_id'][:8]} to investigation",
                            evidence=f"loss_ratio={p.get('loss_ratio', '?')}, "
                                     f"novelty={p.get('novelty_score', '?')}",
                            context=go_context,
                        )
                        nb.record_decision(
                            campaign_id=self._active_campaign_id,
                            decision_type=decision["decision"],
                            subject=f"Promote {p['result_id'][:8]} to investigation",
                            rationale=decision["rationale"],
                            evidence_ids=[p["result_id"]],
                            alternatives=[{"considered": decision.get("alternatives", "")}],
                        )
                        self._emit_event("decision_recorded", {
                            "decision_type": decision["decision"],
                            "subject": p["result_id"][:8],
                            "rationale": decision["rationale"][:100],
                        })
                        if decision["decision"] in ("go", "pivot"):
                            approved_ids.append(p["result_id"])
                    except Exception as e:
                        logger.debug(f"Go/no-go failed for {p['result_id']}: {e}")
                        approved_ids.append(p["result_id"])

                candidate_ids = approved_ids if approved_ids else candidate_ids

            # Add to leaderboard as screening tier (skip if already at screening or above)
            existing_lb = {
                e["result_id"]: e["tier"]
                for e in nb.get_leaderboard(limit=500)
            }
            for p in top:
                if p["result_id"] in candidate_ids:
                    if p["result_id"] in existing_lb and existing_lb[p["result_id"]] in (
                        "screening", "investigation", "validation"
                    ):
                        continue
                    nb.upsert_leaderboard(
                        result_id=p["result_id"],
                        model_source=p.get("model_source") or "graph_synthesis",
                        architecture_desc=p.get("graph_fingerprint", "")[:40],
                        screening_loss_ratio=p.get("loss_ratio"),
                        screening_novelty=p.get("novelty_score"),
                        screening_passed=True,
                        tier="screening",
                        novelty_confidence=p.get("novelty_confidence"),
                    )

            self._pending_investigation = {
                "result_ids": candidate_ids,
                "config": config,
                "hypothesis": (
                    f"Auto-investigation: testing robustness of top "
                    f"{len(candidate_ids)} screening survivors with "
                    f"{config.n_training_programs} training programs each."
                ),
            }

            self._emit_event("auto_investigate_queued", {
                "result_ids": candidate_ids,
                "n_candidates": len(candidate_ids),
                "reason": f"{s1_count} S1 survivors with loss_ratio < 0.5",
            })

            nb.add_entry(ExperimentEntry(
                entry_type="decision",
                title="Auto-Investigation Triggered",
                content=(
                    f"Automatically queuing investigation for {len(candidate_ids)} "
                    f"top performers. Criteria: {s1_count} S1 survivors."
                ),
                metadata={"result_ids": candidate_ids},
            ))

        elif phase == "investigation":
            # After investigation: queue validation if strong candidates
            if not config.auto_validate:
                return

            inv_results = results.get("investigation_results", [])
            strong = [
                r for r in inv_results
                if r.get("robustness", 0) >= config.auto_validate_min_robustness
                and (r.get("best_loss_ratio") or 1.0) < 0.6
                and r.get("baseline_loss_ratio") is not None
                and r.get("baseline_loss_ratio") < config.auto_validate_max_baseline_ratio
                and r.get("novelty_confidence") is not None
                and r.get("novelty_confidence") >= config.auto_validate_min_novelty_confidence
                and not r.get("brittle_risk", False)
                and (
                    r.get("loss_ratio_multiplier") is None
                    or r.get("loss_ratio_multiplier") <= config.investigation_max_loss_ratio_multiplier
                )
            ]

            if not strong:
                return

            candidate_ids = [
                r["result_id"] for r in strong
            ][:config.auto_validate_top_n]

            self._pending_validation = {
                "result_ids": candidate_ids,
                "config": config,
                "hypothesis": (
                    f"Auto-validation: publication-grade testing of "
                    f"{len(candidate_ids)} robust investigation survivors."
                ),
            }

            self._emit_event("auto_validate_queued", {
                "result_ids": candidate_ids,
                "n_candidates": len(candidate_ids),
                "reason": f"{len(strong)} candidates with robustness >= "
                          f"{config.auto_validate_min_robustness}",
            })

            nb.add_entry(ExperimentEntry(
                entry_type="decision",
                title="Auto-Validation Triggered",
                content=(
                    f"Automatically queuing validation for {len(candidate_ids)} "
                    f"robust investigation survivors."
                ),
                metadata={"result_ids": candidate_ids},
            ))

    @staticmethod
    def _validation_config_with_result_ids(
        config: RunConfig,
        result_ids: List[str],
        trigger: str,
    ) -> Dict[str, Any]:
        """Attach validation candidate metadata to persisted experiment config."""
        cfg = config.to_dict()
        ids = [rid for rid in result_ids if rid]
        cfg["validation_result_ids"] = ids
        cfg["validation_candidate_count"] = len(ids)
        cfg["validation_trigger"] = trigger
        return cfg

    def _run_pending_investigation(self):
        """Launch pending auto-investigation if queued."""
        pending = getattr(self, "_pending_investigation", None)
        if pending is None:
            return
        self._pending_investigation = None

        if self.is_running:
            return

        try:
            self.start_investigation(
                result_ids=pending["result_ids"],
                config=pending["config"],
                hypothesis=pending["hypothesis"],
            )
        except Exception as e:
            logger.warning(f"Failed to launch auto-investigation: {e}")

    def _run_pending_validation(self):
        """Launch pending auto-validation if queued."""
        pending = getattr(self, "_pending_validation", None)
        if pending is None:
            return
        self._pending_validation = None

        if self.is_running:
            return

        try:
            self.start_validation(
                result_ids=pending["result_ids"],
                config=pending["config"],
                hypothesis=pending["hypothesis"],
                trigger="auto_escalate",
            )
        except Exception as e:
            logger.warning(f"Failed to launch auto-validation: {e}")

    # ── Evolution & Novelty Search ──

    def start_evolution(self, config: RunConfig,
                        hypothesis: Optional[str] = None) -> str:
        """Start evolutionary search in a background thread."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()
        hypothesis_metadata = self._build_hypothesis_metadata(
            source="user_input" if hypothesis is not None else "unknown",
            llm_used=False,
            fallback_used=False,
            used_context=False,
        )
        if hypothesis is None:
            result = self.aria.formulate_hypothesis(return_metadata=True)
            if isinstance(result, tuple):
                hypothesis, meta = result
                hypothesis_metadata.update(meta or {})
            else:
                hypothesis = result
                hypothesis_metadata["source"] = "rule_based"

        exp_id = nb.start_experiment(
            experiment_type="evolution",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=hypothesis_metadata,
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_generations=config.n_generations,
                aria_message=f"{self.aria.NAME}: Starting evolutionary search...",
            )

        self._emit_event("evolution_started", {
            "experiment_id": exp_id,
            "hypothesis": hypothesis,
            "config": config.to_dict(),
        })

        self._thread = threading.Thread(
            target=self._run_evolution_thread,
            args=(exp_id, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def start_novelty_search(self, config: RunConfig,
                             hypothesis: Optional[str] = None) -> str:
        """Start novelty search in a background thread."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()
        hypothesis_metadata = self._build_hypothesis_metadata(
            source="user_input" if hypothesis is not None else "unknown",
            llm_used=False,
            fallback_used=False,
            used_context=False,
        )
        if hypothesis is None:
            result = self.aria.formulate_hypothesis(return_metadata=True)
            if isinstance(result, tuple):
                hypothesis, meta = result
                hypothesis_metadata.update(meta or {})
            else:
                hypothesis = result
                hypothesis_metadata["source"] = "rule_based"

        exp_id = nb.start_experiment(
            experiment_type="novelty",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=hypothesis_metadata,
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_generations=config.n_generations,
                aria_message=f"{self.aria.NAME}: Starting novelty search...",
            )

        self._emit_event("novelty_started", {
            "experiment_id": exp_id,
            "hypothesis": hypothesis,
            "config": config.to_dict(),
        })

        self._thread = threading.Thread(
            target=self._run_novelty_thread,
            args=(exp_id, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def _make_fitness_fn(self, config: RunConfig):
        """Create fitness function for evolution/novelty search."""
        dev_str = config.device if torch.cuda.is_available() else "cpu"
        dev = torch.device(dev_str)

        def fitness_fn(graph):
            try:
                layer_graphs = [graph] * config.n_layers
                model = compile_model(
                    layer_graphs,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                )
                sandbox_result = safe_eval(
                    model, batch_size=2,
                    seq_len=min(128, config.max_seq_len),
                    vocab_size=config.vocab_size,
                    device=dev_str,
                )
                if not sandbox_result.passed:
                    del model
                    return 0.0

                # Micro-train for fitness
                s1_result = self._micro_train(
                    model,
                    config,
                    dev,
                    seed=self._stable_seed("fitness", graph.fingerprint()),
                )
                del model
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

                if s1_result.get("passed"):
                    lr = s1_result.get("loss_ratio", 1.0)
                    return max(0.0, 1.0 - lr)
                return 0.1  # compiled and stable but didn't learn
            except Exception:
                return 0.0

        return fitness_fn

    def _run_evolution_thread(self, exp_id: str, config: RunConfig,
                               hypothesis: str):
        """Execute evolutionary search in background."""
        nb = self._make_notebook()
        t_start = time.time()
        try:
            from ..search.evolution import EvolutionConfig, evolutionary_search

            grammar = GrammarConfig(
                model_dim=config.model_dim,
                max_depth=min(config.max_depth, 12),
                max_ops=min(config.max_ops, 20),
                residual_prob=config.residual_prob,
            )
            grammar.category_weights["math_space"] = config.math_space_weight

            evo_config = EvolutionConfig(
                population_size=config.population_size,
                n_generations=config.n_generations,
                tournament_size=config.tournament_size,
                mutation_rate=config.mutation_rate,
                crossover_rate=config.crossover_rate,
                elitism=config.elitism,
                fitness_weight=config.fitness_weight,
                novelty_weight=config.novelty_weight,
                grammar_config=grammar,
            )

            fitness_fn = self._make_fitness_fn(config)

            def gen_callback(gen, population):
                if self._stop_event.is_set():
                    return
                fitnesses = [ind.fitness for ind in population]
                avg_fit = sum(fitnesses) / len(fitnesses) if fitnesses else 0
                best_fit = max(fitnesses) if fitnesses else 0
                with self._lock:
                    self._progress.current_generation = gen + 1
                    self._progress.status = "evaluating"
                    self._progress.best_fitness = best_fit
                    self._progress.avg_fitness = avg_fit
                    self._progress.elapsed_seconds = time.time() - t_start
                    self._progress.aria_message = (
                        f"Generation {gen + 1}/{config.n_generations}: "
                        f"best={best_fit:.3f}, avg={avg_fit:.3f}"
                    )
                self._emit_event("evolution_generation", {
                    "experiment_id": exp_id,
                    "generation": gen + 1,
                    "total_generations": config.n_generations,
                    "best_fitness": best_fit,
                    "avg_fitness": avg_fit,
                    "population_size": len(population),
                })
                try:
                    nb.add_entry(ExperimentEntry(
                        entry_type="live_feed",
                        title=f"Evolution generation {gen + 1}/{config.n_generations}",
                        content=(
                            f"Gen {gen + 1}/{config.n_generations}: "
                            f"best={best_fit:.3f}, avg={avg_fit:.3f}, "
                            f"pop={len(population)}"
                        ),
                        experiment_id=exp_id,
                        metadata={
                            "live_feed_type": "evo_gen",
                            "payload": {
                                "experiment_id": exp_id,
                                "generation": gen + 1,
                                "total_generations": config.n_generations,
                                "best_fitness": best_fit,
                                "avg_fitness": avg_fit,
                                "population_size": len(population),
                            },
                        },
                    ))
                except Exception as e:
                    logger.debug("Failed to persist evolution generation feed entry: %s", e)

            def novelty_fn(graph, all_graphs):
                """Structural novelty relative to current population."""
                nov = novelty_score(graph)
                # Penalize duplicates within population
                my_fp = graph.fingerprint()
                dup_count = sum(1 for g in all_graphs
                                if g.fingerprint() == my_fp) - 1
                penalty = max(0, 1 - dup_count * 0.3)
                return nov.structural_novelty * penalty

            population = evolutionary_search(
                fitness_fn=fitness_fn,
                novelty_fn=novelty_fn,
                config=evo_config,
                callback=gen_callback,
            )

            # Record top individuals
            results = {
                "total": len(population),
                "stage0_passed": sum(1 for ind in population if ind.fitness > 0),
                "stage05_passed": sum(1 for ind in population if ind.fitness > 0),
                "stage1_passed": sum(1 for ind in population if ind.fitness > 0.2),
                "novel_count": sum(1 for ind in population if ind.novelty > 0.5),
                "best_loss_ratio": 1.0 - max((ind.fitness for ind in population), default=0),
                "best_novelty_score": max((ind.novelty for ind in population), default=0),
                "survivors": [],
            }

            for ind in population[:20]:
                graph_metrics = self._extract_graph_metrics(ind.graph)
                nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=ind.fingerprint,
                    graph_json=graph_to_json(ind.graph),
                    stage1_passed=ind.fitness > 0.2,
                    stage0_passed=ind.fitness > 0,
                    stage05_passed=ind.fitness > 0,
                    loss_ratio=1.0 - ind.fitness if ind.fitness > 0 else None,
                    novelty_score=ind.novelty,
                    novelty_confidence=0.2,
                    stage_at_death="survived" if ind.fitness > 0.2 else "stage1",
                    **graph_metrics,
                )
                if ind.fitness > 0.2:
                    results["survivors"].append({
                        "fingerprint": ind.fingerprint,
                        "novelty": ind.novelty,
                        "loss_ratio": 1.0 - ind.fitness,
                    })

            nb.update_op_success_rates(exp_id)

            # Rich context for Aria
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)

            # Validate hypothesis
            try:
                validation = self.aria.validate_hypothesis(hypothesis, results, context)
                if validation:
                    nb.add_entry(ExperimentEntry(
                        entry_type="analysis",
                        title="Hypothesis Validation",
                        content=validation.get("explanation", ""),
                        experiment_id=exp_id,
                        metadata={"validated": validation.get("validated", False)},
                    ))
            except Exception as e:
                logger.warning("Hypothesis validation failed for %s: %s", exp_id, e)

            nb.complete_experiment(
                experiment_id=exp_id,
                results=results,
                aria_summary=summary,
                aria_mood=self.aria.state.mood,
                insights=self._analyze_results(results, exp_id, nb, context=context),
                llm_analysis=llm_analysis,
            )

            # Auto-recommend next experiment
            self._auto_recommend(results, config, hypothesis, nb)

            # Auto-scale-up and auto-report
            self._maybe_auto_scale_up(results, config, nb)
            self._maybe_auto_report(config, nb, reason="evolution_complete")

            with self._lock:
                self._progress.status = "completed"
                self._progress.elapsed_seconds = time.time() - t_start
                self._progress.aria_message = summary.split("\n")[-1] if summary else "Evolution complete."

            self._emit_event("evolution_completed", {
                "experiment_id": exp_id,
                "results": results,
                "summary": summary,
            })

        except Exception as e:
            error = traceback.format_exc()
            nb.fail_experiment(exp_id, str(e))
            with self._lock:
                self._progress.status = "failed"
                self._progress.error = str(e)
                self._progress.aria_message = self.aria.react_to_failure(str(e))
            self._emit_event("experiment_failed", {
                "experiment_id": exp_id,
                "error": str(e),
            })
        finally:
            nb.close()
            self._run_pending_scale_up()

    def _run_novelty_thread(self, exp_id: str, config: RunConfig,
                             hypothesis: str):
        """Execute novelty search in background."""
        nb = self._make_notebook()
        t_start = time.time()
        try:
            from ..search.novelty_search import NoveltySearchConfig, novelty_search

            grammar = GrammarConfig(
                model_dim=config.model_dim,
                max_depth=min(config.max_depth, 12),
                max_ops=min(config.max_ops, 20),
                residual_prob=config.residual_prob,
            )
            grammar.category_weights["math_space"] = config.math_space_weight

            ns_config = NoveltySearchConfig(
                archive_size=config.archive_size,
                k_nearest=config.k_nearest,
                archive_threshold=config.archive_threshold,
                novelty_weight=config.novelty_weight,
                fitness_weight=config.fitness_weight,
                population_size=config.population_size,
                n_generations=config.n_generations,
                grammar_config=grammar,
            )

            fitness_fn = self._make_fitness_fn(config)
            dev_str = config.device if torch.cuda.is_available() else "cpu"

            def fingerprint_fn(graph):
                try:
                    layer_graphs = [graph] * config.n_layers
                    model = compile_model(
                        layer_graphs,
                        vocab_size=config.vocab_size,
                        max_seq_len=config.max_seq_len,
                    )
                    fp = compute_fingerprint(
                        model,
                        seq_len=min(64, config.max_seq_len),
                        model_dim=config.model_dim,
                        vocab_size=config.vocab_size,
                        device=dev_str,
                    )
                    del model
                    return fp
                except Exception as e:
                    logger.debug("Fingerprint computation failed: %s", e)
                    return None

            def gen_callback(gen, population, archive):
                if self._stop_event.is_set():
                    return
                fitnesses = [ind.fitness for ind in population]
                novelties = [ind.novelty for ind in population]
                avg_fit = sum(fitnesses) / len(fitnesses) if fitnesses else 0
                best_fit = max(fitnesses) if fitnesses else 0
                with self._lock:
                    self._progress.current_generation = gen + 1
                    self._progress.status = "evaluating"
                    self._progress.best_fitness = best_fit
                    self._progress.avg_fitness = avg_fit
                    self._progress.archive_size = archive.size()
                    self._progress.elapsed_seconds = time.time() - t_start
                    self._progress.aria_message = (
                        f"Generation {gen + 1}/{config.n_generations}: "
                        f"archive={archive.size()}, best_fit={best_fit:.3f}"
                    )
                self._emit_event("novelty_generation", {
                    "experiment_id": exp_id,
                    "generation": gen + 1,
                    "total_generations": config.n_generations,
                    "best_fitness": best_fit,
                    "avg_fitness": avg_fit,
                    "archive_size": archive.size(),
                    "best_novelty": max(novelties) if novelties else 0,
                })
                try:
                    best_novelty = max(novelties) if novelties else 0
                    nb.add_entry(ExperimentEntry(
                        entry_type="live_feed",
                        title=f"Novelty generation {gen + 1}/{config.n_generations}",
                        content=(
                            f"Gen {gen + 1}/{config.n_generations}: "
                            f"best_fit={best_fit:.3f}, archive={archive.size()}, "
                            f"novelty={best_novelty:.3f}"
                        ),
                        experiment_id=exp_id,
                        metadata={
                            "live_feed_type": "nov_gen",
                            "payload": {
                                "experiment_id": exp_id,
                                "generation": gen + 1,
                                "total_generations": config.n_generations,
                                "best_fitness": best_fit,
                                "avg_fitness": avg_fit,
                                "archive_size": archive.size(),
                                "best_novelty": best_novelty,
                            },
                        },
                    ))
                except Exception as e:
                    logger.debug("Failed to persist novelty generation feed entry: %s", e)

            ns_result = novelty_search(
                fitness_fn=fitness_fn,
                fingerprint_fn=fingerprint_fn,
                config=ns_config,
                callback=gen_callback,
            )

            # Record results
            results = {
                "total": ns_result.total_evaluated,
                "stage0_passed": sum(1 for ind in ns_result.best_individuals if ind.fitness > 0),
                "stage05_passed": sum(1 for ind in ns_result.best_individuals if ind.fitness > 0),
                "stage1_passed": sum(1 for ind in ns_result.best_individuals if ind.fitness > 0.2),
                "novel_count": sum(1 for ind in ns_result.best_individuals if ind.novelty > 0.5),
                "best_loss_ratio": None,
                "best_novelty_score": None,
                "survivors": [],
                "archive_size": ns_result.archive_size,
            }

            for ind in ns_result.best_individuals[:20]:
                graph_metrics = self._extract_graph_metrics(ind.graph)
                nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=ind.fingerprint,
                    graph_json=graph_to_json(ind.graph),
                    stage1_passed=ind.fitness > 0.2,
                    stage0_passed=ind.fitness > 0,
                    stage05_passed=ind.fitness > 0,
                    loss_ratio=1.0 - ind.fitness if ind.fitness > 0 else None,
                    novelty_score=ind.novelty,
                    novelty_confidence=0.2,
                    stage_at_death="survived" if ind.fitness > 0.2 else "stage1",
                    **graph_metrics,
                )
                if ind.fitness > 0.2:
                    results["survivors"].append({
                        "fingerprint": ind.fingerprint,
                        "novelty": ind.novelty,
                        "loss_ratio": 1.0 - ind.fitness,
                    })

            if results["survivors"]:
                results["best_loss_ratio"] = min(s["loss_ratio"] for s in results["survivors"])
                results["best_novelty_score"] = max(s["novelty"] for s in results["survivors"])

            nb.update_op_success_rates(exp_id)

            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)

            try:
                validation = self.aria.validate_hypothesis(hypothesis, results, context)
                if validation:
                    nb.add_entry(ExperimentEntry(
                        entry_type="analysis",
                        title="Hypothesis Validation",
                        content=validation.get("explanation", ""),
                        experiment_id=exp_id,
                        metadata={"validated": validation.get("validated", False)},
                    ))
            except Exception:
                pass

            nb.complete_experiment(
                experiment_id=exp_id,
                results=results,
                aria_summary=summary,
                aria_mood=self.aria.state.mood,
                insights=self._analyze_results(results, exp_id, nb, context=context),
                llm_analysis=llm_analysis,
            )

            # Auto-recommend next experiment
            self._auto_recommend(results, config, hypothesis, nb)

            # Auto-scale-up and auto-report
            self._maybe_auto_scale_up(results, config, nb)
            self._maybe_auto_report(config, nb, reason="novelty_search_complete")

            with self._lock:
                self._progress.status = "completed"
                self._progress.elapsed_seconds = time.time() - t_start
                self._progress.aria_message = summary.split("\n")[-1] if summary else "Novelty search complete."

            self._emit_event("novelty_completed", {
                "experiment_id": exp_id,
                "results": results,
                "summary": summary,
                "archive_size": ns_result.archive_size,
            })

        except Exception as e:
            error = traceback.format_exc()
            nb.fail_experiment(exp_id, str(e))
            with self._lock:
                self._progress.status = "failed"
                self._progress.error = str(e)
                self._progress.aria_message = self.aria.react_to_failure(str(e))
            self._emit_event("experiment_failed", {
                "experiment_id": exp_id,
                "error": str(e),
            })
        finally:
            nb.close()
            self._run_pending_scale_up()

    # ── Scale-Up Mode ──

    def start_scale_up(self, result_ids: List[str], config: RunConfig,
                       hypothesis: Optional[str] = None) -> str:
        """Start scale-up validation of specific programs in a background thread."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()
        source = "user_input" if hypothesis is not None else "runner_template"
        if hypothesis is None:
            hypothesis = (
                f"Scale-up validation: testing whether {len(result_ids)} "
                f"top performer(s) maintain their advantage at 10x training scale."
            )

        exp_id = nb.start_experiment(
            experiment_type="scale_up",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source=source,
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=len(result_ids),
                aria_message=f"{self.aria.NAME}: Starting scale-up validation of {len(result_ids)} program(s)...",
            )

        self._emit_event("scale_up_started", {
            "experiment_id": exp_id,
            "hypothesis": hypothesis,
            "result_ids": result_ids,
            "config": {
                "steps": config.scale_up_steps,
                "batch_size": config.scale_up_batch_size,
                "seq_len": config.scale_up_seq_len,
            },
        })

        self._thread = threading.Thread(
            target=self._run_scale_up_thread,
            args=(exp_id, result_ids, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def _run_scale_up_thread(self, exp_id: str, result_ids: List[str],
                              config: RunConfig, hypothesis: str):
        """Execute scale-up training in background."""
        nb = self._make_notebook()
        t_start = time.time()
        try:
            # graph_from_json already imported at module level
            results = {
                "total": len(result_ids), "stage0_passed": 0, "stage05_passed": 0,
                "stage1_passed": 0, "novel_count": 0,
                "best_loss_ratio": None, "best_novelty_score": None,
                "survivors": [],
            }

            dev_str = config.device if torch.cuda.is_available() else "cpu"
            dev = torch.device(dev_str)

            # Create a modified config for scale-up training
            scale_config = RunConfig.from_dict(config.to_dict())
            scale_config.stage1_steps = config.scale_up_steps
            scale_config.stage1_batch_size = config.scale_up_batch_size
            scale_config.max_seq_len = config.scale_up_seq_len

            for prog_idx, source_result_id in enumerate(result_ids):
                if self._stop_event.is_set():
                    break

                with self._lock:
                    self._progress.current_program = prog_idx + 1
                    self._progress.status = "training"
                    self._progress.aria_message = (
                        f"Scale-up {prog_idx + 1}/{len(result_ids)}: "
                        f"training {source_result_id[:8]}... "
                        f"({config.scale_up_steps} steps, batch={config.scale_up_batch_size})"
                    )
                    self._progress.elapsed_seconds = time.time() - t_start

                self._emit_event("scale_up_progress", {
                    "experiment_id": exp_id,
                    "current_program": prog_idx + 1,
                    "total_programs": len(result_ids),
                    "source_result_id": source_result_id,
                    "status": "starting",
                })

                # Fetch source program
                source_program = nb.get_program_detail(source_result_id)
                if source_program is None:
                    self._emit_event("scale_up_progress", {
                        "experiment_id": exp_id,
                        "current_program": prog_idx + 1,
                        "total_programs": len(result_ids),
                        "source_result_id": source_result_id,
                        "status": "skipped",
                        "error": "Source program not found",
                    })
                    continue

                # Reconstruct graph from stored JSON
                graph_json_str = source_program.get("graph_json")
                if not graph_json_str:
                    continue

                try:
                    graph = graph_from_json(graph_json_str)
                except Exception as e:
                    self._emit_event("scale_up_progress", {
                        "experiment_id": exp_id,
                        "current_program": prog_idx + 1,
                        "total_programs": len(result_ids),
                        "source_result_id": source_result_id,
                        "status": "error",
                        "error": f"Graph deserialization failed: {e}",
                    })
                    continue

                # Compile model
                try:
                    layer_graphs = [graph] * config.n_layers
                    model = compile_model(
                        layer_graphs,
                        vocab_size=config.vocab_size,
                        max_seq_len=config.scale_up_seq_len,
                    )
                except Exception as e:
                    self._emit_event("scale_up_progress", {
                        "experiment_id": exp_id,
                        "current_program": prog_idx + 1,
                        "total_programs": len(result_ids),
                        "source_result_id": source_result_id,
                        "status": "error",
                        "error": f"Compilation failed: {e}",
                    })
                    continue

                results["stage0_passed"] += 1
                results["stage05_passed"] += 1

                # Run scale-up training
                s1_result = self._micro_train(
                    model,
                    scale_config,
                    dev,
                    seed=self._stable_seed(exp_id, source_result_id, "scale_up"),
                )

                program_metrics = self._extract_graph_metrics(graph)
                # Store scale-up provenance in model_source (a valid column)
                # rather than as separate columns that don't exist in schema
                program_metrics["model_source"] = "graph_synthesis"

                s1_passed = s1_result.get("passed", False)
                loss_ratio = s1_result.get("loss_ratio")
                final_loss = s1_result.get("final_loss")
                throughput = s1_result.get("throughput")
                training_curve = s1_result.get("training_curve")

                # Training metrics
                for key in ["initial_loss", "min_loss", "loss_improvement_rate",
                            "avg_step_time_ms", "total_train_time_ms",
                            "max_grad_norm", "mean_grad_norm", "grad_norm_std",
                            "n_train_steps", "final_lr"]:
                    program_metrics[key] = s1_result.get(key)

                if s1_passed:
                    results["stage1_passed"] += 1
                    # Baseline comparison at scale
                    if final_loss is not None:
                        try:
                            baseline = self._get_baseline()
                            baseline_steps = int(s1_result.get("n_train_steps") or config.scale_up_steps)
                            baseline_recipe = self._resolve_baseline_recipe(
                                s1_result, default_lr=config.stage1_lr)
                            bl_data_fn, bl_data_tag = self._make_baseline_data_fn(config)
                            baseline_ratio = baseline.compare(
                                final_loss,
                                d_model=config.model_dim,
                                seq_len=min(128, config.scale_up_seq_len),
                                n_steps=max(1, baseline_steps),
                                vocab_size=config.vocab_size,
                                batch_size=config.scale_up_batch_size,
                                lr=baseline_recipe["lr"],
                                device=dev_str,
                                n_layers=config.n_layers,
                                optimizer_name=baseline_recipe["optimizer_name"],
                                weight_decay=baseline_recipe["weight_decay"],
                                momentum=baseline_recipe["momentum"],
                                betas=baseline_recipe["betas"],
                                data_fn=bl_data_fn,
                                data_tag=bl_data_tag,
                            )
                            program_metrics["baseline_loss_ratio"] = baseline_ratio
                        except Exception:
                            pass

                program_metrics["stage_at_death"] = "survived" if s1_passed else "stage1"

                # Diagnostic tasks for S1 survivors
                if s1_passed and model is not None:
                    try:
                        diag = run_diagnostic_suite(model, device=dev_str)
                        program_metrics["diagnostic_tasks_json"] = json.dumps(diag.to_dict())
                        program_metrics["diagnostic_score"] = diag.diagnostic_score
                    except Exception:
                        pass

                # Novelty — compute behavioral fingerprint for S1 survivors
                fp = None
                if s1_passed and model is not None:
                    try:
                        fp = compute_fingerprint(
                            model,
                            seq_len=min(64, config.scale_up_seq_len),
                            model_dim=config.model_dim,
                            vocab_size=config.vocab_size,
                            device=dev_str,
                        )
                    except Exception:
                        pass

                nov = novelty_score(graph, fingerprint=fp)
                n_score = nov.overall_novelty
                if s1_passed and n_score > 0.5:
                    results["novel_count"] += 1
                    results["survivors"].append({
                        "fingerprint": graph.fingerprint(),
                        "novelty": n_score,
                        "loss_ratio": loss_ratio,
                    })

                if loss_ratio and (results["best_loss_ratio"] is None
                                   or loss_ratio < results["best_loss_ratio"]):
                    results["best_loss_ratio"] = loss_ratio
                if n_score and (results["best_novelty_score"] is None
                                or n_score > results["best_novelty_score"]):
                    results["best_novelty_score"] = n_score

                result_id = nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=graph.fingerprint(),
                    graph_json=graph_to_json(graph),
                    stage0_passed=True, stage05_passed=True,
                    stage1_passed=s1_passed,
                    loss_ratio=loss_ratio, final_loss=final_loss,
                    throughput=throughput, novelty_score=n_score,
                    structural_novelty=nov.structural_novelty,
                    behavioral_novelty=nov.behavioral_novelty,
                    most_similar_to=nov.most_similar_to,
                    novelty_confidence=nov.novelty_confidence,
                    **program_metrics,
                )

                if training_curve and result_id:
                    try:
                        nb.store_training_curve(result_id, training_curve)
                    except Exception:
                        pass

                self._emit_event("scale_up_progress", {
                    "experiment_id": exp_id,
                    "current_program": prog_idx + 1,
                    "total_programs": len(result_ids),
                    "source_result_id": source_result_id,
                    "status": "completed",
                    "passed": s1_passed,
                    "loss_ratio": round(loss_ratio, 4) if loss_ratio else None,
                    "final_loss": round(final_loss, 4) if final_loss else None,
                })

                # Cleanup
                del model
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            # Guard: if no programs were processed at all, fail with clear reason
            if results["stage0_passed"] == 0 and results["total"] > 0:
                reason = (f"All {results['total']} source programs were skipped "
                          f"(not found or failed to compile). "
                          f"Result IDs: {', '.join(r[:12] for r in result_ids)}")
                logger.warning("Scale-up produced no results: %s", reason)
                nb.fail_experiment(exp_id, reason)
                with self._lock:
                    self._progress.status = "failed"
                    self._progress.error = reason
                    self._progress.aria_message = self.aria.react_to_failure(reason)
                self._emit_event("experiment_failed", {
                    "experiment_id": exp_id, "error": reason,
                })
                return

            # Complete experiment
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)
            summary = self.aria.experiment_summary(results, context=context)
            llm_analysis = self.aria.analyze_results(results, context=context)
            insights = self._analyze_results(results, exp_id, nb, context=context)

            nb.complete_experiment(
                experiment_id=exp_id,
                results=results,
                aria_summary=summary,
                aria_mood=self.aria.state.mood,
                insights=insights,
                llm_analysis=llm_analysis,
            )

            self._auto_recommend(results, config, hypothesis, nb)

            with self._lock:
                self._progress.status = "completed"
                self._progress.elapsed_seconds = time.time() - t_start
                self._progress.aria_message = summary.split("\n")[-1] if summary else "Scale-up complete."

            self._emit_event("scale_up_completed", {
                "experiment_id": exp_id,
                "results": results,
                "summary": summary,
            })

        except Exception as e:
            nb.fail_experiment(exp_id, str(e))
            with self._lock:
                self._progress.status = "failed"
                self._progress.error = str(e)
                self._progress.aria_message = self.aria.react_to_failure(str(e))
            self._emit_event("experiment_failed", {
                "experiment_id": exp_id,
                "error": str(e),
            })
        finally:
            nb.close()

    def get_dashboard_data(self) -> Dict:
        """Get all data needed for the React dashboard."""
        nb = self._make_notebook()
        try:
            return {
                "aria": self.aria.get_status(),
                "summary": nb.get_dashboard_summary(),
                "recent_experiments": nb.get_recent_experiments(20),
                "top_programs": nb.get_top_programs(20),
                "insights": nb.get_insights(limit=20),
                "recent_entries": nb.get_entries(limit=30),
                "is_running": self.is_running,
                "progress": self.progress.to_dict(),
            }
        finally:
            nb.close()
