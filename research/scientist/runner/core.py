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


class _CoreMixin:
    """Core class definition, __init__, properties, class variables."""

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

    _MAINTENANCE_OPS = {
        "purge_empty_experiments",
        "purge_junk_programs",
        "reset_op_stats",
        "clear_toxic_signatures",
        "vacuum",
        "backfill_failure_signatures",
    }

    _PLATEAU_WINDOW = 5  # cycles to check for progress
    _PLATEAU_MIN_CYCLES = 8  # don't trigger before this many cycles

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

    _SENSITIVITY_PERTURBATIONS = [
        ("lr_half", {"lr_mult": 0.5}),
        ("lr_double", {"lr_mult": 2.0}),
        ("steps_half", {"steps_mult": 0.5}),
        ("steps_double", {"steps_mult": 2.0}),
    ]

    def __init__(self, notebook_path: str = "research/lab_notebook.db"):
        self.notebook_path = notebook_path
        self.aria = get_aria()
        self._math_spaces_registered = False
        self._baseline: Optional[TransformerBaseline] = None

        # Background execution state
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        from ._types import LiveProgress
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
        self._hf_batcher: Optional[CorpusTokenBatcher] = None
        self._hf_signature: Optional[str] = None
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
        self._live_training_context: Optional[Dict[str, str]] = None  # {exp_id, phase}
        self._live_loss_curve: List[Dict] = []  # rolling buffer for dashboard chart
        self._grammar_weight_overrides: Dict[str, float] = {}
        try:
            row = self.notebook.conn.execute(
                "SELECT evidence FROM learning_log "
                "WHERE event_type='chat_grammar_overrides_applied' "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if row and row[0]:
                import json as _json
                meta = _json.loads(row[0])
                overrides = meta.get("overrides") if isinstance(meta, dict) else None
                if isinstance(overrides, dict) and overrides:
                    self._grammar_weight_overrides = overrides
                    logger.info("Restored grammar weight overrides from DB: %s", overrides)
        except Exception:
            pass  # Non-critical: start with empty overrides
        self._last_stagnation_agent_cycle = -10
        self._last_anti_stagnation_cycle = -10
        self._last_chat_config_overrides: Dict[str, Any] = {}
        self._excluded_ops_overrides: Set[str] = set()
        self._op_weights_overrides: Dict[str, float] = {}
        self._structured_sparsity_bias_override: float = 0.0
        try:
            self._healer = CodeHealer(self.notebook_path)
        except Exception:
            self._healer = None
        self._last_healer_integrity_check = 0.0
        self._recent_healer_signatures: Dict[str, float] = {}
        self._pending_heal_retry: Optional[Dict] = None

        self._recover_stale_experiments_on_startup()

    def _make_notebook(self) -> LabNotebook:
        """Create a new notebook connection (thread-safe)."""
        return LabNotebook(self.notebook_path)

    def _ensure_math_spaces(self):
        if not self._math_spaces_registered:
            try:
                from ...mathspaces.registry import register_all_mathspaces
                register_all_mathspaces()
                self._math_spaces_registered = True
            except Exception as e:
                logger.debug("Math spaces registration failed: %s", e)

    def _get_baseline(self) -> TransformerBaseline:
        if self._baseline is None:
            self._baseline = TransformerBaseline()
        return self._baseline

    def _get_corpus_batcher(self, config: RunConfig) -> Optional[CorpusTokenBatcher]:
        """Lazily create or reuse corpus batcher for corpus-mode training."""
        signature = (
            str(config.corpus_path or ""),
            str(config.corpus_format or "auto"),
            str(config.corpus_text_key or "text"),
            str(config.tokenizer_mode or "byte"),
            int(config.corpus_max_chars),
            int(config.vocab_size),
            float(getattr(config, "corpus_train_fraction", 0.9) or 0.9),
            float(getattr(config, "corpus_val_fraction", 0.1) or 0.1),
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
                train_fraction=float(getattr(config, "corpus_train_fraction", 0.9) or 0.9),
                val_fraction=float(getattr(config, "corpus_val_fraction", 0.1) or 0.1),
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

    def _get_hf_batcher(self, config: RunConfig) -> Optional[CorpusTokenBatcher]:
        """Lazily create or reuse a corpus batcher backed by a HuggingFace dataset."""
        ds_name = str(config.hf_dataset or "").strip()
        if not ds_name:
            return None

        subset = str(config.hf_subset or "").strip() or None
        split = str(config.hf_split or "train").strip()
        text_key = str(config.hf_text_key or "text").strip()
        signature = f"{ds_name}|{subset}|{split}|{text_key}|{config.vocab_size}"

        if self._hf_batcher is not None and self._hf_signature == signature:
            return self._hf_batcher

        try:
            from datasets import load_dataset
        except ImportError:
            logger.warning("datasets library not installed; pip install datasets")
            return None

        try:
            ds = load_dataset(ds_name, subset, split=split, trust_remote_code=True)
            texts = []
            char_budget = int(config.corpus_max_chars)
            total = 0
            for row in ds:
                t = row.get(text_key, "")
                if not t:
                    continue
                texts.append(t)
                total += len(t)
                if total >= char_budget:
                    break
            if not texts:
                logger.warning("HuggingFace dataset %s had no text in column '%s'", ds_name, text_key)
                return None

            # Write concatenated text to a temp file and wrap with CorpusBatcher
            import tempfile
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix="hf_", delete=False,
            )
            tmp.write("\n".join(texts))
            tmp.flush()
            tmp.close()

            batcher = CorpusTokenBatcher(
                CorpusConfig(
                    path=tmp.name,
                    fmt="txt",
                    text_key=text_key,
                    tokenizer=str(config.tokenizer_mode or "byte"),
                    max_chars=char_budget,
                    train_fraction=0.9,
                    val_fraction=0.1,
                ),
                vocab_size=int(config.vocab_size),
            )
            self._hf_batcher = batcher
            self._hf_signature = signature
            logger.info("HuggingFace batcher ready: %s (%s), %d chars loaded",
                        ds_name, split, total)
            return batcher
        except Exception as e:
            logger.warning("Failed to load HuggingFace dataset %s: %s", ds_name, e)
            return None

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

    @staticmethod
    def _stable_seed(*parts: Any) -> int:
        """Create a reproducible 31-bit seed from contextual parts."""
        key = "|".join(str(p) for p in parts)
        return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def progress(self) -> LiveProgress:
        with self._lock:
            return LiveProgress(**self._progress.to_dict())

    def _uses_local_llm_backend(self) -> bool:
        """Whether Aria is currently configured to use a local LLM backend."""
        try:
            llm_config = self.aria.get_llm_config()
        except Exception as e:
            logger.debug("Failed to inspect LLM backend for limit policy: %s", e)
            return False
        backend = str((llm_config or {}).get("backend") or "").strip().lower()
        return backend in {"ollama", "local", "lmstudio", "llama.cpp", "llamacpp", "vllm"}

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

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _corpus_version_tag(self, path: str) -> str:
        try:
            stat = os.stat(path)
            name = os.path.basename(path)
            return f"{name}:{stat.st_size}:{stat.st_mtime_ns}"
        except Exception:
            return "missing"

    @staticmethod
    def _norm_map(values: Dict[str, float], higher_is_better: bool = True) -> Dict[str, float]:
        if not values:
            return {}
        vmin = min(values.values())
        vmax = max(values.values())
        if math.isclose(vmin, vmax):
            return {k: 0.5 for k in values}
        out: Dict[str, float] = {}
        for k, v in values.items():
            score = (v - vmin) / (vmax - vmin)
            out[k] = score if higher_is_better else (1.0 - score)
        return out

