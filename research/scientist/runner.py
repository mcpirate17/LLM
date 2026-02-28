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

from ..synthesis.grammar import GrammarConfig, generate_layer_graph, batch_generate
# compile_model routed through native_runner.compile_model_native_first
from .native_runner import (
    compile_model_native_first as compile_model,
    native_runner_capability_report,
    record_native_abi_parity_result,
    reset_native_runner_telemetry,
)
from ..synthesis.validator import validate_graph
from ..synthesis.serializer import graph_to_json, graph_from_json, graph_summary
from ..synthesis.primitives import get_primitive, list_primitives
from ..eval.sandbox import safe_eval
from ..eval.metrics import novelty_score
from ..eval.flops import estimate_flops
from ..eval.baseline import TransformerBaseline
from ..eval.fingerprint import compute_fingerprint, BehavioralFingerprint
from ..eval.diagnostic_tasks import run_diagnostic_suite, DiagnosticSuiteResult
from ..eval.perf_budget import evaluate_perf_budget_gate
from ..eval.pruning import apply_one_shot_pruning, estimate_lm_ce_loss
from ..training.loss_synthesis import synthesize_loss
from ..training.optimizer_synthesis import synthesize_optimizer
from ..training.training_program import synthesize_training_program, synthesize_training_program_batch
from ..training.data_pipeline import CorpusConfig, CorpusTokenBatcher
from ..training.checkpointing import CheckpointManager
from ..orchestrator.executor import WorkerPoolOrchestrator
from .persona import Aria, get_aria
from .notebook import LabNotebook, ExperimentEntry
from .evidence import (
    build_evidence_pack,
    validate_selection_decision_log,
)
from .preregistration import (
    HypothesisPreregistration,
    PreregistrationError,
    validate_preregistration,
)
from ..healer import CodeHealer
from ..healer.core import HealerTaskSpec
from .llm.context import (build_experiment_context,
                          build_rich_context, build_investigation_context,
                          build_validation_context, build_mode_selection_context,
                          build_hypothesis_context, build_go_no_go_context,
                          build_knowledge_extraction_context,
                          build_campaign_report_context,
                          build_campaign_formulation_context,
                          build_manual_start_fallback_context)
from .llm.decision import NextExperimentDecisionPlanner

import logging
logger = logging.getLogger(__name__)

import aria_core
from ..synthesis.primitives import OPCODE_MAP

def _native_proactive_gating(graph) -> Dict[str, Any]:
    """
    Perform high-performance DAG validation and proactive gating using aria-core.
    Identifies stability risks and toxic motifs before compilation.
    """
    try:
        # 1. Map node IDs to 0..N-1 for C++ interop
        nodes = list(graph.nodes.values())
        id_map = {node.id: i for i, node in enumerate(nodes)}
        n_nodes = len(nodes)
        
        # 2. Extract edges
        edges = []
        for node in nodes:
            for iid in node.input_ids:
                if iid in id_map:
                    edges.append([id_map[iid], id_map[node.id]])
        
        # 3. Extract op_codes
        op_codes = []
        for node in nodes:
            op_codes.append(OPCODE_MAP.get(node.op_name, -1))
            
        # 4. Call native engine
        return aria_core.proactive_gating(n_nodes, edges, op_codes)
    except Exception as e:
        logger.debug(f"Native proactive gating failed: {e}")
        return {"passed": True, "reason": "native_gating_error", "error": str(e)}

def _native_runner_progress_report() -> Dict[str, Any]:
    try:
        return native_runner_capability_report()
    except Exception as exc:
        return {
            "enabled": False,
            "strict": False,
            "designer_runtime_available": False,
            "status": f"native_runner_report_error:{exc}",
            "supported_ops": [],
            "unsupported_ops": [],
            "approximate_mappings": {},
            "semantic_warnings": [],
            "semantic_warning_count": 0,
            "mapping_source": "",
        }


def _rebuild_graph_with_overrides(candidate_graph, overrides: Dict[int, Dict[str, Any]]):
    """Rebuild a graph with targeted node op/config overrides."""
    rebuilt = type(candidate_graph)(candidate_graph.model_dim)
    id_map: Dict[int, int] = {}
    topo = candidate_graph.topological_order()
    for old_id in topo:
        node = candidate_graph.nodes[old_id]
        if node.is_input:
            id_map[old_id] = rebuilt.add_input()
            continue
        override = overrides.get(old_id, {})
        op_name = override.get("op_name", node.op_name)
        config = override.get("config", node.config)
        new_inputs = [id_map[i] for i in node.input_ids]
        try:
            new_id = rebuilt.add_op(op_name, new_inputs, config=config)
        except Exception:
            return None
        id_map[old_id] = new_id

    if candidate_graph.output_node is None:
        return None
    out_old = candidate_graph.output_node.id
    out_new = id_map.get(out_old)
    if out_new is None:
        return None
    try:
        rebuilt.set_output(out_new)
    except Exception:
        return None
    rebuilt.metadata = dict(getattr(candidate_graph, "metadata", {}) or {})
    return rebuilt


def propose_ablation_suite(candidate_graph, hypothesis) -> List[Any]:
    """Generate counterfactual ablations by replacing suspected components."""
    if candidate_graph is None:
        return []
    hyp = str(hypothesis or "").lower()
    ops = list_primitives()
    replacement_by_signature: Dict[Tuple[int, str], List[str]] = {}
    for op in ops:
        key = (op.n_inputs, op.shape_rule)
        replacement_by_signature.setdefault(key, []).append(op.name)
    for key in replacement_by_signature:
        replacement_by_signature[key] = sorted(set(replacement_by_signature[key]))

    target_nodes: List[int] = []
    for nid in candidate_graph.topological_order():
        node = candidate_graph.nodes[nid]
        if node.is_input:
            continue
        try:
            prim = get_primitive(node.op_name)
            category = prim.category.value
        except Exception:
            category = ""
        if node.op_name in hyp or category in hyp:
            target_nodes.append(nid)
        elif ("math space" in hyp or "math_space" in hyp) and category == "math_space":
            target_nodes.append(nid)

    if not target_nodes:
        non_input = [nid for nid in candidate_graph.topological_order()
                     if not candidate_graph.nodes[nid].is_input]
        target_nodes = non_input[-2:] if len(non_input) >= 2 else non_input

    ablations: List[Any] = []
    seen: Set[str] = set()
    for nid in target_nodes[:4]:
        node = candidate_graph.nodes[nid]
        try:
            prim = get_primitive(node.op_name)
        except Exception:
            continue
        key = (prim.n_inputs, prim.shape_rule)
        candidates = [name for name in replacement_by_signature.get(key, []) if name != node.op_name]
        if not candidates:
            continue

        # Prefer a non-identical family replacement to produce a meaningful counterfactual.
        replacement = candidates[0]
        for name in candidates:
            try:
                if get_primitive(name).category != prim.category:
                    replacement = name
                    break
            except Exception:
                continue
        rebuilt = _rebuild_graph_with_overrides(
            candidate_graph,
            {nid: {"op_name": replacement, "config": dict(node.config or {})}},
        )
        if rebuilt is None:
            continue
        try:
            fp = rebuilt.fingerprint()
        except Exception:
            continue
        if fp in seen:
            continue
        seen.add(fp)
        ablations.append(rebuilt)
        if len(ablations) >= 4:
            break

    return ablations


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
    enable_perf_tracing: bool = False
    collect_training_curve: bool = False
    gradient_clip_norm: float = 1.0
    optimizer_fused: bool = True
    optimizer_foreach: bool = True
    starvation_check_interval: int = 8
    enable_cuda_graphs: bool = False
    cuda_graph_warmup_steps: int = 3
    loss_check_interval: int = 8
    enable_kernel_profiling: bool = False
    kernel_profile_top_k: int = 20
    # Training data source
    data_mode: str = "corpus"  # "random" | "corpus" | "hydra"
    corpus_path: str = "/home/tim/Projects/LLM/research/micro_corpus.txt"      # TXT or JSONL path for corpus mode
    corpus_format: str = "auto"  # "auto" | "txt" | "jsonl"
    corpus_text_key: str = "text"  # JSONL key when format is jsonl
    tokenizer_mode: str = "byte"  # "byte" | "whitespace"
    corpus_max_chars: int = 200000
    corpus_train_fraction: float = 0.9
    corpus_val_fraction: float = 0.1
    stage1_compute_val_loss: bool = True
    stage1_val_batches: int = 2
    stage1_val_batch_size: int = 4
    stage1_compute_discovery_loss: bool = True
    stage1_discovery_batches: int = 2
    stage1_discovery_batch_size: int = 4
    # HYDRA data loader settings (data_mode="hydra")
    hydra_data_dir: str = "../HYDRA/data"
    hydra_dataset: str = "local_jsonl"  # any HYDRA dataset name
    hydra_project_root: str = "../HYDRA"
    # HuggingFace dataset (data_mode="huggingface")
    hf_dataset: str = ""           # e.g. "roneneldan/TinyStories", "wikitext"
    hf_subset: str = ""            # e.g. "wikitext-2-raw-v1"
    hf_split: str = "train"        # train | validation | test
    hf_text_key: str = "text"      # column name containing text
    # Synthesis grammar
    min_depth: int = 3
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
    # LLM next-step planner (local preferred, remote fallback)
    enable_llm_decision_planner: bool = True
    llm_decision_local_backend: str = ""
    llm_decision_local_model: str = ""
    llm_decision_local_host: str = ""
    llm_decision_remote_backend: str = ""
    llm_decision_remote_model: str = ""
    llm_decision_temperature: float = 0.2
    llm_decision_max_tokens: int = 700
    llm_decision_budget_dollars: float = 0.0
    llm_decision_max_n_programs: int = 200
    llm_decision_max_time_minutes: int = 120
    llm_decision_min_novelty_weight: float = 0.25
    llm_decision_min_family_bonus_weight: float = 0.10
    # Evolution search
    population_size: int = 50
    n_generations: int = 20
    tournament_size: int = 5
    mutation_rate: float = 0.7
    crossover_rate: float = 0.3
    elitism: int = 5
    novelty_weight: float = 0.5
    fitness_weight: float = 0.5
    # Recursive local refinement (winner-tweak loop)
    refinement_top_k: int = 4
    refinement_generations: int = 3
    refinement_mutation_radius: float = 0.35
    refinement_novelty_pressure: float = 0.35
    refinement_min_distance: float = 0.12
    refinement_plateau_patience: int = 2
    refinement_budget_programs: int = 180
    refinement_min_stage1_survivors: int = 2
    refinement_lookback_experiments: int = 4
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
    morph_focus_sparse: bool = False   # force sparse weight-storage options in morphological mode
    morph_sparse_weight_storage: str = ""  # optional explicit sparse storage choice
    morph_compute_routing: str = ""   # optional fixed compute_routing choice for morphology
    morph_channel_mixing: str = ""    # optional fixed channel_mixing choice for morphology
    refine_source_result_ids: str = ""  # comma-separated source result IDs for local fingerprint refinement
    refine_mutations_per_source: int = 4
    refine_intent: str = "balanced"  # balanced|quality|compression|sparsity|novelty
    refine_pool_multiplier: int = 3
    refine_analysis_json: str = ""  # serialized RefinementAnalyzer output for data-driven refinement
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
    breakthrough_raw_threshold: float = 0.70
    breakthrough_normalized_threshold: float = 0.85
    auto_validate_min_novelty_confidence: float = 0.50
    auto_validate_top_n: int = 3
    require_preregistration: bool = True
    auto_preregister: bool = True
    auto_novelty_calibration: bool = True
    novelty_calibration_runs: int = 6
    allow_heuristic_novelty_promotion: bool = False
    heuristic_novelty_justification: str = ""
    # Evidence-based selection policy
    selection_quality_weight: float = 0.40
    selection_novelty_weight: float = 0.25
    selection_efficiency_weight: float = 0.20
    selection_feasibility_weight: float = 0.15
    selection_policy: str = "ucb"  # "ucb" | "epsilon_greedy"
    selection_epsilon: float = 0.20
    selection_ucb_c: float = 1.20
    selection_family_bonus_weight: float = 0.20
    safety_plateau_window: int = 8
    safety_plateau_min_delta: float = 0.01
    switch_epic_breakthrough_confidence_min: float = 0.75
    switch_epic_stagnation_cycles: int = 6
    # External scaling comparison
    enable_scaling_comparison: bool = True
    scaling_reference_families: str = "gpt2"        # comma-separated: "gpt2,mamba"
    scaling_d512_enabled: bool = True                # retrain breakthrough candidates at d=512
    scaling_param_efficiency_target: float = 3.0     # min param efficiency for breakthrough
    scaling_flop_ceiling: float = 2.0                # max FLOP ratio vs reference
    # Checkpoint/resume
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1  # save continuous checkpoint every N experiments
    resume_experiment_id: str = ""  # experiment ID to resume (empty = fresh start)
    keep_checkpoints: bool = False  # keep checkpoints after successful completion
    # Campaign system
    enable_campaigns: bool = True
    knowledge_extraction_interval: int = 3  # every N experiments
    auto_go_no_go: bool = True  # auto-record go/no-go decisions at escalation
    # Stage pass thresholds (overridable by LLM per-cycle)
    stage1_loss_ratio_threshold: float = 0.8
    stage05_stability_threshold: float = 0.5
    investigation_loss_ratio_threshold: float = 0.5
    investigation_robustness_threshold: float = 0.5
    # Pre-investigation gate
    pre_inv_gate_enabled: bool = True
    pre_inv_min_stability: float = 0.3
    pre_inv_max_spectral_norm: float = 50.0
    pre_inv_min_spectral_norm: float = 0.01
    pre_inv_min_improvement_rate: float = 0.0
    pre_inv_top_n: int = 5
    pre_inv_reference_margin: float = 1.5
    pre_inv_probe_enabled: bool = False
    pre_inv_probe_steps_fraction: float = 0.25
    pre_inv_probe_max_lr: float = 0.85
    # Grammar structure probabilities (forwarded to GrammarConfig)
    grammar_split_prob: float = 0.3
    grammar_merge_prob: float = 0.2
    grammar_risky_op_prob: float = 0.1
    grammar_freq_domain_prob: float = 0.0
    # Healer/agent settings
    max_agent_seconds: int = 300
    # LLM consultation in continuous mode
    llm_decision_interval: int = 5  # call Sonnet every N cycles (0 = never in continuous)

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
    # Native runner adapter telemetry
    native_runner: Dict[str, Any] = field(default_factory=_native_runner_progress_report)

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
            from ..morphological_box import ArchSpec
            from ..arch_builder import build_model, BuildConfig
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

            # Simplicity bias: top performers are often shallow/compact.
            # Apply conservative search caps by default under auto-harden.
            if screened.max_depth > 3:
                old = screened.max_depth
                if auto_harden:
                    screened.max_depth = 3
                _record_issue(
                    key="max_depth",
                    severity="medium",
                    reason="Applying simplicity constraint for evolve/novelty: capping max_depth at 3.",
                    old_value=old,
                    suggested_value=3,
                    risk_points=8,
                    adjusted=auto_harden,
                )
            if screened.max_ops > 5:
                old = screened.max_ops
                if auto_harden:
                    screened.max_ops = 5
                _record_issue(
                    key="max_ops",
                    severity="medium",
                    reason="Applying simplicity constraint for evolve/novelty: capping max_ops at 5.",
                    old_value=old,
                    suggested_value=5,
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

    def _get_scaling_reference_manager(self):
        if not hasattr(self, "_scaling_ref_mgr"):
            from ..eval.scaling_reference import ScalingReferenceManager
            cache_path = str(Path(self.notebook_path).parent / "scaling_reference_cache.db")
            self._scaling_ref_mgr = ScalingReferenceManager(cache_path=cache_path)
        return self._scaling_ref_mgr

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

    def _sample_training_input_ids(
        self,
        config: RunConfig,
        dev: torch.device,
        batch_size: int,
        seq_len: int,
        seed: int,
        split: str = "train",
    ) -> torch.Tensor:
        """Sample input IDs from configured data source with deterministic seed."""
        mode = str(config.data_mode or "random").strip().lower()
        generator = torch.Generator(device=dev)
        generator.manual_seed(int(seed))

        if mode == "huggingface":
            batcher = self._get_hf_batcher(config)
            if batcher is not None:
                batch = batcher.sample_batch(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    generator=generator,
                    device=dev,
                    split=split,
                )
                if batch is not None:
                    return batch
            # Fall through to random on failure

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
                    split=split,
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

    def _corpus_version_tag(self, path: str) -> str:
        try:
            stat = os.stat(path)
            name = os.path.basename(path)
            return f"{name}:{stat.st_size}:{stat.st_mtime_ns}"
        except Exception:
            return "missing"

    def _make_baseline_data_fn(self, config: RunConfig, split: str = "train"):
        """Build a data_fn for baseline training when using real data.

        Returns (data_fn, data_tag, cache_data_fn) tuple. data_fn is None for
        random mode (baseline uses its own random tokens). data_tag is a cache
        key suffix. cache_data_fn indicates safe caching for data_fn.
        """
        mode = str(config.data_mode or "random").strip().lower()
        if mode == "huggingface":
            ds_name = str(config.hf_dataset or "").strip()
            subset = str(config.hf_subset or "").strip()
            data_tag = f"hf:{ds_name}:{subset}:{config.hf_split}:{split}"
            step_state = {"step": 0}

            def data_fn(batch_size, seq_len, dev):
                step = step_state["step"]
                step_state["step"] = step + 1
                generator = torch.Generator(device=dev)
                generator.manual_seed(1337 + step)
                batcher = self._get_hf_batcher(config)
                if batcher is not None:
                    batch = batcher.sample_batch(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        generator=generator,
                        device=dev,
                        split=str(split or "train").lower(),
                    )
                    if batch is not None:
                        return batch
                return torch.randint(0, config.vocab_size, (batch_size, seq_len), device=dev, generator=generator)

            return data_fn, data_tag, True
        if mode == "hydra":
            def data_fn(batch_size, seq_len, dev):
                batch = self._get_hydra_batch(config, batch_size, seq_len, dev)
                if batch is not None:
                    return batch
                return torch.randint(0, config.vocab_size, (batch_size, seq_len), device=dev)
            return data_fn, "hydra", False
        if mode == "corpus":
            path = str(config.corpus_path or "").strip()
            version = self._corpus_version_tag(path)
            train_frac = float(getattr(config, "corpus_train_fraction", 0.9) or 0.9)
            val_frac = float(getattr(config, "corpus_val_fraction", 0.1) or 0.1)
            fmt = str(config.corpus_format or "auto")
            text_key = str(config.corpus_text_key or "text")
            tok = str(config.tokenizer_mode or "byte")
            max_chars = int(config.corpus_max_chars)
            split_tag = str(split or "train").lower()
            data_tag = (
                f"corpus:{version}:{fmt}:{text_key}:{tok}:{max_chars}:"
                f"train{train_frac:.3f}:val{val_frac:.3f}:split{split_tag}"
            )
            step_state = {"step": 0}

            def data_fn(batch_size, seq_len, dev):
                step = step_state["step"]
                step_state["step"] = step + 1
                generator = torch.Generator(device=dev)
                generator.manual_seed(1337 + step)
                batcher = self._get_corpus_batcher(config)
                if batcher is not None:
                    batch = batcher.sample_batch(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        generator=generator,
                        device=dev,
                        split=split_tag,
                    )
                    if batch is not None:
                        return batch
                return torch.randint(0, config.vocab_size, (batch_size, seq_len), device=dev, generator=generator)

            return data_fn, data_tag, True
        return None, "random", False

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
            "before": dict(before_progress or {}),
            "after": dict(after_progress or {}),
        }
        if error:
            summary["error"] = str(error)
        return summary

    def _evaluate_switch_epic_guardrails(
        self,
        config: RunConfig,
        nb: LabNotebook,
        cycle_index: int,
    ) -> Dict[str, Any]:
        """Evaluate explicit criteria for switching to a new epic/strategy."""
        confidence_min = float(getattr(config, "switch_epic_breakthrough_confidence_min", 0.75) or 0.75)
        stagnation_cycles = max(3, int(getattr(config, "switch_epic_stagnation_cycles", 6) or 6))

        breakthroughs = nb.get_leaderboard(tier="breakthrough", limit=5, sort_by="composite_score")
        qualified_breakthroughs = []
        for row in breakthroughs:
            conf = float(row.get("novelty_confidence") or 0.0)
            if conf >= confidence_min:
                qualified_breakthroughs.append({
                    "result_id": row.get("result_id"),
                    "composite_score": float(row.get("composite_score") or 0.0),
                    "novelty_confidence": conf,
                })

        recent = list(self._aria_cycle_history[-stagnation_cycles:])
        if recent:
            recent = recent + [dict(self._last_cycle_summary or {})] if self._last_cycle_summary else recent
            recent = recent[-stagnation_cycles:]
        stagnated = bool(recent) and all(int(c.get("delta_stage1_survivors") or 0) == 0 for c in recent)

        should_switch = bool(qualified_breakthroughs) or stagnated
        reasons = []
        if qualified_breakthroughs:
            reasons.append("decision_ready_breakthrough_detected")
        if stagnated:
            reasons.append("stagnation_without_gate_advancement")

        return {
            "should_switch_epic": should_switch,
            "reasons": reasons,
            "criteria": {
                "breakthrough_confidence_min": confidence_min,
                "stagnation_cycles": stagnation_cycles,
            },
            "signals": {
                "qualified_breakthrough_count": len(qualified_breakthroughs),
                "qualified_breakthroughs": qualified_breakthroughs,
                "stagnated_recent_cycles": stagnated,
                "recent_cycle_count": len(recent),
            },
            "evaluated_at_cycle": int(cycle_index),
        }

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

        # Get digest from distiller if available
        _digest = None
        _distiller = getattr(self, "_knowledge_distiller", None)
        if _distiller is not None:
            try:
                _digest = _distiller.get_digest()
            except Exception:
                pass

        mode_rec = self._select_next_mode(config, nb, n_experiments, digest=_digest)
        selected_mode = mode_rec.get("mode", "synthesis")
        mode_reasoning = mode_rec.get("reasoning", "")
        mode_confidence = mode_rec.get("confidence", 0)
        mode_config_overrides = mode_rec.get("config") or {}

        # Extract op_weights and structured_sparsity_bias from mode config
        # (these aren't RunConfig attrs, so _config_with_overrides would ignore them)
        if "op_weights" in mode_config_overrides:
            mode_op_weights = mode_config_overrides.pop("op_weights")
            if isinstance(mode_op_weights, dict):
                self._op_weights_overrides.update({
                    str(k): max(0.01, min(10.0, float(v)))
                    for k, v in mode_op_weights.items()
                    if isinstance(v, (int, float))
                })
                logger.info("Mode recommendation applied op_weights: %s", mode_op_weights)
        if "structured_sparsity_bias" in mode_config_overrides:
            bias = mode_config_overrides.pop("structured_sparsity_bias")
            if isinstance(bias, (int, float)):
                # Store as grammar weight override so _build_grammar_config picks it up
                self._structured_sparsity_bias_override = max(0.0, min(1.0, float(bias)))
                logger.info("Mode recommendation applied structured_sparsity_bias: %.2f", bias)

        cycle_config, override_report = self._config_with_overrides(config, mode_config_overrides)

        if override_report.get("applied"):
            self._emit_event("mode_config_applied", {
                "mode": selected_mode,
                "applied": override_report.get("applied"),
                "ignored": override_report.get("ignored", {}),
                "experiment_number": n_experiments,
            })

        prescreen_mode = "continuous"
        if selected_mode == "evolution":
            prescreen_mode = "evolve"
        elif selected_mode == "novelty":
            prescreen_mode = "novelty"

        cycle_config, cycle_prescreen = self.prescreen_run_config(
            cycle_config,
            mode=prescreen_mode,
            auto_harden=True,
        )
        if cycle_prescreen.get("issue_count", 0) > 0:
            self._emit_event("cycle_prescreen", {
                "mode": selected_mode,
                "report": cycle_prescreen,
                "experiment_number": n_experiments,
            })

        effective_max_time_minutes = self._effective_max_time_minutes(config)

        self._emit_event("mode_selected", {
            "mode": selected_mode,
            "reasoning": mode_reasoning,
            "confidence": mode_confidence,
            "experiment_number": n_experiments,
        })
        logger.info(
            "Cycle %d: mode=%s (confidence=%.2f) — %s",
            n_experiments, selected_mode, mode_confidence,
            mode_reasoning[:120] if mode_reasoning else "no reasoning",
        )

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

        # Per-experiment watchdog: abort if a single cycle exceeds this limit.
        # Modes like novelty/evolution can hang in generation loops; this is the
        # last-resort safety net.  The watchdog sets _stop_event, which the inner
        # loops check, causing a graceful exit rather than a hard kill.
        max_cycle_seconds = int(
            getattr(config, "max_cycle_seconds", 0) or 0
        ) or 1800  # default 30 min
        _watchdog_fired = False

        def _cycle_watchdog():
            nonlocal _watchdog_fired
            _watchdog_fired = True
            logger.error(
                "WATCHDOG: Cycle %d (%s) exceeded %ds — setting stop event",
                n_experiments, selected_mode, max_cycle_seconds,
            )
            self._emit_event("cycle_watchdog", {
                "experiment_number": n_experiments,
                "mode": selected_mode,
                "timeout_seconds": max_cycle_seconds,
            })
            self._stop_event.set()

        watchdog = threading.Timer(max_cycle_seconds, _cycle_watchdog)
        watchdog.daemon = True
        watchdog.start()

        try:
            self._set_aria_cycle_phase(
                "running",
                continuous_active=True,
                cycle_index=n_experiments,
                selected_mode=selected_mode,
                note=f"Running {selected_mode} cycle {n_experiments}.",
            )
            logger.info(
                "Cycle %d: starting %s run [%s] (watchdog=%ds)",
                n_experiments, selected_mode, limit_str, max_cycle_seconds,
            )
            if selected_mode in ("investigation", "validation"):
                self._run_continuous_phase(
                    selected_mode, cycle_config, nb, n_experiments,
                    limit_str, mode_reasoning)
            elif selected_mode == "evolution":
                self._run_continuous_evolution(
                    cycle_config, nb, n_experiments, limit_str, mode_reasoning)
            elif selected_mode == "novelty":
                self._run_continuous_novelty(
                    cycle_config, nb, n_experiments, limit_str, mode_reasoning)
            elif selected_mode == "refinement":
                self._run_continuous_refinement(
                    cycle_config, nb, n_experiments, limit_str, mode_reasoning)
            else:
                self._run_continuous_synthesis(
                    cycle_config, nb, n_experiments, limit_str, mode_reasoning)
        except Exception as e:
            cycle_error = str(e)
            logger.warning("Cycle %d FAILED (%s): %s", n_experiments, selected_mode, e)
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
            watchdog.cancel()
            # If watchdog fired, clear stop event so continuous mode can proceed
            # to the next cycle (the current cycle is already aborted).
            if _watchdog_fired:
                self._stop_event.clear()
                cycle_error = cycle_error or f"Watchdog timeout ({max_cycle_seconds}s)"
                logger.warning(
                    "Cycle %d: watchdog fired, stop event cleared for next cycle",
                    n_experiments,
                )
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
        switch_guardrails = self._evaluate_switch_epic_guardrails(config, nb, n_experiments)
        summary["switch_epic_guardrails"] = switch_guardrails
        if switch_guardrails.get("should_switch_epic"):
            self._emit_event("switch_epic_recommended", switch_guardrails)
            try:
                nb.log_learning_event(
                    "switch_epic_recommended",
                    f"Switch-epic criteria met at cycle {n_experiments}",
                    evidence=json.dumps(switch_guardrails, sort_keys=True),
                )
            except Exception:
                pass
        self._emit_event("aria_cycle_completed", summary)

        delta_s1 = summary.get("delta_stage1_survivors", 0)
        after = summary.get("after", {})
        s0 = after.get("stage0_passed", 0)
        s05 = after.get("stage05_passed", 0)
        s1 = after.get("stage1_passed", 0)
        best_loss = after.get("best_loss_ratio")
        loss_str = f", best loss={best_loss:.4f}" if best_loss else ""
        logger.info(
            "Cycle %d done: S0=%d S0.5=%d S1=%d (ΔS1=%+d)%s",
            n_experiments, s0, s05, s1, delta_s1, loss_str,
        )

        # PROACTIVE: Auto-repair on cycle failure
        if cycle_error:
            self._proactive_cycle_repair(cycle_error, selected_mode, n_experiments, nb)

        # PROACTIVE: Detect stagnation and auto-adjust
        self._proactive_stagnation_check(n_experiments, delta_s1, nb)

        # PROACTIVE: Detect recurring error patterns and auto-fix
        self._proactive_recurring_error_fix(n_experiments, nb)

        # PROACTIVE: Spawn agent to investigate persistent S1 stagnation
        self._proactive_stagnation_agent(n_experiments, nb)
        self._maybe_trigger_integrity_healer(
            nb=nb,
            experiment_id=after_progress.get("experiment_id"),
        )

        return summary

    def _proactive_cycle_repair(self, error: str, mode: str,
                                cycle_index: int, nb: LabNotebook):
        """Spawn a code agent to fix cycle failures autonomously."""
        try:
            from .api import _spawn_code_agent_task, _should_autospawn_self_repair
            if not _should_autospawn_self_repair(error):
                return
            # Rate-limit: don't spawn repair agents more than once per 3 minutes
            now = time.time()
            last = getattr(self, "_last_cycle_repair_spawn", 0)
            if now - last < 180:
                return
            self._last_cycle_repair_spawn = now

            # Build targeted goal based on error type
            import re as _re
            goal_parts = [f"Cycle {cycle_index} ({mode}) failed with: {error[:500]}."]
            # Extract file references from traceback
            file_refs = _re.findall(r'File "([^"]+)", line (\d+)', error)
            if file_refs:
                goal_parts.append(
                    f"Key locations: {', '.join(f'{f}:{l}' for f, l in file_refs[:5])}."
                )
            # Detect error class for targeted advice
            if "ImportError" in error or "ModuleNotFoundError" in error:
                goal_parts.append("This is an import error — check module paths and __init__.py files.")
            elif "CUDA" in error or "RuntimeError" in error:
                goal_parts.append("This may be a CUDA/tensor error — check device placement and tensor shapes.")
            elif "TypeError" in error or "AttributeError" in error:
                goal_parts.append("This is a type/attribute error — check function signatures and object interfaces.")
            goal_parts.append(
                "Investigate root cause. Apply minimal safe fix. "
                "Use local Ollama model if available."
            )
            notebook_path = str(nb._db_path) if hasattr(nb, "_db_path") else ""
            task = _spawn_code_agent_task(
                goal=" ".join(goal_parts),
                notebook_path=notebook_path,
                allow_write=True,
            )
            task_id = task.get("task_id", "unknown")
            logger.info("Proactive repair agent spawned: %s for cycle %d error", task_id, cycle_index)
            nb.log_learning_event(
                "proactive_repair",
                f"Auto-spawned repair agent {task_id} for cycle {cycle_index} failure: {error[:200]}",
                task_id=task_id,
            )
        except Exception as e:
            logger.debug("Proactive cycle repair failed to spawn: %s", e)
        self._invoke_code_healer(
            nb=nb,
            trigger_type="repeated_exception",
            experiment_id=getattr(self._progress, "experiment_id", None),
            scope=f"Cycle failure in mode={mode}: {error[:240]}",
            reproduction_steps=["python -m pytest tests/test_integration.py -k \"start_experiment\" -x --tb=short"],
            acceptance_tests=["python -m pytest tests/test_integration.py -k \"start_experiment\" -x --tb=short"],
            trigger_payload={"mode": mode, "cycle_index": cycle_index, "error": error[:1000]},
        )

    def _proactive_stagnation_check(self, cycle_index: int, delta_s1: int,
                                    nb: LabNotebook):
        """Detect stagnation (consecutive zero-survivor cycles) and auto-adjust."""
        history = getattr(self, "_aria_cycle_history", [])
        if len(history) < 3:
            return

        # Check last 3 cycles for zero survivors
        recent = history[-3:]
        consecutive_zero = all(
            (h.get("delta_stage1_survivors", 0) == 0 and
             (h.get("after") or {}).get("stage1_passed", 0) == 0)
            for h in recent
        )
        if not consecutive_zero:
            return

        # Already applied anti-stagnation recently?
        last_anti = getattr(self, "_last_anti_stagnation_cycle", -10)
        if cycle_index - last_anti < 3:
            return
        self._last_anti_stagnation_cycle = cycle_index

        # Apply anti-stagnation: diversify grammar, reduce depth, try different source
        adjustments = {
            "max_depth": max(4, getattr(self, "_grammar_weight_overrides", {}).get("max_depth", 6) - 2),
            "max_ops": max(4, getattr(self, "_grammar_weight_overrides", {}).get("max_ops", 8) - 2),
        }
        self._last_chat_config_overrides = {
            **(self._last_chat_config_overrides or {}),
            **adjustments,
        }
        # Reset any extreme grammar weights
        if hasattr(self, "_grammar_weight_overrides"):
            for k in list(self._grammar_weight_overrides.keys()):
                v = self._grammar_weight_overrides[k]
                if isinstance(v, (int, float)) and (v > 6.0 or v < 0.3):
                    self._grammar_weight_overrides[k] = 1.0

        nb.log_learning_event(
            "anti_stagnation",
            f"3 consecutive zero-S1 cycles detected at cycle {cycle_index}. "
            f"Auto-reduced depth/ops ({adjustments}), reset extreme grammar weights.",
            adjustments=adjustments,
        )
        logger.info(
            "Anti-stagnation triggered at cycle %d: %s",
            cycle_index, adjustments,
        )

    def _proactive_stagnation_agent(self, cycle_index: int, nb: LabNotebook):
        """Spawn a code agent to investigate persistent S1 stagnation."""
        history = getattr(self, "_aria_cycle_history", [])
        window = 5
        if len(history) < window:
            return

        recent = history[-window:]
        all_zero = all(
            h.get("delta_stage1_survivors", 0) == 0
            for h in recent
        )
        if not all_zero:
            return

        # Rate-limit: skip if recently spawned
        if cycle_index - self._last_stagnation_agent_cycle < 5:
            return

        try:
            from .api import _spawn_code_agent_task
            from .analytics import ExperimentAnalytics

            analytics = ExperimentAnalytics(nb)
            neg = analytics.negative_results_synthesis()
            failed_ops = [op["op_name"] for op in neg.get("failed_ops", [])[:10]]
            weights = analytics.compute_grammar_weights() or {}
            trajectory = analytics.learning_trajectory()
            s1_slope = trajectory.get("s1_slope", 0) if trajectory else 0

            recent_s1 = [
                (h.get("after") or {}).get("stage1_passed", 0)
                for h in recent
            ]

            goal = (
                f"STAGNATION: 0 new S1 survivors for {window} consecutive cycles "
                f"(S1 counts: {recent_s1}, trend slope={s1_slope:.4f}). "
                f"Current grammar weights: {json.dumps(weights, indent=None)}. "
                f"Failed ops (0% S1 rate, >=5 uses): {failed_ops}. "
                f"Read these files and propose a targeted fix:\n"
                f"- synthesis/grammar.py: GrammarConfig.category_weights defaults and generate_layer_graph() — "
                f"are failing categories over-weighted?\n"
                f"- scientist/analytics.py: _compute_weights_from_stats() noise guard and weight formula — "
                f"is the learning signal being suppressed?\n"
                f"- morphological_box.py: dimension constraints in ArchSpec — "
                f"is the config search space too narrow?\n"
                f"- evaluator.py: stage1 pass criteria — is the bar set unreachably high?\n"
                f"Identify the single most likely bottleneck and make a minimal, targeted change."
            )

            notebook_path = str(nb._db_path) if hasattr(nb, "_db_path") else ""
            task = _spawn_code_agent_task(
                goal=goal,
                notebook_path=notebook_path,
                allow_write=True,
            )
            task_id = task.get("task_id", "unknown")
            self._last_stagnation_agent_cycle = cycle_index

            nb.log_learning_event(
                "proactive_stagnation_agent",
                f"Spawned agent {task_id} for S1 stagnation ({window} flat cycles)",
                task_id=task_id, s1_slope=s1_slope,
                failed_ops=failed_ops, recent_s1=recent_s1,
            )
            logger.info(
                "Stagnation agent spawned at cycle %d (task %s, slope=%.4f)",
                cycle_index, task_id, s1_slope,
            )
        except Exception as e:
            logger.debug("Stagnation agent spawn failed: %s", e)
        self._invoke_code_healer(
            nb=nb,
            trigger_type="plateau",
            experiment_id=getattr(self._progress, "experiment_id", None),
            scope=f"Persistent stagnation at cycle {cycle_index}",
            reproduction_steps=["python -m pytest tests/test_selection_policy.py -x --tb=short"],
            acceptance_tests=["python -m pytest tests/test_selection_policy.py -x --tb=short"],
            trigger_payload={"cycle_index": cycle_index, "window": window},
        )

    def _proactive_recurring_error_fix(self, cycle_index: int, nb: LabNotebook):
        """Detect recurring error patterns across cycles and spawn a fix agent.

        If the same error class (first line of traceback) appears 2+ times
        in the last 5 cycles, spawn a code agent to fix the root cause.
        """
        history = getattr(self, "_aria_cycle_history", [])
        if len(history) < 2:
            return

        # Collect errors from recent cycles
        recent = history[-5:]
        errors = []
        for h in recent:
            err = h.get("error")
            if err:
                # Normalize: take the error class/first line
                first_line = str(err).split("\n")[0].strip()[:200]
                errors.append(first_line)

        if len(errors) < 2:
            return

        # Find repeated error patterns (same first 80 chars)
        from collections import Counter
        error_keys = [e[:80] for e in errors]
        counts = Counter(error_keys)
        repeated = [(k, c) for k, c in counts.items() if c >= 2]
        if not repeated:
            return

        # Rate-limit: don't spawn more than 1 recurring-error agent per 10 minutes
        now = time.time()
        last = getattr(self, "_last_recurring_error_agent", 0)
        if now - last < 600:
            return
        self._last_recurring_error_agent = now

        worst_error = max(repeated, key=lambda x: x[1])
        try:
            from .api import _spawn_code_agent_task
            notebook_path = str(nb._db_path) if hasattr(nb, "_db_path") else ""
            task = _spawn_code_agent_task(
                goal=(
                    f"RECURRING ERROR (seen {worst_error[1]}x in last 5 cycles): "
                    f"{worst_error[0]}. "
                    "This error keeps happening. Find the ROOT CAUSE in "
                    "scientist/runner.py, synthesis/, eval/, or training/ and fix it. "
                    "Use local Ollama model if available."
                ),
                notebook_path=notebook_path,
                allow_write=True,
            )
            task_id = task.get("task_id", "unknown")
            nb.log_learning_event(
                "proactive_recurring_error_fix",
                f"Spawned agent {task_id} for recurring error ({worst_error[1]}x): "
                f"{worst_error[0][:200]}",
                task_id=task_id,
                error_pattern=worst_error[0],
                occurrences=worst_error[1],
            )
            logger.info(
                "Recurring error agent spawned: %s for pattern seen %dx: %s",
                task_id, worst_error[1], worst_error[0][:100],
            )
        except Exception as e:
            logger.debug("Failed to spawn recurring error fix agent: %s", e)
        self._invoke_code_healer(
            nb=nb,
            trigger_type="repeated_exception",
            experiment_id=getattr(self._progress, "experiment_id", None),
            scope=f"Recurring error pattern {worst_error[0][:200]}",
            reproduction_steps=["python -m pytest tests/test_integration.py -k \"error\" -x --tb=short"],
            acceptance_tests=["python -m pytest tests/test_integration.py -k \"error\" -x --tb=short"],
            trigger_payload={"error_pattern": worst_error[0], "occurrences": worst_error[1]},
        )

    def _healer_signature_seen(self, error: str, scope: str) -> bool:
        """Return True if this error was already sent to the healer recently."""
        sig = hashlib.md5((error[:200] + scope[:100]).encode()).hexdigest()[:12]
        now = time.time()
        # Expire old signatures (5-minute TTL)
        self._recent_healer_signatures = {
            k: v for k, v in self._recent_healer_signatures.items()
            if now - v < 300
        }
        if sig in self._recent_healer_signatures:
            return True
        self._recent_healer_signatures[sig] = now
        return False

    def _invoke_code_healer(
        self,
        nb: LabNotebook,
        trigger_type: str,
        experiment_id: Optional[str],
        scope: str,
        reproduction_steps: Optional[List[str]] = None,
        acceptance_tests: Optional[List[str]] = None,
        trigger_payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run Code Healer state machine and log output to SQLite."""
        if self._healer is None:
            return None
        if self._healer_signature_seen(
            (trigger_payload or {}).get("error", scope),
            scope,
        ):
            return None
        repro = reproduction_steps or []
        tests = acceptance_tests or ["python -m pytest tests -k integration -x --tb=short"]
        try:
            result = self._healer.open_and_run(
                HealerTaskSpec(
                    experiment_id=experiment_id,
                    trigger_type=trigger_type,
                    scope=scope,
                    reproduction_steps=repro,
                    acceptance_tests=tests,
                    trigger_payload=trigger_payload or {},
                )
            )
            nb.log_learning_event(
                "code_healer_invoked",
                f"CodeHealer handled trigger={trigger_type} state={result.get('state')}",
                trigger_type=trigger_type,
                healer_result=result,
            )
            # If heal succeeded, schedule retry on next continuous cycle
            if result and result.get("verification_ok"):
                self._pending_heal_retry = {
                    "trigger_type": trigger_type,
                    "experiment_id": experiment_id,
                    "scope": scope,
                }
            return result
        except Exception as e:
            nb.log_learning_event(
                "code_healer_failed",
                f"CodeHealer failed for trigger={trigger_type}: {e}",
                trigger_type=trigger_type,
                error=str(e),
            )
            return None

    def _maybe_trigger_integrity_healer(self, nb: LabNotebook, experiment_id: Optional[str]) -> None:
        """Run integrity checks periodically and invoke healer on failures."""
        now = time.time()
        if now - self._last_healer_integrity_check < 300:
            return
        self._last_healer_integrity_check = now
        try:
            from ..tools.novelty_integrity_check import run_integrity_check
            report = run_integrity_check(nb, calibrate_if_missing=False, runs=4)
            if report.get("ok"):
                return
            self._invoke_code_healer(
                nb=nb,
                trigger_type="integrity_failure",
                experiment_id=experiment_id,
                scope="Novelty integrity check failure.",
                reproduction_steps=[
                    f"python tools/novelty_integrity_check.py --db {shlex.quote(str(nb.db_path))}"
                ],
                acceptance_tests=[
                    f"python tools/novelty_integrity_check.py --db {shlex.quote(str(nb.db_path))}"
                ],
                trigger_payload=report,
            )
        except Exception as e:
            logger.debug("Integrity healer check failed to execute: %s", e)

    @staticmethod
    def _config_with_overrides(
        base_config: RunConfig,
        overrides: Dict[str, Any],
    ) -> Tuple[RunConfig, Dict[str, Dict[str, Any]]]:
        """Apply allowed per-cycle mode overrides to a cloned RunConfig."""
        effective = RunConfig.from_dict(base_config.to_dict())
        applied: Dict[str, Any] = {}
        ignored: Dict[str, Any] = {}

        for key, value in (overrides or {}).items():
            if hasattr(effective, key):
                setattr(effective, key, value)
                applied[key] = value
            else:
                ignored[key] = value

        return effective, {"applied": applied, "ignored": ignored}

    def execute_chat_action(self, action: Dict[str, Any], nb) -> Dict[str, Any]:
        """Execute an action dispatched from Aria's chat response.

        Supported types: adjust_config, adjust_grammar, start_experiment, edit_file.
        """
        action_type = str(action.get("type") or "").strip()

        if action_type == "adjust_config":
            changes = action.get("changes") or {}
            if not isinstance(changes, dict) or not changes:
                return {"status": "error", "error": "No changes provided"}
            # Apply via _config_with_overrides on a fresh default config
            base = RunConfig()
            effective, report = self._config_with_overrides(base, changes)
            # Store as the new defaults for future experiments
            self._last_chat_config_overrides = changes
            nb.log_learning_event(
                "chat_config_adjusted",
                f"Aria adjusted config: {report.get('applied', {})}",
                changes=report.get("applied", {}),
                ignored=report.get("ignored", {}),
            )
            return {"status": "applied", "changes": report.get("applied", {}),
                    "ignored": report.get("ignored", {})}

        elif action_type == "adjust_grammar":
            weights = action.get("weights") or {}
            if not isinstance(weights, dict) or not weights:
                return {"status": "error", "error": "No weights provided"}
            # Validate values are numeric
            clean_weights = {}
            for k, v in weights.items():
                try:
                    clean_weights[str(k)] = float(v)
                except (ValueError, TypeError):
                    pass
            if not clean_weights:
                return {"status": "error", "error": "No valid numeric weights"}
            self._grammar_weight_overrides.update(clean_weights)
            nb.log_learning_event(
                "chat_grammar_adjusted",
                f"Aria adjusted grammar weights: {clean_weights}",
                weights=clean_weights,
                all_overrides=dict(self._grammar_weight_overrides),
            )
            return {"status": "applied", "weights": clean_weights}

        elif action_type == "start_experiment":
            if self.is_running:
                return {"status": "busy", "error": "An experiment is already running"}
            mode = str(action.get("mode") or "synthesis").strip().lower()
            config_overrides = action.get("config") or {}
            config = RunConfig()
            if isinstance(config_overrides, dict):
                for k, v in config_overrides.items():
                    if hasattr(config, k):
                        setattr(config, k, v)
            try:
                if mode in {"sparse_morph", "sparse_morphology", "sparse_morphological"}:
                    config.model_source = "morphological_box"
                    config.morph_focus_sparse = True
                    config.n_programs = max(120, int(config.n_programs))
                    config.n_layers = max(1, min(int(config.n_layers), 4))
                    config.max_depth = max(2, min(int(config.max_depth), 6))
                    config.max_ops = max(4, min(int(config.max_ops), 10))
                    exp_id = self.start_experiment(config)
                if mode == "evolution":
                    exp_id = self.start_evolution(config)
                elif mode == "novelty":
                    exp_id = self.start_novelty_search(config)
                elif mode in {"sparse_morph", "sparse_morphology", "sparse_morphological"}:
                    pass
                else:
                    exp_id = self.start_experiment(config)
                return {"status": "started", "experiment_id": exp_id, "mode": mode}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        elif action_type == "edit_file":
            return self._execute_edit_file_action(action, nb)

        elif action_type == "maintain_database":
            return self._execute_maintain_database_action(action, nb)

        else:
            return {"status": "error", "error": f"Unknown action type: {action_type}"}

    def _execute_edit_file_action(self, action: Dict[str, Any], nb) -> Dict[str, Any]:
        """Execute an edit_file action with safety rails."""
        import py_compile
        import shutil

        path = str(action.get("path") or "").strip()
        search = str(action.get("search") or "")
        replace = str(action.get("replace") or "")
        description = str(action.get("description") or "Chat-initiated edit")

        # Safety: reject path traversal
        if ".." in path:
            return {"status": "error", "error": "Path traversal (..) not allowed"}

        # Safety: allow edits only within known project subpaths
        allowed_prefixes = (
            "research/",
            "scientist/", "synthesis/", "eval/", "search/", "training/",
            "dashboard/", "tests/", "tools/", "mathspaces/",
        )
        if not any(path.startswith(prefix) for prefix in allowed_prefixes):
            return {"status": "error", "error": "Path must be under research/ or a known project folder"}

        # Safety: only .py and .js files
        if not (path.endswith(".py") or path.endswith(".js")):
            return {"status": "error", "error": "Only .py and .js files can be edited"}

        # Resolve to absolute path.
        # project_root is typically <repo>/research when running from the package layout.
        # If the incoming path already starts with research/, resolve from repo root;
        # otherwise resolve from project_root directly.
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        repo_root = os.path.dirname(project_root)
        if path.startswith("research/"):
            abs_path = os.path.normpath(os.path.join(repo_root, path))
        else:
            abs_path = os.path.normpath(os.path.join(project_root, path))

        # Double-check resolved path is under project
        if not abs_path.startswith(project_root):
            return {"status": "error", "error": "Resolved path escapes project directory"}

        if not os.path.isfile(abs_path):
            return {"status": "error", "error": f"File not found: {path}"}

        # Read current content
        with open(abs_path, "r") as f:
            content = f.read()

        if search not in content:
            return {"status": "error", "error": "Search string not found in file"}

        # Create backup
        timestamp = int(time.time())
        backup_path = f"{abs_path}.bak.{timestamp}"
        shutil.copy2(abs_path, backup_path)

        # Apply edit
        new_content = content.replace(search, replace, 1)
        with open(abs_path, "w") as f:
            f.write(new_content)

        # Syntax check for .py files
        if path.endswith(".py"):
            try:
                py_compile.compile(abs_path, doraise=True)
            except py_compile.PyCompileError as e:
                # Restore backup
                shutil.copy2(backup_path, abs_path)
                os.remove(backup_path)
                return {"status": "error", "error": f"Syntax error after edit, reverted: {e}"}

        # Log to notebook
        nb.log_learning_event(
            "chat_file_edited",
            f"Aria edited {path}: {description}",
            path=path,
            backup=backup_path,
            description=description,
        )

        return {"status": "applied", "path": path, "backup": backup_path,
                "description": description}

    # ── Database Maintenance Actions ──────────────────────────────────────

    _MAINTENANCE_OPS = {
        "purge_empty_experiments",
        "purge_junk_programs",
        "reset_op_stats",
        "clear_toxic_signatures",
        "vacuum",
        "backfill_failure_signatures",
    }

    def _execute_maintain_database_action(
        self, action: Dict[str, Any], nb: LabNotebook,
    ) -> Dict[str, Any]:
        """Execute a database maintenance operation.

        Allowed operations:
          purge_empty_experiments  — delete failed experiments with no results
          purge_junk_programs      — delete S0 failures with no error classification
          reset_op_stats           — reset op_success_rates for specific ops
          clear_toxic_signatures   — remove failure_signatures for specific ops
          vacuum                   — reclaim disk space
          backfill_failure_signatures — one-time backfill from existing results
        """
        operation = str(action.get("operation") or "").strip()
        if operation not in self._MAINTENANCE_OPS:
            return {
                "status": "error",
                "error": f"Unknown maintenance operation: {operation}. "
                         f"Allowed: {', '.join(sorted(self._MAINTENANCE_OPS))}",
            }

        try:
            if operation == "purge_empty_experiments":
                n = nb.purge_empty_experiments()
                nb.log_learning_event(
                    "maintenance_purge_experiments",
                    f"Aria purged {n} empty failed experiments",
                )
                return {"status": "applied", "deleted_experiments": n}

            elif operation == "purge_junk_programs":
                # Delete S0 failures with no error_type (no learning signal)
                cur = nb.conn.execute(
                    "DELETE FROM program_results "
                    "WHERE (stage0_passed = 0 OR stage0_passed IS NULL) "
                    "AND (error_type IS NULL OR error_type = '')"
                )
                n = cur.rowcount
                nb._maybe_commit()
                nb.log_learning_event(
                    "maintenance_purge_junk",
                    f"Aria purged {n} junk S0 failure records",
                )
                return {"status": "applied", "deleted_programs": n}

            elif operation == "reset_op_stats":
                ops = action.get("ops") or []
                if not isinstance(ops, list) or not ops:
                    return {"status": "error", "error": "Provide 'ops' list of op names to reset"}
                op_names = [str(o).strip() for o in ops if str(o).strip()]
                if not op_names:
                    return {"status": "error", "error": "No valid op names provided"}
                placeholders = ",".join("?" * len(op_names))
                cur = nb.conn.execute(
                    f"DELETE FROM op_success_rates WHERE op_name IN ({placeholders})",
                    op_names,
                )
                n = cur.rowcount
                nb._maybe_commit()
                nb.log_learning_event(
                    "maintenance_reset_op_stats",
                    f"Aria reset op stats for {op_names} ({n} rows)",
                    ops=op_names,
                )
                return {"status": "applied", "ops_reset": op_names, "rows_deleted": n}

            elif operation == "clear_toxic_signatures":
                ops = action.get("ops") or []
                if not isinstance(ops, list) or not ops:
                    return {"status": "error", "error": "Provide 'ops' list of op names to clear signatures for"}
                total = 0
                for op in ops:
                    op = str(op).strip()
                    if not op:
                        continue
                    cur = nb.conn.execute(
                        "DELETE FROM failure_signatures WHERE signature LIKE ?",
                        (f"%{op}%",),
                    )
                    total += cur.rowcount
                nb._maybe_commit()
                nb.log_learning_event(
                    "maintenance_clear_toxic",
                    f"Aria cleared {total} toxic signatures for {ops}",
                    ops=[str(o).strip() for o in ops],
                )
                return {"status": "applied", "signatures_deleted": total, "ops": [str(o).strip() for o in ops]}

            elif operation == "vacuum":
                nb.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                # VACUUM requires isolation_level=None; run on a fresh connection
                import sqlite3
                vac_conn = sqlite3.connect(nb.db_path, isolation_level=None)
                vac_conn.execute("VACUUM")
                vac_conn.close()
                nb.log_learning_event(
                    "maintenance_vacuum",
                    "Aria ran VACUUM to reclaim disk space",
                )
                return {"status": "applied", "operation": "vacuum"}

            elif operation == "backfill_failure_signatures":
                n = nb.backfill_failure_signatures()
                return {"status": "applied", "signatures_created": n}

        except Exception as e:
            logger.warning("Maintenance action %s failed: %s", operation, e)
            return {"status": "error", "error": str(e)[:200]}

        return {"status": "error", "error": "Unreachable"}

    def _compression_focus_override(
        self,
        recommendation: Dict[str, Any],
        fallback_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Bias toward compact/compression runs when compression coverage is thin."""
        mode = str(recommendation.get("mode") or "synthesis").strip().lower()
        if mode in {"investigation", "validation"}:
            return None

        summary = fallback_data.get("compression_summary") or {}
        n_tested = int(summary.get("n_tested") or 0)
        compressed_test_share = float(fallback_data.get("compressed_test_share") or 0.0)
        n_experiments = int(fallback_data.get("n_experiments_in_session") or 0)

        if n_tested < 8:
            return None
        if compressed_test_share >= 0.20:
            return None
        if n_experiments % 3 != 0:
            return None

        compressed_survival = float(summary.get("compressed_survival_rate") or 0.0)
        overall_survival = float(summary.get("overall_survival_rate") or 0.0)
        return {
            "mode": "synthesis",
            "reasoning": (
                "Compression examination injection: compressed coverage is under-target "
                f"({compressed_test_share:.1%} of tested programs). Running a compact synthesis "
                "cycle to improve quality-retention-per-byte evidence before further mode pivots. "
                f"Compressed survival={compressed_survival:.1%}, overall survival={overall_survival:.1%}."
            ),
            "confidence": max(float(recommendation.get("confidence") or 0.0), 0.72),
            "config": {
                "n_programs": max(60, int(fallback_data.get("base_n_programs") or 60)),
                "max_depth": 5,
                "max_ops": 8,
                "math_space_weight": 2.5,
                "residual_prob": 0.82,
                "model_source": "mixed",
                "morph_ratio": 0.85,
            },
            "compression_focus": True,
        }

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
                         hypothesis: Optional[str] = None,
                         preregistration: Optional[Dict[str, Any]] = None,
                         exploratory: bool = False) -> str:
        """Start an experiment in a background thread. Returns experiment ID."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        config, prescreen = self.prescreen_run_config(
            config,
            mode="single",
            auto_harden=True,
        )

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

        # Populate refuted hypotheses cache for similarity gating
        self._populate_refuted_cache(nb)

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

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="synthesis",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=hypothesis_metadata,
            preregistration=preregistration,
            exploratory=exploratory,
            created_by="start_experiment",
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
            "prescreen": prescreen,
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

        config, _ = self.prescreen_run_config(
            config,
            mode="continuous",
            auto_harden=True,
        )

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

        limits = []
        if config.max_experiments > 0:
            limits.append(f"max_experiments={config.max_experiments}")
        if config.max_time_minutes > 0:
            limits.append(f"max_time={config.max_time_minutes}min")
        if config.max_cost_dollars > 0:
            limits.append(f"max_cost=${config.max_cost_dollars:.2f}")
        logger.info(
            "Starting continuous session: %d programs/cycle, dim=%d, "
            "depth=%d, ops=%d, device=%s [%s]",
            config.n_programs, config.model_dim, config.max_depth,
            config.max_ops, config.device,
            ", ".join(limits) if limits else "no limits",
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

    def start_fingerprint_refinement(
        self,
        result_ids: List[str],
        config: RunConfig,
        hypothesis: Optional[str] = None,
    ) -> str:
        """Start local mutation refinement around selected fingerprint sources."""
        ids = [rid.strip() for rid in result_ids if str(rid).strip()]
        if not ids:
            raise ValueError("result_ids required for fingerprint refinement")

        refine_config = RunConfig.from_dict(config.to_dict())
        refine_config.model_source = "fingerprint_refine"
        refine_config.refine_source_result_ids = ",".join(ids)
        if refine_config.refine_mutations_per_source <= 0:
            refine_config.refine_mutations_per_source = 1

        source_stage1_passed = 0
        recent_synthesis_s1_rate = 0.0
        source_rows: List[Dict[str, Any]] = []
        recommendation: Optional[Dict[str, Any]] = None
        try:
            nb = self._make_notebook()
            recent = self._recent_synthesis_health(nb, window=5)
            recent_synthesis_s1_rate = float(recent.get("s1_rate") or 0.0)
            for rid in ids:
                row = nb.get_program_detail(rid)
                if row and row.get("stage1_passed"):
                    source_stage1_passed += 1
                if isinstance(row, dict):
                    source_rows.append(row)

            requested_intent = str(refine_config.refine_intent or "balanced").strip().lower()
            if requested_intent in {"recommended", "auto"}:
                # Auto-run RefinementAnalyzer if no pre-computed analysis
                if not refine_config.refine_analysis_json and source_rows:
                    try:
                        from .analytics import ExperimentAnalytics, RefinementAnalyzer
                        analytics = ExperimentAnalytics(nb)
                        analyzer = RefinementAnalyzer(analytics)
                        primary_row = source_rows[0]
                        primary_id = primary_row.get("result_id", ids[0])
                        analysis = analyzer.analyze_program_for_refinement(primary_id, primary_row)
                        recipe = analysis.get("recipe", {})
                        resolved_intent = recipe.get("recommended_intent", "balanced")
                        recommendation = {
                            "intent": resolved_intent,
                            "rationale": recipe.get("primary_target", ""),
                            "evidence": recipe.get("grammar_hints", {}),
                        }
                        refine_config.refine_analysis_json = json.dumps(analysis)
                    except Exception as e:
                        logger.warning("RefinementAnalyzer failed, falling back: %s", e)
                        resolved_intent, recommendation = self._recommend_refinement_intent(
                            nb, source_rows,
                        )
                else:
                    resolved_intent, recommendation = self._recommend_refinement_intent(
                        nb,
                        source_rows,
                    )
                refine_config.refine_intent = resolved_intent
            nb.close()
        except Exception:
            recent_synthesis_s1_rate = 0.0

        if hypothesis is None:
            intent_spec = self._refinement_intent_spec(refine_config.refine_intent)
            source_rule = (
                f"source_selection_rule=result_ids({len(ids)}) with "
                f"stage1_survivor_sources={source_stage1_passed}/{len(ids)}"
            )
            mutation_plan = (
                "mutation_mechanism=evolution_local_neighborhood("
                f"operators=op_replace|config_tweak|edge_rewire, mutation_rate={refine_config.mutation_rate:.2f}, "
                f"mutations_per_source={refine_config.refine_mutations_per_source}, "
                f"pool_multiplier={max(1, int(refine_config.refine_pool_multiplier or 1))})"
            )
            baseline_s1 = f"recent_synthesis_s1_rate={recent_synthesis_s1_rate:.3f}"
            success_criteria = (
                "success_criteria=(stage0_pass_rate>=0.95 AND stage05_pass_rate>=0.70) "
                "AND (delta_s1_rate>=+0.03_vs_recent OR best_loss_ratio<=0.98*parent_loss_ratio)"
            )
            fallback_plan = (
                "fallback_plan=if(no_stage1_improvement OR no_stage1_sources) "
                "queue_ablation_suite_and_novelty_mode"
            )
            recommendation_clause = ""
            if recommendation:
                recommendation_clause = (
                    " recommended_intent="
                    f"{recommendation.get('intent')}"
                    f" rationale={recommendation.get('rationale')}"
                    f" evidence={recommendation.get('evidence')}"
                    ";"
                )
            hypothesis = (
                "Fingerprint refinement hypothesis: "
                f"{source_rule}; "
                f"{mutation_plan}; "
                f"intent={intent_spec['name']} weights={intent_spec['weights']} "
                f"score={intent_spec['formula']}; "
                f"{recommendation_clause} "
                f"{baseline_s1}; "
                f"{success_criteria}; "
                f"{fallback_plan}."
            )

        return self.start_experiment(refine_config, hypothesis=hypothesis)

    def _recommend_refinement_intent(
        self,
        nb: LabNotebook,
        source_rows: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        """Recommend refinement intent from historical quality/novelty/compression evidence."""
        if not source_rows:
            return "balanced", {
                "intent": "balanced",
                "rationale": "no_source_rows",
                "evidence": {"source_count": 0},
            }

        op_success = self._op_success_lookup(nb)
        sparse_hint_ops = ("sparse", "gate", "topk", "mask", "threshold", "skip", "mixture")

        loss_values: List[float] = []
        novelty_values: List[float] = []
        param_values: List[float] = []
        op_success_values: List[float] = []
        sparse_ratios: List[float] = []

        for row in source_rows:
            loss = row.get("loss_ratio")
            novelty = row.get("novelty_score")
            params = row.get("param_count") or row.get("graph_n_params_estimate")

            if isinstance(loss, (int, float)):
                loss_values.append(float(loss))
            if isinstance(novelty, (int, float)):
                novelty_values.append(float(novelty))
            if isinstance(params, (int, float)) and float(params) > 0:
                param_values.append(float(params))

            ops: List[str] = []
            graph_json = row.get("graph_json")
            if isinstance(graph_json, str) and graph_json.strip():
                try:
                    graph_data = json.loads(graph_json)
                    nodes = graph_data.get("nodes", {}) if isinstance(graph_data, dict) else {}
                    for nd in nodes.values():
                        if not isinstance(nd, dict):
                            continue
                        op_name = str(nd.get("op_name") or "").strip().lower()
                        if not op_name or op_name == "input":
                            continue
                        ops.append(op_name)
                except Exception:
                    ops = []

            if ops:
                scores = [float(op_success.get(op, 0.5)) for op in ops]
                op_success_values.append(sum(scores) / len(scores))
                sparse_ratio = sum(
                    1.0 for op in ops if any(token in op for token in sparse_hint_ops)
                ) / len(ops)
                sparse_ratios.append(float(sparse_ratio))

        mean_loss = (sum(loss_values) / len(loss_values)) if loss_values else None
        mean_novelty = (sum(novelty_values) / len(novelty_values)) if novelty_values else None
        mean_params = (sum(param_values) / len(param_values)) if param_values else None
        mean_op_success = (
            sum(op_success_values) / len(op_success_values)
        ) if op_success_values else None
        mean_sparse_ratio = (
            sum(sparse_ratios) / len(sparse_ratios)
        ) if sparse_ratios else None

        intent = "balanced"
        rationale = "mixed_signals"
        if ((mean_loss is not None and mean_loss >= 0.75)
                or (mean_op_success is not None and mean_op_success < 0.35)):
            intent = "quality"
            rationale = "weak_quality_signal"
        elif (mean_params is not None and mean_params >= 500_000
              and (mean_loss is None or mean_loss <= 0.80)):
            intent = "compression"
            rationale = "high_parameter_budget"
        elif mean_novelty is not None and mean_novelty < 0.45:
            intent = "novelty"
            rationale = "low_novelty_signal"
        elif (mean_sparse_ratio is not None and mean_sparse_ratio < 0.10
              and mean_params is not None and mean_params >= 1_000_000):
            intent = "sparsity"
            rationale = "sparse_operator_gap"
        elif (mean_params is not None and mean_params > 0
              and mean_loss is not None and mean_loss < 0.60):
            # Good quality but check FLOP efficiency
            baseline_params = 6 * 256 ** 2  # ~393K for a minimal 2-layer transformer
            if mean_params > 3 * baseline_params:
                intent = "compression"
                rationale = "low_flop_efficiency"

        recommendation = {
            "intent": intent,
            "rationale": rationale,
            "evidence": {
                "source_count": len(source_rows),
                "mean_loss_ratio": mean_loss,
                "mean_novelty": mean_novelty,
                "mean_params": mean_params,
                "mean_op_success": mean_op_success,
                "mean_sparse_op_ratio": mean_sparse_ratio,
            },
        }
        return intent, recommendation

    def _recent_synthesis_health(self, nb: LabNotebook, window: int = 5) -> Dict[str, float]:
        """Summarize recent synthesis outcomes for fallback decisions."""
        experiments = nb.get_recent_experiments(max(window * 3, window))
        rows = [
            row for row in experiments
            if str(row.get("experiment_type") or "") == "synthesis"
            and str(row.get("status") or "") == "completed"
        ][:window]
        total_programs = sum(max(int(r.get("n_programs_generated") or 0), 0) for r in rows)
        total_s1 = sum(max(int(r.get("n_stage1_passed") or 0), 0) for r in rows)
        rate = (float(total_s1) / float(total_programs)) if total_programs > 0 else 0.0
        return {
            "window": float(len(rows)),
            "total_programs": float(total_programs),
            "total_s1": float(total_s1),
            "s1_rate": float(rate),
        }

    @staticmethod
    def _refinement_candidate_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        """Approximate distance between two candidate programs for diversity gating."""
        loss_a = float(a.get("loss_ratio") or 1.0)
        loss_b = float(b.get("loss_ratio") or 1.0)
        nov_a = float(a.get("novelty_score") or 0.0)
        nov_b = float(b.get("novelty_score") or 0.0)
        ops_a = float(a.get("graph_n_ops") or 0.0)
        ops_b = float(b.get("graph_n_ops") or 0.0)
        fp_a = str(a.get("graph_fingerprint") or "")
        fp_b = str(b.get("graph_fingerprint") or "")
        fp_term = 0.0 if fp_a[:8] == fp_b[:8] and fp_a and fp_b else 0.1
        return abs(loss_a - loss_b) + abs(nov_a - nov_b) + (abs(ops_a - ops_b) / 16.0) + fp_term

    def _select_diverse_refinement_sources(
        self,
        candidates: List[Dict[str, Any]],
        *,
        top_k: int,
        min_distance: float,
        novelty_pressure: float,
    ) -> List[Dict[str, Any]]:
        """Select top-k candidates while preserving pairwise diversity."""
        if not candidates:
            return []
        ranked = []
        for row in candidates:
            loss = float(row.get("loss_ratio") or 1.0)
            novelty = float(row.get("novelty_score") or 0.0)
            quality = max(0.0, 1.0 - min(loss, 1.5))
            score = (1.0 - novelty_pressure) * quality + novelty_pressure * novelty
            ranked.append((score, row))
        ranked.sort(key=lambda x: x[0], reverse=True)

        selected: List[Dict[str, Any]] = []
        for _, row in ranked:
            if any(self._refinement_candidate_distance(row, prev) < min_distance for prev in selected):
                continue
            selected.append(row)
            if len(selected) >= top_k:
                break
        if len(selected) < top_k:
            for _, row in ranked:
                if row in selected:
                    continue
                selected.append(row)
                if len(selected) >= top_k:
                    break
        return selected

    def _build_refinement_plan(
        self,
        nb: LabNotebook,
        config: RunConfig,
    ) -> Optional[Dict[str, Any]]:
        """Build a recursive refinement plan from recent Stage-1 survivors."""
        lookback = max(1, int(config.refinement_lookback_experiments or 1))
        recent = nb.get_recent_experiments(max(lookback * 3, lookback))
        recent_ids = [
            str(row.get("experiment_id") or "")
            for row in recent
            if str(row.get("experiment_id") or "")
        ][:lookback]
        if not recent_ids:
            return None
        if not hasattr(nb, "conn"):
            return None

        placeholders = ",".join(["?"] * len(recent_ids))
        rows = nb.conn.execute(
            f"""SELECT result_id, experiment_id, graph_fingerprint, loss_ratio, novelty_score,
                       stage1_passed, graph_n_ops, timestamp
                FROM program_results
                WHERE stage1_passed = 1
                  AND experiment_id IN ({placeholders})
                ORDER BY loss_ratio ASC NULLS LAST, novelty_score DESC NULLS LAST, timestamp DESC, result_id ASC
                LIMIT ?""",
            [*recent_ids, max(20, int(config.refinement_top_k) * 10)],
        ).fetchall()
        candidates = [dict(r) for r in rows]
        if len(candidates) < max(1, int(config.refinement_min_stage1_survivors or 1)):
            return None

        selected = self._select_diverse_refinement_sources(
            candidates,
            top_k=max(1, int(config.refinement_top_k or 1)),
            min_distance=max(0.01, float(config.refinement_min_distance or 0.01)),
            novelty_pressure=max(0.0, min(1.0, float(config.refinement_novelty_pressure or 0.0))),
        )
        source_ids = [str(row.get("result_id") or "") for row in selected if row.get("result_id")]
        if not source_ids:
            return None

        radius = max(0.05, min(1.0, float(config.refinement_mutation_radius or 0.35)))
        mutation_rate = max(0.10, min(0.95, float(config.mutation_rate) * (0.5 + radius)))
        generations = max(1, int(config.refinement_generations or 1))
        budget_programs = max(int(config.n_programs), int(config.refinement_budget_programs or config.n_programs))
        per_gen = max(4, min(int(config.n_programs), max(4, budget_programs // generations)))
        mutations_per_source = max(1, int(round(2 + 4 * radius)))
        pool_multiplier = max(2, int(round(2 + 3 * float(config.refinement_novelty_pressure or 0.0))))

        return {
            "source_result_ids": source_ids,
            "source_count": len(source_ids),
            "generations": generations,
            "budget_programs": budget_programs,
            "config": {
                "model_source": "fingerprint_refine",
                "refine_source_result_ids": ",".join(source_ids),
                "refine_mutations_per_source": mutations_per_source,
                "refine_pool_multiplier": pool_multiplier,
                "mutation_rate": mutation_rate,
                "n_programs": per_gen,
                "refinement_top_k": int(config.refinement_top_k),
                "refinement_generations": generations,
                "refinement_budget_programs": budget_programs,
                "refinement_plateau_patience": int(config.refinement_plateau_patience),
                "refinement_min_distance": float(config.refinement_min_distance),
                "refinement_novelty_pressure": float(config.refinement_novelty_pressure),
            },
        }

    def _build_next_experiment_summary(
        self,
        nb: LabNotebook,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compact summary payload for LLM next-step planning."""
        recent = nb.get_recent_experiments(8)
        recent_exp_id = str(results.get("experiment_id") or "")
        if not recent_exp_id and recent:
            recent_exp_id = str(recent[0].get("experiment_id") or "")

        stage1_rows: List[Dict[str, Any]] = []
        fail_counts: Dict[str, int] = {}
        if recent_exp_id:
            rows = nb.get_program_results(recent_exp_id, limit=300)
            for row in rows:
                stage = str(row.get("stage_at_death") or "unknown")
                fail_counts[stage] = fail_counts.get(stage, 0) + 1
                if row.get("stage1_passed"):
                    stage1_rows.append(row)
        stage1_rows.sort(
            key=lambda r: (
                float(r.get("loss_ratio") if r.get("loss_ratio") is not None else 1.0),
                -float(r.get("novelty_score") if r.get("novelty_score") is not None else 0.0),
            )
        )
        top = [{
            "result_id": row.get("result_id"),
            "fingerprint": str(row.get("graph_fingerprint") or "")[:16],
            "loss_ratio": row.get("loss_ratio"),
            "novelty_score": row.get("novelty_score"),
            "throughput_tok_s": row.get("throughput_tok_s"),
            "avg_step_time_ms": row.get("avg_step_time_ms"),
            "stability_score": row.get("stability_score"),
        } for row in stage1_rows[:5]]

        eff_rows = [r for r in stage1_rows if isinstance(r.get("throughput_tok_s"), (int, float))]
        stab_rows = [r for r in stage1_rows if isinstance(r.get("stability_score"), (int, float))]
        avg_tp = (sum(float(r.get("throughput_tok_s")) for r in eff_rows) / len(eff_rows)) if eff_rows else None
        avg_stability = (sum(float(r.get("stability_score")) for r in stab_rows) / len(stab_rows)) if stab_rows else None
        novelty_vals = [float(r.get("novelty_score")) for r in stage1_rows if isinstance(r.get("novelty_score"), (int, float))]
        best_loss = min((float(r.get("loss_ratio")) for r in stage1_rows if isinstance(r.get("loss_ratio"), (int, float))), default=None)

        return {
            "recent_experiment_id": recent_exp_id or None,
            "funnel": {
                "total": int(results.get("total") or 0),
                "stage0_passed": int(results.get("stage0_passed") or 0),
                "stage05_passed": int(results.get("stage05_passed") or 0),
                "stage1_passed": int(results.get("stage1_passed") or 0),
            },
            "stage1_survivors": int(len(stage1_rows)),
            "best_loss_ratio": best_loss,
            "best_novelty": max(novelty_vals) if novelty_vals else None,
            "avg_novelty": (sum(novelty_vals) / len(novelty_vals)) if novelty_vals else None,
            "avg_throughput_tok_s": avg_tp,
            "avg_stability_score": avg_stability,
            "top_performers": top,
            "failure_breakdown": fail_counts,
            "recent_experiments": [
                {
                    "experiment_id": str(r.get("experiment_id") or "")[:12],
                    "type": r.get("experiment_type"),
                    "status": r.get("status"),
                    "stage1_passed": int(r.get("n_stage1_passed") or 0),
                    "best_loss_ratio": r.get("best_loss_ratio"),
                    "best_novelty_score": r.get("best_novelty_score"),
                }
                for r in recent[:6]
            ],
        }

    def _refinement_intent_spec(self, intent: str) -> Dict[str, Any]:
        """Canonical intent weighting description used in refinement hypotheses."""
        mode = str(intent or "balanced").lower()
        specs: Dict[str, Dict[str, Any]] = {
            "quality": {
                "name": "quality",
                "weights": {
                    "learned_quality": 0.60,
                    "parent_quality": 0.25,
                    "compression_proxy": 0.15,
                },
                "formula": "0.60*learned_quality + 0.25*parent_quality + 0.15*compression_proxy",
            },
            "compression": {
                "name": "compression",
                "weights": {
                    "compression_proxy": 0.60,
                    "learned_quality": 0.25,
                    "parent_quality": 0.15,
                },
                "formula": "0.60*compression_proxy + 0.25*learned_quality + 0.15*parent_quality",
            },
            "sparsity": {
                "name": "sparsity",
                "weights": {
                    "sparsity_proxy": 0.60,
                    "learned_quality": 0.25,
                    "compression_proxy": 0.15,
                },
                "formula": "0.60*sparsity_proxy + 0.25*learned_quality + 0.15*compression_proxy",
            },
            "novelty": {
                "name": "novelty",
                "weights": {
                    "novelty_proxy": 0.55,
                    "learned_quality": 0.25,
                    "parent_novelty": 0.20,
                },
                "formula": "0.55*novelty_proxy + 0.25*learned_quality + 0.20*parent_novelty",
            },
            "balanced": {
                "name": "balanced",
                "weights": {
                    "learned_quality": 0.35,
                    "compression_proxy": 0.25,
                    "novelty_proxy": 0.20,
                    "parent_signal": 0.20,
                },
                "formula": "0.35*learned_quality + 0.25*compression_proxy + 0.20*novelty_proxy + 0.20*parent_signal",
            },
        }
        return specs.get(mode, specs["balanced"])

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
            
            # Z17: Clear global native-runner counters immediately on stop
            reset_native_runner_telemetry()
            
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
        with self._lock:
            # Z17: Clear any stale progress data from previous runs
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                aria_message=f"{self.aria.NAME}: Starting experiment {exp_id[:8]}...",
            )
            
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

            # Update op success rates and failure signatures after experiment
            nb.update_op_success_rates(exp_id)
            s0_op_counts = results.pop("_s0_op_counts", None)
            if s0_op_counts:
                nb.merge_op_failure_counts(s0_op_counts)
            nb.strip_graph_json_for_failures(exp_id)
            nb.update_failure_signatures(exp_id)

            # Save effective weights + S1 outcome for EMA continuity
            applied_w = results.get("applied_grammar_weights")
            total = results.get("total", 0)
            if applied_w and total > 0:
                s1_rate = results.get("stage1_passed", 0) / total
                nb.save_effective_weights(applied_w, s1_rate, exp_id)

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
            logger.error("Experiment failed (%s): %s\n%s", exp_id, e, error)
            self._invoke_code_healer(
                nb=nb,
                trigger_type="repeated_exception",
                experiment_id=exp_id,
                scope=f"Synthesis/experiment failure: {str(e)[:240]}",
                reproduction_steps=["python -m pytest tests/test_integration.py -k \"start_experiment\" -x --tb=short"],
                acceptance_tests=["python -m pytest tests/test_integration.py -k \"start_experiment\" -x --tb=short"],
                trigger_payload={"mode": "synthesis", "error": str(e)},
            )
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

    _PLATEAU_WINDOW = 5  # cycles to check for progress
    _PLATEAU_MIN_CYCLES = 8  # don't trigger before this many cycles

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

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

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

    def _candidate_tokens(self, row: Dict[str, Any]) -> Set[str]:
        """Extract lightweight semantic tokens for insight matching."""
        tokens: Set[str] = set()
        family = LabNotebook._classify_architecture_family(
            row.get("graph_json"),
            row.get("routing_mode"),
        )
        if family:
            tokens.update(part for part in family.lower().replace("-", " ").split() if len(part) >= 3)

        graph_json = row.get("graph_json")
        if isinstance(graph_json, str) and graph_json:
            try:
                graph_data = json.loads(graph_json)
                nodes = graph_data.get("nodes", {}) if isinstance(graph_data, dict) else {}
                for nd in nodes.values():
                    if not isinstance(nd, dict):
                        continue
                    op_name = str(nd.get("op_name") or "").strip().lower()
                    if not op_name or op_name == "input":
                        continue
                    tokens.add(op_name)
                    tokens.update(part for part in op_name.replace("-", "_").split("_") if len(part) >= 3)
            except Exception:
                pass
        return tokens

    def _resolve_pending_selection_insight_trials(self, nb: LabNotebook) -> None:
        """Resolve pending insight-bundle trials once downstream outcomes are available."""
        try:
            trials = nb.get_pending_selection_insight_trials(limit=200)
        except Exception:
            return
        if not trials:
            return

        leaderboard = nb.get_leaderboard(limit=2000)
        by_result = {
            str(row.get("result_id")): row for row in leaderboard if row.get("result_id")
        }
        for trial in trials:
            context = str(trial.get("context") or "")
            chosen_ids = trial.get("chosen_result_ids_json") or []
            if not isinstance(chosen_ids, list) or not chosen_ids:
                continue
            entries = [by_result.get(str(rid)) for rid in chosen_ids]
            if any(entry is None for entry in entries):
                continue

            rewards: List[float] = []
            resolved = False
            for entry in entries:
                if context == "auto_investigate_screening":
                    inv_pass = entry.get("investigation_passed")
                    inv_loss = entry.get("investigation_loss_ratio")
                    inv_rob = entry.get("investigation_robustness")
                    if inv_pass is None and inv_loss is None:
                        resolved = False
                        rewards = []
                        break
                    passed = 1.0 if bool(inv_pass) else 0.0
                    loss_term = max(0.0, 1.0 - self._to_float(inv_loss, default=1.0))
                    rob_term = max(0.0, min(1.0, self._to_float(inv_rob, default=0.0)))
                    rewards.append(max(0.0, min(1.0, 0.5 * passed + 0.3 * loss_term + 0.2 * rob_term)))
                    resolved = True
                elif context == "auto_validate_investigation":
                    val_pass = entry.get("validation_passed")
                    val_loss = entry.get("validation_loss_ratio")
                    val_base = entry.get("validation_baseline_ratio")
                    val_std = self._to_float(entry.get("validation_multi_seed_std"), default=0.2)
                    if val_pass is None and val_loss is None and val_base is None:
                        resolved = False
                        rewards = []
                        break
                    passed = 1.0 if bool(val_pass) else 0.0
                    if val_base is not None:
                        loss_term = max(0.0, 1.0 - self._to_float(val_base, default=1.0))
                    else:
                        loss_term = max(0.0, 1.0 - self._to_float(val_loss, default=1.0))
                    std_term = max(0.0, min(1.0, 1.0 - val_std))
                    rewards.append(max(0.0, min(1.0, 0.5 * passed + 0.3 * loss_term + 0.2 * std_term)))
                    resolved = True
                else:
                    rewards = []
                    resolved = False
                    break

            if not resolved or not rewards:
                continue

            reward = float(sum(rewards) / len(rewards))
            if reward >= 0.55:
                outcome = "supported"
            elif reward <= 0.45:
                outcome = "not_supported"
            else:
                outcome = "inconclusive"
            nb.resolve_selection_insight_trial(
                trial_id=str(trial.get("trial_id")),
                reward=reward,
                outcome=outcome,
                metadata={
                    "context": context,
                    "n_candidates": len(chosen_ids),
                    "resolved_from": "leaderboard",
                },
            )

    def _selection_supporting_insights(
        self,
        nb: LabNotebook,
        candidates: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, List[str]], List[str]]:
        """Match active insights to candidates and return per-result + global bundles."""
        insights = nb.get_insights(limit=120)
        if not insights:
            return {}, []

        by_result: Dict[str, List[str]] = {}
        global_scores: Dict[str, float] = {}
        for row in candidates:
            rid = str(row.get("result_id") or "")
            if not rid:
                continue
            tokens = self._candidate_tokens(row)
            if not tokens:
                continue
            scored: List[Tuple[float, str]] = []
            for insight in insights:
                insight_id = str(insight.get("insight_id") or "")
                content = str(insight.get("content") or "").lower()
                if not insight_id or not content:
                    continue
                hit_count = sum(1 for token in tokens if token in content)
                if hit_count <= 0:
                    continue
                confidence = self._to_float(insight.get("confidence"), default=0.5)
                score = hit_count * max(0.1, confidence)
                scored.append((score, insight_id))
            if not scored:
                continue
            scored.sort(key=lambda item: item[0], reverse=True)
            matched = [insight_id for _, insight_id in scored[:3]]
            by_result[rid] = matched
            for score, insight_id in scored[:5]:
                global_scores[insight_id] = max(global_scores.get(insight_id, 0.0), score)

        global_ids = [iid for iid, _ in sorted(global_scores.items(), key=lambda kv: kv[1], reverse=True)[:6]]
        return by_result, global_ids

    def _selection_safety_valve(self, nb: LabNotebook, config: RunConfig) -> Optional[Dict[str, Any]]:
        """Trigger novelty/ablation-heavy fallback after repeated stagnation."""
        window = max(3, int(config.safety_plateau_window))
        recent = nb.get_recent_experiments(window)
        if len(recent) < window:
            return None
        ordered = list(reversed(recent))
        loss_vals = [self._to_float(e.get("best_loss_ratio"), default=float("nan")) for e in ordered]
        loss_vals = [v for v in loss_vals if not math.isnan(v)]
        n_stage1 = [int(e.get("n_stage1_passed") or 0) for e in ordered]
        if not loss_vals:
            return None

        first_loss = loss_vals[0]
        best_recent = min(loss_vals)
        loss_gain = max(0.0, first_loss - best_recent)
        no_survivor_progress = all(v <= 0 for v in n_stage1)
        plateau = loss_gain < float(config.safety_plateau_min_delta) and no_survivor_progress
        if not plateau:
            return None

        with self._lock:
            recent_modes = [c.get("mode", "synthesis") for c in self._aria_cycle_history[-window:]]
        novelty_share = (
            sum(1 for m in recent_modes if m == "novelty") / max(1, len(recent_modes))
        )
        mode = "ablation_heavy" if novelty_share >= 0.5 else "novelty"
        return {
            "triggered": True,
            "mode": mode,
            "window": window,
            "loss_gain": round(loss_gain, 6),
            "min_required_gain": float(config.safety_plateau_min_delta),
            "no_survivor_progress": no_survivor_progress,
            "reason": (
                f"No measurable progress over {window} experiments "
                f"(loss gain={loss_gain:.4f}, S1 survivors unchanged)."
            ),
        }

    def _ensure_novelty_calibration(
        self,
        nb: LabNotebook,
        config: RunConfig,
        fp: Optional[Any],
    ) -> Optional[Dict[str, Any]]:
        """Fetch or create baseline novelty calibration for the active reference version."""
        if fp is None:
            return None
        reference_version = getattr(fp, "novelty_reference_version", None)
        if not reference_version:
            return None
        row = nb.get_latest_novelty_calibration(reference_version=reference_version)
        if row is not None:
            return row
        if not config.auto_novelty_calibration:
            return None

        try:
            from ..eval.novelty_calibration import calibrate_baseline_transformer_novelty
            calibration = calibrate_baseline_transformer_novelty(
                n_runs=max(2, int(config.novelty_calibration_runs)),
                seq_len=min(32, int(config.max_seq_len)),
                model_dim=max(16, int(config.model_dim)),
                vocab_size=max(256, min(4096, int(config.vocab_size))),
                device="cpu",
                seed=self._stable_seed("novelty_calibration", reference_version),
            )
            nb.record_novelty_calibration(
                reference_version=calibration.get("reference_version") or reference_version,
                cka_source=calibration.get("cka_source"),
                cka_artifact_version=calibration.get("cka_artifact_version"),
                probe_protocol_hash=calibration.get("probe_protocol_hash"),
                n_runs=int(calibration.get("n_runs") or config.novelty_calibration_runs),
                noise_floor_mean=calibration.get("noise_floor_mean"),
                noise_floor_std=calibration.get("noise_floor_std"),
                confidence_low=calibration.get("confidence_low"),
                confidence_high=calibration.get("confidence_high"),
                distribution=calibration.get("distribution") or {},
                metadata=calibration.get("metadata") or {},
            )
            return nb.get_latest_novelty_calibration(reference_version=reference_version)
        except Exception as e:
            logger.debug("Novelty calibration failed for %s: %s", reference_version, e)
            return None

    def _resolve_novelty_promotion_validity(
        self,
        config: RunConfig,
        valid_for_promotion: bool,
        reason: str,
    ) -> Tuple[bool, str, bool]:
        """Apply explicit override policy for heuristic novelty promotions."""
        valid = bool(valid_for_promotion)
        resolved_reason = str(reason or "unknown")
        requires_justification = not valid
        if valid:
            return True, resolved_reason, False
        if config.allow_heuristic_novelty_promotion and str(config.heuristic_novelty_justification or "").strip():
            return True, f"override:{resolved_reason}", True
        return False, resolved_reason, requires_justification

    @staticmethod
    def _safe_build_evidence_pack(
        nb: LabNotebook,
        recommendation: Dict[str, Any],
        decision_type: str,
    ) -> Dict[str, Any]:
        try:
            return build_evidence_pack(
                nb,
                analytics=None,
                recommendation=recommendation,
                decision_type=decision_type,
            )
        except Exception:
            return {
                "hypothesis": "Insufficient metrics; gather more evidence before confident action.",
                "supporting_metrics": [{
                    "name": "evidence_unavailable",
                    "value": 0.0,
                    "baseline": 0.0,
                    "delta_vs_baseline": 0.0,
                }],
                "uncertainty": {"note": "Evidence pack fallback due to sparse metrics."},
                "confounders": ["Sparse or missing recent experiment metrics."],
                "falsification": ["If next experiment still yields sparse metrics, block automation."],
            }

    def _build_default_preregistration(
        self,
        experiment_type: str,
        config: Dict[str, Any],
        hypothesis: Optional[str],
        exploratory: bool = False,
    ) -> Dict[str, Any]:
        statement = str(hypothesis or f"{experiment_type} batch will improve prioritized objectives.")
        primary_metrics = ["loss_ratio", "stage1_passed"]
        if experiment_type in {"novelty", "evolution"}:
            primary_metrics = ["novelty_score", "stage1_passed"]
        if experiment_type in {"validation", "scale_up"}:
            primary_metrics = ["baseline_loss_ratio", "loss_ratio", "novelty_confidence"]

        prereg = HypothesisPreregistration(
            hypothesis={
                "statement": statement,
                "variables": {
                    "independent": ["architecture_family", "op_composition", "training_recipe"],
                    "dependent": primary_metrics + ["throughput_tok_s", "stability_score"],
                    "controls": ["model_dim", "n_layers", "stage1_steps", "batch_size"],
                },
                "expected_direction": {
                    "loss_ratio": "decrease",
                    "novelty_score": "increase",
                    "throughput_tok_s": "increase",
                    "stability_score": "increase",
                },
                "success_criteria": {
                    "stage1_passed_min": 1,
                    "best_loss_ratio_max": 0.95,
                    "novelty_confidence_min": 0.5,
                },
            },
            analysis_plan={
                "primary_metrics": primary_metrics,
                "secondary_metrics": [
                    "compile_time_ms",
                    "grad_norm_std",
                    "throughput_tok_s",
                    "flops_per_token",
                    "novelty_confidence",
                ],
                "thresholds": {
                    "loss_ratio": {"operator": "<", "value": 1.0},
                    "novelty_confidence": {"operator": ">=", "value": 0.5},
                    "stability_score": {"operator": ">=", "value": 0.5},
                },
                "baseline_comparison": {
                    "method": "relative_loss_ratio",
                    "source": "TransformerBaseline.compare",
                    "delta_operator": "<",
                    "delta_value": 1.0,
                },
            },
            falsification_conditions=[
                "No candidate passes Stage1.",
                "Best loss_ratio does not beat baseline threshold.",
                "Novelty only appears with heuristic fallback and no justification.",
            ],
            confounders_checklist=[
                {"name": "unstable_seed_behavior", "checked": False},
                {"name": "fallback_novelty_mode", "checked": False},
                {"name": "noisy_throughput", "checked": False},
                {"name": "compile_instability", "checked": False},
            ],
            exploratory=exploratory,
        ).to_dict()
        prereg["analysis_plan"]["config_snapshot"] = {
            "n_programs": config.get("n_programs"),
            "stage1_steps": config.get("stage1_steps"),
            "model_dim": config.get("model_dim"),
            "n_layers": config.get("n_layers"),
        }
        return prereg

    def _ensure_preregistration(
        self,
        nb: LabNotebook,
        experiment_type: str,
        config: Dict[str, Any],
        hypothesis: Optional[str],
        preregistration: Optional[Dict[str, Any]] = None,
        exploratory: bool = False,
        created_by: str = "runner",
    ) -> str:
        require_prereg = bool(config.get("require_preregistration", True))
        auto_preregister = bool(config.get("auto_preregister", True))
        payload = preregistration
        if payload is None and auto_preregister:
            payload = self._build_default_preregistration(
                experiment_type=experiment_type,
                config=config,
                hypothesis=hypothesis,
                exploratory=exploratory,
            )
        if require_prereg and payload is None:
            raise PreregistrationError(
                "Experiment blocked: preregistration required but missing."
            )
        if payload is None:
            raise PreregistrationError(
                "Experiment blocked: preregistration payload unavailable."
            )
        validate_preregistration(payload)
        return nb.create_preregistration(
            experiment_type=experiment_type,
            preregistration=payload,
            created_by=created_by,
        )

    def _start_preregistered_experiment(
        self,
        nb: LabNotebook,
        experiment_type: str,
        config: Dict[str, Any],
        hypothesis: Optional[str] = None,
        research_question: Optional[str] = None,
        hypothesis_metadata: Optional[Dict[str, Any]] = None,
        preregistration: Optional[Dict[str, Any]] = None,
        exploratory: bool = False,
        created_by: str = "runner",
    ) -> str:
        prereg_id = self._ensure_preregistration(
            nb=nb,
            experiment_type=experiment_type,
            config=config,
            hypothesis=hypothesis,
            preregistration=preregistration,
            exploratory=exploratory,
            created_by=created_by,
        )
        meta = dict(hypothesis_metadata or {})
        meta["preregistration_id"] = prereg_id
        
        # Z17: Reset global native-runner counters between experiments
        reset_native_runner_telemetry()
        
        return nb.start_experiment(
            experiment_type=experiment_type,
            config=config,
            hypothesis=hypothesis,
            research_question=research_question,
            hypothesis_metadata=meta,
            preregistration_id=prereg_id,
            require_preregistration=bool(config.get("require_preregistration", True)),
        )

    def _score_candidate_pool(
        self,
        candidates: List[Dict[str, Any]],
        config: RunConfig,
        nb: LabNotebook,
        context: str,
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Multi-objective scoring + family uncertainty policy for candidate selection."""
        if not candidates:
            return {
                "summary": {"candidate_count": 0},
                "scored": [],
                "selected": [],
                "reason": "No candidates available.",
                "policy": {"name": config.selection_policy, "exploration": False},
            }

        self._resolve_pending_selection_insight_trials(nb)

        weights = {
            "quality": float(config.selection_quality_weight),
            "novelty": float(config.selection_novelty_weight),
            "efficiency": float(config.selection_efficiency_weight),
            "feasibility": float(config.selection_feasibility_weight),
        }
        weight_sum = sum(max(0.0, w) for w in weights.values()) or 1.0
        for k in list(weights):
            weights[k] = max(0.0, weights[k]) / weight_sum

        quality_raw: Dict[str, float] = {}
        novelty_raw: Dict[str, float] = {}
        efficiency_raw: Dict[str, float] = {}
        feasibility_raw: Dict[str, float] = {}
        families: Dict[str, str] = {}
        by_id: Dict[str, Dict[str, Any]] = {}

        for row in candidates:
            rid = str(row.get("result_id") or "")
            if not rid:
                continue
            by_id[rid] = row
            family = LabNotebook._classify_architecture_family(
                row.get("graph_json"),
                row.get("routing_mode"),
            )
            families[rid] = family

            loss_ratio = self._to_float(row.get("loss_ratio"), default=1.0)
            baseline_ratio = self._to_float(row.get("baseline_loss_ratio"), default=1.0)
            quality_raw[rid] = max(0.0, (1.0 - loss_ratio) + max(0.0, 1.0 - baseline_ratio))

            novelty_raw[rid] = self._to_float(row.get("novelty_score"), default=0.0)

            throughput = self._to_float(row.get("throughput_tok_s"), default=0.0)
            flops = self._to_float(row.get("flops_per_token"), default=0.0)
            mem = self._to_float(row.get("peak_memory_mb"), default=0.0)
            
            # Baseline targets from research
            is_efficient_arch = family.startswith("MoE-") or family.startswith("Adaptive-") or "Mamba" in family
            target_throughput = 10000.0 if is_efficient_arch else 5000.0
            throughput_bonus = max(0.0, throughput / target_throughput)
            
            efficiency_raw[rid] = (throughput_bonus * 5.0) - (0.35 * flops) - (0.15 * mem)
            
            # Add adaptive savings bonus if available
            savings = self._to_float(row.get("depth_savings_ratio"), default=0.0)
            if savings > 0:
                efficiency_raw[rid] += savings * 10.0

            stage0 = 1.0 if int(row.get("stage0_passed") or 0) == 1 else 0.0
            stage05 = 1.0 if int(row.get("stage05_passed") or 0) == 1 else 0.0
            stage1 = 1.0 if int(row.get("stage1_passed") or 0) == 1 else 0.0
            stability = self._to_float(row.get("stability_score"), default=0.0)
            grad_penalty = 0.0
            if int(row.get("has_nan_grad") or 0) == 1:
                grad_penalty += 0.5
            if int(row.get("has_zero_grad") or 0) == 1:
                grad_penalty += 0.3
            feasibility_raw[rid] = max(0.0, (0.2 * stage0 + 0.2 * stage05 + 0.3 * stage1 + 0.3 * stability) - grad_penalty)

        qn = self._norm_map(quality_raw, higher_is_better=True)
        nn = self._norm_map(novelty_raw, higher_is_better=True)
        en = self._norm_map(efficiency_raw, higher_is_better=True)
        fn = self._norm_map(feasibility_raw, higher_is_better=True)

        family_stats = nb.get_selection_family_stats()
        total_trials = sum(int(s.get("n_trials") or 0) for s in family_stats.values())
        family_bonus_raw: Dict[str, float] = {}
        family_uncertainty: Dict[str, float] = {}
        for rid, fam in families.items():
            stat = family_stats.get(fam, {})
            n_trials = int(stat.get("n_trials") or 0)
            mean_reward = self._to_float(stat.get("mean_reward"), default=0.0)
            uncertainty = 1.0 / math.sqrt(n_trials + 1.0)
            ucb = mean_reward + float(config.selection_ucb_c) * math.sqrt(
                math.log(max(total_trials, 1) + 1.0) / (n_trials + 1.0)
            )
            family_uncertainty[rid] = uncertainty
            family_bonus_raw[rid] = uncertainty if config.selection_policy == "epsilon_greedy" else ucb
        family_bonus = self._norm_map(family_bonus_raw, higher_is_better=True)
        unc_norm = self._norm_map(family_uncertainty, higher_is_better=True)
        insight_by_result, supporting_insight_ids = self._selection_supporting_insights(nb, candidates)
        interaction_rows = nb.get_selection_insight_interactions(limit=500)
        interaction_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in interaction_rows:
            key = (
                str(row.get("insight_a") or ""),
                str(row.get("insight_b") or ""),
            )
            interaction_map[key] = row
        insight_interaction_raw: Dict[str, float] = {}
        for rid in by_id:
            matched = insight_by_result.get(rid) or []
            rewards: List[float] = []
            for insight_id in matched:
                stat = interaction_map.get((insight_id, insight_id))
                if stat and int(stat.get("n_trials") or 0) >= 2:
                    rewards.append(self._to_float(stat.get("mean_reward"), default=0.5))
            for i in range(len(matched)):
                for j in range(i + 1, len(matched)):
                    a, b = matched[i], matched[j]
                    if a > b:
                        a, b = b, a
                    stat = interaction_map.get((a, b))
                    if stat and int(stat.get("n_trials") or 0) >= 2:
                        rewards.append(self._to_float(stat.get("mean_reward"), default=0.5))
            if rewards:
                insight_interaction_raw[rid] = float(sum(rewards) / len(rewards))
            else:
                insight_interaction_raw[rid] = 0.5
        insight_interaction = self._norm_map(insight_interaction_raw, higher_is_better=True)

        scored: List[Dict[str, Any]] = []
        for rid in by_id:
            base_score = (
                weights["quality"] * qn.get(rid, 0.0)
                + weights["novelty"] * nn.get(rid, 0.0)
                + weights["efficiency"] * en.get(rid, 0.0)
                + weights["feasibility"] * fn.get(rid, 0.0)
            )
            bonus = family_bonus.get(rid, 0.0)
            total = (1.0 - float(config.selection_family_bonus_weight)) * base_score + (
                float(config.selection_family_bonus_weight) * bonus
            )
            # Small additive term to prefer insight bundles with positive historical interactions.
            interaction_term = (insight_interaction.get(rid, 0.5) - 0.5) * 0.12
            total += interaction_term
            scored.append({
                "result_id": rid,
                "family": families.get(rid, "Unknown"),
                "score": round(total, 6),
                "base_score": round(base_score, 6),
                "components": {
                    "quality": round(qn.get(rid, 0.0), 6),
                    "novelty": round(nn.get(rid, 0.0), 6),
                    "efficiency": round(en.get(rid, 0.0), 6),
                    "feasibility": round(fn.get(rid, 0.0), 6),
                    "insight_interaction": round(insight_interaction.get(rid, 0.5), 6),
                },
                "family_bonus": round(bonus, 6),
                "family_uncertainty": round(unc_norm.get(rid, 0.0), 6),
                "supporting_insight_ids": insight_by_result.get(rid, []),
                "raw": {
                    "loss_ratio": self._to_float(by_id[rid].get("loss_ratio"), default=1.0),
                    "baseline_loss_ratio": self._to_float(by_id[rid].get("baseline_loss_ratio"), default=1.0),
                    "novelty_score": self._to_float(by_id[rid].get("novelty_score"), default=0.0),
                    "throughput_tok_s": self._to_float(by_id[rid].get("throughput_tok_s"), default=0.0),
                    "flops_per_token": self._to_float(by_id[rid].get("flops_per_token"), default=0.0),
                    "peak_memory_mb": self._to_float(by_id[rid].get("peak_memory_mb"), default=0.0),
                    "stability_score": self._to_float(by_id[rid].get("stability_score"), default=0.0),
                },
            })

        scored.sort(key=lambda x: (x["score"], x["base_score"], x["components"]["novelty"]), reverse=True)
        seed = self._stable_seed(context, experiment_id or "none", len(scored), total_trials)
        rng = random.Random(seed)
        exploration = rng.random() < max(0.0, min(1.0, float(config.selection_epsilon)))

        if exploration:
            ranked = sorted(
                scored,
                key=lambda x: (x["family_uncertainty"], x["components"]["novelty"], x["base_score"]),
                reverse=True,
            )
            reason = (
                f"Explore: epsilon trigger in {context}; prioritized high-uncertainty families."
            )
        else:
            ranked = scored
            reason = f"Exploit: selected highest evidence-weighted scores in {context}."

        summary = {
            "candidate_count": len(scored),
            "families": sorted({s["family"] for s in scored}),
            "weights": weights,
            "supporting_insight_ids": supporting_insight_ids,
            "policy": config.selection_policy,
            "epsilon": float(config.selection_epsilon),
            "ucb_c": float(config.selection_ucb_c),
            "exploration": exploration,
            "seed": seed,
        }
        policy = {
            "name": config.selection_policy,
            "exploration": exploration,
            "reason": reason,
            "family_stats": family_stats,
            "supporting_insight_ids": supporting_insight_ids,
        }
        return {
            "summary": summary,
            "scored": scored,
            "selected": ranked,
            "reason": reason,
            "policy": policy,
            "supporting_insight_ids": supporting_insight_ids,
            "supporting_insights_by_result": insight_by_result,
        }

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
            allowed_categories = {
                "principle",
                "anti_pattern",
                "sweet_spot",
                "correlation",
                "tool_insight",
            }

            def _normalize_category(raw: str) -> str:
                value = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
                aliases = {
                    "anti_pattern": "anti_pattern",
                    "anti_patterns": "anti_pattern",
                    "antipattern": "anti_pattern",
                    "anti-pattern": "anti_pattern",
                    "principles": "principle",
                    "sweetspot": "sweet_spot",
                    "sweet_spot": "sweet_spot",
                    "tool": "tool_insight",
                    "toolinsight": "tool_insight",
                    "tool_insights": "tool_insight",
                }
                value = aliases.get(value, value)
                return value if value in allowed_categories else "principle"

            def _canonical_text(raw: str) -> str:
                text = " ".join(str(raw or "").split()).strip().lower()
                text = re.sub(r"\b\d+(?:\.\d+)?%?\b", "#", text)
                text = re.sub(r"[^a-z0-9#\s]+", " ", text)
                return re.sub(r"\s+", " ", text).strip()

            stopwords = {
                "the", "and", "for", "that", "with", "this", "from", "into", "when", "then", "than", "were",
                "been", "have", "has", "had", "are", "was", "show", "shows", "showed", "over", "under",
                "across", "between", "using", "use", "used", "high", "low", "very", "more", "less", "near",
                "around", "recent", "experiments", "experiment", "result", "results", "indicate", "indicates",
                "suggest", "suggests", "mode", "patterns", "pattern", "architecture", "architectures",
            }

            def _tokenize_semantic(raw: str) -> Set[str]:
                canonical = _canonical_text(raw)
                return {
                    tok for tok in canonical.split()
                    if len(tok) > 3 and tok not in stopwords
                }

            def _is_semantic_duplicate(tokens: Set[str], existing_tokens: Set[str]) -> bool:
                if not tokens or not existing_tokens:
                    return False
                inter = len(tokens & existing_tokens)
                if inter < 5:
                    return False
                union = len(tokens | existing_tokens)
                return bool(union) and (inter / union) >= 0.18

            def _is_low_value_entry(title: str, content: str) -> bool:
                title_clean = " ".join(str(title or "").split()).strip()
                content_clean = " ".join(str(content or "").split()).strip()
                title_l = title_clean.lower()
                content_l = content_clean.lower()

                if len(title_clean) < 12 or len(content_clean) < 40:
                    return True
                if "..." in title_clean or "..." in content_clean:
                    return True
                if "1-2 sentences" in content_l or "i will now synthesize" in content_l:
                    return True
                if title_l.startswith("recent experiments show ") or title_l.startswith("all recent experiments show "):
                    return True
                if title_l.startswith("recent synthesis") and "failure" in title_l:
                    return True
                if "[principle/" in title_l or "hybrid? no" in title_l:
                    return True
                if "$" in content_clean or "\\approx" in content_l:
                    return True

                mechanism_tokens = (
                    "depth", "residual", "inverse", "log ", "frequency", "math_space",
                    "parameter", "parallel", "routing", "s1", "loss", "novelty", "baseline",
                )
                action_tokens = (
                    "improve", "improves", "degrade", "degrades", "fail", "fails", "underperform",
                    "correlate", "correlates", "correlation", "predict", "predicts",
                    "optimal", "requires", "avoid", "boost", "increase", "reduce",
                    "enhance", "enhances", "outperform", "outperforms", "suggests", "indicates",
                )
                has_mechanism = any(tok in content_l or tok in title_l for tok in mechanism_tokens)
                has_action = any(tok in content_l for tok in action_tokens)
                has_numeric = bool(re.search(r"\d", content_clean))
                return not (has_mechanism and (has_action or has_numeric))

            recent = nb.get_recent_experiments(config.knowledge_extraction_interval)
            resolved = []
            if self._active_campaign_id:
                all_hyps = nb.get_campaign_hypotheses(self._active_campaign_id)
                resolved = [h for h in all_hyps
                           if h.get("status") in ("confirmed", "refuted")]

            context = build_knowledge_extraction_context(recent, resolved)
            entries = self.aria.extract_knowledge(recent, resolved, context=context)

            existing_entries = nb.get_knowledge()
            existing_by_title: Dict[str, str] = {}
            existing_by_content: Dict[str, str] = {}
            existing_by_semantic: Dict[str, List[Tuple[str, Set[str]]]] = {}
            for row in existing_entries:
                eid = str(row.get("entry_id") or "")
                if not eid:
                    continue
                existing_by_title[_canonical_text(row.get("title") or "")] = eid
                existing_by_content[_canonical_text(row.get("content") or "")] = eid
                category = _normalize_category(str(row.get("category") or "principle"))
                tokens = _tokenize_semantic(f"{row.get('title') or ''} {row.get('content') or ''}")
                if tokens:
                    existing_by_semantic.setdefault(category, []).append((eid, tokens))

            accepted = 0
            skipped_low_value = 0
            deduped = 0

            for entry in entries:
                raw_title = str(entry.get("title") or "").strip()
                raw_content = str(entry.get("content") or "").strip()
                if _is_low_value_entry(raw_title, raw_content):
                    skipped_low_value += 1
                    continue

                category = _normalize_category(entry.get("category", "principle"))
                confidence = float(entry.get("confidence", 0.5) or 0.5)
                confidence = max(0.45, min(0.95, confidence))
                title = " ".join(raw_title.split())
                content = " ".join(raw_content.split())

                title_key = _canonical_text(title)
                content_key = _canonical_text(content)

                existing_entry_id = (
                    existing_by_title.get(title_key)
                    or existing_by_content.get(content_key)
                )
                if not existing_entry_id:
                    semantic_tokens = _tokenize_semantic(f"{title} {content}")
                    for eid, seen_tokens in existing_by_semantic.get(category, []):
                        if _is_semantic_duplicate(semantic_tokens, seen_tokens):
                            existing_entry_id = eid
                            break
                if existing_entry_id:
                    nb.validate_knowledge(existing_entry_id)
                    deduped += 1
                    continue

                evidence = [
                    str(e.get("experiment_id", "")).strip()
                    for e in recent[:5]
                    if str(e.get("experiment_id", "")).strip()
                ]
                new_entry_id = nb.add_knowledge(
                    category=category,
                    title=title,
                    content=content,
                    evidence=evidence,
                    confidence=confidence,
                )
                existing_by_title[title_key] = new_entry_id
                existing_by_content[content_key] = new_entry_id
                semantic_tokens = _tokenize_semantic(f"{title} {content}")
                if semantic_tokens:
                    existing_by_semantic.setdefault(category, []).append((new_entry_id, semantic_tokens))
                accepted += 1

            if entries:
                self._emit_event("knowledge_extracted", {
                    "n_entries": accepted,
                    "categories": list(set(e.get("category", "") for e in entries)),
                    "n_deduped": deduped,
                    "n_skipped_low_value": skipped_low_value,
                })
                logger.info(
                    "Knowledge extracted: accepted=%d deduped=%d skipped_low_value=%d raw=%d",
                    accepted, deduped, skipped_low_value, len(entries),
                )
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
        # Skip per-cycle LLM calls — use rule-based paths to save API costs.
        # LLM is still available for user-initiated chat and campaign formulation.
        self.aria._continuous_mode = True
        self.aria._llm_decision_interval = config.llm_decision_interval

        # Knowledge distiller — background intelligence thread
        distiller = None
        try:
            from .intelligence.distiller import KnowledgeDistiller
            from .intelligence.digest import ExperimentDigest
            db_path = self.notebook_path
            distiller = KnowledgeDistiller(
                db_path=db_path,
                distill_interval_cycles=3,
            )
            # Recover last digest from DB
            try:
                init_nb = self._make_notebook()
                saved = init_nb.get_latest_digest()
                if saved:
                    distiller.set_digest(ExperimentDigest.from_dict(saved))
                    logger.info("Recovered knowledge digest from DB")
                init_nb.close()
            except Exception as e:
                logger.debug("Digest recovery failed: %s", e)
            distiller.start()
            self._knowledge_distiller = distiller
        except Exception as e:
            logger.warning("KnowledgeDistiller init failed (degrading gracefully): %s", e)
            distiller = None
            self._knowledge_distiller = None
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

            # Check for pending heal retry
            if self._pending_heal_retry:
                retry = self._pending_heal_retry
                self._pending_heal_retry = None
                logger.info("Retrying after successful heal: %s", retry.get("scope", "")[:100])
                try:
                    retry_nb = self._make_notebook()
                    retry_nb.log_learning_event(
                        "heal_retry",
                        f"Retrying after heal: {retry['scope'][:200]}",
                    )
                    retry_nb.close()
                except Exception:
                    pass

            # Check limits before starting next experiment
            stop_reason = self._check_continuous_limits(
                config, t_start, n_experiments)
            if stop_reason:
                self.aria._continuous_mode = False
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
                # Stop knowledge distiller
                if distiller is not None:
                    try:
                        distiller.stop()
                    except Exception:
                        pass
                # Launch queued auto-scale-up
                self._run_pending_scale_up()
                return

            n_experiments += 1
            nb = self._make_notebook()
            try:
                self.run_aria_cycle(config, nb, n_experiments, t_start)
                
                # Periodic Gate Performance Summary (Task 9)
                if n_experiments % 5 == 0:
                    try:
                        from .analytics import ExperimentAnalytics
                        analytics = ExperimentAnalytics(nb)
                        stats = analytics.gate_performance_summary()
                        if stats:
                            logger.info(
                                "Gate Performance (Cycle %d): pass_rate=%.2f, violations=%d, corr=%s (n=%d)",
                                n_experiments, 
                                stats.get("stage05_pass_rate", 0),
                                stats.get("causality_violations", 0),
                                f"{stats.get('discovery_validation_correlation'):.2f}" if stats.get("discovery_validation_correlation") is not None else "N/A",
                                stats.get("n_correlation_samples", 0)
                            )
                    except Exception as e:
                        logger.debug("Failed to generate gate performance summary: %s", e)
            finally:
                nb.close()

            # Notify distiller that a cycle completed
            if distiller is not None:
                try:
                    distiller.notify_cycle_complete()
                except Exception:
                    pass

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

            # Purge empty failed experiments between cycles to prevent DB bloat.
            try:
                self.notebook.purge_empty_experiments()
                self.notebook.compact_old_chat()
                self.notebook.backfill_failure_signatures()
            except Exception:
                pass

            if config.rest_between_experiments > 0 and not self._stop_event.is_set():
                time.sleep(config.rest_between_experiments)

        # Stop knowledge distiller
        if distiller is not None:
            try:
                distiller.stop()
            except Exception:
                pass

        # Re-enable LLM for interactive use after continuous mode ends.
        self.aria._continuous_mode = False

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
                          n_experiments: int, digest=None) -> Dict:
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
                digest=digest,
            )

            # Build fallback data for rule-based recommendation
            total_s1 = sum(e.get("n_stage1_passed", 0) for e in recent)
            novelty_scores = [
                e.get("best_novelty_score", 0) for e in recent
                if e.get("best_novelty_score") is not None
            ]
            avg_novelty = (sum(novelty_scores) / len(novelty_scores)
                           if novelty_scores else 0)

            # Count candidates ready for investigation, excluding those already
            # attempted and failed (checked via program_results in investigation experiments)
            _investigated_fps = set()
            try:
                _inv_rows = nb.conn.execute(
                    "SELECT DISTINCT pr.graph_fingerprint "
                    "FROM program_results pr "
                    "JOIN experiments e ON e.experiment_id = pr.experiment_id "
                    "WHERE e.experiment_type = 'investigation'"
                ).fetchall()
                _investigated_fps = {r[0] for r in _inv_rows if r[0]}
            except Exception:
                pass

            investigation_ready = len([
                e for e in leaderboard
                if e.get("tier") == "screening"
                and e.get("screening_loss_ratio") is not None
                and e["screening_loss_ratio"] < config.investigation_loss_ratio_threshold
                and e.get("result_id") not in _investigated_fps
                # Also check by fingerprint from the linked program_result
            ])
            # More robust: filter by fingerprint
            if _investigated_fps:
                _inv_candidates = []
                for e in leaderboard:
                    if (e.get("tier") == "screening"
                            and e.get("screening_loss_ratio") is not None
                            and e["screening_loss_ratio"] < config.investigation_loss_ratio_threshold):
                        # Look up the fingerprint for this result
                        try:
                            fp_row = nb.conn.execute(
                                "SELECT graph_fingerprint FROM program_results WHERE result_id = ?",
                                (e["result_id"],)
                            ).fetchone()
                            fp = fp_row[0] if fp_row else None
                        except Exception:
                            fp = None
                        if fp and fp not in _investigated_fps:
                            _inv_candidates.append(e)
                investigation_ready = len(_inv_candidates)
            validation_ready = len([
                e for e in leaderboard
                if e.get("tier") == "investigation"
                and e.get("investigation_robustness") is not None
                and e["investigation_robustness"] >= config.investigation_robustness_threshold
            ])

            # Gather richer analytics for data-driven rule-based recommendation
            recent_modes = [e.get("experiment_type", "synthesis") for e in recent]
            
            # Z16: Hard-coded fallback logic to prevent pipeline starvation
            # Only promote 'worth it' candidates: Top performer per fingerprint
            # that also meets a 'worthiness' bar (e.g. low loss OR high novelty)
            worth_it_investigation = []
            seen_fps = set()
            
            # Sort leaderboard by best loss to prioritize performance
            sorted_lb = sorted(leaderboard, key=lambda x: x.get("screening_loss_ratio") or 1.0)
            
            for e in sorted_lb:
                if e.get("tier") != "screening": continue
                lr = e.get("screening_loss_ratio") or 1.0
                nov = e.get("screening_novelty") or 0.0
                fp = e.get("graph_fingerprint") or e.get("result_id")
                
                if fp in seen_fps: continue
                seen_fps.add(fp)
                
                # Worthiness Bar:
                # Prefer pre_inv_score when available (from pre-investigation gate)
                # Fall back to legacy heuristic thresholds
                pis = e.get("pre_inv_score")
                if pis is not None:
                    is_worth_it = float(pis) >= 20.0
                else:
                    # Legacy:
                    # 1. Exceptional performance (LR < 0.2)
                    # 2. Good performance + decent novelty (LR < 0.4 AND Nov > 0.4)
                    # 3. High novelty + moderate performance (LR < 0.6 AND Nov > 0.7)
                    is_worth_it = (
                        lr < 0.2 or
                        (lr < 0.4 and nov > 0.4) or
                        (lr < 0.6 and nov > 0.7)
                    )
                
                if is_worth_it and fp not in _investigated_fps:
                    worth_it_investigation.append(e["result_id"])

            investigation_backlog = len(worth_it_investigation)
            
            if investigation_backlog >= 5 and "investigation" not in recent_modes[:3]:
                return {
                    "mode": "investigation",
                    "reasoning": f"Backlog of {investigation_backlog} ELITE investigation-ready candidates detected. Prioritizing unique, high-performing architectures.",
                    "confidence": 1.0,
                    "config": {"n_programs": min(investigation_backlog, 20)}
                }
            
            if validation_ready >= 3 and "validation" not in recent_modes[:3]:
                return {
                    "mode": "validation",
                    "reasoning": f"Pipeline bottleneck at validation: {validation_ready} candidates ready. Switching to multi-seed verification.",
                    "confidence": 1.0,
                    "config": {"n_programs": min(validation_ready, 10)}
                }

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
                "base_n_programs": config.n_programs,
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

            compression_coverage = analytics_data.get("compression_coverage") or {}
            compression_totals = compression_coverage.get("totals") or {}
            n_tested = int(compression_totals.get("n_tested") or 0)
            n_compressed_tested = int(compression_totals.get("n_compressed_tested") or 0)
            n_compressed_survived = int(compression_totals.get("n_compressed_survived") or 0)
            n_survived = int(compression_totals.get("n_survived") or 0)
            compressed_test_share = (
                n_compressed_tested / n_tested if n_tested > 0 else 0.0
            )
            fallback_data["compressed_test_share"] = compressed_test_share
            fallback_data["compression_summary"] = {
                "n_tested": n_tested,
                "n_compressed_tested": n_compressed_tested,
                "n_compressed_survived": n_compressed_survived,
                "n_survived": n_survived,
                "compressed_survival_rate": (
                    n_compressed_survived / n_compressed_tested
                    if n_compressed_tested > 0
                    else 0.0
                ),
                "overall_survival_rate": (
                    n_survived / n_tested if n_tested > 0 else 0.0
                ),
            }

            rec = self.aria.recommend_next_mode(
                context=context, fallback_data=fallback_data, digest=digest,
                op_success_rates=analytics_data.get("op_success_rates"),
                compression_coverage=analytics_data.get("compression_coverage"))

            compression_override = self._compression_focus_override(rec, fallback_data)
            if compression_override is not None:
                rec = compression_override

            trigger = self._selection_safety_valve(nb, config)
            if trigger and trigger.get("triggered"):
                self._invoke_code_healer(
                    nb=nb,
                    trigger_type="plateau",
                    experiment_id=None,
                    scope=f"Safety valve plateau trigger: {trigger.get('reason')}",
                    reproduction_steps=["python -m pytest tests/test_selection_policy.py -x --tb=short"],
                    acceptance_tests=["python -m pytest tests/test_selection_policy.py -x --tb=short"],
                    trigger_payload=trigger,
                )
                if trigger.get("mode") == "novelty":
                    rec["mode"] = "novelty"
                    rec["reasoning"] = (
                        f"{rec.get('reasoning', '')} | Safety valve: {trigger.get('reason')}"
                    ).strip(" |")
                    rec.setdefault("config", {})
                    rec["config"]["n_generations"] = max(4, int(config.n_generations))
                    rec["config"]["population_size"] = max(12, int(config.population_size))
                else:
                    rec["mode"] = "synthesis"
                    rec.setdefault("config", {})
                    rec["config"]["ablation_heavy"] = True
                    rec["config"]["n_programs"] = max(8, int(config.n_programs * 0.6))
                    rec["reasoning"] = (
                        f"{rec.get('reasoning', '')} | Safety valve(ablation-heavy): "
                        f"{trigger.get('reason')}"
                    ).strip(" |")
                rec["safety_valve"] = trigger

            refinement_plan = self._build_refinement_plan(nb, config)
            if (
                refinement_plan
                and rec.get("mode") not in {"investigation", "validation"}
            ):
                rec.setdefault("config", {})
                rec["mode"] = "refinement"
                rec["config"].update(refinement_plan.get("config", {}))
                rec["refinement_plan"] = {
                    "source_result_ids": refinement_plan.get("source_result_ids", []),
                    "source_count": refinement_plan.get("source_count", 0),
                    "generations": refinement_plan.get("generations", 1),
                    "budget_programs": refinement_plan.get("budget_programs", 0),
                }
                rec["confidence"] = max(float(rec.get("confidence", 0.5) or 0.5), 0.7)
                rec["reasoning"] = (
                    f"{rec.get('reasoning', '')} | "
                    f"Recursive refinement on {rec['refinement_plan']['source_count']} "
                    f"diverse Stage-1 winners for {rec['refinement_plan']['generations']} generation(s)."
                ).strip(" |")

            evidence_pack = build_evidence_pack(
                nb,
                analytics=None,
                recommendation=rec,
                decision_type="mode_selection",
                recent_experiments=recent,
            )
            rec["evidence_pack"] = evidence_pack

            nb.add_entry(ExperimentEntry(
                entry_type="decision",
                title=f"Mode Selection: {rec.get('mode', 'synthesis')}",
                content=rec.get("reasoning", ""),
                metadata={
                    "mode": rec.get("mode"),
                    "confidence": rec.get("confidence"),
                    "experiment_number": n_experiments,
                    "evidence_pack": evidence_pack,
                },
            ))

            decision_log = {
                "decision_id": str(uuid.uuid4())[:12],
                "timestamp": time.time(),
                "context": "mode_selection",
                "experiment_id": None,
                "candidate_pool_summary": {
                    "recent_experiments": len(recent),
                    "leaderboard_candidates": len(leaderboard),
                    "total_s1_survivors": total_s1,
                    "avg_novelty": round(avg_novelty, 6),
                },
                "score_breakdown": [{
                    "mode": rec.get("mode"),
                    "confidence": rec.get("confidence"),
                    "quality_signal": total_s1,
                    "novelty_signal": round(avg_novelty, 6),
                }],
                "policy": {
                    "engine": "aria_mode_selection_with_refinement",
                    "safety_valve_triggered": bool(trigger),
                    "safety_valve": trigger,
                    "refinement_plan": rec.get("refinement_plan"),
                },
                "reason": rec.get("reasoning", ""),
                "chosen_experiments": [{
                    "mode": rec.get("mode"),
                    "config": rec.get("config", {}),
                }],
                "trigger": trigger,
            }
            try:
                validate_selection_decision_log(decision_log)
                nb.record_selection_decision(
                    context="mode_selection",
                    experiment_id=None,
                    candidate_pool_summary=decision_log["candidate_pool_summary"],
                    score_breakdown=decision_log["score_breakdown"],
                    policy=decision_log["policy"],
                    reason=decision_log["reason"],
                    chosen_experiments=decision_log["chosen_experiments"],
                    trigger=decision_log["trigger"],
                )
            except Exception as log_err:
                logger.debug("Mode selection decision log failed: %s", log_err)

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

        # Populate refuted hypotheses cache for similarity gating
        self._populate_refuted_cache(nb)

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

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="synthesis",
            config=exp_config,
            hypothesis=hypothesis,
            hypothesis_metadata=hypothesis_metadata,
            created_by="continuous_synthesis",
        )
        logger.info(
            "Experiment %s started (synthesis, %d programs) — hypothesis: %s",
            exp_id[:8], config.n_programs,
            (hypothesis or "none")[:150],
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
        if summary:
            logger.info("Aria summary: %s", summary[:200])
        nb.update_op_success_rates(exp_id)
        s0_op_counts = results.pop("_s0_op_counts", None)
        if s0_op_counts:
            nb.merge_op_failure_counts(s0_op_counts)
        nb.strip_graph_json_for_failures(exp_id)
        nb.update_failure_signatures(exp_id)
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

    def _log_grammar_weight_application(
        self,
        nb: LabNotebook,
        exp_id: str,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float],
        analytics: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Log grammar weight application with reproducible audit query."""
        audit_info: Dict[str, Any] = {}
        try:
            if analytics is not None:
                audit_info = analytics.grammar_weight_audit_info()
        except Exception:
            audit_info = {}
        nb.log_learning_event(
            "grammar_weights_applied",
            f"Applied learned grammar weights for experiment {exp_id}",
            old_weights=old_weights,
            new_weights=new_weights,
            evidence=json.dumps({"audit_query": audit_info}, sort_keys=True),
        )
        return audit_info

    def _run_ablation_experiment(
        self,
        nb: LabNotebook,
        config: RunConfig,
        hypothesis: str,
        ablation_graphs: List[Any],
    ) -> Tuple[List[str], str]:
        """Run Stage 0/0.5/1 evaluation on a generated ablation suite."""
        if not ablation_graphs:
            return ([], "inconclusive")

        evaluable_graphs: List[Any] = []
        dropped_invalid = 0
        dropped_compile = 0
        for graph in ablation_graphs:
            validation = validate_graph(
                graph,
                max_ops=max(1, int(config.max_ops)),
                max_depth=max(1, int(config.max_depth)),
            )
            if not validation.valid:
                dropped_invalid += 1
                continue
            try:
                compile_model(
                    [graph],
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                )
            except Exception:
                dropped_compile += 1
                continue
            evaluable_graphs.append(graph)

        if not evaluable_graphs:
            evidence = {
                "hypothesis": hypothesis,
                "received": len(ablation_graphs),
                "dropped_invalid": dropped_invalid,
                "dropped_compile": dropped_compile,
            }
            nb.log_learning_event(
                "ablation_skipped_no_evaluable_graphs",
                f"Skipped ablation run: no evaluable graphs for {hypothesis}",
                evidence=json.dumps(evidence, sort_keys=True),
            )
            return ([], "skipped_no_evaluable_graphs")

        ab_cfg = config.to_dict()
        ab_cfg["n_programs"] = len(evaluable_graphs)
        ab_cfg["ablation_from_hypothesis"] = hypothesis
        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="ablation",
            config=ab_cfg,
            hypothesis=f"Ablation: {hypothesis}",
            exploratory=True,
            created_by="ablation",
        )

        dev = torch.device(config.device if torch.cuda.is_available() else "cpu")
        dev_str = str(dev)
        stage0_pass = 0
        stage05_pass = 0
        stage1_pass = 0
        result_ids: List[str] = []
        for idx, graph in enumerate(evaluable_graphs):
            try:
                model = compile_model(
                    [graph],
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                ).to(dev)
            except Exception:
                continue
            s0 = self._safe_eval_for_stage(
                model,
                stage_tag="ablation",
                batch_size=2,
                seq_len=min(128, config.max_seq_len),
                vocab_size=config.vocab_size,
                device=dev_str,
            )
            s0_passed = bool(s0.passed)
            s05_passed = bool(s0.stability_score >= config.stage05_stability_threshold) if s0_passed else False
            if s0_passed:
                stage0_pass += 1
            if s05_passed:
                stage05_pass += 1
            s1_passed = False
            final_loss = None
            loss_ratio = None
            if s05_passed:
                s1 = self._micro_train(
                    model,
                    config,
                    dev,
                    seed=self._stable_seed(exp_id, idx, "ablation"),
                )
                s1_passed = bool(s1.get("passed", False))
                final_loss = s1.get("final_loss")
                loss_ratio = s1.get("loss_ratio")
            if s1_passed:
                stage1_pass += 1
            rid = nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=graph.fingerprint(),
                graph_json=graph_to_json(graph),
                stage0_passed=s0_passed,
                stage05_passed=s05_passed,
                stage1_passed=s1_passed,
                stage0_error=s0.error,
                final_loss=final_loss,
                loss_ratio=loss_ratio,
                model_source="ablation",
                perf_report_json=json.dumps(s1.get("perf_report", {})) if s1_passed else None,
                kernel_timings_json=json.dumps(s1.get("kernel_timings_ms", {})) if s1_passed else None,
                starvation_report_json=json.dumps(s1.get("starvation_report", {})) if s1_passed else None,
            )
            result_ids.append(rid)

        total = len(result_ids)
        outcome = "supported" if total > 0 and stage1_pass == 0 else "not_supported"
        if total == 0:
            outcome = "inconclusive"
        nb.complete_experiment(
            experiment_id=exp_id,
            results={
                "total": total,
                "stage0_passed": stage0_pass,
                "stage05_passed": stage05_pass,
                "stage1_passed": stage1_pass,
                "best_loss_ratio": None,
                "best_novelty_score": None,
            },
            aria_summary=f"Ablation outcome: {outcome}",
        )
        return ([exp_id], outcome)

    def _evaluate_grammar_update_gate(
        self,
        nb: LabNotebook,
        analytics: Any,
        config: RunConfig,
    ) -> Dict[str, Any]:
        """Require ablation support OR strong correlation with uncertainty+ablation plan."""
        attribution = analytics.grammar_weight_attribution_report()
        hypothesis_id = self._current_hypothesis_id
        previous = nb.get_attribution_reports(hypothesis_id=hypothesis_id, limit=50)
        has_ablation_support = any(r.get("outcome") == "supported" for r in previous)

        supporting_experiments = [
            e.get("experiment_id")
            for e in nb.get_recent_experiments(5)
            if e.get("experiment_id")
        ]
        strong_corr = bool(attribution.get("strong_correlational_evidence"))
        top_signal = attribution.get("top_signal") or {}
        factor_type = str(top_signal.get("factor_type") or "").strip().lower()
        factor_name = str(top_signal.get("factor_name") or "").strip().lower()
        top_signal_interpretable = bool(
            factor_type
            and factor_name
            and factor_name not in {"unknown", "none", "null", "nan"}
        )
        hypothesis_text = (
            f"signal={top_signal.get('factor_type')}:{top_signal.get('factor_name')}"
            if top_signal_interpretable
            else ""
        )

        ablation_experiments: List[str] = []
        queued_plan: List[str] = []
        ablation_outcome = "none"

        # Dedup: skip if this signal was already ablated (in any hypothesis)
        if strong_corr and top_signal_interpretable and hypothesis_text:
            try:
                already_tested = nb.conn.execute(
                    "SELECT COUNT(*) FROM experiments "
                    "WHERE experiment_type = 'ablation' "
                    "AND hypothesis LIKE ?",
                    (f"%{hypothesis_text}%",),
                ).fetchone()[0]
                if already_tested > 0:
                    logger.info(
                        "Skipping ablation for '%s' — already tested %d time(s)",
                        hypothesis_text, already_tested,
                    )
                    ablation_outcome = "skipped_already_tested"
                    strong_corr = False  # prevent triggering below
            except Exception:
                pass

        if strong_corr and top_signal_interpretable:
            row = nb.conn.execute(
                """SELECT graph_json FROM program_results
                   WHERE stage1_passed = 1 AND graph_json IS NOT NULL
                   ORDER BY loss_ratio ASC NULLS LAST LIMIT 1"""
            ).fetchone()
            if row and row["graph_json"]:
                try:
                    base_graph = graph_from_json(row["graph_json"])
                    suite = propose_ablation_suite(base_graph, hypothesis_text)
                    queued_plan = [g.fingerprint() for g in suite]
                    if suite:
                        ablation_experiments, ablation_outcome = self._run_ablation_experiment(
                            nb=nb,
                            config=config,
                            hypothesis=hypothesis_text,
                            ablation_graphs=suite,
                        )
                except Exception as e:
                    logger.debug("Ablation run failed: %s", e)
        elif strong_corr:
            ablation_outcome = "skipped_low_quality_signal"

        gate_pass = bool(
            has_ablation_support
            or (strong_corr and bool(ablation_experiments))
        )
        if has_ablation_support:
            outcome = "supported"
        elif strong_corr and not top_signal_interpretable:
            outcome = "blocked_low_quality_signal"
        elif strong_corr and ablation_outcome == "skipped_no_evaluable_graphs":
            outcome = "blocked_no_evaluable_ablation"
        elif gate_pass:
            outcome = "correlational_with_plan"
        else:
            outcome = "blocked_weak_evidence"
        report = {
            "gate_pass": gate_pass,
            "has_ablation_support": has_ablation_support,
            "strong_correlational_evidence": strong_corr,
            "top_signal_interpretable": top_signal_interpretable,
            "uncertainty": attribution.get("uncertainty", {}),
            "top_signal": top_signal,
            "queued_ablation_plan": queued_plan,
            "ablation_outcome": ablation_outcome,
            "attribution": attribution,
        }
        nb.record_attribution_report(
            hypothesis_id=hypothesis_id,
            supporting_experiments=supporting_experiments,
            ablation_experiments=ablation_experiments,
            outcome=outcome,
            report=report,
        )
        return report

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

    def _compute_multi_objective_fitness(self, s1_result, sandbox_result, graph, config):
        """Multi-objective fitness: quality + efficiency + speed + learning + compactness."""
        weights = {
            "quality": 0.30,
            "efficiency": 0.25,
            "speed": 0.10,
            "learning_speed": 0.20,
            "compactness": 0.15,
        }

        components = {}

        # Quality: 1 - loss_ratio
        lr = s1_result.get("loss_ratio", 1.0) if s1_result else 1.0
        components["quality"] = max(0.0, 1.0 - lr)

        # Efficiency: prefer fewer params
        max_params = config.model_dim * config.vocab_size * 2
        param_count = getattr(sandbox_result, "param_count", 0) or 0
        if param_count > 0 and max_params > 0:
            components["efficiency"] = max(0.0, 1.0 - min(param_count / max_params, 1.0))
        else:
            components["efficiency"] = 0.0

        # Speed: throughput in tokens/sec
        target_throughput = 50000.0
        throughput = s1_result.get("throughput", 0) if s1_result else 0
        if throughput and throughput > 0:
            components["speed"] = min(throughput / target_throughput, 1.0)
        else:
            components["speed"] = 0.0

        # Learning speed: how fast loss improved
        lir = s1_result.get("loss_improvement_rate", 0) if s1_result else 0
        components["learning_speed"] = max(0.0, min(float(lir or 0), 1.0))

        # Compactness: fewer ops = simpler
        n_ops = len(graph.nodes) if hasattr(graph, "nodes") else 0
        max_ops = max(1, int(config.max_ops))
        components["compactness"] = max(0.0, 1.0 - min(n_ops / max_ops, 1.0))

        # Redistribute weight from missing components to quality
        weighted_sum = 0.0
        missing_weight = 0.0
        for key, w in weights.items():
            val = components[key]
            if val > 0 or key == "quality":
                weighted_sum += val * w
            else:
                missing_weight += w

        # Give missing weight to quality
        if missing_weight > 0:
            weighted_sum += components["quality"] * missing_weight

        return weighted_sum, components

    def _run_continuous_evolution(self, config: RunConfig, nb: LabNotebook,
                                  n_experiments: int, limit_str: str,
                                  mode_reasoning: str):
        """Run evolution search within continuous mode (inline, not threaded)."""
        from ..search.evolution import evolutionary_search, EvolutionConfig

        hypothesis = f"Evolution search: {mode_reasoning}"
        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="evolution",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source="runner_template",
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
            created_by="continuous_evolution",
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
        evo_config = EvolutionConfig(
            population_size=config.n_programs,
            n_generations=config.n_generations,
            grammar_config=self._build_grammar_config(config),
        )

        fitness_cache: dict = {}
        eval_counters = {"total": 0, "s0": 0, "s1": 0}

        def on_evaluate(graph, fitness, sandbox_result, s1_result):
            self._on_program_evaluated(graph, fitness, sandbox_result, s1_result,
                                       eval_counters, nb, exp_id, model_source="evolution")

        fitness_fn = self._make_fitness_fn(
            config, on_evaluate=on_evaluate, fitness_cache=fitness_cache)

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
            stop_check=self._stop_event.is_set,
        )

        results = {
            "total": eval_counters["total"],
            "stage0_passed": eval_counters["s0"],
            "stage05_passed": eval_counters["s0"],
            "stage1_passed": eval_counters["s1"],
            "novel_count": sum(1 for ind in population if ind.novelty > 0.5),
            "best_loss_ratio": 1.0 - max((ind.fitness for ind in population), default=0),
            "best_novelty_score": max((ind.novelty for ind in population), default=0),
            "survivors": [],
        }

        for ind in population[:20]:
            if ind.fitness > 0.2:
                results["survivors"].append({
                    "fingerprint": ind.fingerprint,
                    "novelty": ind.novelty,
                    "loss_ratio": 1.0 - ind.fitness,
                })

        nb.update_op_success_rates(exp_id)
        nb.update_failure_signatures(exp_id)
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
        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="novelty",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source="runner_template",
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
            exploratory=True,
            created_by="continuous_novelty",
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

        grammar = self._build_grammar_config(config)
        ns_config = NoveltySearchConfig(
            population_size=config.n_programs,
            n_generations=config.n_generations,
            grammar_config=grammar,
        )
        dev_str = config.device if torch.cuda.is_available() else "cpu"
        dev = torch.device(dev_str)

        fitness_cache: dict = {}
        fingerprint_cache: dict = {}
        eval_counters = {"total": 0, "s0": 0, "s1": 0}

        def on_evaluate(graph, fitness, sandbox_result, s1_result):
            self._on_program_evaluated(graph, fitness, sandbox_result, s1_result,
                                       eval_counters, nb, exp_id, model_source="novelty")

        def combined_fitness_fn(graph):
            """Compile once, run sandbox + micro-train + fingerprint in one pass."""
            gfp = graph.fingerprint()

            if gfp in fitness_cache:
                return fitness_cache[gfp]

            sandbox_result = None
            s1_result = None
            try:
                layer_graphs = [graph] * config.n_layers
                model = compile_model(
                    layer_graphs,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                )
                sandbox_result = self._safe_eval_for_stage(
                    model,
                    stage_tag="novelty_fitness",
                    batch_size=2,
                    seq_len=min(128, config.max_seq_len),
                    vocab_size=config.vocab_size,
                    device=dev_str,
                )
                if not sandbox_result.passed:
                    del model
                    fitness = 0.0
                    fitness_cache[gfp] = fitness
                    on_evaluate(graph, fitness, sandbox_result, s1_result)
                    return fitness

                # Compute behavioral fingerprint while model is still in memory
                try:
                    bfp = compute_fingerprint(
                        model,
                        seq_len=min(64, config.max_seq_len),
                        model_dim=config.model_dim,
                        vocab_size=config.vocab_size,
                        device=dev_str,
                    )
                    fingerprint_cache[gfp] = bfp
                except Exception as e:
                    logger.debug("Fingerprint computation failed: %s", e)

                s1_result = self._micro_train(
                    model,
                    config,
                    dev,
                    seed=self._stable_seed("fitness", gfp),
                )
                del model
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

                if s1_result.get("passed"):
                    fitness, _components = self._compute_multi_objective_fitness(
                        s1_result, sandbox_result, graph, config)
                else:
                    fitness = 0.1
            except Exception:
                fitness = 0.0

            fitness_cache[gfp] = fitness
            on_evaluate(graph, fitness, sandbox_result, s1_result)
            return fitness

        def fingerprint_fn(graph):
            return fingerprint_cache.get(graph.fingerprint())

        ns_result = novelty_search(
            fitness_fn=combined_fitness_fn,
            fingerprint_fn=fingerprint_fn,
            config=ns_config,
            stop_check=self._stop_event.is_set,
        )

        results = {
            "total": eval_counters["total"],
            "stage0_passed": eval_counters["s0"],
            "stage05_passed": eval_counters["s0"],
            "stage1_passed": eval_counters["s1"],
            "novel_count": sum(1 for ind in ns_result.best_individuals if ind.novelty > 0.5),
            "best_loss_ratio": None,
            "best_novelty_score": None,
            "survivors": [],
            "archive_size": ns_result.archive_size,
        }

        for ind in ns_result.best_individuals[:20]:
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
        nb.update_failure_signatures(exp_id)
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

    def _run_continuous_refinement(
        self,
        config: RunConfig,
        nb: LabNotebook,
        n_experiments: int,
        limit_str: str,
        mode_reasoning: str,
    ):
        """Run recursive local winner-tweak refinement with plateau stopping."""
        plan = self._build_refinement_plan(nb, config)
        if not plan:
            logger.info("Refinement requested but no eligible Stage-1 winners found; falling back to synthesis.")
            self._run_continuous_synthesis(config, nb, n_experiments, limit_str, mode_reasoning)
            return

        source_ids = list(plan.get("source_result_ids", []))
        total_generations = max(1, int(plan.get("generations") or config.refinement_generations or 1))
        budget_remaining = max(int(plan.get("budget_programs") or 0), int(config.n_programs))
        plateau_patience = max(1, int(config.refinement_plateau_patience or 1))
        mutation_radius = max(0.05, min(1.0, float(config.refinement_mutation_radius or 0.35)))
        novelty_pressure = max(0.0, min(1.0, float(config.refinement_novelty_pressure or 0.35)))

        best_loss_seen: Optional[float] = None
        plateau_count = 0
        executed_generations = 0
        history: List[Dict[str, Any]] = []

        for generation in range(total_generations):
            if self._stop_event.is_set() or budget_remaining <= 0 or not source_ids:
                break

            gen_cfg = RunConfig.from_dict(config.to_dict())
            gen_cfg.model_source = "fingerprint_refine"
            gen_cfg.refine_source_result_ids = ",".join(source_ids)
            gen_cfg.refine_mutations_per_source = max(1, int(round(2 + 4 * mutation_radius)))
            gen_cfg.refine_pool_multiplier = max(2, int(round(2 + 3 * novelty_pressure)))
            gen_cfg.mutation_rate = max(0.10, min(0.95, float(config.mutation_rate) * (0.5 + mutation_radius)))
            gen_cfg.n_programs = max(4, min(int(config.n_programs), budget_remaining))

            generation_reason = (
                f"{mode_reasoning} | recursive_refine gen {generation + 1}/{total_generations} "
                f"from {len(source_ids)} seed(s)"
            )
            self._run_continuous_synthesis(
                gen_cfg,
                nb,
                n_experiments,
                limit_str,
                generation_reason,
            )
            budget_remaining -= int(gen_cfg.n_programs)
            executed_generations += 1

            recent = nb.get_recent_experiments(1)
            if not recent:
                break
            current_exp_id = str(recent[0].get("experiment_id") or "")
            if not current_exp_id:
                break
            rows = nb.get_program_results(current_exp_id, limit=400)
            survivors = [row for row in rows if row.get("stage1_passed")]
            if not survivors:
                history.append({
                    "generation": generation + 1,
                    "experiment_id": current_exp_id,
                    "stage1_survivors": 0,
                    "best_loss_ratio": None,
                })
                break

            cur_best = min(
                (float(r.get("loss_ratio")) for r in survivors if isinstance(r.get("loss_ratio"), (int, float))),
                default=None,
            )
            history.append({
                "generation": generation + 1,
                "experiment_id": current_exp_id,
                "stage1_survivors": len(survivors),
                "best_loss_ratio": cur_best,
            })
            if cur_best is not None:
                if best_loss_seen is None or cur_best < best_loss_seen - 1e-4:
                    best_loss_seen = cur_best
                    plateau_count = 0
                else:
                    plateau_count += 1

            selected = self._select_diverse_refinement_sources(
                survivors,
                top_k=max(1, int(config.refinement_top_k or 1)),
                min_distance=max(0.01, float(config.refinement_min_distance or 0.01)),
                novelty_pressure=novelty_pressure,
            )
            source_ids = [str(r.get("result_id") or "") for r in selected if r.get("result_id")]
            if plateau_count >= plateau_patience:
                break

        nb.record_decision(
            campaign_id=self._active_campaign_id,
            decision_type="recursive_refinement",
            subject=f"cycle_{n_experiments}",
            rationale=(
                f"Executed recursive local refinement for {executed_generations}/{total_generations} generation(s); "
                f"plateau_count={plateau_count}, budget_remaining={budget_remaining}."
            ),
            evidence_pack={
                "seed_count": int(plan.get("source_count") or 0),
                "initial_source_result_ids": plan.get("source_result_ids", []),
                "history": history,
                "plateau_patience": plateau_patience,
                "min_distance": float(config.refinement_min_distance),
                "novelty_pressure": novelty_pressure,
            },
        )

    # ── Pre-investigation gate ─────────────────────────────────────────

    def _get_reference_baseline_lr(self, nb: LabNotebook) -> Optional[float]:
        """Fetch best screening_loss_ratio from registered reference architectures."""
        try:
            refs = nb.get_references()
            if not refs:
                return None
            lrs = [float(r["screening_loss_ratio"]) for r in refs
                   if r.get("screening_loss_ratio") is not None]
            return min(lrs) if lrs else None
        except Exception:
            return None

    def _pre_inv_probe(self, config: RunConfig, nb: LabNotebook,
                       result_id: str) -> Optional[float]:
        """Stage C: single-seed probe at reduced step count.

        Runs 1 training program at probe_steps_fraction of investigation_steps.
        Returns loss_ratio or None on failure.
        """
        try:
            details = nb.get_program_details([result_id])
            if not details or not details[0]:
                return None
            source = details[0]
            graph_json = source.get("graph_json")
            if not graph_json:
                return None

            probe_config = RunConfig.from_dict(config.to_dict())
            probe_config.stage1_steps = max(
                50, int(config.investigation_steps * config.pre_inv_probe_steps_fraction))
            probe_config.stage1_batch_size = config.investigation_batch_size
            probe_config.n_programs = 1

            dev_str = config.device if torch.cuda.is_available() else "cpu"
            dev = torch.device(dev_str)

            from research.synthesis.compiler import compile_model
            model = compile_model(graph_json, probe_config, device=dev)
            if model is None:
                return None

            from research.evaluator import evaluate_stage1
            result = evaluate_stage1(model, probe_config, device=dev)
            lr = result.get("loss_ratio") if result else None
            return float(lr) if lr is not None else None
        except Exception as e:
            logger.warning("Pre-inv probe failed for %s: %s", result_id[:8], e)
            return None

    def _pre_investigation_gate(self, config: RunConfig, nb: LabNotebook,
                                leaderboard: list) -> List[str]:
        """Orchestrate three-stage pre-investigation gate.

        Stage A: SQL hard reject (numerical health, stability, gradient path)
        Stage B: Composite readiness score, rank and take top-N
        Stage C: Optional single-seed probe

        Returns filtered, ranked result_ids ready for investigation.
        Falls back to legacy behavior when pre_inv_gate_enabled=False.
        """
        if not config.pre_inv_gate_enabled:
            # Legacy behavior: filter by loss_ratio threshold only
            investigated_fps = nb.get_investigated_fingerprints()
            candidates = [
                e for e in leaderboard
                if e.get("tier") == "screening"
                and e.get("screening_loss_ratio") is not None
                and e["screening_loss_ratio"] < config.investigation_loss_ratio_threshold
            ]
            if investigated_fps:
                candidates = [
                    c for c in candidates
                    if c.get("graph_fingerprint", c.get("architecture_desc", ""))
                    not in investigated_fps
                ]
            return [c["result_id"] for c in candidates[:config.auto_investigate_top_n]
                    if c.get("result_id")]

        # ── Stage A: Hard reject via SQL ──
        ref_lr = self._get_reference_baseline_lr(nb)
        ref_lr_ceiling = None
        if ref_lr is not None:
            ref_lr_ceiling = ref_lr * config.pre_inv_reference_margin

        eligible = nb.get_investigation_eligible(
            max_lr=config.investigation_loss_ratio_threshold,
            min_stability=config.pre_inv_min_stability,
            min_spectral_norm=config.pre_inv_min_spectral_norm,
            max_spectral_norm=config.pre_inv_max_spectral_norm,
            min_improvement_rate=config.pre_inv_min_improvement_rate,
            ref_lr_ceiling=ref_lr_ceiling,
        )

        # Filter out already-investigated fingerprints
        investigated_fps = nb.get_investigated_fingerprints()
        if investigated_fps:
            before = len(eligible)
            eligible = [e for e in eligible
                        if e.get("graph_fingerprint") not in investigated_fps]
            skipped = before - len(eligible)
            if skipped:
                logger.info("Pre-inv gate: skipped %d already-investigated candidates", skipped)

        if not eligible:
            logger.info("Pre-inv gate Stage A: no eligible candidates")
            return []

        logger.info("Pre-inv gate Stage A: %d candidates pass hard filters", len(eligible))

        # ── Stage B: Composite score + rank ──
        for row in eligible:
            row["_pre_inv_score"] = LabNotebook.compute_pre_investigation_score(
                row, best_ref_lr=ref_lr)

        eligible.sort(key=lambda r: r.get("_pre_inv_score", 0), reverse=True)
        top_n = eligible[:config.pre_inv_top_n]

        # Persist scores to leaderboard
        for row in eligible:
            try:
                nb.conn.execute(
                    "UPDATE leaderboard SET pre_inv_score = ? WHERE result_id = ?",
                    (row["_pre_inv_score"], row["result_id"]),
                )
            except Exception:
                pass
        try:
            nb.conn.commit()
        except Exception:
            pass

        logger.info("Pre-inv gate Stage B: top %d scored [%s]",
                     len(top_n),
                     ", ".join(f"{r['result_id'][:8]}={r['_pre_inv_score']:.1f}"
                               for r in top_n))

        # ── Stage C: Optional probe ──
        if config.pre_inv_probe_enabled:
            probed = []
            for row in top_n:
                probe_lr = self._pre_inv_probe(config, nb, row["result_id"])
                if probe_lr is not None and probe_lr > config.pre_inv_probe_max_lr:
                    logger.info("Pre-inv probe rejected %s (lr=%.3f > %.3f)",
                                row["result_id"][:8], probe_lr,
                                config.pre_inv_probe_max_lr)
                    continue
                probed.append(row)
            top_n = probed

        return [r["result_id"] for r in top_n if r.get("result_id")]

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
        # Use pre-investigation gate for candidate selection
        result_ids = self._pre_investigation_gate(config, nb, leaderboard)
        if not result_ids:
            self._run_continuous_synthesis(
                config, nb, n_experiments, limit_str, mode_reasoning)
            return

        # Build context for hypothesis formulation
        inv_details = [d or {} for d in (nb.get_program_details(result_ids) or [])]
        inv_map = {d.get("result_id"): d for d in inv_details if d.get("result_id")}
        inv_context = build_investigation_context(inv_details, leaderboard)
        hypothesis = self.aria.formulate_investigation_hypothesis(
            context=inv_context)
        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="investigation",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source="llm_context",
                llm_used=True,
                fallback_used=False,
                used_context=True,
            ),
            created_by="inline_investigation",
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

            # Fetch all sources at once to avoid N+1 queries
            program_details = [d or {} for d in (nb.get_program_details(result_ids) or [])]
            source_map = {d.get("result_id"): d for d in program_details if d.get("result_id")}

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
                source = inv_map.get(source_result_id)
                if source is None:
                    continue

                graph_json_str = source.get("graph_json")
                arch_spec_json_str = source.get("arch_spec_json")
                model_source = source.get("model_source") or "graph_synthesis"

                # Generate training programs (queue-level scheduling telemetry)
                training_programs, tp_sched = synthesize_training_program_batch(
                    n_programs=config.n_training_programs,
                    n_steps=config.investigation_steps,
                    max_seq_len=config.max_seq_len,
                    seed_offset=prog_idx * 1000,
                )
                results.setdefault("training_program_scheduling", []).append({
                    "result_id": source_result_id,
                    **tp_sched,
                })

                # Test each (model x training_program) pair
                tp_results = []
                for tp_i, tp in enumerate(training_programs):
                    if self._stop_event.is_set():
                        break

                    # Reconstruct model fresh for each training program
                    try:
                        model = self._build_model_from_source(
                            model_source,
                            arch_spec_json_str,
                            graph_json_str,
                            config,
                            seq_len_override=config.max_seq_len,
                        )
                        if model is None:
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

                # Skip candidates where no training program could reconstruct the model
                if not tp_results:
                    logger.debug(
                        f"Investigation: skipping {source_result_id[:8]} — "
                        f"model failed to reconstruct for all {len(training_programs)} programs"
                    )
                    continue

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
                    "training_program_scheduling_avg_ms": tp_sched.get("scheduling_avg_ms"),
                    "training_program_scheduling_max_ms": tp_sched.get("scheduling_max_ms"),
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

                # Brittle risk override: if the investigation LR is good on
                # its own merits (< 0.3), don't let the screening→investigation
                # multiplier veto promotion.  Prevents false positives when
                # screening LR was unrealistically low (e.g. lucky seed).
                investigation_passed = (
                    robustness >= 0.5
                    and (best_lr or 1.0) < 0.5
                    and (not brittle_risk
                         or (best_lr is not None and best_lr < 0.3))
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
                    fp_jacobian_spectral_norm=source.get("fp_jacobian_spectral_norm"),
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
                    novelty_raw_score=source.get("novelty_raw_score"),
                    novelty_z_score=source.get("novelty_z_score"),
                    novelty_reference_version=source.get("novelty_reference_version"),
                    novelty_valid_for_promotion=source.get("novelty_valid_for_promotion"),
                    novelty_validity_reason=source.get("novelty_validity_reason"),
                    novelty_requires_justification=source.get("novelty_requires_justification"),
                    training_program_json=best_tp_json,
                    model_source=model_source,
                    arch_spec_json=arch_spec_json_str,
                )

            # Complete experiment with LLM analysis
            results["perf_report"] = self._build_experiment_perf_report(results)
            results["perf_budget_gate"] = evaluate_perf_budget_gate(results["perf_report"])
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
            and e["investigation_robustness"] >= config.investigation_robustness_threshold
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
        val_details = [d or {} for d in (nb.get_program_details(result_ids) or [])]
        val_map = {d.get("result_id"): d for d in val_details if d.get("result_id")}
        val_context = build_validation_context(
            val_details,
            [e for e in leaderboard if e.get("result_id") in result_ids],
        )
        hypothesis = self.aria.formulate_validation_hypothesis(
            context=val_context)
        exp_id = self._start_preregistered_experiment(
            nb=nb,
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
            created_by="inline_validation",
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

            # Fetch all sources at once to avoid N+1 queries
            program_details = [d or {} for d in (nb.get_program_details(result_ids) or [])]
            source_map = {d.get("result_id"): d for d in program_details if d.get("result_id")}

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
                source = source_map.get(source_result_id)
                if source is None:
                    continue

                graph_json_str = source.get("graph_json")
                arch_spec_json_str = source.get("arch_spec_json")
                model_source = source.get("model_source") or "graph_synthesis"

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
                    init_scheme = "default"
                    try:
                        model = self._build_model_from_source(
                            model_source,
                            arch_spec_json_str,
                            graph_json_str,
                            config,
                            seq_len_override=config.validation_seq_len,
                        )
                        if model is None:
                            continue
                        # Multi-init: use Xavier uniform for the last seed
                        if seed == config.validation_n_seeds - 1:
                            init_scheme = "xavier_uniform"
                            for p in model.parameters():
                                if p.dim() >= 2:
                                    nn.init.xavier_uniform_(p)
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
                    })

                    del model
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

                # Skip candidates where no seed could reconstruct the model
                if not seed_results:
                    logger.debug(
                        f"Inline validation: skipping {source_result_id[:8]} — "
                        f"model failed to reconstruct for all {config.validation_n_seeds} seeds"
                    )
                    continue

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

                # Init sensitivity: std between default and xavier seeds
                init_sensitivity_std = None
                default_losses = [
                    r["loss_ratio"] for r in seed_results
                    if r.get("init_scheme") == "default" and r.get("loss_ratio") is not None
                ]
                xavier_losses = [
                    r["loss_ratio"] for r in seed_results
                    if r.get("init_scheme") == "xavier_uniform" and r.get("loss_ratio") is not None
                ]
                if default_losses and xavier_losses:
                    default_mean = sum(default_losses) / len(default_losses)
                    xavier_mean = sum(xavier_losses) / len(xavier_losses)
                    init_sensitivity_std = abs(default_mean - xavier_mean)

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
                            bl_data_fn, bl_data_tag, bl_cache = self._make_baseline_data_fn(config)
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
                                cache_data_fn=bl_cache,
                            )
                            # Optional: Validation baseline comparison (using val split)
                            v_loss = best_seed.get("validation_loss")
                            if v_loss is not None:
                                try:
                                    v_data_fn, v_data_tag, v_cache = self._make_baseline_data_fn(config, split="val")
                                    v_baseline_ratio = baseline.compare(
                                        v_loss,
                                        d_model=config.model_dim,
                                        seq_len=min(128, int(getattr(config, "validation_seq_len", 128))),
                                        n_steps=max(1, baseline_steps),
                                        vocab_size=config.vocab_size,
                                        batch_size=int(getattr(config, "validation_batch_size", 4)),
                                        lr=baseline_recipe["lr"],
                                        device=dev_str,
                                        n_layers=config.n_layers,
                                        optimizer_name=baseline_recipe["optimizer_name"],
                                        weight_decay=baseline_recipe["weight_decay"],
                                        momentum=baseline_recipe["momentum"],
                                        betas=baseline_recipe["betas"],
                                        data_fn=v_data_fn,
                                        data_tag=v_data_tag,
                                        cache_data_fn=v_cache,
                                    )
                                    program_metrics["validation_baseline_loss_ratio"] = v_baseline_ratio
                                except Exception:
                                    pass
                        except Exception:
                            pass

                # Parameter-normalized baseline comparison
                val_normalized_ratio = None
                val_param_efficiency = None
                source_params = (source.get("param_count")
                                 or source.get("graph_n_params_estimate")
                                 or 0) if source else 0
                if loss_ratios and best_seed is not None and source_params > 0:
                    try:
                        baseline = self._get_baseline()
                        baseline_steps = int(best_seed.get("n_train_steps") or config.validation_steps)
                        baseline_recipe = self._resolve_baseline_recipe(
                            best_seed, default_lr=config.stage1_lr)
                        bl_data_fn, bl_data_tag, bl_cache = self._make_baseline_data_fn(config)
                        norm_result = baseline.compare_normalized(
                            best_seed["final_loss"],
                            program_params=int(source_params),
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
                            cache_data_fn=bl_cache,
                        )
                        val_normalized_ratio = norm_result.get("normalized_ratio")
                        val_param_efficiency = norm_result.get("param_efficiency")
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

                # Determine if breakthrough — requires both raw AND normalized thresholds
                ood_ok = (ood_result is not None
                          and ood_result.get("ood_robustness", 0) >= 0.67)
                hp_ok = (sensitivity_result is not None
                         and sensitivity_result.get("hp_robustness", 0) >= 0.75)
                nov_conf = source.get("novelty_confidence", 0) if source else 0
                novelty_valid = False
                if source:
                    novelty_valid = bool(source.get("novelty_valid_for_promotion"))
                    if not novelty_valid and source.get("cka_source") == "artifact":
                        novelty_valid = True

                raw_threshold = config.breakthrough_raw_threshold
                norm_threshold = config.breakthrough_normalized_threshold
                raw_ok = (val_baseline_ratio is not None
                          and val_baseline_ratio < raw_threshold)
                norm_ok = (val_normalized_ratio is None
                           or val_normalized_ratio < norm_threshold)
                is_breakthrough = (
                    raw_ok
                    and norm_ok
                    and multi_seed_std <= 0.03
                    and len(passed_seeds) >= 5
                    and len(passed_seeds) == config.validation_n_seeds
                    and (ood_result is None or ood_ok)
                    and (sensitivity_result is None or hp_ok)
                    and nov_conf >= 0.5
                    and novelty_valid
                )

                # FLOP gate: reject breakthrough if >5x baseline FLOPs per token
                flop_gated = False
                if is_breakthrough and source_params > 0:
                    candidate_fpt = source_params * 2.0
                    baseline_fpt_gate = 2.0 * config.model_dim ** 2 * config.n_layers
                    if candidate_fpt > 5.0 * baseline_fpt_gate:
                        is_breakthrough = False
                        flop_gated = True
                        logger.info(
                            "FLOP gate downgraded %s: %.0f FPT > 5x baseline %.0f",
                            source_result_id[:8], candidate_fpt, baseline_fpt_gate,
                        )

                # Scaling law comparison gate
                scaling_result = None
                scaling_param_efficiency = None
                scaling_flop_efficiency = None
                scaling_gate_passed_val = None
                scaling_best_family = None
                scaling_confidence = None
                if is_breakthrough and config.enable_scaling_comparison:
                    try:
                        scaling_mgr = self._get_scaling_reference_manager()
                        bl_data_fn, bl_data_tag, _ = self._make_baseline_data_fn(config)
                        candidate_flops = (source.get("flops_forward", 0) or 0)
                        if candidate_flops <= 0:
                            candidate_flops = source_params * 2

                        scaling_result = scaling_mgr.compare_candidate(
                            candidate_loss=best_seed_loss,
                            candidate_params=source_params,
                            candidate_flops=candidate_flops,
                            d_model=config.model_dim,
                            n_steps=config.validation_steps,
                            seq_len=config.validation_seq_len,
                            vocab_size=config.vocab_size,
                            batch_size=config.validation_batch_size,
                            lr=config.stage1_lr,
                            device=dev_str,
                            data_fn=bl_data_fn, data_tag=bl_data_tag,
                            families=config.scaling_reference_families.split(","),
                            param_efficiency_target=config.scaling_param_efficiency_target,
                            flop_ceiling=config.scaling_flop_ceiling,
                        )
                        scaling_param_efficiency = scaling_result.best_param_efficiency
                        scaling_flop_efficiency = scaling_result.flop_efficiency
                        scaling_gate_passed_val = scaling_result.scaling_gate_passed
                        scaling_best_family = scaling_result.best_param_efficiency_family
                        scaling_confidence = scaling_result.confidence

                        if not scaling_result.scaling_gate_passed:
                            is_breakthrough = False
                            logger.info(
                                "Scaling gate downgraded %s: param_eff=%.2f (need %.1f), flop_eff=%.2f",
                                source_result_id[:8],
                                scaling_result.best_param_efficiency,
                                config.scaling_param_efficiency_target,
                                scaling_result.flop_efficiency,
                            )
                    except Exception as e:
                        logger.debug("Scaling comparison failed: %s", e)

                # Quantization eval: test INT8 retention for all validation candidates
                quant_int8_retention = None
                quant_quality_per_byte = None
                if best_seed is not None:
                    try:
                        from ..eval.quantization import evaluate_sparse_quant_quality
                        # Build a fresh model for quant eval
                        if model_source == "morphological_box" and arch_spec_json_str:
                            from ..morphological_box import ArchSpec
                            from ..arch_builder import build_model, BuildConfig
                            _spec = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len)
                            quant_model = build_model(_spec, _bc).to(dev)
                        else:
                            quant_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len).to(dev)
                        # Generate test batches
                        quant_batches = [
                            torch.randint(0, config.vocab_size,
                                          (2, min(128, config.validation_seq_len)),
                                          device=dev)
                            for _ in range(4)
                        ]
                        quant_result = evaluate_sparse_quant_quality(
                            quant_model, quant_batches, dev,
                            target_sparsity=0.5, bits=8)
                        if quant_result is not None:
                            quant_int8_retention = quant_result.get("full_retention")
                            quant_quality_per_byte = quant_result.get("quality_per_byte")
                            if is_breakthrough and quant_int8_retention is not None and quant_int8_retention < 0.80:
                                is_breakthrough = False
                                logger.info(
                                    "Quant gate downgraded %s: INT8 retention=%.3f < 0.80",
                                    source_result_id[:8], quant_int8_retention,
                                )
                        del quant_model
                    except Exception as e:
                        logger.debug("Quantization eval skipped: %s", e)

                # Long-context sweep (informational, non-blocking)
                long_context_score = None
                if best_seed is not None:
                    try:
                        from ..eval.long_context import run_long_context_sweep
                        base_loss_val = best_seed.get("final_loss", 0)
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _asjs_lc = arch_spec_json_str
                            _cfg_lc = config
                            def _make_model_lc():
                                from ..morphological_box import ArchSpec
                                from ..arch_builder import build_model, BuildConfig
                                _sp = ArchSpec(**json.loads(_asjs_lc))
                                _bc2 = BuildConfig(
                                    dim=_cfg_lc.model_dim, n_layers=_cfg_lc.n_layers,
                                    vocab_size=_cfg_lc.vocab_size, max_seq_len=1024)
                                return build_model(_sp, _bc2)
                        else:
                            _gjs_lc = graph_json_str
                            _cfg_lc = config
                            def _make_model_lc():
                                return compile_model(
                                    [graph_from_json(_gjs_lc)] * _cfg_lc.n_layers,
                                    vocab_size=_cfg_lc.vocab_size, max_seq_len=1024)
                        from ..eval.long_context import run_long_context_sweep
                        from ..eval.passkey import evaluate_long_context_retrieval
                        
                        lc_result = run_long_context_sweep(
                            _make_model_lc, config.vocab_size, dev,
                            base_loss=base_loss_val, seq_lens=(512, 1024),
                            n_steps=200, batch_size=2,
                        )
                        
                        # Retrieval test (needle-in-a-haystack)
                        # Use a small validation model for faster retrieval testing
                        retr_model = _make_model_lc().to(dev)
                        retr_result = evaluate_long_context_retrieval(
                            retr_model, config.vocab_size, dev,
                            lengths=[256, 512, 1024]
                        )
                        del retr_model
                        
                        # Combine scaling score and retrieval score (50/50)
                        scaling_score = lc_result.get("long_context_score", 0.0)
                        retrieval_score = retr_result.get("retrieval_score", 0.0)
                        long_context_score = (scaling_score * 0.5) + (retrieval_score * 0.5)
                        
                        logger.info("Long-context check: scaling=%.2f, retrieval=%.2f, combined=%.2f",
                                    scaling_score, retrieval_score, long_context_score)
                    except Exception as e:
                        logger.debug("Long-context sweep skipped: %s", e)

                # Noise sensitivity (informational, non-blocking)
                noise_score = None
                if best_seed is not None:
                    try:
                        from ..eval.noise_sensitivity import evaluate_noise_sensitivity
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _spec_ns = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc_ns = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len)
                            ns_model = build_model(_spec_ns, _bc_ns).to(dev)
                        else:
                            ns_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len).to(dev)
                        ns_batches = [
                            torch.randint(0, config.vocab_size,
                                          (2, min(128, config.validation_seq_len)),
                                          device=dev)
                            for _ in range(4)
                        ]
                        ns_result = evaluate_noise_sensitivity(
                            ns_model, ns_batches, dev)
                        noise_score = ns_result.get("noise_sensitivity_score")
                        del ns_model
                    except Exception as e:
                        logger.debug("Noise sensitivity skipped: %s", e)

                # Activation sparsity analysis (informational, non-blocking)
                activation_sparsity_score = None
                dead_neuron_ratio = None
                if best_seed is not None:
                    try:
                        from ..eval.sparsity import evaluate_activation_sparsity
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _spec_as = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc_as = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len)
                            as_model = build_model(_spec_as, _bc_as).to(dev)
                        else:
                            as_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len).to(dev)
                        as_batches = [
                            torch.randint(0, config.vocab_size,
                                          (2, min(128, config.validation_seq_len)),
                                          device=dev)
                            for _ in range(4)
                        ]
                        as_result = evaluate_activation_sparsity(
                            as_model, as_batches, dev)
                        activation_sparsity_score = as_result.get("activation_sparsity_score")
                        dead_neuron_ratio = as_result.get("dead_neuron_ratio")
                        del as_model
                    except Exception as e:
                        logger.debug("Activation sparsity eval skipped: %s", e)

                # Routing heatmap / collapse detection (informational, non-blocking)
                routing_collapse_score = None
                if best_seed is not None:
                    try:
                        from ..eval.routing_heatmap import evaluate_routing_heatmap
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _spec_rh = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc_rh = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len)
                            rh_model = build_model(_spec_rh, _bc_rh).to(dev)
                        else:
                            rh_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len).to(dev)
                        rh_batches = [
                            torch.randint(0, config.vocab_size,
                                          (2, min(128, config.validation_seq_len)),
                                          device=dev)
                            for _ in range(4)
                        ]
                        rh_result = evaluate_routing_heatmap(
                            rh_model, rh_batches, dev)
                        if rh_result.get("has_routing"):
                            routing_collapse_score = rh_result.get("routing_collapse_score")
                        del rh_model
                    except Exception as e:
                        logger.debug("Routing heatmap eval skipped: %s", e)

                # WikiText perplexity (informational, non-blocking)
                wikitext_perplexity = None
                wikitext_score = None
                if best_seed is not None:
                    try:
                        from ..eval.wikitext_eval import evaluate_wikitext_perplexity
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _spec_wt = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc_wt = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len)
                            wt_model = build_model(_spec_wt, _bc_wt).to(dev)
                        else:
                            wt_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len).to(dev)
                        wt_result = evaluate_wikitext_perplexity(
                            wt_model, config.vocab_size, dev,
                            n_train_steps=200, seq_len=min(128, config.validation_seq_len))
                        wikitext_perplexity = wt_result.get("wikitext_perplexity")
                        wikitext_score = wt_result.get("wikitext_score")
                        if wikitext_perplexity is not None:
                            logger.info("WikiText ppl=%.1f score=%.3f",
                                        wikitext_perplexity, wikitext_score or 0)
                        del wt_model
                    except Exception as e:
                        logger.debug("WikiText eval skipped: %s", e)

                # TinyStories validation (informational, non-blocking)
                tinystories_perplexity = None
                tinystories_score = None
                if best_seed is not None:
                    try:
                        from ..eval.tinystories_eval import evaluate_tinystories
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _spec_ts = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc_ts = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len)
                            ts_model = build_model(_spec_ts, _bc_ts).to(dev)
                        else:
                            ts_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len).to(dev)
                        ts_result = evaluate_tinystories(
                            ts_model, config.vocab_size, dev,
                            n_train_steps=200, seq_len=min(128, config.validation_seq_len))
                        tinystories_perplexity = ts_result.get("tinystories_perplexity")
                        tinystories_score = ts_result.get("tinystories_score")
                        del ts_model
                    except Exception as e:
                        logger.debug("TinyStories eval skipped: %s", e)

                # Cross-task robustness (informational, non-blocking)
                cross_task_score = None
                if best_seed is not None:
                    try:
                        from ..eval.cross_task_eval import evaluate_cross_task_robustness
                        _gjs_ct = graph_json_str
                        _asjs_ct = arch_spec_json_str
                        _ms_ct = model_source
                        _cfg_ct = config
                        def _make_ct_model():
                            if _ms_ct == "morphological_box" and _asjs_ct:
                                _sp = ArchSpec(**json.loads(_asjs_ct))
                                _bc = BuildConfig(
                                    dim=_cfg_ct.model_dim, n_layers=_cfg_ct.n_layers,
                                    vocab_size=_cfg_ct.vocab_size,
                                    max_seq_len=_cfg_ct.validation_seq_len)
                                return build_model(_sp, _bc)
                            return compile_model(
                                [graph_from_json(_gjs_ct)] * _cfg_ct.n_layers,
                                vocab_size=_cfg_ct.vocab_size,
                                max_seq_len=_cfg_ct.validation_seq_len)
                        ct_result = evaluate_cross_task_robustness(
                            _make_ct_model, config.vocab_size, dev,
                            n_train_steps=100, seq_len=min(128, config.validation_seq_len))
                        cross_task_score = ct_result.get("cross_task_score")
                    except Exception as e:
                        logger.debug("Cross-task eval skipped: %s", e)

                # Efficiency wall (informational, non-blocking)
                efficiency_wall_score = None
                max_viable_seq_len = None
                scaling_regime = None
                if best_seed is not None:
                    try:
                        from ..eval.efficiency_wall import evaluate_efficiency_wall
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _spec_ew = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc_ew = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size, max_seq_len=1024)
                            ew_model = build_model(_spec_ew, _bc_ew).to(dev)
                        else:
                            ew_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size, max_seq_len=1024).to(dev)
                        ew_result = evaluate_efficiency_wall(
                            ew_model, config.vocab_size, dev,
                            seq_lens=(64, 128, 256, 512), batch_size=2)
                        efficiency_wall_score = ew_result.get("efficiency_wall_score")
                        max_viable_seq_len = ew_result.get("max_viable_seq_len")
                        scaling_regime = ew_result.get("scaling_regime")
                        del ew_model
                    except Exception as e:
                        logger.debug("Efficiency wall eval skipped: %s", e)

                tier = "breakthrough" if is_breakthrough else "validation"

                validation_entry = {
                    "result_id": source_result_id,
                    "val_loss_ratio": val_loss_ratio,
                    "val_baseline_ratio": val_baseline_ratio,
                    "val_normalized_ratio": val_normalized_ratio,
                    "param_efficiency": val_param_efficiency,
                    "multi_seed_std": multi_seed_std,
                    "seeds_passed": len(passed_seeds),
                    "total_seeds": config.validation_n_seeds,
                    "is_breakthrough": is_breakthrough,
                    "flop_gated": flop_gated,
                    "quant_int8_retention": quant_int8_retention,
                    "quant_quality_per_byte": quant_quality_per_byte,
                    "long_context_score": long_context_score,
                    "noise_sensitivity_score": noise_score,
                    "init_sensitivity_std": init_sensitivity_std,
                    "novelty_confidence": nov_conf,
                    "ood_robustness": ood_result,
                    "sensitivity": sensitivity_result,
                    "activation_sparsity_score": activation_sparsity_score,
                    "dead_neuron_ratio": dead_neuron_ratio,
                    "routing_collapse_score": routing_collapse_score,
                    "wikitext_perplexity": wikitext_perplexity,
                    "wikitext_score": wikitext_score,
                    "tinystories_perplexity": tinystories_perplexity,
                    "tinystories_score": tinystories_score,
                    "cross_task_score": cross_task_score,
                    "efficiency_wall_score": efficiency_wall_score,
                    "max_viable_seq_len": max_viable_seq_len,
                    "scaling_regime": scaling_regime,
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
                            normalized_baseline_ratio=val_normalized_ratio,
                            param_efficiency=val_param_efficiency,
                            quant_int8_retention=quant_int8_retention,
                            quant_quality_per_byte=quant_quality_per_byte,
                            robustness_long_ctx_score=long_context_score,
                            robustness_noise_score=noise_score,
                            init_sensitivity_std=init_sensitivity_std,
                            fp_jacobian_spectral_norm=source.get("fp_jacobian_spectral_norm"),
                            scaling_param_efficiency=scaling_param_efficiency,
                            scaling_flop_efficiency=scaling_flop_efficiency,
                            scaling_gate_passed=scaling_gate_passed_val,
                            scaling_best_family=scaling_best_family,
                            scaling_confidence=scaling_confidence,
                            activation_sparsity_score=activation_sparsity_score,
                            dead_neuron_ratio=dead_neuron_ratio,
                            routing_collapse_score=routing_collapse_score,
                            wikitext_perplexity=wikitext_perplexity,
                            wikitext_score=wikitext_score,
                            tinystories_perplexity=tinystories_perplexity,
                            tinystories_score=tinystories_score,
                            cross_task_score=cross_task_score,
                            efficiency_wall_score=efficiency_wall_score,
                            max_viable_seq_len=max_viable_seq_len,
                            scaling_regime=scaling_regime,
                        )
                        # Store detailed scaling result in external_benchmarks_json
                        if scaling_result is not None:
                            nb.set_external_benchmarks(
                                source_result_id, scaling_result.to_dict())
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
                    novelty_raw_score=source.get("novelty_raw_score"),
                    novelty_z_score=source.get("novelty_z_score"),
                    novelty_reference_version=source.get("novelty_reference_version"),
                    novelty_valid_for_promotion=source.get("novelty_valid_for_promotion"),
                    novelty_validity_reason=source.get("novelty_validity_reason"),
                    novelty_requires_justification=source.get("novelty_requires_justification"),
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
            results["perf_report"] = self._build_experiment_perf_report(results)
            results["perf_budget_gate"] = evaluate_perf_budget_gate(results["perf_report"])
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

        # Z7: Sparsity Ledger
        sparse_ops = {"block_sparse_linear", "nm_sparse_linear", "semi_structured_2_4_linear"}
        dense_ops = {"linear_proj", "linear_proj_down", "linear_proj_up", "fused_linear_gelu"}
        n_sparse = sum(1 for node in graph.nodes.values() if node.op_name in sparse_ops)
        n_dense = sum(1 for node in graph.nodes.values() if node.op_name in dense_ops)
        total_param_ops = n_sparse + n_dense
        metrics["sparsity_ratio"] = n_sparse / total_param_ops if total_param_ops > 0 else 0.0

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

        # Activation Sparsity & Heatmaps
        activation_sparsity = getattr(sandbox_result, "activation_sparsity", None)
        dead_neuron_count = getattr(sandbox_result, "dead_neuron_count", None)
        sparsity_report = getattr(sandbox_result, "sparsity_report", None)
        if activation_sparsity is not None:
            metrics["sparsity_ratio"] = activation_sparsity
        if dead_neuron_count is not None:
            metrics["dead_neuron_count"] = dead_neuron_count
        if sparsity_report:
            metrics["sparsity_report_json"] = json.dumps(sparsity_report)

        # Parse output_range "[min, max]" string
        if sandbox_result.output_range:
            try:
                parts = sandbox_result.output_range.strip("[]").split(",")
                metrics["output_range_min"] = float(parts[0].strip())
                metrics["output_range_max"] = float(parts[1].strip())
            except (ValueError, IndexError):
                pass

        return metrics

    def _extract_architecture_telemetry(self, model: Optional[nn.Module]) -> Dict:
        """Extract sparse, routing, and adaptive telemetry from compiled layer ops."""
        if model is None:
            return {}

        metrics: Dict[str, Any] = {}
        try:
            layers = list(getattr(model, "layers", []) or [])
        except Exception:
            layers = []
        if not layers:
            try:
                topo = getattr(model, "topology", None)
                blocks = getattr(topo, "blocks", None) if topo is not None else None
                if blocks is not None:
                    layers = list(blocks)
            except Exception:
                pass
        routing_mode = None
        spec = getattr(model, "spec", None)
        if spec is not None:
            choices = getattr(spec, "choices", None)
            if isinstance(choices, dict):
                routing_mode = choices.get("compute_routing")
        if routing_mode:
            metrics["routing_mode"] = routing_mode
        
        # 1. Sparse Telemetry
        telemetry_rows: List[Dict[str, Any]] = []
        total_calls = 0
        total_fallback_calls = 0
        kernel_fallback_calls = 0
        density_sum = 0.0
        density_last_values: List[float] = []
        nm_compliant = 0
        nm_total = 0
        sparse_active_params_estimate = 0.0

        # 2. Routing Telemetry (MoE)
        rt_tokens_total = 0
        rt_tokens_processed = 0
        rt_entropy_sum = 0.0
        rt_count = 0
        rt_expert_counts: Optional[torch.Tensor] = None
        
        # 3. Adaptive Telemetry (MoD/MoR)
        at_savings_sum = 0.0
        at_depth_sum = 0.0
        at_count = 0
        recursion_savings_sum = 0.0
        recursion_depth_sum = 0.0
        recursion_count = 0
        recursion_max_depth_sum = 0.0

        for layer in layers:
            # Check for routing/adaptive telemetry on the layer/routing itself (arch_builder style)
            routing = getattr(layer, "routing", None)
            if routing is not None:
                # Routing (MoE)
                rt = getattr(routing, "routing_telemetry", None)
                if isinstance(rt, dict):
                    rt_tokens_total += rt.get("tokens_total", 0)
                    rt_tokens_processed += rt.get("tokens_processed", 0)
                    rt_entropy_sum += rt.get("entropy_sum", 0.0)
                    rt_count += rt.get("count", 0)
                    ec = rt.get("expert_counts")
                    if isinstance(ec, torch.Tensor):
                        if rt_expert_counts is None: rt_expert_counts = ec.clone()
                        else: rt_expert_counts += ec
                
                # Adaptive (MoD/MoR)
                at = getattr(routing, "adaptive_telemetry", None)
                if isinstance(at, dict):
                    at_savings_sum += at.get("savings_sum", 0.0)
                    at_depth_sum += at.get("depth_sum", 0.0)
                    at_count += at.get("count", 0)
                    if routing.__class__.__name__ == "AdaptiveRecursionRouting":
                        recursion_savings_sum += at.get("savings_sum", 0.0)
                        recursion_depth_sum += at.get("depth_sum", 0.0)
                        recursion_count += at.get("count", 0)
                        recursion_max_depth_sum += float(getattr(routing, "max_depth", 0)) * at.get("count", 0)

            # Check for op-level telemetry (compiler style)
            ops = getattr(layer, "ops", None)
            if ops is None:
                continue
            op_values = None
            if isinstance(ops, dict):
                op_values = list(ops.values())
            else:
                try:
                    op_values = list(ops)
                except Exception:
                    # Guard against non-iterable op containers
                    continue
            for compiled_op in op_values:
                # Sparse
                sparse_telemetry = getattr(compiled_op, "sparse_telemetry", None)
                if sparse_telemetry:
                    has_weight = hasattr(compiled_op, "weight")
                    weight_params = float(compiled_op.weight.numel()) if has_weight else 0.0
                    for op_name, stats in sparse_telemetry.items():
                        calls = int(stats.get("calls", 0) or 0)
                        total_calls += calls
                        total_fallback_calls += int(stats.get("fallback_calls", 0) or 0)
                        density_sum += float(stats.get("density_sum", 0.0) or 0.0)
                        last_density = float(stats.get("last_density", 1.0) or 1.0)
                        density_last_values.append(last_density)
                        if stats.get("last_fallback_reason") == "kernel_unavailable":
                            kernel_fallback_calls += int(stats.get("fallback_calls", 0) or 0)
                        if op_name in ("nm_sparse_linear", "semi_structured_2_4_linear"):
                            nm_total += 1
                            if last_density <= 0.51: nm_compliant += 1
                        if weight_params > 0.0:
                            density_for_params = (float(stats.get("density_sum", 0.0)) / calls) if calls > 0 else last_density
                            sparse_active_params_estimate += weight_params * density_for_params
                        telemetry_rows.append({"op_name": op_name, "calls": calls, "last_density": last_density})

                # Routing (MoE)
                rt = getattr(compiled_op, "routing_telemetry", None)
                if isinstance(rt, dict):
                    rt_tokens_total += rt.get("tokens_total", 0)
                    rt_tokens_processed += rt.get("tokens_processed", 0)
                    rt_entropy_sum += rt.get("entropy_sum", 0.0)
                    rt_count += rt.get("count", 0)
                    ec = rt.get("expert_counts")
                    if isinstance(ec, torch.Tensor):
                        if rt_expert_counts is None: rt_expert_counts = ec.clone()
                        else: rt_expert_counts += ec

                # Adaptive
                at = getattr(compiled_op, "adaptive_telemetry", None)
                if isinstance(at, dict):
                    at_savings_sum += at.get("savings_sum", 0.0)
                    at_depth_sum += at.get("depth_sum", 0.0)
                    at_count += at.get("count", 0)

        # Finalize Sparse
        if total_calls > 0:
            metrics["sparse_density_mean"] = density_sum / max(total_calls, 1)
            metrics["sparse_density_last"] = sum(density_last_values) / max(len(density_last_values), 1)
            metrics["sparse_fallback_calls"] = total_fallback_calls
            metrics["sparse_kernel_fallback_calls"] = kernel_fallback_calls
            metrics["sparse_active_params_estimate"] = int(max(0.0, sparse_active_params_estimate))
            metrics["sparse_telemetry_json"] = json.dumps(telemetry_rows)
            if nm_total > 0: metrics["sparse_nm_compliance"] = nm_compliant / nm_total
            # Compression ratio = effective params / dense params
            if sparse_active_params_estimate > 0:
                total_weight_params = sum(
                    float(getattr(op, "weight", torch.empty(0)).numel())
                    for layer_ops in layers for op in layer_ops.values()
                    if hasattr(op, "weight")
                )
                if total_weight_params > 0:
                    metrics["compression_ratio"] = sparse_active_params_estimate / total_weight_params

        # Infer routing_mode from compiled ops if not already set
        if not routing_mode and rt_count > 0:
            for layer in layers:
                ops = getattr(layer, "ops", None)
                if ops is None:
                    continue
                if isinstance(ops, dict):
                    op_values = list(ops.values())
                else:
                    try:
                        op_values = list(ops)
                    except Exception:
                        continue
                for compiled_op in op_values:
                    op_obj = getattr(compiled_op, "op", None)
                    op_name = getattr(op_obj, "name", "") if op_obj else ""
                    if op_name == "moe_2expert":
                        routing_mode = "moe_2expert"
                        break
                    elif op_name == "moe_topk":
                        routing_mode = "moe_topk"
                        break
                    elif op_name == "topk_gate":
                        routing_mode = "topk_gate"
                        break
                    elif op_name in {
                        "mod_topk", "early_exit", "adaptive_recursion",
                        "token_merging", "token_merge", "cascade",
                        "speculative", "route_topk", "route_lanes", "route_recursion",
                    }:
                        routing_mode = op_name
                        break
                if routing_mode:
                    break
            if routing_mode:
                metrics["routing_mode"] = routing_mode
        if rt_count > 0 and not routing_mode:
            routing_mode = "routed"
            metrics["routing_mode"] = routing_mode

        # Finalize Routing
        if rt_count > 0:
            metrics["routing_tokens_total"] = rt_tokens_total
            metrics["routing_tokens_processed"] = rt_tokens_processed
            metrics["routing_utilization_entropy"] = rt_entropy_sum / rt_count
            if rt_tokens_total > 0:
                metrics["routing_drop_rate"] = max(0.0, 1.0 - (rt_tokens_processed / rt_tokens_total))
                metrics["routing_savings_ratio"] = rt_tokens_processed / rt_tokens_total
            if rt_expert_counts is not None:
                metrics["routing_expert_count"] = int(len(rt_expert_counts))
                metrics["routing_expert_utilization_json"] = json.dumps(rt_expert_counts.cpu().tolist())

        # Finalize Adaptive
        if at_count > 0:
            metrics["depth_savings_ratio"] = at_savings_sum / at_count
            if at_depth_sum > 0:
                metrics["effective_depth_ratio"] = at_depth_sum / (at_count * len(layers)) if len(layers) > 0 else 1.0
        if recursion_count > 0:
            metrics["recursion_savings_ratio"] = recursion_savings_sum / recursion_count
            if recursion_depth_sum > 0:
                avg_max_depth = recursion_max_depth_sum / recursion_count if recursion_max_depth_sum > 0 else None
                if avg_max_depth and avg_max_depth > 0:
                    metrics["recursion_depth_ratio"] = recursion_depth_sum / (recursion_count * avg_max_depth)

        return metrics

    @staticmethod
    def _merge_s1_telemetry(program_metrics: Dict[str, Any], s1_result: Dict[str, Any]) -> None:
        telemetry_keys = (
            "routing_mode",
            "routing_tokens_total",
            "routing_tokens_processed",
            "routing_tokens_skipped",
            "routing_drop_rate",
            "routing_utilization_entropy",
            "routing_capacity_overflow_count",
            "routing_confidence_mean",
            "routing_confidence_std",
            "routing_expert_utilization_json",
            "routing_expert_count",
            "routing_savings_ratio",
            "compression_ratio",
            "depth_savings_ratio",
            "effective_depth_ratio",
            "recursion_savings_ratio",
            "recursion_depth_ratio",
        )
        for key in telemetry_keys:
            if key in s1_result and s1_result.get(key) is not None:
                program_metrics[key] = s1_result.get(key)

    def _on_program_evaluated(self, graph, fitness, sandbox_result, s1_result, 
                              eval_counters, nb, exp_id, model_source="evolution"):
        """Unified callback for recording results and updating counters during search."""
        eval_counters["total"] += 1
        if fitness > 0:
            eval_counters["s0"] += 1
        if fitness > 0.2:
            eval_counters["s1"] += 1
            
        try:
            graph_metrics = self._extract_graph_metrics(graph)
            
            # Extract sandbox metrics if available
            if sandbox_result:
                graph_metrics.update(self._extract_sandbox_metrics(sandbox_result))
                
            # Extract S1 and architecture telemetry if available
            if s1_result:
                # Basic training metrics
                for k in ("initial_loss", "final_loss", "min_loss", "throughput",
                          "avg_step_time_ms", "total_train_time_ms",
                          "validation_loss", "validation_loss_ratio", "generalization_gap",
                          "discovery_loss", "discovery_loss_ratio"):
                    if k in s1_result: graph_metrics[k] = s1_result[k]
                self._merge_s1_telemetry(graph_metrics, s1_result)

            nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=graph.fingerprint(),
                graph_json=graph_to_json(graph),
                stage1_passed=fitness > 0.2,
                stage0_passed=fitness > 0,
                stage05_passed=fitness > 0,
                loss_ratio=1.0 - fitness if fitness > 0 else None,
                novelty_score=None,
                novelty_confidence=0.2,
                stage_at_death="survived" if fitness > 0.2 else "stage1",
                model_source=model_source,
                **graph_metrics,
            )
        except Exception as e:
            logger.debug("Failed to record program result: %s", e)

    def _process_orchestrator_results(self, orchestrator, nb, exp_id, results, config):
        """Collect and record all available results from the orchestrator."""
        job_results = orchestrator.get_results()
        if not job_results:
            return
        with nb.batch():
            for jr in job_results:
                self._record_orchestrator_result(jr, nb, exp_id, results, config)

    def _build_experiment_perf_report(
        self,
        results: Dict[str, Any],
        queue_telemetry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Aggregate per-program perf traces into one experiment-level JSON report."""
        perf_traces = results.get("_perf_traces", []) or []
        starvation = results.get("_gpu_starvation", []) or []
        kernel_samples = results.get("_kernel_timing", []) or []

        trace_totals: Dict[str, float] = {}
        trace_counts: Dict[str, int] = {}
        for trace_report in perf_traces:
            summary = (trace_report or {}).get("summary_ms", {})
            if not isinstance(summary, dict):
                continue
            for name, value in summary.items():
                try:
                    val = float(value)
                except Exception:
                    continue
                trace_totals[name] = trace_totals.get(name, 0.0) + val
                trace_counts[name] = trace_counts.get(name, 0) + 1

        trace_avg_ms = {
            name: round(trace_totals[name] / max(1, trace_counts[name]), 4)
            for name in sorted(trace_totals.keys())
        }

        # Aggregate throughput
        throughput_vals = [
            float(t.get("avg_throughput_tok_s", 0.0) or 0.0)
            for t in perf_traces
            if t.get("avg_throughput_tok_s") is not None
        ]
        avg_throughput = sum(throughput_vals) / len(throughput_vals) if throughput_vals else 0.0

        starvation_count = 0
        starvation_total_ms = 0.0
        starvation_max_ms = 0.0
        for item in starvation:
            if not isinstance(item, dict):
                continue
            starvation_count += int(item.get("count", 0) or 0)
            starvation_total_ms += float(item.get("total_stall_ms", 0.0) or 0.0)
            starvation_max_ms = max(starvation_max_ms, float(item.get("max_stall_ms", 0.0) or 0.0))

        op_totals: Dict[str, Dict[str, float]] = {}
        for sample in kernel_samples:
            if not isinstance(sample, dict):
                continue
            
            # Handle new format: mod_name -> ms (float)
            if "top_ops" not in sample:
                for op_name, ms in sample.items():
                    if isinstance(ms, (int, float)):
                        slot = op_totals.setdefault(op_name, {"cpu_ms": 0.0, "cuda_ms": 0.0, "calls": 0.0, "samples": 0.0})
                        slot["cuda_ms"] += float(ms)
                        slot["samples"] += 1.0
                continue
                
            # Handle old format (top_ops)
            for op in sample.get("top_ops", []) or []:
                op_name = str(op.get("op", "unknown"))
                slot = op_totals.setdefault(op_name, {"cpu_ms": 0.0, "cuda_ms": 0.0, "calls": 0.0, "samples": 0.0})
                slot["cpu_ms"] += float(op.get("cpu_ms", 0.0) or 0.0)
                slot["cuda_ms"] += float(op.get("cuda_ms", 0.0) or 0.0)
                slot["calls"] += float(op.get("calls", 0.0) or 0.0)
                slot["samples"] += 1.0

        hotspot_ops = []
        for op_name, agg in op_totals.items():
            samples = max(1.0, agg["samples"])
            hotspot_ops.append({
                "op": op_name,
                "avg_cpu_ms": round(agg["cpu_ms"] / samples, 4),
                "avg_cuda_ms": round(agg["cuda_ms"] / samples, 4),
                "avg_calls": round(agg["calls"] / samples, 2),
            })
        hotspot_ops.sort(key=lambda row: max(row["avg_cuda_ms"], row["avg_cpu_ms"]), reverse=True)

        tp_sched_rows = results.get("training_program_scheduling", []) or []
        tp_avg_ms = [float(r.get("scheduling_avg_ms", 0.0) or 0.0) for r in tp_sched_rows]
        tp_max_ms = [float(r.get("scheduling_max_ms", 0.0) or 0.0) for r in tp_sched_rows]

        return {
            "generated_at": time.time(),
            "programs_profiled": len(perf_traces),
            "trace_avg_ms": trace_avg_ms,
            "avg_throughput_tok_s": round(avg_throughput, 2),
            "gpu_starvation": {
                "event_count": starvation_count,
                "total_stall_ms": round(starvation_total_ms, 4),
                "max_stall_ms": round(starvation_max_ms, 4),
            },
            "kernel_hotspots": hotspot_ops[:10],
            "queue_telemetry": queue_telemetry or {},
            "training_program_scheduling": {
                "n_sources": len(tp_sched_rows),
                "avg_schedule_ms": round(sum(tp_avg_ms) / len(tp_avg_ms), 4) if tp_avg_ms else 0.0,
                "max_schedule_ms": round(max(tp_max_ms), 4) if tp_max_ms else 0.0,
            },
        }

    def _record_orchestrator_result(self, jr, nb, exp_id, results, config):
        """Record a single result from the orchestrator into the notebook."""
        s1_result = jr.s1_result
        program_metrics = jr.payload["metrics"]
        graph = jr.payload["graph"]
        i = jr.index
        
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
        program_metrics["validation_loss"] = s1_result.get("validation_loss")
        program_metrics["validation_loss_ratio"] = s1_result.get("validation_loss_ratio")
        program_metrics["generalization_gap"] = s1_result.get("generalization_gap")
        program_metrics["discovery_loss"] = s1_result.get("discovery_loss")
        program_metrics["discovery_loss_ratio"] = s1_result.get("discovery_loss_ratio")
        program_metrics.update({k: s1_result.get(k) for k in s1_result if k.startswith("pruning_")})
        self._merge_s1_telemetry(program_metrics, s1_result)
        
        # Merge traces
        perf_report = s1_result.get("perf_report", s1_result.get("perf_traces"))
        if perf_report:
            program_metrics["perf_report_json"] = json.dumps(perf_report)
            results.setdefault("_perf_traces", []).append(perf_report)
            
        starvation_report = s1_result.get("starvation_report", s1_result.get("gpu_starvation"))
        if starvation_report:
            program_metrics["starvation_report_json"] = json.dumps(starvation_report)
            results.setdefault("_gpu_starvation", []).append(starvation_report)
            
        kernel_timings = s1_result.get("kernel_timings_ms", s1_result.get("kernel_timing"))
        if kernel_timings:
            program_metrics["kernel_timings_json"] = json.dumps(kernel_timings)
            results.setdefault("_kernel_timing", []).append(kernel_timings)
            
        if getattr(jr, "telemetry", None):
            program_metrics["queue_telemetry_json"] = json.dumps(jr.telemetry)

        if s1_passed:
            results["stage1_passed"] += 1
            with self._lock:
                self._progress.stage1_passed += 1

            logger.info(
                "  ★ S1 SURVIVOR [%d] %s — loss_ratio=%.4f, params=%s",
                i + 1, graph.fingerprint()[:10],
                loss_ratio or 0,
                f"{program_metrics.get('param_count', 0):,}",
            )

            # Compare to baseline (dual-metric: discovery vs validation)
            if final_loss is not None:
                try:
                    baseline = self._get_baseline()
                    baseline_steps = int(s1_result.get("n_train_steps") or config.stage1_steps)
                    baseline_recipe = self._resolve_baseline_recipe(s1_result, default_lr=config.stage1_lr)
                    
                    # 1. Discovery Baseline (Random Tokens)
                    discovery_loss = s1_result.get("discovery_loss")
                    if discovery_loss is not None:
                        try:
                            discovery_steps = min(5, baseline_steps // 10)
                            discovery_ratio = baseline.compare(
                                discovery_loss,
                                d_model=config.model_dim,
                                seq_len=min(128, config.max_seq_len),
                                n_steps=max(1, discovery_steps),
                                vocab_size=config.vocab_size,
                                batch_size=config.stage1_batch_size,
                                lr=baseline_recipe["lr"],
                                device=str(config.device),
                                n_layers=2,
                                data_mode="random",
                                data_tag="discovery_baseline",
                            )
                            program_metrics["discovery_loss_ratio"] = discovery_ratio
                        except Exception as e:
                            logger.debug("Discovery baseline failed: %s", e)

                    # 2. Validation Baseline (Corpus)
                    val_loss = s1_result.get("validation_loss")
                    if val_loss is not None:
                        try:
                            v_data_fn, v_data_tag, v_cache = self._make_baseline_data_fn(config, split="val")
                            v_baseline_ratio = baseline.compare(
                                val_loss,
                                d_model=config.model_dim,
                                seq_len=min(128, config.max_seq_len),
                                n_steps=max(1, baseline_steps),
                                vocab_size=config.vocab_size,
                                batch_size=config.stage1_batch_size,
                                lr=baseline_recipe["lr"],
                                device=str(config.device),
                                n_layers=2,
                                data_fn=v_data_fn,
                                data_mode="corpus",
                                data_tag=v_data_tag,
                                cache_data_fn=v_cache,
                            )
                            program_metrics["validation_baseline_loss_ratio"] = v_baseline_ratio
                            program_metrics["validation_loss_ratio"] = v_baseline_ratio
                        except Exception:
                            pass

                    # 3. Standard Baseline (for backward compatibility / fallback)
                    baseline_ratio = baseline.compare(
                        final_loss,
                        d_model=config.model_dim,
                        seq_len=min(128, config.max_seq_len),
                        n_steps=max(1, baseline_steps),
                        vocab_size=config.vocab_size,
                        batch_size=config.stage1_batch_size,
                        lr=baseline_recipe["lr"],
                        device=str(config.device),
                        n_layers=2,
                        data_mode="corpus" if val_loss is not None else "random",
                        data_tag="standard_baseline",
                    )
                    program_metrics["baseline_loss_ratio"] = baseline_ratio
                except Exception:
                    pass

            # Z12: Diagnostic suite — record metrics for S1 survivors (informational only).
            # The regression gate is NOT applied at screening tier because a single-layer
            # model trained for 50 steps cannot learn the copy/induction tasks the gate
            # requires. The gate should only be applied at investigation/validation tiers
            # where multi-layer models are trained for longer.
            try:
                diag_dev = str(config.device) if torch.cuda.is_available() else "cpu"
                diag_model = compile_model([graph], vocab_size=config.vocab_size, max_seq_len=64)
                diag_result = run_diagnostic_suite(diag_model, device=diag_dev, n_steps=50)
                program_metrics["diagnostic_score"] = diag_result.diagnostic_score
                program_metrics["diagnostic_tasks_json"] = json.dumps(diag_result.to_dict())
            except Exception as e:
                logger.debug("Diagnostic suite failed for %s: %s", graph.fingerprint()[:10], e)

        # Novelty scoring for S1 survivors
        n_score = None
        nov = None
        if s1_passed:
            try:
                fp = None
                fp_dict = s1_result.get("_behavioral_fingerprint")
                if fp_dict is not None:
                    # Option B: reconstruct behavioral fingerprint from S1 worker
                    fp = BehavioralFingerprint()
                    for k, v in fp_dict.items():
                        if hasattr(fp, k):
                            setattr(fp, k, v)

                    calibration_row = self._ensure_novelty_calibration(nb, config, fp)
                    calibration = None
                    if calibration_row:
                        calibration = {
                            "noise_floor_mean": calibration_row.get("noise_floor_mean"),
                            "noise_floor_std": calibration_row.get("noise_floor_std"),
                        }
                    nov = novelty_score(graph, fingerprint=fp, calibration=calibration)
                else:
                    # Option A fallback: structural-only novelty
                    nov = novelty_score(graph)

                n_score = nov.overall_novelty
                novelty_valid, novelty_valid_reason, novelty_requires_justification = (
                    self._resolve_novelty_promotion_validity(
                        config,
                        nov.novelty_valid_for_promotion,
                        nov.novelty_validity_reason,
                    )
                )
                program_metrics["novelty_raw_score"] = nov.raw_novelty
                program_metrics["novelty_z_score"] = nov.novelty_z_score
                program_metrics["novelty_reference_version"] = (
                    nov.novelty_reference_version
                    or (fp.novelty_reference_version if fp is not None else None)
                )
                program_metrics["novelty_valid_for_promotion"] = int(novelty_valid)
                program_metrics["novelty_validity_reason"] = novelty_valid_reason
                program_metrics["novelty_requires_justification"] = int(
                    novelty_requires_justification
                )
            except Exception as e:
                logger.debug("Novelty scoring failed for %s: %s", graph.fingerprint()[:10], e)

        # Record result
        novelty_kwargs = {}
        if nov is not None:
            novelty_kwargs = dict(
                novelty_score=n_score,
                structural_novelty=nov.structural_novelty,
                behavioral_novelty=nov.behavioral_novelty,
                most_similar_to=nov.most_similar_to,
                novelty_confidence=nov.novelty_confidence,
            )
        rid = nb.record_program_result(
            experiment_id=exp_id,
            graph_fingerprint=graph.fingerprint(),
            graph_json=graph_to_json(graph),
            stage0_passed=True,
            stage05_passed=True,
            stage1_passed=s1_passed,
            final_loss=final_loss,
            loss_ratio=loss_ratio,
            throughput_tok_s=throughput,
            **novelty_kwargs,
            **program_metrics,
        )

        if training_curve and rid:
            try:
                nb.store_training_curve(rid, training_curve)
            except Exception:
                pass

        # Update best metrics in experiment summary
        if loss_ratio is not None:
            if results["best_loss_ratio"] is None or loss_ratio < results["best_loss_ratio"]:
                results["best_loss_ratio"] = loss_ratio
        
        try:
            # We can't easily recompute global novelty here without all graphs,
            # but we can take the individual score if it was computed.
            nov = program_metrics.get("novelty_score")
            if nov is not None:
                if results["best_novelty_score"] is None or nov > results["best_novelty_score"]:
                    results["best_novelty_score"] = nov
        except Exception:
            pass

        self._emit_event("program_evaluated", {
            "index": i, "fingerprint": graph.fingerprint()[:10],
            "result": "pass" if s1_passed else "fail",
            "loss_ratio": f"{loss_ratio:.4f}" if loss_ratio is not None else None,
            "result_id": rid,
            "throughput": f"{throughput:.0f}" if throughput else None,
            "params": program_metrics.get("param_count"),
            "memory_mb": f"{program_metrics.get('peak_memory_mb', 0):.1f}" if program_metrics.get("peak_memory_mb") else None,
            "novelty": f"{program_metrics.get('novelty_score', 0):.3f}" if program_metrics.get("novelty_score") is not None else None,
        })

    def _execute_experiment(self, exp_id: str, config: RunConfig,
                            nb: LabNotebook,
                            use_learned_grammar: bool = True) -> Dict:
        """Core experiment logic shared by single and continuous modes."""
        with self._lock:
            # Z17: Explicitly reset progress object at start of execution
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=config.n_programs,
                aria_message=f"{self.aria.NAME}: Initializing experiment {exp_id[:8]}...",
            )
            
        results = {
            "total": 0, "stage0_passed": 0, "stage05_passed": 0,
            "stage1_passed": 0, "novel_count": 0,
            "best_loss_ratio": None, "best_novelty_score": None,
            "survivors": [],
            "skipped_proactive_gating": 0,
            "proactive_gating_failures": [],
        }

        grammar_weights = None
        excluded_ops: set = set()
        op_weights: Dict[str, float] = {}
        failure_blocklist: Dict[str, float] = {}
        champion_bias: Dict[str, float] = {}
        analytics = None
        grammar_gate: Optional[Dict[str, Any]] = None
        if use_learned_grammar:
            try:
                from .analytics import ExperimentAnalytics
                analytics = ExperimentAnalytics(nb)
                last_effective = nb.load_last_effective_weights()
                last_weights = last_effective[0] if last_effective else None
                grammar_weights = analytics.compute_grammar_weights(
                    last_applied=last_weights, alpha=0.6
                )
                if grammar_weights:
                    grammar_gate = self._evaluate_grammar_update_gate(
                        nb=nb,
                        analytics=analytics,
                        config=config,
                    )
                    if not grammar_gate.get("gate_pass"):
                        nb.log_learning_event(
                            "grammar_weights_blocked",
                            f"Blocked grammar weight update for {exp_id}: weak attribution evidence",
                            evidence=json.dumps(grammar_gate, sort_keys=True),
                        )
                        grammar_weights = None
            except Exception as e:
                logger.warning("Failed computing learned grammar weights for %s: %s", exp_id, e)

            # Populate excluded_ops and soft-penalty op_weights from negative results
            op_weights: Dict[str, float] = {}
            try:
                if analytics is not None:
                    neg = analytics.negative_results_synthesis()
                    for op_info in neg.get("failed_ops", []):
                        if (op_info.get("s1_rate", 1) == 0
                                and op_info.get("n_used", 0) >= 5
                                and op_info.get("confidence", 0) >= 0.7):
                            excluded_ops.add(op_info["op_name"])
                    # Soft-penalize weak ops (nonzero but poor S1 rate)
                    for op_info in neg.get("weak_ops", []):
                        op_name = op_info.get("op_name", "")
                        penalty = op_info.get("penalty_weight", 1.0)
                        if op_name and op_name not in excluded_ops:
                            op_weights[op_name] = penalty
                    if excluded_ops:
                        nb.log_learning_event(
                            "excluded_ops_applied",
                            f"Excluded {len(excluded_ops)} ops with 0% S1 rate: "
                            f"{', '.join(sorted(excluded_ops))}",
                            excluded_ops=sorted(excluded_ops),
                        )
                    if op_weights:
                        nb.log_learning_event(
                            "weak_ops_penalized",
                            f"Soft-penalized {len(op_weights)} weak ops: "
                            f"{', '.join(f'{k}={v:.2f}' for k, v in sorted(op_weights.items()))}",
                            op_weights=op_weights,
                        )
            except Exception as e:
                logger.warning("Failed computing excluded/weak ops for %s: %s", exp_id, e)

            # Load failure-signature blocklist (op-pair bigrams with high fail rate)
            failure_blocklist: Dict[str, float] = {}
            try:
                failure_blocklist = nb.get_failure_signature_blocklist()
                if failure_blocklist:
                    nb.log_learning_event(
                        "failure_signatures_loaded",
                        f"Loaded {len(failure_blocklist)} toxic op-pair patterns",
                        signatures=sorted(failure_blocklist.keys())[:10],
                    )
            except Exception as e:
                logger.warning("Failed loading failure signatures for %s: %s", exp_id, e)

            # Champion bias pass: nudge category weights toward proven winners.
            # This biases the search toward high-performing projection/sparse patterns
            # and known-good structural/sequence motifs without hard-coding op-level picks.
            try:
                if analytics is not None:
                    op_rates = analytics.op_success_rates() or {}
                    if op_rates:
                        winning_ops = {"exp", "selective_scan", "tropical_center"}
                        projection_ops = {"low_rank_proj", "shared_basis_proj", "tied_proj"}
                        sparse_ops = {"nm_sparse_linear", "block_sparse_linear", "semi_structured_2_4_linear"}

                        def _is_reliable(op_name: str, min_used: int = 10, min_s1: float = 0.25) -> bool:
                            info = op_rates.get(op_name) or {}
                            n_used = int(info.get("n_used") or 0)
                            s1_rate = float(info.get("s1_rate") or 0.0)
                            return n_used >= min_used and s1_rate >= min_s1

                        has_winners = any(_is_reliable(op) for op in winning_ops)
                        has_projection = any(_is_reliable(op) for op in projection_ops)
                        has_sparse = any(_is_reliable(op) for op in sparse_ops)

                        if has_winners:
                            champion_bias["structural"] = max(champion_bias.get("structural", 1.0), 1.2)
                            champion_bias["sequence"] = max(champion_bias.get("sequence", 1.0), 1.2)
                        if has_projection:
                            champion_bias["parameterized"] = max(champion_bias.get("parameterized", 1.0), 1.4)
                        if has_sparse:
                            champion_bias["parameterized"] = max(champion_bias.get("parameterized", 1.0), 1.5)
                            # Z7: If sparse ops are reliable, nudge the grammar hard toward them
                            champion_bias["_structured_sparsity_bias"] = 0.8

            except Exception as e:
                logger.warning("Failed computing champion bias for %s: %s", exp_id, e)

        # Merge Aria's overrides into excluded_ops and op_weights
        excluded_ops = excluded_ops | self._excluded_ops_overrides
        op_weights = {**op_weights, **self._op_weights_overrides}
        grammar = self._build_grammar_config(config, excluded_ops=excluded_ops, op_weights=op_weights)
        old_weights = dict(grammar.category_weights)

        if grammar_weights:
            old_weights = dict(grammar.category_weights)
            grammar.category_weights.update(grammar_weights)
            self._log_grammar_weight_application(
                nb,
                exp_id,
                old_weights,
                dict(grammar.category_weights),
                analytics=analytics,
            )
            # Persist for observability
            results["applied_grammar_weights"] = dict(grammar.category_weights)
            if grammar_gate:
                results["grammar_weight_attribution"] = grammar_gate

        if champion_bias:
            before_bias = dict(grammar.category_weights)
            for category, multiplier in champion_bias.items():
                if category == "_structured_sparsity_bias":
                    grammar.structured_sparsity_bias = float(multiplier)
                    continue
                base = float(grammar.category_weights.get(category, 1.0))
                grammar.category_weights[category] = round(max(0.5, min(8.0, base * multiplier)), 2)
            nb.log_learning_event(
                "champion_bias_applied",
                f"Applied champion grammar bias for {exp_id}",
                multipliers=champion_bias,
                old_weights=before_bias,
                new_weights=dict(grammar.category_weights),
            )
            results["applied_grammar_weights"] = dict(grammar.category_weights)

        # Apply chat-driven grammar weight overrides (from Aria actions)
        if self._grammar_weight_overrides:
            grammar.category_weights.update(self._grammar_weight_overrides)
            nb.log_learning_event(
                "chat_grammar_overrides_applied",
                f"Applied chat-driven grammar overrides for {exp_id}",
                overrides=dict(self._grammar_weight_overrides),
                final_weights=dict(grammar.category_weights),
            )
            results["applied_grammar_weights"] = dict(grammar.category_weights)
            # Emit SSE so LiveFeed can show learning events
            source_weights = grammar_weights or {}
            n_changed = sum(1 for k in source_weights
                            if old_weights.get(k) != source_weights[k])
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
        if config.model_source == "morphological_box":
            # Morphological box evaluation path (arch_builder models, no graph JSON)
            candidates = self._generate_candidates(config, config.n_programs, "morphological_box")
            results["total"] = len(candidates)

            dev_str = config.device if torch.cuda.is_available() else "cpu"
            dev = torch.device(dev_str)

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

                # Stage 0/0.5
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
                s05_passed = (sandbox_result.stability_score >= config.stage05_stability_threshold
                              and sandbox_result.causality_passed)
                if s0_passed:
                    results["stage0_passed"] += 1
                    with self._lock: self._progress.stage0_passed += 1
                if s05_passed:
                    results["stage05_passed"] += 1
                    with self._lock: self._progress.stage05_passed += 1

                if not s0_passed or not s05_passed:
                    continue

                # Stage 1 (sync, since we already have a compiled model)
                s1_result = self._micro_train(
                    model,
                    config,
                    dev,
                    seed=self._stable_seed(exp_id, i, "morphology"),
                )
                s1_passed = bool(s1_result.get("passed", False))
                if s1_passed:
                    results["stage1_passed"] += 1
                    with self._lock: self._progress.stage1_passed += 1

                program_metrics: Dict[str, Any] = {}
                try:
                    program_metrics.update(self._extract_sandbox_metrics(sandbox_result))
                except Exception:
                    pass
                try:
                    program_metrics["param_count"] = sandbox_result.param_count
                except Exception:
                    pass

                # Merge S1 metrics
                for k in ("initial_loss", "final_loss", "min_loss", "loss_ratio",
                          "throughput", "avg_step_time_ms", "total_train_time_ms",
                          "validation_loss", "validation_loss_ratio", "generalization_gap",
                          "discovery_loss", "discovery_loss_ratio"):
                    if k in s1_result:
                        program_metrics[k] = s1_result.get(k)
                self._merge_s1_telemetry(program_metrics, s1_result)

                nb.record_program_result(
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

            return results

        if config.model_source == "fingerprint_refine":
            graphs = self._generate_refinement_graphs(exp_id, config, nb, grammar)
        else:
            # Project Hephaestus Phase 4: Adaptive Synthesis
            prior = None
            use_adaptive = False
            if use_learned_grammar and analytics is not None:
                try:
                    frontier = analytics.get_efficiency_frontier()
                    if frontier:
                        from ..synthesis.grammar import EfficiencyPrior
                        prior = EfficiencyPrior(frontier)
                        use_adaptive = True
                        nb.log_learning_event(
                            "adaptive_synthesis_enabled",
                            f"Enabling budget-aware adaptive synthesis for {exp_id}",
                            frontier_size=len(frontier),
                        )
                except Exception as e:
                    logger.warning("Failed to initialize efficiency prior: %s", e)
            
            graphs = batch_generate(
                config.n_programs, 
                grammar, 
                use_adaptive_synthesis=use_adaptive,
                prior=prior
            )
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

        logger.info(
            "Experiment %s: generated %d graphs (depth=%d, ops=%d, dim=%d, device=%s)",
            exp_id[:8], len(graphs), grammar.max_depth, grammar.max_ops,
            config.model_dim, config.device,
        )

        nb.add_entry(ExperimentEntry(
            entry_type="observation",
            title=f"Generated {len(graphs)} computation graphs",
            content=f"Grammar: depth={grammar.max_depth}, ops={grammar.max_ops}, "
                    f"dim={config.model_dim}, math_space_weight={config.math_space_weight}",
            experiment_id=exp_id,
        ))

        dev_str = config.device if torch.cuda.is_available() else "cpu"
        dev = torch.device(dev_str)

        # Z12: Detect available GPUs for distributed search
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(num_gpus)]
            # 2 workers per GPU usually helps overlap data loading
            num_workers = num_gpus * 2
        else:
            devices = ["cpu"]
            num_workers = 1

        # Z12: Multi-node distributed workers
        remote_workers = [
            w.strip() for w in os.environ.get("ARIA_REMOTE_WORKERS", "").split(",")
            if w.strip()
        ]

        # Z6: Initialize asynchronous program orchestrator
        orchestrator = WorkerPoolOrchestrator(
            train_fn=lambda m, c, s, d: self._micro_train_async(m, c, s, d),
            num_workers=num_workers,
            max_queue_size=config.n_programs,
            devices=devices,
            remote_workers=remote_workers
        )
        candidate_batch_size = max(1, min(32, int(math.sqrt(max(1, config.n_programs)))))
        results["candidate_batch_size"] = candidate_batch_size

        last_log_time = time.time()

        # Dedup: load fingerprints already evaluated in previous experiments
        # to avoid wasting compute re-testing identical architectures.
        try:
            _existing_fps = {
                r[0] for r in nb.conn.execute(
                    "SELECT DISTINCT graph_fingerprint FROM program_results"
                ).fetchall() if r[0]
            }
        except Exception:
            _existing_fps = set()

        # Pre-filter known fingerprints and adaptively generate more if needed
        original_count = len(graphs)
        _dedup_max_rounds = 3
        _dedup_target = max(1, int(original_count * 0.5))  # want at least 50% novel
        for _dedup_round in range(_dedup_max_rounds):
            novel = []
            seen_this_batch = set()
            for g in graphs:
                fp = g.fingerprint()
                if fp not in _existing_fps and fp not in seen_this_batch:
                    novel.append(g)
                    seen_this_batch.add(fp)
            graphs = novel
            if len(graphs) >= _dedup_target or config.model_source == "fingerprint_refine":
                break
            # Generate extra graphs to compensate for high dedup rate
            shortfall = original_count - len(graphs)
            if shortfall <= 0:
                break
            extra = batch_generate(min(shortfall * 2, original_count), grammar)
            graphs.extend(extra)
            logger.info(
                "Experiment %s dedup round %d: %d novel / %d generated, "
                "added %d extra candidates",
                exp_id[:8], _dedup_round + 1, len(novel), original_count,
                len(extra),
            )

        # Mark all novel fingerprints as seen for within-run dedup
        for g in graphs:
            _existing_fps.add(g.fingerprint())

        dedup_rate = 1.0 - (len(graphs) / max(original_count, 1))
        results["skipped_dedup"] = original_count - len(graphs)
        results["dedup_rate"] = round(dedup_rate, 3)
        results["dedup_novel_count"] = len(graphs)
        results["dedup_known_fingerprints"] = len(_existing_fps)
        results["total"] = len(graphs)  # update to reflect actual novel count

        if dedup_rate > 0.1:
            logger.info(
                "Experiment %s dedup: %d/%d candidates were duplicates (%.0f%% dedup rate), "
                "%d novel candidates remain, %d known fingerprints in DB",
                exp_id[:8], original_count - len(graphs), original_count,
                dedup_rate * 100, len(graphs), len(_existing_fps),
            )
        if dedup_rate > 0.8:
            logger.warning(
                "Experiment %s: grammar diversity exhaustion — %.0f%% dedup rate. "
                "Consider increasing grammar depth/ops or switching to refinement mode.",
                exp_id[:8], dedup_rate * 100,
            )

        with self._lock:
            self._progress.total_programs = len(graphs)

        # Track ops from S0 failures for op_success_rates (not stored in DB)
        _s0_op_counts: Dict[str, Dict[str, int]] = {}  # op -> {n_used, n_s0, n_s05}

        for i, graph in enumerate(graphs):
            if self._stop_event.is_set():
                break

            fp = graph.fingerprint()
            with self._lock:
                self._progress.current_program = i + 1
                self._progress.current_fingerprint = fp[:10]
                self._progress.elapsed_seconds = time.time() - t_start

            # Real-time dedup: skip if evaluated by another process since experiment start
            if nb.has_fingerprint(fp):
                results.setdefault("skipped_dedup_runtime", 0)
                results["skipped_dedup_runtime"] += 1
                self._emit_event("program_evaluated", {
                    "index": i, "fingerprint": fp[:10],
                    "result": "skipped_dedup",
                })
                continue

            # Pre-screen: skip graphs whose op-pair structure is toxic
            if failure_blocklist:
                bigrams = set()
                for nid, node in graph.nodes.items():
                    if node.is_input:
                        continue
                    for inp_id in node.input_ids:
                        parent = graph.nodes.get(inp_id)
                        if parent and not parent.is_input:
                            bigrams.add(f"{parent.op_name}->{node.op_name}")
                if bigrams:
                    toxic_hits = sum(1 for bg in bigrams if bg in failure_blocklist)
                    toxic_ratio = toxic_hits / len(bigrams)
                    if toxic_ratio >= 0.5:
                        results.setdefault("skipped_toxic", 0)
                        results["skipped_toxic"] += 1
                        self._emit_event("program_evaluated", {
                            "index": i, "fingerprint": graph.fingerprint()[:10],
                            "result": "skipped_toxic",
                            "toxic_ratio": f"{toxic_ratio:.2f}",
                        })
                        continue

            # Collect all metrics for this program
            program_metrics: Dict[str, Any] = {}
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

            # Native Proactive Gating (Project Hephaestus)
            # High-performance stability and toxic motif detection
            try:
                native_gating = _native_proactive_gating(graph)
                if not native_gating.get("passed", True):
                    results.setdefault("skipped_proactive_gating", 0)
                    results["skipped_proactive_gating"] += 1
                    
                    # Update metrics with native data
                    program_metrics["proactive_gating_reason"] = native_gating.get("reason")
                    program_metrics["max_depth"] = native_gating.get("max_depth")
                    program_metrics["n_toxic_motifs"] = native_gating.get("n_toxic_motifs")
                    
                    self._emit_event("program_evaluated", {
                        "index": i, "fingerprint": fp[:10],
                        "result": "skipped_proactive",
                        "reason": native_gating.get("reason"),
                        "max_depth": native_gating.get("max_depth"),
                    })
                    continue
            except Exception as e:
                logger.debug("Native proactive gating failed for %s: %s", fp[:10], e)

            # Validate
            validation = validate_graph(
                graph,
                max_ops=max(1, int(config.max_ops)),
                max_depth=max(1, int(config.max_depth)),
            )
            if not validation.valid:
                # Don't store S0 validation failures — they carry no learning
                # signal. Error counts are tracked in results dict and live feed.
                # But DO track op usage for grammar weight adaptation.
                results.setdefault("s0_validation_failures", 0)
                results["s0_validation_failures"] += 1
                for node in graph.nodes.values():
                    if not node.is_input and node.op_name:
                        c = _s0_op_counts.setdefault(node.op_name, {"n_used": 0, "n_s0": 0, "n_s05": 0})
                        c["n_used"] += 1
                self._emit_event("program_evaluated", {
                    "index": i, "fingerprint": fp[:10],
                    "result": "invalid", "error": validation.errors[0] if validation.errors else "",
                })
                continue

            # Compile & Stage 0/0.5
            try:
                # Z13: Defensive pause + GC to stabilize Torch Dynamo context if needed
                if i > 0 and i % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # More aggressive reset every 50 to clear Torch Dynamo cache
                    if i % 50 == 0:
                        try:
                            torch.compiler.reset()
                        except (AttributeError, Exception):
                            pass
                    
                    time.sleep(0.1)

                layer_graphs = [graph] * config.n_layers
                model = compile_model(layer_graphs, vocab_size=config.vocab_size, max_seq_len=config.max_seq_len)
                sandbox_result = self._safe_eval_for_stage(
                    model,
                    stage_tag="candidate_screening",
                    batch_size=2,
                    seq_len=min(128, config.max_seq_len),
                    vocab_size=config.vocab_size,
                    device=dev_str,
                )
                program_metrics.update(self._extract_sandbox_metrics(sandbox_result))
                program_metrics["param_count"] = sandbox_result.param_count
                
                s0_passed = sandbox_result.passed
                s05_passed = (sandbox_result.stability_score >= config.stage05_stability_threshold
                              and sandbox_result.causality_passed)
                
                if s0_passed:
                    results["stage0_passed"] += 1
                    with self._lock: self._progress.stage0_passed += 1
                if s05_passed:
                    results["stage05_passed"] += 1
                    with self._lock: self._progress.stage05_passed += 1

                if not s0_passed or not s05_passed:
                    # Don't store S0/S0.5 failures — error counts are tracked
                    # in results dict and error_type in the live feed event.
                    # But DO track op usage for grammar weight adaptation.
                    error_type = sandbox_result.error_type or "unknown"
                    results.setdefault("failure_error_types", {})
                    results["failure_error_types"][error_type] = (
                        results["failure_error_types"].get(error_type, 0) + 1
                    )
                    for node in graph.nodes.values():
                        if not node.is_input and node.op_name:
                            c = _s0_op_counts.setdefault(node.op_name, {"n_used": 0, "n_s0": 0, "n_s05": 0})
                            c["n_used"] += 1
                            if s0_passed:
                                c["n_s0"] += 1
                            if s05_passed:
                                c["n_s05"] += 1
                    self._emit_event("program_evaluated", {
                        "index": i, "fingerprint": fp[:10],
                        "result": "fail_s0" if not s0_passed else "fail_s05",
                        "error": (sandbox_result.error or "")[:120] if not s0_passed else None,
                        "error_type": error_type,
                        "stability": f"{sandbox_result.stability_score:.2f}" if s0_passed and not s05_passed else None,
                        "params": sandbox_result.param_count if sandbox_result.param_count else None,
                        "memory_mb": f"{sandbox_result.peak_memory_mb:.1f}" if sandbox_result.peak_memory_mb else None,
                        "has_nan": sandbox_result.has_nan_output or sandbox_result.has_nan_grad or None,
                        "has_inf": sandbox_result.has_inf_output or None,
                    })
                    continue

                # Stage 1: Asynchronous Execution (Z6)
                with self._lock:
                    self._progress.current_stage = "queuing_s1"
                
                orchestrator.submit(
                    index=i,
                    graph=graph,
                    config=config,
                    seed=self._stable_seed(exp_id, i, "screening"),
                    payload={
                        "metrics": program_metrics,
                        "graph": graph,
                        "batch_id": i // candidate_batch_size,
                        "queue_kind": "candidate_screening",
                    },
                    model=model # Reuse compiled model
                )
                
            except Exception as e:
                logger.error("Error evaluating graph %d: %s", i, e)
                # Reset CUDA context if this was a fatal CUDA error
                if torch.cuda.is_available():
                    from ..eval.sandbox import is_cuda_fatal
                    if is_cuda_fatal(e):
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                            _probe = torch.zeros(1, device="cuda")
                            del _probe
                            torch.cuda.synchronize()
                            logger.info("CUDA context recovered after fatal error on graph %d", i)
                        except Exception:
                            logger.warning("CUDA context unrecoverable after fatal error on graph %d", i)
                continue
            
            # Periodically process available results to keep the dashboard updated
            self._process_orchestrator_results(orchestrator, nb, exp_id, results, config)

        # Wait for remaining asynchronous Stage 1 evaluations
        with self._lock:
            self._progress.status = "finalizing_evaluations"
            
        while orchestrator.job_queue.unfinished_tasks > 0 or not orchestrator.result_queue.empty():
            if self._stop_event.is_set():
                break
            self._process_orchestrator_results(orchestrator, nb, exp_id, results, config)
            time.sleep(0.5)

        queue_telemetry = orchestrator.get_telemetry()
        orchestrator.shutdown()
        results["queue_telemetry"] = queue_telemetry
        results["perf_report"] = self._build_experiment_perf_report(results, queue_telemetry=queue_telemetry)
        results["perf_budget_gate"] = evaluate_perf_budget_gate(results["perf_report"])
        results.pop("_perf_traces", None)
        results.pop("_gpu_starvation", None)
        results.pop("_kernel_timing", None)
        if _s0_op_counts:
            results["_s0_op_counts"] = _s0_op_counts

        elapsed = time.time() - t_start
        with self._lock:
            self._progress.elapsed_seconds = elapsed
            self._progress.status = "analyzing"
            self._progress.aria_message = self.aria.begin_analysis()

        best = results.get("best_loss_ratio")
        best_str = f", best loss={best:.4f}" if best else ""
        dedup_str = ""
        if results.get("skipped_dedup", 0) > 0:
            dedup_str = f", dedup={results['skipped_dedup']} ({results.get('dedup_rate', 0)*100:.0f}%)"
        logger.info(
            "Experiment %s complete: %d programs → S0=%d → S0.5=%d → S1=%d "
            "(%.1fs)%s%s%s",
            exp_id[:8], results["total"],
            results["stage0_passed"], results["stage05_passed"],
            results["stage1_passed"], elapsed, best_str, dedup_str,
            f", native_gating={results.get('skipped_proactive_gating', 0)}" if results.get('skipped_proactive_gating') else "",
        )

        return results

    def _generate_refinement_graphs(
        self,
        exp_id: str,
        config: RunConfig,
        nb: LabNotebook,
        grammar: GrammarConfig,
    ) -> List:
        """Generate local mutations around selected source result IDs."""
        source_ids = [
            rid.strip() for rid in str(config.refine_source_result_ids or "").split(",")
            if rid.strip()
        ]
        target_n = max(1, int(config.n_programs))
        if not source_ids:
            logger.warning("Refinement mode requested without source IDs; falling back to synthesis generation")
            return batch_generate(target_n, grammar)

        source_pairs: List[Tuple[str, Any, Dict[str, Any]]] = []
        source_stage1_passed = 0
        for source_id in source_ids:
            source = nb.get_program_detail(source_id)
            if not source:
                continue
            graph_json_str = source.get("graph_json")
            if not graph_json_str:
                continue
            try:
                parent_graph = graph_from_json(graph_json_str)
            except Exception:
                continue
            source_pairs.append((source_id, parent_graph, source))
            if source.get("stage1_passed"):
                source_stage1_passed += 1

        if not source_pairs:
            logger.warning(
                "Refinement mode had %d source IDs but no reconstructable graphs; falling back to synthesis",
                len(source_ids),
            )
            return batch_generate(target_n, grammar)

        try:
            from ..search.evolution import _mutate_graph
        except Exception as e:
            logger.warning("Mutation helper unavailable (%s); falling back to synthesis generation", e)
            return batch_generate(target_n, grammar)

        seed = self._stable_seed("fingerprint_refine", exp_id, ",".join(source_ids))
        rng = random.Random(seed)
        per_source = max(1, int(config.refine_mutations_per_source or 1))
        target_pool = max(target_n, target_n * max(1, int(config.refine_pool_multiplier or 1)))
        candidate_pool: List[Tuple[float, Any]] = []
        seen_fingerprints: Set[str] = set()
        op_success = self._op_success_lookup(nb)
        intent = str(config.refine_intent or "balanced").lower()

        # Apply analysis-driven grammar hints if available
        analysis_data: Optional[Dict[str, Any]] = None
        if config.refine_analysis_json:
            try:
                analysis_data = json.loads(config.refine_analysis_json)
                grammar = self._apply_analysis_to_grammar(grammar, analysis_data, intent)
                logger.info(
                    "Experiment %s: applied analysis-driven grammar hints (intent=%s, %d exclude, %d boost)",
                    exp_id[:8], intent,
                    len(analysis_data.get("recipe", {}).get("grammar_hints", {}).get("exclude_ops", [])),
                    len(analysis_data.get("recipe", {}).get("grammar_hints", {}).get("boost_ops", {})),
                )
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.warning("Experiment %s: failed to parse refine_analysis_json: %s", exp_id[:8], e)

        recent_health = self._recent_synthesis_health(nb, window=5)
        zero_s1_regime = (
            source_stage1_passed == 0
            and float(recent_health.get("s1_rate") or 0.0) <= 0.0
        )
        mutated_budget = target_n if not zero_s1_regime else max(1, target_n // 2)
        if zero_s1_regime:
            logger.warning(
                "Refinement detected zero-S1 regime with no survivor sources; "
                "forcing exploration mix (mutated=%d, fallback=%d)",
                mutated_budget,
                max(0, target_n - mutated_budget),
            )

        while len(candidate_pool) < target_pool:
            added_this_round = 0
            for source_id, parent_graph, source_row in source_pairs:
                for _ in range(per_source):
                    if len(candidate_pool) >= target_pool:
                        break
                    try:
                        child = _mutate_graph(parent_graph, grammar, rng)
                    except Exception:
                        continue
                        
                    # Z15: Prune dead branches (unreachable nodes) before validation 
                    # to prevent redundant complexity from bloat mutations.
                    child.prune_dead_branches()
                    
                    validation = validate_graph(
                        child,
                        max_ops=max(1, int(config.max_ops)),
                        max_depth=max(1, int(config.max_depth)),
                    )
                    if not validation.valid:
                        continue

                    fp = child.fingerprint()
                    if fp in seen_fingerprints:
                        continue
                    seen_fingerprints.add(fp)
                    child.metadata.setdefault("refinement", {})
                    child.metadata["refinement"]["source_result_id"] = source_id
                    child.metadata["refinement"]["seed_fingerprint"] = parent_graph.fingerprint()
                    child.metadata["refinement"]["intent"] = intent
                    score, score_breakdown = self._score_refinement_candidate(
                        child,
                        op_success=op_success,
                        intent=intent,
                        source_row=source_row,
                        include_breakdown=True,
                    )
                    child.metadata["refinement"]["intent_score"] = score
                    child.metadata["refinement"]["intent_score_breakdown"] = score_breakdown
                    if analysis_data:
                        recipe = analysis_data.get("recipe", {})
                        child.metadata["refinement"]["analysis_driven"] = True
                        child.metadata["refinement"]["analysis_recipe"] = {
                            "recommended_intent": recipe.get("recommended_intent", "balanced"),
                            "primary_target": recipe.get("primary_target", ""),
                            "confidence": recipe.get("confidence", "low"),
                        }
                    candidate_pool.append((score, child))
                    added_this_round += 1

                if len(candidate_pool) >= target_pool:
                    break

            if added_this_round == 0:
                break

        candidate_pool.sort(key=lambda item: item[0], reverse=True)
        mutated_graphs = [g for _, g in candidate_pool[:mutated_budget]]

        if len(mutated_graphs) < target_n:
            fallback = batch_generate(target_n - len(mutated_graphs), grammar)
            for f in fallback:
                f.metadata.setdefault("refinement", {})
                f.metadata["refinement"]["intent"] = intent
                f.metadata["refinement"]["fallback"] = True
                if zero_s1_regime:
                    f.metadata["refinement"]["fallback_reason"] = "zero_s1_regime"
            mutated_graphs.extend(fallback)

        logger.info(
            "Experiment %s: generated %d refinement graphs from %d source fingerprint(s) [intent=%s pool=%d]",
            exp_id[:8],
            len(mutated_graphs),
            len(source_pairs),
            intent,
            len(candidate_pool),
        )
        return mutated_graphs

    @staticmethod
    def _apply_analysis_to_grammar(
        base_grammar: GrammarConfig,
        analysis: Dict[str, Any],
        intent: str,
    ) -> GrammarConfig:
        """Apply RefinementAnalyzer recipe hints to a grammar config."""
        recipe = analysis.get("recipe", {})
        hints = recipe.get("grammar_hints", {})

        # Exclude risky ops
        exclude_ops = hints.get("exclude_ops", [])
        if exclude_ops:
            base_grammar.excluded_ops = base_grammar.excluded_ops | set(exclude_ops)

        # Boost ops (cap at 3.0)
        boost_ops = hints.get("boost_ops", {})
        for op_name, multiplier in boost_ops.items():
            current = base_grammar.op_weights.get(op_name, 1.0)
            base_grammar.op_weights[op_name] = min(3.0, current * multiplier)

        # Boost categories (×1.5, capped)
        add_categories = hints.get("add_categories", {})
        for cat, multiplier in add_categories.items():
            current = base_grammar.category_weights.get(cat, 1.0)
            base_grammar.category_weights[cat] = min(8.0, current * multiplier)

        return base_grammar

    def _op_success_lookup(self, nb: LabNotebook) -> Dict[str, float]:
        """Return per-op Stage1 success rates for learning-guided refinement."""
        lookup: Dict[str, float] = {}
        try:
            for row in nb.get_op_success_rates():
                n_used = float(row.get("n_used") or 0.0)
                n_s1 = float(row.get("n_stage1_passed") or 0.0)
                if n_used > 0:
                    lookup[str(row.get("op_name"))] = n_s1 / n_used
        except Exception:
            pass
        return lookup

    def _score_refinement_candidate(
        self,
        graph: Any,
        op_success: Dict[str, float],
        intent: str,
        source_row: Optional[Dict[str, Any]] = None,
        include_breakdown: bool = False,
    ) -> Any:
        """Score a refinement candidate using past learning + objective intent."""
        ops: List[str] = []
        for node in graph.nodes.values():
            if not node.is_input:
                ops.append(str(node.op_name))

        n_ops = max(1, int(graph.n_ops()))
        depth = max(1, int(graph.depth()))
        params = max(1.0, float(graph.n_params_estimate()))
        unique_ops = len(set(ops))

        learned_quality = 0.5
        if ops:
            learned_quality = sum(op_success.get(op, 0.5) for op in ops) / len(ops)

        # FLOP-aware compression proxy
        _cfg_dim = 256  # default model_dim
        _cfg_layers = 4  # default n_layers
        try:
            flop_est = estimate_flops(graph, seq_len=128, d_model=_cfg_dim)
            flops_per_token = flop_est.flops_per_token if flop_est and flop_est.flops_per_token > 0 else (params * 2)
        except Exception:
            flops_per_token = params * 2
        baseline_fpt = 2.0 * _cfg_dim ** 2 * _cfg_layers
        flop_efficiency = min(1.0, baseline_fpt / max(flops_per_token, 1.0))
        param_efficiency_proxy = min(1.0, (6 * _cfg_dim ** 2) / max(params, 1.0))
        compression_proxy = 0.5 * flop_efficiency + 0.3 * param_efficiency_proxy + 0.2 / (1.0 + 0.1 * depth)
        novelty_proxy = min(1.0, (unique_ops / max(1, n_ops)) + (0.1 if depth >= 4 else 0.0))

        sparse_hint_ops = (
            "sparse", "gate", "topk", "mask", "threshold", "skip", "mixture"
        )
        sparse_op_bonus = 0.0
        if ops:
            sparse_op_bonus = sum(
                1.0 for op in ops if any(token in op.lower() for token in sparse_hint_ops)
            ) / len(ops)
        sparsity_proxy = min(1.0, 0.7 * compression_proxy + 0.3 * sparse_op_bonus)

        parent_novelty = float((source_row or {}).get("novelty_score") or 0.0)
        parent_quality = 1.0 - float((source_row or {}).get("loss_ratio") or 1.0)

        mode = str(intent or "balanced").lower()
        weighted_terms: Dict[str, float]
        if mode == "quality":
            weighted_terms = {
                "learned_quality": 0.60 * learned_quality,
                "parent_quality": 0.25 * parent_quality,
                "compression_proxy": 0.15 * compression_proxy,
            }
        elif mode == "compression":
            weighted_terms = {
                "compression_proxy": 0.60 * compression_proxy,
                "learned_quality": 0.25 * learned_quality,
                "parent_quality": 0.15 * parent_quality,
            }
        elif mode == "sparsity":
            weighted_terms = {
                "sparsity_proxy": 0.60 * sparsity_proxy,
                "learned_quality": 0.25 * learned_quality,
                "compression_proxy": 0.15 * compression_proxy,
            }
        elif mode == "novelty":
            weighted_terms = {
                "novelty_proxy": 0.55 * novelty_proxy,
                "learned_quality": 0.25 * learned_quality,
                "parent_novelty": 0.20 * parent_novelty,
            }
        else:  # balanced
            weighted_terms = {
                "learned_quality": 0.35 * learned_quality,
                "compression_proxy": 0.25 * compression_proxy,
                "novelty_proxy": 0.20 * novelty_proxy,
                "parent_signal": 0.20 * max(parent_quality, parent_novelty),
            }
        score = float(sum(weighted_terms.values()))
        if not include_breakdown:
            return score

        breakdown = {
            "mode": mode,
            "components": {
                "learned_quality": float(learned_quality),
                "compression_proxy": float(compression_proxy),
                "novelty_proxy": float(novelty_proxy),
                "sparsity_proxy": float(sparsity_proxy),
                "parent_quality": float(parent_quality),
                "parent_novelty": float(parent_novelty),
                "sparse_op_bonus": float(sparse_op_bonus),
            },
            "weighted_terms": {k: float(v) for k, v in weighted_terms.items()},
            "ops": {
                "n_ops": int(n_ops),
                "depth": int(depth),
                "unique_ops": int(unique_ops),
                "params_estimate": float(params),
            },
        }
        return score, breakdown

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

    def _micro_train_async(self, model: nn.Module, config: RunConfig, seed: int, dev: torch.device) -> Dict:
        """Async worker entry point for training a pre-compiled model."""
        try:
            return self._micro_train(model, config, dev, seed=seed)
        except Exception as e:
            return {"error": str(e), "passed": False}

    def _micro_train(self, model: nn.Module, config: RunConfig,
                     dev: torch.device, seed: int = 42) -> Dict:
        """Run Stage 1 micro-training with comprehensive metric capture.

        Uses deterministic seeding per step so all candidates see the same
        training data in the same order, enabling fair comparison (#56).
        """
        from research.scientist.perf import PerfTracer, GPUStarvationDetector, OpKernelProfiler
        trace_enabled = bool(getattr(config, "enable_perf_tracing", False))
        tracer = PerfTracer() if trace_enabled else None
        starvation_detector = GPUStarvationDetector(threshold_ms=2.0)
        op_profiler = OpKernelProfiler(
            enabled=bool(getattr(config, "enable_kernel_profiling", False)),
            top_k=max(1, int(getattr(config, "kernel_profile_top_k", 20) or 20)),
        )
        
        result: Dict[str, Any] = {"passed": False}
        collect_curve = bool(getattr(config, "collect_training_curve", False))
        grad_clip_norm = float(getattr(config, "gradient_clip_norm", 1.0) or 0.0)
        if grad_clip_norm < 0.0:
            grad_clip_norm = 0.0

        trace_totals_ms: Dict[str, float] = {
            "model_setup": 0.0,
            "data_sampling": 0.0,
            "forward_pass": 0.0,
            "backward_pass": 0.0,
            "optimizer_step": 0.0,
        }

        def _trace_ctx(name: str, use_gpu: bool = True):
            return tracer.trace(name, use_gpu=use_gpu) if tracer is not None else nullcontext()

        try:
            setup_t0 = time.perf_counter()
            with _trace_ctx("model_setup"):
                model = model.to(dev)
                model.train()
                opt_kwargs: Dict[str, Any] = {"lr": config.stage1_lr, "weight_decay": 0.01}
                if dev.type == "cuda":
                    use_fused = bool(getattr(config, "optimizer_fused", True))
                    use_foreach = bool(getattr(config, "optimizer_foreach", True))
                    if use_fused:
                        opt_kwargs["fused"] = True
                    elif use_foreach:
                        opt_kwargs["foreach"] = True
                try:
                    optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)
                except Exception:
                    opt_kwargs.pop("fused", None)
                    opt_kwargs.pop("foreach", None)
                    optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)
            trace_totals_ms["model_setup"] += (time.perf_counter() - setup_t0) * 1000.0

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

            step_time_sum_ms = 0.0
            step_count = 0
            grad_norm_sum = 0.0
            grad_norm_sq_sum = 0.0
            grad_norm_max = 0.0
            grad_norm_count = 0
            training_curve: List[Dict] = [] if collect_curve else []
            kernel_profiles: List[Dict[str, Any]] = []

            seq_len = min(128, config.max_seq_len)
            random_mode = str(config.data_mode or "random").strip().lower() == "random"
            _seed_int = int(seed)

            def _make_random_batch(step: int) -> torch.Tensor:
                """Generate a deterministic random batch for a given step."""
                torch.manual_seed(_seed_int * 100_000 + step)
                return torch.randint(
                    0, int(config.vocab_size),
                    (config.stage1_batch_size, seq_len),
                    device=dev,
                )

            # --- Part 1: Discovery Evaluation (Fast) ---
            # Evaluate on a few random batches to get "discovery_loss"
            discovery_steps = min(5, config.stage1_steps // 10)
            discovery_losses = []
            model.eval()
            with torch.no_grad():
                for ds in range(discovery_steps):
                    d_batch = _make_random_batch(ds + 9999) # different offset
                    with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=True):
                        d_logits = model(d_batch)
                        d_loss = F.cross_entropy(d_logits[:, :-1].reshape(-1, config.vocab_size), d_batch[:, 1:].reshape(-1))
                    discovery_losses.append(d_loss.item())
            
            if discovery_losses:
                result["discovery_loss"] = sum(discovery_losses) / len(discovery_losses)
                # Note: discovery_loss_ratio needs a baseline; we'll compute it in _execute_experiment
            
            model.train()
            # --- Part 2: Main Training (Validation Channel) ---
            
            # Implementation of train/val split for Stage 1
            train_steps = int(config.stage1_steps * 0.8)
            val_steps = config.stage1_steps - train_steps

            starvation_interval = max(1, int(getattr(config, "starvation_check_interval", 8) or 8))

            use_cuda_graph = bool(
                dev.type == "cuda"
                and bool(getattr(config, "enable_cuda_graphs", True))
                and random_mode
                and not op_profiler.enabled
                and not trace_enabled
                and not collect_curve
                and int(config.stage1_steps) >= 8
            )

            ran_cuda_graph = False
            if use_cuda_graph:
                try:
                    static_input_ids = torch.empty(
                        (config.stage1_batch_size, seq_len), dtype=torch.long, device=dev
                    )
                    captured_loss = torch.zeros((), device=dev)
                    captured_grad_norm = torch.zeros((), device=dev)
                    warmup_steps = max(1, int(getattr(config, "cuda_graph_warmup_steps", 3) or 3))

                    def _graph_step() -> Tuple[torch.Tensor, torch.Tensor]:
                        with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=True):
                            logits = model(static_input_ids)
                            loss_t = F.cross_entropy(
                                logits[:, :-1].reshape(-1, logits.shape[-1]),
                                static_input_ids[:, 1:].reshape(-1),
                            )
                        optimizer.zero_grad(set_to_none=True)
                        loss_t.backward()
                        if grad_clip_norm > 0.0:
                            grad_norm_t = nn.utils.clip_grad_norm_(
                                model.parameters(), grad_clip_norm, foreach=True
                            )
                        else:
                            grad_norm_t = torch.zeros((), device=dev)
                        optimizer.step()
                        return loss_t, grad_norm_t

                    for wi in range(min(warmup_steps, int(config.stage1_steps))):
                        static_input_ids.copy_(_make_random_batch(wi), non_blocking=True)
                        loss_t, grad_norm_t = _graph_step()
                        captured_loss.copy_(loss_t.detach())
                        captured_grad_norm.copy_(torch.as_tensor(grad_norm_t, device=dev).detach())

                    torch.cuda.synchronize(dev)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        loss_t, grad_norm_t = _graph_step()
                        captured_loss.copy_(loss_t.detach())
                        captured_grad_norm.copy_(torch.as_tensor(grad_norm_t, device=dev).detach())

                    check_interval = max(1, int(getattr(config, "loss_check_interval", 8) or 8))
                    for step in range(config.stage1_steps):
                        if self._stop_event.is_set():
                            break
                        t_step = time.perf_counter()
                        static_input_ids.copy_(_make_random_batch(step), non_blocking=True)
                        graph.replay()
                        t_step_end = time.perf_counter()
                        step_time_ms = (t_step_end - t_step) * 1000.0
                        step_count += 1
                        step_time_sum_ms += step_time_ms
                        total_tokens += static_input_ids.numel()

                        should_check = (step == 0) or (step == config.stage1_steps - 1) or (step % check_interval == 0)
                        if not should_check:
                            continue

                        loss_val = float(captured_loss.item())
                        grad_norm = float(captured_grad_norm.item())
                        if not math.isfinite(loss_val):
                            result["error"] = f"NaN/Inf loss at step {step}"
                            result["n_train_steps"] = step
                            return result
                        if step == 0 and (not math.isfinite(grad_norm) or grad_norm <= 1e-10):
                            result["error"] = "zero_grad_precheck_failed"
                            result["n_train_steps"] = 0
                            result["max_grad_norm"] = grad_norm
                            result["mean_grad_norm"] = grad_norm
                            result["grad_norm_std"] = 0.0
                            return result
                        if step == 0:
                            initial_loss = loss_val
                        final_loss = loss_val
                        min_loss = min(min_loss, loss_val)
                        grad_norm_sum += grad_norm
                        grad_norm_sq_sum += grad_norm * grad_norm
                        grad_norm_max = max(grad_norm_max, grad_norm)
                        grad_norm_count += 1
                    ran_cuda_graph = True
                except Exception as e:
                    result["cuda_graph_fallback_reason"] = str(e)

            if not ran_cuda_graph:
                for step in range(config.stage1_steps):
                    if self._stop_event.is_set():
                        break

                    starvation_sample = (not random_mode) and ((step % starvation_interval) == 0)
                    if starvation_sample:
                        starvation_detector.start_wait()
                    data_t0 = time.perf_counter()
                    with _trace_ctx("data_sampling"):
                        if random_mode:
                            input_ids = _make_random_batch(step)
                        else:
                            input_ids = self._sample_training_input_ids(
                                config=config,
                                dev=dev,
                                batch_size=config.stage1_batch_size,
                                seq_len=seq_len,
                                seed=seed + step,
                            )
                    if starvation_sample:
                        starvation_detector.end_wait()
                    trace_totals_ms["data_sampling"] += (time.perf_counter() - data_t0) * 1000.0

                    t_step = time.perf_counter()

                    step_state: Dict[str, Any] = {}

                    def _run_step() -> None:
                        fwd_t0 = time.perf_counter()
                        with _trace_ctx("forward_pass"):
                            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                                    enabled=(dev.type == "cuda")):
                                logits = model(input_ids)
                                loss = F.cross_entropy(
                                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                                    input_ids[:, 1:].reshape(-1),
                                )
                        trace_totals_ms["forward_pass"] += (time.perf_counter() - fwd_t0) * 1000.0
                        step_state["loss"] = loss

                        bwd_t0 = time.perf_counter()
                        with _trace_ctx("backward_pass"):
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            if grad_clip_norm > 0.0:
                                step_state["grad_norm"] = nn.utils.clip_grad_norm_(
                                    model.parameters(), grad_clip_norm, foreach=(dev.type == "cuda")
                                ).item()
                            else:
                                step_state["grad_norm"] = 0.0
                        trace_totals_ms["backward_pass"] += (time.perf_counter() - bwd_t0) * 1000.0

                        opt_t0 = time.perf_counter()
                        with _trace_ctx("optimizer_step"):
                            optimizer.step()
                        trace_totals_ms["optimizer_step"] += (time.perf_counter() - opt_t0) * 1000.0

                    if step == 0 and op_profiler.enabled:
                        kernel_summary = op_profiler.profile_callable(_run_step)
                        if kernel_summary:
                            kernel_profiles.append({"step": step, **kernel_summary})
                        else:
                            _run_step()
                    else:
                        _run_step()

                    loss = step_state.get("loss")
                    grad_norm = float(step_state.get("grad_norm", 0.0))

                    if loss is None or torch.isnan(loss) or torch.isinf(loss):
                        result["error"] = f"NaN/Inf loss at step {step}"
                        result["n_train_steps"] = step
                        return result

                    if step == 0 and (not math.isfinite(grad_norm) or grad_norm <= 1e-10):
                        result["error"] = "zero_grad_precheck_failed"
                        result["n_train_steps"] = 0
                        result["max_grad_norm"] = grad_norm
                        result["mean_grad_norm"] = grad_norm
                        result["grad_norm_std"] = 0.0
                        return result

                    if dev.type == "cuda" and (trace_enabled or op_profiler.enabled):
                        torch.cuda.synchronize(dev)

                    t_step_end = time.perf_counter()
                    step_time_ms = (t_step_end - t_step) * 1000

                    loss_val = loss.item()
                    if step == 0:
                        initial_loss = loss_val
                    final_loss = loss_val
                    min_loss = min(min_loss, loss_val)
                    total_tokens += input_ids.numel()

                    step_count += 1
                    step_time_sum_ms += step_time_ms
                    grad_norm_sum += grad_norm
                    grad_norm_sq_sum += grad_norm * grad_norm
                    grad_norm_max = max(grad_norm_max, grad_norm)
                    grad_norm_count += 1

                    # Record per-step data
                    if collect_curve:
                        training_curve.append({
                            "step": step,
                            "loss": loss_val,
                            "grad_norm": grad_norm,
                            "step_time_ms": step_time_ms,
                        })

                    # Emit live training step events for dashboard
                    ctx = getattr(self, "_live_training_context", None)
                    if ctx and step % 25 == 0:
                        self._emit_event("training_step", {
                            "experiment_id": ctx.get("exp_id", ""),
                            "step": step,
                            "loss": round(loss_val, 6),
                            "total_steps": config.stage1_steps,
                            "phase": ctx.get("phase", ""),
                        })

                    # Log training progress at start, midpoint, and end
                    total_steps = config.stage1_steps
                    if step == 0 or step == total_steps // 2 or step == total_steps - 1:
                        logger.debug(
                            "    train step %d/%d: loss=%.4f, grad_norm=%.3f, "
                            "step_time=%.1fms",
                            step + 1, total_steps, loss_val, grad_norm, step_time_ms,
                        )

            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t_end = time.perf_counter()
            total_time_ms = (t_end - t_start) * 1000

            # Optional validation loss on heldout corpus split
            validation_loss = None
            validation_loss_ratio = None
            generalization_gap = None
            val_batches = max(1, int(getattr(config, "stage1_val_batches", 0) or 0))
            compute_val = bool(getattr(config, "stage1_compute_val_loss", True))
            val_batch_size = int(getattr(config, "stage1_val_batch_size", 0) or config.stage1_batch_size)
            val_frac = float(getattr(config, "corpus_val_fraction", 0.0) or 0.0)
            if compute_val and val_batches > 0 and val_frac > 0.0:
                if str(config.data_mode or "random").strip().lower() == "corpus":
                    try:
                        model.eval()
                        losses = []
                        with torch.no_grad():
                            for i in range(val_batches):
                                input_ids = self._sample_training_input_ids(
                                    config=config,
                                    dev=dev,
                                    batch_size=val_batch_size,
                                    seq_len=seq_len,
                                    seed=seed + 10_000 + i,
                                    split="val",
                                )
                                if input_ids is None:
                                    continue
                                with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                                        enabled=(dev.type == "cuda")):
                                    logits = model(input_ids)
                                    loss = F.cross_entropy(
                                        logits[:, :-1].reshape(-1, logits.shape[-1]),
                                        input_ids[:, 1:].reshape(-1),
                                    )
                                if loss is not None and torch.isfinite(loss):
                                    losses.append(float(loss.item()))
                        if losses:
                            validation_loss = sum(losses) / len(losses)
                    except Exception as e:
                        result["validation_loss_error"] = str(e)
                    finally:
                        model.train()

            # Optional discovery loss on random tokens (fast triage signal)
            discovery_loss = None
            discovery_loss_ratio = None
            discovery_batches = max(1, int(getattr(config, "stage1_discovery_batches", 0) or 0))
            compute_discovery = bool(getattr(config, "stage1_compute_discovery_loss", True))
            discovery_batch_size = int(getattr(config, "stage1_discovery_batch_size", 0) or config.stage1_batch_size)
            if compute_discovery and discovery_batches > 0:
                try:
                    model.eval()
                    losses = []
                    with torch.no_grad():
                        for i in range(discovery_batches):
                            torch.manual_seed(int(seed) * 10_000 + 3_000 + i)
                            input_ids = torch.randint(
                                0, int(config.vocab_size),
                                (discovery_batch_size, seq_len),
                                device=dev,
                            )
                            with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                                    enabled=(dev.type == "cuda")):
                                logits = model(input_ids)
                                loss = F.cross_entropy(
                                    logits[:, :-1].reshape(-1, logits.shape[-1]),
                                    input_ids[:, 1:].reshape(-1),
                                )
                            if loss is not None and torch.isfinite(loss):
                                losses.append(float(loss.item()))
                    if losses:
                        discovery_loss = sum(losses) / len(losses)
                except Exception as e:
                    result["discovery_loss_error"] = str(e)
                finally:
                    model.train()

            if validation_loss is not None and initial_loss:
                validation_loss_ratio = validation_loss / max(initial_loss, 1e-6)
            if validation_loss is not None and final_loss is not None:
                generalization_gap = validation_loss - final_loss
            if discovery_loss is not None and initial_loss:
                discovery_loss_ratio = discovery_loss / max(initial_loss, 1e-6)

            # Collect perf results
            if tracer is not None:
                result["perf_traces"] = tracer.get_report()
            else:
                result["perf_traces"] = {
                    "summary_ms": {k: round(v, 4) for k, v in trace_totals_ms.items()},
                    "traces": [],
                }
            result["gpu_starvation"] = starvation_detector.get_summary()
            if kernel_profiles:
                result["kernel_timing"] = {
                    "sample_count": len(kernel_profiles),
                    "samples": kernel_profiles,
                    "top_ops": kernel_profiles[0].get("top_ops", []),
                }

            if initial_loss and final_loss:
                result["loss_ratio"] = final_loss / max(initial_loss, 1e-6)
                result["final_loss"] = final_loss
                result["initial_loss"] = initial_loss
                result["min_loss"] = min_loss
                if validation_loss is not None:
                    result["validation_loss"] = validation_loss
                if validation_loss_ratio is not None:
                    result["validation_loss_ratio"] = validation_loss_ratio
                if generalization_gap is not None:
                    result["generalization_gap"] = generalization_gap
                if discovery_loss is not None:
                    result["discovery_loss"] = discovery_loss
                if discovery_loss_ratio is not None:
                    result["discovery_loss_ratio"] = discovery_loss_ratio
                result["throughput"] = total_tokens / (total_time_ms / 1000)
                result["passed"] = result["loss_ratio"] < config.stage1_loss_ratio_threshold

                # Compute improvement rate
                if initial_loss > 0:
                    result["loss_improvement_rate"] = (initial_loss - final_loss) / initial_loss

                # Timing stats
                result["avg_step_time_ms"] = (step_time_sum_ms / step_count) if step_count > 0 else 0.0
                result["total_train_time_ms"] = total_time_ms

                # Gradient norm stats
                if grad_norm_count > 0:
                    result["max_grad_norm"] = grad_norm_max
                    result["mean_grad_norm"] = grad_norm_sum / grad_norm_count
                    mean_gn = result["mean_grad_norm"]
                    var = max((grad_norm_sq_sum / grad_norm_count) - (mean_gn * mean_gn), 0.0)
                    result["grad_norm_std"] = var ** 0.5

                result["n_train_steps"] = step_count
                result["final_lr"] = config.stage1_lr  # constant for now
                if collect_curve:
                    result["training_curve"] = training_curve

                # Extract architecture-specific telemetry (MoE, MoD, MoR, etc.)
                arch_telemetry = self._extract_architecture_telemetry(model)
                result.update(arch_telemetry)

                # Behavioral fingerprint for S1 survivors (novelty scoring)
                if result.get("passed") and model is not None:
                    try:
                        _fp = compute_fingerprint(
                            model,
                            seq_len=min(64, config.max_seq_len),
                            model_dim=config.model_dim,
                            vocab_size=config.vocab_size,
                            device=str(dev),
                        )
                        result["_behavioral_fingerprint"] = _fp.to_dict()
                    except Exception as e_fp:
                        logger.debug("Fingerprint failed in S1 worker: %s", e_fp)

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

        # Finalize performance reports
        try:
            if tracer is not None:
                fallback_perf = tracer.get_report()
            else:
                fallback_perf = {
                    "summary_ms": {k: round(v, 4) for k, v in trace_totals_ms.items()},
                    "traces": [],
                }
            result["perf_report"] = result.get("perf_traces", fallback_perf)
            # Ensure throughput is included in perf_report for experiment-level aggregation
            if isinstance(result.get("throughput"), (int, float)):
                result["perf_report"]["avg_throughput_tok_s"] = float(result["throughput"])
            
            result["starvation_report"] = result.get("gpu_starvation", starvation_detector.get_summary())
            if "kernel_timing" in result:
                result["kernel_timings_ms"] = result["kernel_timing"]
        except Exception as e:
            result["perf_error"] = str(e)

        try:
            result.update(self._extract_architecture_telemetry(model))
        except Exception as e:
            logger.debug("Architecture telemetry extract failed: %s", e)

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

    def _gather_designer_telemetry(self) -> Dict:
        """Fetch telemetry from aria-designer if available."""
        import requests
        base = os.environ.get("ARIA_DESIGNER_PROXY_BASE", "http://127.0.0.1:8091")
        result: Dict = {}
        try:
            r = requests.get(f"{base}/api/v1/integration/bridge-gap-report", timeout=3)
            if r.ok:
                result["bridge_gap_report"] = r.json()
        except Exception:
            pass
        try:
            r = requests.get(f"{base}/api/v1/blocks/builtin", params={"model_dim": 256}, timeout=3)
            if r.ok:
                blocks = r.json()
                result["builtin_blocks"] = [b.get("name") for b in blocks if isinstance(b, dict) and b.get("name")]
        except Exception:
            pass
        return result

    def _gather_analytics_data(self, nb: LabNotebook) -> Dict:
        """Gather all analytics data for rich context."""
        try:
            from .analytics import ExperimentAnalytics
            analytics = ExperimentAnalytics(nb)
            return {
                "op_success_rates": analytics.op_success_rates(),
                "structural_correlations": analytics.structural_correlations(),
                "failure_patterns": analytics.failure_patterns(),
                "compression_coverage": analytics.compression_coverage(),
                "sparse_coverage": analytics.sparse_coverage(),
                "top_op_combinations": analytics.top_op_combinations(10),
                "efficiency_frontier": analytics.efficiency_frontier(),
                "efficiency_frontier_3d": analytics.efficiency_frontier_3d(),
                "grammar_weights": analytics.compute_grammar_weights(),
                "default_weights": analytics.get_current_grammar_weights(),
                "learning_log": nb.get_learning_log(limit=10),
                "insights": nb.get_insights(limit=20),
                "negative_results": analytics.negative_results_synthesis(),
                "decision_outcomes": analytics.decision_outcome_analysis(),
                "designer_telemetry": self._gather_designer_telemetry(),
                "scaling_summary": nb.get_scaling_summary(),
                "gate_health": analytics.gate_health_daily(n_days=7),
            }
        except Exception:
            return {}

    def _get_past_hypotheses(self, nb: LabNotebook, limit: int = 5) -> List[Dict]:
        """Get past hypotheses with their outcomes, including refuted insights.

        Merges two sources:
        1. Recent experiment hypotheses (confirmed/refuted by S1 outcome)
        2. Formally refuted insights from the insights table

        This ensures the system never re-tests directions that were already
        proven unsuccessful.
        """
        experiments = nb.get_recent_experiments(limit * 2)
        past = []
        seen_texts: set = set()
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
            seen_texts.add(hyp[:80].lower())
            if len(past) >= limit:
                break

        # Also pull formally refuted insights so hypotheses that failed
        # in prior campaigns are visible to hypothesis generation.
        try:
            refuted_insights = nb.get_insights(status="refuted", limit=limit)
            for ins in refuted_insights:
                content = ins.get("content", "")
                if not content:
                    continue
                # Skip duplicates already covered by experiment hypotheses
                if content[:80].lower() in seen_texts:
                    continue
                past.append({
                    "hypothesis": content,
                    "confirmed": False,
                    "s1_count": 0,
                    "best_novelty": 0,
                    "experiment_id": None,
                    "source": "refuted_insight",
                    "confidence": ins.get("confidence", 0),
                    "evidence": ins.get("supporting_evidence", ""),
                })
                seen_texts.add(content[:80].lower())
        except Exception:
            pass  # insights table may not exist in older notebooks

        return past

    def _populate_refuted_cache(self, nb: LabNotebook) -> None:
        """Populate the persona's refuted hypothesis cache for similarity gating.

        Merges refuted insights from the insights table with refuted hypotheses
        from negative_results_synthesis so the persona can reject new hypotheses
        that are too similar to proven failures.
        """
        refuted: List[Dict] = []
        try:
            # Source 1: Formally refuted insights
            for ins in nb.get_insights(status="refuted", limit=20):
                content = ins.get("content", "")
                if content:
                    refuted.append({
                        "content": content,
                        "confidence": ins.get("confidence", 0),
                    })

            # Source 2: Refuted hypotheses from negative_results_synthesis
            try:
                from .analytics import ExperimentAnalytics
                analytics = ExperimentAnalytics(nb)
                neg = analytics.negative_results_synthesis()
                for rh in neg.get("refuted_hypotheses", []):
                    content = rh.get("content", "")
                    if content and not any(r.get("content", "")[:80] == content[:80]
                                           for r in refuted):
                        refuted.append({
                            "content": content,
                            "confidence": rh.get("confidence", 0),
                        })
            except Exception:
                pass
        except Exception:
            pass

        self.aria.set_refuted_hypotheses(refuted)

    def _auto_recommend(self, results: Dict, config: RunConfig,
                        hypothesis: str, nb: LabNotebook):
        """Auto-generate a recommendation after experiment completion and APPLY it."""
        try:
            context = self._build_rich_context_for_experiment(
                results, config, hypothesis, nb)
            _analytics = self._gather_analytics_data(nb)
            op_rates = _analytics.get("op_success_rates")
            comp_cov = _analytics.get("compression_coverage")
            heuristic = self.aria.suggest_experiment(
                context, op_success_rates=op_rates,
                compression_coverage=comp_cov) or {}
            summary_payload = self._build_next_experiment_summary(nb, results)
            planner = NextExperimentDecisionPlanner.from_run_config(config)
            plan = planner.propose_plan(
                summary_payload,
                current_cost_dollars=float(self.aria.total_cost or 0.0),
                fallback_plan=heuristic,
            )
            suggestion = {
                "mode": plan.get("mode", heuristic.get("mode", "synthesis")),
                "reasoning": plan.get("reasoning", heuristic.get("reasoning", "")),
                "confidence": float(plan.get("confidence", heuristic.get("confidence", 0.5)) or 0.5),
                "config": plan.get("config", heuristic.get("config", {})),
                "planner": plan.get("planner", {}),
                "guardrails": plan.get("guardrails", {}),
                "summary_excerpt": plan.get("summary_excerpt", {}),
            }
            if suggestion:
                evidence_pack = build_evidence_pack(
                    nb,
                    analytics=None,
                    recommendation=suggestion,
                    decision_type="experiment_recommendation",
                )
                suggestion["evidence_pack"] = evidence_pack
                with self._lock:
                    self._last_recommendation = suggestion
                self._emit_event("aria_recommendation", {
                    "mode": suggestion.get("mode"),
                    "reasoning": suggestion.get("reasoning", ""),
                    "confidence": suggestion.get("confidence", 0),
                    "config": suggestion.get("config", {}),
                    "planner": suggestion.get("planner", {}),
                    "evidence_pack": evidence_pack,
                })
                # Store as notebook entry
                nb.add_entry(ExperimentEntry(
                    entry_type="decision",
                    title="Aria's Next Experiment Recommendation",
                    content=suggestion.get("reasoning", ""),
                    metadata={
                        "mode": suggestion.get("mode"),
                        "confidence": suggestion.get("confidence", 0),
                        "suggested_config": suggestion.get("config", {}),
                        "planner": suggestion.get("planner", {}),
                        "guardrails": suggestion.get("guardrails", {}),
                        "summary_payload": summary_payload,
                        "evidence_pack": evidence_pack,
                    },
                ))
                nb.record_decision(
                    campaign_id=self._active_campaign_id,
                    decision_type="next_experiment_plan",
                    subject=f"experiment:{summary_payload.get('recent_experiment_id') or 'latest'}",
                    rationale=suggestion.get("reasoning", ""),
                    alternatives=[{
                        "heuristic_fallback": heuristic,
                    }],
                    evidence_pack={
                        "mode": suggestion.get("mode"),
                        "confidence": suggestion.get("confidence", 0),
                        "config": suggestion.get("config", {}),
                        "planner": suggestion.get("planner", {}),
                        "guardrails": suggestion.get("guardrails", {}),
                        "summary_payload": summary_payload,
                    },
                )
                # PROACTIVE: Apply suggested config/grammar changes immediately
                self._apply_recommendation(suggestion, nb)
        except Exception as e:
            logger.debug(f"Auto-recommendation failed: {e}")

    def _apply_recommendation(self, suggestion: Dict, nb: LabNotebook):
        """Proactively apply Aria's recommended config and grammar changes.

        Also detects code-level issues in reasoning and spawns repair agents.
        """
        if not suggestion.get("evidence_pack"):
            logger.warning("Skipping recommendation application: missing Evidence Pack.")
            return
        confidence = suggestion.get("confidence", 0)
        reasoning = str(suggestion.get("reasoning") or "")

        # Detect code-level issues in reasoning and spawn agent
        if confidence >= 0.3 and reasoning:
            self._maybe_spawn_agent_from_reasoning(reasoning, nb)

        if confidence < 0.4:
            return  # Low confidence — don't auto-apply config

        suggested_config = suggestion.get("config") or {}
        if not suggested_config:
            return

        # Categorize suggested keys into bins
        GRAMMAR_WEIGHT_KEYS = {"math_space_weight"}
        CATEGORY_WEIGHT_KEY = "category_weights"
        CONFIG_OVERRIDE_KEYS = {
            "n_programs", "model_dim", "max_depth", "max_ops",
            "model_source", "morph_focus_sparse",
            "use_synthesized_training", "novelty_weight",
            "selection_family_bonus_weight", "refinement_top_k",
            "refinement_generations", "refinement_budget_programs",
            "grammar_split_prob", "grammar_merge_prob",
            "grammar_risky_op_prob", "grammar_freq_domain_prob",
            "structured_sparsity_bias", "residual_prob",
            "optimizer_preference",
        }
        OP_CONTROL_KEYS = {"excluded_ops", "op_weights"}

        # Sanity clamps for numeric config values
        CLAMP_RANGES: Dict[str, Tuple[float, float]] = {
            "grammar_split_prob": (0.0, 1.0),
            "grammar_merge_prob": (0.0, 1.0),
            "grammar_risky_op_prob": (0.0, 1.0),
            "grammar_freq_domain_prob": (0.0, 1.0),
            "structured_sparsity_bias": (0.0, 1.0),
            "residual_prob": (0.0, 1.0),
            "n_programs": (4, 500),
            "max_depth": (2, 30),
            "max_ops": (3, 40),
            "model_dim": (32, 1024),
        }
        GRAMMAR_WEIGHT_CLAMP = (0.1, 10.0)  # category weights & math_space_weight
        OP_WEIGHT_CLAMP = (0.01, 10.0)

        def _clamp(val: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, val))

        grammar_overrides = {}
        config_overrides = {}
        for k, v in suggested_config.items():
            if k in GRAMMAR_WEIGHT_KEYS:
                if isinstance(v, (int, float)):
                    grammar_overrides[k] = _clamp(float(v), *GRAMMAR_WEIGHT_CLAMP)
            elif k == CATEGORY_WEIGHT_KEY and isinstance(v, dict):
                # Category weights dict → merge into grammar weight overrides
                for cat_name, weight in v.items():
                    if isinstance(weight, (int, float)):
                        grammar_overrides[cat_name] = _clamp(float(weight), *GRAMMAR_WEIGHT_CLAMP)
            elif k == "excluded_ops" and isinstance(v, list):
                new_excluded = {str(op) for op in v if isinstance(op, str)}
                if new_excluded:
                    self._excluded_ops_overrides |= new_excluded
                    nb.log_learning_event(
                        "auto_excluded_ops",
                        f"Aria excluded ops: {sorted(new_excluded)}",
                        excluded_ops=sorted(new_excluded),
                    )
                    logger.info("Aria auto-excluded ops: %s", sorted(new_excluded))
            elif k == "op_weights" and isinstance(v, dict):
                new_op_weights = {
                    str(op): _clamp(float(w), *OP_WEIGHT_CLAMP)
                    for op, w in v.items()
                    if isinstance(op, str) and isinstance(w, (int, float))
                }
                if new_op_weights:
                    self._op_weights_overrides.update(new_op_weights)
                    nb.log_learning_event(
                        "auto_op_weights",
                        f"Aria adjusted op weights: {new_op_weights}",
                        op_weights=new_op_weights,
                    )
                    logger.info("Aria auto-applied op weights: %s", new_op_weights)
            elif k in CONFIG_OVERRIDE_KEYS:
                if k in CLAMP_RANGES and isinstance(v, (int, float)):
                    lo, hi = CLAMP_RANGES[k]
                    v = type(v)(max(lo, min(hi, v)))
                config_overrides[k] = v

        if grammar_overrides:
            self._grammar_weight_overrides.update(grammar_overrides)
            nb.log_learning_event(
                "auto_grammar_adjusted",
                f"Aria proactively adjusted grammar weights: {grammar_overrides}",
                weights=grammar_overrides,
            )
            logger.info("Aria auto-applied grammar overrides: %s", grammar_overrides)

        if config_overrides:
            self._last_chat_config_overrides = {
                **(self._last_chat_config_overrides or {}),
                **config_overrides,
            }
            nb.log_learning_event(
                "auto_config_adjusted",
                f"Aria proactively adjusted config: {config_overrides}",
                changes=config_overrides,
            )
            logger.info("Aria auto-applied config overrides: %s", config_overrides)

    def _maybe_spawn_agent_from_reasoning(self, reasoning: str, nb: LabNotebook):
        """If Aria's reasoning mentions code issues, spawn a repair agent."""
        import re as _re
        # Detect code-issue signals in reasoning text
        code_issue_patterns = [
            r'\b(?:error|bug|crash|exception|traceback|broken|fails?|failing)\b.*\b(?:in|at|from)\s+\S+\.py\b',
            r'\b(?:fix|repair|patch|update)\b.*\b(?:code|file|module|function|class)\b',
            r'\bImportError\b|\bTypeError\b|\bAttributeError\b|\bNameError\b|\bSyntaxError\b',
            r'\b(?:missing|undefined|unresolved)\s+(?:import|module|function|method|attribute)\b',
        ]
        has_code_issue = any(
            _re.search(pat, reasoning, _re.IGNORECASE) for pat in code_issue_patterns
        )
        if not has_code_issue:
            return
        # Rate-limit: don't spawn more than 1 agent per 5 minutes from reasoning
        now = time.time()
        last = getattr(self, "_last_reasoning_agent_spawn", 0)
        if now - last < 300:
            return
        self._last_reasoning_agent_spawn = now
        try:
            from .api import _spawn_code_agent_task
            notebook_path = str(nb._db_path) if hasattr(nb, "_db_path") else ""
            task = _spawn_code_agent_task(
                goal=(
                    f"Aria's analysis identified a code issue: {reasoning[:600]}. "
                    "Investigate and fix. Use local Ollama model if available."
                ),
                notebook_path=notebook_path,
                allow_write=True,
            )
            task_id = task.get("task_id", "unknown")
            nb.log_learning_event(
                "proactive_reasoning_agent",
                f"Spawned agent {task_id} from recommendation reasoning: {reasoning[:200]}",
                task_id=task_id,
            )
            logger.info("Spawned reasoning-based repair agent: %s", task_id)
        except Exception as e:
            logger.debug("Failed to spawn reasoning-based agent: %s", e)

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
            valid_survivors = [
                s for s in survivors if s.get("novelty_valid_for_promotion", True)
            ]
            if not valid_survivors:
                return
            avg_novelty = (
                sum(s.get("novelty", 0) for s in valid_survivors)
                / len(valid_survivors)
            )
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
        evidence_pack = build_evidence_pack(
            nb,
            analytics=None,
            recommendation={"mode": "scale_up"},
            decision_type="auto_scale_up",
        )
        self._pending_scale_up["evidence_pack"] = evidence_pack

        self._emit_event("auto_scale_up_queued", {
            "result_ids": result_ids,
            "n_programs": len(result_ids),
            "reason": f"{s1_count} S1 survivors with avg novelty >= {config.auto_scale_up_min_novelty}",
            "evidence_pack": evidence_pack,
        })

        nb.add_entry(ExperimentEntry(
            entry_type="decision",
            title="Auto-Scale-Up Triggered",
            content=(
                f"Automatically queuing scale-up validation for {len(result_ids)} "
                f"top performers. Criteria met: {s1_count} S1 survivors."
            ),
            metadata={"result_ids": result_ids, "evidence_pack": evidence_pack},
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
                "efficiency_frontier_3d": analytics.efficiency_frontier_3d(),
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

                sparse_weight_options = (
                    "structured_sparse",
                    "semi_structured_2_4",
                    "block_sparse",
                )

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
                        fixed_choices: Dict[str, str] = {}
                        if bool(getattr(config, "morph_focus_sparse", False)):
                            explicit_sparse = str(getattr(config, "morph_sparse_weight_storage", "") or "").strip()
                            if explicit_sparse in sparse_weight_options:
                                fixed_choices["weight_storage"] = explicit_sparse
                            else:
                                fixed_choices["weight_storage"] = sparse_weight_options[i % len(sparse_weight_options)]
                        fixed_routing = str(getattr(config, "morph_compute_routing", "") or "").strip()
                        if fixed_routing:
                            fixed_choices["compute_routing"] = fixed_routing
                        fixed_channel = str(getattr(config, "morph_channel_mixing", "") or "").strip()
                        if fixed_channel:
                            fixed_choices["channel_mixing"] = fixed_channel

                        spec = roll(seed=i + int(time.time() * 1000) % 100000,
                                    generation=0,
                                    fixed=fixed_choices or None)
                        model = build_model(spec, build_cfg)
                        desc = describe_spec(spec)

                        # Quick smoke test
                        sandbox_result = self._safe_eval_for_stage(
                            model,
                            stage_tag="morph_candidate_gen",
                            batch_size=2,
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
        grammar = self._build_grammar_config(config)

        graphs = batch_generate(n, grammar)
        for graph in graphs:
            if self._stop_event.is_set():
                break
            validation = validate_graph(
                graph,
                max_ops=max(1, int(config.max_ops)),
                max_depth=max(1, int(config.max_depth)),
            )
            if not validation.valid:
                continue
            try:
                layer_graphs = [graph] * config.n_layers
                model = compile_model(
                    layer_graphs,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                )
                sandbox_result = self._safe_eval_for_stage(
                    model,
                    stage_tag="graph_candidate_gen",
                    batch_size=2,
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
        from research.scientist.perf import PerfTracer, GPUStarvationDetector, KernelTimer
        tracer = PerfTracer()
        starvation_detector = GPUStarvationDetector(threshold_ms=2.0)
        kernel_timer = KernelTimer(model, enabled=bool(getattr(config, "enable_kernel_profiling", False)))
        
        result: Dict[str, Any] = {"passed": False}

        try:
            with tracer.trace("model_setup"):
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

                starvation_detector.start_wait()
                with tracer.trace("data_sampling"):
                    input_ids = self._sample_training_input_ids(
                        config=config,
                        dev=dev,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        seed=seed + step,
                    )
                starvation_detector.end_wait()

                t_step = time.perf_counter()

                with tracer.trace("forward_pass"):
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

                with tracer.trace("backward_pass"):
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm_val).item()
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

                # Emit live training step events for dashboard
                ctx = getattr(self, "_live_training_context", None)
                if ctx and step % 25 == 0:
                    self._emit_event("training_step", {
                        "experiment_id": ctx.get("exp_id", ""),
                        "step": step,
                        "loss": round(loss_val, 6),
                        "total_steps": n_steps,
                        "phase": ctx.get("phase", ""),
                    })

            t_end = time.perf_counter()
            total_time_ms = (t_end - t_start) * 1000

            if initial_loss and final_loss:
                result["loss_ratio"] = final_loss / max(initial_loss, 1e-6)
                result["final_loss"] = final_loss
                result["initial_loss"] = initial_loss
                result["min_loss"] = min_loss
                result["throughput"] = total_tokens / (total_time_ms / 1000)
                result["passed"] = result["loss_ratio"] < config.stage1_loss_ratio_threshold

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

                # Extract architecture-specific telemetry (MoE, MoD, MoR, etc.)
                arch_telemetry = self._extract_architecture_telemetry(model)
                result.update(arch_telemetry)

        except Exception as e:
            result["error"] = str(e)

        # Finalize performance reports
        try:
            result["perf_report"] = tracer.get_report()
            # Ensure throughput is included in perf_report for experiment-level aggregation
            if isinstance(result.get("throughput"), (int, float)):
                result["perf_report"]["avg_throughput_tok_s"] = float(result["throughput"])
                
            result["starvation_report"] = starvation_detector.get_summary()
            if kernel_timer.enabled:
                result["kernel_timings_ms"] = kernel_timer.synchronize_and_get_timings()
        except Exception as e:
            result["perf_error"] = str(e)

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
                            hypothesis: Optional[str] = None,
                            preregistration: Optional[Dict[str, Any]] = None,
                            exploratory: bool = False,
                            force: bool = False) -> str:
        """Start investigation phase for selected candidates.

        Args:
            force: Skip tier and already-investigated guards.  Allows
                   re-investigating candidates with different config
                   (e.g. longer steps, different data mode).
        """
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        nb = self._make_notebook()

        if not force:
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
        else:
            logger.info("Force re-investigation: skipping tier/fingerprint guards for %s",
                        ", ".join(r[:8] for r in result_ids))
            # Reset tier to screening so the investigation can re-promote
            for rid in result_ids:
                try:
                    nb.conn.execute(
                        "UPDATE leaderboard SET tier = 'screening', "
                        "investigation_passed = NULL, investigation_loss_ratio = NULL, "
                        "investigation_robustness = NULL, investigation_best_training = NULL "
                        "WHERE result_id = ?", (rid,))
                except Exception:
                    pass
            try:
                nb.conn.commit()
            except Exception:
                pass

        source = "user_input" if hypothesis is not None else "runner_template"
        if hypothesis is None:
            hypothesis = (
                f"Investigation: deep study of {len(result_ids)} screening survivors "
                f"with multiple training programs to test robustness."
            )

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="investigation",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source=source,
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
            preregistration=preregistration,
            exploratory=exploratory,
            created_by="start_investigation",
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
        self._live_training_context = {"exp_id": exp_id, "phase": "investigation"}
        nb = self._make_notebook()
        t_start = time.time()
        ckpt = CheckpointManager(config.checkpoint_dir)

        # Informational: log pre-inv scores for user-triggered investigations
        if config.pre_inv_gate_enabled:
            for rid in result_ids:
                try:
                    row = nb.conn.execute(
                        "SELECT pre_inv_score FROM leaderboard WHERE result_id = ?",
                        (rid,)).fetchone()
                    if row and row[0] is not None:
                        logger.info("Investigation candidate %s pre_inv_score=%.1f",
                                    rid[:8], row[0])
                except Exception:
                    pass

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

            # Fetch all sources at once to avoid N+1 queries
            program_details = [d or {} for d in (nb.get_program_details(result_ids) or [])]
            source_map = {d.get("result_id"): d for d in program_details if d.get("result_id")}

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
                source = source_map.get(source_result_id)
                if source is None:
                    continue

                # Reconstruct model
                graph_json_str = source.get("graph_json")
                arch_spec_json_str = source.get("arch_spec_json")
                model_source = source.get("model_source") or "graph_synthesis"

                # Generate training programs (queue-level scheduling telemetry)
                training_programs, tp_sched = synthesize_training_program_batch(
                    n_programs=config.n_training_programs,
                    n_steps=config.investigation_steps,
                    max_seq_len=config.max_seq_len,
                    seed_offset=prog_idx * 1000,
                )
                results.setdefault("training_program_scheduling", []).append({
                    "result_id": source_result_id,
                    **tp_sched,
                })

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
                            spec_data = self._cached_json_load(arch_spec_json_str)
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

                # Skip candidates where no training program could reconstruct the model
                if not tp_results:
                    logger.debug(
                        f"Threaded investigation: skipping {source_result_id[:8]} — "
                        f"model failed to reconstruct for all {len(training_programs)} programs"
                    )
                    continue

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
                    "training_program_scheduling_avg_ms": tp_sched.get("scheduling_avg_ms"),
                    "training_program_scheduling_max_ms": tp_sched.get("scheduling_max_ms"),
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

                # Brittle risk override: if the investigation LR is good on
                # its own merits (< 0.3), don't let the screening→investigation
                # multiplier veto promotion.  Prevents false positives when
                # screening LR was unrealistically low (e.g. lucky seed).
                investigation_passed = (
                    robustness >= 0.5
                    and (best_lr or 1.0) < 0.5
                    and (not brittle_risk
                         or (best_lr is not None and best_lr < 0.3))
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
                    fp_jacobian_spectral_norm=source.get("fp_jacobian_spectral_norm"),
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
                    novelty_raw_score=source.get("novelty_raw_score"),
                    novelty_z_score=source.get("novelty_z_score"),
                    novelty_reference_version=source.get("novelty_reference_version"),
                    novelty_valid_for_promotion=source.get("novelty_valid_for_promotion"),
                    novelty_validity_reason=source.get("novelty_validity_reason"),
                    novelty_requires_justification=source.get("novelty_requires_justification"),
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
            logger.error("Investigation failed (%s): %s\n%s", exp_id, e, error)
            self._invoke_code_healer(
                nb=nb,
                trigger_type="repeated_exception",
                experiment_id=exp_id,
                scope=f"Investigation failure: {str(e)[:240]}",
                reproduction_steps=["python -m pytest tests/test_integration.py -k \"investigation\" -x --tb=short"],
                acceptance_tests=["python -m pytest tests/test_integration.py -k \"investigation\" -x --tb=short"],
                trigger_payload={"mode": "investigation", "error": str(e)},
            )
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
            self._live_training_context = None
            nb.close()
            self._run_pending_scale_up()

    # ── Validation Phase ──

    def start_validation(self, result_ids: List[str], config: RunConfig,
                         hypothesis: Optional[str] = None,
                         preregistration: Optional[Dict[str, Any]] = None,
                         exploratory: bool = False,
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

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="validation",
            config=self._validation_config_with_result_ids(config, result_ids, trigger),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source=source,
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
            preregistration=preregistration,
            exploratory=exploratory,
            created_by="start_validation",
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
        self._live_training_context = {"exp_id": exp_id, "phase": "validation"}
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

            # Fetch all sources at once to avoid N+1 queries
            program_details = [d or {} for d in (nb.get_program_details(result_ids) or [])]
            source_map = {d.get("result_id"): d for d in program_details if d.get("result_id")}

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
                source = source_map.get(source_result_id)
                if source is None:
                    continue

                graph_json_str = source.get("graph_json")
                arch_spec_json_str = source.get("arch_spec_json")
                model_source = source.get("model_source") or "graph_synthesis"

                # Get best training program from investigation
                leaderboard_entries = nb.get_leaderboard()
                best_tp_json = None
                for entry in leaderboard_entries:
                    if entry.get("result_id") == source_result_id:
                        best_tp_json = entry.get("investigation_best_training")
                        break

                # Multi-seed evaluation (threaded validation)
                seed_results = []
                for seed in range(config.validation_n_seeds):
                    if self._stop_event.is_set():
                        break

                    torch.manual_seed(seed * 42 + 7)

                    # Reconstruct model fresh
                    init_scheme = "default"
                    try:
                        if model_source == "morphological_box" and arch_spec_json_str:
                            from ..morphological_box import ArchSpec
                            from ..arch_builder import build_model, BuildConfig
                            spec_data = self._cached_json_load(arch_spec_json_str)
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
                        # Multi-init: use Xavier uniform for the last seed
                        if seed == config.validation_n_seeds - 1:
                            init_scheme = "xavier_uniform"
                            for p in model.parameters():
                                if p.dim() >= 2:
                                    nn.init.xavier_uniform_(p)
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
                    })

                    del model
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()


                # Skip candidates where no seed could reconstruct the model
                if not seed_results:
                    logger.debug(
                        f"Threaded validation: skipping {source_result_id[:8]} — "
                        f"model failed to reconstruct for all {config.validation_n_seeds} seeds"
                    )
                    continue

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

                # Init sensitivity: std between default and xavier seeds
                init_sensitivity_std = None
                default_losses = [
                    r["loss_ratio"] for r in seed_results
                    if r.get("init_scheme") == "default" and r.get("loss_ratio") is not None
                ]
                xavier_losses = [
                    r["loss_ratio"] for r in seed_results
                    if r.get("init_scheme") == "xavier_uniform" and r.get("loss_ratio") is not None
                ]
                if default_losses and xavier_losses:
                    default_mean = sum(default_losses) / len(default_losses)
                    xavier_mean = sum(xavier_losses) / len(xavier_losses)
                    init_sensitivity_std = abs(default_mean - xavier_mean)

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
                            bl_data_fn, bl_data_tag, bl_cache = self._make_baseline_data_fn(config)
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
                                cache_data_fn=bl_cache,
                            )
                            # Optional: Validation baseline comparison (using val split)
                            v_loss = best_seed.get("validation_loss")
                            if v_loss is not None:
                                try:
                                    v_data_fn, v_data_tag, v_cache = self._make_baseline_data_fn(config, split="val")
                                    v_baseline_ratio = baseline.compare(
                                        v_loss,
                                        d_model=config.model_dim,
                                        seq_len=min(128, int(getattr(config, "validation_seq_len", 128))),
                                        n_steps=max(1, baseline_steps),
                                        vocab_size=config.vocab_size,
                                        batch_size=int(getattr(config, "validation_batch_size", 4)),
                                        lr=baseline_recipe["lr"],
                                        device=dev_str,
                                        n_layers=config.n_layers,
                                        optimizer_name=baseline_recipe["optimizer_name"],
                                        weight_decay=baseline_recipe["weight_decay"],
                                        momentum=baseline_recipe["momentum"],
                                        betas=baseline_recipe["betas"],
                                        data_fn=v_data_fn,
                                        data_tag=v_data_tag,
                                        cache_data_fn=v_cache,
                                    )
                                    program_metrics["validation_baseline_loss_ratio"] = v_baseline_ratio
                                except Exception:
                                    pass
                        except Exception:
                            pass

                # Parameter-normalized baseline comparison
                val_normalized_ratio = None
                val_param_efficiency = None
                source_params = (source.get("param_count")
                                 or source.get("graph_n_params_estimate")
                                 or 0) if source else 0
                if loss_ratios and best_seed is not None and source_params > 0:
                    try:
                        baseline = self._get_baseline()
                        baseline_steps = int(best_seed.get("n_train_steps") or config.validation_steps)
                        baseline_recipe = self._resolve_baseline_recipe(
                            best_seed, default_lr=config.stage1_lr)
                        bl_data_fn, bl_data_tag, bl_cache = self._make_baseline_data_fn(config)
                        norm_result = baseline.compare_normalized(
                            best_seed["final_loss"],
                            program_params=int(source_params),
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
                            cache_data_fn=bl_cache,
                        )
                        val_normalized_ratio = norm_result.get("normalized_ratio")
                        val_param_efficiency = norm_result.get("param_efficiency")
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

                # Determine if breakthrough — requires both raw AND normalized thresholds
                ood_ok = (ood_result is not None
                          and ood_result.get("ood_robustness", 0) >= 0.67)
                hp_ok = (sensitivity_result is not None
                         and sensitivity_result.get("hp_robustness", 0) >= 0.75)
                nov_conf = source.get("novelty_confidence", 0) if source else 0
                novelty_valid = False
                if source:
                    novelty_valid = bool(source.get("novelty_valid_for_promotion"))
                    if not novelty_valid and source.get("cka_source") == "artifact":
                        novelty_valid = True

                raw_threshold = config.breakthrough_raw_threshold
                norm_threshold = config.breakthrough_normalized_threshold
                raw_ok = (val_baseline_ratio is not None
                          and val_baseline_ratio < raw_threshold)
                norm_ok = (val_normalized_ratio is None
                           or val_normalized_ratio < norm_threshold)
                is_breakthrough = (
                    raw_ok
                    and norm_ok
                    and multi_seed_std <= 0.03
                    and len(passed_seeds) >= 5
                    and len(passed_seeds) == config.validation_n_seeds
                    and (ood_result is None or ood_ok)
                    and (sensitivity_result is None or hp_ok)
                    and nov_conf >= 0.5
                    and novelty_valid
                )

                # FLOP gate: reject breakthrough if >5x baseline FLOPs per token
                flop_gated = False
                if is_breakthrough and source_params > 0:
                    candidate_fpt = source_params * 2.0
                    baseline_fpt_gate = 2.0 * config.model_dim ** 2 * config.n_layers
                    if candidate_fpt > 5.0 * baseline_fpt_gate:
                        is_breakthrough = False
                        flop_gated = True
                        logger.info(
                            "FLOP gate downgraded %s: %.0f FPT > 5x baseline %.0f",
                            source_result_id[:8], candidate_fpt, baseline_fpt_gate,
                        )

                # Scaling law comparison gate
                scaling_result = None
                scaling_param_efficiency = None
                scaling_flop_efficiency = None
                scaling_gate_passed_val = None
                scaling_best_family = None
                scaling_confidence = None
                if is_breakthrough and config.enable_scaling_comparison:
                    try:
                        scaling_mgr = self._get_scaling_reference_manager()
                        bl_data_fn, bl_data_tag, _ = self._make_baseline_data_fn(config)
                        candidate_flops = (source.get("flops_forward", 0) or 0)
                        if candidate_flops <= 0:
                            candidate_flops = source_params * 2

                        scaling_result = scaling_mgr.compare_candidate(
                            candidate_loss=best_seed_loss,
                            candidate_params=source_params,
                            candidate_flops=candidate_flops,
                            d_model=config.model_dim,
                            n_steps=config.validation_steps,
                            seq_len=config.validation_seq_len,
                            vocab_size=config.vocab_size,
                            batch_size=config.validation_batch_size,
                            lr=config.stage1_lr,
                            device=dev_str,
                            data_fn=bl_data_fn, data_tag=bl_data_tag,
                            families=config.scaling_reference_families.split(","),
                            param_efficiency_target=config.scaling_param_efficiency_target,
                            flop_ceiling=config.scaling_flop_ceiling,
                        )
                        scaling_param_efficiency = scaling_result.best_param_efficiency
                        scaling_flop_efficiency = scaling_result.flop_efficiency
                        scaling_gate_passed_val = scaling_result.scaling_gate_passed
                        scaling_best_family = scaling_result.best_param_efficiency_family
                        scaling_confidence = scaling_result.confidence

                        if not scaling_result.scaling_gate_passed:
                            is_breakthrough = False
                            logger.info(
                                "Scaling gate downgraded %s: param_eff=%.2f (need %.1f), flop_eff=%.2f",
                                source_result_id[:8],
                                scaling_result.best_param_efficiency,
                                config.scaling_param_efficiency_target,
                                scaling_result.flop_efficiency,
                            )
                    except Exception as e:
                        logger.debug("Scaling comparison failed: %s", e)

                # Quantization eval: test INT8 retention for all validation candidates
                quant_int8_retention = None
                quant_quality_per_byte = None
                if best_seed is not None:
                    try:
                        from ..eval.quantization import evaluate_sparse_quant_quality
                        if model_source == "morphological_box" and arch_spec_json_str:
                            from ..morphological_box import ArchSpec
                            from ..arch_builder import build_model, BuildConfig
                            _spec = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len)
                            quant_model = build_model(_spec, _bc).to(dev)
                        else:
                            quant_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len).to(dev)
                        quant_batches = [
                            torch.randint(0, config.vocab_size,
                                          (2, min(128, config.validation_seq_len)),
                                          device=dev)
                            for _ in range(4)
                        ]
                        quant_result = evaluate_sparse_quant_quality(
                            quant_model, quant_batches, dev,
                            target_sparsity=0.5, bits=8)
                        if quant_result is not None:
                            quant_int8_retention = quant_result.get("full_retention")
                            quant_quality_per_byte = quant_result.get("quality_per_byte")
                            if is_breakthrough and quant_int8_retention is not None and quant_int8_retention < 0.80:
                                is_breakthrough = False
                                logger.info(
                                    "Quant gate downgraded %s: INT8 retention=%.3f < 0.80",
                                    source_result_id[:8], quant_int8_retention,
                                )
                        del quant_model
                    except Exception as e:
                        logger.debug("Quantization eval skipped: %s", e)

                # Long-context sweep (informational, non-blocking)
                long_context_score = None
                if best_seed is not None:
                    try:
                        from ..eval.long_context import run_long_context_sweep
                        base_loss_val = best_seed.get("final_loss", 0)
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _asjs_lc2 = arch_spec_json_str
                            _cfg_lc2 = config
                            def _make_model_lc2():
                                from ..morphological_box import ArchSpec
                                from ..arch_builder import build_model, BuildConfig
                                _sp2 = ArchSpec(**json.loads(_asjs_lc2))
                                _bc3 = BuildConfig(
                                    dim=_cfg_lc2.model_dim, n_layers=_cfg_lc2.n_layers,
                                    vocab_size=_cfg_lc2.vocab_size, max_seq_len=1024)
                                return build_model(_sp2, _bc3)
                        else:
                            _gjs_lc2 = graph_json_str
                            _cfg_lc2 = config
                            def _make_model_lc2():
                                return compile_model(
                                    [graph_from_json(_gjs_lc2)] * _cfg_lc2.n_layers,
                                    vocab_size=_cfg_lc2.vocab_size, max_seq_len=1024)
                        from ..eval.long_context import run_long_context_sweep
                        from ..eval.passkey import evaluate_long_context_retrieval

                        lc_result = run_long_context_sweep(
                            _make_model_lc2, config.vocab_size, dev,
                            base_loss=base_loss_val, seq_lens=(512, 1024),
                            n_steps=200, batch_size=2,
                        )
                        
                        # Retrieval test (needle-in-a-haystack)
                        retr_model = _make_model_lc2().to(dev)
                        retr_result = evaluate_long_context_retrieval(
                            retr_model, config.vocab_size, dev,
                            lengths=[256, 512, 1024]
                        )
                        del retr_model
                        
                        # Combine scaling score and retrieval score (50/50)
                        scaling_score = lc_result.get("long_context_score", 0.0)
                        retrieval_score = retr_result.get("retrieval_score", 0.0)
                        long_context_score = (scaling_score * 0.5) + (retrieval_score * 0.5)
                        
                        logger.info("Long-context check (v2): scaling=%.2f, retrieval=%.2f, combined=%.2f",
                                    scaling_score, retrieval_score, long_context_score)
                    except Exception as e:
                        logger.debug("Long-context sweep skipped: %s", e)

                # Noise sensitivity (informational, non-blocking)
                noise_score = None
                if best_seed is not None:
                    try:
                        from ..eval.noise_sensitivity import evaluate_noise_sensitivity
                        if model_source == "morphological_box" and arch_spec_json_str:
                            from ..morphological_box import ArchSpec
                            from ..arch_builder import build_model, BuildConfig
                            _spec_ns2 = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc_ns2 = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len)
                            ns_model = build_model(_spec_ns2, _bc_ns2).to(dev)
                        else:
                            ns_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len).to(dev)
                        ns_batches = [
                            torch.randint(0, config.vocab_size,
                                          (2, min(128, config.validation_seq_len)),
                                          device=dev)
                            for _ in range(4)
                        ]
                        ns_result = evaluate_noise_sensitivity(
                            ns_model, ns_batches, dev)
                        noise_score = ns_result.get("noise_sensitivity_score")
                        del ns_model
                    except Exception as e:
                        logger.debug("Noise sensitivity skipped: %s", e)

                # Activation sparsity analysis (informational, non-blocking)
                activation_sparsity_score = None
                dead_neuron_ratio = None
                if best_seed is not None:
                    try:
                        from ..eval.sparsity import evaluate_activation_sparsity
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _spec_as = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc_as = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len)
                            as_model = build_model(_spec_as, _bc_as).to(dev)
                        else:
                            as_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len).to(dev)
                        as_batches = [
                            torch.randint(0, config.vocab_size,
                                          (2, min(128, config.validation_seq_len)),
                                          device=dev)
                            for _ in range(4)
                        ]
                        as_result = evaluate_activation_sparsity(
                            as_model, as_batches, dev)
                        activation_sparsity_score = as_result.get("activation_sparsity_score")
                        dead_neuron_ratio = as_result.get("dead_neuron_ratio")
                        del as_model
                    except Exception as e:
                        logger.debug("Activation sparsity eval skipped: %s", e)

                # Routing heatmap / collapse detection (informational, non-blocking)
                routing_collapse_score = None
                if best_seed is not None:
                    try:
                        from ..eval.routing_heatmap import evaluate_routing_heatmap
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _spec_rh = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc_rh = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len)
                            rh_model = build_model(_spec_rh, _bc_rh).to(dev)
                        else:
                            rh_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len).to(dev)
                        rh_batches = [
                            torch.randint(0, config.vocab_size,
                                          (2, min(128, config.validation_seq_len)),
                                          device=dev)
                            for _ in range(4)
                        ]
                        rh_result = evaluate_routing_heatmap(
                            rh_model, rh_batches, dev)
                        if rh_result.get("has_routing"):
                            routing_collapse_score = rh_result.get("routing_collapse_score")
                        del rh_model
                    except Exception as e:
                        logger.debug("Routing heatmap eval skipped: %s", e)

                # WikiText perplexity (informational, non-blocking)
                wikitext_perplexity = None
                wikitext_score = None
                if best_seed is not None:
                    try:
                        from ..eval.wikitext_eval import evaluate_wikitext_perplexity
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _spec_wt = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc_wt = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len)
                            wt_model = build_model(_spec_wt, _bc_wt).to(dev)
                        else:
                            wt_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len).to(dev)
                        wt_result = evaluate_wikitext_perplexity(
                            wt_model, config.vocab_size, dev,
                            n_train_steps=200, seq_len=min(128, config.validation_seq_len))
                        wikitext_perplexity = wt_result.get("wikitext_perplexity")
                        wikitext_score = wt_result.get("wikitext_score")
                        if wikitext_perplexity is not None:
                            logger.info("WikiText ppl=%.1f score=%.3f",
                                        wikitext_perplexity, wikitext_score or 0)
                        del wt_model
                    except Exception as e:
                        logger.debug("WikiText eval skipped: %s", e)

                # TinyStories validation (informational, non-blocking)
                tinystories_perplexity = None
                tinystories_score = None
                if best_seed is not None:
                    try:
                        from ..eval.tinystories_eval import evaluate_tinystories
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _spec_ts = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc_ts = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len)
                            ts_model = build_model(_spec_ts, _bc_ts).to(dev)
                        else:
                            ts_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size,
                                max_seq_len=config.validation_seq_len).to(dev)
                        ts_result = evaluate_tinystories(
                            ts_model, config.vocab_size, dev,
                            n_train_steps=200, seq_len=min(128, config.validation_seq_len))
                        tinystories_perplexity = ts_result.get("tinystories_perplexity")
                        tinystories_score = ts_result.get("tinystories_score")
                        del ts_model
                    except Exception as e:
                        logger.debug("TinyStories eval skipped: %s", e)

                # Cross-task robustness (informational, non-blocking)
                cross_task_score = None
                if best_seed is not None:
                    try:
                        from ..eval.cross_task_eval import evaluate_cross_task_robustness
                        _gjs_ct = graph_json_str
                        _asjs_ct = arch_spec_json_str
                        _ms_ct = model_source
                        _cfg_ct = config
                        def _make_ct_model():
                            if _ms_ct == "morphological_box" and _asjs_ct:
                                _sp = ArchSpec(**json.loads(_asjs_ct))
                                _bc = BuildConfig(
                                    dim=_cfg_ct.model_dim, n_layers=_cfg_ct.n_layers,
                                    vocab_size=_cfg_ct.vocab_size,
                                    max_seq_len=_cfg_ct.validation_seq_len)
                                return build_model(_sp, _bc)
                            return compile_model(
                                [graph_from_json(_gjs_ct)] * _cfg_ct.n_layers,
                                vocab_size=_cfg_ct.vocab_size,
                                max_seq_len=_cfg_ct.validation_seq_len)
                        ct_result = evaluate_cross_task_robustness(
                            _make_ct_model, config.vocab_size, dev,
                            n_train_steps=100, seq_len=min(128, config.validation_seq_len))
                        cross_task_score = ct_result.get("cross_task_score")
                    except Exception as e:
                        logger.debug("Cross-task eval skipped: %s", e)

                # Efficiency wall (informational, non-blocking)
                efficiency_wall_score = None
                max_viable_seq_len = None
                scaling_regime = None
                if best_seed is not None:
                    try:
                        from ..eval.efficiency_wall import evaluate_efficiency_wall
                        if model_source == "morphological_box" and arch_spec_json_str:
                            _spec_ew = ArchSpec(**json.loads(arch_spec_json_str))
                            _bc_ew = BuildConfig(
                                dim=config.model_dim, n_layers=config.n_layers,
                                vocab_size=config.vocab_size, max_seq_len=1024)
                            ew_model = build_model(_spec_ew, _bc_ew).to(dev)
                        else:
                            ew_model = compile_model(
                                [graph_from_json(graph_json_str)] * config.n_layers,
                                vocab_size=config.vocab_size, max_seq_len=1024).to(dev)
                        ew_result = evaluate_efficiency_wall(
                            ew_model, config.vocab_size, dev,
                            seq_lens=(64, 128, 256, 512), batch_size=2)
                        efficiency_wall_score = ew_result.get("efficiency_wall_score")
                        max_viable_seq_len = ew_result.get("max_viable_seq_len")
                        scaling_regime = ew_result.get("scaling_regime")
                        del ew_model
                    except Exception as e:
                        logger.debug("Efficiency wall eval skipped: %s", e)

                tier = "breakthrough" if is_breakthrough else "validation"

                validation_entry = {
                    "result_id": source_result_id,
                    "val_loss_ratio": val_loss_ratio,
                    "val_baseline_ratio": val_baseline_ratio,
                    "val_normalized_ratio": val_normalized_ratio,
                    "param_efficiency": val_param_efficiency,
                    "multi_seed_std": multi_seed_std,
                    "seeds_passed": len(passed_seeds),
                    "total_seeds": config.validation_n_seeds,
                    "is_breakthrough": is_breakthrough,
                    "flop_gated": flop_gated,
                    "quant_int8_retention": quant_int8_retention,
                    "quant_quality_per_byte": quant_quality_per_byte,
                    "long_context_score": long_context_score,
                    "noise_sensitivity_score": noise_score,
                    "init_sensitivity_std": init_sensitivity_std,
                    "novelty_confidence": nov_conf,
                    "ood_robustness": ood_result,
                    "sensitivity": sensitivity_result,
                    "activation_sparsity_score": activation_sparsity_score,
                    "dead_neuron_ratio": dead_neuron_ratio,
                    "routing_collapse_score": routing_collapse_score,
                    "wikitext_perplexity": wikitext_perplexity,
                    "wikitext_score": wikitext_score,
                    "tinystories_perplexity": tinystories_perplexity,
                    "tinystories_score": tinystories_score,
                    "cross_task_score": cross_task_score,
                    "efficiency_wall_score": efficiency_wall_score,
                    "max_viable_seq_len": max_viable_seq_len,
                    "scaling_regime": scaling_regime,
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
                            normalized_baseline_ratio=val_normalized_ratio,
                            param_efficiency=val_param_efficiency,
                            quant_int8_retention=quant_int8_retention,
                            quant_quality_per_byte=quant_quality_per_byte,
                            robustness_long_ctx_score=long_context_score,
                            robustness_noise_score=noise_score,
                            init_sensitivity_std=init_sensitivity_std,
                            fp_jacobian_spectral_norm=source.get("fp_jacobian_spectral_norm"),
                            scaling_param_efficiency=scaling_param_efficiency,
                            scaling_flop_efficiency=scaling_flop_efficiency,
                            scaling_gate_passed=scaling_gate_passed_val,
                            scaling_best_family=scaling_best_family,
                            scaling_confidence=scaling_confidence,
                            activation_sparsity_score=activation_sparsity_score,
                            dead_neuron_ratio=dead_neuron_ratio,
                            routing_collapse_score=routing_collapse_score,
                            wikitext_perplexity=wikitext_perplexity,
                            wikitext_score=wikitext_score,
                            tinystories_perplexity=tinystories_perplexity,
                            tinystories_score=tinystories_score,
                            cross_task_score=cross_task_score,
                            efficiency_wall_score=efficiency_wall_score,
                            max_viable_seq_len=max_viable_seq_len,
                            scaling_regime=scaling_regime,
                        )
                        # Store detailed scaling result
                        if scaling_result is not None:
                            nb.set_external_benchmarks(
                                source_result_id, scaling_result.to_dict())
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
                    novelty_raw_score=source.get("novelty_raw_score"),
                    novelty_z_score=source.get("novelty_z_score"),
                    novelty_reference_version=source.get("novelty_reference_version"),
                    novelty_valid_for_promotion=source.get("novelty_valid_for_promotion"),
                    novelty_validity_reason=source.get("novelty_validity_reason"),
                    novelty_requires_justification=source.get("novelty_requires_justification"),
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
            logger.error("Validation failed (%s): %s\n%s", exp_id, e, error)
            self._invoke_code_healer(
                nb=nb,
                trigger_type="repeated_exception",
                experiment_id=exp_id,
                scope=f"Validation failure: {str(e)[:240]}",
                reproduction_steps=["python -m pytest tests/test_integration.py -k \"validation\" -x --tb=short"],
                acceptance_tests=["python -m pytest tests/test_integration.py -k \"validation\" -x --tb=short"],
                trigger_payload={"mode": "validation", "error": str(e)},
            )
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
            self._live_training_context = None
            nb.close()

    # ── Auto-Escalation Pipeline ──

    def _build_grammar_config(self, config: RunConfig,
                              excluded_ops: Optional[Set[str]] = None,
                              op_weights: Optional[Dict[str, float]] = None) -> GrammarConfig:
        """Create a GrammarConfig from a RunConfig with standardized defaults."""
        from ..synthesis.grammar import GrammarConfig
        
        # Pick up structured_sparsity_bias from mode recommendation or config
        sparsity_bias = getattr(self, "_structured_sparsity_bias_override",
                                getattr(config, "structured_sparsity_bias", 0.0))

        grammar = GrammarConfig(
            model_dim=config.model_dim,
            min_depth=config.min_depth,
            max_depth=min(config.max_depth, 12),
            max_ops=min(config.max_ops, 20),
            residual_prob=config.residual_prob,
            split_prob=config.grammar_split_prob,
            merge_prob=config.grammar_merge_prob,
            risky_op_prob=config.grammar_risky_op_prob,
            freq_domain_prob=config.grammar_freq_domain_prob,
            structured_sparsity_bias=sparsity_bias,
            excluded_ops=excluded_ops or set(),
            op_weights=op_weights or {},
        )
        # Apply specialized weights
        grammar.category_weights["math_space"] = config.math_space_weight

        # Apply Bayesian op priors from compressed learning (optional)
        try:
            from pathlib import Path
            import json as _json
            priors_path = Path("research/runtime/learning/op_priors.json")
            if priors_path.exists():
                payload = _json.loads(priors_path.read_text())
                op_penalties = payload.get("op_penalties", {}) if isinstance(payload, dict) else {}
                if isinstance(op_penalties, dict):
                    for op_name, penalty in op_penalties.items():
                        try:
                            p = float(penalty)
                        except Exception:
                            continue
                        # Convert penalty (0..1) into weight multiplier (1..0.5)
                        mult = max(0.5, 1.0 - 0.5 * max(0.0, min(1.0, p)))
                        grammar.op_weights[op_name] = grammar.op_weights.get(op_name, 1.0) * mult
        except Exception:
            pass

        # Apply cluster-based suggestions (optional)
        try:
            from pathlib import Path
            import json as _json
            sugg_path = Path("research/runtime/learning/cluster_suggestions.json")
            if sugg_path.exists():
                payload = _json.loads(sugg_path.read_text())
                if isinstance(payload, dict):
                    op_weight_suggestions = payload.get("op_weight_suggestions") or payload.get("op_weights") or {}
                    op_penalties = payload.get("op_penalties") or {}
                    op_promotions = payload.get("op_promotions") or {}
                    avoid_patterns = payload.get("avoid_patterns") or []
                    promote_patterns = payload.get("promote_patterns") or []

                    def _apply_mult(op_name: str, mult: float):
                        if not op_name:
                            return
                        m = max(0.2, min(3.0, float(mult)))
                        grammar.op_weights[op_name] = grammar.op_weights.get(op_name, 1.0) * m

                    for op_name, mult in op_weight_suggestions.items():
                        try:
                            _apply_mult(op_name, float(mult))
                        except Exception:
                            continue

                    for op_name, p in op_penalties.items():
                        try:
                            penalty = max(0.0, min(1.0, float(p)))
                        except Exception:
                            continue
                        _apply_mult(op_name, 1.0 - 0.4 * penalty)

                    for op_name, p in op_promotions.items():
                        try:
                            promo = max(0.0, min(1.0, float(p)))
                        except Exception:
                            continue
                        _apply_mult(op_name, 1.0 + 0.4 * promo)

                    def _ops_from_pattern(pat: str):
                        if "->" in pat:
                            parts = [p.strip() for p in pat.split("->", 1)]
                        elif "," in pat:
                            parts = [p.strip() for p in pat.split(",", 1)]
                        else:
                            parts = [pat.strip()]
                        return [p for p in parts if p]

                    for pat in avoid_patterns:
                        for op_name in _ops_from_pattern(str(pat)):
                            _apply_mult(op_name, 0.85)
                    for pat in promote_patterns:
                        for op_name in _ops_from_pattern(str(pat)):
                            _apply_mult(op_name, 1.1)
        except Exception:
            pass

        return grammar

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
            selection = self._score_candidate_pool(
                candidates=top,
                config=config,
                nb=nb,
                context="auto_investigate_screening",
                experiment_id=exp_id,
            )
            scored_by_id = {s["result_id"]: s for s in selection.get("scored", [])}
            ranked = selection.get("selected", [])
            candidate_ids = []
            for item in ranked:
                row = next((p for p in top if p.get("result_id") == item["result_id"]), None)
                if row is None:
                    continue
                if not row.get("stage1_passed"):
                    continue
                if row.get("loss_ratio") is not None and float(row.get("loss_ratio")) >= 0.75:
                    continue
                candidate_ids.append(item["result_id"])
                if len(candidate_ids) >= config.auto_investigate_top_n:
                    break

            if len(candidate_ids) < config.auto_investigate_min_survivors:
                return
            selected_rows = [p for p in top if p.get("result_id") in candidate_ids]
            decision_payload = {
                "decision_id": str(uuid.uuid4())[:12],
                "timestamp": time.time(),
                "context": "auto_investigate_screening",
                "experiment_id": exp_id,
                "candidate_pool_summary": selection.get("summary", {}),
                "score_breakdown": selection.get("scored", []),
                "policy": selection.get("policy", {}),
                "reason": selection.get("reason", ""),
                "chosen_experiments": [
                    {
                        "result_id": rid,
                        "family": scored_by_id.get(rid, {}).get("family"),
                        "score": scored_by_id.get(rid, {}).get("score"),
                    }
                    for rid in candidate_ids
                ],
                "trigger": None,
            }
            try:
                validate_selection_decision_log(decision_payload)
                decision_id = nb.record_selection_decision(
                    context=decision_payload["context"],
                    experiment_id=decision_payload["experiment_id"],
                    candidate_pool_summary=decision_payload["candidate_pool_summary"],
                    score_breakdown=decision_payload["score_breakdown"],
                    policy=decision_payload["policy"],
                    reason=decision_payload["reason"],
                    chosen_experiments=decision_payload["chosen_experiments"],
                    trigger=None,
                )
                supporting_insight_ids = selection.get("supporting_insight_ids") or []
                if supporting_insight_ids:
                    nb.record_selection_insight_trial(
                        decision_id=decision_id,
                        context=decision_payload["context"],
                        insight_ids=supporting_insight_ids,
                        chosen_result_ids=candidate_ids,
                        source_experiment_id=exp_id,
                    )
            except Exception as sel_err:
                logger.debug("Auto-investigate selection logging failed: %s", sel_err)

            # Go/no-go decision for each candidate
            if config.auto_go_no_go and config.enable_campaigns:
                approved_ids = []
                for p in selected_rows:
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
                        evidence_pack = self._safe_build_evidence_pack(
                            nb,
                            recommendation={"mode": "investigation"},
                            decision_type="go_no_go",
                        )
                        nb.record_decision(
                            campaign_id=self._active_campaign_id,
                            decision_type=decision["decision"],
                            subject=f"Promote {p['result_id'][:8]} to investigation",
                            rationale=decision["rationale"],
                            evidence_ids=[p["result_id"]],
                            alternatives=[{"considered": decision.get("alternatives", "")}],
                            evidence_pack=evidence_pack,
                        )
                        self._emit_event("decision_recorded", {
                            "decision_type": decision["decision"],
                            "subject": p["result_id"][:8],
                            "rationale": decision["rationale"][:200],
                            "evidence_pack": evidence_pack,
                        })
                        if decision["decision"] in ("go", "pivot"):
                            approved_ids.append(p["result_id"])
                    except Exception as e:
                        logger.debug(f"Go/no-go failed for {p['result_id']}: {e}")
                        approved_ids.append(p["result_id"])

                candidate_ids = approved_ids if approved_ids else candidate_ids
                selected_rows = [p for p in selected_rows if p.get("result_id") in candidate_ids]

            for rid in candidate_ids:
                score_row = scored_by_id.get(rid)
                if not score_row:
                    continue
                reward = score_row.get("base_score", 0.0)
                nb.update_selection_family_stats(
                    score_row.get("family", "Unknown"),
                    reward=float(reward),
                )

            # Add to leaderboard as screening tier (skip if already at screening or above)
            existing_lb = {
                e["result_id"]: e["tier"]
                for e in nb.get_leaderboard(limit=500)
            }
            for p in selected_rows:
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
                        fp_jacobian_spectral_norm=p.get("fp_jacobian_spectral_norm"),
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
            evidence_pack = self._safe_build_evidence_pack(
                nb,
                recommendation={"mode": "investigation"},
                decision_type="auto_investigate",
            )
            self._pending_investigation["evidence_pack"] = evidence_pack

            self._emit_event("auto_investigate_queued", {
                "result_ids": candidate_ids,
                "n_candidates": len(candidate_ids),
                "reason": f"{s1_count} S1 survivors with loss_ratio < 0.5",
                "evidence_pack": evidence_pack,
            })

            nb.add_entry(ExperimentEntry(
                entry_type="decision",
                title="Auto-Investigation Triggered",
                content=(
                    f"Automatically queuing investigation for {len(candidate_ids)} "
                    f"top performers. Criteria: {s1_count} S1 survivors."
                ),
                metadata={"result_ids": candidate_ids, "evidence_pack": evidence_pack},
            ))

            # Z7: Algorithmic Sparsity Bias Learning
            try:
                sparse_wins = [p for p in top if (p.get("sparsity_ratio") or 0) > 0.3]
                dense_wins = [p for p in top if (p.get("sparsity_ratio") or 0) <= 0.3]
                if sparse_wins and dense_wins:
                    avg_sparse_loss = sum(p.get("loss_ratio", 1.0) for p in sparse_wins) / len(sparse_wins)
                    avg_dense_loss = sum(p.get("loss_ratio", 1.0) for p in dense_wins) / len(dense_wins)
                    
                    if avg_sparse_loss < avg_dense_loss * 0.95: # 5% better
                        delta = 0.1
                        old_bias = config.grammar_config.structured_sparsity_bias
                        config.grammar_config.update_bias(delta)
                        nb.log_learning_event(
                            event_type="grammar_adjustment",
                            description=f"Boosted structured_sparsity_bias by {delta} due to sparse dominance.",
                            old_weights={"bias": old_bias},
                            new_weights={"bias": config.grammar_config.structured_sparsity_bias},
                            evidence=f"avg_sparse_loss={avg_sparse_loss:.4f}, avg_dense_loss={avg_dense_loss:.4f}"
                        )
            except Exception as z7_err:
                logger.debug("Z7 learning logic failed: %s", z7_err)

        elif phase == "investigation":
            # After investigation: queue validation if strong candidates
            if not config.auto_validate:
                return

            inv_results = results.get("investigation_results", [])
            inv_ids = [r.get("result_id") for r in inv_results if r.get("result_id")]
            novelty_meta: Dict[str, Dict[str, Any]] = {}
            if inv_ids:
                placeholders = ",".join("?" for _ in inv_ids)
                rows = nb.conn.execute(
                    f"""SELECT result_id, novelty_valid_for_promotion, cka_source
                        FROM program_results
                        WHERE result_id IN ({placeholders})""",
                    tuple(inv_ids),
                ).fetchall()
                novelty_meta = {row["result_id"]: dict(row) for row in rows}

            strong = []
            for r in inv_results:
                rid = r.get("result_id")
                meta = novelty_meta.get(rid or "", {})
                if not meta:
                    novelty_valid = True  # legacy records pre-date novelty validity fields
                else:
                    novelty_valid = bool(meta.get("novelty_valid_for_promotion"))
                    if not novelty_valid and meta.get("cka_source") == "artifact":
                        novelty_valid = True
                if not novelty_valid and config.allow_heuristic_novelty_promotion:
                    novelty_valid = bool(str(config.heuristic_novelty_justification or "").strip())
                if (
                    r.get("robustness", 0) >= config.auto_validate_min_robustness
                    and (r.get("best_loss_ratio") or 1.0) < 0.6
                    and r.get("baseline_loss_ratio") is not None
                    and r.get("baseline_loss_ratio") < config.auto_validate_max_baseline_ratio
                    and r.get("novelty_confidence") is not None
                    and r.get("novelty_confidence") >= config.auto_validate_min_novelty_confidence
                    and novelty_valid
                    and not r.get("brittle_risk", False)
                    and (
                        r.get("loss_ratio_multiplier") is None
                        or r.get("loss_ratio_multiplier") <= config.investigation_max_loss_ratio_multiplier
                    )
                ):
                    strong.append(r)

            if not strong:
                return

            result_ids_all = [r.get("result_id") for r in strong if r.get("result_id")]
            graph_meta: Dict[str, Dict[str, Any]] = {}
            if result_ids_all:
                placeholders = ",".join("?" for _ in result_ids_all)
                rows = nb.conn.execute(
                    f"""SELECT result_id, graph_json, routing_mode
                        FROM program_results
                        WHERE result_id IN ({placeholders})""",
                    tuple(result_ids_all),
                ).fetchall()
                graph_meta = {row["result_id"]: dict(row) for row in rows}

            prepared_candidates: List[Dict[str, Any]] = []
            for r in strong:
                rid = r.get("result_id")
                if not rid:
                    continue
                meta = graph_meta.get(rid, {})
                prepared_candidates.append({
                    "result_id": rid,
                    "graph_json": meta.get("graph_json"),
                    "routing_mode": meta.get("routing_mode"),
                    "loss_ratio": r.get("best_loss_ratio"),
                    "baseline_loss_ratio": r.get("baseline_loss_ratio"),
                    "novelty_score": r.get("novelty_confidence"),
                    "throughput_tok_s": r.get("throughput_tok_s"),
                    "flops_per_token": r.get("flops_per_token"),
                    "peak_memory_mb": r.get("peak_memory_mb"),
                    "stage0_passed": 1,
                    "stage05_passed": 1,
                    "stage1_passed": 1,
                    "stability_score": r.get("robustness"),
                    "has_nan_grad": 0,
                    "has_zero_grad": 0,
                })

            selection = self._score_candidate_pool(
                candidates=prepared_candidates,
                config=config,
                nb=nb,
                context="auto_validate_investigation",
                experiment_id=results.get("experiment_id"),
            )
            scored_by_id = {s["result_id"]: s for s in selection.get("scored", [])}
            ranked = selection.get("selected", [])
            candidate_ids = [item["result_id"] for item in ranked[:config.auto_validate_top_n]]
            decision_payload = {
                "decision_id": str(uuid.uuid4())[:12],
                "timestamp": time.time(),
                "context": "auto_validate_investigation",
                "experiment_id": results.get("experiment_id"),
                "candidate_pool_summary": selection.get("summary", {}),
                "score_breakdown": selection.get("scored", []),
                "policy": selection.get("policy", {}),
                "reason": selection.get("reason", ""),
                "chosen_experiments": [
                    {
                        "result_id": rid,
                        "family": scored_by_id.get(rid, {}).get("family"),
                        "score": scored_by_id.get(rid, {}).get("score"),
                    }
                    for rid in candidate_ids
                ],
                "trigger": None,
            }
            try:
                validate_selection_decision_log(decision_payload)
                decision_id = nb.record_selection_decision(
                    context=decision_payload["context"],
                    experiment_id=decision_payload["experiment_id"],
                    candidate_pool_summary=decision_payload["candidate_pool_summary"],
                    score_breakdown=decision_payload["score_breakdown"],
                    policy=decision_payload["policy"],
                    reason=decision_payload["reason"],
                    chosen_experiments=decision_payload["chosen_experiments"],
                    trigger=None,
                )
                supporting_insight_ids = selection.get("supporting_insight_ids") or []
                if supporting_insight_ids:
                    nb.record_selection_insight_trial(
                        decision_id=decision_id,
                        context=decision_payload["context"],
                        insight_ids=supporting_insight_ids,
                        chosen_result_ids=candidate_ids,
                        source_experiment_id=str(results.get("experiment_id") or ""),
                    )
            except Exception as sel_err:
                logger.debug("Auto-validate selection logging failed: %s", sel_err)

            for rid in candidate_ids:
                score_row = scored_by_id.get(rid)
                if not score_row:
                    continue
                nb.update_selection_family_stats(
                    score_row.get("family", "Unknown"),
                    reward=float(score_row.get("base_score", 0.0)),
                )

            self._pending_validation = {
                "result_ids": candidate_ids,
                "config": config,
                "hypothesis": (
                    f"Auto-validation: publication-grade testing of "
                    f"{len(candidate_ids)} robust investigation survivors."
                ),
            }
            evidence_pack = self._safe_build_evidence_pack(
                nb,
                recommendation={"mode": "validation"},
                decision_type="auto_validate",
            )
            self._pending_validation["evidence_pack"] = evidence_pack

            self._emit_event("auto_validate_queued", {
                "result_ids": candidate_ids,
                "n_candidates": len(candidate_ids),
                "reason": f"{len(strong)} candidates with robustness >= "
                          f"{config.auto_validate_min_robustness}",
                "evidence_pack": evidence_pack,
            })

            nb.add_entry(ExperimentEntry(
                entry_type="decision",
                title="Auto-Validation Triggered",
                content=(
                    f"Automatically queuing validation for {len(candidate_ids)} "
                    f"robust investigation survivors."
                ),
                metadata={"result_ids": candidate_ids, "evidence_pack": evidence_pack},
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
                        hypothesis: Optional[str] = None,
                        preregistration: Optional[Dict[str, Any]] = None,
                        exploratory: bool = False) -> str:
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

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="evolution",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=hypothesis_metadata,
            preregistration=preregistration,
            exploratory=exploratory,
            created_by="start_evolution",
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
                             hypothesis: Optional[str] = None,
                             preregistration: Optional[Dict[str, Any]] = None,
                             exploratory: bool = False) -> str:
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

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="novelty",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=hypothesis_metadata,
            preregistration=preregistration,
            exploratory=exploratory,
            created_by="start_novelty_search",
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

    def _make_fitness_fn(self, config: RunConfig, *,
                         on_evaluate=None,
                         fitness_cache=None):
        """Create fitness function for evolution/novelty search.

        Args:
            config: Run configuration.
            on_evaluate: Optional callback ``(graph, fitness, sandbox_result, s1_result)``
                fired after every real evaluation (not cache hits).
            fitness_cache: Optional ``Dict[str, float]`` mapping graph fingerprint
                to fitness.  Cache hits skip compilation entirely.
        """
        dev_str = config.device if torch.cuda.is_available() else "cpu"
        dev = torch.device(dev_str)

        def fitness_fn(graph):
            fp = graph.fingerprint()

            # Fast path: return cached fitness without compilation.
            if fitness_cache is not None and fp in fitness_cache:
                return fitness_cache[fp]

            sandbox_result = None
            s1_result = None
            try:
                layer_graphs = [graph] * config.n_layers
                model = compile_model(
                    layer_graphs,
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                )
                sandbox_result = self._safe_eval_for_stage(
                    model,
                    stage_tag="evolution_fitness",
                    batch_size=2,
                    seq_len=min(128, config.max_seq_len),
                    vocab_size=config.vocab_size,
                    device=dev_str,
                )
                if not sandbox_result.passed:
                    del model
                    fitness = 0.0
                    if fitness_cache is not None:
                        fitness_cache[fp] = fitness
                    if on_evaluate:
                        on_evaluate(graph, fitness, sandbox_result, s1_result)
                    return fitness

                # Micro-train for fitness
                s1_result = self._micro_train(
                    model,
                    config,
                    dev,
                    seed=self._stable_seed("fitness", fp),
                )
                del model
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

                if s1_result.get("passed"):
                    fitness, _components = self._compute_multi_objective_fitness(
                        s1_result, sandbox_result, graph, config)
                else:
                    fitness = 0.1  # compiled and stable but didn't learn
            except Exception:
                fitness = 0.0

            if fitness_cache is not None:
                fitness_cache[fp] = fitness
            if on_evaluate:
                on_evaluate(graph, fitness, sandbox_result, s1_result)
            return fitness

        return fitness_fn

    def _run_evolution_thread(self, exp_id: str, config: RunConfig,
                               hypothesis: str):
        """Execute evolutionary search in background."""
        nb = self._make_notebook()
        t_start = time.time()
        try:
            from ..search.evolution import EvolutionConfig, evolutionary_search

            grammar = self._build_grammar_config(config)

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

            fitness_cache: dict = {}
            eval_counters = {"total": 0, "s0": 0, "s1": 0}

            def on_evaluate(graph, fitness, sandbox_result, s1_result):
                self._on_program_evaluated(graph, fitness, sandbox_result, s1_result, 
                                           eval_counters, nb, exp_id, model_source="evolution")

            fitness_fn = self._make_fitness_fn(
                config, on_evaluate=on_evaluate, fitness_cache=fitness_cache)

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

            results = {
                "total": eval_counters["total"],
                "stage0_passed": eval_counters["s0"],
                "stage05_passed": eval_counters["s0"],
                "stage1_passed": eval_counters["s1"],
                "novel_count": sum(1 for ind in population if ind.novelty > 0.5),
                "best_loss_ratio": 1.0 - max((ind.fitness for ind in population), default=0),
                "best_novelty_score": max((ind.novelty for ind in population), default=0),
                "survivors": [],
            }

            for ind in population[:20]:
                if ind.fitness > 0.2:
                    results["survivors"].append({
                        "fingerprint": ind.fingerprint,
                        "novelty": ind.novelty,
                        "loss_ratio": 1.0 - ind.fitness,
                    })

            nb.update_op_success_rates(exp_id)
            nb.update_failure_signatures(exp_id)

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
            logger.error("Evolution failed (%s): %s\n%s", exp_id, e, error)
            self._invoke_code_healer(
                nb=nb,
                trigger_type="repeated_exception",
                experiment_id=exp_id,
                scope=f"Evolution failure: {str(e)[:240]}",
                reproduction_steps=["python -m pytest tests/test_integration.py -k \"evolution\" -x --tb=short"],
                acceptance_tests=["python -m pytest tests/test_integration.py -k \"evolution\" -x --tb=short"],
                trigger_payload={"mode": "evolution", "error": str(e)},
            )
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

            grammar = self._build_grammar_config(config)

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

            dev_str = config.device if torch.cuda.is_available() else "cpu"
            dev = torch.device(dev_str)

            fitness_cache: dict = {}
            fingerprint_cache: dict = {}
            eval_counters = {"total": 0, "s0": 0, "s1": 0}

            def on_evaluate(graph, fitness, sandbox_result, s1_result):
                self._on_program_evaluated(graph, fitness, sandbox_result, s1_result, 
                                           eval_counters, nb, exp_id, model_source="novelty")

            def combined_fitness_fn(graph):
                """Compile once, run sandbox + micro-train + fingerprint in one pass."""
                gfp = graph.fingerprint()
                if gfp in fitness_cache:
                    return fitness_cache[gfp]

                sandbox_result = None
                s1_result = None
                try:
                    layer_graphs = [graph] * config.n_layers
                    model = compile_model(
                        layer_graphs,
                        vocab_size=config.vocab_size,
                        max_seq_len=config.max_seq_len,
                    )
                    sandbox_result = self._safe_eval_for_stage(
                        model,
                        stage_tag="evolution_combined_fitness",
                        batch_size=2,
                        seq_len=min(128, config.max_seq_len),
                        vocab_size=config.vocab_size,
                        device=dev_str,
                    )
                    if not sandbox_result.passed:
                        del model
                        fitness = 0.0
                        fitness_cache[gfp] = fitness
                        on_evaluate(graph, fitness, sandbox_result, s1_result)
                        return fitness

                    # Compute behavioral fingerprint while model is in memory
                    try:
                        bfp = compute_fingerprint(
                            model,
                            seq_len=min(64, config.max_seq_len),
                            model_dim=config.model_dim,
                            vocab_size=config.vocab_size,
                            device=dev_str,
                        )
                        fingerprint_cache[gfp] = bfp
                    except Exception as e:
                        logger.debug("Fingerprint computation failed: %s", e)

                    s1_result = self._micro_train(
                        model,
                        config,
                        dev,
                        seed=self._stable_seed("fitness", gfp),
                    )
                    del model
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

                    if s1_result.get("passed"):
                        fitness, _components = self._compute_multi_objective_fitness(
                            s1_result, sandbox_result, graph, config)
                    else:
                        fitness = 0.1
                except Exception:
                    fitness = 0.0

                fitness_cache[gfp] = fitness
                on_evaluate(graph, fitness, sandbox_result, s1_result)
                return fitness

            def fingerprint_fn(graph):
                return fingerprint_cache.get(graph.fingerprint())

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
                fitness_fn=combined_fitness_fn,
                fingerprint_fn=fingerprint_fn,
                config=ns_config,
                callback=gen_callback,
                stop_check=self._stop_event.is_set,
            )

            results = {
                "total": eval_counters["total"],
                "stage0_passed": eval_counters["s0"],
                "stage05_passed": eval_counters["s0"],
                "stage1_passed": eval_counters["s1"],
                "novel_count": sum(1 for ind in ns_result.best_individuals if ind.novelty > 0.5),
                "best_loss_ratio": None,
                "best_novelty_score": None,
                "survivors": [],
                "archive_size": ns_result.archive_size,
            }

            for ind in ns_result.best_individuals[:20]:
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
            nb.update_failure_signatures(exp_id)

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
            logger.error("Novelty search failed (%s): %s\n%s", exp_id, e, error)
            self._invoke_code_healer(
                nb=nb,
                trigger_type="repeated_exception",
                experiment_id=exp_id,
                scope=f"Novelty search failure: {str(e)[:240]}",
                reproduction_steps=["python -m pytest tests/test_integration.py -k \"novelty\" -x --tb=short"],
                acceptance_tests=["python -m pytest tests/test_integration.py -k \"novelty\" -x --tb=short"],
                trigger_payload={"mode": "novelty", "error": str(e)},
            )
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
                       hypothesis: Optional[str] = None,
                       preregistration: Optional[Dict[str, Any]] = None,
                       exploratory: bool = False) -> str:
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

        exp_id = self._start_preregistered_experiment(
            nb=nb,
            experiment_type="scale_up",
            config=config.to_dict(),
            hypothesis=hypothesis,
            hypothesis_metadata=self._build_hypothesis_metadata(
                source=source,
                llm_used=False,
                fallback_used=False,
                used_context=False,
            ),
            preregistration=preregistration,
            exploratory=exploratory,
            created_by="start_scale_up",
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
                            "n_train_steps", "final_lr",
                            "validation_loss", "validation_loss_ratio", "generalization_gap",
                            "discovery_loss", "discovery_loss_ratio"]:
                    program_metrics[key] = s1_result.get(key)
                self._merge_s1_telemetry(program_metrics, s1_result)

                if s1_passed:
                    results["stage1_passed"] += 1
                    # Baseline comparison at scale
                    if final_loss is not None:
                        try:
                            baseline = self._get_baseline()
                            baseline_steps = int(s1_result.get("n_train_steps") or config.scale_up_steps)
                            baseline_recipe = self._resolve_baseline_recipe(
                                s1_result, default_lr=config.stage1_lr)
                            bl_data_fn, bl_data_tag, bl_cache = self._make_baseline_data_fn(config)
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
                                cache_data_fn=bl_cache,
                            )
                            program_metrics["baseline_loss_ratio"] = baseline_ratio
                            
                            # Optional: Validation baseline comparison (using val split)
                            val_loss = s1_result.get("validation_loss")
                            if val_loss is not None:
                                v_data_fn, v_data_tag, v_cache = self._make_baseline_data_fn(config, split="val")
                                v_baseline_ratio = baseline.compare(
                                    val_loss,
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
                                    data_fn=v_data_fn,
                                    data_tag=v_data_tag,
                                    cache_data_fn=v_cache,
                                )
                                program_metrics["validation_baseline_loss_ratio"] = v_baseline_ratio
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
                calibration_row = None
                if s1_passed and model is not None:
                    try:
                        fp = compute_fingerprint(
                            model,
                            seq_len=min(64, config.scale_up_seq_len),
                            model_dim=config.model_dim,
                            vocab_size=config.vocab_size,
                            device=dev_str,
                        )
                        program_metrics["cka_source"] = fp.cka_source
                        program_metrics["cka_artifact_version"] = fp.cka_artifact_version
                        program_metrics["cka_probe_protocol_hash"] = fp.cka_probe_protocol_hash
                        program_metrics["cka_reference_quality"] = fp.cka_reference_quality
                        calibration_row = self._ensure_novelty_calibration(nb, config, fp)
                    except Exception:
                        pass

                calibration = None
                if calibration_row:
                    calibration = {
                        "noise_floor_mean": calibration_row.get("noise_floor_mean"),
                        "noise_floor_std": calibration_row.get("noise_floor_std"),
                    }
                nov = novelty_score(graph, fingerprint=fp, calibration=calibration)
                n_score = nov.overall_novelty
                novelty_valid, novelty_valid_reason, novelty_requires_justification = (
                    self._resolve_novelty_promotion_validity(
                        config,
                        nov.novelty_valid_for_promotion,
                        nov.novelty_validity_reason,
                    )
                )
                program_metrics["novelty_raw_score"] = nov.raw_novelty
                program_metrics["novelty_z_score"] = nov.novelty_z_score
                program_metrics["novelty_reference_version"] = (
                    nov.novelty_reference_version
                    or (fp.novelty_reference_version if fp is not None else None)
                )
                program_metrics["novelty_valid_for_promotion"] = int(novelty_valid)
                program_metrics["novelty_validity_reason"] = novelty_valid_reason
                program_metrics["novelty_requires_justification"] = int(
                    novelty_requires_justification
                )
                if s1_passed and n_score > 0.5:
                    results["novel_count"] += 1
                    results["survivors"].append({
                        "fingerprint": graph.fingerprint(),
                        "novelty": n_score,
                        "loss_ratio": loss_ratio,
                        "novelty_valid_for_promotion": novelty_valid,
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
            error = traceback.format_exc()
            logger.error("Scale-up failed (%s): %s\n%s", exp_id, e, error)
            self._invoke_code_healer(
                nb=nb,
                trigger_type="repeated_exception",
                experiment_id=exp_id,
                scope=f"Scale-up failure: {str(e)[:240]}",
                reproduction_steps=["python -m pytest tests/test_integration.py -k \"scale_up\" -x --tb=short"],
                acceptance_tests=["python -m pytest tests/test_integration.py -k \"scale_up\" -x --tb=short"],
                trigger_payload={"mode": "scale_up", "error": str(e)},
            )
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
