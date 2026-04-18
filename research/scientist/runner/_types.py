"""Shared dataclass definitions for the runner package.

Contains RunConfig, LiveProgress, and ModelCandidate to avoid circular
imports between __init__.py and submodules.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from research.defaults import (
    MODEL_DIM,
    VOCAB_SIZE,
    MAX_SEQ_LEN,
    VALIDATION_SEQ_LEN,
    STAGE1_STEPS,
    STAGE1_LR,
    STAGE1_BATCH_SIZE,
    INVESTIGATION_STEPS,
    INVESTIGATION_BATCH_SIZE,
    VALIDATION_STEPS,
    VALIDATION_BATCH_SIZE,
    SCALE_UP_STEPS,
    SCALE_UP_BATCH_SIZE,
    SCALE_UP_SEQ_LEN,
)


class ModelCandidate:
    """Unified representation of a candidate model from any source."""

    __slots__ = (
        "source",
        "model",
        "description",
        "graph",
        "graph_json",
        "arch_spec",
        "arch_spec_json",
        "fingerprint",
    )

    def __init__(
        self,
        source: str = "graph_synthesis",
        model: Any = None,
        description: str = "",
        graph: Any = None,
        graph_json: Optional[str] = None,
        arch_spec: Any = None,
        arch_spec_json: Optional[str] = None,
        fingerprint: str = "",
    ):
        self.source = source
        self.model = model
        self.description = description
        self.graph = graph
        self.graph_json = graph_json
        self.arch_spec = arch_spec
        self.arch_spec_json = arch_spec_json
        self.fingerprint = fingerprint


from ._helpers import _native_runner_progress_report


@dataclass(slots=True)
class RunConfig:
    """Configuration for an experiment run."""

    mode: str = "single"
    n_programs: int = 100
    model_dim: int = MODEL_DIM
    n_layers: int = 6
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = MAX_SEQ_LEN
    device: str = "cuda"
    # Stage 1 training
    stage1_steps: int = STAGE1_STEPS
    stage1_lr: float = STAGE1_LR
    stage1_batch_size: int = STAGE1_BATCH_SIZE
    enable_perf_tracing: bool = False
    debug: bool = False  # verbose logging + bypass quality gate + persist all results
    collect_training_curve: bool = True
    gradient_clip_norm: float = 1.0
    optimizer_fused: bool = True
    optimizer_foreach: bool = True
    starvation_check_interval: int = 8
    enable_starvation_monitoring: bool = False
    enable_cuda_graphs: bool = False
    # Early stopping: halt training when loss plateaus
    early_stop_patience: int = 300  # steps without improvement before stopping
    early_stop_min_delta: float = 1e-3  # minimum loss improvement to reset patience
    early_stop_min_steps: int = 100  # don't early-stop before this many steps
    # Inflight training checks: abort hopeless runs early
    inflight_spike_ratio: float = 2.0  # kill if loss > 2x running minimum
    inflight_spike_window: int = 10  # check spike over this many steps
    inflight_grad_norm_limit: float = 100.0  # kill if grad_norm exceeds this
    inflight_grad_norm_strikes: int = 3  # consecutive violations before kill
    cuda_graph_warmup_steps: int = 3
    loss_check_interval: int = 8
    enable_kernel_profiling: bool = False
    kernel_profile_top_k: int = 20
    profile_enabled: bool = False
    profile_dir: str = "profiles"
    profile_wait_steps: int = 0
    profile_warmup_steps: int = 2
    profile_active_steps: int = 4
    profile_record_shapes: bool = True
    profile_memory: bool = True
    profile_with_stack: bool = False
    profile_disable_inflight_checks: bool = False
    profile_disable_post_eval: bool = False
    # Training data source
    data_mode: str = "corpus"  # "random" | "corpus" | "hydra"
    corpus_path: str = "/home/tim/Projects/LLM/research/corpus/wikitext103_train.npy"
    corpus_format: str = "auto"  # "auto" | "txt" | "jsonl"
    corpus_text_key: str = "text"  # JSONL key when format is jsonl
    tokenizer_mode: str = "tiktoken"  # "byte" | "whitespace" | "tiktoken"
    tiktoken_encoding: str = "cl100k_base"  # "gpt2" | "cl100k_base"
    # Progressive screening: cheap qualifying pass at small vocab before
    # expensive full eval.  Filters ~93% of candidates at ~10% of the cost.
    progressive_screening: bool = True
    qualifying_vocab_size: int = 32000
    corpus_max_chars: int = 200000
    corpus_train_fraction: float = 0.9
    corpus_val_fraction: float = 0.1
    stage1_compute_val_loss: bool = True
    stage1_val_batches: int = 2
    stage1_val_batch_size: int = 4
    stage1_compute_discovery_loss: bool = True
    stage1_discovery_batches: int = 2
    stage1_discovery_batch_size: int = 4
    skip_screening_hellaswag: bool = False
    skip_screening_blimp: bool = False
    skip_ar_probe: bool = False
    skip_binding_probes: bool = False
    skip_induction_probe: bool = False
    skip_binding_probe: bool = False
    binding_probe_offload_source_model: bool = False
    binding_probe_train_batch_size: int = 0
    binding_probe_eval_batch_size: int = 0
    skip_post_s1_fingerprint: bool = False
    skip_post_s1_triage: bool = False
    screening_probe_seed: int | None = None
    # HYDRA data loader settings (data_mode="hydra")
    hydra_data_dir: str = "../HYDRA/data"
    hydra_dataset: str = "local_jsonl"  # any HYDRA dataset name
    hydra_project_root: str = "../HYDRA"
    # HuggingFace dataset (data_mode="huggingface")
    hf_dataset: str = ""  # e.g. "roneneldan/TinyStories", "wikitext"
    hf_subset: str = ""  # e.g. "wikitext-2-raw-v1"
    hf_split: str = "train"  # train | validation | test
    hf_text_key: str = "text"  # column name containing text
    # Screening WikiText eval (fast real-token perplexity at screening time)
    skip_screening_wikitext: bool = False  # set True to disable screening WikiText eval
    # Escalation threshold: auto-escalate if ppl_200/ppl_500 exceeds this ratio
    improvement_ratio_escalation_threshold: float = 2.0
    # Optional cheap micro-train gate inserted before full rich S1 screening.
    # Disabled by default so backfills and full-data collection paths keep the
    # original S1 semantics unless they opt in explicitly.
    enable_stage09_cheap_train_gate: bool = False
    # Synthesis grammar
    max_depth: int = 18
    max_ops: int = 24
    math_space_weight: float = 2.0
    residual_prob: float = 0.7
    composition_depth: int = 3  # Minimum template blocks per graph
    _efficiency_mode: bool = False
    _exotic_mode: bool = False
    _routing_first_mode: bool = False
    # Capability-first: trunk+sidecar graphs with explicit retrieval path.
    # Enables ``GrammarConfig.capability_first()`` preset AND flips
    # ``binding_capable_required`` so screening rejects retrieval-dead
    # graphs via gate8. Pairs well with ``ARIA_SCORING_VERSION=v8.1``.
    _capability_first_mode: bool = False
    # Continuous mode
    continuous: bool = False
    max_experiments: int = 100
    rest_between_experiments: int = 5  # seconds
    control_experiment_interval: int = (
        5  # run every Nth synthesis as control (0 disables)
    )
    max_time_minutes: int = 0  # 0 = no limit
    max_cost_dollars: float = 0.0  # 0 = no limit (estimated LLM API cost)
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
    # Exploitation: bias mutation toward winning fingerprint neighborhoods
    exploit_mode: bool = False  # master flag: enables routing + splits + exploitation
    exploit_prob: float = (
        0.2  # probability of archive-guided exploitation per offspring
    )
    local_mutation_prob: float = 0.3  # probability of single-op swap for top-K parents
    exploit_top_k: int = 5  # number of top individuals considered for exploitation
    # Scale-up mode
    scale_up: bool = False
    scale_up_result_ids: str = ""  # comma-separated result IDs
    scale_up_steps: int = SCALE_UP_STEPS
    scale_up_batch_size: int = SCALE_UP_BATCH_SIZE
    scale_up_seq_len: int = SCALE_UP_SEQ_LEN
    # One-shot pruning baseline
    one_shot_pruning_baseline: bool = False
    one_shot_pruning_method: str = "wanda"  # wanda | sparsegpt
    one_shot_pruning_sparsity: float = 0.5
    one_shot_pruning_eval_batches: int = 4
    one_shot_pruning_batch_size: int = 2
    # Automation
    auto_scale_up: bool = True  # auto-trigger scale-up when criteria met
    auto_scale_up_min_survivors: int = 3  # min S1 survivors to trigger
    auto_scale_up_min_novelty: float = 0.5  # min avg novelty of survivors
    auto_scale_up_top_n: int = 5  # how many to scale up
    auto_report: bool = True  # auto-generate report at session end
    auto_report_every_n: int = (
        5  # also generate report every N experiments (continuous)
    )
    # Model source
    model_source: str = (
        "graph_synthesis"  # "graph_synthesis", "morphological_box", "mixed"
    )
    morph_ratio: float = 0.5  # fraction of morphological candidates in mixed mode
    morph_focus_sparse: bool = (
        False  # force sparse weight-storage options in morphological mode
    )
    morph_sparse_weight_storage: str = ""  # optional explicit sparse storage choice
    morph_compute_routing: str = (
        ""  # optional fixed compute_routing choice for morphology
    )
    morph_channel_mixing: str = (
        ""  # optional fixed channel_mixing choice for morphology
    )
    refine_source_result_ids: str = (
        ""  # comma-separated source result IDs for local fingerprint refinement
    )
    refine_mutations_per_source: int = 4
    refine_intent: str = "balanced"  # balanced|quality|compression|sparsity|novelty
    refine_pool_multiplier: int = 3
    refine_analysis_json: str = (
        ""  # serialized RefinementAnalyzer output for data-driven refinement
    )
    # Training program variation
    use_synthesized_training: bool = False  # use random training programs
    n_training_programs: int = 3  # how many to try per candidate (investigation)
    loss_type: str = "cross_entropy"  # "cross_entropy" | "synthesized"
    optimizer_type: str = "adamw"  # "adamw" | "muon" | "synthesized"
    optimizer_betas: tuple = (0.9, 0.95)  # Adam betas (only used for adamw)
    optimizer_weight_decay: float = 0.01  # decoupled weight decay
    # Phase-specific optimizer overrides (empty/"" = inherit from optimizer_type)
    screening_optimizer: str = ""  # optimizer for screening phase
    screening_lr: float = 0.0  # 0 = use stage1_lr
    investigation_optimizer: str = ""  # optimizer for investigation phase
    investigation_lr: float = 0.0  # 0 = use stage1_lr
    # Investigation phase
    investigation_steps: int = INVESTIGATION_STEPS
    investigation_batch_size: int = INVESTIGATION_BATCH_SIZE
    investigation_max_loss_ratio_multiplier: float = 8.0
    # Validation phase
    validation_steps: int = VALIDATION_STEPS
    validation_batch_size: int = VALIDATION_BATCH_SIZE
    validation_seq_len: int = VALIDATION_SEQ_LEN
    validation_n_seeds: int = 5
    # Auto-escalation pipeline
    auto_investigate: bool = (
        True  # Re-enabled: routing models now produce viable candidates
    )
    auto_investigate_min_survivors: int = 5  # Routing models pass S1 reliably
    auto_investigate_top_n: int = 3  # Investigate top 3 per experiment
    auto_validate: bool = True
    auto_validate_min_robustness: float = 0.5
    auto_validate_max_baseline_ratio: float = 0.60
    auto_validate_min_composite_score: float = (
        0.0  # 0 = use best reference as dynamic floor
    )
    # [CALIBRATION] source: judgment — also used with getattr fallback in continuous_validation.py
    #   last reviewed: unknown — flag for calibration sweep
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
    selection_quality_weight: float = 0.35
    selection_novelty_weight: float = 0.25
    selection_efficiency_weight: float = 0.25
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
    scaling_reference_families: str = "gpt2"  # comma-separated: "gpt2,mamba"
    scaling_d512_enabled: bool = True  # retrain breakthrough candidates at d=512
    # [CALIBRATION] source: judgment — also hardcoded in continuous_validation.py scaling gate
    #   last reviewed: unknown — flag for calibration sweep
    scaling_param_efficiency_target: float = (
        3.0  # min param efficiency for breakthrough
    )
    # [CALIBRATION] source: judgment — also hardcoded in continuous_validation.py scaling gate
    #   last reviewed: unknown — flag for calibration sweep
    scaling_flop_ceiling: float = 2.0  # max FLOP ratio vs reference
    # Checkpoint/resume
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1  # save continuous checkpoint every N experiments
    phase_checkpoint_step_interval: int = (
        100  # save validation/investigation train state every N steps
    )
    resume_experiment_id: str = ""  # experiment ID to resume (empty = fresh start)
    keep_checkpoints: bool = False  # keep checkpoints after successful completion
    # Campaign system
    enable_campaigns: bool = True
    knowledge_extraction_interval: int = 3  # every N experiments
    auto_go_no_go: bool = True  # auto-record go/no-go decisions at escalation
    # Stage pass thresholds (overridable by LLM per-cycle)
    stage1_loss_ratio_threshold: float = 0.4
    stage05_stability_threshold: float = 0.5
    investigation_loss_ratio_threshold: float = 0.15
    investigation_robustness_threshold: float = 0.5
    # Adaptive percentile thresholds: opt-in mode where promotion thresholds
    # are computed from recent population distribution instead of fixed values.
    adaptive_thresholds_enabled: bool = False
    screening_promotion_percentile: float = 90.0  # promote top 10% of screening
    investigation_promotion_percentile: float = 90.0  # promote top 10% of investigation
    # Lightning structural floor: minimum structural novelty to proceed past
    # the lightning gate. Below this the graph is too similar to existing
    # population to justify investigation cost.
    lightning_structural_floor: float = 0.10
    # Max composite points from structural-only novelty (no post-investigation
    # CKA completion). Full CKA-backed novelty can reach 40 points.
    # Calibrated against c9c7075e741a8790: structural_novelty=0.381 → ~5.7 pts
    novelty_structural_only_cap: float = 15.0
    # Pre-investigation gate
    pre_inv_gate_enabled: bool = True
    pre_inv_max_lr: float = (
        0.50  # Raised from 0.40: routing models have higher screening lr
    )
    pre_inv_min_stability: float = 0.5
    pre_inv_max_spectral_norm: float = 50.0
    pre_inv_min_spectral_norm: float = 0.01
    pre_inv_min_improvement_rate: float = 0.05  # must show 5% improvement
    pre_inv_top_n: int = 15
    pre_inv_reference_margin: float = 1.5
    pre_inv_probe_enabled: bool = False
    pre_inv_probe_steps_fraction: float = 0.25
    pre_inv_probe_max_lr: float = 0.85
    # Slope reprieve: allows slow-start candidates a 150-step second chance
    # Default disabled — enable only after observing slope distribution
    slope_reprieve_enabled: bool = False
    slope_reprieve_threshold: float = 0.015
    slope_reprieve_consistent_required: bool = True
    slope_reprieve_loss_floor: float = 0.85
    slope_reprieve_max_per_cycle: int = 3
    slope_reprieve_eval_steps: int = 150
    slope_reprieve_score_multiplier: float = 0.75
    # Grammar structure probabilities (forwarded to GrammarConfig)
    grammar_split_prob: float = 0.3
    grammar_merge_prob: float = 0.2
    grammar_risky_op_prob: float = 0.1
    grammar_freq_domain_prob: float = 0.0
    # Custom grammar weights (passed through to GrammarConfig)
    category_weights: Optional[Dict[str, float]] = None
    op_weights: Optional[Dict[str, float]] = None
    template_weights: Optional[Dict[str, float]] = None
    use_learned_grammar_weights: bool = False
    use_learned_candidate_weights: bool = False
    use_screening_signal_weights: bool = False
    allow_unproven_ml_influence: bool = False
    routing_mandatory: bool = True  # require routing/MoE ops in every graph
    persist_screening_failures: bool = (
        True  # keep early failed graphs for data collection — produces
        # hard negatives (good ops, bad structure) that teach the gate
        # model to look beyond op identity
    )
    disable_runtime_dedup: bool = False  # allow repeated fingerprints through screening when collecting template backfill evidence
    # Branching / width control (passed to GrammarConfig)
    min_splits: int = 0  # minimum forced split-merge blocks per graph
    three_way_split_prob: float = 0.0  # probability of 3-way split (vs 2-way)
    branch_depth: int = 1  # depth of processing on each branch (1=shallow, 2+=deep)
    max_recursion_depth: int = 4  # iteration cap for recursive ops
    # Healer/agent settings
    max_agent_seconds: int = 300
    # LLM consultation in continuous mode
    llm_decision_interval: int = (
        5  # call Sonnet every N cycles (0 = never in continuous)
    )
    # Investigation predictor: skip candidates with predicted loss_ratio > max_lr.
    # Ridge regression on 18D fingerprint features, trained in-memory from notebook history.
    investigation_predictor_enabled: bool = True
    investigation_predictor_max_lr: float = 0.7

    # GBM pre-screener: LightGBM on graph-structure features, skips hopeless graphs
    # before expensive eval. Gate threshold: skip if P(pass_s1) < gbm_gate_threshold.
    # F1-optimal operating point is 0.33 (PPV=0.54, recall=0.76, AUC=0.89).
    # Previous 0.1 was the high-recall point (PPV=0.38) — barely filtered anything.
    gbm_prescreener_enabled: bool = True
    gbm_gate_threshold: float = 0.33

    # Thompson sampling for op/template/motif selection via Bayesian posteriors.
    # When True, draws from Beta posterior (explore/exploit). When False, uses
    # posterior mean (exploit-only). Thompson should be the default — it's the
    # primary reason the Bayesian tracker exists.
    use_thompson_sampling: bool = True

    # Force all generated graphs to use this template (bypass pick_template).
    # Set to a template name from TEMPLATES dict, e.g. "transformer_block".
    forced_template: Optional[str] = None

    def copy(self) -> RunConfig:
        """Shallow copy via dataclasses.replace (no dict round-trip)."""
        return dataclasses.replace(self)

    def to_dict(self) -> Dict:
        out = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            # Guard against member_descriptor leaking from slots-based
            # nested dataclasses or class-level attribute access.
            if type(v).__name__ == "member_descriptor":
                v = str(v)
            out[k] = v
        return out

    @classmethod
    def from_dict(cls, d: Dict) -> RunConfig:
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass(slots=True)
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
    native_runner: Dict[str, Any] = field(
        default_factory=_native_runner_progress_report
    )

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ── Validation result types ──


@dataclass(slots=True)
class ExternalEvalResult:
    """Typed output from _run_external_evals.

    Replaces the 28-key untyped dict that was manually unpacked at 3 call sites.
    """

    is_breakthrough: bool = False
    flop_gated: bool = False
    quant_int8_retention: Optional[float] = None
    quant_quality_per_byte: Optional[float] = None
    long_context_score: Optional[float] = None
    long_context_details: Optional[Dict] = None
    noise_score: Optional[float] = None
    ood_result: Optional[Dict] = None
    sensitivity_result: Optional[Dict] = None
    activation_sparsity_score: Optional[float] = None
    dead_neuron_ratio: Optional[float] = None
    routing_collapse_score: Optional[float] = None
    wikitext_perplexity: Optional[float] = None
    wikitext_score: Optional[float] = None
    tinystories_perplexity: Optional[float] = None
    tinystories_score: Optional[float] = None
    cross_task_score: Optional[float] = None
    efficiency_wall_score: Optional[float] = None
    max_viable_seq_len: Optional[int] = None
    scaling_regime: Optional[str] = None
    scaling_param_efficiency: Optional[float] = None
    scaling_flop_efficiency: Optional[float] = None
    scaling_gate_passed_val: Optional[bool] = None
    scaling_best_family: Optional[str] = None
    scaling_confidence: Optional[str] = None
    scaling_result: Optional[Dict] = None
    scaling_d512_param_efficiency: Optional[float] = None
    fp_gromov_delta: Optional[float] = None
    fp_hierarchy_fitness: Optional[float] = None
    # Long-context retrieval sub-scores
    long_ctx_assoc_score: Optional[float] = None
    long_ctx_passkey_score: Optional[float] = None
    long_ctx_multi_hop_score: Optional[float] = None
    long_ctx_retrieval_aggregate: Optional[float] = None
    long_ctx_scaling_score: Optional[float] = None
    long_ctx_combined_score: Optional[float] = None
    # v2 investigation-tier probes (2026-04-18)
    induction_v2_investigation_auc: Optional[float] = None
    induction_v2_investigation_max_gap_acc: Optional[float] = None
    induction_v2_investigation_protocol_version: Optional[str] = None
    binding_v2_investigation_auc: Optional[float] = None
    binding_v2_investigation_max_distance_acc: Optional[float] = None
    binding_v2_investigation_protocol_version: Optional[str] = None
    robustness_checks_attempted: int = 0
    robustness_checks_failed: int = 0


@dataclass(slots=True)
class ValidationEntry:
    """Structured validation result for one candidate.

    Replaces the 25-key dict literal that was duplicated at 2 call sites.
    """

    result_id: str = ""
    val_loss_ratio: Optional[float] = None
    val_baseline_ratio: Optional[float] = None
    val_normalized_ratio: Optional[float] = None
    param_efficiency: Optional[float] = None
    multi_seed_std: float = 0.0
    robustness_score: float = 1.0
    is_unstable: bool = False
    seeds_passed: int = 0
    total_seeds: int = 0
    is_breakthrough: bool = False
    flop_gated: bool = False
    quant_int8_retention: Optional[float] = None
    quant_quality_per_byte: Optional[float] = None
    long_context_score: Optional[float] = None
    noise_sensitivity_score: Optional[float] = None
    init_sensitivity_std: Optional[float] = None
    novelty_confidence: float = 0.0
    ood_robustness: Optional[Dict] = None
    sensitivity: Optional[Dict] = None
    activation_sparsity_score: Optional[float] = None
    dead_neuron_ratio: Optional[float] = None
    routing_collapse_score: Optional[float] = None
    wikitext_perplexity: Optional[float] = None
    wikitext_score: Optional[float] = None
    tinystories_perplexity: Optional[float] = None
    tinystories_score: Optional[float] = None
    cross_task_score: Optional[float] = None
    efficiency_wall_score: Optional[float] = None
    max_viable_seq_len: Optional[int] = None
    scaling_regime: Optional[str] = None

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)


@dataclass(slots=True)
class ValidationMetrics:
    """Output from seed metric computation + baseline comparisons.

    Replaces the untyped dict returned by _validation_compute_metrics.
    """

    val_loss_ratio: Optional[float] = None
    multi_seed_std: float = 0.0
    robustness_score: float = 1.0
    is_unstable: bool = False
    init_sensitivity_std: Optional[float] = None
    val_baseline_ratio: Optional[float] = None
    val_normalized_ratio: Optional[float] = None
    val_param_efficiency: Optional[float] = None
    passed_seeds: list = field(default_factory=list)
    loss_ratios: list = field(default_factory=list)
    best_seed: Optional[Dict] = None
    source_params: int = 0
