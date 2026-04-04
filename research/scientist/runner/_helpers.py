"""Shared helper functions for the runner package.

Centralised here to avoid duplication across submodules.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import time
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ..json_utils import json_safe
from ..thresholds import TIER_RANK

logger = logging.getLogger(__name__)
_REFERENCE_TRAJECTORY_PATH = Path("research/eval/reference_trajectories.json")
_ROUTING_FAST_LANE_OPS: frozenset[str] = frozenset(
    {
        "moe_topk",
        "hetero_moe",
        "arch_router",
        "compute_budget_router",
        "signal_conditioned_compression",
    }
)

# ── Normalized loss_ratio ──
# loss_ratio = final_loss / initial_loss is init-dependent: Kaiming init
# yields initial_loss ~250 while small/ortho init yields ~ln(V).  This makes
# screening ratios (0.008) and investigation ratios (0.24) incomparable for
# the SAME architecture.  Normalizing against ln(vocab_size) — the expected
# cross-entropy of a uniform distribution — gives a consistent, interpretable
# metric across all stages and init schemes.

_DEFAULT_VOCAB_SIZE: int = 32_000
_REFERENCE_INITIAL_LOSS: float = math.log(_DEFAULT_VOCAB_SIZE)  # ~10.37


def _build_source_map(nb: Any, result_ids: List[str]) -> Dict[str, Dict]:
    """Fetch program details for *result_ids* and return a {result_id: detail} map.

    Centralises the repeated ``[d or {} for d in (nb.get_program_details(ids) or [])]``
    pattern used in investigation/validation execution.

    Also reconstructs ``_behavioral_fingerprint`` from ``fingerprint_json``
    so that post-investigation fingerprint completion (CKA) can run.
    """
    import json

    details = [d or {} for d in (nb.get_program_details(result_ids) or [])]
    source_map = {}
    for d in details:
        rid = d.get("result_id")
        if not rid:
            continue
        # Reconstruct _behavioral_fingerprint from fingerprint_json if absent
        if "_behavioral_fingerprint" not in d and d.get("fingerprint_json"):
            try:
                fp_data = d["fingerprint_json"]
                if isinstance(fp_data, str):
                    fp_data = json.loads(fp_data)
                d["_behavioral_fingerprint"] = fp_data
            except (json.JSONDecodeError, TypeError):
                pass
        source_map[rid] = d
    return source_map


def _corpus_type_from_config(config: Any) -> str:
    """Derive corpus type tag from RunConfig for gate calibration."""
    path = str(getattr(config, "corpus_path", "") or "").lower()
    if "wikitext" in path:
        return "wikitext103"
    if "tinystories" in path:
        return "tinystories"
    if "micro_corpus" in path:
        return "micro"
    return "unknown"


_ref_losses_cache: Dict[str, float] = {}
_ref_losses_ts: float = 0.0
_REF_LOSSES_TTL: float = 300.0


def get_reference_losses(db_path: str) -> Dict[str, float]:
    """Pull latest reference losses for gate calibration (cached 300s)."""
    global _ref_losses_cache, _ref_losses_ts
    import sqlite3

    now = time.monotonic()
    if _ref_losses_cache and (now - _ref_losses_ts) < _REF_LOSSES_TTL:
        return _ref_losses_cache

    ref: Dict[str, float] = {}
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        row = conn.execute("""
            SELECT AVG(p.final_loss) as avg_loss
            FROM program_results p
            JOIN leaderboard l ON p.result_id = l.result_id
            WHERE l.reference_name = 'GPT-2'
            AND p.final_loss IS NOT NULL
            AND p.final_loss > 0
            AND p.n_train_steps >= 4000
        """).fetchone()
        if row and row["avg_loss"]:
            ref["gpt2_wikitext103_tiktoken"] = float(row["avg_loss"])
        conn.close()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.debug("get_reference_losses failed: %s", exc)
    _ref_losses_cache = ref
    _ref_losses_ts = now
    return ref


def stage1_learning_gate(
    final_loss: float,
    loss_ratio: float,
    initial_loss: Optional[float],
    n_steps: int,
    corpus_type: str,
    tokenizer: str,
    reference_losses: Dict[str, float],
) -> Tuple[bool, str]:
    """Corpus-aware Stage 1 pass/fail gate.

    Returns (passed, reason).

    Gate logic:
    1. Hard kill: model made things worse (loss_ratio > 1.0)
    2. Hard kill: model clearly didn't learn (ratio > 0.95 after 200+ steps)
    3. Corpus-relative: if WikiText-103 + tiktoken, compare to GPT-2 floor
    4. Absolute floor: final_loss must be below random baseline
    """
    # Random baseline (entropy floor) — used by gates 2 and 4
    # tiktoken cl100k_base: ln(100277) ~ 11.52
    # byte tokenizer 32K vocab: ln(32000) ~ 10.37
    tok_prefix = tokenizer.split("_")[0] if tokenizer else ""
    random_baseline = {"tiktoken": 11.52, "byte": 10.37}.get(tok_prefix, 11.52)

    # Gate 1: Made things worse
    if loss_ratio > 1.0:
        return False, f"loss_ratio={loss_ratio:.3f} > 1.0 — training diverged"

    # Gate 2: Clearly didn't learn — entropy-relative threshold.
    #
    # Problem: fixed 0.95 ratio penalizes complex architectures that start
    # close to the entropy floor. A 12-op graph at init=12 has only 1.6 nats
    # of headroom above ln(vocab), so 5% improvement (0.6 nats) consumes 37%
    # of headroom. A 4-op graph at init=190 has 180 nats of headroom — 5%
    # (9.5 nats) is trivial.
    #
    # Fix: scale threshold by headroom. Require using at least 10% of
    # improvable headroom (init_loss - entropy_floor).
    # At init=190 (headroom=180): need 18 nats drop → threshold=0.905
    # At init=12  (headroom=1.6): need 0.16 nats drop → threshold=0.987
    if n_steps >= 200:
        _init = initial_loss if initial_loss and initial_loss > 0 else 100.0
        _headroom = max(_init - random_baseline, 0.5)
        _min_improvement = _headroom * 0.10
        _raw_thr = 1.0 - _min_improvement / max(_init, 1.0)
        _ratio_threshold = max(0.90, min(0.99, _raw_thr))
        if loss_ratio > _ratio_threshold:
            return False, (
                f"loss_ratio={loss_ratio:.3f} after {n_steps} steps — "
                f"model reduced loss by only {(1 - loss_ratio) * 100:.1f}% "
                f"(threshold={_ratio_threshold:.3f}, headroom={_headroom:.1f})"
            )

    # Gate 3: Corpus-relative gate (WikiText-103 + tiktoken only)
    if corpus_type == "wikitext103" and "tiktoken" in tokenizer:
        gpt2_floor = reference_losses.get("gpt2_wikitext103_tiktoken")
        if gpt2_floor is not None:
            relative_threshold = gpt2_floor * 4.0
            if final_loss > relative_threshold:
                return False, (
                    f"final_loss={final_loss:.3f} > "
                    f"4x GPT-2 floor ({relative_threshold:.3f}) — "
                    f"not competitive on WikiText-103"
                )

    # Gate 4: Absolute random baseline — headroom-aware.
    #
    # Models that start far above the entropy floor (e.g., evolution candidates
    # at init=200+ nats) may plateau just above the baseline after only 750
    # training steps, despite demonstrating strong learning (loss_ratio < 0.2).
    # Relax the floor for proven learners; keep it strict for non-learners.
    if loss_ratio is not None and loss_ratio < 0.20:
        # Strong learner (≥80% loss reduction) — pass regardless of absolute
        # floor.  These need more training steps, not rejection.
        pass
    elif loss_ratio is not None and loss_ratio < 0.50:
        # Moderate learner — allow slight overshoot above baseline.
        if final_loss > random_baseline * 1.05:
            return False, (
                f"final_loss={final_loss:.3f} above relaxed baseline "
                f"({random_baseline * 1.05:.3f}) — moderate learner but "
                f"not close enough (ratio={loss_ratio:.3f})"
            )
    elif final_loss > random_baseline * 0.95:
        return False, (
            f"final_loss={final_loss:.3f} near random baseline "
            f"({random_baseline:.2f}) — model learned nothing meaningful"
        )

    return True, "passed"


def normalized_loss_ratio(
    final_loss: float,
    vocab_size: int = _DEFAULT_VOCAB_SIZE,
) -> float:
    """Compute init-independent loss ratio.

    Returns final_loss / ln(vocab_size), measuring what fraction of
    maximum-entropy loss the model achieves.  Lower is better.
    A value of 0.2 means the model achieved 80% of the possible
    entropy reduction from a uniform distribution over the vocabulary.

    This replaces the old final_loss/initial_loss which was wildly
    init-dependent (Kaiming gave 0.008, small-init gave 0.24 for
    the same architecture and final loss).
    """
    ref = math.log(vocab_size) if vocab_size > 0 else _REFERENCE_INITIAL_LOSS
    return final_loss / max(ref, 1e-6)


# ── Inflight training health checks ──


@dataclass
class InflightState:
    """Mutable state for inflight training checks."""

    __slots__ = ("recent_losses", "grad_strikes", "window")
    recent_losses: List[float]
    grad_strikes: int
    window: int

    def __init__(self, window: int = 20):
        self.recent_losses = []
        self.grad_strikes = 0
        self.window = window


def check_inflight_health(
    step: int,
    loss_val: float,
    grad_norm: float,
    min_loss: float,
    initial_loss: Optional[float],
    total_steps: int,
    state: InflightState,
    spike_ratio: float = 2.0,
    spike_window: int = 10,
    cv_threshold: float = 0.5,
    progress_threshold: float = 0.95,
    grad_norm_limit: float = 100.0,
    grad_norm_strikes: int = 3,
) -> Optional[Dict[str, Any]]:
    """Run all inflight training health checks.

    Returns None if healthy, or a dict with 'error' and 'error_type' if
    the run should be aborted.
    """
    # Track recent losses
    state.recent_losses.append(loss_val)
    if len(state.recent_losses) > state.window:
        state.recent_losses.pop(0)

    # Check 1: loss spike far above running minimum
    if step >= spike_window and min_loss > 0 and loss_val > spike_ratio * min_loss:
        return {
            "error": (
                f"inflight_loss_spike: step {step}, "
                f"loss={loss_val:.4f} > {spike_ratio}x min={min_loss:.4f}"
            ),
            "error_type": "inflight_loss_spike",
        }

    # Check 2: wild oscillation (high CV over recent window)
    w = state.window
    if step >= w and len(state.recent_losses) >= w:
        _mean = sum(state.recent_losses) / w
        if _mean > 0:
            _var = sum((x - _mean) ** 2 for x in state.recent_losses) / w
            _cv = (_var**0.5) / _mean
            if _cv > cv_threshold:
                return {
                    "error": (
                        f"inflight_oscillation: step {step}, "
                        f"CV={_cv:.3f} over last {w} steps "
                        f"(mean={_mean:.2f}, std={_var**0.5:.2f})"
                    ),
                    "error_type": "inflight_oscillation",
                }

    # Check 3a: loss diverging — if loss exceeds initial by 50%, abort
    if step >= spike_window and initial_loss and loss_val > initial_loss * 1.5:
        return {
            "error": (
                f"inflight_divergence: step {step}, "
                f"loss={loss_val:.4f} > 1.5x initial={initial_loss:.4f}"
            ),
            "error_type": "inflight_divergence",
        }

    # Check 3b: no progress at 25% mark
    quarter = total_steps // 4
    if (
        step == quarter
        and initial_loss
        and loss_val >= initial_loss * progress_threshold
    ):
        return {
            "error": (
                f"inflight_no_progress: at step {step}/{total_steps}, "
                f"loss={loss_val:.4f} vs initial={initial_loss:.4f} "
                f"(ratio={loss_val / initial_loss:.3f})"
            ),
            "error_type": "inflight_no_progress",
        }

    # Check 4: persistent gradient explosion
    if grad_norm > grad_norm_limit:
        state.grad_strikes += 1
        if state.grad_strikes >= grad_norm_strikes:
            return {
                "error": (
                    f"inflight_grad_explosion: {grad_norm_strikes} consecutive "
                    f"steps with grad_norm > {grad_norm_limit:.0f} "
                    f"(last={grad_norm:.1f})"
                ),
                "error_type": "inflight_grad_explosion",
            }
    else:
        state.grad_strikes = 0

    return None


def clear_gpu_memory() -> None:
    """Release GPU memory and run garbage collection.

    Centralised cleanup to avoid duplicating torch.cuda.empty_cache() +
    gc.collect() across 13+ call sites in runner submodules.
    Also unloads any Ollama model to free VRAM for training.
    """
    import gc

    try:
        import torch as _torch

        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except (ImportError, RuntimeError) as e:
        logger.debug("GPU memory cleanup skipped: %s", e)
    gc.collect()
    _unload_ollama_if_running()


_ollama_last_unload: float = 0.0


def _unload_ollama_if_running() -> None:
    """Unload Ollama model from GPU if the server is reachable.

    Debounced to at most once per 60 seconds — clear_gpu_memory() is called
    20+ times per cycle and we don't want to spam HTTP requests.
    No-ops silently if Ollama isn't running.
    """
    global _ollama_last_unload
    import time

    now = time.monotonic()
    if now - _ollama_last_unload < 60.0:
        return
    _ollama_last_unload = now

    try:
        import requests

        r = requests.get("http://localhost:11434/api/tags", timeout=1)
        if r.status_code != 200:
            return
        models = r.json().get("models", [])
        if not models:
            return
        for m in models:
            name = m.get("name", "")
            if name:
                requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": name, "prompt": "", "keep_alive": 0},
                    timeout=5,
                )
    except (ImportError, OSError, ValueError) as e:
        logger.debug("Ollama unload skipped: %s", e)


def compute_seed_metrics(
    seed_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate metrics from multi-seed training results.

    Returns dict with: passed_seeds, loss_ratios, val_loss_ratio,
    multi_seed_std, robustness_score, is_unstable, init_sensitivity_std,
    best_seed.
    """
    passed_seeds = [r for r in seed_results if r.get("passed")]
    loss_ratios = [
        r["loss_ratio"] for r in seed_results if r.get("loss_ratio") is not None
    ]

    val_loss_ratio = sum(loss_ratios) / len(loss_ratios) if loss_ratios else None
    multi_seed_std = 0.0
    robustness_score = 1.0
    is_unstable = False

    if len(loss_ratios) > 1:
        mean_lr = val_loss_ratio
        variance = sum((lr - mean_lr) ** 2 for lr in loss_ratios) / len(loss_ratios)
        multi_seed_std = variance**0.5
        if variance > 0.15:
            is_unstable = True
        if mean_lr > 1e-6:
            robustness_score = max(0.0, 1.0 - (multi_seed_std / mean_lr))

    # Init sensitivity: std between default and xavier seeds
    init_sensitivity_std = None
    default_losses: List[float] = []
    xavier_losses: List[float] = []
    for r in seed_results:
        lr = r.get("loss_ratio")
        if lr is None:
            continue
        scheme = r.get("init_scheme")
        if scheme == "default":
            default_losses.append(lr)
        elif scheme == "xavier_uniform":
            xavier_losses.append(lr)
    if default_losses and xavier_losses:
        default_mean = sum(default_losses) / len(default_losses)
        xavier_mean = sum(xavier_losses) / len(xavier_losses)
        init_sensitivity_std = abs(default_mean - xavier_mean)

    # Best seed: lowest final_loss
    best_seed = None
    if loss_ratios:
        best_seed = min(
            (r for r in seed_results if r.get("final_loss") is not None),
            key=lambda r: r["final_loss"],
            default=None,
        )

    return {
        "passed_seeds": passed_seeds,
        "loss_ratios": loss_ratios,
        "val_loss_ratio": val_loss_ratio,
        "multi_seed_std": multi_seed_std,
        "robustness_score": robustness_score,
        "is_unstable": is_unstable,
        "init_sensitivity_std": init_sensitivity_std,
        "best_seed": best_seed,
    }


def screening_wikitext_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract persisted screening WikiText fields from a result dict."""
    fields: Dict[str, Any] = {}
    for key in (
        "wikitext_perplexity",
        "wikitext_score",
        "wikitext_pre_perplexity",
        "wikitext_ppl_improvement",
        "screening_wikitext_status",
        "screening_wikitext_metric_version",
    ):
        value = row.get(key)
        if value is not None:
            fields[key] = value

    budget = row.get("screening_wikitext_budget")
    if budget:
        fields["screening_wikitext_budget_json"] = json.dumps(
            json_safe(budget),
            sort_keys=True,
            separators=(",", ":"),
        )

    variant = row.get("variant")
    if variant is not None:
        fields["screening_wikitext_variant"] = variant

    elapsed = row.get("elapsed_ms")
    if elapsed is not None:
        fields["screening_wikitext_elapsed_ms"] = elapsed

    return fields


def screening_probe_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract persisted screening/probe telemetry from a result dict."""
    fields: Dict[str, Any] = {}
    for key in (
        "rapid_screening_passed",
        "rapid_screening_elapsed_ms",
        "rapid_screening_steps_completed",
        "rapid_screening_max_steps",
        "rapid_screening_degraded",
        "rapid_screening_kill_reason",
        "rapid_screening_kill_step",
        "rapid_screening_kill_metric",
        "rapid_screening_gpu_minutes_saved",
        "ar_auc",
        "ar_final_acc",
        "ar_timed_out",
        "ar_above_chance",
        "induction_auc",
        "induction_probe_train_steps",
        "induction_probe_eval_examples",
        "induction_probe_batch_size",
        "induction_probe_elapsed_ms",
        "binding_auc",
        "binding_probe_eval_examples",
        "binding_probe_elapsed_ms",
        "binding_composite",
        "local_only",
        "hellaswag_acc",
        "hellaswag_status",
        "hellaswag_n_examples",
        "screening_hellaswag_correct",
        "screening_hellaswag_total",
        "screening_hellaswag_elapsed_ms",
        "train_budget_steps",
    ):
        value = row.get(key)
        if value is not None:
            fields[key] = value

    rapid_degraded_reasons = row.get("rapid_screening_degraded_reasons")
    if rapid_degraded_reasons:
        fields["rapid_screening_degraded_reasons_json"] = json.dumps(
            json_safe(rapid_degraded_reasons),
            sort_keys=True,
            separators=(",", ":"),
        )

    rapid_metrics = row.get("rapid_screening_metrics")
    if rapid_metrics:
        fields["rapid_screening_metrics_json"] = json.dumps(
            json_safe(rapid_metrics),
            sort_keys=True,
            separators=(",", ":"),
        )

    induction_gap_accuracies = row.get("induction_gap_accuracies")
    if induction_gap_accuracies:
        fields["induction_gap_accuracies_json"] = json.dumps(
            json_safe(induction_gap_accuracies),
            sort_keys=True,
            separators=(",", ":"),
        )

    induction_gaps = row.get("induction_probe_gaps")
    if induction_gaps:
        fields["induction_probe_gaps_json"] = json.dumps(
            json_safe(induction_gaps),
            sort_keys=True,
            separators=(",", ":"),
        )

    binding_distance_accuracies = row.get("binding_distance_accuracies")
    if binding_distance_accuracies:
        fields["binding_distance_accuracies_json"] = json.dumps(
            json_safe(binding_distance_accuracies),
            sort_keys=True,
            separators=(",", ":"),
        )

    binding_distances = row.get("binding_probe_distances")
    if binding_distances:
        fields["binding_probe_distances_json"] = json.dumps(
            json_safe(binding_distances),
            sort_keys=True,
            separators=(",", ":"),
        )

    return fields


def routing_fast_lane_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract persisted routing fast-lane fields from a result dict."""
    fields: Dict[str, Any] = {}
    for key in (
        "routing_fast_lane_applied",
        "routing_fast_lane_status",
        "routing_fast_lane_metric_version",
        "routing_fast_lane_perplexity",
        "routing_fast_lane_score",
        "routing_fast_lane_pre_perplexity",
        "routing_fast_lane_ppl_improvement",
        "routing_fast_lane_elapsed_ms",
        "routing_fast_lane_slope",
        "routing_fast_lane_slope_consistent",
    ):
        value = row.get(key)
        if value is not None:
            fields[key] = value

    budget = row.get("routing_fast_lane_budget")
    if budget:
        fields["routing_fast_lane_budget_json"] = json.dumps(
            json_safe(budget),
            sort_keys=True,
            separators=(",", ":"),
        )

    routing_ops = row.get("routing_fast_lane_routing_ops")
    if routing_ops:
        fields["routing_fast_lane_routing_ops_json"] = json.dumps(
            sorted({str(op) for op in routing_ops if op}),
            sort_keys=True,
            separators=(",", ":"),
        )

    return fields


def graph_routing_ops(graph: Any) -> List[str]:
    """Return sorted routing-related ops present in a graph-like object."""
    nodes = getattr(graph, "nodes", None)
    ops: Set[str] = set()
    if isinstance(nodes, dict):
        for node in nodes.values():
            op_name = getattr(node, "op_name", None)
            if op_name in _ROUTING_FAST_LANE_OPS:
                ops.add(str(op_name))
    elif isinstance(graph, dict):
        raw_nodes = graph.get("nodes")
        if isinstance(raw_nodes, dict):
            iterable = raw_nodes.values()
        elif isinstance(raw_nodes, list):
            iterable = raw_nodes
        else:
            iterable = []
        for node in iterable:
            if not isinstance(node, dict):
                continue
            op_name = node.get("op_name")
            if op_name in _ROUTING_FAST_LANE_OPS:
                ops.add(str(op_name))
    return sorted(ops)


def trajectory_probe_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract persisted trajectory-probe fields from a benchmark result dict."""
    fields: Dict[str, Any] = {}
    for key in (
        "wikitext_ppl_200",
        "wikitext_ppl_500",
        "wikitext_improvement_ratio",
        "wikitext_eval_steps",
    ):
        value = row.get(key)
        if value is not None:
            fields[key] = value

    if row.get("wikitext_improvement_ratio") is not None:
        fields["wikitext_ppl_improvement_ratio"] = row["wikitext_improvement_ratio"]
    if row.get("eval_budget_steps") is not None:
        fields["eval_budget_steps"] = row["eval_budget_steps"]
    if row.get("evaluation_stage"):
        fields["evaluation_stage"] = row["evaluation_stage"]
    if row.get("capability_tier"):
        fields["capability_tier"] = row["capability_tier"]
    return fields


def _load_best_reference_probe_ppl(step: int) -> Optional[float]:
    """Return the best cached reference PPL at the requested checkpoint."""
    try:
        payload = json.loads(_REFERENCE_TRAJECTORY_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, dict):
        return None
    best = None
    step_key = str(step)
    for trajectory in trajectories.values():
        if not isinstance(trajectory, dict):
            continue
        checkpoints = trajectory.get("checkpoints")
        if not isinstance(checkpoints, dict):
            continue
        point = checkpoints.get(step) or checkpoints.get(step_key)
        if not isinstance(point, dict):
            continue
        try:
            ppl = float(point.get("ppl"))
        except (TypeError, ValueError):
            continue
        if best is None or ppl < best:
            best = ppl
    return best


def _trajectory_probe_capability_tier(
    ppl_500: Optional[float],
    improvement_ratio: Optional[float],
    threshold: float,
) -> str:
    """Classify probe outcome for downstream escalation and UI."""
    if ppl_500 is not None:
        best_ref_ppl = _load_best_reference_probe_ppl(500)
        if best_ref_ppl is not None and ppl_500 <= best_ref_ppl * 1.2:
            return "frontier_signal"
        if best_ref_ppl is not None and ppl_500 <= best_ref_ppl * 1.5:
            return "near_frontier"
    if improvement_ratio is not None and improvement_ratio >= threshold:
        return "slow_burn"
    return "routine"


def apply_adaptive_grad_clip(model: Any, current_clip: float) -> float:
    """Return the effective grad clip norm, respecting model's recommendation.

    Math-space models recommend higher clip values (5.0 vs default 1.0).
    """
    model_clip = getattr(model, "recommended_grad_clip", None)
    if model_clip is not None and model_clip > current_clip:
        return model_clip
    return current_clip


def _native_proactive_gating(graph) -> Dict[str, Any]:
    """
    Perform high-performance DAG validation and proactive gating using aria_core.
    Identifies stability risks and toxic motifs before compilation.
    """
    try:
        import aria_core
        from ...synthesis.primitives import OPCODE_MAP

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
    except (ImportError, RuntimeError, KeyError, TypeError) as e:
        logger.debug("Native proactive gating failed: %s", e)
        return {"passed": True, "reason": "native_gating_error", "error": str(e)}


def _native_runner_progress_report() -> Dict[str, Any]:
    try:
        from ..native.telemetry import native_runner_capability_report

        return native_runner_capability_report()
    except (ImportError, RuntimeError, OSError) as exc:
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


def _rebuild_graph_with_overrides(
    candidate_graph, overrides: Dict[int, Dict[str, Any]]
):
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
        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            logger.debug("Graph rebuild add_op failed: %s", e)
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
    except (ValueError, KeyError, RuntimeError) as e:
        logger.debug("Graph rebuild set_output failed: %s", e)
        return None
    rebuilt.metadata = dict(getattr(candidate_graph, "metadata", {}) or {})
    return rebuilt


def propose_ablation_suite(candidate_graph, hypothesis) -> List[Any]:
    """Generate counterfactual ablations by replacing suspected components."""
    from ...synthesis.primitives import get_primitive, list_primitives

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
        except (KeyError, ValueError) as e:
            logger.debug("get_primitive failed for %s: %s", node.op_name, e)
            category = ""
        if node.op_name in hyp or category in hyp:
            target_nodes.append(nid)
        elif ("math space" in hyp or "math_space" in hyp) and category == "math_space":
            target_nodes.append(nid)

    if not target_nodes:
        non_input = [
            nid
            for nid in candidate_graph.topological_order()
            if not candidate_graph.nodes[nid].is_input
        ]
        target_nodes = non_input[-2:] if len(non_input) >= 2 else non_input

    ablations: List[Any] = []
    seen: Set[str] = set()
    for nid in target_nodes[:4]:
        node = candidate_graph.nodes[nid]
        try:
            prim = get_primitive(node.op_name)
        except (KeyError, ValueError):
            continue
        key = (prim.n_inputs, prim.shape_rule)
        candidates = [
            name
            for name in replacement_by_signature.get(key, [])
            if name != node.op_name
        ]
        if not candidates:
            continue

        # Prefer a non-identical family replacement to produce a meaningful counterfactual.
        replacement = candidates[0]
        for name in candidates:
            try:
                if get_primitive(name).category != prim.category:
                    replacement = name
                    break
            except (KeyError, ValueError):
                continue
        rebuilt = _rebuild_graph_with_overrides(
            candidate_graph,
            {nid: {"op_name": replacement, "config": dict(node.config or {})}},
        )
        if rebuilt is None:
            continue
        try:
            fp = rebuilt.fingerprint()
        except (ValueError, RuntimeError):
            continue
        if fp in seen:
            continue
        seen.add(fp)
        ablations.append(rebuilt)
        if len(ablations) >= 4:
            break

    return ablations


def _build_benchmark_model(
    *,
    config,
    dev,
    model_source: str,
    arch_spec_json_str: str | None,
    graph_json_str: str | None,
    cached_json_load,
) -> Any:
    """Build a model for benchmark evaluation (shared across benchmarks)."""
    if model_source == "morphological_box" and arch_spec_json_str:
        from ...morphological_box import ArchSpec
        from ...arch_builder import BuildConfig, build_model

        spec = ArchSpec(**cached_json_load(arch_spec_json_str))
        build_cfg = BuildConfig(
            dim=config.model_dim,
            n_layers=config.n_layers,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
        )
        return build_model(spec, build_cfg).to(dev)
    elif graph_json_str:
        from ..native_runner import compile_model_native_first as compile_model
        from ...synthesis.serializer import graph_from_json

        return compile_model(
            [graph_from_json(graph_json_str)] * config.n_layers,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
        ).to(dev)
    return None


def _evaluate_investigation_benchmarks(
    *,
    config,
    dev,
    model_source: str,
    arch_spec_json_str: str | None,
    graph_json_str: str | None,
    cached_json_load,
) -> Dict[str, Any]:
    """Run lightweight benchmark evals for investigation survivors.

    Compiles the model once and runs both WikiText and TinyStories evals
    on the same instance to avoid redundant compilation.
    """
    result: Dict[str, Any] = {
        "inv_wikitext_ppl": None,
        "inv_wikitext_score": None,
        "inv_tinystories_ppl": None,
        "inv_tinystories_score": None,
    }

    try:
        model = _build_benchmark_model(
            config=config,
            dev=dev,
            model_source=model_source,
            arch_spec_json_str=arch_spec_json_str,
            graph_json_str=graph_json_str,
            cached_json_load=cached_json_load,
        )
    except (ImportError, RuntimeError, ValueError, TypeError) as exc:
        logger.debug("Benchmark model build failed: %s", exc)
        return result

    if model is None:
        return result

    eval_seq_len = min(128, config.max_seq_len)

    try:
        from ...eval.wikitext_eval import evaluate_wikitext_trajectory

        wt_result = evaluate_wikitext_trajectory(
            model,
            config.vocab_size,
            dev,
            checkpoints=(100, 500, 1000),
            seq_len=eval_seq_len,
        )
        ckpts = wt_result.get("checkpoints") or {}
        ckpt_100 = ckpts.get(100) or ckpts.get("100") or {}
        ckpt_500 = ckpts.get(500) or ckpts.get("500") or {}
        ckpt_1000 = ckpts.get(1000) or ckpts.get("1000") or {}
        ppl_100 = ckpt_100.get("ppl")
        ppl_500 = ckpt_500.get("ppl")
        ppl_1000 = ckpt_1000.get("ppl")
        improvement_ratio = wt_result.get("improvement_ratio")
        result["wikitext_ppl_200"] = ppl_100  # legacy column, now stores @100
        result["wikitext_ppl_500"] = ppl_500
        result["wikitext_improvement_ratio"] = improvement_ratio
        result["wikitext_eval_steps"] = 1000 if ppl_1000 else 500
        result["eval_budget_steps"] = 1000 if ppl_1000 else 500
        # Use ppl@1000 as the screening perplexity (matches v7 anchor)
        result["wikitext_perplexity"] = ppl_1000 or ppl_500 or ppl_100
        result["evaluation_stage"] = "PROBED"
        result["capability_tier"] = _trajectory_probe_capability_tier(
            ppl_1000 or ppl_500,
            improvement_ratio,
            float(
                getattr(config, "improvement_ratio_escalation_threshold", 2.0) or 2.0
            ),
        )
        result["inv_wikitext_ppl"] = (
            wt_result.get("peak_ppl") or ppl_1000 or ppl_500 or ppl_100
        )
        result["inv_wikitext_score"] = (
            ckpt_1000.get("score")
            if ckpt_1000.get("score") is not None
            else ckpt_500.get("score")
            if ckpt_500.get("score") is not None
            else ckpt_100.get("score")
        )
        result["wikitext_trajectory_payload"] = wt_result
        if result["inv_wikitext_ppl"] is not None:
            logger.info(
                "Investigation WikiText-103 probe ppl100=%s ppl500=%s ppl1000=%s ratio=%s tier=%s",
                f"{ppl_100:.1f}" if isinstance(ppl_100, (int, float)) else "n/a",
                f"{ppl_500:.1f}" if isinstance(ppl_500, (int, float)) else "n/a",
                f"{ppl_1000:.1f}" if isinstance(ppl_1000, (int, float)) else "n/a",
                f"{improvement_ratio:.2f}"
                if isinstance(improvement_ratio, (int, float))
                else "n/a",
                result["capability_tier"],
            )
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.debug("Investigation WikiText eval skipped: %s", exc)

    try:
        from ...eval.tinystories_eval import evaluate_tinystories

        ts_result = evaluate_tinystories(
            model,
            config.vocab_size,
            dev,
            n_train_steps=200,
            seq_len=eval_seq_len,
        )
        result["inv_tinystories_ppl"] = ts_result.get("tinystories_perplexity")
        result["inv_tinystories_score"] = ts_result.get("tinystories_score")
        if result["inv_tinystories_ppl"] is not None:
            logger.info(
                "Investigation TinyStories ppl=%.1f score=%.3f",
                result["inv_tinystories_ppl"],
                result["inv_tinystories_score"] or 0,
            )
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.debug("Investigation TinyStories eval skipped: %s", exc)

    try:
        from ...eval.hellaswag_eval import evaluate_hellaswag

        hs_result = evaluate_hellaswag(
            model,
            config.vocab_size,
            dev,
            n_examples=100,
        )
        result["hellaswag_acc"] = hs_result.get("hellaswag_acc")
        result["hellaswag_status"] = hs_result.get("hellaswag_status")
        if result["hellaswag_acc"] is not None:
            logger.info(
                "Investigation HellaSwag acc=%.1f%% (%d/%d, %.0fms)",
                result["hellaswag_acc"] * 100,
                hs_result.get("hellaswag_correct", 0),
                hs_result.get("hellaswag_total", 0),
                hs_result.get("elapsed_ms", 0),
            )
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.debug("Investigation HellaSwag eval skipped: %s", exc)

    # BLiMP linguistic minimal pairs (investigation: 50 per subtask)
    try:
        from ...eval.blimp_eval import evaluate_blimp

        blimp = evaluate_blimp(model, config.vocab_size, dev, n_per_subtask=50)
        result["blimp_overall_accuracy"] = blimp.overall_accuracy
        result["blimp_subtask_accuracies_json"] = json.dumps(blimp.subtask_accuracies)
        result["blimp_n_subtasks"] = blimp.n_subtasks
        result["blimp_status"] = blimp.status
        if blimp.overall_accuracy > 0:
            logger.info(
                "Investigation BLiMP acc=%.1f%% (%d subtasks, %d examples, %.0fms)",
                blimp.overall_accuracy * 100,
                blimp.n_subtasks,
                blimp.n_examples,
                blimp.elapsed_ms,
            )
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.debug("Investigation BLiMP eval skipped: %s", exc)

    # Binding probes: AR + induction + binding range (full suite at investigation)
    try:
        from ...eval.associative_recall import associative_recall_score
        from ...eval.induction_probe import induction_score
        from ...eval.binding_range import binding_range_profile

        ar = associative_recall_score(
            model, n_pairs=20, n_eval=200, n_train_steps=500, batch_size=16, device=dev
        )
        result["ar_auc"] = ar.auc
        result["ar_final_acc"] = ar.final_acc
        result["ar_timed_out"] = ar.timed_out
        result["ar_above_chance"] = ar.above_chance

        ind = induction_score(
            model,
            gaps=(4, 8, 16, 32, 64),
            n_train_steps=1000,
            n_eval=200,
            batch_size=32,
            device=dev,
        )
        result["induction_auc"] = ind.auc
        result["induction_gap_accuracies"] = ind.gap_accuracies

        br = binding_range_profile(
            model, distances=(2, 4, 8, 16, 32, 64), n_eval=200, device=dev
        )
        result["binding_auc"] = br.auc
        result["binding_distance_accuracies"] = br.distance_accuracies

        bc = 0.4 * ar.auc + 0.3 * ind.auc + 0.3 * br.auc
        result["binding_composite"] = round(bc, 4)

        # 3-signal AND: penalty only when ALL signals near zero (conv-3 case).
        # Mamba/RWKV fail induction+AR but may score on binding_auc — no penalty.
        from ...scientist.thresholds import (
            BINDING_AR_SOFT_GATE,
            BINDING_INDUCTION_SOFT_GATE,
            BINDING_BINDING_AUC_SOFT_GATE,
        )

        result["local_only"] = int(
            ar.auc < BINDING_AR_SOFT_GATE
            and ind.auc < BINDING_INDUCTION_SOFT_GATE
            and br.auc < BINDING_BINDING_AUC_SOFT_GATE
        )

        logger.info(
            "Investigation binding probes: ar=%.3f ind=%.3f bind=%.3f bc=%.3f local_only=%s "
            "(%.0f+%.0f+%.0fms)",
            ar.auc,
            ind.auc,
            br.auc,
            bc,
            bool(result["local_only"]),
            ar.elapsed_ms,
            ind.elapsed_ms,
            br.elapsed_ms,
        )

        # Discovery: high AR without standard attention is a priority find
        _attn_ops = {
            "softmax_attention",
            "linear_attention",
            "diff_attention",
            "graph_attention",
            "local_window_attention",
        }
        _graph_str = graph_json_str or ""
        _has_attn = any(op in _graph_str for op in _attn_ops)
        if ar.auc > 0.15 and not _has_attn:
            logger.warning(
                "DISCOVERY: High AR score without full attention — "
                "ar_auc=%.3f, model_source=%s, graph=%s",
                ar.auc,
                model_source,
                _graph_str[:200],
            )
    except (ImportError, RuntimeError, ValueError) as exc:
        logger.debug("Investigation binding probes skipped: %s", exc)

    del model
    return result


# Single-threaded pool for background benchmark evals — avoids blocking the
# investigation loop while still serialising GPU work.
_benchmark_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bench")


def _submit_benchmark_eval(
    *,
    nb,
    exp_id: str,
    source_result_id: str,
    source: Dict[str, Any],
    model_source: str,
    graph_json_str: str | None,
    arch_spec_json_str: str | None,
    n_passed: int,
    best_lr: Any,
    best_tp_json: str | None,
    robustness: float,
    investigation_passed: bool,
    config,
    dev,
    cached_json_load,
    fingerprint_incomplete: bool = False,
) -> Future:
    """Submit benchmark evals + result recording to a background thread.

    The investigation loop can continue to the next candidate immediately
    instead of blocking on 400 training steps per benchmark.

    Creates a fresh LabNotebook connection in the background thread because
    SQLite connections cannot be shared across threads (check_same_thread).
    """
    db_path = str(nb.db_path)

    def _run() -> None:
        benchmark_result = _evaluate_investigation_benchmarks(
            config=config,
            dev=dev,
            model_source=model_source,
            arch_spec_json_str=arch_spec_json_str,
            graph_json_str=graph_json_str,
            cached_json_load=cached_json_load,
        )
        # Create a thread-local notebook for DB writes
        from ..notebook import LabNotebook

        thread_nb = LabNotebook(db_path)
        try:
            _record_investigation_result(
                nb=thread_nb,
                exp_id=exp_id,
                source_result_id=source_result_id,
                source=source,
                model_source=model_source,
                graph_json_str=graph_json_str,
                arch_spec_json_str=arch_spec_json_str,
                n_passed=n_passed,
                best_lr=best_lr,
                best_tp_json=best_tp_json,
                robustness=robustness,
                investigation_passed=investigation_passed,
                benchmark_result=benchmark_result,
                fingerprint_incomplete=fingerprint_incomplete,
            )
            thread_nb.flush_writes()
        finally:
            thread_nb.close()

    return _benchmark_pool.submit(_run)


def _safe_tier(nb, result_id: str, proposed: str) -> str:
    """Return the higher of existing tier and proposed tier to prevent downgrades."""
    try:
        row = nb.conn.execute(
            "SELECT tier FROM leaderboard WHERE result_id = ?", (result_id,)
        ).fetchone()
        if row:
            existing = str(row["tier"] or "screening")
            if TIER_RANK.get(existing, 0) > TIER_RANK.get(proposed, 0):
                return existing
    except (OSError, RuntimeError) as e:
        logger.debug("_safe_tier lookup failed: %s", e)
    return proposed


def _record_investigation_result(
    *,
    nb,
    exp_id: str,
    source_result_id: str,
    source: Dict[str, Any],
    model_source: str,
    graph_json_str: str | None,
    arch_spec_json_str: str | None,
    n_passed: int,
    best_lr: Any,
    best_tp_json: str | None,
    robustness: float,
    investigation_passed: bool,
    benchmark_result: Dict[str, Any],
    fingerprint_incomplete: bool = False,
) -> None:
    """Persist leaderboard and program-results updates for investigation.

    Protects existing investigation data: if the entry already has better
    investigation results (lower loss ratio, higher robustness), those are
    preserved rather than overwritten by a weaker re-investigation.
    """
    # Check if existing investigation results are better — never overwrite with worse
    existing_inv = nb.conn.execute(
        "SELECT investigation_loss_ratio, investigation_robustness, investigation_passed, "
        "investigation_best_training FROM leaderboard WHERE result_id = ?",
        (source_result_id,),
    ).fetchone()
    if existing_inv and existing_inv["investigation_passed"]:
        existing_lr = existing_inv["investigation_loss_ratio"]
        # Never overwrite a passed investigation with a failed one or worse results
        if best_lr is None or (existing_lr is not None and existing_lr <= best_lr):
            best_lr = existing_lr
            robustness = max(
                robustness, float(existing_inv["investigation_robustness"] or 0)
            )
            best_tp_json = existing_inv["investigation_best_training"] or best_tp_json
            investigation_passed = True

    # HellaSwag hard gate: DISABLED — doesn't differentiate at nano scale.

    # Binding probe: informational logging only. No hard gate — probes are
    # too noisy at nano scale (Mamba fluctuates 0.01-0.13 across runs).
    # The soft penalty in compute_composite_v7 handles score reduction.
    _bp_ind = benchmark_result.get("induction_auc")
    if _bp_ind is not None and _bp_ind < 0.03:
        logger.info(
            "Binding probe: %s ind=%.3f (local-only signal, soft penalty applied in scoring)",
            source_result_id[:8],
            _bp_ind,
        )

    trajectory_fields = trajectory_probe_fields(benchmark_result)
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
        tier=_safe_tier(
            nb,
            source_result_id,
            "investigation"
            if investigation_passed
            else "investigation_fingerprint_incomplete"
            if fingerprint_incomplete
            else "investigation_failed",
        ),
        novelty_confidence=source.get("novelty_confidence"),
        fp_jacobian_spectral_norm=source.get("fp_jacobian_spectral_norm"),
        wikitext_perplexity=benchmark_result.get("inv_wikitext_ppl"),
        wikitext_score=benchmark_result.get("inv_wikitext_score"),
        tinystories_perplexity=benchmark_result.get("inv_tinystories_ppl"),
        tinystories_score=benchmark_result.get("inv_tinystories_score"),
        routing_savings_ratio=source.get("routing_savings_ratio"),
        activation_sparsity_score=source.get("activation_sparsity_score"),
        depth_savings_ratio=source.get("depth_savings_ratio"),
        compression_ratio=source.get("compression_ratio"),
        loss_improvement_rate=source.get("loss_improvement_rate"),
        hellaswag_acc=benchmark_result.get("hellaswag_acc"),
        ar_auc=benchmark_result.get("ar_auc"),
        induction_auc=benchmark_result.get("induction_auc"),
        binding_auc=benchmark_result.get("binding_auc"),
        binding_composite=benchmark_result.get("binding_composite"),
        local_only=benchmark_result.get("local_only"),
        **trajectory_fields,
    )

    result_id = nb.record_program_result(
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
        wikitext_perplexity=benchmark_result.get("inv_wikitext_ppl"),
        wikitext_score=benchmark_result.get("inv_wikitext_score"),
        tinystories_perplexity=benchmark_result.get("inv_tinystories_ppl"),
        tinystories_score=benchmark_result.get("inv_tinystories_score"),
        wikitext_ppl_200=benchmark_result.get("wikitext_ppl_200"),
        wikitext_ppl_500=benchmark_result.get("wikitext_ppl_500"),
        wikitext_improvement_ratio=benchmark_result.get("wikitext_improvement_ratio"),
        wikitext_eval_steps=benchmark_result.get("wikitext_eval_steps"),
        hellaswag_acc=benchmark_result.get("hellaswag_acc"),
        hellaswag_status=benchmark_result.get("hellaswag_status"),
        hellaswag_n_examples=benchmark_result.get("hellaswag_total"),
    )
    source_updates = {
        "wikitext_perplexity": benchmark_result.get("inv_wikitext_ppl"),
        "wikitext_score": benchmark_result.get("inv_wikitext_score"),
        "wikitext_ppl_200": benchmark_result.get("wikitext_ppl_200"),
        "wikitext_ppl_500": benchmark_result.get("wikitext_ppl_500"),
        "wikitext_improvement_ratio": benchmark_result.get(
            "wikitext_improvement_ratio"
        ),
        "wikitext_eval_steps": benchmark_result.get("wikitext_eval_steps"),
        "hellaswag_acc": benchmark_result.get("hellaswag_acc"),
        "hellaswag_status": benchmark_result.get("hellaswag_status"),
        "hellaswag_n_examples": benchmark_result.get("hellaswag_total"),
        "ar_auc": benchmark_result.get("ar_auc"),
        "ar_final_acc": benchmark_result.get("ar_final_acc"),
        "ar_timed_out": benchmark_result.get("ar_timed_out"),
        "ar_above_chance": benchmark_result.get("ar_above_chance"),
        "induction_auc": benchmark_result.get("induction_auc"),
        "binding_auc": benchmark_result.get("binding_auc"),
        "binding_composite": benchmark_result.get("binding_composite"),
        "local_only": benchmark_result.get("local_only"),
    }
    set_parts = []
    set_params: List[Any] = []
    for col, value in source_updates.items():
        if value is None:
            continue
        set_parts.append(f"{col} = ?")
        set_params.append(value)
    if set_parts:
        set_params.append(source_result_id)
        nb.conn.execute(
            f"UPDATE program_results SET {', '.join(set_parts)} WHERE result_id = ?",
            set_params,
        )
        nb._maybe_commit()
    try:
        from ...eval.wikitext_eval import trajectory_wikitext_payload

        payload = trajectory_wikitext_payload(
            benchmark_result.get("wikitext_trajectory_payload") or {}
        )
        if payload:
            nb.set_external_benchmarks(result_id, payload)
            if source_result_id != result_id:
                nb.set_external_benchmarks(source_result_id, payload)
    except (ImportError, OSError, ValueError) as e:
        logger.debug("Trajectory wikitext payload persist failed: %s", e)


def _upsert_screening_entry(nb, row: Dict[str, Any]) -> Optional[str]:
    """Create or update a screening-tier leaderboard entry from a program_results row.

    Single source of truth for screening leaderboard creation.
    Returns entry_id on success, None on failure.
    """
    result_id = row.get("result_id")
    if not result_id:
        return None
    wiki_fields = screening_wikitext_fields(row)
    return nb.upsert_leaderboard(
        result_id=result_id,
        model_source=row.get("model_source") or "graph_synthesis",
        architecture_desc=row.get("graph_fingerprint", "")[:40],
        screening_loss_ratio=row.get("loss_ratio"),
        screening_novelty=row.get("novelty_score"),
        screening_passed=True,
        tier="screening",
        novelty_confidence=row.get("novelty_confidence"),
        fp_jacobian_spectral_norm=row.get("fp_jacobian_spectral_norm"),
        routing_savings_ratio=row.get("routing_savings_ratio"),
        activation_sparsity_score=row.get("activation_sparsity_score"),
        depth_savings_ratio=row.get("depth_savings_ratio"),
        compression_ratio=row.get("compression_ratio"),
        **wiki_fields,
    )


# ── SSE Log Bridge ──────────────────────────────────────────────────────
# Bridges Python logging → SSE event queue so dashboard live feed shows
# ── Baseline comparison helper ──
# Replaces the 20-line recipe/compare block that was duplicated 6× across
# execution_validation.py and continuous_validation.py.

logger = logging.getLogger(__name__)


def run_baseline_comparison(
    *,
    get_baseline,
    resolve_recipe,
    make_data_fn,
    candidate_loss: float,
    train_result: dict,
    config,
    dev_str: str,
    split: str = "train",
    normalized: bool = False,
    program_params: int | None = None,
) -> float | dict | None:
    """Run a baseline comparison (raw or parameter-normalized).

    Args:
        get_baseline: callable returning the TransformerBaseline instance.
        resolve_recipe: callable(train_result, default_lr) → recipe dict.
        make_data_fn: callable(config, split) → (data_fn, data_tag, cache).
        candidate_loss: the loss value to compare against baseline.
        train_result: best seed dict with optimizer/lr/steps info.
        config: RunConfig instance.
        dev_str: device string ("cuda", "cpu").
        split: data split ("train" or "val").
        normalized: if True, call compare_normalized instead of compare.
        program_params: required when normalized=True.

    Returns:
        float (loss ratio) for raw comparison, dict for normalized, or None on failure.
    """
    baseline = get_baseline()
    steps = int(train_result.get("n_train_steps") or config.validation_steps)
    recipe = resolve_recipe(train_result, default_lr=config.stage1_lr)
    data_fn, data_tag, cache = make_data_fn(config, split)

    kwargs = dict(
        d_model=config.model_dim,
        seq_len=min(128, config.validation_seq_len),
        n_steps=max(1, steps),
        vocab_size=config.vocab_size,
        batch_size=config.validation_batch_size,
        lr=recipe["lr"],
        device=dev_str,
        n_layers=config.n_layers,
        optimizer_name=recipe["optimizer_name"],
        weight_decay=recipe["weight_decay"],
        momentum=recipe["momentum"],
        betas=recipe["betas"],
        data_fn=data_fn,
        data_tag=data_tag,
        cache_data_fn=cache,
    )

    if normalized:
        return baseline.compare_normalized(
            candidate_loss, program_params=int(program_params), **kwargs
        )
    return baseline.compare(candidate_loss, **kwargs)


# ── Shared post-eval helpers ──
# Deduplicate ~155 lines shared between _run_validation_thread
# and _run_inline_validation.


def build_validation_entry(
    *,
    source_result_id: str,
    metrics,  # ValidationMetrics
    ev_res,  # ExternalEvalResult
    nov_conf: float,
    config,  # RunConfig
):
    """Construct a ValidationEntry from metrics + eval result."""
    from ._types import ValidationEntry

    return ValidationEntry(
        result_id=source_result_id,
        val_loss_ratio=metrics.val_loss_ratio,
        val_baseline_ratio=metrics.val_baseline_ratio,
        val_normalized_ratio=metrics.val_normalized_ratio,
        param_efficiency=metrics.val_param_efficiency,
        multi_seed_std=metrics.multi_seed_std,
        robustness_score=metrics.robustness_score,
        is_unstable=metrics.is_unstable,
        seeds_passed=len(metrics.passed_seeds),
        total_seeds=int(getattr(config, "validation_n_seeds", 5) or 5),
        is_breakthrough=ev_res.is_breakthrough,
        flop_gated=ev_res.flop_gated,
        quant_int8_retention=ev_res.quant_int8_retention,
        quant_quality_per_byte=ev_res.quant_quality_per_byte,
        long_context_score=ev_res.long_context_score,
        noise_sensitivity_score=ev_res.noise_score,
        init_sensitivity_std=metrics.init_sensitivity_std,
        novelty_confidence=nov_conf,
        ood_robustness=ev_res.ood_result,
        sensitivity=ev_res.sensitivity_result,
        activation_sparsity_score=ev_res.activation_sparsity_score,
        dead_neuron_ratio=ev_res.dead_neuron_ratio,
        routing_collapse_score=ev_res.routing_collapse_score,
        wikitext_perplexity=ev_res.wikitext_perplexity,
        wikitext_score=ev_res.wikitext_score,
        tinystories_perplexity=ev_res.tinystories_perplexity,
        tinystories_score=ev_res.tinystories_score,
        cross_task_score=ev_res.cross_task_score,
        efficiency_wall_score=ev_res.efficiency_wall_score,
        max_viable_seq_len=ev_res.max_viable_seq_len,
        scaling_regime=ev_res.scaling_regime,
    )


def promote_validation_candidate(
    *,
    nb,
    source_result_id: str,
    source: dict,
    tier: str,
    metrics,  # ValidationMetrics
    ev_res,  # ExternalEvalResult
    novelty_cap: float | None = None,
) -> None:
    """Promote candidate to tier on leaderboard + store benchmark payload.

    Handles novelty capping (B3) and external benchmark storage.
    """
    from ..shared_utils import coerce_dict_payload

    # B3: cap novelty if CKA was missing
    if novelty_cap is not None:
        _raw_novelty = source.get("novelty_score")
        _raw_confidence = source.get("novelty_confidence")
        if _raw_novelty is not None:
            _raw_novelty = float(_raw_novelty) * novelty_cap
        if _raw_confidence is not None:
            _raw_confidence = float(_raw_confidence) * novelty_cap
        logger.info(
            "validation_novelty_capped: result_id=%s cap=%.2f novelty=%.4f confidence=%.4f",
            source_result_id[:12],
            novelty_cap,
            _raw_novelty or 0.0,
            _raw_confidence or 0.0,
        )
        cap_updates = []
        if _raw_novelty is not None:
            cap_updates.append(("novelty_score", _raw_novelty))
        if _raw_confidence is not None:
            cap_updates.append(("novelty_confidence", _raw_confidence))
        if cap_updates:
            try:
                _set = ", ".join(f"{c} = ?" for c, _ in cap_updates)
                _vals = [v for _, v in cap_updates] + [source_result_id]
                nb._submit_write(
                    f"UPDATE program_results SET {_set} WHERE result_id = ?",
                    _vals,
                )
                nb.flush_writes()
            except (OSError, RuntimeError) as e:
                logger.debug(
                    "B3 novelty cap DB update failed for %s: %s",
                    source_result_id[:12],
                    e,
                )

    entry = nb.get_leaderboard_entry(source_result_id)
    if not entry:
        return

    promote_kwargs = dict(
        entry_id=entry["entry_id"],
        tier=tier,
        validation_loss_ratio=metrics.val_loss_ratio,
        validation_baseline_ratio=metrics.val_baseline_ratio,
        validation_multi_seed_std=metrics.multi_seed_std,
        validation_robustness_score=metrics.robustness_score,
        validation_is_unstable=int(metrics.is_unstable),
        validation_passed=len(metrics.passed_seeds) > 0,
        normalized_baseline_ratio=metrics.val_normalized_ratio,
        param_efficiency=metrics.val_param_efficiency,
        quant_int8_retention=ev_res.quant_int8_retention,
        quant_quality_per_byte=ev_res.quant_quality_per_byte,
        robustness_long_ctx_score=ev_res.long_context_score,
        robustness_noise_score=ev_res.noise_score,
        init_sensitivity_std=metrics.init_sensitivity_std,
        fp_jacobian_spectral_norm=source.get("fp_jacobian_spectral_norm"),
        scaling_param_efficiency=ev_res.scaling_param_efficiency,
        scaling_d512_param_efficiency=ev_res.scaling_d512_param_efficiency,
        scaling_flop_efficiency=ev_res.scaling_flop_efficiency,
        scaling_gate_passed=ev_res.scaling_gate_passed_val,
        scaling_best_family=ev_res.scaling_best_family,
        scaling_confidence=ev_res.scaling_confidence,
        activation_sparsity_score=ev_res.activation_sparsity_score,
        dead_neuron_ratio=ev_res.dead_neuron_ratio,
        routing_collapse_score=ev_res.routing_collapse_score,
        wikitext_perplexity=ev_res.wikitext_perplexity,
        wikitext_score=ev_res.wikitext_score,
        tinystories_perplexity=ev_res.tinystories_perplexity,
        tinystories_score=ev_res.tinystories_score,
        cross_task_score=ev_res.cross_task_score,
        efficiency_wall_score=ev_res.efficiency_wall_score,
        max_viable_seq_len=ev_res.max_viable_seq_len,
        scaling_regime=ev_res.scaling_regime,
    )
    if novelty_cap is not None:
        _raw = source.get("novelty_score")
        if _raw is not None:
            promote_kwargs["screening_novelty"] = float(_raw) * novelty_cap
    nb.promote_to_tier(**promote_kwargs)

    # Store external benchmark payload
    external = {}
    sp = coerce_dict_payload(ev_res.scaling_result)
    if sp is not None:
        external.update(sp)
        external["scaling_comparison"] = sp
    if ev_res.long_context_details is not None:
        external["long_context"] = ev_res.long_context_details
    if external:
        nb.set_external_benchmarks(source_result_id, external)


def run_trajectory_probe(
    *,
    graph_json_str: str | None,
    config,  # RunConfig
    dev,  # torch.device
    dev_str: str,
    nb,
    source_result_id: str,
    tier: str,
    passed_seeds: list,
) -> float | None:
    """Run wikitext trajectory probe and update leaderboard.

    Returns trajectory_composite or None.
    """
    if not graph_json_str or len(passed_seeds) == 0:
        return None

    try:
        from ...eval.wikitext_eval import evaluate_wikitext_trajectory
        from ...synthesis.serializer import graph_from_json
        from ..native_runner import compile_model_native_first as _compile

        traj_graph = graph_from_json(graph_json_str)
        traj_layers = [traj_graph] * config.n_layers
        traj_model = _compile(
            traj_layers, vocab_size=config.vocab_size, max_seq_len=128
        )
        traj_model = traj_model.to(dev)
        traj_result = evaluate_wikitext_trajectory(
            traj_model,
            config.vocab_size,
            dev_str,
            checkpoints=(200, 500, 1000, 2000, 4000),
            seq_len=128,
        )

        # HellaSwag validation probe (200 examples)
        _val_hellaswag_acc = None
        try:
            from ...eval.hellaswag_eval import evaluate_hellaswag

            hs_val = evaluate_hellaswag(
                traj_model, config.vocab_size, dev_str, n_examples=200
            )
            _val_hellaswag_acc = hs_val.get("hellaswag_acc")
            if _val_hellaswag_acc is not None:
                logger.info(
                    "Validation HellaSwag acc=%.1f%% (%d/%d, %.0fms)",
                    _val_hellaswag_acc * 100,
                    hs_val.get("hellaswag_correct", 0),
                    hs_val.get("hellaswag_total", 0),
                    hs_val.get("elapsed_ms", 0),
                )
        except (ImportError, RuntimeError, ValueError) as exc_hs:
            logger.debug("Validation HellaSwag eval skipped: %s", exc_hs)

        # Validation binding probes (full suite, more examples than investigation)
        _val_ar_auc = None
        _val_ind_auc = None
        _val_binding_auc = None
        _val_local_only = None
        try:
            from ...eval.associative_recall import associative_recall_score
            from ...eval.induction_probe import induction_score as _ind_score
            from ...eval.binding_range import binding_range_profile

            _v_ar = associative_recall_score(
                traj_model,
                n_pairs=20,
                n_eval=200,
                n_train_steps=500,
                batch_size=16,
                device=dev_str,
            )
            _val_ar_auc = _v_ar.auc

            _v_ind = _ind_score(
                traj_model,
                gaps=(4, 8, 16, 32, 64),
                n_train_steps=1000,
                n_eval=200,
                batch_size=32,
                device=dev_str,
            )
            _val_ind_auc = _v_ind.auc

            _v_br = binding_range_profile(
                traj_model, distances=(2, 4, 8, 16, 32, 64), n_eval=200, device=dev_str
            )
            _val_binding_auc = _v_br.auc

            from ...scientist.thresholds import (
                BINDING_AR_SOFT_GATE,
                BINDING_INDUCTION_SOFT_GATE,
                BINDING_BINDING_AUC_SOFT_GATE,
            )

            _val_local_only = int(
                _val_ar_auc < BINDING_AR_SOFT_GATE
                and _val_ind_auc < BINDING_INDUCTION_SOFT_GATE
                and _val_binding_auc < BINDING_BINDING_AUC_SOFT_GATE
            )
            _val_bc = 0.4 * _val_ar_auc + 0.3 * _val_ind_auc + 0.3 * _val_binding_auc
            logger.info(
                "Validation binding probes: ar=%.3f ind=%.3f bind=%.3f bc=%.3f local=%s (%.0f+%.0f+%.0fms)",
                _val_ar_auc,
                _val_ind_auc,
                _val_binding_auc,
                _val_bc,
                bool(_val_local_only),
                _v_ar.elapsed_ms,
                _v_ind.elapsed_ms,
                _v_br.elapsed_ms,
            )
        except (ImportError, RuntimeError, ValueError) as exc_bp:
            logger.debug("Validation binding probes skipped: %s", exc_bp)

        del traj_model
        clear_gpu_memory()

        peak_ppl = traj_result.get("peak_ppl")
        steps_div = traj_result.get("steps_to_divergence")
        ckpts = traj_result.get("checkpoints", {})
        ppl_500 = ckpts[500].get("ppl") if 500 in ckpts else None

        entry = nb.get_leaderboard_entry(source_result_id)
        trajectory_composite = None
        if entry:
            update = {}
            if peak_ppl is not None:
                update["peak_ppl"] = peak_ppl
                vocab = config.vocab_size or 32000
                ws = max(0.0, math.log(vocab / peak_ppl) / math.log(vocab))
                update["wikitext_score"] = round(ws, 4)
            if traj_result.get("peak_step") is not None:
                update["peak_step"] = traj_result["peak_step"]
            if steps_div is not None:
                update["steps_to_divergence"] = steps_div
            if ppl_500 is not None:
                update["ppl_500"] = ppl_500
            if _val_hellaswag_acc is not None:
                update["hellaswag_acc"] = _val_hellaswag_acc
            # Binding probe data
            if _val_ar_auc is not None:
                update["ar_auc"] = _val_ar_auc
                update["ar_final_acc"] = _v_ar.final_acc
                update["ar_timed_out"] = int(_v_ar.timed_out)
                update["ar_above_chance"] = int(_v_ar.above_chance)
            if _val_ind_auc is not None:
                update["induction_auc"] = _val_ind_auc
            if _val_binding_auc is not None:
                update["binding_auc"] = _val_binding_auc
            if _val_local_only is not None:
                update["local_only"] = _val_local_only
                update["binding_composite"] = round(
                    0.4 * (_val_ar_auc or 0)
                    + 0.3 * (_val_ind_auc or 0)
                    + 0.3 * (_val_binding_auc or 0),
                    4,
                )
            # No hard gate — soft penalty in scoring handles local-only models.
            # Mamba (frontier SSM) fluctuates across the induction threshold,
            # so a hard gate would produce false positives at nano scale.
            if update:
                nb.promote_to_tier(entry_id=entry["entry_id"], tier=tier, **update)
                row = nb.conn.execute(
                    "SELECT composite_score FROM leaderboard WHERE entry_id = ?",
                    (entry["entry_id"],),
                ).fetchone()
                if row:
                    trajectory_composite = row["composite_score"]

        logger.info(
            "Trajectory probe %s: peak_ppl=%.1f steps_to_div=%s ppl_500=%s composite=%.1f",
            source_result_id[:8],
            peak_ppl or 0,
            steps_div,
            ppl_500,
            trajectory_composite or 0,
        )
        return trajectory_composite
    except Exception as e:  # top-level error boundary: probe must not crash caller
        logger.warning("Trajectory probe failed for %s: %s", source_result_id[:8], e)
        return None


def handle_breakthrough(
    *,
    is_breakthrough: bool,
    trajectory_composite: float | None,
    aria,
    nb,
    exp_id: str,
    source_result_id: str,
    source: dict,
    validation_entry,  # ValidationEntry
    val_loss_ratio: float | None,
    val_baseline_ratio: float | None,
    multi_seed_std: float,
    emit_event,
) -> bool:
    """Check trajectory-aware breakthrough and emit announcement.

    Returns final is_breakthrough value.
    """
    from ..llm.context_experiment import build_validation_context
    from ..notebook import ExperimentEntry

    # [CALIBRATION] source: judgment — 300.0 hardcoded; no config key
    if not is_breakthrough and trajectory_composite is not None:
        if trajectory_composite > 300.0:
            is_breakthrough = True
            logger.info(
                "Trajectory-aware breakthrough: %s composite=%.1f",
                source_result_id[:8],
                trajectory_composite,
            )

    if is_breakthrough:
        entry_dict = (
            validation_entry.to_dict()
            if hasattr(validation_entry, "to_dict")
            else validation_entry
        )
        ctx = build_validation_context([source], [entry_dict])
        announcement = aria.announce_breakthrough(ctx)
        nb.add_entry(
            ExperimentEntry(
                entry_type="insight",
                title="BREAKTHROUGH DETECTED",
                content=announcement,
                experiment_id=exp_id,
                tags=["breakthrough"],
            )
        )
        emit_event(
            "breakthrough_detected",
            {
                "experiment_id": exp_id,
                "result_id": source_result_id,
                "val_loss_ratio": val_loss_ratio,
                "val_baseline_ratio": val_baseline_ratio,
                "multi_seed_std": multi_seed_std,
                "announcement": announcement,
            },
        )

    return is_breakthrough


# ── SSE log handler ──
# log messages without modifying every call site.

_SSE_LOG_DEDUP_WINDOW: float = 5.0  # seconds to suppress identical messages
_SSE_LOG_RATE_LIMIT: int = 10  # max events per second per logger name
_SSE_LOG_RATE_WINDOW: float = 1.0  # sliding window for rate limit


class SSELogHandler(logging.Handler):
    """Logging handler that forwards records to the runner's SSE event queue.

    Guardrails:
    - Only captures ``research.*`` loggers at INFO+
    - Deduplicates identical messages within a time window
    - Rate-limits per logger name to prevent queue saturation
    - Never persists to DB (avoids bloating the notebook)
    """

    __slots__ = (
        "_queue",
        "_dedup",
        "_rate_counts",
        "_rate_window_start",
    )

    def __init__(self, event_queue: queue.Queue):
        super().__init__(level=logging.INFO)
        self._queue = event_queue
        # {message_text: last_emit_ts}
        self._dedup: Dict[str, float] = {}
        # {logger_name: count_in_current_window}
        self._rate_counts: Dict[str, int] = {}
        self._rate_window_start: float = time.monotonic()

    def filter(self, record: logging.LogRecord) -> bool:
        # Only research.* loggers, skip werkzeug/urllib3/etc.
        return record.name.startswith("research.")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record) if self.formatter else record.getMessage()
            now = time.monotonic()

            # ── Dedup: skip identical messages within window ──
            last_seen = self._dedup.get(msg)
            if last_seen is not None and (now - last_seen) < _SSE_LOG_DEDUP_WINDOW:
                return
            self._dedup[msg] = now

            # Prune stale dedup entries periodically (every ~50 messages)
            if len(self._dedup) > 200:
                cutoff = now - _SSE_LOG_DEDUP_WINDOW
                self._dedup = {k: v for k, v in self._dedup.items() if v > cutoff}

            # ── Rate limit per logger name ──
            if (now - self._rate_window_start) >= _SSE_LOG_RATE_WINDOW:
                self._rate_counts.clear()
                self._rate_window_start = now
            count = self._rate_counts.get(record.name, 0)
            if count >= _SSE_LOG_RATE_LIMIT:
                return
            self._rate_counts[record.name] = count + 1

            # ── Push to SSE queue ──
            # Truncate short logger prefix for dashboard display
            short_name = record.name
            if short_name.startswith("research."):
                short_name = short_name[len("research.") :]

            payload = {
                "type": "log_message",
                "data": {
                    "level": record.levelname,
                    "logger": short_name,
                    "message": msg[:500],
                    "timestamp": time.time(),
                },
                "timestamp": time.time(),
            }
            self._queue.put_nowait(payload)
        except queue.Full:
            pass  # drop log events silently when queue is saturated
        except Exception:
            pass  # top-level error boundary: never break the logging pipeline
