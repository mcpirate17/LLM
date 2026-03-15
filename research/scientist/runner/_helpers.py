"""Shared helper functions for the runner package.

Centralised here to avoid duplication across submodules.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)
_REFERENCE_TRAJECTORY_PATH = Path("research/eval/reference_trajectories.json")

# ── Normalized loss_ratio ──
# loss_ratio = final_loss / initial_loss is init-dependent: Kaiming init
# yields initial_loss ~250 while small/ortho init yields ~ln(V).  This makes
# screening ratios (0.008) and investigation ratios (0.24) incomparable for
# the SAME architecture.  Normalizing against ln(vocab_size) — the expected
# cross-entropy of a uniform distribution — gives a consistent, interpretable
# metric across all stages and init schemes.

_DEFAULT_VOCAB_SIZE: int = 32_000
_REFERENCE_INITIAL_LOSS: float = math.log(_DEFAULT_VOCAB_SIZE)  # ~10.37


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
            _cv = (_var ** 0.5) / _mean
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
    if step == quarter and initial_loss and loss_val >= initial_loss * progress_threshold:
        return {
            "error": (
                f"inflight_no_progress: at step {step}/{total_steps}, "
                f"loss={loss_val:.4f} vs initial={initial_loss:.4f} "
                f"(ratio={loss_val/initial_loss:.3f})"
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
    """
    import gc
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


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
            budget,
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
    except Exception as e:
        logger.debug(f"Native proactive gating failed: {e}")
        return {"passed": True, "reason": "native_gating_error", "error": str(e)}


def _native_runner_progress_report() -> Dict[str, Any]:
    try:
        from ..native_runner import native_runner_capability_report
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
    except Exception as exc:
        logger.debug("Benchmark model build failed: %s", exc)
        return result

    if model is None:
        return result

    eval_seq_len = min(128, config.max_seq_len)

    try:
        from ...eval.wikitext_eval import evaluate_wikitext_trajectory

        wt_result = evaluate_wikitext_trajectory(
            model, config.vocab_size, dev,
            checkpoints=(200, 500),
            seq_len=eval_seq_len,
        )
        ckpts = wt_result.get("checkpoints") or {}
        ckpt_200 = ckpts.get(200) or ckpts.get("200") or {}
        ckpt_500 = ckpts.get(500) or ckpts.get("500") or {}
        ppl_200 = ckpt_200.get("ppl")
        ppl_500 = ckpt_500.get("ppl")
        improvement_ratio = wt_result.get("improvement_ratio")
        result["wikitext_ppl_200"] = ppl_200
        result["wikitext_ppl_500"] = ppl_500
        result["wikitext_improvement_ratio"] = improvement_ratio
        result["wikitext_eval_steps"] = 500
        result["eval_budget_steps"] = 500
        result["evaluation_stage"] = "PROBED"
        result["capability_tier"] = _trajectory_probe_capability_tier(
            ppl_500,
            improvement_ratio,
            float(getattr(config, "improvement_ratio_escalation_threshold", 2.0) or 2.0),
        )
        result["inv_wikitext_ppl"] = wt_result.get("peak_ppl") or ppl_500 or ppl_200
        result["inv_wikitext_score"] = (
            ckpt_500.get("score")
            if ckpt_500.get("score") is not None
            else ckpt_200.get("score")
        )
        result["wikitext_trajectory_payload"] = wt_result
        if result["inv_wikitext_ppl"] is not None:
            logger.info(
                "Investigation WikiText probe ppl200=%s ppl500=%s ratio=%s tier=%s",
                f"{ppl_200:.1f}" if isinstance(ppl_200, (int, float)) else "n/a",
                f"{ppl_500:.1f}" if isinstance(ppl_500, (int, float)) else "n/a",
                f"{improvement_ratio:.2f}" if isinstance(improvement_ratio, (int, float)) else "n/a",
                result["capability_tier"],
            )
    except Exception as exc:
        logger.debug("Investigation WikiText eval skipped: %s", exc)

    try:
        from ...eval.tinystories_eval import evaluate_tinystories

        ts_result = evaluate_tinystories(
            model, config.vocab_size, dev,
            n_train_steps=200, seq_len=eval_seq_len,
        )
        result["inv_tinystories_ppl"] = ts_result.get("tinystories_perplexity")
        result["inv_tinystories_score"] = ts_result.get("tinystories_score")
        if result["inv_tinystories_ppl"] is not None:
            logger.info(
                "Investigation TinyStories ppl=%.1f score=%.3f",
                result["inv_tinystories_ppl"], result["inv_tinystories_score"] or 0,
            )
    except Exception as exc:
        logger.debug("Investigation TinyStories eval skipped: %s", exc)

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
            )
            thread_nb.flush_writes()
        finally:
            thread_nb.close()

    return _benchmark_pool.submit(_run)


_TIER_RANK = {"screened_out": 0, "screening": 1, "investigation": 2, "validation": 3, "breakthrough": 4}


def _safe_tier(nb, result_id: str, proposed: str) -> str:
    """Return the higher of existing tier and proposed tier to prevent downgrades."""
    try:
        row = nb.conn.execute(
            "SELECT tier FROM leaderboard WHERE result_id = ?", (result_id,)
        ).fetchone()
        if row:
            existing = str(row["tier"] or "screening")
            if _TIER_RANK.get(existing, 0) > _TIER_RANK.get(proposed, 0):
                return existing
    except Exception:
        pass
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
        (source_result_id,)
    ).fetchone()
    if existing_inv and existing_inv["investigation_passed"]:
        existing_lr = existing_inv["investigation_loss_ratio"]
        if existing_lr is not None and best_lr is not None and existing_lr < best_lr:
            best_lr = existing_lr
            robustness = max(robustness, float(existing_inv["investigation_robustness"] or 0))
            best_tp_json = existing_inv["investigation_best_training"] or best_tp_json
            investigation_passed = True

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
        tier=_safe_tier(nb, source_result_id, "investigation" if investigation_passed else "screened_out"),
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
    )
    source_updates = {
        "wikitext_perplexity": benchmark_result.get("inv_wikitext_ppl"),
        "wikitext_score": benchmark_result.get("inv_wikitext_score"),
        "wikitext_ppl_200": benchmark_result.get("wikitext_ppl_200"),
        "wikitext_ppl_500": benchmark_result.get("wikitext_ppl_500"),
        "wikitext_improvement_ratio": benchmark_result.get("wikitext_improvement_ratio"),
        "wikitext_eval_steps": benchmark_result.get("wikitext_eval_steps"),
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
    except Exception:
        pass


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
