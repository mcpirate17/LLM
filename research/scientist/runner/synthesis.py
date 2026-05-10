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

import json
import copy
import math
import random
import sqlite3
import time
import zlib
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from ...synthesis.grammar import GrammarConfig, batch_generate
from ..native_runner import compile_model_native_first as compile_model
from ...synthesis.validator import validate_graph
from ...synthesis.serializer import graph_to_json, graph_from_json
from ...eval.flops import estimate_flops
from ..notebook import LabNotebook
from ..notebook.graph_artifacts import resolve_graph_json_value
from ..refinement_scoring import oscillation_risk_score
from ..json_utils import json_safe

import logging

logger = logging.getLogger(__name__)

from ._types import LiveProgress, RunConfig

from ._helpers import (
    program_result_kwargs_from_s1,
    propose_ablation_suite,
)


from ...synthesis.op_roles import MOE_OPS
from ..causal_attribution import (
    CausalAblationCandidate,
    run_ablation_suite,
    select_causal_ablation_candidates,
)
from ..runtime_events import publish_runtime_event


def _load_recent_synthesis_experiments(nb: LabNotebook, limit: int):
    return nb.conn.execute(
        """SELECT experiment_id, config_json, results_json
           FROM experiments
           WHERE experiment_type = 'synthesis'
             AND status = 'completed'
           ORDER BY timestamp DESC
           LIMIT ?""",
        (max(8, int(limit)),),
    ).fetchall()


def _load_synthesis_downstream_rates(
    nb: LabNotebook, exp_ids: List[str]
) -> Dict[str, Dict[str, float]]:
    downstream_by_exp: Dict[str, Dict[str, float]] = {}
    chunk_size = 900
    for start in range(0, len(exp_ids), chunk_size):
        chunk = exp_ids[start : start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        down_rows = nb.conn.execute(
            f"""SELECT
                    pr.experiment_id,
                    AVG(COALESCE(l.investigation_passed, 0)) AS investigate_rate,
                    AVG(COALESCE(l.validation_passed, 0)) AS validation_rate
                FROM program_results_compat pr
                LEFT JOIN leaderboard l ON l.result_id = pr.result_id
                WHERE pr.experiment_id IN ({placeholders})
                  AND COALESCE(pr.stage1_passed, 0) = 1
                GROUP BY pr.experiment_id""",
            tuple(chunk),
        ).fetchall()
        for down in down_rows:
            downstream_by_exp[str(down["experiment_id"])] = {
                "investigate_rate": float(down["investigate_rate"] or 0.0),
                "validation_rate": float(down["validation_rate"] or 0.0),
            }
    return downstream_by_exp


def _json_object_or_empty(raw: Any) -> Dict[str, Any]:
    try:
        payload = json.loads(raw) if raw else {}
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _results_payload_or_empty(nb: LabNotebook, raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        payload = nb._decompress(raw)
    except (TypeError, ValueError, zlib.error):
        return _json_object_or_empty(raw)
    return payload if isinstance(payload, dict) else {}


def _synthesis_reward_terms(
    results_payload: Dict[str, Any], downstream: Dict[str, float]
) -> Tuple[float, float, float]:
    total = max(int(results_payload.get("total") or 0), 1)
    s1_rate = float(results_payload.get("stage1_passed") or 0.0) / total
    best_loss = results_payload.get("best_loss_ratio")
    loss_term = 0.0 if best_loss is None else max(0.0, min(1.0, 1.0 - float(best_loss)))
    novelty_term = max(
        0.0,
        min(1.0, float(results_payload.get("best_novelty_score") or 0.0)),
    )
    investigate_rate = float(downstream.get("investigate_rate") or 0.0)
    validation_rate = float(downstream.get("validation_rate") or 0.0)
    reward = (
        0.35 * s1_rate
        + 0.25 * loss_term
        + 0.15 * novelty_term
        + 0.15 * investigate_rate
        + 0.10 * validation_rate
    )
    return reward, s1_rate, investigate_rate, validation_rate


def _new_synthesis_regime_bucket() -> Dict[str, Any]:
    return {
        "n": 0,
        "cumulative_reward": 0.0,
        "recent_weighted_reward": 0.0,
        "recent_weight_sum": 0.0,
        "mean_reward": 0.0,
        "investigate_rate": 0.0,
        "validation_rate": 0.0,
        "s1_rate": 0.0,
    }


def _record_synthesis_regime_reward(
    bucket: Dict[str, Any],
    *,
    reward: float,
    recency_weight: float,
    s1_rate: float,
    investigate_rate: float,
    validation_rate: float,
) -> None:
    bucket["n"] += 1
    bucket["cumulative_reward"] += reward
    bucket["recent_weighted_reward"] += reward * recency_weight
    bucket["recent_weight_sum"] += recency_weight
    bucket["investigate_rate"] += investigate_rate
    bucket["validation_rate"] += validation_rate
    bucket["s1_rate"] += s1_rate


def _finalize_synthesis_regime_stats(stats: Dict[str, Dict[str, Any]]) -> None:
    for bucket in stats.values():
        n = max(int(bucket["n"]), 1)
        bucket["mean_reward"] = float(bucket["cumulative_reward"]) / n
        recent_weight_sum = float(bucket["recent_weight_sum"] or 1.0)
        bucket["recent_mean_reward"] = (
            float(bucket["recent_weighted_reward"]) / recent_weight_sum
        )
        bucket["investigate_rate"] /= n
        bucket["validation_rate"] /= n
        bucket["s1_rate"] /= n


def _graph_is_moe(graph) -> bool:
    """Return True if graph contains any MoE/routing-with-experts op."""
    return any(
        node.op_name in MOE_OPS for node in graph.nodes.values() if not node.is_input
    )


class _SynthesisMixin:
    """Grammar config, ablation, weight management, diversity."""

    _SYNTHESIS_REGIMES: Tuple[str, ...] = (
        "math_space_explore",
        "deep_routing",
        "wide_shallow",
        "efficiency",
        "capability_first",
        "routing_variant",
        "math_space_boost",
        "exotic",
    )

    @classmethod
    def _cycle_regime_name(cls, n_experiments: int) -> str:
        return cls._SYNTHESIS_REGIMES[n_experiments % len(cls._SYNTHESIS_REGIMES)]

    @classmethod
    def _apply_synthesis_regime(
        cls,
        cfg: RunConfig,
        config: RunConfig,
        regime: str,
    ) -> None:
        """Apply one named synthesis regime onto a shallow config copy."""
        # Preserve user-supplied floors
        user_max_depth = config.max_depth
        user_max_ops = config.max_ops
        user_split_prob = config.grammar_split_prob
        user_three_way = config.three_way_split_prob
        user_residual = config.residual_prob

        if regime == "math_space_explore":
            cfg.math_space_weight = max(config.math_space_weight, 1.0)
            cfg.residual_prob = 0.5
            cfg.max_ops = 16
            cfg.composition_depth = 2
        elif regime == "deep_routing":
            cfg.max_depth = 12
            cfg.max_ops = 20
            cfg.residual_prob = 0.8
            cfg.composition_depth = 3
            cfg._routing_first_mode = True
        elif regime == "wide_shallow":
            cfg.max_depth = 8
            cfg.max_ops = 16
            cfg.residual_prob = 0.6
            cfg.composition_depth = 2
        elif regime == "efficiency":
            cfg.max_depth = 10
            cfg.max_ops = 16
            cfg.residual_prob = 0.7
            cfg.composition_depth = 2
            cfg._efficiency_mode = True
        elif regime == "capability_first":
            cfg._capability_first_mode = True
            cfg.max_depth = 12
            cfg.max_ops = 22
            cfg.residual_prob = 0.8
            cfg.composition_depth = 3
        elif regime == "routing_variant":
            cfg.max_depth = 10
            cfg.max_ops = 18
            cfg.residual_prob = 0.7
            cfg.composition_depth = 2
            cfg._routing_first_mode = True
        elif regime == "math_space_boost":
            cfg.math_space_weight = max(config.math_space_weight, 2.5)
            cfg.max_depth = 10
            cfg.max_ops = 16
            cfg.residual_prob = 0.7
            cfg.composition_depth = 2
        else:
            cfg.max_depth = 12
            cfg.max_ops = 20
            cfg.residual_prob = 0.4
            cfg.grammar_split_prob = 0.6
            cfg.math_space_weight = max(config.math_space_weight, 1.5)
            cfg.composition_depth = 3
            cfg._exotic_mode = True

        # Enforce user floors — never shrink below what was requested
        cfg.max_depth = max(cfg.max_depth, user_max_depth)
        cfg.max_ops = max(cfg.max_ops, user_max_ops)
        cfg.grammar_split_prob = max(cfg.grammar_split_prob, user_split_prob)
        cfg.three_way_split_prob = max(cfg.three_way_split_prob, user_three_way)
        cfg.residual_prob = max(cfg.residual_prob, user_residual)

        # If the user ticked capability_first in the dashboard, honor it
        # regardless of which cycle we're on. This lets the automatic
        # cycle 4 exercise it once per rotation, but a user who wants
        # capability_first everywhere can force it.
        if getattr(config, "_capability_first_mode", False):
            cfg._capability_first_mode = True

    def _recent_synthesis_regime_stats(
        self,
        nb: LabNotebook,
        *,
        limit: int = 64,
    ) -> Dict[str, Dict[str, Any]]:
        """Return recent realized reward summaries for adaptive regime choice."""
        rows = _load_recent_synthesis_experiments(nb, limit)
        if not rows:
            return {}

        exp_ids = [str(row["experiment_id"]) for row in rows if row["experiment_id"]]
        downstream_by_exp = (
            _load_synthesis_downstream_rates(nb, exp_ids) if exp_ids else {}
        )

        stats: Dict[str, Dict[str, Any]] = {}
        for recency_idx, row in enumerate(rows):
            config_payload = _json_object_or_empty(row["config_json"])
            results_payload = _results_payload_or_empty(nb, row["results_json"])
            regime = str(
                config_payload.get("adaptive_synthesis_regime")
                or config_payload.get("synthesis_regime")
                or self._cycle_regime_name(recency_idx)
            )
            downstream = downstream_by_exp.get(str(row["experiment_id"]), {})
            reward, s1_rate, investigate_rate, validation_rate = (
                _synthesis_reward_terms(results_payload, downstream)
            )
            bucket = stats.setdefault(
                regime,
                _new_synthesis_regime_bucket(),
            )
            _record_synthesis_regime_reward(
                bucket,
                reward=reward,
                recency_weight=1.0 / float(recency_idx + 1),
                s1_rate=s1_rate,
                investigate_rate=investigate_rate,
                validation_rate=validation_rate,
            )

        _finalize_synthesis_regime_stats(stats)
        return stats

    def _select_synthesis_regime(
        self,
        config: RunConfig,
        n_experiments: int,
        nb: Optional[LabNotebook] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Choose a synthesis regime with realized-outcome UCB instead of mod-8."""
        fallback_regime = self._cycle_regime_name(n_experiments)
        if nb is None:
            return fallback_regime, {
                "policy": "mod8_fallback",
                "regime": fallback_regime,
                "reason": "No notebook available for adaptive regime selection.",
            }

        try:
            regime_stats = self._recent_synthesis_regime_stats(nb, limit=64)
        except (sqlite3.OperationalError, ValueError, TypeError) as e:
            logger.debug("Adaptive regime stats failed: %s", e)
            regime_stats = {}
        if not regime_stats:
            return fallback_regime, {
                "policy": "mod8_fallback",
                "regime": fallback_regime,
                "reason": "No resolved synthesis history available.",
            }

        total_trials = sum(int(item.get("n") or 0) for item in regime_stats.values())
        best_regime = fallback_regime
        best_score = float("-inf")
        score_breakdown: Dict[str, Dict[str, float]] = {}
        for regime in self._SYNTHESIS_REGIMES:
            stat = regime_stats.get(regime, {})
            n_trials = int(stat.get("n") or 0)
            mean_reward = float(stat.get("mean_reward") or 0.0)
            recent_reward = float(stat.get("recent_mean_reward") or mean_reward)
            uncertainty = math.sqrt(
                math.log(max(total_trials, 1) + 1.0) / (n_trials + 1.0)
            )
            score = (0.55 * mean_reward) + (0.25 * recent_reward) + (0.20 * uncertainty)
            score_breakdown[regime] = {
                "n": float(n_trials),
                "mean_reward": round(mean_reward, 6),
                "recent_mean_reward": round(recent_reward, 6),
                "uncertainty": round(uncertainty, 6),
                "score": round(score, 6),
            }
            if score > best_score:
                best_score = score
                best_regime = regime

        return best_regime, {
            "policy": "ucb_realized_reward",
            "regime": best_regime,
            "fallback_regime": fallback_regime,
            "score_breakdown": score_breakdown,
            "history_size": int(total_trials),
        }

    def _diversify_grammar_config(
        self,
        config: RunConfig,
        n_experiments: int,
        nb: Optional[LabNotebook] = None,
    ) -> Tuple[RunConfig, Dict[str, Any]]:
        """Apply an adaptive synthesis regime and return the chosen policy."""
        cfg = copy.copy(config)
        regime, decision = self._select_synthesis_regime(
            config=config,
            n_experiments=n_experiments,
            nb=nb,
        )
        self._apply_synthesis_regime(cfg, config, regime)
        decision = dict(decision)
        decision["regime"] = regime
        return cfg, decision

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
                (json.dumps(json_safe(stored_config)), exp_id),
            )
            nb.conn.commit()
        except (sqlite3.OperationalError, json.JSONDecodeError, ValueError) as e:
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
        except (RuntimeError, AttributeError) as e:
            logger.debug("Grammar weight audit info failed: %s", e)
            audit_info = {}
        self._log_learning_event_compat(
            nb,
            "grammar_weights_applied",
            f"Applied learned grammar weights for experiment {exp_id}",
            old_weights=old_weights,
            new_weights=new_weights,
            evidence=json.dumps(json_safe({"audit_query": audit_info}), sort_keys=True),
        )
        return audit_info

    def _run_ablation_experiment(
        self,
        nb: LabNotebook,
        config: RunConfig,
        hypothesis: str,
        ablation_graphs: List[Any],
        original_loss_ratio: Optional[float] = None,
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
                min_splits=config.min_splits,
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
            except (RuntimeError, ValueError, TypeError) as e:
                logger.debug("Ablation graph compilation failed: %s", e)
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
            self._log_learning_event_compat(
                nb,
                "ablation_skipped_no_evaluable_graphs",
                f"Skipped ablation run: no evaluable graphs for {hypothesis}",
                evidence=json.dumps(json_safe(evidence), sort_keys=True),
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
        best_ablation_lr: Optional[float] = None
        result_ids: List[str] = []
        for idx, graph in enumerate(evaluable_graphs):
            try:
                model = compile_model(
                    [graph],
                    vocab_size=config.vocab_size,
                    max_seq_len=config.max_seq_len,
                ).to(dev)
            except (RuntimeError, ValueError, TypeError) as e:
                logger.debug("Ablation model compilation failed: %s", e)
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
            s05_passed = (
                bool(s0.stability_score >= config.stage05_stability_threshold)
                if s0_passed
                else False
            )
            if s0_passed:
                stage0_pass += 1
            if s05_passed:
                stage05_pass += 1
            s1_passed = False
            s1 = {}
            loss_ratio = None
            if s05_passed:
                s1 = self._micro_train(
                    model,
                    config,
                    dev,
                    seed=self._stable_seed(exp_id, idx, "ablation"),
                )
                s1_passed = bool(s1.get("passed", False))
                loss_ratio = s1.get("loss_ratio")
            if s1_passed:
                stage1_pass += 1
            if loss_ratio is not None and (
                best_ablation_lr is None or loss_ratio < best_ablation_lr
            ):
                best_ablation_lr = loss_ratio

            ablation_kwargs: Dict[str, Any] = (
                program_result_kwargs_from_s1(s1, model_source="ablation")
                if s1
                else {"model_source": "ablation"}
            )
            rid = nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=graph.fingerprint(),
                graph_json=graph_to_json(graph),
                stage0_passed=s0_passed,
                stage05_passed=s05_passed,
                stage1_passed=s1_passed,
                stage0_error=s0.error,
                intentional_rerun_reason="ablation_counterfactual",
                **ablation_kwargs,
            )
            result_ids.append(rid)
            publish_runtime_event(
                notebook_path=self.notebook_path,
                event_type="ablation_child_completed",
                producer="runner.synthesis",
                run_id=exp_id,
                payload={
                    "experiment_id": exp_id,
                    "result_id": rid,
                    "index": idx + 1,
                    "total": len(evaluable_graphs),
                    "stage0_passed": s0_passed,
                    "stage05_passed": s05_passed,
                    "stage1_passed": s1_passed,
                    "loss_ratio": loss_ratio,
                    "hellaswag_acc": ablation_kwargs.get("hellaswag_acc"),
                    "blimp_overall_accuracy": ablation_kwargs.get(
                        "blimp_overall_accuracy"
                    ),
                    "induction_screening_auc": ablation_kwargs.get(
                        "induction_screening_auc"
                    ),
                    "binding_screening_auc": ablation_kwargs.get(
                        "binding_screening_auc"
                    ),
                    "ar_legacy_auc": ablation_kwargs.get("ar_legacy_auc"),
                    "wikitext_perplexity": ablation_kwargs.get("wikitext_perplexity"),
                },
            )

        total = len(result_ids)
        outcome = "supported" if total > 0 and stage1_pass == 0 else "not_supported"
        if total == 0:
            outcome = "inconclusive"

        # Compute ablation delta: how much worse are ablated graphs vs original?
        ablation_best_lr = best_ablation_lr if best_ablation_lr is not None else None
        ablation_delta = None
        if original_loss_ratio is not None and ablation_best_lr is not None:
            ablation_delta = (
                ablation_best_lr - original_loss_ratio
            )  # positive = ablated is worse

        nb.flush_writes()
        experiment_results = {
            "total": total,
            "stage0_passed": stage0_pass,
            "stage05_passed": stage05_pass,
            "stage1_passed": stage1_pass,
            "best_loss_ratio": ablation_best_lr,
            "best_novelty_score": None,
        }
        if ablation_delta is not None:
            experiment_results["ablation_delta"] = ablation_delta
            experiment_results["original_loss_ratio"] = original_loss_ratio
        aria_summary = f"Ablation outcome: {outcome}"
        self._publish_terminal_event(
            producer="runner.synthesis",
            event_type="completed",
            exp_id=exp_id,
            payload={
                "completed_at": time.time(),
                "results": experiment_results,
                "aria_summary": aria_summary,
                "mode": "ablation",
            },
        )
        self._complete_experiment_compat(
            nb=nb,
            experiment_id=exp_id,
            results=experiment_results,
            aria_summary=aria_summary,
        )

        # Log ablation delta as learning event for grammar feedback
        if ablation_delta is not None:
            self._log_learning_event_compat(
                nb,
                "ablation_delta_measured",
                f"Ablation for '{hypothesis}': delta={ablation_delta:.4f} "
                f"(original={original_loss_ratio:.4f}, ablated={ablation_best_lr:.4f})",
                evidence=json.dumps(
                    json_safe(
                        {
                            "hypothesis": hypothesis,
                            "original_loss_ratio": original_loss_ratio,
                            "ablation_best_loss_ratio": ablation_best_lr,
                            "ablation_delta": ablation_delta,
                            "ablation_s1_pass_rate": stage1_pass / max(total, 1),
                            "total_ablation_graphs": total,
                        }
                    ),
                    sort_keys=True,
                ),
            )

        return ([exp_id], outcome)

    def _run_causal_ablation_candidates(
        self,
        nb: LabNotebook,
        config: RunConfig,
        candidates: List[CausalAblationCandidate],
        *,
        max_graphs: int,
    ) -> Dict[str, Any]:
        """Run bounded causal ablations and persist rule evidence."""
        if not candidates:
            return {"planned": 0, "launched": 0, "evidence": []}
        launched = 0
        evidence_rows: List[Dict[str, Any]] = []
        graph_budget = max(1, int(max_graphs or 1))
        for candidate in candidates:
            suite = propose_ablation_suite(candidate.graph, candidate.hypothesis)
            if not suite:
                continue
            suite = suite[:graph_budget]
            result = run_ablation_suite(
                nb=nb,
                runner=self,
                config=config,
                candidate=candidate,
                graphs=suite,
                campaign="causal_attribution",
            )
            if result is None:
                continue
            launched += 1
            evidence_rows.append(result)
            self._log_learning_event_compat(
                nb,
                "causal_ablation_evidence",
                (
                    f"Causal ablation {result['outcome']} for "
                    f"{candidate.rule_type}:{candidate.rule_key}"
                ),
                evidence=json.dumps(json_safe(result), sort_keys=True),
            )
        return {
            "planned": len(candidates),
            "launched": launched,
            "evidence": evidence_rows,
        }

    def _run_causal_ablation_for_result_thread(
        self,
        result_id: str,
        config: RunConfig,
    ) -> None:
        """Background entry point for a manual program-detail ablation run."""
        nb = self._make_notebook()
        try:
            with self._lock:
                self._progress = LiveProgress(
                    experiment_id=f"causal-ablation:{result_id[:12]}",
                    status="evaluating",
                    total_programs=max(1, int(config.causal_ablation_max_graphs or 4)),
                    aria_message="Running causal ablation suite...",
                )
            row = nb.conn.execute(
                """SELECT experiment_id
                   FROM program_results_compat
                   WHERE result_id = ?""",
                (result_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Program not found: {result_id}")
            parent_experiment_id = str(row["experiment_id"] or "")
            candidates = [
                item
                for item in select_causal_ablation_candidates(
                    nb,
                    experiment_id=parent_experiment_id,
                    max_survivors=max(1, int(config.causal_ablation_top_k or 1)),
                    max_signals_per_survivor=max(
                        1, int(config.causal_ablation_max_signals or 2)
                    ),
                )
                if item.parent_result_id == result_id
            ]
            result = self._run_causal_ablation_candidates(
                nb,
                config,
                candidates,
                max_graphs=max(1, int(config.causal_ablation_max_graphs or 4)),
            )
            self._emit_event(
                "causal_ablation_completed",
                {"result_id": result_id, "result": result},
            )
            self._update_progress(
                status="completed",
                aria_message=(
                    f"Causal ablation complete: {result.get('launched', 0)} suite(s)."
                ),
            )
        except Exception as exc:
            logger.exception("Causal ablation failed for %s", result_id)
            self._update_progress(
                status="failed", error=str(exc), aria_message=str(exc)
            )
            self._emit_event(
                "causal_ablation_failed",
                {"result_id": result_id, "error": str(exc)},
            )
        finally:
            nb.close()

    def _maybe_run_causal_ablation_loop(
        self,
        nb: LabNotebook,
        exp_id: str,
        config: RunConfig,
        results: Dict[str, Any],
    ) -> None:
        """Optionally run causal ablations after a synthesis experiment."""
        if not getattr(config, "enable_causal_ablation", False):
            return
        interval = int(getattr(config, "causal_ablation_interval", 0) or 0)
        if interval <= 0:
            return
        completed_count = int(
            nb.conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE status = 'completed'"
            ).fetchone()[0]
            or 0
        )
        if completed_count % interval != 0:
            return
        if int(results.get("stage1_passed") or 0) <= 0:
            return
        candidates = select_causal_ablation_candidates(
            nb,
            experiment_id=exp_id,
            max_survivors=max(1, int(getattr(config, "causal_ablation_top_k", 1) or 1)),
            max_signals_per_survivor=max(
                1, int(getattr(config, "causal_ablation_max_signals", 2) or 2)
            ),
        )
        summary = self._run_causal_ablation_candidates(
            nb,
            config,
            candidates,
            max_graphs=max(
                1, int(getattr(config, "causal_ablation_max_graphs", 4) or 4)
            ),
        )
        if summary.get("launched"):
            results["causal_ablation"] = {
                "planned": summary.get("planned", 0),
                "launched": summary.get("launched", 0),
            }

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
                        hypothesis_text,
                        already_tested,
                    )
                    ablation_outcome = "skipped_already_tested"
                    strong_corr = False  # prevent triggering below
            except sqlite3.OperationalError as e:
                logger.debug("Ablation dedup check failed: %s", e)

        if strong_corr and top_signal_interpretable:
            row = nb.conn.execute("""SELECT graph_json, loss_ratio FROM program_results_compat
                   WHERE stage1_passed = 1 AND graph_json IS NOT NULL
                   ORDER BY loss_ratio ASC NULLS LAST LIMIT 1""").fetchone()
            if row and row["graph_json"]:
                try:
                    graph_json = resolve_graph_json_value(
                        nb.conn,
                        nb.db_path,
                        row["graph_json"],
                    )
                    base_graph = graph_from_json(graph_json)
                    base_loss_ratio = (
                        float(row["loss_ratio"])
                        if row["loss_ratio"] is not None
                        else None
                    )
                    suite = propose_ablation_suite(base_graph, hypothesis_text)
                    queued_plan = [g.fingerprint() for g in suite]
                    if suite:
                        ablation_experiments, ablation_outcome = (
                            self._run_ablation_experiment(
                                nb=nb,
                                config=config,
                                hypothesis=hypothesis_text,
                                ablation_graphs=suite,
                                original_loss_ratio=base_loss_ratio,
                            )
                        )
                except (RuntimeError, ValueError, TypeError) as e:
                    logger.debug("Ablation run failed: %s", e)
        elif strong_corr:
            ablation_outcome = "skipped_low_quality_signal"

        gate_pass = bool(
            has_ablation_support or (strong_corr and bool(ablation_experiments))
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

        return {op: round(count / total, 6) for op, count in sorted(counts.items())}

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
            if isinstance(prev_results_raw, bytes):
                prev_results = nb._decompress(prev_results_raw) or {}
            elif prev_results_raw:
                prev_results = json.loads(prev_results_raw)
            else:
                prev_results = {}
            previous_distribution = prev_results.get("generated_op_distribution")
            if not isinstance(previous_distribution, dict) or not previous_distribution:
                return None

            l1 = self._distribution_l1_distance(
                current_distribution, previous_distribution
            )
            delta_pairs = []
            for op in set(current_distribution.keys()) | set(
                previous_distribution.keys()
            ):
                delta = current_distribution.get(op, 0.0) - previous_distribution.get(
                    op, 0.0
                )
                if abs(delta) > 1e-12:
                    delta_pairs.append((op, delta))
            delta_pairs.sort(key=lambda item: abs(item[1]), reverse=True)
            top_changes = [
                {"op": op, "delta": round(delta, 6)} for op, delta in delta_pairs[:5]
            ]

            return {
                "previous_experiment_id": row["experiment_id"],
                "l1_distance": round(l1, 6),
                "top_op_deltas": top_changes,
            }
        except (
            sqlite3.OperationalError,
            json.JSONDecodeError,
            ValueError,
            KeyError,
        ) as e:
            logger.debug(
                "Failed comparing generated-op distribution for %s: %s", exp_id, e
            )
            return None

    def _compute_multi_objective_fitness(
        self, s1_result, sandbox_result, graph, config
    ):
        """Multi-objective fitness: quality + efficiency + speed + learning."""
        weights = {
            "quality": 0.25,
            "efficiency": 0.45,
            "speed": 0.15,
            "learning_speed": 0.15,
        }

        components = {}

        # Quality: combine loss ratio with absolute loss.
        # Loss ratio alone is misleading — a model can go from 250 → 12 (ratio 0.05)
        # but still be above random baseline.  Penalize final_loss above baseline.
        _fl = s1_result.get("final_loss") if s1_result else None
        _il = s1_result.get("initial_loss") if s1_result else None
        if _fl is not None and _il is not None and _il > 0:
            lr = _fl / _il
        else:
            lr = s1_result.get("loss_ratio", 1.0) if s1_result else 1.0
        ratio_quality = max(0.0, 1.0 - lr)

        # Absolute quality: how far below random baseline?
        # ln(vocab_size) is the expected loss of a random model.
        import math as _math

        _random_baseline = _math.log(max(config.vocab_size, 2))
        if _fl is not None and _fl < _random_baseline:
            # Below baseline: scale 0→1 as final_loss goes from baseline to 0
            absolute_quality = 1.0 - (_fl / _random_baseline)
        else:
            absolute_quality = 0.0

        # Blend: 50% ratio + 50% absolute.  This ensures evolution rewards
        # both improvement rate AND reaching a useful absolute loss level.
        components["quality"] = 0.5 * ratio_quality + 0.5 * absolute_quality

        # Efficiency: use actual efficiency_multiple vs GPT-2.
        # MoE models have more total params but only activate a fraction per
        # token — don't penalize total param count for MoE architectures.
        is_moe = _graph_is_moe(graph)
        max_params = config.model_dim * config.vocab_size * 2
        param_count = getattr(sandbox_result, "param_count", 0) or 0
        try:
            from ..leaderboard_scoring import compute_efficiency_multiple

            eff_result = compute_efficiency_multiple(
                loss_ratio=s1_result.get("loss_ratio") if s1_result else None,
                param_count=param_count or None,
                forward_time_ms=(
                    s1_result.get("forward_time_ms")
                    if s1_result
                    else (
                        getattr(sandbox_result, "forward_time_ms", None)
                        if sandbox_result
                        else None
                    )
                ),
                peak_memory_mb=s1_result.get("peak_memory_mb") if s1_result else None,
                throughput_tok_s=s1_result.get("throughput") if s1_result else None,
                is_moe=is_moe,
            )
            if eff_result and eff_result.get("geomean", 0) > 0:
                components["efficiency"] = min(1.0, eff_result["geomean"] / 5.0)
            elif is_moe:
                # MoE: score on speed/throughput, not param count
                components["efficiency"] = 0.5
            elif param_count > 0 and max_params > 0:
                components["efficiency"] = max(
                    0.0, 1.0 - min(param_count / max_params, 1.0)
                )
            else:
                components["efficiency"] = 0.0
        except (ImportError, RuntimeError, ValueError, TypeError) as e:
            logger.debug("Efficiency multiple computation failed: %s", e)
            if is_moe:
                components["efficiency"] = 0.5
            elif param_count > 0 and max_params > 0:
                components["efficiency"] = max(
                    0.0, 1.0 - min(param_count / max_params, 1.0)
                )
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

    @staticmethod
    def _apply_analysis_to_grammar(
        base_grammar: GrammarConfig,
        analysis: Dict[str, Any],
        intent: str,
    ) -> GrammarConfig:
        """Apply RefinementAnalyzer recipe hints to a grammar config."""
        recipe = analysis.get("recipe", {})
        hints = recipe.get("grammar_hints", {})

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

    def _recent_synthesis_health(
        self, nb: LabNotebook, window: int = 5
    ) -> Dict[str, float]:
        """Summarize recent synthesis outcomes for fallback decisions."""
        experiments = nb.get_recent_experiments(max(window * 3, window))
        rows = [
            row
            for row in experiments
            if str(row.get("experiment_type") or "") == "synthesis"
            and str(row.get("status") or "") == "completed"
        ][:window]
        total_programs = sum(
            max(int(r.get("n_programs_generated") or 0), 0) for r in rows
        )
        total_s1 = sum(max(int(r.get("n_stage1_passed") or 0), 0) for r in rows)
        rate = (float(total_s1) / float(total_programs)) if total_programs > 0 else 0.0
        return {
            "window": float(len(rows)),
            "total_programs": float(total_programs),
            "total_s1": float(total_s1),
            "s1_rate": float(rate),
        }

    def _generate_refinement_graphs(
        self,
        exp_id: str,
        config: RunConfig,
        nb: LabNotebook,
        grammar: GrammarConfig,
    ) -> List:
        """Generate local mutations around selected source result IDs."""
        source_ids = [
            rid.strip()
            for rid in str(config.refine_source_result_ids or "").split(",")
            if rid.strip()
        ]
        target_n = max(1, int(config.n_programs))
        if not source_ids:
            logger.warning(
                "Refinement mode requested without source IDs; falling back to synthesis generation"
            )
            return batch_generate(target_n, grammar).graphs

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
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.debug("Graph deserialization failed: %s", e)
                continue
            source_pairs.append((source_id, parent_graph, source))
            if source.get("stage1_passed"):
                source_stage1_passed += 1

        if not source_pairs:
            logger.warning(
                "Refinement mode had %d source IDs but no reconstructable graphs; falling back to synthesis",
                len(source_ids),
            )
            return batch_generate(target_n, grammar).graphs

        try:
            from ...search.evolution import _mutate_graph
        except (ImportError, AttributeError) as e:
            logger.warning(
                "Mutation helper unavailable (%s); falling back to synthesis generation",
                e,
            )
            return batch_generate(target_n, grammar).graphs

        seed = self._stable_seed("fingerprint_refine", exp_id, ",".join(source_ids))
        rng = random.Random(seed)
        per_source = max(1, int(config.refine_mutations_per_source or 1))
        target_pool = max(
            target_n, target_n * max(1, int(config.refine_pool_multiplier or 1))
        )
        candidate_pool: List[Tuple[float, Any]] = []
        seen_fingerprints: Set[str] = set()
        op_success = self._op_success_lookup(nb)
        intent = str(config.refine_intent or "balanced").lower()

        # Apply analysis-driven grammar hints if available
        analysis_data: Optional[Dict[str, Any]] = None
        if config.refine_analysis_json:
            try:
                analysis_data = json.loads(config.refine_analysis_json)
                grammar = self._apply_analysis_to_grammar(
                    grammar, analysis_data, intent
                )
                logger.info(
                    "Experiment %s: applied analysis-driven grammar hints (intent=%s, %d exclude, %d boost)",
                    exp_id[:8],
                    intent,
                    len(
                        analysis_data.get("recipe", {})
                        .get("grammar_hints", {})
                        .get("exclude_ops", [])
                    ),
                    len(
                        analysis_data.get("recipe", {})
                        .get("grammar_hints", {})
                        .get("boost_ops", {})
                    ),
                )
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.warning(
                    "Experiment %s: failed to parse refine_analysis_json: %s",
                    exp_id[:8],
                    e,
                )

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
                parent_fp = parent_graph.fingerprint()
                for _ in range(per_source):
                    if len(candidate_pool) >= target_pool:
                        break
                    try:
                        child = _mutate_graph(parent_graph, grammar, rng)
                    except (RuntimeError, ValueError, KeyError) as e:
                        logger.debug("Mutation failed: %s", e)
                        continue

                    # Z15: Prune dead branches (unreachable nodes) before validation
                    # to prevent redundant complexity from bloat mutations.
                    child.prune_unreachable_nodes()

                    validation = validate_graph(
                        child,
                        max_ops=max(1, int(config.max_ops)),
                        max_depth=max(1, int(config.max_depth)),
                        min_splits=config.min_splits,
                    )
                    if not validation.valid:
                        continue

                    fp = child.fingerprint()
                    if fp in seen_fingerprints:
                        continue
                    seen_fingerprints.add(fp)
                    child.metadata.setdefault("refinement", {})
                    child.metadata["refinement"]["source_result_id"] = source_id
                    child.metadata["refinement"]["seed_fingerprint"] = parent_fp
                    child.metadata["refinement"]["intent"] = intent
                    score, score_breakdown = self._score_refinement_candidate(
                        child,
                        op_success=op_success,
                        intent=intent,
                        source_row=source_row,
                        include_breakdown=True,
                    )
                    child.metadata["refinement"]["intent_score"] = score
                    child.metadata["refinement"]["intent_score_breakdown"] = (
                        score_breakdown
                    )
                    if analysis_data:
                        recipe = analysis_data.get("recipe", {})
                        child.metadata["refinement"]["analysis_driven"] = True
                        child.metadata["refinement"]["analysis_recipe"] = {
                            "recommended_intent": recipe.get(
                                "recommended_intent", "balanced"
                            ),
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
            fallback = batch_generate(target_n - len(mutated_graphs), grammar).graphs
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
        return (
            abs(loss_a - loss_b)
            + abs(nov_a - nov_b)
            + (abs(ops_a - ops_b) / 16.0)
            + fp_term
        )

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
            flops_per_token = (
                flop_est.flops_per_token
                if flop_est and flop_est.flops_per_token > 0
                else (params * 2)
            )
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug("FLOP estimation failed: %s", e)
            flops_per_token = params * 2
        baseline_fpt = 2.0 * _cfg_dim**2 * _cfg_layers
        flop_efficiency = min(1.0, baseline_fpt / max(flops_per_token, 1.0))
        # MoE: don't penalize total param count — active params are a fraction
        if _graph_is_moe(graph):
            param_efficiency_proxy = 0.5  # neutral — judge on FLOP efficiency instead
        else:
            param_efficiency_proxy = min(1.0, (6 * _cfg_dim**2) / max(params, 1.0))
        compression_proxy = (
            0.5 * flop_efficiency
            + 0.3 * param_efficiency_proxy
            + 0.2 / (1.0 + 0.1 * depth)
        )
        novelty_proxy = min(
            1.0, (unique_ops / max(1, n_ops)) + (0.1 if depth >= 4 else 0.0)
        )

        sparse_hint_ops = (
            "sparse",
            "gate",
            "topk",
            "mask",
            "threshold",
            "skip",
            "mixture",
        )
        sparse_op_bonus = 0.0
        if ops:
            sparse_op_bonus = sum(
                1.0
                for op in ops
                if any(token in op.lower() for token in sparse_hint_ops)
            ) / len(ops)
        sparsity_proxy = min(1.0, 0.7 * compression_proxy + 0.3 * sparse_op_bonus)
        oscillation_risk, stability = oscillation_risk_score(graph)

        parent_novelty = float((source_row or {}).get("novelty_score") or 0.0)
        parent_quality = 1.0 - float((source_row or {}).get("loss_ratio") or 1.0)

        # Fingerprint-aware bonuses
        fingerprint_quality_bonus = 0.0
        behavioral_diversity_bonus = 0.0
        if source_row:
            # Bonus for source with completed behavioral fingerprint
            if source_row.get("fingerprint_json"):
                fingerprint_quality_bonus = 0.1
            # Reward mutations away from transformer-like behavior
            cka_t = source_row.get("fp_cka_vs_transformer")
            if cka_t is not None:
                behavioral_diversity_bonus = 0.05 * max(0.0, 1.0 - float(cka_t))

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
                "oscillation_penalty": -0.06 * oscillation_risk,
            }
        else:  # balanced
            weighted_terms = {
                "learned_quality": 0.35 * learned_quality,
                "compression_proxy": 0.25 * compression_proxy,
                "novelty_proxy": 0.20 * novelty_proxy,
                "parent_signal": 0.20 * max(parent_quality, parent_novelty),
                "oscillation_penalty": -0.10 * oscillation_risk,
            }
        if mode in {"quality", "compression", "sparsity"}:
            weighted_terms["oscillation_penalty"] = -0.10 * oscillation_risk
        # Fingerprint-aware bonuses (additive, independent of mode)
        if fingerprint_quality_bonus > 0:
            weighted_terms["fingerprint_quality_bonus"] = fingerprint_quality_bonus
        if behavioral_diversity_bonus > 0:
            weighted_terms["behavioral_diversity_bonus"] = behavioral_diversity_bonus
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
                **stability,
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

    def _select_diverse_refinement_sources(
        self,
        candidates: List[Dict[str, Any]],
        *,
        top_k: int,
        min_distance: float,
        novelty_pressure: float,
    ) -> List[Dict[str, Any]]:
        """Select top-k candidates while preserving pairwise diversity.

        Uses behavioral fingerprint features when available:
        - hierarchy_fitness bonus: structured representations learn better
        - low CKA-vs-transformer bonus: indicates genuine novelty
        - fingerprint completion bonus: completed fingerprints are more trustworthy
        """
        if not candidates:
            return []
        ranked = []
        for row in candidates:
            loss = float(row.get("loss_ratio") or 1.0)
            novelty = float(row.get("novelty_score") or 0.0)
            quality = max(0.0, 1.0 - min(loss, 1.5))
            score = (1.0 - novelty_pressure) * quality + novelty_pressure * novelty

            # Behavioral fingerprint bonuses
            hierarchy = row.get("fp_interaction_hierarchy")
            if hierarchy is not None:
                score += 0.05 * min(1.0, float(hierarchy))
            cka_transformer = row.get("fp_cka_vs_transformer")
            if cka_transformer is not None:
                # Reward low similarity to transformer (indicates novelty)
                score += 0.03 * max(0.0, 1.0 - float(cka_transformer))
            if row.get("fingerprint_json"):
                score += 0.02  # completed fingerprint bonus

            ranked.append((score, row))
        ranked.sort(key=lambda x: x[0], reverse=True)

        selected: List[Dict[str, Any]] = []
        for _, row in ranked:
            if any(
                self._refinement_candidate_distance(row, prev) < min_distance
                for prev in selected
            ):
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
                       stage1_passed, graph_n_ops, timestamp,
                       fp_interaction_hierarchy, fp_cka_vs_transformer,
                       fp_cka_vs_ssm, fingerprint_json
                FROM program_results_compat
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
            novelty_pressure=max(
                0.0, min(1.0, float(config.refinement_novelty_pressure or 0.0))
            ),
        )
        source_ids = [
            str(row.get("result_id") or "") for row in selected if row.get("result_id")
        ]
        if not source_ids:
            return None

        radius = max(0.05, min(1.0, float(config.refinement_mutation_radius or 0.35)))
        mutation_rate = max(
            0.10, min(0.95, float(config.mutation_rate) * (0.5 + radius))
        )
        generations = max(1, int(config.refinement_generations or 1))
        budget_programs = max(
            int(config.n_programs),
            int(config.refinement_budget_programs or config.n_programs),
        )
        per_gen = max(
            4, min(int(config.n_programs), max(4, budget_programs // generations))
        )
        mutations_per_source = max(1, int(round(2 + 4 * radius)))
        pool_multiplier = max(
            2, int(round(2 + 3 * float(config.refinement_novelty_pressure or 0.0)))
        )

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
                "refinement_novelty_pressure": float(
                    config.refinement_novelty_pressure
                ),
            },
        }
