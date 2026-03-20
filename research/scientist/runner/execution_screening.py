"""Execution mixin: screening thread + core experiment logic."""

from __future__ import annotations

import json
import math
import random
import time
import traceback
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from ..json_utils import json_safe

import torch

from ...synthesis.grammar import GrammarConfig, batch_generate
from ...synthesis.motifs import VALIDATED_MOTIFS
from ...synthesis.templates import DEFAULT_TEMPLATE_WEIGHTS, TEMPLATES
from ..native_runner import compile_model_native_first as compile_model
from ...synthesis.validator import validate_graph
from ..refinement_scoring import rank_synthesis_candidates_by_stability
from ...eval.flops import estimate_flops
from ...eval.perf_budget import evaluate_perf_budget_gate
from ..notebook import LabNotebook, ExperimentEntry

import logging

logger = logging.getLogger(__name__)

from ._types import RunConfig, LiveProgress
from ._helpers import _native_proactive_gating, clear_gpu_memory

try:
    from ..judgment import score_candidate, JudgmentContext

    _HAS_JUDGMENT = True
except ImportError:
    _HAS_JUDGMENT = False

_EXPLORATION_BUDGET = 0.15

# S0.75 initial-loss threshold: architectures with initial CE loss above this
# are killed before rapid screening. Calibrated from diagnosis (2026-03-20):
# architectures with init_loss > 50 have deep unscaled projection chains and
# cannot reach the random-baseline floor (~10.94) within 500 S1 steps.
# Normal architectures start at init_loss ~11–16 (near ln(vocab_size)=11.52).
INITIAL_LOSS_THRESHOLD: float = 50.0

# Number of gradient steps for S0.75 mini-train probe
_S075_PROBE_STEPS: int = 5

_BUCKET_TEMPLATE_BOOSTS: Dict[str, Dict[str, float]] = {
    "attention-heavy": {
        "transformer_block": 1.6,
        "hybrid_parallel": 1.2,
        "residual_block": 0.5,
    },
    "mixing-heavy": {
        "hybrid_parallel": 1.4,
        "sequential": 0.9,
        "residual_block": 0.5,
    },
    "sparse": {
        "sparse_ffn": 1.8,
        "bottleneck": 1.1,
        "moe": 0.8,
    },
    "hybrid": {
        "hybrid_parallel": 1.8,
        "transformer_block": 1.3,
        "parallel_split": 0.8,
    },
    "exotic": {
        "parallel_split": 1.0,
        "gated_residual": 0.7,
        "dense_cascade": 0.5,
    },
}
_TOP_OP_TEMPLATE_HINTS: Dict[str, Dict[str, float]] = {
    "attention": {"transformer_block": 1.1, "hybrid_parallel": 0.7},
    "scan": {"hybrid_parallel": 1.0, "sequential": 0.5},
    "state_space": {"hybrid_parallel": 1.0, "sequential": 0.5},
    "conv": {"hybrid_parallel": 0.8, "sequential": 0.5},
    "sparse": {"sparse_ffn": 1.2, "bottleneck": 0.6},
    "rank": {"bottleneck": 0.8, "sparse_ffn": 0.4},
    "moe": {"moe": 1.2, "gated_residual": 0.4},
    "gate": {"gated_residual": 0.9, "moe": 0.4},
    "norm": {"residual_block": 0.6, "transformer_block": 0.5},
}


def _freeze_op_pair_priors(
    priors: List[Dict[str, Any]],
) -> Tuple[Tuple[str, float], ...]:
    return tuple(
        (
            str(row.get("signature") or ""),
            round(float(row.get("success_rate") or 0.0), 4),
        )
        for row in priors
        if row.get("signature")
    )


def _freeze_fingerprint_buckets(
    buckets: List[Dict[str, Any]],
) -> Tuple[Tuple[str, int, float, Tuple[str, ...]], ...]:
    return tuple(
        (
            str(row.get("bucket") or ""),
            int(row.get("n_graphs") or 0),
            round(float(row.get("s1_rate") or 0.0), 4),
            tuple(
                str(op.get("op_name") or "")
                for op in (row.get("top_ops") or [])
                if op.get("op_name")
            ),
        )
        for row in buckets
        if row.get("bucket")
    )


@lru_cache(maxsize=32)
def _cached_signal_weight_maps(
    op_pair_priors: Tuple[Tuple[str, float], ...],
    fingerprint_buckets: Tuple[Tuple[str, int, float, Tuple[str, ...]], ...],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    pair_rates = {
        signature: success_rate
        for signature, success_rate in op_pair_priors
        if success_rate > 0.3
    }
    motif_weights = {
        motif_name: round(
            sum(
                pair_rates.get(
                    f"{motif.steps[index].op_name}->{motif.steps[index + 1].op_name}",
                    0.0,
                )
                for index in range(len(motif.steps) - 1)
            ),
            4,
        )
        for motif_name, motif in VALIDATED_MOTIFS.items()
    }
    motif_weights = {
        motif_name: weight
        for motif_name, weight in motif_weights.items()
        if weight > 0.0
    }

    template_bonuses: Dict[str, float] = {}
    for bucket_name, n_graphs, s1_rate, top_ops in fingerprint_buckets:
        dominance = max(1.0, float(n_graphs)) * max(0.25, s1_rate)
        for template_name, boost in _BUCKET_TEMPLATE_BOOSTS.get(
            bucket_name, {}
        ).items():
            if template_name in TEMPLATES:
                template_bonuses[template_name] = template_bonuses.get(
                    template_name, 0.0
                ) + (boost * dominance)
        for op_name in top_ops:
            lowered = op_name.lower()
            for token, boosts in _TOP_OP_TEMPLATE_HINTS.items():
                if token not in lowered:
                    continue
                for template_name, boost in boosts.items():
                    if template_name in TEMPLATES:
                        template_bonuses[template_name] = template_bonuses.get(
                            template_name, 0.0
                        ) + (boost * dominance)

    template_weights = {
        template_name: round(
            DEFAULT_TEMPLATE_WEIGHTS.get(template_name, 1.0) + bonus, 4
        )
        for template_name, bonus in template_bonuses.items()
        if bonus > 0.0
    }
    return template_weights, motif_weights


def _apply_insight_adjustments(
    nb: LabNotebook,
    grammar: GrammarConfig,
    template_weights: Dict[str, float],
    motif_weights: Dict[str, float],
) -> None:
    """Apply high-confidence Bayesian insights to grammar config.

    - Structural insights: adjust max_ops, residual_prob
    - Template insights: scale template weights
    - Composition insights: boost motif weights for winning op combos
    """
    _MIN_CONFIDENCE = 0.6

    # ── Structural insights ──
    try:
        structural = nb.get_insights(
            exclude_display_only=True,
            insight_level="structural",
            limit=20,
        )
    except Exception:
        structural = []

    for ins in structural:
        alpha = float(ins.get("alpha") or 1.0)
        beta_ = float(ins.get("beta_") or 1.0)
        conf = alpha / (alpha + beta_)
        if conf < _MIN_CONFIDENCE:
            continue

        subject = str(ins.get("subject_key") or "")
        evidence = ins.get("evidence_json")
        if isinstance(evidence, str):
            try:
                import json as _json

                evidence = _json.loads(evidence)
            except Exception:
                evidence = {}
        if not isinstance(evidence, dict):
            evidence = {}

        if subject == "graph_size_cap" and evidence.get("recommended_max"):
            recommended = int(evidence["recommended_max"])
            grammar.max_ops = min(grammar.max_ops, recommended)
        elif subject == "graph_size_optimal":
            # Nudge composition_depth toward the optimal bucket
            best = evidence.get("best_bucket", "")
            if "7-9" in best:
                grammar.composition_depth = min(grammar.composition_depth, 2)
                grammar.max_ops = min(grammar.max_ops, 12)
        else:
            # ── Data-driven structural rules from profiling ──
            # Composition rates → residual_prob (nudge if residual > sequential)
            comp_rates = evidence.get("composition_rates")
            if isinstance(comp_rates, dict):
                res_info = comp_rates.get("residual")
                seq_info = comp_rates.get("sequential")
                res_rate = (
                    float(res_info.get("rate", 0)) if isinstance(res_info, dict) else 0
                )
                seq_rate = (
                    float(seq_info.get("rate", 0)) if isinstance(seq_info, dict) else 0
                )
                if res_rate > seq_rate:
                    grammar.residual_prob = min(
                        0.85, grammar.residual_prob + conf * 0.1
                    )
            # Also check param_param_residual as standalone evidence
            ppr = evidence.get("param_param_residual")
            if isinstance(ppr, dict) and float(ppr.get("rate", 0)) > 0.7:
                grammar.residual_prob = min(0.85, grammar.residual_prob + conf * 0.05)

            # Corrector ops → boost
            correctors = evidence.get("corrector_ops")
            if isinstance(correctors, dict):
                for op_name, stats in correctors.items():
                    rate = (
                        float(stats.get("correction_rate", 0))
                        if isinstance(stats, dict)
                        else 0
                    )
                    if rate >= 0.5:
                        cur = grammar.op_weights.get(op_name, 1.0)
                        grammar.op_weights[op_name] = cur * (1.0 + conf * 0.15 * rate)

    # ── Template insights ──
    try:
        template_insights = nb.get_insights(
            exclude_display_only=True,
            insight_level="template",
            limit=20,
        )
    except Exception:
        template_insights = []

    for ins in template_insights:
        alpha = float(ins.get("alpha") or 1.0)
        beta_ = float(ins.get("beta_") or 1.0)
        conf = alpha / (alpha + beta_)
        if conf < _MIN_CONFIDENCE:
            continue

        subject = str(ins.get("subject_key") or "")
        # Suppress templates matching the subject
        subject_parts = {
            s.strip().lower()
            for s in subject.replace("+", " ").replace("_", " ").split()
            if len(s.strip()) >= 3
        }
        for tpl_name in list(template_weights.keys()):
            tpl_parts = {s.lower() for s in tpl_name.replace("_", " ").split()}
            if subject_parts & tpl_parts:
                template_weights[tpl_name] *= max(0.2, 1.0 - conf * 0.6)

    # ── Composition insights ──
    try:
        composition = nb.get_insights(
            exclude_display_only=True,
            insight_level="composition",
            limit=50,
        )
    except Exception:
        composition = []

    for ins in composition:
        alpha = float(ins.get("alpha") or 1.0)
        beta_ = float(ins.get("beta_") or 1.0)
        conf = alpha / (alpha + beta_)
        if conf < _MIN_CONFIDENCE:
            continue

        subject = str(ins.get("subject_key") or "")
        evidence = ins.get("evidence_json")
        if isinstance(evidence, str):
            try:
                import json as _json

                evidence = _json.loads(evidence)
            except Exception:
                evidence = {}
        if not isinstance(evidence, dict):
            evidence = {}
        insight_type = str(ins.get("insight_type") or "")
        semantic = str(ins.get("semantic_key") or "")

        # ── Profiling composition rules ──
        if insight_type == "composition_rule" and semantic.startswith("profiling:"):
            _apply_profiling_composition_rule(grammar, subject, evidence, conf)
            continue

        # ── Universal stabilizer / top_op boosting ──
        if insight_type == "top_op" and semantic.startswith("profiling:"):
            stabilizer_set = evidence.get("stabilizer_set") or {}
            for op_name, stats in stabilizer_set.items():
                rate = float(stats.get("rate", 0))
                if rate >= 0.8:
                    cur = grammar.op_weights.get(op_name, 1.0)
                    grammar.op_weights[op_name] = cur * (1.0 + conf * 0.4)
            continue

        # ── Legacy motif boosting ──
        subject_ops = {s.strip() for s in subject.split("+") if s.strip()}
        if not subject_ops:
            continue

        for motif_name, motif in VALIDATED_MOTIFS.items():
            motif_ops = {step.op_name for step in motif.steps}
            if subject_ops & motif_ops:
                base = motif_weights.get(motif_name, motif.lift)
                motif_weights[motif_name] = base * (1.0 + conf * 0.3)


def _apply_profiling_composition_rule(
    grammar: GrammarConfig,
    subject: str,
    evidence: dict,
    conf: float,
) -> None:
    """Apply a profiling-derived composition rule to grammar config.

    Fully data-driven: reads evidence_json to decide what to adjust.
    No hard-coded subject keys — any insight with the right evidence
    structure will be applied.  As Bayesian confidence (alpha/beta)
    changes through experiments, the adjustment strength scales linearly.

    Recognized evidence_json patterns:
      - risk_score + valid_followers → penalize subject op, boost followers
      - best_followers / bridge_ops → boost named ops by their stability rate
      - dampener_ops → boost named ops
      - composition_rates.residual → nudge residual_prob
      - corrector_ops → boost distribution correctors
      - stabilizer_set → boost universal stabilizers
    """
    # ── 1. Op-specific risk rule: penalize the subject op, boost its valid followers ──
    risk = evidence.get("risk_score")
    if risk is not None and float(risk) > 50:
        penalty = max(0.15, 1.0 - (float(risk) / 100.0) * conf)
        cur = grammar.op_weights.get(subject, 1.0)
        grammar.op_weights[subject] = cur * penalty
        # Also boost the ops that rescue it
        for follower in evidence.get("valid_followers", []):
            cur_f = grammar.op_weights.get(follower, 1.0)
            grammar.op_weights[follower] = cur_f * (1.0 + conf * 0.1)
        return

    # ── 2. Composition rates with residual data → nudge residual_prob ──
    comp_rates = evidence.get("composition_rates")
    if isinstance(comp_rates, dict):
        res_info = comp_rates.get("residual")
        seq_info = comp_rates.get("sequential")
        if isinstance(res_info, dict) and isinstance(seq_info, dict):
            res_rate = float(res_info.get("rate", 0))
            seq_rate = float(seq_info.get("rate", 0))
            if res_rate > seq_rate:
                grammar.residual_prob = min(0.85, grammar.residual_prob + conf * 0.1)
        return

    # ── 3. Named op sets with stability rates → boost high-stability ops ──
    for key in ("best_followers", "bridge_ops", "stabilizer_set"):
        named_set = evidence.get(key)
        if isinstance(named_set, dict) and named_set:
            for op_name, stats in named_set.items():
                if isinstance(stats, dict):
                    rate = float(stats.get("rate", 0))
                else:
                    rate = float(stats) if stats else 0
                if rate >= 0.7:
                    cur = grammar.op_weights.get(op_name, 1.0)
                    boost = 1.0 + conf * 0.2 * rate
                    grammar.op_weights[op_name] = cur * boost
            return

    # ── 4. Dampener ops list → boost ──
    dampeners = evidence.get("dampener_ops")
    if isinstance(dampeners, list) and dampeners:
        for op_name in dampeners:
            cur = grammar.op_weights.get(op_name, 1.0)
            grammar.op_weights[op_name] = cur * (1.0 + conf * 0.25)
        return

    # ── 5. Distribution corrector ops → boost ──
    correctors = evidence.get("corrector_ops")
    if isinstance(correctors, dict) and correctors:
        for op_name, stats in correctors.items():
            rate = (
                float(stats.get("correction_rate", 0)) if isinstance(stats, dict) else 0
            )
            if rate >= 0.5:
                cur = grammar.op_weights.get(op_name, 1.0)
                grammar.op_weights[op_name] = cur * (1.0 + conf * 0.15 * rate)
        return

    # ── 6. Insights with valid_followers but no risk_score → informational boost only ──
    valid_followers = evidence.get("valid_followers")
    if isinstance(valid_followers, list) and valid_followers:
        for op_name in valid_followers:
            cur = grammar.op_weights.get(op_name, 1.0)
            grammar.op_weights[op_name] = cur * (1.0 + conf * 0.1)
        return


def _build_signal_weight_maps(
    nb: LabNotebook,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    try:
        op_pair_priors = nb.get_op_pair_priors(min_support=5, limit=50)
        fingerprint_buckets = nb.get_fingerprint_buckets(limit=5)
    except Exception:
        return {}, {}
    if not op_pair_priors and not fingerprint_buckets:
        return {}, {}
    return _cached_signal_weight_maps(
        _freeze_op_pair_priors(op_pair_priors or []),
        _freeze_fingerprint_buckets(fingerprint_buckets or []),
    )


def _judgment_rerank(
    graphs: List,
    nb: LabNotebook,
    log: logging.Logger,
) -> List[Tuple[Any, float]]:
    """Rerank candidates by judgment score, preserving exploration budget.

    Fetches research signals from the notebook, scores each candidate,
    sorts by total_score descending, but reserves the bottom 15% of slots
    for under-sampled candidates (lowest support counts).

    Returns list of (graph, judgment_score) tuples with hard-failure candidates removed.
    Falls back to original order with neutral scores if judgment module unavailable.
    """
    if not _HAS_JUDGMENT or not graphs:
        return [(g, 0.5) for g in graphs]

    # Fetch signals from notebook (fast — cached in DB)
    try:
        signals: Dict[str, Any] = {}
        signals["op_pair_priors"] = nb.get_op_pair_priors(min_support=5, limit=50)
        risk = nb.get_failure_risk_signatures(limit=50)
        signals["failure_risk_signatures"] = risk.get("failure_risk_signatures", [])
        signals["critical_failures"] = risk.get("critical_failures", [])
        signals["fingerprint_buckets"] = nb.get_fingerprint_buckets(limit=5)
        signals["lineage_successors"] = nb.get_lineage_successor_stats(limit=50)
    except Exception as exc:
        log.debug(
            "judgment_rerank: signals fetch failed (%s), using original order", exc
        )
        return [(g, 0.5) for g in graphs]

    ctx = JudgmentContext()
    scored: List[tuple] = []
    skipped = 0

    for graph in graphs:
        try:
            result = score_candidate(graph, ctx, signals)
        except Exception:
            scored.append((graph, 0.5, 0))  # neutral score on error
            continue

        if result.risk_flags:
            log.info(
                "judgment_rerank: discarding candidate with risk flags: %s",
                result.risk_flags,
            )
            skipped += 1
            continue

        support_total = sum(result.support_counts.values())
        scored.append((graph, result.total_score, support_total))

    if skipped:
        log.info(
            "judgment_rerank: filtered %d candidates with critical failures", skipped
        )

    if not scored:
        return [(g, 0.5) for g in graphs]

    # Sort by score descending
    scored.sort(key=lambda t: t[1], reverse=True)

    n = len(scored)
    n_explore = max(1, int(n * _EXPLORATION_BUDGET))
    n_exploit = n - n_explore

    if n > n_explore + 1:
        exploit = scored[:n_exploit]
        explore_pool = scored[n_exploit:]
        # Sort explore pool by support (lowest first = most novel)
        explore_pool.sort(key=lambda t: t[2])
        scored = exploit + explore_pool

    # Decision trace: log judgment summary for observability
    try:
        scores = [t[1] for t in scored]
        nb.log_learning_event(
            "judgment_rerank",
            f"Reranked {n} candidates ({skipped} filtered)",
            n_candidates=n,
            n_filtered=skipped,
            score_min=round(min(scores), 3),
            score_max=round(max(scores), 3),
            score_mean=round(sum(scores) / len(scores), 3),
            n_explore=n_explore,
        )
    except Exception:
        pass

    return [(t[0], t[1]) for t in scored]


class _ExecutionScreeningMixin:
    """Screening experiment thread and core experiment execution."""

    __slots__ = ()

    def _run_experiment_thread(self, exp_id: str, config: RunConfig, hypothesis: str):
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
                results, config, hypothesis, nb
            )

            summary = self.aria.experiment_summary(results, context=context)
            insights = self._analyze_results(results, exp_id, nb, context=context)

            # Store LLM analysis if available
            llm_analysis = self.aria.analyze_results(results, context=context)

            # Validate hypothesis
            try:
                validation = self.aria.validate_hypothesis(hypothesis, results, context)
                if validation:
                    nb.add_entry(
                        ExperimentEntry(
                            entry_type="analysis",
                            title="Hypothesis Validation",
                            content=validation.get("explanation", ""),
                            experiment_id=exp_id,
                            metadata={"validated": validation.get("validated", False)},
                        )
                    )
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

            # Update op success rates and failure signatures after experiment.
            # _s0_op_counts tracks ALL compiled programs (pass + fail) so it
            # is the single source of truth.  Only fall back to
            # update_op_success_rates (program_results scan) when no in-memory
            # counts exist (e.g. investigation/validation modes).
            s0_op_counts = results.pop("_s0_op_counts", None)
            if s0_op_counts:
                nb.merge_op_failure_counts(s0_op_counts)
            else:
                nb.update_op_success_rates(exp_id)
            nb.strip_graph_json_for_failures(exp_id)
            nb.update_failure_signatures(exp_id)

            # Periodic op rehabilitation: test excluded ops in isolation
            try:
                total_exp = nb.conn.execute(
                    "SELECT COUNT(*) FROM experiments"
                ).fetchone()[0]
                if total_exp % 10 == 0:
                    from ...eval.op_rehab import rehabilitate_ops

                    rehab_results = rehabilitate_ops(nb, model_dim=config.model_dim)
                    if rehab_results:
                        logger.info(
                            "Op rehabilitation passed %d ops: %s",
                            len(rehab_results),
                            ", ".join(rehab_results),
                        )
            except Exception as e:
                logger.warning("Op rehabilitation failed: %s", e)

            # Save effective weights + S1 outcome for EMA continuity
            applied_w = results.get("applied_grammar_weights")
            total = results.get("total", 0)
            if applied_w and total > 0:
                s1_rate = results.get("stage1_passed", 0) / total
                nb.save_effective_weights(applied_w, s1_rate, exp_id)

            # Auto-recommend next experiment
            self._auto_recommend(results, config, hypothesis, nb)

            # Flush async writes so auto-escalate can read back S1 survivors
            nb.flush_writes()
            # Auto-escalation pipeline (investigation/validation)
            results["experiment_id"] = exp_id
            self._auto_escalate(results, config, nb, phase="screening")

            # Auto-scale-up if criteria met (legacy, kept for backward compat)
            self._maybe_auto_scale_up(results, config, nb)

            # Auto-report for single experiments
            self._maybe_auto_report(config, nb, reason="experiment_complete")

            self._update_progress(
                status="completed",
                aria_message=summary.split("\n")[-1]
                if summary
                else "Experiment complete.",
            )

            self._emit_event(
                "experiment_completed",
                {
                    "experiment_id": exp_id,
                    "results": results,
                    "summary": summary,
                },
            )

        except Exception as e:
            error = traceback.format_exc()
            logger.error("Experiment failed (%s): %s\n%s", exp_id, e, error)
            self._invoke_code_healer(
                nb=nb,
                trigger_type="repeated_exception",
                experiment_id=exp_id,
                scope=f"Synthesis/experiment failure: {str(e)[:240]}",
                reproduction_steps=[
                    'python -m pytest tests/test_integration.py -k "start_experiment" -x --tb=short'
                ],
                acceptance_tests=[
                    'python -m pytest tests/test_integration.py -k "start_experiment" -x --tb=short'
                ],
                trigger_payload={"mode": "synthesis", "error": str(e)},
            )
            nb.fail_experiment(exp_id, str(e))
            self._update_progress(
                status="failed",
                error=str(e),
                aria_message=self.aria.react_to_failure(str(e)),
            )
            self._emit_event(
                "experiment_failed",
                {
                    "experiment_id": exp_id,
                    "error": str(e),
                },
            )
        finally:
            nb.close()
            # Launch queued auto-scale-up after notebook is closed
            self._run_pending_scale_up()

    def _execute_experiment(
        self,
        exp_id: str,
        config: RunConfig,
        nb: LabNotebook,
        use_learned_grammar: bool = True,
    ) -> Dict:
        """Core experiment logic shared by single and continuous modes."""
        self._live_training_context = {"exp_id": exp_id, "phase": "synthesis"}
        with self._lock:
            # Z17: Explicitly reset progress object at start of execution
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=config.n_programs,
                aria_message=f"{self.aria.NAME}: Initializing experiment {exp_id[:8]}...",
            )

        results = {
            "total": 0,
            "stage0_passed": 0,
            "stage05_passed": 0,
            "rapid_screening_killed": 0,
            "rapid_screening_kill_reasons": {},
            "stage1_passed": 0,
            "novel_count": 0,
            "best_loss_ratio": None,
            "best_novelty_score": None,
            "survivors": [],
            "skipped_proactive_gating": 0,
            "proactive_gating_failures": [],
            "funnel_counts": {
                "raw_generated": 0,
                "post_batch_dedup": 0,
                "judgment_filtered": 0,
                "post_judgment": 0,
                "screening_considered": 0,
                "dropped_runtime_dedup": 0,
                "dropped_toxic": 0,
                "dropped_proactive_gating": 0,
                "dropped_invalid_graph": 0,
                "dropped_runtime_error": 0,
                "stage0_attempted": 0,
                "stage0_passed": 0,
                "dropped_stage0": 0,
                "stage05_passed": 0,
                "dropped_stage05": 0,
                "dropped_s075_high_init": 0,
                "rapid_screen_attempted": 0,
                "dropped_rapid_screening": 0,
                "stage1_queued": 0,
                "stage1_completed": 0,
                "stage1_survived": 0,
                "persisted_rows": 0,
                "dropped_persistence_quality_gate": 0,
            },
        }

        grammar_weights = None
        op_weights: Dict[str, float] = {}
        failure_blocklist: Dict[str, float] = {}
        champion_bias: Dict[str, float] = {}
        template_weights: Dict[str, float] = {}
        motif_weights: Dict[str, float] = {}
        analytics = None
        grammar_gate: Optional[Dict[str, Any]] = None
        if use_learned_grammar:
            try:
                from ..analytics import ExperimentAnalytics

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
                            evidence=json.dumps(
                                json_safe(grammar_gate), sort_keys=True
                            ),
                        )
                        grammar_weights = None
            except Exception as e:
                logger.warning(
                    "Failed computing learned grammar weights for %s: %s", exp_id, e
                )

            # Soft-penalize poorly-performing ops (no hard exclusion — causality
            # sandbox gate catches truly broken ops at eval time)
            op_weights: Dict[str, float] = {}
            try:
                rehab_cache = nb.get_op_rehabilitation_cache()
                if analytics is not None:
                    neg = analytics.negative_results_synthesis()
                    for op_info in neg.get("failed_ops", []):
                        if (
                            op_info.get("s1_rate", 1) == 0
                            and op_info.get("n_used", 0) >= 5
                            and op_info.get("confidence", 0) >= 0.7
                        ):
                            op_name = op_info["op_name"]
                            rehab = rehab_cache.get(op_name)
                            if (
                                rehab
                                and rehab.get("compile_passed")
                                and rehab.get("forward_passed")
                            ):
                                op_weights[op_name] = 0.5
                            elif op_info.get("failure_stage") == "compilation":
                                op_weights[op_name] = 0.15
                            else:
                                op_weights[op_name] = 0.1
                    for op_info in neg.get("weak_ops", []):
                        op_name = op_info.get("op_name", "")
                        penalty = op_info.get("penalty_weight", 1.0)
                        if op_name:
                            op_weights[op_name] = penalty
                    if op_weights:
                        nb.log_learning_event(
                            "weak_ops_penalized",
                            f"Soft-penalized {len(op_weights)} weak ops: "
                            f"{', '.join(f'{k}={v:.2f}' for k, v in sorted(op_weights.items()))}",
                            op_weights=op_weights,
                        )
            except Exception as e:
                logger.warning("Failed computing op penalties for %s: %s", exp_id, e)

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
                logger.warning(
                    "Failed loading failure signatures for %s: %s", exp_id, e
                )

            # Champion bias pass: nudge category weights toward proven winners.
            # This biases the search toward high-performing projection/sparse patterns
            # and known-good structural/sequence motifs without hard-coding op-level picks.
            try:
                if analytics is not None:
                    # Use 7d windowed rates to avoid death spiral from
                    # stale lifetime data poisoning recently-fixed ops
                    _window_cutoff = time.time() - 604800  # 7 days
                    op_rates = analytics.op_success_rates(since_ts=_window_cutoff) or {}
                    if op_rates:
                        winning_ops = {"exp", "selective_scan", "tropical_center"}
                        projection_ops = {
                            "low_rank_proj",
                            "shared_basis_proj",
                            "tied_proj",
                        }
                        sparse_ops = {
                            "nm_sparse_linear",
                            "block_sparse_linear",
                            "semi_structured_2_4_linear",
                        }

                        def _is_reliable(
                            op_name: str, min_used: int = 10, min_s1: float = 0.25
                        ) -> bool:
                            info = op_rates.get(op_name) or {}
                            n_used = int(info.get("n_used") or 0)
                            s1_rate = float(info.get("s1_rate") or 0.0)
                            return n_used >= min_used and s1_rate >= min_s1

                        has_winners = any(_is_reliable(op) for op in winning_ops)
                        has_projection = any(_is_reliable(op) for op in projection_ops)
                        has_sparse = any(_is_reliable(op) for op in sparse_ops)

                        if has_winners:
                            champion_bias["structural"] = max(
                                champion_bias.get("structural", 1.0), 1.2
                            )
                            champion_bias["sequence"] = max(
                                champion_bias.get("sequence", 1.0), 1.2
                            )
                        if has_projection:
                            champion_bias["parameterized"] = max(
                                champion_bias.get("parameterized", 1.0), 1.4
                            )
                        if has_sparse:
                            champion_bias["parameterized"] = max(
                                champion_bias.get("parameterized", 1.0), 1.5
                            )
                            # Z7: If sparse ops are reliable, nudge the grammar hard toward them
                            champion_bias["_structured_sparsity_bias"] = 0.8

            except Exception as e:
                logger.warning("Failed computing champion bias for %s: %s", exp_id, e)

        try:
            template_weights, motif_weights = _build_signal_weight_maps(nb)
        except Exception:
            template_weights, motif_weights = {}, {}

        op_weights = {**op_weights, **self._op_weights_overrides}
        grammar = self._build_grammar_config(config, op_weights=op_weights)
        # Merge learned template/motif weights, but don't overwrite routing_first preset
        if grammar.routing_mandatory:
            # Routing-first: only merge in weights that don't conflict
            for k, v in template_weights.items():
                grammar.template_weights.setdefault(k, v)
        else:
            grammar.template_weights = template_weights
        grammar.motif_weights = motif_weights
        # Apply Bayesian insight adjustments to grammar config
        try:
            _apply_insight_adjustments(
                nb, grammar, grammar.template_weights, grammar.motif_weights
            )
        except Exception as e:
            logger.debug("Insight grammar adjustment failed: %s", e)
        old_weights = dict(grammar.category_weights)

        if grammar_weights:
            old_weights = dict(grammar.category_weights)
            grammar.category_weights.update(grammar_weights)
            n_changed = sum(
                1
                for key, value in grammar_weights.items()
                if old_weights.get(key) != value
            )
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
            self._emit_event(
                "learning_event",
                {
                    "event_type": "grammar_weights_applied",
                    "experiment_id": exp_id,
                    "n_changed": n_changed,
                    "max_depth": int(config.max_depth),
                    "max_ops": int(config.max_ops),
                    "description": (
                        f"Applied learned grammar weights ({n_changed} categories changed; "
                        f"depth<= {int(config.max_depth)}, ops<= {int(config.max_ops)})"
                    ),
                },
            )

        if champion_bias:
            before_bias = dict(grammar.category_weights)
            for category, multiplier in champion_bias.items():
                if category == "_structured_sparsity_bias":
                    grammar.structured_sparsity_bias = float(multiplier)
                    continue
                base = float(grammar.category_weights.get(category, 1.0))
                grammar.category_weights[category] = round(
                    max(0.5, min(8.0, base * multiplier)), 2
                )
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
        else:
            grammar.category_weights["math_space"] = config.math_space_weight

        # Efficiency bias: boost categories that produce compact/efficient architectures
        # Targets sparse, low-rank, MoE, and state-space ops per frontier micronization memo
        _eff_weight = getattr(config, "selection_efficiency_weight", 0.25)
        if _eff_weight >= 0.3:  # only apply when efficiency is prioritized
            _eff_boost = min(1.0 + _eff_weight, 2.0)  # 1.3-2.0x
            for _cat in ("structural", "parameterized"):
                _base = float(grammar.category_weights.get(_cat, 1.0))
                grammar.category_weights[_cat] = round(min(8.0, _base * _eff_boost), 2)
            # Boost specific efficiency-related ops
            for _op in (
                "moe_2expert",
                "moe_topk",
                "block_sparse_linear",
                "bottleneck_proj",
                "linear_proj_down",
                "selective_scan",
            ):
                grammar.op_weights[_op] = grammar.op_weights.get(_op, 1.0) * _eff_boost

        # Hyperbolic promotion: query recent hierarchy fitness from fingerprints
        if analytics is not None:
            try:
                hf = analytics.recent_hierarchy_fitness()
                if hf is not None:
                    grammar._hierarchy_fitness = hf
                    if hf > grammar.hyperbolic_promotion_threshold:
                        logger.info(
                            "Hierarchy detected (fitness=%.3f > %.2f): boosting hyperbolic ops",
                            hf,
                            grammar.hyperbolic_promotion_threshold,
                        )
            except Exception:
                pass

        # Synthesized loss/optimizer exploration (20% of screening experiments)
        if random.random() < 0.2:
            config.loss_type = "synthesized"
        if random.random() < 0.2:
            config.optimizer_type = "synthesized"

        t_start = time.time()

        # Generate graphs
        if config.model_source == "morphological_box":
            self._run_morphological_screening(exp_id, config, nb, results, t_start)
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
                        from ...synthesis.grammar import EfficiencyPrior

                        prior = EfficiencyPrior(frontier)
                        use_adaptive = True
                        nb.log_learning_event(
                            "adaptive_synthesis_enabled",
                            f"Enabling budget-aware adaptive synthesis for {exp_id}",
                            frontier_size=len(frontier),
                        )
                except Exception as e:
                    logger.warning("Failed to initialize efficiency prior: %s", e)

            _bg_result = batch_generate(
                config.n_programs,
                grammar,
                use_adaptive_synthesis=use_adaptive,
                prior=prior,
            )
            graphs = _bg_result.graphs
            results["batch_generate_stats"] = {
                "n_attempted": _bg_result.n_attempted,
                "n_rejected_grammar": _bg_result.n_rejected_grammar,
                "n_rejected_dedup": _bg_result.n_rejected_dedup,
            }
        results["funnel_counts"]["raw_generated"] = len(graphs)
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
                    evidence=json.dumps(json_safe(shift), sort_keys=True),
                )
            else:
                nb.log_learning_event(
                    "architecture_distribution_snapshot",
                    f"Captured generated-op distribution for synthesis experiment {exp_id}",
                    evidence=json.dumps(
                        json_safe({"op_distribution": op_distribution}), sort_keys=True
                    ),
                )

        self._log_generated_graph_observation(nb, exp_id, graphs, grammar, config)
        dev, dev_str, orchestrator, candidate_batch_size = (
            self._prepare_screening_orchestrator(config, results)
        )
        graphs, _existing_fps = self._dedup_graph_candidates(
            nb=nb,
            graphs=graphs,
            grammar=grammar,
            config=config,
            exp_id=exp_id,
            results=results,
        )
        results["funnel_counts"]["post_batch_dedup"] = len(graphs)
        # judgment_scores maps graph id(graph) → score for persistence
        _judgment_scores: Dict[int, float] = {}
        if graphs:
            before_judgment = len(graphs)
            graphs = rank_synthesis_candidates_by_stability(graphs)
            results["stability_reranked"] = True
            ranked = _judgment_rerank(graphs, nb, logger)
            if len(ranked) != before_judgment:
                results["judgment_filtered"] = before_judgment - len(ranked)
            _judgment_scores = {id(g): s for g, s in ranked}
            graphs = [g for g, _ in ranked]
            results["funnel_counts"]["judgment_filtered"] = max(
                0, before_judgment - len(graphs)
            )
        results["funnel_counts"]["post_judgment"] = len(graphs)
        self._update_progress(total_programs=len(graphs))

        # Track ops from S0 failures for op_success_rates (not stored in DB)
        _s0_op_counts: Dict[str, Dict[str, int]] = {}  # op -> {n_used, n_s0, n_s05}

        for i, graph in enumerate(graphs):
            if self._stop_event.is_set():
                break

            results["funnel_counts"]["screening_considered"] += 1
            fp = graph.fingerprint()
            self._update_progress(
                current_program=i + 1,
                current_fingerprint=fp[:10],
                elapsed_seconds=time.time() - t_start,
            )

            # Real-time dedup: skip if evaluated by another process since experiment start
            if nb.has_fingerprint(fp):
                results.setdefault("skipped_dedup_runtime", 0)
                results["skipped_dedup_runtime"] += 1
                results["funnel_counts"]["dropped_runtime_dedup"] += 1
                self._emit_event(
                    "program_evaluated",
                    {
                        "index": i,
                        "fingerprint": fp[:10],
                        "result": "skipped_dedup",
                    },
                )
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
                    toxic_weight = sum(
                        1.0 - failure_blocklist[bg]
                        for bg in bigrams
                        if bg in failure_blocklist
                    )
                    toxic_ratio = toxic_weight / len(bigrams)
                    if toxic_ratio >= 0.5:
                        results.setdefault("skipped_toxic", 0)
                        results["skipped_toxic"] += 1
                        results["funnel_counts"]["dropped_toxic"] += 1
                        self._emit_event(
                            "program_evaluated",
                            {
                                "index": i,
                                "fingerprint": graph.fingerprint()[:10],
                                "result": "skipped_toxic",
                                "toxic_ratio": f"{toxic_ratio:.2f}",
                            },
                        )
                        continue

            # Collect all metrics for this program
            program_metrics: Dict[str, Any] = {}
            program_metrics.update(self._extract_graph_metrics(graph))
            # Persist judgment score for promotion decisions
            j_score = _judgment_scores.get(id(graph))
            if j_score is not None:
                program_metrics["judgment_score"] = j_score

            # Estimate FLOPs
            try:
                flop_est = estimate_flops(
                    graph,
                    seq_len=min(128, config.max_seq_len),
                    d_model=config.model_dim,
                )
                program_metrics["flops_forward"] = flop_est.flops_forward
                program_metrics["flops_per_param"] = flop_est.flops_per_param
                program_metrics["flops_per_token"] = flop_est.flops_per_token
            except Exception as e:
                logger.debug(
                    "FLOP estimate failed for %s: %s", graph.fingerprint()[:10], e
                )

            # Native Proactive Gating (Project Hephaestus)
            # High-performance stability and toxic motif detection
            try:
                native_gating = _native_proactive_gating(graph)
                if not native_gating.get("passed", True):
                    results.setdefault("skipped_proactive_gating", 0)
                    results["skipped_proactive_gating"] += 1
                    results["funnel_counts"]["dropped_proactive_gating"] += 1

                    # Update metrics with native data
                    program_metrics["proactive_gating_reason"] = native_gating.get(
                        "reason"
                    )
                    program_metrics["max_depth"] = native_gating.get("max_depth")
                    program_metrics["n_toxic_motifs"] = native_gating.get(
                        "n_toxic_motifs"
                    )

                    self._emit_event(
                        "program_evaluated",
                        {
                            "index": i,
                            "fingerprint": fp[:10],
                            "result": "skipped_proactive",
                            "reason": native_gating.get("reason"),
                            "max_depth": native_gating.get("max_depth"),
                        },
                    )
                    continue
            except Exception as e:
                logger.debug("Native proactive gating failed for %s: %s", fp[:10], e)

            # Validate
            validation = validate_graph(
                graph,
                max_ops=max(1, int(config.max_ops)),
                max_depth=max(1, int(config.max_depth)),
                min_splits=config.min_splits,
            )
            if not validation.valid:
                # Graph-level validation failures (depth, shape, structure) are
                # NOT op-level failures.  Counting individual ops here inflates
                # n_used without incrementing n_s0, making ubiquitous ops like
                # linear_proj appear broken in the health dashboard.  Track the
                # aggregate count for grammar tuning but don't blame individual
                # ops for structural violations they didn't cause.
                results.setdefault("s0_validation_failures", 0)
                results["s0_validation_failures"] += 1
                results["funnel_counts"]["dropped_invalid_graph"] += 1
                self._emit_event(
                    "program_evaluated",
                    {
                        "index": i,
                        "fingerprint": fp[:10],
                        "result": "invalid",
                        "error": validation.errors[0] if validation.errors else "",
                    },
                )
                continue

            # Compile & Stage 0/0.5
            # Progressive screening: Phase 1 uses cheap qualifying vocab (32K)
            # for S0/S0.5/rapid.  Only Phase 1 survivors get recompiled at
            # the real vocab for S1 training.  This filters ~93% of candidates
            # at ~10% of the cost.
            _use_progressive = (
                config.progressive_screening
                and config.vocab_size > config.qualifying_vocab_size
            )
            _phase1_vocab = (
                config.qualifying_vocab_size if _use_progressive else config.vocab_size
            )

            try:
                results["funnel_counts"]["stage0_attempted"] += 1
                # Z13: Defensive pause + GC to stabilize Torch Dynamo context if needed
                if i > 0 and i % 10 == 0:
                    clear_gpu_memory()

                    # More aggressive reset every 50 to clear Torch Dynamo cache
                    if i % 50 == 0:
                        try:
                            torch.compiler.reset()
                        except (AttributeError, Exception):
                            pass

                    time.sleep(0.1)

                layer_graphs = [graph] * config.n_layers
                model = compile_model(
                    layer_graphs,
                    vocab_size=_phase1_vocab,
                    max_seq_len=config.max_seq_len,
                )
                _eval_timeout = 60 if getattr(config, "_exotic_mode", False) else 30
                sandbox_result = self._safe_eval_for_stage(
                    model,
                    stage_tag="candidate_screening",
                    batch_size=2,
                    seq_len=min(128, config.max_seq_len),
                    vocab_size=_phase1_vocab,
                    device=dev_str,
                    timeout_seconds=_eval_timeout,
                )
                program_metrics.update(self._extract_sandbox_metrics(sandbox_result))
                program_metrics["param_count"] = sandbox_result.param_count

                s0_passed = sandbox_result.passed
                s05_passed = (
                    sandbox_result.stability_score >= config.stage05_stability_threshold
                    and sandbox_result.causality_passed
                )

                if s0_passed:
                    results["stage0_passed"] += 1
                    results["funnel_counts"]["stage0_passed"] += 1
                    with self._lock:
                        self._progress.stage0_passed += 1
                if s05_passed:
                    results["stage05_passed"] += 1
                    results["funnel_counts"]["stage05_passed"] += 1
                    with self._lock:
                        self._progress.stage05_passed += 1

                # Track ALL compiled programs in _s0_op_counts so
                # merge_op_failure_counts sees both passes and failures.
                # Without this, ops that only appear in failing architectures
                # accumulate n_used but never n_stage0_passed, making them
                # look broken even when the op itself works fine.
                for node in graph.nodes.values():
                    if not node.is_input and node.op_name:
                        c = _s0_op_counts.setdefault(
                            node.op_name, {"n_used": 0, "n_s0": 0, "n_s05": 0}
                        )
                        c["n_used"] += 1
                        if s0_passed:
                            c["n_s0"] += 1
                        if s05_passed:
                            c["n_s05"] += 1

                if not s0_passed or not s05_passed:
                    if not s0_passed:
                        results["funnel_counts"]["dropped_stage0"] += 1
                    else:
                        results["funnel_counts"]["dropped_stage05"] += 1
                    # Don't store S0/S0.5 failures — error counts are tracked
                    # in results dict and error_type in the live feed event.
                    error_type = sandbox_result.error_type or "unknown"
                    results.setdefault("failure_error_types", {})
                    results["failure_error_types"][error_type] = (
                        results["failure_error_types"].get(error_type, 0) + 1
                    )
                    self._emit_event(
                        "program_evaluated",
                        {
                            "index": i,
                            "fingerprint": fp[:10],
                            "result": "fail_s0" if not s0_passed else "fail_s05",
                            "error": (sandbox_result.error or "")[:120]
                            if not s0_passed
                            else None,
                            "error_type": error_type,
                            "stability": f"{sandbox_result.stability_score:.2f}"
                            if s0_passed and not s05_passed
                            else None,
                            "params": sandbox_result.param_count
                            if sandbox_result.param_count
                            else None,
                            "memory_mb": f"{sandbox_result.peak_memory_mb:.1f}"
                            if sandbox_result.peak_memory_mb
                            else None,
                            "has_nan": sandbox_result.has_nan_output
                            or sandbox_result.has_nan_grad
                            or None,
                            "has_inf": sandbox_result.has_inf_output or None,
                        },
                    )
                    continue

                # S0.75: Initial-loss pre-screen (5 gradient steps)
                # Architectures with init_loss > 50 have deep unscaled
                # projection chains and waste rapid-screen + S1 budget.
                try:
                    _s075_dev = torch.device(dev_str)
                    model.train()
                    _s075_opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
                    _s075_ids = torch.randint(
                        0, _phase1_vocab, (4, 64), device=_s075_dev
                    )
                    with torch.amp.autocast(
                        device_type=_s075_dev.type,
                        dtype=torch.bfloat16,
                        enabled=(_s075_dev.type == "cuda"),
                    ):
                        _s075_logits = model(_s075_ids)
                        _s075_loss = torch.nn.functional.cross_entropy(
                            _s075_logits[:, :-1].reshape(-1, _s075_logits.size(-1)),
                            _s075_ids[:, 1:].reshape(-1),
                        )
                    _s075_init_loss = _s075_loss.item()
                    program_metrics["s075_initial_loss"] = _s075_init_loss

                    if (
                        not math.isnan(_s075_init_loss)
                        and not math.isinf(_s075_init_loss)
                        and _s075_init_loss > INITIAL_LOSS_THRESHOLD
                    ):
                        results["funnel_counts"]["dropped_s075_high_init"] += 1
                        self._emit_event(
                            "program_evaluated",
                            {
                                "index": i,
                                "fingerprint": fp[:10],
                                "result": "fail_s075",
                                "initial_loss": round(_s075_init_loss, 1),
                                "threshold": INITIAL_LOSS_THRESHOLD,
                            },
                        )
                        del _s075_opt
                        continue

                    # Clean up probe state before rapid screening
                    _s075_opt.zero_grad(set_to_none=True)
                    del _s075_opt
                except Exception as s075_err:
                    logger.warning(
                        "S0.75 probe failed for graph %d, skipping check: %s",
                        i,
                        s075_err,
                    )

                # Rapid Screening: 150-step gradient health check
                # Catches exploding grads, NaN, stalled loss, routing collapse
                # BEFORE committing to full Stage 1 training budget.
                from ...eval.screening_rapid import RapidScreeningCheck

                rapid = RapidScreeningCheck()
                results["funnel_counts"]["rapid_screen_attempted"] += 1
                rapid_result = rapid.run(
                    model,
                    vocab_size=_phase1_vocab,
                    seq_len=min(128, config.max_seq_len),
                    batch_size=2,
                    device=dev_str,
                )
                program_metrics["rapid_screening_passed"] = rapid_result.passed
                program_metrics["rapid_screening_elapsed_ms"] = rapid_result.elapsed_ms
                if rapid_result.degraded:
                    program_metrics["rapid_screening_degraded"] = True
                    program_metrics["rapid_screening_degraded_reasons"] = (
                        rapid_result.degraded_reasons
                    )
                if not rapid_result.passed:
                    results["rapid_screening_killed"] += 1
                    results["funnel_counts"]["dropped_rapid_screening"] += 1
                    kr = rapid_result.kill_reason or "unknown"
                    results["rapid_screening_kill_reasons"][kr] = (
                        results["rapid_screening_kill_reasons"].get(kr, 0) + 1
                    )
                    program_metrics["rapid_screening_kill_reason"] = (
                        rapid_result.kill_reason
                    )
                    program_metrics["rapid_screening_kill_step"] = (
                        rapid_result.kill_step
                    )
                    program_metrics["rapid_screening_kill_metric"] = (
                        rapid_result.kill_metric
                    )
                    self._emit_event(
                        "program_evaluated",
                        {
                            "index": i,
                            "fingerprint": fp[:10],
                            "result": "fail_rapid_screening",
                            "kill_reason": rapid_result.kill_reason,
                            "kill_step": rapid_result.kill_step,
                            "gpu_minutes_saved": rapid_result.gpu_minutes_saved,
                        },
                    )
                    continue

                # Phase 2: recompile at real vocab for S1 training
                if _use_progressive:
                    del model
                    clear_gpu_memory()
                    model = compile_model(
                        layer_graphs,
                        vocab_size=config.vocab_size,
                        max_seq_len=config.max_seq_len,
                    )
                    program_metrics["progressive_phase2_compiled"] = True

                # Stage 1: Asynchronous Execution (Z6)
                self._update_progress(current_stage="queuing_s1")

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
                    model=model,  # Reuse compiled model (at real vocab)
                )
                results["funnel_counts"]["stage1_queued"] += 1

            except Exception as e:
                logger.error("Error evaluating graph %d: %s", i, e)
                results["funnel_counts"]["dropped_runtime_error"] += 1
                # Reset CUDA context if this was a fatal CUDA error
                if torch.cuda.is_available():
                    from ...eval.sandbox import is_cuda_fatal

                    if is_cuda_fatal(e):
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                            _probe = torch.zeros(1, device="cuda")
                            del _probe
                            torch.cuda.synchronize()
                            logger.info(
                                "CUDA context recovered after fatal error on graph %d",
                                i,
                            )
                        except Exception:
                            logger.warning(
                                "CUDA context unrecoverable after fatal error on graph %d",
                                i,
                            )
                continue

            # Periodically process available results to keep the dashboard updated
            self._process_orchestrator_results(
                orchestrator, nb, exp_id, results, config
            )

        # Wait for remaining asynchronous Stage 1 evaluations
        self._update_progress(status="finalizing_evaluations")

        while (
            orchestrator.job_queue.unfinished_tasks > 0
            or not orchestrator.result_queue.empty()
        ):
            if self._stop_event.is_set():
                break
            self._process_orchestrator_results(
                orchestrator, nb, exp_id, results, config
            )
            time.sleep(0.5)

        queue_telemetry = orchestrator.get_telemetry()
        orchestrator.shutdown()
        results["queue_telemetry"] = queue_telemetry
        results["elapsed_seconds"] = time.time() - t_start
        results["perf_report"] = self._build_experiment_perf_report(
            results, queue_telemetry=queue_telemetry
        )
        results["perf_budget_gate"] = evaluate_perf_budget_gate(results["perf_report"])
        results.pop("_perf_traces", None)
        results.pop("_gpu_starvation", None)
        results.pop("_kernel_timing", None)
        if _s0_op_counts:
            results["_s0_op_counts"] = _s0_op_counts

        elapsed = results.get("elapsed_seconds", time.time() - t_start)
        self._update_progress(
            elapsed_seconds=elapsed,
            status="analyzing",
            aria_message=self.aria.begin_analysis(),
        )

        best = results.get("best_loss_ratio")
        best_str = f", best loss={best:.4f}" if best else ""
        dedup_str = ""
        if results.get("skipped_dedup", 0) > 0:
            dedup_str = f", dedup={results['skipped_dedup']} ({results.get('dedup_rate', 0) * 100:.0f}%)"
        rapid_killed = results.get("rapid_screening_killed", 0)
        rapid_str = f", rapid_killed={rapid_killed}" if rapid_killed else ""
        logger.info(
            "Experiment %s complete: %d programs → S0=%d → S0.5=%d → S1=%d "
            "(%.1fs)%s%s%s%s",
            exp_id[:8],
            results["total"],
            results["stage0_passed"],
            results["stage05_passed"],
            results["stage1_passed"],
            elapsed,
            best_str,
            dedup_str,
            rapid_str,
            f", native_gating={results.get('skipped_proactive_gating', 0)}"
            if results.get("skipped_proactive_gating")
            else "",
        )

        self._live_training_context = None
        return results
