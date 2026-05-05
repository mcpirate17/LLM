"""Dashboard mixin: hypothesis framing, analytics gathering, fitness eval.

Owns the code that turns notebook state into the context that feeds LLM
hypothesis generation and evolution-based synthesis."""

from __future__ import annotations

import logging
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..native_runner import compile_model_native_first as compile_model
from ..notebook import LabNotebook
from ..llm.context_experiment import build_rich_context
from ..shared_utils import resolve_device
from ._helpers import clear_gpu_memory
from ._types import RunConfig

logger = logging.getLogger(__name__)

_REPORT_DIR = Path("research/reports")
_META_QUEUE_GLOB = "meta_experiment_queue_*.json"
_META_PROFILE_GLOB = "meta_profile_ml_analysis_*.json"


class _DashboardHypothesisMixin:
    """Hypothesis framing, analytics gathering, fitness eval."""

    def _build_rich_context_for_experiment(
        self,
        results: Dict,
        config: RunConfig,
        hypothesis: str,
        nb: LabNotebook,
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

    # ── Next-experiment summary ──────────────────────────────────────────

    @staticmethod
    def _summary_top_performers(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "result_id": row.get("result_id"),
                "fingerprint": str(row.get("graph_fingerprint") or "")[:16],
                "loss_ratio": row.get("loss_ratio"),
                "novelty_score": row.get("novelty_score"),
                "throughput_tok_s": row.get("throughput_tok_s"),
                "avg_step_time_ms": row.get("avg_step_time_ms"),
                "stability_score": row.get("stability_score"),
            }
            for row in rows[:5]
        ]

    @staticmethod
    def _summary_aggregate_stats(
        stage1_rows: List[Dict[str, Any]],
    ) -> Dict[str, Optional[float]]:
        eff_rows = [
            r
            for r in stage1_rows
            if isinstance(r.get("throughput_tok_s"), (int, float))
        ]
        stab_rows = [
            r for r in stage1_rows if isinstance(r.get("stability_score"), (int, float))
        ]
        novelty_vals = [
            float(r.get("novelty_score"))
            for r in stage1_rows
            if isinstance(r.get("novelty_score"), (int, float))
        ]
        best_loss = min(
            (
                float(r.get("loss_ratio"))
                for r in stage1_rows
                if isinstance(r.get("loss_ratio"), (int, float))
            ),
            default=None,
        )
        return {
            "avg_throughput_tok_s": (
                sum(float(r.get("throughput_tok_s")) for r in eff_rows) / len(eff_rows)
            )
            if eff_rows
            else None,
            "avg_stability_score": (
                sum(float(r.get("stability_score")) for r in stab_rows) / len(stab_rows)
            )
            if stab_rows
            else None,
            "best_novelty": max(novelty_vals) if novelty_vals else None,
            "avg_novelty": (sum(novelty_vals) / len(novelty_vals))
            if novelty_vals
            else None,
            "best_loss_ratio": best_loss,
        }

    @staticmethod
    def _latest_json_report(
        report_dir: Path,
        pattern: str,
    ) -> tuple[Path | None, Dict[str, Any]]:
        candidates = sorted(
            (path for path in report_dir.glob(pattern) if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.debug(
                    "Skipping unreadable meta strategy report %s: %s", path, exc
                )
                continue
            if isinstance(payload, dict):
                return path, payload
        return None, {}

    @classmethod
    def _load_meta_profile_strategy_brief(
        cls,
        report_dir: str | os.PathLike[str] = _REPORT_DIR,
    ) -> Dict[str, Any]:
        """Compact, read-only profile/queue brief for LLM and fallback strategy."""

        root = Path(report_dir)
        queue_path, queue_payload = cls._latest_json_report(root, _META_QUEUE_GLOB)
        ml_path, ml_payload = cls._latest_json_report(root, _META_PROFILE_GLOB)
        if not queue_payload and not ml_payload:
            return {"active": False, "reason": "no_meta_profile_reports"}

        profile_queue = list(queue_payload.get("profile_refresh_queue") or [])
        compression_queue = list(queue_payload.get("compression_safety_queue") or [])
        scaffoldable = [
            row for row in profile_queue if row.get("recommended_scaffold_family")
        ]
        harness_needed = [
            row
            for row in profile_queue
            if row.get("action") == "add_scaffold_family_or_component_profile_harness"
        ]
        top_ops = [str(row.get("op_name") or "") for row in scaffoldable[:6]]
        top_missing_harness_ops = [
            str(row.get("op_name") or "") for row in harness_needed[:5]
        ]
        top_compression_risks = [
            {
                "template_name": row.get("template_name"),
                "selected_motif": row.get("selected_motif"),
                "nano_bind_rate": row.get("nano_bind_rate"),
                "mean_frequency_risk": row.get("mean_frequency_risk"),
                "recommended_variant": row.get("recommended_variant"),
            }
            for row in compression_queue[:5]
        ]
        ml_summary = dict(ml_payload.get("summary") or {})
        ml_recommendations = [
            {
                "target": row.get("target"),
                "feature": row.get("feature"),
                "evidence": row.get("evidence"),
                "recommendation": row.get("recommendation"),
            }
            for row in list(ml_payload.get("recommendations") or [])[:8]
        ]

        op_weights = {op: 1.6 for op in top_ops[:4] if op}
        for op in top_missing_harness_ops:
            # Keep unsupported complex blocks from being over-promoted until
            # they have scaffold/profile coverage.
            if op:
                op_weights.setdefault(op, 0.55)

        return {
            "active": True,
            "source_reports": {
                "queue": str(queue_path) if queue_path else "",
                "ml": str(ml_path) if ml_path else "",
            },
            "recommended_next_mode": "synthesis",
            "strategy_bias": "profile_refresh_guided_routing_compression_synthesis",
            "rationale": (
                "Recent profile-grounded analysis points to routing/compression "
                "coverage gaps and NanoBind-sensitive compression motifs. Bias the "
                "next cycle toward scaffoldable routing/compression candidates while "
                "preserving positional/content mixers after compression."
            ),
            "config_bias": {
                "n_programs": 80,
                "max_ops": 14,
                "max_depth": 10,
                "model_source": "mixed",
                "morph_focus_sparse": True,
                "grammar_merge_prob": 0.18,
                "grammar_freq_domain_prob": 0.35,
                "category_weights": {
                    "functional": 1.35,
                    "frequency": 1.25,
                    "linear_algebra": 1.2,
                    "structural": 1.1,
                },
                "op_weights": op_weights,
            },
            "top_profile_refresh_ops": top_ops,
            "needs_scaffold_harness_ops": top_missing_harness_ops,
            "top_compression_safety_items": top_compression_risks,
            "ml_summary": {
                "n_graphs": ml_summary.get("n_graphs"),
                "n_features": ml_summary.get("n_features"),
                "target_nano_bind_failure_rate": ml_summary.get(
                    "target_nano_bind_failure_rate"
                ),
                "target_routing_improved_rate": ml_summary.get(
                    "target_routing_improved_rate"
                ),
            },
            "ml_recommendations": ml_recommendations,
            "guardrails": {
                "no_hard_gates": True,
                "do_not_promote_unsupported_harness_ops": top_missing_harness_ops,
                "validate_compression_with": [
                    "NanoBind",
                    "controlled_lang_s05",
                    "WikiText",
                    "TinyStories",
                ],
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
            for row in nb.get_program_results(recent_exp_id, limit=300):
                stage = str(row.get("stage_at_death") or "unknown")
                fail_counts[stage] = fail_counts.get(stage, 0) + 1
                if row.get("stage1_passed"):
                    stage1_rows.append(row)
        stage1_rows.sort(
            key=lambda r: (
                float(r.get("loss_ratio") if r.get("loss_ratio") is not None else 1.0),
                -float(
                    r.get("novelty_score")
                    if r.get("novelty_score") is not None
                    else 0.0
                ),
            )
        )

        stats = self._summary_aggregate_stats(stage1_rows)
        return {
            "recent_experiment_id": recent_exp_id or None,
            "funnel": {
                "total": int(results.get("total") or 0),
                "stage0_passed": int(results.get("stage0_passed") or 0),
                "stage05_passed": int(results.get("stage05_passed") or 0),
                "stage1_passed": int(results.get("stage1_passed") or 0),
            },
            "stage1_survivors": int(len(stage1_rows)),
            "best_loss_ratio": stats["best_loss_ratio"],
            "best_novelty": stats["best_novelty"],
            "avg_novelty": stats["avg_novelty"],
            "avg_throughput_tok_s": stats["avg_throughput_tok_s"],
            "avg_stability_score": stats["avg_stability_score"],
            "top_performers": self._summary_top_performers(stage1_rows),
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
            "meta_profile_strategy": self._load_meta_profile_strategy_brief(),
        }

    # ── Analytics + designer telemetry ───────────────────────────────────

    def _gather_analytics_data(self, nb: LabNotebook) -> Dict:
        """Gather all analytics data for rich context."""
        try:
            from ..analytics import ExperimentAnalytics

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
                "hierarchy_fitness": analytics.recent_hierarchy_fitness(),
            }
        except (
            Exception
        ) as e:  # top-level error boundary: analytics must not crash experiment loop
            logger.debug("_gather_analytics_data failed: %s", e)
            return {}

    def _gather_designer_telemetry(self) -> Dict:
        """Fetch telemetry from aria_designer if available."""
        import requests
        from research.defaults import DESIGNER_API_BASE

        base = os.environ.get("ARIA_DESIGNER_PROXY_BASE", DESIGNER_API_BASE)
        result: Dict = {}
        try:
            r = requests.get(f"{base}/api/v1/integration/bridge-gap-report", timeout=3)
            if r.ok:
                result["bridge_gap_report"] = r.json()
        except (OSError, ValueError) as e:
            logger.debug("Designer bridge-gap-report fetch failed: %s", e)
        try:
            r = requests.get(
                f"{base}/api/v1/blocks/builtin", params={"model_dim": 256}, timeout=3
            )
            if r.ok:
                blocks = r.json()
                result["builtin_blocks"] = [
                    b.get("name")
                    for b in blocks
                    if isinstance(b, dict) and b.get("name")
                ]
        except (OSError, ValueError) as e:
            logger.debug("Designer builtin-blocks fetch failed: %s", e)
        return result

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
            past.append(
                {
                    "hypothesis": hyp,
                    "confirmed": s1_count > 0,
                    "s1_count": s1_count,
                    "best_novelty": best_novelty or 0,
                    "experiment_id": exp.get("experiment_id"),
                }
            )
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
                past.append(
                    {
                        "hypothesis": content,
                        "confirmed": False,
                        "s1_count": 0,
                        "best_novelty": 0,
                        "experiment_id": None,
                        "source": "refuted_insight",
                        "confidence": ins.get("confidence", 0),
                        "evidence": ins.get("supporting_evidence", ""),
                    }
                )
                seen_texts.add(content[:80].lower())
        except (OSError, RuntimeError) as e:
            logger.debug(
                "Refuted insights fetch failed: %s", e
            )  # table may not exist in older notebooks

        return past

    # ── Fitness function factory (evolution/novelty search) ──────────────

    def _fitness_from_s1(self, s1_result, sandbox_result, graph, config) -> float:
        """Compute fitness from a completed s1 micro-train result."""
        fl = s1_result.get("final_loss")
        lr = s1_result.get("loss_ratio")
        if fl is not None and lr is not None and lr < 0.95:
            fitness, _ = self._compute_multi_objective_fitness(
                s1_result, sandbox_result, graph, config
            )
            return fitness
        if s1_result.get("passed"):
            fitness, _ = self._compute_multi_objective_fitness(
                s1_result, sandbox_result, graph, config
            )
            return fitness
        return 0.1

    def _make_fitness_fn(
        self, config: RunConfig, *, on_evaluate=None, fitness_cache=None
    ):
        """Create fitness function for evolution/novelty search.

        Args:
            config: Run configuration.
            on_evaluate: Optional callback ``(graph, fitness, sandbox_result, s1_result)``
                fired after every real evaluation (not cache hits).
            fitness_cache: Optional ``Dict[str, float]`` mapping graph fingerprint
                to fitness.  Cache hits skip compilation entirely.
        """
        dev = resolve_device(config.device)
        dev_str = str(dev)

        # Progressive screening: qualify at cheap vocab first
        use_progressive = (
            config.progressive_screening
            and config.vocab_size > config.qualifying_vocab_size
        )
        phase1_vocab = (
            config.qualifying_vocab_size if use_progressive else config.vocab_size
        )

        def fitness_fn(graph):
            fp = graph.fingerprint()
            if fitness_cache is not None and fp in fitness_cache:
                return fitness_cache[fp]

            sandbox_result = None
            s1_result = None
            try:
                layer_graphs = [graph] * config.n_layers
                model = compile_model(
                    layer_graphs,
                    vocab_size=phase1_vocab,
                    max_seq_len=config.max_seq_len,
                )
                sandbox_result = self._safe_eval_for_stage(
                    model,
                    stage_tag="evolution_fitness",
                    batch_size=2,
                    seq_len=min(128, config.max_seq_len),
                    vocab_size=phase1_vocab,
                    device=dev_str,
                )
                if not sandbox_result.passed:
                    del model
                    fitness = 0.0
                else:
                    if use_progressive:
                        del model
                        clear_gpu_memory()
                        model = compile_model(
                            layer_graphs,
                            vocab_size=config.vocab_size,
                            max_seq_len=config.max_seq_len,
                        )
                    s1_result = self._micro_train(
                        model,
                        config,
                        dev,
                        seed=self._stable_seed("fitness", fp),
                    )
                    del model
                    clear_gpu_memory()
                    fitness = self._fitness_from_s1(
                        s1_result, sandbox_result, graph, config
                    )
            except (
                Exception
            ) as e:  # top-level error boundary: fitness eval must not crash evolution
                logger.debug("Fitness evaluation failed for %s: %s", fp[:10], e)
                fitness = 0.0

            if fitness_cache is not None:
                fitness_cache[fp] = fitness
            if on_evaluate:
                on_evaluate(graph, fitness, sandbox_result, s1_result)
            return fitness

        return fitness_fn

    def _maybe_spawn_agent_from_reasoning(self, reasoning: str, nb: LabNotebook):
        """If Aria's reasoning mentions code issues, spawn a repair agent."""
        import re as _re

        # Detect code-issue signals in reasoning text
        code_issue_patterns = [
            r"\b(?:error|bug|crash|exception|traceback|broken|fails?|failing)\b.*\b(?:in|at|from)\s+\S+\.py\b",
            r"\b(?:fix|repair|patch|update)\b.*\b(?:code|file|module|function|class)\b",
            r"\bImportError\b|\bTypeError\b|\bAttributeError\b|\bNameError\b|\bSyntaxError\b",
            r"\b(?:missing|undefined|unresolved)\s+(?:import|module|function|method|attribute)\b",
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
            from ..code_agent import _spawn_code_agent_task

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
            self._log_learning_event_compat(
                nb,
                "proactive_reasoning_agent",
                f"Spawned agent {task_id} from recommendation reasoning: {reasoning[:200]}",
                task_id=task_id,
            )
            logger.info("Spawned reasoning-based repair agent: %s", task_id)
        except (ImportError, RuntimeError, OSError) as e:
            logger.debug("Failed to spawn reasoning-based agent: %s", e)
