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
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch

from ..native_runner import compile_model_native_first as compile_model
from ...synthesis.serializer import graph_to_json
from ._helpers import clear_gpu_memory
from ...eval.metrics import novelty_score
from ...eval.fingerprint import BehavioralFingerprint
from ...eval.diagnostic_tasks import run_diagnostic_suite
from ...eval.perf_budget import evaluate_perf_budget_gate
from ...perf_contract import (
    build_duplicate_work_report,
    build_perf_contract,
    emit_perf_artifact,
)
from ..json_utils import json_safe
from ..notebook import LabNotebook
from ..llm.context_experiment import build_rich_context
from ..shared_utils import resolve_device

import logging

logger = logging.getLogger(__name__)

from ._types import RunConfig


class _DashboardMixin:
    """Dashboard data, live loss curve, telemetry."""

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

    def get_live_loss_curve(self) -> List[Dict]:
        """Return the in-memory training loss curve for the current/last training run."""
        return list(self._live_loss_curve)

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
        from ...morphological_box import roll
        from ...arch_builder import build_model, BuildConfig

        requested_modes = modes or list(self._ROUTING_BENCHMARK_MODES)
        supported_modes = [
            m for m in requested_modes if m in self._ROUTING_BENCHMARK_MODES
        ]
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

        bench_config = config.copy()
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
                    n_steps = int(
                        train_result.get("n_train_steps") or bench_config.stage1_steps
                    )
                    batch_size = int(bench_config.stage1_batch_size)
                    tokens_total = batch_size * seq_len * n_steps
                    eff_factor = float(
                        self._ROUTING_EFFICIENCY_FACTOR.get(routing_mode, 1.0)
                    )

                    run_data.update(
                        {
                            "validation_loss": train_result.get("final_loss"),
                            "tokens_per_sec": train_result.get("throughput"),
                            "routing_stability": self._routing_stability_from_curve(
                                train_result.get("training_curve") or []
                            ),
                            "tokens_total": tokens_total,
                            "effective_token_compute": tokens_total * eff_factor,
                            "loss_ratio": train_result.get("loss_ratio"),
                        }
                    )

                    del model
                    clear_gpu_memory()
                except Exception as exc:
                    run_data["status"] = "error"
                    run_data["error"] = str(exc)

                raw_runs.append(run_data)

        points: List[Dict[str, Any]] = []
        for routing_mode in supported_modes:
            mode_runs = [
                row
                for row in raw_runs
                if row.get("routing_mode") == routing_mode and row.get("status") == "ok"
            ]
            if not mode_runs:
                continue

            def _mean(key: str) -> Optional[float]:
                vals = [float(r[key]) for r in mode_runs if r.get(key) is not None]
                return (sum(vals) / len(vals)) if vals else None

            points.append(
                {
                    "routing_mode": routing_mode,
                    "n_runs": len(mode_runs),
                    "validation_loss": _mean("validation_loss"),
                    "tokens_per_sec": _mean("tokens_per_sec"),
                    "effective_token_compute": _mean("effective_token_compute"),
                    "routing_stability": _mean("routing_stability"),
                }
            )

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

    def _build_experiment_perf_report(
        self,
        results: Dict[str, Any],
        queue_telemetry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Aggregate per-program perf traces into one experiment-level JSON report."""
        perf_traces = results.get("_perf_traces", []) or []
        starvation = results.get("_gpu_starvation", []) or []
        kernel_samples = results.get("_kernel_timing", []) or []

        trace_totals: Dict[str, float] = defaultdict(float)
        trace_counts: Dict[str, int] = defaultdict(int)
        for trace_report in perf_traces:
            summary = (trace_report or {}).get("summary_ms", {})
            if not isinstance(summary, dict):
                continue
            for name, value in summary.items():
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    continue
                trace_totals[name] += val
                trace_counts[name] += 1

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
        avg_throughput = (
            sum(throughput_vals) / len(throughput_vals) if throughput_vals else 0.0
        )

        starvation_count = 0
        starvation_total_ms = 0.0
        starvation_max_ms = 0.0
        for item in starvation:
            if not isinstance(item, dict):
                continue
            starvation_count += int(item.get("count", 0) or 0)
            starvation_total_ms += float(item.get("total_stall_ms", 0.0) or 0.0)
            starvation_max_ms = max(
                starvation_max_ms, float(item.get("max_stall_ms", 0.0) or 0.0)
            )

        op_totals: Dict[str, Dict[str, float]] = {}
        for sample in kernel_samples:
            if not isinstance(sample, dict):
                continue

            # Handle new format: mod_name -> ms (float)
            if "top_ops" not in sample:
                for op_name, ms in sample.items():
                    if isinstance(ms, (int, float)):
                        slot = op_totals.setdefault(
                            op_name,
                            {
                                "cpu_ms": 0.0,
                                "cuda_ms": 0.0,
                                "calls": 0.0,
                                "samples": 0.0,
                            },
                        )
                        slot["cuda_ms"] += float(ms)
                        slot["samples"] += 1.0
                continue

            # Handle old format (top_ops)
            for op in sample.get("top_ops", []) or []:
                op_name = str(op.get("op", "unknown"))
                slot = op_totals.setdefault(
                    op_name,
                    {"cpu_ms": 0.0, "cuda_ms": 0.0, "calls": 0.0, "samples": 0.0},
                )
                slot["cpu_ms"] += float(op.get("cpu_ms", 0.0) or 0.0)
                slot["cuda_ms"] += float(op.get("cuda_ms", 0.0) or 0.0)
                slot["calls"] += float(op.get("calls", 0.0) or 0.0)
                slot["samples"] += 1.0

        hotspot_ops = []
        for op_name, agg in op_totals.items():
            samples = max(1.0, agg["samples"])
            hotspot_ops.append(
                {
                    "op": op_name,
                    "avg_cpu_ms": round(agg["cpu_ms"] / samples, 4),
                    "avg_cuda_ms": round(agg["cuda_ms"] / samples, 4),
                    "avg_calls": round(agg["calls"] / samples, 2),
                }
            )
        hotspot_ops.sort(
            key=lambda row: max(row["avg_cuda_ms"], row["avg_cpu_ms"]), reverse=True
        )

        tp_sched_rows = results.get("training_program_scheduling", []) or []
        tp_avg_ms = [
            float(r.get("scheduling_avg_ms", 0.0) or 0.0) for r in tp_sched_rows
        ]
        tp_max_ms = [
            float(r.get("scheduling_max_ms", 0.0) or 0.0) for r in tp_sched_rows
        ]
        duplicate_work = build_duplicate_work_report(
            repeated_keys={
                "graph_fingerprint_dedup": int(results.get("skipped_dedup", 0) or 0)
            },
            hints=[
                "Large dedup counts indicate search-space waste rather than runtime overhead."
            ],
        )
        report = {
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
                "avg_schedule_ms": round(sum(tp_avg_ms) / len(tp_avg_ms), 4)
                if tp_avg_ms
                else 0.0,
                "max_schedule_ms": round(max(tp_max_ms), 4) if tp_max_ms else 0.0,
            },
        }
        report["duplicate_work"] = duplicate_work
        budget_verdict = evaluate_perf_budget_gate(
            report, budget_profile="research_default"
        )
        contract = build_perf_contract(
            component="research",
            workload="experiment_screening",
            identity={
                "experiment_id": results.get("experiment_id"),
                "total_programs": results.get("total", 0),
                "stage1_passed": results.get("stage1_passed", 0),
            },
            metrics={
                "total_time_ms": round(
                    float(results.get("elapsed_seconds", 0.0) or 0.0) * 1000.0, 4
                ),
                "avg_throughput_tok_s": report["avg_throughput_tok_s"],
                "programs_profiled": report["programs_profiled"],
                "compile_time_ms": trace_avg_ms.get("compile", 0.0),
                "forward_pass_ms": trace_avg_ms.get("forward_pass", 0.0),
                "backward_pass_ms": trace_avg_ms.get("backward_pass", 0.0),
                "optimizer_step_ms": trace_avg_ms.get("optimizer_step", 0.0),
                "queue_submit_wait_ms": float(
                    (queue_telemetry or {}).get("submit_wait_avg_ms", 0.0) or 0.0
                ),
                "queue_scheduling_wait_ms": float(
                    (queue_telemetry or {}).get("scheduling_wait_avg_ms", 0.0) or 0.0
                ),
                "gpu_starvation_max_ms": report["gpu_starvation"]["max_stall_ms"],
            },
            budget_profile="research_default",
            budget_verdict=budget_verdict,
            duplicate_work=duplicate_work,
        )
        artifact_slug = str(
            results.get("experiment_id") or f"research_perf_{int(time.time())}"
        )
        artifact_path = emit_perf_artifact(contract, slug=artifact_slug)
        contract["artifact_path"] = artifact_path
        report["perf_contract"] = contract
        report["perf_budget_gate"] = budget_verdict
        report["perf_artifact_path"] = artifact_path
        return report

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
                -float(
                    r.get("novelty_score")
                    if r.get("novelty_score") is not None
                    else 0.0
                ),
            )
        )
        top = [
            {
                "result_id": row.get("result_id"),
                "fingerprint": str(row.get("graph_fingerprint") or "")[:16],
                "loss_ratio": row.get("loss_ratio"),
                "novelty_score": row.get("novelty_score"),
                "throughput_tok_s": row.get("throughput_tok_s"),
                "avg_step_time_ms": row.get("avg_step_time_ms"),
                "stability_score": row.get("stability_score"),
            }
            for row in stage1_rows[:5]
        ]

        eff_rows = [
            r
            for r in stage1_rows
            if isinstance(r.get("throughput_tok_s"), (int, float))
        ]
        stab_rows = [
            r for r in stage1_rows if isinstance(r.get("stability_score"), (int, float))
        ]
        avg_tp = (
            (sum(float(r.get("throughput_tok_s")) for r in eff_rows) / len(eff_rows))
            if eff_rows
            else None
        )
        avg_stability = (
            (sum(float(r.get("stability_score")) for r in stab_rows) / len(stab_rows))
            if stab_rows
            else None
        )
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
            "avg_novelty": (sum(novelty_vals) / len(novelty_vals))
            if novelty_vals
            else None,
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
        except Exception as e:  # top-level error boundary: analytics must not crash experiment loop
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
            logger.debug("Refuted insights fetch failed: %s", e)  # table may not exist in older notebooks

        return past

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
        _use_progressive = (
            config.progressive_screening
            and config.vocab_size > config.qualifying_vocab_size
        )
        _phase1_vocab = (
            config.qualifying_vocab_size if _use_progressive else config.vocab_size
        )

        def fitness_fn(graph):
            fp = graph.fingerprint()

            # Fast path: return cached fitness without compilation.
            if fitness_cache is not None and fp in fitness_cache:
                return fitness_cache[fp]

            sandbox_result = None
            s1_result = None
            try:
                layer_graphs = [graph] * config.n_layers

                # Phase 1: compile at cheap qualifying vocab for sandbox check
                model = compile_model(
                    layer_graphs,
                    vocab_size=_phase1_vocab,
                    max_seq_len=config.max_seq_len,
                )
                sandbox_result = self._safe_eval_for_stage(
                    model,
                    stage_tag="evolution_fitness",
                    batch_size=2,
                    seq_len=min(128, config.max_seq_len),
                    vocab_size=_phase1_vocab,
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

                # Phase 2: recompile at real vocab for micro-training
                if _use_progressive:
                    del model
                    clear_gpu_memory()
                    model = compile_model(
                        layer_graphs,
                        vocab_size=config.vocab_size,
                        max_seq_len=config.max_seq_len,
                    )

                # Micro-train for fitness (at real vocab)
                s1_result = self._micro_train(
                    model,
                    config,
                    dev,
                    seed=self._stable_seed("fitness", fp),
                )
                del model
                clear_gpu_memory()

                # Always compute real multi-objective fitness when training
                # produced metrics.  Evolution needs uncapped fitness for
                # selection pressure — capping kills the gradient signal.
                # The accounting path (results_analysis.py) separately gates
                # stage1_passed on s1_result["passed"], so false survivors
                # are impossible regardless of fitness value.
                _fl = s1_result.get("final_loss")
                _lr = s1_result.get("loss_ratio")
                if _fl is not None and _lr is not None and _lr < 0.95:
                    fitness, _components = self._compute_multi_objective_fitness(
                        s1_result, sandbox_result, graph, config
                    )
                elif s1_result.get("passed"):
                    fitness, _components = self._compute_multi_objective_fitness(
                        s1_result, sandbox_result, graph, config
                    )
                else:
                    fitness = 0.1
            except Exception as e:  # top-level error boundary: fitness eval must not crash evolution
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
            nb.log_learning_event(
                "proactive_reasoning_agent",
                f"Spawned agent {task_id} from recommendation reasoning: {reasoning[:200]}",
                task_id=task_id,
            )
            logger.info("Spawned reasoning-based repair agent: %s", task_id)
        except Exception as e:
            logger.debug("Failed to spawn reasoning-based agent: %s", e)

    def _process_orchestrator_results(self, orchestrator, nb, exp_id, results, config):
        """Collect and record all available results from the orchestrator."""
        job_results = orchestrator.get_results()
        if not job_results:
            return
        with nb.batch():
            for jr in job_results:
                self._record_orchestrator_result(jr, nb, exp_id, results, config)

    def _record_orchestrator_result(self, jr, nb, exp_id, results, config):
        """Record a single result from the orchestrator into the notebook."""
        s1_result = jr.s1_result
        program_metrics = jr.payload["metrics"]
        graph = jr.payload["graph"]
        i = jr.index

        funnel = results.setdefault("funnel_counts", {})
        funnel["stage1_completed"] = int(funnel.get("stage1_completed", 0)) + 1

        s1_passed = s1_result.get("passed", False)
        loss_ratio = s1_result.get("loss_ratio")
        final_loss = s1_result.get("final_loss")
        throughput = s1_result.get("throughput")
        training_curve = s1_result.get("training_curve")
        from ._helpers import screening_wikitext_fields

        # Training metrics
        program_metrics["initial_loss"] = s1_result.get("initial_loss")
        program_metrics["min_loss"] = s1_result.get("min_loss")
        program_metrics["loss_improvement_rate"] = s1_result.get(
            "loss_improvement_rate"
        )
        program_metrics["avg_step_time_ms"] = s1_result.get("avg_step_time_ms")
        program_metrics["total_train_time_ms"] = s1_result.get("total_train_time_ms")
        program_metrics["max_grad_norm"] = s1_result.get("max_grad_norm")
        program_metrics["mean_grad_norm"] = s1_result.get("mean_grad_norm")
        program_metrics["grad_norm_std"] = s1_result.get("grad_norm_std")
        program_metrics["n_train_steps"] = s1_result.get("n_train_steps")
        program_metrics["final_lr"] = s1_result.get("final_lr")
        program_metrics["validation_loss"] = s1_result.get("validation_loss")
        program_metrics["validation_loss_ratio"] = s1_result.get(
            "validation_loss_ratio"
        )
        program_metrics["generalization_gap"] = s1_result.get("generalization_gap")
        program_metrics["discovery_loss"] = s1_result.get("discovery_loss")
        program_metrics["discovery_loss_ratio"] = s1_result.get("discovery_loss_ratio")
        program_metrics.update(screening_wikitext_fields(s1_result))
        program_metrics.update(
            {k: s1_result.get(k) for k in s1_result if k.startswith("pruning_")}
        )
        # Propagate error info from training result to DB record
        if s1_result.get("error_type"):
            program_metrics["error_type"] = s1_result["error_type"]
        if s1_result.get("error"):
            program_metrics["error_message"] = s1_result["error"]
        self._merge_s1_telemetry(program_metrics, s1_result)

        # Compute efficiency_multiple at screening time.
        # MoE models: skip param count penalty (active params < total params).
        from .synthesis import _graph_is_moe

        try:
            from ..leaderboard_scoring import compute_efficiency_multiple

            eff = compute_efficiency_multiple(
                loss_ratio=s1_result.get("loss_ratio"),
                param_count=program_metrics.get("param_count"),
                forward_time_ms=s1_result.get("forward_time_ms"),
                peak_memory_mb=s1_result.get("peak_memory_mb"),
                throughput_tok_s=s1_result.get("throughput"),
                is_moe=_graph_is_moe(graph) if graph else False,
            )
            if eff:
                program_metrics["efficiency_multiple"] = eff["geomean"]
        except (ImportError, TypeError, ValueError) as e:
            logger.debug("Efficiency multiple computation failed: %s", e)

        # Merge traces
        perf_report = s1_result.get("perf_report", s1_result.get("perf_traces"))
        if perf_report:
            program_metrics["perf_report_json"] = json.dumps(json_safe(perf_report))
            results.setdefault("_perf_traces", []).append(perf_report)

        starvation_report = s1_result.get(
            "starvation_report", s1_result.get("gpu_starvation")
        )
        if starvation_report:
            program_metrics["starvation_report_json"] = json.dumps(
                json_safe(starvation_report)
            )
            results.setdefault("_gpu_starvation", []).append(starvation_report)

        kernel_timings = s1_result.get(
            "kernel_timings_ms", s1_result.get("kernel_timing")
        )
        if kernel_timings:
            program_metrics["kernel_timings_json"] = json.dumps(
                json_safe(kernel_timings)
            )
            results.setdefault("_kernel_timing", []).append(kernel_timings)

        if getattr(jr, "telemetry", None):
            program_metrics["queue_telemetry_json"] = json.dumps(
                json_safe(jr.telemetry)
            )

        if s1_passed:
            results["stage1_passed"] += 1
            funnel["stage1_survived"] = int(funnel.get("stage1_survived", 0)) + 1
            with self._lock:
                self._progress.stage1_passed += 1

            logger.info(
                "  ★ S1 SURVIVOR [%d] %s — loss_ratio=%.4f, params=%s",
                i + 1,
                graph.fingerprint()[:10],
                loss_ratio or 0,
                f"{program_metrics.get('param_count', 0):,}",
            )

            # Compare to baseline (dual-metric: discovery vs validation)
            if final_loss is not None:
                try:
                    baseline = self._get_baseline()
                    baseline_steps = int(
                        s1_result.get("n_train_steps") or config.stage1_steps
                    )
                    baseline_recipe = self._resolve_baseline_recipe(
                        s1_result, default_lr=config.stage1_lr
                    )

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
                            # Keep measured discovery_loss_ratio intact; store
                            # baseline-relative comparison separately.
                            program_metrics["discovery_baseline_ratio"] = (
                                discovery_ratio
                            )
                        except Exception as e:
                            logger.debug("Discovery baseline failed: %s", e)

                    # 2. Validation Baseline (Corpus)
                    val_loss = s1_result.get("validation_loss")
                    if val_loss is not None:
                        try:
                            v_data_fn, v_data_tag, v_cache = (
                                self._make_baseline_data_fn(config, split="val")
                            )
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
                            # Keep measured validation_loss_ratio intact; store
                            # baseline-relative comparison separately.
                            program_metrics["validation_baseline_loss_ratio"] = (
                                v_baseline_ratio
                            )
                            program_metrics["validation_baseline_ratio"] = (
                                v_baseline_ratio
                            )
                        except (RuntimeError, ValueError, TypeError) as e:
                            logger.debug("Validation baseline comparison failed: %s", e)

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
                diag_model = compile_model(
                    [graph], vocab_size=config.vocab_size, max_seq_len=64
                )
                diag_result = run_diagnostic_suite(
                    diag_model, device=diag_dev, n_steps=50
                )
                program_metrics["diagnostic_score"] = diag_result.diagnostic_score
                program_metrics["diagnostic_tasks_json"] = json.dumps(
                    json_safe(diag_result.to_dict())
                )
            except Exception as e:
                logger.debug(
                    "Diagnostic suite failed for %s: %s", graph.fingerprint()[:10], e
                )

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

                    # Persist all behavioral fingerprint fields to DB
                    program_metrics["fingerprint_json"] = json.dumps(
                        json_safe(fp.to_dict())
                    )
                    program_metrics["fp_interaction_locality"] = fp.interaction_locality
                    program_metrics["fp_interaction_sparsity"] = fp.interaction_sparsity
                    program_metrics["fp_interaction_symmetry"] = fp.interaction_symmetry
                    program_metrics["fp_interaction_hierarchy"] = (
                        fp.interaction_hierarchy
                    )
                    program_metrics["fp_intrinsic_dim"] = fp.intrinsic_dim
                    program_metrics["fp_isotropy"] = fp.isotropy
                    program_metrics["fp_rank_ratio"] = fp.rank_ratio
                    program_metrics["fp_jacobian_spectral_norm"] = (
                        fp.jacobian_spectral_norm
                    )
                    program_metrics["fp_jacobian_effective_rank"] = (
                        fp.jacobian_effective_rank
                    )
                    program_metrics["fp_sensitivity_uniformity"] = (
                        fp.sensitivity_uniformity
                    )
                    program_metrics["fp_cka_vs_transformer"] = fp.cka_vs_transformer
                    program_metrics["fp_cka_vs_ssm"] = fp.cka_vs_ssm
                    program_metrics["fp_cka_vs_conv"] = fp.cka_vs_conv
                    program_metrics["fp_hierarchy_fitness"] = fp.hierarchy_fitness
                    program_metrics["fp_gromov_delta"] = fp.gromov_delta

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
                logger.debug(
                    "Novelty scoring failed for %s: %s", graph.fingerprint()[:10], e
                )

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
        # Compute NCD before recording so values go into the initial INSERT
        if training_curve:
            try:
                from ...eval.ncd import compute_graph_ncd

                graph_json_str = graph_to_json(graph)
                ncd_result = compute_graph_ncd(
                    graph_json_str,
                    training_curve,
                    n_params=program_metrics.get("param_count"),
                )
                program_metrics["ncd_score"] = ncd_result["ncd_score"]
                program_metrics["ncd_description_length"] = ncd_result[
                    "description_length"
                ]
                program_metrics["ncd_description_length_per_param"] = ncd_result[
                    "description_length_per_param"
                ]
            except Exception:
                pass

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
        if rid:
            funnel["persisted_rows"] = int(funnel.get("persisted_rows", 0)) + 1
        else:
            funnel["dropped_persistence_quality_gate"] = (
                int(funnel.get("dropped_persistence_quality_gate", 0)) + 1
            )

        if training_curve and rid:
            try:
                nb.store_training_curve(rid, training_curve)
            except Exception:
                pass
        if rid:
            try:
                from ...eval.wikitext_eval import screening_wikitext_payload

                payload = screening_wikitext_payload(s1_result)
                if payload:
                    nb.set_external_benchmarks(rid, payload)
            except Exception as e:
                logger.debug(
                    "Screening benchmark payload persist failed for %s: %s", rid, e
                )

        # Every S1 survivor gets a screening-tier leaderboard entry
        if s1_passed and rid:
            try:
                from ._helpers import _upsert_screening_entry

                _upsert_screening_entry(
                    nb,
                    {
                        "result_id": rid,
                        "model_source": program_metrics.get(
                            "model_source", "graph_synthesis"
                        ),
                        "graph_fingerprint": graph.fingerprint(),
                        "loss_ratio": loss_ratio,
                        "novelty_score": novelty_kwargs.get("novelty_score"),
                        "novelty_confidence": novelty_kwargs.get("novelty_confidence"),
                        "fp_jacobian_spectral_norm": program_metrics.get(
                            "fp_jacobian_spectral_norm"
                        ),
                        "routing_savings_ratio": program_metrics.get(
                            "routing_savings_ratio"
                        ),
                        "activation_sparsity_score": program_metrics.get(
                            "activation_sparsity_score"
                        ),
                        "depth_savings_ratio": program_metrics.get(
                            "depth_savings_ratio"
                        ),
                        "compression_ratio": program_metrics.get("compression_ratio"),
                        "wikitext_perplexity": program_metrics.get(
                            "wikitext_perplexity"
                        ),
                        "wikitext_score": program_metrics.get("wikitext_score"),
                    },
                )
            except Exception as e:
                logger.debug("Screening leaderboard upsert failed for %s: %s", rid, e)

        # Update best metrics in experiment summary
        if loss_ratio is not None:
            if (
                results["best_loss_ratio"] is None
                or loss_ratio < results["best_loss_ratio"]
            ):
                results["best_loss_ratio"] = loss_ratio

        try:
            nov = novelty_kwargs.get("novelty_score") or program_metrics.get(
                "novelty_score"
            )
            if nov is not None:
                if (
                    results["best_novelty_score"] is None
                    or nov > results["best_novelty_score"]
                ):
                    results["best_novelty_score"] = nov
        except Exception:
            pass

        self._emit_event(
            "program_evaluated",
            {
                "index": i,
                "fingerprint": graph.fingerprint()[:10],
                "result": "pass" if s1_passed else "fail",
                "loss_ratio": f"{loss_ratio:.4f}" if loss_ratio is not None else None,
                "result_id": rid,
                "throughput": f"{throughput:.0f}" if throughput else None,
                "params": program_metrics.get("param_count"),
                "memory_mb": f"{program_metrics.get('peak_memory_mb', 0):.1f}"
                if program_metrics.get("peak_memory_mb")
                else None,
                "novelty": f"{program_metrics.get('novelty_score', 0):.3f}"
                if program_metrics.get("novelty_score") is not None
                else None,
            },
        )

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
            str(row.get("result_id")): row
            for row in leaderboard
            if row.get("result_id")
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
                    rewards.append(
                        max(
                            0.0,
                            min(1.0, 0.5 * passed + 0.3 * loss_term + 0.2 * rob_term),
                        )
                    )
                    resolved = True
                elif context == "auto_validate_investigation":
                    val_pass = entry.get("validation_passed")
                    val_loss = entry.get("validation_loss_ratio")
                    val_base = entry.get("validation_baseline_ratio")
                    val_std = self._to_float(
                        entry.get("validation_multi_seed_std"), default=0.2
                    )
                    if val_pass is None and val_loss is None and val_base is None:
                        resolved = False
                        rewards = []
                        break
                    passed = 1.0 if bool(val_pass) else 0.0
                    if val_base is not None:
                        loss_term = max(
                            0.0, 1.0 - self._to_float(val_base, default=1.0)
                        )
                    else:
                        loss_term = max(
                            0.0, 1.0 - self._to_float(val_loss, default=1.0)
                        )
                    std_term = max(0.0, min(1.0, 1.0 - val_std))
                    rewards.append(
                        max(
                            0.0,
                            min(1.0, 0.5 * passed + 0.3 * loss_term + 0.2 * std_term),
                        )
                    )
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
            # Bayesian update: insights that predict well gain confidence
            try:
                trial_insight_ids = trial.get("insight_ids_json") or []
                if isinstance(trial_insight_ids, str):
                    trial_insight_ids = json.loads(trial_insight_ids)
                for insight_id in trial_insight_ids:
                    nb.update_insight_bayesian(
                        str(insight_id),
                        success=(outcome == "supported"),
                    )
            except Exception:
                pass
