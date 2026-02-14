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
import json
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..synthesis.grammar import GrammarConfig, generate_layer_graph, batch_generate
from ..synthesis.compiler import compile_graph, compile_model
from ..synthesis.validator import validate_graph
from ..synthesis.serializer import graph_to_json, graph_summary
from ..eval.sandbox import safe_eval
from ..eval.metrics import novelty_score
from ..training.loss_synthesis import synthesize_loss
from ..training.optimizer_synthesis import synthesize_optimizer
from ..training.training_program import synthesize_training_program
from .persona import Aria, get_aria
from .notebook import LabNotebook, ExperimentEntry
from .llm.context import build_experiment_context, build_history_context


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
    # Synthesis grammar
    max_depth: int = 10
    max_ops: int = 16
    math_space_weight: float = 2.0
    residual_prob: float = 0.7
    # Continuous mode
    continuous: bool = False
    max_experiments: int = 100
    rest_between_experiments: int = 5  # seconds

    def to_dict(self) -> Dict:
        return {
            "n_programs": self.n_programs,
            "model_dim": self.model_dim,
            "n_layers": self.n_layers,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "device": self.device,
            "stage1_steps": self.stage1_steps,
            "stage1_lr": self.stage1_lr,
            "stage1_batch_size": self.stage1_batch_size,
            "max_depth": self.max_depth,
            "max_ops": self.max_ops,
            "math_space_weight": self.math_space_weight,
            "residual_prob": self.residual_prob,
            "continuous": self.continuous,
            "max_experiments": self.max_experiments,
        }

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

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


class ExperimentRunner:
    """Autonomous experiment execution engine with background support."""

    def __init__(self, notebook_path: str = "research/lab_notebook.db"):
        self.notebook_path = notebook_path
        self.aria = get_aria()
        self._math_spaces_registered = False

        # Background execution state
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._progress = LiveProgress()
        self._event_queue: queue.Queue = queue.Queue(maxsize=500)
        self._lock = threading.Lock()

    def _ensure_math_spaces(self):
        if not self._math_spaces_registered:
            try:
                from ..mathspaces.registry import register_all_mathspaces
                register_all_mathspaces()
                self._math_spaces_registered = True
            except Exception:
                pass

    def _make_notebook(self) -> LabNotebook:
        """Create a new notebook connection (thread-safe)."""
        return LabNotebook(self.notebook_path)

    @property
    def progress(self) -> LiveProgress:
        with self._lock:
            return LiveProgress(**self._progress.to_dict())

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

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

        # Pre-generate experiment ID
        nb = self._make_notebook()
        if hypothesis is None:
            hypothesis = self.aria.formulate_hypothesis()

        exp_id = nb.start_experiment(
            experiment_type="synthesis",
            config=config.to_dict(),
            hypothesis=hypothesis,
        )
        nb.close()

        with self._lock:
            self._progress = LiveProgress(
                experiment_id=exp_id,
                status="generating",
                total_programs=config.n_programs,
                aria_message=self.aria.greet(),
            )

        self._emit_event("experiment_started", {
            "experiment_id": exp_id,
            "hypothesis": hypothesis,
            "config": config.to_dict(),
            "aria_greeting": self.aria.greet(),
        })

        self._thread = threading.Thread(
            target=self._run_experiment_thread,
            args=(exp_id, config, hypothesis),
            daemon=True,
        )
        self._thread.start()
        return exp_id

    def start_continuous(self, config: RunConfig) -> str:
        """Start continuous experiment mode in background."""
        if self.is_running:
            raise RuntimeError("An experiment is already running")

        self._ensure_math_spaces()
        self._stop_event.clear()

        config.continuous = True

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

    def stop(self):
        """Stop the current experiment gracefully."""
        self._stop_event.set()
        self.aria.state.mood = "contemplative"
        with self._lock:
            self._progress.status = "stopped"
            self._progress.aria_message = "Stopping... wrapping up current evaluation."
        self._emit_event("experiment_stopping", {})

    # ── Background Threads ──

    def _run_experiment_thread(self, exp_id: str, config: RunConfig,
                                hypothesis: str):
        """Execute a single experiment in background."""
        nb = self._make_notebook()
        try:
            results = self._execute_experiment(exp_id, config, nb)

            # Build context for LLM-enhanced methods
            context = build_experiment_context(results, config.to_dict(), hypothesis)
            history = nb.get_recent_experiments(10)
            if history:
                context += "\n\n" + build_history_context(history)

            summary = self.aria.experiment_summary(results, context=context)
            insights = self._analyze_results(results, exp_id, nb, context=context)

            # Store LLM analysis if available
            llm_analysis = self.aria.analyze_results(results, context=context)

            nb.complete_experiment(
                experiment_id=exp_id,
                results=results,
                aria_summary=summary,
                aria_mood=self.aria.state.mood,
                insights=insights,
                llm_analysis=llm_analysis,
            )

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

    def _run_continuous_thread(self, config: RunConfig):
        """Execute continuous experiments in background."""
        n_experiments = 0
        while n_experiments < config.max_experiments and not self._stop_event.is_set():
            n_experiments += 1
            nb = self._make_notebook()

            hypothesis = self.aria.formulate_hypothesis()
            exp_id = nb.start_experiment(
                experiment_type="synthesis",
                config=config.to_dict(),
                hypothesis=hypothesis,
            )

            with self._lock:
                self._progress = LiveProgress(
                    experiment_id=exp_id,
                    status="generating",
                    total_programs=config.n_programs,
                    aria_message=f"Experiment {n_experiments}/{config.max_experiments}: {hypothesis}",
                )

            self._emit_event("experiment_started", {
                "experiment_id": exp_id,
                "experiment_number": n_experiments,
                "hypothesis": hypothesis,
            })

            try:
                results = self._execute_experiment(exp_id, config, nb)
                context = build_experiment_context(results, config.to_dict(), hypothesis)
                summary = self.aria.experiment_summary(results, context=context)
                insights = self._analyze_results(results, exp_id, nb, context=context)
                llm_analysis = self.aria.analyze_results(results, context=context)
                nb.complete_experiment(
                    experiment_id=exp_id, results=results,
                    aria_summary=summary, aria_mood=self.aria.state.mood,
                    insights=insights, llm_analysis=llm_analysis,
                )
                self._emit_event("experiment_completed", {
                    "experiment_id": exp_id, "results": results,
                })
            except Exception as e:
                nb.fail_experiment(exp_id, str(e))
                self._emit_event("experiment_failed", {
                    "experiment_id": exp_id, "error": str(e),
                })
            finally:
                nb.close()

            if config.rest_between_experiments > 0 and not self._stop_event.is_set():
                time.sleep(config.rest_between_experiments)

        with self._lock:
            self._progress.status = "completed" if not self._stop_event.is_set() else "stopped"
            self._progress.aria_message = (
                f"Continuous session complete: {n_experiments} experiments."
                if not self._stop_event.is_set()
                else f"Stopped after {n_experiments} experiments."
            )

    # ── Core Execution ──

    def _execute_experiment(self, exp_id: str, config: RunConfig,
                            nb: LabNotebook) -> Dict:
        """Core experiment logic shared by single and continuous modes."""
        results = {
            "total": 0, "stage0_passed": 0, "stage05_passed": 0,
            "stage1_passed": 0, "novel_count": 0,
            "best_loss_ratio": None, "best_novelty_score": None,
            "survivors": [],
        }

        grammar = GrammarConfig(
            model_dim=config.model_dim,
            max_depth=config.max_depth,
            max_ops=config.max_ops,
            residual_prob=config.residual_prob,
        )
        grammar.category_weights["math_space"] = config.math_space_weight

        t_start = time.time()

        # Generate graphs
        graphs = batch_generate(config.n_programs, grammar)
        results["total"] = len(graphs)

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

            # Validate
            with self._lock:
                self._progress.current_stage = "validating"

            validation = validate_graph(graph)
            if not validation.valid:
                nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=graph.fingerprint(),
                    graph_json=graph_to_json(graph),
                    stage0_error="; ".join(validation.errors),
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
                nb.record_program_result(
                    experiment_id=exp_id,
                    graph_fingerprint=graph.fingerprint(),
                    graph_json=graph_to_json(graph),
                    stage0_error=str(e),
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

            s0_passed = sandbox_result.passed
            s05_passed = sandbox_result.stability_score >= 0.5

            if s0_passed:
                results["stage0_passed"] += 1
                with self._lock:
                    self._progress.stage0_passed += 1
            if s05_passed:
                results["stage05_passed"] += 1
                with self._lock:
                    self._progress.stage05_passed += 1

            # Stage 1
            s1_passed = False
            loss_ratio = None
            final_loss = None
            throughput = None

            if s0_passed and s05_passed and not self._stop_event.is_set():
                with self._lock:
                    self._progress.current_stage = "stage1"

                s1_result = self._micro_train(model, config, dev)
                s1_passed = s1_result.get("passed", False)
                loss_ratio = s1_result.get("loss_ratio")
                final_loss = s1_result.get("final_loss")
                throughput = s1_result.get("throughput")

                if s1_passed:
                    results["stage1_passed"] += 1
                    with self._lock:
                        self._progress.stage1_passed += 1

            # Novelty
            with self._lock:
                self._progress.current_stage = "novelty"

            nov = novelty_score(graph)
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

            # Record
            nb.record_program_result(
                experiment_id=exp_id,
                graph_fingerprint=graph.fingerprint(),
                graph_json=graph_to_json(graph),
                stage0_passed=s0_passed, stage05_passed=s05_passed,
                stage1_passed=s1_passed, stage0_error=sandbox_result.error,
                param_count=sandbox_result.param_count,
                loss_ratio=loss_ratio, final_loss=final_loss,
                throughput=throughput, novelty_score=n_score,
                structural_novelty=nov.structural_novelty,
                behavioral_novelty=nov.behavioral_novelty,
                most_similar_to=nov.most_similar_to,
            )

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

    def _micro_train(self, model: nn.Module, config: RunConfig,
                     dev: torch.device) -> Dict:
        """Run Stage 1 micro-training."""
        result = {"passed": False}

        try:
            model = model.to(dev)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=config.stage1_lr, weight_decay=0.01)

            initial_loss = None
            final_loss = None
            total_tokens = 0
            t_start = time.perf_counter()

            for step in range(config.stage1_steps):
                if self._stop_event.is_set():
                    break

                input_ids = torch.randint(
                    0, config.vocab_size,
                    (config.stage1_batch_size, min(128, config.max_seq_len)),
                    device=dev,
                )

                with torch.amp.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                        enabled=(dev.type == "cuda")):
                    logits = model(input_ids)
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.shape[-1]),
                        input_ids[:, 1:].reshape(-1),
                    )

                if torch.isnan(loss) or torch.isinf(loss):
                    result["error"] = f"NaN/Inf loss at step {step}"
                    return result

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                loss_val = loss.item()
                if step == 0:
                    initial_loss = loss_val
                final_loss = loss_val
                total_tokens += input_ids.numel()

            t_end = time.perf_counter()

            if initial_loss and final_loss:
                result["loss_ratio"] = final_loss / max(initial_loss, 1e-6)
                result["final_loss"] = final_loss
                result["initial_loss"] = initial_loss
                result["throughput"] = total_tokens / (t_end - t_start)
                result["passed"] = result["loss_ratio"] < 0.8

        except Exception as e:
            result["error"] = str(e)

        return result

    def _analyze_results(self, results: Dict, exp_id: str,
                         nb: LabNotebook, context: str = "") -> List[str]:
        """Analyze experiment results and generate insights."""
        insights = self._rule_based_insights(results, exp_id, nb)
        return insights

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
