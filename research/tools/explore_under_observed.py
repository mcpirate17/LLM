"""
Under-Observed Component Exploration

Guarantees every component with fewer than N observations is preferentially
selected into test graph generation for evidence collection.

Two modes:
  --mode=weighted   Boost under-observed ops via op_weights (fast, statistical)
  --mode=forced     Generate one dedicated graph per target op (slow, exhaustive)

Pipeline per graph: compile → forward → rapid screening → S1 micro-train
Reports: markdown + JSON with per-component coverage proof.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import signal
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Graceful stop on Ctrl+C ──────────────────────────────────────────
_stop_requested = False


def _handle_sigint(signum, frame):
    global _stop_requested
    if _stop_requested:
        # Second Ctrl+C → hard exit
        logger.warning("Second interrupt — aborting immediately.")
        sys.exit(1)
    _stop_requested = True
    logger.info(
        "\nCtrl+C detected — finishing current evaluation then saving results..."
    )


# ── Project imports ──────────────────────────────────────────────────
from research.defaults import MODEL_DIM, N_LAYERS, MAX_SEQ_LEN

# Use 32K vocab for exploration — 100K (tiktoken) is too hard for small
# exploration models (256-dim, 4 layers). ln(100277)=11.52 baseline is
# unreachable in 500 steps. 32K gives ln(32000)=10.37, achievable.
VOCAB_SIZE = 32000
from research.synthesis.grammar import (
    GrammarConfig,
    generate_layer_graph,
    batch_generate,
)
from research.synthesis.grammar_support import OP_TO_TEMPLATE
from research.synthesis.primitives import PRIMITIVE_REGISTRY
from research.synthesis.compiler import compile_model
from research.synthesis.graph import ComputationGraph
from research.synthesis.motifs import VALIDATED_MOTIFS, resolve_step
from research.synthesis.templates import apply_template
from research.eval.sandbox import safe_eval
from research.eval.screening_rapid import RapidScreeningCheck


# ── Data structures ──────────────────────────────────────────────────


@dataclass(slots=True)
class OpCoverage:
    """Per-op coverage tracking through the pipeline."""

    op_name: str
    n_prior_observations: int
    # Pipeline stages
    attempted: int = 0  # graphs generated containing this op
    inserted: int = 0  # op confirmed in final graph (post-prune)
    compile_pass: int = 0
    compile_fail: int = 0
    forward_pass: int = 0
    forward_fail: int = 0
    rapid_pass: int = 0
    rapid_fail: int = 0
    s1_pass: int = 0
    s1_fail: int = 0
    # Reasons for not being inserted
    skip_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "op_name": self.op_name,
            "n_prior_observations": self.n_prior_observations,
            "attempted": self.attempted,
            "inserted": self.inserted,
            "compile_pass": self.compile_pass,
            "compile_fail": self.compile_fail,
            "forward_pass": self.forward_pass,
            "forward_fail": self.forward_fail,
            "rapid_pass": self.rapid_pass,
            "rapid_fail": self.rapid_fail,
            "s1_pass": self.s1_pass,
            "s1_fail": self.s1_fail,
            "skip_reasons": list(set(self.skip_reasons)),
        }


@dataclass(slots=True)
class ExplorationResult:
    """Result of evaluating a single graph."""

    graph_fingerprint: str
    target_ops: List[str]  # ops we wanted in this graph
    ops_present: List[str]  # ops actually in the graph
    graph_json: Optional[str] = None  # for DB recording
    compile_ok: bool = False
    compile_error: Optional[str] = None
    forward_ok: bool = False
    forward_error: Optional[str] = None
    stability_score: float = 0.0
    rapid_ok: bool = False
    rapid_error: Optional[str] = None
    s1_ok: bool = False
    s1_loss_ratio: Optional[float] = None
    s1_initial_loss: Optional[float] = None
    s1_final_loss: Optional[float] = None
    s1_error: Optional[str] = None
    param_count: int = 0
    n_ops: int = 0
    elapsed_s: float = 0.0


# ── Target discovery ─────────────────────────────────────────────────


def discover_targets(
    db_path: str,
    threshold: int = 20,
) -> Dict[str, int]:
    """Find ops with fewer than `threshold` observations.

    Returns {op_name: n_used} for all under-observed ops.
    Also includes ops in PRIMITIVE_REGISTRY that have zero observations
    (never appeared in op_success_rates at all).
    """
    # Ensure math-space ops are registered
    try:
        from research.mathspaces.registry import register_all_mathspaces

        register_all_mathspaces()
    except Exception:
        pass

    all_ops = set(PRIMITIVE_REGISTRY.keys())
    # Remove pseudo-ops that can't be independently placed
    _SKIP_OPS = {"input", "output", "add", "concat", "split"}
    all_ops -= _SKIP_OPS

    observed: Dict[str, int] = {}
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            tables = {
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "op_stats" in tables:
                rows = conn.execute(
                    "SELECT op_name, eval_count FROM op_stats"
                ).fetchall()
                for r in rows:
                    observed[r["op_name"]] = int(r["eval_count"] or 0)
            elif "op_success_rates" in tables:
                rows = conn.execute(
                    "SELECT op_name, n_used FROM op_success_rates"
                ).fetchall()
                for r in rows:
                    observed[r["op_name"]] = int(r["n_used"] or 0)
        finally:
            conn.close()

    targets: Dict[str, int] = {}
    for op_name in sorted(all_ops):
        n = observed.get(op_name, 0)
        if n < threshold:
            targets[op_name] = n

    return targets


# ── Graph ops extraction ─────────────────────────────────────────────


def _ops_in_graph(graph) -> Set[str]:
    """Extract the set of op names from a ComputationGraph."""
    return {
        node.op_name
        for node in graph.nodes.values()
        if node.op_name not in ("input", "output")
    }


# ── Forced-coverage graph generation ─────────────────────────────────


def _find_motifs_containing_op(op_name: str) -> List[str]:
    """Find motif names that contain the target op."""
    from research.synthesis.motifs import ALL_MOTIFS

    return [
        m.name for m in ALL_MOTIFS if any(step.op_name == op_name for step in m.steps)
    ]


def _build_graph_from_motif(motif_name: str, seed: int, model_dim: int):
    """Build a minimal graph from a validated motif."""
    motif = VALIDATED_MOTIFS[motif_name]
    g = ComputationGraph(model_dim=model_dim)
    current = g.add_input()
    rng = random.Random(seed)

    for i, step in enumerate(motif.steps):
        next_op = motif.steps[i + 1].op_name if i + 1 < len(motif.steps) else None
        prev_op = g.nodes[current].op_name if not g.nodes[current].is_input else None
        op_name, config = resolve_step(step, rng, prev_op=prev_op, next_op=next_op)
        config = dict(config or {})

        cur_dim = g.nodes[current].output_shape.dim
        prim = PRIMITIVE_REGISTRY.get(op_name)
        n_inputs = prim.n_inputs if prim else 1
        inputs = [current]
        if n_inputs == 2:
            inp2 = g.add_op("linear_proj", [current], config={"out_dim": cur_dim})
            inputs = [current, inp2]

        if op_name in ("linear_proj", "fused_linear_gelu", "gated_linear"):
            config.setdefault("out_dim", model_dim)
        elif op_name == "linear_proj_down":
            config.setdefault("out_dim", max(cur_dim // 2, 4))
        elif op_name == "linear_proj_up":
            config.setdefault("out_dim", model_dim)
        elif op_name in (
            "nm_sparse_linear",
            "block_sparse_linear",
            "semi_structured_2_4_linear",
            "ternary_projection",
            "kronecker_linear",
        ):
            config.setdefault("out_dim", model_dim)

        current = g.add_op(op_name, inputs, config=config)

    out_dim = g.nodes[current].output_shape.dim
    if out_dim != model_dim:
        current = g.add_op("linear_proj", [current], config={"out_dim": model_dim})

    g.set_output(current)
    return g


def _make_config_for_op(
    op_name: str,
    base_config: Optional[GrammarConfig] = None,
    model_dim: int = MODEL_DIM,
    boost_factor: float = 50.0,
    composition_depth: Optional[int] = None,
    max_depth: Optional[int] = None,
    max_ops: Optional[int] = None,
) -> GrammarConfig:
    """Create a GrammarConfig that biases toward `op_name`.

    Uses GrammarConfig.exploration() which boosts motifs and templates
    containing the target op via _OP_TO_TEMPLATE entries. Does NOT
    override individual op weights — let the template system select
    the right architectural context rather than forcing the op into
    a random slot.
    """
    dim = base_config.model_dim if base_config else model_dim
    cfg = GrammarConfig.exploration(
        target_ops=frozenset({op_name}),
        model_dim=dim,
        boost_factor=boost_factor,
    )
    if composition_depth is not None:
        cfg.composition_depth = composition_depth
    if max_depth is not None:
        cfg.max_depth = max_depth
    if max_ops is not None:
        cfg.max_ops = max_ops
    return cfg


def generate_forced_graph(
    op_name: str,
    seed: int,
    base_config: Optional[GrammarConfig] = None,
    max_retries: int = 100,
    model_dim: int = MODEL_DIM,
    boost_factor: float = 50.0,
    composition_depth: Optional[int] = None,
    max_depth: Optional[int] = None,
    max_ops: Optional[int] = None,
    seen_fingerprints: Optional[set] = None,
) -> Optional[object]:
    """Generate a unique graph containing `op_name` via the grammar's template system.

    Uses generate_layer_graph() with exploration_targets boosting +
    _OP_TO_TEMPLATE entries to select the right dedicated template.
    Skips graphs whose fingerprint is already in seen_fingerprints.
    Auto-widens composition_depth and max_ops when duplicates dominate.

    Returns (graph, retry_count) or (None, max_retries) if all attempts fail.
    """
    seen = seen_fingerprints if seen_fingerprints is not None else set()

    template_name = OP_TO_TEMPLATE.get(op_name)
    if template_name:
        dim = base_config.model_dim if base_config else model_dim
        for attempt in range(max_retries):
            try:
                g = ComputationGraph(model_dim=dim)
                inp = g.add_input()
                out = apply_template(
                    g,
                    inp,
                    random.Random(seed + attempt * 31),
                    template_name=template_name,
                )
                g.set_output(out)
                if op_name not in _ops_in_graph(g):
                    continue
                fp = g.fingerprint()
                if fp in seen:
                    continue
                seen.add(fp)
                return g, attempt
            except (ValueError, RuntimeError):
                continue

    motifs = _find_motifs_containing_op(op_name)
    if motifs:
        dim = base_config.model_dim if base_config else model_dim
        for attempt in range(max_retries):
            motif_name = motifs[attempt % len(motifs)]
            try:
                g = _build_graph_from_motif(
                    motif_name, seed=seed + attempt * 31, model_dim=dim
                )
                if op_name not in _ops_in_graph(g):
                    continue
                fp = g.fingerprint()
                if fp in seen:
                    continue
                seen.add(fp)
                return g, attempt
            except (ValueError, RuntimeError):
                continue

    cfg = _make_config_for_op(
        op_name,
        base_config,
        model_dim=model_dim,
        boost_factor=boost_factor,
        composition_depth=composition_depth,
        max_depth=max_depth,
        max_ops=max_ops,
    )
    base_depth = cfg.composition_depth
    base_max_ops = cfg.max_ops
    base_max_depth = cfg.max_depth
    dup_streak = 0

    for attempt in range(max_retries):
        try:
            g = generate_layer_graph(cfg, seed=seed + attempt * 31)
            if op_name not in _ops_in_graph(g):
                continue
            fp = g.fingerprint()
            if fp in seen:
                dup_streak += 1
                # Auto-widen search after consecutive duplicates
                if dup_streak >= 5 and dup_streak % 5 == 0:
                    new_depth = min(base_depth + 1 + dup_streak // 5, 5)
                    new_max_ops = min(base_max_ops + dup_streak, 30)
                    new_max_d = min(base_max_depth + dup_streak // 3, 20)
                    if new_depth != cfg.composition_depth or new_max_ops != cfg.max_ops:
                        logger.info(
                            "    Auto-widening search: n_blocks=%d→%d, max_ops=%d→%d, max_depth=%d→%d (%d consecutive duplicates)",
                            cfg.composition_depth,
                            new_depth,
                            cfg.max_ops,
                            new_max_ops,
                            cfg.max_depth,
                            new_max_d,
                            dup_streak,
                        )
                    cfg.composition_depth = new_depth
                    cfg.max_ops = new_max_ops
                    cfg.max_depth = new_max_d
                continue
            seen.add(fp)
            return g, attempt
        except (ValueError, RuntimeError):
            continue

    return None, max_retries


# ── Weighted-preference batch generation ─────────────────────────────


def generate_weighted_batch(
    targets: Dict[str, int],
    n_graphs: int,
    base_seed: int = 42,
    base_config: Optional[GrammarConfig] = None,
    model_dim: int = MODEL_DIM,
    composition_depth: Optional[int] = None,
    max_depth: Optional[int] = None,
    max_ops: Optional[int] = None,
) -> List:
    """Generate a batch of graphs with under-observed ops boosted.

    Uses GrammarConfig.exploration() for motif/template boosting, plus
    inverse-observation weighting on op_weights.
    """
    dim = base_config.model_dim if base_config else model_dim
    cfg = GrammarConfig.exploration(
        target_ops=frozenset(targets.keys()),
        model_dim=dim,
        boost_factor=4.0,
    )
    if composition_depth is not None:
        cfg.composition_depth = composition_depth
    if max_depth is not None:
        cfg.max_depth = max_depth
    if max_ops is not None:
        cfg.max_ops = max_ops

    # Layer on inverse-observation weighting: fewer obs → higher per-op weight
    max_obs = max(targets.values()) if targets else 1
    for op_name, n_obs in targets.items():
        weight = 2.0 + 18.0 * (1.0 - n_obs / max(max_obs, 1))
        cfg.op_weights[op_name] = weight

    result = batch_generate(n_graphs, config=cfg, base_seed=base_seed)
    return result.graphs


# ── Pipeline execution ───────────────────────────────────────────────


@dataclass(slots=True)
class _S1Result:
    passed: bool
    loss_ratio: Optional[float] = None
    initial_loss: Optional[float] = None
    final_loss: Optional[float] = None
    error: Optional[str] = None


def _s1_micro_train(
    model: nn.Module,
    device: str,
    vocab_size: int,
    seq_len: int = 128,
    n_steps: int = 500,
    lr: float = 3e-4,
    batch_size: int = 4,
) -> _S1Result:
    """Minimal S1 micro-training."""
    dev = torch.device(device)
    model = model.to(dev)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses = []

    try:
        for step in range(n_steps):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=dev)
            targets = input_ids[:, 1:]
            logits = model(input_ids)[:, :-1].contiguous()
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.contiguous().view(-1)
            )
            if torch.isnan(loss) or torch.isinf(loss):
                return _S1Result(False, error=f"NaN/Inf loss at step {step}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

            if step > 10 and losses[-1] > 500.0:
                ratio = losses[-1] / losses[0] if losses[0] > 0 else None
                return _S1Result(
                    False,
                    loss_ratio=ratio,
                    initial_loss=losses[0],
                    final_loss=losses[-1],
                    error="loss > 500",
                )
    except Exception as e:
        initial = losses[0] if losses else None
        final = losses[-1] if losses else None
        ratio = final / initial if initial and final and initial > 0 else None
        return _S1Result(
            False,
            loss_ratio=ratio,
            initial_loss=initial,
            final_loss=final,
            error=str(e),
        )

    if len(losses) < 2:
        return _S1Result(False, error="insufficient steps")

    loss_ratio = losses[-1] / losses[0] if losses[0] > 0 else float("inf")
    # Entropy-relative threshold: require using 10% of headroom above
    # entropy floor, not a fixed 5% of total loss. Complex architectures
    # start closer to the floor and have less room to improve as a percentage.
    import math

    _entropy_floor = math.log(VOCAB_SIZE)
    _headroom = max(losses[0] - _entropy_floor, 0.5)
    _min_improvement = _headroom * 0.10
    _raw_thr = 1.0 - _min_improvement / max(losses[0], 1.0)
    _ratio_threshold = max(0.90, min(0.99, _raw_thr))
    passed = loss_ratio < _ratio_threshold
    return _S1Result(
        passed, loss_ratio=loss_ratio, initial_loss=losses[0], final_loss=losses[-1]
    )


def evaluate_graph(
    graph,
    device: str = "cpu",
    vocab_size: int = VOCAB_SIZE,
    n_layers: int = N_LAYERS,
    run_s1: bool = True,
    s1_steps: int = 500,
    rapid_steps: int = 150,
) -> ExplorationResult:
    """Run a graph through the full pipeline: compile → forward → rapid → S1."""
    t0 = time.perf_counter()
    ops = sorted(_ops_in_graph(graph))
    result = ExplorationResult(
        graph_fingerprint=graph.fingerprint(),
        target_ops=[],
        ops_present=ops,
        graph_json=json.dumps(graph.to_dict()),
        n_ops=len(ops),
    )

    # ── Compile ──
    try:
        model = compile_model(
            [graph] * n_layers,
            vocab_size=vocab_size,
            max_seq_len=MAX_SEQ_LEN,
        )
        result.compile_ok = True
        result.param_count = sum(p.numel() for p in model.parameters())
    except Exception as e:
        result.compile_error = str(e)[:200]
        result.elapsed_s = time.perf_counter() - t0
        return result

    # ── Forward (safe_eval / S0) ──
    try:
        s0 = safe_eval(
            model,
            batch_size=2,
            seq_len=128,
            vocab_size=vocab_size,
            device=device,
            timeout_seconds=30,
            run_stability_probe=True,
        )
        result.forward_ok = s0.passed
        result.stability_score = s0.stability_score
        if not s0.passed:
            result.forward_error = s0.error[:200] if s0.error else "s0 failed"
    except Exception as e:
        result.forward_error = str(e)[:200]

    if not result.forward_ok:
        _cleanup_model(model, device)
        result.elapsed_s = time.perf_counter() - t0
        return result

    # ── Rapid screening ──
    try:
        rapid = RapidScreeningCheck(max_steps=rapid_steps)
        rapid_result = rapid.run(
            model,
            vocab_size=vocab_size,
            seq_len=128,
            batch_size=2,
            device=device,
        )
        result.rapid_ok = rapid_result.passed
        rapid_result.metrics.get("initial_loss")
        if not rapid_result.passed:
            result.rapid_error = rapid_result.kill_reason
    except Exception as e:
        result.rapid_error = str(e)[:200]

    if not result.rapid_ok or not run_s1:
        _cleanup_model(model, device)
        result.elapsed_s = time.perf_counter() - t0
        return result

    # ── S1 micro-train ──
    # Recompile a fresh model so S1 measures learning from scratch,
    # not from the post-screening state.
    _cleanup_model(model, device)
    try:
        model = compile_model(
            [graph] * n_layers,
            vocab_size=vocab_size,
            max_seq_len=MAX_SEQ_LEN,
        )
    except Exception as e:
        result.s1_error = f"recompile: {e}"
        result.elapsed_s = time.perf_counter() - t0
        return result

    s1 = _s1_micro_train(model, device, vocab_size, n_steps=s1_steps)
    result.s1_ok = s1.passed
    result.s1_loss_ratio = s1.loss_ratio
    result.s1_initial_loss = s1.initial_loss
    result.s1_final_loss = s1.final_loss
    result.s1_error = s1.error

    _cleanup_model(model, device)
    result.elapsed_s = time.perf_counter() - t0
    return result


def _cleanup_model(model: nn.Module, device: str):
    """Release model memory."""
    del model
    gc.collect()
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Coverage tracking ────────────────────────────────────────────────


def update_coverage(
    coverage: Dict[str, OpCoverage],
    graph,
    result: ExplorationResult,
    target_ops: Set[str],
):
    """Update coverage stats for all target ops based on graph eval result."""
    present_ops = _ops_in_graph(graph)

    for op_name in target_ops:
        cov = coverage[op_name]
        if op_name in present_ops:
            cov.inserted += 1
            if result.compile_ok:
                cov.compile_pass += 1
            else:
                cov.compile_fail += 1
            if result.forward_ok:
                cov.forward_pass += 1
            elif result.compile_ok:
                cov.forward_fail += 1
            if result.rapid_ok:
                cov.rapid_pass += 1
            elif result.forward_ok:
                cov.rapid_fail += 1
            if result.s1_ok:
                cov.s1_pass += 1
            elif result.rapid_ok:
                cov.s1_fail += 1


# ── Reporting ────────────────────────────────────────────────────────


def write_reports(
    coverage: Dict[str, OpCoverage],
    results: List[ExplorationResult],
    output_dir: str,
    mode: str,
    threshold: int,
    elapsed_total: float,
):
    """Write markdown and JSON reports."""
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # ── JSON report ──
    json_data = {
        "timestamp": ts,
        "mode": mode,
        "threshold": threshold,
        "n_targets": len(coverage),
        "n_graphs_evaluated": len(results),
        "elapsed_seconds": round(elapsed_total, 1),
        "coverage": {op: cov.to_dict() for op, cov in sorted(coverage.items())},
        "summary": _build_summary(coverage),
        "results": [
            {
                "fingerprint": r.graph_fingerprint,
                "ops_present": r.ops_present,
                "compile_ok": r.compile_ok,
                "forward_ok": r.forward_ok,
                "rapid_ok": r.rapid_ok,
                "s1_ok": r.s1_ok,
                "s1_loss_ratio": r.s1_loss_ratio,
                "param_count": r.param_count,
                "elapsed_s": round(r.elapsed_s, 2),
                "errors": {
                    k: v
                    for k, v in {
                        "compile": r.compile_error,
                        "forward": r.forward_error,
                        "rapid": r.rapid_error,
                        "s1": r.s1_error,
                    }.items()
                    if v
                },
            }
            for r in results
        ],
    }
    json_path = os.path.join(output_dir, f"exploration_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # ── Markdown report ──
    md_path = os.path.join(output_dir, f"exploration_{ts}.md")
    summary = json_data["summary"]
    lines = [
        "# Under-Observed Component Exploration Report",
        "",
        f"**Date**: {ts}",
        f"**Mode**: {mode}",
        f"**Observation threshold**: < {threshold}",
        f"**Target ops**: {len(coverage)}",
        f"**Graphs evaluated**: {len(results)}",
        f"**Elapsed**: {elapsed_total:.1f}s",
        "",
        "## Summary",
        "",
        "| Metric | Count | Rate |",
        "|--------|-------|------|",
        f"| Targets | {summary['n_targets']} | - |",
        f"| Covered (inserted ≥1) | {summary['n_covered']} | {summary['coverage_rate']:.0%} |",
        f"| Compile pass | {summary['n_compile_pass']} | {summary['compile_rate']:.0%} |",
        f"| Forward pass | {summary['n_forward_pass']} | {summary['forward_rate']:.0%} |",
        f"| Rapid pass | {summary['n_rapid_pass']} | {summary['rapid_rate']:.0%} |",
        f"| S1 pass | {summary['n_s1_pass']} | {summary['s1_rate']:.0%} |",
        "",
        "## Per-Component Coverage",
        "",
        "| Op | Prior Obs | Inserted | Compile | Forward | Rapid | S1 | Status |",
        "|-----|-----------|----------|---------|---------|-------|-----|--------|",
    ]

    for op_name in sorted(coverage.keys()):
        cov = coverage[op_name]
        if cov.inserted == 0:
            status = "NOT COVERED"
            reasons = (
                "; ".join(set(cov.skip_reasons))
                if cov.skip_reasons
                else "generation failed"
            )
        elif cov.compile_pass == 0:
            status = "COMPILE FAIL"
        elif cov.forward_pass == 0:
            status = "FORWARD FAIL"
        elif cov.rapid_pass == 0:
            status = "RAPID FAIL"
        elif cov.s1_pass == 0:
            status = "S1 FAIL"
        else:
            status = "PASS"

        lines.append(
            f"| {op_name} | {cov.n_prior_observations} "
            f"| {cov.inserted}/{cov.attempted} "
            f"| {cov.compile_pass} | {cov.forward_pass} "
            f"| {cov.rapid_pass} | {cov.s1_pass} | {status} |"
        )

    # Explain uncovered ops
    uncovered = [op for op, cov in coverage.items() if cov.inserted == 0]
    if uncovered:
        lines.extend(
            [
                "",
                "## Uncovered Ops — Explanation",
                "",
            ]
        )
        for op_name in sorted(uncovered):
            cov = coverage[op_name]
            reasons = (
                set(cov.skip_reasons)
                if cov.skip_reasons
                else {"no motif contains this op; grammar cannot place it"}
            )
            prim = PRIMITIVE_REGISTRY.get(op_name)
            prim_info = (
                f"category={prim.category.value}, space={getattr(prim, 'algebraic_space', 'euclidean')}"
                if prim
                else "not in registry"
            )
            lines.append(f"- **{op_name}** ({prim_info}): {'; '.join(reasons)}")

    lines.extend(["", "---", f"JSON: `{json_path}`"])

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    return md_path, json_path


def _build_summary(coverage: Dict[str, OpCoverage]) -> dict:
    n = len(coverage)
    n_covered = sum(1 for c in coverage.values() if c.inserted > 0)
    n_compile = sum(1 for c in coverage.values() if c.compile_pass > 0)
    n_forward = sum(1 for c in coverage.values() if c.forward_pass > 0)
    n_rapid = sum(1 for c in coverage.values() if c.rapid_pass > 0)
    n_s1 = sum(1 for c in coverage.values() if c.s1_pass > 0)
    return {
        "n_targets": n,
        "n_covered": n_covered,
        "coverage_rate": n_covered / max(n, 1),
        "n_compile_pass": n_compile,
        "compile_rate": n_compile / max(n, 1),
        "n_forward_pass": n_forward,
        "forward_rate": n_forward / max(n, 1),
        "n_rapid_pass": n_rapid,
        "rapid_rate": n_rapid / max(n, 1),
        "n_s1_pass": n_s1,
        "s1_rate": n_s1 / max(n, 1),
    }


# ── DB recording ─────────────────────────────────────────────────────


class _DBRecorder:
    """Real-time DB recorder. Records each result immediately after evaluation."""

    __slots__ = (
        "nb",
        "exp_id",
        "n_recorded",
        "n_s0_passed",
        "n_s1_passed",
        "best_lr",
        "s1_steps",
    )

    def __init__(
        self,
        db_path: str,
        threshold: int,
        mode: str,
        *,
        device: str,
        s1_steps: int,
        rapid_steps: int,
        model_dim: int,
        n_layers: int,
    ):
        from research.scientist.notebook import LabNotebook

        self.nb = LabNotebook(db_path)
        self.exp_id = self.nb.start_experiment(
            experiment_type="forced_exploration",
            config={
                "mode": mode,
                "threshold": threshold,
                "data_mode": "random",
                "tokenizer_mode": "byte",
                "vocab_size": VOCAB_SIZE,
                "device": device,
                "s1_steps": s1_steps,
                "rapid_steps": rapid_steps,
                "model_dim": model_dim,
                "n_layers": n_layers,
                "model_source": "forced_exploration",
            },
            hypothesis=f"Under-observed component coverage (threshold={threshold})",
        )
        self.n_recorded = 0
        self.n_s0_passed = 0
        self.n_s1_passed = 0
        self.best_lr: Optional[float] = None
        self.s1_steps = int(s1_steps)
        logger.info("Created experiment %s for real-time DB recording", self.exp_id)

    def record(self, r: ExplorationResult) -> None:
        """Record a single result immediately."""
        if not r.graph_json:
            return

        error_type = None
        error_msg = None
        if r.compile_error:
            error_type = "compile_error"
            error_msg = r.compile_error
        elif r.forward_error:
            error_type = "forward_error"
            error_msg = r.forward_error
        elif r.rapid_error:
            error_type = "rapid_screening_error"
            error_msg = r.rapid_error
        elif r.s1_error:
            error_type = "s1_error"
            error_msg = r.s1_error

        result_id = self.nb.record_program_result(
            experiment_id=self.exp_id,
            graph_fingerprint=r.graph_fingerprint,
            graph_json=r.graph_json,
            bypass_quality_gate=True,
            stage0_passed=r.forward_ok,
            stage05_passed=r.stability_score > 0.5 if r.stability_score else False,
            stage1_passed=r.s1_ok,
            loss_ratio=r.s1_loss_ratio,
            initial_loss=r.s1_initial_loss,
            final_loss=r.s1_final_loss,
            train_budget_steps=self.s1_steps if r.s1_initial_loss is not None else 0,
            param_count=r.param_count,
            graph_n_ops=r.n_ops,
            graph_n_unique_ops=len(set(r.ops_present)),
            stability_score=r.stability_score,
            error_type=error_type,
            error_message=error_msg,
            model_source="forced_exploration",
        )
        if result_id:
            self.n_recorded += 1
        if r.forward_ok:
            self.n_s0_passed += 1
        if r.s1_ok:
            self.n_s1_passed += 1
        if r.s1_loss_ratio is not None:
            if self.best_lr is None or r.s1_loss_ratio < self.best_lr:
                self.best_lr = r.s1_loss_ratio
        self.nb.flush_writes()

    def finalize(self) -> None:
        """Complete the experiment, update op_success_rates, and close."""
        self.nb.flush_writes()
        self.nb.update_op_success_rates(self.exp_id)
        self.nb.flush_writes()
        self.nb.complete_experiment(
            self.exp_id,
            results={
                "total": self.n_recorded,
                "stage0_passed": self.n_s0_passed,
                "stage1_passed": self.n_s1_passed,
                "best_loss_ratio": self.best_lr,
            },
            aria_summary=f"Forced exploration: {self.n_recorded} programs, "
            f"{self.n_s0_passed} S0, {self.n_s1_passed} S1",
        )
        self.nb.flush_writes()
        logger.info(
            "DB recording complete: %d results (experiment=%s)",
            self.n_recorded,
            self.exp_id,
        )
        self.nb.close()


# ── Main orchestrator ────────────────────────────────────────────────


def run_exploration(
    db_path: str,
    mode: str = "forced",
    threshold: int = 20,
    device: str = "cpu",
    n_graphs_weighted: int = 50,
    max_retries_forced: int = 100,
    graphs_per_op: int = 1,
    run_s1: bool = True,
    s1_steps: int = 500,
    rapid_steps: int = 150,
    output_dir: str = "research/reports",
    base_seed: int = 42,
    dry_run: bool = False,
    record: bool = False,
    force_ops: Optional[List[str]] = None,
    model_dim: int = MODEL_DIM,
    n_layers: int = N_LAYERS,
    composition_depth: Optional[int] = None,
    max_depth: Optional[int] = None,
    max_ops: Optional[int] = None,
    boost_factor: float = 50.0,
) -> Tuple[Dict[str, OpCoverage], List[ExplorationResult]]:
    """Main entry point for under-observed component exploration.

    Args:
        db_path: Path to lab_notebook.db
        mode: "weighted" or "forced"
        threshold: Observation count below which ops are targeted
        device: "cpu" or "cuda"
        n_graphs_weighted: Number of graphs for weighted mode
        max_retries_forced: Max retries per op in forced mode
        graphs_per_op: Graphs to generate per target op in forced mode
        run_s1: Whether to run S1 micro-training
        s1_steps: Number of S1 training steps
        rapid_steps: Rapid screening gradient steps (default 150)
        output_dir: Where to write reports
        base_seed: Random seed
        dry_run: If True, only generate graphs — skip eval pipeline
        record: If True, write results to lab_notebook.db (real-time)
        model_dim: Model dimension (default: 256)
        n_layers: Number of layers in compiled model (default: 4)
        composition_depth: Template blocks stacked per graph (default: 2)
        max_depth: Max graph depth (default: 12)
        max_ops: Max ops per graph (default: 18)
        boost_factor: Weight multiplier for target op templates (default: 50.0)
    """
    t_start = time.perf_counter()

    # 1. Discover targets
    if force_ops:
        # Validate requested ops exist in the registry
        # Ensure math-space ops are registered
        try:
            from research.mathspaces.registry import register_all_mathspaces

            register_all_mathspaces()
        except Exception:
            pass
        unknown = [op for op in force_ops if op not in PRIMITIVE_REGISTRY]
        if unknown:
            logger.error(
                "Unknown ops not in PRIMITIVE_REGISTRY: %s", ", ".join(unknown)
            )
            sys.exit(1)
        # Look up current observation counts for reporting
        all_targets = discover_targets(db_path, threshold=999999)
        targets = {op: all_targets.get(op, 0) for op in force_ops}
        logger.info("Forcing exploration of %d specified ops:", len(targets))
    else:
        logger.info("Discovering under-observed ops (threshold=%d)...", threshold)
        targets = discover_targets(db_path, threshold)
    if not targets:
        logger.info(
            "No under-observed ops found. All ops have >= %d observations.", threshold
        )
        return {}, []

    logger.info("Found %d target ops:", len(targets))
    for op, n in sorted(targets.items(), key=lambda x: x[1]):
        logger.info("  %s: %d observations", op, n)

    # 2. Initialize coverage tracking
    coverage: Dict[str, OpCoverage] = {
        op: OpCoverage(op_name=op, n_prior_observations=n) for op, n in targets.items()
    }
    all_results: List[ExplorationResult] = []
    target_set = set(targets.keys())

    # 3. Set up real-time DB recorder if requested
    recorder = None
    if record and not dry_run:
        recorder = _DBRecorder(
            db_path,
            threshold,
            mode,
            device=device,
            s1_steps=s1_steps,
            rapid_steps=rapid_steps,
            model_dim=model_dim,
            n_layers=n_layers,
        )

    eval_kwargs = dict(
        run_s1=run_s1,
        s1_steps=s1_steps,
        rapid_steps=rapid_steps,
        n_layers=n_layers,
    )
    gen_kwargs = dict(
        model_dim=model_dim,
        composition_depth=composition_depth,
        max_depth=max_depth,
        max_ops=max_ops,
        boost_factor=boost_factor,
    )

    # 4. Generate and evaluate graphs
    if mode == "weighted":
        all_results, coverage = _run_weighted_mode(
            targets,
            coverage,
            target_set,
            n_graphs=n_graphs_weighted,
            base_seed=base_seed,
            device=device,
            dry_run=dry_run,
            recorder=recorder,
            **eval_kwargs,
            **gen_kwargs,
        )
    elif mode == "forced":
        all_results, coverage = _run_forced_mode(
            targets,
            coverage,
            target_set,
            max_retries=max_retries_forced,
            base_seed=base_seed,
            device=device,
            dry_run=dry_run,
            graphs_per_op=graphs_per_op,
            recorder=recorder,
            **eval_kwargs,
            **gen_kwargs,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'weighted' or 'forced'.")

    # 5. Second pass: forced coverage for any ops still at zero in weighted mode
    if mode == "weighted":
        uncovered = [op for op, cov in coverage.items() if cov.inserted == 0]
        if uncovered:
            logger.info(
                "Weighted mode left %d ops uncovered. Running forced pass...",
                len(uncovered),
            )
            forced_targets = {op: targets[op] for op in uncovered}
            forced_results, coverage = _run_forced_mode(
                forced_targets,
                coverage,
                set(uncovered),
                max_retries=max_retries_forced,
                base_seed=base_seed + 10000,
                device=device,
                dry_run=dry_run,
                graphs_per_op=graphs_per_op,
                recorder=recorder,
                **eval_kwargs,
                **gen_kwargs,
            )
            all_results.extend(forced_results)

    # 6. Finalize DB recording
    if recorder is not None:
        recorder.finalize()

    if _stop_requested:
        logger.info(
            "Run interrupted — saving partial results (%d graphs evaluated).",
            len(all_results),
        )

    elapsed = time.perf_counter() - t_start

    # 7. Write reports
    md_path, json_path = write_reports(
        coverage,
        all_results,
        output_dir,
        mode,
        threshold,
        elapsed,
    )
    logger.info("Reports written:")
    logger.info("  Markdown: %s", md_path)
    logger.info("  JSON:     %s", json_path)

    # 7. Print summary
    summary = _build_summary(coverage)
    logger.info(
        "Coverage: %d/%d (%.0f%%) | Compile: %d | Forward: %d | Rapid: %d | S1: %d",
        summary["n_covered"],
        summary["n_targets"],
        summary["coverage_rate"] * 100,
        summary["n_compile_pass"],
        summary["n_forward_pass"],
        summary["n_rapid_pass"],
        summary["n_s1_pass"],
    )

    return coverage, all_results


def _run_weighted_mode(
    targets,
    coverage,
    target_set,
    n_graphs,
    base_seed,
    device,
    run_s1,
    s1_steps,
    rapid_steps,
    dry_run,
    recorder=None,
    n_layers=N_LAYERS,
    model_dim=MODEL_DIM,
    composition_depth=None,
    max_depth=None,
    max_ops=None,
    boost_factor=50.0,
):
    """Generate a batch of graphs with boosted weights for under-observed ops."""
    results = []
    logger.info("Generating %d weighted graphs...", n_graphs)
    graphs = generate_weighted_batch(
        targets,
        n_graphs,
        base_seed,
        model_dim=model_dim,
        composition_depth=composition_depth,
        max_depth=max_depth,
        max_ops=max_ops,
    )
    logger.info("Generated %d graphs. Evaluating...", len(graphs))

    for i, graph in enumerate(graphs):
        if _stop_requested:
            logger.info("  Stopping early (%d/%d graphs evaluated).", i, len(graphs))
            break
        present = _ops_in_graph(graph)
        hit_targets = present & target_set
        for op in hit_targets:
            coverage[op].attempted += 1

        if dry_run:
            r = ExplorationResult(
                graph_fingerprint=graph.fingerprint(),
                target_ops=sorted(hit_targets),
                ops_present=sorted(present),
                compile_ok=True,
                forward_ok=True,
            )
        else:
            logger.info(
                "  [%d/%d] Evaluating graph (%d ops, targets: %s)...",
                i + 1,
                len(graphs),
                len(present),
                sorted(hit_targets),
            )
            r = evaluate_graph(
                graph,
                device=device,
                run_s1=run_s1,
                s1_steps=s1_steps,
                rapid_steps=rapid_steps,
                n_layers=n_layers,
            )
            r.target_ops = sorted(hit_targets)

        update_coverage(coverage, graph, r, target_set)
        results.append(r)
        if recorder is not None and not dry_run:
            recorder.record(r)

    return results, coverage


def _run_forced_mode(
    targets,
    coverage,
    target_set,
    max_retries,
    base_seed,
    device,
    run_s1,
    s1_steps,
    rapid_steps,
    dry_run,
    graphs_per_op=1,
    recorder=None,
    n_layers=N_LAYERS,
    model_dim=MODEL_DIM,
    composition_depth=None,
    max_depth=None,
    max_ops=None,
    boost_factor=50.0,
):
    """Generate dedicated graphs per target op.

    Args:
        graphs_per_op: How many distinct graphs to generate per target op.
            Each uses a different seed for diversity.
    """
    results = []
    n = len(targets)
    total = n * graphs_per_op
    graph_idx = 0
    for idx, (op_name, n_obs) in enumerate(sorted(targets.items())):
        if _stop_requested:
            logger.info("  Stopping early (%d/%d ops explored).", idx, n)
            break
        seen_fingerprints: set = set()
        for gpo in range(graphs_per_op):
            if _stop_requested:
                break
            graph_idx += 1
            coverage[op_name].attempted += 1
            label = f"  [{graph_idx}/{total}] Forcing {op_name}" + (
                f" (run {gpo + 1}/{graphs_per_op})" if graphs_per_op > 1 else ""
            )
            logger.info("%s (prior obs: %d)...", label, n_obs)

            gen_result = generate_forced_graph(
                op_name,
                seed=base_seed + idx * 137 + gpo * 7919,
                max_retries=max_retries,
                model_dim=model_dim,
                boost_factor=boost_factor,
                composition_depth=composition_depth,
                max_depth=max_depth,
                max_ops=max_ops,
                seen_fingerprints=seen_fingerprints,
            )
            graph, retries = gen_result

            if graph is None:
                prim = PRIMITIVE_REGISTRY.get(op_name)
                if prim is None:
                    reason = "not in PRIMITIVE_REGISTRY"
                elif getattr(prim, "n_inputs", 1) > 1:
                    reason = f"binary op (n_inputs={prim.n_inputs}) — requires structural wiring"
                else:
                    motifs = _find_motifs_containing_op(op_name)
                    if not motifs:
                        reason = "no motif contains this op"
                    else:
                        reason = f"in motifs {motifs} but shape/space constraints prevent placement after {max_retries} attempts"
                coverage[op_name].skip_reasons.append(reason)
                logger.warning("    SKIP %s: %s", op_name, reason)
                continue

            present = _ops_in_graph(graph)
            if op_name not in present:
                coverage[op_name].skip_reasons.append(
                    f"generated graph after {retries} retries but op pruned (unreachable)"
                )
                logger.warning("    %s generated but pruned from graph", op_name)
                continue

            if dry_run:
                r = ExplorationResult(
                    graph_fingerprint=graph.fingerprint(),
                    target_ops=[op_name],
                    ops_present=sorted(present),
                    compile_ok=True,
                    forward_ok=True,
                )
            else:
                logger.info(
                    "    Evaluating (retries=%d, %d ops)...", retries, len(present)
                )
                r = evaluate_graph(
                    graph,
                    device=device,
                    run_s1=run_s1,
                    s1_steps=s1_steps,
                    rapid_steps=rapid_steps,
                    n_layers=n_layers,
                )
                r.target_ops = [op_name]

            update_coverage(coverage, graph, r, target_set)
            results.append(r)
            if recorder is not None and not dry_run:
                recorder.record(r)

        if graphs_per_op > 1:
            logger.info(
                "    %s: %d unique graphs generated (dedup tracked %d fingerprints)",
                op_name,
                len(seen_fingerprints),
                len(seen_fingerprints),
            )

    return results, coverage


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Explore under-observed components end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["weighted", "forced"],
        default="forced",
        help="weighted = boost under-observed ops; forced = one graph per op (default: forced)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=20,
        help="Observation threshold (default: 20)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for evaluation (default: cpu)",
    )
    parser.add_argument(
        "--n-graphs",
        type=int,
        default=50,
        help="Number of graphs for weighted mode (default: 50)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=100,
        help="Max retries per op in forced mode (default: 20)",
    )
    parser.add_argument(
        "--graphs-per-op",
        type=int,
        default=1,
        help="Graphs to generate per target op in forced mode (default: 1)",
    )
    parser.add_argument(
        "--rapid-steps",
        type=int,
        default=150,
        help="Rapid screening gradient steps (default: 150)",
    )
    parser.add_argument(
        "--no-s1",
        action="store_true",
        help="Skip S1 micro-training (faster, less thorough)",
    )
    parser.add_argument(
        "--s1-steps",
        type=int,
        default=500,
        help="S1 training steps (default: 500)",
    )
    parser.add_argument(
        "--output-dir",
        default="research/reports",
        help="Output directory for reports (default: research/reports)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--db",
        default="research/lab_notebook.db",
        help="Path to lab_notebook.db (default: research/lab_notebook.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate graphs, skip pipeline evaluation",
    )
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record results to lab_notebook.db (default: on)",
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        metavar="OP",
        help="Force exploration of specific ops by name (ignores --threshold). "
        "Example: --ops softmax_attention linear_proj chebyshev_spectral_mix",
    )
    parser.add_argument(
        "--max-ops",
        type=int,
        default=None,
        metavar="N",
        help="Max ops (components) per graph. Default: 18 (exploration mode)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        metavar="N",
        help="Max graph depth (longest path from input to output). Default: 12",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=None,
        metavar="N",
        help="Template blocks stacked per graph. Each block expands to ~3-8 ops. Default: 2",
    )
    parser.add_argument(
        "--model-dim",
        type=int,
        default=MODEL_DIM,
        metavar="D",
        help=f"Model dimension (default: {MODEL_DIM})",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=N_LAYERS,
        metavar="N",
        help=f"Layers in compiled model (default: {N_LAYERS})",
    )
    parser.add_argument(
        "--boost-factor",
        type=float,
        default=50.0,
        metavar="F",
        help="Template weight multiplier for target ops in forced mode (default: 50.0)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    signal.signal(signal.SIGINT, _handle_sigint)

    coverage, results = run_exploration(
        db_path=args.db,
        mode=args.mode,
        threshold=args.threshold,
        device=args.device,
        n_graphs_weighted=args.n_graphs,
        max_retries_forced=args.max_retries,
        graphs_per_op=args.graphs_per_op,
        run_s1=not args.no_s1,
        s1_steps=args.s1_steps,
        rapid_steps=args.rapid_steps,
        output_dir=args.output_dir,
        base_seed=args.seed,
        dry_run=args.dry_run,
        record=args.record,
        force_ops=args.ops,
        model_dim=args.model_dim,
        n_layers=args.n_layers,
        composition_depth=args.n_blocks,
        max_depth=args.max_depth,
        max_ops=args.max_ops,
        boost_factor=args.boost_factor,
    )

    # Exit code: 0 if all targets covered, 1 if any uncovered
    uncovered = [op for op, cov in coverage.items() if cov.inserted == 0]
    if uncovered:
        logger.warning(
            "%d ops could not be covered: %s",
            len(uncovered),
            ", ".join(sorted(uncovered)),
        )
        sys.exit(1)
    else:
        logger.info("All %d target ops covered.", len(coverage))
        sys.exit(0)


if __name__ == "__main__":
    main()
