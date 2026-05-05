"""Experiment and operational context builders for LLM prompts."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from ._op_registry import grouped_primitive_registry, primitive_registry_size

logger = logging.getLogger(__name__)

_OP_REGISTRY_CACHE: Optional[str] = None


def build_op_reference(
    op_success_rates: Optional[Dict] = None, compression_coverage: Optional[Dict] = None
) -> str:
    """Build a compact op reference for injection into config-producing prompts.

    Combines the primitive registry (valid names) with op success rates so
    the LLM knows exactly which op names exist and how they perform.
    This MUST NOT go through analyst compression — inject it directly.

    *compression_coverage*: output of analytics.compression_coverage(), used
    to add quality retention and compression ratio per technique.
    """
    lines = ["VALID OP NAMES — you MUST only use these exact names in op_weights:"]

    # Registry by category
    try:
        for category, ops in grouped_primitive_registry():
            lines.append(f"  {category}: {', '.join(ops)}")
    except Exception as exc:
        logger.debug("Suppressed error: %s", exc)

    # Success rates for ops with enough data
    if op_success_rates:
        rated = sorted(op_success_rates.items(), key=lambda x: -x[1].get("s1_rate", 0))
        # Routing ops
        routing = [
            (n, s)
            for n, s in rated
            if any(k in n for k in ("gate", "routing", "moe", "router", "expert"))
        ]
        # Compression/efficiency ops
        compress = [
            (n, s)
            for n, s in rated
            if any(
                k in n
                for k in (
                    "sparse",
                    "low_rank",
                    "grouped",
                    "shared_basis",
                    "semi_structured",
                    "nm_sparse",
                    "block_sparse",
                    "factorized",
                    "bottleneck",
                    "tied_proj",
                )
            )
        ]
        if routing:
            lines.append(
                "  Routing ops (S1 rate): "
                + ", ".join(f"{n} ({s.get('s1_rate', 0):.0%})" for n, s in routing)
            )
        if compress:
            lines.append(
                "  Compression ops (S1 rate): "
                + ", ".join(f"{n} ({s.get('s1_rate', 0):.0%})" for n, s in compress)
            )
        # Top 10 overall
        top10 = [(n, s) for n, s in rated if s.get("n_used", 0) >= 20][:10]
        if top10:
            lines.append(
                "  Top 10 by S1 rate (n>=20): "
                + ", ".join(f"{n} ({s.get('s1_rate', 0):.0%})" for n, s in top10)
            )

    # Compression technique quality retention (from analytics)
    if compression_coverage:
        dense_markers = {"dense", "dense_matrix", "standard_float"}
        techniques = compression_coverage.get("techniques") or []
        tech_parts = []
        for tech in techniques:
            name = tech.get("technique", "")
            if name in dense_markers or tech.get("n_tested", 0) < 3:
                continue
            qr = tech.get("avg_quality_retention")
            sr = tech.get("survival_rate", 0)
            parts = [f"{name}: {sr:.0%} survival"]
            if qr is not None:
                parts.append(f"quality={qr:.2f}")
            cr = tech.get("avg_compression_ratio")
            if cr is not None:
                parts.append(f"{cr:.1f}x compression")
            tech_parts.append(", ".join(parts))
        if tech_parts:
            lines.append("  Compression technique performance:")
            for tp in tech_parts:
                lines.append(f"    {tp}")

    lines.append(
        "WARNING: Do NOT invent op names. If an op name is not in the list above, it does not exist."
    )
    return "\n".join(lines)


def _build_op_registry_section() -> str:
    """Build a compact category→ops listing from the primitive registry.

    Cached after first call since the registry is static within a process.
    """
    global _OP_REGISTRY_CACHE
    if _OP_REGISTRY_CACHE is not None:
        return _OP_REGISTRY_CACHE
    try:
        lines = [
            f"Available Ops ({primitive_registry_size()} total, use op_weights to control selection):"
        ]
        for category, ops in grouped_primitive_registry():
            lines.append(f"  {category} ({len(ops)}): {', '.join(ops)}")
        _OP_REGISTRY_CACHE = "\n".join(lines)
    except Exception as exc:
        logger.debug("Falling back to default: %s", exc)
        _OP_REGISTRY_CACHE = ""
    return _OP_REGISTRY_CACHE


def build_experiment_context(
    results: Dict, config: Optional[Dict] = None, hypothesis: Optional[str] = None
) -> str:
    """Build context for a single experiment's results."""
    lines = []

    if hypothesis:
        lines.append(f"Hypothesis: {hypothesis}")

    if config:
        lines.append(
            f"Config: {config.get('n_programs', '?')} programs, "
            f"dim={config.get('model_dim', '?')}, "
            f"depth={config.get('max_depth', '?')}, "
            f"ops={config.get('max_ops', '?')}, "
            f"math_space_weight={config.get('math_space_weight', '?')}"
        )

    total = results.get("total", 0)
    s0 = results.get("stage0_passed", 0)
    s05 = results.get("stage05_passed", 0)
    s1 = results.get("stage1_passed", 0)
    novel = results.get("novel_count", 0)

    lines.append(
        f"\nFunnel: {total} generated -> {s0} S0 ({_pct(s0, total)}) "
        f"-> {s05} S0.5 ({_pct(s05, total)}) "
        f"-> {s1} S1 ({_pct(s1, total)})"
    )
    lines.append(f"Novel survivors (novelty > 0.5): {novel}")

    best_val = results.get("best_validation_loss_ratio")
    best_disc = results.get("best_discovery_loss_ratio")

    if best_val is not None:
        lines.append(f"Best validation loss ratio: {best_val:.4f}")
    if best_disc is not None:
        lines.append(f"Best discovery loss ratio: {best_disc:.4f}")

    if (
        best_val is None
        and best_disc is None
        and results.get("best_loss_ratio") is not None
    ):
        lines.append(f"Best loss ratio: {results['best_loss_ratio']:.4f}")
    if results.get("best_novelty_score") is not None:
        lines.append(f"Best novelty: {results['best_novelty_score']:.3f}")

    validation_results = [
        entry
        for entry in (results.get("validation_results") or [])
        if isinstance(entry, dict)
    ]
    if validation_results:
        passed = results.get("validation_passed_count")
        if passed is None:
            passed = sum(
                1
                for entry in validation_results
                if int(entry.get("seeds_passed") or 0) > 0
            )
        breakthroughs = results.get("breakthrough_count")
        if breakthroughs is None:
            breakthroughs = sum(
                1 for entry in validation_results if bool(entry.get("is_breakthrough"))
            )
        novel_validated = results.get("novel_count", 0)
        lines.append(
            "\nValidation candidates: "
            f"{passed}/{len(validation_results)} passed multi-seed validation; "
            f"{breakthroughs} breakthrough; {novel_validated} with novelty_score > 0.5."
        )
        for entry in validation_results[:5]:
            loss = entry.get("val_loss_ratio")
            loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
            novelty = entry.get("novelty_score")
            novelty_str = (
                f", novelty={novelty:.3f}" if isinstance(novelty, (int, float)) else ""
            )
            baseline = entry.get("val_baseline_ratio")
            baseline_str = (
                f", baseline_ratio={baseline:.4f}"
                if isinstance(baseline, (int, float))
                else ""
            )
            tier = "breakthrough" if entry.get("is_breakthrough") else "validation"
            lines.append(
                f"  - {entry.get('result_id', '?')[:12]}: tier={tier}, "
                f"val_loss_ratio={loss_str}{baseline_str}{novelty_str}, "
                f"seeds={entry.get('seeds_passed', 0)}/{entry.get('total_seeds', 0)}, "
                f"robustness={entry.get('robustness_score', 0):.3f}"
            )

    survivors = results.get("survivors", [])
    if survivors:
        lines.append("\nTop survivors:")
        for s in survivors[:5]:
            loss = s.get("validation_loss_ratio")
            loss_label = "val_loss_ratio"
            if loss is None:
                loss = s.get("loss_ratio", 0)
                loss_label = "loss_ratio"
            ncd = s.get("ncd_score")
            ncd_str = f", ncd={ncd:.3f}" if ncd is not None else ""
            lines.append(
                f"  - {s['fingerprint'][:12]}: "
                f"novelty={s.get('novelty', 0):.3f}, "
                f"{loss_label}={loss:.4f}{ncd_str}"
            )

    return "\n".join(lines)


def build_history_context(experiments: List[Dict], limit: int = 10) -> str:
    """Build context from recent experiment history."""
    lines = ["Recent experiment history:"]

    for exp in experiments[:limit]:
        status = exp.get("status", "?")
        s1 = exp.get("n_stage1_passed", 0)
        total = exp.get("n_programs_generated", 0)
        novelty = exp.get("best_novelty_score")
        val_loss = exp.get("best_validation_loss_ratio")
        disc_loss = exp.get("best_discovery_loss_ratio")
        legacy_loss = exp.get("best_loss_ratio")

        mood = exp.get("aria_mood", "?")

        line = f"  [{status}] {total} programs, {s1} S1 pass"
        if novelty is not None:
            line += f", novelty={novelty:.3f}"
        if val_loss is not None:
            line += f", val_lr={val_loss:.4f}"
        if disc_loss is not None:
            line += f", disc_lr={disc_loss:.4f}"
        if val_loss is None and disc_loss is None and legacy_loss is not None:
            line += f", lr={legacy_loss:.4f}"

        line += f" (mood: {mood})"
        lines.append(line)

    return "\n".join(lines)


def build_program_context(program: Dict) -> str:
    """Build context for a single program's detail."""
    lines = []

    fp = program.get("graph_fingerprint", "unknown")
    lines.append(f"Program fingerprint: {fp}")

    stages = []
    if program.get("stage0_passed"):
        stages.append("S0:PASS")
    else:
        stages.append(f"S0:FAIL ({program.get('stage0_error', 'unknown error')})")
    if program.get("stage05_passed"):
        stages.append("S0.5:PASS")
    if program.get("stage1_passed"):
        stages.append("S1:PASS")
    lines.append(f"Pipeline: {' -> '.join(stages)}")

    if program.get("param_count"):
        lines.append(f"Parameters: {program['param_count']:,}")

    # Priority: Validation > Discovery > Legacy
    val_lr = program.get("validation_loss_ratio")
    disc_lr = program.get("discovery_loss_ratio")
    legacy_lr = program.get("loss_ratio")

    if val_lr is not None:
        lines.append(f"Validation loss ratio: {val_lr:.4f} (primary truth)")
    if disc_lr is not None:
        lines.append(f"Discovery loss ratio: {disc_lr:.4f} (random tokens)")
    if val_lr is None and disc_lr is None and legacy_lr is not None:
        lines.append(f"Loss ratio: {legacy_lr:.4f} (legacy/mixed)")

    if program.get("generalization_gap") is not None:
        lines.append(f"Generalization gap: {program['generalization_gap']:.4f}")
    if program.get("novelty_score") is not None:
        lines.append(
            f"Novelty: {program['novelty_score']:.3f} "
            f"(structural={program.get('structural_novelty', 0):.3f}, "
            f"behavioral={program.get('behavioral_novelty', 0):.3f})"
        )
    if program.get("most_similar_to"):
        lines.append(f"Most similar to: {program['most_similar_to']}")

    return "\n".join(lines)


def _build_rich_op_success_rates_section(analytics_data: Dict) -> Optional[str]:
    op_rates = analytics_data.get("op_success_rates", {})
    if not op_rates:
        return None
    rated = sorted(op_rates.items(), key=lambda x: -x[1].get("s1_rate", 0))
    best = rated[:5]
    worst = rated[-5:] if len(rated) > 5 else []
    lines = ["Op Success Rates (S1):"]
    if best:
        lines.append("  Best:")
        for op, s in best:
            lines.append(
                f"    {op}: S1={s.get('s1_rate', 0):.0%} "
                f"(n={s.get('n_used', 0)}, "
                f"novelty={s.get('avg_novelty') or 0:.3f})"
            )
    if worst:
        lines.append("  Worst:")
        for op, s in worst:
            lines.append(
                f"    {op}: S1={s.get('s1_rate', 0):.0%} (n={s.get('n_used', 0)})"
            )
    return "\n".join(lines)


def _build_rich_structural_correlations_section(analytics_data: Dict) -> Optional[str]:
    correlations = analytics_data.get("structural_correlations", {})
    if not correlations:
        return None
    lines = ["Structural Correlations with S1 Success:"]
    for metric, effect in sorted(correlations.items(), key=lambda x: -abs(x[1])):
        name = metric.replace("graph_", "").replace("_", " ")
        direction = "+" if effect > 0 else "-"
        lines.append(f"  {name}: {direction}{abs(effect):.2f}")
    return "\n".join(lines)


def _build_rich_failure_patterns_section(analytics_data: Dict) -> Optional[str]:
    failures = analytics_data.get("failure_patterns", {})
    if not failures:
        return None
    lines = ["Failure Patterns (error_type x stage):"]
    for err_type, info in sorted(failures.items(), key=lambda x: -x[1].get("total", 0))[
        :5
    ]:
        stages = ", ".join(f"{s}:{c}" for s, c in info.get("by_stage", {}).items())
        lines.append(f"  {err_type}: {info.get('total', 0)} total ({stages})")
    return "\n".join(lines)


def _build_rich_top_op_combinations_section(analytics_data: Dict) -> Optional[str]:
    combos = analytics_data.get("top_op_combinations", [])
    if not combos:
        return None
    lines = ["Top Op Combinations (S1 survivors):"]
    for c in combos[:5]:
        ops = " + ".join(c.get("ops", []))
        lines.append(
            f"  {ops}: {c.get('count', 0)}x (avg novelty {c.get('avg_novelty', 0):.3f})"
        )
    return "\n".join(lines)


def _build_rich_efficiency_frontier_section(analytics_data: Dict) -> Optional[str]:
    frontier = analytics_data.get("efficiency_frontier", [])
    if not frontier:
        return None
    lines = [f"Efficiency Frontier: {len(frontier)} Pareto-optimal programs"]
    best_loss = min(
        (p.get("final_loss") for p in frontier if p.get("final_loss") is not None),
        default=None,
    )
    if best_loss is not None:
        lines.append(f"  Best loss on frontier: {best_loss:.4f}")
    most_eff = min(
        (p for p in frontier if p.get("flops_forward")),
        key=lambda p: p["flops_forward"],
        default=None,
    )
    if most_eff:
        lines.append(
            f"  Most efficient: {most_eff.get('flops_forward', 0):.0f} FLOPs, "
            f"loss={most_eff.get('final_loss', 0):.4f}"
        )
    return "\n".join(lines)


def _build_rich_scaling_gate_section(analytics_data: Dict) -> str:
    scaling = analytics_data.get("scaling_summary", {})
    n_eval = scaling.get("n_evaluated", 0)
    if n_eval <= 0:
        return (
            "SCALING GATE — External Baseline Comparison:\n"
            "  No candidates evaluated yet against GPT-2/Mamba scaling laws.\n"
            "  Goal: achieve 3x parameter efficiency vs standard transformer.\n"
            "  Candidates that pass validation will be compared automatically."
        )
    n_pass = scaling.get("n_gate_passed", 0)
    target = scaling.get("target", 3.0)
    best_eff = scaling.get("best_param_efficiency", 0)
    mean_eff = scaling.get("mean_param_efficiency", 0)
    lines = [
        "SCALING GATE — External Baseline Comparison (CRITICAL):",
        f"  Goal: Architectures must use {target:.0f}x FEWER parameters than GPT-2 for the same loss.",
        f"  {n_eval} candidates evaluated, {n_pass} passed the {target:.0f}x gate.",
        f"  Best param efficiency: {best_eff:.2f}x (need {target:.0f}x)  Mean: {mean_eff:.2f}x",
    ]
    if n_pass == 0:
        gap = target - best_eff
        lines.append(
            f"  *** NO CANDIDATES PASS THE GATE. Best is {gap:.1f}x short of target. ***"
        )
        lines.append(
            "  This means: current architectures are NOT more parameter-efficient than a standard transformer."
        )
        lines.append(
            "  Priority: find architectures that achieve the SAME loss with FEWER parameters."
        )
        lines.append(
            "  Strategies: MoE routing (only activate subset of params), "
            "aggressive sparsity, weight sharing, "
            "sublinear attention, or fundamentally different compute patterns."
        )
    best_e = scaling.get("best_entry", {})
    if best_e:
        lines.append(
            f"  Current best: {best_e.get('fingerprint', '??')} "
            f"({best_e.get('param_efficiency', 0):.2f}x vs {best_e.get('family', 'gpt2')}, "
            f"loss_ratio={best_e.get('loss_ratio', '?')})"
        )
    top_entries = scaling.get("entries", [])
    if len(top_entries) > 1:
        lines.append("  Top evaluated candidates:")
        for e in top_entries[:5]:
            gate_str = "PASS" if e.get("gate") else "FAIL"
            lines.append(
                f"    {e.get('fingerprint', '??')}: "
                f"{e.get('param_eff', 0):.2f}x param, "
                f"{e.get('flop_eff', 0):.2f}x flop, "
                f"lr={e.get('loss_ratio', '?'):.4f} [{gate_str}]"
            )
    return "\n".join(lines)


def _build_rich_grammar_weights_section(analytics_data: Dict) -> Optional[str]:
    grammar_weights = analytics_data.get("grammar_weights")
    default_weights = analytics_data.get("default_weights", {})
    if not (grammar_weights and default_weights):
        return None
    lines = ["Grammar Weights (learned vs default):"]
    for cat in sorted(set(list(grammar_weights.keys()) + list(default_weights.keys()))):
        learned = grammar_weights.get(cat, "—")
        default = default_weights.get(cat, 1.0)
        if isinstance(learned, (int, float)):
            delta = learned - default
            arrow = "^" if delta > 0.1 else ("v" if delta < -0.1 else "=")
            lines.append(f"  {cat}: {default:.1f} -> {learned:.1f} [{arrow}]")
    lines.append("  (Set category_weights in CONFIG to override any of these)")
    return "\n".join(lines)


def _build_rich_learning_log_section(analytics_data: Dict) -> Optional[str]:
    learning_log = analytics_data.get("learning_log", [])
    if not learning_log:
        return None
    lines = [f"Recent Learning Events ({len(learning_log)} most recent):"]
    for entry in learning_log[:5]:
        lines.append(
            f"  [{entry.get('event_type', '?')}] {entry.get('description', '')[:80]}"
        )
    return "\n".join(lines)


def _build_rich_gate_health_section(analytics_data: Dict) -> Optional[str]:
    gate_health = analytics_data.get("gate_health", {})
    gate_summary = gate_health.get("summary", {})
    if not gate_summary:
        return None
    lines = ["Causality Gate Health:"]
    lines.append(f"  Pass rate: {gate_summary.get('stage05_pass_rate', 0):.1%}")
    lines.append(f"  Violations: {gate_summary.get('causality_violations', 0)}")
    corr = gate_summary.get("discovery_validation_correlation")
    if corr is not None:
        lines.append(
            f"  Discovery-Validation correlation: {corr:.3f} "
            f"(n={gate_summary.get('n_correlation_samples', 0)})"
        )
    gate_daily = gate_health.get("daily", [])
    if gate_daily:
        recent = gate_daily[-3:]
        lines.append("  Recent daily gate failure rates:")
        for d in recent:
            lines.append(
                f"    {d['date']}: {d['gate_failure_rate']:.1%} "
                f"({d['models_screened']} screened)"
            )
    return "\n".join(lines)


def _build_rich_active_insights_section(analytics_data: Dict) -> Optional[str]:
    insights = analytics_data.get("insights", [])
    if not insights:
        return None
    lines = ["Active Insights:"]
    for ins in insights[:10]:
        cat = ins.get("category", "general")
        content = ins.get("content", "")[:120]
        conf = ins.get("confidence", 0)
        lines.append(f"  [{cat}] (conf={conf:.1f}) {content}")
    return "\n".join(lines)


def _build_rich_negative_results_section(analytics_data: Dict) -> Optional[str]:
    neg = analytics_data.get("negative_results", {})
    neg_lines = []
    failed_ops = neg.get("failed_ops", [])
    if failed_ops:
        neg_lines.append("Negative Results — Consistently Failing Patterns:")
        for op in failed_ops[:8]:
            neg_lines.append(
                f"  AVOID {op['op_name']}: 0% S1 rate over {op.get('n_used', '?')} "
                f"uses, fails at {op.get('failure_stage', '?')} "
                f"(confidence={op.get('confidence', 0):.2f})"
            )
        neg_lines.append("  (Use op_weights with values <1.0 to penalize these)")
    anti_patterns = neg.get("anti_patterns", [])
    if anti_patterns:
        neg_lines.append(
            "Negative Results — Anti-Patterns:"
            if not neg_lines
            else "  Anti-correlated features:"
        )
        for ap in anti_patterns[:5]:
            neg_lines.append(
                f"  {ap.get('feature', '?')}: correlation={ap.get('correlation', 0):.3f} "
                f"— {ap.get('interpretation', '')}"
            )
    refuted_hyps = neg.get("refuted_hypotheses", [])
    if refuted_hyps:
        if not neg_lines:
            neg_lines.append("Negative Results — Refuted Hypotheses:")
        else:
            neg_lines.append(
                "  Refuted hypotheses (DO NOT re-test similar directions):"
            )
        for rh in refuted_hyps[:5]:
            content = rh.get("content", "")[:120]
            conf = rh.get("confidence", 0)
            neg_lines.append(f"  REFUTED (conf={conf:.2f}): {content}")
    summary = neg.get("summary", "")
    if summary:
        neg_lines.append(f"  Summary: {summary[:200]}")
    return "\n".join(neg_lines) if neg_lines else None


def _build_rich_designer_telemetry_section(analytics_data: Dict) -> Optional[str]:
    designer = analytics_data.get("designer_telemetry", {})
    if not designer:
        return None
    d_lines = ["Designer Integration:"]
    gap = designer.get("bridge_gap_report", {})
    if gap:
        d_lines.append(
            f"  Bridge gap: {gap.get('unsupported_components', 0)} "
            f"of {gap.get('total_components', 0)} components unsupported"
        )
        gaps = gap.get("gaps", [])
        if gaps:
            d_lines.append(
                "  Unsupported: "
                + ", ".join(g.get("component_id", "?") for g in gaps[:8])
            )
    blocks = [b for b in designer.get("builtin_blocks", []) if b]
    if blocks:
        d_lines.append(f"  Available block templates: {', '.join(blocks)}")
    return "\n".join(d_lines)


def _append_rich_analytics_sections(sections: List[str], analytics_data: Dict) -> None:
    for section in [
        _build_rich_op_success_rates_section(analytics_data),
        _build_op_registry_section(),
        _build_rich_structural_correlations_section(analytics_data),
        _build_rich_failure_patterns_section(analytics_data),
        _build_rich_top_op_combinations_section(analytics_data),
        _build_rich_efficiency_frontier_section(analytics_data),
        _build_rich_scaling_gate_section(analytics_data),
        _build_rich_grammar_weights_section(analytics_data),
        _build_rich_learning_log_section(analytics_data),
        _build_rich_gate_health_section(analytics_data),
        _build_rich_active_insights_section(analytics_data),
        _build_rich_negative_results_section(analytics_data),
        _build_rich_designer_telemetry_section(analytics_data),
    ]:
        if section:
            sections.append(section)


def _build_past_hypotheses_section(past_hypotheses: List[Dict]) -> str:
    lines = ["Past Hypothesis Outcomes:"]
    for h in past_hypotheses[:10]:
        status = "CONFIRMED" if h.get("confirmed") else "REFUTED"
        source = h.get("source", "")
        label = f"[{status}]"
        if source == "refuted_insight":
            label = "[REFUTED INSIGHT — avoid similar directions]"
        lines.append(f"  {label} {h.get('hypothesis', '?')[:80]}")
        if h.get("s1_count") is not None:
            lines.append(
                f"    S1 passes: {h['s1_count']}, "
                f"novelty: {h.get('best_novelty', 0):.3f}"
            )
        if source == "refuted_insight" and h.get("evidence"):
            lines.append(f"    Evidence: {h['evidence'][:100]}")
    return "\n".join(lines)


def build_rich_context(
    results: Dict,
    config: Optional[Dict] = None,
    hypothesis: Optional[str] = None,
    analytics_data: Optional[Dict] = None,
    history: Optional[List[Dict]] = None,
    past_hypotheses: Optional[List[Dict]] = None,
    digest: Optional[object] = None,
) -> str:
    """Build comprehensive context including all analytics data.

    Aggregates experiment results, history, op success rates, structural
    correlations, failure patterns, efficiency frontier, grammar weights,
    learning log, and past hypothesis outcomes into a single context string.

    If *digest* (an ExperimentDigest) is provided, a compact knowledge
    section is appended.
    """
    sections = [build_experiment_context(results, config, hypothesis)]
    if history:
        sections.append(build_history_context(history))
    if analytics_data:
        _append_rich_analytics_sections(sections, analytics_data)
    if past_hypotheses:
        sections.append(_build_past_hypotheses_section(past_hypotheses))
    if analytics_data:
        delta_lines = _build_session_delta(analytics_data, history)
        if delta_lines:
            sections.append("\n".join(delta_lines))
        hf = analytics_data.get("hierarchy_fitness")
        if hf is not None and hf > 0.6:
            sections.append(
                f"HIERARCHY DETECTED (fitness={hf:.3f}): "
                f"Representations show tree-like structure. "
                f"Consider hyperbolic ops (poincare_add, exp_map, log_map, "
                f"hyp_linear, hyp_tangent_nonlinear, hyperbolic_norm) "
                f"which are geometrically better suited for hierarchical data."
            )
    if digest is not None:
        inject_digest_context(sections, digest)
    return "\n\n".join(sections)


def _build_session_delta(
    analytics_data: Dict,
    history: Optional[List[Dict]] = None,
) -> List[str]:
    """Build a 'what changed recently' section to prevent stale repetition.

    Highlights delta information so the LLM focuses on new observations
    rather than repeating the same analysis every cycle.
    """
    lines: List[str] = []
    lines.append(
        "Session Delta (focus on NEW information, avoid repeating old observations):"
    )

    # Recent experiment outcomes — summarize the last few
    if history:
        recent = history[:5]
        new_s1 = sum(1 for e in recent if int(e.get("stage1_passed") or 0) > 0)
        total_recent = len(recent)
        modes = [str(e.get("experiment_type") or "synthesis") for e in recent]
        lines.append(
            f"  Last {total_recent} experiments: {new_s1} produced S1 survivors, "
            f"modes: {', '.join(modes)}"
        )
        # Flag if all recent experiments had zero S1
        if new_s1 == 0 and total_recent >= 3:
            lines.append(
                "  WARNING: No new S1 survivors in recent experiments — "
                "current approach may be saturated. Consider changing strategy."
            )

    # Grammar weight movement
    grammar_weights = analytics_data.get("grammar_weights") or {}
    default_weights = analytics_data.get("default_weights") or {}
    if grammar_weights and default_weights:
        big_movers = []
        for cat in grammar_weights:
            learned = grammar_weights.get(cat, 1.0)
            default = default_weights.get(cat, 1.0)
            if isinstance(learned, (int, float)) and isinstance(default, (int, float)):
                ratio = learned / max(default, 0.01)
                if ratio > 2.0 or ratio < 0.5:
                    big_movers.append(
                        f"{cat} ({default:.1f}->{learned:.1f}, "
                        f"{'boosted' if ratio > 1 else 'suppressed'})"
                    )
        if big_movers:
            lines.append(f"  Grammar movers: {'; '.join(big_movers[:5])}")
        else:
            lines.append("  Grammar weights: stable (no large shifts)")

    # Sparsity/compression coverage progress
    sparse_coverage = analytics_data.get("sparse_coverage") or {}
    n_sparse = int(sparse_coverage.get("n_sparse_tested") or 0)
    if n_sparse > 0:
        sparse_surv = sparse_coverage.get("sparse_survival_rate", 0)
        lines.append(
            f"  Sparse coverage: {n_sparse} programs tested, "
            f"{sparse_surv:.1%} survival rate"
        )

    if len(lines) <= 1:
        return []  # Nothing interesting to report
    return lines


def build_investigation_context(candidates: list, leaderboard: list) -> str:
    """Build context for investigation phase LLM prompts."""
    sections = []

    sections.append(
        f"Investigation Phase: {len(candidates)} candidates selected for deep study"
    )

    for i, c in enumerate(candidates[:10]):
        lines = [f"\nCandidate {i + 1}:"]
        lines.append(f"  Result ID: {c.get('result_id', '?')[:12]}")
        lines.append(f"  Source: {c.get('model_source', 'graph_synthesis')}")
        if c.get("architecture_desc"):
            lines.append(f"  Architecture: {c['architecture_desc']}")
        if c.get("loss_ratio") is not None:
            lines.append(f"  Screening loss ratio: {c['loss_ratio']:.4f}")
        if c.get("validation_loss_ratio") is not None:
            lines.append(f"  Validation loss ratio: {c['validation_loss_ratio']:.4f}")
        if c.get("novelty_score") is not None:
            lines.append(f"  Screening novelty: {c['novelty_score']:.3f}")
        if c.get("most_similar_to"):
            lines.append(f"  Most similar to: {c['most_similar_to']}")
        if c.get("param_count"):
            lines.append(f"  Parameters: {c['param_count']:,}")
        sections.append("\n".join(lines))

    if leaderboard:
        lines = [f"\nLeaderboard context ({len(leaderboard)} total entries):"]
        tier_counts: dict = {}
        for entry in leaderboard:
            t = entry.get("tier", "screening")
            tier_counts[t] = tier_counts.get(t, 0) + 1
        for t, count in sorted(tier_counts.items()):
            lines.append(f"  {t}: {count} entries")
        best = max(leaderboard, key=lambda x: x.get("composite_score", 0), default=None)
        if best:
            lines.append(
                f"  Best composite score: {best.get('composite_score', 0):.3f}"
            )
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def build_validation_context(candidates: list, investigation_results: list) -> str:
    """Build context for validation phase LLM prompts."""
    sections = []

    sections.append(f"Validation Phase: {len(candidates)} candidates for final testing")

    for i, c in enumerate(candidates[:5]):
        lines = [f"\nCandidate {i + 1}:"]
        lines.append(f"  Result ID: {c.get('result_id', '?')[:12]}")
        lines.append(f"  Source: {c.get('model_source', 'graph_synthesis')}")
        if c.get("architecture_desc"):
            lines.append(f"  Architecture: {c['architecture_desc']}")
        if c.get("investigation_loss_ratio") is not None:
            lines.append(
                f"  Investigation loss ratio: {c['investigation_loss_ratio']:.4f}"
            )
        if c.get("validation_loss_ratio") is not None:
            lines.append(f"  Validation loss ratio: {c['validation_loss_ratio']:.4f}")
        if c.get("investigation_robustness") is not None:
            lines.append(f"  Robustness: {c['investigation_robustness']:.2f}")
        if c.get("screening_loss_ratio") is not None:
            lines.append(f"  Screening loss ratio: {c['screening_loss_ratio']:.4f}")
        if c.get("screening_novelty") is not None:
            lines.append(f"  Screening novelty: {c['screening_novelty']:.3f}")
        sections.append("\n".join(lines))

    if investigation_results:
        lines = ["\nInvestigation phase summary:"]
        for r in investigation_results[:10]:
            lines.append(
                f"  {r.get('result_id', '?')[:12]}: "
                f"robustness={r.get('robustness', 0):.2f}, "
                f"best_val_lr={r.get('best_validation_loss_ratio', r.get('best_loss_ratio', 0)):.4f}"
            )
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _safe_num(val, default=0):
    """Safely convert a DB value to float, handling bytes/None."""
    if val is None:
        return default
    if isinstance(val, bytes):
        import struct

        try:
            return (
                struct.unpack("f", val)[0]
                if len(val) == 4
                else struct.unpack("d", val)[0]
            )
        except struct.error:
            return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _build_mode_recent_experiments_section(recent_experiments: list) -> str:
    total_s1 = sum(_safe_num(e.get("n_stage1_passed", 0)) for e in recent_experiments)
    total_programs = sum(
        _safe_num(e.get("n_programs_generated", 0)) for e in recent_experiments
    )
    avg_novelty_scores = [
        _safe_num(e.get("best_novelty_score", 0))
        for e in recent_experiments
        if e.get("best_novelty_score") is not None
    ]
    avg_novelty = (
        sum(avg_novelty_scores) / len(avg_novelty_scores) if avg_novelty_scores else 0
    )
    lines = [f"\nRecent experiment history ({len(recent_experiments)} experiments):"]
    lines.append(f"  Total S1 survivors: {total_s1} / {total_programs} programs")
    lines.append(f"  Average best novelty: {avg_novelty:.3f}")
    for exp in recent_experiments[:5]:
        exp_type = exp.get("experiment_type", "synthesis")
        s1 = exp.get("n_stage1_passed", 0)
        total = exp.get("n_programs_generated", 0)
        novelty = _safe_num(exp.get("best_novelty_score"), default=None)
        loss = _safe_num(exp.get("best_validation_loss_ratio"), default=None)
        loss_label = "val_loss"
        if loss is None:
            loss = _safe_num(exp.get("best_loss_ratio"), default=None)
            loss_label = "loss"
        line = f"  [{exp_type}] {s1}/{total} S1"
        if novelty is not None:
            line += f", novelty={novelty:.3f}"
        if loss is not None:
            line += f", {loss_label}={loss:.4f}"
        lines.append(line)
    return "\n".join(lines)


def _append_mode_leaderboard_sections(sections: List[str], leaderboard: list) -> None:
    tier_counts: Dict[str, int] = {}
    tier_best: Dict[str, float] = {}
    for entry in leaderboard:
        rid = str(entry.get("result_id") or "")
        if rid.startswith("ref_"):
            continue
        t = entry.get("tier", "screening")
        tier_counts[t] = tier_counts.get(t, 0) + 1
        score = entry.get("composite_score", 0)
        if score > tier_best.get(t, 0):
            tier_best[t] = score
    lines = [f"\nLeaderboard summary ({len(leaderboard)} total entries):"]
    for t in ["screening", "investigation", "validation", "breakthrough"]:
        if t in tier_counts:
            lines.append(
                f"  {t}: {tier_counts[t]} entries "
                f"(best score: {tier_best.get(t, 0):.3f})"
            )
    sections.append("\n".join(lines))
    screening_candidates = [
        e
        for e in leaderboard
        if e.get("tier") == "screening"
        and not str(e.get("result_id", "")).startswith("ref_")
        and e.get("screening_loss_ratio") is not None
        and e["screening_loss_ratio"] < 0.5
    ]
    if screening_candidates:
        sections.append(
            f"\nInvestigation-ready candidates: {len(screening_candidates)} "
            f"(screening loss_ratio < 0.5)"
        )
    investigation_candidates = [
        e
        for e in leaderboard
        if e.get("tier") == "investigation"
        and not str(e.get("result_id", "")).startswith("ref_")
        and e.get("investigation_robustness") is not None
        and e["investigation_robustness"] >= 0.5
    ]
    if investigation_candidates:
        sections.append(
            f"Validation-ready candidates: {len(investigation_candidates)} "
            f"(investigation robustness >= 0.5)"
        )


def _build_mode_compression_section(compression: Dict, n_tested: int) -> str:
    totals = compression.get("totals") or {}
    n_compressed_tested = int(totals.get("n_compressed_tested") or 0)
    n_compressed_survived = int(totals.get("n_compressed_survived") or 0)
    compressed_share = n_compressed_tested / n_tested
    compressed_survival = (
        n_compressed_survived / n_compressed_tested if n_compressed_tested > 0 else 0.0
    )
    comp_lines = [
        "Compression coverage: "
        f"{n_compressed_tested}/{n_tested} tested ({compressed_share:.1%}), "
        f"compressed survival={compressed_survival:.1%}"
    ]
    dense_markers = {"dense", "dense_matrix", "standard_float"}
    techniques = compression.get("techniques") or []
    for tech in techniques:
        name = tech.get("technique", "")
        if name in dense_markers or tech.get("n_tested", 0) == 0:
            continue
        t_n = tech["n_tested"]
        t_surv = tech.get("survival_rate", 0)
        parts = [f"{name}: {t_n} tested, {t_surv:.0%} survival"]
        qr = tech.get("avg_quality_retention")
        if qr is not None:
            parts.append(f"quality={qr:.2f}")
        mem = tech.get("avg_estimated_memory_mb")
        if mem is not None:
            parts.append(f"mem={mem:.1f}MB")
        cr = tech.get("avg_compression_ratio")
        if cr is not None:
            parts.append(f"compress={cr:.2f}x")
        comp_lines.append(f"  {', '.join(parts)}")
    return "\n".join(comp_lines)


def _build_mode_refuted_hypotheses_section(analytics_data: Dict) -> Optional[str]:
    neg = analytics_data.get("negative_results") or {}
    refuted_hyps = neg.get("refuted_hypotheses", [])
    if not refuted_hyps:
        return None
    lines = ["Refuted Hypotheses (DO NOT re-test similar directions):"]
    for rh in refuted_hyps[:5]:
        content = rh.get("content", "")[:120]
        conf = rh.get("confidence", 0)
        lines.append(f"  REFUTED (conf={conf:.2f}): {content}")
    return "\n".join(lines)


def _build_mode_sparsity_section(analytics_data: Dict, n_tested: int) -> str:
    sparse_summary = analytics_data.get("sparse_coverage") or {}
    n_sparse_tested = int(sparse_summary.get("n_sparse_tested") or 0)
    n_sparse_survived = int(sparse_summary.get("n_sparse_survived") or 0)
    sparse_share = n_sparse_tested / n_tested if n_tested > 0 else 0.0
    sparse_survival = (
        n_sparse_survived / n_sparse_tested if n_sparse_tested > 0 else 0.0
    )
    avg_density = sparse_summary.get("avg_density")
    lines = [
        f"\nSparsity coverage: "
        f"{n_sparse_tested}/{n_tested} tested ({sparse_share:.1%}), "
        f"sparse survival={sparse_survival:.1%}"
    ]
    if avg_density is not None:
        lines[0] += f", avg density={avg_density:.2f}"
    rigl_count = int(sparse_summary.get("n_rigl_runs") or 0)
    pruning_count = int(sparse_summary.get("n_pruning_runs") or 0)
    if rigl_count > 0 or pruning_count > 0:
        lines.append(
            f"  Sparse training: {rigl_count} RigL runs, {pruning_count} pruning baselines"
        )
    return "\n".join(lines)


def _append_mode_analytics_sections(sections: List[str], analytics_data: Dict) -> None:
    op_rates = analytics_data.get("op_success_rates", {})
    if op_rates:
        s1_rates = [v.get("s1_rate", 0) for v in op_rates.values()]
        avg_s1 = sum(s1_rates) / len(s1_rates) if s1_rates else 0
        sections.append(f"\nAverage op S1 rate: {avg_s1:.1%}")
    compression = analytics_data.get("compression_coverage") or {}
    totals = compression.get("totals") or {}
    n_tested = int(totals.get("n_tested") or 0)
    if n_tested > 0:
        sections.append(_build_mode_compression_section(compression, n_tested))
    refuted_section = _build_mode_refuted_hypotheses_section(analytics_data)
    if refuted_section:
        sections.append(refuted_section)
    if n_tested > 0:
        sections.append(_build_mode_sparsity_section(analytics_data, n_tested))


def build_mode_selection_context(
    recent_experiments: list,
    leaderboard: list,
    analytics_data: Optional[Dict] = None,
    current_mode: str = "synthesis",
    n_experiments_in_session: int = 0,
    cost_spent: float = 0.0,
    budget: float = 0.0,
    digest: Optional[object] = None,
) -> str:
    """Build context for mode selection decisions."""
    sections = [
        f"Current mode: {current_mode}",
        f"Experiments completed this session: {n_experiments_in_session}",
    ]
    if budget > 0:
        remaining = budget - cost_spent
        avg_cost = cost_spent / max(n_experiments_in_session, 1)
        est_remaining = int(remaining / avg_cost) if avg_cost > 0 else "unknown"
        sections.append(
            f"Budget: ${cost_spent:.2f} / ${budget:.2f} "
            f"(${remaining:.2f} remaining, ~{est_remaining} experiments left)"
        )
    if recent_experiments:
        sections.append(_build_mode_recent_experiments_section(recent_experiments))
    if leaderboard:
        _append_mode_leaderboard_sections(sections, leaderboard)
    if analytics_data:
        _append_mode_analytics_sections(sections, analytics_data)
    if digest is not None:
        inject_digest_context(sections, digest)
    return "\n\n".join(sections)


def build_manual_start_fallback_context(config: Optional[Dict] = None) -> str:
    """Return minimal context for manual synthesis starts.

    Ensures hypothesis generation still receives non-empty context when
    history/analytics retrieval is unavailable.
    """
    cfg = config or {}
    lines = [
        "Manual Start Context (fallback)",
        "No recent experiment history could be loaded. Use explicit, testable architecture hypotheses.",
        "Prioritize measurable outcomes (loss ratio / novelty / stage-1 survival).",
    ]

    n_programs = cfg.get("n_programs")
    model_dim = cfg.get("model_dim")
    max_depth = cfg.get("max_depth")
    max_ops = cfg.get("max_ops")
    math_space_weight = cfg.get("math_space_weight")

    if n_programs is not None or model_dim is not None:
        lines.append(
            "Planned run: "
            f"n_programs={n_programs if n_programs is not None else '?'}; "
            f"model_dim={model_dim if model_dim is not None else '?'}"
        )

    if max_depth is not None or max_ops is not None or math_space_weight is not None:
        lines.append(
            "Search envelope: "
            f"max_depth={max_depth if max_depth is not None else '?'}; "
            f"max_ops={max_ops if max_ops is not None else '?'}; "
            f"math_space_weight={math_space_weight if math_space_weight is not None else '?'}"
        )

    lines.append("Include a fallback plan if the primary mechanism underperforms.")
    return "\n".join(lines)


def build_go_no_go_context(
    candidate: Dict,
    investigation_results: Optional[List[Dict]] = None,
    campaign_criteria: str = "",
) -> str:
    """Build context for go/no-go decisions."""
    sections = []

    lines = ["Candidate under review:"]
    lines.append(f"  Result ID: {candidate.get('result_id', '?')[:12]}")
    if candidate.get("validation_loss_ratio") is not None:
        lines.append(
            f"  Validation loss ratio: {candidate['validation_loss_ratio']:.4f}"
        )
    elif candidate.get("loss_ratio") is not None:
        lines.append(f"  Loss ratio: {candidate['loss_ratio']:.4f}")
    if candidate.get("novelty_score") is not None:
        lines.append(f"  Novelty: {candidate['novelty_score']:.3f}")
    if candidate.get("investigation_robustness") is not None:
        lines.append(f"  Robustness: {candidate['investigation_robustness']:.2f}")
    if candidate.get("screening_loss_ratio") is not None:
        lines.append(f"  Screening loss ratio: {candidate['screening_loss_ratio']:.4f}")
    if candidate.get("architecture_desc"):
        lines.append(f"  Architecture: {candidate['architecture_desc']}")
    sections.append("\n".join(lines))

    if investigation_results:
        lines = ["Investigation results:"]
        for r in investigation_results[:10]:
            lines.append(
                f"  Program {r.get('result_id', '?')[:8]}: "
                f"loss_ratio={r.get('validation_loss_ratio', r.get('loss_ratio', '?'))}"
            )
        sections.append("\n".join(lines))

    if campaign_criteria:
        sections.append(f"Campaign success criteria: {campaign_criteria}")

    # Include pipeline thresholds so the LLM has a concrete reference
    sections.append(
        "Pipeline thresholds (for reference):\n"
        "  S1 pass: loss_ratio < 0.80\n"
        "  Investigation candidate: loss_ratio < 0.50\n"
        "  Validation gate: loss_ratio < 0.60"
    )

    return "\n\n".join(sections)


def inject_digest_context(sections: list, digest) -> None:
    """Append a compact knowledge digest section (~300 tokens).

    *digest* should be an ExperimentDigest instance (or duck-typed equivalent).
    Fails silently if digest is malformed.
    """
    try:
        lines = ["Knowledge Digest (distilled from historical analysis):"]

        # Training curve profiles
        profiles = getattr(digest, "convergence_profiles", [])
        if profiles:
            parts = []
            for p in profiles:
                parts.append(f"{p.category}={p.count}(S1:{p.s1_pass_rate:.0%})")
            lines.append(f"  Curve profiles: {', '.join(parts)}")

        # Architecture families
        families = getattr(digest, "architecture_families", [])
        if families:
            for f in families[:3]:
                ops_str = ", ".join(f.representative_ops[:4])
                lines.append(
                    f"  Family {f.family_id}: {f.n_members} members "
                    f"[{ops_str}] novelty={f.avg_novelty:.3f} loss={f.avg_loss_ratio:.4f}"
                )

        # Significant config effects
        effects = getattr(digest, "config_effects", [])
        sig = [e for e in effects if e.p_value < 0.05]
        if sig:
            lines.append("  Config effects (p<0.05):")
            for e in sig[:5]:
                lines.append(
                    f"    {e.param_name}->{e.target}: {e.direction} (rho={e.rho:+.3f})"
                )

        # Synergies
        synergies = getattr(digest, "op_synergies", [])
        syn = [s for s in synergies if s.label == "synergistic"][:3]
        anti = [s for s in synergies if s.label == "anti_synergistic"][:3]
        if syn:
            lines.append(
                "  Synergistic pairs: "
                + "; ".join(f"{s.op_a}+{s.op_b}({s.lift:.1f}x)" for s in syn)
            )
        if anti:
            lines.append(
                "  Anti-synergistic: "
                + "; ".join(f"{s.op_a}+{s.op_b}({s.lift:.2f}x)" for s in anti)
            )

        # Efficiency profiles (Pareto-optimal families)
        eff_profiles = getattr(digest, "efficiency_profiles", [])
        pareto = [p for p in eff_profiles if p.pareto_optimal]
        if pareto:
            lines.append("  Pareto-optimal families (best loss/param tradeoff):")
            for p in pareto[:3]:
                lines.append(
                    f"    Family {p.family_id}: {p.avg_params / 1e6:.2f}M params, "
                    f"loss/Mparam={p.loss_per_megaparam:.3f}"
                )

        # Hypothesis closure
        outcomes = getattr(digest, "hypothesis_outcomes", [])
        if outcomes:
            confirmed = sum(1 for h in outcomes if h.outcome == "confirmed")
            refuted = sum(1 for h in outcomes if h.outcome == "refuted")
            lines.append(f"  Hypotheses: {confirmed} confirmed, {refuted} refuted")

        # Recommendations
        recs = getattr(digest, "recommendations", [])
        if recs:
            lines.append("  Strategic recommendations:")
            for r in recs[:5]:
                lines.append(f"    - {r[:120]}")

        if len(lines) > 1:
            sections.append("\n".join(lines))
    except Exception as exc:
        logger.debug("Digest context enrichment failed: %s", exc)


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{n / total * 100:.0f}%"
