"""Briefing context builders for LLM prompts."""

from __future__ import annotations

from typing import Dict, List, Optional


def _loss_parts(row: Dict) -> List[str]:
    val_loss = row.get("best_validation_loss_ratio")
    disc_loss = row.get("best_discovery_loss_ratio")
    legacy_loss = row.get("best_loss_ratio")
    parts: List[str] = []
    if val_loss is not None:
        parts.append(f"val={val_loss:.4f}")
    if disc_loss is not None:
        parts.append(f"disc={disc_loss:.4f}")
    if not parts and legacy_loss is not None:
        parts.append(f"lr={legacy_loss:.4f}")
    return parts


def _format_just_completed(just_completed: Dict) -> str:
    eid = (just_completed.get("experiment_id") or "?")[:12]
    etype = just_completed.get("experiment_type") or just_completed.get("mode") or "?"
    gen = just_completed.get("n_programs_generated") or 0
    s1 = just_completed.get("n_stage1_passed") or 0
    rate = f"{s1 / gen * 100:.1f}%" if gen > 0 else "N/A"
    loss_parts = _loss_parts(just_completed)
    loss_str = f", {', '.join(loss_parts)}" if loss_parts else ""
    summary = just_completed.get("aria_summary") or ""
    summary_str = f"\n  Summary: {summary[:120]}" if summary else ""
    return (
        "*** JUST COMPLETED (analyze this first) ***\n"
        f"  Experiment {eid} ({etype}): {s1}/{gen} S1 survivors ({rate}){loss_str}{summary_str}"
    )


def _format_recent_experiments(recent_experiments: List[Dict]) -> str:
    lines = [f"Recent Experiments ({len(recent_experiments)} most recent):"]
    for exp in recent_experiments[:5]:
        eid = (exp.get("experiment_id") or "?")[:8]
        etype = exp.get("experiment_type", "?")
        status = exp.get("status", "?")
        gen = exp.get("n_programs_generated") or 0
        s1 = exp.get("n_stage1_passed") or 0
        rate = f"{s1 / gen * 100:.1f}%" if gen > 0 else "N/A"
        loss_parts = _loss_parts(exp)
        loss_str = f", {', '.join(loss_parts)}" if loss_parts else ""
        summary = exp.get("aria_summary") or ""
        summary_str = f" — {summary[:60]}" if summary else ""
        lines.append(
            f"  [{eid}] {etype} {status}: {s1}/{gen} S1 ({rate}){loss_str}{summary_str}"
        )
    return "\n".join(lines)


def _format_pipeline(pipeline_tiers: Dict[str, int]) -> str:
    parts = [
        f"{tier}: {pipeline_tiers.get(tier, 0)}"
        for tier in ("screening", "investigation", "validation", "breakthrough")
        if pipeline_tiers.get(tier, 0) > 0
    ]
    return (
        f"Pipeline: {', '.join(parts)}"
        if parts
        else "Pipeline: empty (no candidates yet)"
    )


def _format_learning_trend(learning_trajectory: Dict) -> str:
    trend = learning_trajectory.get("trend", "insufficient_data")
    slope = learning_trajectory.get("slope")
    recent_rate = learning_trajectory.get("recent_s1_rate")
    line = f"Learning Trend: {trend}"
    if slope is not None:
        line += f" (slope: {slope * 100:+.2f}%/experiment)"
    if recent_rate is not None:
        line += f", recent S1 rate: {recent_rate * 100:.1f}%"
    return line


def _format_campaign(campaign: Dict) -> str:
    return (
        f"Active Campaign: {campaign.get('title', '?')}\n"
        f"  Objective: {campaign.get('objective', '?')}\n"
        f"  Status: {campaign.get('status', 'active')}"
    )


def _format_grammar_changes(
    grammar_weights: Dict, default_weights: Dict
) -> Optional[str]:
    deltas = []
    for cat in sorted(grammar_weights.keys()):
        cur = grammar_weights.get(cat, 1.0)
        base = default_weights.get(cat, 1.0)
        if abs(cur - base) > 0.1:
            deltas.append((cat, base, cur, cur - base))
    if not deltas:
        return None
    deltas.sort(key=lambda x: abs(x[3]), reverse=True)
    lines = ["Grammar Weight Changes (biggest shifts):"]
    for cat, base, cur, delta in deltas[:5]:
        lines.append(f"  {cat}: {base:.1f} -> {cur:.1f} ({delta:+.1f})")
    return "\n".join(lines)


def _format_top_programs(top_programs: List[Dict]) -> str:
    lines = [f"Top {len(top_programs)} Programs:"]
    for program in top_programs[:3]:
        fp = (program.get("graph_fingerprint") or "?")[:16]
        loss = program.get("validation_loss_ratio")
        label = "val_loss"
        if loss is None:
            loss = program.get("loss_ratio")
            label = "loss"
        novelty = program.get("novelty_score")
        tier = program.get("tier", "screening")
        parts = [f"{fp} ({tier})"]
        if loss is not None:
            parts.append(f"{label}={loss:.4f}")
        if novelty is not None:
            parts.append(f"novelty={novelty:.3f}")
        lines.append(f"  {', '.join(parts)}")
    return "\n".join(lines)


def _format_scaling_summary(scaling_summary: Dict) -> str:
    n_eval = scaling_summary.get("n_evaluated", 0)
    if n_eval <= 0:
        return "Scaling Gate: no candidates evaluated yet (target: 3x param efficiency vs GPT-2)."
    n_pass = scaling_summary.get("n_gate_passed", 0)
    best = scaling_summary.get("best_param_efficiency", 0)
    target = scaling_summary.get("target", 3.0)
    if n_pass == 0:
        return f"Scaling Gate: 0/{n_eval} pass {target:.0f}x target (best: {best:.2f}x). Priority: improve param efficiency."
    return f"Scaling Gate: {n_pass}/{n_eval} pass {target:.0f}x target (best: {best:.2f}x)."


def _format_sparsity_coverage(sparse_coverage: Dict) -> Optional[str]:
    n_sparse = int(sparse_coverage.get("n_sparse_tested") or 0)
    n_survived = int(sparse_coverage.get("n_sparse_survived") or 0)
    n_total = int(sparse_coverage.get("n_total_tested") or 0)
    if n_total <= 0:
        return None
    share = n_sparse / n_total
    survival = n_survived / n_sparse if n_sparse > 0 else 0.0
    line = f"Sparsity: {n_sparse}/{n_total} programs use sparse ops ({share:.1%}), sparse survival={survival:.1%}"
    avg_d = sparse_coverage.get("avg_density")
    if avg_d is not None:
        line += f", avg density={avg_d:.2f}"
    return line


def _format_reference_comparison(ref_comparison: Dict) -> str:
    refs = ref_comparison.get("references") or []
    lines = ["Reference Baselines (targets to beat):"]
    lines.extend(
        f"  {ref.get('name', '?')}: score={ref.get('score', 0):.1f}" for ref in refs
    )
    if ref_comparison.get("beats_all_references"):
        margin = ref_comparison.get("margin_pct", 0)
        best = ref_comparison.get("best_synthesized_score", 0)
        lines.append(
            f"  *** MILESTONE: Best synthesized model (score={best:.1f}) beats ALL references by {margin}%! ***"
        )
    return "\n".join(lines)


def build_briefing_context(
    recent_experiments: List[Dict],
    pipeline_tiers: Dict[str, int],
    learning_trajectory: Dict,
    campaign: Optional[Dict] = None,
    grammar_weights: Optional[Dict] = None,
    default_weights: Optional[Dict] = None,
    top_programs: Optional[List[Dict]] = None,
    just_completed: Optional[Dict] = None,
    sparse_coverage: Optional[Dict] = None,
    scaling_summary: Optional[Dict] = None,
    ref_comparison: Optional[Dict] = None,
) -> str:
    """Build compact context for the Aria briefing prompt."""
    sections: List[str] = []
    if just_completed:
        sections.append(_format_just_completed(just_completed))
    if recent_experiments:
        sections.append(_format_recent_experiments(recent_experiments))
    if pipeline_tiers is not None:
        sections.append(_format_pipeline(pipeline_tiers))
    if learning_trajectory:
        sections.append(_format_learning_trend(learning_trajectory))
    if campaign:
        sections.append(_format_campaign(campaign))
    if grammar_weights and default_weights:
        grammar = _format_grammar_changes(grammar_weights, default_weights)
        if grammar:
            sections.append(grammar)
    if top_programs:
        sections.append(_format_top_programs(top_programs))
    if scaling_summary:
        sections.append(_format_scaling_summary(scaling_summary))
    if sparse_coverage:
        sparse = _format_sparsity_coverage(sparse_coverage)
        if sparse:
            sections.append(sparse)
    if ref_comparison:
        sections.append(_format_reference_comparison(ref_comparison))
    return "\n\n".join(sections)
