"""
Context Builder for LLM Prompts

Builds structured context strings from notebook data for injection
into prompt templates. Handles missing data gracefully.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def build_experiment_context(results: Dict, config: Optional[Dict] = None,
                             hypothesis: Optional[str] = None) -> str:
    """Build context for a single experiment's results."""
    lines = []

    if hypothesis:
        lines.append(f"Hypothesis: {hypothesis}")

    if config:
        lines.append(f"Config: {config.get('n_programs', '?')} programs, "
                      f"dim={config.get('model_dim', '?')}, "
                      f"depth={config.get('max_depth', '?')}, "
                      f"ops={config.get('max_ops', '?')}, "
                      f"math_space_weight={config.get('math_space_weight', '?')}")

    total = results.get("total", 0)
    s0 = results.get("stage0_passed", 0)
    s05 = results.get("stage05_passed", 0)
    s1 = results.get("stage1_passed", 0)
    novel = results.get("novel_count", 0)

    lines.append(f"\nFunnel: {total} generated -> {s0} S0 ({_pct(s0, total)}) "
                  f"-> {s05} S0.5 ({_pct(s05, total)}) "
                  f"-> {s1} S1 ({_pct(s1, total)})")
    lines.append(f"Novel survivors (novelty > 0.5): {novel}")

    if results.get("best_loss_ratio") is not None:
        lines.append(f"Best loss ratio: {results['best_loss_ratio']:.4f}")
    if results.get("best_novelty_score") is not None:
        lines.append(f"Best novelty: {results['best_novelty_score']:.3f}")

    survivors = results.get("survivors", [])
    if survivors:
        lines.append(f"\nTop survivors:")
        for s in survivors[:5]:
            lines.append(f"  - {s['fingerprint'][:12]}: "
                          f"novelty={s.get('novelty', 0):.3f}, "
                          f"loss_ratio={s.get('loss_ratio', 0):.4f}")

    return "\n".join(lines)


def build_history_context(experiments: List[Dict], limit: int = 10) -> str:
    """Build context from recent experiment history."""
    lines = ["Recent experiment history:"]

    for exp in experiments[:limit]:
        status = exp.get("status", "?")
        s1 = exp.get("n_stage1_passed", 0)
        total = exp.get("n_programs_generated", 0)
        novelty = exp.get("best_novelty_score")
        loss = exp.get("best_loss_ratio")
        mood = exp.get("aria_mood", "?")

        line = f"  [{status}] {total} programs, {s1} S1 pass"
        if novelty is not None:
            line += f", novelty={novelty:.3f}"
        if loss is not None:
            line += f", loss={loss:.4f}"
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
    if program.get("loss_ratio") is not None:
        lines.append(f"Loss ratio: {program['loss_ratio']:.4f}")
    if program.get("novelty_score") is not None:
        lines.append(f"Novelty: {program['novelty_score']:.3f} "
                      f"(structural={program.get('structural_novelty', 0):.3f}, "
                      f"behavioral={program.get('behavioral_novelty', 0):.3f})")
    if program.get("most_similar_to"):
        lines.append(f"Most similar to: {program['most_similar_to']}")

    return "\n".join(lines)


def build_failure_context(programs: List[Dict]) -> str:
    """Build context about failure patterns from a list of programs."""
    total = len(programs)
    if total == 0:
        return "No programs to analyze."

    s0_fail = sum(1 for p in programs if not p.get("stage0_passed"))
    s05_fail = sum(1 for p in programs
                   if p.get("stage0_passed") and not p.get("stage05_passed"))
    s1_fail = sum(1 for p in programs
                  if p.get("stage05_passed") and not p.get("stage1_passed"))
    s1_pass = sum(1 for p in programs if p.get("stage1_passed"))

    lines = [
        f"Total programs: {total}",
        f"Stage 0 failures: {s0_fail} ({_pct(s0_fail, total)})",
        f"Stage 0.5 failures: {s05_fail} ({_pct(s05_fail, total)})",
        f"Stage 1 failures: {s1_fail} ({_pct(s1_fail, total)})",
        f"Stage 1 passes: {s1_pass} ({_pct(s1_pass, total)})",
    ]

    # Error distribution
    errors: Dict[str, int] = {}
    for p in programs:
        err = p.get("stage0_error", "")
        if err:
            # Normalize error to first 60 chars
            key = err[:60].strip()
            errors[key] = errors.get(key, 0) + 1

    if errors:
        lines.append("\nTop error types:")
        for err, count in sorted(errors.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"  [{count}x] {err}")

    return "\n".join(lines)


def build_rich_context(
    results: Dict,
    config: Optional[Dict] = None,
    hypothesis: Optional[str] = None,
    analytics_data: Optional[Dict] = None,
    history: Optional[List[Dict]] = None,
    past_hypotheses: Optional[List[Dict]] = None,
) -> str:
    """Build comprehensive context including all analytics data.

    Aggregates experiment results, history, op success rates, structural
    correlations, failure patterns, efficiency frontier, grammar weights,
    learning log, and past hypothesis outcomes into a single context string.
    """
    sections = []

    # Current experiment results
    sections.append(build_experiment_context(results, config, hypothesis))

    # History
    if history:
        sections.append(build_history_context(history))

    # Analytics data
    if analytics_data:
        # Op success rates
        op_rates = analytics_data.get("op_success_rates", {})
        if op_rates:
            rated = sorted(op_rates.items(), key=lambda x: -x[1].get("s1_rate", 0))
            best = rated[:5]
            worst = rated[-5:] if len(rated) > 5 else []
            lines = ["Op Success Rates (S1):"]
            if best:
                lines.append("  Best:")
                for op, s in best:
                    lines.append(f"    {op}: S1={s.get('s1_rate', 0):.0%} "
                                 f"(n={s.get('n_used', 0)}, "
                                 f"novelty={s.get('avg_novelty') or 0:.3f})")
            if worst:
                lines.append("  Worst:")
                for op, s in worst:
                    lines.append(f"    {op}: S1={s.get('s1_rate', 0):.0%} "
                                 f"(n={s.get('n_used', 0)})")
            sections.append("\n".join(lines))

        # Structural correlations
        correlations = analytics_data.get("structural_correlations", {})
        if correlations:
            lines = ["Structural Correlations with S1 Success:"]
            for metric, effect in sorted(correlations.items(),
                                          key=lambda x: -abs(x[1])):
                name = metric.replace("graph_", "").replace("_", " ")
                direction = "+" if effect > 0 else "-"
                lines.append(f"  {name}: {direction}{abs(effect):.2f}")
            sections.append("\n".join(lines))

        # Failure patterns
        failures = analytics_data.get("failure_patterns", {})
        if failures:
            lines = ["Failure Patterns (error_type x stage):"]
            for err_type, info in sorted(failures.items(),
                                          key=lambda x: -x[1].get("total", 0))[:5]:
                stages = ", ".join(f"{s}:{c}" for s, c
                                   in info.get("by_stage", {}).items())
                lines.append(f"  {err_type}: {info.get('total', 0)} total ({stages})")
            sections.append("\n".join(lines))

        # Top op combinations
        combos = analytics_data.get("top_op_combinations", [])
        if combos:
            lines = ["Top Op Combinations (S1 survivors):"]
            for c in combos[:5]:
                ops = " + ".join(c.get("ops", []))
                lines.append(f"  {ops}: {c.get('count', 0)}x "
                             f"(avg novelty {c.get('avg_novelty', 0):.3f})")
            sections.append("\n".join(lines))

        # Efficiency frontier
        frontier = analytics_data.get("efficiency_frontier", [])
        if frontier:
            lines = [f"Efficiency Frontier: {len(frontier)} Pareto-optimal programs"]
            best_loss = min((p.get("final_loss") for p in frontier
                             if p.get("final_loss") is not None), default=None)
            if best_loss is not None:
                lines.append(f"  Best loss on frontier: {best_loss:.4f}")
            most_eff = min((p for p in frontier if p.get("flops_forward")),
                           key=lambda p: p["flops_forward"], default=None)
            if most_eff:
                lines.append(f"  Most efficient: {most_eff.get('flops_forward', 0):.0f} FLOPs, "
                             f"loss={most_eff.get('final_loss', 0):.4f}")
            sections.append("\n".join(lines))

        # Grammar weights
        grammar_weights = analytics_data.get("grammar_weights")
        default_weights = analytics_data.get("default_weights", {})
        if grammar_weights and default_weights:
            lines = ["Grammar Weights (learned vs default):"]
            for cat in sorted(set(list(grammar_weights.keys()) +
                                   list(default_weights.keys()))):
                learned = grammar_weights.get(cat, "—")
                default = default_weights.get(cat, 1.0)
                if isinstance(learned, (int, float)):
                    delta = learned - default
                    arrow = "^" if delta > 0.1 else ("v" if delta < -0.1 else "=")
                    lines.append(f"  {cat}: {default:.1f} -> {learned:.1f} [{arrow}]")
            sections.append("\n".join(lines))

        # Learning log
        learning_log = analytics_data.get("learning_log", [])
        if learning_log:
            lines = [f"Recent Learning Events ({len(learning_log)} most recent):"]
            for entry in learning_log[:5]:
                lines.append(f"  [{entry.get('event_type', '?')}] "
                             f"{entry.get('description', '')[:80]}")
            sections.append("\n".join(lines))

        # Active insights
        insights = analytics_data.get("insights", [])
        if insights:
            lines = ["Active Insights:"]
            for ins in insights[:10]:
                cat = ins.get("category", "general")
                content = ins.get("content", "")[:120]
                conf = ins.get("confidence", 0)
                lines.append(f"  [{cat}] (conf={conf:.1f}) {content}")
            sections.append("\n".join(lines))

    # Past hypothesis outcomes
    if past_hypotheses:
        lines = ["Past Hypothesis Outcomes:"]
        for h in past_hypotheses[:5]:
            status = "CONFIRMED" if h.get("confirmed") else "REFUTED"
            lines.append(f"  [{status}] {h.get('hypothesis', '?')[:80]}")
            if h.get("s1_count") is not None:
                lines.append(f"    S1 passes: {h['s1_count']}, "
                             f"novelty: {h.get('best_novelty', 0):.3f}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def build_investigation_context(candidates: list, leaderboard: list) -> str:
    """Build context for investigation phase LLM prompts."""
    sections = []

    sections.append(f"Investigation Phase: {len(candidates)} candidates selected for deep study")

    for i, c in enumerate(candidates[:10]):
        lines = [f"\nCandidate {i + 1}:"]
        lines.append(f"  Result ID: {c.get('result_id', '?')[:12]}")
        lines.append(f"  Source: {c.get('model_source', 'graph_synthesis')}")
        if c.get("architecture_desc"):
            lines.append(f"  Architecture: {c['architecture_desc']}")
        if c.get("loss_ratio") is not None:
            lines.append(f"  Screening loss ratio: {c['loss_ratio']:.4f}")
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
        best = max(leaderboard, key=lambda x: x.get("composite_score", 0),
                   default=None)
        if best:
            lines.append(f"  Best composite score: {best.get('composite_score', 0):.3f}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def build_validation_context(candidates: list,
                              investigation_results: list) -> str:
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
            lines.append(f"  Investigation loss ratio: {c['investigation_loss_ratio']:.4f}")
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
                f"best_lr={r.get('best_loss_ratio', 0):.4f}"
            )
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def build_mode_selection_context(
    recent_experiments: list,
    leaderboard: list,
    analytics_data: Optional[Dict] = None,
    current_mode: str = "synthesis",
    n_experiments_in_session: int = 0,
    cost_spent: float = 0.0,
    budget: float = 0.0,
) -> str:
    """Build context for mode selection decisions."""
    sections = []

    sections.append(f"Current mode: {current_mode}")
    sections.append(f"Experiments completed this session: {n_experiments_in_session}")

    if budget > 0:
        remaining = budget - cost_spent
        avg_cost = cost_spent / max(n_experiments_in_session, 1)
        est_remaining = int(remaining / avg_cost) if avg_cost > 0 else "unknown"
        sections.append(
            f"Budget: ${cost_spent:.2f} / ${budget:.2f} "
            f"(${remaining:.2f} remaining, ~{est_remaining} experiments left)"
        )

    # Recent experiment results
    if recent_experiments:
        total_s1 = sum(e.get("n_stage1_passed", 0) for e in recent_experiments)
        total_programs = sum(e.get("n_programs_generated", 0)
                            for e in recent_experiments)
        avg_novelty_scores = [
            e.get("best_novelty_score", 0) for e in recent_experiments
            if e.get("best_novelty_score") is not None
        ]
        avg_novelty = (sum(avg_novelty_scores) / len(avg_novelty_scores)
                       if avg_novelty_scores else 0)

        lines = [f"\nRecent experiment history ({len(recent_experiments)} experiments):"]
        lines.append(f"  Total S1 survivors: {total_s1} / {total_programs} programs")
        lines.append(f"  Average best novelty: {avg_novelty:.3f}")

        for exp in recent_experiments[:5]:
            exp_type = exp.get("experiment_type", "synthesis")
            s1 = exp.get("n_stage1_passed", 0)
            total = exp.get("n_programs_generated", 0)
            novelty = exp.get("best_novelty_score")
            loss = exp.get("best_loss_ratio")
            line = f"  [{exp_type}] {s1}/{total} S1"
            if novelty is not None:
                line += f", novelty={novelty:.3f}"
            if loss is not None:
                line += f", loss={loss:.4f}"
            lines.append(line)
        sections.append("\n".join(lines))

    # Leaderboard summary
    if leaderboard:
        tier_counts: Dict[str, int] = {}
        tier_best: Dict[str, float] = {}
        for entry in leaderboard:
            t = entry.get("tier", "screening")
            tier_counts[t] = tier_counts.get(t, 0) + 1
            score = entry.get("composite_score", 0)
            if score > tier_best.get(t, 0):
                tier_best[t] = score

        lines = [f"\nLeaderboard summary ({len(leaderboard)} total entries):"]
        for t in ["screening", "investigation", "validation", "breakthrough"]:
            if t in tier_counts:
                lines.append(f"  {t}: {tier_counts[t]} entries "
                             f"(best score: {tier_best.get(t, 0):.3f})")
        sections.append("\n".join(lines))

        # Check if investigation/validation candidates exist
        screening_candidates = [
            e for e in leaderboard
            if e.get("tier") == "screening"
            and e.get("screening_loss_ratio") is not None
            and e["screening_loss_ratio"] < 0.5
        ]
        if screening_candidates:
            sections.append(
                f"\nInvestigation-ready candidates: {len(screening_candidates)} "
                f"(screening loss_ratio < 0.5)")

        investigation_candidates = [
            e for e in leaderboard
            if e.get("tier") == "investigation"
            and e.get("investigation_robustness") is not None
            and e["investigation_robustness"] >= 0.5
        ]
        if investigation_candidates:
            sections.append(
                f"Validation-ready candidates: {len(investigation_candidates)} "
                f"(investigation robustness >= 0.5)")

    # Op success rates summary
    if analytics_data:
        op_rates = analytics_data.get("op_success_rates", {})
        if op_rates:
            s1_rates = [v.get("s1_rate", 0) for v in op_rates.values()]
            avg_s1 = sum(s1_rates) / len(s1_rates) if s1_rates else 0
            sections.append(f"\nAverage op S1 rate: {avg_s1:.1%}")

    return "\n\n".join(sections)


def build_hypothesis_context(
    campaign: Optional[Dict] = None,
    recent_hypotheses: Optional[List[Dict]] = None,
    knowledge: Optional[List[Dict]] = None,
    leaderboard: Optional[List[Dict]] = None,
    recent_experiments: Optional[List[Dict]] = None,
) -> str:
    """Build context for structured hypothesis formulation."""
    sections = []

    if campaign:
        sections.append(
            f"Active Campaign: {campaign.get('title', '?')}\n"
            f"Objective: {campaign.get('objective', '?')}\n"
            f"Success Criteria: {campaign.get('success_criteria', '?')}"
        )

    if recent_experiments:
        sections.append(build_history_context(recent_experiments, limit=5))

    if recent_hypotheses:
        lines = ["Recent Hypotheses:"]
        for h in recent_hypotheses[:5]:
            status = h.get("status", "pending")
            lines.append(
                f"  [{status.upper()}] {h.get('prediction', '?')[:80]}"
            )
            if h.get("outcome_summary"):
                lines.append(f"    -> {h['outcome_summary'][:80]}")
        sections.append("\n".join(lines))

    if knowledge:
        lines = ["Knowledge Base (relevant insights):"]
        for k in knowledge[:10]:
            lines.append(
                f"  [{k.get('category', '?')}] {k.get('title', '?')}: "
                f"{k.get('content', '?')[:80]} "
                f"(confidence={k.get('confidence', 0):.1f}, "
                f"validated {k.get('times_validated', 0)}x)"
            )
        sections.append("\n".join(lines))

    if leaderboard:
        tier_counts: Dict[str, int] = {}
        for entry in leaderboard:
            t = entry.get("tier", "screening")
            tier_counts[t] = tier_counts.get(t, 0) + 1
        lines = ["Leaderboard:"]
        for t in ["screening", "investigation", "validation", "breakthrough"]:
            if t in tier_counts:
                lines.append(f"  {t}: {tier_counts[t]} entries")
        sections.append("\n".join(lines))

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
    if candidate.get("loss_ratio") is not None:
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
                f"loss_ratio={r.get('loss_ratio', '?')}"
            )
        sections.append("\n".join(lines))

    if campaign_criteria:
        sections.append(f"Campaign success criteria: {campaign_criteria}")

    return "\n\n".join(sections)


def build_campaign_report_context(
    campaign: Dict,
    experiments: List[Dict],
    hypotheses: List[Dict],
    decisions: List[Dict],
    knowledge: List[Dict],
) -> str:
    """Build context for campaign report generation."""
    sections = []

    sections.append(
        f"Campaign: {campaign.get('title', '?')}\n"
        f"Objective: {campaign.get('objective', '?')}\n"
        f"Success Criteria: {campaign.get('success_criteria', '?')}\n"
        f"Status: {campaign.get('status', '?')}"
    )

    if experiments:
        total_s1 = sum(e.get("n_stage1_passed", 0) for e in experiments)
        total_programs = sum(e.get("n_programs_generated", 0) for e in experiments)
        lines = [f"\nExperiments ({len(experiments)} total):"]
        lines.append(f"  Total programs evaluated: {total_programs}")
        lines.append(f"  Total S1 survivors: {total_s1}")
        for exp in experiments[:10]:
            s1 = exp.get("n_stage1_passed", 0)
            total = exp.get("n_programs_generated", 0)
            exp_type = exp.get("experiment_type", "?")
            lines.append(f"  [{exp_type}] {s1}/{total} S1")
        sections.append("\n".join(lines))

    if hypotheses:
        lines = [f"\nHypothesis Chain ({len(hypotheses)} total):"]
        confirmed = sum(1 for h in hypotheses if h.get("status") == "confirmed")
        refuted = sum(1 for h in hypotheses if h.get("status") == "refuted")
        lines.append(f"  Confirmed: {confirmed}, Refuted: {refuted}")
        for h in hypotheses[:10]:
            lines.append(
                f"  [{h.get('status', '?').upper()}] {h.get('prediction', '?')[:80]}"
            )
        sections.append("\n".join(lines))

    if decisions:
        lines = [f"\nDecisions ({len(decisions)} total):"]
        for d in decisions[:10]:
            lines.append(
                f"  [{d.get('decision_type', '?').upper()}] "
                f"{d.get('subject', '?')[:60]}: {d.get('rationale', '')[:80]}"
            )
        sections.append("\n".join(lines))

    if knowledge:
        lines = [f"\nKnowledge Extracted ({len(knowledge)} entries):"]
        for k in knowledge[:10]:
            lines.append(
                f"  [{k.get('category', '?')}] {k.get('title', '?')}: "
                f"{k.get('content', '?')[:80]}"
            )
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def build_knowledge_extraction_context(
    experiment_results: List[Dict],
    resolved_hypotheses: List[Dict],
) -> str:
    """Build context for knowledge extraction."""
    sections = []

    if experiment_results:
        lines = [f"Recent Experiment Results ({len(experiment_results)} experiments):"]
        for exp in experiment_results[:10]:
            s1 = exp.get("n_stage1_passed", 0)
            total = exp.get("n_programs_generated", 0)
            loss = exp.get("best_loss_ratio")
            exp_type = exp.get("experiment_type", "?")
            line = f"  [{exp_type}] {s1}/{total} S1"
            if loss is not None:
                line += f", best loss_ratio={loss:.4f}"
            lines.append(line)
        sections.append("\n".join(lines))

    if resolved_hypotheses:
        lines = [f"Resolved Hypotheses ({len(resolved_hypotheses)} total):"]
        for h in resolved_hypotheses[:10]:
            lines.append(
                f"  [{h.get('status', '?').upper()}] {h.get('prediction', '?')[:80]}"
            )
            if h.get("outcome_summary"):
                lines.append(f"    Evidence: {h['outcome_summary'][:100]}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def build_campaign_formulation_context(
    recent_experiments: Optional[List[Dict]] = None,
    knowledge: Optional[List[Dict]] = None,
    previous_campaigns: Optional[List[Dict]] = None,
) -> str:
    """Build context for campaign formulation."""
    sections = []

    if previous_campaigns:
        lines = ["Previous Campaigns:"]
        for c in previous_campaigns[:5]:
            lines.append(
                f"  [{c.get('status', '?')}] {c.get('title', '?')}: "
                f"{c.get('objective', '?')[:80]}"
            )
            if c.get("findings_summary"):
                lines.append(f"    Findings: {c['findings_summary'][:100]}")
        sections.append("\n".join(lines))

    if recent_experiments:
        sections.append(build_history_context(recent_experiments, limit=10))

    if knowledge:
        lines = ["Knowledge Base:"]
        for k in knowledge[:10]:
            lines.append(
                f"  [{k.get('category', '?')}] {k.get('title', '?')}: "
                f"{k.get('content', '?')[:80]}"
            )
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def build_briefing_context(
    recent_experiments: List[Dict],
    pipeline_tiers: Dict[str, int],
    learning_trajectory: Dict,
    campaign: Optional[Dict] = None,
    grammar_weights: Optional[Dict] = None,
    default_weights: Optional[Dict] = None,
    top_programs: Optional[List[Dict]] = None,
    just_completed: Optional[Dict] = None,
) -> str:
    """Build compact context for the Aria briefing prompt.

    Targets <2000 tokens to keep LLM calls fast and cheap.
    If *just_completed* is provided, it is highlighted at the top so the
    LLM can reference the experiment that just finished.
    """
    sections = []

    # Just-completed experiment (prominent top section)
    if just_completed:
        eid = (just_completed.get("experiment_id") or "?")[:12]
        etype = just_completed.get("experiment_type") or just_completed.get("mode") or "?"
        gen = just_completed.get("n_programs_generated") or 0
        s1 = just_completed.get("n_stage1_passed") or 0
        rate = f"{s1/gen*100:.1f}%" if gen > 0 else "N/A"
        loss = just_completed.get("best_loss_ratio")
        loss_str = f", best loss ratio={loss:.4f}" if loss is not None else ""
        summary = just_completed.get("aria_summary") or ""
        summary_str = f"\n  Summary: {summary[:120]}" if summary else ""
        sections.append(
            f"*** JUST COMPLETED (analyze this first) ***\n"
            f"  Experiment {eid} ({etype}): {s1}/{gen} S1 survivors ({rate}){loss_str}{summary_str}"
        )

    # Recent experiments (last 5)
    if recent_experiments:
        lines = [f"Recent Experiments ({len(recent_experiments)} most recent):"]
        for exp in recent_experiments[:5]:
            eid = (exp.get("experiment_id") or "?")[:8]
            etype = exp.get("experiment_type", "?")
            status = exp.get("status", "?")
            gen = exp.get("n_programs_generated") or 0
            s1 = exp.get("n_stage1_passed") or 0
            rate = f"{s1/gen*100:.1f}%" if gen > 0 else "N/A"
            loss = exp.get("best_loss_ratio")
            loss_str = f", loss={loss:.4f}" if loss is not None else ""
            summary = exp.get("aria_summary") or ""
            summary_str = f" — {summary[:60]}" if summary else ""
            lines.append(
                f"  [{eid}] {etype} {status}: {s1}/{gen} S1 ({rate}){loss_str}{summary_str}"
            )
        sections.append("\n".join(lines))

    # Pipeline state
    if pipeline_tiers:
        parts = []
        for tier in ["screening", "investigation", "validation", "breakthrough"]:
            count = pipeline_tiers.get(tier, 0)
            if count > 0:
                parts.append(f"{tier}: {count}")
        if parts:
            sections.append(f"Pipeline: {', '.join(parts)}")
        else:
            sections.append("Pipeline: empty (no candidates yet)")

    # Learning trajectory
    if learning_trajectory:
        trend = learning_trajectory.get("trend", "insufficient_data")
        slope = learning_trajectory.get("slope")
        recent_rate = learning_trajectory.get("recent_s1_rate")
        line = f"Learning Trend: {trend}"
        if slope is not None:
            line += f" (slope: {slope*100:+.2f}%/experiment)"
        if recent_rate is not None:
            line += f", recent S1 rate: {recent_rate*100:.1f}%"
        sections.append(line)

    # Active campaign
    if campaign:
        sections.append(
            f"Active Campaign: {campaign.get('title', '?')}\n"
            f"  Objective: {campaign.get('objective', '?')}\n"
            f"  Status: {campaign.get('status', 'active')}"
        )

    # Grammar weight changes (top 5 biggest deltas)
    if grammar_weights and default_weights:
        deltas = []
        for cat in sorted(grammar_weights.keys()):
            cur = grammar_weights.get(cat, 1.0)
            base = default_weights.get(cat, 1.0)
            if abs(cur - base) > 0.1:
                arrow = "+" if cur > base else ""
                deltas.append((cat, base, cur, cur - base))
        if deltas:
            deltas.sort(key=lambda x: abs(x[3]), reverse=True)
            lines = ["Grammar Weight Changes (biggest shifts):"]
            for cat, base, cur, delta in deltas[:5]:
                lines.append(f"  {cat}: {base:.1f} -> {cur:.1f} ({delta:+.1f})")
            sections.append("\n".join(lines))

    # Top programs
    if top_programs:
        lines = [f"Top {len(top_programs)} Programs:"]
        for p in top_programs[:3]:
            fp = (p.get("graph_fingerprint") or "?")[:16]
            loss = p.get("loss_ratio")
            novelty = p.get("novelty_score")
            tier = p.get("tier", "screening")
            parts = [f"{fp} ({tier})"]
            if loss is not None:
                parts.append(f"loss={loss:.4f}")
            if novelty is not None:
                parts.append(f"novelty={novelty:.3f}")
            lines.append(f"  {', '.join(parts)}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{n / total * 100:.0f}%"
