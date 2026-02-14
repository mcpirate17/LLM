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


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{n / total * 100:.0f}%"
