"""Hypothesis and campaign context builders for LLM prompts."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .context_experiment import build_history_context


def _knowledge_canonical(raw: str) -> str:
    text = " ".join(str(raw or "").split()).strip().lower()
    text = re.sub(r"\b\d+(?:\.\d+)?%?\b", "#", text)
    text = re.sub(r"[^a-z0-9#\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_KNOWLEDGE_STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "into", "when", "then", "than", "were", "been",
    "have", "has", "had", "are", "was", "show", "shows", "showed", "over", "under", "across", "between",
    "using", "use", "used", "high", "low", "very", "more", "less", "near", "around", "recent", "experiments",
    "experiment", "result", "results", "indicate", "indicates", "suggest", "suggests", "mode", "patterns",
    "pattern", "architecture", "architectures",
}


def _knowledge_tokens(raw: str) -> set[str]:
    canonical = _knowledge_canonical(raw)
    return {
        tok for tok in canonical.split()
        if len(tok) > 3 and tok not in _KNOWLEDGE_STOPWORDS
    }


def _knowledge_low_signal(row: Dict) -> bool:
    title = " ".join(str(row.get("title") or "").split()).strip().lower()
    content = " ".join(str(row.get("content") or "").split()).strip().lower()
    if not title or not content:
        return True
    if len(title) < 12 or len(content) < 40:
        return True
    if title.startswith("recent experiments show ") or title.startswith("all recent experiments show "):
        return True
    if "..." in title or "..." in content:
        return True
    if "[principle/" in title or "hybrid? no" in title:
        return True
    if "$" in content or "\\approx" in content:
        return True
    return False


def _knowledge_score(row: Dict) -> float:
    eff = float(row.get("effective_confidence", row.get("confidence", 0.5)) or 0.5)
    validated = int(row.get("times_validated") or 0)
    bonus = min(0.08, (max(validated, 0) ** 0.5) * 0.015)
    penalty = 0.22 if _knowledge_low_signal(row) else 0.0
    return eff + bonus - penalty


def _select_knowledge_for_llm(knowledge: List[Dict], limit: int = 6) -> List[Dict]:
    rows = list(knowledge or [])
    deduped: List[Dict] = []
    seen = set()
    semantic_seen: List[set[str]] = []
    for row in rows:
        key = (
            _knowledge_canonical(row.get("title") or ""),
            _knowledge_canonical(row.get("content") or ""),
        )
        if key in seen:
            continue
        row_tokens = _knowledge_tokens(f"{row.get('title') or ''} {row.get('content') or ''}")
        if row_tokens:
            near_dup = False
            for tokens in semantic_seen:
                inter = len(row_tokens & tokens)
                union = len(row_tokens | tokens)
                if inter >= 5 and union and (inter / union) >= 0.22:
                    near_dup = True
                    break
            if near_dup:
                continue
            semantic_seen.append(row_tokens)
        seen.add(key)
        deduped.append(row)
    deduped.sort(
        key=lambda row: (
            _knowledge_score(row),
            int(row.get("times_validated") or 0),
            float(row.get("timestamp") or 0.0),
        ),
        reverse=True,
    )
    selected: List[Dict] = []
    for row in deduped:
        if _knowledge_score(row) < 0.55:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


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
        selected_knowledge = _select_knowledge_for_llm(knowledge, limit=6)
        lines = ["Knowledge Base (relevant insights):"]
        for k in selected_knowledge:
            eff = float(k.get("effective_confidence", k.get("confidence", 0.0)) or 0.0)
            lines.append(
                f"  [{k.get('category', '?')}] {k.get('title', '?')}: "
                f"{k.get('content', '?')[:80]} "
                f"(confidence={eff:.2f}, "
                f"validated {k.get('times_validated', 0)}x)"
            )
        sections.append("\n".join(lines))

    if leaderboard:
        tier_counts: Dict[str, int] = {}
        for entry in leaderboard:
            rid = str(entry.get("result_id") or "")
            if rid.startswith("ref_"):
                continue
            t = entry.get("tier", "screening")
            tier_counts[t] = tier_counts.get(t, 0) + 1
        lines = ["Leaderboard:"]
        for t in ["screening", "investigation", "validation", "breakthrough"]:
            if t in tier_counts:
                lines.append(f"  {t}: {tier_counts[t]} entries")
        sections.append("\n".join(lines))

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
        selected_knowledge = _select_knowledge_for_llm(knowledge, limit=6)
        lines = [f"\nKnowledge Extracted ({len(selected_knowledge)} selected from {len(knowledge)} entries):"]
        for k in selected_knowledge:
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
            loss = exp.get("best_validation_loss_ratio")
            loss_label = "best val_loss_ratio"
            if loss is None:
                loss = exp.get("best_loss_ratio")
                loss_label = "best loss_ratio"
            exp_type = exp.get("experiment_type", "?")
            line = f"  [{exp_type}] {s1}/{total} S1"
            if loss is not None:
                line += f", {loss_label}={loss:.4f}"
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
        selected_knowledge = _select_knowledge_for_llm(knowledge, limit=5)
        lines = ["Knowledge Base:"]
        for k in selected_knowledge:
            lines.append(
                f"  [{k.get('category', '?')}] {k.get('title', '?')}: "
                f"{k.get('content', '?')[:80]}"
            )
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


