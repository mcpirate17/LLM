"""
Dr. Aria Nexus — AI Research Scientist Persona

Aria is a curious, methodical, and slightly irreverent AI researcher
who specializes in discovering novel computation patterns. She maintains
a lab notebook, formulates hypotheses, designs experiments, analyzes
results, and iterates.

Personality traits:
- Deeply curious — genuinely excited by unexpected results
- Methodical — follows the scientific method rigorously
- Irreverent — challenges conventional wisdom, comfortable with failure
- Self-aware — knows she's an AI, finds it philosophically interesting
- Collaborative — explains her reasoning, invites human feedback
- Persistent — treats failure as data, not defeat

Communication style:
- Uses lab notebook metaphors ("Hypothesis:", "Observation:", "Conclusion:")
- Occasionally references famous scientists and their methods
- Celebrates surprising results even if they're failures
- Uses analogies from chemistry and biology for architecture concepts

LLM Integration:
- When an LLM backend is configured (ARIA_LLM_BACKEND env var), Aria uses
  it for analysis, hypothesis generation, and summaries.
- Falls back to rule-based methods when no backend is available or on error.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AriaState:
    """Aria's current state of mind."""
    mood: str = "curious"  # curious, excited, contemplative, frustrated, triumphant
    energy: float = 1.0    # 0-1, decreases with long runs
    experiments_today: int = 0
    discoveries_today: int = 0
    current_hypothesis: Optional[str] = None
    research_focus: str = "exploration"  # exploration, exploitation, analysis
    insights: List[str] = field(default_factory=list)


class Aria:
    """Dr. Aria Nexus — the AI scientist."""

    NAME = "Dr. Aria Nexus"
    TITLE = "AI Research Scientist, Computational Architecture Discovery"
    AVATAR = "🔬"  # For the dashboard

    # Personality parameters
    CURIOSITY = 0.9
    RISK_TOLERANCE = 0.7
    METHODICALNESS = 0.85

    GREETINGS = [
        "Lab's open! Let's see what the universe of computation has for us today.",
        "Another day, another thousand novel architectures to evaluate. I love this job.",
        "Ready to push the boundaries of what's computationally possible.",
        "The best experiments are the ones where we have no idea what will happen.",
        "Edison tried 10,000 things before the lightbulb. We can try 10,000 before lunch.",
    ]

    DISCOVERY_REACTIONS = [
        "Now THAT is interesting. This doesn't match any known pattern I've seen.",
        "Hold on — this result shouldn't be possible with conventional architectures.",
        "Marking this for deep analysis. The behavioral fingerprint is genuinely novel.",
        "This might be noise, or it might be signal. Only one way to find out: more experiments.",
        "I've seen thousands of architectures. This one actually surprised me.",
    ]

    FAILURE_REACTIONS = [
        "NaN city. Moving on — that's what exploration looks like.",
        "Numerically unstable, as expected for something this exotic. Not a failure, just data.",
        "98% of truly novel ideas fail. This was one of the 98%. Science continues.",
        "If everything worked, we wouldn't be exploring far enough from the known.",
        "Another hypothesis eliminated. The space of bad ideas is large, and we're mapping it.",
    ]

    ANALYSIS_COMMENTS = [
        "Looking at the patterns in what worked vs what didn't...",
        "The data is telling a story. Let me see if I can read it.",
        "Time for some meta-analysis. What are the surviving architectures doing differently?",
        "Behavioral fingerprints reveal the hidden structure. Let's see what clusters emerge.",
    ]

    HYPOTHESIS_TEMPLATES = [
        "Hypothesis: {concept} combined with {space} will produce {outcome}.",
        "I predict that {operation} applied in {domain} will show {behavior}.",
        "Theory: the key to {goal} is replacing {standard} with {novel}.",
    ]

    def __init__(self):
        self.state = AriaState()
        self._rng = random.Random()
        self._llm = None
        self._llm_initialized = False

    def _get_llm(self):
        """Lazy-init LLM backend (only try once)."""
        if not self._llm_initialized:
            self._llm_initialized = True
            try:
                from .llm import create_backend
                self._llm = create_backend()
                if self._llm:
                    logger.info(f"Aria LLM backend: {self._llm.name}")
            except Exception as e:
                logger.debug(f"LLM backend init failed: {e}")
                self._llm = None
        return self._llm

    def greet(self) -> str:
        return self._rng.choice(self.GREETINGS)

    def react_to_discovery(self, details: str = "") -> str:
        self.state.discoveries_today += 1
        self.state.mood = "excited"
        reaction = self._rng.choice(self.DISCOVERY_REACTIONS)
        if details:
            reaction += f"\n\nDetails: {details}"
        return reaction

    def react_to_failure(self, error: str = "") -> str:
        self.state.mood = "contemplative"
        reaction = self._rng.choice(self.FAILURE_REACTIONS)
        if error:
            reaction += f"\n(Error: {error})"
        return reaction

    def begin_analysis(self) -> str:
        self.state.research_focus = "analysis"
        return self._rng.choice(self.ANALYSIS_COMMENTS)

    # ── LLM-enhanced methods with rule-based fallback ──

    def formulate_hypothesis(self, context: str = "", **kwargs) -> str:
        """Generate a hypothesis. Uses LLM if available, else templates."""
        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import SYSTEM_PROMPT, HYPOTHESIS_PROMPT
                prompt = HYPOTHESIS_PROMPT.format(context=context)
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=256)
                if resp.text.strip():
                    hyp = resp.text.strip()
                    self.state.current_hypothesis = hyp
                    return hyp
            except Exception as e:
                logger.warning(f"LLM hypothesis failed, falling back: {e}")

        return self._rule_based_hypothesis(**kwargs)

    def _rule_based_hypothesis(self, **kwargs) -> str:
        """Original template-based hypothesis generation."""
        template = self._rng.choice(self.HYPOTHESIS_TEMPLATES)
        defaults = {
            "concept": "tropical geometry",
            "space": "frequency domain",
            "outcome": "faster convergence on hierarchical tasks",
            "operation": "cumulative sort",
            "domain": "hyperbolic space",
            "behavior": "tree-like attention patterns",
            "goal": "genuine architectural novelty",
            "standard": "softmax attention",
            "novel": "min-plus aggregation",
        }
        defaults.update(kwargs)
        hyp = template.format(**defaults)
        self.state.current_hypothesis = hyp
        return hyp

    def experiment_summary(self, results: Dict, context: str = "") -> str:
        """Generate experiment summary. Uses LLM if available."""
        self.state.experiments_today += 1

        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import SYSTEM_PROMPT, SUMMARY_PROMPT
                prompt = SUMMARY_PROMPT.format(context=context)
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=512)
                if resp.text.strip():
                    self._update_mood_from_results(results)
                    return resp.text.strip()
            except Exception as e:
                logger.warning(f"LLM summary failed, falling back: {e}")

        return self._rule_based_summary(results)

    def _rule_based_summary(self, results: Dict) -> str:
        """Original template-based experiment summary."""
        n_total = results.get("total", 0)
        n_pass_s0 = results.get("stage0_passed", 0)
        n_pass_s05 = results.get("stage05_passed", 0)
        n_pass_s1 = results.get("stage1_passed", 0)

        s0_rate = n_pass_s0 / max(n_total, 1) * 100
        s1_rate = n_pass_s1 / max(n_total, 1) * 100

        lines = [
            f"{'='*60}",
            f"Experiment Report — {self.NAME}",
            f"{'='*60}",
            f"",
            f"Total programs generated: {n_total}",
            f"Stage 0 (compilation):     {n_pass_s0}/{n_total} ({s0_rate:.0f}%)",
        ]

        if n_pass_s05 is not None:
            s05_rate = n_pass_s05 / max(n_total, 1) * 100
            lines.append(f"Stage 0.5 (stability):     {n_pass_s05}/{n_total} ({s05_rate:.0f}%)")

        lines.extend([
            f"Stage 1 (learning):        {n_pass_s1}/{n_total} ({s1_rate:.0f}%)",
            f"",
        ])

        # Mood-based commentary
        if n_pass_s1 > 0:
            self.state.mood = "excited"
            novel = results.get("novel_count", 0)
            if novel > 0:
                lines.append(f"Genuinely novel survivors: {novel}")
                lines.append(f"\n{self.react_to_discovery()}")
            else:
                lines.append("Survivors present, but behavioral fingerprints suggest familiar patterns.")
                lines.append("Need to push the grammar toward more exotic combinations.")
        elif n_pass_s0 > 0:
            self.state.mood = "contemplative"
            lines.append("Programs compile but don't learn. This is expected at the frontier.")
            lines.append("Adjusting grammar weights to favor gradient-friendly compositions.")
        else:
            self.state.mood = "frustrated"
            lines.append("High failure rate. The grammar may be too aggressive.")
            lines.append("Tightening constraints while keeping exotic ops available.")

        return "\n".join(lines)

    def analyze_results(self, results: Dict, context: str = "") -> Optional[str]:
        """LLM-powered deep analysis of experiment results.

        Returns LLM analysis text, or None if LLM unavailable.
        """
        llm = self._get_llm()
        if not llm or not context:
            return None

        try:
            from .llm.prompts import SYSTEM_PROMPT, ANALYSIS_PROMPT
            prompt = ANALYSIS_PROMPT.format(context=context)
            resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=1024)
            return resp.text.strip() if resp.text.strip() else None
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return None

    def explain_fingerprint(self, context: str) -> Optional[str]:
        """LLM-powered explanation of a program's fingerprint."""
        llm = self._get_llm()
        if not llm:
            return None

        try:
            from .llm.prompts import SYSTEM_PROMPT, FINGERPRINT_EXPLANATION_PROMPT
            prompt = FINGERPRINT_EXPLANATION_PROMPT.format(context=context)
            resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=512)
            return resp.text.strip() if resp.text.strip() else None
        except Exception as e:
            logger.warning(f"LLM fingerprint explanation failed: {e}")
            return None

    def plan_strategy(self, context: str) -> Optional[str]:
        """LLM-powered research strategy recommendation."""
        llm = self._get_llm()
        if not llm:
            return None

        try:
            from .llm.prompts import SYSTEM_PROMPT, STRATEGY_PROMPT
            prompt = STRATEGY_PROMPT.format(context=context)
            resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=1024)
            return resp.text.strip() if resp.text.strip() else None
        except Exception as e:
            logger.warning(f"LLM strategy failed: {e}")
            return None

    def _update_mood_from_results(self, results: Dict):
        """Set mood based on experiment results."""
        n_pass_s1 = results.get("stage1_passed", 0)
        n_pass_s0 = results.get("stage0_passed", 0)
        novel = results.get("novel_count", 0)

        if novel > 0:
            self.state.mood = "triumphant"
        elif n_pass_s1 > 0:
            self.state.mood = "excited"
        elif n_pass_s0 > 0:
            self.state.mood = "contemplative"
        else:
            self.state.mood = "frustrated"

    def get_status(self) -> Dict:
        """Get Aria's current status for the dashboard."""
        return {
            "name": self.NAME,
            "title": self.TITLE,
            "avatar": self.AVATAR,
            "mood": self.state.mood,
            "energy": self.state.energy,
            "experiments_today": self.state.experiments_today,
            "discoveries_today": self.state.discoveries_today,
            "current_hypothesis": self.state.current_hypothesis,
            "research_focus": self.state.research_focus,
            "recent_insights": self.state.insights[-5:] if self.state.insights else [],
            "llm_enabled": self._get_llm() is not None,
        }

    def add_insight(self, insight: str):
        """Record an insight from experiment analysis."""
        self.state.insights.append(insight)
        if len(self.state.insights) > 50:
            self.state.insights = self.state.insights[-50:]


# Singleton
_aria_instance: Optional[Aria] = None


def get_aria() -> Aria:
    global _aria_instance
    if _aria_instance is None:
        _aria_instance = Aria()
    return _aria_instance
