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
from .persona_analysis import _PersonaAnalysisMixin
from .persona_hypothesis import _PersonaHypothesisMixin
from .persona_llm import _PersonaLLMMixin
from .persona_rules import _PersonaRulesMixin
from .persona_strategy import _PersonaStrategyMixin
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AriaState:
    """Aria's current state of mind."""

    mood: str = "curious"  # curious, excited, contemplative, frustrated, triumphant
    energy: float = 1.0  # 0-1, decreases with long runs
    experiments_today: int = 0
    discoveries_today: int = 0
    current_hypothesis: Optional[str] = None
    research_focus: str = "exploration"  # exploration, exploitation, analysis
    insights: List[str] = field(default_factory=list)


class Aria(
    _PersonaHypothesisMixin,
    _PersonaAnalysisMixin,
    _PersonaStrategyMixin,
    _PersonaLLMMixin,
    _PersonaRulesMixin,
):
    """Dr. Aria Nexus — the AI scientist.

    All domain methods live in mixins:
    - _PersonaHypothesisMixin: hypothesis formulation, critique, breakthrough
    - _PersonaAnalysisMixin: situation reports, briefings, grammar analysis
    - _PersonaStrategyMixin: mode selection, go/no-go, campaigns, knowledge
    - _PersonaLLMMixin: LLM backend management
    - _PersonaRulesMixin: rule-based fallbacks
    """

    NAME = "Dr. Aria Nexus"
    TITLE = "AI Research Scientist, Computational Architecture Discovery"
    AVATAR = "👩‍🔬"  # For the dashboard

    # Personality parameters
    CURIOSITY = 0.9
    RISK_TOLERANCE = 0.7
    METHODICALNESS = 0.85

    PUBLICATION_MIN_SEEDS = 5
    PUBLICATION_MAX_MULTI_SEED_STD = 0.03
    PUBLICATION_MAX_BASELINE_RATIO = 0.90
    PUBLICATION_MIN_OOD_ROBUSTNESS = 0.67
    PUBLICATION_MIN_HP_ROBUSTNESS = 0.75

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
        self._analyst_llm = None
        self._analyst_llm_initialized = False
        # Cost tracking
        self._total_tokens = 0
        self._total_cost = 0.0  # estimated USD
        self._unknown_cost_backends_warned = set()
        # Cooldown tracking for rule-based mode recommendations
        self._last_compression_rec_cycle: int = -10
        self._last_compression_n_tested: int = 0
        self._last_sparse_rec_cycle: int = -10
        self._last_sparse_n_tested: int = 0
        # When True, all per-cycle methods skip LLM and use rule-based paths.
        # Set by runner when entering continuous mode to save API costs.
        self._continuous_mode: bool = False
        # If >0 in continuous mode, call LLM every N cycles for mode selection.
        self._llm_decision_interval: int = 0
        # Cached refuted hypotheses for similarity gating.
        # Populated by runner via set_refuted_hypotheses() before hypothesis
        # generation so the persona can reject near-duplicates of proven failures.
        self._refuted_hypotheses: List[Dict] = []

    def _get_analyst_llm(self):
        """Lazy-init fast analyst LLM backend."""
        if not self._analyst_llm_initialized:
            self._analyst_llm_initialized = True
            from .persona_llm import _init_llm_backend

            self._analyst_llm = _init_llm_backend(is_analyst=True)

        # If no analyst backend configured, fall back to primary
        return self._analyst_llm or self._get_llm()

    # ── Cost tracking ──

    # Rough per-token pricing (USD) for common models
    _COST_PER_TOKEN = {
        "anthropic": 0.000003,  # ~$3/M tokens (Sonnet avg input+output)
        "openai": 0.0000025,  # ~$2.50/M tokens (GPT-4o avg)
        "ollama": 0.0,  # local, free
    }

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_cost(self) -> float:
        return self._total_cost

    def reset_cost_tracking(self):
        """Reset cost counters (e.g., at start of continuous session)."""
        self._total_tokens = 0
        self._total_cost = 0.0

    def get_llm_config(self) -> Dict:
        """Get current LLM configuration for the dashboard."""
        llm = self._get_llm()
        if llm is None:
            return {
                "backend": None,
                "available": False,
                "configured": False,
                "reachable": False,
            }

        reachable = True
        try:
            if hasattr(llm, "is_available"):
                reachable = bool(llm.is_available())
        except Exception as exc:
            logger.debug("LLM availability check failed: %s", exc)
            reachable = False

        config: Dict = {
            "backend": llm.name,
            "available": reachable,
            "configured": True,
            "reachable": reachable,
        }
        if hasattr(llm, "model"):
            config["model"] = llm.model
        if hasattr(llm, "host"):
            config["host"] = llm.host
        # Never expose the full API key
        if hasattr(llm, "api_key") and llm.api_key:
            key = llm.api_key
            config["api_key_set"] = True
            config["api_key_hint"] = (
                key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
            )
        else:
            config["api_key_set"] = False
        return config

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


# Singleton
_aria_instance: Optional[Aria] = None


def get_aria() -> Aria:
    global _aria_instance
    if _aria_instance is None:
        _aria_instance = Aria()
    return _aria_instance
