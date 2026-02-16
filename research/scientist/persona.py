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
import math
import random
import re
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
    AVATAR = "�‍�🔬"  # For the dashboard

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
        # Cost tracking
        self._total_tokens = 0
        self._total_cost = 0.0  # estimated USD
        self._unknown_cost_backends_warned = set()

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

    # ── Cost tracking ──

    # Rough per-token pricing (USD) for common models
    _COST_PER_TOKEN = {
        "anthropic": 0.000003,   # ~$3/M tokens (Sonnet avg input+output)
        "openai": 0.0000025,     # ~$2.50/M tokens (GPT-4o avg)
        "ollama": 0.0,           # local, free
    }

    def _track_cost(self, resp):
        """Accumulate token usage and estimated cost from an LLM response."""
        if resp and resp.tokens_used:
            self._total_tokens += resp.tokens_used
            backend_name = getattr(self._llm, "name", "")
            rate = self._COST_PER_TOKEN.get(backend_name)
            if rate is None:
                rate = self._COST_PER_TOKEN["anthropic"]
                if backend_name and backend_name not in self._unknown_cost_backends_warned:
                    logger.warning(
                        "Unknown LLM backend '%s' for cost estimation; using anthropic default rate.",
                        backend_name,
                    )
                    self._unknown_cost_backends_warned.add(backend_name)
            self._total_cost += resp.tokens_used * rate

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

    def configure_llm(self, backend_name: str, api_key: str = "",
                      model: str = "", host: str = "") -> bool:
        """Configure (or reconfigure) the LLM backend at runtime.

        Returns True if the backend was created successfully.
        """
        from .llm import create_backend_from_config
        try:
            new_backend = create_backend_from_config(
                backend_name, api_key=api_key, model=model, host=host)
            if new_backend and new_backend.is_available():
                self._llm = new_backend
                self._llm_initialized = True
                logger.info(f"Aria LLM reconfigured: {new_backend.name}")
                return True
            elif new_backend:
                # Backend created but not reachable — still set it
                # (might become available later, e.g. Ollama starting up)
                self._llm = new_backend
                self._llm_initialized = True
                logger.warning(f"Aria LLM set to {new_backend.name} but not currently reachable")
                return True
        except Exception as e:
            logger.warning(f"LLM reconfiguration failed: {e}")
        return False

    def get_llm_config(self) -> Dict:
        """Get current LLM configuration for the dashboard."""
        llm = self._get_llm()
        if llm is None:
            return {"backend": None, "available": False}

        config: Dict = {
            "backend": llm.name,
            "available": True,
        }
        if hasattr(llm, "model"):
            config["model"] = llm.model
        if hasattr(llm, "host"):
            config["host"] = llm.host
        # Never expose the full API key
        if hasattr(llm, "api_key") and llm.api_key:
            key = llm.api_key
            config["api_key_set"] = True
            config["api_key_hint"] = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
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

    # ── LLM-enhanced methods with rule-based fallback ──

    def formulate_hypothesis(self, context: str = "", **kwargs) -> str:
        """Generate a hypothesis. Uses LLM if available, else templates."""
        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import SYSTEM_PROMPT, HYPOTHESIS_PROMPT
                prompt = HYPOTHESIS_PROMPT.format(context=context)
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=256)
                self._track_cost(resp)
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
                self._track_cost(resp)
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
            self._track_cost(resp)
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
            self._track_cost(resp)
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
            self._track_cost(resp)
            return resp.text.strip() if resp.text.strip() else None
        except Exception as e:
            logger.warning(f"LLM strategy failed: {e}")
            return None

    def suggest_experiment(self, context: str = "") -> Dict:
        """Suggest an experiment configuration based on data.

        Returns {config: Dict, reasoning: str, confidence: float}.
        Uses LLM with SUGGESTION_PROMPT, falls back to rule-based.
        """
        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import SYSTEM_PROMPT, SUGGESTION_PROMPT
                prompt = SUGGESTION_PROMPT.format(context=context)
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=1024)
                self._track_cost(resp)
                if resp.text.strip():
                    return self._parse_suggestion(resp.text.strip())
            except Exception as e:
                logger.warning(f"LLM suggestion failed, falling back: {e}")

        return self._rule_based_suggestion()

    def _parse_suggestion(self, text: str) -> Dict:
        """Parse LLM suggestion response into structured dict."""
        import json as _json
        import re

        result = {"reasoning": "", "confidence": 0.5, "config": {}}

        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=CONFIDENCE:|CONFIG:|```|$)',
                                     text, re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()

        # Extract confidence
        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', text)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass

        # Extract JSON config
        json_match = re.search(r'```json\s*(\{.+?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                result["config"] = _json.loads(json_match.group(1))
            except _json.JSONDecodeError:
                pass

        if not result["reasoning"]:
            result["reasoning"] = text[:200]

        return result

    def _rule_based_suggestion(self) -> Dict:
        """Rule-based experiment suggestion when LLM unavailable.

        Rotates through diverse configurations to avoid running the same
        experiment parameters every time.
        """
        configs = [
            {
                "reasoning": "Exploring deeper architectures for complex transformations.",
                "config": {
                    "n_programs": 50, "model_dim": 256,
                    "max_depth": 12, "max_ops": 20,
                    "math_space_weight": 2.0, "residual_prob": 0.7,
                },
            },
            {
                "reasoning": "Testing wider, shallower graphs for parallel feature exploration.",
                "config": {
                    "n_programs": 50, "model_dim": 256,
                    "max_depth": 6, "max_ops": 12,
                    "math_space_weight": 1.5, "residual_prob": 0.6,
                },
            },
            {
                "reasoning": "Boosting math space weight for nonlinear diversity.",
                "config": {
                    "n_programs": 50, "model_dim": 256,
                    "max_depth": 10, "max_ops": 16,
                    "math_space_weight": 3.0, "residual_prob": 0.5,
                },
            },
            {
                "reasoning": "Low residual probability to force non-trivial learned paths.",
                "config": {
                    "n_programs": 60, "model_dim": 256,
                    "max_depth": 8, "max_ops": 14,
                    "math_space_weight": 2.0, "residual_prob": 0.3,
                },
            },
            {
                "reasoning": "Minimal operations with high math space for compact architectures.",
                "config": {
                    "n_programs": 50, "model_dim": 256,
                    "max_depth": 8, "max_ops": 10,
                    "math_space_weight": 2.5, "residual_prob": 0.7,
                },
            },
        ]
        idx = self.state.experiments_today % len(configs)
        choice = configs[idx]
        return {
            "reasoning": choice["reasoning"],
            "confidence": 0.3,
            "config": choice["config"],
        }

    def validate_hypothesis(self, hypothesis: str, results: Dict,
                             context: str = "") -> Dict:
        """Validate whether a hypothesis was confirmed or refuted.

        Returns {validated: bool, explanation: str}.
        Uses LLM with VALIDATION_PROMPT, falls back to S1>0 heuristic.
        """
        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import SYSTEM_PROMPT, VALIDATION_PROMPT
                prompt = VALIDATION_PROMPT.format(
                    hypothesis=hypothesis, context=context)
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=512)
                self._track_cost(resp)
                if resp.text.strip():
                    text = resp.text.strip()
                    confirmed = any(w in text.lower() for w in
                                    ["confirmed", "supported", "validated"])
                    return {"validated": confirmed, "explanation": text}
            except Exception as e:
                logger.warning(f"LLM validation failed, falling back: {e}")

        # Rule-based fallback
        s1_passed = results.get("stage1_passed", 0)
        novel = results.get("novel_count", 0)
        confirmed = s1_passed > 0
        if confirmed:
            explanation = (f"Hypothesis partially confirmed: {s1_passed} programs "
                           f"passed Stage 1, {novel} were novel.")
        else:
            explanation = ("Hypothesis refuted: no programs passed Stage 1. "
                           "The proposed approach did not produce learnable architectures.")
        return {"validated": confirmed, "explanation": explanation}

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

    def explain_learning(self, analytics_summary: Dict) -> str:
        """Aria explains what the system has learned from experiments."""
        lines = [f"{'='*50}", f"Learning Report — {self.NAME}", f"{'='*50}", ""]

        # Grammar weight changes
        weights = analytics_summary.get("grammar_weights")
        defaults = analytics_summary.get("default_weights", {})
        if weights and defaults:
            lines.append("Grammar Weight Adjustments:")
            for cat, new_w in sorted(weights.items()):
                old_w = defaults.get(cat, 1.0)
                if abs(new_w - old_w) > 0.1:
                    direction = "increased" if new_w > old_w else "decreased"
                    lines.append(
                        f"  {cat}: {old_w:.1f} -> {new_w:.1f} ({direction})"
                    )
            lines.append("")

        # Top insights
        insights = analytics_summary.get("insights", [])
        if insights:
            lines.append("Key Findings:")
            for insight in insights[:5]:
                lines.append(f"  - {insight}")
            lines.append("")

        # Efficiency frontier
        frontier = analytics_summary.get("frontier_size", 0)
        if frontier > 0:
            lines.append(f"Efficiency frontier: {frontier} Pareto-optimal programs found.")
            lines.append("")

        if not weights and not insights:
            lines.append("Insufficient data for learning yet. Need more experiments.")

        return "\n".join(lines)

    def explain_grammar_weights(
        self,
        default_weights: Dict[str, float],
        learned_weights: Optional[Dict[str, float]],
    ) -> str:
        """Generate a concise plain-language grammar-weight explanation.

        Uses configured LLM backend when available and falls back to a
        deterministic summary when unavailable.
        """
        learned = learned_weights or {}
        if not default_weights:
            return (
                "No grammar-weight baseline is available yet. Run a few experiments so I can "
                "summarize which operation categories are helping or hurting learning."
            )

        llm = self._get_llm()
        if llm:
            try:
                deltas = []
                for category, base in sorted(default_weights.items()):
                    cur = learned.get(category, base)
                    deltas.append(f"- {category}: default={base:.2f}, learned={cur:.2f}, delta={cur - base:+.2f}")
                prompt = (
                    "Summarize these grammar-weight updates for an ML engineer in 3 short sentences. "
                    "Explain which operation categories are being rewarded or penalized and why that "
                    "matters for architecture search.\n\n"
                    + "\n".join(deltas)
                )
                resp = llm.generate(prompt, max_tokens=180)
                self._track_cost(resp)
                if resp.text and resp.text.strip():
                    return resp.text.strip()
            except Exception as e:
                logger.warning("LLM grammar-weight explanation failed, falling back: %s", e)

        delta_rows = []
        for category, base in sorted(default_weights.items()):
            cur = learned.get(category, base)
            delta_rows.append((category, cur - base, cur, base))
        delta_rows.sort(key=lambda row: abs(row[1]), reverse=True)

        increased = [row for row in delta_rows if row[1] > 0.05][:2]
        decreased = [row for row in delta_rows if row[1] < -0.05][:2]

        if not increased and not decreased:
            return (
                "Grammar weights are currently close to default values, which suggests the system has "
                "not yet seen enough consistent evidence to strongly favor specific operation categories."
            )

        parts = []
        if increased:
            winners = ", ".join(
                f"{cat.replace('_', ' ')} (+{delta:.2f})" for cat, delta, _cur, _base in increased
            )
            parts.append(f"The search is rewarding {winners}, because these categories are showing stronger learning outcomes.")
        if decreased:
            losers = ", ".join(
                f"{cat.replace('_', ' ')} ({delta:.2f})" for cat, delta, _cur, _base in decreased
            )
            parts.append(f"It is penalizing {losers}, which likely reflects weaker survival or learning rates in recent experiments.")
        parts.append(
            "In practice, this shifts generation probability toward operation families that are more likely to produce learnable architectures."
        )
        return " ".join(parts)

    def summarize_learning_bullets(self, learning_data: Dict) -> Dict[str, object]:
        """Summarize current learning state into 3-5 concise bullets.

        Uses configured LLM backend when available and falls back to a
        deterministic summary when unavailable.
        """

        summary = learning_data.get("summary") or {}
        grammar_default = learning_data.get("grammar_default") or {}
        grammar_learned = learning_data.get("grammar_learned") or {}
        frontier = learning_data.get("frontier") or []
        clusters = (learning_data.get("clusters") or {}).get("clusters") or []
        recent_experiments = learning_data.get("recent_experiments") or []

        llm = self._get_llm()
        if llm:
            try:
                delta_lines = []
                for category, base in sorted(grammar_default.items()):
                    cur = grammar_learned.get(category, base)
                    delta_lines.append(f"{category}: {base:.2f} -> {cur:.2f}")

                context = (
                    f"Total programs: {summary.get('total_programs_evaluated', 0)}\n"
                    f"Stage1 survivors: {summary.get('stage1_survivors', 0)}\n"
                    f"Survival rate: {summary.get('survival_rate', 0):.4f}\n"
                    f"Frontier size: {len(frontier)}\n"
                    f"Cluster count: {len(clusters)}\n"
                    f"Recent experiments: {len(recent_experiments)}\n"
                    f"Grammar deltas:\n- " + "\n- ".join(delta_lines[:10])
                )
                prompt = (
                    "Write exactly 4 concise bullets for a dashboard card titled 'What I've learned'. "
                    "Each bullet should be one sentence and grounded in the metrics below. "
                    "Avoid hype; focus on actionable interpretation.\n\n"
                    f"{context}"
                )
                resp = llm.generate(prompt, max_tokens=260)
                self._track_cost(resp)
                text = (resp.text or "").strip()
                if text:
                    parsed = []
                    for line in text.splitlines():
                        stripped = line.strip()
                        if not stripped:
                            continue
                        stripped = re.sub(r"^[-*\u2022\d\.\)\s]+", "", stripped).strip()
                        if stripped:
                            parsed.append(stripped)
                    if len(parsed) >= 3:
                        return {"bullets": parsed[:5], "source": "llm"}
            except Exception as e:
                logger.warning("LLM learning-bullet summary failed, falling back: %s", e)

        bullets: List[str] = []

        total = int(summary.get("total_programs_evaluated") or 0)
        survivors = int(summary.get("stage1_survivors") or 0)
        survival_rate = (survivors / max(total, 1)) if total > 0 else 0.0
        bullets.append(
            f"The search has evaluated {total} programs with {survivors} Stage 1 survivors ({survival_rate * 100:.1f}% survival), indicating {'productive' if survival_rate >= 0.03 else 'early-stage'} grammar quality."
        )

        deltas = []
        for category, base in sorted(grammar_default.items()):
            cur = grammar_learned.get(category, base)
            deltas.append((category, float(cur) - float(base)))
        deltas.sort(key=lambda item: abs(item[1]), reverse=True)
        increased = [d for d in deltas if d[1] > 0.05][:2]
        decreased = [d for d in deltas if d[1] < -0.05][:2]
        if increased or decreased:
            parts = []
            if increased:
                parts.append("rewarding " + ", ".join(f"{name.replace('_', ' ')} ({delta:+.2f})" for name, delta in increased))
            if decreased:
                parts.append("downweighting " + ", ".join(f"{name.replace('_', ' ')} ({delta:+.2f})" for name, delta in decreased))
            bullets.append("Grammar adaptation is " + " while ".join(parts) + ".")

        if frontier:
            bullets.append(
                f"The efficiency frontier currently contains {len(frontier)} non-dominated survivor{'s' if len(frontier) != 1 else ''}, which defines the best observed loss-vs-compute trade-offs."
            )

        if clusters:
            best_cluster = max(clusters, key=lambda c: float(c.get("avg_s1_rate") or 0.0))
            bullets.append(
                f"Cluster {best_cluster.get('cluster_id', '?')} is the most productive cohort at {float(best_cluster.get('avg_s1_rate') or 0.0) * 100:.1f}% average S1 pass, suggesting a repeatable design regime."
            )

        if recent_experiments:
            recent = recent_experiments[:5]
            recent_total = sum(int(e.get("n_programs_generated") or 0) for e in recent)
            recent_s1 = sum(int(e.get("n_stage1_passed") or 0) for e in recent)
            recent_rate = recent_s1 / max(recent_total, 1) if recent_total > 0 else 0.0
            bullets.append(
                f"In the most recent experiments, Stage 1 pass rate is {recent_rate * 100:.1f}% ({recent_s1}/{recent_total}), which helps confirm whether recent grammar updates are improving outcomes."
            )

        while len(bullets) < 3:
            bullets.append(
                "Data is still sparse in some analytics slices, so confidence in long-term trends remains provisional."
            )

        return {"bullets": bullets[:5], "source": "rule-based"}

    def generate_report_narrative(self, report_data: Dict) -> str:
        """Generate an executive narrative for the research report.

        Uses LLM if available, falls back to template-based summary.
        """
        llm = self._get_llm()
        if llm:
            try:
                from .llm.prompts import SYSTEM_PROMPT, REPORT_PROMPT
                # Build context string from report data
                context_parts = []
                summary = report_data.get("summary", {})
                if summary:
                    context_parts.append(
                        f"Total experiments: {summary.get('total_experiments', 0)}\n"
                        f"Total programs evaluated: {summary.get('total_programs_evaluated', 0)}\n"
                        f"Stage 1 survivors: {summary.get('stage1_survivors', 0)}\n"
                        f"S1 survival rate: {summary.get('survival_rate', 0):.1%}"
                    )
                top = report_data.get("top_programs", [])
                if top:
                    context_parts.append("Top programs (by loss_ratio):")
                    for p in top[:10]:
                        context_parts.append(
                            f"  - {p.get('graph_fingerprint', '?')[:12]}: "
                            f"loss_ratio={p.get('loss_ratio', '?')}, "
                            f"novelty={p.get('novelty_score', '?')}, "
                            f"similar_to={p.get('most_similar_to', '?')}"
                        )
                op_rates = report_data.get("op_success_rates", [])
                if op_rates:
                    context_parts.append("Op success rates (top 10):")
                    for op in (op_rates[:10] if isinstance(op_rates, list) else []):
                        context_parts.append(
                            f"  - {op.get('op_name', '?')}: "
                            f"s1_rate={op.get('s1_rate', '?')}"
                        )
                failures = report_data.get("failure_patterns", {})
                if failures:
                    context_parts.append(f"Failure patterns: {failures}")
                frontier = report_data.get("efficiency_frontier", [])
                if frontier:
                    context_parts.append(
                        f"Efficiency frontier: {len(frontier)} Pareto-optimal programs"
                    )
                context = "\n".join(context_parts)
                prompt = REPORT_PROMPT.format(context=context)
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=1024)
                self._track_cost(resp)
                if resp.text.strip():
                    return resp.text.strip()
            except Exception as e:
                logger.warning(f"LLM report narrative failed, falling back: {e}")

        # Template-based fallback
        return self._rule_based_report_narrative(report_data)

    def _rule_based_report_narrative(self, report_data: Dict) -> str:
        """Template-based structured markdown report."""
        summary = report_data.get("summary", {})
        total_exp = summary.get("total_experiments", 0)
        completed_exp = summary.get("completed_experiments", 0)
        total_prog = summary.get("total_programs_evaluated", 0)
        s1_passed = summary.get("stage1_survivors", 0)
        top = report_data.get("top_programs", [])
        s1_rate = s1_passed / max(total_prog, 1) * 100
        avg_novelty = summary.get("avg_novelty_score", 0) or 0
        best_novelty = summary.get("top_novelty_score", 0) or 0

        best_lr = top[0].get("loss_ratio", "?") if top else "N/A"

        sections = []

        # Header with key metrics
        sections.append("# Research Report")
        sections.append("")
        sections.append("## Key Metrics")
        sections.append("")
        sections.append(f"| Metric | Value |")
        sections.append(f"|--------|-------|")
        sections.append(f"| Experiments | {completed_exp}/{total_exp} completed |")
        sections.append(f"| Programs evaluated | {total_prog} |")
        sections.append(f"| Stage 1 survivors | {s1_passed} ({s1_rate:.1f}%) |")
        sections.append(f"| Best loss ratio | {best_lr} |")
        sections.append(f"| Avg novelty | {avg_novelty:.3f} |")
        sections.append(f"| Top novelty | {best_novelty:.3f} |")
        sections.append("")

        # Recommendation
        sections.append("## Assessment")
        sections.append("")
        if s1_rate > 15:
            sections.append(
                "The grammar is producing learnable architectures at a strong rate. "
                "Focus on exploitation of top performers and scale-up validation."
            )
        elif s1_rate > 5:
            sections.append(
                "Moderate S1 pass rate. Grammar weight learning should be actively "
                "steering toward productive categories. Consider more experiments."
            )
        elif s1_rate > 0:
            sections.append(
                "Low S1 pass rate — most generated architectures fail to learn. "
                "Grammar weight adjustments should help concentrate on productive ops."
            )
        else:
            sections.append(
                "No programs have passed Stage 1. The search space may need "
                "significant restructuring, or more experiments are needed."
            )
        sections.append("")

        # Recent experiments breakdown
        recent = report_data.get("recent_experiments", [])
        if recent:
            sections.append("## Recent Experiments")
            sections.append("")
            sections.append("| Experiment | Type | Programs | S1 Passed | Best LR | Status |")
            sections.append("|-----------|------|----------|-----------|---------|--------|")
            for exp in recent[:15]:
                exp_id = (exp.get("experiment_id") or "?")[:8]
                exp_type = exp.get("experiment_type", "?")
                n_gen = exp.get("n_programs_generated", 0) or 0
                n_s1 = exp.get("n_stage1_passed", 0) or 0
                blr = exp.get("best_loss_ratio")
                blr_str = f"{blr:.4f}" if blr is not None else "—"
                status = exp.get("status", "?")
                sections.append(f"| {exp_id} | {exp_type} | {n_gen} | {n_s1} | {blr_str} | {status} |")
            sections.append("")

        # Top programs table
        if top:
            sections.append("## Top 10 Programs (by loss ratio)")
            sections.append("")
            sections.append("| Fingerprint | Loss Ratio | Novelty | Confidence | Experiment |")
            sections.append("|------------|------------|---------|------------|------------|")
            for prog in top[:10]:
                fp = (prog.get("graph_fingerprint") or "?")[:12]
                lr = prog.get("loss_ratio")
                lr_str = f"{lr:.4f}" if lr is not None else "?"
                nov = prog.get("novelty_score")
                nov_str = f"{nov:.3f}" if nov is not None else "—"
                nc = prog.get("novelty_confidence")
                nc_str = f"{nc:.2f}" if nc is not None else "—"
                exp_id = (prog.get("experiment_id") or "?")[:8]
                sections.append(f"| {fp} | {lr_str} | {nov_str} | {nc_str} | {exp_id} |")
            sections.append("")

        # Op success rates
        op_rates = report_data.get("op_success_rates", {})
        if op_rates:
            sections.append("## Op Success Rates (top 15 by usage)")
            sections.append("")
            sections.append("| Op | Used | S0% | S0.5% | S1% | Avg Novelty |")
            sections.append("|----|------|-----|-------|-----|-------------|")
            sorted_ops = sorted(op_rates.items(),
                                key=lambda x: x[1]["n_used"], reverse=True)
            for op_name, stats in sorted_ops[:15]:
                n = stats["n_used"]
                s0 = stats.get("s0_rate", 0) * 100
                s05 = stats.get("s05_rate", 0) * 100
                s1 = stats.get("s1_rate", 0) * 100
                nov = stats.get("avg_novelty")
                nov_str = f"{nov:.3f}" if nov else "—"
                sections.append(f"| {op_name} | {n} | {s0:.0f} | {s05:.0f} | {s1:.0f} | {nov_str} |")
            sections.append("")

        # Control experiment comparison (#41)
        gw_data = report_data.get("grammar_weights", {})
        control_cmp = gw_data.get("control_comparison") if isinstance(gw_data, dict) else None
        if control_cmp:
            sections.append("## Control Experiment Analysis")
            sections.append("")
            ctrl = control_cmp["control"]
            lrn = control_cmp["learned"]
            sections.append(f"| Group | Experiments | Programs | S1 Passed | S1 Rate |")
            sections.append(f"|-------|-----------|----------|-----------|---------|")
            sections.append(
                f"| Control (default weights) | {ctrl['experiments']} | "
                f"{ctrl['programs']} | {ctrl['s1_passed']} | "
                f"{ctrl['s1_rate']:.1%} |")
            sections.append(
                f"| Learned weights | {lrn['experiments']} | "
                f"{lrn['programs']} | {lrn['s1_passed']} | "
                f"{lrn['s1_rate']:.1%} |")
            sections.append("")
            sections.append(
                f"**Difference**: {control_cmp['s1_rate_difference']:+.1%} "
                f"(z={control_cmp['z_score']:.2f}, "
                f"{'significant' if control_cmp['significant_at_p05'] else 'not significant'} "
                f"at p<0.05)")
            sections.append("")
            sections.append(f"**Interpretation**: {control_cmp['interpretation']}")
            sections.append("")

        # Grammar weight evolution
        gw_raw = report_data.get("grammar_weights", {})
        if isinstance(gw_raw, dict) and "learned" in gw_raw:
            grammar_weights = gw_raw.get("learned") or {}
            default_weights = gw_raw.get("default") or {}
        else:
            grammar_weights = gw_raw or {}
            default_weights = report_data.get("default_weights", {})
        if grammar_weights:
            sections.append("## Grammar Weights (learned vs default)")
            sections.append("")
            sections.append("| Category | Default | Learned | Change |")
            sections.append("|----------|---------|---------|--------|")
            all_cats = sorted(set(list(grammar_weights.keys()) +
                                  list(default_weights.keys())))
            for cat in all_cats:
                default = default_weights.get(cat, 1.0)
                learned = grammar_weights.get(cat)
                if learned is not None:
                    delta = learned - default
                    arrow = "+" if delta > 0 else ""
                    sections.append(
                        f"| {cat} | {default:.2f} | {learned:.2f} | {arrow}{delta:.2f} |")
                else:
                    sections.append(f"| {cat} | {default:.2f} | — | — |")
            sections.append("")

        return "\n".join(sections)

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

    def formulate_investigation_hypothesis(self, context: str = "") -> str:
        """Generate investigation hypothesis for promising candidates."""
        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import SYSTEM_PROMPT, INVESTIGATION_HYPOTHESIS_PROMPT
                prompt = INVESTIGATION_HYPOTHESIS_PROMPT.format(context=context)
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=512)
                self._track_cost(resp)
                if resp.text.strip():
                    return resp.text.strip()
            except Exception as e:
                logger.warning(f"LLM investigation hypothesis failed: {e}")

        return (
            "Investigation plan: test each candidate with 3 different training "
            "programs (varying loss, optimizer, curriculum). Look for robustness "
            "— candidates that learn with multiple training setups are more likely "
            "to represent genuine architectural innovations rather than lucky "
            "hyperparameter matches."
        )

    def formulate_validation_hypothesis(self, context: str = "") -> str:
        """Generate validation hypothesis for investigation survivors."""
        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import SYSTEM_PROMPT, VALIDATION_ANALYSIS_PROMPT
                prompt = VALIDATION_ANALYSIS_PROMPT.format(context=context)
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=512)
                self._track_cost(resp)
                if resp.text.strip():
                    return resp.text.strip()
            except Exception as e:
                logger.warning(f"LLM validation hypothesis failed: {e}")

        return (
            "Validation hypothesis: candidates that showed robustness across "
            "training programs in investigation will maintain their advantage "
            "at 10x scale with multi-seed evaluation."
        )

    def _extract_breakthrough_metrics_from_context(self, context: str) -> Dict[str, float]:
        """Best-effort parse of validation metrics from free-form context text."""
        parsed: Dict[str, float] = {}
        if not context:
            return parsed

        patterns = {
            "seeds_passed": [r"seeds?_passed\s*[:=]\s*(\d+)", r"seeds\s*[:=]\s*(\d+)\s*/\s*\d+"],
            "total_seeds": [r"total_seeds\s*[:=]\s*(\d+)", r"seeds\s*[:=]\s*\d+\s*/\s*(\d+)"],
            "val_baseline_ratio": [r"val_baseline_ratio\s*[:=]\s*([0-9]*\.?[0-9]+)", r"baseline[^\n]*ratio\s*[:=]\s*([0-9]*\.?[0-9]+)"],
            "multi_seed_std": [r"multi_seed_std\s*[:=]\s*([0-9]*\.?[0-9]+)", r"multi[- ]seed[^\n]*std\s*[:=]\s*([0-9]*\.?[0-9]+)"],
            "ood_robustness": [r"ood_robustness\s*[:=]\s*([0-9]*\.?[0-9]+)"],
            "hp_robustness": [r"hp_robustness\s*[:=]\s*([0-9]*\.?[0-9]+)"],
        }

        for key, key_patterns in patterns.items():
            for pattern in key_patterns:
                m = re.search(pattern, context, re.IGNORECASE)
                if m:
                    try:
                        parsed[key] = float(m.group(1))
                        break
                    except ValueError:
                        continue

        return parsed

    def assess_breakthrough_evidence(
        self,
        context: str = "",
        metrics: Optional[Dict] = None,
    ) -> Dict:
        """Assess whether breakthrough evidence is publication-grade.

        Returns: {label, confidence_band, parsed_metrics, reasons}
        where label is one of: publication_grade, provisional, underspecified.
        """
        merged: Dict[str, float] = {}
        merged.update(self._extract_breakthrough_metrics_from_context(context))
        if metrics:
            for key, value in metrics.items():
                if value is None:
                    continue
                try:
                    merged[key] = float(value)
                except (TypeError, ValueError):
                    continue

        keys_present = set(merged.keys())
        required = {"seeds_passed", "total_seeds", "val_baseline_ratio", "multi_seed_std"}
        if not required.issubset(keys_present):
            return {
                "label": "underspecified",
                "confidence_band": "unknown",
                "parsed_metrics": merged,
                "reasons": ["insufficient_replication_metrics"],
            }

        total_seeds = int(round(merged.get("total_seeds", 0)))
        seeds_passed = int(round(merged.get("seeds_passed", 0)))
        baseline_ratio = float(merged.get("val_baseline_ratio", math.inf))
        multi_seed_std = float(merged.get("multi_seed_std", math.inf))
        ood = merged.get("ood_robustness")
        hp = merged.get("hp_robustness")

        reasons: List[str] = []
        if total_seeds < self.PUBLICATION_MIN_SEEDS:
            reasons.append("seed_count_below_publication_threshold")
        if seeds_passed < total_seeds:
            reasons.append("not_all_seeds_passed")
        if baseline_ratio >= self.PUBLICATION_MAX_BASELINE_RATIO:
            reasons.append("baseline_margin_insufficient")
        if multi_seed_std >= self.PUBLICATION_MAX_MULTI_SEED_STD:
            reasons.append("multi_seed_variability_too_high")
        if ood is not None and ood < self.PUBLICATION_MIN_OOD_ROBUSTNESS:
            reasons.append("ood_robustness_insufficient")
        if hp is not None and hp < self.PUBLICATION_MIN_HP_ROBUSTNESS:
            reasons.append("hp_robustness_insufficient")

        if not reasons:
            if total_seeds >= 8 and multi_seed_std <= 0.02:
                band = "high"
            elif total_seeds >= self.PUBLICATION_MIN_SEEDS and multi_seed_std <= 0.03:
                band = "medium"
            else:
                band = "low"
            return {
                "label": "publication_grade",
                "confidence_band": band,
                "parsed_metrics": merged,
                "reasons": [],
            }

        return {
            "label": "provisional",
            "confidence_band": "low",
            "parsed_metrics": merged,
            "reasons": reasons,
        }

    def announce_breakthrough(self, context: str = "",
                              metrics: Optional[Dict] = None) -> str:
        """Generate breakthrough announcement."""
        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import SYSTEM_PROMPT, BREAKTHROUGH_ANNOUNCEMENT_PROMPT
                prompt = BREAKTHROUGH_ANNOUNCEMENT_PROMPT.format(context=context)
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=512)
                self._track_cost(resp)
                if resp.text.strip():
                    self.state.mood = "triumphant"
                    self.state.discoveries_today += 1
                    return resp.text.strip()
            except Exception as e:
                logger.warning(f"LLM breakthrough announcement failed: {e}")

        evidence = self.assess_breakthrough_evidence(context=context, metrics=metrics)
        self.state.mood = "triumphant"
        self.state.discoveries_today += 1
        if evidence["label"] == "publication_grade":
            return (
                "BREAKTHROUGH DETECTED (publication-grade)! A candidate passed all "
                "three phases and met strict replication thresholds: full multi-seed "
                "pass, tight confidence band, and strong baseline margin. "
                f"Confidence band: {evidence['confidence_band']}."
            )
        if evidence["label"] == "provisional":
            reasons = ", ".join(evidence.get("reasons", [])[:3]) or "replication criteria unmet"
            return (
                "BREAKTHROUGH SIGNAL DETECTED (PROVISIONAL). The candidate is "
                "promising, but publication-grade replication criteria are not fully met yet "
                f"({reasons}). Run additional multi-seed and robustness validation before claiming a breakthrough."
            )
        return (
            "BREAKTHROUGH DETECTED. Evidence packet is currently underspecified for a "
            "publication-grade claim; treat this as a strong internal signal and collect "
            "explicit multi-seed confidence-band metrics before externalizing the claim."
        )

    def recommend_next_mode(self, context: str = "",
                            fallback_data: Optional[Dict] = None) -> Dict:
        """Recommend the next experiment mode based on research progress.

        Returns {mode: str, reasoning: str, confidence: float, config: Dict}.
        Uses LLM with MODE_SELECTION_PROMPT, falls back to rule-based.
        """
        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import SYSTEM_PROMPT, MODE_SELECTION_PROMPT
                prompt = MODE_SELECTION_PROMPT.format(context=context)
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=512)
                self._track_cost(resp)
                if resp.text.strip():
                    return self._parse_mode_recommendation(resp.text.strip())
            except Exception as e:
                logger.warning(f"LLM mode recommendation failed, falling back: {e}")

        return self._rule_based_mode_recommendation(fallback_data or {})

    def _parse_mode_recommendation(self, text: str) -> Dict:
        """Parse LLM mode recommendation response."""
        import json as _json
        import re

        result = {
            "mode": "synthesis",
            "reasoning": "",
            "confidence": 0.5,
            "config": {},
        }

        mode_match = re.search(r'MODE:\s*(\w+)', text)
        if mode_match:
            mode = mode_match.group(1).lower().strip()
            valid_modes = {"synthesis", "evolution", "novelty",
                           "investigation", "validation"}
            if mode in valid_modes:
                result["mode"] = mode

        reasoning_match = re.search(
            r'REASONING:\s*(.+?)(?=CONFIDENCE:|CONFIG|```|$)', text, re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()

        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', text)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass

        json_match = re.search(r'```json\s*(\{.+?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                result["config"] = _json.loads(json_match.group(1))
            except _json.JSONDecodeError:
                pass

        if not result["reasoning"]:
            result["reasoning"] = text[:200]

        return result

    def _rule_based_mode_recommendation(self, data: Dict) -> Dict:
        """Rule-based mode recommendation when LLM is unavailable."""
        total_s1 = data.get("total_s1_survivors", 0)
        avg_novelty = data.get("avg_novelty", 0)
        n_experiments = data.get("n_experiments_in_session", 0)
        investigation_ready = data.get("investigation_ready", 0)
        validation_ready = data.get("validation_ready", 0)

        # Validation candidates waiting -> validate
        if validation_ready > 0:
            return {
                "mode": "validation",
                "reasoning": (f"{validation_ready} candidates passed investigation "
                              "with good robustness. Time to validate at scale."),
                "confidence": 0.8,
                "config": {},
            }

        # Investigation candidates waiting -> investigate
        if investigation_ready >= 2:
            return {
                "mode": "investigation",
                "reasoning": (f"{investigation_ready} screening survivors have "
                              "promising loss ratios. Deepening study with "
                              "multiple training programs."),
                "confidence": 0.7,
                "config": {},
            }

        # No survivors yet -> keep screening
        if total_s1 == 0:
            if n_experiments >= 10:
                return {
                    "mode": "synthesis",
                    "reasoning": ("No S1 survivors after multiple experiments. "
                                  "Recommend pivoting hypothesis or pausing "
                                  "this campaign before spending more budget."),
                    "confidence": 0.8,
                    "config": {
                        "pivot_recommended": True,
                        "stop_recommended": True,
                    },
                }
            return {
                "mode": "synthesis",
                "reasoning": "No S1 survivors yet. Continuing broad exploration.",
                "confidence": 0.6,
                "config": {},
            }

        # Survivors but low novelty -> novelty search
        if total_s1 > 0 and avg_novelty < 0.3:
            return {
                "mode": "novelty",
                "reasoning": (f"Have {total_s1} S1 survivors but avg novelty "
                              f"is only {avg_novelty:.3f}. Using novelty search "
                              "to find behaviorally diverse architectures."),
                "confidence": 0.7,
                "config": {},
            }

        # Good survivors -> evolve
        if total_s1 >= 3:
            return {
                "mode": "evolution",
                "reasoning": (f"{total_s1} diverse S1 survivors provide a good "
                              "seed population. Evolving to optimize."),
                "confidence": 0.6,
                "config": {},
            }

        # Periodic synthesis to avoid getting stuck (every 3rd experiment)
        if n_experiments > 0 and n_experiments % 3 == 0:
            return {
                "mode": "synthesis",
                "reasoning": "Periodic broad exploration to avoid local optima.",
                "confidence": 0.5,
                "config": {},
            }

        # Default: synthesis
        return {
            "mode": "synthesis",
            "reasoning": "Continuing standard screening.",
            "confidence": 0.5,
            "config": {},
        }

    # ── Structured Hypothesis Methods ──

    def formulate_structured_hypothesis(self, context: str = "") -> Dict:
        """Generate a structured hypothesis with all fields.

        Returns {prediction, reasoning, test_method, success_metric, confidence}.
        Falls back to template-based hypothesis.
        """
        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import SYSTEM_PROMPT, STRUCTURED_HYPOTHESIS_PROMPT
                prompt = STRUCTURED_HYPOTHESIS_PROMPT.format(context=context)
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=512)
                self._track_cost(resp)
                if resp.text.strip():
                    return self._parse_structured_hypothesis(resp.text.strip())
            except Exception as e:
                logger.warning(f"LLM structured hypothesis failed, falling back: {e}")

        return self._rule_based_structured_hypothesis()

    def _parse_structured_hypothesis(self, text: str) -> Dict:
        """Parse LLM structured hypothesis response."""
        import re
        result = {
            "prediction": "",
            "reasoning": "",
            "test_method": "",
            "success_metric": "",
            "confidence": 0.5,
        }

        for field in ("prediction", "reasoning", "test_method", "success_metric"):
            pattern = rf'{field.upper().replace("_", ".")}:\s*(.+?)(?=(?:PREDICTION|REASONING|TEST.METHOD|SUCCESS.METRIC|CONFIDENCE):|$)'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                result[field] = match.group(1).strip()

        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', text)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass

        if not result["prediction"]:
            result["prediction"] = text[:200]

        return result

    def _rule_based_structured_hypothesis(self) -> Dict:
        """Template-based structured hypothesis when LLM unavailable.

        Rotates through diverse templates based on experiment count to avoid
        generating identical hypotheses every time.
        """
        templates = [
            {
                "prediction": "Frequency domain operations will discover novel loss surfaces",
                "reasoning": "FFT-based ops explore spectral structure that pointwise ops miss",
                "test_method": "Run synthesis with freq_domain_prob=0.4",
                "success_metric": "s1_pass_rate > 5% and novelty > 0.7",
                "confidence": 0.4,
            },
            {
                "prediction": "Deeper architectures (depth=12) will find lower loss ratios",
                "reasoning": "Deeper graphs can compose more complex transformations",
                "test_method": "Run synthesis with max_depth=12, max_ops=20",
                "success_metric": "best_loss_ratio < 0.4",
                "confidence": 0.35,
            },
            {
                "prediction": "Wider parallel paths improve loss through ensemble-like effects",
                "reasoning": "Multiple parallel branches explore different feature subspaces",
                "test_method": "Run synthesis with max_width=6, split_prob=0.5",
                "success_metric": "s1_pass_rate > 8%",
                "confidence": 0.3,
            },
            {
                "prediction": "Reduction-heavy graphs compress information more effectively",
                "reasoning": "Aggressive reduction forces the network to learn compact representations",
                "test_method": "Run synthesis with reduction category_weight=3.0",
                "success_metric": "best_loss_ratio < 0.35 and s1_pass_rate > 3%",
                "confidence": 0.35,
            },
            {
                "prediction": "Risky operations (inverse, log) unlock unexplored loss basins",
                "reasoning": "Non-monotonic ops create sharper gradients that standard ops cannot",
                "test_method": "Run synthesis with risky_op_prob=0.5",
                "success_metric": "novelty > 0.75",
                "confidence": 0.25,
            },
            {
                "prediction": "Minimal parameterized layers reduce overfitting in small models",
                "reasoning": "Fewer learned parameters force reliance on structural inductive bias",
                "test_method": "Run synthesis with parameterized category_weight=0.5",
                "success_metric": "s1_pass_rate > 10%",
                "confidence": 0.4,
            },
            {
                "prediction": "Split-merge topology variations improve gradient flow diversity",
                "reasoning": "Varied split/merge patterns create different information bottlenecks",
                "test_method": "Run synthesis with split_prob=0.4, merge_mode=weighted",
                "success_metric": "best_loss_ratio < 0.4 and novelty > 0.6",
                "confidence": 0.3,
            },
            {
                "prediction": "Sequence-focused operations capture temporal patterns better",
                "reasoning": "Convolutions and scans along sequence dim exploit local structure",
                "test_method": "Run synthesis with sequence_ops category_weight=2.5",
                "success_metric": "best_loss_ratio < 0.35",
                "confidence": 0.35,
            },
            {
                "prediction": "Math space combinations with high weight yield novel architectures",
                "reasoning": "Mathematical operations (sin, exp, polynomial) add nonlinear diversity",
                "test_method": "Run synthesis with math_space_weight=3.0",
                "success_metric": "s1_pass_rate > 5% and novelty > 0.65",
                "confidence": 0.4,
            },
            {
                "prediction": "Low residual probability forces non-trivial learned transformations",
                "reasoning": "Without residual shortcuts the graph must learn useful operations",
                "test_method": "Run synthesis with residual_prob=0.3",
                "success_metric": "novelty > 0.8",
                "confidence": 0.3,
            },
        ]
        idx = self.state.experiments_today % len(templates)
        return templates[idx]

    def validate_structured_hypothesis(self, hypothesis: Dict,
                                        results: Dict,
                                        context: str = "") -> Dict:
        """Validate a structured hypothesis against results.

        Returns {status, evidence, explanation, follow_up, confidence_after}.
        Falls back to metric-based check.
        """
        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import SYSTEM_PROMPT, HYPOTHESIS_VALIDATION_PROMPT
                prompt = HYPOTHESIS_VALIDATION_PROMPT.format(
                    prediction=hypothesis.get("prediction", ""),
                    reasoning=hypothesis.get("reasoning", ""),
                    success_metric=hypothesis.get("success_metric", ""),
                    context=context,
                )
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=512)
                self._track_cost(resp)
                if resp.text.strip():
                    return self._parse_hypothesis_validation(resp.text.strip())
            except Exception as e:
                logger.warning(f"LLM hypothesis validation failed, falling back: {e}")

        return self._rule_based_hypothesis_validation(hypothesis, results)

    def _parse_hypothesis_validation(self, text: str) -> Dict:
        """Parse LLM hypothesis validation response."""
        import re
        result = {
            "status": "inconclusive",
            "evidence": "",
            "explanation": "",
            "follow_up": None,
            "confidence_after": 0.5,
        }

        status_match = re.search(r'STATUS:\s*(\w+)', text, re.IGNORECASE)
        if status_match:
            s = status_match.group(1).lower()
            if s in ("confirmed", "refuted", "inconclusive"):
                result["status"] = s

        evidence_match = re.search(
            r'EVIDENCE:\s*(.+?)(?=EXPLANATION:|FOLLOW.UP:|CONFIDENCE:|$)',
            text, re.DOTALL | re.IGNORECASE)
        if evidence_match:
            result["evidence"] = evidence_match.group(1).strip()

        expl_match = re.search(
            r'EXPLANATION:\s*(.+?)(?=FOLLOW.UP:|CONFIDENCE:|$)',
            text, re.DOTALL | re.IGNORECASE)
        if expl_match:
            result["explanation"] = expl_match.group(1).strip()

        follow_match = re.search(
            r'FOLLOW.UP:\s*(.+?)(?=CONFIDENCE:|$)',
            text, re.DOTALL | re.IGNORECASE)
        if follow_match:
            fu = follow_match.group(1).strip()
            result["follow_up"] = fu if fu.lower() != "none" else None

        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', text)
        if conf_match:
            try:
                result["confidence_after"] = float(conf_match.group(1))
            except ValueError:
                pass

        return result

    def _rule_based_hypothesis_validation(self, hypothesis: Dict,
                                           results: Dict) -> Dict:
        """Metric-based hypothesis validation when LLM unavailable."""
        import re as _re
        success_metric = hypothesis.get("success_metric", "")
        s1_passed = results.get("stage1_passed", 0)

        # Try to parse "loss_ratio < X" or "s1_pass_rate > X%"
        status = "inconclusive"
        evidence = f"S1 passed: {s1_passed}"

        match = _re.match(r'loss_ratio\s*[<>]=?\s*([\d.]+)', success_metric)
        if match:
            threshold = float(match.group(1))
            best_lr = results.get("best_loss_ratio")
            if best_lr is not None:
                status = "confirmed" if best_lr < threshold else "refuted"
                evidence = f"best_loss_ratio={best_lr:.4f} vs threshold {threshold}"

        match = _re.match(r's1_pass_rate\s*[>]=?\s*([\d.]+)%?', success_metric)
        if match:
            threshold = float(match.group(1)) / 100
            total = results.get("total", 0)
            rate = s1_passed / max(total, 1)
            status = "confirmed" if rate >= threshold else "refuted"
            evidence = f"s1_pass_rate={rate:.1%} vs threshold {threshold:.1%}"

        if status == "inconclusive" and s1_passed > 0:
            status = "confirmed"
            evidence = f"{s1_passed} programs passed S1"

        conf_before = hypothesis.get("confidence", 0.5)
        if status == "confirmed":
            conf_after = min(conf_before + 0.2, 0.95)
        elif status == "refuted":
            conf_after = max(conf_before - 0.3, 0.05)
        else:
            conf_after = conf_before

        return {
            "status": status,
            "evidence": evidence,
            "explanation": f"Hypothesis {status} based on metric check: {evidence}",
            "follow_up": None,
            "confidence_after": conf_after,
        }

    def generate_go_no_go(self, subject: str, evidence: str,
                           context: str = "") -> Dict:
        """Generate a go/no-go decision.

        Returns {decision, rationale, alternatives, next_steps}.
        """
        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import SYSTEM_PROMPT, GO_NO_GO_PROMPT
                prompt = GO_NO_GO_PROMPT.format(context=context)
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=512)
                self._track_cost(resp)
                if resp.text.strip():
                    return self._parse_go_no_go(resp.text.strip())
            except Exception as e:
                logger.warning(f"LLM go/no-go failed, falling back: {e}")

        return self._rule_based_go_no_go(subject, evidence)

    def _parse_go_no_go(self, text: str) -> Dict:
        """Parse LLM go/no-go response."""
        import re
        result = {
            "decision": "go",
            "rationale": "",
            "alternatives": "",
            "next_steps": "",
        }

        dec_match = re.search(r'DECISION:\s*(\w+)', text, re.IGNORECASE)
        if dec_match:
            d = dec_match.group(1).lower()
            if d in ("go", "no_go", "pivot"):
                result["decision"] = d

        for field in ("rationale", "alternatives", "next_steps"):
            pattern = rf'{field.upper().replace("_", ".")}:\s*(.+?)(?=(?:DECISION|RATIONALE|ALTERNATIVES|NEXT.STEPS):|$)'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                result[field] = match.group(1).strip()

        return result

    def _rule_based_go_no_go(self, subject: str, evidence: str) -> Dict:
        """Rule-based go/no-go when LLM unavailable.

        Parses metric values from evidence string and applies thresholds
        instead of rubber-stamping everything as 'go'.
        """
        import re

        # Extract metrics from evidence string (e.g. "loss_ratio=0.45, novelty=0.6")
        lr_match = re.search(r'loss_ratio=([\d.]+)', evidence)
        nov_match = re.search(r'novelty=([\d.]+)', evidence)

        loss_ratio = float(lr_match.group(1)) if lr_match else None
        novelty = float(nov_match.group(1)) if nov_match else None

        decision = "go"
        rationale_parts = []

        if loss_ratio is not None and loss_ratio > 0.5:
            decision = "no_go"
            rationale_parts.append(f"loss_ratio={loss_ratio:.3f} > 0.5 (too weak)")
        elif novelty is not None and novelty < 0.3:
            decision = "no_go"
            rationale_parts.append(f"novelty={novelty:.3f} < 0.3 (not novel enough)")
        elif (loss_ratio is not None and loss_ratio > 0.3
              and novelty is not None and novelty < 0.5):
            decision = "pivot"
            rationale_parts.append(
                f"loss_ratio={loss_ratio:.3f} > 0.3 and novelty={novelty:.3f} < 0.5 "
                f"(mediocre on both axes)")
        else:
            if loss_ratio is not None:
                rationale_parts.append(f"loss_ratio={loss_ratio:.3f}")
            if novelty is not None:
                rationale_parts.append(f"novelty={novelty:.3f}")
            rationale_parts.append("metrics within acceptable range")

        rationale = f"Rule-based {decision}: {'; '.join(rationale_parts)}. {evidence}"

        return {
            "decision": decision,
            "rationale": rationale,
            "alternatives": "No LLM available for detailed analysis",
            "next_steps": ("Proceed to next phase" if decision == "go"
                          else "Consider alternative architectures"
                          if decision == "pivot"
                          else "Candidate rejected — do not escalate"),
        }

    def extract_knowledge(self, results: List[Dict],
                           hypotheses: List[Dict],
                           context: str = "") -> List[Dict]:
        """Extract reusable knowledge from results and hypotheses.

        Returns list of {category, title, content, confidence}.
        """
        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import SYSTEM_PROMPT, KNOWLEDGE_EXTRACTION_PROMPT
                prompt = KNOWLEDGE_EXTRACTION_PROMPT.format(context=context)
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=1024)
                self._track_cost(resp)
                if resp.text.strip():
                    return self._parse_knowledge_entries(resp.text.strip())
            except Exception as e:
                logger.warning(f"LLM knowledge extraction failed, falling back: {e}")

        return self._rule_based_knowledge(results, hypotheses)

    def _parse_knowledge_entries(self, text: str) -> List[Dict]:
        """Parse LLM knowledge extraction response."""
        import re
        entries = []
        blocks = re.split(r'---+', text)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            entry: Dict = {}
            for field in ("category", "title", "content"):
                match = re.search(
                    rf'{field.upper()}:\s*(.+?)(?=(?:CATEGORY|TITLE|CONTENT|CONFIDENCE):|$)',
                    block, re.DOTALL | re.IGNORECASE)
                if match:
                    entry[field] = match.group(1).strip()
            conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', block)
            if conf_match:
                try:
                    entry["confidence"] = float(conf_match.group(1))
                except ValueError:
                    entry["confidence"] = 0.5
            else:
                entry["confidence"] = 0.5

            if entry.get("title") and entry.get("content"):
                entry.setdefault("category", "principle")
                entries.append(entry)

        return entries

    def _rule_based_knowledge(self, results: List[Dict],
                               hypotheses: List[Dict]) -> List[Dict]:
        """Rule-based knowledge extraction when LLM unavailable."""
        entries = []
        # Extract from confirmed hypotheses
        for h in hypotheses:
            if h.get("status") == "confirmed":
                entries.append({
                    "category": "principle",
                    "title": f"Confirmed: {h.get('prediction', '')[:50]}",
                    "content": h.get("outcome_summary", h.get("prediction", ""))[:200],
                    "confidence": h.get("confidence_after", 0.6),
                })
            elif h.get("status") == "refuted":
                entries.append({
                    "category": "anti_pattern",
                    "title": f"Refuted: {h.get('prediction', '')[:50]}",
                    "content": h.get("outcome_summary", h.get("prediction", ""))[:200],
                    "confidence": h.get("confidence_after", 0.6),
                })
        return entries[:5]  # limit to 5

    def compile_campaign_report(self, campaign: Dict,
                                 experiments: List[Dict],
                                 hypotheses: List[Dict],
                                 decisions: List[Dict],
                                 knowledge: List[Dict],
                                 context: str = "") -> str:
        """Compile a cross-experiment campaign report."""
        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import SYSTEM_PROMPT, CAMPAIGN_REPORT_PROMPT
                prompt = CAMPAIGN_REPORT_PROMPT.format(context=context)
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=1500)
                self._track_cost(resp)
                if resp.text.strip():
                    return resp.text.strip()
            except Exception as e:
                logger.warning(f"LLM campaign report failed, falling back: {e}")

        return self._rule_based_campaign_report(
            campaign, experiments, hypotheses, decisions, knowledge)

    def _rule_based_campaign_report(self, campaign: Dict,
                                     experiments: List[Dict],
                                     hypotheses: List[Dict],
                                     decisions: List[Dict],
                                     knowledge: List[Dict]) -> str:
        """Template-based campaign report when LLM unavailable."""
        total_exp = len(experiments)
        total_s1 = sum(e.get("n_stage1_passed", 0) for e in experiments)
        total_programs = sum(e.get("n_programs_generated", 0) for e in experiments)
        confirmed = sum(1 for h in hypotheses if h.get("status") == "confirmed")
        refuted = sum(1 for h in hypotheses if h.get("status") == "refuted")

        lines = [
            f"Campaign Report: {campaign.get('title', 'Untitled')}",
            f"{'=' * 60}",
            f"Objective: {campaign.get('objective', '?')}",
            f"Success Criteria: {campaign.get('success_criteria', '?')}",
            f"Status: {campaign.get('status', '?')}",
            "",
            f"Experiments: {total_exp} completed",
            f"Programs evaluated: {total_programs}",
            f"S1 survivors: {total_s1}",
            f"Hypotheses: {confirmed} confirmed, {refuted} refuted, "
            f"{len(hypotheses) - confirmed - refuted} other",
            f"Decisions: {len(decisions)}",
            f"Knowledge entries: {len(knowledge)}",
        ]
        return "\n".join(lines)

    def formulate_campaign(self, context: str = "") -> Dict:
        """Generate a new campaign title/objective/criteria.

        Returns {title, objective, success_criteria}.
        """
        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import SYSTEM_PROMPT, CAMPAIGN_FORMULATION_PROMPT
                prompt = CAMPAIGN_FORMULATION_PROMPT.format(context=context)
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=512)
                self._track_cost(resp)
                if resp.text.strip():
                    return self._parse_campaign_formulation(resp.text.strip())
            except Exception as e:
                logger.warning(f"LLM campaign formulation failed, falling back: {e}")

        return {
            "title": "Architecture Discovery Campaign",
            "objective": "Discover novel computation patterns that outperform standard attention",
            "success_criteria": "Find 3+ architectures with loss_ratio < 0.5 and novelty > 0.5",
        }

    def _parse_campaign_formulation(self, text: str) -> Dict:
        """Parse LLM campaign formulation response."""
        import re
        result = {
            "title": "Architecture Discovery Campaign",
            "objective": "Discover novel computation patterns",
            "success_criteria": "Find architectures with loss_ratio < 0.5",
        }

        for field in ("title", "objective", "success_criteria"):
            pattern = rf'{field.upper().replace("_", ".")}:\s*(.+?)(?=(?:TITLE|OBJECTIVE|SUCCESS.CRITERIA):|$)'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                result[field] = match.group(1).strip()

        return result

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
