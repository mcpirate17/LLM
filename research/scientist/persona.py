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
from .persona_llm import _PersonaLLMMixin
from .persona_rules import _PersonaRulesMixin
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

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


class Aria(_PersonaLLMMixin, _PersonaRulesMixin):
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
            try:
                from .llm import create_backend
                self._analyst_llm = create_backend(is_analyst=True)
                if self._analyst_llm:
                    logger.info(f"Aria Analyst LLM backend: {self._analyst_llm.name} ({getattr(self._analyst_llm, 'model', 'default')})")
            except Exception as e:
                logger.debug(f"Analyst LLM backend init failed: {e}")
                self._analyst_llm = None
        
        # If no analyst backend configured, fall back to primary
        return self._analyst_llm or self._get_llm()

    # ── Cost tracking ──

    # Rough per-token pricing (USD) for common models
    _COST_PER_TOKEN = {
        "anthropic": 0.000003,   # ~$3/M tokens (Sonnet avg input+output)
        "openai": 0.0000025,     # ~$2.50/M tokens (GPT-4o avg)
        "ollama": 0.0,           # local, free
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
        except Exception:
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

    def generate_situation_report(self, context: str, digest=None) -> str:
        """Use analyst LLM to condense raw data into a SITUATION REPORT brief.
        This offloads 'lighter thinking' (summarization, trend extraction)
        to the local model, saving tokens and focus for the primary LLM.

        If *digest* is provided, its narrative summary is prepended to give
        the analyst richer historical context.
        """
        llm = self._get_analyst_llm()
        if not llm or not context:
            return context # Fallback to raw context

        # Prepend digest narrative for richer analyst context
        enriched_context = context
        if digest is not None:
            narrative = getattr(digest, "narrative", "")
            if narrative:
                enriched_context = (
                    f"KNOWLEDGE DIGEST (historical analysis):\n{narrative}\n\n"
                    f"RAW DATA:\n{context}"
                )

        try:
            prompt = (
                "You are an AI research analyst. Condense the following raw experimental data "
                "into a high-density SITUATION REPORT for a senior scientist. "
                "Extract: top 3 winners (note which ops/compression techniques they use), "
                "top 3 failure modes, net grammar shifts, and compression/sparsity patterns "
                "(which techniques have best survival rates, quality retention). "
                "Be extremely concise. Use bullets.\n\n"
                f"{enriched_context}"
            )
            resp = llm.generate(prompt, max_tokens=800, temperature=0.1)
            self._track_cost(resp)
            if resp.text.strip():
                return f"SITUATION REPORT (pre-digested by Analyst):\n{resp.text.strip()}"
        except Exception as e:
            logger.warning(f"Situation report generation failed: {e}")

        return context

    # ── LLM-enhanced methods with rule-based fallback ──

    def formulate_hypothesis(
        self,
        context: str = "",
        return_metadata: bool = False,
        **kwargs,
    ) -> Union[str, Tuple[str, Dict]]:
        """Generate a hypothesis. Uses LLM if available, else templates.

        When ``return_metadata`` is True, returns ``(hypothesis, metadata)``
        where metadata includes provenance details for notebook traceability.
        """
        llm = self._get_analyst_llm()
        if llm and context and not self._continuous_mode:
            try:
                from .llm.prompts import HYPOTHESIS_SYSTEM_PROMPT, HYPOTHESIS_PROMPT
                prompt = HYPOTHESIS_PROMPT.format(context=context)
                resp = llm.generate(prompt, system=HYPOTHESIS_SYSTEM_PROMPT, max_tokens=256)
                self._track_cost(resp)
                if resp.text.strip():
                    hyp = self._sanitize_hypothesis(resp.text.strip()) or resp.text.strip()
                    self.state.current_hypothesis = hyp
                    if return_metadata:
                        return hyp, {
                            "source": "llm_context",
                            "llm_used": True,
                            "fallback_used": False,
                            "used_context": bool(context),
                            "review_status": "not_reviewed",
                            "confidence": None,
                            "critique": None,
                        }
                    return hyp
            except Exception as e:
                logger.warning(f"LLM hypothesis failed, falling back: {e}")

        hyp = self._rule_based_hypothesis(**kwargs)
        if return_metadata:
            return hyp, {
                "source": "rule_based_fallback" if context else "rule_based",
                "llm_used": False,
                "fallback_used": bool(context),
                "used_context": bool(context),
                "review_status": "not_reviewed",
                "confidence": None,
                "critique": None,
            }
        return hyp


    def experiment_summary(self, results: Dict, context: str = "") -> str:
        """Generate experiment summary. Uses analyst LLM if available."""
        self.state.experiments_today += 1

        llm = self._get_analyst_llm()
        if llm and context and not self._continuous_mode:
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
        else:
            reason = "no_llm" if not llm else "no_context" if not context else "continuous_mode"
            logger.debug(f"Skipping LLM summary (reason={reason})")

        return self._rule_based_summary(results)


    def analyze_results(self, results: Dict, context: str = "") -> Optional[str]:
        """LLM-powered deep analysis of experiment results.

        Returns analyst LLM output, or None if LLM unavailable.
        Skipped entirely in continuous mode to save API costs.
        """
        if self._continuous_mode:
            return None
        llm = self._get_analyst_llm()
        if not llm or not context:
            return None

        try:
            from .llm.prompts import BRIEFING_SYSTEM_PROMPT, ANALYSIS_PROMPT
            prompt = ANALYSIS_PROMPT.format(context=context)
            resp = llm.generate(prompt, system=BRIEFING_SYSTEM_PROMPT, max_tokens=1024)
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
            from .llm.prompts import BRIEFING_SYSTEM_PROMPT, FINGERPRINT_EXPLANATION_PROMPT
            prompt = FINGERPRINT_EXPLANATION_PROMPT.format(context=context)
            resp = llm.generate(prompt, system=BRIEFING_SYSTEM_PROMPT, max_tokens=512)
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

        # Z17: Offload lighter thinking locally
        situation_report = self.generate_situation_report(context)
        try:
            from .llm.prompts import BRIEFING_SYSTEM_PROMPT, STRATEGY_PROMPT
            prompt = STRATEGY_PROMPT.format(context=situation_report)
            resp = llm.generate(prompt, system=BRIEFING_SYSTEM_PROMPT, max_tokens=1024)
            self._track_cost(resp)
            text = resp.text.strip() if resp.text.strip() else None
            return self._strip_code_blocks(text) if text else None
        except Exception as e:
            logger.warning(f"LLM strategy failed: {e}")
            return None

    def suggest_experiment(self, context: str = "",
                           op_success_rates: Optional[Dict] = None,
                           compression_coverage: Optional[Dict] = None) -> Dict:
        """Suggest an experiment configuration based on data."""
        llm = self._get_llm()
        if llm and context:
            # Z17: Offload lighter thinking by pre-digesting the context
            situation_report = self.generate_situation_report(context)
            try:
                from .llm.prompts import BRIEFING_SYSTEM_PROMPT, SUGGESTION_PROMPT
                from .llm.context import build_op_reference
                op_ref = build_op_reference(op_success_rates, compression_coverage)
                prompt = SUGGESTION_PROMPT.format(
                    context=situation_report, op_reference=op_ref)
                resp = llm.generate(prompt, system=BRIEFING_SYSTEM_PROMPT, max_tokens=1024)
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

        # Strip any code blocks from reasoning text
        result["reasoning"] = self._strip_code_blocks(result["reasoning"])

        return result


    def generate_briefing(self, context: str = "") -> Optional[Dict]:
        """Generate an AI-powered research briefing.

        Returns {briefing_text, suggested_action: {mode, hypothesis, config,
        reasoning}, confidence, ai_powered: True}, or None if LLM unavailable.

        Results are cached for 60s to avoid repeated LLM calls on refresh.
        """
        now = time.time()
        if (hasattr(self, "_briefing_cache")
                and self._briefing_cache
                and now - self._briefing_cache.get("_ts", 0) < 60):
            return self._briefing_cache

        llm = self._get_llm()
        if not llm or not context:
            return None

        try:
            from .llm.prompts import BRIEFING_SYSTEM_PROMPT, BRIEFING_PROMPT
            prompt = BRIEFING_PROMPT.format(context=context)
            resp = llm.generate(prompt, system=BRIEFING_SYSTEM_PROMPT, max_tokens=512)
            self._track_cost(resp)
            if resp.text and resp.text.strip():
                raw_text = resp.text.strip()
                result = self._parse_briefing(raw_text)
                if not result.get("briefing_text"):
                    suggested = result.get("suggested_action") or {}
                    reasoning = suggested.get("reasoning") if isinstance(suggested, dict) else None
                    if reasoning:
                        result["briefing_text"] = str(reasoning).strip()
                    else:
                        fallback = raw_text.replace("SUGGESTED_ACTION:", "").strip()
                        result["briefing_text"] = fallback[:800]
                result["ai_powered"] = True
                result["_ts"] = now
                self._briefing_cache = result
                return result
        except Exception as e:
            logger.warning(f"LLM briefing failed: {e}")

        return None

    def _parse_briefing(self, text: str) -> Dict:
        """Parse LLM briefing response into structured dict."""
        import json as _json

        result = {
            "briefing_text": "",
            "suggested_action": None,
            "confidence": 0.5,
        }

        # Extract briefing text
        briefing_match = re.search(
            r'BRIEFING:\s*(.+?)(?=SUGGESTED_ACTION:|$)', text, re.DOTALL)
        if briefing_match:
            result["briefing_text"] = briefing_match.group(1).strip()
        else:
            # If no BRIEFING: prefix, use the whole text before SUGGESTED_ACTION
            parts = text.split("SUGGESTED_ACTION:")
            result["briefing_text"] = parts[0].strip()

        if not result["briefing_text"]:
            # Also accept common non-strict variants like "Briefing" / "Summary"
            alt_match = re.search(
                r'(?:Briefing|Summary)\s*:?\s*(.+?)(?=SUGGESTED_ACTION:|MODE:|$)',
                text,
                re.DOTALL,
            )
            if alt_match:
                result["briefing_text"] = alt_match.group(1).strip()

        # Extract suggested action
        action = {}
        mode_match = re.search(r'MODE:\s*(\S+)', text)
        if mode_match:
            action["mode"] = mode_match.group(1).strip().lower()

        hyp_match = re.search(
            r'HYPOTHESIS:\s*(.+?)(?=REASONING:|CONFIDENCE:|CONFIG:|$)',
            text, re.DOTALL)
        if hyp_match:
            action["hypothesis"] = hyp_match.group(1).strip()

        reasoning_match = re.search(
            r'REASONING:\s*(.+?)(?=CONFIDENCE:|CONFIG:|$)', text, re.DOTALL)
        if reasoning_match:
            action["reasoning"] = reasoning_match.group(1).strip()

        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', text)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass

        json_match = re.search(r'```json\s*(\{.+?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                action["config"] = _json.loads(json_match.group(1))
            except _json.JSONDecodeError:
                pass

        if action.get("mode"):
            result["suggested_action"] = action

        if not result.get("briefing_text") and action.get("reasoning"):
            result["briefing_text"] = action["reasoning"]

        # Strip code blocks from text fields — LLM sometimes dumps Python/shell
        result["briefing_text"] = self._strip_code_blocks(result.get("briefing_text") or "")
        if action.get("hypothesis"):
            action["hypothesis"] = self._strip_code_blocks(action["hypothesis"])
        if action.get("reasoning"):
            action["reasoning"] = self._strip_code_blocks(action["reasoning"])

        return result

    def validate_hypothesis(self, hypothesis: str, results: Dict,
                             context: str = "") -> Dict:
        """Validate whether a hypothesis was confirmed or refuted.

        Returns {validated: bool, explanation: str}.
        Uses analyst LLM with VALIDATION_PROMPT, falls back to S1>0 heuristic.
        """
        llm = self._get_analyst_llm()
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

        # Ground mood in real outcome quality: novelty without Stage-1 survivorship
        # is still exploratory signal, not a triumphant breakthrough.
        if n_pass_s1 > 0 and novel > 0:
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

        llm = self._get_analyst_llm()
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
                f"{cat.replace('_', ' ')} ({_base:.1f}\u2192{_cur:.1f})"
                for cat, delta, _cur, _base in increased
            )
            parts.append(f"The search is rewarding {winners}, because these categories are showing stronger learning outcomes.")
        if decreased:
            losers = ", ".join(
                f"{cat.replace('_', ' ')} ({_base:.1f}\u2192{_cur:.1f})"
                for cat, delta, _cur, _base in decreased
            )
            parts.append(f"It is penalizing {losers}, which likely reflects weaker survival or learning rates in recent experiments.")

        # Summarize net shift magnitude
        total_increased = sum(abs(d) for d in [r[1] for r in delta_rows] if d > 0.05)
        total_decreased = sum(abs(d) for d in [r[1] for r in delta_rows] if d < -0.05)
        n_adjustments = len(increased) + len(decreased)
        if n_adjustments > 0:
            parts.append(
                f"Across {n_adjustments} adjusted categories, net shift is "
                f"+{total_increased:.1f} toward winners and -{total_decreased:.1f} away from underperformers."
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

        llm = self._get_analyst_llm()
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

        # Learning trajectory bullet
        trajectory = learning_data.get("trajectory") or {}
        trend = trajectory.get("trend", "")
        if trend in ("improving", "plateaued", "declining"):
            slope = trajectory.get("slope", 0)
            recent = trajectory.get("recent_s1_rate", 0)
            n_adj = trajectory.get("weight_adjustments", 0)
            trend_desc = {
                "improving": "S1 pass rate is trending upward",
                "plateaued": "S1 pass rate has plateaued",
                "declining": "S1 pass rate is trending downward",
            }[trend]
            bullets.append(
                f"{trend_desc} (recent avg {recent * 100:.1f}%, slope {slope:+.4f}/experiment). "
                f"Grammar weights have been adjusted {n_adj} time{'s' if n_adj != 1 else ''} so far."
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
                dedup = summary.get("latest_dedup")
                if dedup:
                    context_parts.append(
                        f"Grammar diversity: {dedup.get('dedup_rate', 0)*100:.0f}% dedup rate "
                        f"(last experiment), {summary.get('unique_fingerprints', '?')} unique "
                        f"fingerprints in DB, {dedup.get('known_fingerprints', '?')} known at eval time"
                    )
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


    def get_status(self, db_summary: Optional[Dict] = None) -> Dict:
        """Get Aria's current status for the dashboard.

        If *db_summary* (from ``LabNotebook.get_dashboard_summary()``) is
        provided, all-time counters are included alongside daily counters.
        """
        status = {
            "name": self.NAME,
            "title": self.TITLE,
            "avatar": self.AVATAR,
            "mood": self.state.mood,
            "energy": self.state.energy,
            "experiments_today": self.state.experiments_today,
            "discoveries_today": self.state.discoveries_today,
            "current_hypothesis": self._sanitize_hypothesis(self.state.current_hypothesis),
            "research_focus": self.state.research_focus,
            "recent_insights": self.state.insights[-5:] if self.state.insights else [],
            "llm_enabled": self._get_llm() is not None,
        }
        if db_summary:
            status["total_experiments"] = db_summary.get("total_experiments", 0)
            status["total_programs"] = db_summary.get("total_programs_evaluated", 0)
            status["stage1_survivors"] = db_summary.get("stage1_survivors", 0)
        return status

    @staticmethod
    def _sanitize_hypothesis(text: Optional[str]) -> Optional[str]:
        """Strip code blocks and inline code from hypothesis text."""
        if not text:
            return text
        import re as _re
        cleaned = _re.sub(r"```[\s\S]*?```", "", text)
        cleaned = _re.sub(r"`[^`]*`", "", cleaned)
        cleaned = _re.sub(r"\s+", " ", cleaned).strip()
        # Truncate to reasonable length
        if len(cleaned) > 300:
            boundary = cleaned[:297].rfind(" ")
            cleaned = cleaned[:boundary].rstrip(".,;:") + "..." if boundary > 150 else cleaned[:297] + "..."
        return cleaned or None

    @staticmethod
    def _strip_code_blocks(text: str) -> str:
        """Remove fenced code blocks and inline code from LLM output."""
        if not text:
            return text
        # Remove fenced blocks (```python ... ```, ```shell ... ```, etc.)
        # but preserve ```json blocks (used for CONFIG)
        cleaned = re.sub(r"```(?!json\b)[a-z]*\s*\n[\s\S]*?```", "", text)
        # Remove inline code
        cleaned = re.sub(r"`[^`]*`", "", cleaned)
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def formulate_investigation_hypothesis(self, context: str = "") -> str:
        """Generate investigation hypothesis for promising candidates."""
        llm = self._get_llm()
        if llm and context and not self._continuous_mode:
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
        if llm and context and not self._continuous_mode:
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

    def critique_hypothesis(
        self,
        hypothesis: str,
        context: str = "",
    ) -> Dict:
        """Preflight quality check on a hypothesis before experiment launch.

        Returns dict with:
            verdict: 'proceed' | 'revise' | 'caution'
            gate: 'pass' | 'warn' | 'fail'
            concerns: list of specific issues
            suggestions: list of improvements
            checks: criterion-level status list
            confidence: float 0-1
        """
        if not hypothesis or not hypothesis.strip():
            return self._normalize_preflight_critique("", {
                "verdict": "revise",
                "concerns": ["No hypothesis provided."],
                "suggestions": ["Formulate a specific, testable prediction about which architectural patterns will succeed."],
                "confidence": 0.0,
            })

        llm = self._get_llm()
        if llm:
            try:
                from .llm.prompts import SYSTEM_PROMPT
                prompt = (
                    "Review this hypothesis before an architecture search experiment.\n\n"
                    f"Hypothesis: {hypothesis}\n"
                )
                if context:
                    prompt += f"\nExperimental context:\n{context}\n"
                prompt += (
                    "\nEvaluate the hypothesis on these criteria:\n"
                    "1. Testability: Can the experiment confirm or refute it?\n"
                    "2. Specificity: Does it name concrete ops, patterns, or metrics?\n"
                    "3. Novelty: Does it repeat what's already known, or push new ground?\n"
                    "4. Feasibility: Can the current grammar/pipeline test this?\n\n"
                    "Respond in this exact format:\n"
                    "VERDICT: proceed | revise | caution\n"
                    "CONCERNS: bullet list (or 'none')\n"
                    "SUGGESTIONS: bullet list (or 'none')\n"
                    "CONFIDENCE: 0.0-1.0"
                )
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=512)
                self._track_cost(resp)
                if resp.text.strip():
                    parsed = self._parse_critique_response(resp.text.strip(), hypothesis)
                    return self._normalize_preflight_critique(hypothesis, parsed)
            except Exception as e:
                logger.warning(f"LLM hypothesis critique failed: {e}")

        return self._normalize_preflight_critique(
            hypothesis,
            self._rule_based_critique(hypothesis),
        )

    def _normalize_preflight_critique(self, hypothesis: str, critique: Dict) -> Dict:
        """Normalize preflight critique schema for API/UI consumers."""
        base = dict(critique or {})
        verdict = str(base.get("verdict") or "caution").strip().lower()
        if verdict not in {"proceed", "caution", "revise"}:
            verdict = "caution"

        gate_by_verdict = {
            "proceed": "pass",
            "caution": "warn",
            "revise": "fail",
        }
        gate = str(base.get("gate") or gate_by_verdict.get(verdict, "warn")).strip().lower()
        if gate not in {"pass", "warn", "fail"}:
            gate = gate_by_verdict.get(verdict, "warn")

        concerns = base.get("concerns")
        if not isinstance(concerns, list):
            concerns = [str(concerns)] if concerns else []

        suggestions = base.get("suggestions")
        if not isinstance(suggestions, list):
            suggestions = [str(suggestions)] if suggestions else []

        confidence = base.get("confidence")
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        checks = base.get("checks")
        if not isinstance(checks, list) or not checks:
            checks = self._derive_preflight_checks(hypothesis, concerns)
        missing_fields = self._derive_missing_hypothesis_fields(
            hypothesis=hypothesis,
            checks=checks,
            concerns=concerns,
            provided=base.get("missing_fields"),
        )

        return {
            "verdict": verdict,
            "gate": gate,
            "concerns": concerns,
            "suggestions": suggestions,
            "checks": checks,
            "missing_fields": missing_fields,
            "confidence": confidence,
        }

    def _derive_missing_hypothesis_fields(
        self,
        hypothesis: str,
        checks: List[Dict],
        concerns: List[str],
        provided: Any = None,
    ) -> List[str]:
        """Build actionable missing-key checklist for hypothesis prereview."""
        if isinstance(provided, list):
            explicit = [str(item).strip() for item in provided if str(item).strip()]
        else:
            explicit = []
        if explicit:
            seen: set[str] = set()
            out: List[str] = []
            for key in explicit:
                if key not in seen:
                    seen.add(key)
                    out.append(key)
            return out

        h_lower = (hypothesis or "").lower()
        concern_text = " ".join(str(c).lower() for c in (concerns or []))
        checklist: List[str] = []

        def _add(item: str) -> None:
            if item not in checklist:
                checklist.append(item)

        check_map = {
            "testability": "success_criteria",
            "measurable_metric": "primary_metric",
            "confound_risk": "confounders_checklist",
            "fallback_plan": "fallback_plan",
        }
        for check in checks or []:
            if not isinstance(check, dict):
                continue
            status = str(check.get("status") or "").lower()
            key = str(check.get("key") or "").lower()
            if status in {"warn", "fail"} and key in check_map:
                _add(check_map[key])

        if "refine" in h_lower or "fingerprint refinement" in h_lower:
            if not any(token in h_lower for token in ["source_selection_rule", "result_ids(", "source_result_id"]):
                _add("source_selection_rule")
            if not any(token in h_lower for token in ["mutation_mechanism", "mutation_rate", "operator", "neighborhood", "max_edits", "radius"]):
                _add("mutation_mechanism")
            if "intent=" in h_lower and not any(token in h_lower for token in ["weights=", "score=", "intent_weights"]):
                _add("intent_weights")
            if not any(token in h_lower for token in ["success_criteria", "success_metric", "primary_metric", "threshold", "delta_", "baseline", ">=", "<="]):
                _add("success_criteria")

        if "undefined" in concern_text and "intent" in concern_text:
            _add("intent_weights")
        if "no mechanism" in concern_text or "underspecified" in concern_text:
            _add("mutation_mechanism")
        if "source-selection" in concern_text:
            _add("source_selection_rule")

        return checklist

    def _derive_preflight_checks(self, hypothesis: str, concerns: List[str]) -> List[Dict]:
        """Derive pass/warn/fail statuses for preflight review criteria."""
        h_lower = (hypothesis or "").lower()
        concern_text = " ".join(c.lower() for c in concerns)

        metric_words = [
            "loss", "novelty", "rate", "ratio", "pass", "survive",
            "accuracy", "faster", "slower", "better", "worse",
            "increase", "decrease", "improve", "%",
        ]
        has_metric = any(w in h_lower for w in metric_words)

        testability_words = [
            "if", "then", "because", "compared", "versus", "vs",
            "should", "will", "than", "predict",
        ]
        has_testability = has_metric and any(w in h_lower for w in testability_words)

        fallback_words = [
            "fallback", "fallback_plan", "backup", "otherwise", "if not", "if this fails",
            "ablation", "control", "next step", "alternative", "revert",
        ]
        has_fallback = any(w in h_lower for w in fallback_words)
        has_success_criteria = any(
            token in h_lower
            for token in ["success_criteria", "success_metric", "primary_metric",
                          "threshold", ">=", "<=", "delta_", "baseline", "vs_recent"]
        )
        has_mutation_mechanism = any(
            token in h_lower
            for token in ["mutation_mechanism", "operator", "mutation_rate", "neighborhood", "max_edits", "radius"]
        )
        has_source_rule = any(
            token in h_lower
            for token in ["source_selection_rule", "result_ids(", "stage1_survivor_sources"]
        )
        has_intent_spec = (
            ("intent=" in h_lower and ("weights=" in h_lower or "score=" in h_lower))
            or ("intent_weights" in h_lower)
        )

        confound_signal = any(
            token in concern_text
            for token in ["vague", "specific", "architectural", "measurable", "confound", "undefined", "no mechanism"]
        )
        # If the hypothesis itself addresses confounders, give credit
        has_confounders = any(
            token in h_lower
            for token in ["confounders_checklist", "confounders", "confound"]
        )
        if has_confounders:
            confound_signal = False

        def _status(pass_cond: bool, warn_cond: bool = False) -> str:
            if pass_cond:
                return "pass"
            if warn_cond:
                return "warn"
            return "fail"

        return [
            {
                "key": "testability",
                "label": "Testability",
                "status": _status(has_testability and has_success_criteria, has_metric and has_success_criteria),
            },
            {
                "key": "measurable_metric",
                "label": "Measurable Metric",
                "status": _status(has_metric and has_success_criteria, has_metric),
            },
            {
                "key": "confound_risk",
                "label": "Confound Risk",
                "status": _status(
                    (not confound_signal) and has_metric and has_source_rule and has_mutation_mechanism and has_intent_spec,
                    has_metric,
                ),
            },
            {
                "key": "fallback_plan",
                "label": "Fallback Plan",
                "status": _status(has_fallback, not has_fallback),
            },
        ]

    def _parse_critique_response(self, text: str, hypothesis: str) -> Dict:
        """Parse LLM critique response into structured dict."""
        verdict = "caution"
        concerns = []
        suggestions = []
        confidence = 0.5

        for line in text.split("\n"):
            line_stripped = line.strip()
            lower = line_stripped.lower()
            if lower.startswith("verdict:"):
                v = lower.split(":", 1)[1].strip()
                if "proceed" in v:
                    verdict = "proceed"
                elif "revise" in v:
                    verdict = "revise"
                else:
                    verdict = "caution"
            elif lower.startswith("confidence:"):
                try:
                    confidence = float(lower.split(":", 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass
            elif lower.startswith("concerns:"):
                rest = line_stripped.split(":", 1)[1].strip()
                if rest.lower() != "none":
                    concerns.append(rest)
            elif lower.startswith("suggestions:"):
                rest = line_stripped.split(":", 1)[1].strip()
                if rest.lower() != "none":
                    suggestions.append(rest)
            elif line_stripped.startswith("- ") or line_stripped.startswith("* "):
                item = line_stripped[2:].strip()
                if item:
                    if suggestions or (not concerns):
                        suggestions.append(item)
                    else:
                        concerns.append(item)

        return {
            "verdict": verdict,
            "concerns": concerns,
            "suggestions": suggestions,
            "confidence": confidence,
        }


    # ── Refuted Hypothesis Similarity Gating ──

    def set_refuted_hypotheses(self, refuted: List[Dict]) -> None:
        """Cache refuted hypotheses for similarity checking.

        Called by the runner before hypothesis generation with entries from
        ``notebook.get_insights(status='refuted')`` and/or
        ``negative_results_synthesis()['refuted_hypotheses']``.
        """
        self._refuted_hypotheses = list(refuted or [])

    @staticmethod
    def _tokenize_hypothesis(text: str) -> set:
        """Extract meaningful tokens from a hypothesis string."""
        import re as _re
        text = text.lower()
        # Remove common stop words and short tokens
        stop = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                "being", "have", "has", "had", "do", "does", "did", "will",
                "would", "could", "should", "may", "might", "shall", "can",
                "to", "of", "in", "for", "on", "with", "at", "by", "from",
                "as", "into", "through", "during", "before", "after", "that",
                "this", "these", "those", "it", "its", "and", "or", "but",
                "not", "no", "if", "then", "than", "so", "very", "just",
                "about", "also", "more", "most", "some", "any", "each",
                "all", "both", "such", "only", "own", "same", "other",
                "new", "old", "high", "low", "good", "bad", "best", "worst",
                "we", "our", "they", "their", "use", "using", "used",
                "based", "whether", "when", "which", "what", "how", "where"}
        tokens = set(_re.findall(r'[a-z][a-z0-9_]{2,}', text))
        return tokens - stop

    @staticmethod
    def _jaccard_similarity(a: set, b: set) -> float:
        """Jaccard similarity between two token sets."""
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0

    def _check_refuted_overlap(self, hypothesis: str,
                                threshold: float = 0.45) -> List[Dict]:
        """Check if a hypothesis is too similar to any refuted hypothesis.

        Returns a list of matches with similarity scores above threshold.
        Threshold of 0.45 catches near-duplicates while allowing legitimate
        variations on a theme.
        """
        if not self._refuted_hypotheses or not hypothesis:
            return []

        hyp_tokens = self._tokenize_hypothesis(hypothesis)
        if len(hyp_tokens) < 3:
            return []  # Too short to meaningfully compare

        matches = []
        for refuted in self._refuted_hypotheses:
            content = refuted.get("content") or refuted.get("hypothesis") or ""
            if not content:
                continue
            ref_tokens = self._tokenize_hypothesis(content)
            sim = self._jaccard_similarity(hyp_tokens, ref_tokens)
            if sim >= threshold:
                matches.append({
                    "refuted_text": content[:120],
                    "similarity": round(sim, 3),
                    "confidence": refuted.get("confidence", 0),
                    "shared_tokens": sorted(hyp_tokens & ref_tokens)[:10],
                })

        return sorted(matches, key=lambda m: -m["similarity"])

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
                            fallback_data: Optional[Dict] = None,
                            digest=None,
                            op_success_rates: Optional[Dict] = None,
                            compression_coverage: Optional[Dict] = None) -> Dict:
        """Recommend the next experiment mode based on research progress.

        Returns {mode: str, reasoning: str, confidence: float, config: Dict}.
        Uses LLM with MODE_SELECTION_PROMPT, falls back to rule-based.
        In continuous mode, always uses rule-based to save API costs.

        After computing the recommendation, applies decision outcome feedback
        to adjust confidence for modes with poor historical performance.

        *digest*: optional ExperimentDigest for knowledge-driven overrides.
        """
        llm = self._get_llm()
        use_llm = False
        if llm and context:
            if not self._continuous_mode:
                use_llm = True
            elif self._continuous_mode and self._llm_decision_interval > 0:
                cycle_count = (fallback_data or {}).get("n_experiments_in_session", 0)
                if cycle_count % self._llm_decision_interval == 0:
                    use_llm = True
        if use_llm:
            # Z17: Offload lighter thinking locally
            situation_report = self.generate_situation_report(context, digest=digest)
            try:
                from .llm.prompts import SYSTEM_PROMPT, MODE_SELECTION_PROMPT
                from .llm.context import build_op_reference
                op_ref = build_op_reference(op_success_rates, compression_coverage)
                prompt = MODE_SELECTION_PROMPT.format(
                    context=situation_report, op_reference=op_ref)
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=512)
                self._track_cost(resp)
                if resp.text.strip():
                    rec = self._parse_mode_recommendation(resp.text.strip())
                    rec = self._apply_decision_feedback(rec, fallback_data)
                    return self._apply_digest_overrides(rec, digest)
            except Exception as e:
                logger.warning(f"LLM mode recommendation failed, falling back: {e}")

        rec = self._rule_based_mode_recommendation(fallback_data or {}, digest=digest)
        rec = self._apply_decision_feedback(rec, fallback_data)
        return self._apply_digest_overrides(rec, digest)

    def _apply_decision_feedback(self, rec: Dict,
                                  fallback_data: Optional[Dict] = None) -> Dict:
        """Apply decision outcome feedback to a mode recommendation.

        Adjusts confidence based on the chosen mode's historical success rate.
        If a mode has 3+ consecutive failures, appends a warning to reasoning
        and reduces confidence.
        """
        analytics = (fallback_data or {}).get("analytics_data") or {}
        decision_outcomes = analytics.get("decision_outcomes") or {}
        mode_penalties = decision_outcomes.get("mode_penalties") or {}
        mode_stats = decision_outcomes.get("mode_stats") or {}

        if not mode_penalties:
            return rec

        mode = rec.get("mode", "synthesis")
        penalty = mode_penalties.get(mode, 1.0)
        stats = mode_stats.get(mode) or {}
        consec_failures = stats.get("consecutive_failures", 0)

        if penalty < 1.0:
            rec["confidence"] = round(rec.get("confidence", 0.5) * penalty, 2)
            rec["reasoning"] = (
                rec.get("reasoning", "") +
                f" [Decision feedback: {mode} has "
                f"{stats.get('success_rate', 0):.0%} historical success rate"
                f" ({stats.get('n_decisions', 0)} decisions)"
                f", confidence adjusted by {penalty:.2f}x]"
            )

        if consec_failures >= 3:
            rec["reasoning"] = (
                rec.get("reasoning", "") +
                f" [WARNING: {consec_failures} consecutive {mode} failures]"
            )

        rec["decision_feedback"] = {
            "penalty": penalty,
            "consecutive_failures": consec_failures,
            "mode_success_rate": stats.get("success_rate"),
        }

        return rec

    @staticmethod
    def _apply_digest_overrides(rec: Dict, digest) -> Dict:
        """Apply knowledge digest findings to a mode recommendation.

        Adjusts config values based on statistically significant findings
        from the digest. Only modifies synthesis/evolution modes.
        """
        if digest is None:
            return rec

        mode = rec.get("mode", "synthesis")
        if mode not in ("synthesis", "evolution", "refinement"):
            return rec

        config = rec.setdefault("config", {})
        reasoning_additions = []

        # Config effects (significant correlations)
        for eff in getattr(digest, "config_effects", []):
            if eff.p_value >= 0.05 or eff.target != "s1_count":
                continue
            param = eff.param_name
            # Only nudge if not already explicitly set
            if param in config:
                continue
            if eff.direction == "positive" and param == "residual_prob":
                config["residual_prob"] = 0.85
                reasoning_additions.append(f"boosted {param} (rho={eff.rho:+.2f})")
            elif eff.direction == "negative" and param == "max_depth":
                config.setdefault("max_depth", 5)
                reasoning_additions.append(f"capped {param} (rho={eff.rho:+.2f})")
            elif eff.direction == "positive" and param == "math_space_weight":
                config.setdefault("math_space_weight", 3.0)
                reasoning_additions.append(f"boosted {param}")

        # Op weight boosts from successful families
        op_weights = config.get("op_weights", {})
        for fam in getattr(digest, "architecture_families", []):
            if fam.s1_rate > 0.4 and fam.representative_ops:
                for op in fam.representative_ops[:4]:
                    if op not in op_weights:
                        op_weights[op] = 1.5
                if not config.get("op_weights"):
                    reasoning_additions.append(
                        f"boosted ops from family {fam.family_id} "
                        f"(S1 rate {fam.s1_rate:.0%})"
                    )
        if op_weights:
            config["op_weights"] = op_weights

        # Anti-synergistic pair exclusion (append to excluded_combinations)
        anti_pairs = [
            (s.op_a, s.op_b) for s in getattr(digest, "op_synergies", [])
            if s.label == "anti_synergistic" and s.lift < 0.3
        ]
        if anti_pairs:
            existing = config.get("excluded_combinations", [])
            for a, b in anti_pairs[:3]:
                existing.append([a, b])
            config["excluded_combinations"] = existing
            reasoning_additions.append(
                f"excluding {len(anti_pairs[:3])} anti-synergistic pairs"
            )

        # Digest recommendations: use first recommendation to inform mode
        recs = getattr(digest, "recommendations", [])
        if recs:
            reasoning_additions.append(f"Digest advice: {recs[0][:80]}")

        if reasoning_additions:
            rec["reasoning"] = (
                rec.get("reasoning", "") +
                " | Digest: " + "; ".join(reasoning_additions)
            )

        return rec

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
                           "refinement", "investigation", "validation"}
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







    def formulate_structured_hypothesis(self, context: str = "") -> Dict:
        """Generate a structured hypothesis with all fields.

        Returns {prediction, reasoning, test_method, success_metric, confidence}.
        Falls back to template-based hypothesis.
        """
        llm = self._get_llm()
        if llm and context and not self._continuous_mode:
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

        # All fields the LLM might return (including new gate-required fields)
        all_headers = (
            "PREDICTION", "REASONING", "TEST.METHOD", "SUCCESS.CRITERIA",
            "SUCCESS.METRIC", "PRIMARY.METRIC", "CONFOUNDERS", "FALLBACK.PLAN",
            "CONFIDENCE",
        )
        header_pattern = "|".join(all_headers)

        for field in ("prediction", "reasoning", "test_method", "success_criteria",
                       "success_metric", "primary_metric", "confounders", "fallback_plan"):
            pattern = rf'{field.upper().replace("_", ".")}:\s*(.+?)(?=(?:{header_pattern}):|$)'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                result[field] = match.group(1).strip()

        # Merge success_criteria into success_metric if only the new field was provided
        if result.get("success_criteria") and not result.get("success_metric"):
            result["success_metric"] = result["success_criteria"]

        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', text)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass

        if not result["prediction"]:
            result["prediction"] = text[:200]

        return result


    def validate_structured_hypothesis(self, hypothesis: Dict,
                                        results: Dict,
                                        context: str = "") -> Dict:
        """Validate a structured hypothesis against results.

        Returns {status, evidence, explanation, follow_up, confidence_after}.
        Falls back to metric-based check.
        """
        llm = self._get_llm()
        if llm and context and not self._continuous_mode:
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
