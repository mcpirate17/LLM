from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class _PersonaStrategyMixin:
    def plan_strategy(self, context: str) -> Optional[str]:
        """LLM-powered research strategy recommendation."""
        llm = self._get_llm()
        if not llm:
            return None

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

    def suggest_experiment(
        self,
        context: str = "",
        op_success_rates: Optional[Dict] = None,
        compression_coverage: Optional[Dict] = None,
    ) -> Dict:
        """Suggest an experiment configuration based on data."""
        llm = self._get_llm()
        if llm and context:
            situation_report = self.generate_situation_report(context)
            try:
                from .llm.context_experiment import build_op_reference
                from .llm.prompts import BRIEFING_SYSTEM_PROMPT, SUGGESTION_PROMPT

                op_ref = build_op_reference(op_success_rates, compression_coverage)
                prompt = SUGGESTION_PROMPT.format(
                    context=situation_report, op_reference=op_ref
                )
                resp = llm.generate(
                    prompt, system=BRIEFING_SYSTEM_PROMPT, max_tokens=1024
                )
                self._track_cost(resp)
                if resp.text.strip():
                    return self._parse_suggestion(resp.text.strip())
            except Exception as e:
                logger.warning(f"LLM suggestion failed, falling back: {e}")

        return self._rule_based_suggestion()

    def _parse_suggestion(self, text: str) -> Dict:
        """Parse LLM suggestion response into structured dict."""
        import json as _json

        result = {"reasoning": "", "confidence": 0.5, "config": {}}

        reasoning_match = re.search(
            r"REASONING:\s*(.+?)(?=CONFIDENCE:|CONFIG:|```|$)", text, re.DOTALL
        )
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()

        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass

        json_match = re.search(r"```json\s*(\{.+?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                result["config"] = _json.loads(json_match.group(1))
            except _json.JSONDecodeError:
                pass

        if not result["reasoning"]:
            result["reasoning"] = text[:200]

        result["reasoning"] = self._strip_code_blocks(result["reasoning"])
        return result

    def recommend_next_mode(
        self,
        context: str = "",
        fallback_data: Optional[Dict] = None,
        digest=None,
        op_success_rates: Optional[Dict] = None,
        compression_coverage: Optional[Dict] = None,
    ) -> Dict:
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
            situation_report = self.generate_situation_report(context, digest=digest)
            try:
                from .llm.context_experiment import build_op_reference
                from .llm.prompts import MODE_SELECTION_PROMPT, SYSTEM_PROMPT

                op_ref = build_op_reference(op_success_rates, compression_coverage)
                prompt = MODE_SELECTION_PROMPT.format(
                    context=situation_report, op_reference=op_ref
                )
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

    def _apply_decision_feedback(
        self, rec: Dict, fallback_data: Optional[Dict] = None
    ) -> Dict:
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
                rec.get("reasoning", "")
                + f" [Decision feedback: {mode} has "
                + f"{stats.get('success_rate', 0):.0%} historical success rate"
                + f" ({stats.get('n_decisions', 0)} decisions)"
                + f", confidence adjusted by {penalty:.2f}x]"
            )

        if consec_failures >= 3:
            rec["reasoning"] = (
                rec.get("reasoning", "")
                + f" [WARNING: {consec_failures} consecutive {mode} failures]"
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

        for eff in getattr(digest, "config_effects", []):
            if eff.p_value >= 0.05 or eff.target != "s1_count":
                continue
            param = eff.param_name
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

        anti_pairs = [
            (s.op_a, s.op_b)
            for s in getattr(digest, "op_synergies", [])
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

        recs = getattr(digest, "recommendations", [])
        if recs:
            reasoning_additions.append(f"Digest advice: {recs[0][:80]}")

        # Parse actionable signals from digest recommendations
        for rec_text in recs[:3]:
            rec_lower = rec_text.lower()
            # Detect efficiency/compactness recommendations
            if any(
                kw in rec_lower
                for kw in (
                    "compact",
                    "efficien",
                    "sparse",
                    "small",
                    "param",
                    "lightweight",
                    "pruning",
                )
            ):
                for _cat in ("structural", "parameterized"):
                    base = float(
                        op_weights.get(
                            _cat, config.get("category_weights", {}).get(_cat, 1.0)
                        )
                    )
                    config.setdefault("category_weights", {})[_cat] = round(
                        min(8.0, max(base, 2.0)), 2
                    )
                reasoning_additions.append(
                    "applied efficiency bias from digest recommendation"
                )
                break
            # Detect diversity/exploration recommendations
            if any(kw in rec_lower for kw in ("divers", "explor", "novel", "exotic")):
                config.setdefault("grammar_risky_op_prob", 0.2)
                reasoning_additions.append(
                    "boosted risky_op_prob from digest recommendation"
                )
                break

        if reasoning_additions:
            rec["reasoning"] = (
                rec.get("reasoning", "")
                + " | Digest: "
                + "; ".join(reasoning_additions)
            )

        return rec

    def _parse_mode_recommendation(self, text: str) -> Dict:
        """Parse LLM mode recommendation response."""
        import json as _json

        result = {
            "mode": "synthesis",
            "reasoning": "",
            "confidence": 0.5,
            "config": {},
        }

        mode_match = re.search(r"MODE:\s*(\w+)", text)
        if mode_match:
            mode = mode_match.group(1).lower().strip()
            valid_modes = {
                "synthesis",
                "evolution",
                "novelty",
                "refinement",
                "investigation",
                "validation",
            }
            if mode in valid_modes:
                result["mode"] = mode

        reasoning_match = re.search(
            r"REASONING:\s*(.+?)(?=CONFIDENCE:|CONFIG|```|$)", text, re.DOTALL
        )
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()

        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass

        json_match = re.search(r"```json\s*(\{.+?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                result["config"] = _json.loads(json_match.group(1))
            except _json.JSONDecodeError:
                pass

        if not result["reasoning"]:
            result["reasoning"] = text[:200]

        return result

    def generate_go_no_go(self, subject: str, evidence: str, context: str = "") -> Dict:
        """Generate a go/no-go decision.

        Returns {decision, rationale, alternatives, next_steps}.
        """
        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import GO_NO_GO_PROMPT, SYSTEM_PROMPT

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
        result = {
            "decision": "go",
            "rationale": "",
            "alternatives": "",
            "next_steps": "",
        }

        dec_match = re.search(r"DECISION:\s*(\w+)", text, re.IGNORECASE)
        if dec_match:
            d = dec_match.group(1).lower()
            if d in ("go", "no_go", "pivot"):
                result["decision"] = d

        for field in ("rationale", "alternatives", "next_steps"):
            pattern = rf"{field.upper().replace('_', '.')}:\s*(.+?)(?=(?:DECISION|RATIONALE|ALTERNATIVES|NEXT.STEPS):|$)"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                result[field] = match.group(1).strip()

        return result

    def extract_knowledge(
        self, results: List[Dict], hypotheses: List[Dict], context: str = ""
    ) -> List[Dict]:
        """Extract reusable knowledge from results and hypotheses.

        Returns list of {category, title, content, confidence}.
        """
        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import KNOWLEDGE_EXTRACTION_PROMPT, SYSTEM_PROMPT

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
        entries = []
        blocks = re.split(r"---+", text)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            entry: Dict = {}
            for field in ("category", "title", "content"):
                match = re.search(
                    rf"{field.upper()}:\s*(.+?)(?=(?:CATEGORY|TITLE|CONTENT|CONFIDENCE):|$)",
                    block,
                    re.DOTALL | re.IGNORECASE,
                )
                if match:
                    entry[field] = match.group(1).strip()
            conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", block)
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

    def compile_campaign_report(
        self,
        campaign: Dict,
        experiments: List[Dict],
        hypotheses: List[Dict],
        decisions: List[Dict],
        knowledge: List[Dict],
        context: str = "",
    ) -> str:
        """Compile a cross-experiment campaign report."""
        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import CAMPAIGN_REPORT_PROMPT, SYSTEM_PROMPT

                prompt = CAMPAIGN_REPORT_PROMPT.format(context=context)
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=1500)
                self._track_cost(resp)
                if resp.text.strip():
                    return resp.text.strip()
            except Exception as e:
                logger.warning(f"LLM campaign report failed, falling back: {e}")

        return self._rule_based_campaign_report(
            campaign, experiments, hypotheses, decisions, knowledge
        )

    def formulate_campaign(self, context: str = "") -> Dict:
        """Generate a new campaign title/objective/criteria.

        Returns {title, objective, success_criteria}.
        """
        llm = self._get_llm()
        if llm and context:
            try:
                from .llm.prompts import CAMPAIGN_FORMULATION_PROMPT, SYSTEM_PROMPT

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
        result = {
            "title": "Architecture Discovery Campaign",
            "objective": "Discover novel computation patterns",
            "success_criteria": "Find architectures with loss_ratio < 0.5",
        }

        for field in ("title", "objective", "success_criteria"):
            pattern = rf"{field.upper().replace('_', '.')}:\s*(.+?)(?=(?:TITLE|OBJECTIVE|SUCCESS.CRITERIA):|$)"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                result[field] = match.group(1).strip()

        return result
