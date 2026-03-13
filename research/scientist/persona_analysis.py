from __future__ import annotations

import logging
import re
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class _PersonaAnalysisMixin:
    def generate_situation_report(self, context: str, digest=None) -> str:
        """Use analyst LLM to condense raw data into a SITUATION REPORT brief.
        This offloads 'lighter thinking' (summarization, trend extraction)
        to the local model, saving tokens and focus for the primary LLM.

        If *digest* is provided, its narrative summary is prepended to give
        the analyst richer historical context.
        """
        llm = self._get_analyst_llm()
        if not llm or not context:
            return context

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

    def generate_briefing(self, context: str = "") -> Optional[Dict]:
        """Generate an AI-powered research briefing.

        Returns {briefing_text, suggested_action: {mode, hypothesis, config,
        reasoning}, confidence, ai_powered: True}, or None if LLM unavailable.

        Results are cached for 60s to avoid repeated LLM calls on refresh.
        """
        now = time.time()
        if (
            hasattr(self, "_briefing_cache")
            and self._briefing_cache
            and now - self._briefing_cache.get("_ts", 0) < 60
        ):
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

        briefing_match = re.search(r"BRIEFING:\s*(.+?)(?=SUGGESTED_ACTION:|$)", text, re.DOTALL)
        if briefing_match:
            result["briefing_text"] = briefing_match.group(1).strip()
        else:
            parts = text.split("SUGGESTED_ACTION:")
            result["briefing_text"] = parts[0].strip()

        if not result["briefing_text"]:
            alt_match = re.search(
                r"(?:Briefing|Summary)\s*:?\s*(.+?)(?=SUGGESTED_ACTION:|MODE:|$)",
                text,
                re.DOTALL,
            )
            if alt_match:
                result["briefing_text"] = alt_match.group(1).strip()

        action = {}
        mode_match = re.search(r"MODE:\s*(\S+)", text)
        if mode_match:
            action["mode"] = mode_match.group(1).strip().lower()

        hyp_match = re.search(
            r"HYPOTHESIS:\s*(.+?)(?=REASONING:|CONFIDENCE:|CONFIG:|$)",
            text,
            re.DOTALL,
        )
        if hyp_match:
            action["hypothesis"] = hyp_match.group(1).strip()

        reasoning_match = re.search(
            r"REASONING:\s*(.+?)(?=CONFIDENCE:|CONFIG:|$)", text, re.DOTALL
        )
        if reasoning_match:
            action["reasoning"] = reasoning_match.group(1).strip()

        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass

        json_match = re.search(r"```json\s*(\{.+?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                action["config"] = _json.loads(json_match.group(1))
            except _json.JSONDecodeError:
                pass

        if action.get("mode"):
            result["suggested_action"] = action

        if not result.get("briefing_text") and action.get("reasoning"):
            result["briefing_text"] = action["reasoning"]

        result["briefing_text"] = self._strip_code_blocks(result.get("briefing_text") or "")
        if action.get("hypothesis"):
            action["hypothesis"] = self._strip_code_blocks(action["hypothesis"])
        if action.get("reasoning"):
            action["reasoning"] = self._strip_code_blocks(action["reasoning"])

        return result

    def _update_mood_from_results(self, results: Dict):
        """Set mood based on experiment results."""
        n_pass_s1 = results.get("stage1_passed", 0)
        n_pass_s0 = results.get("stage0_passed", 0)
        novel = results.get("novel_count", 0)

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

        weights = analytics_summary.get("grammar_weights")
        defaults = analytics_summary.get("default_weights", {})
        if weights and defaults:
            lines.append("Grammar Weight Adjustments:")
            for cat, new_w in sorted(weights.items()):
                old_w = defaults.get(cat, 1.0)
                if abs(new_w - old_w) > 0.1:
                    direction = "increased" if new_w > old_w else "decreased"
                    lines.append(f"  {cat}: {old_w:.1f} -> {new_w:.1f} ({direction})")
            lines.append("")

        insights = analytics_summary.get("insights", [])
        if insights:
            lines.append("Key Findings:")
            for insight in insights[:5]:
                lines.append(f"  - {insight}")
            lines.append("")

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
                    deltas.append(
                        f"- {category}: default={base:.2f}, learned={cur:.2f}, delta={cur - base:+.2f}"
                    )
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

        return self._rule_based_grammar_weight_explanation(default_weights, learned)

    def _rule_based_grammar_weight_explanation(
        self,
        default_weights: Dict[str, float],
        learned: Dict[str, float],
    ) -> str:
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
                f"{cat.replace('_', ' ')} ({_base:.1f}→{_cur:.1f})"
                for cat, delta, _cur, _base in increased
            )
            parts.append(
                f"The search is rewarding {winners}, because these categories are showing stronger learning outcomes."
            )
        if decreased:
            losers = ", ".join(
                f"{cat.replace('_', ' ')} ({_base:.1f}→{_cur:.1f})"
                for cat, delta, _cur, _base in decreased
            )
            parts.append(
                f"It is penalizing {losers}, which likely reflects weaker survival or learning rates in recent experiments."
            )

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

        llm_result = self._summarize_learning_bullets_llm(
            summary=summary,
            grammar_default=grammar_default,
            grammar_learned=grammar_learned,
            frontier=frontier,
            clusters=clusters,
            recent_experiments=recent_experiments,
        )
        if llm_result:
            return llm_result

        bullets = self._summarize_learning_bullets_rule_based(
            learning_data=learning_data,
            summary=summary,
            grammar_default=grammar_default,
            grammar_learned=grammar_learned,
            frontier=frontier,
            clusters=clusters,
            recent_experiments=recent_experiments,
        )
        return {"bullets": bullets[:5], "source": "rule-based"}

    def _summarize_learning_bullets_llm(
        self,
        summary: Dict,
        grammar_default: Dict,
        grammar_learned: Dict,
        frontier: List,
        clusters: List,
        recent_experiments: List,
    ) -> Optional[Dict[str, object]]:
        llm = self._get_analyst_llm()
        if not llm:
            return None

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
            if not text:
                return None

            parsed = []
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                stripped = re.sub(r"^[-*•\d\.\)\s]+", "", stripped).strip()
                if stripped:
                    parsed.append(stripped)
            if len(parsed) >= 3:
                return {"bullets": parsed[:5], "source": "llm"}
        except Exception as e:
            logger.warning("LLM learning-bullet summary failed, falling back: %s", e)
        return None

    def _summarize_learning_bullets_rule_based(
        self,
        learning_data: Dict,
        summary: Dict,
        grammar_default: Dict,
        grammar_learned: Dict,
        frontier: List,
        clusters: List,
        recent_experiments: List,
    ) -> List[str]:
        bullets: List[str] = []

        total = int(summary.get("total_programs_evaluated") or 0)
        survivors = int(summary.get("stage1_survivors") or 0)
        survival_rate = (survivors / max(total, 1)) if total > 0 else 0.0
        bullets.append(
            f"The search has evaluated {total} programs with {survivors} Stage 1 survivors ({survival_rate * 100:.1f}% survival), indicating {'productive' if survival_rate >= 0.03 else 'early-stage'} grammar quality."
        )

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
                parts.append(
                    "rewarding " + ", ".join(
                        f"{name.replace('_', ' ')} ({delta:+.2f})" for name, delta in increased
                    )
                )
            if decreased:
                parts.append(
                    "downweighting " + ", ".join(
                        f"{name.replace('_', ' ')} ({delta:+.2f})" for name, delta in decreased
                    )
                )
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

        return bullets

    def generate_report_narrative(self, report_data: Dict) -> str:
        """Generate an executive narrative for the research report.

        Uses LLM if available, falls back to template-based summary.
        """
        llm = self._get_llm()
        if llm:
            try:
                from .llm.prompts import SYSTEM_PROMPT, REPORT_PROMPT
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

    def add_insight(self, insight: str):
        """Record an insight from experiment analysis."""
        self.state.insights.append(insight)
        if len(self.state.insights) > 50:
            self.state.insights = self.state.insights[-50:]
