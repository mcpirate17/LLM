from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class _PersonaHypothesisMixin:
    _HYPOTHESIS_STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
        "shall", "can", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
        "into", "through", "during", "before", "after", "that", "this", "these", "those",
        "it", "its", "and", "or", "but", "not", "no", "if", "then", "than", "so", "very",
        "just", "about", "also", "more", "most", "some", "any", "each", "all", "both",
        "such", "only", "own", "same", "other", "new", "old", "high", "low", "good", "bad",
        "best", "worst", "we", "our", "they", "their", "use", "using", "used", "based",
        "whether", "when", "which", "what", "how", "where",
    }

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

    def validate_hypothesis(self, hypothesis: str, results: Dict, context: str = "") -> Dict:
        """Validate whether a hypothesis was confirmed or refuted.

        Returns {validated: bool, explanation: str}.
        Uses analyst LLM with VALIDATION_PROMPT, falls back to S1>0 heuristic.
        """
        llm = self._get_analyst_llm()
        if llm and context:
            try:
                from .llm.prompts import SYSTEM_PROMPT, VALIDATION_PROMPT

                prompt = VALIDATION_PROMPT.format(hypothesis=hypothesis, context=context)
                resp = llm.generate(prompt, system=SYSTEM_PROMPT, max_tokens=512)
                self._track_cost(resp)
                if resp.text.strip():
                    text = resp.text.strip()
                    confirmed = any(
                        w in text.lower() for w in ["confirmed", "supported", "validated"]
                    )
                    return {"validated": confirmed, "explanation": text}
            except Exception as e:
                logger.warning(f"LLM validation failed, falling back: {e}")

        s1_passed = results.get("stage1_passed", 0)
        novel = results.get("novel_count", 0)
        confirmed = s1_passed > 0
        if confirmed:
            explanation = (
                f"Hypothesis partially confirmed: {s1_passed} programs "
                f"passed Stage 1, {novel} were novel."
            )
        else:
            explanation = (
                "Hypothesis refuted: no programs passed Stage 1. "
                "The proposed approach did not produce learnable architectures."
            )
        return {"validated": confirmed, "explanation": explanation}

    @staticmethod
    def _sanitize_hypothesis(text: Optional[str]) -> Optional[str]:
        """Strip code blocks and inline code from hypothesis text."""
        if not text:
            return text
        import re as _re

        cleaned = _re.sub(r"```[\s\S]*?```", "", text)
        cleaned = _re.sub(r"`[^`]*`", "", cleaned)
        cleaned = _re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) > 300:
            boundary = cleaned[:297].rfind(" ")
            if boundary > 150:
                cleaned = cleaned[:boundary].rstrip(".,;:") + "..."
            else:
                cleaned = cleaned[:297] + "..."
        return cleaned or None

    @staticmethod
    def _strip_code_blocks(text: str) -> str:
        """Remove fenced code blocks and inline code from LLM output."""
        if not text:
            return text
        cleaned = re.sub(r"```(?!json\b)[a-z]*\s*\n[\s\S]*?```", "", text)
        cleaned = re.sub(r"`[^`]*`", "", cleaned)
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

    def critique_hypothesis(self, hypothesis: str, context: str = "") -> Dict:
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
            return self._normalize_preflight_critique(
                "",
                {
                    "verdict": "revise",
                    "concerns": ["No hypothesis provided."],
                    "suggestions": [
                        "Formulate a specific, testable prediction about which architectural patterns will succeed."
                    ],
                    "confidence": 0.0,
                },
            )

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
            if not any(
                token in h_lower
                for token in ["source_selection_rule", "result_ids(", "source_result_id"]
            ):
                _add("source_selection_rule")
            if not any(
                token in h_lower
                for token in [
                    "mutation_mechanism",
                    "mutation_rate",
                    "operator",
                    "neighborhood",
                    "max_edits",
                    "radius",
                ]
            ):
                _add("mutation_mechanism")
            if "intent=" in h_lower and not any(
                token in h_lower for token in ["weights=", "score=", "intent_weights"]
            ):
                _add("intent_weights")
            if not any(
                token in h_lower
                for token in [
                    "success_criteria",
                    "success_metric",
                    "primary_metric",
                    "threshold",
                    "delta_",
                    "baseline",
                    ">=",
                    "<=",
                ]
            ):
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
        flags = self._preflight_flags(hypothesis, concerns)

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
                "status": _status(
                    flags["has_testability"] and flags["has_success_criteria"],
                    flags["has_metric"] and flags["has_success_criteria"],
                ),
            },
            {
                "key": "measurable_metric",
                "label": "Measurable Metric",
                "status": _status(flags["has_metric"] and flags["has_success_criteria"], flags["has_metric"]),
            },
            {
                "key": "confound_risk",
                "label": "Confound Risk",
                "status": _status(
                    (not flags["confound_signal"])
                    and flags["has_metric"]
                    and flags["has_source_rule"]
                    and flags["has_mutation_mechanism"]
                    and flags["has_intent_spec"],
                    flags["has_metric"],
                ),
            },
            {
                "key": "fallback_plan",
                "label": "Fallback Plan",
                "status": _status(flags["has_fallback"], not flags["has_fallback"]),
            },
        ]

    def _preflight_flags(self, hypothesis: str, concerns: List[str]) -> Dict[str, bool]:
        """Compute preflight check flags for hypothesis quality gates."""
        h_lower = (hypothesis or "").lower()
        concern_text = " ".join(c.lower() for c in concerns)

        metric_words = ["loss", "novelty", "rate", "ratio", "pass", "survive", "accuracy", "faster", "slower", "better", "worse", "increase", "decrease", "improve", "%"]
        has_metric = any(w in h_lower for w in metric_words)

        testability_words = ["if", "then", "because", "compared", "versus", "vs", "should", "will", "than", "predict"]
        has_testability = has_metric and any(w in h_lower for w in testability_words)

        fallback_words = ["fallback", "fallback_plan", "backup", "otherwise", "if not", "if this fails", "ablation", "control", "next step", "alternative", "revert"]
        has_fallback = any(w in h_lower for w in fallback_words)
        has_success_criteria = any(
            token in h_lower
            for token in ["success_criteria", "success_metric", "primary_metric", "threshold", ">=", "<=", "delta_", "baseline", "vs_recent"]
        )
        has_mutation_mechanism = any(
            token in h_lower
            for token in ["mutation_mechanism", "operator", "mutation_rate", "neighborhood", "max_edits", "radius"]
        )
        has_source_rule = any(token in h_lower for token in ["source_selection_rule", "result_ids(", "stage1_survivor_sources"])
        has_intent_spec = (
            ("intent=" in h_lower and ("weights=" in h_lower or "score=" in h_lower))
            or ("intent_weights" in h_lower)
        )

        confound_signal = any(token in concern_text for token in ["vague", "specific", "architectural", "measurable", "confound", "undefined", "no mechanism"])
        has_confounders = any(token in h_lower for token in ["confounders_checklist", "confounders", "confound"])
        if has_confounders:
            confound_signal = False

        return {
            "has_metric": has_metric,
            "has_testability": has_testability,
            "has_fallback": has_fallback,
            "has_success_criteria": has_success_criteria,
            "has_mutation_mechanism": has_mutation_mechanism,
            "has_source_rule": has_source_rule,
            "has_intent_spec": has_intent_spec,
            "confound_signal": confound_signal,
        }

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
        tokens = set(_re.findall(r"[a-z][a-z0-9_]{2,}", text))
        return tokens - _PersonaHypothesisMixin._HYPOTHESIS_STOPWORDS

    @staticmethod
    def _jaccard_similarity(a: set, b: set) -> float:
        """Jaccard similarity between two token sets."""
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0

    def _check_refuted_overlap(self, hypothesis: str, threshold: float = 0.45) -> List[Dict]:
        """Check if a hypothesis is too similar to any refuted hypothesis.

        Returns a list of matches with similarity scores above threshold.
        Threshold of 0.45 catches near-duplicates while allowing legitimate
        variations on a theme.
        """
        if not self._refuted_hypotheses or not hypothesis:
            return []

        hyp_tokens = self._tokenize_hypothesis(hypothesis)
        if len(hyp_tokens) < 3:
            return []

        matches = []
        for refuted in self._refuted_hypotheses:
            content = refuted.get("content") or refuted.get("hypothesis") or ""
            if not content:
                continue
            ref_tokens = self._tokenize_hypothesis(content)
            sim = self._jaccard_similarity(hyp_tokens, ref_tokens)
            if sim >= threshold:
                matches.append(
                    {
                        "refuted_text": content[:120],
                        "similarity": round(sim, 3),
                        "confidence": refuted.get("confidence", 0),
                        "shared_tokens": sorted(hyp_tokens & ref_tokens)[:10],
                    }
                )

        return sorted(matches, key=lambda m: -m["similarity"])

    def _extract_breakthrough_metrics_from_context(self, context: str) -> Dict[str, float]:
        """Best-effort parse of validation metrics from free-form context text."""
        parsed: Dict[str, float] = {}
        if not context:
            return parsed

        patterns = {
            "seeds_passed": [
                r"seeds?_passed\s*[:=]\s*(\d+)",
                r"seeds\s*[:=]\s*(\d+)\s*/\s*\d+",
            ],
            "total_seeds": [
                r"total_seeds\s*[:=]\s*(\d+)",
                r"seeds\s*[:=]\s*\d+\s*/\s*(\d+)",
            ],
            "val_baseline_ratio": [
                r"val_baseline_ratio\s*[:=]\s*([0-9]*\.?[0-9]+)",
                r"baseline[^\n]*ratio\s*[:=]\s*([0-9]*\.?[0-9]+)",
            ],
            "multi_seed_std": [
                r"multi_seed_std\s*[:=]\s*([0-9]*\.?[0-9]+)",
                r"multi[- ]seed[^\n]*std\s*[:=]\s*([0-9]*\.?[0-9]+)",
            ],
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

    def announce_breakthrough(self, context: str = "", metrics: Optional[Dict] = None) -> str:
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
        result = {
            "prediction": "",
            "reasoning": "",
            "test_method": "",
            "success_metric": "",
            "confidence": 0.5,
        }

        all_headers = (
            "PREDICTION",
            "REASONING",
            "TEST.METHOD",
            "SUCCESS.CRITERIA",
            "SUCCESS.METRIC",
            "PRIMARY.METRIC",
            "CONFOUNDERS",
            "FALLBACK.PLAN",
            "CONFIDENCE",
        )
        header_pattern = "|".join(all_headers)

        for field in (
            "prediction",
            "reasoning",
            "test_method",
            "success_criteria",
            "success_metric",
            "primary_metric",
            "confounders",
            "fallback_plan",
        ):
            pattern = rf"{field.upper().replace('_', '.')}:\s*(.+?)(?=(?:{header_pattern}):|$)"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                result[field] = match.group(1).strip()

        if result.get("success_criteria") and not result.get("success_metric"):
            result["success_metric"] = result["success_criteria"]

        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
        if conf_match:
            try:
                result["confidence"] = float(conf_match.group(1))
            except ValueError:
                pass

        if not result["prediction"]:
            result["prediction"] = text[:200]

        return result

    def validate_structured_hypothesis(self, hypothesis: Dict, results: Dict, context: str = "") -> Dict:
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
        result = {
            "status": "inconclusive",
            "evidence": "",
            "explanation": "",
            "follow_up": None,
            "confidence_after": 0.5,
        }

        status_match = re.search(r"STATUS:\s*(\w+)", text, re.IGNORECASE)
        if status_match:
            s = status_match.group(1).lower()
            if s in ("confirmed", "refuted", "inconclusive"):
                result["status"] = s

        evidence_match = re.search(
            r"EVIDENCE:\s*(.+?)(?=EXPLANATION:|FOLLOW.UP:|CONFIDENCE:|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if evidence_match:
            result["evidence"] = evidence_match.group(1).strip()

        expl_match = re.search(
            r"EXPLANATION:\s*(.+?)(?=FOLLOW.UP:|CONFIDENCE:|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if expl_match:
            result["explanation"] = expl_match.group(1).strip()

        follow_match = re.search(
            r"FOLLOW.UP:\s*(.+?)(?=CONFIDENCE:|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if follow_match:
            fu = follow_match.group(1).strip()
            result["follow_up"] = fu if fu.lower() != "none" else None

        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
        if conf_match:
            try:
                result["confidence_after"] = float(conf_match.group(1))
            except ValueError:
                pass

        return result
