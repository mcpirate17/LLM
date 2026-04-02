"""Results knowledge mixin: knowledge extraction, campaigns, preregistration."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ..native.telemetry import reset_native_runner_telemetry
from ..notebook import LabNotebook
from ..preregistration import (
    HypothesisPreregistration,
    PreregistrationError,
    validate_preregistration,
)
from ..llm.context_hypothesis import (
    build_knowledge_extraction_context,
    build_campaign_formulation_context,
)
from ._types import RunConfig

logger = logging.getLogger(__name__)


class _ResultsKnowledgeMixin:
    """Knowledge extraction, campaign management, preregistration, stale-candidate checks."""

    __slots__ = ()

    def _maybe_extract_knowledge(
        self, config: RunConfig, nb: LabNotebook, n_experiments: int
    ) -> None:
        """Extract knowledge every N experiments."""
        if not config.enable_campaigns:
            return
        if (
            n_experiments <= 0
            or n_experiments % config.knowledge_extraction_interval != 0
        ):
            return

        try:
            allowed_categories = {
                "principle",
                "anti_pattern",
                "sweet_spot",
                "correlation",
                "tool_insight",
            }

            def _normalize_category(raw: str) -> str:
                value = (
                    str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
                )
                aliases = {
                    "anti_pattern": "anti_pattern",
                    "anti_patterns": "anti_pattern",
                    "antipattern": "anti_pattern",
                    "anti-pattern": "anti_pattern",
                    "principles": "principle",
                    "sweetspot": "sweet_spot",
                    "sweet_spot": "sweet_spot",
                    "tool": "tool_insight",
                    "toolinsight": "tool_insight",
                    "tool_insights": "tool_insight",
                }
                value = aliases.get(value, value)
                return value if value in allowed_categories else "principle"

            def _canonical_text(raw: str) -> str:
                text = " ".join(str(raw or "").split()).strip().lower()
                text = re.sub(r"\b\d+(?:\.\d+)?%?\b", "#", text)
                text = re.sub(r"[^a-z0-9#\s]+", " ", text)
                return re.sub(r"\s+", " ", text).strip()

            stopwords = {
                "the",
                "and",
                "for",
                "that",
                "with",
                "this",
                "from",
                "into",
                "when",
                "then",
                "than",
                "were",
                "been",
                "have",
                "has",
                "had",
                "are",
                "was",
                "show",
                "shows",
                "showed",
                "over",
                "under",
                "across",
                "between",
                "using",
                "use",
                "used",
                "high",
                "low",
                "very",
                "more",
                "less",
                "near",
                "around",
                "recent",
                "experiments",
                "experiment",
                "result",
                "results",
                "indicate",
                "indicates",
                "suggest",
                "suggests",
                "mode",
                "patterns",
                "pattern",
                "architecture",
                "architectures",
            }

            def _tokenize_semantic(raw: str) -> Set[str]:
                canonical = _canonical_text(raw)
                return {
                    tok
                    for tok in canonical.split()
                    if len(tok) > 3 and tok not in stopwords
                }

            def _is_semantic_duplicate(
                tokens: Set[str], existing_tokens: Set[str]
            ) -> bool:
                if not tokens or not existing_tokens:
                    return False
                inter = len(tokens & existing_tokens)
                if inter < 5:
                    return False
                union = len(tokens | existing_tokens)
                return bool(union) and (inter / union) >= 0.18

            def _is_low_value_entry(title: str, content: str) -> bool:
                title_clean = " ".join(str(title or "").split()).strip()
                content_clean = " ".join(str(content or "").split()).strip()
                title_l = title_clean.lower()
                content_l = content_clean.lower()

                if len(title_clean) < 12 or len(content_clean) < 40:
                    return True
                if "..." in title_clean or "..." in content_clean:
                    return True
                if "1-2 sentences" in content_l or "i will now synthesize" in content_l:
                    return True
                if title_l.startswith("recent experiments show ") or title_l.startswith(
                    "all recent experiments show "
                ):
                    return True
                if title_l.startswith("recent synthesis") and "failure" in title_l:
                    return True
                if "[principle/" in title_l or "hybrid? no" in title_l:
                    return True
                if "$" in content_clean or "\\approx" in content_l:
                    return True

                mechanism_tokens = (
                    "depth",
                    "residual",
                    "inverse",
                    "log ",
                    "frequency",
                    "math_space",
                    "parameter",
                    "parallel",
                    "routing",
                    "s1",
                    "loss",
                    "novelty",
                    "baseline",
                )
                action_tokens = (
                    "improve",
                    "improves",
                    "degrade",
                    "degrades",
                    "fail",
                    "fails",
                    "underperform",
                    "correlate",
                    "correlates",
                    "correlation",
                    "predict",
                    "predicts",
                    "optimal",
                    "requires",
                    "avoid",
                    "boost",
                    "increase",
                    "reduce",
                    "enhance",
                    "enhances",
                    "outperform",
                    "outperforms",
                    "suggests",
                    "indicates",
                )
                has_mechanism = any(
                    tok in content_l or tok in title_l for tok in mechanism_tokens
                )
                has_action = any(tok in content_l for tok in action_tokens)
                has_numeric = bool(re.search(r"\d", content_clean))
                return not (has_mechanism and (has_action or has_numeric))

            recent = nb.get_recent_experiments(config.knowledge_extraction_interval)
            resolved = []
            if self._active_campaign_id:
                all_hyps = nb.get_campaign_hypotheses(self._active_campaign_id)
                resolved = [
                    h for h in all_hyps if h.get("status") in ("confirmed", "refuted")
                ]

            context = build_knowledge_extraction_context(recent, resolved)
            entries = self.aria.extract_knowledge(recent, resolved, context=context)

            existing_entries = nb.get_knowledge()
            existing_by_title: Dict[str, str] = {}
            existing_by_content: Dict[str, str] = {}
            existing_by_semantic: Dict[str, List[Tuple[str, Set[str]]]] = {}
            for row in existing_entries:
                eid = str(row.get("entry_id") or "")
                if not eid:
                    continue
                existing_by_title[_canonical_text(row.get("title") or "")] = eid
                existing_by_content[_canonical_text(row.get("content") or "")] = eid
                category = _normalize_category(str(row.get("category") or "principle"))
                tokens = _tokenize_semantic(
                    f"{row.get('title') or ''} {row.get('content') or ''}"
                )
                if tokens:
                    existing_by_semantic.setdefault(category, []).append((eid, tokens))

            accepted = 0
            skipped_low_value = 0
            deduped = 0

            for entry in entries:
                raw_title = str(entry.get("title") or "").strip()
                raw_content = str(entry.get("content") or "").strip()
                if _is_low_value_entry(raw_title, raw_content):
                    skipped_low_value += 1
                    continue

                category = _normalize_category(entry.get("category", "principle"))
                confidence = float(entry.get("confidence", 0.5) or 0.5)
                confidence = max(0.45, min(0.95, confidence))
                title = " ".join(raw_title.split())
                content = " ".join(raw_content.split())

                title_key = _canonical_text(title)
                content_key = _canonical_text(content)

                existing_entry_id = existing_by_title.get(
                    title_key
                ) or existing_by_content.get(content_key)
                if not existing_entry_id:
                    semantic_tokens = _tokenize_semantic(f"{title} {content}")
                    for eid, seen_tokens in existing_by_semantic.get(category, []):
                        if _is_semantic_duplicate(semantic_tokens, seen_tokens):
                            existing_entry_id = eid
                            break
                if existing_entry_id:
                    nb.validate_knowledge(existing_entry_id)
                    deduped += 1
                    continue

                evidence = [
                    str(e.get("experiment_id", "")).strip()
                    for e in recent[:5]
                    if str(e.get("experiment_id", "")).strip()
                ]
                new_entry_id = nb.add_knowledge(
                    category=category,
                    title=title,
                    content=content,
                    evidence=evidence,
                    confidence=confidence,
                )
                existing_by_title[title_key] = new_entry_id
                existing_by_content[content_key] = new_entry_id
                semantic_tokens = _tokenize_semantic(f"{title} {content}")
                if semantic_tokens:
                    existing_by_semantic.setdefault(category, []).append(
                        (new_entry_id, semantic_tokens)
                    )
                accepted += 1

            if entries:
                self._emit_event(
                    "knowledge_extracted",
                    {
                        "n_entries": accepted,
                        "categories": list(set(e.get("category", "") for e in entries)),
                        "n_deduped": deduped,
                        "n_skipped_low_value": skipped_low_value,
                    },
                )
                logger.info(
                    "Knowledge extracted: accepted=%d deduped=%d skipped_low_value=%d raw=%d",
                    accepted,
                    deduped,
                    skipped_low_value,
                    len(entries),
                )
        except Exception as e:
            logger.debug(f"Knowledge extraction failed: {e}")

        # ── Refresh intelligence layer on same cadence as knowledge extraction ──
        # Bayesian tracker applies temporal decay; interaction model retrains.
        # Cheap (~2s total), runs every knowledge_extraction_interval experiments.
        try:
            from ..intelligence.temporal_bayesian import TemporalBayesianTracker

            db_path = (
                str(nb.db_path)
                if hasattr(nb, "db_path")
                else "research/lab_notebook.db"
            )
            tracker = TemporalBayesianTracker.from_db(
                db_path=db_path,
                apply_decay=True,
                detect_fixes=True,
            )
            n_ops = len(tracker.op_posteriors)
            logger.debug(
                "Intelligence refresh: Bayesian tracker updated (%d ops)", n_ops
            )
        except Exception as exc:
            logger.debug("Suppressed error: %s", exc)

    def _ensure_campaign(self, config: RunConfig, nb: LabNotebook) -> Optional[str]:
        """Ensure an active campaign exists. Create one if needed."""
        if not config.enable_campaigns:
            return None

        # Check for existing active campaign
        active = nb.get_active_campaigns()
        if active:
            self._active_campaign_id = active[0]["campaign_id"]
            return self._active_campaign_id

        # Create new campaign via Aria
        recent = nb.get_recent_experiments(10)
        knowledge = nb.get_knowledge()
        all_campaigns = nb.conn.execute(
            "SELECT * FROM campaigns ORDER BY timestamp DESC LIMIT 5"
        ).fetchall()
        previous = [dict(r) for r in all_campaigns]

        context = build_campaign_formulation_context(
            recent_experiments=recent,
            knowledge=knowledge,
            previous_campaigns=previous,
        )
        camp_data = self.aria.formulate_campaign(context=context)
        post_hoc_note = (
            "\n\n[POST-HOC] Success criteria were formulated after reviewing "
            "recent experiment outcomes; treat claims as exploratory until "
            "prospective criteria are pre-registered."
        )
        campaign_id = nb.create_campaign(
            title=camp_data["title"],
            objective=camp_data["objective"],
            success_criteria=f"{camp_data['success_criteria']}{post_hoc_note}",
        )
        self._active_campaign_id = campaign_id
        self._emit_event(
            "campaign_created",
            {
                "campaign_id": campaign_id,
                "title": camp_data["title"],
                "objective": camp_data["objective"],
            },
        )
        logger.info(f"Campaign created: {camp_data['title']} ({campaign_id})")
        return campaign_id

    @staticmethod
    def _pipeline_driven_campaign(tiers: dict, reason: str) -> dict:
        """Deterministic campaign formulation based on pipeline state."""
        screening = tiers.get("screening", 0)
        investigation = tiers.get("investigation", 0)
        validation = tiers.get("validation", 0)
        breakthrough = tiers.get("breakthrough", 0)

        if breakthrough > 0:
            return {
                "title": "Scale-Up & Generalization",
                "objective": (
                    f"Validate {breakthrough} breakthrough architecture(s) at "
                    f"larger scale (512+ dim, longer sequences) and on diverse "
                    f"data distributions to confirm generalization."
                ),
                "success_criteria": (
                    "Breakthrough architecture maintains loss_ratio < 0.5 at "
                    "model_dim=512; OOD generalization >= 0.67; "
                    "Reproducible across 5+ random seeds with std <= 0.03"
                ),
            }
        elif validation > 0:
            return {
                "title": "Validation & Robustness",
                "objective": (
                    f"Complete multi-seed validation for {validation} candidate(s) "
                    f"and identify which architectures are robust enough for "
                    f"breakthrough consideration."
                ),
                "success_criteria": (
                    "At least 1 candidate passes validation with multi-seed "
                    "std <= 0.03 and baseline_ratio < 0.90; "
                    "Go/no-go decision recorded for each candidate"
                ),
            }
        elif investigation > 0 or screening > 0:
            total_candidates = investigation + screening
            return {
                "title": "Deep Investigation",
                "objective": (
                    f"Investigate {total_candidates} screening/investigation "
                    f"candidate(s) with extended training to identify which "
                    f"architectures warrant full validation."
                ),
                "success_criteria": (
                    "At least 1 candidate passes investigation with "
                    "loss_ratio < 0.6 and robustness > 0.7; "
                    "Clear go/no-go decision for each investigated candidate"
                ),
            }
        elif reason == "stale":
            return {
                "title": "Novelty Exploration",
                "objective": (
                    "Escape the current search region using evolution and "
                    "novelty search to discover fundamentally different "
                    "architecture patterns."
                ),
                "success_criteria": (
                    "Find 3+ architectures with loss_ratio < 0.5 and "
                    "novelty_score > 0.5; Stage-1 survival rate > 5%"
                ),
            }
        else:
            return {
                "title": "Architecture Discovery",
                "objective": (
                    "Discover novel computation patterns by exploring diverse "
                    "op combinations, math spaces, and weight storage techniques."
                ),
                "success_criteria": (
                    "Find 3+ architectures with loss_ratio < 0.5; "
                    "Stage-1 survival rate > 3%; "
                    "At least 1 go/no-go decision recorded"
                ),
            }

    def _ensure_preregistration(
        self,
        nb: LabNotebook,
        experiment_type: str,
        config: Dict[str, Any],
        hypothesis: Optional[str],
        preregistration: Optional[Dict[str, Any]] = None,
        exploratory: bool = False,
        created_by: str = "runner",
    ) -> str:
        require_prereg = bool(config.get("require_preregistration", True))
        auto_preregister = bool(config.get("auto_preregister", True))
        payload = preregistration
        if payload is None and auto_preregister:
            payload = self._build_default_preregistration(
                experiment_type=experiment_type,
                config=config,
                hypothesis=hypothesis,
                exploratory=exploratory,
            )
        if require_prereg and payload is None:
            raise PreregistrationError(
                "Experiment blocked: preregistration required but missing."
            )
        if payload is None:
            raise PreregistrationError(
                "Experiment blocked: preregistration payload unavailable."
            )
        validate_preregistration(payload)
        return nb.create_preregistration(
            experiment_type=experiment_type,
            preregistration=payload,
            created_by=created_by,
        )

    def _build_default_preregistration(
        self,
        experiment_type: str,
        config: Dict[str, Any],
        hypothesis: Optional[str],
        exploratory: bool = False,
    ) -> Dict[str, Any]:
        statement = str(
            hypothesis
            or f"{experiment_type} batch will improve prioritized objectives."
        )
        primary_metrics = ["loss_ratio", "stage1_passed"]
        if experiment_type in {"novelty", "evolution"}:
            primary_metrics = ["novelty_score", "stage1_passed"]
        if experiment_type in {"validation", "scale_up"}:
            primary_metrics = [
                "baseline_loss_ratio",
                "loss_ratio",
                "novelty_confidence",
            ]

        prereg = HypothesisPreregistration(
            hypothesis={
                "statement": statement,
                "variables": {
                    "independent": [
                        "architecture_family",
                        "op_composition",
                        "training_recipe",
                    ],
                    "dependent": primary_metrics
                    + ["throughput_tok_s", "stability_score"],
                    "controls": ["model_dim", "n_layers", "stage1_steps", "batch_size"],
                },
                "expected_direction": {
                    "loss_ratio": "decrease",
                    "novelty_score": "increase",
                    "throughput_tok_s": "increase",
                    "stability_score": "increase",
                },
                "success_criteria": {
                    "stage1_passed_min": 1,
                    "best_loss_ratio_max": 0.95,
                    "novelty_confidence_min": 0.5,
                },
            },
            analysis_plan={
                "primary_metrics": primary_metrics,
                "secondary_metrics": [
                    "compile_time_ms",
                    "grad_norm_std",
                    "throughput_tok_s",
                    "flops_per_token",
                    "novelty_confidence",
                ],
                "thresholds": {
                    "loss_ratio": {"operator": "<", "value": 1.0},
                    "novelty_confidence": {"operator": ">=", "value": 0.5},
                    "stability_score": {"operator": ">=", "value": 0.5},
                },
                "baseline_comparison": {
                    "method": "relative_loss_ratio",
                    "source": "TransformerBaseline.compare",
                    "delta_operator": "<",
                    "delta_value": 1.0,
                },
            },
            falsification_conditions=[
                "No candidate passes Stage1.",
                "Best loss_ratio does not beat baseline threshold.",
                "Novelty only appears with heuristic fallback and no justification.",
            ],
            confounders_checklist=[
                {"name": "unstable_seed_behavior", "checked": False},
                {"name": "fallback_novelty_mode", "checked": False},
                {"name": "noisy_throughput", "checked": False},
                {"name": "compile_instability", "checked": False},
            ],
            exploratory=exploratory,
        ).to_dict()
        prereg["analysis_plan"]["config_snapshot"] = {
            "n_programs": config.get("n_programs"),
            "stage1_steps": config.get("stage1_steps"),
            "model_dim": config.get("model_dim"),
            "n_layers": config.get("n_layers"),
        }
        return prereg

    def _start_preregistered_experiment(
        self,
        nb: LabNotebook,
        experiment_type: str,
        config: Dict[str, Any],
        hypothesis: Optional[str] = None,
        research_question: Optional[str] = None,
        hypothesis_metadata: Optional[Dict[str, Any]] = None,
        preregistration: Optional[Dict[str, Any]] = None,
        exploratory: bool = False,
        created_by: str = "runner",
    ) -> str:
        prereg_id = self._ensure_preregistration(
            nb=nb,
            experiment_type=experiment_type,
            config=config,
            hypothesis=hypothesis,
            preregistration=preregistration,
            exploratory=exploratory,
            created_by=created_by,
        )
        meta = dict(hypothesis_metadata or {})
        meta["preregistration_id"] = prereg_id

        # Z17: Reset global native-runner counters between experiments
        reset_native_runner_telemetry()

        return nb.start_experiment(
            experiment_type=experiment_type,
            config=config,
            hypothesis=hypothesis,
            research_question=research_question,
            hypothesis_metadata=meta,
            preregistration_id=prereg_id,
            require_preregistration=bool(config.get("require_preregistration", True)),
        )

    def _check_stale_screening_candidates(self, nb: LabNotebook, config: RunConfig):
        """Force investigation if top screening models have high composite scores but are uninvestigated.

        Uses composite_score (not loss_ratio) as the gate — a model that
        excels on efficiency, novelty, or stability deserves investigation
        even if its loss is mediocre.  The threshold is the 25th-percentile
        composite_score of the current investigation tier, so only candidates
        that could plausibly compete on the leaderboard are promoted.
        """
        try:
            # Dynamic threshold: 25th percentile of investigation tier scores
            inv_scores = nb.conn.execute(
                "SELECT l.composite_score FROM leaderboard l"
                " WHERE l.tier IN ('investigation', 'validation')"
                " AND l.composite_score IS NOT NULL"
                " ORDER BY l.composite_score ASC"
            ).fetchall()
            if not inv_scores:
                return None
            score_threshold = inv_scores[len(inv_scores) // 4][0]

            stale = nb.conn.execute(
                """SELECT l.result_id FROM leaderboard l
                   WHERE l.tier = 'screening' AND l.screening_passed = 1
                     AND COALESCE(l.is_reference, 0) = 0
                     AND l.composite_score IS NOT NULL
                     AND l.composite_score >= ?
                     AND l.investigation_loss_ratio IS NULL
                   ORDER BY l.composite_score DESC LIMIT ?""",
                (score_threshold, config.auto_investigate_top_n),
            ).fetchall()
            if stale:
                result_ids = [r["result_id"] for r in stale]
                logger.info(
                    "Stale screening check: %d models with composite_score >= %.1f uninvestigated",
                    len(result_ids),
                    score_threshold,
                )
                return result_ids
        except Exception as e:
            logger.warning("Stale screening check failed: %s", e)
        return None

    def _resolve_novelty_promotion_validity(
        self,
        config: RunConfig,
        valid_for_promotion: bool,
        reason: str,
    ) -> Tuple[bool, str, bool]:
        """Apply explicit override policy for heuristic novelty promotions."""
        valid = bool(valid_for_promotion)
        resolved_reason = str(reason or "unknown")
        requires_justification = not valid
        if valid:
            return True, resolved_reason, False
        if (
            config.allow_heuristic_novelty_promotion
            and str(config.heuristic_novelty_justification or "").strip()
        ):
            return True, f"override:{resolved_reason}", True
        return False, resolved_reason, requires_justification
