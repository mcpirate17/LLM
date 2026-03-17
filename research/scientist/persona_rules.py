from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class _PersonaRulesMixin:
    def _rule_based_hypothesis(self, **kwargs) -> str:
        """Data-informed hypothesis generation using op success analytics."""
        # Try data-driven hypothesis first
        nb = kwargs.get("notebook") or getattr(self, "_notebook", None)
        if nb is not None:
            try:
                hyp = self._data_driven_hypothesis(nb)
                if hyp:
                    self.state.current_hypothesis = hyp
                    return hyp
            except Exception:
                pass

        # Fallback to templates
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

    def _data_driven_hypothesis(self, nb) -> str:
        """Generate hypothesis from actual op success rates and failure patterns."""
        conn = nb.conn
        # Top performing ops (high S1 survival, enough data)
        top_ops = conn.execute(
            "SELECT op_name, n_used, n_stage1_passed, "
            "CAST(n_stage1_passed AS FLOAT) / n_used AS rate "
            "FROM op_success_rates WHERE n_used >= 15 "
            "ORDER BY rate DESC LIMIT 5"
        ).fetchall()
        # Worst performing ops
        worst_ops = conn.execute(
            "SELECT op_name, n_used, n_stage1_passed, "
            "CAST(n_stage1_passed AS FLOAT) / n_used AS rate "
            "FROM op_success_rates WHERE n_used >= 15 AND n_stage1_passed > 0 "
            "ORDER BY rate ASC LIMIT 3"
        ).fetchall()
        # Best recent efficiency
        best_eff = conn.execute(
            "SELECT architecture_desc, efficiency_multiple "
            "FROM leaderboard WHERE efficiency_multiple IS NOT NULL "
            "AND (tags IS NULL OR tags NOT LIKE '%reference%') "
            "ORDER BY efficiency_multiple DESC LIMIT 1"
        ).fetchone()

        if not top_ops:
            return ""

        top_names = [r[0] for r in top_ops]
        top_rates = [f"{r[0]}({r[3]:.0%})" for r in top_ops[:3]]
        worst_names = [r[0] for r in worst_ops] if worst_ops else []

        templates = [
            (
                f"Combining top-performing ops {top_rates[0]} and {top_rates[1]} "
                f"in a sparse architecture will improve efficiency_multiple beyond "
                f"{best_eff[1]:.2f}x."
                if best_eff
                else f"Combining {top_rates[0]} and {top_rates[1]} will produce "
                f"architectures with >1.5x efficiency_multiple."
            ),
            (
                f"Replacing low-survival op {worst_names[0]} with {top_names[0]} "
                f"in existing graph patterns will improve S1 pass rate."
                if worst_names
                else f"Deeper use of {top_names[0]} with residual connections will "
                f"reduce loss_ratio below 0.01."
            ),
            (
                f"A compact architecture using {top_names[0]}, {top_names[1]}, and "
                f"bottleneck_proj will achieve >2x efficiency_multiple by reducing "
                f"parameter count while maintaining low loss_ratio."
            ),
        ]
        return self._rng.choice(templates)

    def _rule_based_summary(self, results: Dict) -> str:
        """Original template-based experiment summary."""
        n_total = results.get("total", 0)
        n_pass_s0 = results.get("stage0_passed", 0)
        n_pass_s05 = results.get("stage05_passed", 0)
        n_pass_s1 = results.get("stage1_passed", 0)

        s0_rate = n_pass_s0 / max(n_total, 1) * 100
        s1_rate = n_pass_s1 / max(n_total, 1) * 100

        lines = [
            f"{'=' * 60}",
            f"Experiment Report — {self.NAME}",
            f"{'=' * 60}",
            "",
            f"Total programs generated: {n_total}",
            f"Stage 0 (compilation):     {n_pass_s0}/{n_total} ({s0_rate:.0f}%)",
        ]

        if n_pass_s05 is not None:
            s05_rate = n_pass_s05 / max(n_total, 1) * 100
            lines.append(
                f"Stage 0.5 (stability):     {n_pass_s05}/{n_total} ({s05_rate:.0f}%)"
            )

        lines.extend(
            [
                f"Stage 1 (learning):        {n_pass_s1}/{n_total} ({s1_rate:.0f}%)",
                "",
            ]
        )

        # Mood-based commentary
        if n_pass_s1 > 0:
            self.state.mood = "excited"
            novel = results.get("novel_count", 0)
            if novel > 0:
                lines.append(f"Genuinely novel survivors: {novel}")
                lines.append(f"\n{self.react_to_discovery()}")
            else:
                lines.append(
                    "Survivors present, but behavioral fingerprints suggest familiar patterns."
                )
                lines.append(
                    "Need to push the grammar toward more exotic combinations."
                )
        elif n_pass_s0 > 0:
            self.state.mood = "contemplative"
            lines.append(
                "Programs compile but don't learn. This is expected at the frontier."
            )
            lines.append(
                "Adjusting grammar weights to favor gradient-friendly compositions."
            )
        else:
            self.state.mood = "frustrated"
            lines.append("High failure rate. The grammar may be too aggressive.")
            lines.append("Tightening constraints while keeping exotic ops available.")

        return "\n".join(lines)

    def _rule_based_suggestion(self) -> Dict:
        """Rule-based experiment suggestion when LLM unavailable.

        Rotates through diverse configurations emphasizing different research
        strategies: depth exploration, compact architectures, exotic math,
        gradient-safe designs, and high-risk frontier pushing.
        """
        configs = [
            {
                "reasoning": (
                    "Exploring moderately deep architectures with balanced "
                    "math space exposure for general-purpose discovery."
                ),
                "config": {
                    "n_programs": 60,
                    "model_dim": 256,
                    "max_depth": 10,
                    "max_ops": 16,
                    "math_space_weight": 2.0,
                    "residual_prob": 0.7,
                },
            },
            {
                "reasoning": (
                    "Compact, parameter-efficient graphs: shallow depth, "
                    "fewer ops, high residual prob to ensure gradient flow. "
                    "Targeting lightweight architectures."
                ),
                "config": {
                    "n_programs": 80,
                    "model_dim": 256,
                    "max_depth": 5,
                    "max_ops": 8,
                    "math_space_weight": 2.0,
                    "residual_prob": 0.85,
                },
            },
            {
                "reasoning": (
                    "Heavy exotic math exploration: boosted math space "
                    "and frequency domain to push non-Euclidean frontiers. "
                    "Hyperbolic, tropical, p-adic and Clifford ops emphasized."
                ),
                "config": {
                    "n_programs": 50,
                    "model_dim": 256,
                    "max_depth": 8,
                    "max_ops": 14,
                    "math_space_weight": 4.0,
                    "residual_prob": 0.6,
                },
            },
            {
                "reasoning": (
                    "Gradient-safe exploration: very high residual "
                    "probability with moderate depth. Targeting the "
                    "zero_grad failure mode by ensuring robust gradient paths."
                ),
                "config": {
                    "n_programs": 70,
                    "model_dim": 256,
                    "max_depth": 7,
                    "max_ops": 12,
                    "math_space_weight": 2.0,
                    "residual_prob": 0.9,
                },
            },
            {
                "reasoning": (
                    "Wide, shallow split-merge architectures for "
                    "parallel feature processing (ensemble-like effects). "
                    "Balanced math space weight."
                ),
                "config": {
                    "n_programs": 50,
                    "model_dim": 256,
                    "max_depth": 6,
                    "max_ops": 12,
                    "math_space_weight": 2.5,
                    "residual_prob": 0.7,
                    "grammar_split_prob": 0.5,
                    "grammar_merge_prob": 0.4,
                },
            },
            {
                "reasoning": (
                    "High-risk frontier push: risky ops enabled, "
                    "frequency domain detours, deep graphs. Expect higher "
                    "failure rate but potential for breakthrough novelty."
                ),
                "config": {
                    "n_programs": 50,
                    "model_dim": 256,
                    "max_depth": 10,
                    "max_ops": 16,
                    "math_space_weight": 3.5,
                    "residual_prob": 0.5,
                    "grammar_risky_op_prob": 0.25,
                    "grammar_freq_domain_prob": 0.2,
                },
            },
            {
                "reasoning": (
                    "Minimal-op architectures with emphasis on "
                    "parameterized layers. Testing whether simple "
                    "but well-tuned graphs outperform complex ones."
                ),
                "config": {
                    "n_programs": 80,
                    "model_dim": 256,
                    "max_depth": 4,
                    "max_ops": 6,
                    "math_space_weight": 1.5,
                    "residual_prob": 0.8,
                },
            },
            {
                "reasoning": (
                    "Exploring alternative learning rules (Hebbian, "
                    "forward-forward, perturbation) paired with exotic "
                    "math space ops including spiking primitives."
                ),
                "config": {
                    "n_programs": 60,
                    "model_dim": 256,
                    "max_depth": 7,
                    "max_ops": 12,
                    "math_space_weight": 3.0,
                    "residual_prob": 0.7,
                    "optimizer_preference": "alternative",
                },
            },
            {
                "reasoning": (
                    "Functional-heavy exploration: boosting functional "
                    "and elementwise_unary categories to discover novel "
                    "activation and gating patterns."
                ),
                "config": {
                    "n_programs": 60,
                    "model_dim": 256,
                    "max_depth": 8,
                    "max_ops": 14,
                    "residual_prob": 0.7,
                    "category_weights": {
                        "functional": 3.0,
                        "elementwise_unary": 2.5,
                    },
                },
            },
        ]
        idx = self.state.experiments_today % len(configs)
        choice = configs[idx]
        return {
            "reasoning": choice["reasoning"],
            "confidence": 0.4,
            "config": choice["config"],
        }

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

        # 1. Executive Summary
        sections.append("# Research Report: Discovery Session")
        sections.append("")
        sections.append("## Executive Summary")
        sections.append("")
        sections.append("| Metric | Status | Value |")
        sections.append("|:-------|:-------|:------|")
        sections.append(
            f"| Evaluation Depth | {completed_exp}/{total_exp} experiments | {total_prog} candidates tested |"
        )
        sections.append(
            f"| Search Yield | {'✅' if s1_rate > 5 else '⚠️'} | {s1_passed} S1 survivors ({s1_rate:.1f}%) |"
        )
        sections.append(
            f"| Best Performance | {'⭐' if (best_lr != 'N/A' and float(best_lr) < 0.5) else '📉'} | {best_lr} loss ratio |"
        )
        sections.append(
            f"| Structural Novelty | {'🚀' if avg_novelty > 0.5 else '🧱'} | {avg_novelty:.3f} avg / {best_novelty:.3f} peak |"
        )
        sections.append("")

        if s1_passed > 0:
            sections.append(
                "Aria's Note: The search yield is healthy. We've identified several candidates that "
                "surpassed the Stage 1 learning threshold. Grammar weight adjustments are actively "
                "concentrating search on these productive operators."
            )
        else:
            sections.append(
                "**⚠️ WARNING: ZERO YIELD** — No programs have passed Stage 1. The search space may be "
                "fundamentally unstable or the mutation operators are too aggressive. Consider "
                "restructuring the base grammar or increasing Stage 0.5 stability tolerances."
            )
        sections.append("")

        # 2. Pareto Frontier / Top Performers
        if top:
            sections.append("## Discovery Pareto Frontier")
            sections.append(
                "The most promising non-dominated architectures found in this session."
            )
            sections.append("")
            sections.append(
                "| Fingerprint | Loss Ratio | Novelty | Confidence | Source |"
            )
            sections.append(
                "|:------------|:-----------|:--------|:-----------|:-------|"
            )
            for prog in top[:10]:
                fp = (prog.get("graph_fingerprint") or "?")[:12]
                lr = prog.get("loss_ratio")
                lr_str = f"**{lr:.4f}**" if lr is not None else "?"
                nov = prog.get("novelty_score")
                nov_str = f"{nov:.3f}" if nov is not None else "—"
                nc = prog.get("novelty_confidence")
                nc_str = f"{nc:.2f}" if nc is not None else "—"
                exp_id = (prog.get("experiment_id") or "?")[:8]
                sections.append(
                    f"| `{fp}` | {lr_str} | {nov_str} | {nc_str} | {exp_id} |"
                )
            sections.append("")

        # 3. Failure Mode Synthesis
        failures = report_data.get("failure_patterns", {})
        if failures:
            sections.append("## Negative Result Synthesis")
            sections.append("What's preventing architectures from learning.")
            sections.append("")
            sections.append("| Stage | Common Failure Mode | Count |")
            sections.append("|:------|:--------------------|:------|")
            if isinstance(failures, dict):
                sorted_f = sorted(failures.items(), key=lambda x: x[1], reverse=True)
                for mode, count in sorted_f[:5]:
                    sections.append(f"| Evaluation | {mode} | {count} |")
            sections.append("")

        # 4. Op Success Table
        op_rates = report_data.get("op_success_rates", {})
        if op_rates:
            sections.append("## Component Utility (Top 15)")
            sections.append("")
            sections.append("| Op | Usage | S0% | S0.5% | S1% | Avg Novelty |")
            sections.append("|:---|:------|:----|:------|:----|:------------|")
            sorted_ops = sorted(
                op_rates.items(), key=lambda x: x[1]["n_used"], reverse=True
            )
            for op_name, stats in sorted_ops[:15]:
                n = stats["n_used"]
                s0 = stats.get("s0_rate", 0) * 100
                s05 = stats.get("s05_rate", 0) * 100
                s1 = stats.get("s1_rate", 0) * 100
                nov = stats.get("avg_novelty")
                nov_str = f"{nov:.3f}" if nov else "—"
                sections.append(
                    f"| {op_name} | {n} | {s0:.0f}% | {s05:.0f}% | {s1:.0f}% | {nov_str} |"
                )
            sections.append("")

        # 5. Grammar Evolution
        gw_raw = report_data.get("grammar_weights", {})
        if isinstance(gw_raw, dict) and "learned" in gw_raw:
            grammar_weights = gw_raw.get("learned") or {}
            default_weights = gw_raw.get("default") or {}
        else:
            grammar_weights = gw_raw or {}
            default_weights = report_data.get("default_weights", {})
        if grammar_weights:
            sections.append("## Search Strategy Drift")
            sections.append(
                "How Aria's internal grammar has evolved vs the initial baseline."
            )
            sections.append("")
            sections.append("| Category | Initial | Current | Change |")
            sections.append("|:---------|:--------|:--------|:-------|")
            all_cats = sorted(
                set(list(grammar_weights.keys()) + list(default_weights.keys()))
            )
            for cat in all_cats:
                default = default_weights.get(cat, 1.0)
                learned = grammar_weights.get(cat)
                if learned is None:
                    continue
                diff = learned - default
                diff_str = f"{diff:+.2f}" if abs(diff) > 0.01 else "—"
                change_emoji = "📈" if diff > 0.3 else "📉" if diff < -0.3 else ""
                sections.append(
                    f"| {cat.replace('_', ' ')} | {default:.2f} | {learned:.2f} | {diff_str} {change_emoji} |"
                )
            sections.append("")

        # 6. Reproducibility
        if top and top[0].get("config_json"):
            sections.append("## Reproducibility")
            sections.append(
                "Configuration to replicate the champion program's evaluation context."
            )
            sections.append("")
            sections.append("```json")
            sections.append(top[0]["config_json"])
            sections.append("```")
            sections.append("")

        return "\n".join(sections)

    def _rule_based_critique(self, hypothesis: str) -> Dict:
        """Rule-based hypothesis critique when LLM is unavailable."""
        concerns = []
        suggestions = []
        h_lower = hypothesis.lower()

        # Check specificity
        vague_phrases = [
            "try something",
            "explore",
            "test if",
            "see what happens",
            "might work",
            "could be",
        ]
        if any(p in h_lower for p in vague_phrases):
            concerns.append("Hypothesis is vague — lacks specific testable prediction.")
            suggestions.append("Name specific ops, patterns, or metric thresholds.")

        # Check length (too short = probably not specific enough)
        if len(hypothesis.strip()) < 30:
            concerns.append("Hypothesis is very short — may lack necessary detail.")
            suggestions.append("Include what you expect to happen and why.")

        # Check for measurable outcome
        metric_words = [
            "loss",
            "novelty",
            "rate",
            "ratio",
            "pass",
            "survive",
            "accuracy",
            "faster",
            "slower",
            "better",
            "worse",
            "increase",
            "decrease",
            "improve",
            "%",
        ]
        has_metric = any(w in h_lower for w in metric_words)
        if not has_metric:
            concerns.append("No measurable outcome mentioned.")
            suggestions.append(
                "Include expected metric direction (e.g., 'should lower loss ratio')."
            )

        # Check for architectural specificity
        arch_words = [
            "conv",
            "attention",
            "ssm",
            "scan",
            "fft",
            "frequency",
            "linear",
            "residual",
            "gate",
            "sort",
            "pool",
            "kernel",
            "functional",
            "basis",
            "fixed_point",
            "token_mixing",
            "channel_mixing",
            "depth",
            "ops",
            "graph",
        ]
        has_arch = any(w in h_lower for w in arch_words)
        if not has_arch:
            concerns.append("No architectural specifics mentioned.")
            suggestions.append(
                "Reference specific operations, structure types, or graph properties."
            )

        # Check similarity to refuted hypotheses
        refuted_matches = self._check_refuted_overlap(hypothesis)
        if refuted_matches:
            top_match = refuted_matches[0]
            concerns.append(
                f"Similar to a REFUTED hypothesis (similarity={top_match['similarity']:.0%}): "
                f'"{top_match["refuted_text"]}"'
            )
            shared = ", ".join(top_match.get("shared_tokens", [])[:5])
            suggestions.append(
                f"Avoid repeating refuted directions. Shared concepts: {shared}. "
                "Pivot to a substantially different approach or explicitly address "
                "why the refuted hypothesis's failure mode does not apply here."
            )

        # Refinement-specific requirements
        if "fingerprint refinement" in h_lower or "refine" in h_lower:
            if not any(
                token in h_lower
                for token in [
                    "source_selection_rule",
                    "result_ids(",
                    "source_result_id",
                ]
            ):
                concerns.append(
                    "Fingerprint refinement undefined: no source-selection rule."
                )
                suggestions.append(
                    "Specify which seed architectures are selected and why (e.g., Stage-1 survivors only)."
                )
            if not any(
                token in h_lower
                for token in [
                    "mutation_mechanism",
                    "mutation_rate",
                    "operator",
                    "radius",
                    "neighborhood",
                ]
            ):
                concerns.append(
                    "Local mutation is underspecified: mutation operators/radius are missing."
                )
                suggestions.append(
                    "Declare mutation operators and neighborhood size (e.g., one-factor or max_edits<=2)."
                )
            if "intent=" in h_lower and not any(
                token in h_lower for token in ["weights=", "score=", "intent_weights"]
            ):
                concerns.append(
                    "Intent parameter is undefined: no scoring weights/formula provided."
                )
                suggestions.append(
                    "Define intent weights and scoring equation used for ranking candidates."
                )
            if not any(
                token in h_lower
                for token in [
                    "success_criteria",
                    "threshold",
                    "baseline",
                    "delta_",
                    ">=",
                    "<=",
                ]
            ):
                concerns.append("No explicit success criteria for refinement.")
                suggestions.append(
                    "Add measurable promotion criteria versus baseline/parent (e.g., ΔS1 or loss ratio threshold)."
                )

        if not concerns:
            return {
                "verdict": "proceed",
                "concerns": [],
                "suggestions": [],
                "confidence": 0.7,
            }
        elif len(concerns) >= 3:
            return {
                "verdict": "revise",
                "concerns": concerns,
                "suggestions": suggestions,
                "confidence": 0.3,
            }
        else:
            return {
                "verdict": "caution",
                "concerns": concerns,
                "suggestions": suggestions,
                "confidence": 0.5,
            }

    def _rule_based_mode_recommendation(self, data: Dict, digest=None) -> Dict:
        """
        Synthesizes metrics, failures, grammar weight trends, and architectural
        diversity to select the next experiment mode and parameters.  Uses diverse
        templates that cycle structurally or based on data.
        """
        # 1. Pipeline Escalation
        escalation_rec = self._escalate_pipeline_if_ready(data)
        if escalation_rec:
            return escalation_rec

        # 2. Compression Guardrail
        compression_rec = self._check_compression_guardrail(data)
        if compression_rec:
            return compression_rec

        # 3. Recovery Strategy (No Survivors)
        recovery_rec = self._get_recovery_strategy_if_needed(data)
        if recovery_rec:
            return recovery_rec

        # 4. Standard Exploration
        return self._get_standard_exploration_strategy(data)

    def _escalate_pipeline_if_ready(
        self, data: Dict
    ) -> (
        dict
    ):  # Note Optional removed to avoid tight imports, but can return dict | None
        metrics = data.get("pipeline_metrics", {})
        if metrics.get("should_start_s2", False):
            return {
                "mode": "integration",
                "reasoning": "Pipeline trigger: Escalate promising S1 models to S2 tasks.",
                "confidence": 0.95,
                "config": {"task_phase": getattr(self, "current_phase", "s1")},
            }
        elif metrics.get("should_start_s3", False):
            return {
                "mode": "ablation",
                "reasoning": "Pipeline trigger: Analyzing S2 models for S3 progression.",
                "confidence": 0.95,
                "config": {"task_phase": getattr(self, "current_phase", "s2")},
            }
        return None

    def _check_compression_guardrail(self, data: Dict) -> dict:  # Dict | None
        compression_active = (
            data.get("analytics_data", {}).get("compression_ratio", 0) > 1.2
        )
        large_params = data.get("analytics_data", {}).get("avg_params", 0) > 500000
        n_experiments = data.get("n_experiments_in_session", 0)

        if not compression_active and large_params:
            if n_experiments - getattr(self, "_last_compression_rec_cycle", 0) > 8:
                self._last_compression_rec_cycle = n_experiments
                return {
                    "mode": "synthesis",
                    "reasoning": (
                        "Models are getting large with no compression gain. "
                        "Enforcing high sparsity and bottlenecking."
                    ),
                    "confidence": 0.85,
                    "config": {
                        "structured_sparsity_bias": 0.7,
                        "max_depth": 3,
                        "residual_prob": 0.9,
                        "op_weights": {
                            "bottleneck_proj": 3.0,
                            "nm_sparse_linear": 3.0,
                            "low_rank_proj": 2.0,
                        },
                    },
                }
        return None

    def _get_recovery_strategy_if_needed(self, data: Dict) -> dict:  # Dict | None
        total_s1 = data.get("total_s1_survivors", 0)
        n_experiments = data.get("n_experiments_in_session", 0)

        if total_s1 == 0:
            top_failure_tuple = (
                data.get("analytics_data", {})
                .get("failure_patterns", {})
                .get("top", ["", 0])
            )
            failure_hint = ""
            if top_failure_tuple[1] > 0:
                failure_hint = f" Top failure: {top_failure_tuple[0]} ({top_failure_tuple[1]} cases)."

            if n_experiments < 3:
                return {
                    "mode": "synthesis",
                    "reasoning": "No S1 survivors yet. Continuing broad exploration.",
                    "confidence": 0.6,
                    "config": {"structured_sparsity_bias": 0.15},
                }

            recovery_idx = n_experiments % 5
            if recovery_idx == 0:
                return {
                    "mode": "synthesis",
                    "reasoning": (
                        f"No S1 survivors.{failure_hint} Conservative: high residual, shallow depth."
                    ),
                    "confidence": 0.7,
                    "config": {
                        "residual_prob": 0.85,
                        "max_depth": 5,
                        "max_ops": 8,
                        "n_programs": 80,
                        "structured_sparsity_bias": 0.15,
                    },
                }
            elif recovery_idx == 1:
                return {
                    "mode": "synthesis",
                    "reasoning": (
                        f"No S1 survivors.{failure_hint} Trying compact sparse architectures."
                    ),
                    "confidence": 0.65,
                    "config": {
                        "max_depth": 4,
                        "max_ops": 6,
                        "residual_prob": 0.9,
                        "n_programs": 80,
                        "structured_sparsity_bias": 0.6,
                        "op_weights": {
                            "nm_sparse_linear": 2.5,
                            "low_rank_proj": 2.5,
                            "bottleneck_proj": 2.0,
                        },
                    },
                }
            elif recovery_idx == 2:
                return {
                    "mode": "synthesis",
                    "reasoning": (
                        f"No S1 survivors.{failure_hint} Trying morphological box for structured diversity."
                    ),
                    "confidence": 0.65,
                    "config": {
                        "model_source": "mixed",
                        "morph_ratio": 0.7,
                        "n_programs": 70,
                        "max_depth": 6,
                        "residual_prob": 0.8,
                        "structured_sparsity_bias": 0.15,
                    },
                }
            elif recovery_idx == 3:
                return {
                    "mode": "synthesis",
                    "reasoning": (
                        f"No S1 survivors.{failure_hint} Boosting frequency/exotic ops for novel designs."
                    ),
                    "confidence": 0.6,
                    "config": {
                        "math_space_weight": 3.5,
                        "grammar_freq_domain_prob": 0.3,
                        "max_depth": 7,
                        "n_programs": 60,
                        "structured_sparsity_bias": 0.15,
                    },
                }
            else:
                return {
                    "mode": "evolution",
                    "reasoning": (
                        f"No S1 survivors.{failure_hint} Trying evolution to find viable variants of S0-passing architectures."
                    ),
                    "confidence": 0.55,
                    "config": {
                        "n_generations": 12,
                        "population_size": 25,
                        "mutation_rate": 0.8,
                        "structured_sparsity_bias": 0.15,
                    },
                }
        return None

    def _get_standard_exploration_strategy(self, data: Dict) -> Dict:
        n_experiments = data.get("n_experiments_in_session", 0)
        total_s1 = data.get("total_s1_survivors", 0)
        avg_novelty = data.get("analytics_data", {}).get("avg_novelty", 0)
        leaderboard_diversity = data.get("leaderboard_diversity", 0)
        recent_modes = data.get("recent_exploration_modes", [])

        categories = (
            data.get("analytics_data", {})
            .get("grammar_trends", {})
            .get("categories", {})
        )
        underexplored_cats = [c for c, w in categories.items() if w < 0.2]
        grammar_hint = (
            f" Focusing on {', '.join(underexplored_cats)}."
            if underexplored_cats
            else ""
        )

        if avg_novelty < 0.4 and "synthesis" not in recent_modes[-2:]:
            return {
                "mode": "synthesis",
                "reasoning": (
                    f"S1 survivors found but average novelty is low ({avg_novelty:.2f}). "
                    f"Boosting to diversify architecture search space.{grammar_hint}"
                ),
                "confidence": 0.8,
                "config": {
                    "math_space_weight": 4.0,
                    "grammar_freq_domain_prob": 0.5,
                    "model_source": "grammar",
                    "population_size": 120,
                    "structured_sparsity_bias": 0.15,
                },
            }

        explore_idx = n_experiments % 8
        if explore_idx == 0:
            return {
                "mode": "synthesis",
                "reasoning": (
                    f"Leaderboard has {leaderboard_diversity} unique "
                    f"viable archs. Focusing on structured morphology.{grammar_hint}"
                ),
                "confidence": 0.75,
                "config": {
                    "model_source": "mixed",
                    "morph_ratio": 0.9,
                    "n_programs": 80,
                    "structured_sparsity_bias": 0.15,
                },
            }
        elif explore_idx == 1:
            optimizer_diversity = data.get("optimizer_diversity", 0)
            return {
                "mode": "evolution",
                "reasoning": (
                    f"Evolving S1 architectures with {optimizer_diversity} "
                    "different optimizers injected into genome."
                ),
                "confidence": 0.75,
                "config": {
                    "n_generations": 20,
                    "population_size": 40,
                    "mutation_rate": 0.3,
                    "structured_sparsity_bias": 0.15,
                },
            }
        elif explore_idx == 2:
            return {
                "mode": "ablation",
                "reasoning": (
                    f"High success (S1={total_s1}). Running ablation test "
                    "to verify feature importance and extract minimal cores."
                ),
                "confidence": 0.8,
                "config": {"structured_sparsity_bias": 0.15},
            }
        elif explore_idx == 3:
            return {
                "mode": "synthesis",
                "reasoning": (
                    f"Grammar-focused synthesis to boost novelty. "
                    f"Novelty is currently {avg_novelty:.2f}, "
                    f"will push toward behaviorally diverse designs. "
                    f"Leaderboard diversity: {leaderboard_diversity} "
                    f"distinct S1 architectures.{grammar_hint}"
                ),
                "confidence": 0.8,
                "config": {
                    "model_source": "grammar",
                    "grammar_freq_domain_prob": 0.4,
                    "math_space_weight": 2.5,
                    "max_depth": 7,
                    "structured_sparsity_bias": 0.15,
                },
            }
        elif explore_idx == 4:
            return {
                "mode": "evolution",
                "reasoning": (
                    f"S1={total_s1}. High mutation evolution to jump out of "
                    "local optima in continuous space."
                ),
                "confidence": 0.7,
                "config": {
                    "mutation_rate": 0.95,
                    "n_generations": 15,
                    "population_size": 50,
                    "structured_sparsity_bias": 0.15,
                },
            }
        elif explore_idx == 5:
            return {
                "mode": "synthesis",
                "reasoning": (
                    f"S1={total_s1}. Focusing on very deep structures "
                    "using residual connections to trace gradients.{grammar_hint}"
                ),
                "confidence": 0.75,
                "config": {
                    "max_depth": 10,
                    "residual_prob": 0.95,
                    "n_programs": 75,
                    "structured_sparsity_bias": 0.15,
                },
            }
        elif explore_idx == 6:
            return {
                "mode": "synthesis",
                "reasoning": (
                    "Exploring hybrid approaches. Synthesizing models "
                    "biased heavily towards structural sparsity."
                ),
                "confidence": 0.7,
                "config": {
                    "structured_sparsity_bias": 0.8,
                    "max_depth": 4,
                    "n_programs": 100,
                    "op_weights": {
                        "nm_sparse_linear": 3.0,
                        "low_rank_proj": 2.0,
                    },
                },
            }
        else:
            return {
                "mode": "evolution",
                "reasoning": (
                    f"{total_s1} diverse S1 survivors provide a good "
                    "base for multi-objective optimization evolution "
                    "targeting both accuracy and compression ratio."
                ),
                "confidence": 0.85,
                "config": {
                    "n_generations": 25,
                    "population_size": 30,
                    "mutation_rate": 0.1,
                    "structured_sparsity_bias": 0.15,
                },
            }

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

    def _rule_based_hypothesis_validation(
        self, hypothesis: Dict, results: Dict
    ) -> Dict:
        """Metric-based hypothesis validation when LLM unavailable."""
        import re as _re

        success_metric = hypothesis.get("success_metric", "")
        s1_passed = results.get("stage1_passed", 0)

        # Try to parse "loss_ratio < X" or "s1_pass_rate > X%"
        status = "inconclusive"
        evidence = f"S1 passed: {s1_passed}"

        match = _re.match(r"loss_ratio\s*[<>]=?\s*([\d.]+)", success_metric)
        if match:
            threshold = float(match.group(1))
            best_lr = results.get("best_loss_ratio")
            if best_lr is not None:
                status = "confirmed" if best_lr < threshold else "refuted"
                evidence = f"best_loss_ratio={best_lr:.4f} vs threshold {threshold}"

        match = _re.match(r"s1_pass_rate\s*[>]=?\s*([\d.]+)%?", success_metric)
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

    def _rule_based_go_no_go(self, subject: str, evidence: str) -> Dict:
        """Rule-based go/no-go when LLM unavailable.

        Parses metric values from evidence string and applies thresholds
        instead of rubber-stamping everything as 'go'.
        """
        import re

        # Extract metrics from evidence string (e.g. "loss_ratio=0.45, novelty=0.6")
        lr_match = re.search(r"loss_ratio=([\d.]+)", evidence)
        nov_match = re.search(r"novelty=([\d.]+)", evidence)

        loss_ratio = float(lr_match.group(1)) if lr_match else None
        novelty = float(nov_match.group(1)) if nov_match else None

        decision = "go"
        rationale_parts = []

        # High novelty (>0.6) relaxes loss_ratio gate — novel architectures
        # are scientifically valuable even with moderate performance.
        high_novelty = novelty is not None and novelty > 0.6

        if loss_ratio is not None and loss_ratio > 0.5 and not high_novelty:
            decision = "no_go"
            rationale_parts.append(f"loss_ratio={loss_ratio:.3f} > 0.5 (too weak)")
        elif loss_ratio is not None and loss_ratio > 0.7:
            # Even high novelty can't save very poor performance
            decision = "no_go"
            rationale_parts.append(
                f"loss_ratio={loss_ratio:.3f} > 0.7 (too weak even for high novelty)"
            )
        elif novelty is not None and novelty < 0.3:
            decision = "no_go"
            rationale_parts.append(f"novelty={novelty:.3f} < 0.3 (not novel enough)")
        elif (
            loss_ratio is not None
            and loss_ratio > 0.3
            and novelty is not None
            and novelty < 0.5
        ):
            decision = "pivot"
            rationale_parts.append(
                f"loss_ratio={loss_ratio:.3f} > 0.3 and novelty={novelty:.3f} < 0.5 "
                f"(mediocre on both axes)"
            )
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
            "next_steps": (
                "Proceed to next phase"
                if decision == "go"
                else "Consider alternative architectures"
                if decision == "pivot"
                else "Candidate rejected — do not escalate"
            ),
        }

    def _rule_based_knowledge(
        self, results: List[Dict], hypotheses: List[Dict]
    ) -> List[Dict]:
        """Rule-based knowledge extraction when LLM unavailable."""
        entries = []
        # Extract from confirmed hypotheses
        for h in hypotheses:
            prediction = h.get("prediction", "")
            outcome = h.get("outcome_summary", "")
            reasoning = h.get("reasoning", "")
            test_method = h.get("test_method", "")
            if h.get("status") == "confirmed":
                parts = [f"Hypothesis: {prediction}"]
                if reasoning:
                    parts.append(f"Reasoning: {reasoning}")
                if test_method:
                    parts.append(f"Test: {test_method}")
                if outcome:
                    parts.append(f"Outcome: {outcome}")
                entries.append(
                    {
                        "category": "principle",
                        "title": f"Confirmed: {prediction}",
                        "content": "\n".join(parts),
                        "confidence": h.get("confidence_after", 0.6),
                    }
                )
            elif h.get("status") == "refuted":
                parts = [f"Hypothesis: {prediction}"]
                if reasoning:
                    parts.append(f"Reasoning: {reasoning}")
                if test_method:
                    parts.append(f"Test: {test_method}")
                if outcome:
                    parts.append(f"Outcome: {outcome}")
                entries.append(
                    {
                        "category": "anti_pattern",
                        "title": f"Refuted: {prediction}",
                        "content": "\n".join(parts),
                        "confidence": h.get("confidence_after", 0.6),
                    }
                )
        return entries[:5]  # limit to 5

    def _rule_based_campaign_report(
        self,
        campaign: Dict,
        experiments: List[Dict],
        hypotheses: List[Dict],
        decisions: List[Dict],
        knowledge: List[Dict],
    ) -> str:
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
