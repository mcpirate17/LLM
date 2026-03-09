from __future__ import annotations
import json
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ...eval.utils import safe_parse_float
from ...synthesis.primitives import get_primitive


class RefinementAnalyzer:
    """Data-driven refinement advisor that examines programs against population success data.

    Produces concrete recommendations (swap/add/remove ops) with evidence,
    behavioral gap analysis, and grammar hints for smarter mutations.
    """

    # Behavioral gap → improvement op suggestions
    _GAP_HINTS: Dict[str, List[str]] = {
        "fp_isotropy": ["rmsnorm", "layer_norm", "batch_norm"],
        "fp_interaction_locality": ["softmax_attention", "state_space", "fourier_mixing"],
        "fp_sensitivity_uniformity": ["residual", "skip_connection", "highway_gate"],
        "fp_rank_ratio": ["low_rank_proj", "linear_proj", "bottleneck_proj"],
        "fp_interaction_sparsity": ["topk_gate", "threshold_gate", "sparse_linear"],
        "fp_intrinsic_dim": ["grouped_linear", "low_rank_proj", "bottleneck_proj"],
        "fp_jacobian_spectral_norm": ["rmsnorm", "layer_norm", "residual", "highway_gate"],
    }

    # Human-readable metric labels
    _METRIC_LABELS: Dict[str, str] = {
        "fp_isotropy": "Isotropy",
        "fp_interaction_locality": "Interaction Locality",
        "fp_interaction_sparsity": "Interaction Sparsity",
        "fp_interaction_symmetry": "Interaction Symmetry",
        "fp_interaction_hierarchy": "Interaction Hierarchy",
        "fp_intrinsic_dim": "Intrinsic Dimensionality",
        "fp_rank_ratio": "Rank Ratio",
        "fp_jacobian_spectral_norm": "Jacobian Spectral Norm",
        "fp_jacobian_effective_rank": "Jacobian Effective Rank",
        "fp_sensitivity_uniformity": "Sensitivity Uniformity",
    }

    # Fingerprint metrics to analyze
    _FP_METRICS = [
        "fp_isotropy", "fp_interaction_locality", "fp_interaction_sparsity",
        "fp_interaction_symmetry", "fp_interaction_hierarchy", "fp_intrinsic_dim",
        "fp_rank_ratio", "fp_sensitivity_uniformity", "fp_jacobian_spectral_norm",
    ]

    def __init__(self, analytics: ExperimentAnalytics):
        self.analytics = analytics
        self.nb = analytics.nb

    def analyze_program_for_refinement(
        self, result_id: str, program_row: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze a program against population data and produce refinement recommendations."""
        graph_json = program_row.get("graph_json")
        if isinstance(graph_json, dict):
            graph_json = json.dumps(graph_json)
        program_ops = self.analytics._extract_ops_fast(graph_json or "") or []

        # Map ops to categories
        op_categories: Dict[str, str] = {}
        for op_name in program_ops:
            try:
                prim = get_primitive(op_name)
                op_categories[op_name] = prim.category.value
            except Exception:
                op_categories[op_name] = "unknown"

        # Fetch population op stats
        op_stats_rows = self.nb.get_op_success_rates()
        pop_s1_rates: Dict[str, float] = {}
        pop_s0_rates: Dict[str, float] = {}
        pop_n_used: Dict[str, int] = {}
        pop_categories: Dict[str, str] = {}

        for row in op_stats_rows:
            op = str(row.get("op_name", ""))
            n_used = int(row.get("n_used") or 0)
            n_s1 = int(row.get("n_stage1_passed") or 0)
            n_s0 = int(row.get("n_stage0_passed") or 0)
            pop_n_used[op] = n_used
            pop_s1_rates[op] = n_s1 / n_used if n_used > 0 else 0.0
            pop_s0_rates[op] = n_s0 / n_used if n_used > 0 else 0.0
            try:
                prim = get_primitive(op)
                pop_categories[op] = prim.category.value
            except Exception:
                pop_categories[op] = "unknown"

        # Compute mean S1 rate across ops with sufficient data
        qualified_rates = [
            pop_s1_rates[op] for op, n in pop_n_used.items() if n >= 5
        ]
        mean_s1_rate = (
            sum(qualified_rates) / len(qualified_rates) if qualified_rates else 0.0
        )

        n_programs_total = sum(pop_n_used.values()) // max(1, len(pop_n_used)) if pop_n_used else 0
        n_stage1_passed = sum(
            int(r.get("n_stage1_passed") or 0) for r in op_stats_rows
        ) // max(1, len(op_stats_rows)) if op_stats_rows else 0

        if not program_ops and not op_stats_rows:
            return self._empty_analysis(result_id, "no_data")

        # Per-op health cards
        op_health = self._build_op_health(
            program_ops, op_categories, pop_s1_rates, pop_s0_rates,
            pop_n_used, pop_categories, mean_s1_rate,
        )

        # Recommended additions
        recommended_additions = self._build_recommended_additions(
            program_ops, pop_s1_rates, pop_n_used, pop_categories, mean_s1_rate,
        )

        # Behavioral gap analysis
        behavioral_gaps = self._build_behavioral_gaps(program_row)

        # Build recipe
        recipe = self._build_recipe(
            op_health, behavioral_gaps, program_row, mean_s1_rate,
        )

        # Brittleness advice (Aria's technical insight)
        brittleness_advice = self._build_brittleness_advice(program_row)

        analysis_quality = "full" if qualified_rates else ("partial" if op_stats_rows else "no_data")

        return {
            "result_id": result_id,
            "graph_fingerprint": program_row.get("graph_fingerprint", ""),
            "program_ops": program_ops,
            "op_health": op_health,
            "recommended_additions": recommended_additions,
            "behavioral_gaps": behavioral_gaps,
            "recipe": recipe,
            "brittleness_advice": brittleness_advice,
            "population_stats": {
                "n_programs_total": n_programs_total,
                "n_stage1_passed": n_stage1_passed,
                "mean_s1_rate": round(mean_s1_rate, 4),
            },
            "analysis_quality": analysis_quality,
        }

    def _build_op_health(
        self,
        program_ops: List[str],
        op_categories: Dict[str, str],
        pop_s1_rates: Dict[str, float],
        pop_s0_rates: Dict[str, float],
        pop_n_used: Dict[str, int],
        pop_categories: Dict[str, str],
        mean_s1_rate: float,
    ) -> List[Dict[str, Any]]:
        """Build per-op health cards with recommendations."""
        cards: List[Dict[str, Any]] = []
        program_op_set = set(program_ops)

        for op_name in program_ops:
            n = pop_n_used.get(op_name, 0)
            s1 = pop_s1_rates.get(op_name, 0.0)
            s0 = pop_s0_rates.get(op_name, 0.0)
            cat = op_categories.get(op_name, "unknown")

            # Classify health
            if n < 5:
                health = "untested"
                recommendation = "investigate"
            elif s0 < 0.5 and n >= 5:
                health = "risky"
                recommendation = "swap"
            elif s1 < mean_s1_rate * 0.5 and n >= 10:
                health = "weak"
                recommendation = "swap"
            elif s1 >= mean_s1_rate * 1.2 and n >= 5:
                health = "strong"
                recommendation = "keep"
            else:
                health = "neutral"
                recommendation = "keep"

            # Find swap candidates: same category, higher S1, not in program
            swap_candidates: List[Dict[str, Any]] = []
            if recommendation in ("swap", "investigate"):
                same_cat_ops = [
                    (op, pop_s1_rates[op])
                    for op, c in pop_categories.items()
                    if c == cat and op not in program_op_set
                    and pop_n_used.get(op, 0) >= 5
                    and pop_s1_rates.get(op, 0.0) > s1
                ]
                same_cat_ops.sort(key=lambda x: x[1], reverse=True)
                swap_candidates = [
                    {"op_name": op, "s1_rate": round(rate, 4)}
                    for op, rate in same_cat_ops[:3]
                ]

            cards.append({
                "op_name": op_name,
                "category": cat,
                "global_s1_rate": round(s1, 4),
                "global_s0_rate": round(s0, 4),
                "n_used": n,
                "health": health,
                "recommendation": recommendation,
                "swap_candidates": swap_candidates,
            })

        return cards

    def _build_recommended_additions(
        self,
        program_ops: List[str],
        pop_s1_rates: Dict[str, float],
        pop_n_used: Dict[str, int],
        pop_categories: Dict[str, str],
        mean_s1_rate: float,
    ) -> List[Dict[str, Any]]:
        """Find high-performing ops not in this program."""
        program_op_set = set(program_ops)
        candidates: List[Dict[str, Any]] = []

        for op_name, s1 in pop_s1_rates.items():
            if op_name in program_op_set:
                continue
            n = pop_n_used.get(op_name, 0)
            if n < 5 or s1 < mean_s1_rate:
                continue
            freq_score = s1 * math.log1p(n)
            candidates.append({
                "op_name": op_name,
                "category": pop_categories.get(op_name, "unknown"),
                "global_s1_rate": round(s1, 4),
                "top_performer_frequency": n,
                "score": freq_score,
                "rationale": f"S1 rate {s1:.1%} across {n} uses, above population mean",
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        for c in candidates:
            del c["score"]
        return candidates[:5]

    def _build_behavioral_gaps(
        self, program_row: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Compare program fingerprint metrics to population S1 survivor means."""
        gaps: List[Dict[str, Any]] = []

        # Fetch S1 survivor stats from recent experiments
        try:
            rows = self.nb.conn.execute(
                """SELECT {} FROM program_results
                   WHERE stage1_passed = 1""".format(
                    ", ".join(self._FP_METRICS)
                )
            ).fetchall()
        except Exception:
            return gaps

        if len(rows) < 3:
            return gaps

        # Compute population means and stds
        pop_stats: Dict[str, Tuple[float, float]] = {}
        for metric in self._FP_METRICS:
            values = [
                float(r[metric]) for r in rows
                if r[metric] is not None
                and not math.isnan(float(r[metric]))
                and not math.isinf(float(r[metric]))
            ]
            if len(values) >= 3:
                mean_val = sum(values) / len(values)
                std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
                pop_stats[metric] = (mean_val, max(std_val, 1e-8))

        for metric, (pop_mean, pop_std) in pop_stats.items():
            prog_val = program_row.get(metric)
            if prog_val is None:
                continue
            try:
                prog_val = float(prog_val)
                if math.isnan(prog_val) or math.isinf(prog_val):
                    continue
            except (TypeError, ValueError):
                continue

            z_score = (prog_val - pop_mean) / pop_std
            abs_z = abs(z_score)
            if abs_z < 1.0:
                continue

            severity = "low" if abs_z < 1.5 else ("medium" if abs_z < 2.0 else "high")
            improvement_ops = self._GAP_HINTS.get(metric, [])

            gaps.append({
                "metric": metric,
                "label": self._METRIC_LABELS.get(metric, metric),
                "program_value": round(prog_val, 4),
                "population_mean": round(pop_mean, 4),
                "population_std": round(pop_std, 4),
                "z_score": round(z_score, 2),
                "severity": severity,
                "improvement_ops": improvement_ops,
            })

        gaps.sort(key=lambda g: abs(g["z_score"]), reverse=True)
        return gaps

    def _build_recipe(
        self,
        op_health: List[Dict[str, Any]],
        behavioral_gaps: List[Dict[str, Any]],
        program_row: Dict[str, Any],
        mean_s1_rate: float,
    ) -> Dict[str, Any]:
        """Combine analysis into actionable recipe with grammar hints."""
        risky_ops = [h for h in op_health if h["health"] == "risky" and h["n_used"] >= 8]
        weak_ops = [h for h in op_health if h["health"] == "weak"]
        significant_gaps = [g for g in behavioral_gaps if g["severity"] in ("medium", "high")]

        # Determine exclude/boost ops
        exclude_ops = [h["op_name"] for h in risky_ops]
        boost_ops: Dict[str, float] = {}
        for h in op_health:
            if h["health"] == "strong":
                boost_ops[h["op_name"]] = min(3.0, 1.5)
            for sc in h.get("swap_candidates", []):
                boost_ops[sc["op_name"]] = min(3.0, 2.0)

        # Categories to boost from gap hints
        add_categories: Dict[str, float] = {}
        for gap in significant_gaps:
            for op in gap.get("improvement_ops", []):
                try:
                    prim = get_primitive(op)
                    cat = prim.category.value
                    add_categories[cat] = max(add_categories.get(cat, 1.0), 1.5)
                except Exception:
                    pass

        # Determine intent
        loss_ratio = program_row.get("loss_ratio")
        param_count = program_row.get("param_count") or program_row.get("graph_n_params_estimate")
        has_reasonable_loss = isinstance(loss_ratio, (int, float)) and float(loss_ratio) < 1.1
        has_high_params = isinstance(param_count, (int, float)) and float(param_count) > 100000

        if risky_ops:
            recommended_intent = "quality"
            primary_target = f"Replace {len(risky_ops)} risky op(s) with proven alternatives"
            confidence = "high" if len(risky_ops) >= 2 else "medium"
        elif len(weak_ops) >= 2:
            recommended_intent = "quality"
            primary_target = f"Improve {len(weak_ops)} underperforming ops"
            confidence = "medium"
        elif significant_gaps:
            recommended_intent = "novelty"
            top_gap = significant_gaps[0]
            primary_target = f"Address {top_gap['label']} gap (z={top_gap['z_score']:+.1f})"
            confidence = "medium" if len(significant_gaps) >= 2 else "low"
        elif has_high_params and has_reasonable_loss:
            recommended_intent = "compression"
            primary_target = "Reduce parameter count while maintaining quality"
            confidence = "medium"
        else:
            recommended_intent = "balanced"
            primary_target = "General improvement across all dimensions"
            confidence = "low"

        # Human summary
        parts: List[str] = []
        if risky_ops:
            parts.append(f"{len(risky_ops)} risky op(s) should be replaced")
        if weak_ops:
            parts.append(f"{len(weak_ops)} weak op(s) could be improved")
        if significant_gaps:
            gap_names = [g["label"] for g in significant_gaps[:3]]
            parts.append(f"behavioral gaps in {', '.join(gap_names)}")
        if not parts:
            parts.append("No major issues detected; balanced refinement recommended")
        human_summary = "; ".join(parts) + "."

        return {
            "recommended_intent": recommended_intent,
            "confidence": confidence,
            "primary_target": primary_target,
            "grammar_hints": {
                "exclude_ops": exclude_ops,
                "boost_ops": boost_ops,
                "add_categories": add_categories,
            },
            "human_summary": human_summary,
        }

    def _build_brittleness_advice(self, program_row: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Aria's technical advice for making architectures less brittle."""
        robustness = program_row.get("investigation_robustness")
        spectral_norm = program_row.get("fp_jacobian_spectral_norm")
        init_std = program_row.get("init_sensitivity_std")
        
        advice_parts = []
        is_brittle = False
        
        if robustness is not None and robustness < 0.5:
            is_brittle = True
            advice_parts.append(
                f"Low robustness ({robustness:.2f}) indicates high sensitivity to training recipes. "
                "Only a fraction of hyperparameter seeds converge successfully."
            )
            
        if spectral_norm is not None and spectral_norm > 15.0:
            is_brittle = True
            advice_parts.append(
                f"High Jacobian spectral norm ({spectral_norm:.1f}) suggests 'exploding' gradient paths "
                "or poor signal propagation. This often leads to training instability."
            )
            
        if init_std is not None and init_std > 0.1:
            is_brittle = True
            advice_parts.append(
                f"High initialization sensitivity ({init_std:.3f}) means the architecture's success "
                "depends heavily on lucky weight initialization."
            )

        if not is_brittle:
            return None

        # Practical remedies
        remedies = [
            "Add 'rmsnorm' or 'layer_norm' before non-linear ops to bound activations.",
            "Use 'residual' or 'skip_connection' paths to improve gradient flow.",
            "Verify that learning rates are sufficiently low for this topology.",
            "If using 'Spectral-Conv', consider adding a small epsilon or weight decay to fourier mixing."
        ]
        
        return {
            "summary": "This architecture is marked as 'Brittle' because it lacks training stability.",
            "diagnosis": " ".join(advice_parts),
            "remedies": remedies,
            "aria_insight": (
                "To make this front-runner reliable, we must tame its variance. "
                "A 0.33 robustness means it's a great 'idea' that currently requires 'perfect' conditions. "
                "Stabilizing it will likely improve its validation score even further."
            )
        }

    def _empty_analysis(self, result_id: str, quality: str) -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            "result_id": result_id,
            "graph_fingerprint": "",
            "program_ops": [],
            "op_health": [],
            "recommended_additions": [],
            "behavioral_gaps": [],
            "recipe": {
                "recommended_intent": "balanced",
                "confidence": "low",
                "primary_target": "Insufficient data for analysis",
                "grammar_hints": {
                    "exclude_ops": [],
                    "boost_ops": {},
                    "add_categories": {},
                },
                "human_summary": "Insufficient population data for analysis.",
            },
            "population_stats": {
                "n_programs_total": 0,
                "n_stage1_passed": 0,
                "mean_s1_rate": 0.0,
            },
            "analysis_quality": quality,
        }

    def strategy_backtest(self) -> Dict[str, Any]:
        """Aggregate cross-experiment outcomes by search intent."""
        rows = self.nb.conn.execute(
            """
            SELECT
                config_json,
                n_programs_generated,
                n_stage1_passed,
                n_stable_passed,
                best_loss_ratio,
                best_novelty_score,
                avg_throughput_tok_s,
                avg_routing_token_retention,
                duration_seconds
            FROM experiments
            WHERE status = 'completed'
            """
        ).fetchall()

        by_intent = defaultdict(lambda: {
            "n_experiments": 0,
            "n_programs": 0,
            "n_s1_passed": 0,
            "n_stable_passed": 0,
            "total_loss": 0.0,
            "loss_count": 0,
            "total_novelty": 0.0,
            "novelty_count": 0,
            "total_throughput": 0.0,
            "throughput_count": 0,
            "total_duration": 0.0,
        })

        for row in rows:
            config = json.loads(row["config_json"]) if row["config_json"] else {}
            intent = config.get("refine_intent") or config.get("mode")
            if not intent or intent not in ("quality", "compression", "sparsity", "novelty", "balanced"):
                # Map some common modes to intents if not explicitly set
                if config.get("mode") == "novelty": intent = "novelty"
                elif config.get("mode") == "scale_up": intent = "quality"
                else: intent = "balanced"
            
            d = by_intent[intent]
            d["n_experiments"] += 1
            d["n_programs"] += (row["n_programs_generated"] or 0)
            d["n_s1_passed"] += (row["n_stage1_passed"] or 0)
            d["n_stable_passed"] += (row["n_stable_passed"] or 0)
            d["total_duration"] += (row["duration_seconds"] or 0)
            
            if row["best_loss_ratio"] is not None:
                d["total_loss"] += row["best_loss_ratio"]
                d["loss_count"] += 1
            if row["best_novelty_score"] is not None:
                d["total_novelty"] += row["best_novelty_score"]
                d["novelty_count"] += 1
            if row["avg_throughput_tok_s"] is not None:
                d["total_throughput"] += row["avg_throughput_tok_s"]
                d["throughput_count"] += 1

        results = []
        for intent, d in by_intent.items():
            results.append({
                "intent": intent,
                "n_experiments": d["n_experiments"],
                "n_programs": d["n_programs"],
                "s1_pass_rate": d["n_s1_passed"] / d["n_programs"] if d["n_programs"] > 0 else 0.0,
                "stable_pass_rate": d["n_stable_passed"] / d["n_programs"] if d["n_programs"] > 0 else 0.0,
                "avg_best_loss": d["total_loss"] / d["loss_count"] if d["loss_count"] > 0 else None,
                "avg_best_novelty": d["total_novelty"] / d["novelty_count"] if d["novelty_count"] > 0 else None,
                "avg_throughput": d["total_throughput"] / d["throughput_count"] if d["throughput_count"] > 0 else None,
                "avg_duration": d["total_duration"] / d["n_experiments"] if d["n_experiments"] > 0 else 0.0,
            })

        return {
            "intents": sorted(results, key=lambda x: -x["n_experiments"]),
            "total_experiments": len(rows)
        }

