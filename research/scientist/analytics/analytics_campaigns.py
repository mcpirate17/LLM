from __future__ import annotations
import json
import re
from typing import Dict, List, Optional


class _CampaignsMixin:
    """Campaign criteria tracking, negative results synthesis, decision analysis."""

    __slots__ = ()

    @staticmethod
    def _parse_criteria_text(success_criteria: str) -> List[str]:
        if not isinstance(success_criteria, str) or not success_criteria.strip():
            return []
        items: List[str] = []
        for part in re.split(r"\n|;|\|", success_criteria):
            text = part.strip()
            if not text:
                continue
            text = re.sub(r"^[-*•]\s*", "", text)
            text = re.sub(r"^\d+[.)]\s*", "", text)
            if text:
                items.append(text)
        return items

    @staticmethod
    def _parse_threshold(text: str) -> Optional[Dict]:
        symbol_match = re.search(r"(<=|>=|<|>|=)\s*(\d+(?:\.\d+)?)(\s*%)?", text)
        if symbol_match:
            return {
                "op": symbol_match.group(1),
                "value": float(symbol_match.group(2)),
                "is_percent": bool(symbol_match.group(3)),
            }

        phrase_patterns = [
            (r"at least\s*(\d+(?:\.\d+)?)(\s*%)?", ">="),
            (r"no more than\s*(\d+(?:\.\d+)?)(\s*%)?", "<="),
            (r"less than\s*(\d+(?:\.\d+)?)(\s*%)?", "<"),
            (r"greater than\s*(\d+(?:\.\d+)?)(\s*%)?", ">"),
        ]
        for pattern, op in phrase_patterns:
            match = re.search(pattern, text)
            if match:
                return {
                    "op": op,
                    "value": float(match.group(1)),
                    "is_percent": bool(match.group(2)),
                }
        return None

    @staticmethod
    def _infer_criterion_type(text: str) -> str:
        if "baseline" in text or "loss ratio" in text:
            return "baseline"
        if "novelty" in text:
            return "novelty"
        if "stage 1" in text or "stage1" in text or "s1" in text or "survivor" in text:
            return "stage1"
        if "decision" in text or "go/no-go" in text or "go no-go" in text:
            return "decision"
        return "unknown"

    @staticmethod
    def _normalize_threshold(
        criterion_type: str, threshold: Optional[Dict]
    ) -> Optional[Dict]:
        if not threshold or criterion_type == "decision":
            return threshold
        should_normalize_as_ratio = (
            bool(threshold.get("is_percent")) or float(threshold.get("value", 0)) > 1
        )
        if not should_normalize_as_ratio:
            return threshold
        value = float(threshold.get("value", 0))
        normalized = dict(threshold)
        normalized["value"] = value / 100.0 if value <= 100 else value
        return normalized

    @staticmethod
    def _compare_threshold(
        observed: Optional[float], threshold: Optional[Dict]
    ) -> Optional[bool]:
        if observed is None or not threshold:
            return None
        op = threshold.get("op")
        value = float(threshold.get("value", 0))
        if op == "<":
            return observed < value
        if op == "<=":
            return observed <= value
        if op == ">":
            return observed > value
        if op == ">=":
            return observed >= value
        if op == "=":
            return abs(observed - value) < 1e-9
        return None

    @staticmethod
    def _threshold_label(criterion_type: str, threshold: Optional[Dict]) -> str:
        if not threshold:
            return ""
        op = threshold.get("op", "")
        value = float(threshold.get("value", 0))
        if criterion_type in {"baseline", "novelty", "stage1"}:
            if criterion_type == "stage1":
                return f" (target {op} {value * 100:.1f}%)"
            return f" (target {op} {value:.3f})"
        return f" (target {op} {value:g})"

    def campaign_success_criteria_tracker(
        self,
        campaign: Dict,
        experiments: List[Dict],
        hypotheses: List[Dict],
        decisions: List[Dict],
    ) -> List[Dict]:
        criteria = self._parse_criteria_text(
            (campaign or {}).get("success_criteria", "")
        )
        if not criteria:
            return []

        baseline_values = [
            float(exp.get("best_baseline_ratio"))
            for exp in experiments
            if isinstance(exp.get("best_baseline_ratio"), (int, float))
        ]
        novelty_values = [
            float(exp.get("best_novelty_score"))
            for exp in experiments
            if isinstance(exp.get("best_novelty_score"), (int, float))
        ]
        stage1_values = []
        for exp in experiments:
            total = exp.get("n_programs_generated") or exp.get("n_programs") or 0
            passed = exp.get("n_stage1_passed") or 0
            if total:
                stage1_values.append(float(passed) / float(total))

        best_baseline_ratio = min(baseline_values) if baseline_values else None
        best_novelty = max(novelty_values) if novelty_values else None
        best_stage1_rate = max(stage1_values) if stage1_values else None
        experiment_count = len(experiments)
        hypothesis_count = len(hypotheses)
        decision_count = len(decisions)

        tracker: List[Dict] = []
        for index, criterion in enumerate(criteria):
            text = criterion.lower()
            criterion_type = self._infer_criterion_type(text)
            threshold = self._normalize_threshold(
                criterion_type, self._parse_threshold(text)
            )
            item = {
                "id": f"{index}-{criterion}",
                "criterion": criterion,
                "criterion_type": criterion_type,
                "status": "not_yet",
                "observed_text": "No mapped metric yet (criterion type not recognized).",
            }

            if criterion_type == "baseline":
                observed = best_baseline_ratio
                passed = (
                    self._compare_threshold(observed, threshold)
                    if threshold
                    else (observed < 1.0 if observed is not None else None)
                )
                item["status"] = (
                    "not_yet"
                    if passed is None
                    else "pass"
                    if passed
                    else "at_risk"
                    if experiment_count > 0
                    else "not_yet"
                )
                if observed is not None:
                    item["observed_text"] = (
                        f"best baseline ratio {observed:.3f}{self._threshold_label(criterion_type, threshold)}"
                    )
                else:
                    item["observed_text"] = "baseline ratio not yet measured"

            elif criterion_type == "novelty":
                observed = best_novelty
                passed = (
                    self._compare_threshold(observed, threshold)
                    if threshold
                    else (observed >= 0.7 if observed is not None else None)
                )
                item["status"] = (
                    "not_yet"
                    if passed is None
                    else "pass"
                    if passed
                    else "at_risk"
                    if experiment_count > 0
                    else "not_yet"
                )
                if observed is not None:
                    item["observed_text"] = (
                        f"best novelty {observed:.3f}{self._threshold_label(criterion_type, threshold)}"
                    )
                else:
                    item["observed_text"] = "novelty signal not yet available"

            elif criterion_type == "stage1":
                observed = best_stage1_rate
                passed = (
                    self._compare_threshold(observed, threshold)
                    if threshold
                    else (observed >= 0.05 if observed is not None else None)
                )
                item["status"] = (
                    "not_yet"
                    if passed is None
                    else "pass"
                    if passed
                    else "at_risk"
                    if experiment_count > 0
                    else "not_yet"
                )
                if observed is not None:
                    item["observed_text"] = (
                        f"best S1 rate {observed * 100:.1f}%{self._threshold_label(criterion_type, threshold)}"
                    )
                else:
                    item["observed_text"] = "S1 evidence not yet available"

            elif criterion_type == "decision":
                observed = float(decision_count)
                passed = (
                    self._compare_threshold(observed, threshold)
                    if threshold
                    else observed > 0
                )
                item["status"] = (
                    "pass"
                    if passed
                    else "at_risk"
                    if hypothesis_count > 0
                    else "not_yet"
                )
                item["observed_text"] = (
                    f"{decision_count} decision{'s' if decision_count != 1 else ''} logged"
                    f"{self._threshold_label(criterion_type, threshold)}"
                )

            tracker.append(item)

        return tracker

    def negative_results_synthesis(self) -> Dict:
        """Aggregate repeatedly failed patterns into a "do not pursue" list.

        Combines zero-success ops, dominant error types, anti-correlated
        structural features, and refuted hypotheses into a single report.
        """
        result: Dict = {
            "failed_ops": [],
            "weak_ops": [],
            "dominant_errors": [],
            "anti_patterns": [],
            "toxic_bigrams": [],
            "refuted_hypotheses": [],
            "summary": "",
        }

        # 1. Ops with 0% S1 rate and sufficient sample size
        op_rates = self.op_success_rates()
        min_usage = 5
        for op_name, stats in sorted(
            op_rates.items(), key=lambda x: -(x[1].get("n_used", 0))
        ):
            n_used = stats.get("n_used", 0)
            n_s0 = stats.get("n_s0", 0)
            s1_rate = stats.get("s1_rate", 0)
            s0_rate = stats.get("s0_rate", 0)

            # CRITICAL FIX: Only label as 'failed' if it ACTUALLY compiled (S0)
            # but failed to learn (S1). We need at least some successful
            # compilations to make a scientific judgment on utility.
            # If s0_rate is low, it's a code/integration bug to fix,
            # not a scientific reason to stop pursuing the component.
            if n_s0 >= 3 and s1_rate == 0 and s0_rate >= 0.8:
                entry = {
                    "op_name": op_name,
                    "n_used": n_used,
                    "n_s0": n_s0,
                    "s0_rate": round(s0_rate, 3),
                    "s1_rate": 0.0,
                    "failure_stage": "learning",
                    "confidence": round(min(0.95, 0.4 + n_s0 / 100), 2),
                }
                # All failed ops get soft penalties (no hard exclusion)
                entry["penalty_weight"] = 0.5
                result["weak_ops"].append(entry)

        # 1b. Weak ops: nonzero but poor S1 rate (soft penalty candidates).
        # These shouldn't be hard-excluded but should be selected less often.
        mean_s1 = 0.0
        s1_vals = [
            s.get("s1_rate", 0)
            for s in op_rates.values()
            if s.get("n_used", 0) >= min_usage
        ]
        if s1_vals:
            mean_s1 = sum(s1_vals) / len(s1_vals)
        weak_threshold = max(mean_s1 * 0.5, 0.20)
        for op_name, stats in sorted(
            op_rates.items(), key=lambda x: x[1].get("s1_rate", 0)
        ):
            n_used = stats.get("n_used", 0)
            s1_rate = stats.get("s1_rate", 0)
            if n_used >= min_usage and 0 < s1_rate <= weak_threshold:
                # Soft penalty: linearly scale from 0.2 (at s1=0) to 1.0 (at threshold)
                penalty = round(max(0.2, s1_rate / weak_threshold), 2)
                result["weak_ops"].append(
                    {
                        "op_name": op_name,
                        "n_used": n_used,
                        "s1_rate": round(s1_rate, 3),
                        "penalty_weight": penalty,
                        "threshold": round(weak_threshold, 3),
                    }
                )

        # 2. Dominant error types (top 10 by count)
        failures = self.failure_patterns()
        total_failures = sum(v["total"] for v in failures.values())
        for error_type, info in sorted(failures.items(), key=lambda x: -x[1]["total"])[
            :10
        ]:
            pct = info["total"] / total_failures if total_failures > 0 else 0
            top_stage = max(
                info.get("by_stage", {}).items(),
                key=lambda x: x[1],
                default=("unknown", 0),
            )
            result["dominant_errors"].append(
                {
                    "error_type": error_type,
                    "count": info["total"],
                    "percentage": round(pct * 100, 1),
                    "primary_stage": top_stage[0],
                    "by_stage": info.get("by_stage", {}),
                }
            )

        # 3. Anti-correlated structural features (negative correlations)
        correlations = self.structural_correlations()
        for metric, effect in sorted(correlations.items(), key=lambda x: x[1]):
            if effect < -0.15:
                name = metric.replace("graph_", "").replace("_", " ")
                result["anti_patterns"].append(
                    {
                        "feature": name,
                        "metric": metric,
                        "correlation": round(effect, 3),
                        "interpretation": (
                            f"Higher {name} is associated with lower S1 success"
                        ),
                    }
                )

        # 4. Refuted hypotheses from insights table
        try:
            rows = self.nb.conn.execute("""
                SELECT content, confidence, supporting_evidence, timestamp
                FROM insights
                WHERE status = 'refuted'
                ORDER BY timestamp DESC
                LIMIT 20
            """).fetchall()
            for r in rows:
                result["refuted_hypotheses"].append(
                    {
                        "content": r["content"],
                        "confidence": r["confidence"],
                        "evidence": r["supporting_evidence"],
                        "timestamp": r["timestamp"],
                    }
                )
        except Exception:
            pass

        # 5. Toxic op-pair bigrams from failure_signatures table
        try:
            blocklist = self.nb.get_failure_signature_blocklist()
            for sig, penalty in sorted(blocklist.items(), key=lambda x: x[1]):
                op1, op2 = sig.split("->") if "->" in sig else (sig, "unknown")
                cat1, cat2 = "unknown", "unknown"
                try:
                    cat1 = get_primitive(op1).category.value
                except Exception:
                    pass
                try:
                    cat2 = get_primitive(op2).category.value
                except Exception:
                    pass

                result["toxic_bigrams"].append(
                    {
                        "pattern": sig,
                        "op1": op1,
                        "op2": op2,
                        "cat1": cat1,
                        "cat2": cat2,
                        "penalty": penalty,
                    }
                )
        except Exception:
            pass

        # 6. Summary text
        n_ops = len(result["failed_ops"])
        n_weak = len(result["weak_ops"])
        n_errs = len(result["dominant_errors"])
        n_anti = len(result["anti_patterns"])
        n_toxic = len(result["toxic_bigrams"])
        n_ref = len(result["refuted_hypotheses"])
        parts = []
        if n_ops:
            op_names = ", ".join(o["op_name"] for o in result["failed_ops"][:5])
            parts.append(f"{n_ops} ops with 0% S1 rate ({op_names})")
        if n_weak:
            weak_names = ", ".join(o["op_name"] for o in result["weak_ops"][:5])
            parts.append(f"{n_weak} weak ops soft-penalized ({weak_names})")
        if n_errs:
            parts.append(
                f"{n_errs} error types, top: {result['dominant_errors'][0]['error_type']}"
                f" ({result['dominant_errors'][0]['count']} occurrences)"
            )
        if n_anti:
            parts.append(f"{n_anti} anti-correlated structural features")
        if n_toxic:
            top_toxic = ", ".join(t["pattern"] for t in result["toxic_bigrams"][:3])
            parts.append(f"{n_toxic} toxic op-pair patterns ({top_toxic})")
        if n_ref:
            parts.append(f"{n_ref} refuted hypotheses")
        result["summary"] = (
            "; ".join(parts) if parts else "No negative results to report yet."
        )

        return result

    def decision_outcome_analysis(self, lookback: int = 30) -> Dict:
        """Analyze which selection decisions led to successful vs failed experiments.

        Joins mode_selection decisions with subsequent experiment outcomes to
        compute per-mode success rates.  Returns a dict with per-mode stats and
        a ``mode_penalties`` dict mapping mode names to penalty multipliers
        (< 1.0 for consistently failing modes).
        """
        result: Dict = {
            "mode_stats": {},
            "mode_penalties": {},
            "total_decisions": 0,
            "analysis_window": lookback,
        }

        try:
            # Get recent mode_selection decisions with their chosen mode
            rows = self.nb.conn.execute(
                """SELECT decision_id, timestamp, chosen_experiments_json
                   FROM selection_decisions
                   WHERE context = 'mode_selection'
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (lookback,),
            ).fetchall()
        except Exception:
            return result

        if not rows:
            return result

        # For each decision, find the experiment that started shortly after
        # and check its outcome
        mode_outcomes: Dict[str, Dict] = {}  # mode -> {total, s1_any, s1_total}
        for row in rows:
            chosen_json = row["chosen_experiments_json"]
            if not chosen_json:
                continue
            try:
                chosen = (
                    json.loads(chosen_json)
                    if isinstance(chosen_json, str)
                    else chosen_json
                )
            except (json.JSONDecodeError, TypeError):
                continue
            if not chosen:
                continue
            mode = (
                chosen[0].get("mode", "synthesis")
                if isinstance(chosen[0], dict)
                else "synthesis"
            )
            decision_ts = row["timestamp"]

            # Find the next completed experiment after this decision
            exp_row = self.nb.conn.execute(
                """SELECT experiment_type, n_stage1_passed, n_programs_generated,
                          best_loss_ratio, best_novelty_score
                   FROM experiments
                   WHERE timestamp >= ? AND status = 'completed'
                   ORDER BY timestamp ASC
                   LIMIT 1""",
                (decision_ts,),
            ).fetchone()

            if exp_row is None:
                continue

            if mode not in mode_outcomes:
                mode_outcomes[mode] = {
                    "total": 0,
                    "s1_any": 0,
                    "s1_total": 0,
                    "programs_total": 0,
                }
            stats = mode_outcomes[mode]
            stats["total"] += 1
            s1 = exp_row["n_stage1_passed"] or 0
            stats["s1_total"] += s1
            stats["programs_total"] += exp_row["n_programs_generated"] or 0
            if s1 > 0:
                stats["s1_any"] += 1

        result["total_decisions"] = sum(s["total"] for s in mode_outcomes.values())

        # Compute per-mode statistics and penalties
        overall_success_rate = 0.0
        total_with_s1 = sum(s["s1_any"] for s in mode_outcomes.values())
        total_decisions = result["total_decisions"]
        if total_decisions > 0:
            overall_success_rate = total_with_s1 / total_decisions

        for mode, stats in mode_outcomes.items():
            n = stats["total"]
            success_rate = stats["s1_any"] / n if n > 0 else 0
            s1_per_program = (
                stats["s1_total"] / stats["programs_total"]
                if stats["programs_total"] > 0
                else 0
            )
            result["mode_stats"][mode] = {
                "n_decisions": n,
                "success_rate": round(success_rate, 3),
                "s1_per_program": round(s1_per_program, 4),
                "s1_total": stats["s1_total"],
                "consecutive_failures": 0,  # filled below
            }

            # Count consecutive recent failures for this mode
            consec = 0
            for row2 in rows:
                chosen2 = row2["chosen_experiments_json"]
                try:
                    c2 = json.loads(chosen2) if isinstance(chosen2, str) else chosen2
                except (json.JSONDecodeError, TypeError):
                    continue
                if not c2 or not isinstance(c2[0], dict):
                    continue
                if c2[0].get("mode") != mode:
                    continue
                exp2 = self.nb.conn.execute(
                    """SELECT n_stage1_passed FROM experiments
                       WHERE timestamp >= ? AND status = 'completed'
                       ORDER BY timestamp ASC LIMIT 1""",
                    (row2["timestamp"],),
                ).fetchone()
                if exp2 and (exp2["n_stage1_passed"] or 0) == 0:
                    consec += 1
                else:
                    break
            result["mode_stats"][mode]["consecutive_failures"] = consec

            # Penalty: reduce weight for modes that consistently fail.
            # Minimum 5 decisions before penalizing to avoid noise.
            # Penalty scales from 1.0 (at or above average) to 0.3 (at 0% success).
            if n >= 5 and overall_success_rate > 0:
                relative = success_rate / max(overall_success_rate, 0.01)
                penalty = round(max(0.3, min(1.0, relative)), 2)
            else:
                penalty = 1.0
            # Extra penalty for recent consecutive failures (3+ in a row)
            if consec >= 3:
                penalty = round(max(0.3, penalty * 0.7), 2)
            result["mode_penalties"][mode] = penalty

        return result
