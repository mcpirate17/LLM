from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
import numpy as np


class _RoutingMixin:
    """Routing health, gating diagnostics, MoE telemetry."""
    __slots__ = ()

    @staticmethod
    def _explain_routing_health(by_mode: List[Dict], total_programs: int,
                                overall_stage1_pass_rate: float) -> str:
        """Generate deterministic plain-language routing interpretation."""
        if not by_mode:
            return (
                "No routing telemetry available yet. Run routed architectures "
                "to estimate drop rate, utilization balance, and confidence."
            )

        def weighted_mean(key: str) -> Optional[float]:
            weighted_sum = 0.0
            weighted_n = 0
            for row in by_mode:
                value = row.get(key)
                n_programs = int(row.get("n_programs") or 0)
                if value is None or n_programs <= 0:
                    continue
                weighted_sum += float(value) * n_programs
                weighted_n += n_programs
            if weighted_n == 0:
                return None
            return weighted_sum / weighted_n

        avg_drop = weighted_mean("avg_drop_rate")
        avg_entropy = weighted_mean("avg_utilization_entropy")
        avg_confidence = weighted_mean("avg_confidence_mean")

        if avg_drop is None:
            drop_desc = "unknown"
            drop_text = "drop rate is unavailable"
        elif avg_drop <= 0.05:
            drop_desc = "low"
            drop_text = f"drop rate is low ({avg_drop * 100:.1f}%)"
        elif avg_drop <= 0.15:
            drop_desc = "moderate"
            drop_text = f"drop rate is moderate ({avg_drop * 100:.1f}%)"
        else:
            drop_desc = "high"
            drop_text = f"drop rate is high ({avg_drop * 100:.1f}%)"

        if avg_entropy is None:
            entropy_text = "utilization balance is unavailable"
        elif avg_entropy >= 1.2:
            entropy_text = f"utilization appears balanced (entropy {avg_entropy:.2f})"
        elif avg_entropy >= 0.8:
            entropy_text = f"utilization is moderately balanced (entropy {avg_entropy:.2f})"
        else:
            entropy_text = f"utilization looks concentrated (entropy {avg_entropy:.2f})"

        if avg_confidence is None:
            confidence_desc = "unknown"
            confidence_text = "confidence is unavailable"
        elif avg_confidence >= 0.7:
            confidence_desc = "strong"
            confidence_text = f"confidence is strong ({avg_confidence:.2f})"
        elif avg_confidence >= 0.5:
            confidence_desc = "moderate"
            confidence_text = f"confidence is moderate ({avg_confidence:.2f})"
        else:
            confidence_desc = "weak"
            confidence_text = f"confidence is weak ({avg_confidence:.2f})"

        best_mode = max(by_mode, key=lambda r: float(r.get("stage1_pass_rate") or 0.0))
        sentence_1 = (
            f"Across {total_programs} routed programs, overall Stage 1 pass rate is "
            f"{overall_stage1_pass_rate * 100:.1f}% and {drop_text}."
        )
        sentence_2 = (
            f"Routing {entropy_text}, while {confidence_text}. "
            f"Best-performing mode is '{best_mode.get('routing_mode', 'unknown')}' "
            f"at {float(best_mode.get('stage1_pass_rate') or 0.0) * 100:.1f}% S1 pass."
        )

        if drop_desc == "high":
            sentence_3 = "High drop suggests routing-capacity pressure; consider reducing overflow and token skipping."
        elif confidence_desc == "weak":
            sentence_3 = "Low confidence suggests uncertain routing choices; prioritize modes with steadier confidence and lower variance."
        elif avg_entropy is not None and avg_entropy < 0.8:
            sentence_3 = "Low entropy suggests expert over-concentration; improve utilization balance to reduce mode-collapse risk."
        else:
            sentence_3 = "Telemetry suggests routing is reasonably healthy and balanced across current modes."

        return f"{sentence_1} {sentence_2} {sentence_3}"

    @staticmethod
    def _routing_sample_size_label(n_programs: int) -> str:
        if n_programs >= 80:
            return "high"
        if n_programs >= 30:
            return "medium"
        return "low"

    @staticmethod
    def _routing_confidence_label(avg_confidence_mean: Optional[float], avg_confidence_std: Optional[float]) -> str:
        if avg_confidence_mean is None:
            return "unknown"
        adjusted = avg_confidence_mean - 0.5 * (avg_confidence_std or 0.0)
        if adjusted >= 0.7:
            return "high"
        if adjusted >= 0.5:
            return "medium"
        return "low"

    @staticmethod
    def _routing_stability_label(avg_confidence_std: Optional[float]) -> str:
        if avg_confidence_std is None:
            return "unknown"
        if avg_confidence_std <= 0.08:
            return "stable"
        if avg_confidence_std <= 0.16:
            return "moderate"
        return "volatile"

    @staticmethod
    def _routing_efficiency_label(token_retention: Optional[float]) -> str:
        if token_retention is None:
            return "unknown"
        if token_retention >= 0.9:
            return "high"
        if token_retention >= 0.75:
            return "medium"
        return "low"

    def routing_mode_comparison(self) -> Dict:
        """Compare routing modes with sample-size and confidence labels."""
        rows = self.nb.conn.execute("""
            SELECT
                COALESCE(NULLIF(routing_mode, ''), 'uniform') as routing_mode,
                COUNT(*) as n_programs,
                SUM(CASE WHEN stage1_passed = 1 THEN 1 ELSE 0 END) as n_stage1_passed,
                AVG(loss_ratio) as avg_loss_ratio,
                AVG(routing_tokens_total) as avg_tokens_total,
                AVG(routing_tokens_processed) as avg_tokens_processed,
                AVG(routing_tokens_skipped) as avg_tokens_skipped,
                AVG(routing_drop_rate) as avg_drop_rate,
                AVG(routing_utilization_entropy) as avg_utilization_entropy,
                AVG(routing_capacity_overflow_count) as avg_capacity_overflow_count,
                AVG(routing_confidence_mean) as avg_confidence_mean,
                AVG(routing_confidence_std) as avg_confidence_std
            FROM program_results
            GROUP BY COALESCE(NULLIF(routing_mode, ''), 'uniform')
            ORDER BY n_programs DESC
        """).fetchall()

        if not rows:
            return {
                "available": False,
                "n_modes": 0,
                "total_programs": 0,
                "routed_programs": 0,
                "uniform_programs": 0,
                "by_mode": [],
                "explanation": (
                    "No routing telemetry available yet. Run routed architectures "
                    "to compare routing modes and confidence stability."
                ),
            }

        by_mode = []
        total_programs = 0
        total_stage1 = 0
        routed_programs = 0
        uniform_programs = 0

        for row in rows:
            n_programs = row["n_programs"] or 0
            n_stage1 = row["n_stage1_passed"] or 0
            total_programs += n_programs
            total_stage1 += n_stage1
            mode = row["routing_mode"] or "uniform"

            if mode == "uniform":
                uniform_programs += n_programs
            else:
                routed_programs += n_programs

            stage1_pass_rate = n_stage1 / max(n_programs, 1)
            avg_drop_rate = row["avg_drop_rate"]
            avg_tokens_total = row["avg_tokens_total"]
            avg_tokens_processed = row["avg_tokens_processed"]
            token_retention = None
            if avg_tokens_total and avg_tokens_total > 0 and avg_tokens_processed is not None:
                token_retention = avg_tokens_processed / avg_tokens_total
            elif avg_drop_rate is not None:
                token_retention = max(0.0, min(1.0, 1.0 - avg_drop_rate))

            avg_conf_mean = row["avg_confidence_mean"]
            avg_conf_std = row["avg_confidence_std"]

            by_mode.append({
                "routing_mode": mode,
                "n_programs": n_programs,
                "stage1_pass_rate": stage1_pass_rate,
                "avg_loss_ratio": row["avg_loss_ratio"],
                "avg_tokens_total": avg_tokens_total,
                "avg_tokens_processed": avg_tokens_processed,
                "avg_tokens_skipped": row["avg_tokens_skipped"],
                "avg_drop_rate": avg_drop_rate,
                "avg_utilization_entropy": row["avg_utilization_entropy"],
                "avg_capacity_overflow_count": row["avg_capacity_overflow_count"],
                "avg_confidence_mean": avg_conf_mean,
                "avg_confidence_std": avg_conf_std,
                "token_retention": token_retention,
                "sample_size_label": self._routing_sample_size_label(n_programs),
                "confidence_label": self._routing_confidence_label(avg_conf_mean, avg_conf_std),
                "stability_label": self._routing_stability_label(avg_conf_std),
                "efficiency_label": self._routing_efficiency_label(token_retention),
            })

        overall_stage1_pass_rate = total_stage1 / max(total_programs, 1)

        return {
            "available": True,
            "n_modes": len(by_mode),
            "total_programs": total_programs,
            "routed_programs": routed_programs,
            "uniform_programs": uniform_programs,
            "overall_stage1_pass_rate": overall_stage1_pass_rate,
            "by_mode": by_mode,
            "explanation": self._explain_routing_health(
                by_mode=by_mode,
                total_programs=total_programs,
                overall_stage1_pass_rate=overall_stage1_pass_rate,
            ),
        }

    def routing_health(self) -> Dict:
        """Aggregate routing telemetry by routing mode.

        Returns structured defaults when routing telemetry is not yet available,
        so dashboard/API consumers can safely render partial states.
        """
        comparison = self.routing_mode_comparison()
        if not comparison.get("available"):
            return {
                "available": False,
                "n_modes": 0,
                "total_programs": 0,
                "by_mode": [],
                "explanation": comparison.get("explanation", "Routing telemetry is unavailable."),
            }
        return {
            "available": True,
            "n_modes": comparison.get("n_modes", 0),
            "total_programs": comparison.get("total_programs", 0),
            "overall_stage1_pass_rate": comparison.get("overall_stage1_pass_rate", 0.0),
            "by_mode": comparison.get("by_mode", []),
            "explanation": comparison.get("explanation", ""),
        }

    @staticmethod
    def _routing_collapse_risk_label(avg_entropy: Optional[float]) -> str:
        if avg_entropy is None:
            return "unknown"
        if avg_entropy >= 1.2:
            return "low"
        if avg_entropy >= 0.8:
            return "medium"
        return "high"

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> Optional[float]:
        if not values:
            return None
        return float(np.percentile(values, percentile * 100))

    def gating_behavior_diagnostics(self) -> Dict:
        """Canonical diagnostics for gated/recursive routing behavior."""
        rows = self.nb.conn.execute("""
            SELECT
                COALESCE(NULLIF(routing_mode, ''), 'uniform') AS routing_mode,
                routing_tokens_total,
                routing_tokens_processed,
                routing_drop_rate,
                routing_utilization_entropy,
                routing_capacity_overflow_count
            FROM program_results
            WHERE routing_mode IS NOT NULL
               OR routing_drop_rate IS NOT NULL
               OR routing_utilization_entropy IS NOT NULL
               OR routing_tokens_total IS NOT NULL
               OR routing_tokens_processed IS NOT NULL
        """).fetchall()

        if not rows:
            return {
                "available": False,
                "total_routed_programs": 0,
                "avg_gate_entropy": None,
                "collapse_risk_counts": {"low": 0, "medium": 0, "high": 0, "unknown": 0},
                "by_mode": [],
                "token_retention_curve_overall": [],
                "explanation": (
                    "No gating telemetry available yet. Run routed/recursive candidates "
                    "to estimate entropy, collapse risk, and token retention behavior."
                ),
            }

        by_mode_raw: Dict[str, Dict[str, Any]] = {}
        total_entropy = 0.0
        entropy_count = 0
        overall_retentions: List[float] = []
        collapse_counts = {"low": 0, "medium": 0, "high": 0, "unknown": 0}

        for row in rows:
            mode = row["routing_mode"] or "uniform"
            bucket = by_mode_raw.setdefault(mode, {
                "n_programs": 0,
                "entropies": [],
                "retentions": [],
                "drop_rates": [],
                "overflows": [],
            })
            bucket["n_programs"] += 1

            entropy = row["routing_utilization_entropy"]
            if entropy is not None:
                entropy = float(entropy)
                bucket["entropies"].append(entropy)
                total_entropy += entropy
                entropy_count += 1

            drop_rate = row["routing_drop_rate"]
            if drop_rate is not None:
                bucket["drop_rates"].append(float(drop_rate))

            overflow = row["routing_capacity_overflow_count"]
            if overflow is not None:
                bucket["overflows"].append(float(overflow))

            retention = None
            total_tokens = row["routing_tokens_total"]
            processed_tokens = row["routing_tokens_processed"]
            if total_tokens is not None and processed_tokens is not None and float(total_tokens) > 0:
                retention = float(processed_tokens) / float(total_tokens)
            elif drop_rate is not None:
                retention = max(0.0, min(1.0, 1.0 - float(drop_rate)))

            if retention is not None:
                retention = max(0.0, min(1.0, retention))
                bucket["retentions"].append(retention)
                overall_retentions.append(retention)

        by_mode: List[Dict[str, Any]] = []
        for mode, bucket in sorted(by_mode_raw.items(), key=lambda item: item[1]["n_programs"], reverse=True):
            entropies = bucket["entropies"]
            retentions = bucket["retentions"]
            avg_entropy = (sum(entropies) / len(entropies)) if entropies else None
            collapse_label = self._routing_collapse_risk_label(avg_entropy)
            collapse_counts[collapse_label] += 1

            avg_retention = (sum(retentions) / len(retentions)) if retentions else None
            token_curve = []
            for quantile, name in ((0.25, "p25"), (0.5, "p50"), (0.75, "p75")):
                value = self._percentile(retentions, quantile)
                if value is not None:
                    token_curve.append({"quantile": name, "retention": value})

            avg_drop = (sum(bucket["drop_rates"]) / len(bucket["drop_rates"])) if bucket["drop_rates"] else None
            avg_overflow = (sum(bucket["overflows"]) / len(bucket["overflows"])) if bucket["overflows"] else None

            by_mode.append({
                "routing_mode": mode,
                "n_programs": bucket["n_programs"],
                "avg_gate_entropy": avg_entropy,
                "collapse_risk_label": collapse_label,
                "avg_token_retention": avg_retention,
                "token_retention_curve": token_curve,
                "avg_drop_rate": avg_drop,
                "avg_capacity_overflow_count": avg_overflow,
            })

        overall_curve = []
        for quantile, name in ((0.25, "p25"), (0.5, "p50"), (0.75, "p75")):
            value = self._percentile(overall_retentions, quantile)
            if value is not None:
                overall_curve.append({"quantile": name, "retention": value})

        avg_gate_entropy = (total_entropy / entropy_count) if entropy_count > 0 else None
        explanation = (
            f"Gating diagnostics over {len(rows)} routed candidates: "
            f"avg gate entropy is {avg_gate_entropy:.2f}. "
            f"Collapse-risk modes (high) = {collapse_counts['high']}."
            if avg_gate_entropy is not None
            else "Gating diagnostics collected, but gate entropy is not yet available."
        )

        return {
            "available": True,
            "total_routed_programs": len(rows),
            "avg_gate_entropy": avg_gate_entropy,
            "collapse_risk_counts": collapse_counts,
            "by_mode": by_mode,
            "token_retention_curve_overall": overall_curve,
            "explanation": explanation,
        }

    def moe_activation_telemetry(self) -> Dict:
        """Aggregate MoE expert utilization and routing quality from experiment history.

        Analyzes programs with routing telemetry to compute:
        - Per-expert utilization distribution
        - Load balance score (entropy-based)
        - Routing quality correlation with loss
        - Capacity overflow trends
        """
        rows = self.nb.conn.execute("""
            SELECT routing_mode, routing_tokens_total, routing_tokens_processed,
                   routing_tokens_skipped, routing_drop_rate,
                   routing_utilization_entropy, routing_capacity_overflow_count,
                   routing_confidence_mean, routing_confidence_std,
                   routing_expert_utilization_json,
                   loss_ratio, stage1_passed, graph_json
            FROM program_results
            WHERE routing_mode IS NOT NULL
        """).fetchall()

        if not rows:
            return {"n_programs": 0, "experts": [], "summary": {}}

        n_programs = len(rows)
        sum_entropy = 0.0
        n_entropy = 0
        sum_drop_rate = 0.0
        n_drop = 0
        sum_overflow = 0
        n_overflow = 0
        sum_confidence = 0.0
        n_confidence = 0
        n_survived = 0
        sum_loss = 0.0
        n_loss = 0
        best_loss = None

        # Per-expert aggregation
        expert_totals: Dict[int, Dict] = {}

        for row in rows:
            record = dict(row)

            if record.get("stage1_passed"):
                n_survived += 1

            loss = self._as_float(record.get("loss_ratio"))
            if loss is not None:
                sum_loss += loss
                n_loss += 1
                if best_loss is None or loss < best_loss:
                    best_loss = loss

            entropy = self._as_float(record.get("routing_utilization_entropy"))
            if entropy is not None:
                sum_entropy += entropy
                n_entropy += 1

            drop_rate = self._as_float(record.get("routing_drop_rate"))
            if drop_rate is not None:
                sum_drop_rate += drop_rate
                n_drop += 1

            overflow = record.get("routing_capacity_overflow_count")
            if overflow is not None:
                sum_overflow += int(overflow)
                n_overflow += 1

            conf = self._as_float(record.get("routing_confidence_mean"))
            if conf is not None:
                sum_confidence += conf
                n_confidence += 1

            # Parse per-expert utilization
            util_json = record.get("routing_expert_utilization_json")
            if util_json:
                try:
                    utilizations = json.loads(util_json)
                    if isinstance(utilizations, list):
                        for eidx, util_val in enumerate(utilizations):
                            bucket = expert_totals.setdefault(eidx, {
                                "expert_id": eidx,
                                "sum_utilization": 0.0,
                                "n_samples": 0,
                                "max_utilization": 0.0,
                                "min_utilization": 1.0,
                            })
                            u = float(util_val)
                            bucket["sum_utilization"] += u
                            bucket["n_samples"] += 1
                            bucket["max_utilization"] = max(bucket["max_utilization"], u)
                            bucket["min_utilization"] = min(bucket["min_utilization"], u)
                except (json.JSONDecodeError, TypeError):
                    pass

        # Build per-expert summary
        experts = []
        for eidx in sorted(expert_totals.keys()):
            b = expert_totals[eidx]
            n = b["n_samples"]
            avg = b["sum_utilization"] / n if n > 0 else 0.0
            experts.append({
                "expert_id": eidx,
                "avg_utilization": round(avg, 4),
                "max_utilization": round(b["max_utilization"], 4),
                "min_utilization": round(b["min_utilization"], 4),
                "n_samples": n,
            })

        # Compute load balance score from average utilizations
        # Perfect balance = all experts have equal utilization
        # Score = 1 - coefficient of variation (capped at 1.0)
        load_balance_score = None
        if experts:
            utils = [e["avg_utilization"] for e in experts]
            mean_u = sum(utils) / len(utils) if utils else 0
            if mean_u > 0 and len(utils) > 1:
                var = sum((u - mean_u) ** 2 for u in utils) / len(utils)
                cv = (var ** 0.5) / mean_u
                load_balance_score = round(max(0.0, min(1.0, 1.0 - cv)), 4)

        summary = {
            "n_programs": n_programs,
            "n_survived": n_survived,
            "survival_rate": round(n_survived / n_programs, 4) if n_programs > 0 else 0.0,
            "avg_loss_ratio": round(sum_loss / n_loss, 4) if n_loss > 0 else None,
            "best_loss_ratio": round(best_loss, 4) if best_loss is not None else None,
            "avg_utilization_entropy": round(sum_entropy / n_entropy, 4) if n_entropy > 0 else None,
            "avg_drop_rate": round(sum_drop_rate / n_drop, 4) if n_drop > 0 else None,
            "avg_capacity_overflow": round(sum_overflow / n_overflow, 2) if n_overflow > 0 else None,
            "avg_routing_confidence": round(sum_confidence / n_confidence, 4) if n_confidence > 0 else None,
            "load_balance_score": load_balance_score,
            "n_experts_observed": len(experts),
        }

        return {
            "n_programs": n_programs,
            "experts": experts,
            "summary": summary,
        }

    def sparse_quant_codesign_summary(self) -> Dict:
        """Aggregate quality-retention-per-byte across programs with sparse+quant data.

        Looks for programs that have both sparsity metrics (sparse_density_*)
        and pruning quality retention data, then estimates combined
        sparse+quant efficiency potential.

        Returns summary with per-program and aggregate statistics.
        """
        rows = self.nb.conn.execute(
            "SELECT result_id, graph_json, stage1_passed, final_loss, "
            "sparse_density_mean, sparse_density_last, "
            "pruning_method, pruning_target_sparsity, pruning_actual_sparsity, "
            "pruning_quality_retention, pruning_dense_eval_loss, "
            "pruning_pruned_eval_loss, pruning_n_params_total "
            "FROM program_results "
            "WHERE (sparse_density_mean IS NOT NULL OR pruning_method IS NOT NULL)"
        ).fetchall()
        if not rows:
            return {"n_programs": 0, "programs": [], "summary": {}}

        programs = []
        sum_retention = 0.0
        sum_compression = 0.0
        n_with_retention = 0
        best_qpb = None  # quality-per-byte
        best_qpb_id = None

        for row in rows:
            pid = row[0]
            graph_json = row[1]
            survived = row[2]
            best_loss = row[3]
            density_mean = row[4]
            density_last = row[5]
            prune_method = row[6]
            prune_target = row[7]
            prune_actual = row[8]
            prune_retention = row[9]
            dense_loss = row[10]
            pruned_loss = row[11]
            n_params = row[12]

            # Detect compression ops in graph
            record = {"graph_json": graph_json or ""}
            compression_ops = self._detect_compression_ops(record)

            # Estimate effective density
            effective_density = density_last or density_mean or 1.0
            if prune_actual is not None and prune_actual > 0:
                effective_density = min(effective_density, 1.0 - prune_actual)

            # Estimate bytes per param under INT8 quant + sparsity
            bytes_sparse_quant_int8 = effective_density * 1.0  # 8-bit = 1 byte
            bytes_sparse_quant_int4 = effective_density * 0.5  # 4-bit = 0.5 byte
            bytes_original = 4.0  # float32

            compression_int8 = bytes_original / max(bytes_sparse_quant_int8, 0.01)
            compression_int4 = bytes_original / max(bytes_sparse_quant_int4, 0.01)

            # Quality retention
            retention = None
            if prune_retention is not None:
                retention = float(prune_retention)
            elif dense_loss is not None and pruned_loss is not None and pruned_loss > 0:
                retention = float(dense_loss) / float(pruned_loss)

            # Quality-per-byte (higher = better)
            qpb_int8 = None
            if retention is not None:
                qpb_int8 = retention * compression_int8
                sum_retention += retention
                sum_compression += compression_int8
                n_with_retention += 1
                if best_qpb is None or qpb_int8 > best_qpb:
                    best_qpb = qpb_int8
                    best_qpb_id = pid

            entry = {
                "program_id": pid,
                "survived": bool(survived),
                "best_loss": round(float(best_loss), 6) if best_loss else None,
                "effective_density": round(float(effective_density), 4),
                "compression_ops": compression_ops,
                "pruning_method": prune_method,
                "quality_retention": round(retention, 4) if retention else None,
                "compression_ratio_int8": round(compression_int8, 2),
                "compression_ratio_int4": round(compression_int4, 2),
                "quality_per_byte_int8": round(qpb_int8, 4) if qpb_int8 else None,
                "n_params": n_params,
            }
            programs.append(entry)

        summary = {
            "n_programs": len(programs),
            "n_with_retention": n_with_retention,
            "avg_quality_retention": (
                round(sum_retention / n_with_retention, 4) if n_with_retention > 0 else None
            ),
            "avg_compression_ratio_int8": (
                round(sum_compression / n_with_retention, 2) if n_with_retention > 0 else None
            ),
            "best_quality_per_byte": round(best_qpb, 4) if best_qpb else None,
            "best_qpb_program_id": best_qpb_id,
        }

        return {
            "n_programs": len(programs),
            "programs": programs,
            "summary": summary,
        }

