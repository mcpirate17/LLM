"""Op, template, motif, and synergy weight computation mixin."""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class _WeightsMixin:
    """Compute contrast-amplified weights for ops, templates, and motifs."""

    __slots__ = ()

    def reproducibility_packet_status(self, program: Dict) -> Dict:
        """Evaluate reproducibility packet completeness for a program."""
        arch_choices = self._extract_arch_choices(program.get("arch_spec_json"))
        checks = [
            ("result_id", bool(program.get("result_id"))),
            ("graph_fingerprint", bool(program.get("graph_fingerprint"))),
            ("arch_spec", bool(arch_choices)),
            (
                "baseline_ratio",
                program.get("validation_baseline_ratio") is not None
                or program.get("baseline_loss_ratio") is not None,
            ),
            (
                "multi_seed_std",
                program.get("validation_multi_seed_std") is not None,
            ),
            ("cka_artifact", program.get("cka_source") == "artifact"),
        ]
        ready_count = sum(1 for _, ok in checks if ok)
        total_checks = len(checks)
        if ready_count == total_checks:
            status = "ready"
        elif ready_count >= 4:
            status = "partial"
        else:
            status = "sparse"
        return {
            "status": status,
            "ready_count": ready_count,
            "total_checks": total_checks,
            "missing": [name for name, ok in checks if not ok],
        }

    def op_success_rates(self, since_ts: float = 0.0) -> Dict[str, Dict]:
        """Get per-op success rates.

        Args:
            since_ts: If > 0, compute rates from program_results within the
                time window (windowed view) instead of the accumulated table.
                This breaks the death spiral where fixed ops remain poisoned
                by stale lifetime data.
        """
        if since_ts > 0:
            rows = self.nb.get_op_success_rates_windowed(since_ts)
        else:
            rows = self.nb.get_op_success_rates()
        result = {}
        for row in rows:
            op = row["op_name"]
            n_used = row["n_used"] or 1
            n_s0 = row.get("n_stage0_passed") or 0

            # S1 success rate should be relative to things that actually
            # passed compilation. If it didn't compile, it's a code issue,
            # not a failure of the architecture's scientific utility.
            s1_rate = (row.get("n_stage1_passed") or 0) / n_s0 if n_s0 > 0 else 0.0

            result[op] = {
                "n_used": n_used,
                "n_s0": n_s0,
                "s0_rate": n_s0 / n_used,
                "s05_rate": (row.get("n_stage05_passed") or 0) / n_used,
                "s1_rate": s1_rate,
                "avg_loss_ratio": row.get("avg_loss_ratio"),
                "avg_novelty": row.get("avg_novelty"),
                "avg_novelty_confidence": row.get("avg_novelty_confidence"),
            }
        return result

    def compute_op_weights(
        self, since_ts: float = 0.0, min_used: int = 5
    ) -> Dict[str, float]:
        """Per-op weights via contrast amplification: (s1_rate/mean)^2, clamped [0.1, 8.0].

        Structural ops (no learnable params) are excluded from the mean
        calculation and get weight 1.0 — they should not be penalized or
        rewarded based on S1 attribution since they are scaffolding.
        """
        from research.synthesis.context_rules import S1_EXEMPT_OPS

        rates = self.op_success_rates(since_ts=since_ts)
        if not rates:
            return {}
        eligible = {
            op: info
            for op, info in rates.items()
            if info["n_used"] >= min_used and op not in S1_EXEMPT_OPS
        }
        if not eligible:
            return {}
        mean_s1 = sum(info["s1_rate"] for info in eligible.values()) / len(eligible)
        if mean_s1 < 1e-6:
            return {}
        weights: Dict[str, float] = {}
        for op, info in eligible.items():
            relative = info["s1_rate"] / mean_s1
            amplified = relative**2
            weights[op] = round(max(0.1, min(8.0, amplified)), 3)
        return weights

    def under_observed_ops(self, threshold: int = 20) -> Dict[str, int]:
        """Return ops with fewer than threshold observations.

        Returns dict of op_name -> n_used. Also includes ops in
        PRIMITIVE_REGISTRY but not tracked in op_success_rates (count=0).
        """
        from research.synthesis.primitives import PRIMITIVE_REGISTRY

        rates = self.op_success_rates()
        result = {}
        for op, info in rates.items():
            if info["n_used"] < threshold:
                result[op] = info["n_used"]

        # Ops in registry but not tracked at all
        tracked = set(rates.keys())
        for name in PRIMITIVE_REGISTRY:
            if name not in tracked and name not in ("input", "output"):
                result[name] = 0

        return result

    def _compute_metadata_weights(
        self,
        metadata_key: str,
        since_ts: float,
        min_used: int,
    ) -> Dict[str, float]:
        """Compute contrast-amplified weights from graph metadata lists.

        Extracts ``metadata_key`` (e.g. ``templates_used``, ``motifs_used``)
        from ``graph_json.metadata`` and computes per-item S1 success rates.
        Returns ``{item_name: weight}`` clamped to ``[0.1, 8.0]``.
        """
        cursor = self.nb.conn.execute(
            "SELECT graph_json, stage1_passed FROM program_results "
            "WHERE stage0_passed = 1 AND timestamp >= ? "
            "AND graph_json IS NOT NULL AND graph_json != '{}'",
            (since_ts,),
        )
        counts: Dict[str, int] = defaultdict(int)
        s1_counts: Dict[str, int] = defaultdict(int)
        for row in cursor:
            try:
                meta = json.loads(row[0]).get("metadata", {})
            except (json.JSONDecodeError, TypeError):
                continue
            items = meta.get(metadata_key)
            if not isinstance(items, list):
                continue
            passed = bool(row[1])
            for item in items:
                if not isinstance(item, str):
                    continue
                counts[item] += 1
                if passed:
                    s1_counts[item] += 1
        stats = {
            name: {"n_used": n, "s1_rate": s1_counts.get(name, 0) / n}
            for name, n in counts.items()
            if n >= min_used
        }
        if not stats:
            return {}
        mean_s1 = sum(s["s1_rate"] for s in stats.values()) / len(stats)
        if mean_s1 < 1e-6:
            return {}
        weights: Dict[str, float] = {}
        for name, s in stats.items():
            relative = s["s1_rate"] / mean_s1
            # Moderate contrast: relative^1.5 (not ^2) to avoid collapsing
            # low-performers too aggressively — they still need search coverage.
            amplified = relative**1.5
            # Confidence discount: shrink toward 1.0 for small sample sizes.
            # At n=min_used the weight is 50% amplified + 50% neutral (1.0).
            # At n=30+ the weight is fully amplified.
            confidence = min(1.0, s["n_used"] / 30.0)
            blended = confidence * amplified + (1.0 - confidence) * 1.0
            weights[name] = round(max(0.3, min(5.0, blended)), 3)
        return weights

    def compute_template_weights(
        self, since_ts: float = 0.0, min_used: int = 3
    ) -> Dict[str, float]:
        """Per-template weights from S1 success rates via contrast amplification."""
        return self._compute_metadata_weights("templates_used", since_ts, min_used)

    def compute_motif_weights(
        self, since_ts: float = 0.0, min_used: int = 3
    ) -> Dict[str, float]:
        """Per-motif weights from S1 success rates via contrast amplification."""
        return self._compute_metadata_weights("motifs_used", since_ts, min_used)

    def compute_template_and_motif_weights(
        self, since_ts: float = 0.0, min_used: int = 3
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute template and motif weights in a single DB query pass."""
        cursor = self.nb.conn.execute(
            "SELECT graph_json, stage1_passed FROM program_results "
            "WHERE stage0_passed = 1 AND timestamp >= ? "
            "AND graph_json IS NOT NULL AND graph_json != '{}'",
            (since_ts,),
        )

        # Single-pass: accumulate counts for both keys simultaneously
        all_counts: Dict[str, Dict[str, int]] = {
            "templates_used": defaultdict(int),
            "motifs_used": defaultdict(int),
        }
        all_s1: Dict[str, Dict[str, int]] = {
            "templates_used": defaultdict(int),
            "motifs_used": defaultdict(int),
        }
        for row in cursor:
            try:
                meta = json.loads(row[0]).get("metadata", {})
            except (json.JSONDecodeError, TypeError):
                continue
            passed = bool(row[1])
            for mk in ("templates_used", "motifs_used"):
                items = meta.get(mk)
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, str):
                        continue
                    all_counts[mk][item] += 1
                    if passed:
                        all_s1[mk][item] += 1

        results: Dict[str, Dict[str, float]] = {}
        for mk in ("templates_used", "motifs_used"):
            counts = all_counts[mk]
            s1_counts = all_s1[mk]
            stats = {
                name: {"n_used": n, "s1_rate": s1_counts.get(name, 0) / n}
                for name, n in counts.items()
                if n >= min_used
            }
            if not stats:
                results[mk] = {}
                continue
            mean_s1 = sum(s["s1_rate"] for s in stats.values()) / len(stats)
            if mean_s1 < 1e-6:
                results[mk] = {}
                continue
            weights: Dict[str, float] = {}
            for name, s in stats.items():
                relative = s["s1_rate"] / mean_s1
                amplified = relative**1.5
                confidence = min(1.0, s["n_used"] / 30.0)
                blended = confidence * amplified + (1.0 - confidence) * 1.0
                weights[name] = round(max(0.3, min(5.0, blended)), 3)
            results[mk] = weights

        return results.get("templates_used", {}), results.get("motifs_used", {})

    def compute_synergy_boosts(
        self,
        min_lift: float = 1.5,
        min_co_occurrences: int = 5,
        boost_cap: float = 3.0,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Boost motif/template weights for ops that are synergistic in S1 survivors.

        For each synergistic pair (A, B) with lift > min_lift:
          - Find motifs containing A -> boost by sqrt(lift)
          - Find motifs containing B -> boost by sqrt(lift)
          - Find templates mapped to A or B -> boost by sqrt(lift)

        sqrt(lift) because both ops' motifs get boosted independently;
        the compound effect when both land in the same graph ~ lift.

        Returns (motif_boosts, template_boosts) -- multiplicative factors.
        """
        from research.scientist.intelligence.analyzer import analyze_op_synergies
        from research.synthesis.motifs import ALL_MOTIFS

        synergies = analyze_op_synergies(self.nb, min_co_occurrences=min_co_occurrences)
        if not synergies:
            return {}, {}

        # Build op -> motif index
        op_to_motifs: Dict[str, List[str]] = defaultdict(list)
        for motif in ALL_MOTIFS:
            for step in motif.steps:
                op_to_motifs[step.op_name].append(motif.name)

        # Use the known _OP_TO_TEMPLATE mapping -- it's a local dict inside
        # generate_layer_graph, so we reconstruct the subset we need.
        _OP_TO_TEMPLATE = {
            "lif_neuron": "spiking_moe_block",
            "sparse_threshold": "spiking_moe_block",
            "spike_rate_code": "spiking_moe_block",
            "split3": "three_way_split",
            "tropical_center": "tropical_center_block",
            "tropical_attention": "tropical_center_block",
            "state_space": "state_space_block",
            "conv_only": "conv_residual_block",
            "gated_delta": "recurrent_delta_block",
            "early_exit": "cascaded_early_exit",
            "n_way_sparse_router": "n_way_moe_block",
        }

        motif_boosts: Dict[str, float] = {}
        template_boosts: Dict[str, float] = {}

        for syn in synergies:
            if syn.label != "synergistic" or syn.lift < min_lift:
                continue
            boost = min(math.sqrt(syn.lift), boost_cap)

            for op in (syn.op_a, syn.op_b):
                # Boost motifs containing this op
                for motif_name in op_to_motifs.get(op, []):
                    motif_boosts[motif_name] = max(
                        motif_boosts.get(motif_name, 1.0), boost
                    )
                # Boost templates mapped to this op
                tpl = _OP_TO_TEMPLATE.get(op)
                if tpl:
                    template_boosts[tpl] = max(template_boosts.get(tpl, 1.0), boost)

        n_syn = sum(1 for s in synergies if s.label == "synergistic")
        if motif_boosts or template_boosts:
            logger.info(
                "Synergy boosts: %d synergistic pairs -> %d motif boosts, %d template boosts",
                n_syn,
                len(motif_boosts),
                len(template_boosts),
            )
        return motif_boosts, template_boosts
