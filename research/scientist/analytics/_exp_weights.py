"""Op, template, motif, and synergy weight computation mixin."""

from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)

# ── Capability-weighted breeding ────────────────────────────────────────
# The analytics weights drive what the grammar breeds toward. Historically a
# graph counted as a "success" iff it cleared the stage-1 (perplexity) gate.
# Foundation analysis 2026-05-24 showed that gate is non-discriminative at the
# screening budget: 96% of stage-1 passers have induction AUC < 0.1 (no genuine
# in-context capability) and position-independent models reach the same wikitext
# perplexity (~700) as real mixers. So s1-only breeding optimises noise.
#
# We instead credit each stage-1 passer by how much in-context capability it
# actually shows (induction screening AUC — the signal that *does* separate real
# sequence models from position-independent ones). Soft, not a hard gate: a
# baseline floor keeps perplexity-passers in play and avoids excluding SSMs that
# acquire capability slowly (see feedback_ssm_capability_floor). Self-correcting
# (Bayesian shrinkage downstream) and reversible via env flag.
_CAPABILITY_BREEDING = os.environ.get("ARIA_DISABLE_CAPABILITY_BREEDING") != "1"
# Gentle tilt, not a hammer. base=0.5 caps the max capability advantage at ~2x
# so families that acquire in-context skill more slowly than attention at the
# 500-step screening probe budget (notably SSMs / linear-attention — see
# feedback_ssm_capability_floor) stay competitively bred rather than collapsing
# to the floor. Verified 2026-05-24: softmax rises, pure-FFN ops fall, and SSM
# ops stay >= ~0.8 instead of the ~0.6 a base=0.25 hammer produced.
_CAP_BASE = 0.5  # credit for clearing s1 with no/low measured capability
_CAP_LO = 0.10  # induction AUC at/below which the capability bonus is 0
_CAP_HI = 0.50  # induction AUC at/above which the capability bonus is full


def _success_credit(row: Dict) -> float:
    """Capability-weighted breeding credit for one graph row, in ``[0, 1]``.

    Returns 0.0 if the graph never passed stage 1. Otherwise a soft blend: a
    baseline ``_CAP_BASE`` for clearing the gate, ramped to 1.0 by the measured
    ``induction_screening_auc_500``. Unmeasured capability gets a neutral mid
    credit. With ``ARIA_DISABLE_CAPABILITY_BREEDING=1`` this collapses to the
    legacy s1-only signal (1.0 for any stage-1 pass).
    """
    if not row.get("stage1_any_passed"):
        return 0.0
    if not _CAPABILITY_BREEDING:
        return 1.0
    auc = row.get("induction_screening_auc_500")
    if auc is None:
        cap = 0.5
    else:
        try:
            a = float(auc)
        except (TypeError, ValueError):
            a = 0.0
        cap = max(0.0, min(1.0, (a - _CAP_LO) / (_CAP_HI - _CAP_LO)))
    return _CAP_BASE + (1.0 - _CAP_BASE) * cap


def _load_deduped_graph_training_rows(db_path):
    from research.scientist.intelligence.ml_corpus import (
        load_deduped_graph_training_rows,
    )

    return load_deduped_graph_training_rows(db_path)


def _scaffold_blend_alpha(support: int) -> float:
    """Conservative influence for scaffold evidence."""
    return max(0.0, min(0.45, (float(support) / 16.0) * 0.45))


# Beta(α, β) prior pseudocount controlling shrinkage strength toward
# `prior_mean`. With strength=6, an item with n=6 attempts is a 50/50
# blend of prior and observed; with n>=30 the posterior is dominated by
# observed evidence. Strength=6 balances "new substrate inherits the
# cohort's median S1 rate" against "established templates aren't held
# back by a stale prior."
_BAYES_PRIOR_STRENGTH = 6.0


def _bayesian_posterior_rate(
    s1_count: float,
    n_used: int,
    prior_mean: float,
    prior_strength: float = _BAYES_PRIOR_STRENGTH,
) -> float:
    """Posterior mean of S1 success rate under a Beta(α, β) prior.

    α = prior_mean * prior_strength, β = (1 - prior_mean) * prior_strength.
    Posterior mean = (α + s1_count) / (α + β + n_used).

    For n_used = 0, returns prior_mean exactly. For n_used >> prior_strength,
    converges to the observed rate. This replaces the hard ``n >= min_used``
    exclusion that left new substrate absent from the weights dict and
    therefore at the mercy of the multiplier chain on incumbents.
    """
    if prior_strength <= 0.0:
        return s1_count / max(n_used, 1)
    prior_mean = max(0.0, min(1.0, prior_mean))
    alpha = prior_mean * prior_strength
    beta = (1.0 - prior_mean) * prior_strength
    denom = alpha + beta + max(n_used, 0)
    if denom <= 0.0:
        return prior_mean
    return (alpha + max(s1_count, 0)) / denom


def _fit_prior_mean(rates: Iterable[float]) -> float:
    """Median observed S1 rate, used as the Beta prior centre.

    Median is robust to a single high-success template skewing the prior.
    Returns 0.0 when the input is empty so callers can short-circuit.
    """
    values = sorted(float(r) for r in rates)
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _amplified_weights_from_counts(
    *,
    counts: Dict[str, int],
    s1_counts: Dict[str, float],
    min_used: int,
    amp_exponent: float = 1.5,
    weight_lo: float = 0.3,
    weight_hi: float = 5.0,
    confidence_scale: float = 30.0,
) -> Dict[str, float]:
    """Beta-Binomial-shrunk, contrast-amplified, confidence-blended weights.

    Steps:
      1. Empirical-Bayes prior_mean = median S1 rate of items with n >= min_used.
      2. Per-item posterior_rate via Beta-Binomial shrinkage toward prior_mean.
      3. Contrast amplification: (posterior_rate / mean_posterior) ** amp_exponent.
      4. Confidence blend toward 1.0 by min(1, n / confidence_scale).
      5. Clamp to [weight_lo, weight_hi].

    Items with n >= 1 appear in the output. Returns {} when there is no
    confident evidence to fit a non-trivial prior (preserves legacy
    short-circuit behaviour).
    """
    confident_rates = [
        s1_counts.get(name, 0) / n
        for name, n in counts.items()
        if n >= max(min_used, 1)
    ]
    prior_mean = _fit_prior_mean(confident_rates)
    if prior_mean < 1e-6:
        return {}
    posterior_rates: Dict[str, float] = {}
    for name, n in counts.items():
        if n < 1:
            continue
        posterior_rates[name] = _bayesian_posterior_rate(
            s1_counts.get(name, 0), n, prior_mean
        )
    if not posterior_rates:
        return {}
    mean_post = sum(posterior_rates.values()) / len(posterior_rates)
    if mean_post < 1e-6:
        return {}
    weights: Dict[str, float] = {}
    for name, post_rate in posterior_rates.items():
        relative = post_rate / mean_post
        amplified = relative**amp_exponent
        confidence = min(1.0, counts[name] / confidence_scale)
        blended = confidence * amplified + (1.0 - confidence) * 1.0
        weights[name] = round(max(weight_lo, min(weight_hi, blended)), 3)
    return weights


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
        """Per-op weights via Bayesian-shrunk contrast amplification.

        Posterior rate uses a Beta(α, β) prior centred on the median S1 rate
        of ops with ``n >= min_used``; the contrast formula
        ``(posterior_rate / mean_posterior)^2`` is then clamped to
        [0.1, 8.0]. Ops with n >= 1 (and scaffold support >= min_used)
        appear in the output so newly-introduced primitives are not
        excluded from the analytics layer's downstream multiplier chain.

        Structural ops (no learnable params) are excluded from the mean
        calculation and get weight 1.0 — they should not be penalized or
        rewarded based on S1 attribution since they are scaffolding.
        """
        from research.synthesis.context_rules import S1_EXEMPT_OPS

        counts: Dict[str, int] = defaultdict(int)
        s1_counts: Dict[str, float] = defaultdict(float)
        for row in self._deduped_graph_rows(since_ts=since_ts):
            if not row.get("stage0_any_passed"):
                continue
            try:
                graph = json.loads(str(row["graph_json"]))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            credit = _success_credit(row)
            for op in self._graph_ops(graph):
                if op in S1_EXEMPT_OPS:
                    continue
                counts[op] += 1
                s1_counts[op] += credit
        if not counts:
            return {}
        try:
            scaffold_stats = self.nb.get_scaffold_component_stats(
                since_ts=since_ts,
                min_support=max(2, min_used // 2),
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            scaffold_stats = {}
        # Blend scaffold prior into observed counts BEFORE shrinkage so the
        # downstream Bayesian update sees a single coherent (s1, n) per op.
        blended_counts: Dict[str, int] = dict(counts)
        blended_s1: Dict[str, float] = dict(s1_counts)
        for op_name, stat in scaffold_stats.items():
            if op_name in S1_EXEMPT_OPS:
                continue
            prior_rate = float(stat.get("prior_rate") or 0.0)
            support = int(stat.get("support") or 0)
            if op_name in blended_counts:
                alpha = _scaffold_blend_alpha(support)
                if alpha <= 0.0:
                    continue
                n = blended_counts[op_name]
                observed_rate = blended_s1.get(op_name, 0) / max(n, 1)
                blended_rate = (1.0 - alpha) * observed_rate + alpha * prior_rate
                blended_s1[op_name] = blended_rate * n
            elif support >= min_used:
                blended_counts[op_name] = support
                blended_s1[op_name] = prior_rate * support
        return _amplified_weights_from_counts(
            counts=blended_counts,
            s1_counts=blended_s1,
            min_used=min_used,
            amp_exponent=2.0,
            weight_lo=0.1,
            weight_hi=8.0,
        )

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
        from ``graph_json.metadata`` and computes per-item posterior S1
        success rates via Beta-Binomial shrinkage. Items with ``n >= 1``
        appear in the output; ``min_used`` is the threshold for inclusion
        in the empirical-Bayes prior fit (median S1 rate of confident items).
        Returns ``{item_name: weight}`` clamped to ``[0.3, 5.0]``.
        """
        rows = [
            row
            for row in self._deduped_graph_rows(since_ts=since_ts)
            if row.get("stage0_any_passed")
        ]
        counts: Dict[str, int] = defaultdict(int)
        s1_counts: Dict[str, float] = defaultdict(float)
        for row in rows:
            try:
                meta = json.loads(str(row["graph_json"])).get("metadata", {})
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            items = meta.get(metadata_key)
            if not isinstance(items, list):
                continue
            credit = _success_credit(row)
            for item in items:
                if not isinstance(item, str):
                    continue
                counts[item] += 1
                s1_counts[item] += credit
        if not counts:
            return {}
        return _amplified_weights_from_counts(
            counts=counts, s1_counts=s1_counts, min_used=min_used
        )

    def compute_template_weights(
        self, since_ts: float = 0.0, min_used: int = 3
    ) -> Dict[str, float]:
        """Per-template weights via Bayesian-shrunk contrast amplification.

        ``min_used`` controls the empirical-Bayes prior fit; templates with
        ``n >= 1`` always appear in the output (with heavy shrinkage toward
        the prior for low N). This replaces the legacy hard-exclusion that
        left freshly-added templates absent from the analytics weight dict.
        """
        return self._compute_metadata_weights("templates_used", since_ts, min_used)

    def compute_trial_template_stats(
        self, since_ts: float = 0.0, min_used: int = 3
    ) -> Dict[str, Dict[str, float]]:
        """Phase C.2 — split per-template stats by `_template_trial` flag.

        Returns:
          {template_name: {"n_trial": int, "s1_trial_rate": float,
                           "n_prod": int,  "s1_prod_rate":  float}}

        Used by the auto-demotion harness to identify trial templates that
        underperform their production peers. Only templates appearing as
        trial picks (graph.metadata["_template_trial"] == True) at least
        `min_used` times are reported.
        """
        rows = [
            row
            for row in self._deduped_graph_rows(since_ts=since_ts)
            if row.get("stage0_any_passed")
        ]
        n_trial: Dict[str, int] = defaultdict(int)
        s1_trial: Dict[str, int] = defaultdict(int)
        n_prod: Dict[str, int] = defaultdict(int)
        s1_prod: Dict[str, int] = defaultdict(int)
        for row in rows:
            try:
                meta = json.loads(str(row["graph_json"])).get("metadata", {})
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            tpls = meta.get("templates_used")
            if not isinstance(tpls, list):
                continue
            is_trial = bool(meta.get("_template_trial"))
            passed = bool(row.get("stage1_any_passed"))
            for tpl in tpls:
                if not isinstance(tpl, str):
                    continue
                if is_trial:
                    n_trial[tpl] += 1
                    if passed:
                        s1_trial[tpl] += 1
                else:
                    n_prod[tpl] += 1
                    if passed:
                        s1_prod[tpl] += 1
        names = {n for n, c in n_trial.items() if c >= min_used}
        return {
            tpl: {
                "n_trial": n_trial[tpl],
                "s1_trial_rate": (s1_trial[tpl] / n_trial[tpl])
                if n_trial[tpl]
                else 0.0,
                "n_prod": n_prod.get(tpl, 0),
                "s1_prod_rate": (
                    s1_prod[tpl] / n_prod[tpl] if n_prod.get(tpl) else 0.0
                ),
            }
            for tpl in sorted(names)
        }

    def compute_motif_weights(
        self, since_ts: float = 0.0, min_used: int = 3
    ) -> Dict[str, float]:
        """Per-motif weights from S1 success rates via contrast amplification."""
        return self._compute_metadata_weights("motifs_used", since_ts, min_used)

    def compute_template_and_motif_weights(
        self, since_ts: float = 0.0, min_used: int = 3
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute template and motif weights in a single DB query pass."""
        rows = [
            row
            for row in self._deduped_graph_rows(since_ts=since_ts)
            if row.get("stage0_any_passed")
        ]

        # Single-pass: accumulate counts for both keys simultaneously
        all_counts: Dict[str, Dict[str, int]] = {
            "templates_used": defaultdict(int),
            "motifs_used": defaultdict(int),
        }
        all_s1: Dict[str, Dict[str, float]] = {
            "templates_used": defaultdict(float),
            "motifs_used": defaultdict(float),
        }
        for row in rows:
            try:
                meta = json.loads(str(row["graph_json"])).get("metadata", {})
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            credit = _success_credit(row)
            for mk in ("templates_used", "motifs_used"):
                items = meta.get(mk)
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, str):
                        continue
                    all_counts[mk][item] += 1
                    all_s1[mk][item] += credit

        templates = _amplified_weights_from_counts(
            counts=all_counts["templates_used"],
            s1_counts=all_s1["templates_used"],
            min_used=min_used,
        )
        motifs = _amplified_weights_from_counts(
            counts=all_counts["motifs_used"],
            s1_counts=all_s1["motifs_used"],
            min_used=min_used,
        )
        return templates, motifs

    def _deduped_graph_rows(self, since_ts: float = 0.0) -> List[Dict]:
        db_path = str(getattr(self.nb, "db_path", Path("research/lab_notebook.db")))
        rows = _load_deduped_graph_training_rows(db_path)
        if since_ts <= 0:
            return rows
        return [
            row for row in rows if float(row.get("latest_timestamp") or 0.0) >= since_ts
        ]

    @staticmethod
    def _graph_ops(graph: Dict) -> set[str]:
        ops: set[str] = set()
        nodes = graph.get("nodes", {})
        if isinstance(nodes, dict):
            iterator = nodes.values()
        elif isinstance(nodes, list):
            iterator = nodes
        else:
            return ops
        for node in iterator:
            if isinstance(node, dict):
                op = node.get("op_name") or node.get("op_type") or node.get("op") or ""
            elif isinstance(node, str):
                op = node
            else:
                continue
            if op and op not in {"input", "output"}:
                ops.add(op)
        return ops

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
