from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Dict, Tuple

from ...synthesis.grammar import GrammarConfig
from ...synthesis.motifs import VALIDATED_MOTIFS
from ...synthesis.templates import DEFAULT_TEMPLATE_WEIGHTS, TEMPLATES
from ..notebook import LabNotebook

logger = logging.getLogger(__name__)

_BUCKET_TEMPLATE_BOOSTS: Dict[str, Dict[str, float]] = {
    "attention-heavy": {
        "transformer_block": 1.6,
        "hybrid_parallel": 1.2,
        "residual_block": 0.5,
    },
    "mixing-heavy": {
        "hybrid_parallel": 1.4,
        "sequential": 0.9,
        "residual_block": 0.5,
    },
    "sparse": {
        "sparse_ffn": 1.8,
        "bottleneck": 1.1,
        "moe": 0.8,
    },
    "hybrid": {
        "hybrid_parallel": 1.8,
        "transformer_block": 1.3,
        "parallel_split": 0.8,
    },
    "exotic": {
        "parallel_split": 1.0,
        "gated_residual": 0.7,
        "dense_cascade": 0.5,
    },
}

_TOP_OP_TEMPLATE_HINTS: Dict[str, Dict[str, float]] = {
    "attention": {"transformer_block": 1.1, "hybrid_parallel": 0.7},
    "scan": {"hybrid_parallel": 1.0, "sequential": 0.5},
    "state_space": {"hybrid_parallel": 1.0, "sequential": 0.5},
    "conv": {"hybrid_parallel": 0.8, "sequential": 0.5},
    "sparse": {"sparse_ffn": 1.2, "bottleneck": 0.6},
    "rank": {"bottleneck": 0.8, "sparse_ffn": 0.4},
    "moe": {"moe": 1.2, "gated_residual": 0.4},
    "gate": {"gated_residual": 0.9, "moe": 0.4},
    "norm": {"residual_block": 0.6, "transformer_block": 0.5},
}

_MIN_CONFIDENCE = 0.6


def _freeze_op_pair_priors(
    priors: list[dict[str, Any]],
) -> Tuple[Tuple[str, float], ...]:
    return tuple(
        (
            str(row.get("signature") or ""),
            round(float(row.get("success_rate") or 0.0), 4),
        )
        for row in priors
        if row.get("signature")
    )


def _freeze_fingerprint_buckets(
    buckets: list[dict[str, Any]],
) -> Tuple[Tuple[str, int, float, Tuple[str, ...]], ...]:
    return tuple(
        (
            str(row.get("bucket") or ""),
            int(row.get("n_graphs") or 0),
            round(float(row.get("s1_rate") or 0.0), 4),
            tuple(
                str(op.get("op_name") or "")
                for op in (row.get("top_ops") or [])
                if op.get("op_name")
            ),
        )
        for row in buckets
        if row.get("bucket")
    )


@lru_cache(maxsize=32)
def _cached_signal_weight_maps(
    op_pair_priors: Tuple[Tuple[str, float], ...],
    fingerprint_buckets: Tuple[Tuple[str, int, float, Tuple[str, ...]], ...],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    pair_rates = {
        signature: success_rate
        for signature, success_rate in op_pair_priors
        if success_rate > 0.3
    }
    motif_weights = {
        motif_name: round(
            sum(
                pair_rates.get(
                    f"{motif.steps[index].op_name}->{motif.steps[index + 1].op_name}",
                    0.0,
                )
                for index in range(len(motif.steps) - 1)
            ),
            4,
        )
        for motif_name, motif in VALIDATED_MOTIFS.items()
    }
    motif_weights = {
        motif_name: weight
        for motif_name, weight in motif_weights.items()
        if weight > 0.0
    }

    template_bonuses: Dict[str, float] = {}
    for bucket_name, n_graphs, s1_rate, top_ops in fingerprint_buckets:
        dominance = max(1.0, float(n_graphs)) * max(0.25, s1_rate)
        for template_name, boost in _BUCKET_TEMPLATE_BOOSTS.get(
            bucket_name, {}
        ).items():
            if template_name in TEMPLATES:
                template_bonuses[template_name] = template_bonuses.get(
                    template_name, 0.0
                ) + (boost * dominance)
        for op_name in top_ops:
            lowered = op_name.lower()
            for token, boosts in _TOP_OP_TEMPLATE_HINTS.items():
                if token not in lowered:
                    continue
                for template_name, boost in boosts.items():
                    if template_name in TEMPLATES:
                        template_bonuses[template_name] = template_bonuses.get(
                            template_name, 0.0
                        ) + (boost * dominance)

    template_weights = {
        template_name: round(
            DEFAULT_TEMPLATE_WEIGHTS.get(template_name, 1.0) + bonus, 4
        )
        for template_name, bonus in template_bonuses.items()
        if bonus > 0.0
    }
    return template_weights, motif_weights


def _decode_evidence_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.debug("Malformed insight evidence JSON: %s", exc)
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def build_signal_weight_maps(
    nb: LabNotebook,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    try:
        op_pair_priors = nb.get_op_pair_priors(min_support=5, limit=50)
        fingerprint_buckets = nb.get_fingerprint_buckets(limit=5)
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug("Failed fetching signal weight data: %s", exc)
        return {}, {}
    if not op_pair_priors and not fingerprint_buckets:
        return {}, {}
    return _cached_signal_weight_maps(
        _freeze_op_pair_priors(op_pair_priors or []),
        _freeze_fingerprint_buckets(fingerprint_buckets or []),
    )


def _apply_profiling_composition_rule(
    grammar: GrammarConfig,
    subject: str,
    evidence: dict[str, Any],
    conf: float,
) -> None:
    risk = evidence.get("risk_score")
    if risk is not None and float(risk) > 50:
        penalty = max(0.15, 1.0 - (float(risk) / 100.0) * conf)
        cur = grammar.op_weights.get(subject, 1.0)
        grammar.op_weights[subject] = cur * penalty
        for follower in evidence.get("valid_followers", []):
            cur_f = grammar.op_weights.get(follower, 1.0)
            grammar.op_weights[follower] = cur_f * (1.0 + conf * 0.1)
        return

    comp_rates = evidence.get("composition_rates")
    if isinstance(comp_rates, dict):
        res_info = comp_rates.get("residual")
        seq_info = comp_rates.get("sequential")
        if isinstance(res_info, dict) and isinstance(seq_info, dict):
            res_rate = float(res_info.get("rate", 0))
            seq_rate = float(seq_info.get("rate", 0))
            if res_rate > seq_rate:
                grammar.residual_prob = min(0.85, grammar.residual_prob + conf * 0.1)
        return

    for key in ("best_followers", "bridge_ops", "stabilizer_set"):
        named_set = evidence.get(key)
        if isinstance(named_set, dict) and named_set:
            for op_name, stats in named_set.items():
                rate = float(stats.get("rate", 0)) if isinstance(stats, dict) else float(stats or 0)
                if rate >= 0.7:
                    cur = grammar.op_weights.get(op_name, 1.0)
                    grammar.op_weights[op_name] = cur * (1.0 + conf * 0.2 * rate)
            return

    dampeners = evidence.get("dampener_ops")
    if isinstance(dampeners, list) and dampeners:
        for op_name in dampeners:
            cur = grammar.op_weights.get(op_name, 1.0)
            grammar.op_weights[op_name] = cur * (1.0 + conf * 0.25)
        return

    correctors = evidence.get("corrector_ops")
    if isinstance(correctors, dict) and correctors:
        for op_name, stats in correctors.items():
            rate = (
                float(stats.get("correction_rate", 0)) if isinstance(stats, dict) else 0
            )
            if rate >= 0.5:
                cur = grammar.op_weights.get(op_name, 1.0)
                grammar.op_weights[op_name] = cur * (1.0 + conf * 0.15 * rate)
        return

    valid_followers = evidence.get("valid_followers")
    if isinstance(valid_followers, list) and valid_followers:
        for op_name in valid_followers:
            cur = grammar.op_weights.get(op_name, 1.0)
            grammar.op_weights[op_name] = cur * (1.0 + conf * 0.1)


def apply_insight_adjustments(
    nb: LabNotebook,
    grammar: GrammarConfig,
    template_weights: Dict[str, float],
    motif_weights: Dict[str, float],
) -> None:
    try:
        structural = nb.get_insights(
            exclude_display_only=True,
            insight_level="structural",
            limit=20,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug("Failed fetching structural insights: %s", exc)
        structural = []

    for ins in structural:
        alpha = float(ins.get("alpha") or 1.0)
        beta_ = float(ins.get("beta_") or 1.0)
        conf = alpha / (alpha + beta_)
        if conf < _MIN_CONFIDENCE:
            continue

        subject = str(ins.get("subject_key") or "")
        evidence = _decode_evidence_json(ins.get("evidence_json"))
        if subject == "graph_size_cap" and evidence.get("recommended_max"):
            grammar.max_ops = min(grammar.max_ops, int(evidence["recommended_max"]))
        elif subject == "graph_size_optimal":
            best = evidence.get("best_bucket", "")
            if "7-9" in best:
                grammar.composition_depth = min(grammar.composition_depth, 2)
                grammar.max_ops = min(grammar.max_ops, 12)
        else:
            comp_rates = evidence.get("composition_rates")
            if isinstance(comp_rates, dict):
                res_info = comp_rates.get("residual")
                seq_info = comp_rates.get("sequential")
                res_rate = float(res_info.get("rate", 0)) if isinstance(res_info, dict) else 0.0
                seq_rate = float(seq_info.get("rate", 0)) if isinstance(seq_info, dict) else 0.0
                if res_rate > seq_rate:
                    grammar.residual_prob = min(
                        0.85, grammar.residual_prob + conf * 0.1
                    )
            ppr = evidence.get("param_param_residual")
            if isinstance(ppr, dict) and float(ppr.get("rate", 0)) > 0.7:
                grammar.residual_prob = min(0.85, grammar.residual_prob + conf * 0.05)
            correctors = evidence.get("corrector_ops")
            if isinstance(correctors, dict):
                for op_name, stats in correctors.items():
                    rate = float(stats.get("correction_rate", 0)) if isinstance(stats, dict) else 0.0
                    if rate >= 0.5:
                        cur = grammar.op_weights.get(op_name, 1.0)
                        grammar.op_weights[op_name] = cur * (1.0 + conf * 0.15 * rate)

    try:
        template_insights = nb.get_insights(
            exclude_display_only=True,
            insight_level="template",
            limit=20,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug("Failed fetching template insights: %s", exc)
        template_insights = []

    for ins in template_insights:
        alpha = float(ins.get("alpha") or 1.0)
        beta_ = float(ins.get("beta_") or 1.0)
        conf = alpha / (alpha + beta_)
        if conf < _MIN_CONFIDENCE:
            continue

        subject = str(ins.get("subject_key") or "")
        subject_parts = {
            part.strip().lower()
            for part in subject.replace("+", " ").replace("_", " ").split()
            if len(part.strip()) >= 3
        }
        for tpl_name in list(template_weights.keys()):
            tpl_parts = {part.lower() for part in tpl_name.replace("_", " ").split()}
            if subject_parts & tpl_parts:
                template_weights[tpl_name] *= max(0.2, 1.0 - conf * 0.6)

    try:
        composition = nb.get_insights(
            exclude_display_only=True,
            insight_level="composition",
            limit=50,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug("Failed fetching composition insights: %s", exc)
        composition = []

    for ins in composition:
        alpha = float(ins.get("alpha") or 1.0)
        beta_ = float(ins.get("beta_") or 1.0)
        conf = alpha / (alpha + beta_)
        if conf < _MIN_CONFIDENCE:
            continue

        subject = str(ins.get("subject_key") or "")
        evidence = _decode_evidence_json(ins.get("evidence_json"))
        insight_type = str(ins.get("insight_type") or "")
        semantic = str(ins.get("semantic_key") or "")
        if insight_type == "composition_rule" and semantic.startswith("profiling:"):
            _apply_profiling_composition_rule(grammar, subject, evidence, conf)
            continue
        if insight_type == "top_op" and semantic.startswith("profiling:"):
            stabilizer_set = evidence.get("stabilizer_set") or {}
            for op_name, stats in stabilizer_set.items():
                rate = float(stats.get("rate", 0))
                if rate >= 0.8:
                    cur = grammar.op_weights.get(op_name, 1.0)
                    grammar.op_weights[op_name] = cur * (1.0 + conf * 0.4)
            continue

        subject_ops = {part.strip() for part in subject.split("+") if part.strip()}
        if not subject_ops:
            continue
        for motif_name, motif in VALIDATED_MOTIFS.items():
            motif_ops = {step.op_name for step in motif.steps}
            if subject_ops & motif_ops:
                motif_weights[motif_name] = motif_weights.get(motif_name, motif.lift) * (
                    1.0 + conf * 0.3
                )
