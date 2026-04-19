from __future__ import annotations

"""Auto-extracted mixin for LabNotebook."""

import json
import math
import statistics
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import time
import uuid
from typing import Any, Dict, List, Optional
from ..json_utils import fast_loads as _json_loads
from ..leaderboard_scoring import (
    compute_efficiency_multiple as _compute_efficiency_multiple,
    compute_pre_investigation_score as _compute_pre_investigation_score,
)

_TEMPLATE_DEF_RE = re.compile(r"^def\s+(tpl_[A-Za-z0-9_]+)\s*\(", re.M)
_EMPTY_DATA_ACCOUNTING_SHAPE = {
    "row_volume": {},
    "run_volume": {},
    "graph_volume": {},
    "filtering": {},
    "training_curve_density": {},
    "leaderboard_tiers": {},
}


@lru_cache(maxsize=1)
def _synthesis_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "synthesis"


@lru_cache(maxsize=1)
def _load_template_source_map() -> Dict[str, str]:
    template_sources: Dict[str, str] = {}
    for template_file in sorted(_synthesis_dir().glob("_templates*.py")):
        try:
            source = template_file.read_text(encoding="utf-8")
        except OSError:
            continue
        matches = list(_TEMPLATE_DEF_RE.finditer(source))
        for idx, match in enumerate(matches):
            name = match.group(1).removeprefix("tpl_")
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(source)
            template_sources[name] = source[start:end]
    return template_sources


@lru_cache(maxsize=1)
def _discover_template_names() -> tuple[str, ...]:
    return tuple(sorted(_load_template_source_map()))


@lru_cache(maxsize=8192)
def _cached_extract_op_bigrams(graph_json: str) -> tuple[str, ...]:
    try:
        data = _json_loads(graph_json)
    except (json.JSONDecodeError, TypeError):
        return ()
    nodes = data.get("nodes", {})
    if not isinstance(nodes, dict):
        return ()
    bigrams: set[str] = set()
    for nd in nodes.values():
        if not isinstance(nd, dict):
            continue
        op = nd.get("op_name", "")
        if not op or op == "input":
            continue
        for inp in nd.get("input_ids", ()):
            parent = nodes.get(str(inp), {})
            if not isinstance(parent, dict):
                continue
            pop = parent.get("op_name", "")
            if pop and pop != "input":
                bigrams.add(f"{pop}->{op}")
    return tuple(sorted(bigrams))


@lru_cache(maxsize=8192)
def _cached_extract_observability_metadata(
    graph_json: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[Dict[str, Any], ...]]:
    try:
        graph = _json_loads(graph_json) if graph_json else {}
    except (json.JSONDecodeError, TypeError):
        return (), (), ()
    metadata = graph.get("metadata", {}) if isinstance(graph, dict) else {}
    templates = metadata.get("templates_used")
    motifs = metadata.get("motifs_used")
    slot_usage = metadata.get("template_slot_usage")
    normalized_templates = (
        tuple(str(item) for item in templates if item is not None)
        if isinstance(templates, list)
        else ()
    )
    normalized_motifs = (
        tuple(str(item) for item in motifs if item is not None)
        if isinstance(motifs, list)
        else ()
    )
    normalized_slots = (
        tuple(slot for slot in slot_usage if isinstance(slot, dict))
        if isinstance(slot_usage, list)
        else ()
    )
    return normalized_templates, normalized_motifs, normalized_slots


@dataclass
class _ObservabilityAccumulator:
    __slots__ = (
        "template_stats",
        "motif_stats",
        "slot_stats",
        "experiment_buckets",
        "loss_values",
        "validation_losses",
        "discovery_losses",
        "motifs_per_graph",
        "templates_per_graph",
    )
    template_stats: Dict[str, Dict[str, Any]]
    motif_stats: Dict[str, Dict[str, Any]]
    slot_stats: Dict[str, Dict[str, Any]]
    experiment_buckets: Dict[str, Dict[str, Any]]
    loss_values: List[float]
    validation_losses: List[float]
    discovery_losses: List[float]
    motifs_per_graph: List[float]
    templates_per_graph: List[float]


_STRUCTURAL_CATEGORY_CACHE: Dict[str, str] = {}

_REFERENCE_TEMPLATES = frozenset(
    {
        "gpt2_reference",
        "mamba_reference",
        "rwkv_block",
        "rwkv_double_norm",
        "rwkv_sparse_chain",
    }
)

_EXOTIC_TEMPLATES = frozenset(
    {
        "normalized_matmul",
        "gated_product",
        "safe_division",
        "cosine_scoring",
        "decay_sequence",
        "hyp_distance_scoring",
        "tropical_residual",
        "tropical_center_block",
        "geometric_product_block",
        "residual_difference",
        "tropical_matmul_block",
        "gated_minimum",
        "spiking_residual_block",
        "spiking_moe_block",
        "hyperbolic_bridge_block",
        "poincare_add_bridge",
        "spiking_stdp_block",
        "reciprocal_gated",
        "sign_ste_gated",
        "log_gated",
    }
)

_ATTENTION_OPS = frozenset(
    {
        "softmax_attention",
        "latent_attention_compressor",
        "graph_attention",
        "local_window_attn",
        "linear_attention",
        "diff_attention",
        "MOTIF_CLASS_ATTENTION",
    }
)

_FFN_OPS = frozenset(
    {
        "swiglu_mlp",
        "gelu_mlp",
        "_FFN_CLASSES",
        "fused_linear_gelu",
    }
)


def _classify_template_structural(name: str) -> str:
    """Classify a template by its structural family.

    Families:
      - reference: GPT-2/Mamba/RWKV baselines
      - exotic: Non-standard math (tropical, spiking, hyperbolic)
      - strong: Has attention + FFN + 2+ residuals (parallel mixing ideal)
      - decent: Has attention + FFN but may be sequential
      - weak: Missing attention or FFN
    """
    if name in _STRUCTURAL_CATEGORY_CACHE:
        return _STRUCTURAL_CATEGORY_CACHE[name]

    if name in _REFERENCE_TEMPLATES:
        _STRUCTURAL_CATEGORY_CACHE[name] = "reference"
        return "reference"

    if name in _EXOTIC_TEMPLATES:
        _STRUCTURAL_CATEGORY_CACHE[name] = "exotic"
        return "exotic"

    # Name-based heuristics for known strong patterns
    # Hybrid templates combine attention with SSM/conv in parallel
    if "_ssm_hybrid" in name or "_conv_hybrid" in name:
        _STRUCTURAL_CATEGORY_CACHE[name] = "strong"
        return "strong"

    src = _load_template_source_map().get(name, "")
    if not src:
        _STRUCTURAL_CATEGORY_CACHE[name] = "weak"
        return "weak"
    # Also check if the function delegates to a known factory
    is_factory = (
        "_tpl_attention_ffn_block" in src
        or "_tpl_attn_op_chain" in src
        or "_make_attn_ffn_template" in src
    )

    has_attention = (
        any(op in src for op in _ATTENTION_OPS)
        or is_factory
        or "_MIXER_CLASSES" in src  # MIXER_CLASSES includes ATTENTION
    )
    has_ffn = any(op in src for op in _FFN_OPS) or is_factory
    n_residuals = src.count("_residual(") + src.count("template_add_residual(")

    # Name-based attention detection for factory-generated templates
    attn_prefixes = (
        "attn_",
        "latent_attn_",
        "local_attn_",
        "diff_attn_",
        "graph_attn_",
        "linear_attn_",
    )
    if any(name.startswith(p) for p in attn_prefixes):
        has_attention = True

    # Factory-generated attention+FFN blocks are at least decent
    if is_factory:
        n_residuals = max(n_residuals, 2)

    # Detect parallel structure
    has_parallel = (
        src.count("[normed]") >= 2
        or src.count("[normed,") >= 1
        or "|| SSM" in src
        or "|| state_space" in src
        or "|| padic" in src
        or "Path A" in src
        or "Path B" in src
        or "MOTIF_CLASS_SSM" in src  # Picks SSM as parallel path
        or ("state_space" in src and "MOTIF_CLASS_ATTENTION" in src)
    )

    if has_attention and has_ffn and n_residuals >= 2 and has_parallel:
        cat = "strong"
    elif has_attention and has_ffn:
        cat = "decent"
    elif has_attention or has_ffn:
        cat = "weak"
    else:
        cat = "weak"

    _STRUCTURAL_CATEGORY_CACHE[name] = cat
    return cat


def _capability_signal_count(row: Dict[str, Any]) -> int:
    hits = 0
    if (
        row.get("avg_validation_loss_ratio") is not None
        and float(row["avg_validation_loss_ratio"]) <= 0.65
    ):
        hits += 1
    if (
        row.get("avg_induction_auc") is not None
        and float(row["avg_induction_auc"]) >= 0.03
    ):
        hits += 1
    if row.get("avg_binding_auc") is not None and float(row["avg_binding_auc"]) >= 0.05:
        hits += 1
    if row.get("avg_ar_auc") is not None and float(row["avg_ar_auc"]) >= 0.05:
        hits += 1
    if (
        row.get("avg_hellaswag_acc") is not None
        and float(row["avg_hellaswag_acc"]) >= 0.30
    ):
        hits += 1
    return hits


def _reference_metric_baselines(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    baselines: Dict[str, float] = {}
    reference_rows = [
        row
        for row in rows
        if row.get("structural_family") == "reference" and (row.get("n_used") or 0) >= 3
    ]
    if not reference_rows:
        return baselines

    lower_is_better = ("avg_validation_loss_ratio", "avg_loss_ratio")
    higher_is_better = (
        "avg_induction_auc",
        "avg_binding_auc",
        "avg_ar_auc",
        "avg_hellaswag_acc",
    )
    for metric in lower_is_better:
        vals = [
            float(row[metric])
            for row in reference_rows
            if row.get(metric) is not None and math.isfinite(float(row[metric]))
        ]
        if vals:
            baselines[metric] = min(vals)
    for metric in higher_is_better:
        vals = [
            float(row[metric])
            for row in reference_rows
            if row.get(metric) is not None and math.isfinite(float(row[metric]))
        ]
        if vals:
            baselines[metric] = max(vals)
    return baselines


def _reference_beating_metrics(
    row: Dict[str, Any],
    baselines: Dict[str, float],
) -> List[str]:
    beats: List[str] = []
    val_lr = row.get("avg_validation_loss_ratio")
    if (
        val_lr is not None
        and "avg_validation_loss_ratio" in baselines
        and float(val_lr) <= baselines["avg_validation_loss_ratio"] * 0.98
    ):
        beats.append("val_loss")
    train_lr = row.get("avg_loss_ratio")
    if (
        train_lr is not None
        and "avg_loss_ratio" in baselines
        and float(train_lr) <= baselines["avg_loss_ratio"] * 0.98
    ):
        beats.append("train_loss")
    metric_pairs = (
        ("avg_induction_auc", "induction"),
        ("avg_binding_auc", "binding"),
        ("avg_ar_auc", "ar"),
        ("avg_hellaswag_acc", "hellaswag"),
    )
    for metric, label in metric_pairs:
        value = row.get(metric)
        baseline = baselines.get(metric)
        if (
            value is not None
            and baseline is not None
            and float(value) >= float(baseline) + 0.005
        ):
            beats.append(label)
    return beats


def _template_label_from_evidence(
    row: Dict[str, Any],
    baselines: Dict[str, float],
) -> str:
    family = str(row.get("structural_family") or "")
    if family == "reference":
        return "reference"
    if family == "exotic":
        return "exotic"

    evidence_level = str(row.get("evidence_level") or "")
    if evidence_level == "insufficient":
        return "untested"
    if evidence_level == "sparse":
        return "data-sparse"

    unique_fingerprints = int(row.get("unique_fingerprints") or 0)
    s1_unique_fingerprints = int(row.get("stage1_unique_fingerprints") or 0)
    repeated_low_loss_count = int(row.get("repeated_low_loss_count") or 0)
    capability_signal_count = int(row.get("capability_signal_count") or 0)
    reference_beats = len(row.get("reference_beating_metrics") or [])
    s1_rate = float(row.get("s1_rate") or 0.0)

    if evidence_level == "building" and (
        unique_fingerprints < 6 or s1_unique_fingerprints < 2
    ):
        return "data-sparse"

    if (
        evidence_level in {"building", "established"}
        and unique_fingerprints >= 12
        and s1_unique_fingerprints >= 6
        and repeated_low_loss_count >= 4
        and s1_rate >= 0.20
        and capability_signal_count >= 2
        and (reference_beats >= 2 or not baselines)
    ):
        return "strong"

    if (
        unique_fingerprints >= 6
        and s1_unique_fingerprints >= 2
        and s1_rate >= 0.12
        and (
            capability_signal_count >= 1
            or repeated_low_loss_count >= 2
            or reference_beats >= 1
        )
    ):
        return "decent"

    return "weak"


def _summarize_template_stat(stat: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize a single template's accumulated stats into a result dict."""
    losses = stat["losses"]
    stage1_losses = stat.get("stage1_losses") or []
    validation_vals = stat["validation_losses"]
    discovery_vals = stat["discovery_losses"]
    novelties = stat["novelties"]
    novelty_confidences = stat["novelty_confidences"]
    induction_aucs = stat["induction_aucs"]
    binding_aucs = stat["binding_aucs"]
    ar_aucs = stat["ar_aucs"]
    hellaswag_accs = stat["hellaswag_accs"]
    screening_hellaswag_accs = stat["screening_hellaswag_accs"]
    reasons = stat["failure_reasons"]
    fast_lane_scores = stat["routing_fast_lane_scores"]
    fast_lane_improvements = stat["routing_fast_lane_improvements"]
    fast_lane_slopes = stat["routing_fast_lane_slopes"]
    top_reason = None
    if reasons:
        top_reason = max(reasons.items(), key=lambda item: item[1])[0]
    n_used = int(stat["n_used"] or 0)
    core = {
        "n_used": n_used,
        "s0_rate": stat["n_stage0"] / max(n_used, 1),
        "s05_rate": stat["n_stage05"] / max(n_used, 1),
        "s1_rate": stat["n_stage1"] / max(n_used, 1),
        "avg_loss_ratio": sum(losses) / len(losses) if losses else None,
        "best_loss_ratio": min(losses) if losses else None,
        "avg_validation_loss_ratio": (
            sum(validation_vals) / len(validation_vals) if validation_vals else None
        ),
        "avg_discovery_loss_ratio": (
            sum(discovery_vals) / len(discovery_vals) if discovery_vals else None
        ),
        "avg_novelty": (sum(novelties) / len(novelties) if novelties else None),
        "avg_novelty_confidence": (
            sum(novelty_confidences) / len(novelty_confidences)
            if novelty_confidences
            else None
        ),
        "avg_induction_auc": (
            sum(induction_aucs) / len(induction_aucs) if induction_aucs else None
        ),
        "avg_binding_auc": (
            sum(binding_aucs) / len(binding_aucs) if binding_aucs else None
        ),
        "avg_ar_auc": sum(ar_aucs) / len(ar_aucs) if ar_aucs else None,
        "avg_hellaswag_acc": (
            sum(hellaswag_accs) / len(hellaswag_accs) if hellaswag_accs else None
        ),
        "avg_screening_hellaswag_acc": (
            sum(screening_hellaswag_accs) / len(screening_hellaswag_accs)
            if screening_hellaswag_accs
            else None
        ),
        "screening_wikitext_ok_rate": (
            int(stat.get("screening_wikitext_ok") or 0)
            / max(int(stat.get("screening_wikitext_runs") or 0), 1)
            if stat.get("screening_wikitext_runs")
            else None
        ),
        "screening_metric_coverage": {
            "induction": len(induction_aucs),
            "binding": len(binding_aucs),
            "associative_recall": len(ar_aucs),
            "hellaswag": len(hellaswag_accs) + len(screening_hellaswag_accs),
            "wikitext": int(stat.get("screening_wikitext_runs") or 0),
        },
        "slot_count": int(stat.get("slot_count") or 0),
        "routing_fast_lane_runs": int(stat.get("routing_fast_lane_runs") or 0),
        "routing_fast_lane_ok_rate": (
            int(stat.get("routing_fast_lane_ok") or 0)
            / max(int(stat.get("routing_fast_lane_runs") or 0), 1)
            if stat.get("routing_fast_lane_runs")
            else None
        ),
        "routing_fast_lane_positive_rate": (
            int(stat.get("routing_fast_lane_positive") or 0)
            / max(int(stat.get("routing_fast_lane_runs") or 0), 1)
            if stat.get("routing_fast_lane_runs")
            else None
        ),
        "routing_fast_lane_avg_score": (
            sum(fast_lane_scores) / len(fast_lane_scores) if fast_lane_scores else None
        ),
        "routing_fast_lane_avg_improvement": (
            sum(fast_lane_improvements) / len(fast_lane_improvements)
            if fast_lane_improvements
            else None
        ),
        "routing_fast_lane_avg_slope": (
            sum(fast_lane_slopes) / len(fast_lane_slopes) if fast_lane_slopes else None
        ),
        "evidence_level": (
            "insufficient"
            if n_used < 3
            else "sparse"
            if n_used < 10
            else "building"
            if n_used < 30
            else "established"
        ),
    }
    n_used = int(core["n_used"])
    s0_rate = core["s0_rate"]
    s05_rate = core["s05_rate"]
    s1_rate = core["s1_rate"]
    avg_loss_ratio = core["avg_loss_ratio"]
    avg_validation_loss_ratio = core["avg_validation_loss_ratio"]
    avg_induction_auc = core["avg_induction_auc"]
    avg_binding_auc = core["avg_binding_auc"]
    avg_hellaswag_acc = core["avg_hellaswag_acc"]
    evidence_level = core["evidence_level"]

    diagnosis: List[str] = []
    actions: List[str] = []
    if evidence_level == "insufficient":
        diagnosis.append("Too little evidence to rank confidently.")
        actions.append("Backfill this template before changing weights.")
    elif s0_rate < 0.5:
        diagnosis.append("Most runs fail before stable screening begins.")
        actions.append("Audit template wiring and unsafe op combinations.")
    elif s05_rate + 0.15 < s0_rate:
        diagnosis.append(
            "Candidates clear S0 but drop during the stability band before S1."
        )
        actions.append("Tighten motif compatibility and lane constraints.")
    elif s1_rate < 0.15:
        diagnosis.append("Template consumes budget but rarely reaches Stage 1.")
        actions.append("Downweight until slot and motif evidence improves.")
    elif s1_rate > 0.4:
        diagnosis.append("Template is producing Stage-1 survivors consistently.")
        actions.append("Use as a reference family for nearby sparse templates.")
    if (
        avg_validation_loss_ratio is not None
        and avg_loss_ratio is not None
        and avg_validation_loss_ratio > avg_loss_ratio * 1.15
    ):
        diagnosis.append(
            "Validation materially trails training, suggesting brittle generalization."
        )
        actions.append("Reduce brittle motif mixes or extend slow-starter screening.")
    if avg_induction_auc is not None and avg_induction_auc < 0.02 and s1_rate >= 0.2:
        diagnosis.append("Survivors train, but induction evidence remains weak.")
        actions.append("Bias backfills toward longer-range token-interaction motifs.")
    if avg_binding_auc is not None and avg_binding_auc < 0.05 and s1_rate >= 0.2:
        diagnosis.append("Binding/copy behavior is weak relative to survivor rate.")
        actions.append("Probe slot choices that preserve non-local token access.")
    if avg_hellaswag_acc is not None and avg_hellaswag_acc <= 0.27:
        diagnosis.append("Commonsense signal is near noise floor.")
        actions.append("Do not trust perplexity-only wins from this family.")
    if (
        stat.get("routing_fast_lane_runs")
        and (stat.get("routing_fast_lane_positive") or 0) >= 2
        and s1_rate < 0.2
    ):
        diagnosis.append("Fast-lane probes are positive despite poor short-run S1.")
        actions.append("Treat it as a slow starter and extend targeted backfills.")
    if top_reason and len(diagnosis) < 3:
        diagnosis.append(f"Most common failure mode is {top_reason}.")
    if not actions:
        actions.append("Keep sampling while collecting more slot-level evidence.")
    repeated_low_loss_count = sum(1 for v in stage1_losses if v <= 0.45)
    very_low_loss_count = sum(1 for v in stage1_losses if v <= 0.40)
    structural_category = _classify_template_structural(stat["name"])
    return {
        "name": stat["name"],
        "structural_family": structural_category,
        "structural_category": structural_category,
        **core,
        "unique_fingerprints": int(len(stat.get("fingerprints") or ())),
        "stage1_unique_fingerprints": int(len(stat.get("stage1_fingerprints") or ())),
        "survivor_loss_median": (
            statistics.median(stage1_losses) if stage1_losses else None
        ),
        "repeated_low_loss_count": repeated_low_loss_count,
        "very_low_loss_count": very_low_loss_count,
        "repeated_low_loss_family": repeated_low_loss_count >= 3,
        "top_failure_reason": top_reason,
        "failure_reasons": dict(sorted(reasons.items(), key=lambda item: -item[1])[:3]),
        "diagnosis": diagnosis[:3],
        "actions": actions[:3],
    }


def _empty_template_stat(name: str, slot_count: int) -> Dict[str, Any]:
    """Construct a canonical zero-run template stat payload."""
    return {
        "name": str(name),
        "n_used": 0,
        "n_stage0": 0,
        "n_stage05": 0,
        "n_stage1": 0,
        "losses": [],
        "stage1_losses": [],
        "fingerprints": set(),
        "stage1_fingerprints": set(),
        "validation_losses": [],
        "discovery_losses": [],
        "novelties": [],
        "novelty_confidences": [],
        "induction_aucs": [],
        "binding_aucs": [],
        "ar_aucs": [],
        "hellaswag_accs": [],
        "screening_hellaswag_accs": [],
        "screening_wikitext_runs": 0,
        "screening_wikitext_ok": 0,
        "failure_reasons": {},
        "slot_count": int(slot_count or 0),
        "routing_fast_lane_runs": 0,
        "routing_fast_lane_ok": 0,
        "routing_fast_lane_positive": 0,
        "routing_fast_lane_scores": [],
        "routing_fast_lane_improvements": [],
        "routing_fast_lane_slopes": [],
    }
