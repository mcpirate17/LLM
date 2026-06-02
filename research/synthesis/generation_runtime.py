"""Reusable runtime state for synthesis graph generation."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .grammar_support import (
    DBOpWeightCache,
    DBTemplateWeightCache,
    EFFICIENCY_TEMPLATES,
    OP_TO_TEMPLATE,
    SlotAdaptationCache,
    blend_template_weights_with_db,
    compute_motif_weights_from_op_weights,
)
from .dynamic_template_registry import load_dynamic_template_candidates
from .routing_decision_priors import load_routing_decision_priors

_DEFAULT_DYNAMIC_TEMPLATE_CANDIDATE_PATH = (
    "research/data/synthesis_candidates/dynamic_component_candidates.json"
)


@dataclass(frozen=True, slots=True)
class GenerationRuntimeContext:
    tpl_weights: Mapping[str, float] | None
    motif_weights: Mapping[str, float] | None
    effective_op_weights: Mapping[str, float] | None
    first_tpl_weights: Mapping[str, float] | None
    use_efficiency_first: bool
    slot_motif_weight_multipliers: Mapping[str, Mapping[str, float]] | None
    slot_motif_denylist: Mapping[str, tuple[str, ...]] | None
    slot_adaptations: Mapping[str, list] | None
    routing_decision_priors: Mapping[str, Any] | None
    dynamic_template_candidates: tuple[Any, ...] | None


_db_weight_cache = DBTemplateWeightCache(ttl=60.0)
_db_op_weight_cache = DBOpWeightCache(ttl=60.0)
_slot_adaptation_cache = SlotAdaptationCache(ttl=120.0)
_RUNTIME_CONTEXT_CACHE_MAX = 128
_RUNTIME_CONTEXT_DB_TTL_SECONDS = 60.0
_runtime_context_cache: OrderedDict[tuple[Any, ...], GenerationRuntimeContext] = (
    OrderedDict()
)


def normalize_generation_config(config: Any) -> Any:
    """Apply generation-only config coercions once per batch or single graph."""
    if not config.forced_template and config.template_weights:
        positive_templates = [
            name for name, weight in config.template_weights.items() if weight > 0.0
        ]
        if len(positive_templates) == 1:
            config = replace(config, forced_template=positive_templates[0])

    if config.forced_template and config.composition_depth > 1:
        config = replace(config, composition_depth=1)
    return config


def _template_weights(config: Any) -> Mapping[str, float] | None:
    db_tpl_weights = _db_weight_cache.get() if config.use_db_weights else None
    if config.template_weights:
        return blend_template_weights_with_db(
            dict(config.template_weights), db_tpl_weights
        )
    return db_tpl_weights or None


def _effective_op_weights(config: Any) -> dict[str, float]:
    effective_op_weights: dict[str, float] = (
        dict(config.op_weights) if config.op_weights else {}
    )
    if config.use_db_weights:
        db_op_weights = _db_op_weight_cache.get()
        if db_op_weights:
            for op_name, weight in db_op_weights.items():
                effective_op_weights.setdefault(op_name, weight)
    return effective_op_weights


def _motif_weights(
    config: Any, effective_op_weights: Mapping[str, float]
) -> dict[str, float]:
    motif_weights: dict[str, float] = (
        dict(config.motif_weights) if config.motif_weights else {}
    )
    if effective_op_weights:
        cached_factors = compute_motif_weights_from_op_weights(effective_op_weights)
        for motif_name, (factor, default_lift) in cached_factors.items():
            current = motif_weights.get(motif_name, default_lift)
            motif_weights[motif_name] = current * factor
    return motif_weights


def _apply_exploration_targets(
    config: Any,
    tpl_weights: Mapping[str, float] | None,
    motif_weights: dict[str, float],
) -> Mapping[str, float] | None:
    if not config.exploration_targets:
        return tpl_weights

    from .motifs import ALL_MOTIFS

    for motif in ALL_MOTIFS:
        motif_ops = {step.op_name for step in motif.steps}
        if motif_ops & config.exploration_targets:
            current = motif_weights.get(motif.name, motif.lift)
            motif_weights[motif.name] = current * config.exploration_boost_factor

    if tpl_weights is None:
        from .templates import DEFAULT_TEMPLATE_WEIGHTS

        tpl_weights = dict(DEFAULT_TEMPLATE_WEIGHTS)
    else:
        tpl_weights = dict(tpl_weights)

    for op_name in config.exploration_targets:
        tpl_name = OP_TO_TEMPLATE.get(op_name)
        if tpl_name and tpl_name in tpl_weights:
            tpl_weights[tpl_name] *= config.exploration_boost_factor
    return tpl_weights


def _efficiency_first_weights(
    config: Any,
    tpl_weights: Mapping[str, float] | None,
) -> tuple[Mapping[str, float] | None, bool]:
    if config.structured_sparsity_bias <= 0.5 or not tpl_weights:
        return None, False
    first_tpl_weights = {
        k: (v if k in EFFICIENCY_TEMPLATES else 0.0) for k, v in tpl_weights.items()
    }
    if any(v > 0 for v in first_tpl_weights.values()):
        return first_tpl_weights, True
    return None, False


def _slot_motif_weight_multipliers(
    config: Any,
) -> Mapping[str, Mapping[str, float]] | None:
    if not config.slot_motif_weight_multipliers:
        return None
    return {
        str(slot_key): {str(name): float(weight) for name, weight in weights.items()}
        for slot_key, weights in config.slot_motif_weight_multipliers.items()
        if weights
    }


def _slot_motif_denylist(config: Any) -> Mapping[str, tuple[str, ...]] | None:
    if not config.slot_motif_denylist:
        return None
    return {
        str(slot_key): tuple(sorted({str(name) for name in denied if str(name)}))
        for slot_key, denied in config.slot_motif_denylist.items()
        if denied
    }


def _sorted_mapping_items(mapping: Mapping[Any, Any]) -> tuple[tuple[Any, Any], ...]:
    return tuple(
        sorted((key, _freeze_config_value(value)) for key, value in mapping.items())
    )


def _freeze_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _sorted_mapping_items(value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_freeze_config_value(item) for item in value))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config_value(item) for item in value)
    return value


def _runtime_context_cache_key(config: Any) -> tuple[Any, ...]:
    db_bucket = (
        int(time.monotonic() // _RUNTIME_CONTEXT_DB_TTL_SECONDS)
        if config.use_db_weights
        else None
    )
    routing_prior_token = (
        _path_mtime_cache_token(getattr(config, "routing_decision_prior_path", ""))
        if getattr(config, "use_routing_decision_priors", False)
        else None
    )
    dynamic_template_token = (
        _path_mtime_cache_token(
            getattr(
                config,
                "dynamic_template_candidate_path",
                _DEFAULT_DYNAMIC_TEMPLATE_CANDIDATE_PATH,
            )
        )
        if getattr(config, "use_dynamic_template_candidates", False)
        else None
    )
    return (
        db_bucket,
        config.forced_template,
        config.composition_depth,
        config.use_db_weights,
        config.structured_sparsity_bias,
        config.exploration_boost_factor,
        _freeze_config_value(config.template_weights),
        _freeze_config_value(config.op_weights),
        _freeze_config_value(config.motif_weights),
        _freeze_config_value(config.exploration_targets),
        _freeze_config_value(config.slot_motif_weight_multipliers),
        _freeze_config_value(config.slot_motif_denylist),
        bool(getattr(config, "use_routing_decision_priors", False)),
        routing_prior_token,
        float(getattr(config, "routing_decision_prior_strength", 1.0) or 0.0),
        bool(getattr(config, "use_dynamic_template_candidates", False)),
        dynamic_template_token,
        float(getattr(config, "dynamic_template_candidate_prob", 0.0) or 0.0),
        float(getattr(config, "dynamic_template_candidate_strength", 1.0) or 0.0),
        int(getattr(config, "dynamic_template_max_candidates", 32) or 0),
        int(getattr(config, "dynamic_template_min_lowered_ops", 8) or 0),
    )


def _path_mtime_cache_token(path_or_dir: str) -> tuple[str, int | None]:
    path = Path(path_or_dir)
    artifact = path / "latest.json" if path.is_dir() else path
    try:
        return str(artifact), artifact.stat().st_mtime_ns
    except OSError:
        return str(artifact), None


def build_generation_runtime_context(config: Any) -> GenerationRuntimeContext:
    """Precompute generation state reused across batch candidates."""
    tpl_weights = _template_weights(config)
    effective_op_weights = _effective_op_weights(config)
    motif_weights = _motif_weights(config, effective_op_weights)
    tpl_weights = _apply_exploration_targets(config, tpl_weights, motif_weights)
    first_tpl_weights, use_efficiency_first = _efficiency_first_weights(
        config, tpl_weights
    )
    slot_adaptations = (
        _slot_adaptation_cache.get() or None if config.use_db_weights else None
    )
    routing_decision_priors = None
    if getattr(config, "use_routing_decision_priors", False):
        routing_decision_priors = load_routing_decision_priors(
            getattr(config, "routing_decision_prior_path", "")
        )
    dynamic_template_candidates = None
    if getattr(config, "use_dynamic_template_candidates", False):
        dynamic_template_candidates = load_dynamic_template_candidates(
            getattr(
                config,
                "dynamic_template_candidate_path",
                _DEFAULT_DYNAMIC_TEMPLATE_CANDIDATE_PATH,
            ),
            max_candidates=getattr(config, "dynamic_template_max_candidates", 32),
            min_lowered_ops=getattr(config, "dynamic_template_min_lowered_ops", 8),
        )
    return GenerationRuntimeContext(
        tpl_weights=tpl_weights,
        motif_weights=motif_weights or None,
        effective_op_weights=effective_op_weights or None,
        first_tpl_weights=first_tpl_weights,
        use_efficiency_first=use_efficiency_first,
        slot_motif_weight_multipliers=_slot_motif_weight_multipliers(config),
        slot_motif_denylist=_slot_motif_denylist(config),
        slot_adaptations=slot_adaptations,
        routing_decision_priors=routing_decision_priors,
        dynamic_template_candidates=dynamic_template_candidates or None,
    )


def runtime_context_for_config(config: Any) -> GenerationRuntimeContext:
    """Return reusable generation state for direct repeated single-graph calls."""
    key = _runtime_context_cache_key(config)
    try:
        context = _runtime_context_cache.pop(key)
    except KeyError:
        context = build_generation_runtime_context(config)
        if len(_runtime_context_cache) >= _RUNTIME_CONTEXT_CACHE_MAX:
            _runtime_context_cache.popitem(last=False)
    _runtime_context_cache[key] = context
    return context
