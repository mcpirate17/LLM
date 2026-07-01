"""Compact dynamic math-sweep feature extraction for fab ledger consumers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations
from typing import Any


DESCRIPTOR_DELTA_KEYS: tuple[str, ...] = (
    "long_range_reach",
    "content_dependence",
    "content_match_gating",
    "effective_rank",
    "causality_violation",
    "spectral_radius",
)

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "math_sweep_version": ("math_sweep_version", "op_math_sweep_version", "version"),
    "math_sweep_passed": ("math_sweep_passed", "op_math_sweep_passed", "passed"),
    "math_variant_selected": (
        "math_variant_selected",
        "op_math_variant_selected",
        "variant_selected",
    ),
    "math_variant_family": (
        "math_variant_family",
        "op_math_variant_family",
        "math_sweep_selected_family",
        "selected_family",
        "variant_family",
    ),
    "math_variant_transform": (
        "math_variant_transform",
        "op_math_variant_transform",
        "math_sweep_selected_transform",
        "selected_transform",
        "variant_transform",
    ),
    "math_variant_target": (
        "math_variant_target",
        "op_math_variant_target",
        "op_physics_target",
        "target",
        "selected_target",
    ),
    "math_variant_score": (
        "math_variant_score",
        "op_math_variant_score",
        "math_sweep_score",
        "score",
        "variant_score",
    ),
    "math_variant_failure_reason": (
        "math_variant_failure_reason",
        "op_math_variant_failure_reason",
        "failure_reason",
        "variant_failure_reason",
    ),
    "math_variant_artifact_ref": (
        "math_variant_artifact_ref",
        "math_sweep_artifact",
        "math_sweep_artifact_ref",
        "op_math_variant_artifact_ref",
        "artifact_ref",
    ),
    "math_variant_stability_band": (
        "math_variant_stability_band",
        "physics_stability_band",
        "op_math_variant_stability_band",
        "stability_band",
    ),
    "math_variant_family_pair": (
        "math_variant_family_pair",
        "op_math_variant_family_pair",
        "family_pair",
    ),
}
_BOOL_FIELDS = {
    "math_sweep_passed",
    "math_variant_selected",
    "math_variant_target_improved",
    "math_variant_rank_collapsed",
    "math_variant_self_dominance_collapsed",
    "math_variant_softmax_twin_regression",
    "math_variant_causality_failed",
    "math_variant_spectral_unstable",
}
_NUMERIC_FIELDS = {
    "math_variant_score",
    *(f"math_variant_delta_{key}" for key in DESCRIPTOR_DELTA_KEYS),
}
_SWEEP_PREFIXES = (
    "math_sweep_",
    "math_variant_",
    "op_math_sweep_",
    "op_math_variant_",
)


def extract_math_sweep_metadata(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized, flat math-sweep metadata from axes or ledger metadata."""

    sources = _mapping_sources(source)
    if not _has_sweep_signal(sources):
        return {}

    out: dict[str, Any] = {}
    for field, aliases in _FIELD_ALIASES.items():
        value = _lookup(sources, aliases)
        if value is None:
            continue
        out[field] = _coerce_field(field, value)

    axes = _lookup(
        sources,
        ("math_variant_axes", "op_math_variant_axes", "math_sweep_selected_axes"),
    )
    if axes is None:
        axes = _lookup(sources, ("selected_axes", "variant_axes"))
    if isinstance(axes, Mapping):
        out["math_variant_axes"] = dict(axes)

    descriptor_delta = _lookup(
        sources, ("math_sweep_descriptor_delta", "descriptor_delta")
    )
    delta_map = descriptor_delta if isinstance(descriptor_delta, Mapping) else {}
    for key in DESCRIPTOR_DELTA_KEYS:
        field = f"math_variant_delta_{key}"
        value = _lookup(sources, (field, f"op_{field}"))
        if value is None:
            value = delta_map.get(key)
        if value is None and key == "content_match_gating":
            value = delta_map.get("content_gating")
        if value is not None:
            out[field] = float(value)

    score = _optional_float(out.get("math_variant_score"))
    if score is not None and "math_variant_target_improved" not in out:
        out["math_variant_target_improved"] = score > 0.0

    reason = str(out.get("math_variant_failure_reason") or "")
    out.setdefault("math_variant_rank_collapsed", reason == "rank_collapse")
    out.setdefault(
        "math_variant_self_dominance_collapsed",
        reason == "self_dominance_collapse",
    )
    out.setdefault(
        "math_variant_softmax_twin_regression",
        reason == "softmax_twin_regression",
    )
    out.setdefault("math_variant_causality_failed", reason == "causality_violation")
    out.setdefault("math_variant_spectral_unstable", reason == "spectral_instability")
    return out


def math_sweep_axis_values(source: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Categorical sweep features for Beta-Binomial axis lift."""

    meta = extract_math_sweep_metadata(source)
    if not meta:
        return ()
    axes: list[tuple[str, str]] = []
    for field in (
        "math_variant_family",
        "math_variant_transform",
        "math_variant_target",
        "math_variant_stability_band",
        "math_variant_failure_reason",
    ):
        value = meta.get(field)
        if isinstance(value, str) and value:
            axes.append((field, value))

    explicit_pair = meta.get("math_variant_family_pair")
    if isinstance(explicit_pair, str) and explicit_pair:
        axes.append(("math_variant_family_pair", explicit_pair))
    else:
        families = _string_values(meta.get("math_variant_family"))
        for left, right in combinations(sorted(set(families)), 2):
            axes.append(("math_variant_family_pair", f"{left}+{right}"))

    for key in DESCRIPTOR_DELTA_KEYS:
        field = f"math_variant_delta_{key}"
        value = _optional_float(meta.get(field))
        if value is not None:
            axes.append((field, delta_bucket(value)))

    target_improved = meta.get("math_variant_target_improved")
    if isinstance(target_improved, bool):
        axes.append(("math_variant_target_improved", str(target_improved).lower()))
    return tuple(axes)


def math_sweep_surrogate_features(source: Mapping[str, Any]) -> dict[str, float]:
    """Numeric and categorical sweep features for the ledger surrogate."""

    meta = extract_math_sweep_metadata(source)
    if not meta:
        return {}
    features: dict[str, float] = {}
    for axis, value in math_sweep_axis_values(meta):
        features[f"{axis}={value}"] = 1.0
    for field in _NUMERIC_FIELDS:
        value = _optional_float(meta.get(field))
        if value is not None:
            features[field] = value
    for field in _BOOL_FIELDS:
        value = meta.get(field)
        if isinstance(value, bool):
            features[field] = 1.0 if value else 0.0
    return features


def math_sweep_failure_reason(source: Mapping[str, Any]) -> str | None:
    """Return a pre-assembly sweep failure reason if the metadata says it failed."""

    meta = extract_math_sweep_metadata(source)
    if not meta:
        return None
    passed = meta.get("math_sweep_passed")
    reason = str(meta.get("math_variant_failure_reason") or "").strip()
    if passed is False:
        return reason or "math_sweep_failed"
    return reason or None


def delta_bucket(value: float, *, eps: float = 1e-6) -> str:
    if value > eps:
        return "up"
    if value < -eps:
        return "down"
    return "flat"


def _mapping_sources(source: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    sources: list[Mapping[str, Any]] = [source]
    axes = source.get("math_axes")
    if isinstance(axes, Mapping):
        sources.append(axes)
    nested = source.get("math_sweep")
    if isinstance(nested, Mapping):
        sources.append(nested)
    return tuple(sources)


def _has_sweep_signal(sources: tuple[Mapping[str, Any], ...]) -> bool:
    for source in sources:
        if isinstance(source.get("math_sweep"), Mapping):
            return True
        if any(isinstance(key, str) and key.startswith(_SWEEP_PREFIXES) for key in source):
            return True
    return False


def _lookup(sources: tuple[Mapping[str, Any], ...], keys: Iterable[str]) -> Any:
    for source in sources:
        for key in keys:
            if key in source:
                return source[key]
    return None


def _coerce_field(field: str, value: Any) -> Any:
    if field in _BOOL_FIELDS:
        return _coerce_bool(value)
    if field in _NUMERIC_FIELDS:
        return float(value)
    return value


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part for part in value.split("+") if part)
    if isinstance(value, Iterable):
        return tuple(str(part) for part in value if str(part))
    return ()
