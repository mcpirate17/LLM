from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Mapping, Sequence, Tuple

from research.scientist.native.core import _try_import_rust_scheduler


TemplateNameOrder = Tuple[str, ...]
WeightVector = Tuple[float, ...]


@lru_cache(maxsize=8)
def _compile_template_selector_handle(
    names: TemplateNameOrder,
    default_weights: WeightVector,
):
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "compile_template_selector_handle"):
        return None
    try:
        return rust.compile_template_selector_handle(list(names), list(default_weights))
    except Exception:
        return None


def pick_template_index_native(
    names: TemplateNameOrder,
    default_weights: WeightVector,
    override_weights: Mapping[str, float] | None,
    *,
    allowed_names: Iterable[str] | None = None,
    exploration_budget: float,
    exploration_draw: float,
    selection_draw: float,
) -> tuple[int, bool] | None:
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "select_template_index_compiled"):
        return None
    handle = _compile_template_selector_handle(names, default_weights)
    if handle is None:
        return None
    try:
        index, explored = rust.select_template_index_compiled(
            handle,
            float(exploration_budget),
            float(exploration_draw),
            float(selection_draw),
            dict(override_weights) if override_weights else None,
            list(allowed_names) if allowed_names is not None else None,
        )
        return int(index), bool(explored)
    except Exception:
        return None


def pick_template_index_python(
    names: Sequence[str],
    default_weights: Sequence[float],
    override_weights: Mapping[str, float] | None,
    *,
    allowed_names: Iterable[str] | None = None,
    exploration_budget: float,
    exploration_draw: float,
    selection_draw: float,
) -> tuple[int, bool]:
    if not names:
        raise ValueError("template selector has no names")

    allowed_name_set = set(allowed_names) if allowed_names is not None else None
    allowed_indices = [
        idx
        for idx, name in enumerate(names)
        if allowed_name_set is None or name in allowed_name_set
    ]
    if not allowed_indices:
        allowed_indices = list(range(len(names)))
    allowed_index_set = set(allowed_indices)

    effective_weights = [
        (
            _effective_weight(name, default_weights[idx], override_weights)
            if idx in allowed_index_set
            else 0.0
        )
        for idx, name in enumerate(names)
    ]
    weighted_pool = [idx for idx in allowed_indices if effective_weights[idx] > 0.0]
    if not weighted_pool:
        weighted_pool = allowed_indices

    if exploration_budget > 0.0 and exploration_draw < exploration_budget:
        return _uniform_pick(weighted_pool, selection_draw), True

    total = sum(effective_weights)
    if total <= 0.0:
        return _uniform_pick(allowed_indices, selection_draw), False

    threshold = max(0.0, min(0.999999999999, selection_draw)) * total
    for idx, weight in enumerate(effective_weights):
        if weight <= 0.0:
            continue
        if threshold < weight:
            return idx, False
        threshold -= weight
    return len(names) - 1, False


def _effective_weight(
    name: str,
    default_weight: float,
    override_weights: Mapping[str, float] | None,
) -> float:
    weight = (
        float(override_weights.get(name, default_weight))
        if override_weights
        else float(default_weight)
    )
    if weight <= 0.0 or weight != weight:
        return 0.0
    return weight


def _uniform_pick(indices: Sequence[int], draw: float) -> int:
    if not indices:
        return 0
    return indices[_uniform_index(len(indices), draw)]


def _uniform_index(length: int, draw: float) -> int:
    if length <= 1:
        return 0
    scaled = int(max(0.0, min(0.999999999999, draw)) * float(length))
    return min(length - 1, scaled)
