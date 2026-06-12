from __future__ import annotations

from functools import lru_cache
from collections.abc import Mapping as MappingABC
from typing import Iterable, Iterator, Mapping, Sequence, Tuple

from research.scientist.native.core import _try_import_rust_scheduler


TemplateNameOrder = Tuple[str, ...]
WeightVector = Tuple[float, ...]
OverrideItems = tuple[tuple[str, float], ...]


class TemplateWeightOverrides(MappingABC[str, float]):
    """Immutable weight map carrying the native selector's precomputed key."""

    __slots__ = ("override_items", "_values")

    def __init__(self, override_items: OverrideItems):
        self.override_items = override_items
        self._values = dict(override_items)

    def __getitem__(self, key: str) -> float:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def get(self, key: str, default: float | None = None) -> float | None:
        return self._values.get(key, default)


@lru_cache(maxsize=256)
def make_template_weight_overrides(
    override_items: OverrideItems,
) -> TemplateWeightOverrides:
    return TemplateWeightOverrides(override_items)


@lru_cache(maxsize=8)
def _compile_template_selector_handle(
    names: TemplateNameOrder,
    default_weights: WeightVector,
):
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "compile_template_selector_handle"):
        return None
    # A failure here is a bug in the Rust layer or our inputs — raise, don't
    # silently degrade every subsequent selection to the Python path.
    return rust.compile_template_selector_handle(list(names), list(default_weights))


@lru_cache(maxsize=128)
def _allowed_indices(
    names: TemplateNameOrder,
    allowed_names: tuple[str, ...] | None,
) -> list[int] | None:
    if allowed_names is None:
        return None
    index_by_name = {name: idx for idx, name in enumerate(names)}
    indices = sorted(
        {index_by_name[name] for name in allowed_names if name in index_by_name}
    )
    return list(indices) if indices else None


@lru_cache(maxsize=256)
def _override_arrays(
    names: TemplateNameOrder,
    override_items: tuple[tuple[str, float], ...] | None,
) -> tuple[list[int], list[float]] | None:
    if not override_items:
        return None
    index_by_name = {name: idx for idx, name in enumerate(names)}
    indices: list[int] = []
    weights: list[float] = []
    for name, weight in override_items:
        idx = index_by_name.get(name)
        if idx is None:
            continue
        indices.append(idx)
        weights.append(float(weight))
    if not indices:
        return None
    return indices, weights


def _override_key(
    override_weights: Mapping[str, float] | None,
) -> OverrideItems | None:
    if not override_weights:
        return None
    prepared = getattr(override_weights, "override_items", None)
    if prepared is not None:
        return prepared
    return tuple(
        (str(name), float(weight)) for name, weight in override_weights.items()
    )


def _allowed_key(allowed_names: Iterable[str] | None) -> tuple[str, ...] | None:
    if allowed_names is None:
        return None
    if isinstance(allowed_names, frozenset):
        return _allowed_key_from_frozenset(allowed_names)
    if isinstance(allowed_names, tuple):
        return allowed_names
    return tuple(sorted(str(name) for name in allowed_names))


@lru_cache(maxsize=128)
def _allowed_key_from_frozenset(allowed_names: frozenset[str]) -> tuple[str, ...]:
    return tuple(sorted(str(name) for name in allowed_names))


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
    arrays_fn = getattr(rust, "select_template_index_compiled_arrays", None)
    if arrays_fn is None:
        return None
    handle = _compile_template_selector_handle(names, default_weights)
    if handle is None:
        return None
    override = _override_arrays(names, _override_key(override_weights))
    if override is None:
        override_indices = None
        override_values = None
    else:
        override_indices, override_values = override
    allowed_indices = _allowed_indices(names, _allowed_key(allowed_names))
    # A failure past this point is a bug (the Rust layer exists and claims
    # this API) — raise rather than silently falling back to Python.
    index, explored = arrays_fn(
        handle,
        float(exploration_budget),
        float(exploration_draw),
        float(selection_draw),
        override_indices,
        override_values,
        allowed_indices,
    )
    return int(index), bool(explored)


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
