from __future__ import annotations

import random
import time
from statistics import median

import pytest

from research.scientist.native.core import _try_import_rust_scheduler
from research.synthesis.native_template_selection import (
    pick_template_index_native,
    pick_template_index_python,
)
from research.synthesis.templates import (
    DEFAULT_TEMPLATE_WEIGHTS,
    ROUTING_TEMPLATES,
    _TEMPLATE_DEFAULT_WEIGHT_VECTOR,
    _TEMPLATE_NAME_ORDER,
)


pytestmark = pytest.mark.native


def _sample_override_weights(seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    names = list(_TEMPLATE_NAME_ORDER)
    rng.shuffle(names)
    picked = names[:12]
    overrides: dict[str, float] = {}
    for idx, name in enumerate(picked):
        if idx % 4 == 0:
            overrides[name] = 0.0
        elif idx % 4 == 1:
            overrides[name] = 0.25
        elif idx % 4 == 2:
            overrides[name] = 3.5
        else:
            overrides[name] = 7.0
    return overrides


def _median_ms(fn, *, repeats: int = 5) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        end = time.perf_counter_ns()
        samples.append((end - start) / 1_000_000.0)
    return median(samples)


def test_native_template_selector_matches_python_helper():
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "select_template_index_compiled"):
        pytest.skip("aria_scheduler template selector not available")

    cases = [
        (None, 0.0),
        (None, 0.2),
        (_sample_override_weights(7), 0.0),
        (_sample_override_weights(11), 0.15),
        (_sample_override_weights(19), 0.15),
    ]
    for overrides, exploration_budget in cases:
        rng = random.Random(1234)
        for _ in range(256):
            exploration_draw = rng.random() if exploration_budget > 0.0 else 1.0
            selection_draw = rng.random()
            allowed_names = (
                tuple(sorted(ROUTING_TEMPLATES))
                if overrides is not None and len(overrides) % 2 == 0
                else None
            )
            expected = pick_template_index_python(
                _TEMPLATE_NAME_ORDER,
                _TEMPLATE_DEFAULT_WEIGHT_VECTOR,
                overrides,
                allowed_names=allowed_names,
                exploration_budget=exploration_budget,
                exploration_draw=exploration_draw,
                selection_draw=selection_draw,
            )
            actual = pick_template_index_native(
                _TEMPLATE_NAME_ORDER,
                _TEMPLATE_DEFAULT_WEIGHT_VECTOR,
                overrides,
                allowed_names=allowed_names,
                exploration_budget=exploration_budget,
                exploration_draw=exploration_draw,
                selection_draw=selection_draw,
            )
            assert actual == expected


def test_native_template_selector_perf_gate():
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "select_template_index_compiled"):
        pytest.skip("aria_scheduler template selector not available")

    iterations = 20_000
    exploration_budget = 0.15
    overrides = DEFAULT_TEMPLATE_WEIGHTS
    rng = random.Random(9876)
    draws = [(rng.random(), rng.random()) for _ in range(iterations)]

    for exploration_draw, selection_draw in draws[:256]:
        pick_template_index_python(
            _TEMPLATE_NAME_ORDER,
            _TEMPLATE_DEFAULT_WEIGHT_VECTOR,
            overrides,
            exploration_budget=exploration_budget,
            exploration_draw=exploration_draw,
            selection_draw=selection_draw,
        )
        pick_template_index_native(
            _TEMPLATE_NAME_ORDER,
            _TEMPLATE_DEFAULT_WEIGHT_VECTOR,
            overrides,
            exploration_budget=exploration_budget,
            exploration_draw=exploration_draw,
            selection_draw=selection_draw,
        )

    def run_python() -> None:
        for exploration_draw, selection_draw in draws:
            pick_template_index_python(
                _TEMPLATE_NAME_ORDER,
                _TEMPLATE_DEFAULT_WEIGHT_VECTOR,
                overrides,
                exploration_budget=exploration_budget,
                exploration_draw=exploration_draw,
                selection_draw=selection_draw,
            )

    def run_native() -> None:
        for exploration_draw, selection_draw in draws:
            pick_template_index_native(
                _TEMPLATE_NAME_ORDER,
                _TEMPLATE_DEFAULT_WEIGHT_VECTOR,
                overrides,
                exploration_budget=exploration_budget,
                exploration_draw=exploration_draw,
                selection_draw=selection_draw,
            )

    python_ms = _median_ms(run_python)
    native_ms = _median_ms(run_native)
    speedup = python_ms / native_ms

    assert speedup >= 1.10, (
        f"native template selector too slow: python={python_ms:.3f}ms "
        f"native={native_ms:.3f}ms speedup={speedup:.3f}x"
    )


def test_native_template_selector_subset_perf_gate_vs_zeroed_dict():
    rust = _try_import_rust_scheduler()
    if rust is None or not hasattr(rust, "select_template_index_compiled"):
        pytest.skip("aria_scheduler template selector not available")

    iterations = 20_000
    exploration_budget = 0.15
    allowed_names = tuple(sorted(ROUTING_TEMPLATES))
    allowed_set = set(allowed_names)
    old_zeroed_weights = {
        name: (DEFAULT_TEMPLATE_WEIGHTS.get(name, 1.0) if name in allowed_set else 0.0)
        for name in _TEMPLATE_NAME_ORDER
    }
    rng = random.Random(54321)
    draws = [(rng.random(), rng.random()) for _ in range(iterations)]

    for exploration_draw, selection_draw in draws[:256]:
        pick_template_index_python(
            _TEMPLATE_NAME_ORDER,
            _TEMPLATE_DEFAULT_WEIGHT_VECTOR,
            old_zeroed_weights,
            exploration_budget=exploration_budget,
            exploration_draw=exploration_draw,
            selection_draw=selection_draw,
        )
        pick_template_index_native(
            _TEMPLATE_NAME_ORDER,
            _TEMPLATE_DEFAULT_WEIGHT_VECTOR,
            DEFAULT_TEMPLATE_WEIGHTS,
            allowed_names=allowed_names,
            exploration_budget=exploration_budget,
            exploration_draw=exploration_draw,
            selection_draw=selection_draw,
        )

    def run_old_python_zeroed() -> None:
        for exploration_draw, selection_draw in draws:
            pick_template_index_python(
                _TEMPLATE_NAME_ORDER,
                _TEMPLATE_DEFAULT_WEIGHT_VECTOR,
                old_zeroed_weights,
                exploration_budget=exploration_budget,
                exploration_draw=exploration_draw,
                selection_draw=selection_draw,
            )

    def run_native_subset() -> None:
        for exploration_draw, selection_draw in draws:
            pick_template_index_native(
                _TEMPLATE_NAME_ORDER,
                _TEMPLATE_DEFAULT_WEIGHT_VECTOR,
                DEFAULT_TEMPLATE_WEIGHTS,
                allowed_names=allowed_names,
                exploration_budget=exploration_budget,
                exploration_draw=exploration_draw,
                selection_draw=selection_draw,
            )

    old_python_ms = _median_ms(run_old_python_zeroed)
    native_ms = _median_ms(run_native_subset)
    speedup = old_python_ms / native_ms

    assert speedup >= 1.25, (
        f"native subset selector too slow vs zeroed-dict path: "
        f"python_zeroed={old_python_ms:.3f}ms native_subset={native_ms:.3f}ms "
        f"speedup={speedup:.3f}x"
    )
