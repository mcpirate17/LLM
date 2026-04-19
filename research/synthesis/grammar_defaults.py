from __future__ import annotations

DEFAULT_CATEGORY_WEIGHTS: dict[str, float] = {
    "elementwise_unary": 2.0,
    "elementwise_binary": 1.5,
    "reduction": 0.8,
    "linear_algebra": 1.0,
    "structural": 1.0,
    "parameterized": 2.0,
    "mixing": 1.5,
    "sequence": 1.2,
    "frequency": 1.0,
    "math_space": 1.5,
    "functional": 3.0,
}


def default_category_weights() -> dict[str, float]:
    return dict(DEFAULT_CATEGORY_WEIGHTS)
