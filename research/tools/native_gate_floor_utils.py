"""Shared native/reciprocal/slot gate floor helpers."""

from __future__ import annotations


DEFAULT_NATIVE_GATE_FLOORS = (0.05, 0.05, 0.10, 0.25, 0.05, 0.15, 0.15, 0.20)
DEFAULT_NATIVE_GATE_FLOORS_CSV = ",".join(str(value) for value in DEFAULT_NATIVE_GATE_FLOORS)


def parse_float_csv(raw: str) -> tuple[float, ...]:
    vals = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not vals:
        raise ValueError("expected at least one comma-separated float")
    return vals
