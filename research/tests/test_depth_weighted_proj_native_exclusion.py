"""Pin ``depth_weighted_proj`` out of native dispatch until backward lands.

The 2026-04-16 live run surfaced a runtime crash:

    RuntimeError: unsupported op: no backward kernel for op: depth_weighted_proj

The C forward kernel exists but the Rust backward arm is missing, so
letting the native dispatcher handle this op produces a fast forward
followed by a crash the first time a gradient flows through it. Five
ops alias onto ``depth_weighted_proj`` (``gated_lane_blend``,
``route_lanes``, ``depth_gated_transform``, ``route_recursion``,
``adaptive_recursion``), all of which inherit the bug.

This regression pins the op *out* of ``_NATIVE_C_KERNEL_OPS`` so a
well-meaning re-add would fail the test suite. Delete this test once
``ffi::aria_depth_weighted_proj_backward_f32`` and the matching arm in
``NativeKernelDispatch::dispatch_backward`` land in aria-scheduler.
"""

from __future__ import annotations

from research.scientist.native.dispatch import (
    _NATIVE_C_KERNEL_OPS,
    _NATIVE_OP_ALIASES,
)


DEPTH_WEIGHTED_ALIASES = {
    "adaptive_recursion",
    "gated_lane_blend",
    "route_lanes",
    "depth_gated_transform",
    "route_recursion",
}


def test_depth_weighted_proj_is_excluded_from_native_c_kernels() -> None:
    """Native dispatch must not claim this op until backward is implemented."""
    assert "depth_weighted_proj" not in _NATIVE_C_KERNEL_OPS, (
        "depth_weighted_proj re-added to _NATIVE_C_KERNEL_OPS but the Rust "
        "backward kernel is still missing — any graph that routes a gradient "
        "through it will crash with 'no backward kernel'. Remove this test "
        "and the allowlist entry together once aria_depth_weighted_proj_"
        "backward_f32 ships."
    )


def test_depth_weighted_aliases_still_resolve_to_canonical_name() -> None:
    """Aliases should still map to the canonical op name.

    Excluding from native dispatch doesn't mean renaming — the Python
    fallback path still uses the canonical name. The alias table must
    remain intact so template-generated graphs using
    ``gated_lane_blend`` resolve correctly through the PyTorch code
    path.
    """
    for alias in DEPTH_WEIGHTED_ALIASES:
        assert _NATIVE_OP_ALIASES.get(alias) == "depth_weighted_proj", (
            f"Alias {alias!r} no longer resolves to depth_weighted_proj — "
            "capability-first and routing-first templates rely on this."
        )
